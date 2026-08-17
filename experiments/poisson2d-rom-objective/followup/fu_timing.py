"""Poisson-2D online-cost ladder in N on ONE GPU, sequentially.

The K-latent hard-BC coordinate decoder (trained at N=64) is meshfree, and the
weak-form / NNLS-EQ ROM with a MESHFREE candidate pool never touches the mesh:
its online cost is (m decoder evaluations + an M' x m matvec + a K x K solve)
per LM iteration.  The only N-dependent online piece is the source-side term
Lambda^{-1} Phi^T f (M' numbers), which we time separately as 'preprocessing'
(a spectral projection of the input, O(n) work, no solver).

Per N in NS: 16 held-out test sources at that resolution (FOM = CG on the
N-grid, the testbed's own solver / tolerance) -> FOM solve time (source 0,
median of TIME_REPS after 2 warm-ups, block_until_ready), preprocessing time,
ROM solve time (JITTED lax.while_loop LM, budget GN_ITERS, mean-latent init;
same acceptance rule as pro_common.lm_generic), iterations, per-iteration
time, and the ROM error vs the FD solution at that N (all 16 sources; the
decoder was trained on the N=64 grid, so this doubles as ROM mesh transfer).

Usage: PKL=<hard-bc K8 pkl, N=64> [NS=32,64,128,256,512] [M=64] [MQ=256] [GN_ITERS=60]
       [TIME_REPS=7] [N_TEST=16] N=64 python fu_timing.py <out.json>
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pro_common as pc  # noqa: E402
from pro_common import mp, F64  # noqa: E402

PKL = os.environ["PKL"]
OUT = sys.argv[1]
NS = [int(v) for v in os.environ.get("NS", "32,64,128,256,512").split(",")]
M_MODES = int(os.environ.get("M", "64"))
MQ = int(os.environ.get("MQ", "256"))
GN_ITERS = int(os.environ.get("GN_ITERS", "60"))
TIME_REPS = int(os.environ.get("TIME_REPS", "7"))
N_TEST = int(os.environ.get("N_TEST", "16"))
EQ_SNAPS = int(os.environ.get("EQ_SNAPS", "64"))
EQ_PERTURB = int(os.environ.get("EQ_PERTURB", "3"))
EQ_ROWS = int(os.environ.get("EQ_ROWS", "3072"))
EQ_CAND = int(os.environ.get("EQ_CAND_OFF", "4096"))
INIT = os.environ.get("INIT", "mean")


def time_fn(fn, reps=TIME_REPS, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), [float(t) for t in ts]


def eq_meshfree(dec, grid, Z_tr, K, N_TRAIN, cx, cy, w, a, M, m, rng):
    """NNLS-EQ on a fixed off-grid candidate pool for the WEAK form (decoder-
    output snapshots) -- the `nnlsoff` scheme of pro_colloc.py, verbatim logic."""
    t0 = time.time()
    cand_off = rng.uniform(0.0, 1.0, size=(EQ_CAND, 2))
    idx = rng.choice(N_TRAIN, size=min(EQ_SNAPS, N_TRAIN), replace=False)
    cand = jnp.asarray(cand_off)
    snap_fn = jax.jit(lambda z: dec(z, cand))
    full_fn = jax.jit(lambda z: dec(z, grid.coords_int))
    snaps, fulls = [], []
    for i in idx:
        z = jnp.asarray(Z_tr[i])
        for zz in [z] + [z + 0.05 * jnp.asarray(rng.standard_normal(K)) for _ in range(EQ_PERTURB)]:
            snaps.append(np.asarray(snap_fn(zz))); fulls.append(np.asarray(full_fn(zz)))
    R = np.stack(snaps); Rf = np.stack(fulls)
    spec_c = dict(kind="weak", alpha=0.0, M=M)
    PhiT_c, _ = pc.colloc_mode_table(grid, spec_c, "offgrid", cand_off)
    PhiT_f, _ = pc.colloc_mode_table(grid, spec_c, "offgrid", np.asarray(grid.coords_int))
    Phi = np.asarray(PhiT_c).T
    b = (Rf @ np.asarray(PhiT_f).T * grid.dx ** 2).reshape(-1)
    n_c = cand_off.shape[0]
    G = np.einsum("sp,pm->smp", R, Phi).reshape(-1, n_c)
    sc = np.linalg.norm(G, axis=1) + 1e-300
    G, b = G / sc[:, None], b / sc
    rows = rng.choice(G.shape[0], size=min(G.shape[0], EQ_ROWS), replace=False)
    wts, rnorm, n_outer = pc.nnls_capped(G[rows], b[rows], max_support=m)
    supp = np.nonzero(wts > 0)[0]
    if len(supp) >= m:
        keep = supp[np.argsort(-wts[supp])[:m]]; padded = 0
    else:
        rest = np.setdiff1d(np.arange(n_c), supp); score = np.abs(R).mean(0)
        pad = rest[np.argsort(-score[rest])[:m - len(supp)]]
        keep = np.concatenate([supp, pad]); padded = len(pad)
    Gk = G[:, keep]
    wq, rnorm_final, _ = pc.nnls_capped(Gk, b, max_support=len(keep))
    if np.any(wq <= 0):
        wq = np.where(wq > 0, wq, 1e-8 * max(wq.max(), 1e-300))
    rnorm_final = float(np.linalg.norm(Gk @ wq - b))
    info = dict(support=int(len(supp)), padded=int(padded), rnorm_capped=float(rnorm),
                rnorm_final=rnorm_final, b_norm=float(np.linalg.norm(b)), n_rows=int(len(rows)),
                secs=time.time() - t0, fit_grid_N=grid.N)
    print(f"  NNLS-EQ meshfree m={m} M={M}: support {len(supp)} (+{padded}) rnorm final {rnorm_final:.2e}/{np.linalg.norm(b):.2e} [{info['secs']:.0f}s]", flush=True)
    return cand_off[keep], wq, info


def make_lm_jit(dec, K, pts, wq, PhiT, Wl, budget):
    """Jitted LM on r(z) = Wl*(PhiT @ (wq * dec(z, pts))) - f_m; same acceptance,
    damping and stopping rules as pro_common.lm_generic (use_rel_dec=True)."""
    pts = jnp.asarray(pts); wq = jnp.asarray(wq); PhiT = jnp.asarray(PhiT); Wl = jnp.asarray(Wl)

    def r_of(z, f_m):
        return Wl * (PhiT @ (wq * dec(z, pts))) - f_m
    rJ = lambda z, f_m: (r_of(z, f_m), jax.jacfwd(r_of)(z, f_m))
    rn_fn = lambda z, f_m: jnp.linalg.norm(r_of(z, f_m))

    def lm(z0, f_m):
        r0, J0 = rJ(z0, f_m)
        v0 = jnp.linalg.norm(r0)
        init = (z0, J0, r0, v0, jnp.asarray(1e-6, F64), jnp.int32(0), jnp.int32(0), jnp.int32(1),
                jnp.where(jnp.isfinite(v0), jnp.int32(0), jnp.int32(5)))

        def cond(s):
            return (s[8] == 0) & (s[5] < budget)

        def body(s):
            z, J, r, val, lam, att, acc, nJ, _ = s
            H = J.T @ J; g = J.T @ r
            D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K, dtype=F64)
            dz = jnp.linalg.solve(H + lam * D, -g)
            finite = jnp.all(jnp.isfinite(dz))
            z_new = z + jnp.where(finite, dz, 0.0)
            v_new = rn_fn(z_new, f_m)
            accept = finite & jnp.isfinite(v_new) & (v_new < val)
            rel_dec = jnp.where(accept, (val - v_new) / (jnp.abs(val) + 1e-300), 1.0)
            step = jnp.linalg.norm(dz) / (1.0 + jnp.linalg.norm(z))
            r2, J2 = jax.lax.cond(accept, lambda: rJ(z_new, f_m), lambda: (r, J))
            z = jnp.where(accept, z_new, z); val = jnp.where(accept, v_new, val)
            lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12), jnp.minimum(lam * 10.0, 1e12))
            acc = acc + accept.astype(jnp.int32); nJ = nJ + accept.astype(jnp.int32)
            reason = jnp.where(accept & ((rel_dec < 1e-12) | (step < 1e-13)), 1,
                               jnp.where((~accept) & (lam >= 1e12), 3, 0)).astype(jnp.int32)
            return (z, J2, r2, val, lam, att + 1, acc, nJ, reason)

        z, J, r, val, lam, att, acc, nJ, reason = jax.lax.while_loop(cond, body, init)
        return z, val, nJ, acc, att, reason

    return jax.jit(lm)


def main():
    print(f"jax_backend={jax.default_backend()} device={jax.devices()[0]} NS={NS} M={M_MODES} m={MQ}", flush=True)
    d, cfg, stages_all, Z_tr, HARD_BC = pc.load_pkl(PKL)
    K = cfg["K_LAT"]; N0 = mp.N; N_TRAIN = mp.N_TRAIN
    dec = pc.make_decoder(stages_all[:1], hard_bc=bool(HARD_BC))
    grid0 = pc.Grid(N0)
    cx, cy, w, a, _ = mp.sample_params()
    rng = np.random.default_rng(mp.SEED + 12345)
    pts, wq, eq_info = eq_meshfree(dec, grid0, Z_tr, K, N_TRAIN, cx, cy, w, a, M_MODES, MQ, rng)
    z_mean = jnp.asarray(Z_tr.mean(0))
    report = dict(config=dict(pkl=os.path.basename(PKL), pkl_config=cfg, hard_bc=HARD_BC, K=K,
                              M=M_MODES, m=MQ, gn_iters=GN_ITERS, time_reps=TIME_REPS, n_test=N_TEST,
                              init=INIT, ns=NS, backend=jax.default_backend(), device=str(jax.devices()[0]),
                              cg_tol=mp.CG_TOL),
                  eq_info=eq_info, rows=[])
    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)
    for n in NS:
        t0 = time.time()
        grid = pc.Grid(n)
        spec = dict(kind="weak", alpha=1.0, M=M_MODES)
        PhiT, Wl = pc.colloc_mode_table(grid, spec, "offgrid", pts)      # continuum modes at the EQ points
        # FOM: CG on the n-grid for the 16 test sources (also the reference fields)
        op = lambda v: mp.neg_lap_interior(v, n)
        solve_one = jax.jit(lambda F: jax.scipy.sparse.linalg.cg(op, F, tol=mp.CG_TOL, maxiter=mp.CG_MAXITER)[0])
        Fs = np.stack([mp.source_interior(n, cx[N_TRAIN + i], cy[N_TRAIN + i], w[N_TRAIN + i], a[N_TRAIN + i])
                       for i in range(N_TEST)])
        U_int = np.asarray(jax.lax.map(solve_one, jnp.asarray(Fs)))
        res = float(np.max([np.linalg.norm(np.asarray(op(jnp.asarray(U_int[i]))) - Fs[i]) / np.linalg.norm(Fs[i]) for i in range(N_TEST)]))
        F0 = jnp.asarray(Fs[0])
        fom_med, fom_all = time_fn(lambda: solve_one(F0).block_until_ready())
        # preprocessing: Lambda^{-1} Phi^T f (continuum modes, grid rule) for source 0
        def pre_once():
            return np.asarray(pc.weak_source_term(grid, spec, "offgrid", Fs[0]))
        pre_med, _ = time_fn(pre_once, reps=3, warm=1)
        f_ms = [jnp.asarray(pc.weak_source_term(grid, spec, "offgrid", Fs[i])) for i in range(N_TEST)]
        # ROM: jitted LM
        lm = make_lm_jit(dec, K, pts, wq, PhiT, Wl, GN_ITERS)
        z0 = z_mean
        def rom_once():
            z, val, nJ, acc, att, reason = lm(z0, f_ms[0]); z.block_until_ready(); return z, val, nJ, acc, att, reason
        z, val, nJ, acc, att, reason = rom_once()
        rom_med, rom_all = time_fn(rom_once)
        # errors on all 16 sources vs the FD solution at this n
        coords = grid.coords
        errs, iters = [], []
        for i in range(N_TEST):
            zi, vi, nJi, acci, atti, ri = lm(z0, f_ms[i])
            u_full = np.asarray(dec(zi, coords)).reshape(n, n)
            u_ref = np.zeros((n, n)); u_ref[1:-1, 1:-1] = U_int[i]
            errs.append(float(np.linalg.norm(u_full - u_ref) / np.linalg.norm(u_ref))); iters.append(int(nJi))
        row = dict(N=n, n_dof=(n - 2) ** 2, fom_cg_s=fom_med, fom_all=fom_all, fom_max_rel_residual=res,
                   preprocess_s=pre_med, rom_solve_s=rom_med, rom_all=rom_all, rom_iters_src0=int(nJ),
                   rom_attempts_src0=int(att), rom_s_per_iter=rom_med / max(int(nJ), 1),
                   rom_reason_src0=int(reason), rom_iters_mean=float(np.mean(iters)),
                   rom_rel_l2_mean=float(np.mean(errs)), rom_rel_l2_med=float(np.median(errs)),
                   rom_rel_l2_max=float(np.max(errs)), n_modes=int(PhiT.shape[0]),
                   speedup_solve_only=fom_med / rom_med, speedup_with_preprocess=fom_med / (rom_med + pre_med),
                   secs=time.time() - t0)
        report["rows"].append(row); save()
        print(f"RESULT N={n:4d} FOM CG {fom_med*1e3:8.2f} ms  ROM {rom_med*1e3:6.2f} ms ({int(nJ)} iters, "
              f"{row['rom_s_per_iter']*1e3:.3f} ms/iter)  pre {pre_med*1e3:.2f} ms  speedup {row['speedup_solve_only']:.1f}x "
              f"(w/ pre {row['speedup_with_preprocess']:.1f}x)  ROM err {row['rom_rel_l2_mean']:.3e} [{row['secs']:.0f}s]", flush=True)
    report["complete"] = True; save()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
