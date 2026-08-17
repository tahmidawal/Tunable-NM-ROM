"""Poisson-2D online-cost measurements on ONE GPU, all ladder points measured
SEQUENTIALLY IN ONE PROCESS (cross-N / cross-k ratios from different GPUs are
not comparable).

Protocol for every reported time: 2 warm-ups, then the median of TIME_REPS (7)
`block_until_ready`-synchronised repetitions, same device, same process.  The
FOM baseline is the testbed's own jitted CG at the testbed's own tolerance --
the function that produced the truth -- warmed and compiled the same way, and
its converged residual is asserted below FOM_RES_TOL before anything is timed.

  MODE=n : N ladder at fixed (K, M, m).  The coordinate decoder is meshfree, so
           the same N=64 checkpoint is used at every N.  The NNLS-EQ weights are
           REFIT ON EACH N's GRID, so the decoder-side quadrature target and the
           source-side projection Lambda^-1 Phi^T f use the SAME grid rule at
           every N (fitting once at N=64 and pairing it with an N=512 source
           rule would mix two discretisations of the same continuum integral).
           Times: FOM CG, input preprocessing, ROM latent solve, full-field
           decode; plus iterations, termination reasons and the ROM error.
  MODE=k : k ladder at N=64, fixed (M, m): per-iteration cost and iterations to
           termination for each checkpoint in PKLS, and the linear POD control
           (whose online solve is one precomputed pseudo-inverse matvec).
  MODE=m : m / M ladder at N=64, fixed K: per-solve and per-iteration cost for
           each (M, m, pool).  ACCURACY for these ladders is measured by
           pro_colloc.py (the authoritative path); the error column here is a
           cross-check on a smaller test set.

There is no absolute residual tolerance in the reference LM (pro_common.lm_generic
stops on relative decrease / step size / budget), so the iteration counts below
are ITERATIONS TO TERMINATION and the termination reasons are reported with
them.  Setting REL_TOL>0 adds an invariant absolute stop ||r|| <= REL_TOL*||f_m||.

Usage: PKL=<pkl> MODE=n NS=32,64,128,256,512 [M=64] [MQ=256] ... python fu_timing.py <out.json>
       PKLS=<pkl,...> MODE=k [POD_KS=...] ...            python fu_timing.py <out.json>
       PKL=<pkl> MODE=m [MS_LADDER=64,...] [M_LADDER=...] python fu_timing.py <out.json>
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

MODE = os.environ.get("MODE", "n")
OUT = sys.argv[1]
NS = [int(v) for v in os.environ.get("NS", "32,64,128,256,512").split(",")]
M_MODES = int(os.environ.get("M", "64"))
MQ = int(os.environ.get("MQ", "256"))
GN_ITERS = int(os.environ.get("GN_ITERS", "60"))
TIME_REPS = int(os.environ.get("TIME_REPS", "7"))
WARM = int(os.environ.get("TIME_WARM", "2"))
N_TEST = int(os.environ.get("N_TEST", "16"))
EQ_SNAPS = int(os.environ.get("EQ_SNAPS", "64"))
EQ_PERTURB = int(os.environ.get("EQ_PERTURB", "3"))
EQ_ROWS = int(os.environ.get("EQ_ROWS", "3072"))
EQ_CAND = int(os.environ.get("EQ_CAND_OFF", "4096"))
INIT = os.environ.get("INIT", "mean")                 # mean | nearest
REL_TOL = float(os.environ.get("REL_TOL", "0.0"))
FOM_RES_TOL = float(os.environ.get("FOM_RES_TOL", "1e-10"))
POOLS = [p for p in os.environ.get("POOLS", "offgrid").split(",") if p]
MS_LADDER = [int(v) for v in os.environ.get("MS_LADDER", "64,128,256,512,1024").split(",") if v]
M_LADDER = [int(v) for v in os.environ.get("M_LADDER", "16,32,64,128,256").split(",") if v]
POD_KS = [int(v) for v in os.environ.get("POD_KS", "2,4,6,8,12,16,24,32,48,64").split(",") if v]
EQ_SEED = int(os.environ.get("EQ_SEED", str(mp.SEED + 20259)))


def time_fn(fn, reps=None, warm=None):
    reps = TIME_REPS if reps is None else reps
    warm = WARM if warm is None else warm
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), [float(t) for t in ts]


# --------------------------------------------------------------- NNLS-EQ
def eq_fit(dec, grid, Z_tr, K, M, m, pool):
    """NNLS-EQ quadrature weights for the weak form, mirroring the `nnls` /
    `nnlsoff` schemes of pro_colloc.py (decoder-OUTPUT snapshots; targets = the
    exact full-grid projections at THIS grid).  The snapshot indices, latent
    perturbations and row subset come from a FIXED stream (EQ_SEED), so every
    (M, m, pool) in a ladder is fitted on the same snapshots."""
    t0 = time.time()
    r_eq = np.random.default_rng(EQ_SEED)
    n_tr = Z_tr.shape[0]
    idx = r_eq.choice(n_tr, size=min(EQ_SNAPS, n_tr), replace=False)
    if pool == "offgrid":
        cand_np = np.random.default_rng(mp.SEED + 12345).uniform(0.0, 1.0, size=(EQ_CAND, 2))
        spec_c = dict(kind="weak", alpha=0.0, M=M)
        PhiT_c, _ = pc.colloc_mode_table(grid, spec_c, "offgrid", cand_np)
        PhiT_f, _ = pc.colloc_mode_table(grid, spec_c, "offgrid", np.asarray(grid.coords_int))
        Phi = np.asarray(PhiT_c).T                                   # (n_c, M')
        Phi_f = np.asarray(PhiT_f).T * grid.dx ** 2                  # grid rule at THIS grid
        cand = jnp.asarray(cand_np)
    else:
        cand_np = np.asarray(grid.coords_int)
        mask = np.asarray(grid.mode_mask(M)).astype(bool)
        I, Jm = np.nonzero(mask)
        S = np.asarray(grid.S)
        Phi = S[grid.ix_full - 1][:, I] * S[grid.iy_full - 1][:, Jm]  # (n_i^2, M')
        Phi_f = Phi
        cand = grid.coords_int
    snap_fn = jax.jit(lambda z: dec(z, cand))
    full_fn = jax.jit(lambda z: dec(z, grid.coords_int))
    snaps, fulls = [], []
    for i in idx:
        z = jnp.asarray(Z_tr[i])
        for zz in [z] + [z + 0.05 * jnp.asarray(r_eq.standard_normal(K)) for _ in range(EQ_PERTURB)]:
            snaps.append(np.asarray(snap_fn(zz)))
            fulls.append(snaps[-1] if pool == "grid" else np.asarray(full_fn(zz)))
    R = np.stack(snaps); Rf = np.stack(fulls)
    b = (Rf @ Phi_f).reshape(-1)
    n_c = cand_np.shape[0]
    G = np.einsum("sp,pm->smp", R, Phi).reshape(-1, n_c)
    sc = np.linalg.norm(G, axis=1) + 1e-300
    G, b = G / sc[:, None], b / sc
    rows = r_eq.choice(G.shape[0], size=min(G.shape[0], EQ_ROWS), replace=False)
    wts, rnorm, _ = pc.nnls_capped(G[rows], b[rows], max_support=m)
    supp = np.nonzero(wts > 0)[0]
    if len(supp) >= m:
        keep = supp[np.argsort(-wts[supp])[:m]]; padded = 0
    else:
        rest = np.setdiff1d(np.arange(n_c), supp)
        pad = rest[np.argsort(-np.abs(R).mean(0)[rest])[:m - len(supp)]]
        keep = np.concatenate([supp, pad]); padded = len(pad)
    Gk = G[:, keep]
    wq, _, _ = pc.nnls_capped(Gk, b, max_support=len(keep))
    wq = np.where(wq > 0, wq, 1e-8 * max(wq.max(), 1e-300))
    res = Gk @ wq - b
    rel_rows = np.abs(res) / (np.abs(b) + 1e-300)
    info = dict(M=M, m=int(len(keep)), pool=pool, grid_N=grid.N, support=int(len(supp)),
                padded=int(padded), rnorm_capped=float(rnorm), rnorm_final=float(np.linalg.norm(res)),
                b_norm=float(np.linalg.norm(b)),
                rel_fit=float(np.linalg.norm(res) / np.linalg.norm(b)),
                row_rel_median=float(np.median(rel_rows)), row_rel_p95=float(np.quantile(rel_rows, 0.95)),
                row_rel_max=float(np.max(rel_rows)), n_rows=int(len(rows)), n_cand=int(n_c),
                secs=time.time() - t0)
    print(f"  NNLS-EQ {pool} M={M} m={m} @N={grid.N}: support {len(supp)} (+{padded}) "
          f"rel fit {info['rel_fit']:.2e} (row p95 {info['row_rel_p95']:.1e}, max "
          f"{info['row_rel_max']:.1e}) [{info['secs']:.0f}s]", flush=True)
    return cand_np[keep], wq, info


# --------------------------------------------------------------- jitted LM
def make_lm_jit(dec, K, pts, wq, PhiT, Wl, budget, rel_tol=0.0):
    """Jitted LM on r(z) = Wl*(PhiT @ (wq * dec(z, pts))) - f_m: the same
    residual as pro_common.make_colloc_objective's weak core, with the same
    acceptance, damping and stopping rules as pro_common.lm_generic
    (use_rel_dec=True), plus an OPTIONAL invariant absolute stop
    ||r|| <= rel_tol*||f_m|| (rel_tol=0 disables it, matching the reference).
    Reason codes: 0 budget, 1 converged (rel-dec/step), 2 tol, 3 lambda_max,
    5 nan_at_init."""
    pts = jnp.asarray(pts); wq = jnp.asarray(wq); PhiT = jnp.asarray(PhiT); Wl = jnp.asarray(Wl)

    def r_of(z, f_m):
        return Wl * (PhiT @ (wq * dec(z, pts))) - f_m
    rJ = lambda z, f_m: (r_of(z, f_m), jax.jacfwd(r_of)(z, f_m))
    rn_fn = lambda z, f_m: jnp.linalg.norm(r_of(z, f_m))

    def lm(z0, f_m):
        tol = rel_tol * jnp.linalg.norm(f_m)
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
            v_new = jnp.where(finite, rn_fn(z_new, f_m), jnp.inf)
            accept = finite & jnp.isfinite(v_new) & (v_new < val)
            rel_dec = jnp.where(accept, (val - v_new) / (jnp.abs(val) + 1e-300), 1.0)
            step = jnp.linalg.norm(dz) / (1.0 + jnp.linalg.norm(z))
            r2, J2 = jax.lax.cond(accept, lambda: rJ(z_new, f_m), lambda: (r, J))
            z = jnp.where(accept, z_new, z); val = jnp.where(accept, v_new, val)
            lam = jnp.where(accept, jnp.maximum(lam / 3.0, 1e-12), jnp.minimum(lam * 10.0, 1e12))
            acc = acc + accept.astype(jnp.int32); nJ = nJ + accept.astype(jnp.int32)
            reason = jnp.where(accept & (val <= tol) & (tol > 0), jnp.int32(2),
                       jnp.where(accept & ((rel_dec < 1e-12) | (step < 1e-13)), jnp.int32(1),
                        jnp.where((~accept) & (lam >= 1e12), jnp.int32(3), jnp.int32(0))))
            return (z, J2, r2, val, lam, att + 1, acc, nJ, reason)

        z, J, r, val, lam, att, acc, nJ, reason = jax.lax.while_loop(cond, body, init)
        return z, val, nJ, acc, att, reason

    return jax.jit(lm)


# --------------------------------------------------------------- shared pieces
def load():
    d, cfg, stages_all, Z_tr, HARD_BC = pc.load_pkl(os.environ["PKL"] if MODE != "k"
                                                    else os.environ["PKLS"].split(",")[0])
    return cfg, stages_all, Z_tr, HARD_BC


def sources(n, cx, cy, w, a, N_TRAIN):
    return np.stack([mp.source_interior(n, cx[N_TRAIN + i], cy[N_TRAIN + i], w[N_TRAIN + i],
                                        a[N_TRAIN + i]) for i in range(N_TEST)])


def fom_solve(n, Fs):
    """The testbed's own jitted CG at the testbed's tolerance -- the function
    that generated the truth.  Aborts if the converged residual is too large."""
    op = lambda v: mp.neg_lap_interior(v, n)
    solve_one = jax.jit(lambda F: jax.scipy.sparse.linalg.cg(op, F, tol=mp.CG_TOL,
                                                             maxiter=mp.CG_MAXITER)[0])
    U_int = np.asarray(jax.lax.map(solve_one, jnp.asarray(Fs)))
    res = float(np.max([np.linalg.norm(np.asarray(op(jnp.asarray(U_int[i]))) - Fs[i])
                        / np.linalg.norm(Fs[i]) for i in range(Fs.shape[0])]))
    if not np.isfinite(res) or res > FOM_RES_TOL:
        raise SystemExit(f"N={n}: FOM CG rel residual {res:.2e} > {FOM_RES_TOL:.0e} -- "
                         f"the baseline at this N is not converged, refusing to time it")
    return solve_one, U_int, res


def rom_arm(dec, grid, K, Z_tr, M, m, pool, Fs, U_int, z0, label):
    """Fit the EQ rule, build the jitted LM, time one solve (median of
    TIME_REPS), and report iterations / reasons / error over the N_TEST
    sources."""
    n = grid.N
    spec = dict(kind="weak", alpha=1.0, M=M)
    if pool == "full":
        pts = np.asarray(grid.coords_int); wq = np.ones(pts.shape[0]); info = None
    else:
        pts, wq, info = eq_fit(dec, grid, Z_tr, K, M, m, pool)
    kind = "grid" if pool == "grid" else "offgrid"
    if pool == "full":
        kind = "grid"
    PhiT, Wl = pc.colloc_mode_table(grid, spec, kind, pts)
    f_ms = [jnp.asarray(pc.weak_source_term(grid, spec, kind, Fs[i])) for i in range(N_TEST)]
    lm = make_lm_jit(dec, K, pts, wq, PhiT, Wl, GN_ITERS, REL_TOL)

    def once():
        z, val, nJ, acc, att, reason = lm(z0, f_ms[0]); z.block_until_ready()
    rom_med, rom_all = time_fn(once)
    errs, iters, atts, reasons = [], [], [], []
    coords = grid.coords
    for i in range(N_TEST):
        zi, vi, nJi, acci, atti, ri = lm(z0, f_ms[i])
        u_full = np.asarray(dec(zi, coords)).reshape(n, n)
        u_ref = np.zeros((n, n)); u_ref[1:-1, 1:-1] = U_int[i]
        errs.append(float(np.linalg.norm(u_full - u_ref) / np.linalg.norm(u_ref)))
        iters.append(int(nJi)); atts.append(int(atti)); reasons.append(int(ri))
    out = dict(label=label, M=M, m=int(len(wq)), pool=pool, n_modes=int(PhiT.shape[0]),
               rom_solve_s=rom_med, rom_all=rom_all, rom_iters_mean=float(np.mean(iters)),
               rom_attempts_mean=float(np.mean(atts)),
               rom_s_per_iter=rom_med / max(iters[0], 1),
               rom_s_per_attempt=rom_med / max(atts[0], 1),
               rom_reasons={str(r): reasons.count(r) for r in set(reasons)},
               rom_rel_l2_mean=float(np.mean(errs)), rom_rel_l2_med=float(np.median(errs)),
               rom_rel_l2_max=float(np.max(errs)), eq_info=info)
    return out


def main():
    print(f"jax_backend={jax.default_backend()} device={jax.devices()[0]} MODE={MODE} "
          f"M={M_MODES} m={MQ} reps={TIME_REPS} warm={WARM} rel_tol={REL_TOL}", flush=True)
    cfg, stages_all, Z_tr, HARD_BC = load()
    K = cfg["K_LAT"]; N0 = mp.N; N_TRAIN = mp.N_TRAIN
    dec = pc.make_decoder(stages_all[:1], hard_bc=bool(HARD_BC))
    cx, cy, w, a, z_par = mp.sample_params()
    zt = np.asarray(z_par)
    nn_idx = np.argmin(((zt[N_TRAIN:, None, :] - zt[None, :N_TRAIN, :]) ** 2).sum(-1), axis=1)
    z_mean = jnp.asarray(Z_tr.mean(0))
    z_init = z_mean if INIT == "mean" else jnp.asarray(Z_tr[nn_idx[0]])
    report = dict(config=dict(mode=MODE, pkl_config=cfg, hard_bc=HARD_BC, K=K, M=M_MODES, m=MQ,
                              gn_iters=GN_ITERS, time_reps=TIME_REPS, time_warm=WARM,
                              n_test=N_TEST, init=INIT, rel_tol=REL_TOL, ns=NS,
                              backend=jax.default_backend(), device=str(jax.devices()[0]),
                              cg_tol=mp.CG_TOL, fom_res_tol=FOM_RES_TOL, eq_seed=EQ_SEED),
                  rows=[])

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    if MODE == "n":
        for n in NS:
            t0 = time.time()
            grid = pc.Grid(n)
            Fs = sources(n, cx, cy, w, a, N_TRAIN)
            solve_one, U_int, res = fom_solve(n, Fs)
            F0 = jnp.asarray(Fs[0])
            fom_med, fom_all = time_fn(lambda: solve_one(F0).block_until_ready())
            spec = dict(kind="weak", alpha=1.0, M=M_MODES)
            arm = rom_arm(dec, grid, K, Z_tr, M_MODES, MQ, "offgrid", Fs, U_int, z_init,
                          f"N{n}")
            # input preprocessing (Lambda^-1 Phi^T f) and full-field decode, same protocol
            pre_med, _ = time_fn(lambda: np.asarray(pc.weak_source_term(grid, spec, "offgrid", Fs[0])))
            dec_full = jax.jit(lambda z: dec(z, grid.coords))
            dec_med, _ = time_fn(lambda: dec_full(z_init).block_until_ready())
            row = dict(N=n, n_dof=(n - 2) ** 2, fom_cg_s=fom_med, fom_all=fom_all,
                       fom_max_rel_residual=res, preprocess_s=pre_med, decode_full_field_s=dec_med,
                       speedup_solve_only=fom_med / arm["rom_solve_s"],
                       speedup_with_preprocess=fom_med / (arm["rom_solve_s"] + pre_med),
                       speedup_end_to_end=fom_med / (arm["rom_solve_s"] + pre_med + dec_med),
                       secs=time.time() - t0, **arm)
            report["rows"].append(row); save()
            print(f"RESULT N={n:4d} FOM CG {fom_med*1e3:8.2f} ms  ROM {arm['rom_solve_s']*1e3:6.2f} ms "
                  f"({arm['rom_iters_mean']:.1f} iters, {arm['rom_s_per_iter']*1e3:.3f} ms/iter)  "
                  f"pre {pre_med*1e3:.2f} ms  decode {dec_med*1e3:.2f} ms  "
                  f"speedup solve {row['speedup_solve_only']:.1f}x / e2e {row['speedup_end_to_end']:.1f}x  "
                  f"ROM err {arm['rom_rel_l2_mean']:.3e} [{row['secs']:.0f}s]", flush=True)
    elif MODE == "k":
        n = N0
        grid = pc.Grid(n)
        Fs = sources(n, cx, cy, w, a, N_TRAIN)
        solve_one, U_int, res = fom_solve(n, Fs)
        F0 = jnp.asarray(Fs[0])
        fom_med, fom_all = time_fn(lambda: solve_one(F0).block_until_ready())
        report["fom_cg_s"] = fom_med; report["fom_all"] = fom_all
        report["fom_max_rel_residual"] = res
        print(f"FOM CG N={n}: {fom_med*1e3:.2f} ms", flush=True)
        for path in os.environ["PKLS"].split(","):
            d_, cfg_, stages_, Ztr_, hb_ = pc.load_pkl(path)
            Kk = cfg_["K_LAT"]
            dk = pc.make_decoder(stages_[:1], hard_bc=bool(hb_))
            z0 = jnp.asarray(Ztr_.mean(0)) if INIT == "mean" else jnp.asarray(Ztr_[nn_idx[0]])
            arm = rom_arm(dk, grid, Kk, Ztr_, M_MODES, MQ, "offgrid", Fs, U_int, z0,
                          os.path.basename(path))
            arm.update(kind="coord", k=Kk, ckpt=os.path.basename(path),
                       train_seed=cfg_.get("train_seed"),
                       speedup_solve_only=fom_med / arm["rom_solve_s"])
            report["rows"].append(arm); save()
            print(f"RESULT coord k={Kk:3d} ROM {arm['rom_solve_s']*1e3:6.2f} ms "
                  f"({arm['rom_iters_mean']:.1f} iters, {arm['rom_s_per_iter']*1e3:.3f} ms/iter) "
                  f"err {arm['rom_rel_l2_mean']:.3e}", flush=True)
        # POD control: the ROM is LINEAR in the coefficients, so the online solve is one
        # precomputed pseudo-inverse matvec (the exact minimiser of the SAME objective)
        U_tr = np.asarray(mp.build_snapshots(n)[0])[:N_TRAIN][:, grid.ix_full * n + grid.iy_full]
        Vfull, sv, _ = np.linalg.svd(U_tr.T, full_matrices=False)
        spec = dict(kind="weak", alpha=1.0, M=M_MODES)
        PhiT_g, Wl_g = pc.colloc_mode_table(grid, spec, "grid", np.asarray(grid.coords_int))
        f_ms = [np.asarray(pc.weak_source_term(grid, spec, "grid", Fs[i])) for i in range(N_TEST)]
        for k in POD_KS:
            V = Vfull[:, :k]
            A_ = np.asarray(PhiT_g) @ V                          # (M', k)
            pinv = jnp.asarray(np.linalg.pinv(A_))
            b0 = jnp.asarray(f_ms[0])
            apply_ = jax.jit(lambda b: pinv @ b)
            med, all_ = time_fn(lambda: apply_(b0).block_until_ready())
            errs = []
            for i in range(N_TEST):
                c = np.asarray(apply_(jnp.asarray(f_ms[i])))
                u_full = np.zeros((n, n)); u_full[1:-1, 1:-1] = (V @ c).reshape(grid.n_i, grid.n_i)
                u_ref = np.zeros((n, n)); u_ref[1:-1, 1:-1] = U_int[i]
                errs.append(float(np.linalg.norm(u_full - u_ref) / np.linalg.norm(u_ref)))
            row = dict(kind="pod", k=k, M=M_MODES, m=grid.n_i ** 2, pool="full",
                       n_modes=int(A_.shape[0]), rom_solve_s=med, rom_all=all_,
                       rom_iters_mean=1.0, rom_s_per_iter=med, square_system=bool(A_.shape[0] <= k),
                       cond=float(np.linalg.cond(A_)), rank=int(np.linalg.matrix_rank(A_)),
                       rom_rel_l2_mean=float(np.mean(errs)), rom_rel_l2_med=float(np.median(errs)),
                       speedup_solve_only=fom_med / med)
            report["rows"].append(row); save()
            print(f"RESULT pod   k={k:3d} solve {med*1e6:7.1f} us (1 matvec, cond {row['cond']:.1e}) "
                  f"err {row['rom_rel_l2_mean']:.3e}", flush=True)
    else:                                          # MODE == "m"
        n = N0
        grid = pc.Grid(n)
        Fs = sources(n, cx, cy, w, a, N_TRAIN)
        solve_one, U_int, res = fom_solve(n, Fs)
        F0 = jnp.asarray(Fs[0])
        fom_med, _ = time_fn(lambda: solve_one(F0).block_until_ready())
        report["fom_cg_s"] = fom_med; report["fom_max_rel_residual"] = res
        arms = [(M_MODES, m, pool) for m in MS_LADDER for pool in POOLS] + \
               [(M_MODES, None, "full")] + \
               [(M, 4 * M, pool) for M in M_LADDER for pool in POOLS] + \
               [(M, None, "full") for M in M_LADDER]
        seen = set()
        for M, m, pool in arms:
            key = (M, m, pool)
            if key in seen:
                continue
            seen.add(key)
            arm = rom_arm(dec, grid, K, Z_tr, M, m or grid.n_i ** 2, pool, Fs, U_int, z_init,
                          f"M{M}_m{m or 'full'}_{pool}")
            arm["speedup_solve_only"] = fom_med / arm["rom_solve_s"]
            report["rows"].append(arm); save()
            print(f"RESULT M={M:4d} m={arm['m']:5d} {pool:8s} solve {arm['rom_solve_s']*1e3:7.2f} ms "
                  f"({arm['rom_iters_mean']:.1f} iters, {arm['rom_s_per_iter']*1e3:.3f} ms/iter) "
                  f"err {arm['rom_rel_l2_mean']:.3e}", flush=True)
    report["complete"] = True; save()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
