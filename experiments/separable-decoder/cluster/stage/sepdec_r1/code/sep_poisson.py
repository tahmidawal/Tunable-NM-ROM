"""Separable-decoder Poisson-2D cell: train (no POD), weak NM-ROM solve.

Two solve arms through the SAME incumbent trust-LM weak solver
(ctol_tol.lm_tau_poisson):
  meshfree : dec(z, pts) evaluates the feature network inside the loop
  cached   : dec_fast(z, .) = G_q @ h(z), G_q = features at the EQ nodes,
             cached once -- no spatial network in the compiled iteration.
GATE 0: the two arms' weak residual/Jacobian and solve outputs must agree to
<= 1e-12 relative.  Timing and error come from the same invocation.  The FOM
CG iso-accuracy ladder is timed in the same job.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import sep_common as sc

import jax
import jax.numpy as jnp

import pro_common as pc                      # noqa: E402  (path set by sc)
from pro_common import mp                    # noqa: E402
import ctol_eq                                # noqa: E402
import ctol_tol                               # noqa: E402

F64 = jnp.float64

N = int(os.environ.get("N", "64"))
K = int(os.environ.get("K", "16"))
R = int(os.environ.get("R", "64"))
M_MODES = int(os.environ.get("M", str(4 * K)))
MQ = int(os.environ.get("MQ", str(4 * M_MODES)))
STEPS = int(os.environ.get("STEPS", "30000"))
LR = float(os.environ.get("LR", "1e-3"))
N_TEST = int(os.environ.get("N_TEST", "16"))
GN_ITERS = int(os.environ.get("GN_ITERS", "60"))
TAUS = [float(v) for v in os.environ.get("TAUS", "1e-3,1e-2").split(",")]
TR_FACTOR = float(os.environ.get("TR_FACTOR", "1.0"))
SEED0 = int(os.environ.get("SEED0", "0"))
OUT = os.environ.get("OUT", "sep_poisson.json")
CKPT = os.environ.get("CKPT", f"sep_poisson_N{N}_K{K}_R{R}.pkl")
FOM_LADDER = [float(v) for v in os.environ.get(
    "FOM_LADDER", "1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,1e-6").split(",")]


def main():
    dev = jax.devices()[0]
    sc.log(f"jax_backend={dev.platform} device={dev} x64={jax.config.jax_enable_x64} "
           f"N={N} K={K} R={R} M={M_MODES} m={MQ} steps={STEPS} seed={SEED0}")
    t_all = time.time()
    report = dict(config=dict(
        pde="poisson2d", N=N, k=K, r=R, M=M_MODES, m=MQ, steps=STEPS, lr=LR,
        taus=TAUS, n_test=N_TEST, gn_iters=GN_ITERS, tr_factor=TR_FACTOR,
        seed=SEED0, data_seed=mp.SEED, cg_tol=mp.CG_TOL,
        arch="separable: FourierFeat-MLP g(x)->R^r  x  MLP-head h(z)->R^r, "
             "hard poly BC; NO POD anywhere",
        objective=f"weak alpha=1 M={M_MODES}, NNLS-EQ m={MQ} grid nodes",
        solver="ctol_tol.lm_tau_poisson (incumbent trust-LM), both arms",
        x64=True, matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local")), rows=[], complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    # ------------------ data (regenerated from seed) -------------------------
    grid = pc.Grid(N)
    n_i = grid.n_i
    int_idx = np.asarray(grid.ix_full * N + grid.iy_full)
    U_all = np.asarray(mp.build_snapshots(N)[0])
    U_tr = U_all[:mp.N_TRAIN][:, int_idx]
    cx, cy, w, a, _z = mp.sample_params()
    Fs = np.stack([mp.source_interior(N, cx[mp.N_TRAIN + i], cy[mp.N_TRAIN + i],
                                      w[mp.N_TRAIN + i], a[mp.N_TRAIN + i])
                   for i in range(N_TEST)])
    solve_one = jax.jit(lambda F: jax.scipy.sparse.linalg.cg(
        lambda v: mp.neg_lap_interior(v, N), F, tol=mp.CG_TOL,
        maxiter=mp.CG_MAXITER)[0])
    U_int = np.asarray(jax.lax.map(solve_one, jnp.asarray(Fs)))
    res = float(np.max([np.linalg.norm(np.asarray(
        mp.neg_lap_interior(jnp.asarray(U_int[i]), N)) - Fs[i])
        / np.linalg.norm(Fs[i]) for i in range(N_TEST)]))
    sc.log(f"  truth: {N_TEST} test sources, FOM CG rel residual {res:.2e}")
    assert res < 1e-10, "unconverged truth"
    U_int = U_int.reshape(N_TEST, -1)
    tn = np.array([np.linalg.norm(U_int[i]) for i in range(N_TEST)])
    coords_int = np.asarray(grid.coords_int)

    # ------------------ train ------------------------------------------------
    params, Z_tr, tinfo = sc.train_autodecoder(
        jax.random.PRNGKey(SEED0), coords_int, U_tr, K, R,
        steps=STEPS, lr=LR, tag=f"poisson N={N} k={K} r={R}")
    report["train"] = tinfo
    dec = sc.SeparableDecoder(params, K, R)
    sc.save_pkl(CKPT, params, Z_tr, report["config"])

    # held-out representation oracle on 4 test fields (mean/max)
    om = []
    zbar = Z_tr.mean(0)
    for i in range(min(4, N_TEST)):
        _, val = sc.oracle_fit(dec, coords_int, U_int[i], [zbar], budget=150)
        om.append(val)
    report["oracle_test_rel_l2"] = dict(mean=float(np.mean(om)),
                                        max=float(np.max(om)), n=len(om))
    sc.log(f"  test oracle rel-L2: mean {np.mean(om):.3e} max {np.max(om):.3e}")
    save()

    # ------------------ weak form + EQ --------------------------------------
    spec = dict(kind="weak", alpha=1.0, M=M_MODES)
    mask = np.asarray(grid.mode_mask(M_MODES)).astype(bool)
    I, Jm = np.nonzero(mask)
    S_ = np.asarray(grid.S)
    Phi_f = S_[grid.ix_full - 1][:, I] * S_[grid.iy_full - 1][:, Jm]
    cand_pos = ctol_eq.candidate_pool(n_i * n_i)
    cand_j = jnp.asarray(coords_int[cand_pos])
    u_cand = jax.jit(lambda z: dec(z, cand_j))
    u_full = jax.jit(lambda z: dec(z, jnp.asarray(coords_int)))
    keep, wq, eq_info = ctol_eq.eq_fit_poisson(
        u_cand, u_full, Phi_f[cand_pos], Phi_f, Z_tr, K, MQ,
        f"sep poisson N={N} k={K} M={M_MODES} m={MQ}", pc.nnls_capped)
    report["eq"] = {k_: v for k_, v in eq_info.items()
                    if isinstance(v, (int, float, str, bool, type(None)))}
    node_pos = cand_pos[keep]
    pts_np = coords_int[node_pos]
    PhiT, Wl = pc.colloc_mode_table(grid, spec, "grid", pts_np)
    f_ms = [jnp.asarray(np.asarray(pc.weak_source_term(grid, spec, "grid", Fs[i])))
            for i in range(N_TEST)]

    # ------------------ the two arms ----------------------------------------
    train_radius = float(np.max(np.linalg.norm(Z_tr - Z_tr.mean(0), axis=1)))
    trust = TR_FACTOR * train_radius if TR_FACTOR > 0 else np.inf
    z0 = jnp.asarray(Z_tr.mean(0))

    G_q = dec.feat_at(pts_np)                       # (m, r) cached bank
    h_fn = dec.head_fn()
    dec_fast = lambda z, xy: G_q @ h_fn(z)          # ignores xy: nodes are baked in
    G_full = dec.feat_at(coords_int)                # (n_i^2, r) readout bank
    u_full_fast = jax.jit(lambda z: G_full @ h_fn(z))

    arms = dict(meshfree=(dec, u_full),
                cached=(dec_fast, u_full_fast))

    # GATE 0: identity of the two arms through the SAME weak residual
    def r_of(dfn, z, f_m):
        return jnp.asarray(Wl) * (jnp.asarray(PhiT) @
                                  (jnp.asarray(wq) * dfn(z, jnp.asarray(pts_np)))) - f_m
    g0 = []
    rng = np.random.default_rng(SEED0)
    for _ in range(5):
        zt = jnp.asarray(Z_tr[rng.integers(len(Z_tr))] +
                         0.05 * rng.standard_normal(K))
        ra = r_of(dec, zt, f_ms[0]); rb = r_of(dec_fast, zt, f_ms[0])
        Ja = jax.jacfwd(lambda z: r_of(dec, z, f_ms[0]))(zt)
        Jb = jax.jacfwd(lambda z: r_of(dec_fast, z, f_ms[0]))(zt)
        g0.append(max(float(jnp.max(jnp.abs(ra - rb)) / (jnp.max(jnp.abs(ra)) + 1e-300)),
                      float(jnp.max(jnp.abs(Ja - Jb)) / (jnp.max(jnp.abs(Ja)) + 1e-300))))
    report["gate0_max_rel_dev"] = float(np.max(g0))
    sc.log(f"  GATE 0 (meshfree vs cached weak r/J identity): max rel dev {np.max(g0):.2e}")
    assert np.max(g0) < 1e-12, "gate 0 failed: cached arm is not the same discrete map"

    # ------------------ solves: same solver, both arms, all taus ------------
    for arm, (dfn, ufull) in arms.items():
        lm, _ = ctol_tol.lm_tau_poisson(dfn, K, pts_np, wq, PhiT, Wl,
                                        GN_ITERS, trust_delta=trust)

        def pipe_fn(f_m, tau, _lm=lm, _uf=ufull):
            out = _lm(z0, f_m, tau)
            return (_uf(out[0]),) + out[1:]
        pipe = jax.jit(pipe_fn)
        ctol_tol.burn_in(1.5)
        for tau in TAUS:
            per_t, per_err, per_jac, per_reason = [], [], [], []
            for i in range(N_TEST):
                u, val, v0, nJ, acc, att, rsn = pipe(f_ms[i], tau)
                med, _ = sc.time_fn(lambda _f=f_ms[i]:
                                    pipe(_f, tau)[0].block_until_ready())
                per_t.append(med)
                per_err.append(float(np.linalg.norm(np.asarray(u) - U_int[i]) / tn[i]))
                per_jac.append(int(nJ)); per_reason.append(int(rsn))
            cens = [r_ not in ctol_tol.POISSON_TAU_OK for r_ in per_reason]
            row = dict(pde="poisson2d", method=f"sep_{arm}", N=N, k=K, r=R,
                       M=M_MODES, m=int(len(wq)), tau=tau,
                       time_ms=float(np.median(per_t)) * 1e3,
                       time_ms_all=[t * 1e3 for t in per_t],
                       err_rel_l2=float(np.mean(per_err)),
                       err_rel_l2_max=float(np.max(per_err)),
                       jac_evals=float(np.mean(per_jac)),
                       censored_frac=float(np.mean(cens)),
                       n_sources=N_TEST, trust_delta=trust)
            report["rows"].append(row)
            sc.log(f"   {arm:8s} tau={tau:.0e}  solve+decode {row['time_ms']:8.3f} ms  "
                   f"jac {row['jac_evals']:5.1f}  err {row['err_rel_l2']:.3e}  "
                   f"cens {row['censored_frac']*100:3.0f}%")
            save()

    # ------------------ FOM iso-accuracy ladder (same job) ------------------
    for tol in sorted(set(FOM_LADDER), reverse=True):
        s1 = jax.jit(lambda F, _t=tol: jax.scipy.sparse.linalg.cg(
            lambda v: mp.neg_lap_interior(v, N), F, tol=_t,
            maxiter=mp.CG_MAXITER)[0])
        errs, ts = [], []
        for i in range(N_TEST):
            Fi = jnp.asarray(Fs[i])
            u = np.asarray(s1(Fi)).reshape(-1)
            errs.append(float(np.linalg.norm(u - U_int[i]) / tn[i]))
            med, _ = sc.time_fn(lambda _F=Fi: s1(_F).block_until_ready())
            ts.append(med)
        report.setdefault("fom", []).append(dict(
            fom_tol=tol, time_ms=float(np.median(ts)) * 1e3,
            err_rel_l2=float(np.mean(errs)), err_rel_l2_max=float(np.max(errs))))
        sc.log(f"   FOM CG tol={tol:.0e}: {np.median(ts)*1e3:8.3f} ms  "
               f"err {np.mean(errs):.3e}")
        save()

    report["complete"] = True
    report["total_seconds"] = time.time() - t_all
    save()
    sc.log(f"DONE poisson [{time.time()-t_all:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main()
