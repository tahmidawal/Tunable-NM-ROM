"""Separable-decoder Poisson-2D cell: train (no POD), weak NM-ROM solve.

Two solve arms through the SAME incumbent trust-LM weak solver
(ctol_tol.lm_tau_poisson):
  meshfree : dec(z, pts) evaluates the feature network inside the loop
  cached   : dec_fast(z, .) = G_q @ h(z), G_q = features at the EQ nodes,
             cached once -- no spatial network in the compiled iteration.
GATE 0: the two arms' weak residual/Jacobian and solve outputs must agree to
<= 1e-12 relative.

N-scaling round (2026-08-23): implements the MANDATORY MEASUREMENT RULES from
HANDOFF.md / AUDIT-2026-08-23.md:
  - the timed ROM path is end-to-end: grid source -> weak projection f_m ->
    LM solve -> FULL-GRID decode (the projection is inside the timed jit and
    verified against the incumbent pc.weak_source_term <= 1e-12);
  - raw timing repetitions retained for every timing site;
  - balanced AB/BA paired blocks for ROM vs classical baselines;
  - the timed call's output is compared against the error-bearing call's
    output and the max deviation stored;
  - stop-reason distributions recorded next to every row;
  - a FRESH-SEED test cohort (never part of the seed-0 draw) is evaluated
    alongside the same-seed held-out cohort;
  - the classical ladder includes CG and the exact dense-spectral solve
    (this rectangle is separable; that control keeps the comparison honest).
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
N_TEST_FRESH = int(os.environ.get("N_TEST_FRESH", "16"))
FRESH_SEED = int(os.environ.get("FRESH_SEED", "20260823"))
GN_ITERS = int(os.environ.get("GN_ITERS", "60"))
TAUS = [float(v) for v in os.environ.get("TAUS", "1e-3,1e-2").split(",")]
TAU_MAIN = float(os.environ.get("TAU_MAIN", "1e-2"))
TR_FACTOR = float(os.environ.get("TR_FACTOR", "1.0"))
SEED0 = int(os.environ.get("SEED0", "0"))
REPS = int(os.environ.get("REPS", "7"))
PAIR_REPS = int(os.environ.get("PAIR_REPS", "4"))
OUT = os.environ.get("OUT", "sep_poisson.json")
CKPT = os.environ.get("CKPT", f"sep_poisson_N{N}_K{K}_R{R}.pkl")
FOM_LADDER = [float(v) for v in os.environ.get(
    "FOM_LADDER", "1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,1e-6").split(",")]
ARCH = sc.arch_from_env()

REASON_NAMES = {0: "budget", 1: "stalled", 2: "tol", 3: "lambda_max", 5: "nan"}


def main():
    dev = jax.devices()[0]
    sc.log(f"jax_backend={dev.platform} device={dev} x64={jax.config.jax_enable_x64} "
           f"N={N} K={K} R={R} M={M_MODES} m={MQ} steps={STEPS} seed={SEED0} "
           f"arch={ARCH}")
    t_all = time.time()
    report = dict(config=dict(
        pde="poisson2d", N=N, k=K, r=R, M=M_MODES, m=MQ, steps=STEPS, lr=LR,
        taus=TAUS, tau_main=TAU_MAIN, n_test=N_TEST, n_test_fresh=N_TEST_FRESH,
        fresh_seed=FRESH_SEED, gn_iters=GN_ITERS, tr_factor=TR_FACTOR,
        seed=SEED0, data_seed=mp.SEED, n_train=mp.N_TRAIN, cg_tol=mp.CG_TOL,
        reps=REPS, pair_reps=PAIR_REPS, arch=ARCH,
        arch_desc="separable: FourierFeat-MLP g(x)->R^r  x  MLP-head h(z)->R^r, "
                  "hard poly BC; NO POD anywhere",
        objective=f"weak alpha=1 M={M_MODES}, NNLS-EQ m={MQ} grid nodes",
        solver="ctol_tol.lm_tau_poisson (incumbent trust-LM), both arms",
        timed_path="end-to-end: grid source -> weak projection -> LM -> "
                   "full-grid decode, one jit",
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
    del U_all
    coords_int = np.asarray(grid.coords_int)

    solve_truth = jax.jit(lambda F: jax.scipy.sparse.linalg.cg(
        lambda v: mp.neg_lap_interior(v, N), F, tol=mp.CG_TOL,
        maxiter=mp.CG_MAXITER)[0])

    def make_cohort(name, params_idx0, seed=None, n_srcs=0):
        """Test sources + CG truth.  Same-seed cohort: indices >= N_TRAIN of
        the mp.SEED draw (held-out).  Fresh cohort: its own seed's draw."""
        if seed is None:
            cx, cy, w, a, _z = mp.sample_params()
            sel = range(params_idx0, params_idx0 + n_srcs)
            Fs = np.stack([mp.source_interior(N, cx[i], cy[i], w[i], a[i])
                           for i in sel])
        else:
            cx, cy, w, a, _z = mp.sample_params(seed=seed, m=n_srcs)
            Fs = np.stack([mp.source_interior(N, cx[i], cy[i], w[i], a[i])
                           for i in range(n_srcs)])
        U_int = np.asarray(jax.lax.map(solve_truth, jnp.asarray(Fs)))
        res = float(np.max([np.linalg.norm(np.asarray(
            mp.neg_lap_interior(jnp.asarray(U_int[i]), N)) - Fs[i])
            / np.linalg.norm(Fs[i]) for i in range(n_srcs)]))
        sc.log(f"  truth[{name}]: {n_srcs} sources, FOM CG rel residual {res:.2e}")
        assert res < 1e-10, f"unconverged truth ({name})"
        U_int = U_int.reshape(n_srcs, -1)
        tn = np.array([np.linalg.norm(U_int[i]) for i in range(n_srcs)])
        return dict(name=name, Fs=Fs, U=U_int, tn=tn, n=n_srcs,
                    truth_residual=res)

    cohorts = [make_cohort("held_out_seed0", mp.N_TRAIN, n_srcs=N_TEST),
               make_cohort("fresh_seed", 0, seed=FRESH_SEED, n_srcs=N_TEST_FRESH)]
    report["cohorts"] = [dict(name=c["name"], n=c["n"],
                              truth_residual=c["truth_residual"])
                         for c in cohorts]

    # ------------------ train ------------------------------------------------
    params, Z_tr, tinfo = sc.train_autodecoder(
        jax.random.PRNGKey(SEED0), coords_int, U_tr, K, R,
        steps=STEPS, lr=LR, tag=f"poisson N={N} k={K} r={R}", **ARCH)
    report["train"] = tinfo
    dec = sc.SeparableDecoder(params, K, R)
    sc.save_pkl(CKPT, params, Z_tr, report["config"])
    save()

    # ------------- representation oracle (span-split fit, direct check) ------
    G_full = dec.feat_at(coords_int)                # (n_i^2, r) readout bank
    h_fn = dec.head_fn()
    u_full_fast = jax.jit(lambda z: G_full @ h_fn(z))
    zbar = Z_tr.mean(0)
    rng0 = np.random.default_rng(SEED0)
    o_inits = [zbar] + [Z_tr[i] for i in
                        rng0.choice(len(Z_tr), size=min(8, len(Z_tr)),
                                    replace=False)]
    oracle = sc.make_span_fitter(G_full, h_fn, K, o_inits, iters=150)
    for c in cohorts:
        om, om_est = [], []
        for i in range(min(4, c["n"])):
            u_t = jnp.asarray(c["U"][i])
            z_o, rel_est, _ = oracle(u_t)
            direct = float(jnp.linalg.norm(u_full_fast(z_o) - u_t) / c["tn"][i])
            om.append(direct); om_est.append(float(rel_est))
        report[f"oracle_{c['name']}"] = dict(
            mean=float(np.mean(om)), max=float(np.max(om)), n=len(om),
            rel_est_mean=float(np.mean(om_est)),
            split_vs_direct_max_dev=float(np.max(np.abs(np.array(om) -
                                                        np.array(om_est)))))
        sc.log(f"  oracle[{c['name']}] rel-L2: mean {np.mean(om):.3e} "
               f"max {np.max(om):.3e} (split-vs-direct dev "
               f"{np.max(np.abs(np.array(om) - np.array(om_est))):.1e})")
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

    # jitted weak source projection == pc.weak_source_term (verified below);
    # it charges the online source->f_m cost inside the timed pipe
    Sd = jnp.asarray(grid.S)
    lam_sel = jnp.asarray(np.asarray(grid.lam)[I, Jm])
    Ij, Jj = jnp.asarray(I), jnp.asarray(Jm)

    def fm_of_F(F2d):
        C = Sd.T @ F2d @ Sd
        return C[Ij, Jj] * lam_sel ** (-1.0)          # alpha = 1

    f_ms, dev_fm = [], 0.0
    for c in cohorts:
        c["F_dev"] = [jnp.asarray(c["Fs"][i]) for i in range(c["n"])]
        for i in range(c["n"]):
            ref = np.asarray(pc.weak_source_term(grid, spec, "grid", c["Fs"][i]))
            mine = np.asarray(fm_of_F(c["F_dev"][i]))
            dev_fm = max(dev_fm, float(np.max(np.abs(mine - ref))
                                       / (np.max(np.abs(ref)) + 1e-300)))
    report["fm_projection_max_rel_dev"] = dev_fm
    sc.log(f"  jitted weak_source_term vs incumbent: max rel dev {dev_fm:.2e}")
    assert dev_fm < 1e-12, "source projection deviates from incumbent"
    f_ms = [jnp.asarray(np.asarray(pc.weak_source_term(
        grid, spec, "grid", cohorts[0]["Fs"][i]))) for i in range(cohorts[0]["n"])]

    # ------------------ the two arms ----------------------------------------
    train_radius = float(np.max(np.linalg.norm(Z_tr - Z_tr.mean(0), axis=1)))
    trust = TR_FACTOR * train_radius if TR_FACTOR > 0 else np.inf
    z0 = jnp.asarray(Z_tr.mean(0))

    G_q = dec.feat_at(pts_np)                       # (m, r) cached bank
    dec_fast = lambda z, xy: G_q @ h_fn(z)          # ignores xy: nodes baked in

    arms = dict(meshfree=(dec, jax.jit(u_full)),
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

    # ------------- solves: same solver, both arms, all taus, both cohorts ----
    pipes = {}
    for arm, (dfn, ufull) in arms.items():
        lm, _ = ctol_tol.lm_tau_poisson(dfn, K, pts_np, wq, PhiT, Wl,
                                        GN_ITERS, trust_delta=trust)

        def pipe_fn(F2d, tau, _lm=lm, _uf=ufull):
            out = _lm(z0, fm_of_F(F2d), tau)
            return (_uf(out[0]),) + out[1:]
        pipes[arm] = jax.jit(pipe_fn)

    ctol_tol.burn_in(1.5)
    for c in cohorts:
        for arm in arms:
            pipe = pipes[arm]
            if c["name"] == "fresh_seed" and arm == "meshfree":
                continue          # gate 0 + same-seed rows already pin the arm
            for tau in TAUS:
                per_t, per_raw, per_err, per_jac, per_reason, per_dev = \
                    [], [], [], [], [], []
                for i in range(c["n"]):
                    Fi = c["F_dev"][i]
                    u, val, v0, nJ, acc, att, rsn = pipe(Fi, tau)
                    u_err = np.asarray(u)
                    med, ts = sc.time_fn(lambda _f=Fi:
                                         pipe(_f, tau)[0].block_until_ready(),
                                         reps=REPS)
                    u_timed = np.asarray(pipe(Fi, tau)[0])
                    per_dev.append(float(np.max(np.abs(u_timed - u_err))))
                    per_t.append(med)
                    per_raw.append([t * 1e3 for t in ts])
                    per_err.append(float(np.linalg.norm(u_err - c["U"][i])
                                         / c["tn"][i]))
                    per_jac.append(int(nJ)); per_reason.append(int(rsn))
                cens = [r_ not in ctol_tol.POISSON_TAU_OK for r_ in per_reason]
                rdist = {REASON_NAMES.get(r_, str(r_)): per_reason.count(r_)
                         for r_ in set(per_reason)}
                row = dict(pde="poisson2d", method=f"sep_{arm}", cohort=c["name"],
                           N=N, k=K, r=R, M=M_MODES, m=int(len(wq)), tau=tau,
                           time_ms=float(np.median(per_t)) * 1e3,
                           time_ms_all=[t * 1e3 for t in per_t],
                           time_ms_raw=per_raw,
                           timed_vs_error_max_dev=float(np.max(per_dev)),
                           err_rel_l2=float(np.mean(per_err)),
                           err_rel_l2_max=float(np.max(per_err)),
                           err_rel_l2_all=per_err,
                           jac_evals=float(np.mean(per_jac)),
                           jac_evals_all=per_jac,
                           censored_frac=float(np.mean(cens)),
                           stop_reasons=rdist,
                           n_sources=c["n"], trust_delta=trust)
                report["rows"].append(row)
                sc.log(f"   [{c['name']}] {arm:8s} tau={tau:.0e}  e2e "
                       f"{row['time_ms']:8.3f} ms  jac {row['jac_evals']:5.1f}  "
                       f"err {row['err_rel_l2']:.3e}  cens "
                       f"{row['censored_frac']*100:3.0f}%  reasons {rdist}")
                save()

    # ------------- FOM baselines: CG ladder + exact spectral, both cohorts ---
    def cg_at(tol):
        return jax.jit(lambda F, _t=tol: jax.scipy.sparse.linalg.cg(
            lambda v: mp.neg_lap_interior(v, N), F, tol=_t,
            maxiter=mp.CG_MAXITER)[0])

    lam_full = jnp.asarray(np.asarray(grid.lam))
    spectral = jax.jit(lambda F: (Sd @ ((Sd.T @ F @ Sd) / lam_full) @ Sd.T))

    report["fom"] = []
    for c in cohorts:
        for tol in sorted(set(FOM_LADDER), reverse=True):
            sol = cg_at(tol)
            errs, ts_med, ts_raw = [], [], []
            for i in range(c["n"]):
                Fi = c["F_dev"][i]
                u = np.asarray(sol(Fi)).reshape(-1)
                errs.append(float(np.linalg.norm(u - c["U"][i]) / c["tn"][i]))
                med, ts = sc.time_fn(lambda _F=Fi: sol(_F).block_until_ready(),
                                     reps=REPS)
                ts_med.append(med); ts_raw.append([t * 1e3 for t in ts])
            report["fom"].append(dict(
                cohort=c["name"], fom="cg", fom_tol=tol,
                time_ms=float(np.median(ts_med)) * 1e3,
                time_ms_all=[t * 1e3 for t in ts_med], time_ms_raw=ts_raw,
                err_rel_l2=float(np.mean(errs)),
                err_rel_l2_max=float(np.max(errs)), err_rel_l2_all=errs))
            sc.log(f"   [{c['name']}] FOM cg tol={tol:.0e}: "
                   f"{np.median(ts_med)*1e3:8.3f} ms  err {np.mean(errs):.3e}")
            save()
        errs, ts_med, ts_raw = [], [], []
        for i in range(c["n"]):
            Fi = c["F_dev"][i]
            u = np.asarray(spectral(Fi)).reshape(-1)
            errs.append(float(np.linalg.norm(u - c["U"][i]) / c["tn"][i]))
            med, ts = sc.time_fn(lambda _F=Fi: spectral(_F).block_until_ready(),
                                 reps=REPS)
            ts_med.append(med); ts_raw.append([t * 1e3 for t in ts])
        report["fom"].append(dict(
            cohort=c["name"], fom="spectral_dense", fom_tol=None,
            time_ms=float(np.median(ts_med)) * 1e3,
            time_ms_all=[t * 1e3 for t in ts_med], time_ms_raw=ts_raw,
            err_rel_l2=float(np.mean(errs)),
            err_rel_l2_max=float(np.max(errs)), err_rel_l2_all=errs))
        sc.log(f"   [{c['name']}] FOM spectral_dense: "
               f"{np.median(ts_med)*1e3:8.3f} ms  err {np.mean(errs):.3e}")
        save()

    # ------------- balanced AB/BA paired block: cached ROM vs baselines ------
    pipe_c = pipes["cached"]
    report["paired"] = []
    for c in cohorts:
        for tag, sol, tol in ([("cg", cg_at(t), t) for t in
                               sorted(set(FOM_LADDER), reverse=True)]
                              + [("spectral_dense", spectral, None)]):
            rows_p = []
            for i in range(c["n"]):
                Fi = c["F_dev"][i]
                pr = sc.time_pair(
                    lambda _f=Fi: pipe_c(_f, TAU_MAIN)[0].block_until_ready(),
                    lambda _f=Fi, _s=sol: _s(_f).block_until_ready(),
                    reps=PAIR_REPS)
                rows_p.append(pr)
            report["paired"].append(dict(
                cohort=c["name"], rom=f"sep_cached tau={TAU_MAIN}",
                baseline=tag, fom_tol=tol,
                rom_ms=float(np.median([r_["a_ms"] for r_ in rows_p])),
                base_ms=float(np.median([r_["b_ms"] for r_ in rows_p])),
                per_source=rows_p))
            sc.log(f"   [{c['name']}] paired ROM(tau={TAU_MAIN}) "
                   f"{report['paired'][-1]['rom_ms']:8.3f} ms vs {tag}"
                   f"{'' if tol is None else f' tol={tol:.0e}'} "
                   f"{report['paired'][-1]['base_ms']:8.3f} ms")
            save()

    report["complete"] = True
    report["total_seconds"] = time.time() - t_all
    save()
    sc.log(f"DONE poisson [{time.time()-t_all:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main()
