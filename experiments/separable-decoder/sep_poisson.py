"""Separable-decoder Poisson-2D cell(s): train (no POD), weak NM-ROM solve.

N-scaling round (2026-08-23).  Differences from the N=64 first cell, each one
mandated by AUDIT-2026-08-23 / HANDOFF.md:
  * the timed ROM invocation now includes the source-to-f_m projection AND the
    full-grid decode, and the reported error is extracted from a timed
    invocation's own output (max deviation to the error-bearing call recorded);
  * raw timing repetitions are retained for every arm, with a balanced
    (alternating-order) schedule across ROM and CG arms (sep_common.time_multi);
  * per-source stopping-reason histograms are reported next to every error;
  * a FRESH-SEED test cohort arm (new parameter draw, truth solved in-job)
    confirms the same-seed held-out cohort;
  * several (K, R, arch) cells run in one job from ONE in-job data build
    (CELLS env), so the seed-regenerated data is shared, never cached on disk.

Two solve arms through the SAME incumbent trust-LM weak solver
(ctol_tol.lm_tau_poisson):
  meshfree : dec(z, pts) evaluates the feature network inside the loop
  cached   : dec_fast(z, .) = G_q @ h(z), G_q = features at the EQ nodes.
GATE 0 (asserted per cell): the two arms' weak residual/Jacobian agree to
<= 1e-12 relative.
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
# cells: "K:R[:NFF[:FFSCALE[:STEPS]]]" comma-separated
CELLS = os.environ.get("CELLS", "")
K_DEF = int(os.environ.get("K", "16"))
R_DEF = int(os.environ.get("R", str(4 * K_DEF)))
NFF = int(os.environ.get("NFF", "64"))
FF_SCALE = float(os.environ.get("FF_SCALE", "4.0"))
G_HID = int(os.environ.get("G_HID", "128"))
H_HID = int(os.environ.get("H_HID", "128"))
G_LAY = int(os.environ.get("G_LAY", "2"))
H_LAY = int(os.environ.get("H_LAY", "2"))
STEPS = int(os.environ.get("STEPS", "30000"))
LR = float(os.environ.get("LR", "1e-3"))
N_TEST = int(os.environ.get("N_TEST", "16"))
GN_ITERS = int(os.environ.get("GN_ITERS", "60"))
TAUS = [float(v) for v in os.environ.get("TAUS", "1e-3,1e-2").split(",")]
TR_FACTOR = float(os.environ.get("TR_FACTOR", "1.0"))
SEED0 = int(os.environ.get("SEED0", "0"))
FRESH_SEED = int(os.environ.get("FRESH_SEED", "20260823"))
OUT_DIR = os.environ.get("OUT_DIR", ".")
TIME_REPS = int(os.environ.get("TIME_REPS", "7"))
FOM_LADDER = [float(v) for v in os.environ.get(
    "FOM_LADDER", "1e-1,3e-2,1e-2,3e-3,1e-3,1e-6").split(",")]


def parse_cells():
    if not CELLS:
        return [dict(K=K_DEF, R=R_DEF, nff=NFF, ffs=FF_SCALE, steps=STEPS)]
    out = []
    for tok in CELLS.split(","):
        p = tok.split(":")
        k = int(p[0])
        out.append(dict(K=k, R=int(p[1]) if len(p) > 1 and p[1] else 4 * k,
                        nff=int(p[2]) if len(p) > 2 and p[2] else NFF,
                        ffs=float(p[3]) if len(p) > 3 and p[3] else FF_SCALE,
                        steps=int(p[4]) if len(p) > 4 and p[4] else STEPS))
    return out


def cg_truth(Fs, n, n_test):
    solve_one = jax.jit(lambda F: jax.scipy.sparse.linalg.cg(
        lambda v: mp.neg_lap_interior(v, n), F, tol=mp.CG_TOL,
        maxiter=mp.CG_MAXITER)[0])
    U_int = np.asarray(jax.lax.map(solve_one, jnp.asarray(Fs)))
    res = float(np.max([np.linalg.norm(np.asarray(
        mp.neg_lap_interior(jnp.asarray(U_int[i]), n)) - Fs[i])
        / np.linalg.norm(Fs[i]) for i in range(n_test)]))
    assert res < 1e-10, f"unconverged truth: {res:.2e}"
    return U_int.reshape(n_test, -1), res


REASON_NAMES = {0: "budget", 1: "converged", 2: "tau", 3: "lambda_max",
                5: "nan_at_init"}


def run_cell(cell, data, report_common):
    K, R = cell["K"], cell["R"]
    M_MODES = 4 * K
    MQ = 4 * M_MODES
    tag = f"K{K}_R{R}_nff{cell['nff']}_ffs{cell['ffs']:g}"
    OUT = os.path.join(OUT_DIR, f"sep_poisson_{tag}.json")
    CKPT = os.path.join(OUT_DIR, f"sep_poisson_N{N}_{tag}.pkl")
    t_cell = time.time()
    grid, U_tr, Fs, U_int, tn, Fs_fresh, U_fresh, tn_fresh, coords_int = data
    n_i = grid.n_i
    dev = jax.devices()[0]
    report = dict(config=dict(
        pde="poisson2d", N=N, k=K, r=R, M=M_MODES, m=MQ, steps=cell["steps"],
        lr=LR, n_ff=cell["nff"], ff_scale=cell["ffs"], g_hidden=G_HID,
        h_hidden=H_HID, g_layers=G_LAY, h_layers=H_LAY,
        taus=TAUS, n_test=N_TEST, gn_iters=GN_ITERS, tr_factor=TR_FACTOR,
        seed=SEED0, fresh_seed=FRESH_SEED, data_seed=mp.SEED, cg_tol=mp.CG_TOL,
        time_reps=TIME_REPS,
        arch="separable: FourierFeat-MLP g(x)->R^r  x  MLP-head h(z)->R^r, "
             "hard poly BC; NO POD anywhere",
        objective=f"weak alpha=1 M={M_MODES}, NNLS-EQ m={MQ} grid nodes",
        solver="ctol_tol.lm_tau_poisson (incumbent trust-LM), both arms",
        timing="pipe = f_m projection + LM solve + full-grid decode; balanced "
               "alternating-order schedule over all ROM and CG arms; raw reps "
               "retained; error extracted from a timed invocation's output",
        x64=True, matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local"),
        **report_common), rows=[], complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    # ------------------ train ------------------------------------------------
    params, Z_tr, tinfo = sc.train_autodecoder(
        jax.random.PRNGKey(SEED0), coords_int, U_tr, K, R,
        steps=cell["steps"], lr=LR, tag=f"poisson N={N} {tag}",
        n_ff=cell["nff"], ff_scale=cell["ffs"], g_hidden=G_HID,
        h_hidden=H_HID, g_layers=G_LAY, h_layers=H_LAY)
    report["train"] = tinfo
    dec = sc.SeparableDecoder(params, K, R)
    sc.save_pkl(CKPT, params, Z_tr, report["config"])

    # held-out representation oracle on 4 held-out + 4 fresh fields
    zbar = Z_tr.mean(0)
    om, omf = [], []
    for i in range(min(4, N_TEST)):
        om.append(sc.oracle_fit(dec, coords_int, U_int[i], [zbar], budget=150)[1])
        omf.append(sc.oracle_fit(dec, coords_int, U_fresh[i], [zbar], budget=150)[1])
    report["oracle_test_rel_l2"] = dict(mean=float(np.mean(om)),
                                        max=float(np.max(om)), n=len(om))
    report["oracle_fresh_rel_l2"] = dict(mean=float(np.mean(omf)),
                                         max=float(np.max(omf)), n=len(omf))
    sc.log(f"  oracle rel-L2: held-out mean {np.mean(om):.3e} max {np.max(om):.3e}"
           f" | fresh mean {np.mean(omf):.3e} max {np.max(omf):.3e}")
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

    # in-timed source projection f_m(F2d) = lam^-alpha * (S^T F S)[mask]
    lam_sel = jnp.asarray(np.asarray(grid.lam)[I, Jm])
    S_j = jnp.asarray(S_)
    I_j, Jm_j = jnp.asarray(I), jnp.asarray(Jm)

    def f_m_of(F2d):
        C = S_j.T @ F2d @ S_j
        return C[I_j, Jm_j] * lam_sel ** (-1.0)

    f_m_of_j = jax.jit(f_m_of)
    # prove equivalence to the incumbent weak_source_term, in-job
    dev_src = 0.0
    for i in range(N_TEST):
        a = np.asarray(f_m_of_j(jnp.asarray(Fs[i])))
        b = np.asarray(pc.weak_source_term(grid, spec, "grid", Fs[i]))
        dev_src = max(dev_src, float(np.max(np.abs(a - b))
                                     / (np.max(np.abs(b)) + 1e-300)))
    report["source_projection_max_rel_dev"] = dev_src
    assert dev_src < 1e-12, "in-timed source projection is not the incumbent map"

    f_ms = [f_m_of_j(jnp.asarray(Fs[i])) for i in range(N_TEST)]
    f_ms_fresh = [f_m_of_j(jnp.asarray(Fs_fresh[i])) for i in range(N_TEST)]

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

    # ------------------ error passes: same-invocation outputs ----------------
    # pipe = f_m projection + incumbent LM solve + full-grid decode (all timed
    # later THROUGH THE SAME jitted callable on the same inputs)
    pipes = {}
    for arm, (dfn, ufull) in arms.items():
        lm, _ = ctol_tol.lm_tau_poisson(dfn, K, pts_np, wq, PhiT, Wl,
                                        GN_ITERS, trust_delta=trust)

        def pipe_fn(F2d, tau, _lm=lm, _uf=ufull):
            out = _lm(z0, f_m_of(F2d), tau)
            return (_uf(out[0]),) + out[1:]
        pipes[arm] = jax.jit(pipe_fn)

    F2d_dev = [jnp.asarray(Fs[i]) for i in range(N_TEST)]
    F2d_fresh_dev = [jnp.asarray(Fs_fresh[i]) for i in range(N_TEST)]
    err_fields = {}          # (arm, tau) -> list of solved fields (numpy)
    for arm in arms:
        for tau in TAUS:
            per_err, per_jac, per_reason, fields = [], [], [], []
            for i in range(N_TEST):
                u, val, v0, nJ, acc, att, rsn = pipes[arm](F2d_dev[i], tau)
                u = np.asarray(u)
                fields.append(u)
                per_err.append(float(np.linalg.norm(u - U_int[i]) / tn[i]))
                per_jac.append(int(nJ)); per_reason.append(int(rsn))
            err_fields[(arm, tau)] = fields
            cens = [r_ not in ctol_tol.POISSON_TAU_OK for r_ in per_reason]
            hist = {}
            for r_ in per_reason:
                nm = REASON_NAMES.get(r_, str(r_))
                hist[nm] = hist.get(nm, 0) + 1
            row = dict(pde="poisson2d", method=f"sep_{arm}", cohort="heldout",
                       N=N, k=K, r=R, M=M_MODES, m=int(len(wq)), tau=tau,
                       err_rel_l2=float(np.mean(per_err)),
                       err_rel_l2_max=float(np.max(per_err)),
                       err_rel_l2_all=per_err,
                       jac_evals=float(np.mean(per_jac)), jac_evals_all=per_jac,
                       censored_frac=float(np.mean(cens)), stop_reasons=hist,
                       n_sources=N_TEST, trust_delta=trust)
            report["rows"].append(row)
            sc.log(f"   {arm:8s} tau={tau:.0e}  err {row['err_rel_l2']:.3e} "
                   f"(max {row['err_rel_l2_max']:.3e})  jac {row['jac_evals']:5.1f}  "
                   f"cens {row['censored_frac']*100:3.0f}%  reasons {hist}")
            save()

    # fresh-seed cohort (cached arm): same solver, same z0, new parameter draw
    for tau in TAUS:
        per_err, per_jac, per_reason = [], [], []
        for i in range(N_TEST):
            u, val, v0, nJ, acc, att, rsn = pipes["cached"](F2d_fresh_dev[i], tau)
            u = np.asarray(u)
            per_err.append(float(np.linalg.norm(u - U_fresh[i]) / tn_fresh[i]))
            per_jac.append(int(nJ)); per_reason.append(int(rsn))
        cens = [r_ not in ctol_tol.POISSON_TAU_OK for r_ in per_reason]
        hist = {}
        for r_ in per_reason:
            nm = REASON_NAMES.get(r_, str(r_))
            hist[nm] = hist.get(nm, 0) + 1
        row = dict(pde="poisson2d", method="sep_cached", cohort="fresh_seed",
                   N=N, k=K, r=R, M=M_MODES, m=int(len(wq)), tau=tau,
                   err_rel_l2=float(np.mean(per_err)),
                   err_rel_l2_max=float(np.max(per_err)),
                   err_rel_l2_all=per_err,
                   jac_evals=float(np.mean(per_jac)), jac_evals_all=per_jac,
                   censored_frac=float(np.mean(cens)), stop_reasons=hist,
                   n_sources=N_TEST, trust_delta=trust, fresh_seed=FRESH_SEED)
        report["rows"].append(row)
        sc.log(f"   cached/FRESH tau={tau:.0e}  err {row['err_rel_l2']:.3e} "
               f"(max {row['err_rel_l2_max']:.3e})  cens {row['censored_frac']*100:3.0f}%")
        save()

    # ------------------ FOM CG ladder: error pass (untimed) ------------------
    cg_solvers = {}
    fom_rows = []
    for tol in sorted(set(FOM_LADDER), reverse=True):
        s1 = jax.jit(lambda F, _t=tol: jax.scipy.sparse.linalg.cg(
            lambda v: mp.neg_lap_interior(v, N), F, tol=_t,
            maxiter=mp.CG_MAXITER)[0])
        cg_solvers[tol] = s1
        errs = []
        for i in range(N_TEST):
            u = np.asarray(s1(F2d_dev[i])).reshape(-1)
            errs.append(float(np.linalg.norm(u - U_int[i]) / tn[i]))
        fom_rows.append(dict(fom_tol=tol, err_rel_l2=float(np.mean(errs)),
                             err_rel_l2_max=float(np.max(errs)),
                             err_rel_l2_all=errs))
        sc.log(f"   FOM CG tol={tol:.0e}: err {np.mean(errs):.3e}")
    report["fom"] = fom_rows
    save()

    # ------------------ balanced timing block --------------------------------
    # thunks per source: every ROM (arm, tau) + every CG tol, alternating order
    timing = dict(reps=TIME_REPS, warm=2,
                  schedule="sep_common.time_multi alternating forward/reverse",
                  per_source={})
    timed_dev = {}          # (arm, tau) -> max |timed u - error-pass u|
    for i in range(N_TEST):
        ctol_tol.burn_in(1.0)
        last_out = {}
        thunks = {}
        for arm in arms:
            for tau in TAUS:
                nm = f"sep_{arm}_tau{tau:g}"
                def th(_a=arm, _t=tau, _i=i, _nm=nm):
                    u = pipes[_a](F2d_dev[_i], _t)[0]
                    u.block_until_ready()
                    last_out[_nm] = u
                thunks[nm] = th
        for tol in sorted(set(FOM_LADDER), reverse=True):
            nm = f"cg_tol{tol:g}"
            def th(_t=tol, _i=i):
                cg_solvers[_t](F2d_dev[_i]).block_until_ready()
            thunks[nm] = th
        raw, order = sc.time_multi(thunks, reps=TIME_REPS, warm=2)
        timing["per_source"][str(i)] = {nm: [t * 1e3 for t in ts]
                                        for nm, ts in raw.items()}
        if i == 0:
            timing["order_source0"] = order
        for arm in arms:
            for tau in TAUS:
                nm = f"sep_{arm}_tau{tau:g}"
                d = float(np.max(np.abs(np.asarray(last_out[nm])
                                        - err_fields[(arm, tau)][i])))
                key = (arm, tau)
                timed_dev[key] = max(timed_dev.get(key, 0.0), d)
    # summaries: per-method median of per-source medians
    meds = {}
    for nm in timing["per_source"]["0"]:
        per_src = [float(np.median(timing["per_source"][str(i)][nm]))
                   for i in range(N_TEST)]
        meds[nm] = dict(median_ms=float(np.median(per_src)),
                        per_source_median_ms=per_src)
    timing["summary"] = meds
    report["timing"] = timing
    # attach medians + timed-vs-error deviation to the error rows
    for row in report["rows"]:
        if row.get("cohort") != "heldout":
            continue
        nm = f"{row['method']}_tau{row['tau']:g}"
        if nm in meds:
            row["time_ms"] = meds[nm]["median_ms"]
            arm = row["method"].replace("sep_", "")
            row["timed_vs_err_max_abs_dev"] = timed_dev[(arm, row["tau"])]
    for fr in report["fom"]:
        nm = f"cg_tol{fr['fom_tol']:g}"
        if nm in meds:
            fr["time_ms"] = meds[nm]["median_ms"]
    for nm, d in meds.items():
        sc.log(f"   timed {nm:24s} {d['median_ms']:9.3f} ms")
    report["complete"] = True
    report["cell_seconds"] = time.time() - t_cell
    save()
    sc.log(f"CELL DONE poisson {tag} [{time.time()-t_cell:.0f}s] -> {OUT}")


def main():
    dev = jax.devices()[0]
    cells = parse_cells()
    sc.log(f"jax_backend={dev.platform} device={dev} x64={jax.config.jax_enable_x64} "
           f"N={N} cells={cells} seed={SEED0}")
    t_all = time.time()

    # ------------------ data (regenerated from seed, ONCE per job) -----------
    grid = pc.Grid(N)
    int_idx = np.asarray(grid.ix_full * N + grid.iy_full)
    U_all = np.asarray(mp.build_snapshots(N)[0])
    U_tr = U_all[:mp.N_TRAIN][:, int_idx]
    del U_all
    cx, cy, w, a, _z = mp.sample_params()
    Fs = np.stack([mp.source_interior(N, cx[mp.N_TRAIN + i], cy[mp.N_TRAIN + i],
                                      w[mp.N_TRAIN + i], a[mp.N_TRAIN + i])
                   for i in range(N_TEST)])
    U_int, res = cg_truth(Fs, N, N_TEST)
    sc.log(f"  truth: {N_TEST} held-out sources, FOM CG rel residual {res:.2e}")
    tn = np.array([np.linalg.norm(U_int[i]) for i in range(N_TEST)])
    # fresh-seed cohort: an entirely new parameter draw, truth solved in-job
    cxf, cyf, wf, af, _zf = mp.sample_params(seed=FRESH_SEED, m=N_TEST)
    Fs_fresh = np.stack([mp.source_interior(N, cxf[i], cyf[i], wf[i], af[i])
                         for i in range(N_TEST)])
    U_fresh, res_f = cg_truth(Fs_fresh, N, N_TEST)
    sc.log(f"  truth: {N_TEST} fresh-seed sources (seed {FRESH_SEED}), "
           f"CG rel residual {res_f:.2e}")
    tn_fresh = np.array([np.linalg.norm(U_fresh[i]) for i in range(N_TEST)])
    coords_int = np.asarray(grid.coords_int)
    data = (grid, U_tr, Fs, U_int, tn, Fs_fresh, U_fresh, tn_fresh, coords_int)
    report_common = dict(truth_res=res, truth_res_fresh=res_f)

    for cell in cells:
        run_cell(cell, data, report_common)
    sc.log(f"ALL CELLS DONE poisson [{time.time()-t_all:.0f}s]")


if __name__ == "__main__":
    main()
