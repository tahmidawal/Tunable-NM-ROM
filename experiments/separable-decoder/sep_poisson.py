"""Separable-decoder Poisson-2D cell: train (no POD), weak NM-ROM solve.

Two solve arms through the SAME incumbent trust-LM weak solver
(ctol_tol.lm_tau_poisson):
  meshfree : dec(z, pts) evaluates the feature network inside the loop
  cached   : dec_fast(z, .) = G_q @ h(z), G_q = features at the EQ nodes,
             cached once -- no spatial network in the compiled iteration.
GATE 0: the two arms' weak residual/Jacobian must agree to <= 1e-12 relative.

MEASUREMENT RULES (HANDOFF 2026-08-23; each fixes an N=64 audit FAIL):
  - the timed ROM pipe includes the online source projection f -> f_m AND the
    full-grid decode readout, so it times what a user gets from a grid source;
  - errors and counters are extracted FROM a timed invocation (the last timed
    rep), and the max deviation between the first and last timed reps' outputs
    is recorded;
  - ALL raw timing repetitions are retained per (method, source, rep);
  - balanced ordering: the unit list (both ROM arms x taus + the whole FOM CG
    ladder) is executed REPS times, in reversed order on odd reps;
  - stop-reason distributions are recorded next to every error;
  - a fresh-seed test cohort (FRESH_SEED, never touched by training or model
    selection at any N) is evaluated alongside the seed-0 held-out cohort.
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
FRESH_SEED = int(os.environ.get("FRESH_SEED", "1"))
REPS = int(os.environ.get("REPS", "7"))
OUT = os.environ.get("OUT", "sep_poisson.json")
CKPT = os.environ.get("CKPT", f"sep_poisson_N{N}_K{K}_R{R}.pkl")
FOM_LADDER = [float(v) for v in os.environ.get(
    "FOM_LADDER", "1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,1e-6").split(",")]
ARCH = sc.arch_from_env()

REASON_NAMES = {0: "budget", 1: "stalled", 2: "tol", 3: "lambda_max",
                5: "nan_at_init"}


def main():
    dev = jax.devices()[0]
    sc.log(f"jax_backend={dev.platform} device={dev} x64={jax.config.jax_enable_x64} "
           f"N={N} K={K} R={R} M={M_MODES} m={MQ} steps={STEPS} seed={SEED0} "
           f"arch={ARCH}")
    t_all = time.time()
    report = dict(config=dict(
        pde="poisson2d", N=N, k=K, r=R, M=M_MODES, m=MQ, steps=STEPS, lr=LR,
        taus=TAUS, n_test=N_TEST, gn_iters=GN_ITERS, tr_factor=TR_FACTOR,
        seed=SEED0, data_seed=mp.SEED, fresh_seed=FRESH_SEED, reps=REPS,
        cg_tol=mp.CG_TOL, arch_cfg=ARCH,
        arch="separable: FourierFeat-MLP g(x)->R^r  x  MLP-head h(z)->R^r, "
             "hard poly BC; NO POD anywhere",
        objective=f"weak alpha=1 M={M_MODES}, NNLS-EQ m={MQ} grid nodes",
        solver="ctol_tol.lm_tau_poisson (incumbent trust-LM), both arms",
        timing="pipe = source projection + LM solve + full-grid decode; "
               "errors from the LAST TIMED rep; raw reps retained; balanced "
               "unit order (reversed on odd reps)",
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
    coords_int = np.asarray(grid.coords_int)

    solve_one = jax.jit(lambda F: jax.scipy.sparse.linalg.cg(
        lambda v: mp.neg_lap_interior(v, N), F, tol=mp.CG_TOL,
        maxiter=mp.CG_MAXITER)[0])

    def make_cohort(name, cxs, cys, ws, azs):
        Fs = np.stack([mp.source_interior(N, cxs[i], cys[i], ws[i], azs[i])
                       for i in range(N_TEST)])
        U_int = np.asarray(jax.lax.map(solve_one, jnp.asarray(Fs)))
        res = float(np.max([np.linalg.norm(np.asarray(
            mp.neg_lap_interior(jnp.asarray(U_int[i]), N)) - Fs[i])
            / np.linalg.norm(Fs[i]) for i in range(N_TEST)]))
        sc.log(f"  truth[{name}]: {N_TEST} test sources, FOM CG rel residual "
               f"{res:.2e}")
        assert res < 1e-10, "unconverged truth"
        U_int = U_int.reshape(N_TEST, -1)
        tn = np.array([np.linalg.norm(U_int[i]) for i in range(N_TEST)])
        return dict(name=name, Fs=Fs, U=U_int, tn=tn, truth_residual=res)

    # held-out cohort: same-seed draw, indices N_TRAIN.. (validation-style)
    cx, cy, w, a, _z = mp.sample_params()
    cohorts = [make_cohort("held_seed0",
                           cx[mp.N_TRAIN:], cy[mp.N_TRAIN:],
                           w[mp.N_TRAIN:], a[mp.N_TRAIN:])]
    # fresh-seed confirmation cohort: an entirely new draw, never used for any
    # training or model selection at any resolution
    cxf, cyf, wf, af, _zf = mp.sample_params(seed=FRESH_SEED, m=N_TEST)
    cohorts.append(make_cohort(f"fresh_seed{FRESH_SEED}", cxf, cyf, wf, af))
    report["cohorts"] = {c["name"]: dict(n=N_TEST,
                                         truth_residual=c["truth_residual"])
                         for c in cohorts}

    # ------------------ train ------------------------------------------------
    params, Z_tr, tinfo = sc.train_autodecoder(
        jax.random.PRNGKey(SEED0), coords_int, U_tr, K, R,
        steps=STEPS, lr=LR, tag=f"poisson N={N} k={K} r={R}", **ARCH)
    report["train"] = tinfo
    dec = sc.SeparableDecoder(params, K, R)
    sc.save_pkl(CKPT, params, Z_tr, report["config"])

    # held-out representation oracle on 4 fields per cohort (mean/max)
    zbar = Z_tr.mean(0)
    report["oracle_test_rel_l2"] = {}
    for c in cohorts:
        om = []
        for i in range(min(4, N_TEST)):
            _, val = sc.oracle_fit(dec, coords_int, c["U"][i], [zbar],
                                   budget=150)
            om.append(val)
        report["oracle_test_rel_l2"][c["name"]] = dict(
            mean=float(np.mean(om)), max=float(np.max(om)), n=len(om))
        sc.log(f"  test oracle rel-L2 [{c['name']}]: mean {np.mean(om):.3e} "
               f"max {np.max(om):.3e}")
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

    # online source projection matrix: f_m = Lambda^{-alpha} Phi_f^T f.  Built
    # once from the mode table (model-independent), charged INSIDE the timed
    # pipe -- the N=64 audit flagged it as an uncharged online ROM cost.
    lam_sel = np.asarray(grid.lam)[I, Jm]
    Psrc = jnp.asarray((Phi_f * (lam_sel ** -1.0)[None, :]).T)     # (M', n_i^2)
    f_ref = np.asarray(pc.weak_source_term(grid, spec, "grid",
                                           cohorts[0]["Fs"][0]))
    f_new = np.asarray(Psrc @ jnp.asarray(cohorts[0]["Fs"][0].reshape(-1)))
    psrc_dev = float(np.max(np.abs(f_new - f_ref))
                     / (np.max(np.abs(f_ref)) + 1e-300))
    report["src_projection_max_rel_dev"] = psrc_dev
    assert psrc_dev < 1e-12, "online source projection != weak_source_term"

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
    f_m0 = jnp.asarray(f_ref)

    def r_of(dfn, z, f_m):
        return jnp.asarray(Wl) * (jnp.asarray(PhiT) @
                                  (jnp.asarray(wq) * dfn(z, jnp.asarray(pts_np)))) - f_m
    g0 = []
    rng = np.random.default_rng(SEED0)
    for _ in range(5):
        zt = jnp.asarray(Z_tr[rng.integers(len(Z_tr))] +
                         0.05 * rng.standard_normal(K))
        ra = r_of(dec, zt, f_m0); rb = r_of(dec_fast, zt, f_m0)
        Ja = jax.jacfwd(lambda z: r_of(dec, z, f_m0))(zt)
        Jb = jax.jacfwd(lambda z: r_of(dec_fast, z, f_m0))(zt)
        g0.append(max(float(jnp.max(jnp.abs(ra - rb)) / (jnp.max(jnp.abs(ra)) + 1e-300)),
                      float(jnp.max(jnp.abs(Ja - Jb)) / (jnp.max(jnp.abs(Ja)) + 1e-300))))
    report["gate0_max_rel_dev"] = float(np.max(g0))
    sc.log(f"  GATE 0 (meshfree vs cached weak r/J identity): max rel dev {np.max(g0):.2e}")
    assert np.max(g0) < 1e-12, "gate 0 failed: cached arm is not the same discrete map"

    # ------------------ timed pipes: source -> f_m -> LM -> full decode -----
    pipes = {}
    for arm, (dfn, ufull) in arms.items():
        lm, _ = ctol_tol.lm_tau_poisson(dfn, K, pts_np, wq, PhiT, Wl,
                                        GN_ITERS, trust_delta=trust)

        def pipe_fn(F_vec, tau, _lm=lm, _uf=ufull):
            f_m = Psrc @ F_vec
            out = _lm(z0, f_m, tau)
            return (_uf(out[0]),) + out[1:]
        pipes[arm] = jax.jit(pipe_fn)

    # FOM CG ladder solvers (same job, same GPU, full interior field out)
    cg = {}
    for tol in sorted(set(FOM_LADDER), reverse=True):
        cg[tol] = jax.jit(lambda F, _t=tol: jax.scipy.sparse.linalg.cg(
            lambda v: mp.neg_lap_interior(v, N), F.reshape(n_i, n_i), tol=_t,
            maxiter=mp.CG_MAXITER)[0].reshape(-1))

    # timed units: every (arm, tau) and every ladder member.  One invocation =
    # one (source -> field) solve; outputs of the FIRST and LAST timed rep are
    # kept (error comes from the LAST timed rep, dev(first, last) is recorded).
    units = []
    for arm in ("meshfree", "cached"):
        for tau in TAUS:
            units.append((f"sep_{arm}", tau,
                          lambda Fv, _p=pipes[arm], _t=tau: _p(Fv, _t)))
    for tol in sorted(set(FOM_LADDER), reverse=True):
        units.append(("fom_cg", tol,
                      lambda Fv, _s=cg[tol]: (_s(Fv),)))

    for c in cohorts:
        Fv = [jnp.asarray(c["Fs"][i].reshape(-1)) for i in range(N_TEST)]
        # warm every unit on every source (compile + autotune)
        for _name, _p, fn in units:
            for i in range(N_TEST):
                out = fn(Fv[i]); out[0].block_until_ready()
        ctol_tol.burn_in(1.5)
        raw = {}      # (name, par) -> [source][rep] seconds
        first, last = {}, {}
        for rep in range(REPS):
            order = units if rep % 2 == 0 else list(reversed(units))
            for name, par, fn in order:
                key = (name, par)
                store = raw.setdefault(key, [[] for _ in range(N_TEST)])
                for i in range(N_TEST):
                    t0 = time.perf_counter()
                    out = fn(Fv[i])
                    out[0].block_until_ready()
                    store[i].append(time.perf_counter() - t0)
                    if rep == 0:
                        first[(key, i)] = np.asarray(out[0])
                    if rep == REPS - 1:
                        last[(key, i)] = [np.asarray(o) for o in out]

        for name, par, _fn in units:
            key = (name, par)
            per_med = [float(np.median(raw[key][i])) for i in range(N_TEST)]
            errs = [float(np.linalg.norm(last[(key, i)][0] - c["U"][i])
                          / c["tn"][i]) for i in range(N_TEST)]
            devs = [float(np.max(np.abs(last[(key, i)][0] - first[(key, i)])))
                    for i in range(N_TEST)]
            row = dict(pde="poisson2d", method=name, cohort=c["name"], N=N,
                       k=K, r=R, M=M_MODES, m=int(len(wq)),
                       time_ms=float(np.median(per_med)) * 1e3,
                       time_ms_raw=[[t * 1e3 for t in s] for s in raw[key]],
                       err_rel_l2=float(np.mean(errs)),
                       err_rel_l2_max=float(np.max(errs)),
                       err_rel_l2_all=errs,
                       dev_first_last_max=float(np.max(devs)),
                       n_sources=N_TEST, reps=REPS)
            if name.startswith("sep_"):
                jacs = [int(last[(key, i)][3]) for i in range(N_TEST)]
                rsn = [int(last[(key, i)][6]) for i in range(N_TEST)]
                cens = [r_ not in ctol_tol.POISSON_TAU_OK for r_ in rsn]
                row.update(tau=par, jac_evals=float(np.mean(jacs)),
                           jac_evals_all=jacs,
                           censored_frac=float(np.mean(cens)),
                           stop_reasons={REASON_NAMES.get(r_, str(r_)):
                                         rsn.count(r_) for r_ in set(rsn)},
                           trust_delta=trust)
                sc.log(f"   {c['name']:12s} {name:12s} tau={par:.0e}  "
                       f"{row['time_ms']:8.3f} ms  jac {row['jac_evals']:5.1f}  "
                       f"err {row['err_rel_l2']:.3e}  "
                       f"cens {row['censored_frac']*100:3.0f}%  "
                       f"dev {row['dev_first_last_max']:.1e}")
            else:
                row.update(fom_tol=par)
                sc.log(f"   {c['name']:12s} FOM CG tol={par:.0e}: "
                       f"{row['time_ms']:8.3f} ms  err {row['err_rel_l2']:.3e}  "
                       f"dev {row['dev_first_last_max']:.1e}")
            report["rows"].append(row)
            save()

    report["complete"] = True
    report["total_seconds"] = time.time() - t_all
    save()
    sc.log(f"DONE poisson [{time.time()-t_all:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main()
