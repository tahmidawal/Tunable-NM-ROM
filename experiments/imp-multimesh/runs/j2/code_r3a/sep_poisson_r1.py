"""N=256 push, ROUND 1, Poisson cell (PUSH-PLAN.md).

One decoder (K16/R64/200k by default -- the verified Pareto cell), then the
round-1 lever matrix over the SAME incumbent weak objective:

  EQ sets   : ctrl (M=4K, m=16K, incumbent NNLS fit -- asserted bit-equal to
              ctol_eq.eq_fit_poisson), m512 (m=32K), M128 (M=8K, m=32K),
              tail (ctrl system, tail-reweighted NNLS -- sep_solvers)
  solvers   : lm60 (incumbent budget), lm300 (budget only),
              rst300 (budget + restart-on-stall)  [sep_solvers; rst300 with
              restarts disabled is asserted to reproduce the incumbent]
  inits     : zbar (incumbent), enc (offline f_m -> z encoder, TRAINING data
              only, applied INSIDE the timed pipe)
  adq       : per-query adaptive quadrature on the fresh cohort (cheap solve
              -> extend EQ system with rows at z* -> refit -> re-solve),
              timed end-to-end including the host NNLS refit

plus the full error ladder per cohort: train recon / representation oracle
(all test fields, multi-init, incl. each arm's solve-z) / weak-EQ optimum
(rst-LM from the oracle latent, tau=0, DIAGNOSTIC: oracle init uses truth) /
solver output.

All N=64-audit measurement rules stay in force: source->f_m projection inside
the timed pipe, full-field decode inside the pipe, balanced AB/BA sweeps, raw
reps retained, error from captured timed invocations, both cohorts, stop
reasons + censoring per row, same-job CG ladder.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import sep_common as sc
import sep_solvers as ss

import jax
import jax.numpy as jnp

import pro_common as pc                      # noqa: E402  (path set by sc)
from pro_common import mp                    # noqa: E402
import ctol_eq                                # noqa: E402
import ctol_tol                               # noqa: E402

F64 = jnp.float64

N = int(os.environ.get("N", "64"))
K = int(os.environ.get("K", "16"))
R = int(os.environ.get("R", str(4 * K)))
STEPS = int(os.environ.get("STEPS", "30000"))
LR = float(os.environ.get("LR", "1e-3"))
N_TEST = int(os.environ.get("N_TEST", "16"))
GN_ITERS = int(os.environ.get("GN_ITERS", "60"))
BUDGET_HI = int(os.environ.get("BUDGET_HI", "300"))
N_RESTARTS = int(os.environ.get("N_RESTARTS", "6"))
RESTART_SIG = float(os.environ.get("RESTART_SIG", "0.3"))
TAUS = [float(v) for v in os.environ.get("TAUS", "1e-3,1e-2").split(",")]
TR_FACTOR = float(os.environ.get("TR_FACTOR", "1.0"))
SEED0 = int(os.environ.get("SEED0", "0"))
FRESH_SEED = int(os.environ.get("FRESH_SEED", "777"))
REPS = int(os.environ.get("REPS", "9"))
WARM = int(os.environ.get("WARM", "2"))
OUT = os.environ.get("OUT", "sep_poisson_r1.json")
CKPT = os.environ.get("CKPT", f"sep_poisson_r1_N{N}_K{K}_R{R}.pkl")
FOM_LADDER = [float(v) for v in os.environ.get(
    "FOM_LADDER", "5e-1,3e-1,1e-1,3e-2,1e-2,1e-3,1e-6").split(",")]
M_CTRL = int(os.environ.get("M_CTRL", str(4 * K)))
MQ_CTRL = int(os.environ.get("MQ_CTRL", str(16 * K)))
M_BIG = int(os.environ.get("M_BIG", str(8 * K)))
MQ_BIG = int(os.environ.get("MQ_BIG", str(32 * K)))
TAIL_CAP = float(os.environ.get("TAIL_CAP", "3e-2"))
TAIL_ROUNDS = int(os.environ.get("TAIL_ROUNDS", "3"))
ENC_STEPS = int(os.environ.get("ENC_STEPS", "8000"))
ORACLE_BUDGET = int(os.environ.get("ORACLE_BUDGET", "250"))
WOPT_BUDGET = int(os.environ.get("WOPT_BUDGET", "600"))
ADQ_N = int(os.environ.get("ADQ_N", "8"))
ADQ_PERT = int(os.environ.get("ADQ_PERT", "3"))
ARCH = sc.arch_from_env()


def eq_fit_ext(u_cand, u_full, Phi_c, Phi_f, Z_snap, k_lat, m, label):
    """ctol_eq.eq_fit_poisson with the SAME rng stream / snapshot rule /
    _solve_nnls tail, additionally returning the raw system (G, b, pad_score)
    for the tail-reweight and adaptive-quadrature arms.  Bit-equality with the
    reference is asserted by the caller for the control set."""
    t0 = time.time()
    r_eq = np.random.default_rng(ctol_eq.EQ_SEED)
    n_tr = Z_snap.shape[0]
    idx = r_eq.choice(n_tr, size=min(ctol_eq.EQ_SNAPS, n_tr), replace=False)
    pert = 0.05 * np.asarray(Z_snap, dtype=np.float64).std(axis=0)
    snaps, fulls = [], []
    for i in idx:
        z = jnp.asarray(Z_snap[i])
        for zz in [z] + [z + jnp.asarray(pert * r_eq.standard_normal(k_lat))
                         for _ in range(ctol_eq.EQ_PERTURB)]:
            snaps.append(np.asarray(u_cand(zz)))
            fulls.append(np.asarray(u_full(zz)))
    Rm = np.stack(snaps)
    Rf = np.stack(fulls)
    b = (Rf @ Phi_f).reshape(-1)
    G = np.einsum("sp,pm->smp", Rm, Phi_c).reshape(-1, Rm.shape[1])
    del Rf
    pad_score = np.abs(Rm).mean(0)
    keep, wq, info = ctol_eq._solve_nnls(
        G.copy(), b.copy(), m, r_eq, pc.nnls_capped, label,
        dict(kind="weak_poisson"), t0, pad_score=pad_score)
    return keep, wq, info, G, b, pad_score, pert


def main():
    dev = jax.devices()[0]
    sc.log(f"jax_backend={dev.platform} device={dev} x64={jax.config.jax_enable_x64} "
           f"ROUND1 N={N} K={K} R={R} steps={STEPS} seed={SEED0} arch={ARCH or 'default'}")
    t_all = time.time()
    report = dict(config=dict(
        pde="poisson2d", round=1, N=N, k=K, r=R, steps=STEPS, lr=LR,
        M_ctrl=M_CTRL, m_ctrl=MQ_CTRL, M_big=M_BIG, m_big=MQ_BIG,
        taus=TAUS, n_test=N_TEST, gn_iters=GN_ITERS, budget_hi=BUDGET_HI,
        n_restarts=N_RESTARTS, restart_sig=RESTART_SIG,
        tail_cap=TAIL_CAP, tail_rounds=TAIL_ROUNDS, enc_steps=ENC_STEPS,
        oracle_budget=ORACLE_BUDGET, wopt_budget=WOPT_BUDGET,
        adq_n=ADQ_N, adq_pert=ADQ_PERT, tr_factor=TR_FACTOR,
        seed=SEED0, fresh_seed=FRESH_SEED, data_seed=mp.SEED, cg_tol=mp.CG_TOL,
        reps=REPS, warm=WARM, arch_overrides=ARCH,
        x64=True, matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local")),
        rows=[], gates={}, complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    # ------------------ data (regenerated from seed) -------------------------
    grid = pc.Grid(N)
    n_i = grid.n_i
    int_idx = np.asarray(grid.ix_full * N + grid.iy_full)
    U_all = np.asarray(mp.build_snapshots(N)[0])
    U_tr = U_all[:mp.N_TRAIN][:, int_idx]
    cx, cy, w, a, _z = mp.sample_params()
    Fs_held = np.stack([mp.source_interior(N, cx[mp.N_TRAIN + i], cy[mp.N_TRAIN + i],
                                           w[mp.N_TRAIN + i], a[mp.N_TRAIN + i])
                        for i in range(N_TEST)])
    cxf, cyf, wf, af, _zf = mp.sample_params(seed=FRESH_SEED, m=N_TEST)
    Fs_fresh = np.stack([mp.source_interior(N, cxf[i], cyf[i], wf[i], af[i])
                         for i in range(N_TEST)])
    Fs = np.concatenate([Fs_held, Fs_fresh])
    cohort_of = (["heldout_seed0"] * N_TEST + [f"fresh_seed{FRESH_SEED}"] * N_TEST)
    n_src = Fs.shape[0]
    solve_one = jax.jit(lambda F: jax.scipy.sparse.linalg.cg(
        lambda v: mp.neg_lap_interior(v, N), F, tol=mp.CG_TOL,
        maxiter=mp.CG_MAXITER)[0])
    U_int = np.asarray(jax.lax.map(solve_one, jnp.asarray(Fs)))
    res = float(np.max([np.linalg.norm(np.asarray(
        mp.neg_lap_interior(jnp.asarray(U_int[i]), N)) - Fs[i])
        / np.linalg.norm(Fs[i]) for i in range(n_src)]))
    sc.log(f"  truth: {n_src} test sources (2 cohorts), FOM CG rel residual {res:.2e}")
    assert res < 1e-10, "unconverged truth"
    U_int = U_int.reshape(n_src, -1)
    tn = np.array([np.linalg.norm(U_int[i]) for i in range(n_src)])
    coords_int = np.asarray(grid.coords_int)
    # training sources (for the encoder features; TRAINING data only)
    Fs_tr = np.stack([mp.source_interior(N, cx[i], cy[i], w[i], a[i])
                      for i in range(mp.N_TRAIN)])

    # ------------------ train ------------------------------------------------
    params, Z_tr, tinfo = sc.train_autodecoder(
        jax.random.PRNGKey(SEED0), coords_int, U_tr, K, R,
        steps=STEPS, lr=LR, tag=f"poisson r1 N={N} k={K} r={R}", **ARCH)
    report["train"] = tinfo
    dec = sc.SeparableDecoder(params, K, R)
    sc.save_pkl(CKPT, params, Z_tr, report["config"])
    save()

    h_fn = dec.head_fn()
    G_full = dec.feat_at(coords_int)                    # (n_i^2, r) readout
    u_full_fast = jax.jit(lambda z: G_full @ h_fn(z))
    train_radius = float(np.max(np.linalg.norm(Z_tr - Z_tr.mean(0), axis=1)))
    trust = TR_FACTOR * train_radius if TR_FACTOR > 0 else np.inf
    zbar = jnp.asarray(Z_tr.mean(0))
    z_std = np.asarray(Z_tr, dtype=np.float64).std(axis=0)

    # ------------------ EQ sets ---------------------------------------------
    cand_pos = ctol_eq.candidate_pool(n_i * n_i)
    cand_xy = coords_int[cand_pos]
    G_cand = dec.feat_at(cand_xy)                       # cached bank at pool
    u_cand_mesh = jax.jit(lambda z: dec(z, jnp.asarray(cand_xy)))
    u_full_mesh = jax.jit(lambda z: dec(z, jnp.asarray(coords_int)))
    # cached == meshfree identity on the whole candidate pool (feeds every EQ
    # set and the adq refits)
    rng0 = np.random.default_rng(SEED0)
    zt = jnp.asarray(Z_tr[rng0.integers(len(Z_tr))] + 0.05 * rng0.standard_normal(K))
    pool_dev = float(jnp.max(jnp.abs(G_cand @ h_fn(zt) - u_cand_mesh(zt)))
                     / (jnp.max(jnp.abs(u_cand_mesh(zt))) + 1e-300))
    report["gates"]["pool_cached_vs_meshfree"] = pool_dev
    assert pool_dev < 1e-12, "cached bank differs from meshfree on the pool"

    def mode_tables(M):
        mask = np.asarray(grid.mode_mask(M)).astype(bool)
        I, Jm = np.nonzero(mask)
        S_ = np.asarray(grid.S)
        Phi_f = S_[grid.ix_full - 1][:, I] * S_[grid.iy_full - 1][:, Jm]
        S_j = jnp.asarray(grid.S)
        I_j, Jm_j = jnp.asarray(I), jnp.asarray(Jm)
        Wsrc = jnp.asarray(np.asarray(grid.lam)[I, Jm] ** (-1.0))

        def f_m_of(F2d):
            C = S_j.T @ F2d @ S_j
            return C[I_j, Jm_j] * Wsrc
        return I, Jm, Phi_f, jax.jit(f_m_of)

    Itab = {M: mode_tables(M) for M in sorted({M_CTRL, M_BIG})}

    eq_sets = {}

    def finish_eq(name, M, keep, wq, info, extra=None):
        I, Jm, Phi_f, f_m_of = Itab[M]
        node_pos = cand_pos[keep]
        pts_np = coords_int[node_pos]
        spec = dict(kind="weak", alpha=1.0, M=M)
        PhiT, Wl = pc.colloc_mode_table(grid, spec, "grid", pts_np)
        f_ms = [jnp.asarray(np.asarray(pc.weak_source_term(grid, spec, "grid", Fs[i])))
                for i in range(n_src)]
        fm_dev = max(float(jnp.max(jnp.abs(f_m_of(jnp.asarray(Fs[i])) - f_ms[i]))
                           / (float(jnp.max(jnp.abs(f_ms[i]))) + 1e-300))
                     for i in (0, N_TEST, n_src - 1))
        assert fm_dev < 1e-12, f"f_m projection mismatch for {name}"
        G_q = G_cand[jnp.asarray(keep)]
        # GATE 0 on THIS node set: meshfree vs cached weak residual/Jacobian
        def r_mesh(z, f_m):
            return jnp.asarray(Wl) * (jnp.asarray(PhiT) @
                                      (jnp.asarray(wq) * dec(z, jnp.asarray(pts_np)))) - f_m
        def r_cach(z, f_m):
            return jnp.asarray(Wl) * (jnp.asarray(PhiT) @
                                      (jnp.asarray(wq) * (G_q @ h_fn(z)))) - f_m
        g0 = []
        rng = np.random.default_rng(SEED0)
        for _ in range(3):
            zt = jnp.asarray(Z_tr[rng.integers(len(Z_tr))] + 0.05 * rng.standard_normal(K))
            ra, rb = r_mesh(zt, f_ms[0]), r_cach(zt, f_ms[0])
            Ja = jax.jacfwd(lambda z: r_mesh(z, f_ms[0]))(zt)
            Jb = jax.jacfwd(lambda z: r_cach(z, f_ms[0]))(zt)
            g0.append(max(float(jnp.max(jnp.abs(ra - rb)) / (jnp.max(jnp.abs(ra)) + 1e-300)),
                          float(jnp.max(jnp.abs(Ja - Jb)) / (jnp.max(jnp.abs(Ja)) + 1e-300))))
        gate0 = float(np.max(g0))
        sc.log(f"  GATE 0 [{name}]: max rel dev {gate0:.2e}  (fm dev {fm_dev:.2e})")
        assert gate0 < 1e-12, f"gate 0 failed for EQ set {name}"
        eq_sets[name] = dict(M=M, m=int(len(keep)), keep=np.asarray(keep),
                             wq=jnp.asarray(wq), PhiT=jnp.asarray(PhiT),
                             Wl=jnp.asarray(Wl), G_q=G_q, pts_np=pts_np,
                             f_ms=f_ms, f_m_of=f_m_of, info=info, gate0=gate0,
                             fm_dev=fm_dev, Phi_f=Phi_f)
        eq_rep = {k_: v for k_, v in info.items()
                  if isinstance(v, (int, float, str, bool, type(None)))}
        eq_rep.update(M=int(M), m=int(len(keep)), gate0=gate0, fm_dev=fm_dev)
        report.setdefault("eq", {})[name] = eq_rep
        if extra:
            report["eq"][name].update(extra)
        save()

    # control set: my ext fit must reproduce ctol_eq.eq_fit_poisson bit-for-bit
    I0, Jm0, Phi_f0, _ = Itab[M_CTRL]
    keep_c, wq_c, info_c, G_c, b_c, pad_c, pert_c = eq_fit_ext(
        u_cand_mesh, u_full_mesh, Phi_f0[cand_pos], Phi_f0, Z_tr, K, MQ_CTRL,
        f"r1 ctrl N={N} M={M_CTRL} m={MQ_CTRL}")
    keep_ref, wq_ref, info_ref = ctol_eq.eq_fit_poisson(
        u_cand_mesh, u_full_mesh, Phi_f0[cand_pos], Phi_f0, Z_tr, K, MQ_CTRL,
        "reference re-fit", pc.nnls_capped)
    assert np.array_equal(np.asarray(keep_c), np.asarray(keep_ref)), \
        "ext EQ fit selected different nodes than the incumbent"
    wq_dev = float(np.max(np.abs(wq_c - wq_ref)) / (np.max(np.abs(wq_ref)) + 1e-300))
    report["gates"]["eq_ext_vs_incumbent_wq"] = wq_dev
    assert wq_dev < 1e-12, "ext EQ weights differ from the incumbent fit"
    finish_eq("ctrl", M_CTRL, keep_c, wq_c, info_c)

    keep_m5, wq_m5, info_m5, _, _, _, _ = eq_fit_ext(
        u_cand_mesh, u_full_mesh, Phi_f0[cand_pos], Phi_f0, Z_tr, K, MQ_BIG,
        f"r1 m512 N={N} M={M_CTRL} m={MQ_BIG}")
    finish_eq("m512", M_CTRL, keep_m5, wq_m5, info_m5)

    I1, Jm1, Phi_f1, _ = Itab[M_BIG]
    keep_M1, wq_M1, info_M1, _, _, _, _ = eq_fit_ext(
        u_cand_mesh, u_full_mesh, Phi_f1[cand_pos], Phi_f1, Z_tr, K, MQ_BIG,
        f"r1 M128 N={N} M={M_BIG} m={MQ_BIG}")
    finish_eq("M128", M_BIG, keep_M1, wq_M1, info_M1)

    keep_t, wq_t, info_t = ss.tail_reweight_fit(
        G_c, b_c, MQ_CTRL, pc.nnls_capped, seed=ctol_eq.EQ_SEED + 1,
        cap=TAIL_CAP, rounds=TAIL_ROUNDS, pad_score=pad_c,
        label=f"r1 tail N={N} M={M_CTRL} m={MQ_CTRL}")
    finish_eq("tail", M_CTRL, keep_t, wq_t, info_t)

    # ------------------ encoder (TRAINING data only) -------------------------
    fm_of_ctrl = eq_sets["ctrl"]["f_m_of"]
    X_tr = np.stack([np.asarray(fm_of_ctrl(jnp.asarray(Fs_tr[i])))
                     for i in range(mp.N_TRAIN)])
    enc_params, enc_apply, enc_info = ss.fit_code_encoder(
        jax.random.PRNGKey(SEED0 + 7), X_tr, Z_tr, steps=ENC_STEPS,
        tag=f"poisson f_m->z N={N}")
    report["encoder"] = enc_info

    # ------------------ solvers + agreement gates ---------------------------
    noise = jnp.asarray(RESTART_SIG * z_std * np.random.default_rng(SEED0 + 123)
                        .standard_normal((N_RESTARTS, K)))
    lm60 = ss.lm_tau_cached_restart(h_fn, K, GN_ITERS, None)
    lm300 = ss.lm_tau_cached_restart(h_fn, K, BUDGET_HI, None)
    rst300 = ss.lm_tau_cached_restart(h_fn, K, BUDGET_HI, noise)
    wopt_lm = ss.lm_tau_cached_restart(h_fn, K, WOPT_BUDGET, noise)

    e0 = eq_sets["ctrl"]
    dec_fast_ctrl = lambda z, xy: e0["G_q"] @ h_fn(z)
    lm_inc, _ = ctol_tol.lm_tau_poisson(dec_fast_ctrl, K, e0["pts_np"],
                                        np.asarray(e0["wq"]),
                                        np.asarray(e0["PhiT"]),
                                        np.asarray(e0["Wl"]),
                                        GN_ITERS, trust_delta=trust)
    lm_mesh, _ = ctol_tol.lm_tau_poisson(dec, K, e0["pts_np"],
                                         np.asarray(e0["wq"]),
                                         np.asarray(e0["PhiT"]),
                                         np.asarray(e0["Wl"]),
                                         GN_ITERS, trust_delta=trust)
    out_inc = lm_inc(zbar, e0["f_ms"][0], TAUS[0])
    out_new = lm60(zbar, e0["G_q"], e0["wq"], e0["PhiT"], e0["Wl"],
                   e0["f_ms"][0], TAUS[0], trust)
    sdev = float(np.linalg.norm(np.asarray(out_new[0]) - np.asarray(out_inc[0]))
                 / (1.0 + np.linalg.norm(np.asarray(out_inc[0]))))
    report["gates"]["solver_norestart_vs_incumbent"] = dict(
        rel_dz=sdev, inc_reason=int(out_inc[6]), new_reason=int(out_new[6]),
        inc_nj=int(out_inc[3]), new_nj=int(out_new[3]))
    sc.log(f"  SOLVER GATE (repairs disabled vs incumbent): rel|dz| {sdev:.2e} "
           f"reasons {int(out_inc[6])}/{int(out_new[6])}")
    assert sdev < 1e-10, "repaired solver with repairs off != incumbent"

    # ------------------ timed pipes -----------------------------------------
    # uniform captured schema: (u, z, val, v0, nJ, att, reason, rs)
    def make_pipe(eq, lm, init_kind):
        G_q, wq, PhiT, Wl = eq["G_q"], eq["wq"], eq["PhiT"], eq["Wl"]
        f_m_of = eq["f_m_of"]

        def pipe(F2d, tau):
            f_m = f_m_of(F2d)
            if init_kind == "enc":
                z0 = enc_apply(enc_params, f_m)
            else:
                z0 = zbar
            z, val, v0, nJ, acc, att, reason, rs = lm(
                z0, G_q, wq, PhiT, Wl, f_m, tau, trust)
            return G_full @ h_fn(z), z, val, v0, nJ, att, reason, rs
        return jax.jit(pipe)

    def mesh_pipe(F2d, tau):
        f_m = e0["f_m_of"](F2d)
        z, val, v0, nJ, acc, att, reason = lm_mesh(zbar, f_m, tau)
        return (dec(z, jnp.asarray(coords_int)), z, val, v0, nJ, att, reason,
                jnp.int32(0))

    arms = {
        "mesh|ctrl|lm60|zbar": jax.jit(mesh_pipe),
        "cach|ctrl|lm60|zbar": make_pipe(e0, lm60, "zbar"),
        "cach|ctrl|lm300|zbar": make_pipe(e0, lm300, "zbar"),
        "cach|ctrl|rst300|zbar": make_pipe(e0, rst300, "zbar"),
        "cach|ctrl|lm60|enc": make_pipe(e0, lm60, "enc"),
        "cach|ctrl|rst300|enc": make_pipe(e0, rst300, "enc"),
        "cach|m512|rst300|zbar": make_pipe(eq_sets["m512"], rst300, "zbar"),
        "cach|M128|rst300|zbar": make_pipe(eq_sets["M128"], rst300, "zbar"),
        "cach|tail|rst300|zbar": make_pipe(eq_sets["tail"], rst300, "zbar"),
    }
    cg_solvers = {tol: jax.jit(lambda F, _t=tol: jax.scipy.sparse.linalg.cg(
        lambda v: mp.neg_lap_interior(v, N), F, tol=_t,
        maxiter=mp.CG_MAXITER)[0]) for tol in sorted(set(FOM_LADDER), reverse=True)}

    acc_t = {}
    ctol_tol.burn_in(1.5)
    for i in range(n_src):
        Fi = jnp.asarray(Fs[i])
        subs = []
        for aname, p in arms.items():
            for tau in TAUS:
                def fn(_p=p, _F=Fi, _tau=tau):
                    out = _p(_F, _tau)
                    out[0].block_until_ready()
                    return out
                subs.append((f"{aname}|{tau:.0e}", fn))
        for tol, s1 in cg_solvers.items():
            def fn(_s=s1, _F=Fi):
                u = _s(_F)
                u.block_until_ready()
                return u
            subs.append((f"fom_cg|{tol:.0e}", fn))
        raw, results = sc.balanced_time(subs, reps=REPS, warm=WARM)
        for name in raw:
            acc_t.setdefault(name, []).append((raw[name], results[name]))
        if i == 0:
            sc.log(f"   timing block: {len(subs)} subjects x {REPS} reps "
                   f"(+{WARM} warm), AB/BA")

    cohorts = ["heldout_seed0", f"fresh_seed{FRESH_SEED}"]
    solve_z = {}                # arm name -> (n_src, K) solved latents (tau[0])
    for aname in arms:
        for tau in TAUS:
            name = f"{aname}|{tau:.0e}"
            zs = []
            for cname in cohorts:
                idxs = [i for i in range(n_src) if cohort_of[i] == cname]
                per_t, per_err, per_jac, per_reason, per_rs, per_val, raw_all = \
                    [], [], [], [], [], [], {}
                for i in idxs:
                    times, out = acc_t[name][i]
                    u = np.asarray(out[0]).reshape(-1)
                    per_err.append(float(np.linalg.norm(u - U_int[i]) / tn[i]))
                    per_jac.append(int(out[4]))
                    per_reason.append(int(out[6]))
                    per_rs.append(int(out[7]))
                    per_val.append(float(out[2] / (out[3] + 1e-300)))
                    per_t.append(float(np.median(times)))
                    raw_all[str(i)] = [float(t) for t in times]
                    if tau == TAUS[0]:
                        zs.append(np.asarray(out[1]))
                cens = [r_ != 2 for r_ in per_reason]
                reasons_hist = {str(r_): per_reason.count(r_) for r_ in set(per_reason)}
                mesh, eqn, sol, ini = aname.split("|")
                row = dict(pde="poisson2d", method=aname, arm=mesh, eq_set=eqn,
                           solver=sol, init=ini, N=N, k=K, r=R,
                           M=eq_sets.get(eqn, e0)["M"] if eqn in eq_sets else M_CTRL,
                           m=eq_sets.get(eqn, e0)["m"] if eqn in eq_sets else MQ_CTRL,
                           tau=tau, cohort=cname,
                           time_ms=float(np.median(per_t)) * 1e3,
                           time_ms_all=[t * 1e3 for t in per_t],
                           time_raw_s=raw_all,
                           err_rel_l2=float(np.mean(per_err)),
                           err_rel_l2_max=float(np.max(per_err)),
                           obj_rel_mean=float(np.mean(per_val)),
                           jac_evals=float(np.mean(per_jac)),
                           restarts_mean=float(np.mean(per_rs)),
                           censored_frac=float(np.mean(cens)),
                           stop_reasons=reasons_hist,
                           n_sources=len(idxs), trust_delta=trust)
                report["rows"].append(row)
                sc.log(f"   {aname:24s} tau={tau:.0e} [{cname:14s}] "
                       f"{row['time_ms']:8.3f} ms  jac {row['jac_evals']:5.1f}  "
                       f"err {row['err_rel_l2']:.3e}  obj {row['obj_rel_mean']:.2e}  "
                       f"cens {row['censored_frac']*100:3.0f}%")
            if tau == TAUS[0]:
                solve_z[aname] = np.stack(zs)
            save()

    for tol in cg_solvers:
        name = f"fom_cg|{tol:.0e}"
        for cname in cohorts:
            idxs = [i for i in range(n_src) if cohort_of[i] == cname]
            errs, ts, raw_all = [], [], {}
            for i in idxs:
                times, u = acc_t[name][i]
                u = np.asarray(u).reshape(-1)
                errs.append(float(np.linalg.norm(u - U_int[i]) / tn[i]))
                ts.append(float(np.median(times)))
                raw_all[str(i)] = [float(t) for t in times]
            report.setdefault("fom", []).append(dict(
                fom_tol=tol, cohort=cname, time_ms=float(np.median(ts)) * 1e3,
                time_ms_all=[t * 1e3 for t in ts], time_raw_s=raw_all,
                err_rel_l2=float(np.mean(errs)), err_rel_l2_max=float(np.max(errs)),
                n_sources=len(idxs)))
            sc.log(f"   FOM CG tol={tol:.0e} [{cname:14s}]: "
                   f"{np.median(ts)*1e3:8.3f} ms  err {np.mean(errs):.3e}")
        save()

    # ------------------ error ladder: oracle + weak-EQ optimum ---------------
    oracle_lm = ss.make_oracle_lm(u_full_fast, K, budget=ORACLE_BUDGET)
    targets = jnp.asarray(U_int)
    enc_inits = jnp.stack([enc_apply(enc_params, e0["f_m_of"](jnp.asarray(Fs[i])))
                           for i in range(n_src)])
    init_sets = [jnp.tile(zbar[None], (n_src, 1)), enc_inits,
                 jnp.asarray(solve_z["cach|ctrl|rst300|enc"])]
    z_or, v_or = ss.oracle_multi_init(oracle_lm, init_sets, targets)
    or_rel = np.asarray(v_or) / tn
    report["oracle"] = {}
    for cname in cohorts:
        idxs = [i for i in range(n_src) if cohort_of[i] == cname]
        report["oracle"][cname] = dict(
            mean=float(np.mean(or_rel[idxs])), max=float(np.max(or_rel[idxs])),
            per_source=[float(v) for v in or_rel[idxs]], n=len(idxs),
            inits="zbar + encoder + solve-z (multi-init, all fields)")
        sc.log(f"  ORACLE [{cname}]: mean {np.mean(or_rel[idxs]):.3e} "
               f"max {np.max(or_rel[idxs]):.3e}  (n={len(idxs)})")
    save()

    # weak-EQ optimum (DIAGNOSTIC: init at oracle z, truth-informed init;
    # never a reported solve).  tau=0 -> run to stall/budget with restarts.
    report["weak_opt"] = {}
    for eqn in ("ctrl", "M128"):
        eq = eq_sets[eqn]
        errs, objs = [], []
        for i in range(n_src):
            out = wopt_lm(z_or[i], eq["G_q"], eq["wq"], eq["PhiT"], eq["Wl"],
                          eq["f_ms"][i], 0.0, trust)
            u = np.asarray(u_full_fast(out[0])).reshape(-1)
            errs.append(float(np.linalg.norm(u - U_int[i]) / tn[i]))
            objs.append(float(out[1]))
        errs = np.asarray(errs)
        for cname in cohorts:
            idxs = [i for i in range(n_src) if cohort_of[i] == cname]
            report["weak_opt"].setdefault(eqn, {})[cname] = dict(
                err_mean=float(np.mean(errs[idxs])), err_max=float(np.max(errs[idxs])),
                per_source=[float(v) for v in errs[idxs]],
                note="rst-LM from ORACLE latent, tau=0 -- diagnostic rung only")
            sc.log(f"  WEAK-OPT [{eqn}|{cname}]: err mean {np.mean(errs[idxs]):.3e} "
                   f"max {np.max(errs[idxs]):.3e}")
    save()

    # ------------------ adaptive quadrature (fresh cohort, timed honestly) ---
    adq_rows = []
    Phi_c0 = Phi_f0[cand_pos]
    for qi in range(min(ADQ_N, N_TEST)):
        i = N_TEST + qi                                  # fresh cohort index
        t0 = time.perf_counter()
        Fi = jnp.asarray(Fs[i])
        f_m = e0["f_m_of"](Fi)
        z1 = lm60(zbar, e0["G_q"], e0["wq"], e0["PhiT"], e0["Wl"], f_m,
                  TAUS[0], trust)[0]
        rng_q = np.random.default_rng(ctol_eq.EQ_SEED + 100 + qi)
        zs_new = [np.asarray(z1)] + [np.asarray(z1) + pert_c *
                                     rng_q.standard_normal(K)
                                     for _ in range(ADQ_PERT)]
        R_new = np.stack([np.asarray(G_cand @ h_fn(jnp.asarray(zz)))
                          for zz in zs_new])
        Rf_new = np.stack([np.asarray(u_full_fast(jnp.asarray(zz)))
                           for zz in zs_new])
        b_new = (Rf_new @ Phi_f0).reshape(-1)
        G_new = np.einsum("sp,pm->smp", R_new, Phi_c0).reshape(-1, R_new.shape[1])
        keep2, wq2, dinfo = ss.adq_extend_fit(
            G_c, b_c, G_new, b_new, MQ_CTRL, pc.nnls_capped,
            seed=ctol_eq.EQ_SEED + 100 + qi, pad_score=None)
        pts2 = coords_int[cand_pos[keep2]]
        PhiT2 = jnp.asarray(Phi_f0[cand_pos[keep2]].T)
        Wl2 = jnp.asarray(np.ones(PhiT2.shape[0]))
        G_q2 = G_cand[jnp.asarray(keep2)]
        out = rst300(zbar, G_q2, jnp.asarray(wq2), PhiT2, Wl2, f_m,
                     TAUS[0], trust)
        u = np.asarray(G_full @ h_fn(out[0])).reshape(-1)
        u.reshape(-1)
        dt = time.perf_counter() - t0
        if qi == 0:
            # gate the per-query EQ rule once: meshfree == cached on new nodes,
            # and the Phi_f-row mode table == the incumbent colloc_mode_table
            spec = dict(kind="weak", alpha=1.0, M=M_CTRL)
            PhiT_ref, Wl_ref = pc.colloc_mode_table(grid, spec, "grid", pts2)
            tab_dev = max(float(jnp.max(jnp.abs(PhiT2 - jnp.asarray(PhiT_ref)))),
                          float(jnp.max(jnp.abs(Wl2 - jnp.asarray(Wl_ref)))))
            zt = jnp.asarray(Z_tr[0] + 0.05 * np.random.default_rng(1).standard_normal(K))
            ra = Wl2 * (PhiT2 @ (jnp.asarray(wq2) * dec(zt, jnp.asarray(pts2)))) - f_m
            rb = Wl2 * (PhiT2 @ (jnp.asarray(wq2) * (G_q2 @ h_fn(zt)))) - f_m
            g0 = float(jnp.max(jnp.abs(ra - rb)) / (jnp.max(jnp.abs(ra)) + 1e-300))
            report["gates"]["adq_eq_gate0"] = g0
            report["gates"]["adq_mode_table_dev"] = tab_dev
            sc.log(f"  ADQ gates: mode-table dev {tab_dev:.2e}, gate0 {g0:.2e}")
            assert g0 < 1e-12 and tab_dev < 1e-12
        err = float(np.linalg.norm(u - U_int[i]) / tn[i])
        adq_rows.append(dict(src=i, cohort=cohort_of[i], err_rel_l2=err,
                             time_s_total=dt, obj_rel=float(out[1] / (out[2] + 1e-300)),
                             reason=int(out[6]), restarts=int(out[7]),
                             eq_diag=dinfo))
        sc.log(f"   ADQ src {i}: err {err:.3e}  total {dt:.1f}s "
               f"(EQ p95 {dinfo['row_rel_p95']:.1e} max {dinfo['row_rel_max']:.1e})")
    report["adq"] = dict(
        rows=adq_rows,
        note="per-query adaptive EQ: cheap solve -> extend system with rows at "
             "z* -> NNLS refit (host) -> re-solve; total wall time reported -- "
             "NOT competitive as a speed arm, accuracy diagnostic only")
    save()

    report["complete"] = True
    report["total_seconds"] = time.time() - t_all
    save()
    sc.log(f"DONE poisson r1 [{time.time()-t_all:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main()
