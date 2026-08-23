"""N=256 push, ROUND 2, Poisson cell — the 1e-3 representation push.

Round-1 verdict (runs/push_r1_poisson): solver output == weak-EQ optimum ==
oracle at M=128; REPRESENTATION is the only binding rung.  POD-floor verdict
(runs/pod_floor): R=64 floors at fresh 3.2e-2 (the r1 oracle sat ON it);
R=512 over a 2048-sample dense family floors at fresh 4.8e-4 mean — so the
~1e-3 ladder needs R~512 AND denser training data.

This cell (architecture+optimization authorized by the 2026-08-23 directive):
  * data: canonical seed-0 train (512) + N_EXTRA fresh-seed-4242 sources
    appended (cohort definitions untouched), held/fresh test cohorts as ever
  * decoder: R per env (512), optional multi-scale Fourier features
    (FF_SCALES), wider tracks (G_HIDDEN/H_HIDDEN), trained by
    sep_solvers.train_autodecoder_v2 (point-subsampled steps, AdamW masked
    weight decay, EMA, full-batch last phase, wall-time cap)
  * objective certification at scale: EQ sets M in {64, 128, 256} with
    m = 4M, plus a tail-reweighted M256 variant; per-set weak-EQ optimum vs
    the L2 oracle IS the objective-truncation error and is reported per set
  * solve arms through the incumbent trust-LM (restart variant), encoder
    init, both taus; audit measurement rules (in-pipe projection + decode,
    AB/BA, raw reps, captured-invocation errors, both cohorts, censoring)
  * gates: pool cached==meshfree, gate 0 per EQ set, ext-EQ==incumbent,
    solver-repairs-off==incumbent.  PURE NEURAL — no POD in the model.
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
R = int(os.environ.get("R", "512"))
STEPS = int(os.environ.get("STEPS", "300000"))
LR = float(os.environ.get("LR", "1e-3"))
P_SUB = int(os.environ.get("P_SUB", "16384"))
WD = float(os.environ.get("WD", "1e-5"))
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.999"))
FULL_LAST = int(os.environ.get("FULL_LAST", "20000"))
TIME_CAP = float(os.environ.get("TIME_CAP", "0"))
LAM_ORTH = float(os.environ.get("LAM_ORTH", "1e-4"))
N_EXTRA = int(os.environ.get("N_EXTRA", "2048"))
EXTRA_SEED = int(os.environ.get("EXTRA_SEED", "4242"))
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
OUT = os.environ.get("OUT", "sep_poisson_r2.json")
CKPT = os.environ.get("CKPT", f"sep_poisson_r2_N{N}_K{K}_R{R}.pkl")
FOM_LADDER = [float(v) for v in os.environ.get(
    "FOM_LADDER", "5e-1,3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-6").split(",")]
EQ_MS = [int(v) for v in os.environ.get("EQ_MS", "64,128,256").split(",")]
TAIL_CAP = float(os.environ.get("TAIL_CAP", "3e-2"))
TAIL_ROUNDS = int(os.environ.get("TAIL_ROUNDS", "2"))
ENC_STEPS = int(os.environ.get("ENC_STEPS", "12000"))
ORACLE_BUDGET = int(os.environ.get("ORACLE_BUDGET", "300"))
WOPT_BUDGET = int(os.environ.get("WOPT_BUDGET", "600"))
ARCH = sc.arch_from_env()


def eq_fit_ext(u_cand, u_full, Phi_c, Phi_f, Z_snap, k_lat, m, label):
    """ctol_eq.eq_fit_poisson + raw system return (identical rng stream and
    NNLS tail; asserted bit-equal to the incumbent for the control set)."""
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
    return keep, wq, info, G, b, pad_score


def main():
    dev = jax.devices()[0]
    sc.log(f"jax_backend={dev.platform} device={dev} x64={jax.config.jax_enable_x64} "
           f"ROUND2 N={N} K={K} R={R} steps={STEPS} p_sub={P_SUB} wd={WD} "
           f"n_extra={N_EXTRA} arch={ARCH or 'default'}")
    t_all = time.time()
    report = dict(config=dict(
        pde="poisson2d", round=2, N=N, k=K, r=R, steps=STEPS, lr=LR,
        p_sub=P_SUB, wd=WD, ema_decay=EMA_DECAY, full_last=FULL_LAST,
        time_cap=TIME_CAP, lam_orth=LAM_ORTH, n_extra=N_EXTRA,
        extra_seed=EXTRA_SEED, eq_Ms=EQ_MS, taus=TAUS, n_test=N_TEST,
        gn_iters=GN_ITERS, budget_hi=BUDGET_HI, n_restarts=N_RESTARTS,
        restart_sig=RESTART_SIG, tail_cap=TAIL_CAP, tail_rounds=TAIL_ROUNDS,
        enc_steps=ENC_STEPS, oracle_budget=ORACLE_BUDGET,
        wopt_budget=WOPT_BUDGET, tr_factor=TR_FACTOR, seed=SEED0,
        fresh_seed=FRESH_SEED, data_seed=mp.SEED, cg_tol=mp.CG_TOL,
        reps=REPS, warm=WARM, arch_overrides=ARCH,
        x64=True, matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local")),
        rows=[], gates={}, complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    # ------------------ data -------------------------------------------------
    grid = pc.Grid(N)
    n_i = grid.n_i
    int_idx = np.asarray(grid.ix_full * N + grid.iy_full)
    U_all = np.asarray(mp.build_snapshots(N)[0])
    U_tr0 = U_all[:mp.N_TRAIN][:, int_idx]
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

    # EXTRA training snapshots from a separate seed (canonical draw untouched)
    cxe, cye, we, ae, _ = mp.sample_params(seed=EXTRA_SEED, m=N_EXTRA)
    U_extra = []
    for i0 in range(0, N_EXTRA, 256):
        Fe = np.stack([mp.source_interior(N, cxe[i], cye[i], we[i], ae[i])
                       for i in range(i0, min(i0 + 256, N_EXTRA))])
        U_extra.append(np.asarray(jax.lax.map(solve_one, jnp.asarray(Fe))
                                  ).reshape(len(Fe), -1))
    U_extra = np.concatenate(U_extra)
    rese = float(np.max([np.linalg.norm(np.asarray(
        mp.neg_lap_interior(jnp.asarray(U_extra[i].reshape(n_i, n_i)), N))
        - mp.source_interior(N, cxe[i], cye[i], we[i], ae[i]))
        / np.linalg.norm(mp.source_interior(N, cxe[i], cye[i], we[i], ae[i]))
        for i in range(0, N_EXTRA, 97)]))
    assert rese < 1e-10, "unconverged extra training snapshots"
    U_tr = np.concatenate([U_tr0, U_extra])
    Fs_tr_params = (np.concatenate([cx[:mp.N_TRAIN], cxe]),
                    np.concatenate([cy[:mp.N_TRAIN], cye]),
                    np.concatenate([w[:mp.N_TRAIN], we]),
                    np.concatenate([a[:mp.N_TRAIN], ae]))
    sc.log(f"  training set: {U_tr.shape[0]} snapshots "
           f"({mp.N_TRAIN} canonical + {N_EXTRA} seed-{EXTRA_SEED})")
    report["config"]["n_train_total"] = int(U_tr.shape[0])

    # ------------------ train (v2) ------------------------------------------
    params, Z_tr, tinfo = ss.train_autodecoder_v2(
        jax.random.PRNGKey(SEED0), coords_int, U_tr, K, R,
        steps=STEPS, lr=LR, lam_orth=LAM_ORTH, weight_decay=WD, p_sub=P_SUB,
        ema_decay=EMA_DECAY, full_last=FULL_LAST, time_cap=TIME_CAP,
        tag=f"poisson r2 N={N} k={K} r={R}", **ARCH)
    report["train"] = tinfo
    dec = sc.SeparableDecoder(params, K, R)
    sc.save_pkl(CKPT, params, Z_tr, report["config"])
    save()

    h_fn = dec.head_fn()
    G_full = dec.feat_at(coords_int)
    u_full_fast = jax.jit(lambda z: G_full @ h_fn(z))
    train_radius = float(np.max(np.linalg.norm(Z_tr - Z_tr.mean(0), axis=1)))
    trust = TR_FACTOR * train_radius if TR_FACTOR > 0 else np.inf
    zbar = jnp.asarray(Z_tr.mean(0))
    z_std = np.asarray(Z_tr, dtype=np.float64).std(axis=0)

    # ------------------ EQ sets (certification ladder in M) ------------------
    cand_pos = ctol_eq.candidate_pool(n_i * n_i)
    cand_xy = coords_int[cand_pos]
    G_cand = dec.feat_at(cand_xy)
    u_cand_mesh = jax.jit(lambda z: dec(z, jnp.asarray(cand_xy)))
    u_full_mesh = jax.jit(lambda z: dec(z, jnp.asarray(coords_int)))
    rng0 = np.random.default_rng(SEED0)
    zt = jnp.asarray(Z_tr[rng0.integers(len(Z_tr))] + 0.05 * rng0.standard_normal(K))
    pool_dev = float(jnp.max(jnp.abs(G_cand @ h_fn(zt) - u_cand_mesh(zt)))
                     / (jnp.max(jnp.abs(u_cand_mesh(zt))) + 1e-300))
    report["gates"]["pool_cached_vs_meshfree"] = pool_dev
    assert pool_dev < 1e-12

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

    Itab = {M: mode_tables(M) for M in sorted(set(EQ_MS))}
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
                             f_ms=f_ms, f_m_of=f_m_of, info=info, gate0=gate0)
        eq_rep = {k_: v for k_, v in info.items()
                  if isinstance(v, (int, float, str, bool, type(None)))}
        eq_rep.update(M=int(M), m=int(len(keep)), gate0=gate0, fm_dev=fm_dev)
        if extra:
            eq_rep.update(extra)
        report.setdefault("eq", {})[name] = eq_rep
        save()

    G_keep = {}
    for Mi in EQ_MS:
        I_, Jm_, Phi_f_, _ = Itab[Mi]
        keep_i, wq_i, info_i, G_i, b_i, pad_i = eq_fit_ext(
            u_cand_mesh, u_full_mesh, Phi_f_[cand_pos], Phi_f_, Z_tr, K, 4 * Mi,
            f"r2 M{Mi} N={N} m={4 * Mi}")
        name = "ctrl" if Mi == EQ_MS[0] else f"M{Mi}"
        if Mi == EQ_MS[0]:
            keep_ref, wq_ref, _ = ctol_eq.eq_fit_poisson(
                u_cand_mesh, u_full_mesh, Phi_f_[cand_pos], Phi_f_, Z_tr, K,
                4 * Mi, "reference re-fit", pc.nnls_capped)
            assert np.array_equal(np.asarray(keep_i), np.asarray(keep_ref))
            wq_dev = float(np.max(np.abs(wq_i - wq_ref))
                           / (np.max(np.abs(wq_ref)) + 1e-300))
            report["gates"]["eq_ext_vs_incumbent_wq"] = wq_dev
            assert wq_dev < 1e-12
        finish_eq(name, Mi, keep_i, wq_i, info_i)
        if Mi == max(EQ_MS):
            keep_t, wq_t, info_t = ss.tail_reweight_fit(
                G_i, b_i, 4 * Mi, pc.nnls_capped, seed=ctol_eq.EQ_SEED + 1,
                cap=TAIL_CAP, rounds=TAIL_ROUNDS, pad_score=pad_i,
                label=f"r2 tail M{Mi}")
            finish_eq(f"M{Mi}t", Mi, keep_t, wq_t, info_t)
        del G_i, b_i

    # ------------------ encoder (all training sources) -----------------------
    fm_of_ctrl = eq_sets["ctrl"]["f_m_of"]
    cxa, cya, wa, aa = Fs_tr_params
    X_tr = np.stack([np.asarray(fm_of_ctrl(jnp.asarray(
        mp.source_interior(N, cxa[i], cya[i], wa[i], aa[i]))))
        for i in range(len(cxa))])
    enc_params, enc_apply, enc_info = ss.fit_code_encoder(
        jax.random.PRNGKey(SEED0 + 7), X_tr, Z_tr, steps=ENC_STEPS,
        tag=f"poisson r2 f_m->z N={N}")
    report["encoder"] = enc_info

    # ------------------ solvers + gates --------------------------------------
    noise = jnp.asarray(RESTART_SIG * z_std * np.random.default_rng(SEED0 + 123)
                        .standard_normal((N_RESTARTS, K)))
    lm60 = ss.lm_tau_cached_restart(h_fn, K, GN_ITERS, None)
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
        rel_dz=sdev, inc_reason=int(out_inc[6]), new_reason=int(out_new[6]))
    sc.log(f"  SOLVER GATE: rel|dz| {sdev:.2e}")
    assert sdev < 1e-10

    # ------------------ timed pipes ------------------------------------------
    def make_pipe(eq, lm, init_kind):
        G_q, wq, PhiT, Wl = eq["G_q"], eq["wq"], eq["PhiT"], eq["Wl"]
        f_m_of = eq["f_m_of"]

        def pipe(F2d, tau):
            f_m = f_m_of(F2d)
            z0 = enc_apply(enc_params, f_m) if init_kind == "enc" else zbar
            z, val, v0, nJ, acc, att, reason, rs = lm(
                z0, G_q, wq, PhiT, Wl, f_m, tau, trust)
            return G_full @ h_fn(z), z, val, v0, nJ, att, reason, rs
        return jax.jit(pipe)

    def mesh_pipe(F2d, tau):
        f_m = e0["f_m_of"](F2d)
        z, val, v0, nJ, acc, att, reason = lm_mesh(zbar, f_m, tau)
        return (dec(z, jnp.asarray(coords_int)), z, val, v0, nJ, att, reason,
                jnp.int32(0))

    arms = {"mesh|ctrl|lm60|zbar": jax.jit(mesh_pipe),
            "cach|ctrl|lm60|enc": make_pipe(e0, lm60, "enc"),
            "cach|ctrl|rst300|enc": make_pipe(e0, rst300, "enc"),
            "cach|ctrl|rst300|zbar": make_pipe(e0, rst300, "zbar")}
    for name in eq_sets:
        if name != "ctrl":
            arms[f"cach|{name}|rst300|enc"] = make_pipe(eq_sets[name], rst300,
                                                        "enc")
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
    solve_z = {}
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
                           M=eq_sets[eqn]["M"], m=eq_sets[eqn]["m"], tau=tau,
                           cohort=cname,
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
                       f"err {row['err_rel_l2']:.3e}  cens "
                       f"{row['censored_frac']*100:3.0f}%")
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

    # ------------------ ladder: oracle + weak-EQ optimum per set -------------
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
            inits="zbar + encoder + solve-z")
        sc.log(f"  ORACLE [{cname}]: mean {np.mean(or_rel[idxs]):.3e} "
               f"max {np.max(or_rel[idxs]):.3e}")
    save()

    report["weak_opt"] = {}
    for eqn, eq in eq_sets.items():
        errs = []
        for i in range(n_src):
            out = wopt_lm(z_or[i], eq["G_q"], eq["wq"], eq["PhiT"], eq["Wl"],
                          eq["f_ms"][i], 0.0, trust)
            u = np.asarray(u_full_fast(out[0])).reshape(-1)
            errs.append(float(np.linalg.norm(u - U_int[i]) / tn[i]))
        errs = np.asarray(errs)
        for cname in cohorts:
            idxs = [i for i in range(n_src) if cohort_of[i] == cname]
            report["weak_opt"].setdefault(eqn, {})[cname] = dict(
                err_mean=float(np.mean(errs[idxs])),
                err_max=float(np.max(errs[idxs])),
                per_source=[float(v) for v in errs[idxs]],
                note="rst-LM from ORACLE latent, tau=0 -- diagnostic rung; "
                     "gap vs oracle IS the objective-truncation error")
            sc.log(f"  WEAK-OPT [{eqn}|{cname}]: err mean {np.mean(errs[idxs]):.3e} "
                   f"max {np.max(errs[idxs]):.3e}")
    save()

    report["complete"] = True
    report["total_seconds"] = time.time() - t_all
    save()
    sc.log(f"DONE poisson r2 [{time.time()-t_all:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main()
