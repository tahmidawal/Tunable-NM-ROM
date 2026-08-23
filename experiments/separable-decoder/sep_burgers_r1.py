"""N=256 push, ROUND 1, Burgers cell (PUSH-PLAN.md).

One decoder (K16/R64/60k -- the verified Pareto cell, same rng stream as the
n256_j2 run so training and the control EQ set reproduce it), then the round-1
lever matrix over the SAME incumbent weak objective and tolerance rule:

  EQ sets : ctrl (M=4K, m=16K, incumbent bc.fit_eq_weights), m512 (m=32K),
            M128 (M=8K, m=32K), tail (ctrl system, tail-reweighted NNLS;
            baseline round asserted bit-equal to the incumbent fit)
  IC      : ic_ctrl (incumbent 12-candidate LM), ic_high (all-candidate,
            3x budget -- the one-time-cost ceiling), ic_enc (offline
            u0-at-EQ-nodes -> z0 encoder trained on TRAINING states only,
            + short LM refine)
  rollout : roll_ctrl (incumbent lm_step_jit scan), roll_extrap (incumbent
            kernel + safeguarded 2-step latent warm-start extrapolation),
            roll_adapt (adaptive trust region + restart-on-stall),
            roll_adapt_extrap  [sep_solvers; each repair disabled reproduces
            the incumbent, asserted before results are recorded]

Ladder diagnostics (truth used, clearly labelled, never in a solve path):
per-state representation oracle along every test trajectory, and a
single-step weak-EQ-optimum probe (step from the ORACLE latent of the true
previous state).

All N=64-audit measurement rules stay in force: END-TO-END timing including
the IC fit and full-grid decode, split reported; balanced AB/BA sweeps; raw
reps retained; error from captured timed invocations; strong tol-Newton
classical ladder in-job; truth generator labelled OVER-SOLVED; stop-reason
distributions everywhere.
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

import blat_common as bc                     # noqa: E402  (path set by sc)
import ctol_tol                               # noqa: E402

F64 = jnp.float64

N = int(os.environ.get("N", "64"))
K = int(os.environ.get("K", "16"))
R = int(os.environ.get("R", str(4 * K)))
M_CTRL = int(os.environ.get("M_CTRL", str(4 * K)))
MQ_CTRL = int(os.environ.get("MQ_CTRL", str(16 * K)))
M_BIG = int(os.environ.get("M_BIG", str(8 * K)))
MQ_BIG = int(os.environ.get("MQ_BIG", str(32 * K)))
STEPS = int(os.environ.get("STEPS", "30000"))
LR = float(os.environ.get("LR", "1e-3"))
N_TEST = int(os.environ.get("N_TEST", "4"))
MAX_SNAPS = int(os.environ.get("MAX_SNAPS", "8192"))
TR_FACTOR = float(os.environ.get("TR_FACTOR", "0.01"))
SEED0 = int(os.environ.get("SEED0", "0"))
REPS = int(os.environ.get("REPS", "5"))
WARM = int(os.environ.get("WARM", "2"))
IC_TOP = int(os.environ.get("IC_TOP", "12"))
IC_TOP_HI = int(os.environ.get("IC_TOP_HI", "48"))
IC_BUDGET_HI = int(os.environ.get("IC_BUDGET_HI", "300"))
IC_ENC_BUDGET = int(os.environ.get("IC_ENC_BUDGET", "50"))
ENC_STEPS = int(os.environ.get("ENC_STEPS", "8000"))
EXTRAP = float(os.environ.get("EXTRAP", "1.0"))
ADAPT_GROW = float(os.environ.get("ADAPT_GROW", "2.0"))
ADAPT_SHRINK = float(os.environ.get("ADAPT_SHRINK", "0.5"))
DMIN_FACTOR = float(os.environ.get("DMIN_FACTOR", "1e-4"))
DMAX_FACTOR = float(os.environ.get("DMAX_FACTOR", "1.0"))
N_STEP_RESTARTS = int(os.environ.get("N_STEP_RESTARTS", "2"))
STEP_RESTART_SIG = float(os.environ.get("STEP_RESTART_SIG", "0.05"))
TAIL_CAP = float(os.environ.get("TAIL_CAP", "3e-2"))
TAIL_ROUNDS = int(os.environ.get("TAIL_ROUNDS", "3"))
ORACLE_BUDGET = int(os.environ.get("ORACLE_BUDGET", "150"))
SSTEP_TS = [int(v) for v in os.environ.get("SSTEP_TS", "1,10,25,50").split(",")]
SSTEP_BUDGET = int(os.environ.get("SSTEP_BUDGET", "120"))
NEWTON_TOLS = [float(v) for v in os.environ.get(
    "NEWTON_TOLS", "1e-2,1e-3,1e-5,1e-8").split(",")]
LIN_FACTOR = float(os.environ.get("LIN_FACTOR", "1e-2"))
MAX_NEWTON = int(os.environ.get("MAX_NEWTON", "20"))
OUT_PREFIX = os.environ.get("OUT_PREFIX", "")
REASON_NAMES = {0: "budget", 1: "tol", 2: "stalled", 3: "lambda_max",
                4: "tol_at_init", 5: "nan_at_init"}


def make_tol_rollout(n, ntol, lin_tol, max_newton):
    """STRONG classical baseline (identical to sep_burgers.py): the truth
    generator's own residual/BiCGStab, Newton terminated at
    ||R|| <= ntol*||u_prev|| instead of a fixed 8 iterations."""
    _, residual = bc.bf.make_rollout(n)

    def step(u_prev, nu):
        u_scale = jnp.maximum(jnp.linalg.norm(u_prev), 1e-300)

        def cond(s):
            u, it, rn = s
            return (rn > ntol * u_scale) & (it < max_newton)

        def body(s):
            u, it, _ = s
            r = residual(u, u_prev, nu)
            Jv = lambda v: jax.jvp(lambda uu: residual(uu, u_prev, nu),
                                   (u,), (v,))[1]
            du, _ = jax.scipy.sparse.linalg.bicgstab(
                Jv, -r, tol=lin_tol, maxiter=bc.bf.LIN_MAXITER)
            ok = jnp.isfinite(du).all()
            u2 = u + jnp.where(ok, du, 0.0)
            rn2 = jnp.linalg.norm(residual(u2, u_prev, nu))
            return (u2, it + 1, rn2)

        r0 = jnp.linalg.norm(residual(u_prev, u_prev, nu))
        u, it, rn = jax.lax.while_loop(cond, body, (u_prev, jnp.int32(0), r0))
        return u, it, rn / u_scale

    def roll(u0, nu):
        def body(u, _):
            u2, it, rel = step(u, nu)
            return u2, (u2, it, rel)
        _, (snaps, its, rels) = jax.lax.scan(body, u0, None,
                                             length=bc.NUM_STEPS)
        return jnp.concatenate([u0[None], snaps], axis=0), its, rels

    return jax.jit(roll)


def build_Gb_burgers(dec, n, M, Z_snap):
    """The G/b system of bc.fit_eq_weights (kind='weak', pool='grid'),
    byte-for-byte the same construction, returned raw (pre-normalization) for
    the tail-reweight arm.  The baseline refit through sep_solvers._nnls_rows
    on all rows is asserted equal to the incumbent fit's output."""
    kx, ky, Phi, lam, lamc = bc.test_modes(n, M)
    coords = jnp.asarray(bc.grid_coords(n))
    interior = bc.interior_indices(n)
    xy_int = coords[jnp.asarray(interior)]
    Phi_np = np.asarray(Phi)
    u_full = jax.jit(lambda z: dec(z, xy_int))
    Gs, bs = [], []
    for z in Z_snap:
        z = jnp.asarray(z, dtype=F64)
        uf = u_full(z)
        Nf = bc.upwind_adv_field(uf, n)
        for v_f, v_c in ((np.asarray(uf), np.asarray(uf)),
                         (np.asarray(Nf), np.asarray(Nf))):
            bs.append(Phi_np.T @ v_f)
            Gs.append(Phi_np.T * v_c[None, :])
    G = np.concatenate(Gs, axis=0)
    b = np.concatenate(bs)
    return G, b, interior


def main():
    dev = jax.devices()[0]
    sc.log(f"jax_backend={dev.platform} device={dev} "
           f"x64={jax.config.jax_enable_x64} ROUND1 N={N} K={K} R={R} "
           f"steps={STEPS} seed={SEED0}")
    t_all = time.time()
    OUT = f"{OUT_PREFIX}sep_burgers_r1_K{K}_R{R}.json"
    CKPT = f"{OUT_PREFIX}sep_burgers_r1_N{N}_K{K}_R{R}.pkl"
    ARCH = sc.arch_from_env()

    # ------------------ data (regenerated from seed) -------------------------
    d = bc.build_data(N)
    U = np.asarray(d["U"], dtype=np.float64)
    n_traj, T, n2 = U.shape
    interior = bc.interior_indices(N)
    coords = np.asarray(bc.grid_coords(N))
    U_test = np.asarray(d["U_test"], dtype=np.float64)
    nu_test = np.asarray(d["nu_test"], dtype=np.float64)
    S_flat = U.reshape(n_traj * T, n2)

    report = dict(config=dict(
        pde="burgers2d", round=1, N=N, k=K, r=R, M_ctrl=M_CTRL, m_ctrl=MQ_CTRL,
        M_big=M_BIG, m_big=MQ_BIG, steps=STEPS, lr=LR, n_test=N_TEST,
        gn_budget=bc.GN_BUDGET, gn_tol=bc.GN_TOL, num_steps=bc.NUM_STEPS,
        dt=bc.DT, tr_factor=TR_FACTOR, seed=SEED0, data_seed=bc.SEED,
        test_seed=bc.TEST_SEED, max_snaps=MAX_SNAPS, reps=REPS, warm=WARM,
        ic_top=IC_TOP, ic_top_hi=IC_TOP_HI, ic_budget=bc.IC_BUDGET,
        ic_budget_hi=IC_BUDGET_HI, ic_enc_budget=IC_ENC_BUDGET,
        enc_steps=ENC_STEPS, extrap=EXTRAP, adapt_grow=ADAPT_GROW,
        adapt_shrink=ADAPT_SHRINK, dmin_factor=DMIN_FACTOR,
        dmax_factor=DMAX_FACTOR, n_step_restarts=N_STEP_RESTARTS,
        step_restart_sig=STEP_RESTART_SIG, tail_cap=TAIL_CAP,
        tail_rounds=TAIL_ROUNDS, oracle_budget=ORACLE_BUDGET,
        sstep_ts=SSTEP_TS, sstep_budget=SSTEP_BUDGET,
        newton_tols=NEWTON_TOLS, lin_factor=LIN_FACTOR, max_newton=MAX_NEWTON,
        arch_overrides=ARCH,
        x64=True, matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local")),
        rows=[], gates={}, complete=False)
    report["data"] = dict(n_traj=int(n_traj), T=int(T), n2=int(n2),
                          fingerprint=bc.data_fingerprint(U),
                          max_fom_rel_residual=d.get("max_fom_rel_residual"))

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    # per-cell rng stream IDENTICAL to sep_burgers.run_cell (pick, eq_pick)
    rng = np.random.default_rng(SEED0)
    n_states = n_traj * T
    if n_states > MAX_SNAPS:
        pick = np.sort(rng.choice(n_states, MAX_SNAPS, replace=False))
    else:
        pick = np.arange(n_states)
    S_tr = S_flat[pick][:, interior]
    report["data"]["n_states_trained"] = int(S_tr.shape[0])
    coords_int = coords[interior]

    # ------------------ train (reproduces the n256_j2 decoder) ---------------
    params, Z_tr, tinfo = sc.train_autodecoder(
        jax.random.PRNGKey(SEED0), coords_int, S_tr, K, R,
        steps=STEPS, lr=LR, tag=f"burgers r1 N={N} k={K} r={R}", **ARCH)
    report["train"] = tinfo
    dec = sc.SeparableDecoder(params, K, R)
    sc.save_pkl(CKPT, params, Z_tr, report["config"])
    save()

    train_radius = float(np.max(np.linalg.norm(Z_tr - Z_tr.mean(0), axis=1)))
    bc.TR_DELTA = (TR_FACTOR * train_radius) if TR_FACTOR > 0 else np.inf
    report["config"]["trust_delta"] = float(bc.TR_DELTA)
    z_std = np.asarray(Z_tr, dtype=np.float64).std(axis=0)
    zbar = Z_tr.mean(0)
    h_fn = dec.head_fn()
    G_all = dec.feat_at(coords)                                   # (n^2, r)
    coords_j = jnp.asarray(coords)
    interior_j = jnp.asarray(interior)

    # ------------------ EQ sets ---------------------------------------------
    eq_pick = np.sort(rng.choice(len(Z_tr), min(64, len(Z_tr)), replace=False))
    Z_eq = Z_tr[eq_pick]
    colloc_ctrl = bc.fit_eq_weights(dec, N, M_CTRL, MQ_CTRL, Z_eq, kind="weak",
                                    pool="grid")
    colloc_m512 = bc.fit_eq_weights(dec, N, M_CTRL, MQ_BIG, Z_eq, kind="weak",
                                    pool="grid")
    colloc_M128 = bc.fit_eq_weights(dec, N, M_BIG, MQ_BIG, Z_eq, kind="weak",
                                    pool="grid")
    # tail arm: same G/b system, baseline asserted == incumbent, then reweight
    G_t, b_t, _ = build_Gb_burgers(dec, N, M_CTRL, Z_eq)
    sc0 = np.linalg.norm(G_t, axis=1) + 1e-300
    keep_b, wq_b, _ = ss._nnls_rows(G_t / sc0[:, None], b_t / sc0,
                                    MQ_CTRL, bc.nnls_capped,
                                    np.random.default_rng(0),
                                    G_t.shape[0], None)
    idx_b = interior[keep_b]
    assert np.array_equal(np.sort(idx_b), np.sort(np.asarray(colloc_ctrl["idx"]))), \
        "tail-arm G/b builder does not reproduce the incumbent EQ node set"
    ord_map = np.argsort(idx_b)
    ord_ref = np.argsort(np.asarray(colloc_ctrl["idx"]))
    wq_dev = float(np.max(np.abs(wq_b[ord_map]
                                 - np.asarray(colloc_ctrl["w"])[ord_ref]))
                   / (np.max(np.abs(colloc_ctrl["w"])) + 1e-300))
    report["gates"]["tail_builder_vs_incumbent_wq"] = wq_dev
    sc.log(f"  tail-builder baseline vs incumbent EQ fit: wq dev {wq_dev:.2e}")
    assert wq_dev < 1e-10, "tail-arm baseline differs from incumbent EQ fit"
    keep_t, wq_t, info_t = ss.tail_reweight_fit(
        G_t, b_t, MQ_CTRL, bc.nnls_capped, seed=1, cap=TAIL_CAP,
        rounds=TAIL_ROUNDS, eq_rows=G_t.shape[0],
        label=f"r1 burgers tail N={N} M={M_CTRL} m={MQ_CTRL}")
    colloc_tail = dict(kind="grid", idx=interior[keep_t], w=wq_t, info=info_t)
    del G_t, b_t
    collocs = dict(ctrl=colloc_ctrl, m512=colloc_m512, M128=colloc_M128,
                   tail=colloc_tail)
    report["eq"] = {}
    for name, cl in collocs.items():
        report["eq"][name] = {k_: v for k_, v in cl["info"].items()
                              if isinstance(v, (int, float, str, bool,
                                                type(None)))}

    # ------------------ cached ops per EQ set + gate 0 -----------------------
    dx = 1.0 / (N - 1)
    eq_ops = {}
    for name, cl in collocs.items():
        M_this = M_CTRL if name != "M128" else M_BIG
        kx, ky, Phi, lam, _lamc = bc.test_modes(N, M_this)
        idx = np.asarray(cl["idx"])
        m = idx.size
        w_q = jnp.asarray(cl["w"], dtype=F64)
        pos = np.searchsorted(interior, idx)
        assert np.all(interior[pos] == idx)
        Phi_q = jnp.asarray(Phi[pos]) * w_q[:, None]
        lam_j = jnp.asarray(lam, dtype=F64)
        st = bc.stencil_indices(idx, N)
        G_st = dec.feat_at(coords[st.reshape(-1)]).reshape(m, 5, dec.r)

        def mk(G_st=G_st, Phi_q=Phi_q, lam_j=lam_j):
            def u_and_N_fast(z):
                us = jnp.einsum("msr,r->ms", G_st, h_fn(z))
                c, xp, xm, yp, ym = (us[:, 0], us[:, 1], us[:, 2], us[:, 3],
                                     us[:, 4])
                ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
                uy = jnp.where(c > 0.0, (c - ym) / dx, (yp - c) / dx)
                return c, c * (ux + uy)

            def prev_of_fast(z):
                return jnp.einsum("mr,r->m", G_st[:, 0, :], h_fn(z))

            def r_w_fast(z, prev_c, nu):
                wt = (1.0 + bc.DT * nu * lam_j) ** (-bc.WEAK_ALPHA)
                u, Nu = u_and_N_fast(z)
                pu = Phi_q.T @ u
                return wt * (Phi_q.T @ (u - prev_c)
                             + bc.DT * ((Phi_q.T @ Nu) + nu * lam_j * pu))

            def d_c_fast(z):
                return u_and_N_fast(z)[0]

            def rJ_fast(z, prev_c, nu):
                return (r_w_fast(z, prev_c, nu),
                        jax.jacfwd(r_w_fast)(z, prev_c, nu),
                        Phi_q.T @ jax.jacfwd(d_c_fast)(z))

            def full_fast(z):
                return G_all @ h_fn(z)
            return r_w_fast, rJ_fast, prev_of_fast, full_fast
        r_w_fast, rJ_fast, prev_of_fast, full_fast = mk()
        ops_fast = bc._finish_ops(rJ_fast, r_w_fast, prev_of_fast, full_fast,
                                  m, "lspg")
        ops_fast["M"] = M_this
        ops_fast["tol_scale"] = float(np.sqrt(interior.size))
        ops_fast["colloc_used"] = cl
        ops_ref = bc.make_weak_ops(dec, N, cl, kind="weak", M=M_this,
                                   solver="lspg")
        g0 = []
        grng = np.random.default_rng(SEED0 + 50)
        for _ in range(5):
            zt = jnp.asarray(Z_tr[grng.integers(len(Z_tr))]
                             + 0.05 * grng.standard_normal(K))
            zp = jnp.asarray(Z_tr[grng.integers(len(Z_tr))])
            prev_c = ops_ref["prev_of"](zp)
            nu = float(np.median(nu_test))
            ra, Ja, _ = ops_ref["rJ"](zt, prev_c, nu)
            rb, Jb, _ = ops_fast["rJ"](zt, prev_c, nu)
            pa = ops_ref["prev_of"](zt)
            pb = ops_fast["prev_of"](zt)
            g0.append(max(
                float(jnp.max(jnp.abs(ra - rb)) / (jnp.max(jnp.abs(ra)) + 1e-300)),
                float(jnp.max(jnp.abs(Ja - Jb)) / (jnp.max(jnp.abs(Ja)) + 1e-300)),
                float(jnp.max(jnp.abs(pa - pb)) / (jnp.max(jnp.abs(pa)) + 1e-300))))
        gate0 = float(np.max(g0))
        sc.log(f"  GATE 0 [{name}]: max rel dev {gate0:.2e}")
        assert gate0 < 1e-12, f"gate 0 failed for EQ set {name}"
        report["eq"][name]["gate0"] = gate0
        eq_ops[name] = dict(ops_fast=ops_fast, ops_ref=ops_ref, idx=idx,
                            pos=pos, r_w=r_w_fast, m=m)
        save()

    e0 = eq_ops["ctrl"]
    ops0 = e0["ops_fast"]
    idx0_j = jnp.asarray(e0["idx"])
    G_q0 = dec.feat_at(coords[e0["idx"]])                # (m, r) EQ centres

    # ------------------ IC encoder (TRAINING states only) --------------------
    X_tr = S_tr[:, e0["pos"]]                            # values at ctrl nodes
    enc_params, enc_apply, enc_info = ss.fit_code_encoder(
        jax.random.PRNGKey(SEED0 + 7), X_tr, Z_tr, steps=ENC_STEPS,
        tag=f"burgers u@EQ->z N={N}")
    report["encoder"] = enc_info

    # ------------------ IC fits ---------------------------------------------
    t0_mask = (pick % T) == 0
    Z0_states = Z_tr[t0_mask] if t0_mask.any() else Z_tr[:8]
    cands = np.concatenate([Z0_states, zbar[None]], axis=0)
    cands_j = jnp.asarray(cands)
    report["ic"] = dict(n_candidates=int(cands.shape[0]))

    def make_ic_multistart(ic_top, budget):
        ic_top = min(ic_top, cands.shape[0])

        def fit(u0):
            u0_eq = u0[idx0_j]
            scores = jax.vmap(lambda z: jnp.linalg.norm(G_q0 @ h_fn(z) - u0_eq))(
                cands_j)
            _, top = jax.lax.top_k(-scores, ic_top)
            z0s = cands_j[top]

            def f(z):
                return G_all @ h_fn(z) - u0
            lm = ctol_tol.lm_tau_generic(f, K, budget)
            outs = jax.vmap(lambda z_: lm(z_, 0.0))(z0s)
            zs, rns, nJs = outs[0], outs[1], outs[3]
            b = jnp.argmin(jnp.where(jnp.isfinite(rns), rns, jnp.inf))
            return zs[b], rns[b], jnp.sum(nJs)
        return fit

    def ic_enc_fit(u0):
        z0 = enc_apply(enc_params, u0[idx0_j])

        def f(z):
            return G_all @ h_fn(z) - u0
        lm = ctol_tol.lm_tau_generic(f, K, IC_ENC_BUDGET)
        z, rn, _, nJ, *_ = lm(z0, 0.0)
        return z, rn, nJ

    ic_fits = dict(ctrl=make_ic_multistart(IC_TOP, bc.IC_BUDGET),
                   high=make_ic_multistart(IC_TOP_HI, IC_BUDGET_HI),
                   enc=ic_enc_fit)
    ic_fit_jit = {a: jax.jit(f) for a, f in ic_fits.items()}

    # IC identity gate (incumbent path, as in the scaling round)
    u0_gate = jnp.asarray(U_test[0, 0], dtype=F64)

    def f_gate(z):
        return dec(z, coords_j) - u0_gate
    lm_gate = ctol_tol.lm_tau_generic(f_gate, K, bc.IC_BUDGET)
    ic_dev = ctol_tol.check_tau_agreement(
        lm_gate, lambda *a: bc.fit_ic(*a), (jnp.asarray(zbar), 0.0),
        (dec, N, U_test[0, 0], {"mean": zbar}), "ic-jit vs fit_ic", tol=1e-9)
    report["ic"]["jit_vs_incumbent_rel_dev"] = float(ic_dev)
    sc.log(f"  IC solver identity: {ic_dev:.2e}")

    # ------------------ repaired rollouts + agreement gates ------------------
    noise_step = jnp.asarray(STEP_RESTART_SIG * z_std
                             * np.random.default_rng(SEED0 + 123)
                             .standard_normal((N_STEP_RESTARTS, K)))
    step_ad = ss.make_step_adaptive(e0["r_w"], K, noise_step,
                                    grow=ADAPT_GROW, shrink=ADAPT_SHRINK)
    step_ad_off = ss.make_step_adaptive(e0["r_w"], K, None, grow=1.0,
                                        shrink=1.0)
    delta0 = jnp.asarray(float(bc.TR_DELTA), dtype=F64)
    dmin = jnp.asarray(DMIN_FACTOR * train_radius, dtype=F64)
    dmax = jnp.asarray(DMAX_FACTOR * train_radius, dtype=F64)

    # gate: adaptive step with repairs disabled == incumbent lm_step_jit
    zg = jnp.asarray(Z_tr[3])
    pg = ops0["prev_of"](jnp.asarray(Z_tr[5]))
    nug = float(np.median(nu_test))
    tolg = 1e-9 * float(np.sqrt(np.mean(U_test[0, 0][interior] ** 2))) \
        * ops0["tol_scale"]
    oi = ops0["step_jit"](zg, pg, nug, tolg, bc.GN_BUDGET)
    oa = step_ad_off(zg, pg, nug, tolg, bc.GN_BUDGET, delta0, delta0, delta0)
    sdev = float(np.linalg.norm(np.asarray(oa[0]) - np.asarray(oi[0]))
                 / (1.0 + np.linalg.norm(np.asarray(oi[0]))))
    report["gates"]["step_adaptive_off_vs_incumbent"] = dict(
        rel_dz=sdev, inc_reason=int(oi[4]), new_reason=int(oa[4]),
        inc_nj=int(oi[2]), new_nj=int(oa[2]))
    sc.log(f"  STEP GATE (adaptive repairs off vs incumbent): rel|dz| {sdev:.2e}")
    assert sdev < 1e-10, "adaptive step with repairs off != incumbent"

    rolls = {}
    rolls["ctrl"] = None                                # incumbent rollout_jit
    rolls["extrap"] = ss.make_rollout_v2("incumbent", ops=ops0,
                                         num_steps=bc.NUM_STEPS, extrap=EXTRAP)
    rolls["adapt"] = ss.make_rollout_v2("adaptive", step_ad=step_ad,
                                        rn_fn=ops0["rn"],
                                        prev_of=ops0["prev_of"],
                                        num_steps=bc.NUM_STEPS, extrap=0.0)
    rolls["adapt_extrap"] = ss.make_rollout_v2("adaptive", step_ad=step_ad,
                                               rn_fn=ops0["rn"],
                                               prev_of=ops0["prev_of"],
                                               num_steps=bc.NUM_STEPS,
                                               extrap=EXTRAP)
    roll_v2_off = ss.make_rollout_v2("incumbent", ops=ops0,
                                     num_steps=bc.NUM_STEPS, extrap=0.0)

    # gate: v2 rollout with no extrapolation vs incumbent rollout_jit.  The
    # kernel itself is bit-identical (STEP GATE above, rel|dz| = 0); composing
    # 50 kernels through a nested-jit scan reorders floating-point ops at the
    # jit boundary, which can flip individual LM accept decisions and produce
    # a small latent divergence (same class as the audited 3e-10 timed-vs-
    # untimed deviation at N=64).  Recorded, bounded well below any reported
    # error scale; the CONTROL arm still uses the incumbent rollout verbatim.
    us_g = jnp.full((bc.NUM_STEPS,), tolg, dtype=F64)
    Zi = ops0["rollout_jit"](zg, nug, us_g, bc.GN_BUDGET)[0]
    Zv = roll_v2_off(zg, nug, us_g, bc.GN_BUDGET, delta0, dmin, dmax)[0]
    rdev = float(jnp.max(jnp.abs(Zi - Zv)))
    report["gates"]["rollout_v2_off_vs_incumbent"] = rdev
    sc.log(f"  ROLLOUT GATE (v2 no-extrap vs incumbent): max |dZ| {rdev:.2e} "
           f"(fp-path divergence only; kernel gate is exact)")
    assert rdev < 1e-6, "v2 rollout without repairs diverges from incumbent"

    # variant rollouts for the non-ctrl EQ sets (best combo arm)
    rolls_eq = {}
    for name in ("m512", "M128", "tail"):
        e = eq_ops[name]
        sa = ss.make_step_adaptive(e["r_w"], K, noise_step, grow=ADAPT_GROW,
                                   shrink=ADAPT_SHRINK)
        rolls_eq[name] = ss.make_rollout_v2(
            "adaptive", step_ad=sa, rn_fn=e["ops_fast"]["rn"],
            prev_of=e["ops_fast"]["prev_of"], num_steps=bc.NUM_STEPS,
            extrap=EXTRAP)

    # ------------------ e2e pipelines ---------------------------------------
    def decode_all(Zf):
        return jax.vmap(h_fn)(Zf) @ G_all.T
    decode_jit = jax.jit(decode_all)

    def decode_all_mesh(Zf):
        return jax.vmap(lambda z: dec(z, coords_j))(Zf)

    def make_e2e(ic_name, roll_name, eq_name="ctrl", mesh=False):
        ops = eq_ops[eq_name]["ops_fast"] if not mesh \
            else eq_ops[eq_name]["ops_ref"]
        ic_fit = ic_fits[ic_name] if not mesh else None
        if mesh:
            def ic_fit(u0):
                u0_eq = u0[idx0_j]
                scores = jax.vmap(lambda z: jnp.linalg.norm(
                    dec(z, coords_j[idx0_j]) - u0_eq))(cands_j)
                _, top = jax.lax.top_k(-scores, min(IC_TOP, cands.shape[0]))
                z0s = cands_j[top]

                def f(z):
                    return dec(z, coords_j) - u0
                lm = ctol_tol.lm_tau_generic(f, K, bc.IC_BUDGET)
                outs = jax.vmap(lambda z_: lm(z_, 0.0))(z0s)
                zs, rns, nJs = outs[0], outs[1], outs[3]
                b = jnp.argmin(jnp.where(jnp.isfinite(rns), rns, jnp.inf))
                return zs[b], rns[b], jnp.sum(nJs)
        if roll_name == "ctrl" or mesh:
            def roll_fn(z0, nu, us):
                Z, rns, nJs, reasons = ops["rollout_jit"](z0, nu, us,
                                                          bc.GN_BUDGET)
                return Z, rns, nJs, reasons, jnp.zeros_like(nJs)
        else:
            rv = rolls[roll_name] if eq_name == "ctrl" else rolls_eq[eq_name]

            def roll_fn(z0, nu, us):
                return rv(z0, nu, us, bc.GN_BUDGET, delta0, dmin, dmax)
        dec_all = decode_all_mesh if mesh else decode_all
        tol_scale = ops["tol_scale"]

        def e2e(u0, nu):
            z0, ic_rn, ic_nJ = ic_fit(u0)
            u_scale = jnp.sqrt(jnp.mean(u0[interior_j] ** 2))
            us = jnp.full((bc.NUM_STEPS,),
                          bc.GN_TOL * u_scale * tol_scale, dtype=F64)
            Z, rns, nJs, reasons, rss = roll_fn(z0, nu, us)
            Zfull = jnp.concatenate([z0[None], Z], axis=0)
            F = dec_all(Zfull)
            return F, z0, Z, rns, nJs, reasons, rss, ic_rn, ic_nJ
        return jax.jit(e2e)

    e2e_arms = {
        "cach|ctrl|ic_ctrl|roll_ctrl": make_e2e("ctrl", "ctrl"),
        "mesh|ctrl|ic_ctrl|roll_ctrl": make_e2e("ctrl", "ctrl", mesh=True),
        "cach|ctrl|ic_ctrl|roll_adapt": make_e2e("ctrl", "adapt"),
        "cach|ctrl|ic_ctrl|roll_extrap": make_e2e("ctrl", "extrap"),
        "cach|ctrl|ic_ctrl|roll_adapt_extrap": make_e2e("ctrl", "adapt_extrap"),
        "cach|ctrl|ic_enc|roll_adapt_extrap": make_e2e("enc", "adapt_extrap"),
        "cach|ctrl|ic_high|roll_ctrl": make_e2e("high", "ctrl"),
        "cach|m512|ic_enc|roll_adapt_extrap": make_e2e("enc", "adapt_extrap",
                                                       "m512"),
        "cach|M128|ic_enc|roll_adapt_extrap": make_e2e("enc", "adapt_extrap",
                                                       "M128"),
        "cach|tail|ic_enc|roll_adapt_extrap": make_e2e("enc", "adapt_extrap",
                                                       "tail"),
    }

    # ------------------ classical baselines ---------------------------------
    fom_roll, _ = bc.bf.make_rollout(N)
    tol_rolls = {ntol: make_tol_rollout(N, ntol,
                                        max(ntol * LIN_FACTOR, 1e-12),
                                        MAX_NEWTON)
                 for ntol in NEWTON_TOLS}

    # ------------------ per-trajectory balanced timing -----------------------
    n_test = min(N_TEST, U_test.shape[0])
    per_arm_rows = {a: [] for a in e2e_arms}
    base_rows = {ntol: [] for ntol in NEWTON_TOLS}
    tg_rows = []
    ctol_tol.burn_in(1.5)
    for i in range(n_test):
        u0_np = U_test[i, 0]
        u0 = jnp.asarray(u0_np, dtype=F64)
        nu = float(nu_test[i])
        tnorm = np.linalg.norm(U_test[i], axis=1)
        u_scale = float(np.sqrt(np.mean(u0_np[interior] ** 2)))
        pre = {a: e2e_arms[a](u0, nu) for a in e2e_arms}
        z0_ctrl = pre["cach|ctrl|ic_ctrl|roll_ctrl"][1]
        us_ctrl = jnp.full((bc.NUM_STEPS,),
                           bc.GN_TOL * u_scale * ops0["tol_scale"], dtype=F64)

        subs = []
        arm_names = list(e2e_arms)
        # interleave ROM arms with the classical ladder for AB/BA balance
        for j, a in enumerate(arm_names):
            subs.append((f"e2e|{a}",
                         lambda _u=u0, _n=nu, _f=e2e_arms[a]:
                         (lambda o: (o[0].block_until_ready(), o)[1])(_f(_u, _n))))
            if j < len(NEWTON_TOLS):
                ntol = NEWTON_TOLS[j]
                subs.append((f"fom_ntol_{ntol:.0e}",
                             lambda _u=u0, _n=nu, _r=tol_rolls[ntol]:
                             (lambda o: (o[0].block_until_ready(), o)[1])(
                                 _r(_u, _n))))
        subs.append(("fom_newton8_truthgen",
                     lambda _u=jnp.asarray(U_test[i:i + 1, 0]),
                            _n=jnp.asarray(nu_test[i:i + 1]):
                     (lambda o: (o[0].block_until_ready(), o)[1])(
                         fom_roll(_u, _n))))
        for a in ("ctrl", "high", "enc"):
            subs.append((f"split_ic_{a}",
                         lambda _u=u0, _f=ic_fit_jit[a]:
                         (lambda o: (o[0].block_until_ready(), o)[1])(_f(_u))))
        for rname in ("ctrl", "extrap", "adapt", "adapt_extrap"):
            if rname == "ctrl":
                def rf(_z=z0_ctrl, _n=nu, _us=us_ctrl):
                    o = ops0["rollout_jit"](_z, _n, _us, bc.GN_BUDGET)
                    o[0].block_until_ready()
                    return o
            else:
                def rf(_z=z0_ctrl, _n=nu, _us=us_ctrl, _r=rolls[rname]):
                    o = _r(_z, _n, _us, bc.GN_BUDGET, delta0, dmin, dmax)
                    o[0].block_until_ready()
                    return o
            subs.append((f"split_roll_{rname}", rf))
        Zfull_ctrl = jnp.concatenate([z0_ctrl[None],
                                      pre["cach|ctrl|ic_ctrl|roll_ctrl"][2]],
                                     axis=0)
        subs.append(("split_decode",
                     lambda _Z=Zfull_ctrl:
                     (lambda o: (o.block_until_ready(), o)[1])(decode_jit(_Z))))
        raw, results = sc.balanced_time(subs, reps=REPS, warm=WARM)

        for a in e2e_arms:
            F, z0_t, Z_t, rns, nJs, reasons, rss, ic_rn, ic_nJ = \
                results[f"e2e|{a}"]
            Fh = np.asarray(F)
            per_time = np.linalg.norm(Fh - U_test[i], axis=1) / tnorm
            det_dev = float(jnp.max(jnp.abs(Z_t - pre[a][2])))
            reasons_np = [int(v) for v in np.asarray(reasons)]
            row = dict(
                traj=i, nu=nu,
                ic_rel=float(ic_rn) / float(np.linalg.norm(u0_np)),
                ic_jac_total=int(ic_nJ),
                traj_rel=float(np.mean(per_time)),
                traj_rel_frob=float(np.linalg.norm(Fh - U_test[i])
                                    / np.linalg.norm(U_test[i])),
                per_time_max=float(np.max(per_time)),
                n_finite_steps=int(np.sum(np.all(np.isfinite(Fh), axis=1)) - 1),
                jac_total=int(np.sum(np.asarray(nJs))),
                restarts_total=int(np.sum(np.asarray(rss))),
                stop_reasons={REASON_NAMES[r_]: reasons_np.count(r_)
                              for r_ in set(reasons_np)},
                e2e_ms=float(np.median(raw[f"e2e|{a}"])) * 1e3,
                e2e_raw_s=[float(t) for t in raw[f"e2e|{a}"]],
                timed_vs_untimed_max_latent_dev=det_dev)
            per_arm_rows[a].append(row)
            sc.log(f"   {a:38s} traj {i}: ic {row['ic_rel']:.2e} "
                   f"err {row['traj_rel']:.3e}  jac {row['jac_total']}  "
                   f"rs {row['restarts_total']}  e2e {row['e2e_ms']:8.2f} ms")
        for ntol in NEWTON_TOLS:
            snaps, its, rels = results[f"fom_ntol_{ntol:.0e}"]
            Sh = np.asarray(snaps)
            per_time = np.linalg.norm(Sh - U_test[i], axis=1) / tnorm
            its_np = np.asarray(its)
            rels_np = np.asarray(rels)
            base_rows[ntol].append(dict(
                traj=i, nu=nu, traj_rel=float(np.mean(per_time)),
                per_time_max=float(np.max(per_time)),
                newton_iters_total=int(np.sum(its_np)),
                steps_converged=int(np.sum(rels_np <= ntol)),
                steps_at_cap=int(np.sum(its_np >= MAX_NEWTON)),
                time_ms=float(np.median(raw[f"fom_ntol_{ntol:.0e}"])) * 1e3,
                time_raw_s=[float(t) for t in raw[f"fom_ntol_{ntol:.0e}"]]))
        snaps, res = results["fom_newton8_truthgen"]
        Sh = np.asarray(snaps)[:, 0, :]
        per_time = np.linalg.norm(Sh - U_test[i], axis=1) / tnorm
        tg_rows.append(dict(
            traj=i, nu=nu, traj_rel=float(np.mean(per_time)),
            max_step_rel_res=float(np.max(np.asarray(res))),
            time_ms=float(np.median(raw["fom_newton8_truthgen"])) * 1e3,
            time_raw_s=[float(t) for t in raw["fom_newton8_truthgen"]]))
        # per-trajectory split rows
        splits = dict(traj=i)
        for a in ("ctrl", "high", "enc"):
            z_s, rn_s, nJ_s = results[f"split_ic_{a}"]
            splits[f"ic_{a}_ms"] = float(np.median(raw[f"split_ic_{a}"])) * 1e3
            splits[f"ic_{a}_raw_s"] = [float(t) for t in raw[f"split_ic_{a}"]]
            splits[f"ic_{a}_rel"] = float(rn_s) / float(np.linalg.norm(u0_np))
            splits[f"ic_{a}_jac"] = int(nJ_s)
        for rname in ("ctrl", "extrap", "adapt", "adapt_extrap"):
            o = results[f"split_roll_{rname}"]
            splits[f"roll_{rname}_ms"] = \
                float(np.median(raw[f"split_roll_{rname}"])) * 1e3
            splits[f"roll_{rname}_raw_s"] = [float(t)
                                             for t in raw[f"split_roll_{rname}"]]
            splits[f"roll_{rname}_jac"] = int(np.sum(np.asarray(o[2])))
        splits["decode_ms"] = float(np.median(raw["split_decode"])) * 1e3
        report.setdefault("splits", []).append(splits)
        sc.log(f"   splits traj {i}: ic ctrl {splits['ic_ctrl_ms']:.1f} / high "
               f"{splits['ic_high_ms']:.1f} / enc {splits['ic_enc_ms']:.1f} ms "
               f"(rel {splits['ic_ctrl_rel']:.2e} / {splits['ic_high_rel']:.2e}"
               f" / {splits['ic_enc_rel']:.2e}); roll "
               + " ".join(f"{r_}:{splits[f'roll_{r_}_ms']:.1f}ms"
                          f"/j{splits[f'roll_{r_}_jac']}"
                          for r_ in ("ctrl", "extrap", "adapt", "adapt_extrap")))
        save()

    # ------------------ aggregate rows ---------------------------------------
    for a in e2e_arms:
        rows = per_arm_rows[a]
        errs = [r_["traj_rel"] for r_ in rows if np.isfinite(r_["traj_rel"])]
        agg_reasons = {}
        for r_ in rows:
            for k_, v in r_["stop_reasons"].items():
                agg_reasons[k_] = agg_reasons.get(k_, 0) + v
        mesh, eqn, icn, rn_ = a.split("|")
        report["rows"].append(dict(
            pde="burgers2d", method=a, arm=mesh, eq_set=eqn, ic=icn,
            roll=rn_, N=N, k=K, r=R,
            err_traj_rel_mean=float(np.mean(errs)) if errs else None,
            err_traj_rel_max=float(np.max(errs)) if errs else None,
            ic_rel_mean=float(np.mean([r_["ic_rel"] for r_ in rows])),
            ic_rel_max=float(np.max([r_["ic_rel"] for r_ in rows])),
            e2e_ms_median=float(np.median([r_["e2e_ms"] for r_ in rows])),
            jac_total_mean=float(np.mean([r_["jac_total"] for r_ in rows])),
            restarts_mean=float(np.mean([r_["restarts_total"] for r_ in rows])),
            stop_reasons=agg_reasons,
            n_blowups=int(sum(r_["n_finite_steps"] < bc.NUM_STEPS
                              for r_ in rows)),
            per_traj=rows, n_test=n_test))
    for ntol in NEWTON_TOLS:
        rows = base_rows[ntol]
        report["rows"].append(dict(
            pde="burgers2d", method="fom_newton_tol", N=N, newton_tol=ntol,
            lin_tol=max(ntol * LIN_FACTOR, 1e-12), max_newton=MAX_NEWTON,
            err_traj_rel_mean=float(np.mean([r_["traj_rel"] for r_ in rows])),
            err_traj_rel_max=float(np.max([r_["traj_rel"] for r_ in rows])),
            time_ms_median=float(np.median([r_["time_ms"] for r_ in rows])),
            newton_iters_mean=float(np.mean([r_["newton_iters_total"]
                                             for r_ in rows])),
            steps_converged_frac=float(np.mean(
                [r_["steps_converged"] / bc.NUM_STEPS for r_ in rows])),
            per_traj=rows, n_test=n_test))
    report["rows"].append(dict(
        pde="burgers2d", method="fom_newton8_truthgen", N=N, oversolved=True,
        note="fixed-8-Newton truth generator; NEVER a headline baseline",
        err_traj_rel_mean=float(np.mean([r_["traj_rel"] for r_ in tg_rows])),
        time_ms_median=float(np.median([r_["time_ms"] for r_ in tg_rows])),
        per_traj=tg_rows, n_test=n_test))
    save()

    # ------------------ ladder diagnostics (truth used, labelled) ------------
    full_fast = jax.jit(lambda z: G_all @ h_fn(z))
    oracle_lm = ss.make_oracle_lm(full_fast, K, budget=ORACLE_BUDGET)
    report["oracle"] = []
    z_or_all = {}
    for i in range(n_test):
        targets = jnp.asarray(U_test[i], dtype=F64)             # (T+1, n^2)
        enc_inits = jax.vmap(lambda u: enc_apply(enc_params, u[idx0_j]))(targets)
        init_sets = [jnp.tile(jnp.asarray(zbar)[None], (targets.shape[0], 1)),
                     enc_inits]
        z_or, v_or = ss.oracle_multi_init(oracle_lm, init_sets, targets)
        rel = np.asarray(v_or) / np.linalg.norm(U_test[i], axis=1)
        z_or_all[i] = np.asarray(z_or)
        report["oracle"].append(dict(
            traj=i, mean=float(np.mean(rel)), max=float(np.max(rel)),
            t0=float(rel[0]), per_time=[float(v) for v in rel],
            note="per-state representation oracle (multi-init: zbar+encoder), "
                 "DIAGNOSTIC only"))
        sc.log(f"  ORACLE traj {i}: mean {np.mean(rel):.3e} max {np.max(rel):.3e} "
               f"t0 {rel[0]:.3e}")
    save()

    # single-step weak-EQ optimum: step from the ORACLE latent of the TRUE
    # previous state (isolates objective bias from error accumulation)
    step_diag = ss.make_step_adaptive(e0["r_w"], K, noise_step,
                                      grow=ADAPT_GROW, shrink=ADAPT_SHRINK)
    sstep = []
    for i in range(n_test):
        nu = float(nu_test[i])
        u_scale = float(np.sqrt(np.mean(U_test[i, 0][interior] ** 2)))
        tol_abs = bc.GN_TOL * u_scale * ops0["tol_scale"]
        for t in SSTEP_TS:
            if t < 1 or t > bc.NUM_STEPS:
                continue
            z_prev = jnp.asarray(z_or_all[i][t - 1])
            prev_c = ops0["prev_of"](z_prev)
            o = step_diag(z_prev, prev_c, nu, tol_abs, SSTEP_BUDGET,
                          delta0, dmin, dmax)
            u_hat = np.asarray(full_fast(o[0]))
            err = float(np.linalg.norm(u_hat - U_test[i, t])
                        / np.linalg.norm(U_test[i, t]))
            or_err = float(np.linalg.norm(
                np.asarray(full_fast(jnp.asarray(z_or_all[i][t])))
                - U_test[i, t]) / np.linalg.norm(U_test[i, t]))
            sstep.append(dict(traj=i, t=t, err=err, oracle_err=or_err,
                              rn=float(o[1]), reason=int(o[4]),
                              restarts=int(o[7])))
    report["single_step_weak_opt"] = dict(
        rows=sstep,
        note="one adaptive-LM implicit step from the oracle latent of the "
             "true previous state; DIAGNOSTIC (oracle init uses truth)")
    sc.log("  single-step weak-opt: " + " ".join(
        f"t={r_['t']}:{r_['err']:.2e}(or {r_['oracle_err']:.2e})"
        for r_ in sstep[:len(SSTEP_TS)]))
    save()

    report["complete"] = True
    report["total_seconds"] = time.time() - t_all
    save()
    sc.log(f"DONE burgers r1 [{time.time()-t_all:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main()
