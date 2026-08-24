"""N=256 push, ROUND 2, Burgers cell — the PRIMARY 1e-3 trajectory push.

Round-1 verdict (runs/push_r1_burgers): every arm identical at 2.47e-2;
single-step weak-opt == per-state oracle at every probed t (no compounding,
no solver/objective/IC slack); early-time states are the hardest (oracle
t0-t1 up to 1.6e-1); encoder-IC (3.5 ms) and extrapolated warm start
(55 ms rollout) are pure speed wins; adaptive-TR restarts are a pure loss.

This cell, per the priority-flip directive:
  * decoder R per the POD-floor verdict (env), v2 trainer (point-subsampled
    AdamW + EMA + full-batch tail), MORE training states with the early-time
    window fully included (all t<=T_EARLY states of every training
    trajectory + uniform rest up to MAX_SNAPS) — evidence-driven from the
    r1 oracle-vs-t profile; training data only, cohorts untouched
  * IC: encoder (trained on the training states) + LM refine, plus the
    incumbent multistart control; ic floors reported against the t0 oracle
  * rollouts: incumbent control + safeguarded extrapolation (champion);
    STEP_TOL ladder {1e-9 control, 1e-6, 1e-5} through the SAME incumbent
    absolute-tolerance rule (tol * rms(u0) * sqrt(n_i^2)) so stepping can
    terminate on TOLERANCE at the 1e-3 scale — stop reasons reported per arm
  * objective certification: EQ sets M in {64,128,256}, m = 4M; per-set
    single-step weak-opt vs per-state oracle tracking (the gap IS the
    objective-truncation error at stepping scale)
  * REQUIRED per-step error-accumulation table: per_time rel-L2 arrays are
    stored in the JSON for every e2e arm and every trajectory, plus the
    per-state oracle per_time — so any 1e-3 failure is attributable to
    representation vs compounding.
All audit measurement rules, gate 0 per EQ set, and the solver/rollout
identity gates from r1 are retained.  PURE NEURAL — no POD in the model.
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
ROUND = int(os.environ.get("ROUND", "2"))
BATCHED = int(os.environ.get("BATCHED", "0"))
K = int(os.environ.get("K", "16"))
R = int(os.environ.get("R", "256"))
STEPS = int(os.environ.get("STEPS", "200000"))
LR = float(os.environ.get("LR", "1e-3"))
P_SUB = int(os.environ.get("P_SUB", "8192"))
WD = float(os.environ.get("WD", "1e-5"))
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.999"))
FULL_LAST = int(os.environ.get("FULL_LAST", "10000"))
TIME_CAP = float(os.environ.get("TIME_CAP", "0"))
LAM_ORTH = float(os.environ.get("LAM_ORTH", "1e-4"))
MAX_SNAPS = int(os.environ.get("MAX_SNAPS", "16384"))
T_EARLY = int(os.environ.get("T_EARLY", "5"))
EQ_MS = [int(v) for v in os.environ.get("EQ_MS", "64,128,256").split(",")]
STEP_TOLS = [float(v) for v in os.environ.get(
    "STEP_TOLS", "1e-9,1e-6,1e-5").split(",")]
N_TEST = int(os.environ.get("N_TEST", "8"))
TR_FACTOR = float(os.environ.get("TR_FACTOR", "0.01"))
SEED0 = int(os.environ.get("SEED0", "0"))
REPS = int(os.environ.get("REPS", "5"))
WARM = int(os.environ.get("WARM", "2"))
IC_TOP = int(os.environ.get("IC_TOP", "12"))
IC_ENC_BUDGET = int(os.environ.get("IC_ENC_BUDGET", "50"))
ENC_STEPS = int(os.environ.get("ENC_STEPS", "12000"))
EXTRAP = float(os.environ.get("EXTRAP", "1.0"))
ORACLE_BUDGET = int(os.environ.get("ORACLE_BUDGET", "150"))
SSTEP_TS = [int(v) for v in os.environ.get("SSTEP_TS", "1,2,3,5,10,25,50").split(",")]
SSTEP_BUDGET = int(os.environ.get("SSTEP_BUDGET", "120"))
NEWTON_TOLS = [float(v) for v in os.environ.get(
    "NEWTON_TOLS", "1e-2,1e-3,1e-5,1e-8").split(",")]
LIN_FACTOR = float(os.environ.get("LIN_FACTOR", "1e-2"))
MAX_NEWTON = int(os.environ.get("MAX_NEWTON", "20"))
OUT_PREFIX = os.environ.get("OUT_PREFIX", "")
REASON_NAMES = {0: "budget", 1: "tol", 2: "stalled", 3: "lambda_max",
                4: "tol_at_init", 5: "nan_at_init"}


def make_tol_rollout(n, ntol, lin_tol, max_newton):
    """STRONG classical baseline (identical to sep_burgers_r1)."""
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


def main():
    dev = jax.devices()[0]
    sc.log(f"jax_backend={dev.platform} device={dev} "
           f"x64={jax.config.jax_enable_x64} ROUND2 N={N} K={K} R={R} "
           f"steps={STEPS} max_snaps={MAX_SNAPS} t_early={T_EARLY} seed={SEED0}")
    t_all = time.time()
    OUT = f"{OUT_PREFIX}sep_burgers_r2_K{K}_R{R}.json"
    CKPT = f"{OUT_PREFIX}sep_burgers_r2_N{N}_K{K}_R{R}.pkl"
    ARCH = sc.arch_from_env()

    d = bc.build_data(N)
    U = np.asarray(d["U"], dtype=np.float64)
    n_traj, T, n2 = U.shape
    interior = bc.interior_indices(N)
    coords = np.asarray(bc.grid_coords(N))
    U_test = np.asarray(d["U_test"], dtype=np.float64)
    nu_test = np.asarray(d["nu_test"], dtype=np.float64)

    report = dict(config=dict(
        pde="burgers2d", round=ROUND, N=N, k=K, r=R, steps=STEPS, lr=LR,
        p_sub=P_SUB, wd=WD, ema_decay=EMA_DECAY, full_last=FULL_LAST,
        time_cap=TIME_CAP, lam_orth=LAM_ORTH, max_snaps=MAX_SNAPS,
        t_early=T_EARLY, eq_Ms=EQ_MS, step_tols=STEP_TOLS, n_test=N_TEST,
        gn_budget=bc.GN_BUDGET, num_steps=bc.NUM_STEPS, dt=bc.DT,
        tr_factor=TR_FACTOR, seed=SEED0, data_seed=bc.SEED,
        test_seed=bc.TEST_SEED, reps=REPS, warm=WARM, ic_top=IC_TOP,
        ic_budget=bc.IC_BUDGET, ic_enc_budget=IC_ENC_BUDGET,
        enc_steps=ENC_STEPS, extrap=EXTRAP, oracle_budget=ORACLE_BUDGET,
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

    # training pick: ALL early-time states + uniform rest (training data only)
    rng = np.random.default_rng(SEED0)
    n_states = n_traj * T
    tidx_of = np.arange(n_states) % T
    early = np.nonzero(tidx_of <= T_EARLY)[0]
    rest = np.nonzero(tidx_of > T_EARLY)[0]
    if early.size >= MAX_SNAPS:
        pick = np.sort(rng.choice(early, MAX_SNAPS, replace=False))
    else:
        extra = rng.choice(rest, min(MAX_SNAPS - early.size, rest.size),
                           replace=False)
        pick = np.sort(np.concatenate([early, extra]))
    S_tr = U.reshape(n_states, n2)[pick][:, interior]
    n_early_pick = int(np.sum(tidx_of[pick] <= T_EARLY))
    report["data"]["n_states_trained"] = int(S_tr.shape[0])
    report["data"]["n_early_states_in_pick"] = n_early_pick
    coords_int = coords[interior]
    sc.log(f"  training states: {S_tr.shape[0]} ({n_early_pick} early "
           f"t<={T_EARLY} + {S_tr.shape[0]-n_early_pick} uniform)")

    # ------------------ train (v2) ------------------------------------------
    params, Z_tr, tinfo = ss.train_autodecoder_v2(
        jax.random.PRNGKey(SEED0), coords_int, S_tr, K, R,
        steps=STEPS, lr=LR, lam_orth=LAM_ORTH, weight_decay=WD, p_sub=P_SUB,
        ema_decay=EMA_DECAY, full_last=FULL_LAST, time_cap=TIME_CAP,
        tag=f"burgers r2 N={N} k={K} r={R}", **ARCH)
    report["train"] = tinfo
    dec = sc.SeparableDecoder(params, K, R)
    sc.save_pkl(CKPT, params, Z_tr, report["config"])
    save()

    train_radius = float(np.max(np.linalg.norm(Z_tr - Z_tr.mean(0), axis=1)))
    bc.TR_DELTA = (TR_FACTOR * train_radius) if TR_FACTOR > 0 else np.inf
    report["config"]["trust_delta"] = float(bc.TR_DELTA)
    zbar = Z_tr.mean(0)
    h_fn = dec.head_fn()
    G_all = dec.feat_at(coords)
    coords_j = jnp.asarray(coords)
    interior_j = jnp.asarray(interior)

    # ------------------ EQ sets + cached ops + gate 0 ------------------------
    eq_pick = np.sort(rng.choice(len(Z_tr), min(64, len(Z_tr)), replace=False))
    Z_eq = Z_tr[eq_pick]
    dx = 1.0 / (N - 1)
    eq_ops = {}
    for Mi in EQ_MS:
        name = "ctrl" if Mi == EQ_MS[0] else f"M{Mi}"
        cl = bc.fit_eq_weights(dec, N, Mi, 4 * Mi, Z_eq, kind="weak",
                               pool="grid")
        kx, ky, Phi, lam, _ = bc.test_modes(N, Mi)
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
        r_w_fast, rJ_fast, prev_of_fast, full_fast_ = mk()
        ops_fast = bc._finish_ops(rJ_fast, r_w_fast, prev_of_fast, full_fast_,
                                  m, "lspg")
        ops_fast["M"] = Mi
        ops_fast["tol_scale"] = float(np.sqrt(interior.size))
        ops_fast["colloc_used"] = cl
        ops_ref = bc.make_weak_ops(dec, N, cl, kind="weak", M=Mi,
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
        info_rep = {k_: v for k_, v in cl["info"].items()
                    if isinstance(v, (int, float, str, bool, type(None)))}
        info_rep["gate0"] = gate0
        report.setdefault("eq", {})[name] = info_rep
        eq_ops[name] = dict(ops_fast=ops_fast, ops_ref=ops_ref, idx=idx,
                            pos=pos, r_w=r_w_fast, m=m)
        save()

    e0 = eq_ops["ctrl"]
    ops0 = e0["ops_fast"]
    idx0_j = jnp.asarray(e0["idx"])
    G_q0 = dec.feat_at(coords[e0["idx"]])

    # ------------------ encoder + IC fits ------------------------------------
    X_tr = S_tr[:, e0["pos"]]
    enc_params, enc_apply, enc_info = ss.fit_code_encoder(
        jax.random.PRNGKey(SEED0 + 7), X_tr, Z_tr, steps=ENC_STEPS,
        tag=f"burgers r2 u@EQ->z N={N}")
    report["encoder"] = enc_info

    t0_mask = (pick % T) == 0
    Z0_states = Z_tr[t0_mask] if t0_mask.any() else Z_tr[:8]
    cands = np.concatenate([Z0_states, zbar[None]], axis=0)
    cands_j = jnp.asarray(cands)
    report["ic"] = dict(n_candidates=int(cands.shape[0]))

    def ic_ctrl_fit(u0):
        u0_eq = u0[idx0_j]
        scores = jax.vmap(lambda z: jnp.linalg.norm(G_q0 @ h_fn(z) - u0_eq))(
            cands_j)
        _, top = jax.lax.top_k(-scores, min(IC_TOP, cands.shape[0]))
        z0s = cands_j[top]

        def f(z):
            return G_all @ h_fn(z) - u0
        lm = ctol_tol.lm_tau_generic(f, K, bc.IC_BUDGET)
        outs = jax.vmap(lambda z_: lm(z_, 0.0))(z0s)
        zs, rns, nJs = outs[0], outs[1], outs[3]
        b = jnp.argmin(jnp.where(jnp.isfinite(rns), rns, jnp.inf))
        return zs[b], rns[b], jnp.sum(nJs)

    def ic_enc_fit(u0):
        z0 = enc_apply(enc_params, u0[idx0_j])

        def f(z):
            return G_all @ h_fn(z) - u0
        lm = ctol_tol.lm_tau_generic(f, K, IC_ENC_BUDGET)
        z, rn, _, nJ, *_ = lm(z0, 0.0)
        return z, rn, nJ

    ic_fits = dict(ctrl=ic_ctrl_fit, enc=ic_enc_fit)
    ic_fit_jit = {a: jax.jit(f) for a, f in ic_fits.items()}

    u0_gate = jnp.asarray(U_test[0, 0], dtype=F64)

    def f_gate(z):
        return dec(z, coords_j) - u0_gate
    lm_gate = ctol_tol.lm_tau_generic(f_gate, K, bc.IC_BUDGET)
    ic_dev = ctol_tol.check_tau_agreement(
        lm_gate, lambda *a: bc.fit_ic(*a), (jnp.asarray(zbar), 0.0),
        (dec, N, U_test[0, 0], {"mean": zbar}), "ic-jit vs fit_ic", tol=1e-9)
    report["ic"]["jit_vs_incumbent_rel_dev"] = float(ic_dev)
    sc.log(f"  IC solver identity: {ic_dev:.2e}")

    # ------------------ rollouts (ctrl + extrap champion) --------------------
    roll_extrap = {name: ss.make_rollout_v2(
        "incumbent", ops=eq_ops[name]["ops_fast"], num_steps=bc.NUM_STEPS,
        extrap=EXTRAP) for name in eq_ops}
    roll_v2_off = ss.make_rollout_v2("incumbent", ops=ops0,
                                     num_steps=bc.NUM_STEPS, extrap=0.0)
    zg = jnp.asarray(Z_tr[3])
    nug = float(np.median(nu_test))
    tolg = STEP_TOLS[0] * float(np.sqrt(np.mean(U_test[0, 0][interior] ** 2))) \
        * ops0["tol_scale"]
    us_g = jnp.full((bc.NUM_STEPS,), tolg, dtype=F64)
    delta0 = jnp.asarray(float(bc.TR_DELTA), dtype=F64)
    Zi = ops0["rollout_jit"](zg, nug, us_g, bc.GN_BUDGET)[0]
    Zv = roll_v2_off(zg, nug, us_g, bc.GN_BUDGET, delta0, delta0, delta0)[0]
    rdev = float(jnp.max(jnp.abs(Zi - Zv)))
    report["gates"]["rollout_v2_off_vs_incumbent"] = rdev
    sc.log(f"  ROLLOUT GATE (v2 no-extrap vs incumbent): max |dZ| {rdev:.2e}")
    assert rdev < 1e-6

    # ------------------ e2e pipelines ----------------------------------------
    def decode_all(Zf):
        return jax.vmap(h_fn)(Zf) @ G_all.T
    decode_jit = jax.jit(decode_all)

    def decode_all_mesh(Zf):
        return jax.vmap(lambda z: dec(z, coords_j))(Zf)

    def make_e2e(ic_name, roll_name, eq_name="ctrl", step_tol=STEP_TOLS[0],
                 mesh=False):
        ops = eq_ops[eq_name]["ops_fast"] if not mesh \
            else eq_ops[eq_name]["ops_ref"]
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
        else:
            ic_fit = ic_fits[ic_name]
        if roll_name == "ctrl" or mesh:
            def roll_fn(z0, nu, us):
                Z, rns, nJs, reasons = ops["rollout_jit"](z0, nu, us,
                                                          bc.GN_BUDGET)
                return Z, rns, nJs, reasons
        else:
            rv = roll_extrap[eq_name]

            def roll_fn(z0, nu, us):
                Z, rns, nJs, reasons, _ = rv(z0, nu, us, bc.GN_BUDGET,
                                             delta0, delta0, delta0)
                return Z, rns, nJs, reasons
        dec_all = decode_all_mesh if mesh else decode_all
        tol_scale = ops["tol_scale"]

        def e2e(u0, nu):
            z0, ic_rn, ic_nJ = ic_fit(u0)
            u_scale = jnp.sqrt(jnp.mean(u0[interior_j] ** 2))
            us = jnp.full((bc.NUM_STEPS,),
                          step_tol * u_scale * tol_scale, dtype=F64)
            Z, rns, nJs, reasons = roll_fn(z0, nu, us)
            Zfull = jnp.concatenate([z0[None], Z], axis=0)
            F = dec_all(Zfull)
            return F, z0, Z, rns, nJs, reasons, ic_rn, ic_nJ
        return jax.jit(e2e)

    e2e_arms = {
        "cach|ctrl|ic_ctrl|roll_ctrl|t1e-9": make_e2e("ctrl", "ctrl"),
        "mesh|ctrl|ic_ctrl|roll_ctrl|t1e-9": make_e2e("ctrl", "ctrl",
                                                      mesh=True),
        "cach|ctrl|ic_enc|roll_extrap|t1e-9": make_e2e("enc", "extrap"),
    }
    for stol in STEP_TOLS[1:]:
        e2e_arms[f"cach|ctrl|ic_enc|roll_extrap|t{stol:.0e}"] = \
            make_e2e("enc", "extrap", step_tol=stol)
    for name in eq_ops:
        if name != "ctrl":
            e2e_arms[f"cach|{name}|ic_enc|roll_extrap|t1e-9"] = \
                make_e2e("enc", "extrap", eq_name=name)

    fom_roll, _ = bc.bf.make_rollout(N)
    tol_rolls = {ntol: make_tol_rollout(N, ntol,
                                        max(ntol * LIN_FACTOR, 1e-12),
                                        MAX_NEWTON)
                 for ntol in NEWTON_TOLS}

    # ------------------ timing + metrics -------------------------------------
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
        pre = {a: e2e_arms[a](u0, nu) for a in e2e_arms}

        subs = []
        arm_names = list(e2e_arms)
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
        for a in ("ctrl", "enc"):
            subs.append((f"split_ic_{a}",
                         lambda _u=u0, _f=ic_fit_jit[a]:
                         (lambda o: (o[0].block_until_ready(), o)[1])(_f(_u))))
        Zfull_ctrl = jnp.concatenate(
            [pre["cach|ctrl|ic_ctrl|roll_ctrl|t1e-9"][1][None],
             pre["cach|ctrl|ic_ctrl|roll_ctrl|t1e-9"][2]], axis=0)
        subs.append(("split_decode",
                     lambda _Z=Zfull_ctrl:
                     (lambda o: (o.block_until_ready(), o)[1])(decode_jit(_Z))))
        raw, results = sc.balanced_time(subs, reps=REPS, warm=WARM)

        for a in e2e_arms:
            F, z0_t, Z_t, rns, nJs, reasons, ic_rn, ic_nJ = results[f"e2e|{a}"]
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
                per_time=[float(v) for v in per_time],      # REQUIRED table
                per_time_max=float(np.max(per_time)),
                n_finite_steps=int(np.sum(np.all(np.isfinite(Fh), axis=1)) - 1),
                jac_total=int(np.sum(np.asarray(nJs))),
                step_rn_final=[float(v) for v in np.asarray(rns)[-3:]],
                stop_reasons={REASON_NAMES[r_]: reasons_np.count(r_)
                              for r_ in set(reasons_np)},
                e2e_ms=float(np.median(raw[f"e2e|{a}"])) * 1e3,
                e2e_raw_s=[float(t) for t in raw[f"e2e|{a}"]],
                timed_vs_untimed_max_latent_dev=det_dev)
            per_arm_rows[a].append(row)
            sc.log(f"   {a:42s} traj {i}: ic {row['ic_rel']:.2e} "
                   f"err {row['traj_rel']:.3e}  jac {row['jac_total']}  "
                   f"e2e {row['e2e_ms']:8.2f} ms  {row['stop_reasons']}")
        for ntol in NEWTON_TOLS:
            snaps, its, rels = results[f"fom_ntol_{ntol:.0e}"]
            Sh = np.asarray(snaps)
            per_time = np.linalg.norm(Sh - U_test[i], axis=1) / tnorm
            its_np = np.asarray(its)
            rels_np = np.asarray(rels)
            base_rows[ntol].append(dict(
                traj=i, nu=nu, traj_rel=float(np.mean(per_time)),
                per_time=[float(v) for v in per_time],
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
        splits = dict(traj=i)
        for a in ("ctrl", "enc"):
            z_s, rn_s, nJ_s = results[f"split_ic_{a}"]
            splits[f"ic_{a}_ms"] = float(np.median(raw[f"split_ic_{a}"])) * 1e3
            splits[f"ic_{a}_raw_s"] = [float(t) for t in raw[f"split_ic_{a}"]]
            splits[f"ic_{a}_rel"] = float(rn_s) / float(np.linalg.norm(u0_np))
            splits[f"ic_{a}_jac"] = int(nJ_s)
        splits["decode_ms"] = float(np.median(raw["split_decode"])) * 1e3
        report.setdefault("splits", []).append(splits)
        save()

    for a in e2e_arms:
        rows = per_arm_rows[a]
        errs = [r_["traj_rel"] for r_ in rows if np.isfinite(r_["traj_rel"])]
        agg_reasons = {}
        for r_ in rows:
            for k_, v in r_["stop_reasons"].items():
                agg_reasons[k_] = agg_reasons.get(k_, 0) + v
        parts = a.split("|")
        report["rows"].append(dict(
            pde="burgers2d", method=a, arm=parts[0], eq_set=parts[1],
            ic=parts[2], roll=parts[3], step_tol=parts[4], N=N, k=K, r=R,
            err_traj_rel_mean=float(np.mean(errs)) if errs else None,
            err_traj_rel_max=float(np.max(errs)) if errs else None,
            ic_rel_mean=float(np.mean([r_["ic_rel"] for r_ in rows])),
            ic_rel_max=float(np.max([r_["ic_rel"] for r_ in rows])),
            e2e_ms_median=float(np.median([r_["e2e_ms"] for r_ in rows])),
            jac_total_mean=float(np.mean([r_["jac_total"] for r_ in rows])),
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

    # ------------------ batched multi-query (round 3 speed) ------------------
    if BATCHED:
        u0b = jnp.asarray(U_test[:n_test, 0], dtype=F64)
        nub = jnp.asarray(nu_test[:n_test], dtype=F64)
        b_names = [a for a in ("cach|ctrl|ic_enc|roll_extrap|t1e-9",
                               f"cach|M{max(EQ_MS)}|ic_enc|roll_extrap|t1e-9")
                   if a in e2e_arms]
        subs = []
        for aname in b_names:
            be = jax.jit(jax.vmap(e2e_arms[aname]))

            def fn(_b=be):
                out = _b(u0b, nub)
                out[0].block_until_ready()
                return out
            subs.append((f"batched|{aname}", fn))
        for ntol in (1e-2, 1e-3):
            if ntol not in tol_rolls:
                continue
            br = jax.jit(jax.vmap(tol_rolls[ntol]))

            def fn(_b=br):
                out = _b(u0b, nub)
                out[0].block_until_ready()
                return out
            subs.append((f"batched|fom_ntol_{ntol:.0e}", fn))
        ctol_tol.burn_in(1.0)
        raw_b, res_b = sc.balanced_time(subs, reps=REPS, warm=WARM)
        report["batched"] = []
        for name in raw_b:
            times = raw_b[name]
            ent = dict(subject=name, n_queries=n_test,
                       total_ms_median=float(np.median(times)) * 1e3,
                       amortized_ms=float(np.median(times)) * 1e3 / n_test,
                       raw_s=[float(t) for t in times])
            out = res_b[name]
            Fb = np.asarray(out[0])                    # (n_test, T+1, n2)
            errs = [float(np.mean(np.linalg.norm(Fb[i] - U_test[i], axis=1)
                                  / np.linalg.norm(U_test[i], axis=1)))
                    for i in range(n_test)]
            ent.update(err_traj_rel_mean=float(np.mean(errs)),
                       err_traj_rel_max=float(np.max(errs)))
            if name.startswith("batched|cach"):
                # equivalence check vs the single-query run: the per-time
                # error curves must agree (fields themselves are not retained
                # per trajectory in the single-query rows)
                aname = name.split("batched|")[1]
                dev = 0.0
                for i in range(n_test):
                    pt_b = np.linalg.norm(Fb[i] - U_test[i], axis=1) \
                        / np.linalg.norm(U_test[i], axis=1)
                    pt_s = np.asarray(per_arm_rows[aname][i]["per_time"])
                    dev = max(dev, float(np.max(np.abs(pt_b - pt_s))))
                ent["batched_vs_single_max_pertime_dev"] = dev
            report["batched"].append(ent)
            sc.log(f"   BATCHED {name}: total {ent['total_ms_median']:.2f} ms "
                   f"-> {ent['amortized_ms']:.3f} ms/traj  "
                   f"err {ent['err_traj_rel_mean']:.3e}")
        save()

    # ------------------ ladder diagnostics (truth used, labelled) ------------
    full_fast = jax.jit(lambda z: G_all @ h_fn(z))
    oracle_lm = ss.make_oracle_lm(full_fast, K, budget=ORACLE_BUDGET)
    report["oracle"] = []
    z_or_all = {}
    for i in range(n_test):
        targets = jnp.asarray(U_test[i], dtype=F64)
        enc_inits = jax.vmap(lambda u: enc_apply(enc_params, u[idx0_j]))(targets)
        init_sets = [jnp.tile(jnp.asarray(zbar)[None], (targets.shape[0], 1)),
                     enc_inits]
        z_or, v_or = ss.oracle_multi_init(oracle_lm, init_sets, targets)
        rel = np.asarray(v_or) / np.linalg.norm(U_test[i], axis=1)
        z_or_all[i] = np.asarray(z_or)
        report["oracle"].append(dict(
            traj=i, mean=float(np.mean(rel)), max=float(np.max(rel)),
            t0=float(rel[0]), per_time=[float(v) for v in rel],
            note="per-state representation oracle, DIAGNOSTIC only"))
        sc.log(f"  ORACLE traj {i}: mean {np.mean(rel):.3e} max {np.max(rel):.3e} "
               f"t0 {rel[0]:.3e}")
    save()

    # per-set single-step weak-opt (objective certification at stepping scale)
    report["single_step_weak_opt"] = {}
    for name in eq_ops:
        ops_n = eq_ops[name]["ops_fast"]
        rows_ss = []
        for i in range(n_test):
            nu = float(nu_test[i])
            u_scale = float(np.sqrt(np.mean(U_test[i, 0][interior] ** 2)))
            tol_abs = STEP_TOLS[0] * u_scale * ops_n["tol_scale"]
            for t in SSTEP_TS:
                if t < 1 or t > bc.NUM_STEPS:
                    continue
                z_prev = jnp.asarray(z_or_all[i][t - 1])
                prev_c = ops_n["prev_of"](z_prev)
                z2, rn, nJ, acc, reason, att = ops_n["step_jit"](
                    z_prev, prev_c, nu, tol_abs, SSTEP_BUDGET)
                u_hat = np.asarray(full_fast(z2))
                err = float(np.linalg.norm(u_hat - U_test[i, t])
                            / np.linalg.norm(U_test[i, t]))
                or_err = float(np.linalg.norm(
                    np.asarray(full_fast(jnp.asarray(z_or_all[i][t])))
                    - U_test[i, t]) / np.linalg.norm(U_test[i, t]))
                rows_ss.append(dict(traj=i, t=t, err=err, oracle_err=or_err,
                                    rn=float(rn), reason=int(reason)))
        report["single_step_weak_opt"][name] = rows_ss
        em = np.mean([r_["err"] for r_ in rows_ss])
        om = np.mean([r_["oracle_err"] for r_ in rows_ss])
        sc.log(f"  single-step weak-opt [{name}]: mean err {em:.3e} vs mean "
               f"oracle {om:.3e} (gap = objective truncation)")
    save()

    report["complete"] = True
    report["total_seconds"] = time.time() - t_all
    save()
    sc.log(f"DONE burgers r2 [{time.time()-t_all:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main()
