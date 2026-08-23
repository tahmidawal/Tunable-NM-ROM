"""Separable-decoder Burgers-2D cell: train (no POD), weak NM-ROM rollout.

Two arms through the SAME incumbent weak operators and LM stepper:
  meshfree : blat_common.make_weak_ops(dec, ...) -- the network runs in-loop
  cached   : identical formulas with the stencil feature banks G_st (m,5,r)
             cached once; assembled by blat_common._finish_ops so the solver,
             acceptance rule, and rollout are bit-identical code.
GATE 0: weak residual/Jacobian of the two arms agree <= 1e-12 relative.
The FOM-exact sign-upwind stencil is preserved exactly.

N-scaling round (2026-08-23): implements the MANDATORY MEASUREMENT RULES from
HANDOFF.md / AUDIT-2026-08-23.md:
  - END-TO-END timed ROM path: known u0 -> IC latent fit -> 50-step latent
    rollout -> FULL-GRID decode of all 51 states, one jit; the IC-fit /
    rollout+decode split is reported separately;
  - the IC fit is the span-split multi-start LM (sep_common.make_span_fitter)
    -- the exact same data-misfit objective as the incumbent bc.fit_ic, with
    the N-dependent work reduced to one cached (r x n^2) matvec; its result
    is verified per-trajectory by a direct full-grid decode, and against the
    incumbent fitter on trajectory 0;
  - raw timing repetitions retained everywhere; balanced AB/BA pairing of the
    ROM against each classical baseline arm;
  - TWO classical baselines, same job, same GPU: the truth-generating fixed
    NEWTON_ITERS rollout (labelled OVER-SOLVED, never a headline comparator)
    and a tolerance-terminated Newton ladder (the strong classical stepper);
  - stop-reason distributions and per-step Newton counts recorded;
  - timed-vs-error-path latent deviation stored per trajectory.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import sep_common as sc

import jax
import jax.numpy as jnp

import blat_common as bc                     # noqa: E402  (path set by sc)
import ctol_eq                                # noqa: E402
import ctol_tol                               # noqa: E402  (burn_in only)

F64 = jnp.float64

N = int(os.environ.get("N", "64"))
K = int(os.environ.get("K", "16"))
R = int(os.environ.get("R", "64"))
M_MODES = int(os.environ.get("M", str(4 * K)))
MQ = int(os.environ.get("MQ", str(4 * M_MODES)))
STEPS = int(os.environ.get("STEPS", "30000"))
LR = float(os.environ.get("LR", "1e-3"))
N_TEST = int(os.environ.get("N_TEST", "4"))
MAX_SNAPS = int(os.environ.get("MAX_SNAPS", "8192"))
TR_FACTOR = float(os.environ.get("TR_FACTOR", "0.01"))
SEED0 = int(os.environ.get("SEED0", "0"))
REPS = int(os.environ.get("REPS", "5"))
FOM_REPS = int(os.environ.get("FOM_REPS", "3"))
PAIR_REPS = int(os.environ.get("PAIR_REPS", "3"))
ICF_ITERS = int(os.environ.get("ICF_ITERS", "40"))
ICF_STARTS = int(os.environ.get("ICF_STARTS", "16"))
NEWTON_TOLS = [float(v) for v in os.environ.get(
    "NEWTON_TOLS", "1e-2,1e-3,1e-4").split(",")]
LINB_FRAC = float(os.environ.get("LINB_FRAC", "0.05"))
NEWTON_MAX = int(os.environ.get("NEWTON_MAX", "12"))
OUT = os.environ.get("OUT", "sep_burgers.json")
CKPT = os.environ.get("CKPT", f"sep_burgers_N{N}_K{K}_R{R}.pkl")
ARCH = sc.arch_from_env()


def make_tol_newton(n):
    """Tolerance-terminated Newton rollout on the FULL grid -- the strong
    classical baseline (rule 5).  Same residual, upwind stencil, dt and
    BiCGStab inner solver as the truth generator, but each implicit step stops
    as soon as ||R(u)|| <= ntol * ||u_prev|| instead of running a fixed 8
    Newton iterations at LIN_TOL=1e-10.  Purely classical: no learned
    component anywhere."""
    _, residual = bc.bf.make_rollout(n)
    lin_maxiter = bc.bf.LIN_MAXITER
    # same exact-Helmholtz preconditioner as the (patched) truth generator:
    # boundary Jacobian rows are identity, interior stiff part is
    # I + dt*nu*(-lap_h), inverted exactly in the sine basis
    dxl = 1.0 / (n - 1)
    _pp = np.arange(1, n - 1)
    S_pc = jnp.asarray(np.sqrt(2.0 / (n - 1))
                       * np.sin(np.pi * np.outer(_pp, _pp) / (n - 1)))
    _l1 = (4.0 / dxl**2) * np.sin(np.pi * _pp / (2 * (n - 1))) ** 2
    lam_pc = jnp.asarray(_l1[:, None] + _l1[None, :])

    def step(u_prev, nu, ntol, lin_tol):
        u_scale = jnp.maximum(jnp.linalg.norm(u_prev), 1e-300)

        def cond(s):
            _, it, rn = s
            return (rn > ntol * u_scale) & (it < NEWTON_MAX)

        def body(s):
            u, it, rn = s
            r = residual(u, u_prev, nu)
            Jv = lambda v: jax.jvp(
                lambda uu: residual(uu, u_prev, nu), (u,), (v,))[1]
            def Minv(v):
                V = v.reshape(n, n)
                C = S_pc.T @ V[1:-1, 1:-1] @ S_pc
                return V.at[1:-1, 1:-1].set(
                    S_pc @ (C / (1.0 + bc.DT * nu * lam_pc)) @ S_pc.T
                ).reshape(-1)
            du, _ = jax.scipy.sparse.linalg.bicgstab(
                Jv, -r, tol=lin_tol, maxiter=lin_maxiter, M=Minv)
            ok = jnp.isfinite(du).all()
            u2 = u + jnp.where(ok, du, 0.0)
            rn2 = jnp.linalg.norm(residual(u2, u_prev, nu))
            good = jnp.isfinite(rn2)
            u = jnp.where(good, u2, u)
            rn = jnp.where(good, rn2, rn)
            it2 = jnp.where(good & ok, it + 1, jnp.int32(NEWTON_MAX))
            return (u, it2, rn)

        rn0 = jnp.linalg.norm(residual(u_prev, u_prev, nu))
        u, its, rn = jax.lax.while_loop(
            cond, body, (u_prev, jnp.int32(0), rn0))
        return u, its, rn / u_scale

    def roll(u0, nu, ntol, lin_tol):
        def body(u, _):
            u2, its, rel = step(u, nu, ntol, lin_tol)
            return u2, (u2, its, rel)
        _, (snaps, its, rels) = jax.lax.scan(body, u0, None,
                                             length=bc.NUM_STEPS)
        snaps = jnp.concatenate([u0[None], snaps], axis=0)   # (T+1, n^2)
        return snaps, its, rels

    return jax.jit(roll)


def traj_metrics(F, U_true):
    """The project's trajectory metric: mean over the 51 per-time rel-L2
    (incl. IC) plus the trajectory Frobenius rel-L2 (audit note)."""
    per = np.linalg.norm(F - U_true, axis=1) / np.linalg.norm(U_true, axis=1)
    return (float(np.mean(per)), float(np.max(per)),
            float(np.linalg.norm(F - U_true) / np.linalg.norm(U_true)))


def main():
    dev = jax.devices()[0]
    sc.log(f"jax_backend={dev.platform} device={dev} x64={jax.config.jax_enable_x64} "
           f"N={N} K={K} R={R} M={M_MODES} m={MQ} steps={STEPS} seed={SEED0} "
           f"n_train={bc.bf.N_TRAIN}+{bc.bf.N_VAL} arch={ARCH}")
    t_all = time.time()
    report = dict(config=dict(
        pde="burgers2d", N=N, k=K, r=R, M=M_MODES, m=MQ, steps=STEPS, lr=LR,
        n_test=N_TEST, gn_budget=bc.GN_BUDGET, gn_tol=bc.GN_TOL,
        num_steps=bc.NUM_STEPS, dt=bc.DT, tr_factor=TR_FACTOR, seed=SEED0,
        data_seed=bc.SEED, test_seed=bc.TEST_SEED, max_snaps=MAX_SNAPS,
        n_train_traj=bc.bf.N_TRAIN, n_val_traj=bc.bf.N_VAL,
        newton_iters_truth=bc.bf.NEWTON_ITERS, lin_tol_truth=bc.bf.LIN_TOL,
        newton_tols=NEWTON_TOLS, linb_frac=LINB_FRAC, newton_max=NEWTON_MAX,
        reps=REPS, fom_reps=FOM_REPS, pair_reps=PAIR_REPS,
        icf_iters=ICF_ITERS, icf_starts=ICF_STARTS, arch=ARCH,
        arch_desc="separable: FourierFeat-MLP g(x)->R^r  x  MLP-head h(z)->R^r, "
                  "hard poly BC; NO POD anywhere",
        objective=f"incumbent weak upwind M={M_MODES}, NNLS-EQ m={MQ} grid nodes",
        solver="blat_common lm_step_jit / rollout_jit (incumbent), both arms",
        timed_path="end-to-end: u0 -> span-split IC fit -> rollout_jit -> "
                   "full-grid decode of all 51 states, one jit; split reported",
        x64=True, matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local")), rows=[], complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    # ------------------ data (regenerated from seed) -------------------------
    d = bc.build_data(N)
    U = np.asarray(d["U"], dtype=np.float64)            # (n_traj, T, n^2)
    U_test = np.asarray(d["U_test"], dtype=np.float64)  # (N_TEST, T, n^2)
    nu_test = np.asarray(d["nu_test"], dtype=np.float64)
    n_traj, T, n2 = U.shape
    interior = bc.interior_indices(N)
    coords = np.asarray(bc.grid_coords(N))
    coords_int = coords[interior]
    report["data"] = dict(n_traj=int(n_traj), T=int(T), n2=int(n2),
                          fingerprint=bc.data_fingerprint(U),
                          max_fom_rel_residual=d.get("max_fom_rel_residual"))

    # training states: every (trajectory, time) state, interior values
    S_all = U.reshape(n_traj * T, n2)[:, interior]
    del d
    rng = np.random.default_rng(SEED0)
    if S_all.shape[0] > MAX_SNAPS:
        pick = np.sort(rng.choice(S_all.shape[0], MAX_SNAPS, replace=False))
    else:
        pick = np.arange(S_all.shape[0])
    S_tr = np.ascontiguousarray(S_all[pick])
    del S_all
    report["data"]["n_states_total"] = int(n_traj * T)
    report["data"]["n_states_trained"] = int(S_tr.shape[0])
    report["data"]["state_subsample"] = dict(
        rule=f"rng(SEED0={SEED0}).choice(n_states_total, {MAX_SNAPS}, "
             f"replace=False), sorted", indices_sha_first16=[int(v) for v in pick[:16]],
        n=int(pick.size))

    # ------------------ train ------------------------------------------------
    params, Z_tr, tinfo = sc.train_autodecoder(
        jax.random.PRNGKey(SEED0), coords_int, S_tr, K, R,
        steps=STEPS, lr=LR, tag=f"burgers N={N} k={K} r={R}", **ARCH)
    report["train"] = tinfo
    dec = sc.SeparableDecoder(params, K, R)
    sc.save_pkl(CKPT, params, Z_tr, report["config"])
    save()

    # trust region exactly as the accepted recipe: factor x training radius
    train_radius = float(np.max(np.linalg.norm(Z_tr - Z_tr.mean(0), axis=1)))
    bc.TR_DELTA = (TR_FACTOR * train_radius) if TR_FACTOR > 0 else np.inf
    report["config"]["trust_delta"] = float(bc.TR_DELTA)

    # ------------------ EQ (capped candidate pool) + incumbent weak ops ------
    kx, ky, Phi, lam, _lamc = bc.test_modes(N, M_MODES)
    n_i2 = interior.size
    eq_pick = np.sort(rng.choice(len(Z_tr), min(64, len(Z_tr)), replace=False))
    cand_pos = ctol_eq.candidate_pool(n_i2)
    xy_int_j = jnp.asarray(coords_int)
    u_full_int = jax.jit(lambda z: dec(z, xy_int_j))
    adv_full = jax.jit(lambda uf: bc.upwind_adv_field(uf, N))
    keep, wq_np, eq_info = ctol_eq.eq_fit_burgers(
        u_full_int, adv_full, Phi, cand_pos, Z_tr[eq_pick], K, MQ,
        f"sep burgers N={N} k={K} M={M_MODES} m={MQ}", bc.nnls_capped)
    colloc = dict(kind="grid", idx=interior[cand_pos[keep]], w=wq_np,
                  info=eq_info)
    report["eq"] = {k_: v for k_, v in eq_info.items()
                    if isinstance(v, (int, float, str, bool, type(None)))}
    ops_ref = bc.make_weak_ops(dec, N, colloc, kind="weak", M=M_MODES,
                               solver="lspg")

    # ------------------ cached fast ops (same formulas, banks cached) --------
    idx = np.asarray(colloc["idx"])
    m = idx.size
    w_q = jnp.asarray(colloc["w"], dtype=F64)
    pos = np.searchsorted(interior, idx)
    assert np.all(interior[pos] == idx)
    Phi_q = jnp.asarray(Phi[pos]) * w_q[:, None]                  # (m, M')
    lam_j = jnp.asarray(lam, dtype=F64)
    st = bc.stencil_indices(idx, N)                               # (m, 5)
    G_st = dec.feat_at(coords[st.reshape(-1)]).reshape(m, 5, dec.r)
    G_all = dec.feat_at(coords)                                   # (n^2, r)
    h_fn = dec.head_fn()
    dx = 1.0 / (N - 1)

    def u_and_N_fast(z):
        us = jnp.einsum("msr,r->ms", G_st, h_fn(z))
        c, xp, xm, yp, ym = us[:, 0], us[:, 1], us[:, 2], us[:, 3], us[:, 4]
        ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
        uy = jnp.where(c > 0.0, (c - ym) / dx, (yp - c) / dx)
        return c, c * (ux + uy)

    def prev_of_fast(z):
        return jnp.einsum("mr,r->m", G_st[:, 0, :], h_fn(z))

    def r_w_fast(z, prev_c, nu):
        wt = (1.0 + bc.DT * nu * lam_j) ** (-bc.WEAK_ALPHA)
        u, Nu = u_and_N_fast(z)
        pu = Phi_q.T @ u
        return wt * (Phi_q.T @ (u - prev_c) + bc.DT * ((Phi_q.T @ Nu)
                                                       + nu * lam_j * pu))

    def d_c_fast(z):
        return u_and_N_fast(z)[0]

    def rJ_fast(z, prev_c, nu):
        return (r_w_fast(z, prev_c, nu), jax.jacfwd(r_w_fast)(z, prev_c, nu),
                Phi_q.T @ jax.jacfwd(d_c_fast)(z))

    def full_fast(z):
        return G_all @ h_fn(z)

    ops_fast = bc._finish_ops(rJ_fast, r_w_fast, prev_of_fast, full_fast, m,
                              "lspg")
    ops_fast["M"] = M_MODES
    ops_fast["tol_scale"] = float(np.sqrt(interior.size))
    ops_fast["colloc_used"] = colloc

    # ------------------ GATE 0: identity of the two arms ---------------------
    g0 = []
    for _ in range(5):
        zt = jnp.asarray(Z_tr[rng.integers(len(Z_tr))]
                         + 0.05 * rng.standard_normal(K))
        zp = jnp.asarray(Z_tr[rng.integers(len(Z_tr))])
        prev_c = ops_ref["prev_of"](zp)
        nu = float(np.median(nu_test))
        ra, Ja, _ = ops_ref["rJ"](zt, prev_c, nu)
        rb, Jb, _ = ops_fast["rJ"](zt, prev_c, nu)
        pa = ops_ref["prev_of"](zt); pb = ops_fast["prev_of"](zt)
        g0.append(max(
            float(jnp.max(jnp.abs(ra - rb)) / (jnp.max(jnp.abs(ra)) + 1e-300)),
            float(jnp.max(jnp.abs(Ja - Jb)) / (jnp.max(jnp.abs(Ja)) + 1e-300)),
            float(jnp.max(jnp.abs(pa - pb)) / (jnp.max(jnp.abs(pa)) + 1e-300))))
    report["gate0_max_rel_dev"] = float(np.max(g0))
    sc.log(f"  GATE 0 (incumbent vs cached weak r/J identity): "
           f"max rel dev {np.max(g0):.2e}")
    assert np.max(g0) < 1e-12, "gate 0 failed"
    save()

    # ------------------ IC fitter (span split, multi-start, jitted) ----------
    t0_mask = (pick % T) == 0            # codes of t=0 training states
    Z0_states = Z_tr[t0_mask] if t0_mask.any() else Z_tr[:ICF_STARTS]
    zbar = Z_tr.mean(0)
    ic_inits = [zbar] + [z for z in Z0_states[:ICF_STARTS - 1]]
    while len(ic_inits) < ICF_STARTS:
        ic_inits.append(Z_tr[rng.integers(len(Z_tr))])
    report["config"]["ic_inits"] = f"mean + {len(ic_inits)-1} t=0 train codes"
    ic_fit = sc.make_span_fitter(G_all, h_fn, K, ic_inits, iters=ICF_ITERS)

    interior_j = jnp.asarray(interior)
    tol_scale = ops_fast["tol_scale"]

    def e2e_fn(u0, nu):
        z0, _rel, _v = ic_fit(u0)
        u_scale = jnp.sqrt(jnp.mean(u0[interior_j] ** 2))
        us = jnp.full((bc.NUM_STEPS,), bc.GN_TOL * u_scale * tol_scale,
                      dtype=F64)
        Z, rns, nJs, reasons = ops_fast["rollout_jit"](z0, nu, us,
                                                       bc.GN_BUDGET)
        Zall = jnp.concatenate([z0[None], Z], axis=0)
        U_out = sc.head(dec.params, Zall) @ G_all.T          # (T+1, n^2)
        return U_out, Z, nJs, reasons

    e2e = jax.jit(e2e_fn)

    def roll_dec_fn(z0, nu, us):
        Z, rns, nJs, reasons = ops_fast["rollout_jit"](z0, nu, us,
                                                       bc.GN_BUDGET)
        Zall = jnp.concatenate([z0[None], Z], axis=0)
        return sc.head(dec.params, Zall) @ G_all.T

    roll_dec = jax.jit(roll_dec_fn)

    # ------------------ per-trajectory: IC, error rollouts, e2e timing -------
    ctol_tol.burn_in(1.5)
    n_test = min(N_TEST, U_test.shape[0])
    traj_state = []
    for i in range(n_test):
        u0_dev = jnp.asarray(U_test[i, 0])
        u_scale = float(np.sqrt(np.mean(U_test[i, 0][interior] ** 2)))
        z0, ic_rel_est, _ = ic_fit(u0_dev)
        ic_rel = float(jnp.linalg.norm(G_all @ h_fn(z0) - u0_dev)
                       / jnp.linalg.norm(u0_dev))
        st_i = dict(traj=i, z0=z0, u0=u0_dev, u_scale=u_scale,
                    ic_rel=ic_rel, ic_rel_est=float(ic_rel_est),
                    nu=float(nu_test[i]))
        if i == 0:
            zi, ic_rel_inc, ic_info = bc.fit_ic(
                dec, N, U_test[i, 0], {"mean": zbar}, budget=30)
            st_i["ic_rel_incumbent_mean_init_b30"] = float(ic_rel_inc)
            sc.log(f"   IC check traj 0: span-fit {ic_rel:.3e} vs incumbent "
                   f"(mean init, budget 30) {ic_rel_inc:.3e}")
        traj_state.append(st_i)
        sc.log(f"   IC traj {i}: rel {ic_rel:.3e} (est {float(ic_rel_est):.3e})")

    for arm, ops in (("meshfree", ops_ref), ("cached", ops_fast)):
        rows = []
        for stt in traj_state:
            i = stt["traj"]
            ro = bc.rollout(dec, N, ops, stt["z0"], stt["nu"], stt["u_scale"],
                            U_true=U_test[i])
            row = dict(traj=i, ic_rel=stt["ic_rel"],
                       ic_rel_est=stt["ic_rel_est"],
                       traj_rel=ro.get("traj_rel"),
                       traj_rel_frob=ro.get("traj_rel_frob"),
                       per_time_mean=float(np.nanmean(ro["per_time"])),
                       per_time_max=float(np.nanmax(ro["per_time"])),
                       n_done=int(ro["n_done"]),
                       jac_total=int(np.sum(ro["iters"])),
                       reasons={r_: ro["reasons"].count(r_)
                                for r_ in set(ro["reasons"])})
            if arm == "cached":
                stt["Z_err"] = ro["Z"]
                us = jnp.full((bc.NUM_STEPS,),
                              bc.GN_TOL * stt["u_scale"] * tol_scale, dtype=F64)
                # end-to-end timed (u0 -> IC fit -> rollout -> full decode)
                med_e2e, ts_e2e = sc.time_fn(
                    lambda _u=stt["u0"], _n=stt["nu"]:
                    e2e(_u, _n)[0].block_until_ready(), reps=REPS)
                # split: IC fit alone / rollout+decode alone
                med_ic, ts_ic = sc.time_fn(
                    lambda _u=stt["u0"]: ic_fit(_u)[0].block_until_ready(),
                    reps=REPS)
                med_rd, ts_rd = sc.time_fn(
                    lambda _z=stt["z0"], _n=stt["nu"], _us=us:
                    roll_dec(_z, _n, _us).block_until_ready(), reps=REPS)
                # timed-vs-error-path equivalence (rule 4)
                _, Z_t, nJs_t, reasons_t = e2e(stt["u0"], stt["nu"])
                Z_err = np.asarray(ro["Z"])[1:]
                L_ = min(Z_err.shape[0], int(Z_t.shape[0]))
                dev = float(np.max(np.abs(np.asarray(Z_t)[:L_] - Z_err[:L_]))) \
                    if L_ > 0 else float("nan")
                row.update(e2e_ms=med_e2e * 1e3,
                           e2e_ms_raw=[t * 1e3 for t in ts_e2e],
                           icfit_ms=med_ic * 1e3,
                           icfit_ms_raw=[t * 1e3 for t in ts_ic],
                           rolldec_ms=med_rd * 1e3,
                           rolldec_ms_raw=[t * 1e3 for t in ts_rd],
                           timed_vs_error_max_latent_dev=dev,
                           timed_jac_total=int(jnp.sum(nJs_t)))
                stt["us"] = us
            else:
                # meshfree: latent rollout only (reference arm; NOT end-to-end)
                us = jnp.full((bc.NUM_STEPS,),
                              bc.GN_TOL * stt["u_scale"] * tol_scale, dtype=F64)
                med, ts = sc.time_fn(
                    lambda _z=stt["z0"], _n=stt["nu"], _us=us:
                    ops["rollout_jit"](_z, _n, _us, bc.GN_BUDGET)[0]
                    .block_until_ready(), reps=REPS)
                row.update(rollout_only_ms=med * 1e3,
                           rollout_only_ms_raw=[t * 1e3 for t in ts])
            rows.append(row)
            sc.log(f"   {arm:8s} traj {i}: ic {row['ic_rel']:.2e}  err "
                   f"{row.get('traj_rel', float('nan')):.3e}  jac "
                   f"{row['jac_total']}  n_done {row['n_done']}"
                   + (f"  e2e {row['e2e_ms']:8.2f} ms (ic {row['icfit_ms']:.2f}"
                      f" + roll+dec {row['rolldec_ms']:.2f})"
                      if arm == "cached" else ""))
        errs = [r_["traj_rel"] for r_ in rows if r_["traj_rel"] is not None
                and np.isfinite(r_["traj_rel"])]
        agg_reasons = {}
        for r_ in rows:
            for k_, v_ in r_["reasons"].items():
                agg_reasons[k_] = agg_reasons.get(k_, 0) + v_
        agg = dict(
            pde="burgers2d", method=f"sep_{arm}", N=N, k=K, r=R, M=M_MODES,
            m=int(m), err_traj_rel_mean=float(np.mean(errs)) if errs else None,
            err_traj_rel_max=float(np.max(errs)) if errs else None,
            jac_total_mean=float(np.mean([r_["jac_total"] for r_ in rows])),
            n_blowups=int(sum(r_["n_done"] < bc.NUM_STEPS for r_ in rows)),
            stop_reasons=agg_reasons, per_traj=rows, n_test=n_test)
        if arm == "cached":
            agg["e2e_ms_median"] = float(np.median([r_["e2e_ms"] for r_ in rows]))
            agg["icfit_ms_median"] = float(np.median([r_["icfit_ms"] for r_ in rows]))
            agg["rolldec_ms_median"] = float(np.median([r_["rolldec_ms"] for r_ in rows]))
        else:
            agg["rollout_only_ms_median"] = float(
                np.median([r_["rollout_only_ms"] for r_ in rows]))
        report["rows"].append(agg)
        save()

    # ------------------ classical baselines (same job, same GPU) -------------
    # (a) truth generator: fixed NEWTON_ITERS Newton + BiCGStab 1e-10 --
    #     OVER-SOLVED by construction; recorded, never a headline comparator.
    fom_roll, _ = bc.bf.make_rollout(N)
    fom_rows = []
    for stt in traj_state:
        i = stt["traj"]
        U0 = jnp.asarray(U_test[i:i + 1, 0])
        nu1 = jnp.asarray(nu_test[i:i + 1])
        snaps, res = fom_roll(U0, nu1)
        med, ts = sc.time_fn(lambda _u=U0, _n=nu1:
                             fom_roll(_u, _n)[0].block_until_ready(),
                             reps=FOM_REPS, warm=1)
        fom_rows.append(dict(traj=i, time_ms=med * 1e3,
                             time_ms_raw=[t * 1e3 for t in ts],
                             max_step_rel_res=float(jnp.max(res))))
        sc.log(f"   FOM truth(newton8,lin1e-10) traj {i}: {med*1e3:9.2f} ms  "
               f"max step rel res {float(jnp.max(res)):.1e}")
    report["fom_truth"] = dict(
        label="truth-generating fixed-Newton rollout; OVER-SOLVED",
        newton_iters=bc.bf.NEWTON_ITERS, lin_tol=bc.bf.LIN_TOL,
        time_ms_median=float(np.median([r_["time_ms"] for r_ in fom_rows])),
        per_traj=fom_rows)
    save()

    # (b) tolerance-terminated Newton ladder (strong classical stepper)
    tol_newton = make_tol_newton(N)
    report["fom_tolnewton"] = []
    report["paired"] = []
    for ntol in NEWTON_TOLS:
        lin_tol = max(LINB_FRAC * ntol, 1e-12)
        rows_tn, rows_pair = [], []
        for stt in traj_state:
            i = stt["traj"]
            u0 = stt["u0"]
            nuj = stt["nu"]
            snaps, its, rels = tol_newton(u0, nuj, ntol, lin_tol)
            F = np.asarray(snaps)
            e_mean, e_max, e_frob = traj_metrics(F, U_test[i])
            med, ts = sc.time_fn(
                lambda _u=u0, _n=nuj, _t=ntol, _l=lin_tol:
                tol_newton(_u, _n, _t, _l)[0].block_until_ready(),
                reps=FOM_REPS, warm=1)
            rows_tn.append(dict(
                traj=i, time_ms=med * 1e3, time_ms_raw=[t * 1e3 for t in ts],
                traj_rel=e_mean, per_time_max=e_max, traj_rel_frob=e_frob,
                newton_total=int(jnp.sum(its)),
                newton_per_step_max=int(jnp.max(its)),
                worst_step_rel_res=float(jnp.max(rels))))
            # balanced AB/BA pair: ROM end-to-end vs tol-Newton (rule 3/5)
            pr = sc.time_pair(
                lambda _u=u0, _n=nuj: e2e(_u, _n)[0].block_until_ready(),
                lambda _u=u0, _n=nuj, _t=ntol, _l=lin_tol:
                tol_newton(_u, _n, _t, _l)[0].block_until_ready(),
                reps=PAIR_REPS, warm=1)
            rows_pair.append(dict(traj=i, **pr))
            sc.log(f"   tol-Newton ntol={ntol:.0e} traj {i}: {med*1e3:9.2f} ms  "
                   f"err {e_mean:.3e}  newton {int(jnp.sum(its))}  "
                   f"paired ROM {pr['a_ms']:.2f} / TN {pr['b_ms']:.2f} ms")
        report["fom_tolnewton"].append(dict(
            ntol=ntol, lin_tol=lin_tol,
            time_ms_median=float(np.median([r_["time_ms"] for r_ in rows_tn])),
            err_traj_rel_mean=float(np.mean([r_["traj_rel"] for r_ in rows_tn])),
            err_traj_rel_max=float(np.max([r_["traj_rel"] for r_ in rows_tn])),
            newton_total_mean=float(np.mean([r_["newton_total"] for r_ in rows_tn])),
            per_traj=rows_tn))
        report["paired"].append(dict(
            rom="sep_cached end-to-end", baseline=f"tol_newton ntol={ntol:.0e}",
            rom_ms=float(np.median([r_["a_ms"] for r_ in rows_pair])),
            base_ms=float(np.median([r_["b_ms"] for r_ in rows_pair])),
            per_traj=rows_pair))
        save()

    report["complete"] = True
    report["total_seconds"] = time.time() - t_all
    save()
    sc.log(f"DONE burgers [{time.time()-t_all:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main()
