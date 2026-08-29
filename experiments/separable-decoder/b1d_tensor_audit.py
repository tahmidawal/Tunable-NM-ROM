"""E1 — local audit of the precomputed quadratic advection tensor against the
full-grid sign-upwind oracle, from a COMMITTED sep_b1d_scale checkpoint
(no training, no node fitting).  CPU is fine (JAX_PLATFORMS=cpu).

Per N it reports, and writes to OUT (json):
  TB   build determinism: T from two chunk orders, max rel diff
  TA   algebraic identity: h^T T h == Phi^T (u . D^- u) on every decoded
       training state (must be ~1e-14: this is what 'exact on u>0 points'
       reduces to, the two stencils coincide pointwise wherever u > 0)
  T0   h^T T h vs Phi^T N_upwind(G h) on the decoded training states whose
       field is positive EVERYWHERE (exact, ~1e-14) and pointwise identity of
       the two stencils on u>0 points
  TS   mismatch statistics on ALL decoded training states (median/mean/max
       relative), fraction of grid points with u<=0, min u
  TJ   Jacobian mismatch (dr/dz tensor vs oracle jacfwd) and gradient cosine
       at 32 perturbed states + 32 unperturbed training states
  TC   contraction condition  sum_jk |T_ijk h_j h_k| / |q_i|
  TL   oracle host-loop LM rollout of the 8 regenerated test trajectories
       with EVERY LM candidate (accepted and rejected) recorded: sign
       pattern (min u, #points u<=0), tensor-vs-oracle mismatch of q, r, J
       at each candidate; the host loop is cross-gated against the device
       reference rollout (gate V analog).
Env: N, CKPT_CACHE, OUT, QOUT (npy of Q for the in-job cross-machine check).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import b1d_common as b1
import b1d_fast_common as fc
import b1d_tensor_common as tc

F64 = jnp.float64
N = int(os.environ.get("N", "256"))
CKPT_CACHE = os.environ["CKPT_CACHE"]
OUT = os.environ.get("OUT", f"/tmp/b1d_tensor_audit_n{N}.json")
QOUT = os.environ.get("QOUT", "")
N_JAC = int(os.environ.get("N_JAC", "32"))
log = b1.log


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def stats(v):
    v = np.asarray(v, dtype=np.float64)
    return dict(median=float(np.median(v)), mean=float(np.mean(v)),
                p95=float(np.quantile(v, 0.95)), max=float(np.max(v)),
                min=float(np.min(v)), n=int(v.size))


def main():
    dev = jax.devices()[0]
    log(f"jax_backend={dev.platform} device={dev} B1D-TENSOR-AUDIT N={N}")
    t_all = time.time()
    su = fc.Setup(CKPT_CACHE, N)
    dx = su.dx
    G = np.asarray(su.G_int)
    Phi = np.asarray(su.Phi_j)
    R_, M_ = G.shape[1], Phi.shape[1]
    rep = dict(config=dict(N=N, K=fc.K, R=R_, M=M_, n_interior=int(su.n_i),
                           ckpt=CKPT_CACHE, backend=dev.platform,
                           jax_version=jax.__version__,
                           matmul_precision=os.environ.get(
                               "JAX_DEFAULT_MATMUL_PRECISION")),
               gates={})

    # ---------------- TB: build in two chunk orders --------------------------
    t0 = time.time()
    T1 = tc.build_T(Phi, G, dx, chunk=256, reverse=False)
    T2 = tc.build_T(Phi, G, dx, chunk=97, reverse=True)
    gTB = float(np.max(np.abs(T1 - T2)) / np.max(np.abs(T1)))
    rep["gates"]["TB_build_order_rel"] = gTB
    rep["build_secs"] = time.time() - t0
    log(f"  GATE TB (two chunk orders, max rel diff): {gTB:.2e}  "
        f"[{time.time()-t0:.2f}s, T shape {T1.shape}]")
    T = T1
    Q = tc.symmetrize(T)
    if QOUT:
        np.save(QOUT, Q)
    Qj = jnp.asarray(Q)
    absT = np.abs(T)

    # ---------------- decoded training states -------------------------------
    Z_tr = np.asarray(su.Z_tr)
    H = np.asarray(su.h_fn(jnp.asarray(Z_tr)))                 # (S, R)
    U = H @ G.T                                                # (S, n_i)
    upw = jax.jit(jax.vmap(lambda u: b1.upwind_adv_field_1d(u, N)))
    Nup = np.asarray(upw(jnp.asarray(U)))
    DmU = tc.backward_diff_bank(U.T, dx).T                     # (S, n_i)
    q_or = Nup @ Phi                                           # (S, M)
    q_alg = (U * DmU) @ Phi
    q_T = np.asarray(jax.jit(lambda Hh: 0.5 * jnp.einsum(
        "ijk,sj,sk->si", Qj, Hh, Hh))(jnp.asarray(H)))
    nq = np.linalg.norm(q_or, axis=1) + 1e-300
    gTA = float(np.max(np.linalg.norm(q_T - q_alg, axis=1) / nq))
    rep["gates"]["TA_algebraic_identity_max_rel"] = gTA
    log(f"  GATE TA (h^T T h == Phi^T(u D^-u), all {len(H)} training "
        f"states): {gTA:.2e}")

    minu = U.min(axis=1)
    pos_states = minu > 0
    mis = np.linalg.norm(q_T - q_or, axis=1) / nq
    rep["TS_train_states"] = dict(
        mismatch_rel=stats(mis),
        frac_points_u_le_0=float(np.mean(U <= 0)),
        frac_states_all_positive=float(np.mean(pos_states)),
        n_states_all_positive=int(np.sum(pos_states)),
        min_u=float(U.min()),
        min_u_per_state=stats(minu))
    if np.any(pos_states):
        gT0 = float(np.max(mis[pos_states]))
    else:
        gT0 = None
    # pointwise: stencils coincide on u>0 points (the literal 'u>0 points')
    mask = U > 0
    pw = np.max(np.abs(Nup[mask] - (U * DmU)[mask])) / np.max(np.abs(Nup))
    rep["gates"]["T0_all_positive_states_max_rel"] = gT0
    rep["gates"]["T0_pointwise_u_gt_0_max_rel"] = float(pw)
    log(f"  GATE T0 (tensor vs oracle on the {int(np.sum(pos_states))} "
        f"all-positive states): "
        f"{'n/a' if gT0 is None else f'{gT0:.2e}'} ; pointwise stencil "
        f"identity on u>0 points: {pw:.2e}")
    ts = rep["TS_train_states"]
    log(f"  TS (all training states): mismatch median {np.median(mis):.2e}"
        f" mean {np.mean(mis):.2e} max {np.max(mis):.2e}"
        f" | frac points u<=0 {ts['frac_points_u_le_0']:.3%}  min u "
        f"{ts['min_u']:.3e}  all-positive states {ts['frac_states_all_positive']:.1%}")

    # ---------------- TC: contraction condition -----------------------------
    absq = np.abs(q_T) + 1e-300
    S_abs = np.einsum("ijk,sj,sk->si", absT, np.abs(H), np.abs(H))
    ratio = S_abs / absq
    ratio_inf = S_abs / (np.max(np.abs(q_T), axis=1, keepdims=True) + 1e-300)
    rep["TC_contraction"] = dict(
        per_entry_max=float(ratio.max()), per_entry_median=float(np.median(ratio)),
        per_entry_p95=float(np.quantile(ratio, 0.95)),
        vs_qmax_max=float(ratio_inf.max()),
        vs_qmax_median=float(np.median(ratio_inf)),
        note="per_entry: sum_jk|T_ijk h_j h_k|/|q_i| (max over i, states); "
             "vs_qmax: same numerator over max_i|q_i| of that state")
    log(f"  TC contraction sum|T h h|/|q_i|: max {ratio.max():.2e} median "
        f"{np.median(ratio):.2e} p95 {np.quantile(ratio,0.95):.2e}; "
        f"vs max_i|q_i|: max {ratio_inf.max():.2e}")

    # ---------------- residual closures --------------------------------------
    r_or = su.make_full_rw()
    r_T = su.make_tensor_rw(Q)
    rJ_or = jax.jit(lambda z, p, nu: (r_or(z, p, nu),
                                      jax.jacfwd(r_or)(z, p, nu)))
    rJ_T = jax.jit(lambda z, p, nu: (r_T(z, p, nu),
                                     jax.jacfwd(r_T)(z, p, nu)))
    u_of = jax.jit(lambda z: su.G_int @ su.h_fn(z))

    # ---------------- TJ: Jacobian mismatch at 32+32 states ------------------
    rng = np.random.default_rng(fc.SEED0 + 500)
    rows = []
    for si in range(2 * N_JAC):
        i = rng.integers(len(Z_tr))
        z = Z_tr[i] + (0.05 * rng.standard_normal(fc.K) if si >= N_JAC
                       else 0.0)
        zp = Z_tr[rng.integers(len(Z_tr))]
        nu = float(np.exp(rng.uniform(np.log(0.01), np.log(0.1))))
        pv = su.prev_of(jnp.asarray(zp))
        ro, Jo = [np.asarray(v) for v in rJ_or(jnp.asarray(z), pv, nu)]
        rt, Jt = [np.asarray(v) for v in rJ_T(jnp.asarray(z), pv, nu)]
        uu = np.asarray(u_of(jnp.asarray(z)))
        rows.append(dict(perturbed=bool(si >= N_JAC), nu=nu,
                         r_rel=rel(rt, ro), J_rel=rel(Jt, Jo),
                         g_rel=rel(Jt.T @ rt, Jo.T @ ro),
                         g_cos=cosine(Jt.T @ rt, Jo.T @ ro),
                         min_u=float(uu.min()),
                         n_neg=int(np.sum(uu <= 0))))
    for lab, sel in (("unperturbed", [r_ for r_ in rows if not r_["perturbed"]]),
                     ("perturbed_0.05", [r_ for r_ in rows if r_["perturbed"]])):
        d = dict(r_rel=stats([r_["r_rel"] for r_ in sel]),
                 J_rel=stats([r_["J_rel"] for r_ in sel]),
                 g_rel=stats([r_["g_rel"] for r_ in sel]),
                 g_cos_min=float(min(r_["g_cos"] for r_ in sel)),
                 min_u=float(min(r_["min_u"] for r_ in sel)),
                 frac_states_with_neg=float(np.mean([r_["n_neg"] > 0
                                                     for r_ in sel])))
        rep[f"TJ_{lab}"] = d
        log(f"  TJ [{lab}] r rel med {d['r_rel']['median']:.2e} max "
            f"{d['r_rel']['max']:.2e} | J rel med {d['J_rel']['median']:.2e} "
            f"max {d['J_rel']['max']:.2e} | grad cos min {d['g_cos_min']:.6f}"
            f" | min u {d['min_u']:.2e}")
    rep["TJ_rows"] = rows

    # ---------------- TL: oracle host-loop rollout, every LM candidate -------
    U_test, nu_test = fc.gen_test(N)
    interior = su.interior
    Tn = b1.NUM_STEPS + 1
    ic_ref = fc.make_ic_ref(su)
    ops_or = fc.make_device_ref(su, r_or)
    rn_or = ops_or["rn"]
    TR_DELTA = su.TR_DELTA
    cand_rows = []
    traj_rows = []
    for ti in range(fc.N_TEST):
        nu = float(nu_test[ti])
        u0 = U_test[ti, 0]
        u0i = jnp.asarray(u0[interior])
        u_scale = float(np.sqrt(np.mean(u0[interior] ** 2)))
        tol_abs = fc.STEP_TOL * u_scale * float(np.sqrt(su.n_i))
        z0, v0 = ic_ref(u0i)
        Zd, rnd, nJd, red = ops_or["rollout"](z0, nu, tol_abs, fc.GN_BUDGET)
        Zd = np.asarray(Zd)
        zs = [np.asarray(z0)]
        errs = [rel(np.asarray(u_of(z0)), u0[interior])]
        n_cand = 0

        def probe(z, prev_c, accepted, step, kind):
            ro, Jo = [np.asarray(v) for v in rJ_or(z, prev_c, nu)]
            rt, Jt = [np.asarray(v) for v in rJ_T(z, prev_c, nu)]
            uu = np.asarray(u_of(z))
            # advection term alone
            hz = np.asarray(su.h_fn(z))
            qo = Phi.T @ np.asarray(b1.upwind_adv_field_1d(jnp.asarray(uu), N))
            qt = tc.q_of(Q, hz)
            cand_rows.append(dict(
                traj=ti, step=step, kind=kind, accepted=bool(accepted),
                min_u=float(uu.min()), n_neg=int(np.sum(uu <= 0)),
                rn_over_tol=float(np.linalg.norm(ro) / tol_abs),
                q_rel=rel(qt, qo), r_rel=rel(rt, ro), J_rel=rel(Jt, Jo),
                g_cos=cosine(Jt.T @ rt, Jo.T @ ro),
                g_scaled=float(np.linalg.norm(Jt.T @ rt - Jo.T @ ro)
                               / (np.linalg.norm(Jo) * np.linalg.norm(ro)
                                  + 1e-300)),
                g_stationarity=float(np.linalg.norm(Jo.T @ ro)
                                     / (np.linalg.norm(Jo)
                                        * np.linalg.norm(ro) + 1e-300))))

        for t in range(1, Tn):
            z_prev = zs[-1]
            z_init = z_prev if len(zs) < 2 else \
                z_prev + fc.EXTRAP * (zs[-1] - zs[-2])
            prev_c = su.prev_of(jnp.asarray(z_prev, dtype=F64))
            z = jnp.asarray(z_init, dtype=F64)
            r, J = ops_or["rJ"](z, prev_c, nu)
            probe(z, prev_c, True, t, "init")
            rn = float(jnp.linalg.norm(r))
            lam_lm = 1e-6
            if not np.isfinite(rn) or rn <= tol_abs:
                zs.append(np.asarray(z))
                errs.append(rel(np.asarray(u_of(z)), U_test[ti, t][interior]))
                continue
            for attempt in range(1, fc.GN_BUDGET + 1):
                Hm = J.T @ J
                g = J.T @ r
                D = jnp.diag(jnp.diag(Hm)) + 1e-30 * jnp.eye(fc.K, dtype=F64)
                dz = jnp.linalg.solve(Hm + lam_lm * D, -g)
                if not bool(jnp.all(jnp.isfinite(dz))):
                    lam_lm = min(lam_lm * 10.0, 1e12)
                    if lam_lm >= 1e12:
                        break
                    continue
                ndz = float(jnp.linalg.norm(dz))
                if ndz > TR_DELTA:
                    lam_lm = min(lam_lm * 10.0, 1e12)
                    if lam_lm >= 1e12:
                        break
                    continue
                if ndz <= 1e-12 * (1.0 + float(jnp.linalg.norm(z))):
                    break
                z_new = z + dz
                rn_new = float(rn_or(z_new, prev_c, nu))
                acc = bool(np.isfinite(rn_new) and rn_new < rn)
                n_cand += 1
                probe(z_new, prev_c, acc, t, "cand")
                if acc:
                    rel_dec = (rn - rn_new) / rn
                    z, rn = z_new, rn_new
                    if rn <= tol_abs:
                        break
                    r, J = ops_or["rJ"](z, prev_c, nu)
                    lam_lm = max(lam_lm / 3.0, 1e-12)
                    if rel_dec < fc.STALL:
                        break
                else:
                    lam_lm = min(lam_lm * 10.0, 1e12)
                    if lam_lm >= 1e12:
                        break
            zs.append(np.asarray(z))
            errs.append(rel(np.asarray(u_of(jnp.asarray(zs[-1]))),
                            U_test[ti, t][interior]))
        Zh = np.stack(zs[1:])
        latdev = float(np.max(np.abs(Zh - Zd)))
        rows_t = [c for c in cand_rows if c["traj"] == ti]
        traj_rows.append(dict(
            traj=ti, nu=nu, err_host=float(np.mean(errs)),
            latdev_host_vs_device=latdev, n_candidates=n_cand,
            n_cand_with_neg=int(sum(1 for c in rows_t
                                    if c["kind"] == "cand" and c["n_neg"] > 0)),
            n_accepted_with_neg=int(sum(1 for c in rows_t
                                        if c["kind"] == "cand" and c["accepted"]
                                        and c["n_neg"] > 0)),
            n_rejected=int(sum(1 for c in rows_t
                               if c["kind"] == "cand" and not c["accepted"])),
            min_u=float(min(c["min_u"] for c in rows_t)),
            r_rel_max=float(max(c["r_rel"] for c in rows_t)),
            q_rel_max=float(max(c["q_rel"] for c in rows_t)),
            g_cos_min=float(min(c["g_cos"] for c in rows_t)),
            g_cos_min_nonstationary=float(min(
                [c["g_cos"] for c in rows_t if c["g_stationarity"] > 1e-2]
                or [float("nan")])),
            g_scaled_max=float(max(c["g_scaled"] for c in rows_t))))
        tr = traj_rows[-1]
        log(f"  TL traj {ti}: host err {tr['err_host']:.6e} latdev(host vs "
            f"device) {latdev:.1e} | cands {n_cand} (rejected "
            f"{tr['n_rejected']}) with u<=0: {tr['n_cand_with_neg']} "
            f"(accepted {tr['n_accepted_with_neg']}) | min u {tr['min_u']:.2e}"
            f" | r rel max {tr['r_rel_max']:.2e} q rel max "
            f"{tr['q_rel_max']:.2e} grad cos min {tr['g_cos_min']:.6f}")
    allc = [c for c in cand_rows if c["kind"] == "cand"]
    alli = [c for c in cand_rows if c["kind"] == "init"]
    rep["TL_candidates"] = dict(
        per_traj=traj_rows,
        n_candidates=len(allc),
        n_rejected=int(sum(1 for c in allc if not c["accepted"])),
        n_with_neg=int(sum(1 for c in allc if c["n_neg"] > 0)),
        n_accepted_with_neg=int(sum(1 for c in allc
                                    if c["accepted"] and c["n_neg"] > 0)),
        n_init_with_neg=int(sum(1 for c in alli if c["n_neg"] > 0)),
        n_init=len(alli),
        min_u=float(min(c["min_u"] for c in cand_rows)),
        q_rel=stats([c["q_rel"] for c in cand_rows]),
        r_rel=stats([c["r_rel"] for c in cand_rows]),
        J_rel=stats([c["J_rel"] for c in cand_rows]),
        g_cos_min=float(min(c["g_cos"] for c in cand_rows)),
        g_cos_min_nonstationary=float(min(
            [c["g_cos"] for c in cand_rows if c["g_stationarity"] > 1e-2]
            or [float("nan")])),
        n_nonstationary=int(sum(1 for c in cand_rows
                                if c["g_stationarity"] > 1e-2)),
        g_scaled=stats([c["g_scaled"] for c in cand_rows]),
        g_cos_note="the LM stalls (reason 2) at least-squares stationary "
                   "points where J^T r ~ 0 while |r| >> tol; the cosine of "
                   "a vanishing gradient carries no information there. "
                   "g_scaled = |g_T - g_or| / (|J_or|_F |r_or|); "
                   "g_cos_min_nonstationary restricts to candidates with "
                   "|J^T r| / (|J| |r|) > 1e-2",
        latdev_host_vs_device_max=float(max(t_["latdev_host_vs_device"]
                                            for t_ in traj_rows)),
        rows=cand_rows)
    tl = rep["TL_candidates"]
    log(f"  TL ALL: {tl['n_candidates']} LM candidates ({tl['n_rejected']} "
        f"rejected) + {tl['n_init']} inits; with u<=0 somewhere: "
        f"{tl['n_with_neg']} cands ({tl['n_accepted_with_neg']} accepted), "
        f"{tl['n_init_with_neg']} inits; min u {tl['min_u']:.2e}; tensor-vs-"
        f"oracle q rel median {tl['q_rel']['median']:.2e} max "
        f"{tl['q_rel']['max']:.2e}; r rel median {tl['r_rel']['median']:.2e} "
        f"max {tl['r_rel']['max']:.2e}; J rel max {tl['J_rel']['max']:.2e}; "
        f"grad cos min {tl['g_cos_min']:.6f} (non-stationary "
        f"{tl['n_nonstationary']} cands: {tl['g_cos_min_nonstationary']:.6f}"
        f"; scaled grad mismatch max {tl['g_scaled']['max']:.2e}); "
        f"host-vs-device latdev "
        f"{tl['latdev_host_vs_device_max']:.1e}")
    rep["secs_total"] = time.time() - t_all
    json.dump(rep, open(OUT, "w"), indent=1, default=float)
    log(f"DONE -> {OUT} [{time.time()-t_all:.0f}s]")


if __name__ == "__main__":
    main()
