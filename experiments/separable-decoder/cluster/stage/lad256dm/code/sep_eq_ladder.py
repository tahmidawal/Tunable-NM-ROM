"""EQ FIDELITY LADDER -- how much does the empirical quadrature distort what the
solver consumes?  Trains nothing.  Loads a committed checkpoint, fits the
usual EQ sets, and compares every solver quantity computed with the SAMPLED
quadrature (m nodes, NNLS weights -- what runs online) against the same
quantity computed EXACTLY on the full interior grid (what the quadrature
approximates).  Design doc 2026-08-22 section 6.1, rungs (b)/(c1)/(c2)/(c3);
meeting 2026-08-25 ("L2 on the points, the integral, the gradient, the
Hessian -- measured separately").

Rungs, per evaluated state (z, prev, nu, lambda):

  (a)  recon_pts / recon_full  -- decoder-vs-truth RMS error on the m nodes
                                  (quadrature-weighted) vs on the full grid.
                                  Oracle states only (needs a truth field).
  (b)  ||R_s - R_f|| / ||R_f||  -- the weak residual ("is the integral right")
       split into its linear part (mass, previous state, Laplacian: all of
       the form Phi^T u) and its advection part (Phi^T N(u)), each relative
       to ||R_f||.  The linear part could be made EXACT by precomputing
       Phi^T G; the advection part cannot (sign-upwind).
  (c1) ||g_s - g_f|| / ||g_f||,  g = J^T R  -- the objective gradient, plus
       its cosine, plus an ABSOLUTE version ||g_s - g_f|| / (||J_f||_F ||R_f||)
       that does not blow up near a stationary point.
  (c2) ||H_s - H_f||_F / ||H_f||_F,  H = J^T J  -- the GN normal operator,
       plus the relative error of H_s v for v = the full-grid step direction.
  (c3) ||dz_s - dz_f|| / ||dz_f||   -- the damped LM step at the SAME lambda
       and the same diagonal scaling the solver uses, plus its cosine.

States evaluated:

  solver-path : every LM iterate (z, lambda) of a real ROM rollout of each
                test trajectory on the SAMPLED ops (the online solver, with
                the round-4 optimized settings: Gram/adaptive-stall not needed
                here, we use the incumbent LM rule with a configurable stall),
                bucketed by time index.  Off-manifold, exactly where the
                solver goes.  The previous state is the ROM's own previous
                latent, as online.
  oracle      : the full-grid least-squares code of each TRUTH test state
                (diagnostic; truth is used only to place z on the manifold,
                never in any solve path), with prev = oracle code of the
                previous truth state.  On-manifold, at the solution.
  train-snap  : a few training codes with prev = the training code of the
                previous snapshot (same trajectory).

Everything the sampled side uses is built by the SAME code as
sep_speed_r4/r5 (copied verbatim) and gated against blat_common.make_weak_ops
at <=1e-12 (gate 0).  The full-grid side is gated against make_weak_ops with
idx=interior, w=None (exact grid sums) at <=1e-12 wherever the grid is small
enough to run that reference (N<=512); at N=1024 the identity is the same
code path (bank product + upwind_adv_field, both already gated elsewhere).

Banks and Phi are explicit jit ARGUMENTS everywhere (captured-constant
landmine, CLAUDE.md).
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

import blat_common as bc                     # noqa: E402
import ctol_eq                                # noqa: E402
import ctol_tol                               # noqa: E402
from sep_burgers_r3 import build_test_full    # noqa: E402

F64 = jnp.float64

CKPT = os.environ["CKPT"]
N = int(os.environ.get("N", "256"))
N_TEST = int(os.environ.get("N_TEST", "4"))
EQ_MS = [int(v) for v in os.environ.get("EQ_MS", "64,256").split(",")]
EQ_M_FACTOR = int(os.environ.get("EQ_M_FACTOR", "4"))
EQ_CAND_CAP = int(os.environ.get("EQ_CAND_CAP", "65536"))
EQ_TAIL = int(os.environ.get("EQ_TAIL", "0"))
EQ_TAIL_CAP = float(os.environ.get("EQ_TAIL_CAP", "3e-2"))
EQ_TAIL_ROUNDS = int(os.environ.get("EQ_TAIL_ROUNDS", "3"))
SEED0 = int(os.environ.get("SEED0", "0"))
TR_FACTOR = float(os.environ.get("TR_FACTOR", "0.01"))
STEP_TOL = float(os.environ.get("STEP_TOL", "1e-9"))
STALL = float(os.environ.get("STALL", "1e-3"))
EXTRAP = float(os.environ.get("EXTRAP", "1.0"))
IC_BUDGET = int(os.environ.get("IC_BUDGET", "100"))
ORACLE_TS = [int(v) for v in os.environ.get(
    "ORACLE_TS", "0,1,2,3,5,10,25,50").split(",")]
N_TRAIN_SNAP = int(os.environ.get("N_TRAIN_SNAP", "16"))
FEAT_CHUNK = int(os.environ.get("FEAT_CHUNK", "0"))
FULL_GATE_MAX_N = int(os.environ.get("FULL_GATE_MAX_N", "512"))
OUT_PREFIX = os.environ.get("OUT_PREFIX", "")
OUT_TAG = os.environ.get("OUT_TAG", "")

T_BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 9), (10, 24), (25, 50)]


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def main():
    dev = jax.devices()[0]
    sc.log(f"jax_backend={dev.platform} device={dev} EQ-LADDER N={N} "
           f"ckpt={os.path.basename(CKPT)}")
    t_all = time.time()
    params, Z_tr, cfg = sc.load_pkl(CKPT)
    K, R = int(cfg["k"]), int(cfg["r"])
    assert int(cfg["N"]) == N, f"ckpt N={cfg['N']} != N={N}"
    dec = sc.SeparableDecoder(params, K, R)
    h_fn = dec.head_fn()
    tag = OUT_TAG or f"N{N}_K{K}_R{R}"
    OUT = f"{OUT_PREFIX}sep_eq_ladder_{tag}.json"

    interior = bc.interior_indices(N)
    coords = np.asarray(bc.grid_coords(N))
    n_i2 = interior.size
    T = bc.NUM_STEPS + 1
    dx = 1.0 / (N - 1)
    feat_chunk = FEAT_CHUNK or (0 if N <= 512 else 131072)

    report = dict(config=dict(
        pde="burgers2d", kind="eq_fidelity_ladder", N=N, k=K, r=R,
        ckpt=os.path.basename(CKPT), ckpt_cfg=cfg, n_test=N_TEST, eq_Ms=EQ_MS,
        eq_m_factor=EQ_M_FACTOR, eq_tail=bool(EQ_TAIL), eq_tail_cap=EQ_TAIL_CAP,
        eq_tail_rounds=EQ_TAIL_ROUNDS, step_tol=STEP_TOL, stall=STALL,
        extrap=EXTRAP, ic_budget=IC_BUDGET, gn_budget=bc.GN_BUDGET,
        oracle_ts=ORACLE_TS, n_train_snap=N_TRAIN_SNAP, num_steps=bc.NUM_STEPS,
        dt=bc.DT, weak_alpha=bc.WEAK_ALPHA, seed=SEED0, test_seed=bc.TEST_SEED,
        x64=True, matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        backend=dev.platform, gpu=getattr(dev, "device_kind", str(dev)),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME", "local")),
        gates={}, eq={}, records=[], rollout=[], complete=False)

    def save():
        json.dump(report, open(OUT, "w"), indent=1, default=float)

    U_test, nu_test, worst_res_test = build_test_full(N, N_TEST, sc.log)
    report["data"] = dict(n_test=int(N_TEST),
                          max_fom_rel_residual_test=worst_res_test)
    save()

    train_radius = float(np.max(np.linalg.norm(Z_tr - Z_tr.mean(0), axis=1)))
    bc.TR_DELTA = (TR_FACTOR * train_radius) if TR_FACTOR > 0 else np.inf
    report["config"]["trust_delta"] = float(bc.TR_DELTA)
    zbar = Z_tr.mean(0)

    G_all = dec.feat_at(coords, chunk=feat_chunk)
    interior_j = jnp.asarray(interior)
    G_int = G_all[interior_j]
    coords_int = coords[interior]
    del G_all

    # ---------------- full-grid reference ops (exact grid sums) ------------
    # Same weak residual as make_weak_ops(kind='weak') with idx=interior and
    # unit weights, written on the cached bank so that N=1024 fits in memory:
    #   u = G_int h(z); N(u) by the FOM upwind stencil; Phi^T(.) exact.
    # Phi and G_int are explicit arguments.
    adv_full = jax.jit(lambda uf: bc.upwind_adv_field(uf, N))

    def make_full(Mi):
        kx, ky, Phi, lam, _ = bc.test_modes(N, Mi)
        Phi_j = jnp.asarray(Phi)
        lam_j = jnp.asarray(lam, dtype=F64)

        def parts(Gb, Ph, z, prev_full, nu):
            """Returns (R_f, lin, adv) with R_f = wt*(lin + DT*adv)."""
            w_ = (1.0 + bc.DT * nu * lam_j) ** (-bc.WEAK_ALPHA)
            u = Gb @ h_fn(z)
            Nu = bc.upwind_adv_field(u, N)
            pu = Ph.T @ u
            lin = Ph.T @ (u - prev_full) + bc.DT * nu * lam_j * pu
            adv = Ph.T @ Nu
            return w_ * (lin + bc.DT * adv), w_ * lin, w_ * bc.DT * adv

        def r_f(Gb, Ph, z, prev_full, nu):
            return parts(Gb, Ph, z, prev_full, nu)[0]

        def rJ_f(Gb, Ph, z, prev_full, nu):
            R, lin, adv = parts(Gb, Ph, z, prev_full, nu)
            J = jax.jacfwd(r_f, argnums=2)(Gb, Ph, z, prev_full, nu)
            return R, J, lin, adv

        return dict(Phi=Phi_j, lam=lam_j, Phi_np=np.asarray(Phi),
                    rJ=jax.jit(rJ_f), r=jax.jit(r_f), M=Mi)

    u_full_j = jax.jit(lambda Gb, z: Gb @ h_fn(z))

    # ---------------- EQ sets + sampled ops (verbatim sep_speed_r5) ---------
    rng = np.random.default_rng(SEED0)
    eq_pick = np.sort(rng.choice(len(Z_tr), min(64, len(Z_tr)), replace=False))
    Z_eq = Z_tr[eq_pick]
    cand_pos = ctol_eq.candidate_pool(n_i2, cap=EQ_CAND_CAP)
    u_full_int = lambda z: u_full_j(G_int, z)
    _zb = jnp.asarray(Z_tr[eq_pick[0]])
    _a, _b = u_full_int(_zb), dec(_zb, jnp.asarray(coords_int))
    dv = float(jnp.max(jnp.abs(_a - _b)) / (jnp.max(jnp.abs(_b)) + 1e-300))
    report["gates"]["eq_bank_vs_meshfree"] = dv
    sc.log(f"  EQ-fit bank vs meshfree full-interior decode: {dv:.2e}")
    assert dv < 1e-12
    del _a, _b

    eq_ops = {}
    full_ops = {}
    for Mi in EQ_MS:
        name = "ctrl" if Mi == EQ_MS[0] else f"M{Mi}"
        kx, ky, Phi, lam, _ = bc.test_modes(N, Mi)
        m_want = EQ_M_FACTOR * Mi
        lbl = f"eq-ladder N={N} k={K} M={Mi} m={m_want}"
        if EQ_TAIL:
            G_eq, b_eq, pad_sc = ss.build_eq_system_burgers(
                u_full_int, adv_full, np.asarray(Phi), cand_pos, Z_eq)
            keep, wq_np, eq_info = ss.tail_reweight_fit(
                G_eq, b_eq, m_want, bc.nnls_capped, SEED0 + 900,
                cap=EQ_TAIL_CAP, rounds=EQ_TAIL_ROUNDS, pad_score=pad_sc,
                label=lbl)
            del G_eq, b_eq
        else:
            keep, wq_np, eq_info = ctol_eq.eq_fit_burgers(
                u_full_int, adv_full, np.asarray(Phi), cand_pos, Z_eq, K,
                m_want, lbl, bc.nnls_capped)
        cl = dict(kind="grid", idx=interior[cand_pos[keep]], w=wq_np,
                  info=eq_info)
        idx = np.asarray(cl["idx"])
        m = idx.size
        w_q = jnp.asarray(cl["w"], dtype=F64)
        pos = np.searchsorted(interior, idx)
        assert np.all(interior[pos] == idx)
        Phi_q = jnp.asarray(np.asarray(Phi)[pos]) * w_q[:, None]
        lam_j = jnp.asarray(lam, dtype=F64)
        st = bc.stencil_indices(idx, N)
        G_st = dec.feat_at(coords[st.reshape(-1)]).reshape(m, 5, dec.r)
        del Phi

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

            def parts_s(z, prev_c, nu):
                w_ = (1.0 + bc.DT * nu * lam_j) ** (-bc.WEAK_ALPHA)
                u, Nu = u_and_N_fast(z)
                pu = Phi_q.T @ u
                lin = Phi_q.T @ (u - prev_c) + bc.DT * nu * lam_j * pu
                adv = Phi_q.T @ Nu
                return w_ * (lin + bc.DT * adv), w_ * lin, w_ * bc.DT * adv

            def r_w_fast(z, prev_c, nu):
                return parts_s(z, prev_c, nu)[0]

            def d_c_fast(z):
                return u_and_N_fast(z)[0]

            def rJ_fast(z, prev_c, nu):
                return (r_w_fast(z, prev_c, nu),
                        jax.jacfwd(r_w_fast)(z, prev_c, nu),
                        Phi_q.T @ jax.jacfwd(d_c_fast)(z))

            def rJ_parts(z, prev_c, nu):
                R, lin, adv = parts_s(z, prev_c, nu)
                return R, jax.jacfwd(r_w_fast)(z, prev_c, nu), lin, adv

            def full_fast(z):
                return G_st[:, 0, :] @ h_fn(z)
            return (r_w_fast, rJ_fast, prev_of_fast, full_fast, rJ_parts)
        (r_w_f, rJ_f, prev_f, full_f, rJp_f) = mk()
        ops_fast = bc._finish_ops(rJ_f, r_w_f, prev_f, full_f, m, "lspg")
        ops_fast["M"] = Mi
        ops_fast["tol_scale"] = float(np.sqrt(n_i2))
        ops_ref = bc.make_weak_ops(dec, N, cl, kind="weak", M=Mi, solver="lspg")
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
            g0.append(max(
                float(jnp.max(jnp.abs(ra - rb)) / (jnp.max(jnp.abs(ra)) + 1e-300)),
                float(jnp.max(jnp.abs(Ja - Jb)) / (jnp.max(jnp.abs(Ja)) + 1e-300))))
        gate0 = float(np.max(g0))
        sc.log(f"  GATE 0 [{name}]: {gate0:.2e}")
        assert gate0 < 1e-12
        info_rep = {k_: v for k_, v in eq_info.items()
                    if isinstance(v, (int, float, str, bool, type(None)))}
        info_rep["gate0"] = gate0
        info_rep["m"] = int(m)
        report["eq"][name] = info_rep

        fo = make_full(Mi)
        # gate F: full-grid reference == make_weak_ops on the whole interior
        if N <= FULL_GATE_MAX_N:
            cl_full = dict(kind="grid", idx=interior, w=None)
            ops_full_ref = bc.make_weak_ops(dec, N, cl_full, kind="weak",
                                            M=Mi, solver="lspg")
            gF = []
            for _ in range(3):
                zt = jnp.asarray(Z_tr[grng.integers(len(Z_tr))]
                                 + 0.05 * grng.standard_normal(K))
                zp = jnp.asarray(Z_tr[grng.integers(len(Z_tr))])
                nu = float(np.median(nu_test))
                pf = u_full_j(G_int, zp)
                ra, Ja, _ = ops_full_ref["rJ"](zt, ops_full_ref["prev_of"](zp), nu)
                rb, Jb, _, _ = fo["rJ"](G_int, fo["Phi"], zt, pf, nu)
                gF.append(max(
                    float(jnp.max(jnp.abs(ra - rb)) / (jnp.max(jnp.abs(ra)) + 1e-300)),
                    float(jnp.max(jnp.abs(Ja - Jb)) / (jnp.max(jnp.abs(Ja)) + 1e-300))))
            gateF = float(np.max(gF))
            sc.log(f"  GATE F (full-grid ref vs make_weak_ops on interior) "
                   f"[{name}]: {gateF:.2e}")
            assert gateF < 1e-12
            report["eq"][name]["gateF"] = gateF
            del ops_full_ref
        else:
            report["eq"][name]["gateF"] = None
            sc.log(f"  GATE F skipped at N={N} (> FULL_GATE_MAX_N)")
        eq_ops[name] = dict(ops_fast=ops_fast, idx=idx, pos=pos, m=m,
                            rJ_parts=jax.jit(rJp_f), prev=jax.jit(prev_f),
                            w=np.asarray(wq_np), lam=lam_j)
        full_ops[name] = fo
        save()

    # ---------------- ladder at one state ------------------------------------
    def ladder(name, z, z_prev, nu, lam_lm, u_true=None):
        e = eq_ops[name]
        fo = full_ops[name]
        zj = jnp.asarray(z, dtype=F64)
        zpj = jnp.asarray(z_prev, dtype=F64)
        prev_full = u_full_j(G_int, zpj)
        prev_c = e["prev"](zpj)
        Rs, Js, lin_s, adv_s = [np.asarray(v) for v in e["rJ_parts"](zj, prev_c, nu)]
        Rf, Jf, lin_f, adv_f = [np.asarray(v) for v in
                                fo["rJ"](G_int, fo["Phi"], zj, prev_full, nu)]
        gs, gf = Js.T @ Rs, Jf.T @ Rf
        Hs, Hf = Js.T @ Js, Jf.T @ Jf
        Ds = np.diag(np.diag(Hs)) + 1e-30 * np.eye(K)
        Df = np.diag(np.diag(Hf)) + 1e-30 * np.eye(K)
        dzs = np.linalg.solve(Hs + lam_lm * Ds, -gs)
        dzf = np.linalg.solve(Hf + lam_lm * Df, -gf)
        # GN (undamped) step too, lambda-independent
        dzs0 = np.linalg.lstsq(Js, -Rs, rcond=None)[0]
        dzf0 = np.linalg.lstsq(Jf, -Rf, rcond=None)[0]
        nRf = np.linalg.norm(Rf)
        rec = dict(
            eq=name, nu=float(nu), lam=float(lam_lm),
            R_f_norm=float(nRf), R_s_norm=float(np.linalg.norm(Rs)),
            b_resid=rel(Rs, Rf),
            b_lin=float(np.linalg.norm(lin_s - lin_f) / (nRf + 1e-300)),
            b_adv=float(np.linalg.norm(adv_s - adv_f) / (nRf + 1e-300)),
            b_lin_self=rel(lin_s, lin_f), b_adv_self=rel(adv_s, adv_f),
            J_rel=float(np.linalg.norm(Js - Jf) / (np.linalg.norm(Jf) + 1e-300)),
            c1_grad=rel(gs, gf), c1_cos=cosine(gs, gf),
            c1_abs=float(np.linalg.norm(gs - gf)
                         / (np.linalg.norm(Jf) * nRf + 1e-300)),
            g_f_norm=float(np.linalg.norm(gf)),
            c2_hess=float(np.linalg.norm(Hs - Hf) / (np.linalg.norm(Hf) + 1e-300)),
            c2_hv=rel(Hs @ dzf, Hf @ dzf),
            c3_step=rel(dzs, dzf), c3_cos=cosine(dzs, dzf),
            c3_gn=rel(dzs0, dzf0), c3_gn_cos=cosine(dzs0, dzf0),
            dz_f_norm=float(np.linalg.norm(dzf)),
        )
        if u_true is not None:
            uf = np.asarray(u_full_j(G_int, zj))
            ut = np.asarray(u_true)[interior]
            err_full = np.linalg.norm(uf - ut) / (np.linalg.norm(ut) + 1e-300)
            w = e["w"]; p = e["pos"]
            d = (uf[p] - ut[p])
            err_pts = np.sqrt((w * d * d).sum() / ((w * ut[p] * ut[p]).sum() + 1e-300))
            err_pts_unw = np.linalg.norm(d) / (np.linalg.norm(ut[p]) + 1e-300)
            rec.update(a_recon_full=float(err_full), a_recon_pts=float(err_pts),
                       a_recon_pts_unweighted=float(err_pts_unw))
        return rec

    # ---------------- oracle codes of the truth test states ------------------
    oracle = ss.make_oracle_lm_banked(h_fn, K, budget=200)
    orng = np.random.default_rng(SEED0 + 11)

    def oracle_codes(targets_int, n_init=8):
        B = targets_int.shape[0]
        tj = jnp.asarray(targets_int, dtype=F64)
        z0_sets = [np.tile(zbar[None], (B, 1))]
        for _ in range(n_init):
            z0_sets.append(Z_tr[orng.integers(len(Z_tr), size=B)])
        z, v = ss.oracle_multi_init_banked(oracle, G_int, z0_sets, tj)
        return np.asarray(z), np.asarray(v)

    # ---------------- IC fit (no encoder; multi-init full-grid LM on u0) -----
    # u0 is the given initial condition, not a solution -- allowed in a solve
    # path.  Inits: zbar + 8 random training codes; best residual wins.
    def ic_fit(u0_int):
        z, v = oracle_codes(u0_int[None], n_init=8)
        return z[0], float(v[0])

    # ---------------- host LM loop that records every iterate ----------------
    def lm_step_recording(name, z0, z_prev, nu, tol_abs, budget, t_idx, traj):
        """solve_step's LSPG rule (lam0=1e-6, /3 accept, x10 reject, trust
        radius, stall) with a configurable stall threshold, evaluating the
        ladder at EVERY iterate (z, lam) before the step is taken."""
        e = eq_ops[name]
        ops = e["ops_fast"]
        prev_c = e["prev"](jnp.asarray(z_prev, dtype=F64))
        z = jnp.asarray(z0, dtype=F64)
        r, J, _ = ops["rJ"](z, prev_c, nu)
        rn = float(jnp.linalg.norm(r))
        lam = 1e-6
        reason = "budget"
        n_it = 0
        for attempt in range(1, budget + 1):
            rec = ladder(name, np.asarray(z), z_prev, nu, lam)
            rec.update(kind="solver", t=int(t_idx), traj=int(traj),
                       iter=int(attempt), rn=rn)
            report["records"].append(rec)
            n_it += 1
            H = J.T @ J
            g = J.T @ r
            D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K, dtype=F64)
            dz = jnp.linalg.solve(H + lam * D, -g)
            if not bool(jnp.all(jnp.isfinite(dz))):
                lam = min(lam * 10.0, 1e12)
                if lam >= 1e12:
                    reason = "nan_step"; break
                continue
            ndz = float(jnp.linalg.norm(dz))
            if ndz > bc.TR_DELTA:
                lam = min(lam * 10.0, 1e12)
                if lam >= 1e12:
                    reason = "trust_lambda_max"; break
                continue
            if ndz <= 1e-12 * (1.0 + float(jnp.linalg.norm(z))):
                reason = "stalled"; break
            z_new = z + dz
            rn_new = float(ops["rn"](z_new, prev_c, nu))
            if np.isfinite(rn_new) and rn_new < rn:
                rel_dec = (rn - rn_new) / rn
                z, rn = z_new, rn_new
                if rn <= tol_abs:
                    reason = "tol"; break
                r, J, _ = ops["rJ"](z, prev_c, nu)
                lam = max(lam / 3.0, 1e-12)
                if rel_dec < STALL:
                    reason = "stalled"; break
            else:
                lam = min(lam * 10.0, 1e12)
                if lam >= 1e12:
                    reason = "lambda_max"; break
        return np.asarray(z), rn, reason, n_it

    # ---------------- solver-path states: real rollouts ----------------------
    for name in eq_ops:
        for ti in range(N_TEST):
            nu = float(nu_test[ti])
            u0 = U_test[ti, 0]
            z0, v0 = ic_fit(u0[interior])
            u_scale = float(np.sqrt(np.mean(u0[interior] ** 2)))
            tol_abs = STEP_TOL * u_scale * float(np.sqrt(n_i2))
            zs = [z0]
            errs, reasons, its = [], [], []
            uf = np.asarray(u_full_j(G_int, jnp.asarray(z0)))
            errs.append(float(np.linalg.norm(uf - u0[interior])
                              / np.linalg.norm(u0[interior])))
            t1 = time.time()
            for t in range(1, T):
                z_prev = zs[-1]
                z_init = z_prev if len(zs) < 2 or EXTRAP == 0 else \
                    z_prev + EXTRAP * (zs[-1] - zs[-2])
                z_new, rn, reason, n_it = lm_step_recording(
                    name, z_init, z_prev, nu, tol_abs, bc.GN_BUDGET, t, ti)
                zs.append(z_new)
                reasons.append(reason); its.append(n_it)
                uf = np.asarray(u_full_j(G_int, jnp.asarray(z_new)))
                ut = U_test[ti, t][interior]
                errs.append(float(np.linalg.norm(uf - ut) / np.linalg.norm(ut)))
            report["rollout"].append(dict(
                eq=name, traj=ti, nu=nu, ic_resid=v0, err_per_t=errs,
                err_mean=float(np.mean(errs)), reasons=reasons, iters=its,
                n_iters_total=int(np.sum(its)), secs=time.time() - t1))
            sc.log(f"  [{name}] traj {ti}: rollout err {np.mean(errs):.3e} "
                   f"(t0 {errs[0]:.2e}, t1 {errs[1]:.2e}, t50 {errs[-1]:.2e}), "
                   f"{int(np.sum(its))} LM iterates recorded "
                   f"[{time.time()-t1:.0f}s]")
            save()

    # ---------------- oracle states (on-manifold at the truth) ---------------
    for ti in range(N_TEST):
        nu = float(nu_test[ti])
        ts = [t for t in ORACLE_TS if t < T]
        need = sorted(set(ts) | set(max(t - 1, 0) for t in ts))
        tgt = np.stack([U_test[ti, t][interior] for t in need])
        zc, vc = oracle_codes(tgt, n_init=8)
        code = {t: zc[i] for i, t in enumerate(need)}
        for name in eq_ops:
            for t in ts:
                tp = max(t - 1, 0)
                rec = ladder(name, code[t], code[tp], nu, 1e-6,
                             u_true=U_test[ti, t])
                rec.update(kind="oracle", t=int(t), traj=int(ti), iter=0,
                           oracle_resid_rel=float(
                               vc[need.index(t)] / np.linalg.norm(tgt[need.index(t)])))
                report["records"].append(rec)
        sc.log(f"  oracle traj {ti}: codes at t={ts} placed")
        save()

    # ---------------- training snapshots -------------------------------------
    if N_TRAIN_SNAP > 0:
        if cfg.get("hfit_pick") is not None:
            pick = np.asarray(cfg["hfit_pick"], dtype=np.int64)
        else:
            cfg_ms = int(cfg.get("max_snaps", 16384))
            cfg_te = int(cfg.get("t_early", 5))
            n_traj = int(cfg.get("n_traj") or (bc.bf.N_TRAIN + bc.bf.N_VAL))
            rng2 = np.random.default_rng(SEED0)
            tidx_of = np.arange(n_traj * T) % T
            early = np.nonzero(tidx_of <= cfg_te)[0]
            rest = np.nonzero(tidx_of > cfg_te)[0]
            if early.size >= cfg_ms:
                pick = np.sort(rng2.choice(early, cfg_ms, replace=False))
            else:
                extra = rng2.choice(rest, min(cfg_ms - early.size, rest.size),
                                    replace=False)
                pick = np.sort(np.concatenate([early, extra]))
        if pick.size == len(Z_tr):
            pos_of = {int(s): i for i, s in enumerate(pick)}
            cands = [i for i, s in enumerate(pick)
                     if (s % T) >= 1 and int(s - 1) in pos_of]
            srng = np.random.default_rng(SEED0 + 23)
            chosen = srng.choice(cands, min(N_TRAIN_SNAP, len(cands)),
                                 replace=False)
            cx, cy, w_, a_, nu_all, _ = bc.bf.sample_params(seed=bc.SEED)
            ex_seed = int(cfg.get("hfit_extra_seed", 0) or 0)
            ex_traj = int(cfg.get("hfit_extra_traj", 0) or 0)
            if ex_seed:
                exd = bc.bf.sample_params(seed=ex_seed, m=ex_traj)
                nu_all = np.concatenate([nu_all, exd[4]])
            for i in chosen:
                s = int(pick[i]); tr = s // T; t = s % T
                nu = float(nu_all[tr])
                for name in eq_ops:
                    rec = ladder(name, Z_tr[i], Z_tr[pos_of[s - 1]], nu, 1e-6)
                    rec.update(kind="train", t=int(t), traj=int(tr), iter=0)
                    report["records"].append(rec)
            sc.log(f"  training snapshots: {len(chosen)} states")
        else:
            sc.log("  training snapshots skipped (pick != len(Z_tr))")
        save()

    # ---------------- summary by (eq, kind, t-bucket) ------------------------
    keys = ["b_resid", "b_lin", "b_adv", "J_rel", "c1_grad", "c1_cos", "c1_abs",
            "c2_hess", "c2_hv", "c3_step", "c3_cos", "c3_gn", "c3_gn_cos",
            "a_recon_full", "a_recon_pts"]
    summary = []
    for name in eq_ops:
        for kind in ("solver", "oracle", "train"):
            for lo, hi in T_BUCKETS:
                rs = [r for r in report["records"]
                      if r["eq"] == name and r["kind"] == kind
                      and lo <= r["t"] <= hi]
                if not rs:
                    continue
                row = dict(eq=name, kind=kind, t_lo=lo, t_hi=hi, n=len(rs))
                for k_ in keys:
                    vals = [r[k_] for r in rs if k_ in r]
                    if vals:
                        row[k_ + "_mean"] = float(np.mean(vals))
                        row[k_ + "_median"] = float(np.median(vals))
                        row[k_ + "_max"] = float(np.max(vals)) if "cos" not in k_ \
                            else float(np.min(vals))
                summary.append(row)
    report["summary"] = summary
    for row in summary:
        if row["kind"] != "solver":
            continue
        sc.log(f"  {row['eq']:5s} solver t={row['t_lo']:2d}-{row['t_hi']:2d} "
               f"n={row['n']:4d}  (b) {row['b_resid_mean']:.2e} "
               f"[lin {row['b_lin_mean']:.1e} adv {row['b_adv_mean']:.1e}]  "
               f"(c1) {row['c1_grad_mean']:.2e} cos {row['c1_cos_mean']:.4f}  "
               f"(c2) {row['c2_hess_mean']:.2e}  (c3) {row['c3_step_mean']:.2e} "
               f"cos {row['c3_cos_mean']:.4f}")
    report["complete"] = True
    report["secs_total"] = time.time() - t_all
    save()
    sc.log(f"DONE -> {OUT} [{time.time()-t_all:.0f}s]")


if __name__ == "__main__":
    main()
