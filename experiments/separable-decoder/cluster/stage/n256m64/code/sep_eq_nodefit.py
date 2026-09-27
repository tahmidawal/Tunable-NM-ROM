"""STAGE 3 -- LEARNED CONTINUOUS QUADRATURE NODES (design doc section 6.2).

Built from sep_eq_gradfit.py; the collection, row build, grad baseline and the
whole certification ladder are that driver verbatim.  New here: the m node
positions become CONTINUOUS trainable parameters (sigmoid box reparam,
min-separation penalty, fixed cardinality m), optimized by Adam against a
FROZEN full-grid teacher with the weights re-solved by the inner NNLS at the
current nodes every REFIT_EVERY steps (variable projection -- the convex
family is always the fallback).  The loss IS the ladder: rung (b) plus rung
(c1, frozen-J_f form), averaged over the same off-manifold fit states the
stage-2 rows use.  g is meshfree, so g(x_j), the dx-stencil values and the
sine modes at x_j are all differentiable in x_j; the upwind where(c>0) switch
is piecewise and its subgradient is used as-is.

Gates: gate C (at the grid init nodes the continuous machinery reproduces the
grid ops to <=1e-12) and gate L (linear part exact, node-independent); gate 0
still runs for the grid variants.  Success bar (handoff): beat the stage-2
'grad' set on held-out (c1)/(c3) AND on test rollout error at the same m --
a convex-not-beaten outcome is a reportable negative.

Variants reported: adv (stage-1), grad (stage-2 baseline), node (learned).
The learned set is saved to *_nodes.npz (X, w, X0, w0).

--- stage-2 header follows ---

STAGE 2 -- SAME-TARGET NNLS FOR ADVECTION: four quadratures, one ladder.

Built from sep_eq_ladder.py (exp/2026-08-26-eq-learned); everything after the
EQ-fit section -- the ladder rungs, solver-path recording, oracle states,
training snapshots, summary -- is that driver verbatim.  Instead of one fit at
several M, this driver fits FOUR node/weight sets at ONE fixed (M, m), all run
online with the EXACT-LINEAR residual (EXLIN):

  inc  : the incumbent two-block state fit at training codes (u and N(u) rows)
  adv  : stage-1 advection-only state fit at training codes
  path : advection rows at OFF-MANIFOLD states -- every LM iterate of real ROM
         rollouts on TRAINING trajectories (nu-quantile picks, analytic ICs,
         no test data anywhere) plus a few training codes
  grad : the path rows PLUS gradient-teacher rows with the full-grid Jacobian
         FROZEN:  J_f^T R_s(w) -> J_f^T R_f, which under EXLIN reduces to
         DT (diag(wt) J_f)^T [Phi_c^T diag(N_c) w - Phi^T N_f] = 0 -- LINEAR
         in w (the literal J_s^T R_s is QUADRATIC in w; explainer section E).
         Gradient rows are weighted GRAD_W (default sqrt(M/K), equalizing the
         two blocks' total squared row count) after per-row normalization.

Certification is the LADDER, never the NNLS rel-fit: rungs (b)/(c1)/(c2)/(c3)
on (i) HELD-OUT fit-side iterates (kind='heldout', collected but excluded from
the row build), (ii) solver-path iterates of the 4 test trajectories, (iii)
oracle states, (iv) training snapshots -- for each of the four sets, plus the
4 rollout errors per set.  This is the baseline table stage 3 must beat.

--- original ladder header follows ---

EQ FIDELITY LADDER -- how much does the empirical quadrature distort what the
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

EXLIN=1 (exp/2026-08-26-eq-learned): the SAMPLED side computes the linear
terms EXACTLY as (Phi^T G_int) h(z) with A = Phi^T G_int precomputed (M, R);
only the advection term Phi^T N(u) stays on the m nodes.  The full-grid side
is unchanged.  Gate 0 (bit-identity of the whole sampled residual to
make_weak_ops) then no longer applies to the exlin residual BY DESIGN; it is
still asserted for the incumbent-form ops on the same node set (code
identity), and two new gates replace it for the exlin ops:
  gate L: exlin linear part == full-grid linear part   (exactness, <=1e-12)
  gate A: exlin advection part == incumbent advection part on the same
          nodes (nothing changed there, <=1e-12)
EQ_ADV_ONLY=1 fits the m nodes on the advection row blocks only
(exlin_common.eq_fit_burgers_adv) -- the freed budget serves one term.
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
import exlin_common as xc                     # noqa: E402
from sep_burgers_r3 import build_test_full    # noqa: E402

F64 = jnp.float64

CKPT = os.environ["CKPT"]
N = int(os.environ.get("N", "256"))
N_TEST = int(os.environ.get("N_TEST", "4"))
EQ_M = int(os.environ.get("EQ_M", "64"))
EQ_M_FACTOR = int(os.environ.get("EQ_M_FACTOR", "4"))
EQ_CAND_CAP = int(os.environ.get("EQ_CAND_CAP", "65536"))
EXLIN = 1                          # this driver is exact-linear only
VARIANTS = os.environ.get("VARIANTS", "adv,grad,node").split(",")
N_FIT_TRAJ = int(os.environ.get("N_FIT_TRAJ", "8"))
N_ROW_STATES = int(os.environ.get("N_ROW_STATES", str(max(24, 6144 // EQ_M))))
N_CODE_STATES = int(os.environ.get("N_CODE_STATES", "16"))
N_HELDOUT = int(os.environ.get("N_HELDOUT", "64"))
GRAD_W = float(os.environ.get("GRAD_W", "0"))   # 0 -> sqrt(M/K) at runtime
STEPS = int(os.environ.get("STEPS", "3000"))
LR = float(os.environ.get("LR", "3e-3"))
REFIT_EVERY = int(os.environ.get("REFIT_EVERY", "500"))
ALPHA_B = float(os.environ.get("ALPHA_B", "1.0"))
ALPHA_G = float(os.environ.get("ALPHA_G", "1.0"))
MINSEP_W = float(os.environ.get("MINSEP_W", "0"))    # 0 -> 1/dx^2 at runtime
MINSEP_D = float(os.environ.get("MINSEP_D", "0"))    # 0 -> dx/2 at runtime
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
    global GRAD_W
    if GRAD_W == 0.0:
        GRAD_W = float(np.sqrt(EQ_M / K))
    tag = OUT_TAG or f"N{N}_K{K}_R{R}"
    OUT = f"{OUT_PREFIX}sep_eq_nodefit_{tag}.json"

    interior = bc.interior_indices(N)
    coords = np.asarray(bc.grid_coords(N))
    n_i2 = interior.size
    T = bc.NUM_STEPS + 1
    dx = 1.0 / (N - 1)
    feat_chunk = FEAT_CHUNK or (0 if N <= 512 else 131072)
    global MINSEP_W, MINSEP_D
    if MINSEP_D == 0.0:
        MINSEP_D = 0.5 * dx
    if MINSEP_W == 0.0:
        MINSEP_W = 1.0 / (dx * dx)
    if "node" in VARIANTS:
        assert "grad" in VARIANTS, "node needs the grad baseline (init + gate C)"

    report = dict(config=dict(
        pde="burgers2d", kind="eq_nodefit", N=N, k=K, r=R,
        ckpt=os.path.basename(CKPT), ckpt_cfg=cfg, n_test=N_TEST, eq_M=EQ_M,
        eq_m_factor=EQ_M_FACTOR, exlin=True, variants=VARIANTS,
        n_fit_traj=N_FIT_TRAJ, n_row_states_cfg=N_ROW_STATES,
        n_code_states=N_CODE_STATES, n_heldout_cfg=N_HELDOUT,
        grad_w=GRAD_W, node_steps=STEPS, node_lr=LR,
        refit_every=REFIT_EVERY, alpha_b=ALPHA_B, alpha_g=ALPHA_G,
        step_tol=STEP_TOL, stall=STALL,
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

    # =======================================================================
    # STAGE 2 -- four quadratures at ONE fixed (M, m), all with the EXACT-
    # LINEAR residual online:
    #   inc  : incumbent two-block state fit at training codes (u and N(u))
    #   adv  : stage-1 advection-only state fit at training codes
    #   path : advection rows at OFF-MANIFOLD states (LM iterates of real ROM
    #          rollouts on TRAINING trajectories, several nu) + training codes
    #   grad : path rows + gradient-teacher rows with the full-grid Jacobian
    #          FROZEN (J_f^T R_s(w) -> J_f^T R_f; LINEAR in w -- section E)
    # =======================================================================
    eq_ops = {}
    full_ops = {}
    Mi = EQ_M
    kx, ky, Phi, lam, _ = bc.test_modes(N, Mi)
    Phi_np = np.asarray(Phi)
    del Phi
    lam_np = np.asarray(lam, dtype=np.float64)
    lam_j = jnp.asarray(lam_np, dtype=F64)
    m_want = EQ_M_FACTOR * Mi
    fo = make_full(Mi)
    A_j = jnp.asarray(Phi_np).T @ G_int          # (M, R) exact-linear matrix
    grng = np.random.default_rng(SEED0 + 50)
    gateF_state = {"done": False}
    lbl0 = f"gradfit N={N} k={K} M={Mi} m={m_want}"

    def build_variant(name, keep, wq_np, eq_info):
        cl = dict(kind="grid", idx=interior[cand_pos[keep]], w=wq_np,
                  info=eq_info)
        idx = np.asarray(cl["idx"])
        m = idx.size
        w_q = jnp.asarray(cl["w"], dtype=F64)
        pos = np.searchsorted(interior, idx)
        assert np.all(interior[pos] == idx)
        Phi_q = jnp.asarray(Phi_np[pos]) * w_q[:, None]
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

            def parts_s(z, prev_c, nu):
                w_ = (1.0 + bc.DT * nu * lam_j) ** (-bc.WEAK_ALPHA)
                u, Nu = u_and_N_fast(z)
                pu = Phi_q.T @ u
                lin = Phi_q.T @ (u - prev_c) + bc.DT * nu * lam_j * pu
                adv = Phi_q.T @ Nu
                return w_ * (lin + bc.DT * adv), w_ * lin, w_ * bc.DT * adv

            def r_w_fast(z, prev_c, nu):
                return parts_s(z, prev_c, nu)[0]

            def rJ_fast(z, prev_c, nu):
                return (r_w_fast(z, prev_c, nu),
                        jax.jacfwd(r_w_fast)(z, prev_c, nu), None)

            def rJ_parts(z, prev_c, nu):
                R, lin, adv = parts_s(z, prev_c, nu)
                return R, jax.jacfwd(r_w_fast)(z, prev_c, nu), lin, adv
            return (r_w_fast, rJ_fast, prev_of_fast, rJ_parts)
        (r_w_f, rJ_f, prev_f, rJp_f) = mk()
        ops_ref = bc.make_weak_ops(dec, N, cl, kind="weak", M=Mi,
                                   solver="lspg")
        rJ_f_j = jax.jit(rJ_f)
        g0 = []
        for _ in range(5):
            zt = jnp.asarray(Z_tr[grng.integers(len(Z_tr))]
                             + 0.05 * grng.standard_normal(K))
            zp = jnp.asarray(Z_tr[grng.integers(len(Z_tr))])
            prev_c = ops_ref["prev_of"](zp)
            nu = float(np.median(nu_test))
            ra, Ja, _ = ops_ref["rJ"](zt, prev_c, nu)
            rb, Jb, _ = rJ_f_j(zt, prev_c, nu)
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

        # gate F is node-independent: run it once, on the first variant
        if not gateF_state["done"]:
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
                    ra, Ja, _ = ops_full_ref["rJ"](zt,
                                                   ops_full_ref["prev_of"](zp),
                                                   nu)
                    rb, Jb, _, _ = fo["rJ"](G_int, fo["Phi"], zt, pf, nu)
                    gF.append(max(
                        float(jnp.max(jnp.abs(ra - rb))
                              / (jnp.max(jnp.abs(ra)) + 1e-300)),
                        float(jnp.max(jnp.abs(Ja - Jb))
                              / (jnp.max(jnp.abs(Ja)) + 1e-300))))
                gateF = float(np.max(gF))
                sc.log(f"  GATE F (full-grid ref vs make_weak_ops on "
                       f"interior): {gateF:.2e}")
                assert gateF < 1e-12
                report["gates"]["gateF"] = gateF
                del ops_full_ref
            else:
                report["gates"]["gateF"] = None
                sc.log(f"  GATE F skipped at N={N} (> FULL_GATE_MAX_N)")
            gateF_state["done"] = True

        # -------- exact-linear sampled ops (the ONLY online form here) ------
        def mk_ex(G_st=G_st, Phi_q=Phi_q, lam_j=lam_j, A=A_j):
            def u_and_N_fast(z):
                us = jnp.einsum("msr,r->ms", G_st, h_fn(z))
                c, xp, xm, yp, ym = (us[:, 0], us[:, 1], us[:, 2],
                                     us[:, 3], us[:, 4])
                ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
                uy = jnp.where(c > 0.0, (c - ym) / dx, (yp - c) / dx)
                return c, c * (ux + uy)

            def prev_of_ex(z):
                return A @ h_fn(z)          # (M,) exact Phi^T u(z_prev)

            def parts_ex(z, prev_m, nu):
                w_ = (1.0 + bc.DT * nu * lam_j) ** (-bc.WEAK_ALPHA)
                Ah = A @ h_fn(z)
                _, Nu = u_and_N_fast(z)
                lin = (Ah - prev_m) + bc.DT * nu * lam_j * Ah
                adv = Phi_q.T @ Nu
                return (w_ * (lin + bc.DT * adv), w_ * lin,
                        w_ * bc.DT * adv)

            def r_w_ex(z, prev_m, nu):
                return parts_ex(z, prev_m, nu)[0]

            def d_c_ex(z):
                return u_and_N_fast(z)[0]

            def rJ_ex(z, prev_m, nu):
                return (r_w_ex(z, prev_m, nu),
                        jax.jacfwd(r_w_ex)(z, prev_m, nu),
                        Phi_q.T @ jax.jacfwd(d_c_ex)(z))

            def rJ_parts_ex(z, prev_m, nu):
                R, lin, adv = parts_ex(z, prev_m, nu)
                return R, jax.jacfwd(r_w_ex)(z, prev_m, nu), lin, adv

            def full_ex(z):
                return G_st[:, 0, :] @ h_fn(z)
            return (r_w_ex, rJ_ex, prev_of_ex, full_ex, rJ_parts_ex)
        (r_w_x, rJ_x, prev_x, full_x, rJp_x) = mk_ex()
        ops_ex = bc._finish_ops(rJ_x, r_w_x, prev_x, full_x, m, "lspg")
        ops_ex["M"] = Mi
        ops_ex["tol_scale"] = float(np.sqrt(n_i2))
        rJp_x_j = jax.jit(rJp_x)
        rJp_inc_j = jax.jit(rJp_f)
        prev_inc_j = jax.jit(prev_f)
        prev_ex_j = jax.jit(prev_x)
        gL, gA = [], []
        for _ in range(5):
            zt = jnp.asarray(Z_tr[grng.integers(len(Z_tr))]
                             + 0.05 * grng.standard_normal(K))
            zp = jnp.asarray(Z_tr[grng.integers(len(Z_tr))])
            nu = float(np.median(nu_test))
            pf = u_full_j(G_int, zp)
            _, _, lin_f, _ = fo["rJ"](G_int, fo["Phi"], zt, pf, nu)
            _, _, lin_x, adv_x = rJp_x_j(zt, prev_ex_j(zp), nu)
            _, _, _, adv_i = rJp_inc_j(zt, prev_inc_j(zp), nu)
            gL.append(float(jnp.max(jnp.abs(lin_x - lin_f))
                            / (jnp.max(jnp.abs(lin_f)) + 1e-300)))
            gA.append(float(jnp.max(jnp.abs(adv_x - adv_i))
                            / (jnp.max(jnp.abs(adv_i)) + 1e-300)))
        gateL, gateA = float(np.max(gL)), float(np.max(gA))
        sc.log(f"  GATE L (exlin linear vs full-grid linear) [{name}]: "
               f"{gateL:.2e}")
        sc.log(f"  GATE A (exlin advection vs incumbent advection) "
               f"[{name}]: {gateA:.2e}")
        assert gateL < 1e-12 and gateA < 1e-12
        report["eq"][name]["gateL"] = gateL
        report["eq"][name]["gateA"] = gateA
        eq_ops[name] = dict(ops_fast=ops_ex, idx=idx, pos=pos, m=m,
                            rJ_parts=rJp_x_j, prev=prev_ex_j,
                            w=np.asarray(wq_np), lam=lam_j)
        full_ops[name] = fo
        save()

    # ---------------- variants (a): code-state fits -------------------------
    if "inc" in VARIANTS:
        keep, wq_np, eq_info = ctol_eq.eq_fit_burgers(
            u_full_int, adv_full, Phi_np, cand_pos, Z_eq, K, m_want,
            lbl0 + " inc", bc.nnls_capped)
        build_variant("inc", keep, wq_np, eq_info)
    keep, wq_np, eq_info = xc.eq_fit_burgers_adv(
        u_full_int, adv_full, Phi_np, cand_pos, Z_eq, K, m_want,
        lbl0 + " adv-only", bc.nnls_capped)
    build_variant("adv", keep, wq_np, eq_info)

    # ---------------- collect LM iterates on TRAINING trajectories ----------
    # Real ROM rollouts with the 'adv' quadrature on training ICs (analytic
    # blob_ic, canonical seed draw), N_FIT_TRAJ trajectories chosen as nu
    # quantiles.  Every LM iterate (z, lam) is recorded BEFORE its step --
    # exactly the off-manifold states the solver visits.  No test data.
    cx_f, cy_f, w_f, a_f, nu_f, _ = bc.bf.sample_params(seed=bc.SEED)
    order_f = np.argsort(nu_f)
    fit_traj = order_f[np.linspace(0, len(order_f) - 1,
                                   N_FIT_TRAJ).astype(int)]
    report["config"]["fit_traj"] = [int(t) for t in fit_traj]
    report["config"]["fit_nus"] = [float(nu_f[t]) for t in fit_traj]

    oracle_fit = ss.make_oracle_lm_banked(h_fn, K, budget=200)
    firng = np.random.default_rng(SEED0 + 71)

    def fit_ic(u0_int):
        tj = jnp.asarray(u0_int[None], dtype=F64)
        z0s = [np.tile(zbar[None], (1, 1))]
        for _ in range(8):
            z0s.append(Z_tr[firng.integers(len(Z_tr), size=1)])
        z, v = ss.oracle_multi_init_banked(oracle_fit, G_int, z0s, tj)
        return np.asarray(z)[0]

    e_adv = eq_ops["adv"]
    it_states = []          # (z, z_prev, nu, lam, t, traj)
    t_coll = time.time()
    for tr in fit_traj:
        u0 = bc.bf.blob_ic(N, cx_f[tr], cy_f[tr], w_f[tr], a_f[tr])
        nu = float(nu_f[tr])
        z0 = fit_ic(u0[interior])
        u_scale = float(np.sqrt(np.mean(u0[interior] ** 2)))
        tol_abs = STEP_TOL * u_scale * float(np.sqrt(n_i2))
        ops = e_adv["ops_fast"]
        zs = [z0]
        n_rec = 0
        for t in range(1, T):
            z_prev = zs[-1]
            z_init = z_prev if len(zs) < 2 or EXTRAP == 0 else \
                z_prev + EXTRAP * (zs[-1] - zs[-2])
            prev_c = e_adv["prev"](jnp.asarray(z_prev, dtype=F64))
            z = jnp.asarray(z_init, dtype=F64)
            r, J, _ = ops["rJ"](z, prev_c, nu)
            rn = float(jnp.linalg.norm(r))
            lam_lm = 1e-6
            for attempt in range(1, bc.GN_BUDGET + 1):
                it_states.append((np.asarray(z), np.asarray(z_prev), nu,
                                  lam_lm, t, int(tr)))
                n_rec += 1
                H = J.T @ J
                g = J.T @ r
                D = jnp.diag(jnp.diag(H)) + 1e-30 * jnp.eye(K, dtype=F64)
                dz = jnp.linalg.solve(H + lam_lm * D, -g)
                if not bool(jnp.all(jnp.isfinite(dz))):
                    lam_lm = min(lam_lm * 10.0, 1e12)
                    if lam_lm >= 1e12:
                        break
                    continue
                ndz = float(jnp.linalg.norm(dz))
                if ndz > bc.TR_DELTA:
                    lam_lm = min(lam_lm * 10.0, 1e12)
                    if lam_lm >= 1e12:
                        break
                    continue
                if ndz <= 1e-12 * (1.0 + float(jnp.linalg.norm(z))):
                    break
                z_new = z + dz
                rn_new = float(ops["rn"](z_new, prev_c, nu))
                if np.isfinite(rn_new) and rn_new < rn:
                    rel_dec = (rn - rn_new) / rn
                    z, rn = z_new, rn_new
                    if rn <= tol_abs:
                        break
                    r, J, _ = ops["rJ"](z, prev_c, nu)
                    lam_lm = max(lam_lm / 3.0, 1e-12)
                    if rel_dec < STALL:
                        break
                else:
                    lam_lm = min(lam_lm * 10.0, 1e12)
                    if lam_lm >= 1e12:
                        break
            zs.append(np.asarray(z))
        sc.log(f"  collect traj {tr} (nu {nu:.3e}): {n_rec} iterates")
    sc.log(f"  collected {len(it_states)} fit-side LM iterates "
           f"[{time.time()-t_coll:.0f}s]")
    report["config"]["n_collected_iterates"] = len(it_states)

    # ---------------- row states + held-out split ---------------------------
    sel_rng = np.random.default_rng(SEED0 + 130)
    n_iter_rows = max(0, N_ROW_STATES - N_CODE_STATES)
    pick_rows = sel_rng.choice(len(it_states),
                               size=min(n_iter_rows, len(it_states)),
                               replace=False)
    rest = np.setdiff1d(np.arange(len(it_states)), pick_rows)
    hold_pick = sel_rng.choice(rest, size=min(N_HELDOUT, rest.size),
                               replace=False)
    heldout_states = [it_states[i] for i in hold_pick]
    code_rng = np.random.default_rng(SEED0 + 131)
    code_pick = code_rng.choice(len(Z_eq), size=min(N_CODE_STATES, len(Z_eq)),
                                replace=False)
    nu_quant = np.quantile(nu_f, np.linspace(0.05, 0.95,
                                             max(1, N_CODE_STATES)))
    row_states = [it_states[i] for i in pick_rows]
    row_states += [(Z_eq[j], Z_eq[j], float(nu_quant[jj % len(nu_quant)]),
                    1e-6, 0, -1) for jj, j in enumerate(code_pick)]
    report["config"]["n_row_states"] = len(row_states)
    report["config"]["n_heldout"] = len(heldout_states)

    # ---------------- same-target row systems -------------------------------
    t_rows = time.time()
    state_blocks, grad_blocks, teachers = [], [], []
    for (zr, zpr, nur, _, _, _) in row_states:
        rows, tgt, N_c = xc.build_adv_row_block(u_full_int, adv_full, Phi_np,
                                                cand_pos, zr)
        state_blocks.append((rows, tgt, N_c))
        pf = u_full_j(G_int, jnp.asarray(zpr, dtype=F64))
        Rf, J_f, _, _ = fo["rJ"](G_int, fo["Phi"], jnp.asarray(zr, dtype=F64),
                                 pf, nur)
        J_f = np.asarray(J_f)
        wt_r = (1.0 + bc.DT * nur * lam_np) ** (-bc.WEAK_ALPHA)
        grad_blocks.append(xc.build_grad_row_block(J_f, wt_r, bc.DT, rows,
                                                   tgt))
        # frozen teacher tensors for the node loss (anti-collusion: nothing
        # here is re-evaluated during node training)
        teachers.append(dict(
            h=np.asarray(h_fn(jnp.asarray(zr, dtype=F64))), t=tgt, wt=wt_r,
            W=bc.DT * (wt_r[:, None] * J_f).T,
            nRf=float(np.linalg.norm(np.asarray(Rf))),
            nJR=float(np.linalg.norm(J_f) * np.linalg.norm(np.asarray(Rf)))))
    sc.log(f"  built {len(state_blocks)} state row blocks "
           f"(+{len(grad_blocks)} grad blocks) + frozen teachers "
           f"[{time.time()-t_rows:.0f}s]")

    if "path" in VARIANTS:
        keep, wq_np, eq_info = xc.nnls_same_target(
            state_blocks, [], m_want, bc.nnls_capped, ctol_eq.EQ_SEED,
            grad_w=0.0, eq_rows=ctol_eq.EQ_ROWS, label=lbl0 + " path")
        build_variant("path", keep, wq_np, eq_info)
    keep_g, w_g, info_g = xc.nnls_same_target(
        state_blocks, grad_blocks, m_want, bc.nnls_capped,
        ctol_eq.EQ_SEED, grad_w=GRAD_W, eq_rows=ctol_eq.EQ_ROWS,
        label=lbl0 + f" grad(w={GRAD_W:g})")
    if "grad" in VARIANTS:
        build_variant("grad", keep_g, w_g, info_g)
    del state_blocks, grad_blocks

    # =======================================================================
    # STAGE 3 -- learned CONTINUOUS node positions (design doc section 6.2).
    # FROZEN decoder, FROZEN full-grid teacher.  The only new parameters are
    # the m node positions, reparameterized into the open box by a sigmoid;
    # the weights are re-solved by the inner NNLS at the current nodes every
    # REFIT_EVERY steps (variable projection), so the convex family is always
    # the fallback.  Loss = the ladder rungs vs the frozen teacher:
    #   alpha_b * ||wt*DT*(a_s - t_s)||^2 / ||R_f||^2          (rung b)
    # + alpha_g * ||W_s (a_s - t_s)||^2 / (||J_f|| ||R_f||)^2  (rung c1, abs)
    # + minsep penalty, averaged over the fit states, where
    #   a_s(X, w) = Phi(X)^T ( w .* N_s(X) ),
    # Phi(X) the continuous sine modes with the grid normalization
    # (blat_common.modes_at) and N_s(X) the dx-stencil sign-upwind advection
    # of the frozen decoder at the continuous nodes (g is meshfree).  The
    # upwind where(c>0) switch is piecewise in X; its subgradient is used
    # as-is (stated in the report).
    # =======================================================================
    if "node" in VARIANTS:
        import optax
        X0 = coords[interior[cand_pos[keep_g]]].astype(np.float64)   # (m, 2)
        w_node = np.asarray(w_g, dtype=np.float64)
        S = len(teachers)
        H_S = jnp.asarray(np.stack([t_["h"] for t_ in teachers]))    # (S, R)
        T_S = jnp.asarray(np.stack([t_["t"] for t_ in teachers]))    # (S, M)
        WT_S = jnp.asarray(np.stack([t_["wt"] for t_ in teachers]))  # (S, M)
        W_S = jnp.asarray(np.stack([t_["W"] for t_ in teachers]))    # (S,K,M)
        nRf_S = jnp.asarray([t_["nRf"] for t_ in teachers])
        nJR_S = jnp.asarray([t_["nJR"] for t_ in teachers])
        kx_j = jnp.asarray(np.asarray(kx, dtype=np.float64))
        ky_j = jnp.asarray(np.asarray(ky, dtype=np.float64))
        LO, HI = float(dx), float(1.0 - dx)
        OFF = jnp.asarray(np.array([[0.0, 0.0], [dx, 0.0], [-dx, 0.0],
                                    [0.0, dx], [0.0, -dx]]))         # (5, 2)
        dparams = dec.params

        def x_of(theta):
            return LO + (HI - LO) * jax.nn.sigmoid(theta)            # (m, 2)

        def theta_of(X):
            p = np.clip((X - LO) / (HI - LO), 1e-6, 1 - 1e-6)
            return np.log(p / (1.0 - p))

        def adv_at(X, Hb):
            """N_s at continuous nodes for all S states: (S, m)."""
            Xs = (X[:, None, :] + OFF[None, :, :]).reshape(-1, 2)
            G5 = sc.features(dparams, Xs).reshape(-1, 5, R)          # (m,5,R)
            U5 = jnp.einsum("mfr,sr->smf", G5, Hb)               # (S,m,5)
            c, xp, xm, yp, ym = (U5[..., 0], U5[..., 1], U5[..., 2],
                                 U5[..., 3], U5[..., 4])
            ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
            uy = jnp.where(c > 0.0, (c - ym) / dx, (yp - c) / dx)
            return c * (ux + uy)

        def phi_at(X):
            sx = jnp.sin(jnp.pi * kx_j[None, :] * X[:, 0:1])
            sy = jnp.sin(jnp.pi * ky_j[None, :] * X[:, 1:2])
            return (sx * sy) / ((N - 1) / 2.0)                       # (m, M)

        def a_of(X, w, Hb):
            PhiX = phi_at(X)
            Nn = adv_at(X, Hb)
            return (w[None, :] * Nn) @ PhiX                          # (S, M)

        def minsep(X):
            d2 = jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, -1)
            d2 = d2 + jnp.eye(X.shape[0]) * 1e9
            return jnp.sum(jax.nn.relu(MINSEP_D - jnp.sqrt(d2)) ** 2)

        def loss_fn(theta, w, Hb):
            X = x_of(theta)
            a = a_of(X, w, Hb)
            d = a - T_S
            Lb = jnp.mean(jnp.sum((WT_S * bc.DT * d) ** 2, -1) / nRf_S ** 2)
            Lg = jnp.mean(jnp.sum(jnp.einsum("skm,sm->sk", W_S, d) ** 2, -1)
                          / nJR_S ** 2)
            return (ALPHA_B * Lb + ALPHA_G * Lg + MINSEP_W * minsep(X),
                    (Lb, Lg))
        grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
        a_of_j = jax.jit(a_of)
        adv_at_j = jax.jit(adv_at)
        phi_at_j = jax.jit(phi_at)

        # gate C: at the (grid) init nodes the continuous machinery must
        # reproduce the grid machinery of the 'grad' variant
        X0_j = jnp.asarray(X0)
        w0_j = jnp.asarray(w_node)
        e_g = eq_ops["grad"]
        gC = []
        for si in range(min(3, S)):
            zr, zpr, nur = row_states[si][0], row_states[si][1], row_states[si][2]
            wt_r = (1.0 + bc.DT * nur * lam_np) ** (-bc.WEAK_ALPHA)
            a_cont = np.asarray(a_of_j(X0_j, w0_j, H_S))[si]
            _, _, _, adv_grid = e_g["rJ_parts"](
                jnp.asarray(zr, dtype=F64),
                e_g["prev"](jnp.asarray(zpr, dtype=F64)), nur)
            a_grid = np.asarray(adv_grid) / (wt_r * bc.DT)     # undo wt*DT
            gC.append(float(np.max(np.abs(a_cont - a_grid))
                            / (np.max(np.abs(a_grid)) + 1e-300)))
        gateC = float(np.max(gC))
        sc.log(f"  GATE C (continuous machinery at grid init vs grid ops): "
               f"{gateC:.2e}")
        assert gateC < 1e-12
        report["gates"]["gateC"] = gateC

        def refit_w(X):
            """Inner NNLS at current nodes: the same loss, linear in w."""
            PhiX = np.asarray(phi_at_j(jnp.asarray(X)))          # (m, M)
            Nn = np.asarray(adv_at_j(jnp.asarray(X), H_S))       # (S, m)
            Gr, br = [], []
            for si in range(S):
                base = PhiX.T * Nn[si][None, :]                  # (M, m)
                sw = (np.asarray(WT_S[si]) * bc.DT
                      / float(nRf_S[si]))[:, None]
                Gr.append(np.sqrt(ALPHA_B) * sw * base)
                br.append(np.sqrt(ALPHA_B) * sw[:, 0]
                          * np.asarray(T_S[si]))
                Wg = np.asarray(W_S[si]) / float(nJR_S[si])      # (K, M)
                Gr.append(np.sqrt(ALPHA_G) * (Wg @ base))
                br.append(np.sqrt(ALPHA_G) * (Wg @ np.asarray(T_S[si])))
            Gr = np.concatenate(Gr, axis=0)
            br = np.concatenate(br)
            rng_w = np.random.default_rng(SEED0 + 777)
            pad = np.abs(Nn).mean(0)
            keep_w, ww, _ = ss._nnls_rows(Gr, br, Gr.shape[1], bc.nnls_capped,
                                          rng_w, ctol_eq.EQ_ROWS, pad)
            w_full = np.zeros(Gr.shape[1])
            w_full[keep_w] = ww
            return w_full

        theta = jnp.asarray(theta_of(X0))
        opt = optax.adam(LR)
        opt_state = opt.init(theta)
        w_cur = jnp.asarray(w_node)
        best = dict(loss=np.inf, X=X0, w=np.asarray(w_node), step=-1)
        t_opt = time.time()
        hist = []
        for it in range(STEPS + 1):
            if it % REFIT_EVERY == 0 and it > 0:
                w_np = refit_w(np.asarray(x_of(theta)))
                w_cur = jnp.asarray(w_np)
            (L, (Lb, Lg)), gth = grad_fn(theta, w_cur, H_S)
            Lf = float(L)
            if Lf < best["loss"]:
                best = dict(loss=Lf, X=np.asarray(x_of(theta)),
                            w=np.asarray(w_cur), step=it)
            if it % max(1, STEPS // 20) == 0:
                hist.append(dict(step=it, loss=Lf, Lb=float(Lb),
                                 Lg=float(Lg)))
                sc.log(f"  node-opt step {it:5d}  loss {Lf:.4e} "
                       f"(b {float(Lb):.3e}, g {float(Lg):.3e}) "
                       f"[{time.time()-t_opt:.0f}s]")
            if it == STEPS:
                break
            upd, opt_state = opt.update(gth, opt_state)
            theta = optax.apply_updates(theta, upd)
        # final weight refit at the best nodes
        w_best = refit_w(best["X"])
        Lb0 = float(grad_fn(jnp.asarray(theta_of(X0)), jnp.asarray(w_node),
                            H_S)[0][0])
        Lend = float(grad_fn(jnp.asarray(theta_of(best["X"])),
                             jnp.asarray(w_best), H_S)[0][0])
        report["node_opt"] = dict(
            steps=STEPS, lr=LR, refit_every=REFIT_EVERY, alpha_b=ALPHA_B,
            alpha_g=ALPHA_G, minsep_w=MINSEP_W, minsep_d=MINSEP_D,
            loss_init=Lb0, loss_best=best["loss"], loss_final_refit=Lend,
            best_step=best["step"], hist=hist, gateC=gateC,
            secs=time.time() - t_opt,
            n_moved=float(np.mean(np.linalg.norm(best["X"] - X0, axis=1)
                                  > 1e-9)),
            mean_move=float(np.mean(np.linalg.norm(best["X"] - X0, axis=1))),
            max_move=float(np.max(np.linalg.norm(best["X"] - X0, axis=1))),
            w_nonzero=int(np.sum(w_best > 0)))
        sc.log(f"  node-opt DONE: loss {Lb0:.4e} -> {Lend:.4e} "
               f"(best step {best['step']}), mean node move "
               f"{report['node_opt']['mean_move']:.2e}, "
               f"{report['node_opt']['w_nonzero']}/{len(w_best)} weights > 0 "
               f"[{time.time()-t_opt:.0f}s]")

        # ---- certification ops for the learned set (continuous nodes) -----
        X_n = best["X"]
        w_n_j = jnp.asarray(w_best, dtype=F64)
        Xs_n = (X_n[:, None, :] + np.asarray(OFF)[None, :, :]).reshape(-1, 2)
        G_st_n = dec.feat_at(Xs_n).reshape(len(X_n), 5, dec.r)
        Phi_q_n = jnp.asarray(np.asarray(phi_at_j(jnp.asarray(X_n)))) \
            * w_n_j[:, None]

        def mk_node(G_st=G_st_n, Phi_q=Phi_q_n, lam_j=lam_j, A=A_j):
            def u_and_N_fast(z):
                us = jnp.einsum("msr,r->ms", G_st, h_fn(z))
                c, xp, xm, yp, ym = (us[:, 0], us[:, 1], us[:, 2],
                                     us[:, 3], us[:, 4])
                ux = jnp.where(c > 0.0, (c - xm) / dx, (xp - c) / dx)
                uy = jnp.where(c > 0.0, (c - ym) / dx, (yp - c) / dx)
                return c, c * (ux + uy)

            def prev_of_ex(z):
                return A @ h_fn(z)

            def parts_ex(z, prev_m, nu):
                w_ = (1.0 + bc.DT * nu * lam_j) ** (-bc.WEAK_ALPHA)
                Ah = A @ h_fn(z)
                _, Nu = u_and_N_fast(z)
                lin = (Ah - prev_m) + bc.DT * nu * lam_j * Ah
                adv = Phi_q.T @ Nu
                return (w_ * (lin + bc.DT * adv), w_ * lin,
                        w_ * bc.DT * adv)

            def r_w_ex(z, prev_m, nu):
                return parts_ex(z, prev_m, nu)[0]

            def d_c_ex(z):
                return u_and_N_fast(z)[0]

            def rJ_ex(z, prev_m, nu):
                return (r_w_ex(z, prev_m, nu),
                        jax.jacfwd(r_w_ex)(z, prev_m, nu),
                        Phi_q.T @ jax.jacfwd(d_c_ex)(z))

            def rJ_parts_ex(z, prev_m, nu):
                Rr, lin, adv = parts_ex(z, prev_m, nu)
                return Rr, jax.jacfwd(r_w_ex)(z, prev_m, nu), lin, adv

            def full_ex(z):
                return G_st[:, 0, :] @ h_fn(z)
            return (r_w_ex, rJ_ex, prev_of_ex, full_ex, rJ_parts_ex)
        (r_w_n, rJ_n, prev_n, full_n, rJp_n) = mk_node()
        ops_n = bc._finish_ops(rJ_n, r_w_n, prev_n, full_n, len(X_n), "lspg")
        ops_n["M"] = Mi
        ops_n["tol_scale"] = float(np.sqrt(n_i2))
        rJp_n_j = jax.jit(rJp_n)
        prev_n_j = jax.jit(prev_n)
        # gate L holds for any node set (linear part is node-independent)
        gL = []
        for _ in range(5):
            zt = jnp.asarray(Z_tr[grng.integers(len(Z_tr))]
                             + 0.05 * grng.standard_normal(K))
            zp = jnp.asarray(Z_tr[grng.integers(len(Z_tr))])
            nu = float(np.median(nu_test))
            pf = u_full_j(G_int, zp)
            _, _, lin_f, _ = fo["rJ"](G_int, fo["Phi"], zt, pf, nu)
            _, _, lin_x, _ = rJp_n_j(zt, prev_n_j(zp), nu)
            gL.append(float(jnp.max(jnp.abs(lin_x - lin_f))
                            / (jnp.max(jnp.abs(lin_f)) + 1e-300)))
        gateL = float(np.max(gL))
        sc.log(f"  GATE L [node]: {gateL:.2e}")
        assert gateL < 1e-12
        report["eq"]["node"] = dict(
            kind="learned_nodes", m=int(len(X_n)), gateL=gateL, gateC=gateC,
            rel_fit=None, loss_init=Lb0, loss_final=Lend,
            w_nonzero=int(np.sum(w_best > 0)))
        eq_ops["node"] = dict(ops_fast=ops_n, idx=None, pos=None,
                              m=len(X_n), rJ_parts=rJp_n_j, prev=prev_n_j,
                              w=np.asarray(w_best), lam=lam_j)
        full_ops["node"] = fo
        np.savez(OUT.replace(".json", "_nodes.npz"), X=X_n, w=w_best, X0=X0,
                 w0=np.asarray(w_node))
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
        if u_true is not None and e.get("pos") is not None:
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

    # ---------------- held-out fit-side iterates (certification) -------------
    for name in eq_ops:
        for (zh, zph, nuh, lamh, th, trh) in heldout_states:
            rec = ladder(name, zh, zph, nuh, lamh)
            rec.update(kind="heldout", t=int(th), traj=int(trh), iter=0)
            report["records"].append(rec)
    sc.log(f"  held-out iterates evaluated: {len(heldout_states)} states x "
           f"{len(eq_ops)} sets")
    save()

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
        for kind in ("solver", "heldout", "oracle", "train"):
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
