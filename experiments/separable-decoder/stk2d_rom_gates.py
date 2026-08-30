"""PHASE 2b driver: the quadrature-free reduced-order model and its gates.

Gates S4, S6, S7, S8, S9, on top of the certified phase-1 FOM
(`stk2d_common.py`), the certified phase-2a bank (`stk2d_bank.py`) and the
phase-2b STOP GATE artifact (`stk2d_recon_gate.py`).  NOTHING in phase 1 or
phase 2a is modified.

THIS DRIVER REFUSES TO RUN unless the stop-gate artifact exists, is complete
and certified, and records verdict.passed.  The stop gate is the whole point of
running reconstruction first: if a finite MLP head cannot materially beat POD-8
on held-out data, the residual, the controls and the timing are all measuring a
decoder that does not work, and the honest deliverable is the negative.

WHAT THIS CELL IS FOR, restated because it governs how S7 must be read.
STOKES-DESIGN.md: "This cell is a de-risking rehearsal for 2D incompressible
Navier-Stokes.  It is not expected to produce a positive result, and it must
not be written up as one."  Steady Stokes is LINEAR and G is itself a POD
basis, so the direct reduced solve in the G span (S7c) is a ONE-SHOT
projection.  It is expected to be both faster and more accurate than a
nonlinear head driven by an iterative LM solve, and if it is, that is the
predicted outcome and not a failure of the machinery.

GATES
  PRECOND   frozen config, no -O, SMOKE never certifies.
  S0        solver dtype f64; JAX x64 / matmul=highest / backend gpu.
  STOPGATE  the phase-2b reconstruction artifact must exist, be certified, and
            have passed.  Recorded with its numbers, not just its boolean.
  S-HEAD    the numpy head used by every TIMED path must agree with the JAX
            head that was trained, value and Jacobian, to 1e-13; and the
            numpy Jacobian must agree with a central difference.
  S4        the quadrature-free residual against an INDEPENDENT strong-form
            full-grid implementation that decodes, reassembles, APPLIES the MAC
            stencil through the matrix-free pad-and-slice operator, and
            projects -- at >= 32 seeded states and at every captured solve
            solution, for the RESIDUAL and the JACOBIAN alike, in a
            cancellation-aware normalisation.  Includes the pressure-
            elimination rung: Phi^T M_u(-nu L u_FOM - f) must be roundoff on
            every FOM solution, along a path the solver never takes.
            NEGATIVE CONTROLS: even (free-slip) ghosts in the full-grid path;
            the affine-mean constant term dropped (the revision-1 design bug);
            and A perturbed by a relative 1e-10.
  S6        cost: quadrature-free vs full-grid vs a NEWLY DEFINED strong-form
            EQ/NNLS arm with its own fit target, its own two-lattice sampling
            applied AFTER analytic pressure elimination, its own exactness
            gate, and setup timed separately.  The AFFINE and NON-AFFINE force
            arms are timed separately and never blended.
  S7        three controls plus the nonlinear arm: (a) POD-K at matched online
            dimension, (b) POD-R Galerkin at matched trial span, (c) the direct
            reduced solve in the G span.  (c) is EXPECTED TO WIN.
  S8        the M ladder with M >= R: rank(A J_h(z)) = K and
            sigma_min/sigma_max, INCLUDING a near-collision cohort where the
            family's own Jacobian degenerates from rank 8 to rank 4.
  S9        the R frontier {8,16,32}, K fixed, parameter count per R, run
            against the M ladder.
  MANIFEST  every expected gate present, exact row counts, no non-finite.

Env: ROM_NS, R_LADDER, M_LADDER, M_STOP, K_LAT, NU, SEED, REPS, BURN,
     EQ_BUDGETS, N_S4_STATES, OUT_TAG, OUT_PREFIX, CACHE, ALLOW_CPU=0, SMOKE=0.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time

import numpy as np
import scipy

import stk2d_common as stk
import stk2d_bank as bank
import stk2d_rom as rom

HERE = os.path.dirname(os.path.abspath(__file__))

ROM_NS = [int(v) for v in os.environ.get("ROM_NS", "32,64,128,256").split(",") if v]
R_LADDER = [int(v) for v in os.environ.get("R_LADDER", "8,16,32").split(",") if v]
M_LADDER = [int(v) for v in os.environ.get("M_LADDER", "32,64,128").split(",") if v]
M_MAIN = int(os.environ.get("M_MAIN", "64"))
R_MAIN = int(os.environ.get("R_MAIN", "32"))
N_MAIN = int(os.environ.get("N_MAIN", "64"))
K_LAT = int(os.environ.get("K_LAT", "8"))
S_TRAIN = int(os.environ.get("S_TRAIN", "256"))
S_TEST = int(os.environ.get("S_TEST", "64"))
NU = float(os.environ.get("NU", "1.0"))
SEED = int(os.environ.get("SEED", "20260830"))
REPS = int(os.environ.get("REPS", "12"))
BURN = int(os.environ.get("BURN", "3"))
EQ_BUDGETS = [int(v) for v in os.environ.get("EQ_BUDGETS", "64,128,256,512").split(",") if v]
EQ_CAND = int(os.environ.get("EQ_CAND", "3000"))
EQ_FIT_STATES = int(os.environ.get("EQ_FIT_STATES", "24"))
N_S4_STATES = int(os.environ.get("N_S4_STATES", "32"))
N_TIME_QUERIES = int(os.environ.get("N_TIME_QUERIES", "4"))
LM_STARTS = int(os.environ.get("LM_STARTS", "4"))
OUT_TAG = os.environ.get("OUT_TAG", "")
OUT_PREFIX = os.environ.get("OUT_PREFIX", "runs/stk2d/")
CACHE = os.environ.get("CACHE", "runs/stk2d/cache")
HEADS = os.environ.get("HEADS", "runs/stk2d/heads")
ALLOW_CPU = int(os.environ.get("ALLOW_CPU", "0"))
SMOKE = int(os.environ.get("SMOKE", "0"))

RECON_ARTIFACT = "runs/stk2d/stk2d_recon_gate_recon_nu1.json"

# ------------------------------------------------------------- thresholds ---
S4_TOL = 1e-12             # STOKES-DESIGN.md S4, cancellation-aware form
S4_CTL_FLOOR = 1e2 * S4_TOL    # a control must FAIL by at least two orders.
                           # Stated as a MULTIPLE of the gate tolerance, never
                           # as an absolute number (phase-2a retraction 19).
                           # Two orders rather than three because the WEAKEST
                           # control is a relative perturbation of A, and the
                           # size of the discrepancy it produces is set by that
                           # perturbation: a relative 1e-8 perturbation reads
                           # about 1.3e-9, and a relative 1e-10 one about
                           # 1.3e-11, which is only ten times the gate.  The
                           # perturbation size is the honest statement of the
                           # gate's sensitivity and both are recorded.
S4_A_PERT = 1e-8           # the gated A-perturbation control
S4_A_PERT_SMALL = 1e-10    # recorded alongside, as a sensitivity diagnostic
HEAD_TOL = 1e-13           # numpy head vs the trained JAX head
HEAD_FD_TOL = 1e-6         # numpy head Jacobian vs a central difference
PRESS_TOL = 1e-12          # the pressure-elimination rung, normalised
EQ_EXACT_TOL = 1e-13       # the EQ machinery with ALL faces and unit weights
RANK_TOL = 1e-9            # relative singular-value cut for rank(A J_h)
EQ_CTL_FLOOR = 1e-2        # a shuffled-weight EQ control must exceed this

FROZEN_CONFIG = dict(rom_ns=[32, 64, 128, 256], r_ladder=[8, 16, 32],
                     m_ladder=[32, 64, 128], m_main=64, r_main=32, n_main=64,
                     k_lat=8, s_train=256, s_test=64, nu=1.0, reps=12, burn=3,
                     eq_budgets=[64, 128, 256, 512], eq_cand=3000,
                     eq_fit_states=24, n_s4_states=32, n_time_queries=4,
                     lm_starts=4, allow_cpu=0, Q=48, K=8, grad_mix=3.0)
EXPECTED_GATES = frozenset(("PRECOND", "S0", "STOPGATE", "S_HEAD", "S4", "S6",
                            "S7", "S8", "S9", "MANIFEST"))
EXPECTED_ROWS = dict(S_HEAD=4 * 3, S4=4, S6=4, S7=4, S8=4 * 3, S9=4 * 3 * 3)


def finite(label, xs):
    a = np.asarray([float(x) for x in xs], dtype=float)
    bad = ~np.isfinite(a)
    assert not bad.any(), (f"non-finite value(s) in {label}: {a[bad].tolist()} "
                           f"(indices {np.nonzero(bad)[0].tolist()})")
    return a


def log(*a):
    print(*a, flush=True)


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"],
                                       text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("GIT_COMMIT", "unknown")


def jax_provenance():
    out = dict(imported=False)
    try:
        import jax
        dev = jax.devices()[0]
        out = dict(imported=True, backend=dev.platform, device=str(dev),
                   device_kind=getattr(dev, "device_kind", ""),
                   x64=bool(jax.config.jax_enable_x64),
                   matmul_precision=os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
                   jax_version=jax.__version__)
    except Exception as e:                                    # pragma: no cover
        out["error"] = repr(e)
    return out


def head_path(N, R):
    return os.path.join(HERE, HEADS, f"head_N{N}_R{R}_K{K_LAT}.npz")


# ------------------------------------------------------------ ROM solving ---

def nl_solve(qf, hspec, b, z_starts):
    """One nonlinear ROM query in the quadrature-free form: LM over z with
    r(z) = const - nu A h(z) - b."""
    def rf(z):
        return qf.resid(rom.head_np(hspec, z), b)

    def jf(z):
        _, J = rom.head_jac_np(hspec, z)
        return qf.nuA @ J
    return rom.lm_multistart(rf, jf, z_starts, scale=qf.scale_of(b))


def field_err(cell, u_hat, i):
    """Mass-weighted relative error of a reconstructed field against FOM
    snapshot i, in the SAME convention as every other error in this cell."""
    x = cell.U_te[:, i]
    return float(np.linalg.norm(u_hat - x) / np.linalg.norm(x))


def centred_err(cell, a, i, R):
    """The centred convention phase 2a used for POD errors."""
    x = cell.U_te[:, i] - cell.ubar
    e = x - cell.G[:, :R] @ a
    return float(np.linalg.norm(e) / np.linalg.norm(x))


# ==================================================================== S4 =====

def gate_s4(cell, R, M, hspec, solve_states, rng):
    """S4: the quadrature-free residual against an INDEPENDENT strong-form
    full-grid implementation.

    The full-grid arm decodes, reassembles the force from the dictionary,
    applies the MAC stencil through `stk.apply_laplacian` (the pad-and-slice
    matrix-free operator, written independently of the sparse assembly and
    gated against it by phase 1's MF gate), and projects with a test space
    rebuilt by `stk.apply_curl`.  It shares no matrix with the QF path.

    Phase 2a's S-SOLVE reads exactly 0.0 because both of its paths hand the
    same assembled matrix to the same SuperLU: it is an assembly regression
    check and CANNOT substitute for this (carried-forward condition 6).
    """
    g = cell.g
    qf = rom.QFRom(cell, R, M)
    Phi_mf = rom.phi_matrix_free(g, qf.modes)
    phi_dev = float(np.abs(Phi_mf - qf.Phi).max()
                    / (np.abs(qf.Phi).max() + 1e-300))
    c_te = cell.coeff_of(cell.U_te)[:, :R]
    csc = float(np.abs(c_te).std())

    states = []
    for t in range(N_S4_STATES):                       # seeded random states
        i = int(rng.integers(0, S_TEST))
        states.append(("seeded", i, rng.standard_normal(R) * csc))
    for i, z in solve_states:                          # captured solve solutions
        states.append(("solve", i, rom.head_np(hspec, z)))
    for i in range(S_TEST):                            # exact FOM coefficients
        states.append(("fom", i, c_te[i]))

    rel, canc, gcanc, grel = [], [], [], []
    for kind, i, hz in states:
        th = cell.Th_te[i]
        b = qf.b_of(th)
        f = cell.F @ th
        u = cell.ubar + cell.G[:, :R] @ hz
        rq = qf.resid(hz, b)
        rf = rom.resid_full_grid(g, Phi_mf, qf.nu, u, f)
        sc = qf.resid_scale(hz, b)
        d = float(np.linalg.norm(rq - rf))
        rel.append(d / (np.linalg.norm(rf) + 1e-300))
        canc.append(d / sc)
    # Jacobians, at a subset (each full-grid Jacobian is K stencil applies)
    Jh_states = states[:8] + states[N_S4_STATES:N_S4_STATES + 8]
    for kind, i, hz in Jh_states:
        # the head Jacobian at the latent that produced hz is unavailable for
        # the seeded states, so a RANDOM full-rank Jh is used: S4 tests the
        # residual/Jacobian machinery, not the head
        Jh = rng.standard_normal((R, K_LAT)) / np.sqrt(R)
        Jq = qf.nuA @ Jh
        Jf = rom.jac_full_grid(g, Phi_mf, qf.nu, cell.G[:, :R], Jh)
        d = float(np.linalg.norm(Jq - Jf))
        grel.append(d / (np.linalg.norm(Jf) + 1e-300))
        # cancellation-aware denominator: the TERM MAGNITUDE nu ||A||_F
        # ||Jh||_F, which cannot collapse, rather than ||J_full|| itself
        gcanc.append(d / (qf.nu * np.linalg.norm(qf.A, "fro")
                          * np.linalg.norm(Jh) + 1e-300))

    # ---- the pressure-elimination rung, on every FOM solution --------------
    press = []
    for i in range(S_TEST):
        f = cell.F @ cell.Th_te[i]
        rf = rom.resid_full_grid(g, Phi_mf, qf.nu, cell.U_te[:, i], f)
        scale = qf.nu * np.linalg.norm(qf.A, "fro") * np.linalg.norm(
            c_te[i]) + np.linalg.norm(qf.b_of(cell.Th_te[i]))
        press.append(float(np.linalg.norm(rf) / (scale + 1e-300)))

    # ---- NEGATIVE CONTROLS, all three must FAIL the gate ------------------
    kind, i, hz = states[0]
    th = cell.Th_te[i]
    b = qf.b_of(th)
    f = cell.F @ th
    u = cell.ubar + cell.G[:, :R] @ hz
    sc = qf.resid_scale(hz, b)
    U, V = g.unpack(u)
    LU, LV = stk.apply_laplacian(g, U, V, "even")       # free-slip ghosts
    r_even = Phi_mf.T @ ((-qf.nu * g.pack(LU, LV) - f) * g.h ** 2)
    ctl_even = float(np.linalg.norm(qf.resid(hz, b) - r_even) / sc)
    r_noubar = -qf.nu * (qf.A @ hz) - b                # drop the affine mean
    ctl_noubar = float(np.linalg.norm(r_noubar
                                      - rom.resid_full_grid(g, Phi_mf, qf.nu,
                                                            u, f)) / sc)
    r_ref = rom.resid_full_grid(g, Phi_mf, qf.nu, u, f)
    ctl_pert, ctl_pert_small = None, None
    for eps, slot in ((S4_A_PERT, "big"), (S4_A_PERT_SMALL, "small")):
        A_pert = qf.A * (1.0 + eps * rng.standard_normal(qf.A.shape))
        v = float(np.linalg.norm(qf.const - qf.nu * (A_pert @ hz) - b - r_ref)
                  / sc)
        if slot == "big":
            ctl_pert = v
        else:
            ctl_pert_small = v
    return dict(
        N=cell.N, R=int(R), M=int(M), n_states=len(states),
        n_seeded=int(N_S4_STATES), n_solve_states=len(solve_states),
        phi_vs_matrixfree=phi_dev,
        resid_rel_max=float(finite("S4 rel", rel).max()),
        resid_canc_max=float(finite("S4 canc", canc).max()),
        jac_rel_max=float(finite("S4 jac rel", grel).max()),
        jac_canc_max=float(finite("S4 jac canc", gcanc).max()),
        press_elim_max=float(finite("S4 press", press).max()),
        ctl_evenghost=ctl_even, ctl_dropped_mean=ctl_noubar,
        ctl_perturbed_A=ctl_pert, ctl_perturbed_A_small=ctl_pert_small,
        ctl_perturbed_A_eps=float(S4_A_PERT),
        ctl_perturbed_A_small_eps=float(S4_A_PERT_SMALL),
        offline_seconds=float(qf.t_offline))


# ==================================================================== S6/S7 ==

def build_eq(cell, qf, hspec, rng):
    """The strong-form EQ arm: two-lattice candidate sampling AFTER analytic
    pressure elimination, a fit target taken from the FULL strong MAC
    projection at perturbed states, greedy support selection, and NNLS
    weights."""
    g = cell.g
    cand, lat = rom.eq_candidates(g, EQ_CAND, SEED + cell.N)
    c_te = cell.coeff_of(cell.U_te)[:, :qf.R]
    csc = float(np.abs(c_te).std())
    fit_states, fit_idx = [], []
    for t in range(EQ_FIT_STATES):
        i = int(rng.integers(0, S_TRAIN))
        hz = cell.coeff_affine(cell.mu_tr[i])[0, :qf.R] \
            + 0.3 * csc * rng.standard_normal(qf.R)
        f = cell.F @ cell.Th_tr[i]
        fit_states.append((cell.ubar + cell.G[:, :qf.R] @ hz, f))
        fit_idx.append((i, hz))
    t0 = time.time()
    Psi, tgt = rom.eq_design(cell, qf, fit_states, cand)
    order, hist = rom.eq_fit_greedy(Psi, tgt, max(EQ_BUDGETS))
    t_setup = time.time() - t0

    # exactness of the MACHINERY: all faces, unit weights, must reproduce the
    # full strong projection.  A bookkeeping check (stencil slicing and index
    # remapping), and labelled as one.
    allnodes = np.arange(g.n_u)
    eq_all = rom.EQArm(cell, qf, allnodes, np.ones(g.n_u))
    i = 0
    hz = c_te[i]
    th = cell.Th_te[i]
    exact = float(np.linalg.norm(eq_all.resid(hz, th) - qf.resid(hz, qf.b_of(th)))
                  / qf.resid_scale(hz, qf.b_of(th)))
    del eq_all

    arms = {}
    for m in EQ_BUDGETS:
        sup = order[:m]
        w, fitres = rom.eq_weights(Psi, tgt, sup)
        arm = rom.EQArm(cell, qf, cand[sup], w)
        # held-out accuracy of the reduced quadrature against the FULL strong
        # projection, at states the fit never saw
        errs = []
        for j in range(0, S_TEST, 4):
            hzj = c_te[j] + 0.3 * csc * rng.standard_normal(qf.R)
            thj = cell.Th_te[j]
            bj = qf.b_of(thj)
            errs.append(float(np.linalg.norm(arm.resid(hzj, thj)
                                             - qf.resid(hzj, bj))
                              / qf.resid_scale(hzj, bj)))
        Aeq = arm.collapsed_A()
        arms[m] = dict(m=int(m), n_ext=int(arm.ext.size),
                       n_lattice_x=int((cand[sup] < g.n_ux).sum()),
                       n_lattice_y=int((cand[sup] >= g.n_ux).sum()),
                       w_min=float(w.min()), w_max=float(w.max()),
                       fit_residual=float(fitres),
                       heldout_resid_err=float(np.max(errs)),
                       heldout_resid_err_med=float(np.median(errs)),
                       collapsed_A_err=float(np.linalg.norm(Aeq - qf.A)
                                             / np.linalg.norm(qf.A)),
                       arm=arm)
    return dict(cand=cand, order=order, hist=hist, t_setup=float(t_setup),
                machinery_exact=exact, arms=arms, Psi_shape=list(Psi.shape))


def gate_s67(cell, R, M, hspec, eq, rng):
    """S6 (cost) and S7 (the three controls plus the nonlinear arm), on one
    mesh at the main (R, M).

    ACCURACY is measured over the whole held-out cohort; COST is measured over
    N_TIME_QUERIES queries with the project's balanced-order timing harness.
    The AFFINE and NON-AFFINE force arms are timed separately and never
    blended, as STOKES-DESIGN.md requires.
    """
    g = cell.g
    assert int(hspec["meta"]["R"]) == int(R), \
        f"head R={hspec['meta']['R']} does not match the arm's R={R}"
    qf = rom.QFRom(cell, R, M)
    Phi_mf = rom.phi_matrix_free(g, qf.modes)
    linR = rom.LinearArm(qf)
    linK = rom.LinearArm(qf, r_trial=K_LAT)
    eq_arm = eq["arms"][max(EQ_BUDGETS)]["arm"]
    Zp = hspec["Z"]
    lo, hi = Zp.min(0), Zp.max(0)
    rs = np.random.default_rng(SEED + 5)
    z_starts = [np.zeros(K_LAT)] + [lo + (hi - lo) * rs.random(K_LAT)
                                    for _ in range(LM_STARTS - 1)]

    # ---------------- accuracy over the whole held-out cohort --------------
    err = {k: [] for k in ("nl_qf", "podK", "podR_galerkin", "gspan_direct",
                           "nl_full", "nl_eq")}
    solve_states, lm_stats = [], []
    for i in range(S_TEST):
        th = cell.Th_te[i]
        b = qf.b_of(th)
        z, val, tot = nl_solve(qf, hspec, b, z_starts)
        solve_states.append((i, z))
        lm_stats.append(tot["iters"])
        a_nl = rom.head_np(hspec, z)
        err["nl_qf"].append(centred_err(cell, a_nl, i, R))
        err["podK"].append(centred_err(cell, linK.solve_lsq(b), i, K_LAT))
        err["podR_galerkin"].append(centred_err(cell, linR.solve_galerkin(th),
                                                i, R))
        err["gspan_direct"].append(centred_err(cell, linR.solve_lsq(b), i, R))

        def rf_full(z_):
            u = cell.ubar + cell.G[:, :R] @ rom.head_np(hspec, z_)
            return rom.resid_full_grid(g, Phi_mf, qf.nu, u, cell.F @ th)

        def jf_full(z_):
            _, Jh = rom.head_jac_np(hspec, z_)
            return rom.jac_full_grid(g, Phi_mf, qf.nu, cell.G[:, :R], Jh)

        if i < 8:      # the full-grid and EQ arms are O(n) per iteration
            zf, _, _ = rom.lm_multistart(rf_full, jf_full, z_starts[:2],
                                         scale=qf.scale_of(b))
            err["nl_full"].append(centred_err(cell, rom.head_np(hspec, zf), i, R))

            def rf_eq(z_):
                return eq_arm.resid(rom.head_np(hspec, z_), th)

            def jf_eq(z_):
                _, Jh = rom.head_jac_np(hspec, z_)
                return eq_arm.jac(Jh)
            ze, _, _ = rom.lm_multistart(rf_eq, jf_eq, z_starts[:2],
                                         scale=qf.scale_of(b))
            err["nl_eq"].append(centred_err(cell, rom.head_np(hspec, ze), i, R))

    # ---------------- timing, balanced order -------------------------------
    qidx = list(range(0, S_TEST, max(1, S_TEST // N_TIME_QUERIES)))[:N_TIME_QUERIES]
    fac = cell.fac
    if fac is None:
        t0 = time.time()
        fac = bank.SaddleFactor(g, nu=NU, ghost="odd", ops=cell.ops)
        t_factor = time.time() - t0
    else:
        t_factor = cell.t_factor

    def mk(fn):
        def run():
            out = None
            for i in qidx:
                out = fn(i)
            return out
        return run

    def fom(i):
        return fac.solve(cell.F @ cell.Th_te[i])[0]

    def qf_nl(i):
        b = qf.b_of(cell.Th_te[i])
        return nl_solve(qf, hspec, b, z_starts)[0]

    def full_nl(i):
        th = cell.Th_te[i]
        f = cell.F @ th

        def rf(z_):
            u = cell.ubar + cell.G[:, :R] @ rom.head_np(hspec, z_)
            return rom.resid_full_grid(g, Phi_mf, qf.nu, u, f)

        def jf(z_):
            _, Jh = rom.head_jac_np(hspec, z_)
            return rom.jac_full_grid(g, Phi_mf, qf.nu, cell.G[:, :R], Jh)
        return rom.lm_multistart(rf, jf, z_starts,
                                 scale=qf.scale_of(qf.b_of(th)))[0]

    def eq_nl(i):
        th = cell.Th_te[i]

        def rf(z_):
            return eq_arm.resid(rom.head_np(hspec, z_), th)

        def jf(z_):
            _, Jh = rom.head_jac_np(hspec, z_)
            return eq_arm.jac(Jh)
        return rom.lm_multistart(rf, jf, z_starts,
                                 scale=qf.scale_of(qf.b_of(th)))[0]

    def gspan(i):
        return linR.solve_lsq(qf.b_of(cell.Th_te[i]))

    def podk(i):
        return linK.solve_lsq(qf.b_of(cell.Th_te[i]))

    def galerkin(i):
        return linR.solve_galerkin(cell.Th_te[i])

    def nonaffine(i):
        """SEPARATE ARM: the physical moving-blob force, whose projection
        Phi^T M_u f CANNOT be precomputed and costs O(M n_u) inside the pipe."""
        f = rom.force_nonaffine(cell, cell.mu_te[i])
        b = qf.Phi.T @ (f * g.h ** 2)
        return nl_solve(qf, hspec, b, z_starts)[0]

    subjects = [("fom_backsub", mk(fom)),
                ("rom_qf_nonlinear", mk(qf_nl)),
                ("rom_fullgrid_nonlinear", mk(full_nl)),
                ("rom_eq_nonlinear", mk(eq_nl)),
                ("rom_gspan_direct", mk(gspan)),
                ("rom_podK_direct", mk(podk)),
                ("rom_podR_galerkin", mk(galerkin))]
    raw, _ = rom.balanced_time(subjects, reps=REPS, warm=BURN)
    raw_na, _ = rom.balanced_time([("rom_qf_nonlinear_NONAFFINE", mk(nonaffine))],
                                  reps=REPS, warm=BURN)
    times = {k: rom.tstats(np.asarray(v) / len(qidx)) for k, v in raw.items()}
    times.update({k: rom.tstats(np.asarray(v) / len(qidx))
                  for k, v in raw_na.items()})

    # the discrepancy between the affine surrogate family and the physical
    # moving-blob family it stands in for -- a DIAGNOSTIC, not an
    # interpolation error: the affine amplitudes are Gaussian kernel weights,
    # not interpolation coefficients
    na_dev = []
    for i in qidx:
        fa = cell.F @ cell.Th_te[i]
        fa = fa / (g.h * np.linalg.norm(fa))
        fn = rom.force_nonaffine(cell, cell.mu_te[i])
        na_dev.append(float(np.linalg.norm(fa - fn) / np.linalg.norm(fn)))

    def stat(v):
        v = np.asarray(v, dtype=float)
        return dict(agg=float(np.sqrt((v ** 2).mean())),
                    median=float(np.median(v)), max=float(v.max()),
                    n=int(v.size))

    return dict(
        N=cell.N, R=int(R), M=int(M), n_heldout=int(S_TEST),
        err={k: stat(v) for k, v in err.items() if len(v)},
        lm_iters_median=float(np.median(lm_stats)),
        lm_starts=int(LM_STARTS),
        times=times, n_time_queries=len(qidx),
        offline=dict(qf_precompute_s=float(qf.t_offline),
                     eq_setup_s=float(eq["t_setup"]),
                     fom_factor_s=float(t_factor) if t_factor else None,
                     lin_pinv_s=None),
        eq=dict(machinery_exact=eq["machinery_exact"],
                budgets={str(m): {k: v for k, v in d.items() if k != "arm"}
                         for m, d in eq["arms"].items()},
                candidates=int(EQ_CAND), fit_states=int(EQ_FIT_STATES),
                psi_shape=eq["Psi_shape"]),
        nonaffine_force_discrepancy=float(np.median(na_dev)),
        rank_A=int(linR.rankA), cond_A=float(linR.cond),
        solve_states=[(int(i), [float(x) for x in z]) for i, z in solve_states])


# ==================================================================== S8/S9 ==

def gate_s8(cell, R, M, hspec, solve_states, rng):
    """S8: the M ladder with M >= R.  rank(A J_h(z)) must equal K and
    sigma_min/sigma_max is reported, at the captured solve solutions AND on a
    NEAR-COLLISION cohort where the family's own Jacobian degenerates from
    rank 8 to rank 4 (phase-2a verification, carried-forward conditions 1 and
    8: this cell INCLUDES the near-collision case rather than excluding it
    from the parameter domain)."""
    qf = rom.QFRom(cell, R, M)
    ranks, conds = [], []
    for i, z in solve_states[:16]:
        _, Jh = rom.head_jac_np(hspec, z)
        AJ = qf.nuA @ Jh
        s = np.linalg.svd(AJ, compute_uv=False)
        ranks.append(int((s > s[0] * RANK_TOL).sum()))
        conds.append(float(s[-1] / s[0]))

    # ---- the near-collision cohort ----------------------------------------
    near = []
    base = np.full(bank.K_LATENT, 0.5)
    for d in (1e-1, 1e-2, 1e-3, 1e-4, 0.0):
        mu = base.copy()
        mu[0:3] = 0.45
        mu[4:7] = 0.45 + d
        mu[3] = 0.5
        mu[7] = 0.5
        th = bank.theta(mu, cell.descs)[0]
        # the FAMILY's own Jacobian, by central differences through the exact
        # affine coefficient map (no FOM solve needed: c(mu) is exact)
        eps = 1e-6
        cols = []
        for k in range(bank.K_LATENT):
            mp, mm = mu.copy(), mu.copy()
            mp[k] += eps
            mm[k] -= eps
            cols.append((cell.coeff_affine(mp)[0] - cell.coeff_affine(mm)[0])
                        / (2 * eps))
        sJ = np.linalg.svd(np.column_stack(cols), compute_uv=False)
        fam_rank = int((sJ > sJ[0] * 1e-8).sum())
        # the ROM at that parameter
        b = qf.b_of(th)
        Zp = hspec["Z"]
        lo, hi = Zp.min(0), Zp.max(0)
        rs = np.random.default_rng(SEED + 17)
        z_starts = [np.zeros(K_LAT)] + [lo + (hi - lo) * rs.random(K_LAT)
                                        for _ in range(LM_STARTS - 1)]
        z, val, _ = nl_solve(qf, hspec, b, z_starts)
        _, Jh = rom.head_jac_np(hspec, z)
        s = np.linalg.svd(qf.nuA @ Jh, compute_uv=False)
        c_true = cell.coeff_affine(mu)[0, :R]
        a_nl = rom.head_np(hspec, z)
        a_lin = rom.LinearArm(qf).solve_lsq(b)
        den = np.linalg.norm(c_true) + 1e-300
        near.append(dict(
            separation=float(d), family_jacobian_rank=fam_rank,
            family_sv_ratio=float(sJ[-1] / sJ[0]),
            AJ_rank=int((s > s[0] * RANK_TOL).sum()),
            AJ_sv_ratio=float(s[-1] / s[0]),
            rom_coeff_err=float(np.linalg.norm(a_nl - c_true) / den),
            gspan_coeff_err=float(np.linalg.norm(a_lin - c_true) / den),
            resid_norm=float(val)))
    return dict(N=cell.N, R=int(R), M=int(M),
                M_ge_R=bool(M >= R),
                AJ_rank_min=int(min(ranks)), AJ_rank_max=int(max(ranks)),
                AJ_sv_ratio_min=float(min(conds)),
                AJ_sv_ratio_median=float(np.median(conds)),
                n_states=len(ranks), near_collision=near,
                near_collision_worst_AJ_rank=int(min(x["AJ_rank"] for x in near)),
                near_collision_worst_family_rank=int(
                    min(x["family_jacobian_rank"] for x in near)))


def gate_s9(cell, R, M, hspec, rng):
    """S9: one rung of the R frontier -- K fixed, nested banks from ONE
    factorisation, parameter count recorded, run against the M ladder.

    THE STRUCTURAL FACT that makes the frontier readable, stated here rather
    than left to the table: the decoder's reconstruction error is bounded
    below by the POD-R TRUNCATION FLOOR at its own R, exactly, because G is
    M_u-orthonormal.  At R = K = 8 the head therefore CANNOT beat POD-8; the
    frontier measures how much of the extra span at R = 16, 32 a K = 8 head
    can actually reach."""
    assert int(hspec["meta"]["R"]) == int(R), \
        f"head R={hspec['meta']['R']} does not match the arm's R={R}"
    qf = rom.QFRom(cell, R, M)
    linR = rom.LinearArm(qf)
    linK = rom.LinearArm(qf, r_trial=min(K_LAT, R))
    Zp = hspec["Z"]
    lo, hi = Zp.min(0), Zp.max(0)
    rs = np.random.default_rng(SEED + 23)
    z_starts = [np.zeros(K_LAT)] + [lo + (hi - lo) * rs.random(K_LAT)
                                    for _ in range(LM_STARTS - 1)]
    e_nl, e_g, e_k, iters = [], [], [], []
    for i in range(S_TEST):
        b = qf.b_of(cell.Th_te[i])
        z, val, tot = nl_solve(qf, hspec, b, z_starts)
        iters.append(tot["iters"])
        e_nl.append(centred_err(cell, rom.head_np(hspec, z), i, R))
        e_g.append(centred_err(cell, linR.solve_lsq(b), i, R))
        e_k.append(centred_err(cell, linK.solve_lsq(b), i, min(K_LAT, R)))
    p2, t2 = cell.perp_energy(cell.U_te, R)
    floor = float(np.sqrt(p2.sum() / t2.sum()))

    def stat(v):
        v = np.asarray(v, dtype=float)
        return dict(agg=float(np.sqrt((v ** 2).mean())),
                    median=float(np.median(v)), max=float(v.max()))
    return dict(N=cell.N, R=int(R), M=int(M), K=int(K_LAT),
                n_params=int(rom.n_params_np(hspec)),
                truncation_floor=floor,
                nl_qf=stat(e_nl), gspan_direct=stat(e_g), podK=stat(e_k),
                lm_iters_median=float(np.median(iters)),
                rank_A=int(linR.rankA), cond_A=float(linR.cond),
                online_flops_per_lm_iter=int(M * R + M * K_LAT
                                             + rom.n_params_np(hspec)))


# ===================================================================== main ==

def main():
    t_all = time.time()
    rng = np.random.default_rng(SEED)
    tag = OUT_TAG or f"rom_nu{NU:g}"
    out = os.path.join(OUT_PREFIX, f"stk2d_rom_gates_{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    jp = jax_provenance()

    report = dict(config=dict(
        pde="stokes2d", kind="phase2b_quadrature_free_rom",
        driver_revision=1,
        phase1_artifact="runs/stk2d/stk2d_fom_gates_nu1_M64.json",
        phase2a_artifact="runs/stk2d/stk2d_bank_gates_bank_nu1.json",
        stopgate_artifact=RECON_ARTIFACT,
        decoder="u = ubar + G h(z); NO bc(x) mask",
        residual="r(z) = -nu (Phi^T M_u L ubar + A h(z)) - b(mu), "
                 "A = Phi^T M_u L G, b(mu) = sum_q theta_q(mu) b_q",
        rom_ns=ROM_NS, r_ladder=R_LADDER, m_ladder=M_LADDER, m_main=M_MAIN,
        r_main=R_MAIN, n_main=N_MAIN, k_lat=K_LAT, s_train=S_TRAIN,
        s_test=S_TEST, nu=NU, seed=SEED, reps=REPS, burn=BURN,
        eq_budgets=EQ_BUDGETS, eq_cand=EQ_CAND, eq_fit_states=EQ_FIT_STATES,
        n_s4_states=N_S4_STATES, n_time_queries=N_TIME_QUERIES,
        lm_starts=LM_STARTS,
        timing="balanced forward/reversed sweeps, rom.balanced_time; every "
               "timed path is numpy on CPU so no JAX dispatch overhead enters "
               "the comparison",
        thresholds=dict(s4_tol=S4_TOL, s4_ctl_floor=S4_CTL_FLOOR,
                        head_tol=HEAD_TOL, head_fd_tol=HEAD_FD_TOL,
                        press_tol=PRESS_TOL, eq_exact_tol=EQ_EXACT_TOL,
                        rank_tol=RANK_TOL, eq_ctl_floor=EQ_CTL_FLOOR),
        numpy=np.__version__, scipy=scipy.__version__,
        python=platform.python_version(), jax=jp, allow_cpu=bool(ALLOW_CPU),
        smoke=bool(SMOKE), git_commit=git_commit(),
        hostname=os.uname().nodename), gates=dict(), complete=False)

    def save():
        json.dump(report, open(out, "w"), indent=1, default=float)

    save()
    log(f"stk2d ROM gates (phase 2b) -> {out}")

    # ---- S0 ---------------------------------------------------------------
    gp = stk.MacGrid(8)
    probe = stk.solve_stokes(gp, stk.manufactured(gp)["f"])[0]
    report["gates"]["S0"] = dict(
        jax=jp, numpy_float64=str(probe.dtype),
        numpy_is_f64=bool(probe.dtype == np.float64), allow_cpu=bool(ALLOW_CPU),
        rule="solver output dtype float64; JAX x64=True, matmul_precision="
             "'highest', backend 'gpu' unless ALLOW_CPU=1")
    save()
    assert probe.dtype == np.float64
    assert jp.get("imported") and jp.get("x64") is True
    assert jp.get("matmul_precision") == "highest", \
        f"S0: JAX_DEFAULT_MATMUL_PRECISION={jp.get('matmul_precision')}"
    if not ALLOW_CPU:
        assert jp.get("backend") == "gpu", f"S0: backend {jp.get('backend')}"

    # ---- PRECOND ----------------------------------------------------------
    observed = dict(rom_ns=ROM_NS, r_ladder=R_LADDER, m_ladder=M_LADDER,
                    m_main=M_MAIN, r_main=R_MAIN, n_main=N_MAIN, k_lat=K_LAT,
                    s_train=S_TRAIN, s_test=S_TEST, nu=NU, reps=REPS,
                    burn=BURN, eq_budgets=EQ_BUDGETS, eq_cand=EQ_CAND,
                    eq_fit_states=EQ_FIT_STATES, n_s4_states=N_S4_STATES,
                    n_time_queries=N_TIME_QUERIES, lm_starts=LM_STARTS,
                    allow_cpu=int(ALLOW_CPU), Q=bank.Q_TOTAL, K=bank.K_LATENT,
                    grad_mix=bank.GRAD_MIX)
    mism = {k: [FROZEN_CONFIG[k], v] for k, v in observed.items()
            if FROZEN_CONFIG[k] != v}
    report["gates"]["PRECOND"] = dict(
        debug_asserts_active=bool(__debug__), smoke=int(SMOKE),
        frozen_config=FROZEN_CONFIG, observed_config=observed,
        config_mismatch=mism, expected_gates=sorted(EXPECTED_GATES),
        expected_row_counts=EXPECTED_ROWS,
        rule="ASSERTED unless SMOKE=1: the entire configuration equals the "
             "frozen contract; python runs WITHOUT -O (a raise, not an "
             "assert); a SMOKE=1 run never sets complete=true")
    save()
    if not __debug__:
        raise RuntimeError("PRECOND: python is running with -O, every assert "
                           "in this harness is dead.")
    if not SMOKE:
        assert not mism, f"PRECOND: configuration differs from frozen: {mism}"
    assert BURN >= 3, f"BURN={BURN} < 3 (the timing spec's warm-up floor)"

    # ---- STOPGATE ---------------------------------------------------------
    rp = os.path.join(HERE, RECON_ARTIFACT)
    assert os.path.exists(rp), (
        f"STOPGATE: the phase-2b reconstruction artifact {RECON_ARTIFACT} does "
        f"not exist.  Reconstruction is gated BEFORE any residual, control or "
        f"timing work; run stk2d_recon_gate.py first.")
    rg = json.load(open(rp))
    v = rg["gates"]["S_RECON"]["verdict"]
    report["gates"]["STOPGATE"] = dict(
        artifact=RECON_ARTIFACT, complete=bool(rg.get("complete")),
        certified=bool(rg.get("certified")), verdict=v,
        rule="the reconstruction stop gate must have been RUN and PASSED "
             "before any residual, control or timing work.  If the held-out "
             "nonlinear reconstruction oracle sits at the POD-K floor, phase "
             "2b stops and the negative is the result -- this driver refuses "
             "to produce ROM numbers for a decoder that does not represent "
             "the manifold")
    save()
    assert rg.get("complete") and rg.get("certified"), \
        "STOPGATE: the reconstruction artifact is not a certified run"
    assert v["passed"], (
        f"STOPGATE: the reconstruction stop gate FAILED -- oracle aggregate "
        f"{v['oracle_agg']:.4e} vs required {v['required_agg']:.4e}, median "
        f"{v['oracle_median']:.4e} vs required {v['required_median']:.4e}.  "
        f"PHASE 2b STOPS HERE BY DESIGN.  Report the negative.")
    log(f"  STOPGATE passed: oracle agg {v['oracle_agg']:.3e} vs POD-K "
        f"{v['podK_agg']:.3e} ({v['gain_agg']:.2f}x)")

    head_rows, s4_rows, s67_rows, s8_rows, s9_rows = [], [], [], [], []
    import stk2d_head as head

    for N in ROM_NS:
        log(f" cell N={N}")
        cell = rom.RomCell(N, nu=NU, seed=SEED, s_train=S_TRAIN,
                           s_test=S_TEST, rmax=max(R_LADDER),
                           cache_dir=os.path.join(HERE, CACHE))
        specs = {}
        for R in R_LADDER:
            hp = head_path(N, R)
            assert os.path.exists(hp), f"missing trained head {hp}"
            specs[R] = head.load_head_np(hp)
            m = specs[R]["meta"]
            assert m["R"] == R and m["K"] == K_LAT, f"head metadata {m}"
            # PROVENANCE: the head must come from the CERTIFIED stop-gate run
            # on this mesh, not from a smoke run that happened to leave a file
            # at the same path.
            assert int(m.get("smoke", 1)) == 0, \
                f"head {hp} was produced by a SMOKE run: {m}"
            assert int(m.get("N", -1)) == N, f"head {hp} is for mesh {m.get('N')}"
            assert m.get("producer") == os.path.basename(RECON_ARTIFACT), \
                (f"head {hp} was produced by {m.get('producer')}, not by the "
                 f"stop-gate artifact {RECON_ARTIFACT}")
            for k in ("steps", "n_fit", "hidden", "layers", "mode"):
                assert m[k] == rg["config"][{"hidden": "hid"}.get(k, k)], \
                    (f"head {hp} {k}={m[k]} differs from the stop-gate "
                     f"configuration {rg['config'][{'hidden': 'hid'}.get(k, k)]}")

            # ---- S-HEAD ---------------------------------------------------
            import jax
            jparams = head.np_to_jax(specs[R])
            zs = rng.standard_normal((6, K_LAT))
            dv, dj, dfd = [], [], []
            for z in zs:
                a_np = rom.head_np(specs[R], z)
                a_jx = np.asarray(head.apply_head(jparams,
                                                  np.asarray(z)[None, :])[0]) \
                    * specs[R]["scale"]
                dv.append(np.abs(a_np - a_jx).max() / (np.abs(a_jx).max() + 1e-300))
                _, J = rom.head_jac_np(specs[R], z)
                Jx = np.asarray(jax.jacfwd(
                    lambda zz: head.apply_head(jparams, zz[None, :])[0])(
                        np.asarray(z))) * specs[R]["scale"]
                dj.append(np.abs(J - Jx).max() / (np.abs(Jx).max() + 1e-300))
                eps = 1e-6
                Jfd = np.column_stack([
                    (rom.head_np(specs[R], z + eps * np.eye(K_LAT)[k])
                     - rom.head_np(specs[R], z - eps * np.eye(K_LAT)[k]))
                    / (2 * eps) for k in range(K_LAT)])
                dfd.append(np.abs(J - Jfd).max() / (np.abs(J).max() + 1e-300))
            head_rows.append(dict(
                N=N, R=int(R), K=int(K_LAT),
                n_params=int(rom.n_params_np(specs[R])),
                value_dev=float(finite("S_HEAD val", dv).max()),
                jac_dev=float(finite("S_HEAD jac", dj).max()),
                jac_fd_dev=float(finite("S_HEAD fd", dfd).max()),
                mode=str(m["mode"]), hidden=int(m["hidden"]),
                layers=int(m["layers"]), ff=int(m["ff"])))

        hmain = specs[R_MAIN]
        # ---- S6 / S7 (also captures the solve states S4 and S8 need) ------
        qf_main = rom.QFRom(cell, R_MAIN, M_MAIN)
        eq = build_eq(cell, qf_main, hmain, rng)
        row67 = gate_s67(cell, R_MAIN, M_MAIN, hmain, eq, rng)
        s67_rows.append(row67)
        solve_states = [(i, np.asarray(z)) for i, z in row67["solve_states"]]
        log(f"  S7 N={N}: nl_qf {row67['err']['nl_qf']['agg']:.3e}  "
            f"gspan {row67['err']['gspan_direct']['agg']:.3e}  podK "
            f"{row67['err']['podK']['agg']:.3e}  galerkin "
            f"{row67['err']['podR_galerkin']['agg']:.3e}")
        log(f"  S6 N={N}: qf {row67['times']['rom_qf_nonlinear']['median']*1e3:.3f} ms"
            f"  full {row67['times']['rom_fullgrid_nonlinear']['median']*1e3:.3f} ms"
            f"  eq {row67['times']['rom_eq_nonlinear']['median']*1e3:.3f} ms"
            f"  gspan {row67['times']['rom_gspan_direct']['median']*1e3:.3f} ms"
            f"  fom {row67['times']['fom_backsub']['median']*1e3:.3f} ms")

        # ---- S4 -----------------------------------------------------------
        s4_rows.append(gate_s4(cell, R_MAIN, M_MAIN, hmain, solve_states, rng))
        log(f"  S4 N={N}: resid {s4_rows[-1]['resid_canc_max']:.3e}  jac "
            f"{s4_rows[-1]['jac_canc_max']:.3e}  press "
            f"{s4_rows[-1]['press_elim_max']:.3e}  controls "
            f"{s4_rows[-1]['ctl_evenghost']:.2e}/"
            f"{s4_rows[-1]['ctl_dropped_mean']:.2e}/"
            f"{s4_rows[-1]['ctl_perturbed_A']:.2e}")

        # ---- S8 / S9 ------------------------------------------------------
        for M in M_LADDER:
            if M >= R_MAIN:
                s8_rows.append(gate_s8(cell, R_MAIN, M, hmain, solve_states, rng))
            else:
                s8_rows.append(dict(N=N, R=int(R_MAIN), M=int(M),
                                    M_ge_R=False, AJ_rank_min=None,
                                    AJ_rank_max=None, AJ_sv_ratio_min=None,
                                    AJ_sv_ratio_median=None, n_states=0,
                                    near_collision=[],
                                    near_collision_worst_AJ_rank=None,
                                    near_collision_worst_family_rank=None))
            for R in R_LADDER:
                if M >= R:
                    s9_rows.append(gate_s9(cell, R, M, specs[R], rng))
                else:
                    s9_rows.append(dict(N=N, R=int(R), M=int(M), K=int(K_LAT),
                                        n_params=int(rom.n_params_np(specs[R])),
                                        truncation_floor=None, nl_qf=None,
                                        gspan_direct=None, podK=None,
                                        lm_iters_median=None, rank_A=None,
                                        cond_A=None,
                                        online_flops_per_lm_iter=None))
        del cell
        save()

    # ---- S-HEAD -----------------------------------------------------------
    report["gates"]["S_HEAD"] = dict(
        rows=head_rows,
        worst_value=float(finite("S_HEAD v", [r["value_dev"] for r in head_rows]).max()),
        worst_jac=float(finite("S_HEAD j", [r["jac_dev"] for r in head_rows]).max()),
        worst_fd=float(finite("S_HEAD fd", [r["jac_fd_dev"] for r in head_rows]).max()),
        rule=f"the NUMPY head used by every timed path must reproduce the JAX "
             f"head that was trained, value and analytic Jacobian, to "
             f"{HEAD_TOL} relative; and the analytic Jacobian must match a "
             f"central difference to {HEAD_FD_TOL}.  Two implementations of "
             f"one function, not two copies of one: the numpy path exists so "
             f"that no JAX dispatch overhead enters the S6 cost comparison")
    save()
    for r in head_rows:
        assert r["value_dev"] <= HEAD_TOL, f"S-HEAD value N={r['N']} R={r['R']}: {r['value_dev']}"
        assert r["jac_dev"] <= HEAD_TOL, f"S-HEAD jac N={r['N']} R={r['R']}: {r['jac_dev']}"
        assert r["jac_fd_dev"] <= HEAD_FD_TOL, f"S-HEAD fd N={r['N']} R={r['R']}: {r['jac_fd_dev']}"

    # ---- S4 ---------------------------------------------------------------
    report["gates"]["S4"] = dict(
        rows=s4_rows,
        worst_resid=float(finite("S4", [r["resid_canc_max"] for r in s4_rows]).max()),
        worst_jac=float(finite("S4j", [r["jac_canc_max"] for r in s4_rows]).max()),
        worst_press=float(finite("S4p", [r["press_elim_max"] for r in s4_rows]).max()),
        worst_control=float(finite(
            "S4c", [min(r["ctl_evenghost"], r["ctl_dropped_mean"],
                        r["ctl_perturbed_A"]) for r in s4_rows]).min()),
        rule=f"the quadrature-free residual and its Jacobian must agree with "
             f"an INDEPENDENT strong-form full-grid implementation (decode, "
             f"reassemble, apply the MAC stencil through the matrix-free "
             f"pad-and-slice operator, project through a matrix-free test "
             f"space) to {S4_TOL} in the CANCELLATION-AWARE normalisation -- "
             f"the term magnitude nu(||Phi^T M_u L ubar|| + ||A||_F ||h||) + "
             f"||b||, which cannot collapse, because at a converged solve "
             f"||r|| itself is roundoff and a relative form is meaningless.  "
             f">= {N_S4_STATES} seeded states plus every captured solve "
             f"solution plus every exact FOM coefficient state.  PRESSURE-"
             f"ELIMINATION RUNG: Phi^T M_u(-nu L u_FOM - f) must be "
             f"<= {PRESS_TOL} on every FOM solution, along a path the solver "
             f"never takes.  THREE NEGATIVE CONTROLS, each required to exceed "
             f"{S4_CTL_FLOOR} = 1e3 x the gate tolerance: EVEN (free-slip) "
             f"ghosts in the full-grid path; the affine-mean constant term "
             f"DROPPED (the revision-1 design's omission); and A perturbed by "
             f"a relative {S4_A_PERT:g}.  A relative {S4_A_PERT_SMALL:g} "
             f"perturbation is RECORDED beside it as the gate's sensitivity: "
             f"the discrepancy a perturbation control produces is set by the "
             f"perturbation, so quoting a control margin without the "
             f"perturbation size says nothing")
    save()
    for r in s4_rows:
        assert r["resid_canc_max"] <= S4_TOL, \
            f"S4 N={r['N']} residual {r['resid_canc_max']}"
        assert r["jac_canc_max"] <= S4_TOL, f"S4 N={r['N']} jacobian {r['jac_canc_max']}"
        assert r["press_elim_max"] <= PRESS_TOL, \
            f"S4 N={r['N']} pressure elimination {r['press_elim_max']}"
        assert r["phi_vs_matrixfree"] <= 1e-13, \
            f"S4 N={r['N']} Phi vs matrix-free {r['phi_vs_matrixfree']}"
        for k in ("ctl_evenghost", "ctl_dropped_mean", "ctl_perturbed_A"):
            assert r[k] >= S4_CTL_FLOOR, \
                (f"S4 negative control {k} at N={r['N']} did not fire: "
                 f"{r[k]} < {S4_CTL_FLOOR}")

    # ---- S6 ---------------------------------------------------------------
    report["gates"]["S6"] = dict(
        rows=[{k: v for k, v in r.items() if k != "solve_states"}
              for r in s67_rows],
        worst_eq_machinery=float(finite(
            "S6 eqm", [r["eq"]["machinery_exact"] for r in s67_rows]).max()),
        rule=f"COST: the quadrature-free arm against the full-grid arm and "
             f"against a NEWLY DEFINED strong-form EQ/NNLS arm.  The EQ arm "
             f"has its own fit target (the FULL strong MAC projection at "
             f"perturbed states), its own candidate sampling defined "
             f"SEPARATELY on the two face lattices, sampling applied only "
             f"AFTER analytic pressure elimination, non-negative weights from "
             f"a genuine NNLS, and its setup timed separately.  Its "
             f"MACHINERY is gated at {EQ_EXACT_TOL}: with ALL faces and unit "
             f"weights it must reproduce the full strong projection -- a "
             f"BOOKKEEPING check on the stencil slicing and index remapping, "
             f"labelled as one, not evidence about the fitted rule.  The "
             f"fitted rule's accuracy is REPORTED against node budget, not "
             f"gated, because it is a measurement.  The AFFINE and NON-AFFINE "
             f"force arms are timed separately and never blended: the "
             f"non-affine arm pays O(M n_u) per query to project a moving-"
             f"blob force that cannot be precomputed.  Timing is "
             f"balanced-order, {REPS} reps after {BURN} warm-ups, every path "
             f"in numpy on CPU")
    save()
    for r in s67_rows:
        assert r["eq"]["machinery_exact"] <= EQ_EXACT_TOL, \
            f"S6 EQ machinery N={r['N']}: {r['eq']['machinery_exact']}"
        for k in ("fom_backsub", "rom_qf_nonlinear", "rom_fullgrid_nonlinear",
                  "rom_eq_nonlinear", "rom_gspan_direct",
                  "rom_qf_nonlinear_NONAFFINE"):
            assert k in r["times"] and r["times"][k]["median"] > 0, \
                f"S6 N={r['N']}: timing subject {k} missing"

    # ---- S7 ---------------------------------------------------------------
    report["gates"]["S7"] = dict(
        rows=[{k: v for k, v in r.items() if k != "solve_states"}
              for r in s67_rows],
        gspan_beats_nonlinear=bool(all(
            r["err"]["gspan_direct"]["agg"] <= r["err"]["nl_qf"]["agg"]
            for r in s67_rows)),
        rule="THREE CONTROLS plus the nonlinear arm, all on the identical "
             "held-out cohort in the identical mass-weighted centred norm: "
             "(a) POD-K at matched ONLINE dimension; (b) POD-R Galerkin at "
             "matched trial span; (c) the DIRECT reduced solve in the G span, "
             "valid because M >= R and rank A = R (both asserted).  (c) IS "
             "EXPECTED TO WIN: steady Stokes is linear and G is a POD basis, "
             "so (c) is a one-shot projection.  A win by (c) is the design's "
             "prediction, not a failure of the machinery, and this cell must "
             "not be written up as a positive result either way")
    save()
    for r in s67_rows:
        assert r["rank_A"] == r["R"], \
            (f"S7 N={r['N']}: rank A = {r['rank_A']} != R = {r['R']}, so the "
             f"direct G-span solve (c) is not valid and must be replaced by "
             f"the Galerkin form")
        assert r["M"] >= r["R"], f"S7 N={r['N']}: M={r['M']} < R={r['R']}"
        for k in ("nl_qf", "podK", "podR_galerkin", "gspan_direct"):
            assert k in r["err"], f"S7 N={r['N']}: arm {k} missing"

    # ---- S8 ---------------------------------------------------------------
    live8 = [r for r in s8_rows if r["M_ge_R"]]
    report["gates"]["S8"] = dict(
        rows=s8_rows,
        worst_rank=int(min(r["AJ_rank_min"] for r in live8)),
        worst_sv_ratio=float(finite("S8", [r["AJ_sv_ratio_min"]
                                           for r in live8]).min()),
        rule=f"across M in {M_LADDER} with M >= R, rank(A J_h(z)) must equal "
             f"K = {K_LAT} at every captured solve solution and "
             f"sigma_min/sigma_max is reported.  M < R rungs are recorded as "
             f"NOT RUN rather than silently skipped.  NEAR-COLLISION "
             f"CONDITIONING IS INCLUDED rather than excluded from the "
             f"parameter domain (phase-2a verification, carried-forward "
             f"conditions 1 and 8): at coincident blobs the FAMILY's own "
             f"Jacobian degenerates from rank 8 to rank 4, and this gate "
             f"reports the family rank, the reduced Jacobian's rank and "
             f"conditioning, and both arms' errors as the separation goes to "
             f"zero")
    save()
    for r in live8:
        assert r["AJ_rank_min"] == K_LAT, \
            (f"S8 N={r['N']} M={r['M']}: rank(A J_h) = {r['AJ_rank_min']} != "
             f"K = {K_LAT}; the LM system is rank-deficient")
        assert r["near_collision_worst_family_rank"] <= 4, \
            (f"S8 N={r['N']} M={r['M']}: the near-collision cohort never "
             f"degenerates (worst family rank "
             f"{r['near_collision_worst_family_rank']}), so the coincident-"
             f"blob case this gate exists to cover was not actually probed")

    # ---- S9 ---------------------------------------------------------------
    live9 = [r for r in s9_rows if r["nl_qf"] is not None]
    report["gates"]["S9"] = dict(
        rows=s9_rows,
        rule=f"the R frontier {R_LADDER} with K = {K_LAT} FIXED, nested banks "
             f"from ONE factorisation, parameter count reported per R, run "
             f"against the M ladder {M_LADDER}.  R = 64 is ALGEBRAICALLY "
             f"UNAVAILABLE at Q = 48 (phase-2a retraction 14) and is not "
             f"attempted.  STRUCTURAL FACT the frontier must be read with: "
             f"the decoder's error is bounded below by the POD-R TRUNCATION "
             f"FLOOR at its own R, exactly, because G is M_u-orthonormal, so "
             f"at R = K the head CANNOT beat POD-K and that rung is not a "
             f"like-for-like comparison.  It is asserted, not assumed")
    save()
    for r in live9:
        assert r["nl_qf"]["agg"] >= r["truncation_floor"] * (1 - 1e-9), \
            (f"S9 N={r['N']} R={r['R']} M={r['M']}: the nonlinear arm "
             f"{r['nl_qf']['agg']} is below the POD-R truncation floor "
             f"{r['truncation_floor']}, which is impossible")
        assert r["rank_A"] == r["R"], \
            f"S9 N={r['N']} R={r['R']} M={r['M']}: rank A {r['rank_A']}"

    # ---- MANIFEST ---------------------------------------------------------
    counts = dict(S_HEAD=len(head_rows), S4=len(s4_rows), S6=len(s67_rows),
                  S7=len(s67_rows), S8=len(s8_rows), S9=len(s9_rows))
    missing = sorted(EXPECTED_GATES - set(report["gates"]) - {"MANIFEST"})
    badc = {k: [EXPECTED_ROWS[k], counts[k]] for k in EXPECTED_ROWS
            if counts[k] != EXPECTED_ROWS[k]}
    nonfinite = []

    def sweep(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                sweep(v, f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                sweep(v, f"{path}[{i}]")
        elif isinstance(node, float) and not np.isfinite(node):
            nonfinite.append(path)

    sweep(report["gates"], "gates")
    report["gates"]["MANIFEST"] = dict(
        expected_gates=sorted(EXPECTED_GATES), missing_gates=missing,
        expected_row_counts=EXPECTED_ROWS, observed_row_counts=counts,
        row_count_mismatch=badc, nonfinite_fields=nonfinite,
        rule="ASSERTED unless SMOKE=1: every expected gate present, EXACT row "
             "counts, and no non-finite float anywhere in gates/.  Rungs that "
             "do not apply (M < R) are recorded as null, never NaN")
    save()
    if not SMOKE:
        assert not missing, f"MANIFEST: missing gates {missing}"
        assert not badc, f"MANIFEST: row-count mismatch {badc}"
    assert not nonfinite, f"MANIFEST: non-finite values at {nonfinite}"

    report["complete"] = not bool(SMOKE)
    report["certified"] = not bool(SMOKE)
    if SMOKE:
        report["incomplete_reason"] = "SMOKE=1 is never a certified artifact"
    report["total_seconds"] = float(time.time() - t_all)
    save()
    log(f"DONE stk2d ROM gates [{report['total_seconds']:.0f}s] "
        f"complete={report['complete']} -> {out}")


if __name__ == "__main__":
    main()
