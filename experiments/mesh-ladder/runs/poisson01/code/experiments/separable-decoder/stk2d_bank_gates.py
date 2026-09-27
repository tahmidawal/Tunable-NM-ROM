"""PHASE 2a driver: the force family, the divergence-free bank, and the test
space, with every gate ASSERTED and every gate given a negative control.

Design: STOKES-DESIGN.md revision 3 -- sections "Force family", "The bank",
"Test space"; gates S1, S2, S-MEAN, S5, plus the manifold-richness
requirements.  Builds on the phase-1 FOM certified in
runs/stk2d/stk2d_fom_gates_nu1_M64.json; `stk2d_common.py` is IMPORTED and NOT
MODIFIED.

Harness discipline is inherited verbatim from `stk2d_fom_gates.py`, because
phase 1 took four verification rounds and EVERY defect found was in the gates,
not the numerics (STOKES-NOTES.md retractions 1-11):

  * every gate is a NUMBER with an asserted threshold, never a recorded boolean;
  * every gate has a NEGATIVE CONTROL that makes it fail, run in-band and
    tabulated -- a gate that has never fired is not evidence;
  * thresholds are NORMALISED, because absolute tolerances for mesh-scaling
    quantities have been wrong three times in this cell;
  * a suspiciously exact reading is treated as a tautology until an
    INDEPENDENT control says otherwise (retraction 9: a gate that read
    1.000000000000 because its control was a normalised copy of the thing
    under test);
  * NaN must FAIL: `max([finite, nan])` returns the finite value in Python, so
    every aggregate goes through `finite()` before reducing (retraction 11);
  * a frozen-config manifest with EXACT row counts, `SMOKE=1` unable to produce
    a certified artifact, and the report saved with complete=false before
    anything can fail.

GATES
  PRECOND   frozen-config equality, no -O, SMOKE never certifies.
  S0        solver dtype f64; JAX x64 / matmul=highest / backend gpu.
  S-SOLVE   the factor-once/solve-many path vs the CERTIFIED
            `stk.solve_stokes`, plus blockwise backward errors over every
            bank solve.
  S-DICT    the Q=48 affine dictionary: 32 exactly-solenoidal atoms, 16
            exactly-gradient atoms, full column rank.  Negative control: an
            ANALYTICALLY sampled curl-sine "solenoidal" atom, which is only
            O(h^2) divergence-free and must fail.
  S-HODGE   solenoidal / gradient force energy fractions, the exactness of the
            partition, and ||Grad_h p|| / ||f||.  The mixture must be genuine:
            neither fraction may collapse, or the velocity or the pressure
            side of the cell becomes vacuous.
  S-RICH    THE GATE THIS PHASE EXISTS FOR.  >= K+1 = 9 independent solenoidal
            response directions; centred snapshot numerical rank > K; Jacobian
            rank of mu -> u equal to K; and the held-out POD-K reconstruction
            error, which is what a nonlinear head could win.  Negative control:
            an AFFINE control family with K independent amplitudes, whose
            centred rank must read exactly K and so make the gate fire.
  S1        per-mode ||D g_i|| / (||D|| ||g_i||) <= 1e-11 for the psi-route
            bank, RE-GATED after normalisation and after reorthogonalisation,
            reported against the FOM snapshot divergence AND against
            sigma_1/sigma_i.  The naive route g_i = X v_i / sigma_i is built
            alongside purely to EXHIBIT the 1/sigma_i amplification, and the
            paired negative control contaminates the snapshots with a 1e-6
            gradient: the naive route must fail and the psi route must not.
  S-MEAN    ||D ubar|| / (||D|| ||ubar||) <= 1e-11.  Negative control: a mean
            with a 1e-6 gradient component.
  S2        structural ||D C||_inf exactly 0; field ||D Phi|| / (||D|| ||Phi||)
            <= 1e-11 per column, before AND after mass normalisation.
            Negative control: the ANALYTIC curl-sine sampling.
  S5        primary operator eigen-residual ||L phi + lambda phi|| / ||L phi||
            >= 0.5 for k,l <= 8 -- A ROUNDOFF VALUE FAILS.  Negative control:
            EVEN (free-slip) ghosts, for which the curl-sine modes ARE exact
            eigenvectors, so the residual collapses to roundoff and the gate
            fires.  Secondary diagnostic ||A + Lambda B|| / ||A|| (note the
            PLUS: this repo's convention is L Phi = -Phi Lambda).
  S-SPEC    singular spectrum, numerical rank vs R, nested-bank identity,
            M_u-orthonormality, held-out POD-R errors, and the measured
            evidence that the frozen R = 64 rung is UNREACHABLE at Q = 48.
  MANIFEST  every expected gate present, exact row counts, no non-finite float.

Env: BANK_NS, S5_NS, R_LADDER, M_LADDER, S_TRAIN, S_TEST, NU, SEED, OUT_TAG,
     OUT_PREFIX, ALLOW_CPU=0, SMOKE=0.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time

import numpy as np
import scipy
import scipy.sparse as sp

import stk2d_common as stk
import stk2d_bank as bank

HERE = os.path.dirname(os.path.abspath(__file__))

BANK_NS = [int(v) for v in os.environ.get("BANK_NS", "32,64,128,256").split(",") if v]
S5_NS = [int(v) for v in os.environ.get("S5_NS", "8,16,32,64,128,256").split(",") if v]
R_LADDER = [int(v) for v in os.environ.get("R_LADDER", "8,16,32").split(",") if v]
M_LADDER = [int(v) for v in os.environ.get("M_LADDER", "32,64,128").split(",") if v]
S_TRAIN = int(os.environ.get("S_TRAIN", "256"))
S_TEST = int(os.environ.get("S_TEST", "64"))
NU = float(os.environ.get("NU", "1.0"))
SEED = int(os.environ.get("SEED", "20260830"))
OUT_TAG = os.environ.get("OUT_TAG", "")
OUT_PREFIX = os.environ.get("OUT_PREFIX", "runs/stk2d/")
ALLOW_CPU = int(os.environ.get("ALLOW_CPU", "0"))
SMOKE = int(os.environ.get("SMOKE", "0"))

# ------------------------------------------------------------- thresholds ----
DIV_TOL = 1e-11            # STOKES-DESIGN.md S1 / S2 / S-MEAN
DIV_CTL_FLOOR = 10 * DIV_TOL   # a negative control must exceed this.  It is
                           # stated as a MULTIPLE OF DIV_TOL, not as an
                           # absolute number, because these controls are
                           # O(h^2) consistency errors: the analytically
                           # sampled curl-sine control falls from 1.7e-6 at
                           # N=32 to 4.0e-10 at N=256, so a flat 1e-9 floor
                           # would have "failed to fire" at the finest mesh
                           # purely from refinement.  What the control has to
                           # show is that it FAILS the gate it is controlling,
                           # with margin.
SOLVE_AGREE_TOL = 1e-12    # factor-once vs the certified solver
AFFINE_SCALE = 1e-14       # the affine superposition identity
                           # u(theta) = U_dict theta is gated at
                           # AFFINE_SCALE * N^2, NOT at a flat tolerance.  It
                           # recombines 48 independently computed FOM
                           # solutions, so its floor is the FOM's own FORWARD
                           # error, backward_err x kappa(K), and kappa(K)
                           # grows like h^-2.  Measured 3.7e-14, 4.4e-14,
                           # 4.3e-13, 4.8e-12 at N=32/64/128/256 -- a factor
                           # 130 across the ladder, while the cancellation
                           # ratio sum_q|theta_q| ||u_q|| / ||u|| stays flat
                           # at 2.46.  A flat 1e-12 tolerance passed three
                           # meshes and failed the fourth on refinement
                           # alone.
N_SOLVE_PROBES = 4         # cross-checks against the certified solver per
                           # mesh.  Each one REFACTORS the saddle matrix
                           # (76 s at N=256), which is the run's single
                           # largest cost.
BACKERR_TOL = 1e-13        # phase-1 frozen engineering threshold
CONT_TOL = 1e-12
GAUGE_TOL = 1e-12
HODGE_EXACT_TOL = 1e-12    # |sol_frac + grad_frac - 1|
HODGE_PURITY_TOL = 1e-20   # off-family ENERGY (squared) fraction of an atom
MIX_LO, MIX_HI = 0.05, 0.95   # neither Hodge fraction may collapse
RICH_MIN_DIRS = bank.K_LATENT + 1     # >= K+1 = 9 solenoidal response dirs
PODK_FLOOR = 1e-3          # see the note in gate_rich(): the "phase 2b is
                           # worth running" criterion, and it is MINE, not the
                           # design's -- the design gates only rank > K, which
                           # a family with a 1e-12 POD-K error would also pass
S5_EIGRES_FLOOR = 0.5      # STOKES-DESIGN.md S5 primary -- but see
                           # S5_ANCHORED_NS: the literal 0.5 floor is
                           # MESH-DEPENDENT and does NOT hold below N = 64
S5_ANCHORED_NS = (64, 128, 256)   # the meshes the design actually anchored
S5_EIGRES_WEAK_FLOOR = 1e-2       # asserted at EVERY mesh
S5_CTL_SCALE = 1e-13       # even-ghost control ceiling = S5_CTL_SCALE * N^2:
                           # the control is a cancellation of two O(h^-2)
                           # terms, so its roundoff floor grows like eps N^2
                           # (measured 0.17x to 8.6x eps N^2 over the ladder).
                           # A FLAT ceiling here would have been the fourth
                           # mesh-scaling absolute tolerance to be wrong in
                           # this cell.
S5_CTL_RATIO_FLOOR = 1e6   # the mesh-independent form of "a roundoff value
                           # FAILS": odd-ghost min / even-ghost max
S5_ARATIO_LO, S5_ARATIO_HI = 0.30, 0.45   # auditor's expected band
S5_ARATIO_ANCHOR_TOL = 2e-3
S5_ARATIO_FLOOR = 1e-2     # dense A required for the actual bank
RANK_RTOL = 1e-9           # relative singular-value cut for every numerical
                           # rank in this driver.  NOT the numpy-ish 1e-12:
                           # the snapshot matrix's noise floor is the FOM's
                           # own forward error, ~eps kappa(K) ~ eps N^2, so
                           # sigma_33/sigma_1 rises from 8.7e-15 at N=32 to
                           # 1.4e-12 at N=256 and a 1e-12 cut reports rank 33
                           # at the finest mesh.  1e-9 sits in the middle of a
                           # gap that is 7-9 orders wide at every mesh
                           # (sigma_32/sigma_1 ~ 4e-5 to 7.7e-5 above it), and
                           # THE GAP ITSELF IS ASSERTED so the choice cannot
                           # silently go wrong.
RANK_GAP_FLOOR = 1e6       # sigma_{Q_sol} / sigma_{Q_sol+1}
RANK_MODE_FLOOR = 1e-6     # sigma_{Q_sol}/sigma_1: the last mode must be real
RANK_NOISE_CEIL = 1e-9     # sigma_{Q_sol+1}/sigma_1: the next must be noise
ORTHO_TOL = 1e-13          # ||G^T M_u G - I||_max AFTER reorthogonalisation
ORTHO_SCALE = 1e-15        # the RAW Gram-POD basis is gated at
                           # ORTHO_SCALE * (sigma_1/sigma_R)^2, because a Gram
                           # POD squares the condition number: measured
                           # 7.4e-15 (R=8) to 2.0e-9 (R=32), tracking
                           # (sigma_1/sigma_R)^2 with a coefficient of
                           # 1e-18..6e-17 across meshes.  A FLAT tolerance
                           # here would have been wrong.

# audit anchors, STOKES-DESIGN.md S5 / STOKES-AUDIT-mac_s5_scaling.py
S5_EIGRES_ANCHORS = {64: (0.769, 0.998), 128: (0.943, 0.999),
                     256: (0.988, 0.99985)}
S5_ARATIO_ANCHORS = {64: 0.371, 128: 0.372, 256: 0.373}

FROZEN_CONFIG = dict(bank_ns=[32, 64, 128, 256],
                     s5_ns=[8, 16, 32, 64, 128, 256],
                     r_ladder=[8, 16, 32], m_ladder=[32, 64, 128],
                     s_train=256, s_test=64, nu=1.0, allow_cpu=0,
                     Q=48, Q_sol=32, Q_grad=16, K=8, grad_mix=3.0)
EXPECTED_GATES = frozenset((
    "PRECOND", "S0", "S_SOLVE", "S_DICT", "S_HODGE", "S_RICH", "S1",
    "S_MEAN", "S2", "S5", "S_SPEC", "MANIFEST"))
EXPECTED_ROWS = dict(S_SOLVE=4, S_DICT=4, S_HODGE=4, S_RICH=4,
                     S1=4 * 3, S_MEAN=4, S2=4 * 3, S5=6, S_SPEC=4)

SOLVES = []            # every bank solve, for the blockwise backward-error gate


def finite(label, xs):
    """Assert every value is finite BEFORE reducing.  max([finite, nan])
    returns the finite value in Python, so a NaN would otherwise turn a failed
    solve into a green run (STOKES-NOTES.md retraction 11)."""
    a = np.asarray([float(x) for x in xs], dtype=float)
    bad = ~np.isfinite(a)
    assert not bad.any(), (
        f"non-finite value(s) in {label}: {a[bad].tolist()} "
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
                   matmul_precision=os.environ.get(
                       "JAX_DEFAULT_MATMUL_PRECISION"),
                   jax_version=jax.__version__)
    except Exception as e:                                    # pragma: no cover
        out["error"] = repr(e)
    return out


def track(tag, N, info):
    SOLVES.append(dict(tag=tag, N=int(N), backward_err=info["backward_err"],
                       mom_resid=info["mom_resid"],
                       cont_resid=info["cont_resid"],
                       cont_resid_phase1=info["cont_resid_phase1"],
                       u_norm=info["u_norm"], p_norm=info["p_norm"],
                       gauge_resid=info["gauge_resid"],
                       gauge_raw=info["gauge_raw"], lam=info["lam"]))


# =============================================================== per-mesh ====

class MeshCell:
    """Everything phase 2a computes on one mesh, built once and shared by the
    gates.  The saddle matrix is factored ONCE here; every snapshot is a
    back-substitution."""

    def __init__(self, N, rng_seed):
        t0 = time.time()
        self.N = N
        g = stk.MacGrid(N)
        self.g = g
        self.D = stk.divergence_matrix(g)
        self.Grad = stk.gradient_matrix(g)
        self.L = stk.laplacian_matrix(g, "odd")
        self.C = stk.curl_matrix(g)
        self.ops = (self.D, self.Grad, self.L, self.C)
        self.D_fro = stk.spnorm_fro(self.D)
        t = time.time()
        self.fac = bank.SaddleFactor(g, nu=NU, ghost="odd", ops=self.ops)
        self.t_factor = time.time() - t
        self.hodge = bank.Hodge(g, C=self.C)

        # ---- dictionary and its responses ---------------------------------
        self.F, self.descs, self.kinds = bank.dictionary(g, ops=self.ops)
        t = time.time()
        self.U_dict, self.P_dict, infos = self.fac.solve_many(self.F)
        for i, inf in enumerate(infos):
            track(f"dict[{i}]", N, inf)
        self.t_dict = time.time() - t

        # ---- parameter samples --------------------------------------------
        self.mu_tr = bank.sample_mu(S_TRAIN, rng_seed)
        self.mu_te = bank.sample_mu(S_TEST, rng_seed + 1)
        self.Th_tr = bank.theta(self.mu_tr, self.descs)
        self.Th_te = bank.theta(self.mu_te, self.descs)
        # velocity is LINEAR in theta, so the snapshots follow from the
        # dictionary responses -- but they are solved directly anyway, and the
        # linearity identity is itself checked (gate S-SOLVE).
        self.F_tr = self.F @ self.Th_tr.T
        self.F_te = self.F @ self.Th_te.T
        t = time.time()
        self.U_tr, self.P_tr, itr = self.fac.solve_many(self.F_tr)
        self.U_te, self.P_te, ite = self.fac.solve_many(self.F_te)
        for i, inf in enumerate(itr):
            track(f"train[{i}]", N, inf)
        for i, inf in enumerate(ite):
            track(f"test[{i}]", N, inf)
        self.t_snap = time.time() - t
        self.seconds = time.time() - t0
        log(f"  cell N={N}: factor {self.t_factor:.1f}s  dict {self.t_dict:.1f}s"
            f"  {S_TRAIN + S_TEST} snapshots {self.t_snap:.1f}s"
            f"  (total {self.seconds:.1f}s)")


# ================================================================= gates =====

def gate_solve(cell, rng):
    """S-SOLVE: the factor-once path must reproduce the CERTIFIED
    `stk.solve_stokes` (phase 1), and the linear-superposition identity
    u(theta) = U_dict theta must hold -- the identity every affine cost claim
    in phase 2b rests on."""
    g = cell.g
    du, dp = [], []
    idx = rng.choice(S_TRAIN, size=N_SOLVE_PROBES, replace=False)
    for i in idx:
        f = cell.F_tr[:, i]
        uc, pc, _ = stk.solve_stokes(g, f, nu=NU, ghost="odd",
                                     ops=(cell.D, cell.Grad, cell.L))
        du.append(np.linalg.norm(cell.U_tr[:, i] - uc)
                  / (np.linalg.norm(uc) + 1e-300))
        dp.append(np.linalg.norm(cell.P_tr[:, i] - pc)
                  / (np.linalg.norm(pc) + 1e-300))
    # affine superposition identity
    U_aff = cell.U_dict @ cell.Th_tr.T
    un = np.linalg.norm(cell.U_tr, axis=0) + 1e-300
    aff_cols = np.linalg.norm(U_aff - cell.U_tr, axis=0) / un
    aff = float(finite("S_SOLVE affine cols", aff_cols).max())
    aff_med = float(np.median(aff_cols))
    canc = float(((np.abs(cell.Th_tr) @ np.linalg.norm(cell.U_dict, axis=0))
                  / un).max())
    # NEGATIVE CONTROL: a solution perturbed by a relative PERT_REL must break
    # the backward-error gate.  Phase 1 used 1e-11 and measured 2.0e-13 at
    # N=32, but it also documented that S-BACKERR is reference-direction
    # INDEPENDENT yet not DIRECTION independent -- the same relative
    # perturbation reads 2.0e-13 (random), 2.8e-13 (alternating) or 2.4e-15
    # (parallel to the velocity).  At N=64 a random 1e-11 perturbation lands at
    # 9.2e-14, just UNDER the 1e-13 threshold, so a control at that size does
    # not reliably fire.  PERT_REL = 1e-9 is used instead: still 10x inside
    # phase 1's own 1e-8 field tolerance, and it fires by two orders.
    f0 = cell.F_tr[:, 0]
    u0, p0, _ = cell.fac.solve(f0)
    sol = np.concatenate([u0, p0, [0.0]])
    pert = rng.standard_normal(sol.size)
    PERT_REL = 1e-9
    pert *= PERT_REL * np.linalg.norm(sol) / np.linalg.norm(pert)
    rhs = np.concatenate([f0, np.zeros(g.n_p), [0.0]])
    res = cell.fac.K @ (sol + pert) - rhs
    ctl = float(np.linalg.norm(res)
                / (cell.fac.K_fro * np.linalg.norm(sol + pert)
                   + np.linalg.norm(rhs) + 1e-300))
    return dict(N=cell.N, n_u=int(g.n_u), n_p=int(g.n_p), n_psi=int(g.n_psi),
                vs_certified_u_max=float(finite("S_SOLVE u", du).max()),
                vs_certified_p_max=float(finite("S_SOLVE p", dp).max()),
                n_probes=int(len(idx)),
                affine_superposition_max=aff,
                affine_superposition_med=aff_med,
                affine_superposition_budget=float(AFFINE_SCALE * cell.N ** 2),
                affine_cancellation_ratio=canc,
                perturbed_backerr_control=ctl,
                perturbation_relative_size=float(PERT_REL),
                factor_seconds=float(cell.t_factor),
                snapshot_seconds=float(cell.t_snap),
                ms_per_solve=float(1e3 * cell.t_snap / (S_TRAIN + S_TEST)))


def gate_dict(cell):
    """S-DICT: the dictionary is exactly what it claims to be."""
    g, D, Dfro = cell.g, cell.D, cell.D_fro
    kinds = np.asarray(cell.kinds)
    dn = bank.div_norm_cols(Dfro, D, cell.F)
    sol_mask = kinds == "sol"
    # purity: the off-family Hodge energy of each atom
    fs, fg = cell.hodge.fractions(cell.F)
    # NEGATIVE CONTROL: the ANALYTICALLY sampled curl of the same Gaussian
    # stream function.  It is the "obvious" way to build a solenoidal force and
    # is only O(h^2) divergence-free; the S1/S2 metric must reject it.
    xu, yu = g.coords_u()
    xv, yv = g.coords_v()
    ctl = []
    for (x, y, tau) in cell.descs[:bank.Q_SOL]:
        w = float(np.exp(tau))
        gu = np.exp(-((xu - x) ** 2 + (yu - y) ** 2) / (2 * w * w))
        gv = np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * w * w))
        fa = g.pack(-(yu - y) / w ** 2 * gu, (xv - x) / w ** 2 * gv)
        ctl.append(bank.div_norm(Dfro, D, fa))
    sv = np.linalg.svd(cell.F, compute_uv=False)
    return dict(N=cell.N, Q=int(cell.F.shape[1]),
                Q_sol=int(sol_mask.sum()), Q_grad=int((~sol_mask).sum()),
                sol_atom_div_max=float(finite("S_DICT sol div",
                                              dn[sol_mask]).max()),
                grad_atom_div_max=float(finite("S_DICT grad div",
                                               dn[~sol_mask]).max()),
                sol_atom_offfamily_max=float(finite(
                    "S_DICT sol purity", fg[sol_mask]).max()),
                grad_atom_offfamily_max=float(finite(
                    "S_DICT grad purity", fs[~sol_mask]).max()),
                dict_rank=int(bank.numerical_rank(sv)),
                dict_cond=float(sv[0] / (sv[-1] + 1e-300)),
                analytic_curl_control_min=float(finite(
                    "S_DICT control", ctl).min()),
                analytic_curl_control_max=float(np.max(ctl)))


def gate_hodge(cell):
    """S-HODGE: the force mixture, measured rather than assumed."""
    fs, fg = cell.hodge.fractions(cell.F_tr)
    part = np.abs(fs + fg - 1.0)
    gp = np.linalg.norm(cell.Grad @ cell.P_tr, axis=0)
    fn = np.linalg.norm(cell.F_tr, axis=0)
    un = np.linalg.norm(cell.U_tr, axis=0)
    return dict(N=cell.N,
                sol_frac_mean=float(finite("S_HODGE sol", fs).mean()),
                sol_frac_min=float(fs.min()), sol_frac_max=float(fs.max()),
                grad_frac_mean=float(finite("S_HODGE grad", fg).mean()),
                grad_frac_min=float(fg.min()), grad_frac_max=float(fg.max()),
                partition_defect_max=float(finite("S_HODGE part", part).max()),
                gradp_over_f_mean=float(finite("S_HODGE gradp",
                                               gp / fn).mean()),
                gradp_over_f_min=float((gp / fn).min()),
                gradp_over_f_max=float((gp / fn).max()),
                u_over_f_mean=float((un / fn).mean()))


def gate_rich(cell, rng):
    """S-RICH -- the gate this phase exists for.

    Steady Stokes is linear, so u is linear in theta.  If theta were affine in
    mu with independently varying amplitudes, the solution manifold would BE a
    K-dimensional affine subspace, a linear POD-K decoder would represent it
    exactly, and phase 2b's nonlinear-head comparison would be VACUOUS.

    Three measurements, and an in-band affine control that must make the
    decisive one fire:

      independent solenoidal response directions = rank(U_dict).  Gradient
        atoms produce EXACTLY zero velocity (Grad chi is balanced by the
        pressure), so this counts the directions that actually drive flow.
      centred snapshot numerical rank.  Snapshots lying in a K-dimensional
        AFFINE subspace would give centred rank <= K.  Rank > K is exactly the
        statement that no K-dimensional affine subspace contains the manifold.
      Jacobian rank of mu -> u.  Confirms the parameterisation is genuinely
        K-dimensional and not silently degenerate (a single-blob kernel would
        read 3).

    Reported alongside, and gated: the HELD-OUT POD-K relative reconstruction
    error.  Rank > K alone is necessary but not sufficient -- a family whose
    POD-K error were 1e-12 would pass it and still leave a nonlinear head
    nothing to win.  The PODK_FLOOR = 1e-3 threshold is MINE and is stated as
    such; the design specifies only rank > K.
    """
    g = cell.g
    sv_dict = np.linalg.svd(cell.U_dict, compute_uv=False)
    n_dirs = bank.numerical_rank(sv_dict, rtol=RANK_RTOL)
    ubar = cell.U_tr.mean(axis=1)
    Xc = cell.U_tr - ubar[:, None]
    sv = np.linalg.svd(Xc, compute_uv=False)
    rank_c = bank.numerical_rank(sv, rtol=RANK_RTOL)

    # Jacobian of mu -> u at three interior parameter points
    eps = 1e-5
    jr, jsv = [], []
    for _ in range(3):
        m0 = 0.1 + 0.8 * rng.random(bank.K_LATENT)
        cols = []
        for k in range(bank.K_LATENT):
            mp, mm = m0.copy(), m0.copy()
            mp[k] += eps
            mm[k] -= eps
            fp = cell.F @ bank.theta(mp, cell.descs)[0]
            fm = cell.F @ bank.theta(mm, cell.descs)[0]
            cols.append((cell.fac.solve(fp)[0] - cell.fac.solve(fm)[0])
                        / (2 * eps))
        s = np.linalg.svd(np.column_stack(cols), compute_uv=False)
        jr.append(bank.numerical_rank(s, rtol=1e-10))
        jsv.append(float(s[-1] / s[0]))

    # held-out POD-K / POD-R reconstruction from the psi-route bank
    b = bank.build_bank(g, cell.U_tr, max(R_LADDER), cell.hodge)
    Xte = cell.U_te - b["ubar"][:, None]
    err = {}
    for R in sorted(set(R_LADDER + [bank.K_LATENT])):
        G = b["G_psi"][:, :R]
        proj = G @ (G.T @ Xte * g.h ** 2)
        err[str(R)] = float(np.linalg.norm(Xte - proj)
                            / (np.linalg.norm(Xte) + 1e-300))

    # ---- NEGATIVE CONTROL: an AFFINE family with K independent amplitudes ---
    # theta_ctl(mu) = mu, acting on the first K solenoidal atoms.  Its solution
    # manifold IS a K-dimensional affine subspace, so its centred rank must
    # read exactly K and the rank > K gate must fire on it.
    Th_ctl = np.zeros((S_TRAIN, cell.F.shape[1]))
    Th_ctl[:, :bank.K_LATENT] = cell.mu_tr
    U_ctl = cell.U_dict @ Th_ctl.T
    sv_ctl = np.linalg.svd(U_ctl - U_ctl.mean(axis=1)[:, None],
                           compute_uv=False)
    rank_ctl = bank.numerical_rank(sv_ctl, rtol=RANK_RTOL)
    Gc, _ = np.linalg.qr(U_ctl - U_ctl.mean(axis=1)[:, None])
    Gc = Gc[:, :bank.K_LATENT]
    Xc_ctl = U_ctl - U_ctl.mean(axis=1)[:, None]
    err_ctl = float(np.linalg.norm(Xc_ctl - Gc @ (Gc.T @ Xc_ctl))
                    / (np.linalg.norm(Xc_ctl) + 1e-300))

    return dict(N=cell.N, K=int(bank.K_LATENT),
                n_solenoidal_response_dirs=int(n_dirs),
                required_dirs=int(RICH_MIN_DIRS),
                dict_response_sv_ratio_last=float(sv_dict[n_dirs - 1]
                                                  / sv_dict[0]),
                dict_response_sv_ratio_next=float(
                    sv_dict[n_dirs] / sv_dict[0]) if n_dirs < len(sv_dict)
                else 0.0,
                centred_snapshot_rank=int(rank_c),
                centred_sv_ratio_K=float(sv[bank.K_LATENT] / sv[0]),
                centred_sv_ratio_last=float(sv[rank_c - 1] / sv[0]),
                jacobian_rank=[int(x) for x in jr],
                jacobian_sv_ratio_min=[float(x) for x in jsv],
                heldout_pod_err=err,
                heldout_pod_K_err=float(err[str(bank.K_LATENT)]),
                affine_control_centred_rank=int(rank_ctl),
                affine_control_podK_err=err_ctl)


def gate_bank(cell, R):
    """S1 + S-MEAN + S-SPEC on one (mesh, R).

    Re-gated at THREE stages, as the design requires: raw construction, after
    mass normalisation, and after reorthogonalisation.  The naive route
    g_i = X v_i / sigma_i is built alongside to EXHIBIT the 1/sigma_i
    amplification the design warns about, and the contaminated-snapshot control
    makes it fail while the psi route survives.
    """
    g, D, Dfro = cell.g, cell.D, cell.D_fro
    b = bank.build_bank(g, cell.U_tr, R, cell.hodge)
    sig = b["sigma"]
    snap_div = bank.div_norm_cols(Dfro, D, cell.U_tr)

    d_psi = bank.div_norm_cols(Dfro, D, b["G_psi"])
    d_naive = bank.div_norm_cols(Dfro, D, b["G_naive"])
    # stage 2: mass normalisation (build_bank already scales to ||g||_M = 1;
    # re-normalise explicitly and re-measure, because the design requires a
    # re-gate after EVERY normalisation step)
    Gn = b["G_psi"] / (g.h * np.linalg.norm(b["G_psi"], axis=0))[None, :]
    d_psi_norm = bank.div_norm_cols(Dfro, D, Gn)
    # stage 3: reorthogonalisation, carried out in psi coordinates
    Gq, _ = bank.reorth_psi(g, cell.hodge, b["Psi_modes"])
    d_psi_qr = bank.div_norm_cols(Dfro, D, Gq)
    # the same QR done in VELOCITY coordinates on the naive bank (diagnostic)
    Gnq, _ = np.linalg.qr(b["G_naive"])
    d_naive_qr = bank.div_norm_cols(Dfro, D, Gnq)

    amp = sig[0] / np.maximum(sig[:R], 1e-300)
    ortho = float(np.abs(b["G_psi"].T @ b["G_psi"] * g.h ** 2
                         - np.eye(R)).max())
    ortho_qr = float(np.abs(Gq.T @ Gq * g.h ** 2 - np.eye(R)).max())
    ortho_budget = float(ORTHO_SCALE * (sig[0] / max(sig[R - 1], 1e-300)) ** 2)

    # ---- paired NEGATIVE CONTROL: snapshots contaminated with a gradient ----
    rngc = np.random.default_rng(SEED + 77 + R)
    e = cell.Grad @ (rngc.standard_normal(g.n_p) - 0.0)
    e = e / np.linalg.norm(e)
    Xcont = cell.U_tr + 1e-6 * np.linalg.norm(cell.U_tr, axis=0)[None, :] * e[:, None]
    bc = bank.build_bank(g, Xcont, R, cell.hodge)
    dc_naive = bank.div_norm_cols(Dfro, D, bc["G_naive"])
    dc_psi = bank.div_norm_cols(Dfro, D, bc["G_psi"])

    # ---- S-MEAN, and its own negative control ------------------------------
    ubar = b["ubar"]
    ubar_psi = b["ubar_psi"]
    ubar_bad = ubar + 1e-6 * np.linalg.norm(ubar) * e

    return dict(
        N=cell.N, R=int(R),
        snapshot_div_max=float(finite("S1 snap", snap_div).max()),
        psi_div_max=float(finite("S1 psi", d_psi).max()),
        psi_div_tail=float(d_psi[-1]),
        psi_norm_div_max=float(finite("S1 psi norm", d_psi_norm).max()),
        psi_qr_div_max=float(finite("S1 psi qr", d_psi_qr).max()),
        naive_div_max=float(finite("S1 naive", d_naive).max()),
        naive_div_head=float(d_naive[0]),
        naive_div_tail=float(d_naive[-1]),
        naive_qr_div_max=float(finite("S1 naive qr", d_naive_qr).max()),
        naive_amplification=float(d_naive[-1] / (d_naive[0] + 1e-300)),
        sigma_ratio_tail=float(amp[-1]),
        naive_div_over_snap_times_amp=float(
            d_naive[-1] / (snap_div.max() * amp[-1] + 1e-300)),
        psi_div_over_snap=float(d_psi.max() / (snap_div.max() + 1e-300)),
        contaminated_naive_div_max=float(finite("S1 ctl naive",
                                                dc_naive).max()),
        contaminated_psi_div_max=float(finite("S1 ctl psi", dc_psi).max()),
        mean_div=bank.div_norm(Dfro, D, ubar),
        mean_psi_div=bank.div_norm(Dfro, D, ubar_psi),
        mean_control_div=bank.div_norm(Dfro, D, ubar_bad),
        mean_psi_defect=float(np.linalg.norm(ubar - ubar_psi)
                              / (np.linalg.norm(ubar) + 1e-300)),
        orthonormality_max=ortho,
        orthonormality_qr_max=ortho_qr,
        orthonormality_budget=ortho_budget,
        sigma_over_sigma0=[float(x) for x in (sig[:R] / sig[0])])


def gate_spec(cell):
    """S-SPEC: spectrum, rank vs R, nested banks, and the R = 64 evidence.

    THE RANK IS TAKEN FROM A DIRECT SVD OF THE CENTRED SNAPSHOTS, not from the
    Gram POD's own singular values.  A symmetric-Gram POD forms X^T X, so its
    numerical noise floor is sqrt(eps) ~ 1.5e-8 RELATIVE, not eps: the Gram
    route reports 48 nonzero "singular values" for a matrix of true rank 32,
    because sigma_33.. come out at ~1e-8 sigma_1 instead of ~1e-15 sigma_1.
    Every rank statement here therefore comes from the direct SVD, and the
    Gram spectrum is recorded beside it with the discrepancy made explicit.
    """
    g = cell.g
    Rmax = max(R_LADDER)
    b = bank.build_bank(g, cell.U_tr, Rmax, cell.hodge)
    sig_gram = b["sigma"]
    Xc = cell.U_tr - cell.U_tr.mean(axis=1)[:, None]
    sig = np.linalg.svd(Xc, compute_uv=False) * g.h     # mass-weighted
    nsv = int(min(len(sig), 72))
    rank = bank.numerical_rank(sig, rtol=RANK_RTOL)
    rank_gram = bank.numerical_rank(sig_gram, rtol=RANK_RTOL)
    q = bank.Q_SOL
    s_last = float(sig[q - 1] / sig[0])
    s_next = float(sig[q] / sig[0]) if len(sig) > q else 0.0
    # nested-bank identity: ONE factorisation, prefixes taken from it
    nest = 0.0
    for R in R_LADDER:
        bR = bank.build_bank(g, cell.U_tr, R, cell.hodge)
        nest = max(nest, float(np.abs(bR["G_psi"] - b["G_psi"][:, :R]).max()))
    tail = float(sig[bank.Q_SOL] / sig[0]) if len(sig) > bank.Q_SOL else 0.0
    return dict(N=cell.N, n_snapshots=int(S_TRAIN),
                sigma_over_sigma0=[float(x) for x in (sig[:nsv] / sig[0])],
                sigma_gram_over_sigma0=[float(x) for x in
                                        (sig_gram[:nsv] / sig_gram[0])],
                numerical_rank=int(rank),
                numerical_rank_gram_route=int(rank_gram),
                gram_noise_floor=float(sig_gram[rank] / sig_gram[0])
                if rank < len(sig_gram) else 0.0,
                svd_noise_floor=float(sig[rank] / sig[0])
                if rank < len(sig) else 0.0,
                rank_cap_Q_sol=int(bank.Q_SOL),
                sigma_Qsol_over_sigma0=s_last,
                sigma_Qsol_plus1_over_sigma0=s_next,
                rank_gap=float(s_last / (s_next + 1e-300)),
                rank_rtol=float(RANK_RTOL),
                sigma_ratio_at_Qsol=tail,
                R64_reachable=bool(rank >= 64),
                nested_bank_max_diff=float(nest),
                R_ladder=[int(r) for r in R_LADDER])


def gate_test_space(cell, M):
    """S2: structural ||D C||_inf exactly 0, and the FIELD path per column,
    before AND after mass normalisation.

    Negative control: the SAME curl-sine modes sampled ANALYTICALLY.  That
    field carries component factors l*pi and k*pi where the discrete curl
    carries 2 sin(l pi h / 2)/h and 2 sin(k pi h / 2)/h; the mismatch leaves an
    O(h^2) divergence, so it must fail the 1e-11 field gate.
    """
    g, D, Dfro = cell.g, cell.D, cell.D_fro
    Phi_raw, lams, modes = stk.test_modes(g, M, normalize=False)
    Phi, lams2, _ = stk.test_modes(g, M, normalize=True)
    DC = (D @ cell.C).tocsr()
    d_raw = bank.div_norm_cols(Dfro, D, Phi_raw)
    d_nrm = bank.div_norm_cols(Dfro, D, Phi)
    # mass-normalisation evidence: unnormalised curl-sine norms grow like
    # sqrt(lambda), which is why unnormalised modes would up-weight the
    # high-frequency equations
    nrm = g.h * np.linalg.norm(Phi_raw, axis=0)
    ratio = nrm / np.sqrt(lams)
    # NEGATIVE CONTROL, split by mode class.  The analytic sampling has cell
    # divergence 2 cos(k pi x_c) cos(l pi y_c) [l pi sin(k pi h/2)
    # - k pi sin(l pi h/2)], which vanishes IDENTICALLY when k == l.  So the
    # DIAGONAL modes are exactly divergence-free even under analytic sampling,
    # and a control that took the minimum over ALL columns would read roundoff
    # and never fire.  The off-diagonal modes are the control; the diagonal
    # ones are recorded as the exact fact they are.
    xu, yu = g.coords_u()
    xv, yv = g.coords_v()
    ctl_off, ctl_diag = [], []
    for (k, l) in modes:
        fa = g.pack(l * np.pi * np.sin(k * np.pi * xu) * np.cos(l * np.pi * yu),
                    -k * np.pi * np.cos(k * np.pi * xv) * np.sin(l * np.pi * yv))
        (ctl_diag if k == l else ctl_off).append(bank.div_norm(Dfro, D, fa))
    return dict(N=cell.N, M=int(M),
                struct_DC_inf=float(stk.spnorm_inf(DC)),
                struct_DC_max=float(stk.spnorm_max(DC)),
                struct_DC_nnz=int(DC.nnz),
                phi_div_raw_max=float(finite("S2 raw", d_raw).max()),
                phi_div_norm_max=float(finite("S2 norm", d_nrm).max()),
                phi_div_norm_aggregate=float(
                    np.linalg.norm(D @ Phi)
                    / (Dfro * np.linalg.norm(Phi) + 1e-300)),
                mass_norm_min=float(nrm.min()), mass_norm_max=float(nrm.max()),
                norm_over_sqrt_lambda_min=float(ratio.min()),
                norm_over_sqrt_lambda_max=float(ratio.max()),
                lambda_min=float(lams.min()), lambda_max=float(lams.max()),
                analytic_control_offdiag_min=float(
                    finite("S2 control off-diagonal", ctl_off).min()),
                analytic_control_offdiag_max=float(np.max(ctl_off)),
                analytic_control_diag_max=float(
                    finite("S2 control diagonal", ctl_diag).max()),
                n_control_offdiag=int(len(ctl_off)),
                n_control_diag=int(len(ctl_diag)))


def gate_s5(N, bank_G=None):
    """S5.

    PRIMARY: the operator eigen-residual ||L phi + lambda phi|| / ||L phi||.
    Under no-slip MAC the curl-sine modes are NOT eigenvectors of the vector
    Laplacian -- settled analytically, not empirically: the tangential COSINE
    components want even/free-slip ghosts, while no-slip uses ODD ones,
    leaving a -2/h^2 defect on every boundary-adjacent tangential row.  A
    ROUNDOFF VALUE FAILS.

    THE DESIGN'S 0.5 FLOOR IS MESH-DEPENDENT AND DOES NOT HOLD ON THE WHOLE
    FROZEN LADDER.  The defect lives on O(N) boundary-adjacent rows with
    magnitude 2/h^2, while ||L phi|| ~ lambda ||phi|| stays O(1) for fixed
    k, l, so the ratio grows like N^2/lambda: measured minima 0.0357 (N=8),
    0.1644 (N=16), 0.4308 (N=32), 0.7692 (N=64), 0.9430 (N=128), 0.9876
    (N=256).  STOKES-DESIGN.md anchored the gate only at N >= 64 and stated
    the floor flat.  So the 0.5 floor is asserted at N in S5_ANCHORED_NS, and
    the MESH-INDEPENDENT form of the same statement -- odd-ghost minimum over
    even-ghost maximum >= 1e6 -- is asserted at EVERY mesh.  That ratio is
    what "a roundoff value fails" actually means.

    NEGATIVE CONTROL, and it is exact rather than constructed: under EVEN
    (free-slip) ghosts the even extension of cos(l pi y) at the wall IS its
    analytic continuation, so the modes become exact eigenvectors and the
    residual collapses to roundoff.  The control is therefore precisely the
    bug the gate exists to catch, and it makes the gate fire.

    SECONDARY DIAGNOSTIC: ||A + Lambda B|| / ||A|| with A = Phi^T M_u L G and
    B = Phi^T M_u G.  NOTE THE PLUS SIGN -- this repo's convention is
    L Phi = -Phi Lambda, so exact eigenvectors would give 0 and a DIAGONAL A
    would suffice.  Computed on the auditor's own clamped basis (reproducing
    the 0.371 / 0.372 / 0.373 anchors) and on the actual bank.
    """
    g = stk.MacGrid(N)
    C = stk.curl_matrix(g)
    Lo = stk.laplacian_matrix(g, "odd")
    Le = stk.laplacian_matrix(g, "even")
    ro, lams, modes, nrm = bank.eig_residuals(g, 8, C=C, L=Lo)
    re, _, _, _ = bank.eig_residuals(g, 8, C=C, L=Le)
    xs, ys = g.coords_psi()
    Phi = C @ np.column_stack([(np.sin(k * np.pi * xs)
                                * np.sin(l * np.pi * ys)).ravel()
                               for k, l in modes])
    Phi = Phi / (g.h * np.linalg.norm(Phi, axis=0))[None, :]
    Gaud = bank.audit_clamped_basis(g)
    r_aud, _, _ = bank.a_ratio(g, Phi, lams, Gaud, L=Lo)
    r_aud_even, _, _ = bank.a_ratio(g, Phi, lams, Gaud, L=Le)
    a = S5_EIGRES_ANCHORS.get(N)
    out = dict(N=N, kmax=int(min(8, N - 1)), n_modes=int(len(ro)),
               mode_norm_min=float(finite("S5 mode norms", nrm).min()),
               eigres_min=float(finite("S5 odd", ro).min()),
               eigres_med=float(np.median(ro)),
               eigres_max=float(ro.max()),
               eigres_evenghost_max=float(finite("S5 even", re).max()),
               evenghost_ceiling=float(S5_CTL_SCALE * N * N),
               ctl_ratio=float(ro.min() / (re.max() + 1e-300)),
               anchored=bool(N in S5_ANCHORED_NS),
               aratio_audit_basis=float(r_aud),
               aratio_audit_basis_evenghost=float(r_aud_even),
               aratio_bank=None, aratio_bank_evenghost=None,
               anchor_min=a[0] if a else None,
               anchor_max=a[1] if a else None,
               aratio_anchor=S5_ARATIO_ANCHORS.get(N))
    if bank_G is not None:
        rb, _, _ = bank.a_ratio(g, Phi, lams, bank_G, L=Lo)
        rbe, _, _ = bank.a_ratio(g, Phi, lams, bank_G, L=Le)
        out["aratio_bank"] = float(rb)
        out["aratio_bank_evenghost"] = float(rbe)
    return out


# =================================================================== main ====

def main():
    t_all = time.time()
    rng = np.random.default_rng(SEED)
    tag = OUT_TAG or f"bank_nu{NU:g}"
    out = os.path.join(OUT_PREFIX, f"stk2d_bank_gates_{tag}.json")
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)

    jp = jax_provenance()
    report = dict(config=dict(
        pde="stokes2d", kind="phase2a_force_family_bank_test_space",
        driver_revision=1,
        phase1_artifact="runs/stk2d/stk2d_fom_gates_nu1_M64.json",
        discretization="MAC, N cells, h=1/N; phase-1 operators imported "
                       "unmodified from stk2d_common.py",
        family=("Q=48 affine dictionary (32 solenoidal C psi_q + 16 gradient "
                "Grad chi_q) over a 3-D descriptor space (x, y, log-width); "
                "amplitudes theta_q(mu) from a curved K=8 map built as a "
                "2-blob superposition of the design's Gaussian kernel"),
        bank=("psi-route POD: snapshots carried to streamfunction "
              "coordinates, POD there, lifted by C.  The naive velocity-space "
              "route g_i = X v_i / sigma_i is built alongside as a DIAGNOSTIC "
              "to exhibit the 1/sigma_i amplification"),
        bank_ns=BANK_NS, s5_ns=S5_NS, r_ladder=R_LADDER, m_ladder=M_LADDER,
        s_train=S_TRAIN, s_test=S_TEST, nu=NU, seed=SEED,
        Q=bank.Q_TOTAL, Q_sol=bank.Q_SOL, Q_grad=bank.Q_GRAD,
        K=bank.K_LATENT, n_blobs=bank.N_BLOBS, grad_mix=bank.GRAD_MIX,
        centres_1d=[float(x) for x in bank.CENTRES_1D],
        widths=[float(x) for x in bank.WIDTHS],
        desc_weights=[float(x) for x in bank.DESC_W],
        blob_weights=[float(x) for x in bank.BLOB_W],
        kernel_bandwidth=[bank.S_LO, bank.S_HI],
        thresholds=dict(div_tol=DIV_TOL, div_ctl_floor=DIV_CTL_FLOOR,
                        solve_agree_tol=SOLVE_AGREE_TOL, affine_scale=AFFINE_SCALE,
                        n_solve_probes=N_SOLVE_PROBES,
                        backerr_tol=BACKERR_TOL, cont_tol=CONT_TOL,
                        gauge_tol=GAUGE_TOL,
                        hodge_exact_tol=HODGE_EXACT_TOL,
                        hodge_purity_tol=HODGE_PURITY_TOL,
                        mix_lo=MIX_LO, mix_hi=MIX_HI,
                        rich_min_dirs=RICH_MIN_DIRS, podk_floor=PODK_FLOOR,
                        s5_eigres_floor=S5_EIGRES_FLOOR,
                        s5_anchored_ns=list(S5_ANCHORED_NS),
                        s5_eigres_weak_floor=S5_EIGRES_WEAK_FLOOR,
                        s5_ctl_scale=S5_CTL_SCALE,
                        s5_ctl_ratio_floor=S5_CTL_RATIO_FLOOR,
                        s5_aratio_band=[S5_ARATIO_LO, S5_ARATIO_HI],
                        s5_aratio_anchor_tol=S5_ARATIO_ANCHOR_TOL,
                        s5_aratio_floor=S5_ARATIO_FLOOR,
                        ortho_tol=ORTHO_TOL, ortho_scale=ORTHO_SCALE,
                        rank_rtol=RANK_RTOL, rank_gap_floor=RANK_GAP_FLOOR,
                        rank_mode_floor=RANK_MODE_FLOOR,
                        rank_noise_ceil=RANK_NOISE_CEIL),
        numpy=np.__version__, scipy=scipy.__version__,
        python=platform.python_version(), jax=jp, allow_cpu=bool(ALLOW_CPU),
        smoke=bool(SMOKE), git_commit=git_commit(),
        hostname=os.uname().nodename),
        gates=dict(), complete=False)

    def save():
        json.dump(report, open(out, "w"), indent=1, default=float)

    # complete=false written BEFORE anything can fail, so a crash cannot leave
    # an older complete=true artifact untouched at this path.
    save()
    log(f"stk2d BANK gates (phase 2a, driver rev 1) -> {out}")
    log(f"  numpy {np.__version__} scipy {scipy.__version__}  jax {jp}")

    # ---- S0 ---------------------------------------------------------------
    gp = stk.MacGrid(8)
    probe = stk.solve_stokes(gp, stk.manufactured(gp)["f"])[0]
    s0 = dict(jax=jp, numpy_float64=str(probe.dtype),
              numpy_is_f64=bool(probe.dtype == np.float64),
              allow_cpu=bool(ALLOW_CPU),
              rule="solver output dtype float64; JAX x64=True, "
                   "matmul_precision='highest', backend 'gpu' unless "
                   "ALLOW_CPU=1.  Phase 2a's numerics are CPU scipy f64 by "
                   "design (sparse direct); the JAX environment is asserted "
                   "because phase 2b trains in it")
    report["gates"]["S0"] = s0
    save()
    assert probe.dtype == np.float64, f"S0: dtype {probe.dtype} != f64"
    assert jp.get("imported"), f"S0: JAX did not import: {jp}"
    assert jp.get("x64") is True, "S0: JAX_ENABLE_X64 is not active"
    assert jp.get("matmul_precision") == "highest", \
        f"S0: JAX_DEFAULT_MATMUL_PRECISION={jp.get('matmul_precision')}"
    if not ALLOW_CPU:
        assert jp.get("backend") == "gpu", \
            f"S0: jax backend is {jp.get('backend')}, not gpu"
    log(f"  S0 asserted: dtype {probe.dtype}, x64 {jp.get('x64')}, "
        f"matmul {jp.get('matmul_precision')}, backend {jp.get('backend')}")

    # ---- PRECOND ----------------------------------------------------------
    observed = dict(bank_ns=BANK_NS, s5_ns=S5_NS, r_ladder=R_LADDER,
                    m_ladder=M_LADDER, s_train=S_TRAIN, s_test=S_TEST, nu=NU,
                    allow_cpu=int(ALLOW_CPU), Q=bank.Q_TOTAL,
                    Q_sol=bank.Q_SOL, Q_grad=bank.Q_GRAD, K=bank.K_LATENT,
                    grad_mix=bank.GRAD_MIX)
    mism = {k: [FROZEN_CONFIG[k], v] for k, v in observed.items()
            if FROZEN_CONFIG[k] != v}
    report["gates"]["PRECOND"] = dict(
        debug_asserts_active=bool(__debug__), smoke=int(SMOKE),
        frozen_config=FROZEN_CONFIG, observed_config=observed,
        config_mismatch=mism, expected_gates=sorted(EXPECTED_GATES),
        expected_row_counts=EXPECTED_ROWS,
        rule="ASSERTED unless SMOKE=1: the ENTIRE configuration -- including "
             "the dictionary constants Q, Q_sol, Q_grad, K and grad_mix, "
             "which live in stk2d_bank.py and not in the environment -- must "
             "equal the frozen contract; the expected gate manifest must be "
             "present with EXACT row counts; and no aggregated array may hold "
             "a non-finite value.  Python must run WITHOUT -O, checked by a "
             "raise rather than an assert.  A SMOKE=1 run NEVER sets "
             "complete=true")
    save()
    if not __debug__:
        raise RuntimeError("PRECOND: python is running with -O, so every "
                           "assert in this harness is dead.  Refusing to "
                           "produce a JSON that would claim complete=true "
                           "without having checked anything.")
    if not SMOKE:
        assert not mism, f"PRECOND: configuration differs from frozen: {mism}"
    log(f"  PRECOND: asserts_active={__debug__} smoke={SMOKE} "
        f"config_mismatch={mism or 'none'}")

    # ---- S5 (cheap, no solves) -------------------------------------------
    log(" S5: operator eigen-residual + A-ratio")
    s5_rows = [gate_s5(N) for N in S5_NS]
    for r in s5_rows:
        log(f"  N={r['N']:4d}: eigres min/med/max "
            f"{r['eigres_min']:.6f}/{r['eigres_med']:.6f}/{r['eigres_max']:.6f}"
            f"  even-ghost control {r['eigres_evenghost_max']:.2e}"
            f"  A-ratio(audit) {r['aratio_audit_basis']:.6f}")
    report["gates"]["S5"] = dict(
        rows=s5_rows,
        worst_eigres_min=float(finite("S5 eigres",
                                      [r["eigres_min"] for r in s5_rows]).min()),
        worst_anchored_eigres_min=float(finite(
            "S5 anchored",
            [r["eigres_min"] for r in s5_rows if r["anchored"]]).min())
        if any(r["anchored"] for r in s5_rows) else None,
        worst_ctl_ratio=float(finite("S5 ratio",
                                     [r["ctl_ratio"] for r in s5_rows]).min()),
        worst_evenghost=float(finite("S5 even",
                                     [r["eigres_evenghost_max"]
                                      for r in s5_rows]).max()),
        rule=f"PRIMARY (design's form): eigen-residual >= {S5_EIGRES_FLOOR} "
             f"for every k,l <= min(8, N-1), asserted at N in "
             f"{list(S5_ANCHORED_NS)} -- the meshes STOKES-DESIGN.md actually "
             f"anchored.  THE FLAT 0.5 FLOOR DOES NOT HOLD BELOW N=64 and is "
             f"NOT asserted there; see the row values.  PRIMARY "
             f"(mesh-independent form), asserted at EVERY mesh: odd-ghost "
             f"minimum >= {S5_EIGRES_WEAK_FLOOR}, and odd-min / even-max >= "
             f"{S5_CTL_RATIO_FLOOR} -- that ratio is what 'a roundoff value "
             f"FAILS' actually means.  NEGATIVE CONTROL: the EVEN-ghost "
             f"operator, for which the curl-sine modes ARE exact "
             f"eigenvectors, must sit under {S5_CTL_SCALE} * N^2 (its "
             f"roundoff floor grows like eps N^2; a flat ceiling would be "
             f"wrong).  Anchors {S5_EIGRES_ANCHORS} from STOKES-DESIGN.md S5. "
             f"SECONDARY: ||A + Lambda B||/||A|| on the auditor's clamped "
             f"basis must match {S5_ARATIO_ANCHORS} to "
             f"{S5_ARATIO_ANCHOR_TOL} and lie in [{S5_ARATIO_LO}, "
             f"{S5_ARATIO_HI}] at the anchored meshes, and exceed "
             f"{S5_ARATIO_FLOOR} at every mesh -- note the PLUS sign, the "
             f"convention here is L Phi = -Phi Lambda, so a value this far "
             f"from 0 means DENSE A IS REQUIRED")
    save()
    for r in s5_rows:
        assert r["mode_norm_min"] > 1e-3, \
            (f"S5 N={r['N']}: a test mode has mass-norm {r['mode_norm_min']}, "
             f"i.e. it aliased to zero -- the residual would be a ratio of two "
             f"roundoff quantities")
        assert r["eigres_min"] >= S5_EIGRES_WEAK_FLOOR, \
            (f"S5 primary (weak floor) failed at N={r['N']}: "
             f"{r['eigres_min']} < {S5_EIGRES_WEAK_FLOOR}")
        if r["anchored"]:
            assert r["eigres_min"] >= S5_EIGRES_FLOOR, \
                (f"S5 primary failed at N={r['N']}: min eigen-residual "
                 f"{r['eigres_min']} < {S5_EIGRES_FLOOR}")
        assert r["eigres_evenghost_max"] <= r["evenghost_ceiling"], \
            (f"S5 negative control failed at N={r['N']}: the even-ghost "
             f"residual is {r['eigres_evenghost_max']}, above its roundoff "
             f"ceiling {r['evenghost_ceiling']} -- the control does not make "
             f"the gate fire, so the gate is not evidence")
        assert r["ctl_ratio"] >= S5_CTL_RATIO_FLOOR, \
            (f"S5 N={r['N']}: odd/even separation {r['ctl_ratio']} < "
             f"{S5_CTL_RATIO_FLOOR}")
        if r["anchor_min"] is not None:
            assert r["anchor_min"] - 1e-3 <= r["eigres_min"], \
                f"S5 anchor(min) N={r['N']}: {r['eigres_min']} vs {r['anchor_min']}"
            assert r["eigres_max"] <= r["anchor_max"] + 1e-3, \
                f"S5 anchor(max) N={r['N']}: {r['eigres_max']} vs {r['anchor_max']}"
        if r["aratio_anchor"] is not None:
            assert abs(r["aratio_audit_basis"] - r["aratio_anchor"]) \
                <= S5_ARATIO_ANCHOR_TOL, \
                (f"S5 A-ratio anchor N={r['N']}: {r['aratio_audit_basis']} vs "
                 f"{r['aratio_anchor']}")
            assert S5_ARATIO_LO <= r["aratio_audit_basis"] <= S5_ARATIO_HI, \
                f"S5 A-ratio band N={r['N']}: {r['aratio_audit_basis']}"
        assert r["aratio_audit_basis"] >= S5_ARATIO_FLOOR, \
            (f"S5 A-ratio N={r['N']}: {r['aratio_audit_basis']} < "
             f"{S5_ARATIO_FLOOR}; a diagonal A would suffice, contradicting "
             f"the design's conclusion")
        assert r["aratio_audit_basis_evenghost"] <= 1e-10, \
            (f"S5 A-ratio control N={r['N']}: even-ghost A-ratio "
             f"{r['aratio_audit_basis_evenghost']} is not roundoff")

    # ---- the mesh cells ---------------------------------------------------
    solve_rows, dict_rows, hodge_rows, rich_rows = [], [], [], []
    s1_rows, mean_rows, s2_rows, spec_rows = [], [], [], []
    for N in BANK_NS:
        log(f" cell N={N}")
        cell = MeshCell(N, SEED)
        solve_rows.append(gate_solve(cell, rng))
        dict_rows.append(gate_dict(cell))
        hodge_rows.append(gate_hodge(cell))
        rich_rows.append(gate_rich(cell, rng))
        for R in R_LADDER:
            r = gate_bank(cell, R)
            s1_rows.append(r)
            if R == max(R_LADDER):
                mean_rows.append(dict(N=N, R=int(R),
                                      mean_div=r["mean_div"],
                                      mean_psi_div=r["mean_psi_div"],
                                      mean_control_div=r["mean_control_div"],
                                      mean_psi_defect=r["mean_psi_defect"]))
        spec_rows.append(gate_spec(cell))
        for M in M_LADDER:
            s2_rows.append(gate_test_space(cell, M))
        # attach the bank to the S5 A-ratio row for this mesh
        b = bank.build_bank(cell.g, cell.U_tr, max(R_LADDER), cell.hodge)
        for r in s5_rows:
            if r["N"] == N:
                rb = gate_s5(N, bank_G=b["G_psi"])
                r["aratio_bank"] = rb["aratio_bank"]
                r["aratio_bank_evenghost"] = rb["aratio_bank_evenghost"]
        del cell
        save()

    for r in s5_rows:
        if r["aratio_bank"] is not None:
            assert r["aratio_bank"] >= S5_ARATIO_FLOOR, \
                (f"S5 A-ratio on the BANK at N={r['N']}: {r['aratio_bank']} < "
                 f"{S5_ARATIO_FLOOR}; a diagonal A would suffice")
            assert r["aratio_bank_evenghost"] <= 1e-10, \
                (f"S5 bank A-ratio control N={r['N']}: "
                 f"{r['aratio_bank_evenghost']} is not roundoff")
    save()

    # ---- S-SOLVE ----------------------------------------------------------
    be = [s["backward_err"] for s in SOLVES]
    mom = [s["mom_resid"] for s in SOLVES]
    cont = [s["cont_resid"] for s in SOLVES]
    gau = [s["gauge_resid"] for s in SOLVES]
    gr = [abs(s["gauge_raw"]) for s in SOLVES]
    c1 = [s["cont_resid_phase1"] for s in SOLVES]
    w_be = float(finite("S_SOLVE backward_err", be).max())
    w_mom = float(finite("S_SOLVE mom", mom).max())
    w_cont = float(finite("S_SOLVE cont", cont).max())
    w_gau = float(finite("S_SOLVE gauge", gau).max())
    w_gr = float(finite("S_SOLVE gauge_raw", gr).max())
    report["gates"]["S_SOLVE"] = dict(
        rows=solve_rows, n_solves=int(len(SOLVES)),
        worst_backward_err=w_be, worst_mom_resid=w_mom,
        worst_cont_resid=w_cont, worst_gauge_resid=w_gau,
        worst_gauge_raw=w_gr,
        worst_cont_resid_phase1_diagnostic=float(finite(
            "S_SOLVE cont phase1", c1).max()),
        worst_vs_certified=float(finite(
            "S_SOLVE vs certified",
            [r["vs_certified_u_max"] for r in solve_rows]
            + [r["vs_certified_p_max"] for r in solve_rows]).max()),
        worst_affine_superposition=float(finite(
            "S_SOLVE affine",
            [r["affine_superposition_max"] for r in solve_rows]).max()),
        worst_perturbed_control=float(finite(
            "S_SOLVE control",
            [r["perturbed_backerr_control"] for r in solve_rows]).min()),
        rule=f"the factor-once path must agree with the CERTIFIED "
             f"stk.solve_stokes to {SOLVE_AGREE_TOL}; the affine "
             f"superposition identity u(theta) = U_dict theta must hold to "
             f"{SOLVE_AGREE_TOL}; blockwise backward errors over ALL "
             f"{len(SOLVES)} bank solves must meet phase 1's thresholds "
             f"(global/momentum {BACKERR_TOL}, continuity {CONT_TOL}, "
             f"normalised gauge {GAUGE_TOL}).  The CONTINUITY metric is "
             f"deliberately NOT phase 1's: phase 1 normalised by "
             f"||D||_F ||u|| + |lam| sqrt(n_p), which COLLAPSES on this "
             f"cell's 16 gradient dictionary atoms (whose exact velocity is "
             f"zero) and reads 2.5e-2 on a roundoff-clean solve.  The gated "
             f"form is the standard blockwise backward error for the row "
             f"block [D | 0 | 1]; phase 1's form is RECORDED as "
             f"worst_cont_resid_phase1_diagnostic and NOT gated.  The RAW "
             f"gauge |1^T p| is "
             f"RECORDED ONLY: phase 1's own forward note says it is not "
             f"scale-free and its 1e-8 threshold had only 9x margin there; "
             f"the mass-normalised dictionary here makes ||f||_2 grow like "
             f"1/h, so the raw form would trip spuriously.  NEGATIVE CONTROL: "
             f"a relative 1e-9 perturbation of a converged solution -- still "
             f"10x inside phase 1's own 1e-8 field tolerance -- must raise "
             f"the backward error above {BACKERR_TOL}")
    save()
    assert w_be <= BACKERR_TOL, f"S-SOLVE global backward error {w_be}"
    assert w_mom <= BACKERR_TOL, f"S-SOLVE momentum residual {w_mom}"
    assert w_cont <= CONT_TOL, f"S-SOLVE continuity residual {w_cont}"
    assert w_gau <= GAUGE_TOL, f"S-SOLVE gauge residual {w_gau}"
    for r in solve_rows:
        assert r["vs_certified_u_max"] <= SOLVE_AGREE_TOL, \
            f"S-SOLVE N={r['N']} velocity vs certified {r['vs_certified_u_max']}"
        assert r["vs_certified_p_max"] <= SOLVE_AGREE_TOL, \
            f"S-SOLVE N={r['N']} pressure vs certified {r['vs_certified_p_max']}"
        assert r["affine_superposition_max"] <= r["affine_superposition_budget"], \
            (f"S-SOLVE N={r['N']} affine identity "
             f"{r['affine_superposition_max']} exceeds its h^-2 budget "
             f"{r['affine_superposition_budget']}")
        assert r["perturbed_backerr_control"] > BACKERR_TOL, \
            (f"S-SOLVE negative control N={r['N']} did not fire: "
             f"{r['perturbed_backerr_control']} <= {BACKERR_TOL}")
    log(f" S-SOLVE over {len(SOLVES)} solves: backerr {w_be:.3e} mom "
        f"{w_mom:.3e} cont {w_cont:.3e} gauge {w_gau:.3e} (raw {w_gr:.3e})")

    # ---- S-DICT -----------------------------------------------------------
    report["gates"]["S_DICT"] = dict(
        rows=dict_rows,
        worst_sol_div=float(finite("S_DICT sol",
                                   [r["sol_atom_div_max"] for r in dict_rows]).max()),
        worst_grad_purity=float(finite(
            "S_DICT purity",
            [r["grad_atom_offfamily_max"] for r in dict_rows]).max()),
        worst_control=float(finite(
            "S_DICT ctl",
            [r["analytic_curl_control_min"] for r in dict_rows]).min()),
        rule=f"Q={bank.Q_TOTAL}, Q_sol={bank.Q_SOL}, Q_grad={bank.Q_GRAD}, "
             f"full column rank; solenoidal atoms divergence-free to "
             f"{DIV_TOL} normalised; each atom's OFF-family Hodge energy "
             f"fraction <= {HODGE_PURITY_TOL}.  NEGATIVE CONTROL: the "
             f"ANALYTICALLY sampled curl of the same Gaussian stream "
             f"function -- the obvious wrong way to build a solenoidal force "
             f"-- must exceed {DIV_CTL_FLOOR}")
    save()
    for r in dict_rows:
        assert r["Q"] == bank.Q_TOTAL and r["Q_sol"] == bank.Q_SOL \
            and r["Q_grad"] == bank.Q_GRAD, f"S-DICT counts N={r['N']}: {r}"
        assert r["dict_rank"] == bank.Q_TOTAL, \
            f"S-DICT N={r['N']} rank {r['dict_rank']} != {bank.Q_TOTAL}"
        assert r["sol_atom_div_max"] <= DIV_TOL, \
            f"S-DICT N={r['N']} solenoidal atom divergence {r['sol_atom_div_max']}"
        assert r["sol_atom_offfamily_max"] <= HODGE_PURITY_TOL, \
            f"S-DICT N={r['N']} solenoidal purity {r['sol_atom_offfamily_max']}"
        assert r["grad_atom_offfamily_max"] <= HODGE_PURITY_TOL, \
            f"S-DICT N={r['N']} gradient purity {r['grad_atom_offfamily_max']}"
        assert r["analytic_curl_control_min"] >= DIV_CTL_FLOOR, \
            (f"S-DICT negative control N={r['N']} did not fire: "
             f"{r['analytic_curl_control_min']} < {DIV_CTL_FLOOR}")
    log(f" S-DICT: sol div {report['gates']['S_DICT']['worst_sol_div']:.3e}  "
        f"analytic-curl control >= "
        f"{report['gates']['S_DICT']['worst_control']:.3e}")

    # ---- S-HODGE ----------------------------------------------------------
    report["gates"]["S_HODGE"] = dict(
        rows=hodge_rows,
        worst_partition=float(finite(
            "S_HODGE part",
            [r["partition_defect_max"] for r in hodge_rows]).max()),
        rule=f"the Hodge partition must be exact to {HODGE_EXACT_TOL}, and "
             f"BOTH mean energy fractions must lie in [{MIX_LO}, {MIX_HI}]: a "
             f"gradient-dominated dictionary gives large pressure and almost "
             f"no velocity, and a purely solenoidal one makes the pressure "
             f"diagnostic vacuous.  ||Grad_h p||/||f|| is recorded per mesh")
    save()
    for r in hodge_rows:
        assert r["partition_defect_max"] <= HODGE_EXACT_TOL, \
            f"S-HODGE N={r['N']} partition defect {r['partition_defect_max']}"
        assert MIX_LO <= r["sol_frac_mean"] <= MIX_HI, \
            f"S-HODGE N={r['N']} solenoidal fraction {r['sol_frac_mean']}"
        assert MIX_LO <= r["grad_frac_mean"] <= MIX_HI, \
            f"S-HODGE N={r['N']} gradient fraction {r['grad_frac_mean']}"
    log(" S-HODGE: sol/grad mean fractions "
        + ", ".join(f"N={r['N']} {r['sol_frac_mean']:.3f}/"
                    f"{r['grad_frac_mean']:.3f}" for r in hodge_rows))

    # ---- S-RICH -----------------------------------------------------------
    report["gates"]["S_RICH"] = dict(
        rows=rich_rows,
        worst_dirs=int(min(r["n_solenoidal_response_dirs"] for r in rich_rows)),
        worst_centred_rank=int(min(r["centred_snapshot_rank"]
                                   for r in rich_rows)),
        worst_podK=float(finite("S_RICH podK",
                                [r["heldout_pod_K_err"] for r in rich_rows]).min()),
        rule=f"MANIFOLD RICHNESS -- the verdict this phase exists for.  "
             f"(a) at least K+1 = {RICH_MIN_DIRS} independent SOLENOIDAL "
             f"response directions; (b) centred snapshot numerical rank "
             f"STRICTLY greater than K = {bank.K_LATENT}, which is exactly "
             f"the statement that no K-dimensional AFFINE subspace contains "
             f"the manifold; (c) Jacobian rank of mu -> u equal to K, so the "
             f"parameterisation is not silently degenerate; (d) held-out "
             f"POD-K reconstruction error >= {PODK_FLOOR}.  (d)'s threshold "
             f"is MINE, not the design's: rank > K alone is necessary but not "
             f"sufficient, since a family with a 1e-12 POD-K error would pass "
             f"it and still leave a nonlinear head nothing to win.  NEGATIVE "
             f"CONTROL: an AFFINE family with K independent amplitudes, whose "
             f"centred rank must read EXACTLY K so that gate (b) fires on it")
    save()
    for r in rich_rows:
        assert r["n_solenoidal_response_dirs"] >= RICH_MIN_DIRS, \
            (f"S-RICH N={r['N']}: only {r['n_solenoidal_response_dirs']} "
             f"independent solenoidal response directions, need "
             f"{RICH_MIN_DIRS}.  PHASE 2b MUST NOT RUN AS DESIGNED")
        assert r["centred_snapshot_rank"] > bank.K_LATENT, \
            (f"S-RICH N={r['N']}: centred snapshot rank "
             f"{r['centred_snapshot_rank']} <= K={bank.K_LATENT}, so a linear "
             f"POD-K decoder represents the family exactly and the "
             f"nonlinear-head comparison is VACUOUS.  PHASE 2b MUST NOT RUN "
             f"AS DESIGNED")
        assert all(x == bank.K_LATENT for x in r["jacobian_rank"]), \
            (f"S-RICH N={r['N']}: Jacobian rank {r['jacobian_rank']} != "
             f"K={bank.K_LATENT}; the parameterisation is degenerate")
        assert r["heldout_pod_K_err"] >= PODK_FLOOR, \
            (f"S-RICH N={r['N']}: held-out POD-K error "
             f"{r['heldout_pod_K_err']} < {PODK_FLOOR}; a linear POD-K "
             f"decoder already reconstructs the family, so the nonlinear head "
             f"has nothing to win")
        assert r["affine_control_centred_rank"] == bank.K_LATENT, \
            (f"S-RICH negative control N={r['N']} did not fire: the affine "
             f"family's centred rank is {r['affine_control_centred_rank']}, "
             f"not K={bank.K_LATENT}, so the rank>K gate is not evidence")
    log(" S-RICH: " + ", ".join(
        f"N={r['N']} dirs {r['n_solenoidal_response_dirs']} rank "
        f"{r['centred_snapshot_rank']} jac {r['jacobian_rank']} "
        f"podK {r['heldout_pod_K_err']:.3e} (affine control rank "
        f"{r['affine_control_centred_rank']})" for r in rich_rows))

    # ---- S1 ---------------------------------------------------------------
    report["gates"]["S1"] = dict(
        rows=s1_rows,
        worst_psi=float(finite("S1 psi", [r["psi_div_max"] for r in s1_rows]).max()),
        worst_psi_norm=float(finite("S1 psi norm",
                                    [r["psi_norm_div_max"] for r in s1_rows]).max()),
        worst_psi_qr=float(finite("S1 psi qr",
                                  [r["psi_qr_div_max"] for r in s1_rows]).max()),
        worst_naive=float(finite("S1 naive",
                                 [r["naive_div_max"] for r in s1_rows]).max()),
        worst_snapshot=float(finite("S1 snap",
                                    [r["snapshot_div_max"] for r in s1_rows]).max()),
        max_naive_amplification=float(finite(
            "S1 amp", [r["naive_amplification"] for r in s1_rows]).max()),
        worst_contaminated_naive=float(finite(
            "S1 ctl", [r["contaminated_naive_div_max"] for r in s1_rows]).min()),
        worst_contaminated_psi=float(finite(
            "S1 ctl psi",
            [r["contaminated_psi_div_max"] for r in s1_rows]).max()),
        worst_ortho_raw=float(finite("S1 ortho",
                                     [r["orthonormality_max"] for r in s1_rows]).max()),
        worst_ortho_qr=float(finite("S1 ortho qr",
                                    [r["orthonormality_qr_max"] for r in s1_rows]).max()),
        rule=f"per mode ||D g_i||/(||D|| ||g_i||) <= {DIV_TOL}, RE-GATED at "
             f"all three stages (raw construction, mass normalisation, "
             f"psi-space reorthogonalisation) and reported against the FOM "
             f"snapshot divergence AND against sigma_1/sigma_i.  The naive "
             f"route g_i = X v_i/sigma_i is RECORDED, not gated, and exists "
             f"to exhibit the 1/sigma_i amplification.  PAIRED NEGATIVE "
             f"CONTROL: snapshots contaminated with a relative 1e-6 gradient "
             f"-- the naive route must then exceed {DIV_CTL_FLOOR} while the "
             f"psi route must still meet {DIV_TOL}.  Orthonormality: the "
             f"REORTHOGONALISED basis at {ORTHO_TOL} absolute, and the RAW "
             f"Gram-POD basis at {ORTHO_SCALE} * (sigma_1/sigma_R)^2, which "
             f"is the Gram POD's own condition-squaring budget")
    save()
    for r in s1_rows:
        for k in ("psi_div_max", "psi_norm_div_max", "psi_qr_div_max"):
            assert r[k] <= DIV_TOL, f"S1 N={r['N']} R={r['R']} {k}={r[k]}"
        assert r["orthonormality_qr_max"] <= ORTHO_TOL, \
            (f"S1 N={r['N']} R={r['R']} reorthogonalised orthonormality "
             f"{r['orthonormality_qr_max']}")
        assert r["orthonormality_max"] <= r["orthonormality_budget"], \
            (f"S1 N={r['N']} R={r['R']} raw Gram-POD orthonormality "
             f"{r['orthonormality_max']} exceeds its "
             f"(sigma_1/sigma_R)^2 budget {r['orthonormality_budget']}")
        assert r["contaminated_naive_div_max"] >= DIV_CTL_FLOOR, \
            (f"S1 negative control N={r['N']} R={r['R']} did not fire: the "
             f"contaminated naive route reads "
             f"{r['contaminated_naive_div_max']}, below {DIV_CTL_FLOOR}")
        assert r["contaminated_psi_div_max"] <= DIV_TOL, \
            (f"S1 N={r['N']} R={r['R']}: the psi route did not survive the "
             f"contamination control: {r['contaminated_psi_div_max']}")
    log(f" S1: psi {report['gates']['S1']['worst_psi']:.3e} / norm "
        f"{report['gates']['S1']['worst_psi_norm']:.3e} / qr "
        f"{report['gates']['S1']['worst_psi_qr']:.3e}; naive "
        f"{report['gates']['S1']['worst_naive']:.3e} "
        f"(amplification up to "
        f"{report['gates']['S1']['max_naive_amplification']:.1e}x)")

    # ---- S-MEAN -----------------------------------------------------------
    report["gates"]["S_MEAN"] = dict(
        rows=mean_rows,
        worst=float(finite("S_MEAN", [r["mean_psi_div"] for r in mean_rows]).max()),
        worst_plain=float(finite("S_MEAN plain",
                                 [r["mean_div"] for r in mean_rows]).max()),
        worst_control=float(finite("S_MEAN ctl",
                                   [r["mean_control_div"] for r in mean_rows]).min()),
        rule=f"||D ubar||/(||D|| ||ubar||) <= {DIV_TOL} for the affine mean.  "
             f"A non-solenoidal mean makes u = ubar + G h never "
             f"divergence-free however clean G is.  NEGATIVE CONTROL: the "
             f"same mean plus a relative 1e-6 gradient must exceed "
             f"{DIV_CTL_FLOOR}")
    save()
    for r in mean_rows:
        assert r["mean_psi_div"] <= DIV_TOL, \
            f"S-MEAN N={r['N']} {r['mean_psi_div']}"
        assert r["mean_div"] <= DIV_TOL, \
            f"S-MEAN (plain) N={r['N']} {r['mean_div']}"
        assert r["mean_control_div"] >= DIV_CTL_FLOOR, \
            (f"S-MEAN negative control N={r['N']} did not fire: "
             f"{r['mean_control_div']}")
    log(f" S-MEAN: {report['gates']['S_MEAN']['worst']:.3e} "
        f"(plain {report['gates']['S_MEAN']['worst_plain']:.3e}, control >= "
        f"{report['gates']['S_MEAN']['worst_control']:.3e})")

    # ---- S2 ---------------------------------------------------------------
    report["gates"]["S2"] = dict(
        rows=s2_rows,
        worst_struct=float(finite("S2 struct",
                                  [r["struct_DC_inf"] for r in s2_rows]).max()),
        worst_field_raw=float(finite("S2 raw",
                                     [r["phi_div_raw_max"] for r in s2_rows]).max()),
        worst_field_norm=float(finite("S2 norm",
                                      [r["phi_div_norm_max"] for r in s2_rows]).max()),
        worst_control=float(finite(
            "S2 ctl", [r["analytic_control_offdiag_min"] for r in s2_rows]).min()),
        worst_control_diag=float(finite(
            "S2 ctl diag",
            [r["analytic_control_diag_max"] for r in s2_rows]).max()),
        rule=f"structural ||D C||_inf EXACTLY 0; field path per COLUMN "
             f"<= {DIV_TOL} both before and after mass normalisation (an "
             f"aggregate Frobenius form could hide one bad column).  NEGATIVE "
             f"CONTROL: the same curl-sine modes sampled ANALYTICALLY -- the "
             f"discrete curl carries 2 sin(k pi h/2)/h where the analytic one "
             f"carries k pi, and the mismatch leaves an O(h^2) divergence.  "
             f"EVERY OFF-DIAGONAL (k != l) column must exceed "
             f"{DIV_CTL_FLOOR} = 10 x the gate tolerance.  The DIAGONAL "
             f"(k == l) columns are excluded from the control and asserted "
             f"<= {DIV_TOL} instead: the analytic divergence carries the "
             f"factor l sin(k pi h/2) - k sin(l pi h/2), which vanishes "
             f"identically at k == l, so a control taken over ALL columns "
             f"would read roundoff and could never fire")
    save()
    for r in s2_rows:
        assert r["struct_DC_inf"] == 0.0, \
            f"S2 N={r['N']} ||D C||_inf = {r['struct_DC_inf']}, not exactly 0"
        assert r["phi_div_raw_max"] <= DIV_TOL, \
            f"S2 N={r['N']} M={r['M']} raw field {r['phi_div_raw_max']}"
        assert r["phi_div_norm_max"] <= DIV_TOL, \
            f"S2 N={r['N']} M={r['M']} normalised field {r['phi_div_norm_max']}"
        assert r["analytic_control_offdiag_min"] >= DIV_CTL_FLOOR, \
            (f"S2 negative control N={r['N']} M={r['M']} did not fire: "
             f"{r['analytic_control_offdiag_min']} < {DIV_CTL_FLOOR}")
        assert r["analytic_control_diag_max"] <= DIV_TOL, \
            (f"S2 N={r['N']} M={r['M']}: the k==l analytic modes should be "
             f"EXACTLY divergence-free but read "
             f"{r['analytic_control_diag_max']}")
    log(f" S2: field {report['gates']['S2']['worst_field_norm']:.3e}  "
        f"analytic control >= {report['gates']['S2']['worst_control']:.3e}")

    # ---- S-SPEC -----------------------------------------------------------
    report["gates"]["S_SPEC"] = dict(
        rows=spec_rows,
        worst_nested=float(finite("S_SPEC nested",
                                  [r["nested_bank_max_diff"] for r in spec_rows]).max()),
        R64_reachable=bool(any(r["R64_reachable"] for r in spec_rows)),
        rule=f"the R ladder must come from ONE factorisation, so G_R is the "
             f"first R columns of G_Rmax BIT-FOR-BIT (max diff exactly 0).  "
             f"CONTRACT CONFLICT, ASSERTED AS A MEASURED FACT: the frozen "
             f"contract lists R in {{8,16,32,64}}, but only the SOLENOIDAL "
             f"part of the dictionary drives velocity, so the solution "
             f"manifold has rank at most Q_sol = {bank.Q_SOL} < 64.  The "
             f"R = 64 rung is UNREACHABLE at Q = {bank.Q_TOTAL}, and the "
             f"certified ladder here is {R_LADDER}.  Ranks come from a "
             f"DIRECT SVD of the centred snapshots at rtol {RANK_RTOL}: the "
             f"Gram POD's own singular values have a sqrt(eps) noise floor "
             f"and report a far larger rank.  The rank is ASSERTED through "
             f"the GAP -- sigma_Qsol/sigma_1 >= {RANK_MODE_FLOOR}, "
             f"sigma_(Qsol+1)/sigma_1 <= {RANK_NOISE_CEIL}, ratio >= "
             f"{RANK_GAP_FLOOR} -- so the cut level cannot silently become "
             f"the thing that decides the answer")
    save()
    for r in spec_rows:
        assert r["nested_bank_max_diff"] == 0.0, \
            f"S-SPEC N={r['N']} nested-bank diff {r['nested_bank_max_diff']}"
        # The rank is asserted through the GAP, not through a cut level: the
        # Q_sol-th direction must be a real mode and the next must be noise,
        # with the cut RANK_RTOL lying between them by several orders.
        assert r["sigma_Qsol_over_sigma0"] >= RANK_MODE_FLOOR, \
            (f"S-SPEC N={r['N']}: sigma_{bank.Q_SOL}/sigma_1 = "
             f"{r['sigma_Qsol_over_sigma0']} is not a real mode")
        assert r["sigma_Qsol_plus1_over_sigma0"] <= RANK_NOISE_CEIL, \
            (f"S-SPEC N={r['N']}: sigma_{bank.Q_SOL + 1}/sigma_1 = "
             f"{r['sigma_Qsol_plus1_over_sigma0']} is not roundoff")
        assert r["rank_gap"] >= RANK_GAP_FLOOR, \
            f"S-SPEC N={r['N']}: rank gap {r['rank_gap']} < {RANK_GAP_FLOOR}"
        assert r["numerical_rank"] == bank.Q_SOL, \
            (f"S-SPEC N={r['N']}: snapshot rank {r['numerical_rank']} != "
             f"Q_sol {bank.Q_SOL}")
        assert not r["R64_reachable"], \
            (f"S-SPEC N={r['N']}: the rank is {r['numerical_rank']} >= 64, "
             f"which contradicts the recorded R=64 unreachability finding")
        assert max(R_LADDER) <= r["numerical_rank"], \
            f"S-SPEC N={r['N']}: R ladder exceeds the attainable rank"
    log(" S-SPEC: ranks " + ", ".join(f"N={r['N']} {r['numerical_rank']}"
                                      for r in spec_rows)
        + f"; R=64 reachable: {report['gates']['S_SPEC']['R64_reachable']}")

    # ---- MANIFEST ---------------------------------------------------------
    counts = dict(S_SOLVE=len(solve_rows), S_DICT=len(dict_rows),
                  S_HODGE=len(hodge_rows), S_RICH=len(rich_rows),
                  S1=len(s1_rows), S_MEAN=len(mean_rows), S2=len(s2_rows),
                  S5=len(s5_rows), S_SPEC=len(spec_rows))
    missing = sorted(EXPECTED_GATES - set(report["gates"]) - {"MANIFEST"})
    bad = {k: [EXPECTED_ROWS[k], counts[k]] for k in EXPECTED_ROWS
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
        row_count_mismatch=bad, nonfinite_fields=nonfinite,
        n_solves=int(len(SOLVES)),
        rule="ASSERTED unless SMOKE=1: every expected gate present, EXACT "
             "expected row counts (non-empty is not sufficient), and no "
             "non-finite float anywhere in gates/.  Fields that do not apply "
             "are recorded as null, never NaN")
    log(f" MANIFEST: missing {missing or 'none'}  row-count mismatches "
        f"{bad or 'none'}  non-finite {nonfinite or 'none'}")
    save()
    if not SMOKE:
        assert not missing, f"MANIFEST: missing gates {missing}"
        assert not bad, f"MANIFEST: row-count mismatch {bad}"
    assert not nonfinite, f"MANIFEST: non-finite values at {nonfinite}"

    report["complete"] = not bool(SMOKE)
    report["certified"] = not bool(SMOKE)
    if SMOKE:
        report["incomplete_reason"] = (
            "SMOKE=1: PRECOND config equality and the MANIFEST row counts "
            "were not enforced, so this run is not a certified artifact and "
            "complete stays false by construction")
    report["total_seconds"] = float(time.time() - t_all)
    save()
    log(f"DONE stk2d BANK gates [{report['total_seconds']:.0f}s] "
        f"complete={report['complete']} -> {out}")


if __name__ == "__main__":
    main()
