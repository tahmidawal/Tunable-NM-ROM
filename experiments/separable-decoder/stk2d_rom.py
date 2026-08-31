"""PHASE 2b: the vector-valued decoder, the quadrature-free residual, and the
reduced-order solve.  Gates S4, S6, S7, S8, S9.

Design: STOKES-DESIGN.md revision 4 (the r4 frozen-contract amendments are
binding).  Builds strictly ON TOP of the certified phase-1 FOM
(`stk2d_common.py`) and the certified phase-2a bank (`stk2d_bank.py`): NO
operator, NO solver and NO bank-construction routine in either module is
modified here.

WHAT THIS PHASE IS FOR, restated because it governs how the numbers must be
read.  STOKES-DESIGN.md: "This cell is a de-risking rehearsal for 2D
incompressible Navier-Stokes.  It is not expected to produce a positive result,
and it must not be written up as one."  Steady Stokes is LINEAR and G is itself
a POD basis, so the direct reduced solve in the G span (gate S7c) is a one-shot
projection and is EXPECTED to beat the nonlinear head on both accuracy and
cost.  The deliverable is verified machinery plus honest gate numbers.

--------------------------------------------------------------- the decoder

    u(z) = ubar + G h(z),      G in R^{n_u x R},  h: R^K -> R^R.

NO bc(x) mask.  The mask is removed because multiplying a divergence-free field
by a scalar mask destroys divergence-freeness: div(bc . Gh) = grad(bc).(Gh) +
bc div(Gh), and the first term is not zero.  The bank already carries both
div-free-ness (phase 2a S1: ||D g_i|| ~ 1e-19) and the no-slip boundary
condition (every column is C psi with psi vanishing on the boundary, and the
boundary-normal velocities are eliminated degrees of freedom).

Because G is M_u-ORTHONORMAL (phase 2a, reorthogonalised to 1.3e-15) the whole
decoder problem collapses to COEFFICIENT space:

    ||u - ubar - G h(z)||_M^2 = ||c - h(z)||_2^2 + ||perp||_M^2,
    c = G^T M_u (u - ubar),   perp = (I - G G^T M_u)(u - ubar).

`perp` is the POD-R truncation floor and is INDEPENDENT of the head.  This is
why the head can NEVER beat POD-R at its own R: the reconstruction error is
bounded below by the truncation floor, exactly.  It is stated here rather than
discovered in the R-frontier table.

--------------------------------------------------- the quadrature-free residual

    r(z) = -nu ( Phi^T M_u L ubar + A h(z) ) - b(mu),   A = Phi^T M_u L G,

with A (M x R) and every b_q = Phi^T M_u f_q precomputed ONCE, and
b(mu) = sum_q theta_q(mu) b_q assembled per query in O(MQ).  Pressure never
appears: Phi is divergence-free, so Phi^T M_u Grad p = -(D Phi)^T M_p p = 0
exactly (phase 1 gate S-PRESS, 1.5e-15).

The honest per-query cost is  O(MQ) (assemble b)  +  per LM iteration
[ head evaluation + head Jacobian + O(MR) + O(MK) ].  "Cost per residual is MR"
alone omits the head and the query-time assembly and is not quoted here.

------------------------------------------------------- the affine coefficient map

Steady Stokes is linear and the dictionary is affine, so

    u(mu) = U_dict theta(mu)     (phase 2a gate S-SOLVE, gated at 1e-14 N^2)
    c(mu) = Cd theta(mu) - cbar, Cd = G^T M_u U_dict, cbar = G^T M_u ubar

EXACTLY.  Training coefficient vectors therefore cost one 48-column matvec each
once the 48 dictionary responses are solved, and the head's training cohort can
be enlarged without any further FOM work.  The BANK is still built from the
frozen phase-2a 256-snapshot training set and is bit-for-bit the phase-2a bank,
so carried-forward condition 3 (rerun S1 if the bank changes) is not triggered.

All numerics f64.  numpy/scipy on CPU for the operators and every TIMED path
(a sparse direct solve is the right tool, and a numpy timed pipe is comparable
across arms); JAX on GPU for training the head, with x64 on.
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.optimize as sopt

import stk2d_common as stk
import stk2d_bank as bank

PI = np.pi
HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================ the mesh cell ==

class RomCell:
    """Everything phase 2b needs on one mesh: the phase-2a snapshots, the
    phase-2a bank (reorthogonalised psi route), and the affine coefficient map.

    The dictionary responses, the 256 training snapshots and the 64 held-out
    snapshots are all DIRECT FOM solves through the phase-2a factor-once path,
    which phase 2a gated bit-for-bit against the certified `stk.solve_stokes`.
    They are cached on disk keyed by the full configuration; the cache is a
    convenience only and is regenerated whenever any key changes.
    """

    def __init__(self, N, nu=1.0, seed=20260830, s_train=256, s_test=64,
                 rmax=32, cache_dir=None, verbose=True):
        t0 = time.time()
        self.N, self.nu, self.seed = int(N), float(nu), int(seed)
        self.s_train, self.s_test, self.rmax = int(s_train), int(s_test), int(rmax)
        g = stk.MacGrid(N)
        self.g = g
        self.h = g.h
        self.D = stk.divergence_matrix(g)
        self.Grad = stk.gradient_matrix(g)
        self.L = stk.laplacian_matrix(g, "odd")
        self.C = stk.curl_matrix(g)
        self.ops = (self.D, self.Grad, self.L, self.C)
        self.D_fro = stk.spnorm_fro(self.D)
        self.hodge = bank.Hodge(g, C=self.C)
        self.F, self.descs, self.kinds = bank.dictionary(g, ops=self.ops)
        self.mu_tr = bank.sample_mu(s_train, seed)
        self.mu_te = bank.sample_mu(s_test, seed + 1)
        self.Th_tr = bank.theta(self.mu_tr, self.descs)
        self.Th_te = bank.theta(self.mu_te, self.descs)

        key = self._cache_key()
        path = (os.path.join(cache_dir, f"stk2d_cell_{key}.npz")
                if cache_dir else None)
        self.fac = None
        self.t_factor = None
        if path and os.path.exists(path):
            d = np.load(path)
            self.U_dict, self.U_tr, self.U_te = d["U_dict"], d["U_tr"], d["U_te"]
            self.from_cache = True
        else:
            t = time.time()
            self.fac = bank.SaddleFactor(g, nu=nu, ghost="odd", ops=self.ops)
            self.t_factor = time.time() - t
            self.U_dict, _, _ = self.fac.solve_many(self.F)
            self.U_tr, _, _ = self.fac.solve_many(self.F @ self.Th_tr.T)
            self.U_te, _, _ = self.fac.solve_many(self.F @ self.Th_te.T)
            self.from_cache = False
            if path:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                np.savez(path, U_dict=self.U_dict, U_tr=self.U_tr,
                         U_te=self.U_te)

        # ---- the phase-2a bank, reorthogonalised in the induced metric -----
        b = bank.build_bank(g, self.U_tr, rmax, self.hodge)
        Gq, Psi_q = bank.reorth_psi(g, self.hodge, b["Psi_modes"])
        self.ubar = b["ubar_psi"]        # ubar = C psibar: the psi-route mean
        self.ubar_plain = b["ubar"]
        self.G = Gq
        self.Psi_bank = Psi_q
        self.sigma = b["sigma"]
        self.G_raw = b["G_psi"]

        # ---- the exact affine coefficient map ------------------------------
        h2 = g.h ** 2
        self.Cd = h2 * (self.G.T @ self.U_dict)       # (R, Q)
        self.cbar = h2 * (self.G.T @ self.ubar)       # (R,)
        self.seconds = time.time() - t0
        if verbose:
            src = "cache" if self.from_cache else f"solve {self.seconds:.1f}s"
            print(f"  cell N={N}: n_u={g.n_u} ({src})", flush=True)

    def _cache_key(self):
        s = (f"N{self.N}_nu{self.nu:g}_seed{self.seed}_tr{self.s_train}"
             f"_te{self.s_test}_Q{bank.Q_TOTAL}_K{bank.K_LATENT}"
             f"_gm{bank.GRAD_MIX}")
        return s + "_" + hashlib.sha1(s.encode()).hexdigest()[:8]

    # -- coefficient helpers ------------------------------------------------
    def coeff_affine(self, mu):
        """c(mu) = Cd theta(mu) - cbar, EXACT (phase 2a's affine identity)."""
        return bank.theta(np.atleast_2d(mu), self.descs) @ self.Cd.T \
            - self.cbar[None, :]

    def coeff_of(self, U):
        """c = G^T M_u (U - ubar), columnwise; returns (S, R)."""
        return (self.g.h ** 2) * ((np.asarray(U) - self.ubar[:, None]).T @ self.G)

    def perp_energy(self, U, R=None):
        """||(I - G_R G_R^T M_u)(u - ubar)||_M^2 per column: the POD-R
        truncation floor, which no head at that R can beat, and the total
        centred energy ||u - ubar||_M^2 per column.

        THE RESIDUAL IS FORMED EXPLICITLY, not as ||x||^2 - ||c||^2.  The
        Pythagorean form is catastrophic here: at R = 32 the truncation energy
        is 1e-28 of the total, i.e. twelve orders BELOW the f64 resolution of
        the subtraction, and it returns 2.7e-8 relative instead of the true
        3.1e-14 -- which is a plausible-looking number, not an obvious one
        (retraction 24)."""
        R = self.rmax if R is None else int(R)
        X = np.asarray(U) - self.ubar[:, None]
        c = (self.g.h ** 2) * (X.T @ self.G[:, :R])
        E = X - self.G[:, :R] @ c.T
        h = self.g.h
        return ((h * np.linalg.norm(E, axis=0)) ** 2,
                (h * np.linalg.norm(X, axis=0)) ** 2)


# ============================================================== the decoder ==
# A plain MLP head h: R^K -> R^R, silu activations, linear skip.  Written twice
# on purpose: once in JAX (training, autodiff) and once in numpy (every TIMED
# path, so no JAX dispatch overhead enters the cost comparison).  Gate S4/S6
# assert the two agree.

def _silu_np(x):
    """silu(x) = x * sigmoid(x), with a branch-stable sigmoid.

    The naive 1/(1+exp(-x)) overflows for x <~ -709 and raises a RuntimeWarning
    on every LM step that wanders far from the data.  The value it produces is
    still correct (exp overflows to inf, the sigmoid underflows to 0), but a
    warning that fires routinely trains the reader to ignore warnings, and the
    LM DOES wander that far from a cold multi-start."""
    x = np.asarray(x, dtype=float)
    sg = np.empty_like(x)
    pos = x >= 0
    sg[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    sg[~pos] = e / (1.0 + e)
    return x * sg, sg


def _feat_np(spec, z):
    """zf = [z, sin(2 pi z B), cos(2 pi z B)] when latent Fourier features are
    in use, else zf = z.  Returns (zf, dzf/dz)."""
    B = spec.get("B")
    z = np.atleast_1d(np.asarray(z, dtype=float))
    K = z.shape[-1]
    if B is None:
        return z, np.eye(K)
    a = 2.0 * PI * (z @ B)
    zf = np.concatenate([z, np.sin(a), np.cos(a)], axis=-1)
    dz = np.vstack([np.eye(K),
                    2.0 * PI * (np.cos(a)[:, None] * B.T),
                    -2.0 * PI * (np.sin(a)[:, None] * B.T)])
    return zf, dz


def head_np(spec, z):
    """h(z) in numpy, in PHYSICAL coefficient units (the training scale is
    folded in).  Written separately from the JAX head on purpose: every TIMED
    path in this cell runs in numpy, so no JAX dispatch overhead enters the
    cost comparison.  Gate S-HEAD asserts the two agree."""
    W, b, skip, sc = spec["layers"], spec["biases"], spec["skip"], spec["scale"]
    z = np.asarray(z, dtype=float).ravel()
    x, _ = _feat_np(spec, z)
    for i in range(len(W) - 1):
        x, _ = _silu_np(x @ W[i] + b[i])
    return sc * (x @ W[-1] + b[-1] + z @ skip)


def head_jac_np(spec, z):
    """(h(z), dh/dz) in numpy, physical units, by an explicit forward chain
    rule.  Returns (R,), (R, K)."""
    W, b, skip, sc = spec["layers"], spec["biases"], spec["skip"], spec["scale"]
    z = np.asarray(z, dtype=float).ravel()
    x, J = _feat_np(spec, z)
    for i in range(len(W) - 1):
        a = x @ W[i] + b[i]
        x, sg = _silu_np(a)
        dphi = sg * (1.0 + a * (1.0 - sg))        # d/da [a sigma(a)]
        J = (W[i].T @ J) * dphi[:, None]
    out = x @ W[-1] + b[-1] + z @ skip
    return sc * out, sc * (W[-1].T @ J + skip.T)


def n_params_np(spec):
    n = sum(w.size for w in spec["layers"]) + sum(b.size for b in spec["biases"])
    n += spec["skip"].size
    if spec.get("B") is not None:
        n += spec["B"].size
    return int(n)


# =================================================== the quadrature-free ROM ==

class QFRom:
    """The precomputed quadrature-free reduced problem on one (mesh, R, M).

        r(z) = -nu ( phi_L_ubar + A h(z) ) - Bq theta(mu),   A = Phi^T M_u L G

    Every object here is built ONCE, offline.  Online, a query costs
    O(MQ) for b(mu) plus, per LM iteration, one head evaluation, one head
    Jacobian, and O(MR + MK).  Nothing scales with n_u.
    """

    def __init__(self, cell, R, M, nu=None):
        g = cell.g
        self.cell, self.R, self.M = cell, int(R), int(M)
        self.nu = float(cell.nu if nu is None else nu)
        t = time.time()          # the offline clock includes building Phi:
                                 # the test space is part of the precompute
        Phi, lams, modes = stk.test_modes(g, M)
        self.Phi, self.lams, self.modes = Phi, lams, modes
        h2 = g.h ** 2
        G = cell.G[:, :R]
        self.G = G
        self.A = Phi.T @ ((cell.L @ G) * h2)                     # (M, R)
        self.phi_L_ubar = Phi.T @ ((cell.L @ cell.ubar) * h2)    # (M,)
        self.Bq = Phi.T @ (cell.F * h2)                          # (M, Q)
        self.t_offline = time.time() - t
        self.const = -self.nu * self.phi_L_ubar
        self.nuA = -self.nu * self.A

    # -- online -------------------------------------------------------------
    def b_of(self, th):
        return self.Bq @ np.asarray(th)

    def resid(self, hz, b):
        return self.const + self.nuA @ np.asarray(hz) - b

    def scale_of(self, b):
        """The query's residual magnitude scale, used as the LM stopping
        reference.  Depends only on b, so it is computable before the solve."""
        return float(self.nu * np.linalg.norm(self.phi_L_ubar)
                     + np.linalg.norm(b) + 1e-300)

    def resid_scale(self, hz, b):
        """A term-magnitude scale that CANNOT collapse, used as the
        cancellation-aware denominator: near a solve stop ||r|| itself is
        roundoff and a relative form is meaningless."""
        return float(self.nu * (np.linalg.norm(self.phi_L_ubar)
                                + np.linalg.norm(self.A, "fro")
                                * np.linalg.norm(hz))
                     + np.linalg.norm(b) + 1e-300)


def phi_matrix_free(g, modes):
    """Phi built by the INDEPENDENT pad-and-slice vertex curl
    `stk.apply_curl`, mass-normalised by its own computation.  Shares only the
    (k, l) MODE LIST with `stk.test_modes` -- a specification, not a
    computation.  Used by the full-grid arm of gate S4 so that arm never
    touches the sparse assembly."""
    xs, ys = g.coords_psi()
    cols = []
    for k, l in modes:
        PSI = np.sin(k * PI * xs) * np.sin(l * PI * ys)
        U, V = stk.apply_curl(g, PSI)
        col = g.pack(U, V)
        cols.append(col / (g.h * np.linalg.norm(col)))
    return np.column_stack(cols)


def resid_full_grid(g, Phi_mf, nu, u, f):
    """The INDEPENDENT strong-form full-grid residual:

        decode -> reassemble -> apply the MAC stencil -> project.

    `stk.apply_laplacian` is the pad-and-slice matrix-free implementation,
    written independently of the sparse assembly and gated against it by phase
    1 (gate MF, 1.1e-16).  Phase 2a's S-SOLVE reads exactly 0.0 because both
    of its paths hand the same assembled matrix to the same SuperLU; it is an
    assembly regression check and CANNOT substitute for this (phase-2a
    verification, carried-forward condition 6)."""
    U, V = g.unpack(np.asarray(u))
    LU, LV = stk.apply_laplacian(g, U, V, "odd")
    Lu = g.pack(LU, LV)
    return Phi_mf.T @ ((-nu * Lu - np.asarray(f)) * g.h ** 2)


# ================================================================= LM solve ==

def lm_solve(resid_fn, jac_fn, z0, scale=1.0, max_iter=100, rtol=1e-12,
             xtol=1e-11, lam0=1e-6):
    """Damped Levenberg-Marquardt on ||r(z)||_2.

    THE STOPPING RULE IS RELATIVE, and it has to be.  ||r|| here is an
    unnormalised projected residual whose magnitude is set by ||b(mu)|| (about
    190 at N=32), so an ABSOLUTE tolerance of 1e-13 can never be met and every
    query silently runs the full iteration budget -- which would corrupt the
    S6 cost comparison rather than any accuracy number.  `scale` is the
    term-magnitude scale from `QFRom.scale_of`.

    Returns (z, ||r||, iters, n_resid, n_jac)."""
    z = np.asarray(z0, dtype=float).copy()
    r = resid_fn(z)
    val = float(np.linalg.norm(r))
    J = jac_fn(z)
    lam = lam0
    nres, njac, it = 1, 1, 0
    K = z.size
    stop = rtol * float(scale)
    for it in range(1, max_iter + 1):
        if val <= stop:
            break
        H = J.T @ J
        gr = J.T @ r
        dz = np.linalg.solve(H + lam * np.diag(np.diag(H)) + 1e-30 * np.eye(K),
                             -gr)
        zn = z + dz
        rn = resid_fn(zn)
        nres += 1
        vn = float(np.linalg.norm(rn))
        if np.isfinite(vn) and vn < val:
            z, r, val = zn, rn, vn
            J = jac_fn(z)
            njac += 1
            lam = max(lam / 3.0, 1e-14)
            if np.linalg.norm(dz) <= xtol * (1.0 + np.linalg.norm(z)):
                break
        else:
            lam = min(lam * 10.0, 1e14)
            if lam >= 1e14:
                break
    return z, val, it, nres, njac


def lm_multistart(resid_fn, jac_fn, z_starts, scale=1.0, **kw):
    """Multi-start LM.  Every start is run and the best residual kept; the
    COST reported for this arm includes every start, because the ROM has no
    way to know in advance which one converges."""
    best = None
    tot = dict(iters=0, nres=0, njac=0)
    for z0 in z_starts:
        z, val, it, nr, nj = lm_solve(resid_fn, jac_fn, z0, scale=scale, **kw)
        tot["iters"] += it
        tot["nres"] += nr
        tot["njac"] += nj
        if best is None or val < best[1]:
            best = (z, val)
    return best[0], best[1], tot


# ========================================================= the linear arms ===

class LinearArm:
    """The three S7 controls, all built from the SAME precomputed pieces.

    (a) POD-K  : trial span = the first K columns of G, matched ONLINE
                 dimension with the nonlinear head.
    (b) POD-R Galerkin : G^T M_u (-nu L)(ubar + G a) = G^T M_u f, an R x R
                 solve.  Legitimate here because G is divergence-free, so the
                 pressure drops out of a G-tested equation exactly.
    (c) direct G-span least squares : min_a ||A a + nu Phi^T M_u L ubar
                 + b(mu)||_2, an M x R least-squares solve, valid because
                 M >= R and rank A = R (both ASSERTED).  This is the control
                 the design says is EXPECTED TO WIN, and it is the reason this
                 cell is a rehearsal and not a result.
    """

    def __init__(self, qf, r_trial=None):
        self.qf = qf
        R = qf.R if r_trial is None else int(r_trial)
        self.R = R
        A = qf.A[:, :R]
        self.A = A
        self.nu = qf.nu
        # the residual is r = const - nu A a - b, so the least-squares solution
        # is a = pinv(-nu A) (b - const).  The -nu belongs INSIDE the pseudo-
        # inverse; leaving it out flips the sign and reads a relative error of
        # exactly 2.0 against the true coefficients (caught by the S7 check
        # against the FOM, not by inspection).
        self.pinv = np.linalg.pinv(-self.nu * A)      # offline (M x R -> R x M)
        s = np.linalg.svd(A, compute_uv=False)
        self.sv = s
        self.rankA = int((s > s[0] * 1e-12).sum())
        self.cond = float(s[0] / max(s[-1], 1e-300))
        # Galerkin: G^T M_u L G  (R x R), and G^T M_u L ubar, G^T M_u f_q
        cell = qf.cell
        h2 = cell.g.h ** 2
        G = cell.G[:, :R]
        self.KG = G.T @ ((cell.L @ G) * h2)
        self.KG_lu = np.linalg.inv(self.KG)
        self.g_L_ubar = G.T @ ((cell.L @ cell.ubar) * h2)
        self.Gq_f = G.T @ (cell.F * h2)

    def solve_lsq(self, b):
        """(c) / (a): one matvec with the precomputed pseudo-inverse."""
        return self.pinv @ (b - self.qf.const)

    def resid_of(self, a, b):
        """The same residual the nonlinear arm minimises, at a linear-arm
        coefficient vector, so the arms are compared on one quantity."""
        return self.qf.const - self.nu * (self.A @ a) - b

    def solve_galerkin(self, th):
        """(b): the R x R reduced Stokes solve, b assembled in O(RQ)."""
        return self.KG_lu @ (-(self.Gq_f @ th) / self.nu - self.g_L_ubar)


# ============================== empirical quadrature (strong form) ============

def eq_candidates(g, n_cand, seed):
    """Candidate face indices, drawn SEPARATELY on the two face lattices, as
    STOKES-DESIGN.md S6 requires.  Returns (idx, lattice_label)."""
    rng = np.random.default_rng(seed)
    nx = g.n_ux
    take = min(n_cand // 2, nx)
    a = rng.choice(nx, size=take, replace=False)
    b = nx + rng.choice(g.n_uy, size=min(n_cand - take, g.n_uy), replace=False)
    idx = np.concatenate([np.sort(a), np.sort(b)])
    lat = np.concatenate([np.zeros(a.size, int), np.ones(b.size, int)])
    return idx, lat


def eq_design(cell, qf, states, cand):
    """The EQ fit matrix.

    The strong-form projected residual is
        r_m = sum_j Phi[j,m] h^2 ( -nu (L u)[j] - f[j] ),
    a SUM OVER FACES.  Empirical quadrature replaces that sum by a weighted
    sum over a small node set, with weights fitted so the reduced sum
    reproduces the full one on training states.  Sampling is applied AFTER
    analytic pressure elimination -- the sampled quantity is the momentum
    residual only, and the pressure is already gone because Phi is
    divergence-free.

    Returns (Psi, targets) with Psi[(i,m), j] the per-face contribution and
    targets[(i,m)] the exact full-grid value.
    """
    g = cell.g
    h2 = g.h ** 2
    rows, tgt = [], []
    for (u, f) in states:
        U, V = g.unpack(u)
        LU, LV = stk.apply_laplacian(g, U, V, "odd")
        w = (-qf.nu * g.pack(LU, LV) - f) * h2         # per-face contribution
        rows.append(qf.Phi[cand, :].T * w[cand][None, :])   # (M, n_cand)
        tgt.append(qf.Phi.T @ w)                            # (M,)
    return np.vstack(rows), np.concatenate(tgt)


def eq_fit_greedy(Psi, tgt, m):
    """Non-negative greedy (ECSW / empirical-quadrature) selection of an
    ORDERED node support of size `m`.

    Selection uses the largest positive correlation of a candidate with the
    current residual -- the non-negativity of the criterion is what makes the
    support usable by a non-negative weight fit -- with an unconstrained
    least-squares residual update, which is cheap.  The WEIGHTS are then fitted
    by a genuine NNLS on the chosen support (`eq_weights`), once per node
    budget, because a quadrature rule with negative weights is not a
    quadrature rule.  Refitting NNLS inside every greedy step is equivalent
    here and costs minutes per mesh.
    """
    Psi = np.asarray(Psi, dtype=float)
    tgt = np.asarray(tgt, dtype=float)
    nt = np.linalg.norm(tgt) + 1e-300
    active, hist = [], []
    r = tgt.copy()
    # Incremental normal equations with a tiny ridge.  Calling
    # `np.linalg.lstsq` inside every greedy step uses the SVD driver and takes
    # minutes per mesh at a 512-node budget; the selection criterion does not
    # need that accuracy, and the WEIGHTS that are actually used come from the
    # NNLS in `eq_weights`, not from here.
    Gm = np.zeros((int(m), int(m)))
    Pt = np.zeros((int(m), Psi.shape[0]))
    for step in range(int(m)):
        corr = Psi.T @ r
        if active:
            corr[active] = -np.inf
        j = int(np.argmax(corr))
        if not np.isfinite(corr[j]) or corr[j] <= 0:
            break
        col = Psi[:, j]
        Pt[step] = col
        Gm[step, :step + 1] = Pt[:step + 1] @ col
        Gm[:step + 1, step] = Gm[step, :step + 1]
        active.append(j)
        n = step + 1
        A = Gm[:n, :n]
        w = np.linalg.solve(A + 1e-12 * np.trace(A) / n * np.eye(n),
                            Pt[:n] @ tgt)
        r = tgt - Pt[:n].T @ w
        hist.append(float(np.linalg.norm(r) / nt))
    return np.asarray(active, dtype=int), hist


def eq_weights(Psi, tgt, support):
    """NON-NEGATIVE least squares for the quadrature weights on a fixed
    support.  Returns (w, relative fit residual)."""
    w, _ = sopt.nnls(Psi[:, support], tgt)
    r = tgt - Psi[:, support] @ w
    return w, float(np.linalg.norm(r) / (np.linalg.norm(tgt) + 1e-300))


class EQArm:
    """The online strong-form EQ residual.

    Online it must (i) decode u on the union of the sampled faces' 5-point
    stencils, (ii) apply those rows of the MAC Laplacian, (iii) assemble f on
    the sampled faces from the affine dictionary, and (iv) contract with the
    weighted Phi rows.  Every cost is O(m R + m Q + m M) with m the node
    budget: grid-independent, but with a much larger constant than the
    quadrature-free O(MR), and it is only APPROXIMATE.
    """

    def __init__(self, cell, qf, nodes, w):
        self.cell, self.qf = cell, qf
        self.nodes = np.asarray(nodes, dtype=int)
        self.w = np.asarray(w, dtype=float)
        Lsub = cell.L[self.nodes, :].tocsr()
        ext = np.unique(Lsub.indices)
        self.ext = ext
        remap = -np.ones(cell.g.n_u, dtype=int)
        remap[ext] = np.arange(ext.size)
        self.Lsub = sp.csr_matrix(
            (Lsub.data, remap[Lsub.indices], Lsub.indptr),
            shape=(self.nodes.size, ext.size))
        self.G_ext = np.ascontiguousarray(cell.G[ext, :qf.R])
        self.ubar_ext = cell.ubar[ext].copy()
        self.F_nodes = np.ascontiguousarray(cell.F[self.nodes, :])
        h2 = cell.g.h ** 2
        self.PhiW = np.ascontiguousarray(
            (qf.Phi[self.nodes, :] * (self.w * h2)[:, None]))   # (m, M)
        self.nu = qf.nu

    def resid(self, hz, th):
        u_ext = self.ubar_ext + self.G_ext @ hz
        lu = self.Lsub @ u_ext
        f = self.F_nodes @ th
        return self.PhiW.T @ (-self.nu * lu - f)

    def jac(self, Jh):
        """dr/dz through the SAMPLED path, evaluated as a nonlinear operator
        would require it: lift the head Jacobian to the stencil rows, apply the
        sampled operator rows, contract with the weighted test rows.

        FINDING, stated where it is implemented rather than buried in a table:
        because this operator is LINEAR, PhiW^T (-nu Lsub G_ext) is a constant
        (M x R) matrix and could be precomputed once -- at which point the EQ
        arm collapses to EXACTLY the quadrature-free form, with an APPROXIMATE
        matrix in place of the exact one.  On a linear PDE empirical quadrature
        therefore has no cost story at all; it can only lose accuracy.  The
        general form is timed here because it is the one that carries to
        Navier-Stokes."""
        return self.PhiW.T @ (-self.nu * (self.Lsub @ (self.G_ext @ Jh)))

    def collapsed_A(self):
        """The (M, R) matrix the linear case collapses to -- the EQ arm's
        approximation of A, comparable entry-for-entry with the exact one."""
        return self.PhiW.T @ (-self.nu * (self.Lsub @ self.G_ext))


def jac_full_grid(g, Phi_mf, nu, G, Jh):
    """dr/dz through the INDEPENDENT full-grid path: lift, apply the MAC
    stencil to each latent direction, project.  O(n_u K) applies."""
    Y = np.asarray(G) @ np.asarray(Jh)              # (n_u, K)
    cols = []
    for k in range(Y.shape[1]):
        U, V = g.unpack(Y[:, k])
        LU, LV = stk.apply_laplacian(g, U, V, "odd")
        cols.append(g.pack(LU, LV))
    return Phi_mf.T @ ((-nu * np.column_stack(cols)) * g.h ** 2)


# ================================= the NON-AFFINE (moving-centre) force arm ==

def force_nonaffine(cell, mu):
    """The PHYSICAL moving-blob force the affine dictionary stands in for: two
    genuinely moving, genuinely rescaling Gaussian stream-function blobs,

        f(mu) = sum_b w_b C psi(x; m_b(mu), width e^{tau_b(mu)}),

    built on the grid at query time.  Its projection Phi^T M_u f CANNOT be
    precomputed, so b(mu) costs O(M n_u) PER QUERY -- which is exactly why
    STOKES-DESIGN.md makes this a separate, separately-reported arm whose
    projection is timed INSIDE the pipe and never blended into the affine
    numbers.  Same eight parameters, same kernel geometry as the affine
    family; it is not an interpolant of it, and the discrepancy between the
    two is reported as a diagnostic rather than called an approximation
    error."""
    g = cell.g
    xs, ys = g.coords_psi()
    mu = np.asarray(mu, dtype=float).ravel()
    psi = np.zeros(g.n_psi)
    for b in range(bank.N_BLOBS):
        m = mu[4 * b:4 * b + 4]
        cx = bank.M_LO + (bank.M_HI - bank.M_LO) * m[0]
        cy = bank.M_LO + (bank.M_HI - bank.M_LO) * m[1]
        tau = bank.TAU[0] + (bank.TAU[1] - bank.TAU[0]) * m[2]
        w = float(np.exp(tau))
        psi += bank.BLOB_W[b] * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2)
                                       / (2 * w * w)).ravel()
    f = cell.C @ psi
    return f / (g.h * np.linalg.norm(f) + 1e-300)


# ================================================================== timing ===

def balanced_time(subjects, reps=12, warm=3, capture=True):
    """The project's MANDATORY-MEASUREMENT-RULES harness, reimplemented here
    rather than imported so this cell shares no code with the collocated
    Burgers/Poisson modules (see `sep_common.balanced_time`, HANDOFF.md rule 3):
    every subject is warmed `warm` times, then timed `reps` times in a BALANCED
    order -- the full list swept forward on even repetitions and reversed on
    odd ones -- so no subject is systematically first or last.  ALL raw
    repetition times are returned, never only medians."""
    assert warm >= 3, f"warm={warm} < 3 (the spec's warm-up floor)"
    raw = {name: [] for name, _ in subjects}
    results = {}
    for name, fn in subjects:
        for _ in range(warm):
            results[name] = fn()
    for rep in range(reps):
        order = subjects if rep % 2 == 0 else list(reversed(subjects))
        for name, fn in order:
            t0 = time.perf_counter()
            res = fn()
            raw[name].append(time.perf_counter() - t0)
            if capture:
                results[name] = res
    return raw, results


def tstats(ts):
    a = np.asarray(ts, dtype=float)
    return dict(median=float(np.median(a)), min=float(a.min()),
                max=float(a.max()), mean=float(a.mean()), n=int(a.size))
