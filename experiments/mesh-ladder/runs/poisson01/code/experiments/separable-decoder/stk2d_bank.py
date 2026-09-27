"""PHASE 2a: the force family, the divergence-free bank, and the test space.

Design: STOKES-DESIGN.md revision 3, sections "Force family", "The bank",
"Test space"; gates S1, S2, S-MEAN, S5 and the manifold-richness requirements.
Builds strictly ON TOP of the certified phase-1 FOM in `stk2d_common.py` --
NO operator and NO solver in that module is modified here.

WHAT THIS PHASE DECIDES.  Steady Stokes is linear, so after pressure
elimination the velocity is linear in the force amplitudes.  If the parameters
entered the amplitudes affinely and varied independently, the solution manifold
would BE a linear subspace, a linear POD-K decoder would represent it exactly,
and phase 2b's nonlinear-head comparison would be vacuous.  The family below is
therefore a FIXED AFFINE DICTIONARY (so b(mu) stays precomputable) driven by a
CURVED K-parameter amplitude map, and the curvature is GATED, not assumed.

------------------------------------------------------------------ dictionary

Q = 48 fixed force shapes, each carrying a fixed DESCRIPTOR c_q in a 3-D
descriptor space (x, y, tau) with tau = log(blob width):

    32 SOLENOIDAL atoms   f_q = C psi_q,    psi_q a vertex Gaussian at (x,y)
                                            of width e^tau,
                          tau in {log 0.09, log 0.16}, (x,y) on a 4x4 grid;
    16 GRADIENT atoms     f_q = Grad_h chi_q, chi_q a mean-zero cell-centre
                                            Gaussian, same 4x4 grid, tau = the
                                            coarse level.

Both families are EXACT by construction, not to O(h^2):  range(C) = ker D and
range(Grad) is its exact M_u-orthogonal complement (phase 1, gate S-STRUCT:
||D C||_inf is exactly 0, and M_u Grad = -D^T M_p exactly).  So the Hodge split
of any dictionary force is exact, and it is MEASURED here rather than assumed.

Each atom is mass-normalised, then the gradient atoms are scaled by the frozen
mixture constant GRAD_MIX so that the two Hodge energies are comparable.  A
gradient-dominated dictionary would give large pressure and almost no velocity;
a purely solenoidal one would make the pressure diagnostic vacuous.

------------------------------------------------------------- amplitude map

mu in [0,1]^K, K = 8, drives TWO moving blobs in the descriptor space:

    theta_q(mu) = sum_{b=1,2} w_b exp( -|| (c_q - m_b(mu)) .* W ||^2
                                       / (2 s_b(mu)^2) )

    m_b(mu) = (0.15 + 0.7 mu_{4b-3},  0.15 + 0.7 mu_{4b-2},
               tau_lo + (tau_hi - tau_lo) mu_{4b-1})
    s_b(mu) = S_LO (S_HI/S_LO)^{mu_{4b}}        (log-uniform kernel bandwidth)

This is the design's EIM-style affine approximation of a physical moving-blob
family: the amplitudes are a curved function of mu while
f(mu) = sum_q theta_q(mu) f_q stays affine in the dictionary, so
b(mu) = sum_q theta_q(mu) Phi^T M_u f_q is precomputable.

DEVIATION FROM THE DESIGN, STATED PLAINLY.  STOKES-DESIGN.md writes a SINGLE
exponential, theta_q = exp(-||c_q - m(mu)||^2 / 2 s(mu)^2).  With one blob the
map mu -> theta factors through (m, s), so its image is a 3-parameter manifold
whatever K is, and K = 8 would be six directions of pure degeneracy: the
Jacobian d u / d mu would have rank 3, not 8.  Two blobs with FOUR parameters
each is the smallest superposition of the design's own kernel that is genuinely
K = 8 dimensional.  The blob weights w = (1.0, 0.7) are unequal so the family
has no exact blob-permutation symmetry.  Gate S-RICH measures the Jacobian rank
and would read 3 or 4 if this were wrong.

------------------------------------------------------------------- the bank

The bank is built in STREAMFUNCTION COORDINATES, not velocity coordinates, and
that is the whole point of gate S1.

The design's own warning: for a Gram POD g_i = X v_i / sigma_i, one has
D g_i = (D X) v_i / sigma_i, so a snapshot divergence residual eps becomes
eps / sigma_i in the tail modes.  Here that amplification is REAL and MEASURED
(the naive route is built and gated alongside the psi route purely to exhibit
it): at N = 128 the naive route's per-mode divergence climbs from 2.2e-16 in
mode 1 to 1.7e-12 in mode 32, tracking sigma_1/sigma_32 = 2.2e4.

The psi route removes it by construction.  Every snapshot lies in range(C), so
write u_i = C psi_i with psi_i = (C^T C)^{-1} C^T u_i, do the identical POD in
psi coordinates under the induced metric C^T M_u C (which gives the IDENTICAL
singular values and modes, since the two metrics agree on range C), and set
G = C Psi_pod.  Then D G telescopes cell by cell in floating point and reads
~1e-19 regardless of sigma_i.

------------------------------------------------------------ the performance

For a fixed mesh and fixed nu the bordered saddle matrix is the SAME for every
snapshot; only the right-hand side changes.  `SaddleFactor` factors it ONCE per
mesh with SuperLU and back-substitutes per sample.  It is gated bit-for-bit
against the certified `stk.solve_stokes` (gate S-SOLVE).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import stk2d_common as stk

PI = np.pi

# ------------------------------------------------------- FROZEN family ------
Q_TOTAL = 48
Q_SOL = 32
Q_GRAD = 16
K_LATENT = 8
N_BLOBS = 2
CENTRES_1D = np.linspace(0.2, 0.8, 4)          # 4x4 spatial dictionary grid
WIDTHS = np.array([0.09, 0.16])                # two dictionary scale levels
TAU = np.log(WIDTHS)
DESC_W = np.array([1.0, 1.0, 0.35])            # descriptor metric (tau axis)
BLOB_W = np.array([1.0, 0.7])                  # unequal: no permutation symmetry
M_LO, M_HI = 0.15, 0.85                        # blob centre range in x and y
S_LO, S_HI = 0.10, 0.45                        # kernel bandwidth range
GRAD_MIX = 3.0                                 # gradient-atom mixture weight


def descriptors():
    """(Q,3) descriptors and the matching kind list, in the frozen order.

    Rows 0..31 solenoidal (4x4 centres x 2 scale levels), rows 32..47 gradient
    (4x4 centres, coarse scale level).
    """
    descs, kinds = [], []
    for tau in TAU:
        for x in CENTRES_1D:
            for y in CENTRES_1D:
                descs.append((x, y, tau))
                kinds.append("sol")
    for x in CENTRES_1D:
        for y in CENTRES_1D:
            descs.append((x, y, TAU[0]))
            kinds.append("grad")
    descs = np.asarray(descs, dtype=float)
    assert descs.shape == (Q_TOTAL, 3)
    assert kinds.count("sol") == Q_SOL and kinds.count("grad") == Q_GRAD
    return descs, kinds


def dictionary(g: stk.MacGrid, ops=None):
    """The (n_u, Q) affine force dictionary.  Solenoidal atoms are C psi_q,
    gradient atoms are Grad_h chi_q; each is mass-normalised and the gradient
    atoms are then scaled by GRAD_MIX."""
    C = stk.curl_matrix(g) if ops is None else ops[3]
    Gr = stk.gradient_matrix(g) if ops is None else ops[1]
    descs, kinds = descriptors()
    xs, ys = g.coords_psi()
    xp, yp = g.coords_p()
    cols = []
    for (x, y, tau), kd in zip(descs, kinds):
        w = float(np.exp(tau))
        if kd == "sol":
            psi = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * w * w)).ravel()
            f = C @ psi
            k = 1.0
        else:
            chi = np.exp(-((xp - x) ** 2 + (yp - y) ** 2) / (2 * w * w)).ravel()
            chi = chi - chi.mean()             # mean-zero: pure gauge removed
            f = Gr @ chi
            k = GRAD_MIX
        cols.append(k * f / (g.h * np.linalg.norm(f)))
    return np.column_stack(cols), descs, kinds


def theta(mu, descs):
    """The curved amplitude map, theta: R^K -> R^Q.  `mu` may be (K,) or (S,K)."""
    mu = np.atleast_2d(np.asarray(mu, dtype=float))
    assert mu.shape[1] == K_LATENT
    out = np.zeros((mu.shape[0], descs.shape[0]))
    for b in range(N_BLOBS):
        m = mu[:, 4 * b:4 * b + 4]
        c = np.column_stack([
            M_LO + (M_HI - M_LO) * m[:, 0],
            M_LO + (M_HI - M_LO) * m[:, 1],
            TAU[0] + (TAU[1] - TAU[0]) * m[:, 2]])
        s = S_LO * (S_HI / S_LO) ** m[:, 3]
        d = (descs[None, :, :] - c[:, None, :]) * DESC_W[None, None, :]
        out += BLOB_W[b] * np.exp(-(d ** 2).sum(2) / (2 * s[:, None] ** 2))
    return out


def sample_mu(n, seed):
    """Scrambled-Sobol-free, fully deterministic stratified sample of [0,1]^K:
    an independently permuted Latin hypercube.  Reproducible from `seed`."""
    rng = np.random.default_rng(seed)
    u = (np.arange(n)[:, None] + rng.random((n, K_LATENT))) / n
    for k in range(K_LATENT):
        u[:, k] = u[rng.permutation(n), k]
    return u


def forces(F, mu, descs):
    """(n_u, S) affine forces f(mu_i) = sum_q theta_q(mu_i) f_q."""
    return F @ theta(mu, descs).T


# ---------------------------------------------------- factor-once solver -----

class SaddleFactor:
    """Factor the bordered saddle matrix ONCE per mesh, back-substitute per
    right-hand side.  The matrix is built by the certified phase-1
    `stk.stokes_saddle`; only the solve strategy differs from
    `stk.solve_stokes`, and gate S-SOLVE checks the two agree."""

    def __init__(self, g: stk.MacGrid, nu=1.0, ghost="odd", ops=None):
        self.g = g
        self.nu = float(nu)
        if ops is None:
            ops = (stk.divergence_matrix(g), stk.gradient_matrix(g),
                   stk.laplacian_matrix(g, ghost), stk.curl_matrix(g))
        self.D, self.Grad, self.L, self.C = ops
        self.K = stk.stokes_saddle(g, nu=nu, ghost=ghost,
                                   ops=(self.D, self.Grad, self.L)).tocsc()
        self.lu = spla.splu(self.K)
        self.K_fro = stk.spnorm_fro(self.K)
        self.L_fro = stk.spnorm_fro(self.L)
        self.D_fro = stk.spnorm_fro(self.D)
        self.Grad_fro = stk.spnorm_fro(self.Grad)
        self.n_solves = 0

    def solve(self, f):
        """Returns (u, p, info) with the SAME blockwise backward-error fields
        as `stk.solve_stokes`, so phase 1's S-BACKERR thresholds apply."""
        g = self.g
        f = np.asarray(f, dtype=float).ravel()
        rhs = np.concatenate([f, np.zeros(g.n_p), [0.0]])
        sol = self.lu.solve(rhs)
        self.n_solves += 1
        u = sol[:g.n_u]
        p = sol[g.n_u:g.n_u + g.n_p]
        lam = float(sol[-1])
        res = self.K @ sol - rhs
        nu_u, nrm_p, nrm_f = (np.linalg.norm(u), np.linalg.norm(p),
                              np.linalg.norm(f))
        r_gauge = float(res[-1])
        nrm_sol = np.linalg.norm(sol)
        r_cont = res[g.n_u:g.n_u + g.n_p]
        # CONTINUITY NORMALISATION -- deliberately NOT phase 1's.
        # Phase 1 used ||r_cont|| / (||D||_F ||u|| + |lam| sqrt(n_p)).  That
        # denominator COLLAPSES on this cell's 16 GRADIENT dictionary atoms,
        # whose exact velocity is zero (Grad chi is balanced entirely by the
        # pressure, so ||u|| ~ 1e-17): the metric then reads 2.5e-2 on a solve
        # whose absolute continuity residual is pure roundoff.  This is the
        # standard blockwise normwise backward error for the row block
        # [D | 0 | 1] with a zero right-hand side, whose denominator cannot
        # collapse.  Phase 1's form is RECORDED as cont_resid_phase1 and is
        # NOT gated here, with this reason.
        info = dict(
            lam=lam,
            backward_err=float(np.linalg.norm(res)
                               / (self.K_fro * nrm_sol
                                  + np.linalg.norm(rhs) + 1e-300)),
            mom_resid=float(np.linalg.norm(res[:g.n_u])
                            / (self.nu * self.L_fro * nu_u
                               + self.Grad_fro * nrm_p + nrm_f + 1e-300)),
            cont_resid=float(np.linalg.norm(r_cont)
                             / (np.sqrt(self.D_fro ** 2 + g.n_p) * nrm_sol
                                + 1e-300)),
            cont_resid_phase1=float(np.linalg.norm(r_cont)
                                    / (self.D_fro * nu_u
                                       + abs(lam) * np.sqrt(g.n_p) + 1e-300)),
            u_norm=float(nu_u), p_norm=float(nrm_p), f_norm=float(nrm_f),
            gauge_raw=r_gauge,
            gauge_resid=float(abs(r_gauge)
                              / (np.sqrt(g.n_p) * nrm_p + 1e-300)))
        return u, p, info

    def solve_many(self, F):
        """(n_u, S) forces -> (n_u, S) velocities, (n_p, S) pressures, infos."""
        F = np.asarray(F, dtype=float)
        U = np.empty((self.g.n_u, F.shape[1]))
        P = np.empty((self.g.n_p, F.shape[1]))
        infos = []
        for i in range(F.shape[1]):
            u, p, info = self.solve(F[:, i])
            U[:, i] = u
            P[:, i] = p
            infos.append(info)
        return U, P, infos


# ------------------------------------------------------ Hodge machinery ------

class Hodge:
    """Exact discrete Hodge split on this MAC layout.

        R^{n_u} = range(C)  (+)_{M_u}  range(Grad),     dim (N-1)^2 + (N^2-1)

    Both facts are phase-1 gate results, not assumptions: ||D C||_inf is
    EXACTLY 0 and M_u Grad = -D^T M_p exactly, so range(Grad) is the exact
    M_u-orthogonal complement of ker D = range(C).  The solenoidal part is
    obtained by the vertex Poisson solve (C^T C) psi = C^T x.
    """

    def __init__(self, g: stk.MacGrid, C=None):
        self.g = g
        self.C = stk.curl_matrix(g) if C is None else C
        self.lu = spla.splu((self.C.T @ self.C).tocsc())

    def psi_of(self, X):
        """Streamfunction coordinates of X (columnwise)."""
        return self.lu.solve(self.C.T @ np.asarray(X))

    def split(self, X):
        """(solenoidal part, gradient part) of X, columnwise."""
        X = np.asarray(X)
        Xs = self.C @ self.psi_of(X)
        return Xs, X - Xs

    def fractions(self, X):
        """(solenoidal energy fraction, gradient energy fraction) per column."""
        X = np.atleast_2d(np.asarray(X).T).T
        Xs, Xg = self.split(X)
        n2 = (X ** 2).sum(0) + 1e-300
        return (Xs ** 2).sum(0) / n2, (Xg ** 2).sum(0) / n2


# --------------------------------------------------------------- the bank ----

def _gram_pod(Xc, h, R):
    """Symmetric-Gram POD in the mass metric.  Returns (V, sigma) with
    sigma the mass-weighted singular values (sigma_i = h * s_i)."""
    Gram = (Xc.T @ Xc) * (h ** 2)
    w, V = np.linalg.eigh(Gram)
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    sig = np.sqrt(np.maximum(w, 0.0))
    return V[:, :R], sig


def build_bank(g, Xu, R, hodge, centred=True):
    """Build the phase-2a bank.  Returns a dict with BOTH routes.

    psi route  (THE BANK): POD carried out in streamfunction coordinates and
        lifted, G = C Psi.  Divergence-free to telescoping roundoff, with NO
        1/sigma_i amplification.
    naive route (DIAGNOSTIC ONLY, built to exhibit the failure the design
        warns about): the same POD applied directly to velocity snapshots,
        g_i = X v_i / sigma_i.
    """
    h = g.h
    ubar = Xu.mean(axis=1) if centred else np.zeros(g.n_u)
    Xc = Xu - ubar[:, None]
    Psi_c = hodge.psi_of(Xc)                 # streamfunction snapshots
    psibar = hodge.psi_of(ubar[:, None])[:, 0]
    V, sig = _gram_pod(Xc, h, R)
    # sig_i = h * s_i with s_i the plain singular values of Xc, so the mode
    # with UNIT MASS NORM (h ||.||_2 = 1) is Xc v_i / sig_i.
    scale = sig[:R]
    Gp = hodge.C @ (Psi_c @ V / scale[None, :])
    Gn = Xc @ V / scale[None, :]
    return dict(ubar=ubar, ubar_psi=hodge.C @ psibar, psibar=psibar,
                sigma=sig, V=V, G_psi=Gp, G_naive=Gn, Psi_modes=Psi_c @ V / scale[None, :])


def reorth_psi(g, hodge, Psi_modes, n_passes=2):
    """Reorthogonalisation carried out in STREAMFUNCTION coordinates under the
    INDUCED metric C^T M_u C, then lifted, so the result is M_u-orthonormal in
    velocity space AND still exactly in range(C).

    A plain `np.linalg.qr` on Psi_modes would orthonormalise in the psi 2-norm,
    which is the WRONG inner product: the lifted basis then reads
    ||G^T M_u G - I||_max ~ 0.98.  This is Cholesky-QR in the induced metric:
    S = (C Psi)^T M_u (C Psi) = R^T R, Psi <- Psi R^{-1}.  Two passes, because
    one Cholesky-QR pass loses accuracy like cond(Psi)^2 and the Gram POD has
    already squared the spectrum once.

    Gate S1 re-checks the divergence AFTER this step, as the design requires.
    """
    Psi = np.asarray(Psi_modes)
    for _ in range(n_passes):
        S = (hodge.C @ Psi).T @ (hodge.C @ Psi) * (g.h ** 2)
        R = np.linalg.cholesky(S).T
        Psi = np.linalg.solve(R.T, Psi.T).T
    return hodge.C @ Psi, Psi


# ------------------------------------------------------------- S5 helpers ----

def eig_residuals(g, kmax=8, ghost="odd", C=None, L=None):
    """||L phi + lambda phi|| / ||L phi|| for phi = C psi_{k,l}, k,l <= kmax.

    Under no-slip MAC the curl-sine modes are NOT eigenvectors of the vector
    Laplacian: the tangential cosine components want EVEN (free-slip) ghosts,
    while no-slip uses ODD ones, leaving a -2/h^2 defect on every
    boundary-adjacent tangential row.  A ROUNDOFF value therefore FAILS this
    gate -- it would mean even ghosts, a wrong L, or omitted boundary terms.

    kmax IS CLAMPED TO N-1.  STOKES-DESIGN.md writes "k, l <= 8" flat, but
    sin(k pi x) sampled on the interior vertices x_i = i/N is IDENTICALLY ZERO
    for k = N: at N = 8 the k = 8 modes are numerical noise of size 1e-16 and
    the residual becomes a ratio of two roundoff quantities, which read
    0.0357 (odd) and 0.134 (even) -- i.e. the negative control silently stops
    being roundoff and the gate stops meaning anything.  The per-mode norms
    are returned so the caller can assert the modes are non-degenerate.
    """
    C = stk.curl_matrix(g) if C is None else C
    L = stk.laplacian_matrix(g, ghost) if L is None else L
    kmax = int(min(kmax, g.N - 1))
    xs, ys = g.coords_psi()
    rels, lams, modes, norms = [], [], [], []
    for k in range(1, kmax + 1):
        for l in range(1, kmax + 1):
            psi = (np.sin(k * PI * xs) * np.sin(l * PI * ys)).ravel()
            phi = C @ psi
            lam = 4.0 / g.h ** 2 * (np.sin(k * PI / (2 * g.N)) ** 2
                                    + np.sin(l * PI / (2 * g.N)) ** 2)
            Lp = L @ phi
            rels.append(float(np.linalg.norm(Lp + lam * phi)
                              / (np.linalg.norm(Lp) + 1e-300)))
            lams.append(float(lam))
            modes.append((k, l))
            norms.append(float(g.h * np.linalg.norm(phi)))
    return (np.asarray(rels), np.asarray(lams), modes, np.asarray(norms))


def audit_clamped_basis(g):
    """The auditor's own six clamped stream functions sin^2(a pi x) sin^2(b pi y),
    used ONLY to reproduce STOKES-AUDIT-mac_s5_scaling.py's ||A + Lambda B||
    anchor (0.371 / 0.372 / 0.373 at N = 64/128/256).  Not the bank."""
    C = stk.curl_matrix(g)
    xs, ys = g.coords_psi()
    cols = [C @ (np.sin(a * PI * xs) ** 2 * np.sin(b * PI * ys) ** 2).ravel()
            for a, b in [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]]
    return np.column_stack(cols)


def a_ratio(g, Phi, lams_phi, Gbasis, L=None):
    """Secondary S5 diagnostic ||A + Lambda B|| / ||A|| with
    A = Phi^T M_u L G and B = Phi^T M_u G.  NOTE THE PLUS SIGN: this repo's
    convention is L Phi = -Phi Lambda, so exact eigenvectors would give 0 and a
    diagonal A would suffice.  A value of order 1e-1 means DENSE A is required.
    """
    L = stk.laplacian_matrix(g, "odd") if L is None else L
    Gq, _ = np.linalg.qr(np.asarray(Gbasis))
    A = Phi.T @ ((L @ Gq) * g.h ** 2)
    B = Phi.T @ (Gq * g.h ** 2)
    num = float(np.linalg.norm(A + np.asarray(lams_phi)[:, None] * B))
    den = float(np.linalg.norm(A) + 1e-300)
    return num / den, num, den


# ------------------------------------------------------------- diagnostics ---

def div_norm(D_fro, D, x):
    """The phase-1 convention: ||D x||_2 / (||D||_F ||x||_2)."""
    x = np.asarray(x)
    return float(np.linalg.norm(D @ x) / (D_fro * np.linalg.norm(x) + 1e-300))


def div_norm_cols(D_fro, D, X):
    X = np.asarray(X)
    DX = D @ X
    return (np.linalg.norm(DX, axis=0)
            / (D_fro * np.linalg.norm(X, axis=0) + 1e-300))


def numerical_rank(sig, rtol=1e-12):
    sig = np.asarray(sig, dtype=float)
    return int((sig > sig[0] * rtol).sum()) if sig.size and sig[0] > 0 else 0
