"""Staggered MAC steady-Stokes 2D full-order model -- operators, manufactured
solution, and the direct saddle-point solve.

Phase 1 of the 2026-08-30 Stokes cell (`exp/2026-08-30-stokes-vector`).
Design: STOKES-DESIGN.md revision 3, section "Discretization" + "Frozen
contract".  Adversarial audits: STOKES-DESIGN-AUDIT-r1/-r2-codex-gpt56sol.md.
The auditor's own numpy reference is archived as STOKES-AUDIT-mac_check.py /
STOKES-AUDIT-mac_s5_scaling.py; the operators here reproduce it entry-for-entry
(gate REF in stk2d_fom_gates.py).

CONVENTIONS -- these are NOT the conventions of any other script in this
directory and must never share a reshape, mask, or spacing with them.  The
collocated Burgers/Poisson code here uses (N-2)^2 interior points and
h = 1/(N-1); this module uses N CELLS and h = 1/N.

    p    cell centres      (N, N)        x=(i+1/2)h, y=(j+1/2)h   i,j=0..N-1
    u_x  interior vertical faces (N-1, N)  x=i h,     y=(j+1/2)h  i=1..N-1, j=0..N-1
    u_y  interior horizontal faces (N, N-1) x=(i+1/2)h, y=j h     i=0..N-1, j=1..N-1
    psi  interior vertices  (N-1, N-1)    x=i h,     y=j h        i,j=1..N-1

    n_u = 2 N (N-1)   (active velocity DOF; boundary-NORMAL faces are
                       identically zero and are ELIMINATED, not stored)
    n_p = N^2         (one gauge null mode, fixed by the mean-zero constraint)
    n_psi = (N-1)^2

The velocity vector is packed as concatenate([u_x.ravel(), u_y.ravel()])
in C order.  Mass matrices are M_u = M_p = h^2 I on this uniform layout.

BOUNDARY CONDITIONS.  Homogeneous no-slip.  Normal components sit exactly on
the wall and are eliminated.  TANGENTIAL components sit half a cell inside the
wall and are closed with ODD ghosts,

    u_x[i, -1] = -u_x[i, 0]        (wall value 0 at y = 0)

which turns the wall-adjacent y-diagonal into -3/h^2 instead of -2/h^2, i.e. a
total diagonal of -5/h^2 rather than the interior -4/h^2.

    ghost="odd"   -> no-slip     (THE contract)
    ghost="even"  -> FREE-SLIP   (diagonal -3/h^2; provided ONLY as a
                                  deliberate negative control -- both audits
                                  flag accidental free-slip as the failure mode
                                  that looks healthy and silently invalidates
                                  everything downstream)

OPERATORS.  D (divergence, n_p x n_u), Grad (n_u x n_p), L (componentwise
vector Laplacian, n_u x n_u), C (vertex-curl, n_u x n_psi).  D and Grad are
assembled INDEPENDENTLY from their own stencils -- neither is built as the
transpose of the other -- so that the weighted adjointness gate
M_u Grad = -D^T M_p is a real measurement and not a tautology.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

PI = np.pi


# ----------------------------------------------------------------- grid ------

@dataclass(frozen=True)
class MacGrid:
    """N cells per side on (0,1)^2, h = 1/N."""
    N: int

    @property
    def h(self) -> float:
        return 1.0 / self.N

    # --- shapes / counts ---
    @property
    def shape_u(self):
        return (self.N - 1, self.N)

    @property
    def shape_v(self):
        return (self.N, self.N - 1)

    @property
    def shape_p(self):
        return (self.N, self.N)

    @property
    def shape_psi(self):
        return (self.N - 1, self.N - 1)

    @property
    def n_ux(self):
        return (self.N - 1) * self.N

    @property
    def n_uy(self):
        return self.N * (self.N - 1)

    @property
    def n_u(self):
        return 2 * self.N * (self.N - 1)

    @property
    def n_p(self):
        return self.N * self.N

    @property
    def n_psi(self):
        return (self.N - 1) * (self.N - 1)

    # --- lattice coordinates (2-D arrays matching the field shapes) ---
    def coords_u(self):
        h = self.h
        x = (np.arange(1, self.N) * h)[:, None]
        y = ((np.arange(self.N) + 0.5) * h)[None, :]
        return np.broadcast_to(x, self.shape_u), np.broadcast_to(y, self.shape_u)

    def coords_v(self):
        h = self.h
        x = ((np.arange(self.N) + 0.5) * h)[:, None]
        y = (np.arange(1, self.N) * h)[None, :]
        return np.broadcast_to(x, self.shape_v), np.broadcast_to(y, self.shape_v)

    def coords_p(self):
        h = self.h
        x = ((np.arange(self.N) + 0.5) * h)[:, None]
        y = ((np.arange(self.N) + 0.5) * h)[None, :]
        return np.broadcast_to(x, self.shape_p), np.broadcast_to(y, self.shape_p)

    def coords_psi(self):
        h = self.h
        x = (np.arange(1, self.N) * h)[:, None]
        y = (np.arange(1, self.N) * h)[None, :]
        return (np.broadcast_to(x, self.shape_psi),
                np.broadcast_to(y, self.shape_psi))

    # --- packing ---
    def pack(self, U, V):
        return np.concatenate([np.asarray(U).ravel(), np.asarray(V).ravel()])

    def unpack(self, u):
        u = np.asarray(u)
        return (u[:self.n_ux].reshape(self.shape_u),
                u[self.n_ux:].reshape(self.shape_v))

    # --- index helper arrays used by the sparse assembly ---
    def _iu(self):
        i, j = np.meshgrid(np.arange(1, self.N), np.arange(self.N),
                           indexing="ij")
        return i.ravel(), j.ravel(), np.arange(self.n_ux)

    def _iv(self):
        i, j = np.meshgrid(np.arange(self.N), np.arange(1, self.N),
                           indexing="ij")
        return i.ravel(), j.ravel(), self.n_ux + np.arange(self.n_uy)


def _coo(vals, rows, cols, shape):
    return sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                 np.concatenate(cols))),
                         shape=shape).tocsr()


# ------------------------------------------------------------ operators ------

def divergence_matrix(g: MacGrid) -> sp.csr_matrix:
    """D : cell divergence, (n_p, n_u).  (D u)_{ij} = (u[i+1,j]-u[i,j])/h
    + (v[i,j+1]-v[i,j])/h with boundary-normal faces identically zero."""
    N, h = g.N, g.h
    iu, ju, qu = g._iu()
    iv, jv, qv = g._iv()
    pid = lambda i, j: i * N + j                                  # noqa: E731
    r = [pid(iu - 1, ju), pid(iu, ju), pid(iv, jv - 1), pid(iv, jv)]
    c = [qu, qu, qv, qv]
    v = [np.full(g.n_ux, 1.0 / h), np.full(g.n_ux, -1.0 / h),
         np.full(g.n_uy, 1.0 / h), np.full(g.n_uy, -1.0 / h)]
    return _coo(v, r, c, (g.n_p, g.n_u))


def gradient_matrix(g: MacGrid) -> sp.csr_matrix:
    """Grad : face pressure gradient, (n_u, n_p).  Assembled from its OWN
    stencil, independently of D (that is what makes S-ADJ a measurement)."""
    N, h = g.N, g.h
    iu, ju, qu = g._iu()
    iv, jv, qv = g._iv()
    pid = lambda i, j: i * N + j                                  # noqa: E731
    r = [qu, qu, qv, qv]
    c = [pid(iu, ju), pid(iu - 1, ju), pid(iv, jv), pid(iv, jv - 1)]
    v = [np.full(g.n_ux, 1.0 / h), np.full(g.n_ux, -1.0 / h),
         np.full(g.n_uy, 1.0 / h), np.full(g.n_uy, -1.0 / h)]
    return _coo(v, r, c, (g.n_u, g.n_p))


def laplacian_matrix(g: MacGrid, ghost: str = "odd") -> sp.csr_matrix:
    """L : componentwise vector Laplacian, (n_u, n_u).

    ghost="odd"  -> no-slip tangential wall closure  (wall diagonal -5/h^2)
    ghost="even" -> FREE-SLIP negative control       (wall diagonal -3/h^2)
    """
    if ghost not in ("odd", "even"):
        raise ValueError(f"ghost must be 'odd' or 'even', got {ghost!r}")
    s = -1.0 if ghost == "odd" else +1.0
    N, h = g.N, g.h
    iu, ju, qu = g._iu()
    iv, jv, qv = g._iv()
    r, c, v = [], [], []

    # --- u_x ---------------------------------------------------------------
    # x: u sits ON the grid lines i=0..N with prescribed value 0 at i=0,N.
    diag = np.full(g.n_ux, -2.0)
    m = iu - 1 >= 1
    r.append(qu[m]); c.append(qu[m] - N); v.append(np.ones(int(m.sum())))
    m = iu + 1 <= N - 1
    r.append(qu[m]); c.append(qu[m] + N); v.append(np.ones(int(m.sum())))
    # y: tangential, cell-centred, ghost u[i,-1] = s*u[i,0].
    diag = diag + np.where((ju == 0) | (ju == N - 1), -2.0 + s, -2.0)
    m = ju - 1 >= 0
    r.append(qu[m]); c.append(qu[m] - 1); v.append(np.ones(int(m.sum())))
    m = ju + 1 <= N - 1
    r.append(qu[m]); c.append(qu[m] + 1); v.append(np.ones(int(m.sum())))
    r.append(qu); c.append(qu); v.append(diag)

    # --- u_y ---------------------------------------------------------------
    diagv = np.full(g.n_uy, -2.0)
    diagv = diagv + np.where((iv == 0) | (iv == N - 1), -2.0 + s, -2.0)
    m = iv - 1 >= 0
    r.append(qv[m]); c.append(qv[m] - (N - 1)); v.append(np.ones(int(m.sum())))
    m = iv + 1 <= N - 1
    r.append(qv[m]); c.append(qv[m] + (N - 1)); v.append(np.ones(int(m.sum())))
    m = jv - 1 >= 1
    r.append(qv[m]); c.append(qv[m] - 1); v.append(np.ones(int(m.sum())))
    m = jv + 1 <= N - 1
    r.append(qv[m]); c.append(qv[m] + 1); v.append(np.ones(int(m.sum())))
    r.append(qv); c.append(qv); v.append(diagv)

    v = [x / h ** 2 for x in v]
    return _coo(v, r, c, (g.n_u, g.n_u))


def curl_matrix(g: MacGrid) -> sp.csr_matrix:
    """C : vertex streamfunction -> velocity, (n_u, n_psi).

        u[i,j+1/2] = (psi[i,j+1] - psi[i,j]) / h
        v[i+1/2,j] = -(psi[i+1,j] - psi[i,j]) / h

    with psi = 0 on the boundary vertices.  The cell divergence of C psi
    telescopes exactly, boundary cells included.
    """
    N, h = g.N, g.h
    iu, ju, qu = g._iu()
    iv, jv, qv = g._iv()
    sid = lambda i, j: (i - 1) * (N - 1) + (j - 1)                # noqa: E731
    r, c, v = [], [], []
    m = (ju + 1 >= 1) & (ju + 1 <= N - 1)
    r.append(qu[m]); c.append(sid(iu[m], ju[m] + 1))
    v.append(np.full(int(m.sum()), 1.0 / h))
    m = (ju >= 1) & (ju <= N - 1)
    r.append(qu[m]); c.append(sid(iu[m], ju[m]))
    v.append(np.full(int(m.sum()), -1.0 / h))
    m = (iv + 1 >= 1) & (iv + 1 <= N - 1)
    r.append(qv[m]); c.append(sid(iv[m] + 1, jv[m]))
    v.append(np.full(int(m.sum()), -1.0 / h))
    m = (iv >= 1) & (iv <= N - 1)
    r.append(qv[m]); c.append(sid(iv[m], jv[m]))
    v.append(np.full(int(m.sum()), 1.0 / h))
    return _coo(v, r, c, (g.n_u, g.n_psi))


def mass_u(g: MacGrid) -> sp.csr_matrix:
    return sp.identity(g.n_u, format="csr") * (g.h ** 2)


def mass_p(g: MacGrid) -> sp.csr_matrix:
    return sp.identity(g.n_p, format="csr") * (g.h ** 2)


# ------------------------------- matrix-free (INDEPENDENT implementation) ----
# Written deliberately in the pad-and-slice style of the auditor's
# STOKES-AUDIT-mac_s5_scaling.py, not by reusing the sparse assembly, so that
# agreement between the two is a genuine cross-check of the stencils.

def apply_laplacian(g: MacGrid, U, V, ghost: str = "odd"):
    s = -1.0 if ghost == "odd" else +1.0
    N, h = g.N, g.h
    Ug = np.zeros((N + 1, N + 2))
    Ug[1:N, 1:N + 1] = U                      # rows 0 and N are the x-walls (0)
    Ug[1:N, 0] = s * U[:, 0]                  # y ghosts (tangential)
    Ug[1:N, N + 1] = s * U[:, -1]
    LU = (Ug[2:N + 1, 1:N + 1] + Ug[0:N - 1, 1:N + 1]
          + Ug[1:N, 2:N + 2] + Ug[1:N, 0:N] - 4.0 * U) / h ** 2

    Vg = np.zeros((N + 2, N + 1))
    Vg[1:N + 1, 1:N] = V                      # cols 0 and N are the y-walls (0)
    Vg[0, 1:N] = s * V[0, :]                  # x ghosts (tangential)
    Vg[N + 1, 1:N] = s * V[-1, :]
    LV = (Vg[2:N + 2, 1:N] + Vg[0:N, 1:N]
          + Vg[1:N + 1, 2:N + 1] + Vg[1:N + 1, 0:N - 1] - 4.0 * V) / h ** 2
    return LU, LV


def apply_divergence(g: MacGrid, U, V):
    N, h = g.N, g.h
    Uf = np.zeros((N + 1, N))
    Uf[1:N, :] = U
    Vf = np.zeros((N, N + 1))
    Vf[:, 1:N] = V
    return (Uf[1:, :] - Uf[:-1, :]) / h + (Vf[:, 1:] - Vf[:, :-1]) / h


def apply_gradient(g: MacGrid, P):
    N, h = g.N, g.h
    return (P[1:, :] - P[:-1, :]) / h, (P[:, 1:] - P[:, :-1]) / h


def apply_curl(g: MacGrid, PSI):
    """PSI is the (N-1,N-1) interior block; boundary vertices are zero."""
    N, h = g.N, g.h
    F = np.zeros((N + 1, N + 1))
    F[1:N, 1:N] = PSI
    U = (F[1:N, 1:] - F[1:N, :-1]) / h
    V = -(F[1:, 1:N] - F[:-1, 1:N]) / h
    return U, V


# ------------------------------------------------ manufactured solution ------

def manufactured(g: MacGrid, nu: float = 1.0):
    """The frozen manufactured solution (STOKES-DESIGN.md "Frozen contract").

        psi = sin^2(pi x) sin^2(pi y)
        u   = ( pi sin^2(pi x) sin(2 pi y),  -pi sin(2 pi x) sin^2(pi y) )
        p   = sin(2 pi x) + cos(2 pi y)                      (mean zero)
        f   = -nu Lap(u) + grad(p)                           (analytic)

    u is divergence-free AND vanishes on all four walls (psi and its normal
    derivative both vanish there), i.e. genuine NO-SLIP.  f is evaluated
    analytically on the two face lattices; nothing is differenced numerically.
    """
    xu, yu = g.coords_u()
    xv, yv = g.coords_v()
    xp, yp = g.coords_p()

    U = PI * np.sin(PI * xu) ** 2 * np.sin(2 * PI * yu)
    V = -PI * np.sin(2 * PI * xv) * np.sin(PI * yv) ** 2
    P = np.sin(2 * PI * xp) + np.cos(2 * PI * yp)

    # Lap u = pi [ 2 pi^2 cos(2 pi x) sin(2 pi y) - 4 pi^2 sin^2(pi x) sin(2 pi y) ]
    LapU = (2 * PI ** 3 * np.cos(2 * PI * xu) * np.sin(2 * PI * yu)
            - 4 * PI ** 3 * np.sin(PI * xu) ** 2 * np.sin(2 * PI * yu))
    LapV = (4 * PI ** 3 * np.sin(2 * PI * xv) * np.sin(PI * yv) ** 2
            - 2 * PI ** 3 * np.sin(2 * PI * xv) * np.cos(2 * PI * yv))
    Px = 2 * PI * np.cos(2 * PI * xu) + 0.0 * yu
    Py = -2 * PI * np.sin(2 * PI * yv) + 0.0 * xv

    FU = -nu * LapU + Px
    FV = -nu * LapV + Py
    return dict(U=U, V=V, P=P, FU=FU, FV=FV, LapU=LapU, LapV=LapV,
                u=g.pack(U, V), p=P.ravel(), f=g.pack(FU, FV))


# ---------------------------------------------------------------- solve ------

def stokes_saddle(g: MacGrid, nu: float = 1.0, ghost: str = "odd",
                  ops=None):
    """The bordered saddle-point matrix, size n_u + n_p + 1.

        [ -nu L   Grad   0 ] [u]   [f]
        [   D      0     1 ] [p] = [0]
        [   0     1^T    0 ] [lam]  [0]

    Row 3 is the MEAN-ZERO PRESSURE GAUGE.  The unbordered saddle matrix has a
    one-dimensional null space (0, const) and left null space (0, const); the
    border removes both.  Since 1^T D = 0 the multiplier lam is exactly zero at
    the solution, which is reported as a consistency witness.
    """
    D, Grad, L = ops if ops is not None else (
        divergence_matrix(g), gradient_matrix(g), laplacian_matrix(g, ghost))
    one = sp.csr_matrix(np.ones((g.n_p, 1)))
    return sp.bmat([[-nu * L, Grad, None],
                    [D, None, one],
                    [None, one.T, None]], format="csc")


def solve_stokes(g: MacGrid, f, nu: float = 1.0, ghost: str = "odd",
                 ops=None):
    """Direct sparse LU solve of the bordered saddle system.  Returns
    (u, p, info) with p in the mean-zero gauge.

    `info` carries the GLOBAL normalised backward error and, separately, the
    THREE BLOCKWISE residuals.  The blocks are reported separately because the
    global metric cannot see the bordered rows: ||K||_F is dominated by the
    O(h^-2) momentum block, so a violated continuity or mean-zero-gauge row is
    invisible in it (STOKES-PHASE1C-VERIFY-codex.md: a constant 1e-8 pressure
    offset at N=128 leaves a gauge-row residual of 1.6384e-4 while the global
    metric reads 4.46e-14 and passes).
    """
    if ops is None:
        ops = (divergence_matrix(g), gradient_matrix(g),
               laplacian_matrix(g, ghost))
    D, Grad, L = ops
    K = stokes_saddle(g, nu=nu, ghost=ghost, ops=ops)
    rhs = np.concatenate([np.asarray(f).ravel(), np.zeros(g.n_p), [0.0]])
    sol = spla.spsolve(K, rhs)
    u = sol[:g.n_u]
    p = sol[g.n_u:g.n_u + g.n_p]
    lam = float(sol[-1])
    res = K @ sol - rhs
    # Normalised BACKWARD error of the linear solve.  Unlike ||res||/||rhs||
    # (which carries the h^-2 growth of ||K||), this is the standard
    # scale-free quantity that a backward-stable factorisation bounds by
    # O(nnz^(1/2) * u_mach) independently of the mesh, and it is completely
    # independent of the manufactured solution.
    kf = float(np.sqrt((sp.csr_matrix(K).data ** 2).sum()))
    fro = lambda A: float(np.sqrt((sp.csr_matrix(A).data ** 2).sum()))  # noqa
    nu_u, nrm_p, nrm_f = (np.linalg.norm(u), np.linalg.norm(p),
                          np.linalg.norm(np.asarray(f).ravel()))
    r_mom = res[:g.n_u]
    r_cont = res[g.n_u:g.n_u + g.n_p]
    r_gauge = float(res[-1])                       # == 1^T p, the gauge row
    info = dict(lam=lam,
                lin_resid_rel=float(np.linalg.norm(res)
                                    / (np.linalg.norm(rhs) + 1e-300)),
                backward_err=float(np.linalg.norm(res)
                                   / (kf * np.linalg.norm(sol)
                                      + np.linalg.norm(rhs) + 1e-300)),
                K_fro=kf,
                # --- blockwise ---
                mom_resid=float(np.linalg.norm(r_mom)
                                / (nu * fro(L) * nu_u + fro(Grad) * nrm_p
                                   + nrm_f + 1e-300)),
                cont_resid=float(np.linalg.norm(r_cont)
                                 / (fro(D) * nu_u
                                    + abs(lam) * np.sqrt(g.n_p) + 1e-300)),
                gauge_raw=r_gauge,
                gauge_resid=float(abs(r_gauge)
                                  / (np.sqrt(g.n_p) * nrm_p + 1e-300)),
                p_mean=float(p.mean()),
                saddle_dim=int(K.shape[0]), saddle_nnz=int(K.nnz))
    return u, p, info


# ----------------------------------------------------------- test space ------

def test_modes(g: MacGrid, M: int, normalize: bool = True):
    """Phi = C psi_{k,l}, psi_{k,l} = sin(k pi x) sin(l pi y) on interior
    vertices, ordered by lambda_{k,l} = 4/h^2 (sin^2(k pi/2N) + sin^2(l pi/2N))
    ascending, mass-normalized in the kinetic inner product ||.||_{M_u}.

    Returns (Phi, lambdas, modes) with Phi of shape (n_u, M)."""
    N, h = g.N, g.h
    ks = np.arange(1, N)
    lam1 = 4.0 / h ** 2 * np.sin(ks * PI / (2 * N)) ** 2
    K_, L_ = np.meshgrid(ks, ks, indexing="ij")
    lam = lam1[:, None] + lam1[None, :]
    order = np.argsort(lam.ravel(), kind="stable")[:M]
    kk = K_.ravel()[order]
    ll = L_.ravel()[order]
    lams = lam.ravel()[order]

    xs, ys = g.coords_psi()
    C = curl_matrix(g)
    cols = []
    for k, l in zip(kk, ll):
        psi = (np.sin(k * PI * xs) * np.sin(l * PI * ys)).ravel()
        cols.append(C @ psi)
    Phi = np.column_stack(cols)
    if normalize:
        nrm = h * np.linalg.norm(Phi, axis=0)          # ||.||_{M_u} = h ||.||_2
        Phi = Phi / nrm[None, :]
    return Phi, lams, list(zip(kk.tolist(), ll.tolist()))


# ---------------------------------------------------------------- norms ------

def mass_norm(g: MacGrid, x):
    """||x||_{M} = sqrt(x^T (h^2 I) x) = h ||x||_2  (uniform MAC layout)."""
    return float(g.h * np.linalg.norm(np.asarray(x).ravel()))


def mass_rel(g: MacGrid, a, b):
    """Mass-weighted relative error ||a-b||_M / ||b||_M."""
    return float(mass_norm(g, np.asarray(a).ravel() - np.asarray(b).ravel())
                 / (mass_norm(g, b) + 1e-300))


def spnorm_fro(A):
    A = sp.csr_matrix(A)
    return float(np.sqrt((A.data ** 2).sum())) if A.nnz else 0.0


def spnorm_inf(A):
    A = sp.csr_matrix(A)
    return float(abs(A).sum(axis=1).max()) if A.nnz else 0.0


def spnorm_max(A):
    A = sp.csr_matrix(A)
    return float(np.abs(A.data).max()) if A.nnz else 0.0


# --------------------------------------- closed-form DISCRETE solution -------

def exact_discrete(g: MacGrid):
    r"""The EXACT solution of the discrete saddle system for the frozen
    manufactured data (odd ghosts only).  Not in STOKES-DESIGN.md; derived and
    verified in phase 1.

    Write $t = \pi h$.  On this MAC layout the sampled manufactured fields
    satisfy three exact discrete identities:

      1. $D\,u_{ex} = 0$ EXACTLY (not merely to $O(h^2)$).  The two cell
         differences are $\pm\pi\sin(t)\sin(2\pi x_c)\sin(2\pi y_c)$ and cancel.
      2. $L_h\,u_{ex} = \gamma\,(\Delta u)|_{\text{lattice}}$ with
         $\gamma = \sin^2(t)/t^2$.  Both components reduce to the same factor:
         $\sin(2\pi y)$ on cell centres is an exact ODD-ghost eigenvector with
         eigenvalue $-4\sin^2(t)/h^2$, and $\sin^2(\pi x)$ on grid lines (whose
         true endpoint values are $0$, matching the eliminated normal faces)
         differences to $\mu\cos(2\pi x)/2$.
      3. $\mathrm{Grad}_h\,p_{ex} = \delta\,(\nabla p)|_{\text{lattice}}$ with
         $\delta = \sin(t)/t$, and $p_{ex}$ already has exactly zero cell mean.

    Since $f = -\nu\Delta u + \nabla p$ and $\Delta u$, $\nabla p$ are linearly
    independent on the lattice, $u_h = \gamma^{-1}u_{ex}$, $p_h=\delta^{-1}p_{ex}$
    solves the discrete system, for EVERY $\nu$.  Hence

      $\|u_h-u_{ex}\|/\|u_{ex}\| = (t/\sin t)^2 - 1$,
      $\|p_h-p_{ex}\|/\|p_{ex}\| = (t/\sin t) - 1$,

    which are exactly the audit's anchors 5.303e-2 / 2.617e-2 (N=8),
    1.295e-2 / 6.455e-3 (N=16), 3.219e-3 / 1.608e-3 (N=32) and
    8.036e-4 / 4.017e-4 (N=64).  The FOM can therefore be certified to MACHINE
    PRECISION against an analytic discrete solution, not merely to two digits
    of an observed order.  It also means the discretization error of this
    particular manufactured pair is a PURE AMPLITUDE error, exactly parallel to
    the exact solution -- see STOKES-NOTES.md.

    Returns (u_h, p_h, a, b) with a = 1/gamma, b = 1/delta.
    """
    t = PI * g.h
    a = (t / np.sin(t)) ** 2
    b = t / np.sin(t)
    mf = manufactured(g)
    p = mf["p"] - mf["p"].mean()
    return a * mf["u"], b * p, float(a), float(b)


# ------------------------------------ generic manufactured solution ----------

def manufactured_generic(g: MacGrid, nu: float = 1.0):
    r"""The SECOND, GENERIC manufactured solution, added after the phase-1
    verification (`STOKES-PHASE1-VERIFY-codex.md`) found the frozen one
    degenerate: its discrete error is a uniform scalar amplitude factor, so its
    convergence table adds almost nothing beyond the three closed-form
    identities.  This one has error/solution cosine ~0.912, so it genuinely
    tests the SPATIAL structure of the discretization error.

        psi_g = sin^2(pi x) sin^2(2 pi y) + 0.3 sin^2(3 pi x) sin^2(pi y)
        u_g   = ( d psi_g/dy , -d psi_g/dx )
        p_g   = sin(4 pi x) + 0.37 cos(6 pi y) + 0.21 sin(2 pi x) cos(4 pi y)
        f_g   = -nu Lap(u_g) + grad(p_g)

    Explicitly,

        u = 2 pi sin^2(pi x) sin(4 pi y) + 0.3 pi sin^2(3 pi x) sin(2 pi y)
        v = -pi sin(2 pi x) sin^2(2 pi y) - 0.9 pi sin(6 pi x) sin^2(pi y)

    which is divergence-free by construction and vanishes on all four walls
    (every x-factor vanishes at x=0,1 and every y-factor at y=0,1), i.e.
    genuine no-slip.  Every derivative below is analytic; gate MMSF in
    stk2d_fom_gates.py checks each one against a high-accuracy finite
    difference of u and p, so an algebra slip cannot pass silently.
    """
    xu, yu = g.coords_u()
    xv, yv = g.coords_v()
    xp, yp = g.coords_p()
    s, c = np.sin, np.cos

    U = (2 * PI * s(PI * xu) ** 2 * s(4 * PI * yu)
         + 0.3 * PI * s(3 * PI * xu) ** 2 * s(2 * PI * yu))
    V = (-PI * s(2 * PI * xv) * s(2 * PI * yv) ** 2
         - 0.9 * PI * s(6 * PI * xv) * s(PI * yv) ** 2)
    P = (s(4 * PI * xp) + 0.37 * c(6 * PI * yp)
         + 0.21 * s(2 * PI * xp) * c(4 * PI * yp))

    # d2/dx2 sin^2(a pi x) = 2 a^2 pi^2 cos(2 a pi x)
    LapU = (4 * PI ** 3 * c(2 * PI * xu) * s(4 * PI * yu)
            - 32 * PI ** 3 * s(PI * xu) ** 2 * s(4 * PI * yu)
            + 5.4 * PI ** 3 * c(6 * PI * xu) * s(2 * PI * yu)
            - 1.2 * PI ** 3 * s(3 * PI * xu) ** 2 * s(2 * PI * yu))
    LapV = (4 * PI ** 3 * s(2 * PI * xv) * s(2 * PI * yv) ** 2
            - 8 * PI ** 3 * s(2 * PI * xv) * c(4 * PI * yv)
            + 32.4 * PI ** 3 * s(6 * PI * xv) * s(PI * yv) ** 2
            - 1.8 * PI ** 3 * s(6 * PI * xv) * c(2 * PI * yv))
    Px = 4 * PI * c(4 * PI * xu) + 0.42 * PI * c(2 * PI * xu) * c(4 * PI * yu)
    Py = (-2.22 * PI * s(6 * PI * yv)
          - 0.84 * PI * s(2 * PI * xv) * s(4 * PI * yv))

    FU = -nu * LapU + Px
    FV = -nu * LapV + Py
    return dict(U=U, V=V, P=P, FU=FU, FV=FV, LapU=LapU, LapV=LapV,
                Px=Px, Py=Py,
                u=g.pack(U, V), p=P.ravel(), f=g.pack(FU, FV))


# Point-evaluable forms of the two manufactured families, used ONLY by the
# forcing-consistency gate (finite differences of u and p at arbitrary points).

def _mms_frozen_point(x, y):
    s, c = np.sin, np.cos
    u = PI * s(PI * x) ** 2 * s(2 * PI * y)
    v = -PI * s(2 * PI * x) * s(PI * y) ** 2
    p = s(2 * PI * x) + c(2 * PI * y)
    return u, v, p


def _mms_generic_point(x, y):
    s, c = np.sin, np.cos
    u = 2 * PI * s(PI * x) ** 2 * s(4 * PI * y) \
        + 0.3 * PI * s(3 * PI * x) ** 2 * s(2 * PI * y)
    v = -PI * s(2 * PI * x) * s(2 * PI * y) ** 2 \
        - 0.9 * PI * s(6 * PI * x) * s(PI * y) ** 2
    p = s(4 * PI * x) + 0.37 * c(6 * PI * y) \
        + 0.21 * s(2 * PI * x) * c(4 * PI * y)
    return u, v, p


MMS_FAMILIES = {
    "frozen": dict(build=manufactured, point=_mms_frozen_point,
                   label="psi=sin^2(pi x) sin^2(pi y); p=sin(2 pi x)+cos(2 pi y)"
                         " (STOKES-DESIGN.md frozen contract)"),
    "generic": dict(build=manufactured_generic, point=_mms_generic_point,
                    label="psi=sin^2(pi x)sin^2(2 pi y)+0.3 sin^2(3 pi x)"
                          "sin^2(pi y); p=sin(4 pi x)+0.37 cos(6 pi y)"
                          "+0.21 sin(2 pi x)cos(4 pi y) (added 2026-08-30 "
                          "after STOKES-PHASE1-VERIFY-codex.md)"),
}


def mms_forcing_consistency(family: str, n_pts: int = 512, seed: int = 7,
                            eps: float = 1e-4):
    """Check the ANALYTIC Laplacian / gradient / divergence / no-slip of a
    manufactured family against high-accuracy finite differences of its own
    point-evaluable u, v, p.  Fourth-order central stencils at eps=1e-4 give
    ~1e-9 relative accuracy in f64, which is far tighter than any algebra slip
    (a wrong sign or coefficient shows as O(1)).  Returns relative errors."""
    fam = MMS_FAMILIES[family]
    pt = fam["point"]
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.05, 0.95, n_pts)
    y = rng.uniform(0.05, 0.95, n_pts)

    def d2(f, ax):
        dx = eps if ax == 0 else 0.0
        dy = eps if ax == 1 else 0.0
        vals = [f(x + k * dx, y + k * dy) for k in (-2, -1, 0, 1, 2)]
        return [(-a + 16 * b - 30 * c0 + 16 * d - e) / (12 * eps ** 2)
                for a, b, c0, d, e in zip(*vals)]

    def d1(f, ax):
        dx = eps if ax == 0 else 0.0
        dy = eps if ax == 1 else 0.0
        vals = [f(x + k * dx, y + k * dy) for k in (-2, -1, 1, 2)]
        return [(a - 8 * b + 8 * d - e) / (12 * eps)
                for a, b, d, e in zip(*vals)]

    uxx, vxx, pxx = d2(pt, 0)
    uyy, vyy, pyy = d2(pt, 1)
    ux, vx, px = d1(pt, 0)
    uy, vy, py = d1(pt, 1)
    lap_u_fd, lap_v_fd = uxx + uyy, vxx + vyy

    # analytic forms, evaluated at the same scattered points
    if family == "frozen":
        LapU = (2 * PI ** 3 * np.cos(2 * PI * x) * np.sin(2 * PI * y)
                - 4 * PI ** 3 * np.sin(PI * x) ** 2 * np.sin(2 * PI * y))
        LapV = (4 * PI ** 3 * np.sin(2 * PI * x) * np.sin(PI * y) ** 2
                - 2 * PI ** 3 * np.sin(2 * PI * x) * np.cos(2 * PI * y))
        Px = 2 * PI * np.cos(2 * PI * x)
        Py = -2 * PI * np.sin(2 * PI * y)
    else:
        s, c = np.sin, np.cos
        LapU = (4 * PI ** 3 * c(2 * PI * x) * s(4 * PI * y)
                - 32 * PI ** 3 * s(PI * x) ** 2 * s(4 * PI * y)
                + 5.4 * PI ** 3 * c(6 * PI * x) * s(2 * PI * y)
                - 1.2 * PI ** 3 * s(3 * PI * x) ** 2 * s(2 * PI * y))
        LapV = (4 * PI ** 3 * s(2 * PI * x) * s(2 * PI * y) ** 2
                - 8 * PI ** 3 * s(2 * PI * x) * c(4 * PI * y)
                + 32.4 * PI ** 3 * s(6 * PI * x) * s(PI * y) ** 2
                - 1.8 * PI ** 3 * s(6 * PI * x) * c(2 * PI * y))
        Px = 4 * PI * c(4 * PI * x) + 0.42 * PI * c(2 * PI * x) * c(4 * PI * y)
        Py = (-2.22 * PI * s(6 * PI * y)
              - 0.84 * PI * s(2 * PI * x) * s(4 * PI * y))

    def r(a, b):
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))

    # continuous divergence and the wall traces
    div = ux + vy
    tb = np.linspace(0.0, 1.0, 257)
    zeros = np.concatenate([np.concatenate(pt(np.zeros_like(tb), tb)[:2]),
                            np.concatenate(pt(np.ones_like(tb), tb)[:2]),
                            np.concatenate(pt(tb, np.zeros_like(tb))[:2]),
                            np.concatenate(pt(tb, np.ones_like(tb))[:2])])
    scale = float(np.max(np.abs(np.concatenate(pt(x, y)[:2]))))
    return dict(
        family=family, n_pts=int(n_pts), eps=eps,
        lap_u_rel=r(LapU, lap_u_fd), lap_v_rel=r(LapV, lap_v_fd),
        grad_px_rel=r(Px, px), grad_py_rel=r(Py, py),
        div_rel=float(np.linalg.norm(div)
                      / (np.linalg.norm(np.abs(ux) + np.abs(vy)) + 1e-300)),
        wall_trace_max=float(np.max(np.abs(zeros)) / scale))
