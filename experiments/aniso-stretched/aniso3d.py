"""3D extension: -div(K grad u) = f on the unit cube, K = eps I + (1-eps) b b^T, degree-p tensor elements.

Family A3: b = (cos el cos az, cos el sin az, sin el) constant in space.
mu = (az_deg, el_deg, log10 eps, cx, cy, cz).
"""
import numpy as np, scipy.sparse as sp
import aniso as an, hq

LO = np.array([15.0, 15.0, -4.0, 0.3, 0.3, 0.3])
HI = np.array([75.0, 75.0, -2.0, 0.7, 0.7, 0.7])
SEEDS = {"dev": 3101, "test": 3102, "train": 3103}


def sample_params(cohort, n):
    return LO + np.random.default_rng(SEEDS[cohort]).random((n, 6)) * (HI - LO)


def ktensor(mu):
    az, el, eps = np.deg2rad(mu[0]), np.deg2rad(mu[1]), 10.0 ** mu[2]
    b = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return eps * np.eye(3) + (1 - eps) * np.outer(b, b)


def source(mu, x, y, z):
    r2 = (x[:, None, None] - mu[3]) ** 2 + (y[None, :, None] - mu[4]) ** 2 + (z[None, None, :] - mu[5]) ** 2
    return 1.0 + an.AMP * np.exp(-r2 / (2 * an.SIGMA ** 2))


class Space:
    """Degree-p space on an N^3 tensor grid (stretch beta). Holds the 6 affine operator terms."""

    def __init__(self, N, p, beta=0.0):
        self.N, self.p, self.beta = N, p, beta
        xv = an.nodes(N, beta); h = np.diff(xv)
        q, w = np.polynomial.legendre.leggauss(p + 1)
        V, D = hq.ref1d(p, q)
        self.n1 = n1 = N * p + 1
        g = (np.arange(N)[:, None] * p + np.arange(p + 1)[None, :])
        # 1D global matrices: mass M, stiffness S, mixed C[i,j] = int phi_i' phi_j
        def glob(Lmat):   # Lmat (N, p+1, p+1)
            r = np.broadcast_to(g[:, :, None], Lmat.shape).ravel(); c = np.broadcast_to(g[:, None, :], Lmat.shape).ravel()
            return sp.coo_matrix((Lmat.ravel(), (r, c)), shape=(n1, n1)).tocsr()
        Mref = np.einsum("q,qi,qj->ij", w, V, V); Sref = np.einsum("q,qi,qj->ij", w, D, D); Cref = np.einsum("q,qi,qj->ij", w, D, V)
        self.M = glob(Mref[None] * (h / 2)[:, None, None])[1:-1, 1:-1]
        self.S = glob(Sref[None] * (2 / h)[:, None, None])[1:-1, 1:-1]
        self.C = glob(np.broadcast_to(Cref, (N, p + 1, p + 1)).copy())[1:-1, 1:-1]
        self.xq = (xv[:-1, None] + 0.5 * (q + 1) * h[:, None]).ravel()           # quadrature points (N*(p+1),)
        # load operator: nodal dofs <- quadrature values, per direction
        Lq = np.einsum("q,qi->iq", w, V)[None] * (h / 2)[:, None, None]            # (N, p+1, nq)
        r = np.broadcast_to(g[:, :, None], Lq.shape).ravel()
        c = np.broadcast_to((np.arange(N)[:, None] * (p + 1) + np.arange(p + 1)[None, :])[:, None, :], Lq.shape).ravel()
        self.Lq = sp.coo_matrix((Lq.ravel(), (r, c)), shape=(n1, N * (p + 1))).tocsr()[1:-1]
        self.n = n1 - 2
        k3 = lambda a, b, c: sp.kron(sp.kron(a, b, format="csr"), c, format="csr")
        M, S, C = self.M, self.S, self.C
        self.terms = {
            (0, 0): k3(S, M, M), (1, 1): k3(M, S, M), (2, 2): k3(M, M, S),
            (0, 1): k3(C, C.T, M) + k3(C.T, C, M), (0, 2): k3(C, M, C.T) + k3(C.T, M, C), (1, 2): k3(M, C, C.T) + k3(M, C.T, C),
        }
        self._idx = None

    def matrix(self, mu):
        K = ktensor(mu)
        A = None
        for (i, j), T in self.terms.items():
            A = K[i, j] * T if A is None else A + K[i, j] * T
        return A.tocsr()

    def load(self, mu):
        F = source(mu, self.xq, self.xq, self.xq)
        F = np.einsum("ia,abc->ibc", self.Lq.toarray(), F) if self.Lq.shape[1] < 400 else (self.Lq @ F.reshape(F.shape[0], -1)).reshape(-1, F.shape[1], F.shape[2])
        F = np.moveaxis((self.Lq @ np.moveaxis(F, 1, 0).reshape(F.shape[1], -1)).reshape(-1, F.shape[0], F.shape[2]), 0, 1)
        F = np.moveaxis((self.Lq @ np.moveaxis(F, 2, 0).reshape(F.shape[2], -1)).reshape(-1, F.shape[0], F.shape[1]), 0, 2)
        return np.ascontiguousarray(F).ravel()

    def interp(self, xe):
        return hq.interp1d(self.N, self.beta, self.p, xe)[:, 1:-1]


def apply3(P, U):
    """(P x P x P) U for U of shape (n,n,n) -> (m,m,m)."""
    U = np.asarray(P @ U.reshape(U.shape[0], -1)).reshape(-1, U.shape[1], U.shape[2])
    U = np.moveaxis(np.asarray(P @ np.moveaxis(U, 1, 0).reshape(U.shape[1], -1)).reshape(-1, U.shape[0], U.shape[2]), 0, 1)
    U = np.moveaxis(np.asarray(P @ np.moveaxis(U, 2, 0).reshape(U.shape[2], -1)).reshape(-1, U.shape[0], U.shape[1]), 0, 2)
    return U


class Evaluator3:
    def __init__(self, n=128):
        self.x = np.linspace(0, 1, n + 1)
        w = an.lumped1d(self.x)
        self.W = w[:, None, None] * w[None, :, None] * w[None, None, :]

    def field(self, space, u):
        return apply3(space.interp(self.x), u.reshape(space.n, space.n, space.n))

    def errors(self, Ue, Uref):
        d = Ue - Uref
        return dict(l2w=float(np.sqrt((self.W * d * d).sum() / (self.W * Uref * Uref).sum())),
                    linf=float(np.abs(d).max() / np.abs(Uref).max()))
