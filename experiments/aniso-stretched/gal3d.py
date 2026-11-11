"""Galerkin ROM for Family A3 from any nodal basis Phi (n^3 x R) on the truth space (p, N).

Offline: six affine reduced operators Phi^T T_ij Phi; load = constant part + separable Gaussian part, the latter through a
1D POD of the 1D Gaussian load vectors (m modes) -> tensor (m, m, m, R). Online cost is independent of the grid:
O(n m) for the 1D load coefficients, O(R m^3) contraction, O(6 R^2) assembly, O(R^3) Cholesky.
"""
import time
import numpy as np, scipy.linalg as sla
import jax, jax.numpy as jnp
import aniso as an, aniso3d as a3, solve3d as s3

PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def unitK(i, j):
    K = np.zeros((3, 3)); K[i, j] = K[j, i] = 1.0
    return K


class Galerkin:
    def __init__(self, S: "s3.Solver", Phi, m_load=16, chunk=16):
        """Phi: (n^3, R) float64 numpy. Builds reduced operators with the matrix-free operator of S."""
        t0 = time.perf_counter()
        n = S.n; R = Phi.shape[1]; self.S, self.R, self.n = S, R, n
        PhiJ = jnp.asarray(Phi)
        mv = jax.jit(jax.vmap(lambda K, u: S.matvec(K, u).ravel(), in_axes=(None, 3), out_axes=1))
        self.Aij = []
        for (i, j) in PAIRS:
            K = jnp.asarray(unitK(i, j)); cols = []
            for c in range(0, R, chunk):
                cols.append(mv(K, PhiJ[:, c:c + chunk].reshape(n, n, n, -1)))
            self.Aij.append(np.asarray(PhiJ.T @ jnp.concatenate(cols, axis=1)))
        self.Aij = np.stack(self.Aij)                                    # (6, R, R)
        sp = S.sp; self.Lq = sp.Lq.toarray(); self.xq = sp.xq
        l1 = self.Lq @ np.ones(self.Lq.shape[1])
        self.b0 = Phi.T @ (l1[:, None, None] * l1[None, :, None] * l1[None, None, :]).ravel()
        cs = np.linspace(0.3, 0.7, 401)
        Lc = self.Lq @ np.exp(-(self.xq[:, None] - cs[None, :]) ** 2 / (2 * an.SIGMA ** 2))   # (n, 401)
        Uq, s, _ = np.linalg.svd(Lc, full_matrices=False)
        self.Q = Uq[:, :m_load]; self.load_trunc = float(s[m_load] / s[0]) if m_load < len(s) else 0.0
        Qj = jnp.asarray(self.Q)
        T = jnp.einsum("abcr,ai,bj,ck->ijkr", PhiJ.reshape(n, n, n, R), Qj, Qj, Qj)
        self.T = np.asarray(T)                                           # (m, m, m, R)
        self.offline_seconds = time.perf_counter() - t0

    def restrict(self, r):
        g = object.__new__(Galerkin)
        g.__dict__.update(self.__dict__); g.R = r
        g.Aij = np.ascontiguousarray(self.Aij[:, :r, :r]); g.b0 = self.b0[:r].copy(); g.T = np.ascontiguousarray(self.T[..., :r])
        return g

    def online(self, mu):
        """Coefficients for parameter mu (numpy, CPU)."""
        az, el, eps = np.deg2rad(mu[0]), np.deg2rad(mu[1]), 10.0 ** mu[2]
        ce = np.cos(el); b = np.array([ce * np.cos(az), ce * np.sin(az), np.sin(el)])
        K = (1 - eps) * np.outer(b, b); K[0, 0] += eps; K[1, 1] += eps; K[2, 2] += eps
        A = (K[0, 0] * self.Aij[0] + K[1, 1] * self.Aij[1] + K[2, 2] * self.Aij[2]
             + K[0, 1] * self.Aij[3] + K[0, 2] * self.Aij[4] + K[1, 2] * self.Aij[5])
        g = np.exp(-(self.xq[None, :] - mu[3:6, None]) ** 2 / (2 * an.SIGMA ** 2))        # (3, nq)
        a = (g @ self.Lq.T) @ self.Q                                                       # (3, m)
        rhs = self.b0 + an.AMP * (a[2] @ np.tensordot(a[1], np.tensordot(a[0], self.T, 1), 1))
        return sla.cho_solve(sla.cho_factor(A, check_finite=False), rhs, check_finite=False)

    def timed(self, mu, rep=200):
        self.online(mu); ts = []
        for _ in range(rep):
            t = time.perf_counter(); c = self.online(mu); ts.append(time.perf_counter() - t)
        return c, dict(med_ms=1e3 * float(np.median(ts)), min_ms=1e3 * min(ts), max_ms=1e3 * max(ts))
