"""Matrix-free Kronecker operator and PCG solvers (JAX) for Family A3 (see PROTOCOL amendment 1).

Preconditioners: none, Jacobi, fast diagonalisation (FD) of the diagonal part sum_i K_ii d_i^2.
"""
import numpy as np, scipy.linalg as sla, time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import aniso3d as a3


class Solver:
    def __init__(self, N, p, dtype=jnp.float64):
        self.sp = sp = a3.Space(N, p)
        self.n = sp.n
        M, S, C = (np.asarray(m.todense()) for m in (sp.M, sp.S, sp.C))
        lam, V = sla.eigh(S, M)                      # V^T M V = I, V^T S V = diag(lam)
        self.np_mats = (M, S, C)
        self.M, self.S, self.C, self.V, self.lam = (jnp.asarray(a, dtype) for a in (M, S, C, V, lam))
        self.dM, self.dS = jnp.asarray(np.diag(M), dtype), jnp.asarray(np.diag(S), dtype)
        self.dtype = dtype
        self._solve = jax.jit(self._pcg, static_argnames=("prec",))

    # axis applications
    @staticmethod
    def _x(A, U): return jnp.tensordot(A, U, axes=(1, 0))
    @staticmethod
    def _y(A, U): return jnp.einsum("jb,abc->ajc", A, U)
    @staticmethod
    def _z(A, U): return U @ A.T

    def matvec(self, K, U):
        M, S, C = self.M, self.S, self.C; Ct = C.T
        x, y, z = self._x, self._y, self._z
        Mz, Sz, Cz, Ctz = z(M, U), z(S, U), z(C, U), z(Ct, U)
        MyMz, SyMz, CyMz, CtyMz = y(M, Mz), y(S, Mz), y(C, Mz), y(Ct, Mz)
        out = x(S, K[0, 0] * MyMz) + x(M, K[1, 1] * SyMz + K[2, 2] * y(M, Sz)
                                         + K[1, 2] * (y(C, Ctz) + y(Ct, Cz)))
        out = out + K[0, 1] * (x(C, CtyMz) + x(Ct, CyMz))
        out = out + K[0, 2] * (x(C, y(M, Ctz)) + x(Ct, y(M, Cz)))
        return out

    def _prec(self, prec, K):
        if prec == "none":
            return lambda r: r
        if prec == "jacobi":
            dM, dS = self.dM, self.dS
            d = (K[0, 0] * dS[:, None, None] * dM[None, :, None] * dM[None, None, :]
                 + K[1, 1] * dM[:, None, None] * dS[None, :, None] * dM[None, None, :]
                 + K[2, 2] * dM[:, None, None] * dM[None, :, None] * dS[None, None, :])
            return lambda r: r / d
        V, lam = self.V, self.lam; Vt = V.T
        den = K[0, 0] * lam[:, None, None] + K[1, 1] * lam[None, :, None] + K[2, 2] * lam[None, None, :]
        x, y, z = self._x, self._y, self._z
        return lambda r: x(V, y(V, z(V, x(Vt, y(Vt, z(Vt, r))) / den)))

    def _pcg(self, K, b, tol, maxit, prec="fd"):
        A = lambda u: self.matvec(K, u)
        Pinv = self._prec(prec, K)
        bn = jnp.linalg.norm(b)
        x0 = jnp.zeros_like(b); r0 = b; z0 = Pinv(r0)
        def cond(s): return (s[5] < maxit) & (jnp.linalg.norm(s[1]) > tol * bn)
        def body(s):
            x, r, z, p, rz, k = s
            Ap = A(p); al = rz / jnp.vdot(p, Ap)
            x = x + al * p; r = r - al * Ap; z = Pinv(r); rz2 = jnp.vdot(r, z)
            return (x, r, z, z + (rz2 / rz) * p, rz2, k + 1)
        x, r, z, p, rz, k = jax.lax.while_loop(cond, body, (x0, r0, z0, z0, jnp.vdot(r0, z0), 0))
        return x, k, jnp.linalg.norm(r) / bn

    def load(self, mu):
        return jnp.asarray(self.sp.load(mu).reshape(self.n, self.n, self.n), self.dtype)

    def solve(self, mu, b=None, tol=1e-8, maxit=20000, prec="fd"):
        K = jnp.asarray(a3.ktensor(mu), self.dtype)
        if b is None: b = self.load(mu)
        x, k, res = self._solve(K, b, tol, maxit, prec=prec)
        return x, int(k), float(res)

    def timed(self, mu, tol, prec, rep=5, maxit=20000):
        K = jnp.asarray(a3.ktensor(mu), self.dtype); b = self.load(mu)
        x, k, res = self._solve(K, b, tol, maxit, prec=prec); x.block_until_ready()
        ts = []
        for _ in range(rep):
            t = time.perf_counter(); x, k, res = self._solve(K, b, tol, maxit, prec=prec); x.block_until_ready(); ts.append(time.perf_counter() - t)
        return np.asarray(x), int(k), dict(med_ms=1e3 * float(np.median(ts)), min_ms=1e3 * min(ts), max_ms=1e3 * max(ts))
