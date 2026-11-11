"""Matrix-free Kronecker operator and PCG solvers (JAX) for Family A3 (see PROTOCOL amendment 1).

Preconditioners: none, Jacobi, fast diagonalisation (FD) of the diagonal part sum_i K_ii d_i^2.
"""
import numpy as np, scipy.linalg as sla, time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import aniso3d as a3


class Solver:
    def __init__(self, N, p, dtype=jnp.float64):
        self.sp = sp = a3.Space(N, p); self.p_ = p
        self.n = sp.n
        M, S, C = (np.asarray(m.todense()) for m in (sp.M, sp.S, sp.C))
        lam, V = sla.eigh(S, M)                      # V^T M V = I, V^T S V = diag(lam)
        self.np_mats = (M, S, C)
        self.M, self.S, self.C, self.V, self.lam = (jnp.asarray(a, dtype) for a in (M, S, C, V, lam))
        self.dM, self.dS = jnp.asarray(np.diag(M), dtype), jnp.asarray(np.diag(S), dtype)
        self.dtype = dtype
        # element-local operators for the convolution form of the operator (uniform grids):
        # A u = G^T (A_loc (G u)), G = gather of the (p+1)^3 local dofs of each element (stride p)
        q, w = np.polynomial.legendre.leggauss(p + 1)
        import hq as _hq
        Vq, Dq = _hq.ref1d(p, q); h = 1.0 / N
        m = np.einsum("q,qi,qj->ij", w, Vq, Vq) * h / 2; s_ = np.einsum("q,qi,qj->ij", w, Dq, Dq) * 2 / h; c = np.einsum("q,qi,qj->ij", w, Dq, Vq)
        k3 = lambda a, b, c_: np.einsum("ad,be,cf->abcdef", a, b, c_).reshape((p + 1) ** 3, (p + 1) ** 3)
        loc = [k3(s_, m, m), k3(m, s_, m), k3(m, m, s_), k3(c, c.T, m) + k3(c.T, c, m), k3(c, m, c.T) + k3(c.T, m, c), k3(m, c, c.T) + k3(m, c.T, c)]
        self.loc = jnp.asarray(np.stack(loc), dtype)                  # (6, L, L), order xx yy zz xy xz yz
        L = (p + 1) ** 3
        self._E = jnp.asarray(np.eye(L).reshape(L, 1, p + 1, p + 1, p + 1), dtype)
        self._full = (1, 1, N * p + 1, N * p + 1, N * p + 1)
        gather = lambda Uf: jax.lax.conv_general_dilated(Uf, self._E, (p, p, p), "VALID")
        self._scatter = jax.linear_transpose(gather, jnp.zeros(self._full, dtype))
        self.conv = __import__("os").environ.get("OPFORM", "conv") != "dense"
        self._solve = jax.jit(self._pcg, static_argnames=("prec",))

    # axis applications
    @staticmethod
    def _x(A, U): return jnp.tensordot(A, U, axes=(1, 0))
    @staticmethod
    def _y(A, U): return jnp.einsum("jb,abc->ajc", A, U)
    @staticmethod
    def _z(A, U): return U @ A.T

    def matvec(self, K, U):
        if self.conv:
            p = self.p_; L = (p + 1) ** 3
            Aloc = (K[0, 0] * self.loc[0] + K[1, 1] * self.loc[1] + K[2, 2] * self.loc[2]
                    + K[0, 1] * self.loc[3] + K[0, 2] * self.loc[4] + K[1, 2] * self.loc[5])
            Uf = jnp.pad(U, 1)[None, None]
            y = jax.lax.conv_general_dilated(Uf, Aloc.reshape(L, 1, p + 1, p + 1, p + 1), (p, p, p), "VALID")
            return self._scatter(y)[0][0, 0, 1:-1, 1:-1, 1:-1]
        return self.matvec_dense(K, U)

    def matvec_dense(self, K, U):
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
