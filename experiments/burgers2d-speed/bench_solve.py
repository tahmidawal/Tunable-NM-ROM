"""Solve micro-benchmark (engineering diagnostics only): variants of the damped normal-equation solve at R' = 128/384,
each repeated N times inside ONE jitted fori_loop, both compile modes; and their agreement with cho_factor/cho_solve."""
import time
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import b2fast as B2

N = 200


def blocked_chol(A, nb):
    """Right-looking blocked Cholesky, diagonal blocks by lax.linalg.cholesky, panels by triangular solves."""
    n = A.shape[0]
    Lm = jnp.zeros_like(A)
    for k in range(0, n, nb):
        Akk = A[k:k + nb, k:k + nb] - Lm[k:k + nb, :k] @ Lm[k:k + nb, :k].T
        Lkk = jax.lax.linalg.cholesky(Akk)
        Lm = Lm.at[k:k + nb, k:k + nb].set(Lkk)
        if k + nb < n:
            B = A[k + nb:, k:k + nb] - Lm[k + nb:, :k] @ Lm[k:k + nb, :k].T
            X = jax.lax.linalg.triangular_solve(Lkk, B, left_side=False, lower=True, transpose_a=True)
            Lm = Lm.at[k + nb:, k:k + nb].set(X)
    return Lm


def unrolled_chol(A):
    """Cholesky-Crout by columns with jnp ops in a fori_loop (one small kernel set per column)."""
    n = A.shape[0]

    def col(j, Lm):
        v = A[:, j] - Lm @ Lm[j, :]
        d = jnp.sqrt(v[j])
        c = jnp.where(jnp.arange(n) >= j, v / d, 0.)
        return Lm.at[:, j].set(c)
    return jax.lax.fori_loop(0, n, col, jnp.zeros_like(A))


def tri_solve_pair(Lm, g):
    y = jax.lax.linalg.triangular_solve(Lm, g[:, None], left_side=True, lower=True)
    return jax.lax.linalg.triangular_solve(Lm, y, left_side=True, lower=True, transpose_a=True)[:, 0]


variants = {
    'cho_factor+cho_solve (current)': lambda H, g: jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(H, lower=True), g),
    'potrf only': lambda H, g: jnp.linalg.cholesky(H)[:, 0] + g * 0,
    'potrf batched [None]': lambda H, g: jax.scipy.linalg.cho_solve((jnp.linalg.cholesky(H[None])[0], True), g),
    'potrf + inverse-free trsm pair': lambda H, g: tri_solve_pair(jnp.linalg.cholesky(H), g),
    'blocked chol nb=32 + trsm': lambda H, g: tri_solve_pair(blocked_chol(H, 32), g),
    'blocked chol nb=64 + trsm': lambda H, g: tri_solve_pair(blocked_chol(H, 64), g),
    'crout fori chol + trsm': lambda H, g: tri_solve_pair(unrolled_chol(H), g),
    'LU jnp.linalg.solve': lambda H, g: jnp.linalg.solve(H, g),
}
for Rp in (128, 384):
    rng = np.random.default_rng(0)
    J = jnp.asarray(rng.normal(size=(4 * Rp, Rp)))
    H0 = J.T @ J
    g0 = jnp.asarray(rng.normal(size=Rp))
    ref = variants['cho_factor+cho_solve (current)'](H0 + jnp.diag(1e-6 * jnp.diag(H0)), g0)
    for name, f in variants.items():
        for gr in (False, True):
            def body(i, x, f=f):
                H = H0 + jnp.diag((1e-6 + 1e-12 * x[0] ** 2) * jnp.diag(H0))
                return x + 1e-9 * f(H, g0)
            loop = lambda x: jax.lax.fori_loop(0, N, body, x)
            g = jax.jit(loop, compiler_options=B2.GRAPHS) if gr else jax.jit(loop)
            x0 = jnp.zeros(Rp)
            jax.block_until_ready(g(x0))
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                jax.block_until_ready(g(x0))
                ts.append(time.perf_counter() - t0)
            sol = jax.jit(f)(H0 + jnp.diag(1e-6 * jnp.diag(H0)), g0)
            dev = float(jnp.linalg.norm(sol - ref) / jnp.linalg.norm(ref)) if 'only' not in name else float('nan')
            print(f'R{Rp} {name:34s} graphs={gr!s:5s} {1e6 * np.median(ts) / N:8.1f} us  rel-diff {dev:.1e}', flush=True)
print('BENCH DONE', jax.devices()[0].device_kind, flush=True)
