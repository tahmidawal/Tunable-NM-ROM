"""Component micro-benchmark (engineering diagnostics only, never a result): per-call device time of the pieces of
one linear-rung LM time step on synthetic arrays of the deployed shapes (m = 3969 lattice nodes, M = 4R'), each
repeated N times inside ONE jitted fori_loop, in both compile modes."""
import time
import sys

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import hops as H
import b2fast as B2

N = 200
L = 256
out = []


def timeit(label, f, *args, graphs=False):
    g = jax.jit(f, compiler_options=B2.GRAPHS) if graphs else jax.jit(f)
    jax.block_until_ready(g(*args))
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        jax.block_until_ready(g(*args))
        ts.append(time.perf_counter() - t0)
    us = 1e6 * np.median(ts) / N
    out.append((label, graphs, us))
    print(f'{label:40s} graphs={graphs!s:5s} {us:9.1f} us/call', flush=True)


for Rp in (128, 384):
    M = 4 * Rp
    kx, ky, lam = H.modes_lean(L, M)
    rng = np.random.default_rng(0)
    m = 63 * 63
    G5 = jnp.asarray(rng.normal(size=(m, 5, Rp)) * .1)
    A = jnp.asarray(rng.normal(size=(M, Rp)))
    ij, w = H.lattice_rule(L, 64)
    Pq = jnp.asarray(H.phi_rows(L, kx, ky, ij) * w[:, None])
    sep = B2.lattice_tables(L, 64, kx, ky)
    lamj = jnp.asarray(lam)
    dt, nu = .005, .05
    dadv = jax.vmap(jax.grad(lambda u: H.advect_points(u, L)))

    def res(wv, sepd):
        us = jnp.einsum('msr,r->ms', G5, wv)
        ah = A @ wv
        return (ah + dt * (B2.proj_vec(H.advect_points(us, L), sepd) + nu * lamj * ah)) / (1 + dt * nu * lamj)

    def evalJ(wv, sepd):
        us = jnp.einsum('msr,r->ms', G5, wv)
        ah = A @ wv
        S = 1. / (1 + dt * nu * lamj)
        r = (ah + dt * (B2.proj_vec(H.advect_points(us, L), sepd) + nu * lamj * ah)) * S
        dA = jnp.einsum('ms,msr->mr', dadv(us), G5)
        return r, A + (dt * S)[:, None] * B2.proj_cols(dA, sepd)

    def evalJ_dense(wv, P):
        us = jnp.einsum('msr,r->ms', G5, wv)
        ah = A @ wv
        S = 1. / (1 + dt * nu * lamj)
        r = (ah + dt * (P.T @ H.advect_points(us, L) + nu * lamj * ah)) * S
        dA = jnp.einsum('ms,msr->mr', dadv(us), G5)
        return r, A + (dt * S)[:, None] * (P.T @ dA)

    w0 = jnp.asarray(rng.normal(size=Rp) * .01)
    loop = lambda body: (lambda x, *a: jax.lax.fori_loop(0, N, lambda i, c: body(c, *a), x))

    def b_res3(x, sepd):
        cand = jnp.stack((x, 1.001 * x, .999 * x))
        return x + 1e-12 * jnp.sum(jax.vmap(lambda v: jnp.linalg.norm(res(v, sepd)))(cand))

    def b_evalJ(x, sepd):
        r, J = evalJ(x, sepd)
        return x + 1e-12 * (J.T @ r)

    def b_evalJd(x, P):
        r, J = evalJ_dense(x, P)
        return x + 1e-12 * (J.T @ r)

    Hm = jnp.asarray(rng.normal(size=(M, Rp)))
    Hm = Hm.T @ Hm + jnp.eye(Rp)

    def b_chol(x, Hm):
        c = jax.scipy.linalg.cho_factor(Hm + 1e-9 * jnp.diag(x ** 2), lower=True)
        return x + 1e-12 * jax.scipy.linalg.cho_solve(c, x)

    def b_lu(x, Hm):
        return x + 1e-12 * jnp.linalg.solve(Hm + 1e-9 * jnp.diag(x ** 2), x)

    def b_normal(x, sepd):
        r, J = evalJ(x, sepd)
        Hh = J.T @ J
        return x + 1e-12 * (Hh @ x)

    def b_while(x, sepd):
        def one(c):
            v, k = c
            return (v * 1.0000001, k + 1)
        v, _ = jax.lax.while_loop(lambda c: c[1] < 1, one, (x, 0))
        return v

    for gr in (False, True):
        timeit(f'R{Rp} residual x3 (vmapped)', loop(b_res3), w0, sep, graphs=gr)
        timeit(f'R{Rp} evalJ separable (E1)', loop(b_evalJ), w0, sep, graphs=gr)
        timeit(f'R{Rp} evalJ dense Pq (parent)', loop(b_evalJd), w0, Pq, graphs=gr)
        timeit(f'R{Rp} evalJ + J^T J', loop(b_normal), w0, sep, graphs=gr)
        timeit(f'R{Rp} cholesky factor+solve', loop(b_chol), w0, Hm, graphs=gr)
        timeit(f'R{Rp} LU solve', loop(b_lu), w0, Hm, graphs=gr)
        timeit(f'R{Rp} empty 1-trip while loop', loop(b_while), w0, sep, graphs=gr)
print('BENCH DONE', jax.devices()[0].device_kind, flush=True)
