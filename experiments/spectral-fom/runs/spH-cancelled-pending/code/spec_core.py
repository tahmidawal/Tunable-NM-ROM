"""spectral-fom: fast-transform full-order solvers on the paper's own discretisations.

Every solver here solves the SAME discrete system as the paper's iterative FOM (5-/7-point
Dirichlet Laplacian scaled by n^2), exactly, by diagonalising it with the orthonormal DST-I.
Two DST-I implementations are provided and both are timed; the faster one is the spectral FOM:

  * `fft`  -- odd extension + real FFT of length 2(N+1), orthonormal (the construction of
             experiments/multiresolution-poisson/core.py:37 `dst1` and heat-bank-knob core.py:71
             `dst_axis`, re-implemented here so the file is self-contained);
  * `mm`   -- the dense orthonormal sine matrix S (N x N, symmetric, S @ S = I), applied by matmul
             along each axis.  O(N^{d+1}) work but a handful of large GEMMs.

Nothing here reads a neural network.  Large arrays are always explicit jit arguments.
"""
from __future__ import annotations

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp


# ------------------------------------------------------------------ DST-I -----

def dst1_fft(x, axis):
    """Orthonormal DST-I along `axis` (self-inverse)."""
    x = jnp.moveaxis(x, axis, -1)
    N = x.shape[-1]
    zero = jnp.zeros(x.shape[:-1] + (1,), x.dtype)
    ext = jnp.concatenate((zero, x, zero, -x[..., ::-1]), axis=-1)
    y = -jnp.fft.rfft(ext, axis=-1).imag[..., 1:N + 1] / jnp.sqrt(2.0 * (N + 1))
    return jnp.moveaxis(y, -1, axis)


def dstn_fft(x):
    for ax in range(x.ndim):
        x = dst1_fft(x, ax)
    return x


def sine_matrix(N):
    """Orthonormal DST-I matrix, N = n-1 interior points: S[j,k] = sqrt(2/n) sin(pi j k / n)."""
    n = N + 1
    j = np.arange(1, n)
    return np.sqrt(2.0 / n) * np.sin(np.pi * np.outer(j, j) / n)


def dstn_mm(x, S):
    if x.ndim == 2:
        return S @ x @ S                      # S symmetric
    # 3D: contract each axis with S
    x = jnp.einsum('ai,ijk->ajk', S, x)
    x = jnp.einsum('bj,ajk->abk', S, x)
    return jnp.einsum('ck,abk->abc', S, x)


def eig_axis(n):
    k = np.arange(1, n)
    return 4.0 * n ** 2 * np.sin(np.pi * k / (2 * n)) ** 2


def eig_grid(n, d):
    l = eig_axis(n)
    if d == 2:
        return l[:, None] + l[None, :]
    return l[:, None, None] + l[None, :, None] + l[None, None, :]


def interior(x):
    return x[(slice(1, -1),) * x.ndim]


def pad1(x):
    return jnp.pad(x, 1)


# --------------------------------------------------------------- Poisson ------

def make_poisson(n, d, variant):
    """Exact solve of  n^2 * (2d u - sum neighbours) = f  on the interior, zero Dirichlet.
    Returns fn(host-shaped full nodal source on device) -> full nodal field, plus its jit args."""
    lam = jnp.asarray(eig_grid(n, d))
    if variant == 'fft':
        @jax.jit
        def solve(source, lam):
            return pad1(dstn_fft(dstn_fft(interior(source)) / lam))
        return lambda s: solve(s, lam)
    S = jnp.asarray(sine_matrix(n - 1))

    @jax.jit
    def solve_mm(source, lam, S):
        return pad1(dstn_mm(dstn_mm(interior(source), S) / lam, S))
    return lambda s: solve_mm(s, lam, S)


# ------------------------------------------------------------------ Heat ------

def _dst_batched_fft(x, d):
    """Orthonormal DST-I over the last d axes of a batched array."""
    for ax in range(1, d + 1):
        x = dst1_fft(x, x.ndim - ax)
    return x


def _dst_batched_mm(x, S, d):
    if d == 2:
        return S @ x @ S                                  # broadcasts over the leading batch axis
    x = jnp.einsum('ai,...ijk->...ajk', S, x)
    x = jnp.einsum('bj,...ajk->...abk', S, x)
    return jnp.einsum('ck,...abk->...abc', S, x)


def make_heat(n, d, nu, times, variant, scheme, dt=None):
    """Exact modal solution of the heat lane's semi-discrete problem u' = -nu A u (A = the n^2-scaled Dirichlet
    Laplacian, zero Dirichlet) on the INTERIOR grid (the lane's query scope: interior (n-1)^d field in -> all output
    fields out).  scheme='exp': exact propagation exp(-nu t lam) (= the lane's same-grid truth, core.make_propagate);
    scheme='cn': Crank-Nicolson with step dt applied exactly in modal space, factor g^s,
    g = (1 - dt nu lam/2)/(1 + dt nu lam/2).  t = 0 is returned as the input itself (both factors are 1 there).
    The later outputs are transformed as one batch."""
    lam = eig_grid(n, d)
    assert times[0] == 0.0
    later = list(times[1:])
    if scheme == 'exp':
        fac = np.stack([np.exp(-nu * t * lam) for t in later])
    else:
        g = (1 - 0.5 * dt * nu * lam) / (1 + 0.5 * dt * nu * lam)
        steps = [int(round(t / dt)) for t in later]
        assert all(abs(s_ * dt - t) < 1e-12 for s_, t in zip(steps, later)), (times, dt)
        fac = np.stack([g ** s_ for s_ in steps])
    fac = jnp.asarray(fac)
    if variant == 'fft':
        @jax.jit
        def run(u0, fac):
            c = _dst_batched_fft(u0[None], d)
            return jnp.concatenate((u0[None], _dst_batched_fft(c * fac, d)))
        return lambda u: run(u, fac)
    S = jnp.asarray(sine_matrix(n - 1))

    @jax.jit
    def run_mm(u0, fac, S):
        c = _dst_batched_mm(u0[None], S, d)
        return jnp.concatenate((u0[None], _dst_batched_mm(c * fac, S, d)))
    return lambda u: run_mm(u, fac, S)


def make_poisson_interior(n, d, variant):
    """Interior (n-1)^d forcing in -> interior field out (the cube lane's query scope)."""
    lam = jnp.asarray(eig_grid(n, d))
    if variant == 'fft':
        @jax.jit
        def solve(f, lam):
            return dstn_fft(dstn_fft(f) / lam)
        return lambda f: solve(f, lam)
    S = jnp.asarray(sine_matrix(n - 1))

    @jax.jit
    def solve_mm(f, lam, S):
        return dstn_mm(dstn_mm(f, S) / lam, S)
    return lambda f: solve_mm(f, lam, S)


# --------------------------------------------------------------- Burgers ------

def make_burgers(L, dt, residual, m=0, max_iter=400, horizon=.25, spacing=.05, reg=1e-12, dst='fft', predictor=False):
    """Backward-Euler 2D Burgers on the paper's stencil (the `residual` passed in is the paper's own
    `mr-burgers2d/engines.residual`, imported unchanged), each step solved to ||r|| <= ntol ||prev|| (the
    paper's Newton stopping rule) by the fixed-point iteration  u <- u - H^{-1} r(u),  H = I + dt nu A
    (the modal Helmholtz inverse, applied exactly by DST-I).  Its fixed point is the backward-Euler
    solution.  m = 0: plain (Picard / chord) iteration; m > 0: Anderson acceleration with memory m.
    max_iter = 1 with ntol = 0 is the one-sweep IMEX scheme (explicit upwind advection, implicit diffusion).
    dst = 'fft' | 'mm' selects the DST-I implementation inside H^{-1} (same operator, round-off apart).
    predictor=True (Picard only): the first iterate of each step is the linear extrapolation 2 u_n - u_{n-1}
    (u_0 on the first step) instead of u_n; the stopping rule is unchanged, so the solution quality is too.
    query(full (L+1)^2 initial field, nu, ntol) -> (6 output fields, iterations per step, final rel. res.)"""
    assert not (predictor and m)
    k = np.arange(1, L)
    l1 = 4.0 * L ** 2 * np.sin(np.pi * k / (2 * L)) ** 2
    lam = jnp.asarray(l1[:, None] + l1[None, :])
    nsub = int(round(spacing / dt))
    nout = int(round(horizon / spacing))
    assert abs(nsub * dt - spacing) < 1e-12
    N = (L - 1) ** 2

    S = jnp.asarray(sine_matrix(L - 1)) if dst == 'mm' else jnp.zeros((0, 0))

    def hinv(v, nu, lam, S):
        if dst == 'mm':
            return dstn_mm(dstn_mm(v.reshape(L - 1, L - 1), S) / (1.0 + dt * nu * lam), S).reshape(-1)
        return dstn_fft(dstn_fft(v.reshape(L - 1, L - 1)) / (1.0 + dt * nu * lam)).reshape(-1)

    def step(prev, nu, ntol, lam, S, guess=None):
        thr = ntol * jnp.maximum(jnp.linalg.norm(prev), 1e-300)
        start = prev if guess is None else guess
        r0 = residual(start, prev, nu, dt, L)
        if m == 0:
            def body(s):
                u, r, it = s
                u = u - hinv(r, nu, lam, S)
                return u, residual(u, prev, nu, dt, L), it + 1
            u, r, it = jax.lax.while_loop(
                lambda s: (jnp.linalg.norm(s[1]) > thr) & (s[2] < max_iter) & jnp.all(jnp.isfinite(s[1][:1])),
                body, (start, r0, jnp.int32(0)))
            return u, (it, jnp.linalg.norm(r) / jnp.maximum(jnp.linalg.norm(prev), 1e-300))
        # Anderson (type II): x_{k+1} = g_k - dG^T gamma, gamma = argmin || f_k - dF^T gamma ||
        def body(s):
            x, r, fprev, gprev, dF, dG, it = s
            f = -hinv(r, nu, lam, S)
            g = x + f
            slot = (it - 1) % m
            push = it > 0
            dF = jnp.where(push, dF.at[slot].set(f - fprev), dF)
            dG = jnp.where(push, dG.at[slot].set(g - gprev), dG)
            valid = (jnp.arange(m) < jnp.minimum(it, m)).astype(x.dtype)
            A = (dF @ dF.T) * valid[:, None] * valid[None, :]
            scale = jnp.maximum(jnp.trace(A), 1e-300)
            A = A + (reg * scale + (1.0 - valid)) * jnp.eye(m)
            gam = jnp.linalg.solve(A, (dF @ f) * valid)
            xn = g - gam @ dG
            return xn, residual(xn, prev, nu, dt, L), f, g, dF, dG, it + 1
        z = jnp.zeros((m, N), prev.dtype)
        s = jax.lax.while_loop(
            lambda s: (jnp.linalg.norm(s[1]) > thr) & (s[6] < max_iter) & jnp.all(jnp.isfinite(s[1][:1])),
            body, (prev, r0, jnp.zeros_like(prev), jnp.zeros_like(prev), z, z, jnp.int32(0)))
        u, r, it = s[0], s[1], s[6]
        return u, (it, jnp.linalg.norm(r) / jnp.maximum(jnp.linalg.norm(prev), 1e-300))

    @jax.jit
    def query(u0, nu, ntol, lam, S):
        if predictor:
            def pstep(c, _):
                u, up = c
                un, info = step(u, nu, ntol, lam, S, guess=2.0 * u - up)
                return (un, u), info

            def block(c, _):
                c, (it, rr) = jax.lax.scan(pstep, c, None, length=nsub)
                return c, (jnp.pad(c[0].reshape(L - 1, L - 1), 1), it, rr)
            x0 = u0[1:-1, 1:-1].reshape(-1)
            _, (fields, it, rr) = jax.lax.scan(block, (x0, x0), None, length=nout)
        else:
            def block(u, _):
                u, (it, rr) = jax.lax.scan(lambda u, _: step(u, nu, ntol, lam, S), u, None, length=nsub)
                return u, (jnp.pad(u.reshape(L - 1, L - 1), 1), it, rr)
            _, (fields, it, rr) = jax.lax.scan(block, u0[1:-1, 1:-1].reshape(-1), None, length=nout)
        return jnp.concatenate((jnp.pad(u0[1:-1, 1:-1], 1)[None], fields)), it.reshape(-1), rr.reshape(-1)
    return lambda u0, nu, ntol: query(u0, nu, ntol, lam, S)
