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

def make_heat(n, d, nu, times, variant, scheme, dt=None):
    """Exact modal solution of the paper's semi-discrete heat problem u' = -nu A u (A = the n^2-scaled
    Dirichlet Laplacian), zero Dirichlet, on the interior.  Returns all output fields (len(times), full
    nodal grid).  scheme='exp': exact propagation exp(-nu t lam); scheme='cn': Crank-Nicolson with step
    dt applied exactly in modal space, g = (1 - dt nu lam/2)/(1 + dt nu lam/2), g^steps per output."""
    lam = eig_grid(n, d)
    if scheme == 'exp':
        fac = np.stack([np.exp(-nu * t * lam) for t in times])
    else:
        g = (1 - 0.5 * dt * nu * lam) / (1 + 0.5 * dt * nu * lam)
        steps = [int(round(t / dt)) for t in times]
        assert all(abs(s * dt - t) < 1e-12 for s, t in zip(steps, times)), (times, dt)
        fac = np.stack([g ** s for s in steps])
    fac = jnp.asarray(fac)
    if variant == 'fft':
        @jax.jit
        def run(u0, fac):
            c = dstn_fft(interior(u0))
            return jax.lax.map(lambda f: pad1(dstn_fft(c * f)), fac)
        return lambda u: run(u, fac)
    S = jnp.asarray(sine_matrix(n - 1))

    @jax.jit
    def run_mm(u0, fac, S):
        c = dstn_mm(interior(u0), S)
        return jax.lax.map(lambda f: pad1(dstn_mm(c * f, S)), fac)
    return lambda u: run_mm(u, fac, S)
