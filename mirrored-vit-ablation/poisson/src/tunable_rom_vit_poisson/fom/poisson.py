"""Full-order model for the parametric Poisson equation.

    -Delta u(x) = F(x; mu)  on  [0, 1]^d
    u = 0                    on  boundary

with a single tensor-product sinusoid source of amplitude 10 and
parametric wavenumbers `mu = (k1, ..., kd)`, each in [1, 3].

Discretization: uniform Cartesian grid, second-order centered FD
Laplacian (5-point in 2D, 7-point in 3D). Boundary rows are identity
so that boundary DOFs stay at zero throughout CG. The FOM solver is
matrix-free `jax.scipy.sparse.linalg.cg` applied to the negative
discrete Laplacian.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np


CG_TOL = 1e-6
CG_MAXITER = 1000


def _boundary_mask(N: int, d: int) -> jnp.ndarray:
    if d == 2:
        m = np.ones((N, N), dtype=np.float32)
        m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 0.0
    else:
        m = np.ones((N, N, N), dtype=np.float32)
        m[0, :, :] = m[-1, :, :] = 0.0
        m[:, 0, :] = m[:, -1, :] = 0.0
        m[:, :, 0] = m[:, :, -1] = 0.0
    return jnp.asarray(m)


def _neg_laplacian_2d(u_grid, dx):
    """-Laplacian with identity on boundary."""
    out = jnp.zeros_like(u_grid)
    c = u_grid[1:-1, 1:-1]
    n = u_grid[2:, 1:-1]
    s = u_grid[:-2, 1:-1]
    e = u_grid[1:-1, 2:]
    w = u_grid[1:-1, :-2]
    out = out.at[1:-1, 1:-1].set((4 * c - n - s - e - w) / dx**2)
    out = out.at[0, :].set(u_grid[0, :])
    out = out.at[-1, :].set(u_grid[-1, :])
    out = out.at[:, 0].set(u_grid[:, 0])
    out = out.at[:, -1].set(u_grid[:, -1])
    return out


def _neg_laplacian_3d(u_grid, dx):
    out = jnp.zeros_like(u_grid)
    c = u_grid[1:-1, 1:-1, 1:-1]
    xp = u_grid[2:, 1:-1, 1:-1]
    xm = u_grid[:-2, 1:-1, 1:-1]
    yp = u_grid[1:-1, 2:, 1:-1]
    ym = u_grid[1:-1, :-2, 1:-1]
    zp = u_grid[1:-1, 1:-1, 2:]
    zm = u_grid[1:-1, 1:-1, :-2]
    out = out.at[1:-1, 1:-1, 1:-1].set((6 * c - xp - xm - yp - ym - zp - zm) / dx**2)
    # Identity on boundary slabs.
    out = out.at[0, :, :].set(u_grid[0, :, :])
    out = out.at[-1, :, :].set(u_grid[-1, :, :])
    out = out.at[:, 0, :].set(u_grid[:, 0, :])
    out = out.at[:, -1, :].set(u_grid[:, -1, :])
    out = out.at[:, :, 0].set(u_grid[:, :, 0])
    out = out.at[:, :, -1].set(u_grid[:, :, -1])
    return out


@dataclass
class PoissonFOM:
    N: int
    spatial_dim: int
    L: float = 1.0
    amplitude: float = 10.0

    def __post_init__(self):
        self.dx = self.L / (self.N - 1)
        self.mask = _boundary_mask(self.N, self.spatial_dim)
        self.num_nodes = self.N**self.spatial_dim

    def neg_laplacian(self, u_grid):
        if self.spatial_dim == 2:
            return _neg_laplacian_2d(u_grid, self.dx)
        return _neg_laplacian_3d(u_grid, self.dx)

    def K_op(self, u_flat):
        u_grid = u_flat.reshape((self.N,) * self.spatial_dim)
        return self.neg_laplacian(u_grid).reshape(-1)

    def cg_solve(self, F_flat, x0=None, tol=CG_TOL, maxiter=CG_MAXITER):
        if x0 is None:
            x0 = jnp.zeros_like(F_flat)
        u, _ = jax.scipy.sparse.linalg.cg(self.K_op, F_flat, x0=x0, tol=tol, maxiter=maxiter)
        return u * self.mask.reshape(-1)


def source_field(fom: PoissonFOM, freqs: Sequence[float]) -> jnp.ndarray:
    """F(x; mu) = A * prod_i sin(k_i * pi * x_i), flat."""
    d = fom.spatial_dim
    assert len(freqs) == d
    axes = [jnp.linspace(0.0, fom.L, fom.N) for _ in range(d)]
    grid = jnp.meshgrid(*axes, indexing="ij")
    f = fom.amplitude * jnp.ones_like(grid[0])
    for g, k in zip(grid, freqs):
        f = f * jnp.sin(k * jnp.pi * g)
    return f.reshape(-1)


def sample_parameters(rng: np.random.Generator, spatial_dim: int) -> dict:
    freqs = rng.uniform(1.0, 3.0, size=spatial_dim).tolist()
    return {"freqs": freqs}
