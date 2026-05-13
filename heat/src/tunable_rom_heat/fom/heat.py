"""Full-order model for the parametric Heat equation.

    du/dt = kappa * laplacian(u)  on  [0, 1]^d
    u = 0                          on  the boundary
    u(x, 0) = sum of 1..3 Gaussian blobs

Discretization: uniform Cartesian grid, second-order centered finite
differences (5-point in 2D, 7-point in 3D). Time integration: backward
Euler with dt = 0.005, 50 steps. Each implicit step is solved with CG
against the matrix-free implicit operator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np


DT = 0.005
NUM_STEPS = 50
CG_TOL = 1e-6
CG_MAXITER = 1000


def _build_boundary_mask(N: int, d: int) -> jnp.ndarray:
    """1.0 on interior nodes, 0.0 on the boundary."""
    if d == 2:
        mask = np.ones((N, N), dtype=np.float32)
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0.0
    elif d == 3:
        mask = np.ones((N, N, N), dtype=np.float32)
        mask[0, :, :] = mask[-1, :, :] = 0.0
        mask[:, 0, :] = mask[:, -1, :] = 0.0
        mask[:, :, 0] = mask[:, :, -1] = 0.0
    else:
        raise ValueError(f"d must be 2 or 3, got {d}")
    return jnp.asarray(mask)


def _laplacian_2d(u_grid: jnp.ndarray, dx: float) -> jnp.ndarray:
    """5-point centered Laplacian, identity on the boundary."""
    N = u_grid.shape[0]
    lap = jnp.zeros_like(u_grid)
    c = u_grid[1:-1, 1:-1]
    n = u_grid[2:, 1:-1]
    s = u_grid[:-2, 1:-1]
    e = u_grid[1:-1, 2:]
    w = u_grid[1:-1, :-2]
    lap = lap.at[1:-1, 1:-1].set((n + s + e + w - 4 * c) / dx**2)
    return lap


def _laplacian_3d(u_grid: jnp.ndarray, dx: float) -> jnp.ndarray:
    """7-point centered Laplacian, identity on the boundary."""
    lap = jnp.zeros_like(u_grid)
    c = u_grid[1:-1, 1:-1, 1:-1]
    xp = u_grid[2:, 1:-1, 1:-1]
    xm = u_grid[:-2, 1:-1, 1:-1]
    yp = u_grid[1:-1, 2:, 1:-1]
    ym = u_grid[1:-1, :-2, 1:-1]
    zp = u_grid[1:-1, 1:-1, 2:]
    zm = u_grid[1:-1, 1:-1, :-2]
    lap = lap.at[1:-1, 1:-1, 1:-1].set((xp + xm + yp + ym + zp + zm - 6 * c) / dx**2)
    return lap


@dataclass
class HeatFOM:
    """Heat equation FOM on a (N,)^d grid."""

    N: int
    spatial_dim: int  # 2 or 3
    L: float = 1.0

    def __post_init__(self):
        self.dx = self.L / (self.N - 1)
        self.mask = _build_boundary_mask(self.N, self.spatial_dim)
        self.num_nodes = self.N**self.spatial_dim

    def laplacian(self, u_grid):
        if self.spatial_dim == 2:
            return _laplacian_2d(u_grid, self.dx)
        return _laplacian_3d(u_grid, self.dx)

    def implicit_op(self, u_flat, kappa):
        """A(u) = u - dt*kappa*lap(u), with identity on the boundary."""
        u_grid = u_flat.reshape((self.N,) * self.spatial_dim)
        Lu = self.laplacian(u_grid)
        Au = u_grid - DT * kappa * Lu
        # Keep boundary identity (so u_boundary = u_prev_boundary = 0).
        Au = jnp.where(self.mask > 0, Au, u_grid)
        return Au.reshape(-1)

    def step(self, u_prev_flat, kappa):
        """One implicit-Euler step via CG."""
        A = lambda v: self.implicit_op(v, kappa)
        u_next, _ = jax.scipy.sparse.linalg.cg(
            A, u_prev_flat, x0=u_prev_flat, tol=CG_TOL, maxiter=CG_MAXITER
        )
        # Re-enforce Dirichlet (CG can leak slightly).
        u_next = u_next * self.mask.reshape(-1)
        return u_next

    def rollout(self, u0_flat, kappa, num_steps: int = NUM_STEPS):
        def body(_, u):
            return self.step(u, kappa)
        u_final = jax.lax.fori_loop(0, num_steps, body, u0_flat)
        return u_final

    def rollout_trajectory(self, u0_flat, kappa, num_steps: int = NUM_STEPS):
        """Return all snapshots (num_steps+1, N**d) — useful for AE training data."""
        snaps = [u0_flat]
        u = u0_flat
        for _ in range(num_steps):
            u = self.step(u, kappa)
            snaps.append(u)
        return jnp.stack(snaps, axis=0)


def make_initial_condition(
    fom: HeatFOM,
    centers: Sequence[Sequence[float]],
    amplitudes: Sequence[float],
    widths: Sequence[float],
) -> jnp.ndarray:
    """Sum of axis-aligned Gaussian blobs, masked to be zero on the boundary."""
    d = fom.spatial_dim
    axes = [jnp.linspace(0.0, fom.L, fom.N) for _ in range(d)]
    grid = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)  # (N,)*d + (d,)
    u = jnp.zeros((fom.N,) * d)
    for c, a, w in zip(centers, amplitudes, widths):
        r2 = jnp.sum((grid - jnp.asarray(c)) ** 2, axis=-1)
        u = u + a * jnp.exp(-r2 / (2 * w**2))
    u = u * fom.mask
    return u.reshape(-1)


def sample_parameters(rng: np.random.Generator, spatial_dim: int) -> dict:
    """Latin-hypercube-style draw of one trajectory's parameters.

    Returns kappa (log-uniform in [0.01, 0.5]) and 1..3 Gaussian blobs.
    """
    n_gauss = int(np.round(1 + 2 * rng.random()))  # 1, 2, or 3
    kappa = float(np.exp(rng.uniform(np.log(0.01), np.log(0.5))))
    centers = [list(rng.uniform(0.15, 0.85, size=spatial_dim)) for _ in range(n_gauss)]
    amplitudes = [float(rng.uniform(1.0, 10.0)) for _ in range(n_gauss)]
    widths = [float(rng.uniform(0.05, 0.20)) for _ in range(n_gauss)]
    return {"kappa": kappa, "centers": centers, "amplitudes": amplitudes, "widths": widths}


def generate_trajectory(fom: HeatFOM, params: dict, num_steps: int = NUM_STEPS):
    """Generate a single (num_steps+1, N**d) trajectory from sampled parameters."""
    u0 = make_initial_condition(fom, params["centers"], params["amplitudes"], params["widths"])
    snaps = fom.rollout_trajectory(u0, params["kappa"], num_steps=num_steps)
    return snaps
