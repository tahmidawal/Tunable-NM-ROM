"""Analytical parametric Poisson — both on-grid snapshots and off-mesh ground truth.

PDE:
    -Delta u(x) = A * prod_i sin(k_i pi x_i)   on [0,1]^d
    u = 0 on the boundary

Closed-form solution:
    u(x; mu) = (A / (pi^2 * sum k_i^2)) * prod_i sin(k_i pi x_i)

This is exact for any x in [0,1]^d (not just grid nodes), which is why
we use it as the off-mesh reference for the INR decoders.

Parameters mu = (k_1, ..., k_d), each in [1, 3]; amplitude A = 10.
Matches the conventions of poisson/src/tunable_rom_poisson/fom/poisson.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


AMPLITUDE = 10.0
K_LOW, K_HIGH = 1.0, 3.0


@dataclass
class PoissonAnalytical:
    """Analytical Poisson solver. Stateless w.r.t. params; only N + d."""
    N: int
    spatial_dim: int = 2
    L: float = 1.0
    amplitude: float = AMPLITUDE

    def __post_init__(self):
        self.dx = self.L / (self.N - 1)
        self.num_nodes = self.N ** self.spatial_dim
        self._axes = [np.linspace(0.0, self.L, self.N, dtype=np.float32)
                      for _ in range(self.spatial_dim)]

    def u_on_grid(self, freqs: np.ndarray) -> np.ndarray:
        """Return u evaluated on the full uniform grid, flattened. (N**d,)."""
        d = self.spatial_dim
        assert len(freqs) == d
        denom = (np.pi ** 2) * float(np.sum(np.asarray(freqs) ** 2))
        coeff = self.amplitude / denom
        grids = np.meshgrid(*self._axes, indexing="ij")
        u = coeff * np.ones_like(grids[0])
        for g, k in zip(grids, freqs):
            u = u * np.sin(k * np.pi * g)
        # Boundary is implicitly zero from sin(k*pi*0)=sin(k*pi*1)=0 (for integer k);
        # for non-integer k, the analytical solution is still 0 at x=0 (sin(0)=0)
        # but not at x=1 unless k is an integer. We do NOT enforce a zero mask
        # because here freqs are integers — the source is sin(k pi x) and so is
        # the solution.
        return u.astype(np.float32).reshape(-1)

    def u_at_points(self, freqs: np.ndarray, x_query: np.ndarray) -> np.ndarray:
        """Evaluate u at arbitrary points x_query: (M, d) -> (M,)."""
        d = self.spatial_dim
        assert x_query.shape[-1] == d
        denom = (np.pi ** 2) * float(np.sum(np.asarray(freqs) ** 2))
        coeff = self.amplitude / denom
        prod = np.ones(x_query.shape[0], dtype=np.float32)
        for j, k in enumerate(freqs):
            prod = prod * np.sin(k * np.pi * x_query[:, j])
        return (coeff * prod).astype(np.float32)


def sample_freqs(rng: np.random.Generator, spatial_dim: int = 2) -> np.ndarray:
    """Sample mu = (k_1, ..., k_d) uniform in [1, 3]^d. Matches the paper."""
    return rng.uniform(K_LOW, K_HIGH, size=spatial_dim).astype(np.float32)


def generate_dataset(
    N: int,
    spatial_dim: int,
    n_samples: int,
    seed: int = 42,
):
    """Generate (U_grid, freqs):
        U_grid: (n_samples, N**d)  flattened analytical fields on the FD grid
        freqs:  (n_samples, d)
    """
    fom = PoissonAnalytical(N=N, spatial_dim=spatial_dim)
    rng = np.random.default_rng(seed)
    U = np.empty((n_samples, fom.num_nodes), dtype=np.float32)
    F = np.empty((n_samples, spatial_dim), dtype=np.float32)
    for i in range(n_samples):
        freqs = sample_freqs(rng, spatial_dim)
        F[i] = freqs
        U[i] = fom.u_on_grid(freqs)
    return U, F
