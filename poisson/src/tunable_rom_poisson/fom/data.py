"""Two data generators for the Poisson NM-ROM.

`generate_analytical`: closed-form solution
    u(x; mu) = (A / (pi^2 * sum k_i^2)) * prod_i sin(k_i * pi * x_i)
of the continuous PDE. Cheap to evaluate. Used for N <= 128.

`generate_cg`: CG-discrete solution
    -K @ u = F   solved by `jax.scipy.sparse.linalg.cg`
of the discretised problem. Required for N >= 256 because the
analytical and FEM/CG solutions differ by an O(dx^2) consistency gap
that becomes comparable to the AE-reconstruction error and pollutes
the test-time rel-L2 measurement.

Rule: the data generator must agree with the test-time FOM operator.
If the benchmark is `cg_solve(K, F)`, generate with `generate_cg`.
"""
from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
import numpy as np

from .poisson import PoissonFOM, source_field, sample_parameters


def _analytical_u(fom: PoissonFOM, freqs):
    d = fom.spatial_dim
    axes = [jnp.linspace(0.0, fom.L, fom.N) for _ in range(d)]
    grid = jnp.meshgrid(*axes, indexing="ij")
    denom = (jnp.pi ** 2) * sum(k**2 for k in freqs)
    u = (fom.amplitude / denom) * jnp.ones_like(grid[0])
    for g, k in zip(grid, freqs):
        u = u * jnp.sin(k * jnp.pi * g)
    u = u * fom.mask
    return u.reshape(-1)


def generate_analytical(N: int, spatial_dim: int, n_samples: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (U, freqs) where U has shape (n_samples, N**d)."""
    fom = PoissonFOM(N=N, spatial_dim=spatial_dim)
    rng = np.random.default_rng(seed)
    U = np.empty((n_samples, fom.num_nodes), dtype=np.float32)
    freqs = np.empty((n_samples, spatial_dim), dtype=np.float32)
    for i in range(n_samples):
        params = sample_parameters(rng, spatial_dim)
        U[i] = np.asarray(_analytical_u(fom, params["freqs"]))
        freqs[i] = params["freqs"]
    return U, freqs


def generate_cg(N: int, spatial_dim: int, n_samples: int, seed: int = 42, warm_start: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Generate FEM-consistent snapshots via CG. Optionally warm-start CG with the analytical solution."""
    fom = PoissonFOM(N=N, spatial_dim=spatial_dim)
    rng = np.random.default_rng(seed)
    U = np.empty((n_samples, fom.num_nodes), dtype=np.float32)
    freqs = np.empty((n_samples, spatial_dim), dtype=np.float32)
    for i in range(n_samples):
        params = sample_parameters(rng, spatial_dim)
        F = source_field(fom, params["freqs"])
        x0 = _analytical_u(fom, params["freqs"]) if warm_start else None
        u = fom.cg_solve(F, x0=x0)
        U[i] = np.asarray(u)
        freqs[i] = params["freqs"]
    return U, freqs
