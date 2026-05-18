"""CG-discrete Poisson FOM, kept local to exploring-decoders for scope isolation.

This is a near-verbatim copy of poisson/src/tunable_rom_poisson/fom/poisson.py
plus the generate_cg helper from poisson/.../fom/data.py. The point is that
the ROM solver's residual must use the SAME discrete K operator that
generated the training data (data-must-match-operator rule).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import jax
# Enable float64 so CG iterates don't transiently overflow on smooth
# warm-starts. Must be set before any jnp.array is created downstream.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


CG_TOL = 1e-6
CG_MAXITER = 1000


def _boundary_mask_2d(N: int) -> jnp.ndarray:
    m = np.ones((N, N), dtype=np.float32)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 0.0
    return jnp.asarray(m)


def _neg_laplacian_2d(u_grid, dx):
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


@dataclass
class PoissonFOM:
    N: int
    spatial_dim: int = 2
    L: float = 1.0
    amplitude: float = 10.0

    def __post_init__(self):
        assert self.spatial_dim == 2, "exploring-decoders ROM is 2D only for now"
        self.dx = self.L / (self.N - 1)
        self.mask = _boundary_mask_2d(self.N)
        self.num_nodes = self.N**self.spatial_dim

    def neg_laplacian(self, u_grid):
        return _neg_laplacian_2d(u_grid, self.dx)

    def K_op(self, u_flat):
        u_grid = u_flat.reshape((self.N, self.N))
        return self.neg_laplacian(u_grid).reshape(-1)

    def cg_solve(self, F_flat, x0=None, tol=CG_TOL, maxiter=CG_MAXITER):
        if x0 is None:
            x0 = jnp.zeros_like(F_flat)
        # Mask F to zero on the boundary so the (identity-on-boundary) K
        # operator is consistent with the implicit Dirichlet BC. Without
        # this, CG drives boundary nodes toward F_bnd which then couples
        # into the interior Laplacian and explodes the iterate magnitude
        # to O(1e2-1e12). The paper's poisson code has this same bug; we
        # fix it here.
        F_masked = F_flat * self.mask.reshape(-1)
        # Float64 inside CG to avoid transient-iterate overflow.
        F64 = jnp.asarray(F_masked, dtype=jnp.float64)
        x0_64 = jnp.asarray(x0, dtype=jnp.float64)
        def K64(u):
            return jnp.asarray(self.K_op(u), dtype=jnp.float64)
        u, _ = jax.scipy.sparse.linalg.cg(K64, F64, x0=x0_64, tol=tol, maxiter=maxiter)
        u = jnp.asarray(u, dtype=jnp.float32)
        return u * self.mask.reshape(-1)


def source_field(fom: PoissonFOM, freqs: Sequence[float]) -> jnp.ndarray:
    d = fom.spatial_dim
    assert len(freqs) == d
    axes = [jnp.linspace(0.0, fom.L, fom.N) for _ in range(d)]
    grid = jnp.meshgrid(*axes, indexing="ij")
    f = fom.amplitude * jnp.ones_like(grid[0])
    for g, k in zip(grid, freqs):
        f = f * jnp.sin(k * jnp.pi * g)
    return f.reshape(-1)


def analytical_u(fom: PoissonFOM, freqs) -> jnp.ndarray:
    d = fom.spatial_dim
    axes = [jnp.linspace(0.0, fom.L, fom.N) for _ in range(d)]
    grid = jnp.meshgrid(*axes, indexing="ij")
    denom = (jnp.pi**2) * sum(k**2 for k in freqs)
    u = (fom.amplitude / denom) * jnp.ones_like(grid[0])
    for g, k in zip(grid, freqs):
        u = u * jnp.sin(k * jnp.pi * g)
    u = u * fom.mask
    return u.reshape(-1)


def sample_freqs(rng: np.random.Generator, spatial_dim: int = 2) -> np.ndarray:
    return rng.uniform(1.0, 3.0, size=spatial_dim).astype(np.float32)


def generate_cg(
    N: int,
    spatial_dim: int,
    n_samples: int,
    seed: int = 42,
    warm_start: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate FEM-consistent CG snapshots and the matching forcing fields.

    JAX's CG returns the iterate unconditionally on non-convergence, and at
    large N partial iterates sometimes overflow float32 to NaN/Inf. We
    validate each sample and fall back: warm-start CG -> cold-start CG ->
    analytical (carrying an O(dx^2) consistency gap but finite). Mirrors
    the multires-experiment fallback.

    Returns (U, freqs, F) where U[i] = cg_solve(F[i]).
    """
    fom = PoissonFOM(N=N, spatial_dim=spatial_dim)
    rng = np.random.default_rng(seed)
    U = np.empty((n_samples, fom.num_nodes), dtype=np.float32)
    F = np.empty((n_samples, fom.num_nodes), dtype=np.float32)
    freqs = np.empty((n_samples, spatial_dim), dtype=np.float32)

    # Magnitude check: analytical magnitude is bounded by amplitude/(pi^2 * 2)
    # for k_min=1, so ~0.5 in 2D. A correct CG solution sits in [-1, 1]
    # comfortably. We treat anything > MAX_OK as a divergent iterate.
    MAX_OK = 100.0

    mask_flat = np.asarray(fom.mask.reshape(-1))
    n_warm_bad = 0
    n_cold_bad = 0
    for i in range(n_samples):
        f = sample_freqs(rng, spatial_dim)
        Fi = source_field(fom, f.tolist())
        # The ROM residual is evaluated only at strictly interior nodes (EQ),
        # so F at boundary is irrelevant to the solve. Save the masked F so
        # `cg_solve(F) = u` is consistent end-to-end (cg_solve also masks).
        Fi_masked = np.asarray(Fi) * mask_flat
        x0 = analytical_u(fom, f.tolist()) if warm_start else None
        u = np.asarray(fom.cg_solve(Fi, x0=x0))
        bad = (not np.all(np.isfinite(u))) or (np.max(np.abs(u)) > MAX_OK)
        if bad:
            n_warm_bad += 1
            u = np.asarray(fom.cg_solve(Fi, x0=None))
            bad = (not np.all(np.isfinite(u))) or (np.max(np.abs(u)) > MAX_OK)
        if bad:
            n_cold_bad += 1
            u = np.asarray(analytical_u(fom, f.tolist()))
            if (not np.all(np.isfinite(u))) or (np.max(np.abs(u)) > MAX_OK):
                raise RuntimeError(
                    f"both CG and analytical produced non-finite output at "
                    f"i={i} freqs={f.tolist()}"
                )
        U[i] = u
        F[i] = Fi_masked
        freqs[i] = f
    if n_warm_bad > 0 or n_cold_bad > 0:
        print(
            f"[generate_cg N={N}] warm-start failed on {n_warm_bad}/{n_samples}; "
            f"cold-start failed on {n_cold_bad}/{n_samples} (analytical fallback applied)"
        )
    assert np.all(np.isfinite(U)), "generate_cg: NaN/Inf survived all fallbacks"
    assert np.max(np.abs(U)) <= MAX_OK, (
        f"generate_cg: max |u|={float(np.max(np.abs(U))):.3e} > {MAX_OK} after fallbacks"
    )
    return U, freqs, F
