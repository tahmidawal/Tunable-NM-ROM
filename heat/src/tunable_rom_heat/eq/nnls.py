"""Empirical-quadrature (EQ) node and weight selection via NNLS.

The reduced residual is integrated over a sparse subset of grid nodes,
chosen so that the EQ-weighted integral reproduces the full-grid
integral over a representative set of training snapshots.

Key implementation note: building the design matrix via
`jax.jacfwd(decoder)(z)` on the full grid is O(N**d * k) in memory and
OOMs at N>=128. We exploit the CP decoder structure: the field at any
node n is a contraction of small per-axis factors `W_x[r,ix(n)]`,
`W_y[r,iy(n)]`, `W_z[r,iz(n)]`. The decoder Jacobian factors as
    dU/dz |_n = (Jh(z) @ V_full[:, n])
where Jh = d(MLP_out)/dz is the SMALL rank x k Jacobian. We assemble
the design matrix as one BLAS GEMM, never materializing the full
N**d x k Jacobian.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import nnls


def _flat_index(ijk: np.ndarray, N: int, d: int) -> np.ndarray:
    """Convert (n, d) integer coords to flat indices in row-major order."""
    if d == 2:
        return ijk[:, 0] * N + ijk[:, 1]
    return ijk[:, 0] * N * N + ijk[:, 1] * N + ijk[:, 2]


def _stencil_offsets(N: int, d: int) -> np.ndarray:
    """Flat-index offsets for the (2d+1)-point stencil [center, -x, +x, ...]."""
    if d == 2:
        return np.asarray([0, -1, 1, -N, N], dtype=np.int64)
    return np.asarray([0, -1, 1, -N, N, -N * N, N * N], dtype=np.int64)


def _interior_flat_indices(N: int, d: int) -> np.ndarray:
    """Flat indices of all strictly interior nodes [1..N-2]^d."""
    rng = np.arange(1, N - 1)
    coords = np.stack(np.meshgrid(*([rng] * d), indexing="ij"), axis=-1).reshape(-1, d)
    return _flat_index(coords, N, d), coords


def build_v_eq(
    decoder_params: dict,
    eq_flat_indices: np.ndarray,
    N: int,
    spatial_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute V_eq for stencil evaluation.

    Returns:
        v_eq_st: (rank, n_eq * (2d+1)) — CP basis at center + neighbours.
        stencil_indices: (n_eq, 2d+1) flat indices used for gather.
    """
    offsets = _stencil_offsets(N, spatial_dim)
    stencil_indices = eq_flat_indices[:, None] + offsets[None, :]
    flat = stencil_indices.reshape(-1)
    # Recover per-axis indices.
    if spatial_dim == 2:
        ix = flat // N
        iy = flat % N
        v = decoder_params["W_x"][:, ix] * decoder_params["W_y"][:, iy]
    else:
        ix = flat // (N * N)
        iy = (flat // N) % N
        iz = flat % N
        v = (
            decoder_params["W_x"][:, ix]
            * decoder_params["W_y"][:, iy]
            * decoder_params["W_z"][:, iz]
        )
    return np.asarray(v), stencil_indices


def compute_eq_weights(
    model,
    params: dict,
    snapshots: np.ndarray,
    N: int,
    spatial_dim: int,
    n_eq_samples: int = 16,
    min_eq_points: int = 64,
    weight_tol: float = 1e-10,
    rng: np.random.Generator | None = None,
):
    """Select EQ nodes + weights via NNLS over training snapshots.

    Design-matrix rows: |u_i - decode(encode(u_i))| at strictly interior
    nodes, for n_eq_samples training snapshots. Weights are normalised
    to sum to 1.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    M = snapshots.shape[0]
    sample_idx = rng.choice(M, size=min(n_eq_samples, M), replace=False)

    int_flat, _ = _interior_flat_indices(N, spatial_dim)

    G = np.empty((sample_idx.size, int_flat.size), dtype=np.float32)
    for i, idx in enumerate(sample_idx):
        u = jnp.asarray(snapshots[idx])
        u_hat = model.apply({"params": params}, u)
        R = np.abs(np.asarray(u - u_hat))
        G[i] = R[int_flat]

    b = G.sum(axis=1)
    w, _ = nnls(G, b, maxiter=5000)

    keep = np.where(w > weight_tol)[0]
    if keep.size < min_eq_points:
        order = np.argsort(-w)
        keep = order[:min_eq_points]
    eq_flat = int_flat[keep]
    eq_weights = w[keep]
    eq_weights = eq_weights / eq_weights.sum()
    return eq_flat, eq_weights
