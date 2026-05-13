"""EQ node + weight selection for the Poisson NM-ROM.

Restricted to STRICTLY interior nodes ([1, N-2]^d), so the (2d+1)-point
stencil never touches the boundary at runtime. Otherwise identical in
structure to the Heat EQ module.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import nnls


def _flat_index(ijk, N, d):
    if d == 2:
        return ijk[:, 0] * N + ijk[:, 1]
    return ijk[:, 0] * N * N + ijk[:, 1] * N + ijk[:, 2]


def _stencil_offsets(N, d):
    if d == 2:
        return np.asarray([0, -1, 1, -N, N], dtype=np.int64)
    return np.asarray([0, -1, 1, -N, N, -N * N, N * N], dtype=np.int64)


def _interior_indices(N, d):
    rng = np.arange(1, N - 1)
    coords = np.stack(np.meshgrid(*([rng] * d), indexing="ij"), axis=-1).reshape(-1, d)
    return _flat_index(coords, N, d), coords


def build_v_eq(
    decoder_params: dict,
    eq_flat_indices: np.ndarray,
    N: int,
    spatial_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    offsets = _stencil_offsets(N, spatial_dim)
    stencil_indices = eq_flat_indices[:, None] + offsets[None, :]
    flat = stencil_indices.reshape(-1)
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
    snapshots: np.ndarray,                  # (M, N**d)
    K_op_numpy,                             # callable: u_flat -> K @ u (numpy)
    N: int,
    spatial_dim: int,
    n_eq_samples: int = 200,
    min_eq_points: int = 2000,
    weight_tol: float = 1e-10,
    rng: np.random.Generator | None = None,
):
    """Select interior EQ nodes via NNLS on |K u_i| over a sample of training snapshots.

    The design matrix row for snapshot i is `|K @ u_i|` restricted to
    strictly interior nodes. Weights are normalised to sum to 1.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    M = snapshots.shape[0]
    sample_idx = rng.choice(M, size=min(n_eq_samples, M), replace=False)

    int_flat, _ = _interior_indices(N, spatial_dim)
    G = np.empty((sample_idx.size, int_flat.size), dtype=np.float32)
    for i, idx in enumerate(sample_idx):
        Ku = np.abs(K_op_numpy(snapshots[idx]))
        G[i] = Ku[int_flat]
    b = G.sum(axis=1)
    w, _ = nnls(G, b, maxiter=10000)

    keep = np.where(w > weight_tol)[0]
    if keep.size < min_eq_points:
        order = np.argsort(-w)
        keep = order[:min_eq_points]
    eq_flat = int_flat[keep]
    eq_w = w[keep]
    eq_w = eq_w / eq_w.sum()
    return eq_flat, eq_w
