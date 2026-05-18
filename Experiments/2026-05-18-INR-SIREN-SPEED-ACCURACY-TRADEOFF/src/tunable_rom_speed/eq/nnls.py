"""EQ node + weight selection for the decoder-agnostic NM-ROM.

Verbatim recipe from poisson/src/tunable_rom_poisson/eq/nnls.py: NNLS on
|K @ u_i| over a sample of training snapshots, restricted to strictly
interior nodes [1, N-2]^d so the 5-point stencil never touches the
boundary at runtime.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import nnls


def _flat_index_2d(ij, N):
    return ij[:, 0] * N + ij[:, 1]


def _interior_indices_2d(N: int):
    rng = np.arange(1, N - 1)
    coords = np.stack(np.meshgrid(rng, rng, indexing="ij"), axis=-1).reshape(-1, 2)
    return _flat_index_2d(coords, N), coords


def build_stencil(eq_flat_indices: np.ndarray, N: int) -> np.ndarray:
    """Return (n_eq, 5) array of flat indices [center, -x, +x, -y, +y]."""
    offsets = np.asarray([0, -1, 1, -N, N], dtype=np.int64)
    return eq_flat_indices[:, None] + offsets[None, :]


def compute_eq_weights(
    snapshots: np.ndarray,                  # (M, N**d)
    K_op_numpy,                             # callable: u_flat -> K @ u (numpy)
    N: int,
    spatial_dim: int = 2,
    n_eq_samples: int = 200,
    min_eq_points: int = 2000,
    weight_tol: float = 1e-10,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select interior EQ nodes via NNLS on |K u_i| over a sample of snapshots.

    Returns (eq_flat_indices, eq_weights) with sum(weights) = 1.
    """
    assert spatial_dim == 2, "exploring-decoders EQ is 2D-only"
    if rng is None:
        rng = np.random.default_rng(0)
    M = snapshots.shape[0]
    sample_idx = rng.choice(M, size=min(n_eq_samples, M), replace=False)

    int_flat, _ = _interior_indices_2d(N)
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
    return eq_flat.astype(np.int64), eq_w.astype(np.float32)
