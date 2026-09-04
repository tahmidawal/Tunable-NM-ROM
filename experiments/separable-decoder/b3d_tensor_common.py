"""Precomputed quadratic advection tensor for the 3D Burgers weak residual
(2026-09-03; port of b2d_tensor_common.py).

On a NON-NEGATIVE field the sign-upwind stencil is the fixed backward
difference on all three axes (at u_c = 0 the branch is multiplied by zero), so
with the frozen interior bank G (n_i, R) and DG = D-x G + D-y G + D-z G:

    Phi^T (u . (D-x u + D-y u + D-z u)) = h^T T h,
    T[i, j, k] = sum_x Phi[x, i] G[x, j] DG[x, k],

ONE tensor for all three axes.  T is not symmetric in (j, k); the stored
table is Q = T + T^(j<->k): q(h) = 0.5 h^T Q h (= h^T T h), dq/dh = Q h.
Built in f64, blocked over x (the n_i x R^2 intermediate is never
materialised: at N=129, R=128 it would be 275 GB); each block is contracted
on device through an explicit-argument jit and accumulated in numpy f64.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

F64 = jnp.float64


def backward_diff_bank_3d(G, n):
    """(D-x + D-y + D-z) G on the interior grid with ghost zeros, applied
    column-wise to the bank G (n_i, R) (numpy or jax), interior flat order
    i*(n-2)^2 + j*(n-2) + k."""
    xp = jnp if isinstance(G, jnp.ndarray) else np
    ni = n - 2
    dx = 1.0 / (n - 1)
    R = G.shape[1]
    G4 = G.reshape(ni, ni, ni, R)
    z = xp.zeros((1, ni, ni, R), dtype=G.dtype)
    Gx = xp.concatenate([z, G4[:-1]], axis=0)
    z = xp.zeros((ni, 1, ni, R), dtype=G.dtype)
    Gy = xp.concatenate([z, G4[:, :-1]], axis=1)
    z = xp.zeros((ni, ni, 1, R), dtype=G.dtype)
    Gz = xp.concatenate([z, G4[:, :, :-1]], axis=2)
    return ((3.0 * G4 - Gx - Gy - Gz) / dx).reshape(ni ** 3, R)


@jax.jit
def _chunk_T(P, Gc, Dc):
    """(chunk, M), (chunk, R), (chunk, R) -> (M, R, R) partial sum, f64."""
    c, R = Gc.shape
    prod = (Gc[:, :, None] * Dc[:, None, :]).reshape(c, R * R)
    return (P.T @ prod).reshape(P.shape[1], R, R)


def build_T(Phi, G, n, chunk=4096, reverse=False):
    """T[i,j,k] = sum_x Phi[x,i] G[x,j] (DG)[x,k] over the interior, f64,
    accumulated over x-blocks of `chunk` rows (reverse order if `reverse`).
    Phi (n_i, M), G (n_i, R) numpy or jax.  Returns numpy (M, R, R)."""
    Phi = jnp.asarray(Phi, dtype=F64)
    G = jnp.asarray(G, dtype=F64)
    DG = backward_diff_bank_3d(G, n)
    n_i, R = G.shape
    M = Phi.shape[1]
    T = np.zeros((M, R, R), dtype=np.float64)
    starts = list(range(0, n_i, chunk))
    if reverse:
        starts = starts[::-1]
    for s in starts:
        e = min(s + chunk, n_i)
        T += np.asarray(_chunk_T(Phi[s:e], G[s:e], DG[s:e]))
    return T


def symmetrize(T):
    return T + T.swapaxes(1, 2)


def q_of(Q, h):
    v = Q @ h
    return 0.5 * (v @ h)


def dq_dh(Q, h):
    return Q @ h
