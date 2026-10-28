"""Precomputed quadratic advection tensor for the 2D Burgers weak residual
(2026-08-29, port of b1d_tensor_common.py; branch exp/2026-08-29-b2d-tensor).

The FOM advection is the non-conservative sign-upwind
    N(u) = u * (u_x + u_y),
    u_x = (u[i,j] - u[i-1,j])/dx  where u[i,j] > 0,  else (u[i+1,j] - u[i,j])/dx
    u_y = (u[i,j] - u[i,j-1])/dx  where u[i,j] > 0,  else (u[i,j+1] - u[i,j])/dx
with ghost zeros on all four walls -- exactly `blat_common.upwind_adv_field`
(both axes switch on the SAME centre value; interior flat index i*(n-2)+j
with i the x index).  On a field that is positive everywhere the stencil is
the fixed backward difference on BOTH axes, and the projected term is then
an exact quadratic in the head output h:

    Phi^T (u . (D^-_x u + D^-_y u)) = sum_{j,k} h_j h_k T[:, j, k],
    T[i, j, k] = sum_x Phi[x, i] G[x, j] (DG)[x, k],   DG = D^-_x G + D^-_y G.

ONE tensor covers both axes.  T is NOT symmetric in (j, k) (bank vs
differenced bank), so the stored table is Q = T + T.swapaxes(1, 2):
    q(h) = 0.5 * sum_j (Q h)[:, j] h_j  (= h^T T h),   dq/dh = Q h  (M x R),
and dq/dz = (Q h) @ dh/dz, which is what forward-mode AD computes.

Everything is f64 and BLOCKED over x: the n x R^2 intermediate is never
materialised for the whole grid (at N=1024, n ~ 1.04e6 and R = 64 it would be
34 GB); each block holds (chunk, R^2).  The build is deterministic up to
floating-point summation order; two chunkings are compared (gate TB).
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

F64 = jnp.float64


def backward_diff_bank_2d(G, n):
    """(D^-_x G + D^-_y G) on the interior grid with ghost zeros on the walls:
    `upwind_adv_field`'s c>0 branch applied column-wise to the bank.  G is
    (n_i^2, R) numpy or jax, interior flat order i*(n-2)+j."""
    ni = n - 2
    dx = 1.0 / (n - 1)
    xp = jnp if isinstance(G, jnp.ndarray) else np
    R = G.shape[1]
    Gg = G.reshape(ni, ni, R)
    Gx = xp.concatenate([xp.zeros((1, ni, R), dtype=Gg.dtype), Gg[:-1]], axis=0)
    Gy = xp.concatenate([xp.zeros((ni, 1, R), dtype=Gg.dtype), Gg[:, :-1]], axis=1)
    return ((Gg - Gx) + (Gg - Gy)).reshape(ni * ni, R) / dx


@jax.jit
def _chunk_T(P, Gc, Dc):
    """(chunk, M), (chunk, R), (chunk, R) -> (M, R, R) partial sum, f64."""
    prod = (Gc[:, :, None] * Dc[:, None, :]).reshape(Gc.shape[0], -1)
    return (P.T @ prod).reshape(P.shape[1], Gc.shape[1], Dc.shape[1])


def build_T(Phi, G, n, chunk=16384, reverse=False, device=True):
    """T[i,j,k] = sum_x Phi[x,i] G[x,j] (DG)[x,k] over the interior, f64,
    accumulated over x-chunks of `chunk` rows (reverse order if `reverse`).
    Phi (n_i2, M), G (n_i2, R) -- numpy or device arrays.  Returns numpy
    (M, R, R).  With device=True each block is contracted on the accelerator
    (explicit arguments, nothing captured); the accumulation is in numpy f64."""
    Phi = jnp.asarray(Phi, dtype=F64) if device else np.asarray(Phi, np.float64)
    G = jnp.asarray(G, dtype=F64) if device else np.asarray(G, np.float64)
    DG = backward_diff_bank_2d(G, n)
    n_pts, R = G.shape
    M = Phi.shape[1]
    T = np.zeros((M, R, R), dtype=np.float64)
    starts = list(range(0, n_pts, chunk))
    if reverse:
        starts = starts[::-1]
    for s in starts:
        e = min(s + chunk, n_pts)
        if device:
            T += np.asarray(_chunk_T(Phi[s:e], G[s:e], DG[s:e]))
        else:
            prod = (G[s:e, :, None] * DG[s:e, None, :]).reshape(e - s, R * R)
            T += (Phi[s:e].T @ prod).reshape(M, R, R)
    return T


def symmetrize(T):
    """Q = T + T^(j<->k); q(h) = 0.5 h^T Q h = h^T T h, dq/dh = Q h."""
    return T + T.swapaxes(1, 2)


def q_of(Q, h):
    """Advection term from the symmetrised table (works for numpy and jnp)."""
    v = jnp.einsum("ijk,k->ij", Q, h) if isinstance(Q, jnp.ndarray) else Q @ h
    return 0.5 * (v @ h)


def dq_dh(Q, h):
    return Q @ h                                # (M, R)
