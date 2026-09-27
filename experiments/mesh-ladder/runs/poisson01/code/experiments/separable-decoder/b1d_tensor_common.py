"""Precomputed quadratic advection tensor for the 1D Burgers weak residual
(2026-08-29, sample-free nonlinear residual experiment).

The FOM advection is the non-conservative sign-upwind N(u) = u * u_x with
    u_x = (u[x] - u[x-1])/dx   where u[x] > 0     (backward, ghost zero at x=0)
    u_x = (u[x+1] - u[x])/dx   otherwise          (forward, ghost zero at x=n-1)
(see b1d_common.upwind_adv_field_1d).  On a field that is positive everywhere
the stencil is the fixed backward difference, and the projected term
Phi^T N(G h) is then an exact quadratic in the head output h:

    Phi^T (u . D^- u) = sum_{j,k} h_j h_k T[:, j, k],
    T[i, j, k] = sum_x Phi[x, i] G[x, j] (D^- G)[x, k].

T is NOT symmetric in (j, k) (bank vs differenced bank), so the stored table
is Q = T + T.swapaxes(1, 2):  q(h) = 0.5 * sum_j (Q h)[:, j] h_j  (= h^T T h)
and dq/dh = Q h  (M x R), hence dq/dz = (Q h) @ dh/dz.

Everything is built in f64 and blocked over x (never an n x R^2 temporary).
The build is deterministic up to floating-point summation order; two chunk
orders are compared in the audit (gate TB).
"""
from __future__ import annotations

import numpy as np


def backward_diff_bank(G, dx):
    """(D^- G)[x] = (G[x] - G[x-1]) / dx on the interior grid with G[-1] = 0
    (the ghost zero at the left wall), exactly upwind_adv_field_1d's backward
    branch applied column-wise to the bank."""
    G = np.asarray(G, dtype=np.float64)
    Gm = np.concatenate([np.zeros((1, G.shape[1])), G[:-1]], axis=0)
    return (G - Gm) / dx


def build_T(Phi, G, dx, chunk=256, reverse=False):
    """T[i,j,k] = sum_x Phi[x,i] G[x,j] (D^-G)[x,k], f64, accumulated over
    x-chunks of size `chunk` (in reverse chunk order if `reverse`).  Returns
    (M, R, R)."""
    Phi = np.asarray(Phi, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    n, R = G.shape
    M = Phi.shape[1]
    DG = backward_diff_bank(G, dx)
    T = np.zeros((M, R, R), dtype=np.float64)
    starts = list(range(0, n, chunk))
    if reverse:
        starts = starts[::-1]
    for s in starts:
        e = min(s + chunk, n)
        prod = (G[s:e, :, None] * DG[s:e, None, :]).reshape(e - s, R * R)
        T += (Phi[s:e].T @ prod).reshape(M, R, R)
    return T


def symmetrize(T):
    """Q = T + T^(j<->k); q(h) = 0.5 h^T Q h = h^T T h, dq/dh = Q h."""
    return T + T.swapaxes(1, 2)


def q_of(Q, h):
    """Advection term from the symmetrized table (numpy, f64)."""
    v = Q @ h                                   # (M, R)
    return 0.5 * (v @ h)


def dq_dh(Q, h):
    return Q @ h                                # (M, R)
