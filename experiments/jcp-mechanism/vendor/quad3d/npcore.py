"""quadrature-burgers3d: independent NumPy/SciPy primitives for refine_q.py and audit_q.py (no JAX anywhere).

Re-implemented from the definitions, not imported from the JAX code: the bank and its (1,1,1) forward derivative,
the mode order and shell completion, the orthonormal 3D DST-I (scipy), the sign-upwind stencil, the 65-node lattice.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scipy.fft


def load_bank(model_dir):
    b = pickle.loads((Path(model_dir) / 'bank.pkl').read_bytes())
    to_np = lambda t: [(np.asarray(w), np.asarray(bb)) for w, bb in t]
    p = dict(freq=np.asarray(b['params']['freq']), net=to_np(b['params']['net']), scale=float(b['params']['scale']))
    return p, np.asarray(b['rotation'])


def _sig(a):
    return 0.5 * (1.0 + np.tanh(0.5 * a))


def bank_np(p, T, x, deriv=False, chunk=1 << 14):
    """G_hat(x) = scale * mu(x) * MLP(RFF(x)) @ T, and optionally (d/dx + d/dy + d/dz) G_hat(x) by forward mode."""
    outs, douts = [], []
    for s in range(0, len(x), chunk):
        xx = x[s:s + chunk]
        ang = 2 * np.pi * (xx @ p['freq'])
        dang = 2 * np.pi * (np.ones(3) @ p['freq'])[None, :]
        h = np.concatenate((np.sin(ang), np.cos(ang)), -1)
        dh = np.concatenate((np.cos(ang) * dang, -np.sin(ang) * dang), -1)
        for w, b in p['net'][:-1]:
            a = h @ w + b
            da = dh @ w
            sg = _sig(a)
            h = a * sg
            dh = (sg * (1 + a * (1 - sg))) * da
        w, b = p['net'][-1]
        g = h @ w + b
        dg = dh @ w
        q = xx * (1 - xx)
        mu = 64.0 * np.prod(q, axis=-1)
        dq = 1 - 2 * xx
        dmu = 64.0 * (dq[:, 0] * q[:, 1] * q[:, 2] + q[:, 0] * dq[:, 1] * q[:, 2] + q[:, 0] * q[:, 1] * dq[:, 2])
        outs.append((p['scale'] * mu)[:, None] * g @ T)
        if deriv:
            douts.append(p['scale'] * (dmu[:, None] * g + mu[:, None] * dg) @ T)
    G = np.concatenate(outs, 0)
    return (G, np.concatenate(douts, 0)) if deriv else G


def lattice65_coords():
    ax = np.arange(1, 64) / 64.0
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)


def mesh_coords(n):
    ax = np.arange(1, n - 1) / (n - 1)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)


def modes(n, M_req):
    """M = M_req completed to the end of its discrete-eigenvalue shell; (kx, ky, kz) 1-based and lambda."""
    p = np.arange(1, n - 1)
    l1 = 4.0 * (n - 1) ** 2 * np.sin(np.pi * p / (2 * (n - 1))) ** 2
    lam = (l1[:, None, None] + l1[None, :, None] + l1[None, None, :]).ravel()
    order = np.argsort(lam, kind='stable')
    M = M_req
    lm = lam[order[M - 1]]
    while M < lam.size and abs(lam[order[M]] - lm) <= 1e-9 * lm:
        M += 1
    ni = n - 2
    k = np.stack(np.unravel_index(order[:M], (ni, ni, ni)), 1) + 1
    return k, lam[order[:M]]


def upwind_np(u, n):
    ni = n - 2
    v = u.reshape(ni, ni, ni)
    pd = np.pad(v, 1)
    s = np.zeros_like(v)
    for ax in range(3):
        lo = [slice(1, -1)] * 3
        hi = [slice(1, -1)] * 3
        lo[ax] = slice(None, -2)
        hi[ax] = slice(2, None)
        s += np.where(v > 0, v - pd[tuple(lo)], pd[tuple(hi)] - v)
    return (v * s * (n - 1)).ravel()


def tested_np(F, n, k):
    """Phi^T F for a field F on the interior nodes: orthonormal DST-I (scipy) on three axes, at modes k."""
    ni = n - 2
    d = scipy.fft.dstn(F.reshape(ni, ni, ni), type=1, norm='ortho')
    return d[k[:, 0] - 1, k[:, 1] - 1, k[:, 2] - 1]


def offmesh_adv_np(G, D, X, w, n, k, Cs):
    """N^{3/2} sum_q w_q psi_m(x_q) (G c)(D c) for coefficient rows Cs."""
    psi = 2.0 ** 1.5 * np.ones((len(X), len(k)))
    for a in range(3):
        psi *= np.sin(np.pi * np.outer(X[:, a], k[:, a]))
    P = (n - 1) ** 1.5 * w[:, None] * psi
    return ((Cs @ G.T) * (Cs @ D.T)) @ P


def worst_evolved(F, R):
    """max over t >= 1 of ||F(t) - R(t)|| / ||R(0)|| (rows = output times)."""
    e = np.linalg.norm(F - R, axis=1) / np.linalg.norm(R[0])
    return e
