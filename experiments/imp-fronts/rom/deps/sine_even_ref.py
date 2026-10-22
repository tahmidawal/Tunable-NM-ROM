"""imp-fronts arm (c) 'sine_even': the ONE reference implementation of the bank formula (DESIGN Amendments 19-20),
NumPy only, for every consumer to test against (Stage A/B/rotation sep_common copies in this tree; the ROM-side
sep_common and the NumPy evaluator npbank with its hand gradient).

    u_r(x, y) = s * sin(pi x) sin(pi y) * g_r( cos(pi Kx_1 x), ..., cos(pi Kx_n x), cos(pi Ky_1 y), ..., cos(pi Ky_n y) )

params (the checkpoint's `params` dict): Kc (2, n_ff) fixed integer frequencies (Kc = |round(2 B0)| of the seed-1 init
draw; stored as float64, rounded on use), g = [(W, b), ...] the MLP (SiLU on hidden layers, linear output),
out_scale = s. params['B'] is present for format compatibility and is NOT used. Dispatch rule shared by every
consumer: "Kc" in params  <=>  cfg['arch'] == 'sine_even'; otherwise the incumbent 16x(1-x)y(1-y) * MLP(sin/cos 2 pi xB).

    features(params, xy)       -> (P, R) values
    features_grad(params, xy)  -> (G, Gx, Gy), each (P, R): values and exact x, y derivatives (forward-mode by hand)
"""
from __future__ import annotations

import numpy as np


def _silu(z):
    return z / (1.0 + np.exp(-z))


def _dsilu(z):
    s = 1.0 / (1.0 + np.exp(-z))
    return s * (1.0 + z * (1.0 - s))


def _inputs(params, xy):
    K = np.round(np.asarray(params['Kc'], dtype=np.float64))
    x, y = xy[:, 0:1], xy[:, 1:2]
    ff = np.concatenate([np.cos(np.pi * x * K[0]), np.cos(np.pi * y * K[1])], axis=1)
    zx, zy = np.zeros_like(x * K[0]), np.zeros_like(y * K[1])
    dfx = np.concatenate([-np.pi * K[0] * np.sin(np.pi * x * K[0]), zy], axis=1)
    dfy = np.concatenate([zx, -np.pi * K[1] * np.sin(np.pi * y * K[1])], axis=1)
    return ff, dfx, dfy


def _mlp_jvp(g, a, tangents):
    ts = list(tangents)
    for i, (w, b) in enumerate(g):
        w, b = np.asarray(w, np.float64), np.asarray(b, np.float64)
        z = a @ w + b
        tz = [t @ w for t in ts]
        if i < len(g) - 1:
            d = _dsilu(z)
            a, ts = _silu(z), [d * t for t in tz]
        else:
            a, ts = z, tz
    return a, ts


def features(params, xy):
    xy = np.asarray(xy, np.float64)
    ff, _, _ = _inputs(params, xy)
    gv, _ = _mlp_jvp(params['g'], ff, [])
    env = np.sin(np.pi * xy[:, 0]) * np.sin(np.pi * xy[:, 1])
    return float(params['out_scale']) * env[:, None] * gv


def features_grad(params, xy):
    xy = np.asarray(xy, np.float64)
    ff, dfx, dfy = _inputs(params, xy)
    gv, (gx, gy) = _mlp_jvp(params['g'], ff, [dfx, dfy])
    sx, sy = np.sin(np.pi * xy[:, 0]), np.sin(np.pi * xy[:, 1])
    cx, cy = np.cos(np.pi * xy[:, 0]), np.cos(np.pi * xy[:, 1])
    s = float(params['out_scale'])
    env = sx * sy
    G = s * env[:, None] * gv
    Gx = s * ((np.pi * cx * sy)[:, None] * gv + env[:, None] * gx)
    Gy = s * ((np.pi * sx * cy)[:, None] * gv + env[:, None] * gy)
    return G, Gx, Gy
