"""Independent NumPy evaluation of the frozen coordinate-network bank and its exact spatial gradient (no JAX).

features(x) = out_scale * bc(x) * MLP([sin 2 pi x B, cos 2 pi x B]),  bc = 16 x (1-x) y (1-y),  SiLU hidden layers.
The gradient is propagated by hand (forward mode, chain + product rule). Used by the audit to recompute rho and the
off-mesh tested advection without the driver's code path.
"""
import pickle

import numpy as np


def load(ckpt):
    ck = pickle.load(open(ckpt, 'rb'))
    p = ck['params']
    return dict(B=np.asarray(p['B'], float), g=[(np.asarray(w, float), np.asarray(b, float)) for w, b in p['g']],
                out_scale=float(np.asarray(p['out_scale'])))


def _silu(h):
    s = 1. / (1. + np.exp(-h))
    return h * s, s * (1. + h * (1. - s))


def features_grad(P, X):
    """Values (m, R) and d/dx, d/dy (m, R) at points X (m, 2)."""
    X = np.asarray(X, float)
    ang = 2. * np.pi * X @ P['B']
    s, c = np.sin(ang), np.cos(ang)
    a = np.concatenate([s, c], 1)
    da = [np.concatenate([c * (2. * np.pi * P['B'][j]), -s * (2. * np.pi * P['B'][j])], 1) for j in (0, 1)]
    for W, b in P['g'][:-1]:
        h = a @ W + b
        a, ds = _silu(h)
        da = [ds * (d @ W) for d in da]
    W, b = P['g'][-1]
    o = a @ W + b
    do = [d @ W for d in da]
    x, y = X[:, 0:1], X[:, 1:2]
    bc = 16. * x * (1 - x) * y * (1 - y)
    dbc = [16. * (1 - 2 * x) * y * (1 - y), 16. * x * (1 - x) * (1 - 2 * y)]
    k = P['out_scale']
    return k * bc * o, k * (dbc[0] * o + bc * do[0]), k * (dbc[1] * o + bc * do[1])
