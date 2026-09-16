"""Independent pure-NumPy/SciPy re-implementation used by both audits.

Imports neither JAX nor any driver module: the separable decoder, the discrete
sine solve and the exact reference are rewritten here from the equations so that
an audit can disagree with the driver.
"""
import pickle

import numpy as np
from scipy.fft import dstn


# ------------------------------------------------------------------ decoder ---

def load(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['params'], np.asarray(d['Z_tr']), d['cfg']


def silu(x):
    return x / (1.0 + np.exp(-x))


def mlp(layers, x):
    for w, b in layers[:-1]:
        x = silu(x @ np.asarray(w) + np.asarray(b))
    w, b = layers[-1]
    return x @ np.asarray(w) + np.asarray(b)


def bc_poly(xy):
    return 16.0 * xy[:, 0] * (1.0 - xy[:, 0]) * xy[:, 1] * (1.0 - xy[:, 1])


def features(params, xy):
    ang = 2.0 * np.pi * (xy @ np.asarray(params['B']))
    ff = np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)
    return (float(params['out_scale']) * bc_poly(xy))[:, None] * mlp(params['g'], ff)


def head(params, z):
    z = np.atleast_2d(np.asarray(z))
    return mlp(params['h'], z) + z @ np.asarray(params['h_lin'])


def coords(intervals):
    p = np.arange(1, intervals) / intervals
    xx, yy = np.meshgrid(p, p, indexing='ij')
    return np.column_stack((xx.ravel(), yy.ravel()))


# ------------------------------------------------------------------- physics --

def eigenvalues(intervals):
    p = np.arange(1, intervals)
    l = 4.0 * intervals ** 2 * np.sin(np.pi * p / (2 * intervals)) ** 2
    return l[:, None] + l[None, :]


def source_interior(intervals, param):
    cx, cy, w, a = param
    x = np.linspace(0.0, 1.0, intervals + 1)
    X, Y = np.meshgrid(x, x, indexing='ij')
    Xi, Yi = X[1:-1, 1:-1], Y[1:-1, 1:-1]
    return a * np.exp(-((Xi - cx) ** 2 + (Yi - cy) ** 2) / (2 * w ** 2))


def solve(intervals, param):
    """Exact zero-Dirichlet FD solution on the `intervals`-interval mesh."""
    f = source_interior(intervals, param)
    c = dstn(f, type=1, norm='ortho', workers=1)
    return np.pad(dstn(c / eigenvalues(intervals), type=1, norm='ortho', workers=1), 1)


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(b))


def restrict(field, source_intervals, target_intervals):
    assert source_intervals % target_intervals == 0
    s = source_intervals // target_intervals
    return np.asarray(field)[::s, ::s]


def source_params(seed, count):
    rng = np.random.default_rng(seed)
    cx = rng.uniform(0.15, 0.85, count)
    cy = rng.uniform(0.15, 0.85, count)
    w = np.exp(rng.uniform(np.log(0.02), np.log(0.1), count))
    a = rng.uniform(0.5, 2.0, count)
    return np.column_stack((cx, cy, w, a))
