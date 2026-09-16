"""Shared NumPy-only pieces of the head-refinement audits. No JAX, no GPU."""
import numpy as np


def split_theta(theta, shapes):
    arrs, off = [], 0
    for s in shapes:
        k = int(np.prod(s))
        arrs.append(np.asarray(theta[off:off + k], dtype=float).reshape(s))
        off += k
    assert off == len(theta), (off, len(theta))
    return arrs


def head_np(theta, z, shapes):
    """`sep_common.head` in pure NumPy: a SiLU MLP plus a linear skip."""
    arrs = split_theta(theta, shapes)
    mlp, h_lin = arrs[:-1], arrs[-1]
    x = np.asarray(z, dtype=float)
    for i in range(0, len(mlp) - 2, 2):
        t = x @ mlp[i] + mlp[i + 1]
        x = t / (1.0 + np.exp(-t))
    x = x @ mlp[-2] + mlp[-1]
    return x + np.asarray(z, dtype=float) @ h_lin


def median(x):
    return float(np.median(np.asarray(x, dtype=float)))
