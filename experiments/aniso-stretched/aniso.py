"""Q1 finite elements for -div(K grad u) = f on a stretched tensor grid (see PROTOCOL.md)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

SIGMA = 0.1
AMP = 10.0
RANGES = {
    "A": dict(theta=(15.0, 75.0), loge=(-4.0, -2.0), c=(0.3, 0.7)),
    "B": dict(theta=(10.0, 40.0), loge=(-4.0, -2.0), c=(0.3, 0.7)),
}
SEEDS = {"dev": 1101, "test": 1102, "train": 1103}


def sample_params(family: str, cohort: str, n: int) -> np.ndarray:
    """Rows (theta_deg, log10 eps, cx, cy)."""
    r = RANGES[family]
    rng = np.random.default_rng(SEEDS[cohort] + (0 if family == "A" else 7919))
    u = rng.random((n, 4))
    lo = np.array([r["theta"][0], r["loge"][0], r["c"][0], r["c"][0]])
    hi = np.array([r["theta"][1], r["loge"][1], r["c"][1], r["c"][1]])
    return lo + u * (hi - lo)


def centre_param(family: str) -> np.ndarray:
    r = RANGES[family]
    return np.array([np.mean(r["theta"]), np.mean(r["loge"]), 0.5, 0.5])


def nodes(N: int, beta: float) -> np.ndarray:
    xi = np.linspace(0.0, 1.0, N + 1)
    if beta == 0:
        return xi
    x = 0.5 * (1.0 + np.tanh(beta * (2 * xi - 1)) / np.tanh(beta))
    x[0], x[-1] = 0.0, 1.0
    return x


def tensor(family: str, mu, xc, yc):
    """K components (a, b, c) = (Kxx, Kxy, Kyy) at element centres, arrays (Nx, Ny)."""
    th, loge = np.deg2rad(mu[0]), mu[1]
    eps = 10.0 ** loge
    if family == "A":
        t = np.full((xc.size, yc.size), th)
    else:
        t = th * np.sin(np.pi * xc)[:, None] * np.sin(np.pi * yc)[None, :]
    c2, s2 = np.cos(2 * t), np.sin(2 * t)
    a = 0.5 * (1 + eps) + 0.5 * (1 - eps) * c2
    c = 0.5 * (1 + eps) - 0.5 * (1 - eps) * c2
    b = 0.5 * (1 - eps) * s2
    return a, b, c


def source(mu, x, y):
    return 1.0 + AMP * np.exp(-((x[:, None] - mu[2]) ** 2 + (y[None, :] - mu[3]) ** 2) / (2 * SIGMA**2))


def mass1d(x):
    h = np.diff(x)
    n = x.size
    d = np.zeros(n)
    d[:-1] += h / 3
    d[1:] += h / 3
    return sp.diags([h / 6, d, h / 6], [-1, 0, 1], format="csr")


def lumped1d(x):
    h = np.diff(x)
    w = np.zeros(x.size)
    w[:-1] += h / 2
    w[1:] += h / 2
    return w


_S = np.array([[1.0, -1.0], [-1.0, 1.0]])
_M = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0
_D = np.array([[-0.5, -0.5], [0.5, 0.5]])  # D[p,p'] = int phi_p' phi_p'
_LXX = np.kron(_S, _M)   # times hy/hx
_LYY = np.kron(_M, _S)   # times hx/hy
_LXY = np.kron(_D.T, _D) + np.kron(_D, _D.T)


def assemble(x, y, a, b, c):
    """Interior stiffness matrix (CSR) for element-wise (a,b,c). Node (i,j) -> i*(Ny+1)+j."""
    Nx, Ny = x.size - 1, y.size - 1
    hx, hy = np.diff(x)[:, None], np.diff(y)[None, :]
    ke = ((a * hy / hx)[..., None, None] * _LXX + (c * hx / hy)[..., None, None] * _LYY
          + b[..., None, None] * _LXY)                       # (Nx,Ny,4,4)
    i = np.arange(Nx)[:, None]
    j = np.arange(Ny)[None, :]
    loc = np.stack([(i + p) * (Ny + 1) + (j + q) for p in (0, 1) for q in (0, 1)], axis=-1)  # (Nx,Ny,4)
    rows = np.broadcast_to(loc[..., :, None], ke.shape).ravel()
    cols = np.broadcast_to(loc[..., None, :], ke.shape).ravel()
    n = (Nx + 1) * (Ny + 1)
    A = sp.coo_matrix((ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    inter = np.zeros((Nx + 1, Ny + 1), bool)
    inter[1:-1, 1:-1] = True
    idx = np.flatnonzero(inter.ravel())
    return A[idx][:, idx].tocsr()


def load(x, y, mu):
    F = source(mu, x, y)
    B = mass1d(x) @ F @ mass1d(y)          # symmetric 1D masses
    return np.ascontiguousarray(B[1:-1, 1:-1]).ravel()


def problem(family, mu, N, bx, by):
    x, y = nodes(N, bx), nodes(N, by)
    xc, yc = 0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1])
    A = assemble(x, y, *tensor(family, mu, xc, yc))
    return x, y, A, load(x, y, mu)


def interp1d(xc, xe):
    """Sparse linear interpolation matrix from nodes xc to points xe."""
    k = np.clip(np.searchsorted(xc, xe, side="right") - 1, 0, xc.size - 2)
    t = (xe - xc[k]) / (xc[k + 1] - xc[k])
    r = np.arange(xe.size)
    return sp.csr_matrix((np.r_[1 - t, t], (np.r_[r, r], np.r_[k, k + 1])), shape=(xe.size, xc.size))


def to_full(u, N):
    U = np.zeros((N + 1, N + 1))
    U[1:-1, 1:-1] = u.reshape(N - 1, N - 1)
    return U


class Evaluator:
    """Interpolate a coarse nodal field to the evaluation grid and score it."""

    def __init__(self, N=2048, beta=2.5):
        self.x = nodes(N, beta)
        self.w = lumped1d(self.x)
        self._P = {}

    def P(self, N, b):
        key = (N, b)
        if key not in self._P:
            self._P[key] = interp1d(nodes(N, b), self.x)
        return self._P[key]

    def interp(self, U, N, bx, by):
        return np.asarray(self.P(N, bx) @ (self.P(N, by) @ U.T).T)

    def errors(self, Ue, Uref):
        d = Ue - Uref
        W = self.w[:, None] * self.w[None, :]
        return dict(
            l2w=float(np.sqrt((W * d * d).sum() / (W * Uref * Uref).sum())),
            l2n=float(np.linalg.norm(d) / np.linalg.norm(Uref)),
            linf=float(np.abs(d).max() / np.abs(Uref).max()),
        )
