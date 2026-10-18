"""Independent NumPy evaluation of a head (no JAX, no driver code path) for the jcp-heads offline checks.

  * head(z): the 512 x 2 SiLU MLP plus the linear skip (sep_common.head without latent Fourier features), folded into the
    rotated basis exactly as bkfast.fold_head does: c = h(z) @ Lrot[:512].T;
  * coefficients of saved latent states, the scoring-lattice decode (npcheck.decode257 at r = 512);
  * the LSPG LMM residual in z and its stationarity ratio gamma with a central finite-difference Jacobian in z (an
    independent route: neither the solver's analytic Jacobian nor JAX autodiff); used for the budget-0 control and spot
    checks, where gamma is far from the 1e-3 bar.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

import npcheck as NC

HERE = Path(__file__).resolve().parent
ROT = HERE / 'vendor' / 'quad2d' / 'inputs' / 'rotation_R512.npz'


def load_head(path):
    """(layers [(W, b)], h_lin) of a head checkpoint: compact npz or full pkl."""
    path = Path(path)
    if path.suffix == '.npz':
        z = np.load(path)
        n = int(z['n_layers'])
        return [(np.asarray(z[f'h{i}_w'], float), np.asarray(z[f'h{i}_b'], float)) for i in range(n)], np.asarray(z['h_lin'], float)
    ck = pickle.load(open(path, 'rb'))
    p = ck['params']
    return [(np.asarray(w, float), np.asarray(b, float)) for w, b in p['h']], np.asarray(p['h_lin'], float)


def silu(x):
    return x / (1. + np.exp(-x))


class Head:
    def __init__(self, path):
        self.layers, self.h_lin = load_head(path)
        self.Lrot = np.load(ROT)['L']
        self.K = self.h_lin.shape[0]

    def h(self, z):
        z = np.asarray(z, float)
        x = z
        for W, b in self.layers[:-1]:
            x = silu(x @ W + b)
        W, b = self.layers[-1]
        return x @ W + b + z @ self.h_lin

    def c(self, z):
        """Rotated-basis coefficients (bkfast.fold_head: output L[:R'] h(z))."""
        return self.h(z) @ self.Lrot[:512].T


def residual(head, z, hist_c, A, lam, t, co, dt, nu):
    """LSPG LMM residual r(z) with history coefficients (c_n, c_{n-1}) and rule tables t (npcheck.tables)."""
    a0, a1, a2, b0, b1 = co
    cn, cm = hist_c
    c = head.c(z)
    Ac = A @ c
    Fn = NC.N(cn, t) + nu * lam * (A @ cn) if b1 != 0 else 0.
    R = a0 * Ac + a1 * (A @ cn) + a2 * (A @ cm) + dt * (b0 * (NC.N(c, t) + nu * lam * Ac) + b1 * Fn)
    return R / (a0 + dt * b0 * nu * lam)


def gammas(head, states, A, lam, t, scheme, dt, nu, eps=1e-6):
    """gamma_k of every step from a saved latent trajectory, central-difference Jacobian in z."""
    co_s = NC.SCHEMES[scheme]
    out = []
    C = [head.c(s) for s in states]
    for k in range(len(states) - 1):
        co = NC.BE_CO if k < co_s[5] else co_s[:5]
        hist = (C[k], C[max(k - 1, 0)])
        zn = np.asarray(states[k + 1], float)
        r = residual(head, zn, hist, A, lam, t, co, dt, nu)
        J = np.empty((len(r), len(zn)))
        for i in range(len(zn)):
            e = np.zeros_like(zn)
            e[i] = eps * (1. + abs(zn[i]))
            J[:, i] = (residual(head, zn + e, hist, A, lam, t, co, dt, nu) - residual(head, zn - e, hist, A, lam, t, co, dt, nu)) / (2 * e[i])
        out.append(float(np.linalg.norm(J.T @ r) / (np.linalg.norm(J) * np.linalg.norm(r) + 1e-300)))
    return np.array(out)
