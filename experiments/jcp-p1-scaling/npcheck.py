"""Independent NumPy evaluator (no JAX, no driver code path) for the jcp-p1-scaling acceptance checks (DESIGN 5, 6, 8).

  * Gauss-Legendre tensor rules from numpy.polynomial.legendre (not hari_quadrature);
  * bank values and exact gradients from the vendored NumPy bank (`npbank`, hand-propagated forward mode);
  * the tested advection N(c) = Psi^T[(Gq c)(Gs c)], its Jacobian, the LSPG LMM residual and its Jacobian (written out
    by hand), the stationarity ratio gamma, the successor-rule rho, and the decode at the 257^2 scoring lattice.
A and lam (mesh operators) are inputs (saved by the job); everything rule- and solver-dependent is recomputed here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor' / 'quad2d'))
import npbank as NB  # noqa: E402

ROOT = HERE.parents[1]
CKPT = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
ROT = HERE / 'vendor' / 'quad2d' / 'inputs' / 'rotation_R512.npz'
SCHEMES = dict(BE=(1., -1., 0., 1., 0., 0), CNR=(1., -1., 0., .5, .5, 2))
BE_CO = (1., -1., 0., 1., 0.)
_P, _T = None, None


def bank():
    global _P, _T
    if _P is None:
        _P = NB.load(CKPT)
        _T = np.load(ROT)['T']
    return _P, _T


def gauss_rule(p):
    t, w = np.polynomial.legendre.leggauss(p)
    x, w = (t + 1) / 2, w / 2
    X = np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2)
    return X, np.outer(w, w).ravel()


def values_grads(X, r, chunk=4096):
    P, T = bank()
    out = [[], [], []]
    for s in range(0, len(X), chunk):
        v, gx, gy = NB.features_grad(P, X[s:s + chunk])
        for i, a in enumerate((v, gx, gy)):
            out[i].append(a @ T[:, :r])
    return [np.concatenate(o) for o in out]


def tables(X, w, r, kx, ky, L):
    Gq, Gx, Gy = values_grads(X, r)
    x, y = X[:, 0:1], X[:, 1:2]
    psi = 2. * np.sin(np.asarray(kx, float)[None] * np.pi * x) * np.sin(np.asarray(ky, float)[None] * np.pi * y)
    return dict(Gq=Gq, Gs=Gx + Gy, Psi=L * w[:, None] * psi)


def N(c, t):
    return t['Psi'].T @ ((t['Gq'] @ c) * (t['Gs'] @ c))


def dN(c, t):
    u, s = t['Gq'] @ c, t['Gs'] @ c
    return t['Psi'].T @ (s[:, None] * t['Gq'] + u[:, None] * t['Gs'])


def rho(c, t, tp):
    a, b = N(c, t), N(c, tp)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def gammas(states, A, lam, t, scheme, dt, nu):
    """gamma_k of every step from the saved internal trajectory (row k+1 = state after step k), written out by hand."""
    co_s = SCHEMES[scheme]
    out = []
    for k in range(len(states) - 1):
        a0, a1, a2, b0, b1 = BE_CO if k < co_s[5] else co_s[:5]
        ck, cm, cn = states[k], states[max(k - 1, 0)], states[k + 1]
        D = a0 + dt * b0 * nu * lam
        Fn = N(ck, t) + nu * lam * (A @ ck) if b1 != 0 else 0.
        Acn = A @ cn
        r = (a0 * Acn + a1 * (A @ ck) + a2 * (A @ cm) + dt * (b0 * (N(cn, t) + nu * lam * Acn) + b1 * Fn)) / D
        J = (a0 * A + dt * b0 * (dN(cn, t) + (nu * lam)[:, None] * A)) / D[:, None]
        out.append(float(np.linalg.norm(J.T @ r) / (np.linalg.norm(J) * np.linalg.norm(r) + 1e-300)))
    return np.array(out)


_G257 = {}


def decode257(W, r):
    """Fields (n, 257, 257) at x = i/256 (zero walls) from coefficients W (n, r): the bank at the scoring lattice."""
    if r not in _G257:
        x = np.arange(257) / 256.
        X = np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2)
        P, T = bank()
        V = np.concatenate([NB.features_grad(P, X[s:s + 8192])[0] for s in range(0, len(X), 8192)])
        V[(X[:, 0] == 0) | (X[:, 0] == 1) | (X[:, 1] == 0) | (X[:, 1] == 1)] = 0.
        _G257[r] = V @ T[:, :r]
    return (np.asarray(W) @ _G257[r].T).reshape(len(W), 257, 257)


def fdist(fa, fb, n0):
    return max(float(np.linalg.norm(fa[k] - fb[k])) / n0 for k in range(1, 6))
