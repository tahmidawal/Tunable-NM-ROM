"""imp-geometry POD arms (R-PODG and C-ECSW; DESIGN 5.1-5.2, A1).

POD of the FD training snapshots on the data grid (grid inner product h^2), method of snapshots (Gram in f64).
Space for rom.py: P = Phi (interior nodes), S = (D_x + D_y)_h Phi (Shortley-Weller rows on H), w = h^2, M = I,
K = Phi^T (-A_h) Phi h^2 (non-symmetric on H), IC on the 1/n_init sub-lattice, decoder = lattice rows.
ECSW: weights over FD nodes reproducing the tested nonlinear vector (Galerkin: Mt = r; LSPG: Mt = min(4r, R)) on the
projected training states, by the jcp-hr-baselines greedy NNLS (vendor/nnls.py, byte-identical), supports kappa r.
Online ECSW cost: only the selected rows of Phi and of D_h Phi are used (precomputed rows; no stencil neighbours are
needed online).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import geom as G
import fd
import rom as RM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor'))

ECSW_SEED = 7_199_001
N_CAND = 32768
ECSW_KAPPA = (2, 4, 8)


def pod(U, h, R, chunk=32768):
    """U (S, n) snapshots on the host (interior nodes); returns Phi (n, R) orthonormal in the h^2-weighted inner
    product, and the singular values. Gram and basis are formed in column chunks on the device (memory-bounded)."""
    U = np.asarray(U)
    S, n = U.shape
    C = jnp.zeros((S, S))
    for j in range(0, n, chunk):
        B = jnp.asarray(U[:, j:j + chunk])
        C = C + B @ B.T
    C = C * (h * h)
    lam, V = jnp.linalg.eigh(C)
    lam, V = lam[::-1], V[:, ::-1]
    R = min(R, int(jnp.sum(lam > lam[0] * 1e-26)))
    W = V[:, :R] / jnp.sqrt(lam[:R])[None, :]
    Phi = np.zeros((n, R))
    for j in range(0, n, chunk):
        Phi[j:j + chunk] = np.asarray(jnp.asarray(U[:, j:j + chunk]).T @ W)
    # one re-orthonormalisation pass in the weighted inner product (f64)
    Q, Rq = jnp.linalg.qr(jnp.asarray(Phi) * h)
    Phi = np.asarray(Q / h * jnp.sign(jnp.diag(Rq))[None, :])
    return Phi, np.sqrt(np.maximum(np.asarray(lam), 0))


def project(U, Phi, h, chunk=32768):
    """Coefficients U Phi h^2 in column chunks (host arrays)."""
    Z = np.zeros((U.shape[0], Phi.shape[1]))
    for j in range(0, U.shape[1], chunk):
        Z += np.asarray(jnp.asarray(U[:, j:j + chunk]) @ jnp.asarray(Phi[j:j + chunk]))
    return Z * h * h


def cols_op(gr, Phi, op):
    """Apply a grid operator (fd.lap / fd.dsum) to every column of Phi (n_in, R) -> (n_in, R)."""
    m = gr.mask_np
    out = np.zeros_like(Phi)
    f = jax.jit(lambda U: jax.vmap(lambda u: op(gr.c, u))(U))
    for s in range(0, Phi.shape[1], 64):
        cols = Phi[:, s:s + 64].T
        Ug = np.zeros((cols.shape[0],) + m.shape)
        Ug[:, m] = cols
        out[:, s:s + 64] = np.asarray(f(jnp.asarray(Ug)))[:, m].T
    return out


def pod_space(dom, n, Phi, n_init):
    gr = fd.Grid(dom, n)
    h = gr.h
    m = gr.mask_np
    S = cols_op(gr, Phi, fd.dsum)
    LPhi = cols_op(gr, Phi, fd.lap)
    K = -(Phi.T @ LPhi) * h * h
    Mm = (Phi.T @ Phi) * h * h
    # initial-condition sub-lattice (n multiple of n_init)
    assert n % n_init == 0
    s = n // n_init
    sub = np.zeros_like(m)
    sub[s - 1::s, s - 1::s] = True
    flat_sub = sub[m]
    Xin = np.stack([gr.X[m], gr.Y[m]], -1)
    Pi = Phi[flat_sub]
    wi = np.full(int(flat_sub.sum()), (s * h) ** 2)
    # decoder: lattice rows
    I, J = gr.lat_idx()
    pos = -np.ones(m.shape, int)
    pos[m] = np.arange(int(m.sum()))
    Pl = Phi[pos[I, J]]
    assert np.all(pos[I, J] >= 0)
    sp = RM.Space(Phi, S, np.full(Phi.shape[0], h * h), Mm, K, Pi, wi, Pl, name=f'pod-{dom}-n{n}',
                  meta=dict(dom=dom, ic_points=Xin[flat_sub], node_points=Xin, n=n, n_init=n_init,
                            K_asym=float(np.linalg.norm(K - K.T) / np.linalg.norm(K)),
                            M_dev=float(np.linalg.norm(Mm - np.eye(len(Mm))))))
    return sp


def ecsw_fit(sp, r, Mt, Z, kappas=ECSW_KAPPA, seed=ECSW_SEED, n_cand=N_CAND, max_seconds=3600):
    """Fit ECSW node weights for the tested nonlinear vector Theta^T W (u * s) at reduced states Z (n_states, R_max).
    Returns {kappa: dict(nodes, weights, relres, size)} and the candidate set."""
    import nnls as NN
    P = sp.P
    n = P.shape[0]
    Zr = Z[:, :r]
    U = Zr @ P[:, :r].T                 # (states, n) reduced fields at the nodes
    Sv = Zr @ sp.S[:, :r].T
    contrib_norm = np.sqrt(np.sum((P[:, :Mt] ** 2), axis=1)) * np.mean(np.abs(U * Sv), axis=0)
    rng = np.random.default_rng([seed, r, Mt])
    if n <= n_cand:
        cand = np.arange(n)
    else:
        half = n_cand // 2
        uni = rng.choice(n, half, replace=False)
        p = contrib_norm / contrib_norm.sum()
        imp = rng.choice(n, n_cand, replace=False, p=p)
        cand = np.unique(np.concatenate([uni, imp]))[:n_cand]
    w = sp.w
    rows, b = [], []
    for zi in range(Zr.shape[0]):
        g = P[cand, :Mt] * (U[zi, cand] * Sv[zi, cand])[:, None]          # (cand, Mt)
        full = P[:, :Mt].T @ (w * U[zi] * Sv[zi])                         # exact tested vector
        sc = 1.0 / (np.linalg.norm(full) + 1e-300)
        rows.append(g.T * sc)
        b.append(full * sc)
    Gm = np.concatenate(rows)
    bv = np.concatenate(b)
    targets = sorted({min(int(k * r), len(cand)) for k in kappas})
    o = NN.greedy_nnls(Gm, bv, supports=targets, cap=max(targets), max_seconds=max_seconds)
    out = {}
    for k in kappas:
        t = min(int(k * r), len(cand))
        x = o['records'].get(('s', t))
        if x is None and o['reason'] == 'kkt' and len(o['support']):
            pos = o['weights'] > 0
            x = dict(support=o['support'][pos], weights=o['weights'][pos], relres=o['relres'], size=int(pos.sum()))
        out[k] = None if x is None else dict(nodes=cand[x['support']], weights=np.asarray(x['weights']),
                                             relres=float(x['relres']), size=int(x['size']))
    return out, dict(reason=o['reason'], n_cand=len(cand), rows=Gm.shape[0])
