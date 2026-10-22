"""G-OPS (DESIGN A3.2): manufactured operator check with exact trigonometric integrals.

Basis: the six sine modes (1,1), (2,1), (1,2), (2,2), (3,1), (1,3) (already L2-orthonormal: <u,v> = (1/4) sum b b',
so each mode is 2 sin(a pi x) sin(b pi y)). Exact Galerkin tensor T[k,i,j] = int psi_k psi_i (d_x + d_y) psi_j from 1-D
integrals by a 200-point Gauss-Legendre rule on [0, 1] (exact to round-off for these degrees; independent of the
engine's composite 2-D rule). Gated: the engine (point form and tensor form, rom.py, on the reference rule (4, 16)) and
the engine's K vs exact; the FOM linear symbol vs -pi^2 (k^2 + l^2). Reported only: pscore.nonlinear (N = 8, M = 12)
vs the exact sine coefficients of -u (u_x + u_y) (a finite pseudo-spectral quadrature difference).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor'))
import pscore as P   # noqa: E402
import podsine as PS  # noqa: E402
import rom as RM      # noqa: E402
import spec as SP     # noqa: E402

MODES = ((1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3))


def run(n_states=10, seed=11):
    n = 3
    Phi = np.zeros((len(MODES), n, n))
    for i, (a, b) in enumerate(MODES):
        Phi[i, a - 1, b - 1] = 2.0
    x, w = np.polynomial.legendre.leggauss(200)
    x, w = 0.5 * (x + 1), 0.5 * w
    k = np.arange(1, 8)
    S = np.sin(np.pi * np.outer(x, k))                         # (200, 7)
    C = np.cos(np.pi * np.outer(x, k)) * (np.pi * k)
    Isss = np.einsum('q,qa,qb,qc->abc', w, S, S, S)            # int sin a sin b sin c
    Issc = np.einsum('q,qa,qb,qc->abc', w, S, S, C)            # int sin a sin b (d/dx sin c)
    F = Phi                                                    # (R, 3, 3) on modes 1..3
    Is3, Ic3 = Isss[:n, :n, :n], Issc[:n, :n, :n]
    Tx = np.einsum('kab,icd,jef,ace,bdf->kij', F, F, F, Ic3, Is3)
    Ty = np.einsum('kab,icd,jef,ace,bdf->kij', F, F, F, Is3, Ic3)
    Tex = Tx + Ty
    Kex = PS.exact_K(Phi)
    sp = PS.pod_space(Phi, 4, 16, name='gops')
    r = len(MODES)
    rng = np.random.default_rng(seed)
    Zs = rng.standard_normal((n_states, r))
    Tt = np.asarray(RM.tensor(sp.P, sp.P, sp.S, sp.w))
    e_pt, e_tn, e_fom = 0.0, 0.0, 0.0
    for z in Zs:
        ex = np.einsum('kij,i,j->k', Tex, z, z)
        pt = sp.P.T @ (sp.w * (sp.P @ z) * (sp.S @ z))
        tn = np.einsum('kij,i,j->k', Tt, z, z)
        e_pt = max(e_pt, float(np.linalg.norm(pt - ex) / np.linalg.norm(ex)))
        e_tn = max(e_tn, float(np.linalg.norm(tn - ex) / np.linalg.norm(ex)))
        # FOM: exact sine coefficients of -u (u_x + u_y) on modes 1..7 vs pscore.nonlinear(N = 8)
        b = np.tensordot(z, Phi, axes=1)                       # (3, 3)
        bp = np.zeros((7, 7))
        bp[:3, :3] = b
        Uq = S[:, :3] @ b @ S[:, :3].T
        Gq = C[:, :3] @ b @ S[:, :3].T + S[:, :3] @ b @ C[:, :3].T
        ex_c = -4.0 * np.einsum('pq,p,q,pa,qb->ab', Uq * Gq, w, w, S, S)
        fo = np.asarray(P.nonlinear(jnp.asarray(bp), 8, 2)[0])
        e_fom = max(e_fom, float(np.linalg.norm(fo - ex_c) / np.linalg.norm(ex_c)))
    dK = float(np.linalg.norm(sp.K - Kex) / np.linalg.norm(Kex))
    dM = float(np.linalg.norm(sp.M - np.eye(r)))                # exact M = I (orthonormal sine modes)
    kk = np.arange(1, 8.0)
    dL = float(np.max(np.abs(np.asarray(SP.lam_grid(8)) + np.pi ** 2 * (kk[:, None] ** 2 + kk[None, :] ** 2))))
    return dict(point_vs_exact=e_pt, tensor_vs_exact=e_tn, K_vs_exact=dK, M_vs_exact=dM, fom_linear_symbol=dL,
                fom_nonlinear_vs_exact_reported=e_fom,
                passed=bool(e_pt <= 1e-10 and e_tn <= 1e-10 and dK <= 1e-12 and dM <= 1e-12 and dL <= 1e-12))


if __name__ == '__main__':
    print(run())
