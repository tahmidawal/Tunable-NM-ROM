"""Shared pieces for the EXACT-LINEAR-TERMS experiments (exp/2026-08-26-eq-learned).

The separable decoder makes every linear weak term exactly computable:
Phi^T u = (Phi^T G) h(z) with A = Phi^T G an (M, R) matrix precomputed once
offline, so mass, previous-state and Laplacian terms need no quadrature at
all.  Only the advection term Phi^T N(u) (sign-upwind, quadratic) still needs
the m-node NNLS quadrature.  Report 2026-08-25-eq-fidelity-ladder.md section 3
item 1; explainer 2026-08-25 section D.

This module holds the advection-only NNLS fit.  The exact-linear residual
closures live in the drivers (sep_eq_ladder.py EXLIN=1, sep_burgers_exlin.py)
next to the incumbent closures they replace, in the project's usual
copied-verbatim style.
"""
from __future__ import annotations

import time

import numpy as np

import jax.numpy as jnp

import ctol_eq


def eq_fit_burgers_adv(u_full, adv_full, Phi_full, cand_pos, Z_snap, K, m,
                       label, nnls_capped):
    """`ctol_eq.eq_fit_burgers` restricted to the ADVECTION row blocks.

    Identical assembly and identical NNLS sequence (same EQ_SEED rng, same
    row subsample size, same support padding, same final refit), but each
    snapshot contributes ONE row block (N(u), the FOM-exact sign-upwind
    advection field) instead of two (u and N(u)).  The u rows are gone
    because the linear terms are computed exactly and no longer consume the
    node budget.  Targets stay the exact full-grid projections Phi^T N(u).
    """
    t0 = time.time()
    r_eq = np.random.default_rng(ctol_eq.EQ_SEED)
    n_s = Z_snap.shape[0]
    idx = r_eq.choice(n_s, size=min(ctol_eq.EQ_SNAPS, n_s), replace=False)
    Gs, bs, snap_c = [], [], []
    Phi_c = Phi_full[cand_pos]                                  # (n_c, M)
    for i in idx:
        z = jnp.asarray(Z_snap[i])
        uf = np.asarray(u_full(z))
        Nf = np.asarray(adv_full(jnp.asarray(uf)))
        bs.append(Phi_full.T @ Nf)                               # (M,)
        Gs.append(Phi_c.T * Nf[cand_pos][None, :])               # (M, n_c)
        snap_c.append(Nf[cand_pos])
    pad_score = np.abs(np.stack(snap_c)).mean(0)                 # reference rule
    G = np.concatenate(Gs, axis=0)
    b = np.concatenate(bs)
    return ctol_eq._solve_nnls(G, b, m, r_eq, nnls_capped, label,
                               dict(kind="weak_burgers_adv_only"), t0,
                               pad_score=pad_score)
