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


# ===========================================================================
# Stage 2: same-target rows (advection rows at arbitrary states + gradient-
# teacher rows with the full-grid Jacobian FROZEN -- the convex form; the
# literal J_s^T R_s is QUADRATIC in w, explainer 2026-08-25 section E).
# ===========================================================================
def build_adv_row_block(u_full, adv_full, Phi_full, cand_pos, z):
    """One advection-projection row block at state z: (M, n_c) rows
    Phi_c^T * N(u_z)|cand with target Phi^T N(u_z) (exact full grid).
    Returns (rows, target, N_cand)."""
    uf = np.asarray(u_full(jnp.asarray(z)))
    Nf = np.asarray(adv_full(jnp.asarray(uf)))
    rows = Phi_full[cand_pos].T * Nf[cand_pos][None, :]          # (M, n_c)
    return rows, Phi_full.T @ Nf, Nf[cand_pos]


def build_grad_row_block(J_f, wt, dt, adv_rows, adv_target):
    """Gradient-teacher rows at one state, FROZEN full-grid factor:

        J_f^T (R_s(w) - R_f) = DT * (diag(wt) J_f)^T
                               [ Phi_c^T diag(N_c) w  -  Phi^T N_f ]

    (the linear terms cancel exactly under EXLIN), so the K rows are the
    state's advection row block left-multiplied by DT*(diag(wt) J_f)^T and
    the K targets the same transform of the state's advection target.  LINEAR
    in w -- the convex family.  J_f (M, K) full-grid weak Jacobian at the
    state, wt (M,) the solver's mode weights at the state's nu."""
    W = dt * (wt[:, None] * J_f).T                               # (K, M)
    return W @ adv_rows, W @ adv_target


def nnls_same_target(state_blocks, grad_blocks, m, nnls_capped, seed,
                     grad_w=0.0, eq_rows=3072, label=""):
    """Assemble state row blocks (+ optionally gradient row blocks, weighted
    grad_w after per-row normalization) and run the reference NNLS sequence
    (sep_solvers._nnls_rows: capped Lawson-Hanson on a subsample -> support
    padding -> nonnegative refit on ALL rows).  pad_score = mean |N_c| over
    the fit states, the reference rule.  Diagnostics are computed on the
    UNWEIGHTED normalized rows of each block separately."""
    import time as _time
    import sep_solvers as _ss
    t0 = _time.time()
    Gs = [b[0] for b in state_blocks]
    bs = [b[1] for b in state_blocks]
    pad_score = np.abs(np.stack([b[2] for b in state_blocks])).mean(0)
    n_state = sum(g.shape[0] for g in Gs)
    if grad_w > 0.0 and grad_blocks:
        Gs += [b[0] for b in grad_blocks]
        bs += [b[1] for b in grad_blocks]
    G = np.concatenate(Gs, axis=0)
    b = np.concatenate(bs)
    scale = np.linalg.norm(G, axis=1) + 1e-300
    Gn = G / scale[:, None]
    bn = b / scale
    wgt = np.ones(Gn.shape[0])
    wgt[n_state:] = grad_w if grad_w > 0.0 else 1.0
    rng = np.random.default_rng(seed)
    keep, wq, padded = _ss._nnls_rows(Gn * wgt[:, None], bn * wgt, m,
                                      nnls_capped, rng, eq_rows, pad_score)
    info = dict(kind="same_target", grad_w=float(grad_w),
                n_state_rows=int(n_state),
                n_grad_rows=int(Gn.shape[0] - n_state),
                m=int(len(keep)), padded=int(padded),
                secs=_time.time() - t0)
    for blk, lo, hi in (("state", 0, n_state), ("grad", n_state, Gn.shape[0])):
        if hi <= lo:
            continue
        d = _ss.eq_diag(Gn[lo:hi], bn[lo:hi], keep, wq)
        info.update({f"{blk}_{k}": v for k, v in d.items()})
    info["rel_fit"] = info["state_rel_fit"]
    print(f"  NNLS-ST {label}: {n_state} state rows + "
          f"{info['n_grad_rows']} grad rows (w={grad_w:g})  state rel fit "
          f"{info['state_rel_fit']:.2e}  grad rel fit "
          f"{info.get('grad_rel_fit', float('nan')):.2e} "
          f"[{info['secs']:.0f}s]", flush=True)
    return keep, wq, info
