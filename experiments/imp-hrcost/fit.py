"""imp-hrcost fitted rules (DESIGN 2.2, A1.5): support-targeted greedy NNLS (jcp-hr-baselines nnls.py, copied
unchanged) on the ECSW system of projected advection contributions over the Gauss 128^2 candidate pool.

  g_qj = p_q * u_q(c_j) s_q(c_j)   (p_q: the family's per-point row without the quadrature weight, DESIGN A2.2:
                                    F3a Q^T(L psi_q), F3b L psi_q, F2F3 G(x_q))
  C[(j, :), q] = g_qj / ||b_j||,   b = C w_parent,   min_{w >= 0} ||C w - b||, first crossing of supports kappa r.

Snapshots c_j come from parent-arm rollouts of training draws only (hrcore.train_draw_2d(seed=0, count=576) rows 0-15,
replacements 16, 17, ... if a parent rollout is invalid); disjointness is asserted on descriptors (C7).
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import jax
import jax.numpy as jnp

import nnls as NN

KAPPAS = (4, 8, 16)
NDRAW, MAX_REPL = 16, 4
SNAP_STEPS = {'F3a': tuple(range(0, 51, 5)), 'F2F3': tuple(range(0, 51, 5)), 'F3b': (0, 10, 30, 50)}   # dt0 steps
CAP, DEADLINE_S = 6400, 12 * 60       # DESIGN A3.5


# ---------------------------------------------------------------- training draws (copied from hrcore) ----

def train_draw_2d(seed=0, count=576):
    """jcp-hr-baselines hrcore.train_draw_2d, text copied: rows (cx, cy, w, a, nu)."""
    rng = np.random.default_rng(seed)
    cx = rng.uniform(0.15, 0.85, count)
    cy = rng.uniform(0.15, 0.85, count)
    w = rng.uniform(0.05, 0.20, count)
    a = rng.uniform(0.5, 2.0, count)
    nu = np.exp(rng.uniform(np.log(0.01), np.log(0.1), count))
    return np.stack([cx, cy, w, a, nu], 1)


def sha(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def assert_not_dev_val(rows, dev_val_desc):
    """C7: no fitting row equals (to 1e-12) any dev6 / val32 / test64 descriptor row."""
    rows = np.asarray(rows, float)
    for j, r_ in enumerate(np.asarray(dev_val_desc, float)):
        assert not np.any(np.all(np.abs(rows - r_[None]) <= 1e-12, axis=1)), ('fit leakage', j)
    return True


# ------------------------------------------------------------------------------------- ECSW system ----

def build_system(rows_unw, Gq, Gs, w_parent, snaps):
    """(C on device (n k, m), b (n k,), info). rows_unw (m, k): unweighted per-point rows of the family (A2.2);
    snaps (n, r) host array."""
    S = jnp.asarray(snaps)
    a = (S @ Gq.T) * (S @ Gs.T)                                  # (n, m)
    wp = jnp.asarray(w_parent)
    bj = (a * wp[None]) @ rows_unw                               # (n, k) = sum_q w_q p_q a_jq
    nb = jnp.linalg.norm(bj, axis=1)
    keep = np.asarray(nb > 1e-12 * jnp.max(nb))
    if not np.isfinite(np.asarray(bj)).all():
        raise FloatingPointError('non-finite ECSW target')
    a, bj, nb = a[keep], bj[keep], nb[keep]
    n, r = bj.shape
    C = (rows_unw.T[None, :, :] * (a / nb[:, None])[:, None, :]).reshape(n * r, -1)
    b = np.asarray((bj / nb[:, None]).reshape(-1))
    info = dict(rows=int(C.shape[0]), cols=int(C.shape[1]), bytes=int(C.size * 8), dropped=int((~keep).sum()),
                snapshots=int(n))
    return jax.block_until_ready(C), b, info


def fit(C, b, Rp, log=None, kappas=KAPPAS):
    t0 = time.perf_counter()
    res = NN.greedy_nnls(C, b, supports=[k * Rp for k in kappas], cap=CAP, max_seconds=DEADLINE_S, log=log)
    out = dict(seconds=time.perf_counter() - t0, reason=res['reason'], relres_final=float(res['relres']),
               support_final=int(len(res['support'])), drift_ok=bool(res.get('drift_ok', False)),
               drift_events=res.get('drift_events'), rules={})
    for k in kappas:
        rec = res['records'].get(('s', k * Rp))
        if rec is None:
            out['rules'][k] = None                               # "not reached" (incomplete evidence, A1.5)
        else:
            out['rules'][k] = dict(idx=np.asarray(rec['support'], int), w=np.asarray(rec['weights'], float),
                                   relres=float(rec['relres']), size=int(rec['size']))
    return out


def check_residual(rows_unw, Gq, Gs, w_parent, snaps, idx, w_hat):
    """A2.5: the fit residual recomputed by an independent path (support columns rebuilt from the raw tables, snapshot by
    snapshot; target recomputed), relative to the normalised target, as the NNLS objective."""
    idx = np.asarray(idx, int)
    wp, wh = jnp.asarray(w_parent), jnp.asarray(w_hat)
    num, den = 0., 0.
    blocks = []
    for c in np.asarray(snaps):
        cj = jnp.asarray(c)
        a = (Gq @ cj) * (Gs @ cj)
        bj = rows_unw.T @ (wp * a)
        blocks.append((a, bj, float(jnp.linalg.norm(bj))))
    nmax = max(b_[2] for b_ in blocks)
    for a, bj, nb in blocks:
        if not nb > 1e-12 * nmax:            # the same drop rule as build_system
            continue
        fj = rows_unw[idx].T @ (wh * a[idx])
        num += float(jnp.sum((fj - bj) ** 2)) / nb ** 2
        den += 1.
    return float(np.sqrt(num / den))
