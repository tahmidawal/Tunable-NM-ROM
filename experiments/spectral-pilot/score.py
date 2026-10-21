"""spectral-pilot: three-way scoring against a target with a conditional allowance U (DESIGN A2.5, A3.1) and the
deterministic matcher checks SW1a/SW1b (A3.3). PILOT: small, not a result."""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

TARGETS = [3e-2, 1e-2, 3e-3, 1e-3]


def classify(e, U, tau):
    """'meets' if e + U <= tau, 'misses' if e - U > tau, else 'unresolved'. e = worst error (inf = invalid)."""
    if not np.isfinite(e):
        return 'misses'
    if e + U <= tau:
        return 'meets'
    if e - U > tau:
        return 'misses'
    return 'unresolved'


def sw1b():
    fx = [((0.5, 0.1, 1.0), 'meets'), ((1.5, 0.1, 1.0), 'misses'), ((0.95, 0.1, 1.0), 'unresolved'),
          ((0.9, 0.1, 1.0), 'meets'), ((1.1, 0.1, 1.0), 'unresolved')]
    out = [dict(e=e, U=U, tau=t, want=w, got=classify(e, U, t)) for (e, U, t), w in fx]
    return dict(rows=out, passed=all(o['want'] == o['got'] for o in out))


def sw1a(Bref, err_fn, eps_list=(1e-3, 1e-2), seed=3):
    """Reference + perturbation of exact relative size eps at every scored time -> scored error must equal eps."""
    rng = np.random.default_rng(seed)
    Bref = np.asarray(Bref)
    rows = []
    for eps in eps_list:
        Bp = Bref.copy()
        for j in range(len(Bp)):
            n0 = np.linalg.norm(Bref[j, 0])
            for k in range(1, Bp.shape[1]):
                d = rng.standard_normal(Bp.shape[2:])
                Bp[j, k] += eps * n0 * d / np.linalg.norm(d)
        e = np.asarray(err_fn(jnp.asarray(Bp), jnp.asarray(Bref)))
        rows.append(dict(eps=eps, max_rel_dev=float(np.max(np.abs(e / eps - 1)))))
    return dict(rows=rows, passed=all(r['max_rel_dev'] <= 1e-10 for r in rows))
