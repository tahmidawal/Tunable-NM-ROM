"""spectral-pilot Part A diagnostics (DESIGN A1.6 G7, A2.1): the 1024-point IC rule against pscore.ic_coefs_2d
(8192 points), front sharpness max|grad u| / max|u| at each output time and the remaining energy ||u(t5)|| / ||u0||,
from the references. PILOT: small, not a result.

  python diag_a.py A1 A2 A3
"""
from __future__ import annotations

import json
import sys

import numpy as np
import jax
import jax.numpy as jnp

import common as C

P = C.P


def grad_max(b, N):
    """max |grad u| and max |u| on the 3/2-padded grid (the pscore transforms)."""
    M = 3 * N // 2
    bp = jnp.pad(b, ((0, M - N), (0, M - N)))
    n = M - 1
    S = lambda x, ax: P.along(P.sine_synth_last, x, ax, M)
    Cc = lambda x, ax: P.along(P.cos_synth_last, x, ax, M)
    Ty = S(bp, 1)
    u = S(Ty, 0)
    ux = Cc(P.kvec(n, 2, 0) * Ty, 0)
    uy = S(Cc(P.kvec(n, 2, 1) * bp, 1), 0)
    return jnp.max(jnp.sqrt(ux ** 2 + uy ** 2)), jnp.max(jnp.abs(u))


for lv in sys.argv[1:]:
    d = C.RUNS / 'A' / lv
    ref = json.loads((d / 'ref.json').read_text())
    Nref = ref['Nref']
    te = np.asarray(ref['test_desc'])
    B = np.load(d / 'ref_coef.npy')
    rule = C.ICRule(Nref)
    g7 = []
    for dd in te:
        gx, gy, a = rule.coefs(jnp.asarray(dd))
        b1 = np.asarray(a * jnp.outer(gx, gy))
        b2 = P.ic_coefs_2d(dd, Nref)
        g7.append(float(np.linalg.norm(b1 - b2) / np.linalg.norm(b2)))
    f = jax.jit(lambda b: grad_max(b, Nref))
    sharp = np.array([[float(x / y) for x, y in [f(jnp.asarray(B[j, k])) for k in range(6)]] for j in range(len(B))])
    energy = np.sqrt(np.sum(B[:, 5] ** 2, axis=(-1, -2)) / np.sum(B[:, 0] ** 2, axis=(-1, -2)))
    out = dict(level=lv, G7_max=max(g7), G7_pass=bool(max(g7) <= 1e-12), sharp=sharp.tolist(),
               sharp_t0_median=float(np.median(sharp[:, 0])), sharp_max_median=float(np.median(sharp[:, 1:].max(1))),
               sharp_max_worst=float(sharp[:, 1:].max()),
               energy_t5=energy.tolist(), energy_median=float(np.median(energy)),
               s_median=float(np.median(C.steep(te))), env=C.env_record(), label=C.PILOT)
    C.dump(d / 'diag.json', out)
    print(lv, 'G7 %.2e' % out['G7_max'], 'sharp t0 %.1f, t>0 median %.1f worst %.1f' % (out['sharp_t0_median'], out['sharp_max_median'], out['sharp_max_worst']),
          'energy %.3f' % out['energy_median'], 's_med %.1f' % out['s_median'])
