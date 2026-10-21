"""spectral-pilot Part A: the spectral FOM accuracy ladder for one level (DESIGN 4.1, A1.5, A2.2, A3.1).
PILOT: small, not a result.

  python fom_a.py A2 [--threads 10]

Writes runs/A/<level>/fom.json: per (N, n) per-case continuous-L2 errors against the reference, the trajectory
distance to the n/2 run, and SW1a/SW1b. Search heuristics (registered): n ascending until the trajectory distance
<= 1e-5 (A2.2); sizes ascending, stopped once two consecutive sizes meet every target (larger N only cost more per
step and need at least as many stable steps); unexplored sizes are reported.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import jax.numpy as jnp

import common as C
import score as SC
from gen_a import run_batched

N_LADDER = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
N_STEPS = [5 * 2 ** k for k in range(10)]
DIST_TOL = 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('level')
    ap.add_argument('--threads', type=int, default=10)
    a = ap.parse_args()
    lv = a.level
    d = C.RUNS / 'A' / lv
    ref = json.loads((d / 'ref.json').read_text())
    Nref = ref['Nref']
    te = np.asarray(ref['test_desc'])
    Bref = jnp.asarray(np.load(d / 'ref_coef.npy'))
    den = np.sqrt(np.sum(np.asarray(Bref[:, 0]) ** 2, axis=(-1, -2)))
    U = ref['checks']['space']['max'] + ref['checks']['time']['max']
    pad = json.loads((d / 'pad.json').read_text()) if (d / 'pad.json').exists() else None
    if pad:
        U += pad['max']
    rep = dict(level=lv, env=C.env_record(), Nref=Nref, U=U, U_has_pad=bool(pad), label=C.PILOT,
               SW1a=SC.sw1a(np.asarray(Bref), C.err_coef), SW1b=SC.sw1b(), sizes={}, unexplored=[])
    print(lv, 'SW1a', rep['SW1a']['passed'], 'SW1b', rep['SW1b']['passed'], flush=True)
    all_met_run = 0
    for N in [x for x in N_LADDER if x <= 3 * Nref // 4]:
        if all_met_run >= 2:
            rep['unexplored'].append(N)
            continue
        rows, prev = [], None
        t0 = time.perf_counter()
        for n in N_STEPS:
            B = run_batched(C.make_fom(N, n, 5, lattice=False), te, a.threads)
            Bp = C.pad_to(jnp.asarray(B), Nref)
            e = np.asarray(C.err_coef(Bp, Bref))
            dist = None
            if prev is not None:
                dd = np.sqrt(np.sum((B - prev) ** 2, axis=(-1, -2))) / den[:, None]
                dist = float(np.max(dd)) if np.isfinite(dd).all() else float('inf')
            rows.append(dict(n=n, err=e.tolist(), worst=float(np.max(e)), median=float(np.median(e)), dist=dist))
            if dist is not None and dist <= DIST_TOL and np.isfinite(e).all():
                break
            prev = B if np.isfinite(B).all() else None
        rep['sizes'][N] = dict(rows=rows, seconds=time.perf_counter() - t0)
        print(lv, 'FOM N=%d' % N, ' '.join('%d:%.2e' % (r['n'], r['worst']) for r in rows),
              '(%.0f s)' % rep['sizes'][N]['seconds'], flush=True)
        best = min(r['worst'] for r in rows)
        all_met_run = all_met_run + 1 if all(SC.classify(best, U, t) == 'meets' for t in SC.TARGETS) else 0
        C.dump(d / 'fom.json', rep)
    C.dump(d / 'fom.json', rep)
    print(lv, 'done', flush=True)


if __name__ == '__main__':
    main()
