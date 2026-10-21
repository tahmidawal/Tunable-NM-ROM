"""spectral-pilot Part B: FD FOM accuracy ladders on the L-shape (DESIGN 4.3, A1.5, A1.8, A2.2, A3.1).
PILOT: small, not a result.

  python fom_b.py [--procs 6]

F-IMEX (SBDF2 + SuperLU, both factorisations) and F-RK4 (JAX), L in {32, 64, 128, 192, 256, 320, 384, 448, 512};
scored on the i/64 lattice (L = 32 on the i/32 lattice, diagnostic). Step search n = 5 * 2^k ascending, stopped by
the trajectory-distance rule (1e-5); F-RK4 starts at the first n above the classical-RK4 diffusion limit
n >= 0.25 * 8 nu_max L^2 / 2.785 (smaller n are unstable; skipping them is a search heuristic, reported). Size search
stops once two consecutive sizes meet every target. Writes runs/B/fom.json.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import jax
import jax.numpy as jnp

import common as C
import fd_l as F
import score as SC
from gen_b import run, sub

L_LADDER = [32, 64, 128, 192, 256, 320, 384, 448, 512]
DIST_TOL = 1e-5


def ladder(runfn, ns, den, label):
    rows, prev = [], None
    for n in ns:
        e, V = runfn(n)
        dist = None
        if prev is not None:
            dd = np.sqrt(np.sum((V - prev) ** 2, axis=-1)) / den[:, None]
            dist = float(np.max(dd)) if np.isfinite(dd).all() else float('inf')
        rows.append(dict(n=n, err=[float(x) for x in e], worst=float(np.max(e)), median=float(np.median(e)), dist=dist))
        if dist is not None and dist <= DIST_TOL and np.isfinite(e).all():
            break
        prev = V if np.isfinite(V).all() else None
    print(label, ' '.join('%d:%.2e' % (r['n'], r['worst']) for r in rows), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--procs', type=int, default=6)
    a = ap.parse_args()
    d = C.RUNS / 'B'
    ref = json.loads((d / 'ref.json').read_text())
    te = np.asarray(ref['test_desc'])
    R256 = np.load(d / 'ref_lat256.npy')
    U = ref['U_B'] if ref['U_B'] is not None else ref['U_numerical']
    rep = dict(env=C.env_record(), label=C.PILOT, U=U, scoring=ref['scoring'], families={}, unexplored={})
    numax = float(te[:, 4].max())
    for fam in ('IMEX', 'RK4'):
        rep['families'][fam] = {}
        rep['unexplored'][fam] = []
        run_met = 0
        for L in L_LADDER:
            if run_met >= 2:
                rep['unexplored'][fam].append(L)
                continue
            lat = 32 if L == 32 else 64
            Rl = R256[..., sub(256, lat)]
            den = np.sqrt(np.sum(Rl[:, 0] ** 2, axis=-1))
            t0 = time.perf_counter()
            if fam == 'IMEX':
                ns = [5 * 2 ** k for k in range(1, 10)]                 # 10 .. 2560
                fn = lambda n: (lambda V: (F.err_lat(V, Rl), V))(run(te, L, n, a.procs, lat=lat))
            else:
                nst = 0.25 * 8 * numax * L * L / 2.785
                ns = [5 * 2 ** k for k in range(0, 14) if 5 * 2 ** k >= nst]

                def fn(n, L=L, lat=lat, Rl=Rl):
                    q = F.make_rk4(L, n, out_lattice=False)
                    I, J = F.lattice_idx(L, lat)
                    V = np.stack([np.asarray(q(jnp.asarray(dd)))[:, I, J] for dd in te])
                    return F.err_lat(V, Rl), V
            rows = ladder(fn, ns, den, f'B {fam} L={L}')
            rep['families'][fam][L] = dict(rows=rows, lattice=lat, seconds=time.perf_counter() - t0,
                                           diagnostic_only=(L == 32))
            if L != 32:
                best = min(r['worst'] for r in rows)
                run_met = run_met + 1 if all(SC.classify(best, U, t) == 'meets' for t in SC.TARGETS) else 0
            C.dump(d / 'fom.json', rep)
    C.dump(d / 'fom.json', rep)
    print('B FOM done', flush=True)


if __name__ == '__main__':
    main()
