"""imp-geometry interaction diagnostic (DESIGN A1.2): D_geo and feature-window energy of L / H against the box, FD
SBDF2 at per-unit n on val[0:ncases] of each (family, level). Writes <out>/interaction.json.

  python interaction.py --out <dir> [--n 256] [--ncases 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geom as G      # noqa: E402
import fd             # noqa: E402
import ladder as LD   # noqa: E402


def feature_centre(dom):
    return (G.LC, G.LC) if dom == 'L' else G.HC


def run(n, ncases, nsteps):
    res = {}
    for fam in G.FAMS:
        for lev in G.LEVELS:
            D = G.draw(fam, lev, 'val', ncases)
            out = {}
            for dom in G.DOMS:
                P, I, J = G.lattice(dom)
                s = n // G.LAT
                q = fd.make_sbdf2(dom, n, nsteps, 1e-9, out_idx=(I * s - 1, J * s - 1))
                O, st, sec = LD.run_cohort(q, D)
                out[dom] = (P, O)
            PB, UB = out['B']
            keyB = {tuple(np.round(p * G.LAT).astype(int)): i for i, p in enumerate(PB)}
            for dom in ('L', 'H'):
                P, U = out[dom]
                idx = np.array([keyB[tuple(np.round(p * G.LAT).astype(int))] for p in P])
                n0 = np.linalg.norm(U[:, 0], axis=1)
                Dg = np.max(np.linalg.norm(U[:, 1:] - UB[:, 1:, idx], axis=2), axis=1) / n0
                c = feature_centre(dom)
                near = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) < 0.2
                fe = np.max(np.linalg.norm(U[:, 1:, near], axis=2) ** 2 / np.linalg.norm(U[:, 1:], axis=2) ** 2, axis=1)
                zero = np.max(np.linalg.norm(U[:, 1:], axis=2), axis=1) / n0
                res[f'{dom}|{fam}|{lev}'] = dict(D_geo=Dg.tolist(), D_geo_median=float(np.median(Dg)),
                                                 feature_energy_median=float(np.median(fe)),
                                                 zero_baseline_worst=float(zero.max()), passes=bool(np.median(Dg) >= 0.05))
                print(dom, fam, lev, f'D_geo median {np.median(Dg):.3f} min {Dg.min():.3f} max {Dg.max():.3f}; '
                      f'feature energy median {np.median(fe):.3f}', flush=True)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--ncases', type=int, default=8)
    ap.add_argument('--nsteps', type=int, default=2000)
    ap.add_argument('--variant', default=G.VARIANT)
    ap.add_argument('--T', type=float, default=G.T_END)
    a = ap.parse_args()
    G.configure(a.variant)
    G.T_END = a.T
    r = run(a.n, a.ncases, a.nsteps)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'interaction_{a.variant}_T{a.T}.json').write_text(json.dumps(dict(variant=a.variant, T=a.T, n=a.n, ncases=a.ncases, nsteps=a.nsteps, cells=r,
                                                                  backend=jax.default_backend()), indent=1))
