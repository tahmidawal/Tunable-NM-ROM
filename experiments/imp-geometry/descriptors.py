"""imp-geometry frozen descriptors (DESIGN A3.1): every (family, level, split) array of the frozen variant, its sha256 and
the rejection acceptance rate. Writes runs/descriptors.json. Pure NumPy (no JAX)."""
import hashlib
import json
from pathlib import Path

import numpy as np

import geom as G


def acceptance(fam, lev, split, n_probe=20000):
    seed = G.seed_of(fam, lev, split)
    F = G.FAMS[fam]
    r = np.random.default_rng([seed, 10 ** 6])
    c = r.uniform(F['c0'], F['c0'] + F['W'], size=(n_probe, 2))
    return float(np.mean([not G.rejected(x, y) for x, y in c]))


def main():
    out = dict(variant=G.VARIANT, spec=G.VARIANTS[G.VARIANT], cohorts={})
    for fam in G.FAMS:
        for lev in G.LEVELS:
            for split in G.SPLITS:
                D = G.draw(fam, lev, split)
                out['cohorts'][f'{fam}|{lev}|{split}'] = dict(
                    seed=G.seed_of(fam, lev, split), n=len(D), sha256=hashlib.sha256(np.ascontiguousarray(D).tobytes()).hexdigest(),
                    acceptance_rate=acceptance(fam, lev, split), s_min=float(G.steep(D).min()), s_max=float(G.steep(D).max()))
    assert G.check_cohorts()
    p = Path(__file__).resolve().parent / 'runs' / 'descriptors.json'
    p.write_text(json.dumps(out, indent=1))
    print(p, len(out['cohorts']))


if __name__ == '__main__':
    main()
