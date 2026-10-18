"""jcp-dims: the one new 3D rule file (DESIGN 2.3): lat262144, a CBC rank-1 lattice generated ONCE locally with the
vendored, unchanged generators of vendor/quad3d/rules.py (cbc_lattice_vector3 + rank1_lattice3, shift seed 0), exactly
as speed3d's rules3d/make_rules.py generated lat65536 / lat131072; committed with sha256 and its P2 merit. Every other
lane rule (shift draws, Sobol / Smolyak / MC / Halton draws, bad-vector fixtures) is generated in-job from committed
generators (rules_dims.py) and identified by the sha256 of its points and weights.

    python rules3d/make_rules_dims.py   ->  rules3d/rules_dims.npz, rules3d/rules_dims.json
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'vendor' / 'quad3d'))
import rules as RV  # noqa: E402  vendored generators, unchanged

LATTICES = (262144,)


def main():
    rules, meta = {}, {}
    for n in LATTICES:
        t0 = time.time()
        z, P = RV.cbc_lattice_vector3(n)
        X, w = RV.rank1_lattice3(n, z, seed=0)
        X = np.ascontiguousarray(X, dtype=np.float64)
        w = np.ascontiguousarray(w, dtype=np.float64)
        assert X.shape == (n, 3) and np.all((X >= 0) & (X <= 1))
        name = f'lat{n}'
        rules[name] = (X, w)
        meta[name] = dict(family='cbc_lattice', m=n, z=[int(v) for v in z], p2=P, p2_check=RV.p2_merit3(z, n),
                          shift_seed=0, weight_sum=float(w.sum()), seconds=time.time() - t0,
                          sha256=hashlib.sha256(X.tobytes() + w.tobytes()).hexdigest())
        print(name, meta[name], flush=True)
    out = HERE / 'rules_dims.npz'
    np.savez(out, **{f'{k}_X': x[0] for k, x in rules.items()}, **{f'{k}_w': x[1] for k, x in rules.items()})
    (HERE / 'rules_dims.json').write_text(json.dumps(dict(rules=meta, npz_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                                                         generator='vendor/quad3d/rules.py (unchanged)'), indent=1) + '\n')


if __name__ == '__main__':
    main()
