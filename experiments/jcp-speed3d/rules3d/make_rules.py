"""jcp-wide-bank: the extra 3D rules of DESIGN.md (section 2, A1, A2-12b), generated ONCE locally with the vendored,
unchanged generators of vendor/quad3d/rules.py (Hari's mathematics), committed with SHA256; never regenerated on the
cluster. Existing rules (lat256, lat4096..lat32768, gl16/24/32/64/80, smol8, ...) are read from the vendored
rules.npz byte for byte and are NOT duplicated here.

    python rules3d/make_rules.py   ->  rules3d/rules_extra.npz, rules3d/rules_extra.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'vendor' / 'quad3d'))
import rules as RV  # noqa: E402  vendored generators, unchanged

LATTICES = (2048, 65536, 131072)
GAUSS = (12, 20, 40, 48, 56, 64)     # gl64 is also in the vendored file (continuum check); regenerated here as the
                                     # J4 escalation successor and cross-checked against the vendored copy below


def main():
    rules, meta = {}, {}

    def add(name, family, X, w, **extra):
        X = np.ascontiguousarray(X, dtype=np.float64)
        w = np.ascontiguousarray(w, dtype=np.float64)
        assert X.ndim == 2 and X.shape[1] == 3 and len(w) == len(X) and np.all((X >= 0) & (X <= 1))
        rules[name] = (X, w)
        meta[name] = dict(family=family, m=int(len(w)), weight_sum=float(w.sum()), w_min=float(w.min()),
                          sha256=hashlib.sha256(X.tobytes() + w.tobytes()).hexdigest(), **extra)
        print(name, meta[name], flush=True)

    for n in LATTICES:
        z, P = RV.cbc_lattice_vector3(n)
        add(f'lat{n}', 'cbc_lattice', *RV.rank1_lattice3(n, z, seed=0), z=[int(v) for v in z], p2=P,
            p2_check=RV.p2_merit3(z, n), shift_seed=0)
    for p in GAUSS:
        add(f'gl{p}', 'gauss_tensor', *RV.gauss_tensor3(p), p=p)
    v = np.load(HERE.parent / 'vendor' / 'quad3d' / 'rules' / 'rules.npz')
    assert np.array_equal(v['gl64_X'], rules['gl64'][0]) and np.array_equal(v['gl64_w'], rules['gl64'][1]), 'gl64 drift'
    out = HERE / 'rules_extra.npz'
    np.savez(out, **{f'{k}_X': x[0] for k, x in rules.items()}, **{f'{k}_w': x[1] for k, x in rules.items()})
    (HERE / 'rules_extra.json').write_text(json.dumps(dict(rules=meta, npz_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                                                          generator='vendor/quad3d/rules.py (unchanged)'), indent=1) + '\n')


if __name__ == '__main__':
    main()
