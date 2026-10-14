"""jcp-combined DESIGN A11 (POST HOC): the Gauss tensor rules 96^3 and 112^3 for the W1024 escalation extension, generated
ONCE locally with the vendored, unchanged generator vendor/quad3d/rules.py (gauss_tensor3), one file per rule (each
below the 50 MB mirror limit), committed with SHA256 and pinned in c3common.RULE_FILES; never regenerated on the cluster.
gl64 is regenerated as a cross-check against the vendored copy.

    python rules3d/make_rules_ext2.py  ->  rules3d/rules_gl96.npz, rules3d/rules_gl112.npz, rules3d/rules_ext2.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'vendor' / 'quad3d'))
import rules as RV  # noqa: E402  vendored generators, unchanged


def main():
    v = np.load(HERE.parent / 'vendor' / 'quad3d' / 'rules' / 'rules.npz')
    X64, w64 = RV.gauss_tensor3(64)
    assert np.array_equal(v['gl64_X'], X64) and np.array_equal(v['gl64_w'], w64), 'generator drift (gl64)'
    meta = {}
    for p in (96, 112):
        X, w = (np.ascontiguousarray(a, dtype=np.float64) for a in RV.gauss_tensor3(p))
        assert X.shape == (p ** 3, 3) and len(w) == p ** 3 and np.all((X >= 0) & (X <= 1))
        assert abs(w.sum() - 1.) < 1e-12
        out = HERE / f'rules_gl{p}.npz'
        np.savez(out, **{f'gl{p}_X': X, f'gl{p}_w': w})
        meta[f'gl{p}'] = dict(family='gauss_tensor', p=p, m=int(len(w)), weight_sum=float(w.sum()), w_min=float(w.min()),
                              sha256=hashlib.sha256(X.tobytes() + w.tobytes()).hexdigest(),
                              npz=out.name, npz_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
        print(f'gl{p}', meta[f'gl{p}'], flush=True)
    (HERE / 'rules_ext2.json').write_text(json.dumps(dict(rules=meta, generator='vendor/quad3d/rules.py (unchanged)',
                                                          gl64_crosscheck='equal'), indent=1) + '\n')


if __name__ == '__main__':
    main()
