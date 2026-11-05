"""Generate the 3D rule ladder (DESIGN §4, A2.5) once, locally, with the vendored quadrature-burgers3d `rules.py` generators
(unchanged), and write rules3d.npz + rules3d_manifest.json. Rules shared with October are checked against the October
rules.json sha256 (same generator, same seeds), and Hari's CBC vectors are asserted for n = 4096 and 32768."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QB = HERE / 'vendor/tree/experiments/quadrature-burgers3d'
sys.path.insert(0, str(QB))
import rules as RL  # noqa: E402
from make_configs import CBC3, GL3, SOB3, KOR3  # noqa: E402


def main():
    rules, meta = {}, {}
    old = json.loads((QB / 'rules/rules.json').read_text())['rules']

    def add(name, family, X, w, **extra):
        X = np.ascontiguousarray(X, dtype=np.float64)
        w = np.ascontiguousarray(w, dtype=np.float64)
        assert X.ndim == 2 and X.shape[1] == 3 and len(w) == len(X) and np.all((X >= 0) & (X <= 1)), name
        h = hashlib.sha256(X.tobytes() + w.tobytes()).hexdigest()
        if name in old:
            assert old[name]['sha256'] == h, ('differs from October', name)
        rules[name] = (X, w)
        meta[name] = dict(family=family, m=int(len(w)), weight_sum=float(w.sum()), sha256=h,
                          october_identical=name in old, **extra)
        print(name, len(w), flush=True)

    for n in CBC3:
        z, P = RL.cbc_lattice_vector3(n)
        if n in RL.HARI_CBC:
            assert tuple(int(v) for v in z) == RL.HARI_CBC[n], (n, z)
        add(f'lat{n}', 'lattice', *RL.rank1_lattice3(n, z, seed=0), z=[int(v) for v in z], p2=P, shift_seed=0)
    for p in GL3:
        add(f'gl{p}', 'gauss', *RL.gauss_tensor3(p), p=p)
    for m in SOB3:
        add(f'sob{m}', 'sobol', *RL.sobol3(m, seed=0), scramble_seed=0)
    for n in KOR3:
        z, P = RL.korobov_search3(n)
        add(f'kor{n}', 'korobov', *RL.rank1_lattice3(n, z, seed=0), z=[int(v) for v in z], p2=P, shift_seed=0)
    add('smol8', 'control', *RL.smolyak3(8), level=8)
    out = HERE / 'rules3d.npz'
    np.savez(out, **{f'{k}_X': v[0] for k, v in rules.items()}, **{f'{k}_w': v[1] for k, v in rules.items()})
    (HERE / 'rules3d_manifest.json').write_text(json.dumps(dict(
        note='3D rules generated locally by vendored rules.py; sha256(X.tobytes() + w.tobytes())',
        npz_sha256=hashlib.sha256(out.read_bytes()).hexdigest(), rules=meta), indent=1) + '\n')


if __name__ == '__main__':
    main()
