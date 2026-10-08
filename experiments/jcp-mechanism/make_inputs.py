"""jcp-mechanism: pack the saved reached states used by A2 (DESIGN.md section 4) into inputs/a2_states.npz, with the
sha256 of every source file in inputs/a2_states.json. Sources are read-only outputs of the two 2026-10-01 lanes:

  3D fixed  : validation job val65 (job 4732813), arm tensor_R{512,256}, cases 0-7, steps k = 1, 3, ..., 25 (13 per case)
  3D own    : val65 / val129 / val257, arm tensor_R{512,256}, cases 0-3, the same steps (each evaluated at its own mesh)
  2D fixed  : dev job dv1024 (job 4735709), population_{acc,fast} (lat64-reached), dev6 cases, steps k = 2, 4, ..., 50
  2D own    : dv256 / dv1024 / dv4096, population_{acc,fast}, dev6 cases, steps k = 3, 6, ..., 48

Usage: /home/tahmid/Dev/.venv/bin/python make_inputs.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parents[2]
Q3 = WT / '2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/runs'
Q2 = WT / '2026-10-01-quadrature-study/experiments/quadrature-study/runs'
K3 = list(range(1, 26, 2))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    arrs, prov = {}, {}
    for Rp in (512, 256):
        for tag, meshes, cases in (('fixed', ('65',), range(8)), ('own', ('65', '129', '257'), range(4))):
            for n in meshes:
                rows, lab = [], []
                for j in cases:
                    f = Q3 / f'val{n}/code/output/fields/tensor_R{Rp}_c{j}.npz'
                    prov[str(f.relative_to(WT))] = sha(f)
                    W = np.load(f)['internal']
                    assert W.shape == (26, Rp), W.shape
                    rows.append(W[K3])
                    lab += [f'{j}|{k}' for k in K3]
                arrs[f'd3_{tag}_n{n}_R{Rp}'] = np.concatenate(rows, 0)
                arrs[f'd3_{tag}_n{n}_R{Rp}_labels'] = np.array(lab)
    for s in ('acc', 'fast'):
        for tag, meshes, ks in (('fixed', ('1024',), range(2, 51, 2)), ('own', ('256', '1024', '4096'), range(3, 49, 3))):
            for L in meshes:
                f = Q2 / f'dv{L}/archive/output/population_{s}.npz'
                prov[str(f.relative_to(WT))] = sha(f)
                z = np.load(f)
                lab = [str(x) for x in z['labels']]
                want = [f'dev6|{c}|{k}' for c in range(6) for k in ks]
                idx = [lab.index(w) for w in want]
                arrs[f'd2_{tag}_L{L}_{s}'] = np.asarray(z['C'])[idx]
                arrs[f'd2_{tag}_L{L}_{s}_labels'] = np.array(want)
    out = HERE / 'inputs' / 'a2_states.npz'
    out.parent.mkdir(exist_ok=True)
    np.savez(out, **arrs)
    meta = dict(sources_sha256=prov, npz_sha256=sha(out), shapes={k: list(v.shape) for k, v in arrs.items()},
                doc=__doc__)
    (HERE / 'inputs' / 'a2_states.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(meta['shapes'], indent=1), meta['npz_sha256'])


if __name__ == '__main__':
    main()
