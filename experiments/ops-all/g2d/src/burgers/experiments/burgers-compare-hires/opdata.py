"""Operator training / validation data at the TARGET mesh (burgers-compare-hires, DESIGN.md section 4).

The 256^2 operator panel trained on the pinned 128-case training index (SHA256 5333584b...) and selected on
the pinned 32-case validation index (468b9e70...). This script keeps exactly those cases -- the same case ids,
seeds and generation descriptors, read from the pinned indices, never re-drawn (the GB10/cluster 1-ulp `exp`
landmine) -- and regenerates each trajectory at the target mesh with the SAME solver setting the panel uses
as its same-grid reference: Newton-BiCGStab, FFT-Helmholtz preconditioner, dt 0.005, ntol 1e-6, ltol 1e-8
(`fft_tight`, iterative_paths.make_fom). So the operators are trained on exactly the field every arm is graded
against, at the mesh they are graded at.

    python opdata.py --mesh 1024 --out data

Writes data/train/index.json, data/validation/index.json and one NPZ per case in the model-facing contract of
ops/dataset.py (input (1,L+1,L+1), target (6,1,L+1,L+1), parameters [nu], times), all float64.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import iterative_paths as ip

TIMES = np.array([0., .05, .1, .15, .2, .25])
PINNED = {'train': '5333584b7162df622ec7bc2b08e68d8003036b1b05abbea520249ed408fc8d49',
          'validation': '468b9e70df3392c4b5bbb41381c9062d3d697ac4772b7236c4ee59d0ca73ebad'}
DEV6_SHA = '108f12dc8f9e6a64daf94f55861c616a29810134a52726e8a683a2a43892dc8a'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh', type=int, required=True)
    p.add_argument('--pinned', default='inputs/pinned')
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int, default=None, help='local smoke only')
    p.add_argument('--allow-cpu', action='store_true')
    a = p.parse_args()
    if not a.allow_cpu:
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    print(f'jax_backend={jax.default_backend()} x64={jax.config.jax_enable_x64}', flush=True)
    L, dt, ntol, ltol = a.mesh, .005, 1e-6, 1e-8
    fn, pre = ip.make_fom(L, dt, 'fft')
    out = Path(a.out)
    # the dev6 evaluation cohort must be disjoint from everything trained on (checked on descriptors)
    dev6 = np.concatenate((e.params_draw(7090702, 4), e.params_draw(911702, 2)))
    summary = dict(mesh=L, dt=dt, ntol=ntol, ltol=ltol, solver='iterative_paths.make_fom(L, dt, "fft") = fft_tight',
                   dev6_sha256=hashlib.sha256(np.ascontiguousarray(dev6).tobytes()).hexdigest(), splits={})
    for split in ('train', 'validation'):
        src = Path(a.pinned) / f'{split}-index.json'
        got = sha(src)
        assert got == PINNED[split], (split, got)
        index = json.loads(src.read_text())
        recs = index['records'][:a.limit] if a.limit else index['records']
        d = out / split
        d.mkdir(parents=True, exist_ok=False)
        rows, secs, worst = [], [], 0.
        for r in recs:
            g = r['generation_descriptors']
            phys = np.array([g['cx'], g['cy'], g['width'], g['amplitude'], g['nu']], dtype=np.float64)
            assert not np.any(np.all(np.isclose(dev6, phys[None]), axis=1)), ('dev6 overlap', r['case_id'])
            u0 = e.initial(L, phys)
            t0 = time.perf_counter()
            v = fn(jnp.asarray(u0), float(phys[4]), ntol, ltol, *pre)
            jax.block_until_ready(v)
            secs.append(time.perf_counter() - t0)
            f = np.asarray(v[0], dtype=np.float64)
            rn = np.asarray(v[2])
            assert f.shape == (6, L + 1, L + 1) and np.isfinite(f).all(), f.shape
            assert rn.max() <= ntol * (1 + 1e-9), ('not converged', r['case_id'], float(rn.max()))
            worst = max(worst, float(rn.max()))
            for edge in (f[:, 0, :], f[:, -1, :], f[:, :, 0], f[:, :, -1]):
                assert not np.any(edge), 'boundary must be exactly zero'
            target = f[:, None]
            x = np.ascontiguousarray(target[0])
            path = d / f"{r['case_id']}.npz"
            np.savez(path, input=x, target=target, parameters=np.array([phys[4]]), times=TIMES)
            rows.append(dict(case_id=r['case_id'], split=split, case_index=r['case_index'], seed=r['seed'],
                             path=path.name, sha256=sha(path), mesh=L, generation_descriptors=g,
                             reference=dict(intervals=L, dt=dt, ntol=ntol, ltol=ltol, preconditioner='fft',
                                            max_relative_residual=float(rn.max()),
                                            newton_total=int(np.sum(np.asarray(v[1]))), seconds=secs[-1])))
            print('CASE', split, r['case_id'], round(secs[-1], 2), flush=True)
        (d / 'index.json').write_text(json.dumps(dict(
            schema_version=1, pde='burgers', kind='burgers-compare-hires target-mesh regeneration', split=split,
            count=len(rows), mesh=L, complete=True, source_index_sha256=got, records=rows,
            derivation=('the pinned 256^2 operator cases (same ids, seeds, descriptors) solved at the target mesh '
                        'with the panel\'s own same-grid reference setting fft_tight')), indent=1) + '\n')
        summary['splits'][split] = dict(cases=len(rows), source_index_sha256=got, index_sha256=sha(d / 'index.json'),
                                        median_seconds=float(np.median(secs)), max_seconds=float(np.max(secs)),
                                        worst_relative_residual=worst)
    (out / 'opdata-summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print('OPDATA DONE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
