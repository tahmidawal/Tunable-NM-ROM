"""g2d: Burgers operator training at a mesh whose f64 training set cannot be written to the shared disk (4096^2).

    python train_mem.py --mesh 4096 --out <dir> --arms fno-large,unet-refine,tsol-refine,don-small [--wall-seconds 3000]
    (run from experiments/burgers-compare-hires; PYTHONPATH as in the g2d submit script)

Data: exactly opdata.py (burgers-compare-hires @ 07c0e3e8): the pinned 128 training / 32 validation cases (index
SHA256 asserted), each solved at the target mesh with fft_tight (Newton-BiCGStab, FFT preconditioner, dt 0.005,
ntol 1e-6, ltol 1e-8), convergence and exact-zero boundary asserted, dev6 disjointness asserted -- but written into
preallocated host float64 arrays instead of per-case NPZ files (129 GB at 4096^2; the group share has ~370 GB free
for four groups). Records carry the same case ids / seeds / descriptors; per-case SHA256 of the field bytes is recorded.

Training: ops/train.py `train()` UNCHANGED, called once per arm in this process with three substitutions made from
outside the module: (1) `dataset.load_pair` returns the in-memory records, (2) `train.arrays` returns the in-memory
arrays (torch views, no copy), (3) `train.normalization` is computed in chunks (same statistics up to summation order;
the unchunked version would need ~70 GB of temporaries), and (4) the per-case validation prediction NPZ files
(0.8 GB each) are not written. Everything else -- model, AdamW, schedule, micro-batching, 3000 s wall budget, best
validation checkpoint, result.json -- is train.py's own code. An arm that raises (e.g. CUDA OOM at micro-batch 1) is
recorded with its error and the next arm runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import iterative_paths as ip

PINNED = {'train': '5333584b7162df622ec7bc2b08e68d8003036b1b05abbea520249ed408fc8d49',
          'validation': '468b9e70df3392c4b5bbb41381c9062d3d697ac4772b7236c4ee59d0ca73ebad'}
TIMES = np.array([0., .05, .1, .15, .2, .25])


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def generate(L, limit=None):
    assert jax.default_backend() == 'gpu' or os.environ.get('G2D_ALLOW_CPU')
    fn, pre = ip.make_fom(L, .005, 'fft')
    ntol, ltol = 1e-6, 1e-8
    dev6 = np.concatenate((e.params_draw(7090702, 4), e.params_draw(911702, 2)))
    data, recs, summary = {}, {}, {}
    for split in ('train', 'validation'):
        src = Path('inputs/pinned') / f'{split}-index.json'
        assert sha(src) == PINNED[split], split
        rows = json.loads(src.read_text())['records']
        rows = rows[:limit] if limit else rows
        target = np.empty((len(rows), 6, 1, L + 1, L + 1), dtype=np.float64)
        params = np.empty((len(rows), 1), dtype=np.float64)
        out, secs = [], []
        for i, r in enumerate(rows):
            g = r['generation_descriptors']
            phys = np.array([g['cx'], g['cy'], g['width'], g['amplitude'], g['nu']], dtype=np.float64)
            assert not np.any(np.all(np.isclose(dev6, phys[None]), axis=1)), ('dev6 overlap', r['case_id'])
            t0 = time.perf_counter()
            v = fn(jnp.asarray(e.initial(L, phys)), float(phys[4]), ntol, ltol, *pre)
            f = np.asarray(v[0], dtype=np.float64)
            rn = np.asarray(v[2])
            del v
            secs.append(time.perf_counter() - t0)
            assert f.shape == (6, L + 1, L + 1) and np.isfinite(f).all()
            assert rn.max() <= ntol * (1 + 1e-9), ('not converged', r['case_id'], float(rn.max()))
            for edge in (f[:, 0, :], f[:, -1, :], f[:, :, 0], f[:, :, -1]):
                assert not np.any(edge)
            target[i, :, 0] = f
            params[i, 0] = phys[4]
            out.append(dict(case_id=r['case_id'], split=split, case_index=r['case_index'], seed=r['seed'], mesh=L,
                            generation_descriptors=g, field_sha256=hashlib.sha256(f.tobytes()).hexdigest(),
                            max_relative_residual=float(rn.max()), seconds=secs[-1]))
            del f
            print('CASE', split, r['case_id'], round(secs[-1], 2), flush=True)
        data[split] = dict(target=target, parameters=params)
        recs[split] = out
        summary[split] = dict(cases=len(out), source_index_sha256=PINNED[split], median_seconds=float(np.median(secs)),
                              host_bytes=int(target.nbytes))
    return data, recs, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--arms', required=True)
    p.add_argument('--wall-seconds', type=float, default=3000.)
    p.add_argument('--limit', type=int, default=None, help='local smoke only')
    a = p.parse_args()
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    data, recs, summary = generate(a.mesh, a.limit)
    summary.update(mesh=a.mesh, seconds=time.perf_counter() - t0, solver='iterative_paths.make_fom(L, dt, "fft") = fft_tight',
                   storage='host memory only (no per-case files)')
    (out / 'opdata-summary.json').write_text(json.dumps(dict(summary=summary, records=recs), indent=1) + '\n')
    print('DATA DONE', json.dumps(summary), flush=True)
    jax.clear_caches()

    import torch
    sys.path.insert(0, str(Path('ops').resolve()))
    import dataset
    import train as T

    tensors = {}
    for split in ('train', 'validation'):
        tgt = torch.from_numpy(data[split]['target'])
        tensors[split] = dict(target=tgt, input=tgt[:, 0], parameters=torch.from_numpy(data[split]['parameters']))
    ids = {split: [r['case_id'] for r in recs[split]] for split in recs}

    def load_pair(train_path, validation_path):
        return 'burgers', recs['train'], recs['validation']

    def arrays(records):
        key = 'train' if [r['case_id'] for r in records] == ids['train'] else 'validation'
        assert [r['case_id'] for r in records] == ids[key]
        return tensors[key]

    def normalization(d, chunk=8):
        x, pr, y = d['input'], d['parameters'], d['target']
        n, hw = x.shape[0], x.shape[-1] * x.shape[-2]
        s1 = s2 = t2 = 0.
        for i in range(0, n, chunk):
            xb = x[i:i + chunk].double()
            s1 += float(xb.sum()); s2 += float(xb.square().sum())
            t2 += float(y[i:i + chunk].square().sum())
        cnt = n * hw
        m0 = s1 / cnt
        v0 = (s2 - cnt * m0 * m0) / (cnt - 1)
        pv = pr[:, 0].double()
        m1 = float(pv.mean())
        v1 = float(((pv - m1) ** 2).sum()) * hw / (cnt - 1)
        mean = torch.tensor([m0, m1], dtype=torch.float64).reshape(1, 2, 1, 1)
        std = torch.tensor([max(v0, 0.) ** .5, max(v1, 0.) ** .5], dtype=torch.float64).reshape(1, 2, 1, 1).clamp_min(1e-12)
        scale = torch.tensor((t2 / y.numel()) ** .5, dtype=torch.float64).clamp_min(1e-12)
        return tuple(v.cuda() for v in (mean, std, scale))

    class NP:
        def __getattr__(self, k):
            return getattr(np, k)

        def savez(self, path, **kw):
            if str(path).endswith('.prediction.npz'):
                return
            return np.savez(path, **kw)

    dataset.load_pair = load_pair
    T.arrays = arrays
    T.normalization = normalization
    T.np = NP()

    records = []
    for arm in a.arms.split(','):
        cfgp = Path('inputs/opconfigs') / f'{arm}.json'
        args = argparse.Namespace(train_index=Path('inputs/pinned/train-index.json'),
                                  validation_index=Path('inputs/pinned/validation-index.json'),
                                  config=cfgp, out=out / 'train' / arm, wall_seconds=a.wall_seconds, smoke=False)
        start = time.time()
        rec = dict(arm=arm, config=str(cfgp), job_id=os.environ.get('SLURM_JOB_ID'), started_unix=start)
        try:
            T.train(args)
            r = json.loads((args.out / 'result.json').read_text())
            rec.update(ok=True, epochs=r['epochs_completed'], stop_reason=r['stop_reason'],
                       validation_mean_case_max=r['validation']['mean_case_max'],
                       validation_worst_case_max=r['validation']['worst_case_max'],
                       micro_batch_final=r.get('micro_batch_final'), best_checkpoint_sha256=r['best_checkpoint_sha256'])
        except Exception as exc:                                     # noqa: BLE001  (recorded, next arm runs)
            rec.update(ok=False, error_type=type(exc).__name__, error=str(exc)[:2000], traceback=traceback.format_exc()[-4000:])
            hist = args.out / 'history.json'
            rec['epochs_completed_before_error'] = len(json.loads(hist.read_text())) if hist.exists() else 0
        rec['ended_unix'] = time.time()
        records.append(rec)
        torch.cuda.empty_cache()
        (out / 'worker.json').write_text(json.dumps(dict(arms=records), indent=1) + '\n')
        print('ARM', json.dumps({k: v for k, v in rec.items() if k != 'traceback'}), flush=True)
    print('TRAIN_MEM FINISHED', flush=True)


if __name__ == '__main__':
    main()
