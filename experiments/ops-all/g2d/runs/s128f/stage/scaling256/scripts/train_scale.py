"""g2d scaling256: train one operator at 256^2 on the first N trajectories of the pinned generator (DESIGN.md).

    python train_scale.py --arm fno-w96f32 --n-train 576 --out <dir> [--wall-seconds 7200]
    (run from experiments/burgers-compare-hires; PYTHONPATH as in the g2d submit script)

Derived from ../../scripts/train_mem.py. Data: training case i has seed SeedSequence([20260914, 22, 1, i]) (the pinned
index's `case_seed`), descriptors params_draw(seed, 1); the first 128 are asserted equal to the pinned train index;
validation = the pinned 32. Every trajectory solved at 256 intervals with fft_tight, in host memory. Training:
ops/train.py `train()` unchanged (same substitutions as train_mem.py), config = the arm's config with both patiences
held constant in optimisation steps. Afterwards the best checkpoint is scored on the training and validation sets
with train.py's own `evaluate` (per-case max over times of ||u-u_ref||/||u0||; worst/mean/median over cases).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def case_seed(i):
    return int(np.random.SeedSequence([20260914, 22, 1, i]).generate_state(1, dtype=np.uint32)[0])


def generate(L, n_train):
    assert jax.default_backend() == 'gpu' or os.environ.get('G2D_ALLOW_CPU')
    fn, pre = ip.make_fom(L, .005, 'fft')
    ntol, ltol = 1e-6, 1e-8
    dev6 = np.concatenate((e.params_draw(7090702, 4), e.params_draw(911702, 2)))
    pin = {s: json.loads((Path('inputs/pinned') / f'{s}-index.json').read_text())['records'] for s in PINNED}
    for s in PINNED:
        assert sha(Path('inputs/pinned') / f'{s}-index.json') == PINNED[s], s
    rows = {'train': [], 'validation': []}
    for i in range(n_train):
        sd = case_seed(i)
        ph = e.params_draw(sd, 1)[0]
        g = dict(zip(('cx', 'cy', 'width', 'amplitude', 'nu'), map(float, ph)))
        if i < 128:
            assert pin['train'][i]['seed'] == sd, i
            ref = np.array([pin['train'][i]['generation_descriptors'][k] for k in ('cx', 'cy', 'width', 'amplitude', 'nu')])
            assert np.allclose(ph, ref, rtol=1e-12, atol=0), (i, ph, ref)
        rows['train'].append(dict(case_id=f'burgers-train-{i:05d}', split='train', case_index=i, seed=sd, mesh=L,
                                  generation_descriptors=g, phys=ph))
    for r in pin['validation']:
        g = r['generation_descriptors']
        ph = np.array([g[k] for k in ('cx', 'cy', 'width', 'amplitude', 'nu')])
        rows['validation'].append(dict(case_id=r['case_id'], split='validation', case_index=r['case_index'], seed=r['seed'],
                                       mesh=L, generation_descriptors=g, phys=ph))
    vset = {r['seed'] for r in rows['validation']}
    assert not vset & {r['seed'] for r in rows['train']}, 'train/validation seed overlap'
    for r in rows['train'] + rows['validation']:
        assert not np.any(np.all(np.isclose(dev6, r['phys'][None]), axis=1)), ('dev6 overlap', r['case_id'])
    data, summary = {}, {}
    for split, rs in rows.items():
        target = np.empty((len(rs), 6, 1, L + 1, L + 1), dtype=np.float64)
        params = np.empty((len(rs), 1), dtype=np.float64)
        secs = []
        for i, r in enumerate(rs):
            ph = r.pop('phys')
            t0 = time.perf_counter()
            v = fn(jnp.asarray(e.initial(L, ph)), float(ph[4]), ntol, ltol, *pre)
            f = np.asarray(v[0], dtype=np.float64)
            rn = np.asarray(v[2])
            secs.append(time.perf_counter() - t0)
            assert f.shape == (6, L + 1, L + 1) and np.isfinite(f).all()
            assert rn.max() <= ntol * (1 + 1e-9), ('not converged', r['case_id'], float(rn.max()))
            for edge in (f[:, 0, :], f[:, -1, :], f[:, :, 0], f[:, :, -1]):
                assert not np.any(edge)
            target[i, :, 0] = f
            params[i, 0] = ph[4]
            r.update(field_sha256=hashlib.sha256(f.tobytes()).hexdigest(), max_relative_residual=float(rn.max()))
        data[split] = dict(target=target, parameters=params)
        summary[split] = dict(cases=len(rs), median_seconds=float(np.median(secs)), host_bytes=int(target.nbytes))
    return data, rows, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', required=True)
    p.add_argument('--n-train', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--wall-seconds', type=float, default=7200.)
    p.add_argument('--mesh', type=int, default=256)
    a = p.parse_args()
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    data, recs, summary = generate(a.mesh, a.n_train)
    summary.update(mesh=a.mesh, n_train=a.n_train, seconds=time.perf_counter() - t0,
                   solver='iterative_paths.make_fom(L, dt, "fft") = fft_tight at the target mesh', storage='host memory only')
    (out / 'data-summary.json').write_text(json.dumps(dict(summary=summary, records=recs), indent=1) + '\n')
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
    ids = {s: [r['case_id'] for r in recs[s]] for s in recs}

    def load_pair(train_path, validation_path):
        return 'burgers', recs['train'], recs['validation']

    def arrays(records):
        key = 'train' if [r['case_id'] for r in records] == ids['train'] else 'validation'
        assert [r['case_id'] for r in records] == ids[key]
        return tensors[key]

    def normalization(d, chunk=64):
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

    cfg = json.loads((Path('inputs/opconfigs') / f'{a.arm}.json').read_text())
    spe = math.ceil(a.n_train / cfg['batch_size'])
    base_spe = math.ceil(128 / cfg['batch_size'])
    cfg['plateau_patience'] = max(1, round(int(cfg.get('plateau_patience', 20)) * base_spe / spe))
    cfg['patience'] = max(1, round(int(cfg['patience']) * base_spe / spe))
    cfg['g2d_scaling'] = dict(n_train=a.n_train, steps_per_epoch=spe, note='patiences held constant in optimisation steps (N=128 values)')
    cfgp = out / f'config-{a.arm}-N{a.n_train}.json'
    cfgp.write_text(json.dumps(cfg, indent=1) + '\n')
    args = argparse.Namespace(train_index=Path('inputs/pinned/train-index.json'),
                              validation_index=Path('inputs/pinned/validation-index.json'),
                              config=cfgp, out=out / 'train' / a.arm, wall_seconds=a.wall_seconds, smoke=False)
    rec = dict(arm=a.arm, n_train=a.n_train, config=cfg, job_id=os.environ.get('SLURM_JOB_ID'))
    try:
        T.train(args)
        r = json.loads((args.out / 'result.json').read_text())
        rec.update(ok=True, epochs=r['epochs_completed'], best_epoch=r['best_epoch'], stop_reason=r['stop_reason'],
                   optimisation_steps=r['optimisation_steps'], training_seconds=r['training_seconds'],
                   micro_batch_final=r.get('micro_batch_final'), best_checkpoint_sha256=r['best_checkpoint_sha256'])
        import model as adapter
        ck = torch.load(args.out / 'best.pt', map_location='cuda', weights_only=False)
        net = adapter.make_model('burgers', ck['config'])
        net.load_state_dict(ck['model'])
        net.eval()
        norm = tuple(v.cuda() for v in ck['normalization'])
        for split in ('train', 'validation'):
            m = T.evaluate(net, tensors[split], norm, 'burgers', 8)
            rec[f'{split}_best_checkpoint'] = {k: m[k] for k in ('worst_case_max', 'mean_case_max', 'median_case_max', 'p95_case_max')}
        hist = json.loads((args.out / 'history.json').read_text())
        rec['train_msre_best_epoch'] = hist[r['best_epoch']]['train_mean_squared_relative_error']
        rec['curve'] = [dict(epoch=h['epoch'], wall=h['wall_seconds'], train_msre=h['train_mean_squared_relative_error'],
                             val_mean=h['validation']['mean_case_max'], val_worst=h['validation']['worst_case_max'],
                             lr=h['learning_rate']) for h in hist]
    except Exception as exc:                                       # noqa: BLE001 (recorded)
        rec.update(ok=False, error_type=type(exc).__name__, error=str(exc)[:2000], traceback=traceback.format_exc()[-4000:])
    (out / 'scale.json').write_text(json.dumps(rec, indent=1) + '\n')
    print('SCALE DONE', json.dumps({k: v for k, v in rec.items() if k not in ('curve', 'traceback', 'config')}), flush=True)


if __name__ == '__main__':
    main()
