"""Train ONE operator network for Poisson 2D (square or L-shape) at ONE target mesh.

Adapted from exp/2026-09-23-heat-compare-hires @ 9a5baf9a `ops/train_ops.py` (same optimiser, wall-time
cosine learning rate, gradient clip, micro-batch probe with gradient accumulation, best-validation
checkpoint). Changes: the Poisson contract (pops.py), one network per job, training draws = the fit split of
the NM-ROM's training cohort (optionally only its first `--train-count` draws, declared), validation = the
first `--val-count` draws of the NM-ROM's validation split. No evaluation draw is generated here except to
assert disjointness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

import pops as P

STOP = False


def on_signal(signum, frame):
    global STOP
    STOP = True


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Data:
    """Sources are analytic (regenerated per batch); targets exact DST (square, per batch) or
    precomputed batched CG (L-shape, stored on the host in float32 for training, float64 for validation)."""

    def __init__(self, problem, n, train, val, cg_batch):
        self.problem, self.n, self.train, self.val = problem, n, train, val
        self.mask = P.domain_mask(problem, n)
        self.stats = {}
        if problem == 'lshape':
            t0 = time.monotonic()
            self.train_targets = torch.empty((len(train), n + 1, n + 1), dtype=torch.float32, pin_memory=True)
            for s in range(0, len(train), cg_batch):
                u = P.solve(problem, P.sources(problem, n, train[s:s + cg_batch]))
                self.train_targets[s:s + len(u)] = u.float().cpu()
                print('CG data', s + len(u), '/', len(train), round(time.monotonic() - t0, 1), flush=True)
            self.val_targets = torch.cat([P.solve(problem, P.sources(problem, n, val[s:s + cg_batch])).cpu()
                                          for s in range(0, len(val), cg_batch)])
            self.stats['data_seconds'] = time.monotonic() - t0

    def batch(self, ids, split='train'):
        draws = (self.train if split == 'train' else self.val)[ids]
        f = P.sources(self.problem, self.n, draws)
        if self.problem == 'square':
            return f, P.square_solve(f)
        tgt = self.train_targets if split == 'train' else self.val_targets
        return f, tgt[torch.as_tensor(ids)].cuda(non_blocking=True).double()


def normalisation(data, count=64, chunk=8):
    s1 = s2 = t2 = 0.0
    c = tc = 0
    for i in range(0, count, chunk):
        ids = np.arange(i, min(i + chunk, count, len(data.train)))
        if len(ids) == 0:
            break
        f, u = data.batch(ids)
        inside = data.mask.bool().expand_as(f)
        s1 += float(f[inside].sum()); s2 += float(f[inside].square().sum()); c += int(inside.sum())
        t2 += float(u[inside].square().sum()); tc += int(inside.sum())
    mean = s1 / c
    std = math.sqrt(max(s2 / c - mean * mean, 1e-24))
    scale = math.sqrt(t2 / tc)
    return tuple(torch.tensor(v, dtype=torch.float64, device='cuda') for v in (mean, std, scale))


def evaluate(net, data, norm):
    net.eval()
    errors = []
    with torch.no_grad():
        for i in range(len(data.val)):
            f, u = data.batch(np.array([i]), 'val')
            errors.append(float(P.relative(P.predict(net, f, *norm, data.mask), u)[0]))
    e = np.asarray(errors)
    if not np.isfinite(e).all():
        raise RuntimeError('nonfinite validation error')
    return dict(errors=e.tolist(), mean=float(e.mean()), median=float(np.median(e)), worst=float(e.max()))


MEM_SHARE = 1.0   # fraction of the device this process may use (co-scheduled networks, declared)


def probe_micro(config, data, norm, candidates=(8, 4, 2, 1)):
    for m in candidates:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            net = P.make_model(config)
            f, u = data.batch(np.arange(m))
            loss = P.relative(P.predict(net, f, *norm, data.mask), u).square().mean()
            loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            del net, f, u, loss
            torch.cuda.empty_cache()
            if peak < 0.80 * MEM_SHARE * torch.cuda.get_device_properties(0).total_memory or m == 1:
                return m, peak
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not isinstance(exc, torch.OutOfMemoryError) and not any(k in str(exc) for k in ('out of memory', 'ALLOC', 'CUBLAS_STATUS_ALLOC', 'CUDNN_STATUS')):
                raise
            print('probe micro-batch', m, 'does not fit:', type(exc).__name__, str(exc)[:200], flush=True)
            net = f = u = loss = None
        import gc
        gc.collect()          # free the failed attempt's graph before the next (smaller) probe
        torch.cuda.empty_cache()
        print('probe after cleanup: allocated', torch.cuda.memory_allocated(), flush=True)
    raise torch.OutOfMemoryError('no micro-batch fits (micro-batch 1 out of memory)')


def train_net(name, config, data, norm, arm, wall):
    arm.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    micro, probe_peak = probe_micro(config, data, norm)
    torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    net = P.make_model(config)
    P.check_dtypes(net)
    opt = torch.optim.AdamW(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    lr0, floor = config['learning_rate'], 1e-5
    warm = int(config.get('warmup_steps', 0))
    gen = torch.Generator().manual_seed(config['seed'])
    best, history, step, stop_reason = float('inf'), [], 0, 'epoch_cap'
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    ntrain = len(data.train)
    for epoch in range(config['epochs']):
        net.train()
        order = torch.randperm(ntrain, generator=gen).numpy()
        losses = []
        for s in range(0, len(order), config['batch_size']):
            ids = order[s:s + config['batch_size']]
            frac = min(1.0, (time.monotonic() - started) / wall)
            lr = floor + (lr0 - floor) * 0.5 * (1 + math.cos(math.pi * frac))
            if step < warm:
                lr = lr * (step + 1) / warm
            for g in opt.param_groups:
                g['lr'] = lr
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for m in range(0, len(ids), micro):
                sub = ids[m:m + micro]
                f, u = data.batch(sub)
                loss = P.relative(P.predict(net, f, *norm, data.mask), u).square().mean() * len(sub) / len(ids)
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite training loss')
                loss.backward()
                total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            step += 1
            losses.append(total)
            if STOP or time.monotonic() - started >= wall:
                break
        metrics = evaluate(net, data, norm)
        if metrics['mean'] < best:
            best = metrics['mean']
            ck = dict(model=net.state_dict(), config=config, epoch=epoch, step=step, best=best, mesh=data.n,
                      problem=data.problem, normalization=[v.cpu() for v in norm], family=P.family_of(config))
            torch.save(ck, arm / 'best.tmp'); (arm / 'best.tmp').replace(arm / 'best.pt')
        history.append(dict(epoch=epoch, steps=step, train_loss=float(np.mean(losses)), validation=metrics,
                            learning_rate=lr, wall_seconds=time.monotonic() - started, best_so_far=best,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        dump(arm / 'history.json', history)
        print(json.dumps(dict(arm=name, epoch=epoch, steps=step, train=history[-1]['train_loss'],
                              val_mean=metrics['mean'], val_worst=metrics['worst'], lr=lr,
                              wall=round(history[-1]['wall_seconds'], 1))), flush=True)
        if STOP:
            stop_reason = 'signal'; break
        if time.monotonic() - started >= wall:
            stop_reason = 'wall_budget'; break
    seconds = time.monotonic() - started
    net2, norm2, ck = P.load(arm / 'best.pt')
    final = evaluate(net2, data, norm2)
    return dict(complete=True, name=name, family=P.family_of(config), config=config, mesh=data.n,
                problem=data.problem, micro_batch=micro, probe_peak_bytes=probe_peak, best_epoch=ck['epoch'],
                best_step=ck['step'], epochs_completed=len(history), optimisation_steps=step,
                stop_reason=stop_reason, training_seconds=seconds, wall_budget_seconds=wall,
                validation_best_checkpoint=final, best_checkpoint_sha256=sha(arm / 'best.pt'),
                parameter_dtype=str(getattr(net2, 'parameter_dtype', torch.float64)),
                real_parameter_count=sum(p.numel() * (2 if p.is_complex() else 1) for p in net2.parameters()),
                normalization=[float(v) for v in norm], peak_allocated_bytes=torch.cuda.max_memory_allocated())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problem', choices=['square', 'lshape'], required=True)
    ap.add_argument('--mesh', type=int, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--wall-seconds', type=float, default=3000.)
    ap.add_argument('--train-count', type=int, default=0, help='0 = the whole fit split')
    ap.add_argument('--val-count', type=int, default=16)
    ap.add_argument('--cg-batch', type=int, default=32)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--mem-share', type=float, default=1.0, help='per-process device memory fraction (co-scheduled jobs)')
    ap.add_argument('--co-scheduled', default='', help='names of networks sharing this GPU (recorded)')
    a = ap.parse_args()
    global MEM_SHARE
    MEM_SHARE = a.mem_share
    if not a.smoke and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('production training runs in a cluster allocation')
    signal.signal(signal.SIGTERM, on_signal); signal.signal(signal.SIGUSR1, on_signal)
    env = P.configure()
    if a.mem_share < 1.0:
        torch.cuda.set_per_process_memory_fraction(a.mem_share)
    n = a.mesh
    train, val = P.training_draws(a.problem)
    evald = P.evaluation_draws(a.problem)
    for s in (train, val):
        assert not any(np.allclose(r, e) for r in s for e in evald), 'training/validation draw equals an evaluation draw'
    full_train = len(train)
    if a.train_count:
        train = train[:a.train_count]
    val = val[:a.val_count]
    cfg = json.loads(a.config.read_text())
    if cfg.get('family') == 'transolver':
        cfg['patch'] = max(1, n // 64)
    if a.smoke:
        cfg = dict(cfg, **cfg.get('smoke_overrides', {}))
    a.out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    prov = dict(environment=env, problem=a.problem, mesh=n, train_count=len(train), fit_split_size=full_train,
                val_count=len(val), train_sha=hashlib.sha256(np.ascontiguousarray(train).tobytes()).hexdigest(),
                sources={p.name: sha(p) for p in sorted(here.glob('*.py'))}, config=cfg,
                config_sha256=sha(a.config), wall_seconds=a.wall_seconds, job_id=os.environ.get('SLURM_JOB_ID'),
                node=os.environ.get('SLURMD_NODENAME'), evaluation_cohort_used_for_training=False,
                mem_share=a.mem_share, co_scheduled_on_same_gpu=[x for x in a.co_scheduled.split(',') if x])
    dump(a.out / 'provenance.json', prov)
    name = a.config.stem
    result = dict(complete=False, name=name, family=P.family_of(cfg), problem=a.problem, mesh=n, mem_share=a.mem_share,
                  co_scheduled_on_same_gpu=[x for x in a.co_scheduled.split(',') if x])
    try:
        t0 = time.monotonic()
        data = Data(a.problem, n, train, val, a.cg_batch)
        norm = normalisation(data)
        result['data'] = dict(seconds=time.monotonic() - t0, **data.stats)
        print('DATA READY', json.dumps(result['data']), [float(v) for v in norm], flush=True)
        result.update(train_net(name, cfg, data, norm, a.out / name, a.wall_seconds))
    except torch.OutOfMemoryError as exc:
        result.update(failure='out_of_memory', failure_message=str(exc)[:2000],
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
        print('FAILED out_of_memory', str(exc)[:500], flush=True)
    dump(a.out / 'result.json', result)
    print('TRAIN OP COMPLETE' if result['complete'] else 'TRAIN OP FAILED', flush=True)


if __name__ == '__main__':
    main()
