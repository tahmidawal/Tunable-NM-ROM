"""Train the four operator families for heat 2D at ONE target mesh, one after another, in one job.

Protocol (Burgers operator panel, exp/2026-09-22-ops-timing-panel / ops-tune-grid, adapted; the
deviations are declared in DESIGN.md section 5):
* published configurations per family, batch 8, AdamW(lr 1e-3, wd 1e-4), gradient clip 1.0,
  a fixed wall budget per arm (Burgers: 3000 s), checkpoint = best validation mean-case-max;
* training draws: the NM-ROM's own training seed (791000, 512 draws), so the operators see the
  same initial conditions our bank and head were trained on (data parity); validation: the
  NM-ROM's validation seed (791001, 16 draws). No evaluation cohort is generated here;
* DEVIATION (declared): learning rate follows a cosine in ELAPSED WALL TIME to 1e-5 (the Burgers
  plateau rule counts epochs, and an epoch here costs 16-64x a Burgers epoch, so it would never
  fire); Transolver keeps its warm-up as 160 optimisation steps (= Burgers' 10 epochs x 16 steps);
* DEVIATION (declared): micro-batches with gradient accumulation keep the effective batch at 8
  where 8 does not fit; the chosen micro-batch is recorded;
* DEVIATION (declared): Transolver patch = n/64, so the token grid stays 65x65 as at Burgers 256^2.
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

import heatops as H

STOP = False
EVAL_COHORTS = ((791099, 16), (790711, 4), (790716, 8))


def on_signal(signum, frame):
    global STOP
    STOP = True


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def batch(n, draws):
    traj = H.interior_fields(n, draws)                                # B 6 n-1 n-1 f64
    full = H.nodal(traj)
    return full[:, :1], full[:, 1:]


def normalisation(n, draws, chunk=16):
    s1 = s2 = t2 = 0.0
    count = tcount = 0
    for i in range(0, len(draws), chunk):
        x, y = batch(n, draws[i:i + chunk])
        s1 += float(x.sum()); s2 += float(x.square().sum()); count += x.numel()
        t2 += float(y.square().sum()); tcount += y.numel()
    mean = s1 / count
    std = math.sqrt(max(s2 / count - mean * mean, 1e-24))
    scale = math.sqrt(t2 / tcount)
    return tuple(torch.tensor(v, dtype=torch.float64, device='cuda') for v in (mean, std, scale))


def evaluate(net, n, draws, norm):
    net.eval()
    errors = []
    with torch.no_grad():
        for i in range(len(draws)):
            x, y = batch(n, draws[i:i + 1])
            pred = H.predict(net, x, *norm)
            errors.append(H.current_relative(pred[:, 1:], y)[0].cpu().numpy())
    e = np.asarray(errors)
    if not np.isfinite(e).all():
        raise RuntimeError('nonfinite validation error')
    per_case = e.max(1)
    return dict(errors=e.tolist(), mean_case_max=float(per_case.mean()), median_case_max=float(np.median(per_case)),
                worst_case_max=float(per_case.max()))


def probe_micro(config, n, draws, norm, candidates=(8, 4, 2, 1)):
    for m in candidates:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        net = H.make_model(config)
        try:
            x, y = batch(n, draws[:m])
            loss = H.current_relative(H.predict(net, x, *norm)[:, 1:], y).square().mean()
            loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            del net, x, y, loss
            torch.cuda.empty_cache()
            if peak < 0.80 * torch.cuda.get_device_properties(0).total_memory or m == 1:
                return m, peak
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not isinstance(exc, torch.OutOfMemoryError) and not any(k in str(exc) for k in ('out of memory', 'ALLOC', 'CUBLAS_STATUS_ALLOC')):
                raise
            print('probe micro-batch', m, 'does not fit:', type(exc).__name__, flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    raise RuntimeError('no micro-batch fits')


def train_arm(name, config, n, train, val, norm, out, wall, smoke):
    arm = out / name
    arm.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    torch.cuda.reset_peak_memory_stats()
    micro, probe_peak = probe_micro(config, n, train, norm, (1,) if smoke else (8, 4, 2, 1))
    torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    net = H.make_model(config)
    H.check_dtypes(net)
    opt = torch.optim.AdamW(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    lr0, floor = config['learning_rate'], 1e-5
    warm = int(config.get('warmup_steps', 0))
    gen = torch.Generator().manual_seed(config['seed'])
    best, history, step, stop_reason = float('inf'), [], 0, 'epoch_cap'
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(config['epochs']):
        net.train()
        order = torch.randperm(len(train), generator=gen).numpy()
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
                x, y = batch(n, train[sub])
                loss = H.current_relative(H.predict(net, x, *norm)[:, 1:], y).square().mean() * len(sub) / len(ids)
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
        metrics = evaluate(net, n, val, norm)
        improved = metrics['mean_case_max'] < best
        if improved:
            best = metrics['mean_case_max']
            ck = dict(model=net.state_dict(), config=config, epoch=epoch, step=step, best=best, mesh=n,
                      normalization=[v.cpu() for v in norm], family=H.family_of(config))
            torch.save(ck, arm / 'best.tmp'); (arm / 'best.tmp').replace(arm / 'best.pt')
        history.append(dict(epoch=epoch, steps=step, train_loss=float(np.mean(losses)), validation=metrics, learning_rate=lr,
                            wall_seconds=time.monotonic() - started, best_so_far=best,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        dump(arm / 'history.json', history)
        print(json.dumps(dict(arm=name, epoch=epoch, steps=step, train=history[-1]['train_loss'],
                              val_mean_case_max=metrics['mean_case_max'], val_worst=metrics['worst_case_max'],
                              lr=lr, wall=round(history[-1]['wall_seconds'], 1))), flush=True)
        if STOP:
            stop_reason = 'signal'; break
        if time.monotonic() - started >= wall:
            stop_reason = 'wall_budget'; break
    seconds = time.monotonic() - started
    net2, norm2, ck = H.load(arm / 'best.pt')
    final = evaluate(net2, n, val, norm2)
    result = dict(complete=True, name=name, family=H.family_of(config), config=config, mesh=n, micro_batch=micro,
                  probe_peak_bytes=probe_peak, best_epoch=ck['epoch'], best_step=ck['step'], epochs_completed=len(history),
                  optimisation_steps=step, stop_reason=stop_reason, training_seconds=seconds, wall_budget_seconds=wall,
                  validation_best_checkpoint=final, best_checkpoint_sha256=sha(arm / 'best.pt'),
                  parameter_dtype=str(getattr(net2, 'parameter_dtype', torch.float64)),
                  real_parameter_count=sum(p.numel() * (2 if p.is_complex() else 1) for p in net2.parameters()),
                  normalization=[float(v) for v in norm])
    dump(arm / 'result.json', result)
    del net, net2, opt
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh', type=int, required=True)
    ap.add_argument('--configs', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--wall-seconds', type=float, default=3000.)
    ap.add_argument('--train', type=int, nargs=2, default=(791000, 512))
    ap.add_argument('--validation', type=int, nargs=2, default=(791001, 16))
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if not args.smoke and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('production training runs in a cluster allocation')
    signal.signal(signal.SIGTERM, on_signal); signal.signal(signal.SIGUSR1, on_signal)
    env = H.configure()
    n = args.mesh
    train = H.family(*args.train); val = H.family(*args.validation)
    evald = np.concatenate([H.family(s, c) for s, c in EVAL_COHORTS])
    for a in (train, val):
        assert not any(np.array_equal(r, e) for r in a for e in evald), 'training/validation draw equals an evaluation draw'
    assert not any(np.array_equal(r, e) for r in train for e in val)
    args.out.mkdir(parents=True, exist_ok=True)
    norm = normalisation(n, train)
    here = Path(__file__).resolve().parent
    configs = {}
    for c in args.configs:
        cfg = json.loads(c.read_text())
        if cfg.get('family') == 'transolver':
            cfg['patch'] = max(1, n // 64)
        if args.smoke:
            cfg = dict(cfg, **cfg.get('smoke_overrides', {}))
        configs[c.stem] = cfg
    dump(args.out / 'provenance.json', dict(environment=env, mesh=n, train=args.train, validation=args.validation,
         sources={p.name: sha(p) for p in sorted(here.glob('*.py'))}, configs=configs,
         config_sha256={c.stem: sha(c) for c in args.configs}, wall_seconds=args.wall_seconds,
         job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'),
         normalization=[float(v) for v in norm], evaluation_cohorts_generated=False))
    results = {}
    for name, cfg in configs.items():
        if STOP:
            break
        results[name] = train_arm(name, cfg, n, train, val, norm, args.out, args.wall_seconds, args.smoke)
        print('ARM DONE', name, json.dumps({k: results[name][k] for k in ('stop_reason', 'epochs_completed', 'optimisation_steps', 'micro_batch')}),
              'val_mean_case_max', results[name]['validation_best_checkpoint']['mean_case_max'], flush=True)
    dump(args.out / 'summary.json', dict(complete=len(results) == len(configs), results=results))
    print('TRAIN OPS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
