"""Train ONE operator arm for ONE (problem, mesh) cell. Several of these processes may share one GPU
(each with its own torch memory fraction, --mem-frac); every one keeps its own 3000 s wall budget.

Adapted from experiments/ns3d-operators/train_op.py (@ 766c3247): same optimiser and protocol —
AdamW(lr 1e-3, wd 1e-4), batch 8 (micro-batches + accumulation if 8 does not fit), grad clip 1.0, cosine
learning rate in elapsed wall time to 1e-5, Transolver warm-up 160 steps, 3000 s wall budget, epoch cap 4000,
patience 250 epochs, validation every epoch, checkpoint = best validation mean over cases of the per-case
metric (Poisson: relative L2 error; heat: max over output times of the same-grid relative L2 error).
Data: the NM-ROM's own training seed and draw count (Poisson 920410 x 512; heat 921000 x 2048), generated on
the fly on the GPU in float64 at the target mesh (problems.py); split index % 8 == 7 -> validation.
The evaluation cohorts (Poisson 920411 / 920499; heat 921777 / 921099) are never generated here; their rows
are asserted disjoint from the training draws.
Loss: mean over batch (and the five evolved times for heat) of the squared relative error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
STOP = False
VAL_EVERY = 8


def on_signal(signum, frame):
    global STOP
    STOP = True


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1) + '\n')
    tmp.replace(path)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--problem', choices=('poisson', 'heat'), required=True)
    ap.add_argument('--mesh', type=int, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--wall-seconds', type=float, default=3000.)
    ap.add_argument('--mem-frac', type=float, default=0.9)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--probe-only', action='store_true', help='memory probe at micro-batch 1 only, write probe.json, exit')
    args = ap.parse_args()
    if not args.smoke and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('production training runs in a cluster allocation')
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGUSR1, on_signal)
    import torch
    import ops_l3d as O
    import problems as PB
    env = O.configure()
    torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    budget_bytes = args.mem_frac * torch.cuda.get_device_properties(0).total_memory
    n = args.mesh
    prob = PB.Problem(args.problem, n)
    config = json.loads(args.config.read_text())
    name = args.config.stem
    config.update(cin=prob.cin, cout=prob.cout)
    if config['family'] == 'transolver':
        config['patch'] = max(1, n // 16)  # token grid 16^3 at every mesh
    if args.smoke:
        config = dict(config, **config.get('smoke_overrides', {}))
    args.out.mkdir(parents=True, exist_ok=True)
    for existing in ('best.pt', 'result.json', 'history.json'):
        if (args.out / existing).exists():
            raise RuntimeError(f'refusing: {args.out / existing} already exists')

    spec = PB.POISSON if args.problem == 'poisson' else PB.HEAT
    count = 64 if args.smoke else spec['train_count']
    draws = PB.family(spec['train_seed'], count)
    rows = {tuple(r) for r in np.round(draws, 12)}
    cohorts = ([('validation', spec['validation_seed'], spec['validation_count']), ('final', spec['final_seed'], spec['final_count'])]
               if args.problem == 'poisson' else
               [('validation', spec['validation_seed'], 16), ('heldout', spec['heldout_seed'], spec['heldout_count'])])
    overlap = {c: int(sum(tuple(r) in rows for r in np.round(PB.family(s, k), 12))) for c, s, k in cohorts}
    if any(overlap.values()):
        raise RuntimeError(f'training draws overlap an evaluation cohort: {overlap}')
    is_val = (np.arange(count) % VAL_EVERY) == VAL_EVERY - 1
    tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)

    # normalisation from the training split only (RMS of input and target fields)
    s_in = s_out = 0.0
    c_in = c_out = 0
    with torch.no_grad():
        nb = 16 if n <= 128 else 1   # 256^3: generate one draw at a time (memory only; same sums up to order)
        for s in range(0, len(tr_idx), nb):
            x, y = prob.batch(draws[tr_idx[s:s + nb]])
            s_in += float(x.square().sum()); c_in += x.numel()
            s_out += float(y.square().sum()); c_out += y.numel()
    norm = (torch.tensor(math.sqrt(s_in / c_in), dtype=torch.float64, device='cuda'),
            torch.tensor(math.sqrt(s_out / c_out), dtype=torch.float64, device='cuda'))

    def evaluate(net, norm=norm):
        net.eval()
        errs = []
        with torch.no_grad():
            vb = 4 if n <= 128 else 1
            for s in range(0, len(va_idx), vb):
                x, y = prob.batch(draws[va_idx[s:s + vb]])
                errs.append(prob.errors(prob.predict(net, x, norm), y, x).cpu().numpy())
        e = np.concatenate(errs)
        if not np.isfinite(e).all():
            raise RuntimeError('nonfinite validation error')
        per = e if e.ndim == 1 else e.max(1)
        return dict(errors=e.tolist(), mean_case=float(per.mean()), median_case=float(np.median(per)),
                    worst_case=float(per.max()))

    def loss_of(net, ids):
        x, y = prob.batch(draws[ids])
        e = prob.errors(prob.predict(net, x, norm), y, x)
        if e.ndim == 2:
            e = e[:, 1:]
        return e.square().mean()

    seed = int(config['seed'])
    batch_size = int(config['batch_size'])
    micro = None
    probe = {}
    for m in ((1,) if (args.smoke or args.probe_only) else (8, 4, 2, 1)):
        m = min(m, batch_size)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(seed)
        net = popt = None
        try:
            net = O.make_model(config)
            popt = torch.optim.AdamW(net.parameters(), lr=1e-12, weight_decay=0.0)
            for _ in range(2):
                popt.zero_grad(set_to_none=True)
                loss_of(net, tr_idx[:m]).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                popt.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            probe[str(m)] = peak
            ok = peak < 0.85 * budget_bytes or m == 1
        except torch.OutOfMemoryError:
            probe[str(m)] = 'oom'
            ok = False
        del net, popt
        torch.cuda.empty_cache()
        if ok:
            micro = m
            break
    if args.probe_only:
        dump(args.out / 'probe.json', dict(arm=name, problem=args.problem, mesh=n, config=config, memory_fraction=args.mem_frac,
                                           budget_bytes=budget_bytes, probe_peak_bytes=probe, fits_micro_batch_1=micro is not None,
                                           job_id=os.environ.get('SLURM_JOB_ID'), gpu=env['gpu']))
        print('PROBE', json.dumps(probe), flush=True)
        return
    result_base = dict(arm=name, problem=args.problem, mesh=n, family=config['family'], config=config,
                       memory_fraction=args.mem_frac, probe_peak_bytes=probe, environment=env,
                       job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'))
    if micro is None:
        dump(args.out / 'result.json', dict(result_base, complete=False, failure='no micro-batch fits (OOM at micro-batch 1)'))
        raise RuntimeError('no micro-batch fits')

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    net = O.make_model(config)
    O.check_dtypes(net)
    opt = torch.optim.AdamW(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    lr0, floor = float(config['learning_rate']), 1e-5
    warm = int(config.get('warmup_steps', 0))
    gen = torch.Generator().manual_seed(seed)
    wall = float(args.wall_seconds)
    best, best_epoch, history, step, stop_reason = float('inf'), -1, [], 0, 'epoch_cap'
    patience = int(config.get('patience', 250))
    provenance = dict(result_base, config_sha256=sha_file(args.config),
                      sources={p.name: sha_file(p) for p in sorted(HERE.glob('*.py'))},
                      train_seed=spec['train_seed'], train_cases=count, split='index % 8 == 7 -> validation',
                      train_count=int(len(tr_idx)), validation_count=int(len(va_idx)),
                      overlap_with_evaluation_cohorts=overlap,
                      draws_sha256=hashlib.sha256(np.ascontiguousarray(draws).tobytes()).hexdigest(),
                      normalization=[float(v) for v in norm], micro_batch=micro,
                      wall_budget_seconds=wall, smoke=bool(args.smoke), real_parameter_count=O.parameter_count(net))
    dump(args.out / 'provenance.json', provenance)
    print(json.dumps({k: provenance[k] for k in ('arm', 'problem', 'mesh', 'micro_batch', 'real_parameter_count')}), flush=True)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(int(config['epochs'])):
        net.train()
        order = tr_idx[torch.randperm(len(tr_idx), generator=gen).numpy()]
        losses = []
        for s in range(0, len(order), batch_size):
            ids = order[s:s + batch_size]
            frac = min(1.0, (time.monotonic() - started) / wall)
            lr = floor + (lr0 - floor) * 0.5 * (1 + math.cos(math.pi * frac))
            if step < warm:
                lr = lr * (step + 1) / warm
            for g in opt.param_groups:
                g['lr'] = lr
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for m0 in range(0, len(ids), micro):
                sub = ids[m0:m0 + micro]
                loss = loss_of(net, sub) * len(sub) / len(ids)
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
        metrics = evaluate(net)
        if metrics['mean_case'] < best:
            best, best_epoch = metrics['mean_case'], epoch
            O.save_checkpoint(args.out / 'best.tmp', net, config, norm,
                              dict(epoch=epoch, step=step, best=best, mesh=n, arm=name, problem=args.problem))
            (args.out / 'best.tmp').replace(args.out / 'best.pt')
        history.append(dict(epoch=epoch, steps=step, train_loss=float(np.mean(losses)),
                            validation={k: v for k, v in metrics.items() if k != 'errors'},
                            learning_rate=lr, wall_seconds=time.monotonic() - started, best_so_far=best,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        dump(args.out / 'history.json', history)
        if epoch < 5 or epoch % 10 == 0:
            print(json.dumps(dict(arm=name, epoch=epoch, steps=step, train=history[-1]['train_loss'],
                                  val_mean=metrics['mean_case'], val_worst=metrics['worst_case'],
                                  lr=lr, wall=round(history[-1]['wall_seconds'], 1))), flush=True)
        if STOP:
            stop_reason = 'signal'; break
        if time.monotonic() - started >= wall:
            stop_reason = 'wall_budget'; break
        if epoch - best_epoch >= patience:
            stop_reason = 'patience'; break
    seconds = time.monotonic() - started
    del net, opt
    torch.cuda.empty_cache()
    net2, norm2, ck = O.load_checkpoint(args.out / 'best.pt')
    final = evaluate(net2, norm2)
    reproduced = abs(final['mean_case'] - best) <= 1e-6 * max(best, 1e-12)
    result = dict(result_base, complete=bool(stop_reason != 'signal' and reproduced),
                  validation_reproduced_from_checkpoint=bool(reproduced), micro_batch=micro,
                  best_epoch=ck['epoch'], best_step=ck['step'], epochs_completed=len(history),
                  optimisation_steps=step, stop_reason=stop_reason, training_seconds=seconds,
                  finalisation_seconds=time.monotonic() - started - seconds,
                  wall_budget_seconds=wall, validation_best_checkpoint=final,
                  best_checkpoint_sha256=sha_file(args.out / 'best.pt'),
                  parameter_dtype=str(getattr(net2, 'parameter_dtype', torch.float64)),
                  real_parameter_count=O.parameter_count(net2), normalization=[float(v) for v in norm])
    dump(args.out / 'result.json', result)
    print('TRAIN DONE', json.dumps({k: result[k] for k in ('arm', 'problem', 'mesh', 'stop_reason', 'epochs_completed',
                                                              'optimisation_steps', 'best_epoch')}),
          'val_mean', final['mean_case'], 'val_worst', final['worst_case'], flush=True)


if __name__ == '__main__':
    main()
