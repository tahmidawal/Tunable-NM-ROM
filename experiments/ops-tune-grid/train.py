"""One capacity experiment; case-level validation and complete checkpoints.

Copied from the parent FNO lane's driver. Differences, all guarded so that an
FNO config behaves exactly as before: `family` dispatch through `model.make_model`,
smoke bounds through `model.smoke_bounded`, an optional per-epoch linear
learning-rate warm-up (`warmup_epochs`, default 0 = the parent behaviour), and
explicit stop-reason flags in `result.json`.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import dataset
import model as adapter

STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def save_checkpoint(path, value):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def arrays(records):
    data = [dataset.read_case(r) for r in records]
    return {key: torch.from_numpy(np.stack([d[key] for d in data]))
            for key in ('input', 'target', 'parameters')}


def normalization(data):
    x, p, y = (data[k] for k in ('input', 'parameters', 'target'))
    raw = torch.cat((x, p[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])), 1)
    mean = raw.mean((0, 2, 3), keepdim=True)
    std = raw.std((0, 2, 3), keepdim=True).clamp_min(1e-12)
    scale = y.square().mean().sqrt().clamp_min(1e-12)
    return tuple(v.cuda() for v in (mean, std, scale))


def batch(data, ids):
    return tuple(data[key][ids].cuda() for key in ('input', 'parameters', 'target'))


def evaluate(model, data, norm, pde, size):
    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(data['input']), size):
            x, p, y = batch(data, slice(start, start + size))
            pred = adapter.predict(model, x, p, *norm, pde)
            errors.extend(adapter.relative_errors(pred, y, x, pde).cpu().tolist())
    error = np.asarray(errors)
    if not np.isfinite(error).all():
        raise RuntimeError('Validation produced nonfinite errors')
    per_case = error.max(axis=1)
    return dict(errors=errors, mean_case_max=float(per_case.mean()),
                median_case_max=float(np.median(per_case)), worst_case_max=float(per_case.max()),
                p95_case_max=float(np.quantile(per_case, .95)),
                above_threshold_counts={str(t): int((per_case > t).sum()) for t in (.01, .02, .05)})


def train(args):
    if not os.environ.get('SLURM_JOB_ID') and not args.smoke:
        raise RuntimeError('Production training must run in a cluster allocation')
    config = json.loads(args.config.read_text())
    if args.smoke:
        if config['epochs'] > 2 or not adapter.smoke_bounded(config):
            raise ValueError('Smoke mode is bounded to two tiny epochs')
    environment = adapter.configure()
    pde, training_records, validation_records = dataset.load_pair(args.train_index, args.validation_index)
    if args.smoke and (len(training_records) > 4 or training_records[0]['mesh'] > 64):
        raise ValueError('Smoke data must be tiny')
    args.out.mkdir(parents=True, exist_ok=False)
    source = {p.name: dataset.sha256(p) for p in Path(__file__).parent.glob('*.py')}
    write_json(args.out / 'provenance.json', dict(environment=environment, sources=source,
        config=config, config_sha256=dataset.sha256(args.config),
        train_index_sha256=dataset.sha256(args.train_index), validation_index_sha256=dataset.sha256(args.validation_index),
        train_case_ids=[r['case_id'] for r in training_records],
        validation_case_ids=[r['case_id'] for r in validation_records],
        job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'),
        seed=config['seed'], pde=pde, mesh=training_records[0]['mesh'],
        final_cohort_opened=False, target='declared dataset target; physical-reference evaluation is separate',
        supplied_input='sampled field and known physical coefficients; no Gaussian descriptors',
        timing='training resource accounting only; no FOM/ROM speed ratio'))
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    np.random.seed(config['seed'])
    training, validation = arrays(training_records), arrays(validation_records)
    norm = normalization(training)
    model = adapter.make_model(pde, config)
    adapter.check_dtypes(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    # ops-tune-grid DESIGN 5.4: `schedule` defaults to 'plateau', the parent lane's
    # ReduceLROnPlateau(0.5, patience 20, min 1e-5), so an untuned config is unchanged.
    # 'cosine' decays over the configured epoch cap to the same 1e-5 floor and, unlike the
    # plateau rule, ignores the validation score, so it never sees the validation set.
    schedule = config.get('schedule', 'plateau')
    if schedule == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=20, min_lr=1e-5)
    elif schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-5)
    else:
        raise ValueError(f'unknown schedule: {schedule}')
    generator = torch.Generator().manual_seed(config['seed'])
    best, stale, history = float('inf'), 0, []
    warmup = int(config.get('warmup_epochs', 0))
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(config['epochs']):
        if epoch < warmup:
            for group in optimizer.param_groups:
                group['lr'] = config['learning_rate'] * (epoch + 1) / warmup
        model.train()
        order = torch.randperm(len(training_records), generator=generator)
        losses = []
        for start in range(0, len(order), config['batch_size']):
            x, p, y = batch(training, order[start:start + config['batch_size']])
            optimizer.zero_grad(set_to_none=True)
            pred = adapter.predict(model, x, p, *norm, pde)
            errors = adapter.relative_errors(pred, y, x, pde)
            if pde == 'burgers':
                errors = errors[:, 1:]
            loss = errors.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            if epoch == 0 and start == 0:
                adapter.check_dtypes(model, gradients=True)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append((float(loss.detach()), len(x)))
        metrics = evaluate(model, validation, norm, pde, config['batch_size'])
        score = metrics['mean_case_max']
        scheduler.step(score) if schedule == 'plateau' else scheduler.step()
        improved = score < best
        stale = 0 if improved else stale + 1
        if improved:
            best = score
        record = dict(epoch=epoch, train_mean_squared_relative_error=sum(l*n for l,n in losses)/sum(n for _,n in losses),
                      validation=metrics, learning_rate=optimizer.param_groups[0]['lr'],
                      wall_seconds=time.monotonic()-started, best_so_far=best,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
        history.append(record)
        write_json(args.out / 'history.json', history)
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            config=config, pde=pde, epoch=epoch, best=best, normalization=[x.cpu() for x in norm],
            train_index_sha256=dataset.sha256(args.train_index), validation_index_sha256=dataset.sha256(args.validation_index),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), permutation_rng=generator.get_state())
        if improved:
            save_checkpoint(args.out / 'best.pt', checkpoint)
        save_checkpoint(args.out / 'last.pt', checkpoint)
        print(json.dumps({k:record[k] for k in ('epoch','train_mean_squared_relative_error','wall_seconds','best_so_far')}), flush=True)
        if STOP:
            stop_reason = 'signal'
            break
        if stale >= config['patience']:
            stop_reason = 'early_stopping'
            break
        if time.monotonic()-started >= args.wall_seconds:
            stop_reason = 'wall_budget'
            break
    else:
        stop_reason = 'epoch_cap'
    restored = torch.load(args.out / 'best.pt', map_location='cuda', weights_only=False)
    model.load_state_dict(restored['model'])
    # The reported errors and the saved prediction fields come from ONE batch-1 pass, so
    # the audit's recomputation from the saved fields is exact by construction even for a
    # float32 network, whose batched kernels need not be batch-invariant.
    model.eval()
    errors = []
    with torch.no_grad():
        for i, row in enumerate(validation_records):
            x, p, y = batch(validation, slice(i, i+1))
            prediction = adapter.predict(model, x, p, *norm, pde)
            errors.append(adapter.relative_errors(prediction, y, x, pde).cpu().tolist()[0])
            np.savez(args.out / f"{row['case_id']}.prediction.npz", prediction=prediction.cpu().numpy()[0])
    error = np.asarray(errors)
    if not np.isfinite(error).all():
        raise RuntimeError('Final validation produced nonfinite errors')
    per_case = error.max(axis=1)
    metrics = dict(errors=errors, mean_case_max=float(per_case.mean()),
                   median_case_max=float(np.median(per_case)), worst_case_max=float(per_case.max()),
                   p95_case_max=float(np.quantile(per_case, .95)),
                   above_threshold_counts={str(t): int((per_case > t).sum()) for t in (.01, .02, .05)},
                   batched_selection_score=history[restored['epoch']]['validation']['mean_case_max'])
    np.savez(args.out / 'validation-errors.npz', errors=error.astype(np.float64))
    write_json(args.out / 'result.json', dict(complete=True, pde=pde, best_epoch=restored['epoch'],
        family=adapter.family_of(config), parameter_dtype=str(getattr(model, 'parameter_dtype', torch.float64)),
        epochs_completed=len(history), stop_reason=stop_reason, stopped_by_signal=stop_reason == 'signal',
        stopped_by_wall_budget=stop_reason == 'wall_budget',
        stopped_by_early_stopping=stop_reason == 'early_stopping', stopped_by_epoch_cap=stop_reason == 'epoch_cap',
        wall_budget_seconds=args.wall_seconds, warmup_epochs=warmup,
        validation=metrics, parameter_tensor_elements=sum(p.numel() for p in model.parameters()),
        real_parameter_count=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters()),
        best_checkpoint_sha256=dataset.sha256(args.out/'best.pt'),
        training_seconds=time.monotonic()-started, scientific_status='single-seed development capacity screen; no paired speed claim'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('train-index', 'validation-index', 'config', 'out'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--wall-seconds', type=float, default=7200.)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGUSR1, stop)
    train(args)
