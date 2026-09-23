"""One capacity/hyperparameter experiment; case-level validation and complete checkpoints.

Copied from `experiments/ops-deeponet-b2d/train.py` (itself the parent FNO lane's driver).
Every change below is guarded so that a config without the new keys trains exactly as before:

  * `validate_every_steps` + `patience_evaluations` -- validation cadence and early stopping
    measured in optimisation steps instead of epochs, so one stopping rule holds across a
    training set that grows from 128 to 4608 cases (DESIGN section 4.1). Absent => the
    inherited per-epoch cadence with `patience` epochs.
  * `schedule: "cosine"` -- cosine decay driven by the fraction of the WALL budget elapsed,
    after `warmup_steps` of linear warm-up (DESIGN section 4.2). Absent => the inherited
    `ReduceLROnPlateau`.
  * `output_scale_mode: "per_time"` -- one output scale per evolved output time instead of one
    global scale (DESIGN section 4.3). Absent => the inherited global scale.
  * `--pool` / `--pool-limit` -- read the training split from `build_pool.py`'s packed arrays
    and train on the first N cases in `case_index` order (DESIGN section 4.4). Absent => the
    inherited per-case read.

Statistics needed to describe a run honestly -- the training loss at the selected evaluation
and at the end, how long the learning rate sat at its floor, what actually stopped the run --
are recorded in `result.json` rather than asserted anywhere downstream.
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
import dataset
import model as adapter

STOP = False
# Validation is batched at 8 for every arm. The inherited driver reused `config['batch_size']`,
# which would make the batch-32 arm's selection score come from different kernels than every
# other arm's (design audit finding 31). 8 is the inherited value for every published arm.
VALIDATION_BATCH = 8


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


def pool_arrays(pool, train_index, validation_index, limit):
    """`arrays` for a packed pool.

    `build_pool.py` ran `dataset.load_pair` over the whole split when it built the pool -- the
    per-case checksum, schema, dtype, boundary, output-time and train/validation disjointness
    gates all fired there -- and recorded both index hashes. Re-running that here would re-read
    17 GB once per arm; asserting both hashes instead binds this run to that verification.
    """
    pool = Path(pool)
    manifest = json.loads((pool / 'cases.json').read_text())
    if not manifest.get('disjointness_verified'):
        raise RuntimeError('pool does not record a completed load_pair verification')
    if manifest['index_sha256'] != dataset.sha256(train_index):
        raise RuntimeError('pool was built from a different training index')
    if manifest['validation_index_sha256'] != dataset.sha256(validation_index):
        raise RuntimeError('pool was built against a different validation index')
    count = manifest['count'] if limit is None else int(limit)
    if not 1 <= count <= manifest['count']:
        raise RuntimeError(f'--pool-limit must be in [1, {manifest["count"]}]')
    out = {}
    for key in ('input', 'target', 'parameters'):
        values = np.load(pool / f'{key}.npy', mmap_mode='r')[:count]
        out[key] = torch.from_numpy(np.ascontiguousarray(values))
    return out, manifest, count


def index_metadata(path, split):
    """Records of one split without reading its case files; only for a pool-verified split."""
    index = json.loads(Path(path).read_text())
    if index.get('pde') not in ('poisson', 'burgers') or index.get('complete') is not True:
        raise ValueError('index must be a complete poisson/burgers index')
    records = [r for r in index.get('records', index.get('cases')) if r['split'] == split]
    return index['pde'], records


def normalization(data, mode='global', chunk=64):
    """Inherited statistics, accumulated in chunks so a 17 GB training set does not
    materialise a second copy, and an optional per-output-time output scale."""
    x, p, y = (data[k] for k in ('input', 'parameters', 'target'))
    channels = x.shape[1] + p.shape[1]
    total = torch.zeros(channels, dtype=torch.float64)
    elements = 0
    for start in range(0, len(x), chunk):
        xi, pi = x[start:start + chunk], p[start:start + chunk]
        raw = torch.cat((xi, pi[:, :, None, None].expand(-1, -1, xi.shape[-2], xi.shape[-1])), 1)
        total += raw.sum((0, 2, 3))
        elements += raw.shape[0] * raw.shape[2] * raw.shape[3]
    mean = (total / elements).reshape(1, -1, 1, 1)
    # Second pass for the variance: the inherited `std` is torch's sample (Bessel-corrected)
    # standard deviation, and a one-pass sum-of-squares form would cancel badly at 4608 cases.
    squares = torch.zeros(channels, dtype=torch.float64)
    for start in range(0, len(x), chunk):
        xi, pi = x[start:start + chunk], p[start:start + chunk]
        raw = torch.cat((xi, pi[:, :, None, None].expand(-1, -1, xi.shape[-2], xi.shape[-1])), 1)
        squares += (raw - mean).square().sum((0, 2, 3))
    std = (squares / (elements - 1)).sqrt().reshape(1, -1, 1, 1).clamp_min(1e-12)
    if mode == 'global':
        squared, count = 0., 0
        for start in range(0, len(y), chunk):
            block = y[start:start + chunk]
            squared += float(block.square().sum())
            count += block.numel()
        scale = torch.tensor(squared / count, dtype=torch.float64).sqrt().clamp_min(1e-12)
    elif mode == 'per_time':
        times = y.shape[1]
        squared = torch.zeros(times, dtype=torch.float64)
        count = 0
        for start in range(0, len(y), chunk):
            block = y[start:start + chunk]
            squared += block.square().sum(dim=(0, 2, 3, 4))
            count += block.shape[0] * block.shape[2] * block.shape[3] * block.shape[4]
        # The network emits the EVOLVED times only; time 0 is the supplied field.
        scale = (squared[1:] / count).sqrt().clamp_min(1e-12).reshape(1, -1, 1, 1)
    else:
        raise ValueError(mode)
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


def learning_rate_now(config, step, elapsed, wall_seconds):
    """Cosine decay on the fraction of the WALL budget elapsed, after linear warm-up."""
    base, floor = config['learning_rate'], config.get('min_learning_rate', 1e-6)
    warmup = int(config.get('warmup_steps', 0))
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    fraction = min(max(elapsed / max(wall_seconds, 1e-9), 0.), 1.)
    return floor + (base - floor) * .5 * (1. + math.cos(math.pi * fraction))


def train(args):
    if not os.environ.get('SLURM_JOB_ID') and not args.smoke:
        raise RuntimeError('Production training must run in a cluster allocation')
    config = json.loads(args.config.read_text())
    if args.smoke:
        if config['epochs'] > 2 or not adapter.smoke_bounded(config):
            raise ValueError('Smoke mode is bounded to two tiny epochs')
    environment = adapter.configure()
    pool_manifest = None
    if args.pool:
        training, pool_manifest, count = pool_arrays(args.pool, args.train_index,
                                                     args.validation_index, args.pool_limit)
        pde, training_records = index_metadata(args.train_index, 'train')
        training_records = training_records[:count]
        if pool_manifest['case_ids'][:count] != [r['case_id'] for r in training_records]:
            raise RuntimeError('pool case order differs from the index order')
        validation_pde, validation_rows = dataset.load_index(args.validation_index, ('validation',))
        validation_records = validation_rows['validation']
        if validation_pde['pde'] != pde:
            raise RuntimeError('Train and validation PDEs differ')
    else:
        if args.pool_limit is not None:
            raise RuntimeError('--pool-limit requires --pool')
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
        training_cases=len(training_records),
        pool=dict(path=str(args.pool), limit=args.pool_limit,
                  index_sha256=pool_manifest['index_sha256'],
                  array_sha256=pool_manifest['array_sha256']) if pool_manifest else None,
        job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'),
        seed=config['seed'], pde=pde, mesh=training_records[0]['mesh'],
        final_cohort_opened=False, target='declared dataset target; physical-reference evaluation is separate',
        supplied_input='sampled field and known physical coefficients; no Gaussian descriptors',
        timing='no timing block runs in this lane; no speed statement is admissible from it'))
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    np.random.seed(config['seed'])
    if not args.pool:
        training = arrays(training_records)
    validation = arrays(validation_records)
    norm = normalization(training, config.get('output_scale_mode', 'global'))
    model = adapter.make_model(pde, config)
    adapter.check_dtypes(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    cosine = config.get('schedule', 'plateau') == 'cosine'
    cadence = config.get('validate_every_steps')
    plateau_patience = config.get('plateau_patience_evaluations', 20 if not cadence else 8)
    scheduler = None if cosine else torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.5, patience=plateau_patience, min_lr=config.get('min_learning_rate', 1e-5))
    generator = torch.Generator().manual_seed(config['seed'])
    best, stale, history = float('inf'), 0, []
    warmup = int(config.get('warmup_epochs', 0))
    patience = config['patience_evaluations'] if cadence else config['patience']
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    step, stop_reason, epoch = 0, None, 0

    def checkpoint_and_check(epoch, losses):
        """One validation evaluation: score, schedule, checkpoint, and the stop decision."""
        nonlocal best, stale
        metrics = evaluate(model, validation, norm, pde, VALIDATION_BATCH)
        score = metrics['mean_case_max']
        if scheduler is not None:
            scheduler.step(score)
        improved = score < best
        stale = 0 if improved else stale + 1
        if improved:
            best = score
        record = dict(epoch=epoch, step=step,
                      train_mean_squared_relative_error=(sum(l * n for l, n in losses) / sum(n for _, n in losses))
                      if losses else float('nan'),
                      validation=metrics, learning_rate=optimizer.param_groups[0]['lr'],
                      wall_seconds=time.monotonic() - started, best_so_far=best,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
        history.append(record)
        write_json(args.out / 'history.json', history)
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler is not None else None,
            config=config, pde=pde, epoch=epoch, step=step, best=best,
            normalization=[v.cpu() for v in norm], training_cases=len(training_records),
            train_index_sha256=dataset.sha256(args.train_index), validation_index_sha256=dataset.sha256(args.validation_index),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), permutation_rng=generator.get_state())
        if improved:
            save_checkpoint(args.out / 'best.pt', checkpoint)
        save_checkpoint(args.out / 'last.pt', checkpoint)
        print(json.dumps({k: record[k] for k in ('epoch', 'step', 'train_mean_squared_relative_error',
                                                 'wall_seconds', 'best_so_far')}), flush=True)
        if STOP:
            return 'signal'
        if stale >= patience:
            return 'early_stopping'
        if time.monotonic() - started >= args.wall_seconds:
            return 'wall_budget'
        return None

    for epoch in range(config['epochs']):
        if warmup and epoch < warmup and not cosine:
            for group in optimizer.param_groups:
                group['lr'] = config['learning_rate'] * (epoch + 1) / warmup
        model.train()
        order = torch.randperm(len(training_records), generator=generator)
        losses = []
        for start in range(0, len(order), config['batch_size']):
            if cosine:
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate_now(config, step, time.monotonic() - started, args.wall_seconds)
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
            if step == 0:
                adapter.check_dtypes(model, gradients=True)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append((float(loss.detach()), len(x)))
            step += 1
            due = cadence and (step % cadence == 0 or STOP
                               or time.monotonic() - started >= args.wall_seconds)
            if due:
                stop_reason = checkpoint_and_check(epoch, losses)
                losses = []
                model.train()
                if stop_reason:
                    break
        if stop_reason:
            break
        if not cadence:
            stop_reason = checkpoint_and_check(epoch, losses)
            if stop_reason:
                break
    else:
        stop_reason = stop_reason or 'epoch_cap'
    if stop_reason is None:
        stop_reason = 'epoch_cap'
    if not history:
        raise RuntimeError('No validation evaluation completed; the budget is too small for the cadence')
    restored = torch.load(args.out / 'best.pt', map_location='cuda', weights_only=False)
    model.load_state_dict(restored['model'])
    # The reported errors and the saved prediction fields come from ONE batch-1 pass, so
    # the audit's recomputation from the saved fields is exact by construction even for a
    # float32 network, whose batched kernels need not be batch-invariant.
    model.eval()
    errors = []
    with torch.no_grad():
        for i, row in enumerate(validation_records):
            x, p, y = batch(validation, slice(i, i + 1))
            prediction = adapter.predict(model, x, p, *norm, pde)
            errors.append(adapter.relative_errors(prediction, y, x, pde).cpu().tolist()[0])
            np.savez(args.out / f"{row['case_id']}.prediction.npz", prediction=prediction.cpu().numpy()[0])
    error = np.asarray(errors)
    if not np.isfinite(error).all():
        raise RuntimeError('Final validation produced nonfinite errors')
    per_case = error.max(axis=1)
    subset = min(len(training_records), 128)
    training_metrics = evaluate(model, {k: v[:subset] for k, v in training.items()},
                                norm, pde, VALIDATION_BATCH)
    training_metrics['cases'] = subset
    training_metrics['note'] = ('the first 128 training cases at the selected checkpoint, same '
                                'metric as validation; the generalisation gap, not a selection input')
    position = next(i for i, h in enumerate(history) if h['step'] == restored['step'])
    selected = history[position]
    floor = config.get('min_learning_rate', 1e-5)
    metrics = dict(errors=errors, mean_case_max=float(per_case.mean()),
                   median_case_max=float(np.median(per_case)), worst_case_max=float(per_case.max()),
                   p95_case_max=float(np.quantile(per_case, .95)),
                   above_threshold_counts={str(t): int((per_case > t).sum()) for t in (.01, .02, .05)},
                   batched_selection_score=selected['validation']['mean_case_max'])
    np.savez(args.out / 'validation-errors.npz', errors=error.astype(np.float64))
    write_json(args.out / 'result.json', dict(complete=True, pde=pde, best_epoch=restored['epoch'],
        best_step=restored['step'], family=adapter.family_of(config),
        parameter_dtype=str(getattr(model, 'parameter_dtype', torch.float64)),
        epochs_completed=epoch + 1, steps_completed=step, evaluations_completed=len(history),
        training_cases=len(training_records), train_index_sha256=dataset.sha256(args.train_index),
        validation_every_steps=cadence, patience=patience,
        schedule='cosine' if cosine else 'plateau',
        output_scale_mode=config.get('output_scale_mode', 'global'),
        stop_reason=stop_reason, stopped_by_signal=stop_reason == 'signal',
        stopped_by_wall_budget=stop_reason == 'wall_budget',
        stopped_by_early_stopping=stop_reason == 'early_stopping', stopped_by_epoch_cap=stop_reason == 'epoch_cap',
        wall_budget_seconds=args.wall_seconds, warmup_epochs=warmup,
        warmup_steps=int(config.get('warmup_steps', 0)),
        train_loss_at_best=selected['train_mean_squared_relative_error'],
        train_loss_final=history[-1]['train_mean_squared_relative_error'],
        evaluations_at_minimum_learning_rate=sum(1 for h in history if h['learning_rate'] <= floor * 1.000001),
        evaluations_after_best=len(history) - 1 - position,
        validation=metrics, training_subset=training_metrics, parameter_tensor_elements=sum(p.numel() for p in model.parameters()),
        real_parameter_count=sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()),
        best_checkpoint_sha256=dataset.sha256(args.out / 'best.pt'),
        training_seconds=time.monotonic() - started,
        scientific_status='single-seed development arm; no paired speed claim and no timing block'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('train-index', 'validation-index', 'config', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--wall-seconds', type=float, default=7200.)
    parser.add_argument('--pool', default=None)
    parser.add_argument('--pool-limit', type=int, default=None)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGUSR1, stop)
    train(args)
