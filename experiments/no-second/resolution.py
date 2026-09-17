"""The resolution knob: evaluate frozen operators at coarser grids (DESIGN §A5).

No training happens here. Each frozen, validation-selected checkpoint is evaluated on the
same cases at 256 / 128 / 64 / 32 intervals, graded exactly as the Burgers lane grades its
own coarse-mesh FOM arms — restrict the supplied field by stride, run at that grid, prolong
every output time back to the 257² evaluation grid by the same aligned bilinear map, score
with the same fixed-initial metric against the same reference — and timed in this one job.

The `interp-floor` control applies restriction+prolongation to the *reference itself*: the
error a perfect operator would incur at that rung.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import dataset
import model as adapter

RUNGS = (256, 128, 64, 32)


def restrict(field, intervals):
    """Exact nested sampling; matches engines.restrict."""
    source = field.shape[-1] - 1
    assert source % intervals == 0
    stride = source // intervals
    return np.ascontiguousarray(field[..., ::stride, ::stride], dtype=np.float64)


def prolong(field, target):
    """Aligned bilinear prolongation to `target` intervals.

    A NumPy transcription of `engines.output_field`: coarse boundaries and nested nodes are
    exact. `check_prolongation` verifies it against that JAX original before any use.
    """
    a = np.asarray(field, dtype=np.float64)
    L = a.shape[-1] - 1
    if target == L:
        return a
    x = np.arange(target + 1, dtype=np.float64) * L / target
    lo = np.minimum(x.astype(np.int64), L - 1)
    f = x - lo
    a = (1 - f[:, None]) * a[..., lo, :] + f[:, None] * a[..., lo + 1, :]
    return (1 - f[None, :]) * a[..., :, lo] + f[None, :] * a[..., :, lo + 1]


def check_prolongation(tolerance=1e-12):
    """Compare against the Burgers lane's own `engines.output_field`."""
    import importlib.util
    import jax
    jax.config.update('jax_enable_x64', True)
    path = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-14-no-burgers/'
                'experiments/mr-burgers2d/engines.py')
    if not path.exists():  # on the cluster the reference source is staged beside us
        path = Path(__file__).with_name('engines_output_field.py')
    spec = importlib.util.spec_from_file_location('engines_ref', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worst = 0.0
    rng = np.random.default_rng(20260917)
    for L in (32, 64, 128):
        interior = rng.standard_normal((L - 1) * (L - 1))
        mine_in = np.zeros((L + 1, L + 1))
        mine_in[1:-1, 1:-1] = interior.reshape(L - 1, L - 1)
        theirs = np.asarray(module.output_field(interior, L, 256))
        mine = prolong(mine_in, 256)
        worst = max(worst, float(np.abs(theirs - mine).max()))
    assert worst <= tolerance, f'prolongation differs from engines.output_field by {worst}'
    return dict(max_abs_difference=worst, tolerance=tolerance, source=str(path),
                source_sha256=dataset.sha256(path))


def fixed_initial_errors(prediction, target):
    denominator = np.linalg.norm(target[0, 0, 1:-1, 1:-1])
    assert np.isfinite(denominator) and denominator > 0
    return np.linalg.norm((prediction - target)[:, 0, 1:-1, 1:-1].reshape(len(target), -1), axis=1) / denominator


def summarise(per_case_times):
    """worst-over-evolved, worst-over-all and the t=0 term, kept separate (rule 9)."""
    e = np.asarray(per_case_times, dtype=np.float64)          # (case, time)
    all_times = e.max(axis=1)
    evolved = e[:, 1:].max(axis=1)
    initial = e[:, 0]
    def stats(x):
        return dict(mean=float(x.mean()), median=float(np.median(x)), maximum=float(x.max()),
                    p95=float(np.quantile(x, .95)))
    return dict(worst_over_all_times=stats(all_times), worst_over_evolved_times=stats(evolved),
                initial_time_term=stats(initial), per_case_all=all_times.tolist(),
                per_case_evolved=evolved.tolist(), per_case_initial=initial.tolist(),
                per_case_per_time=e.tolist())


def timed(function, repetitions, burn_in):
    for _ in range(burn_in):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return samples


def load(entry, environment):
    checkpoint = torch.load(entry['checkpoint'], map_location='cuda', weights_only=False)
    actual = dataset.sha256(entry['checkpoint'])
    assert actual == entry['sha256'], f"{entry['name']}: checkpoint hash {actual} != recorded {entry['sha256']}"
    network = adapter.make_model(checkpoint['pde'], checkpoint['config'])
    network.load_state_dict(checkpoint['model'])
    adapter.check_dtypes(network)
    network.eval()
    norm = tuple(v.cuda() for v in checkpoint['normalization'])
    return network, norm, checkpoint


def main(args):
    environment = adapter.configure()
    prolongation = check_prolongation()
    print(json.dumps(dict(prolongation_check=prolongation)), flush=True)
    manifest = json.loads(args.checkpoints.read_text())
    _, splits = dataset.load_index(args.validation_index, ('validation',))
    records = splits['validation']
    cohort_records = []
    if args.cohort_index and args.cohort_index.exists():
        _, csplits = dataset.load_index(args.cohort_index, ('calibration',))
        cohort_records = csplits['calibration']
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- interpolation floor: a perfect operator at each rung ------------------------
    floors = {}
    for cohort_name, rows in (('validation', records), ('diagnosis', cohort_records)):
        if not rows:
            continue
        for rung in RUNGS:
            errs = []
            for row in rows:
                case = dataset.read_case(row)
                target = case['target']
                approx = prolong(restrict(target[:, 0], rung), 256)[:, None]
                errs.append(fixed_initial_errors(approx, target).tolist())
            floors[f'{cohort_name}@{rung}'] = summarise(errs)
        print(json.dumps({k: round(v['worst_over_evolved_times']['maximum'] * 100, 4)
                          for k, v in floors.items() if k.startswith(cohort_name)}), flush=True)

    results, timings = {}, {}
    for entry in manifest['checkpoints']:
        network, norm, checkpoint = load(entry, environment)
        pde = checkpoint['pde']
        for rung in RUNGS:
            key = f"{entry['name']}@{rung}"
            for cohort_name, rows in (('validation', records), ('diagnosis', cohort_records)):
                if not rows:
                    continue
                errs = []
                destination = args.out / 'fields' / key / cohort_name
                destination.mkdir(parents=True, exist_ok=True)
                with torch.no_grad():
                    for row in rows:
                        case = dataset.read_case(row)
                        target = case['target']
                        coarse = restrict(case['input'], rung)
                        field = torch.from_numpy(np.ascontiguousarray(coarse))[None].cuda()
                        parameters = torch.from_numpy(case['parameters'])[None].cuda()
                        prediction = adapter.predict(network, field, parameters, *norm, pde)
                        fine = prolong(prediction.cpu().numpy()[0][:, 0], 256)[:, None]
                        errs.append(fixed_initial_errors(fine, target).tolist())
                        np.savez_compressed(destination / f"{row['case_id']}.prediction.npz", prediction=fine)
                results[f'{key}|{cohort_name}'] = summarise(errs)
            # ---- timing: same job, same GPU, this model's own curve ------------------
            case = dataset.read_case(records[0])
            coarse = restrict(case['input'], rung)
            field = torch.from_numpy(np.ascontiguousarray(coarse))[None].cuda()
            parameters = torch.from_numpy(case['parameters'])[None].cuda()
            torch.cuda.synchronize()

            def query():
                return adapter.predict(network, field, parameters, *norm, pde)

            per_case = [timed(query, args.repetitions, args.burn_in) for _ in range(args.timing_cases)]
            samples = np.asarray(per_case, dtype=np.float64)
            timings[key] = dict(repetitions=list(samples.shape),
                                device_query_pooled_median_ms=float(np.median(samples) * 1e3),
                                device_query_mean_ms=float(samples.mean() * 1e3),
                                device_query_p05_ms=float(np.quantile(samples, .05) * 1e3),
                                device_query_p95_ms=float(np.quantile(samples, .95) * 1e3))
            np.savez(args.out / f'timing-{key.replace("@", "-")}.npz', samples=samples)
            print(json.dumps({key: dict(worst_evolved_pct=round(results[f'{key}|validation']['worst_over_evolved_times']['maximum'] * 100, 4),
                                        median_ms=round(timings[key]['device_query_pooled_median_ms'], 4))}), flush=True)
        del network, checkpoint
        torch.cuda.empty_cache()

    (args.out / 'resolution.json').write_text(json.dumps(dict(
        environment=environment, prolongation_check=prolongation, rungs=list(RUNGS),
        checkpoints=manifest['checkpoints'], interpolation_floor=floors, results=results, timings=timings,
        validation_index_sha256=dataset.sha256(args.validation_index),
        cohort_index_sha256=dataset.sha256(args.cohort_index) if cohort_records else None,
        validation_cases=[r['case_id'] for r in records],
        diagnosis_cases=[r['case_id'] for r in cohort_records],
        repetitions_per_case=args.repetitions, burn_in_per_block=args.burn_in, timing_cases=args.timing_cases,
        grading='restrict the supplied field by stride, run the frozen operator at that grid, prolong every '
                'output time back to 256 intervals by the aligned bilinear map engines.output_field uses, and '
                'score with the Burgers lane fixed-initial metric against the same reference on the 256 grid',
        timing_scope='same job, same GPU; ratios are admissible only within one model\'s own resolution curve',
        training='none; every checkpoint is frozen and hash-verified'), indent=2) + '\n')
    print('RESOLUTION FINISHED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', required=True, type=Path)
    parser.add_argument('--validation-index', required=True, type=Path)
    parser.add_argument('--cohort-index', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--repetitions', type=int, default=30)
    parser.add_argument('--burn-in', type=int, default=20)
    parser.add_argument('--timing-cases', type=int, default=8)
    main(parser.parse_args())
