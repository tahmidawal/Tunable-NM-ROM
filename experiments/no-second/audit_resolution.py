"""Independent NumPy audit of the resolution-ladder job (DESIGN §A5). No torch, no jax.

    python audit_resolution.py res01

Recomputes every reported error from the saved prolonged prediction fields; re-derives the
interpolation floor from the references themselves; re-derives every timing median from the
retained repetition arrays; and enforces the §A5 gates, above all the **top-rung gate**: at
256 intervals each checkpoint must reproduce the already-published validation numbers for
that checkpoint (this lane's audits for the U-Net and Transolver, the parent lane's for the
FNO) to 1e-9, or the job is void. Applies the pre-registered R-USABLE / R-DEGENERATE labels.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

LANE = Path(__file__).resolve().parent
PUBLISHED = {
    'unet-refine': (LANE / 'runs/unet01/audit.json', 'arms'),
    'tsol-refine': (LANE / 'runs/tsol01/audit.json', 'arms'),
    'fno-large': (LANE.parent / 'neural-operator-audit/runs/fno_burgers02/field-audit.json', 'models'),
}
SPEEDUP_GATE = 1.5     # §A5 R-USABLE
ERROR_GATE = 2.0       # §A5 R-USABLE
TOP_RUNG_TOLERANCE = 1e-9


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def restrict(field, intervals):
    source = field.shape[-1] - 1
    assert source % intervals == 0
    stride = source // intervals
    return np.ascontiguousarray(field[..., ::stride, ::stride], dtype=np.float64)


def prolong(field, target):
    a = np.asarray(field, dtype=np.float64)
    L = a.shape[-1] - 1
    if target == L:
        return a
    x = np.arange(target + 1, dtype=np.float64) * L / target
    lo = np.minimum(x.astype(np.int64), L - 1)
    f = x - lo
    a = (1 - f[:, None]) * a[..., lo, :] + f[:, None] * a[..., lo + 1, :]
    return (1 - f[None, :]) * a[..., :, lo] + f[None, :] * a[..., :, lo + 1]


def fixed_initial_errors(prediction, target):
    denominator = np.linalg.norm(target[0, 0, 1:-1, 1:-1])
    assert np.isfinite(denominator) and denominator > 0
    return np.linalg.norm((prediction - target)[:, 0, 1:-1, 1:-1].reshape(len(target), -1), axis=1) / denominator


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    return dict(mean=float(x.mean()), median=float(np.median(x)), maximum=float(x.max()),
                p95=float(np.quantile(x, .95)))


def published_worst_evolved(name):
    """The already-audited worst-over-evolved-times for this checkpoint at full resolution."""
    path, key = PUBLISHED[name]
    record = json.loads(path.read_text())[key][name]
    per_time = np.asarray(record['per_time_errors'], dtype=np.float64)
    return float(per_time[:, 1:].max()), float(per_time.max()), path


def main(attempt):
    run = LANE / 'runs' / attempt
    root = run / 'archive' / attempt
    stated = json.loads((root / 'out/resolution/resolution.json').read_text())
    provenance = json.loads((run / 'PROVENANCE.json').read_text())
    log = next(iter(sorted((root / 'logs').glob('*.out')))).read_text()
    job_id = re.search(r'job=(\d+)', log).group(1)
    assert 'jax_backend=gpu' in log and log.rstrip().endswith('ALL-DONE')
    assert re.search(r'checkpoints_verified=(\d+)', log), 'checkpoint cache was not verified in-job'
    gpu = re.search(r'\n(NVIDIA [^,]+), ', log).group(1)
    assert stated['prolongation_check']['max_abs_difference'] <= stated['prolongation_check']['tolerance']

    rows = json.loads((root / 'data/validation/index.json').read_text())['records']
    cohort_rows = json.loads((root / 'data/diagnosis-cohort/index.json').read_text())['records']
    cohorts = {'validation': (rows, root / 'data/validation'),
               'diagnosis': (cohort_rows, root / 'data/diagnosis-cohort')}

    # ---- interpolation floor, recomputed from the references themselves -----------------
    floor = {}
    for cohort, (records, data_root) in cohorts.items():
        for rung in stated['rungs']:
            errs = []
            for row in records:
                with np.load(data_root / row['path']) as case:
                    target = case['target'].copy()
                approx = prolong(restrict(target[:, 0], rung), 256)[:, None]
                errs.append(fixed_initial_errors(approx, target))
            e = np.asarray(errs)
            floor[f'{cohort}@{rung}'] = dict(worst_over_evolved_times=stats(e[:, 1:].max(axis=1)),
                                             worst_over_all_times=stats(e.max(axis=1)),
                                             initial_time_term=stats(e[:, 0]))
            claimed = stated['interpolation_floor'][f'{cohort}@{rung}']['worst_over_evolved_times']['maximum']
            assert abs(claimed - floor[f'{cohort}@{rung}']['worst_over_evolved_times']['maximum']) <= 1e-12

    # ---- every model at every rung, recomputed from the saved prolonged fields ----------
    models = {}
    for entry in stated['checkpoints']:
        name = entry['name']
        for rung in stated['rungs']:
            key = f'{name}@{rung}'
            for cohort, (records, data_root) in cohorts.items():
                errs = []
                for row in records:
                    with np.load(data_root / row['path']) as case:
                        target = case['target'].copy()
                    with np.load(root / 'out/resolution/fields' / key / cohort / f"{row['case_id']}.prediction.npz") as saved:
                        prediction = saved['prediction'].copy()
                    assert prediction.shape == target.shape and prediction.dtype == np.float64
                    assert np.isfinite(prediction).all()
                    for edge in (prediction[..., 0, :], prediction[..., -1, :], prediction[..., :, 0], prediction[..., :, -1]):
                        assert not np.any(edge), f'{key} {cohort} {row["case_id"]}: boundary must stay exactly zero'
                    errs.append(fixed_initial_errors(prediction, target))
                e = np.asarray(errs)
                mine = dict(worst_over_evolved_times=stats(e[:, 1:].max(axis=1)),
                            worst_over_all_times=stats(e.max(axis=1)),
                            initial_time_term=stats(e[:, 0]),
                            per_case_evolved=e[:, 1:].max(axis=1).tolist(),
                            per_case_all=e.max(axis=1).tolist(), per_case_initial=e[:, 0].tolist())
                theirs = stated['results'][f'{key}|{cohort}']
                for block in ('worst_over_evolved_times', 'worst_over_all_times', 'initial_time_term'):
                    for stat in ('mean', 'median', 'maximum'):
                        a, b = mine[block][stat], theirs[block][stat]
                        assert abs(a - b) <= 1e-11 * max(abs(a), 1e-300) + 1e-13, (key, cohort, block, stat, a, b)
                models[f'{key}|{cohort}'] = mine
            # timing re-derived from retained repetitions
            samples = np.load(root / 'out/resolution' / f'timing-{name}-{rung}.npz')['samples']
            assert np.isfinite(samples).all() and (samples > 0).all()
            recomputed = float(np.median(samples) * 1e3)
            assert abs(recomputed - stated['timings'][key]['device_query_pooled_median_ms']) <= 1e-9 * max(1., recomputed)
            models[f'{key}|timing'] = dict(device_query_pooled_median_ms=recomputed,
                                           device_query_p05_ms=float(np.quantile(samples, .05) * 1e3),
                                           device_query_p95_ms=float(np.quantile(samples, .95) * 1e3),
                                           device_query_mean_ms=float(samples.mean() * 1e3),
                                           repetitions=list(samples.shape),
                                           timed_region='on-device supplied field and parameters to the on-device '
                                                        'complete six-time trajectory (the parent lane\'s device query); '
                                                        'the device-to-host copy is not included')

    # ---- §A5 top-rung gate: rung 256 must reproduce the published numbers ---------------
    top = {}
    for entry in stated['checkpoints']:
        name = entry['name']
        mine = models[f'{name}@256|validation']
        pub_evolved, pub_all, path = published_worst_evolved(name)
        gap_evolved = abs(mine['worst_over_evolved_times']['maximum'] - pub_evolved)
        gap_all = abs(mine['worst_over_all_times']['maximum'] - pub_all)
        top[name] = dict(published_worst_evolved=pub_evolved, reproduced_worst_evolved=mine['worst_over_evolved_times']['maximum'],
                         gap_evolved=gap_evolved, published_worst_all=pub_all,
                         reproduced_worst_all=mine['worst_over_all_times']['maximum'], gap_all=gap_all,
                         source=str(path.relative_to(LANE.parents[1])), passed=bool(max(gap_evolved, gap_all) <= TOP_RUNG_TOLERANCE))
        assert top[name]['passed'], f'TOP-RUNG GATE FAILED for {name}: {top[name]}'

    # ---- pre-registered R-USABLE / R-DEGENERATE labels ----------------------------------
    labels = {}
    for entry in stated['checkpoints']:
        name = entry['name']
        base_err = models[f'{name}@256|validation']['worst_over_evolved_times']['maximum']
        base_ms = models[f'{name}@256|timing']['device_query_pooled_median_ms']
        rungs = {}
        for rung in stated['rungs']:
            if rung == 256:
                continue
            err = models[f'{name}@{rung}|validation']['worst_over_evolved_times']['maximum']
            ms = models[f'{name}@{rung}|timing']['device_query_pooled_median_ms']
            speedup = base_ms / ms if ms > 0 else float('inf')
            rungs[rung] = dict(worst_evolved=err, error_ratio=err / base_err, median_ms=ms,
                               same_job_speedup=speedup,
                               meets_error_gate=bool(err <= ERROR_GATE * base_err),
                               meets_speed_gate=bool(speedup >= SPEEDUP_GATE),
                               floor_worst_evolved=floor[f'validation@{rung}']['worst_over_evolved_times']['maximum'],
                               error_over_floor=err / floor[f'validation@{rung}']['worst_over_evolved_times']['maximum'])
        usable = [r for r, v in rungs.items() if v['meets_error_gate'] and v['meets_speed_gate']]
        first = max(r for r in rungs)  # the first rung below 256
        labels[name] = dict(base_worst_evolved=base_err, base_median_ms=base_ms, rungs=rungs,
                            usable_rungs=sorted(usable, reverse=True),
                            label='R-USABLE' if usable else 'R-DEGENERATE',
                            first_rung_below=first, first_rung_passes=bool(rungs[first]['meets_error_gate'] and rungs[first]['meets_speed_gate']),
                            criterion=f'R-USABLE iff some rung has same-job speedup >= {SPEEDUP_GATE} and '
                                      f'worst-over-evolved error <= {ERROR_GATE}x the model\'s own 256 value')

    result = dict(attempt=attempt, job_id=job_id, gpu=gpu, source_commit=provenance['source_commit'],
                  rungs=stated['rungs'], checkpoints=stated['checkpoints'], prolongation_check=stated['prolongation_check'],
                  top_rung_gate=top, interpolation_floor=floor, models=models, labels=labels,
                  validation_cases=len(rows), diagnosis_cases=len(cohort_rows),
                  training_in_this_job=False,
                  timing_scope='same job, same GPU; the only ratios formed are within one model\'s own '
                               'resolution curve, never across jobs, models-from-other-jobs, the ROM or the FOM',
                  grading=stated['grading'], passed=True,
                  limitation='One mesh ladder, one PDE, one seed, the validation-selected checkpoint of each '
                             'family; the labels describe these checkpoints, not neural operators in general.')
    (run / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(job=job_id, gpu=gpu, top_rung={k: v['passed'] for k, v in top.items()},
                          labels={k: v['label'] for k, v in labels.items()}), indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
