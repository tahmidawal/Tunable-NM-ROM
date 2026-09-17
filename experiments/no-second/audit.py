"""Independent NumPy audit of one collected attempt. Imports neither torch nor jax
nor any driver file.

    python audit.py <attempt>

Recomputes every reported per-case, per-time fixed-initial error from the saved
prediction fields and the archived validation / cohort cases; checks that the
supplied initial state is returned bitwise, the boundary is exactly zero and
every prediction is finite float64; verifies checkpoint hashes; asserts the
train/validation index hashes equal the FNO job's; re-derives the timing medians
from the retained repetition arrays; and records for every arm its parameters,
epochs, best epoch, budget and which condition ended it. Writes
`runs/<attempt>/audit.json`, the ONLY file the report generator reads for this lane.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

LANE = Path(__file__).resolve().parent
# The FNO job's index hashes (its provenance.json, reproduced in its report sources).
FNO_TRAIN_INDEX = '5333584b7162df622ec7bc2b08e68d8003036b1b05abbea520249ed408fc8d49'
FNO_VALIDATION_INDEX = '468b9e70df3392c4b5bbb41381c9062d3d697ac4772b7236c4ee59d0ca73ebad'
FNO_COHORT_INDEX = json.loads((LANE.parent / 'neural-operator-audit/runs/fno_burgers02/field-audit.json')
                              .read_text())['diagnosis_cohort']['cohort_index_sha256']
# The Poisson FNO job's (3702464) index hashes, from its archived provenance.json.
FNO_POISSON_TRAIN_INDEX = 'd20a5994af09b2dd5631186a9e920bd2444b1bea86e6c9521841d398f1bf1013'
FNO_POISSON_VALIDATION_INDEX = '65cc277bf30b55fce7e6b976514c4a1a2be3c664e9cc3193dee95f7e9100500d'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def statistics(values):
    x = np.asarray(values, dtype=np.float64)
    q1, q3 = np.quantile(x, [.25, .75])
    return dict(mean=float(x.mean()), median=float(np.median(x)), maximum=float(x.max()),
                p95=float(np.quantile(x, .95)), minimum=float(x.min()),
                upper_tukey_outliers=int((x > q3 + 1.5 * (q3 - q1)).sum()),
                above_threshold_counts={str(t): int((x > t).sum()) for t in (.01, .02, .05)})


def fixed_initial_errors(prediction, target):
    """Burgers lane metric: interior l2 discrepancy over the interior l2 norm of the
    reference initial field, per requested output time."""
    denominator = np.linalg.norm(target[0, 0, 1:-1, 1:-1])
    assert np.isfinite(denominator) and denominator > 0
    return np.linalg.norm((prediction - target)[:, 0, 1:-1, 1:-1].reshape(len(target), -1), axis=1) / denominator


def check_prediction(prediction, target, supplied):
    assert prediction.shape == target.shape and prediction.dtype == np.float64
    assert np.isfinite(prediction).all()
    for edge in (prediction[..., 0, :], prediction[..., -1, :], prediction[..., :, 0], prediction[..., :, -1]):
        assert not np.any(edge), 'boundary must be exactly zero'
    assert np.array_equal(prediction[0], supplied), 'supplied initial state must be returned exactly'
    errors = fixed_initial_errors(prediction, target)
    assert errors[0] == 0
    return errors


PROTOCOL = dict(epochs=4000, patience=250, batch_size=8, weight_decay=0.0001)
LEARNING_RATES = {0.001, 0.0003}


def stop_reason(result):
    reason = result['stop_reason']
    assert reason in ('signal', 'early_stopping', 'wall_budget', 'epoch_cap')
    assert result[f'stopped_by_{reason}'] is True
    return reason


def audit_arm_poisson(folder, rows, data_root):
    """Poisson: whole-field discrepancy over the target norm against the declared discrete
    target, and the same against the physical-reference sidecar (the parent's two metrics)."""
    result = json.loads((folder / 'result.json').read_text())
    provenance = json.loads((folder / 'provenance.json').read_text())
    assert provenance['train_index_sha256'] == FNO_POISSON_TRAIN_INDEX, folder
    assert provenance['validation_index_sha256'] == FNO_POISSON_VALIDATION_INDEX, folder
    assert result['pde'] == 'poisson'
    history = json.loads((folder / 'history.json').read_text())
    assert len(history) == result['epochs_completed']
    config = provenance['config']
    for key, value in dict(epochs=500, patience=80, batch_size=8, weight_decay=0.0001).items():
        assert config[key] == value, (folder, key, config[key])
    assert config['learning_rate'] in LEARNING_RATES and config['seed'] == 20260914 and config['family'] in ('unet', 'transolver')
    discrete, physical, declared = [], [], []
    for index, row in enumerate(rows):
        with np.load(data_root / row['path']) as case:
            target = case['target'].copy()
        assert sha(data_root / row['path']) == row['sha256']
        with np.load(folder / (row['case_id'] + '.prediction.npz')) as saved:
            prediction = saved['prediction'].copy()
        assert prediction.shape == target.shape and prediction.dtype == np.float64 and np.isfinite(prediction).all()
        for edge in (prediction[..., 0, :], prediction[..., -1, :], prediction[..., :, 0], prediction[..., :, -1]):
            assert not np.any(edge)
        e = float(np.linalg.norm(prediction - target) / np.linalg.norm(target))
        stated = float(result['validation']['errors'][index][0])
        assert abs(e - stated) <= 1e-11 * max(e, 1e-300) + 1e-13, (row['case_id'], e, stated)
        discrete.append(e)
        declared.append(stated)
        ref = row['reference']
        assert sha(data_root / ref['path']) == ref['sha256']
        with np.load(data_root / ref['path']) as saved:
            reference = saved['target'].copy()
        physical.append(float(np.linalg.norm(prediction - reference) / np.linalg.norm(reference)))
    assert sha(folder / 'best.pt') == result['best_checkpoint_sha256']
    best = history[result['best_epoch']]
    assert best['validation']['mean_case_max'] == result['validation']['batched_selection_score']
    tolerance = 1e-5 if result['parameter_dtype'] == 'torch.float32' else 1e-11
    selection_gap = abs(best['validation']['mean_case_max'] - result['validation']['mean_case_max'])
    assert selection_gap <= tolerance * result['validation']['mean_case_max'], (folder, selection_gap)
    return dict(complete=True, family=result['family'], parameter_dtype=result['parameter_dtype'], pde='poisson',
                config=config, config_sha256=provenance['config_sha256'], seed=provenance['seed'],
                real_parameter_count=result['real_parameter_count'], best_epoch=result['best_epoch'],
                epochs_completed=result['epochs_completed'], stop_reason=stop_reason(result),
                wall_budget_seconds=result['wall_budget_seconds'], training_seconds=result['training_seconds'],
                warmup_epochs=result.get('warmup_epochs', 0), final_learning_rate=history[-1]['learning_rate'],
                batched_selection_score=result['validation']['batched_selection_score'],
                batch1_vs_batch8_selection_gap=selection_gap, best_checkpoint_sha256=result['best_checkpoint_sha256'],
                peak_allocated_bytes=max(h['peak_allocated_bytes'] for h in history),
                discrete=statistics(discrete), physical_candidate=statistics(physical),
                discrete_errors=discrete, physical_candidate_errors=physical, declared_errors=declared,
                validation_case_ids=[r['case_id'] for r in rows],
                train_index_sha256=provenance['train_index_sha256'],
                validation_index_sha256=provenance['validation_index_sha256'])


def audit_arm(folder, rows, data_root):
    result = json.loads((folder / 'result.json').read_text())
    provenance = json.loads((folder / 'provenance.json').read_text())
    assert provenance['train_index_sha256'] == FNO_TRAIN_INDEX, folder
    assert provenance['validation_index_sha256'] == FNO_VALIDATION_INDEX, folder
    assert provenance['final_cohort_opened'] is False
    history = json.loads((folder / 'history.json').read_text())
    assert len(history) == result['epochs_completed']
    config = provenance['config']
    for key, value in PROTOCOL.items():
        assert config[key] == value, (folder, key, config[key])
    assert config['learning_rate'] in LEARNING_RATES and config['seed'] in (20260914, 20260915), folder
    assert config['family'] in ('unet', 'transolver')
    per_case, per_time, declared = [], [], []
    for index, row in enumerate(rows):
        with np.load(data_root / row['path']) as case:
            target, supplied, times = case['target'].copy(), case['input'].copy(), case['times'].copy()
        assert sha(data_root / row['path']) == row['sha256']
        assert times.size == 6 and times[0] == 0
        with np.load(folder / (row['case_id'] + '.prediction.npz')) as saved:
            prediction = saved['prediction'].copy()
        errors = check_prediction(prediction, target, supplied)
        stated = np.asarray(result['validation']['errors'][index], dtype=np.float64)
        assert np.allclose(errors, stated, rtol=1e-11, atol=1e-13), (row['case_id'], errors, stated)
        per_case.append(float(errors.max()))
        per_time.append(errors.tolist())
        declared.append(stated.tolist())
    assert sha(folder / 'best.pt') == result['best_checkpoint_sha256']
    with np.load(folder / 'validation-errors.npz') as saved:
        assert np.allclose(saved['errors'], np.asarray(per_time), rtol=1e-11, atol=1e-13)
    # The selection score in history came from the batch-8 pass; the reported numbers come
    # from the batch-1 pass whose fields are saved. Batched float32 kernels need not be
    # batch-invariant, so the pre-registered tolerance is 1e-5 relative for float32 and
    # 1e-11 for float64.
    best = history[result['best_epoch']]
    assert best['validation']['mean_case_max'] == result['validation']['batched_selection_score']
    tolerance = 1e-5 if result['parameter_dtype'] == 'torch.float32' else 1e-11
    selection_gap = abs(best['validation']['mean_case_max'] - result['validation']['mean_case_max'])
    assert selection_gap <= tolerance * result['validation']['mean_case_max'], (folder, selection_gap)
    return dict(complete=True, family=result['family'], parameter_dtype=result['parameter_dtype'],
                config=provenance['config'], config_sha256=provenance['config_sha256'],
                seed=provenance['seed'], real_parameter_count=result['real_parameter_count'],
                best_epoch=result['best_epoch'], epochs_completed=result['epochs_completed'],
                stop_reason=stop_reason(result), wall_budget_seconds=result['wall_budget_seconds'],
                training_seconds=result['training_seconds'], warmup_epochs=result.get('warmup_epochs', 0),
                final_learning_rate=history[-1]['learning_rate'],
                batched_selection_score=result['validation']['batched_selection_score'],
                batch1_vs_batch8_selection_gap=selection_gap,
                best_checkpoint_sha256=result['best_checkpoint_sha256'],
                peak_allocated_bytes=max(h['peak_allocated_bytes'] for h in history),
                fixed_initial=statistics(per_case), case_maximum_errors=per_case,
                per_time_errors=per_time, worst_per_time=np.asarray(per_time).max(axis=0).tolist(),
                declared_errors=declared, validation_case_ids=[r['case_id'] for r in rows],
                train_index_sha256=provenance['train_index_sha256'],
                validation_index_sha256=provenance['validation_index_sha256'])


def audit_cohort(root, prefix):
    folder = root / 'out/diagnosis-cohort'
    index = root / 'data/diagnosis-cohort/index.json'
    if not (folder / 'cohort-result.json').exists():
        return dict(present=False)
    assert sha(index) == FNO_COHORT_INDEX, 'cohort index must be the FNO job\'s'
    stated = json.loads((folder / 'cohort-result.json').read_text())
    assert stated['cohort_index_sha256'] == FNO_COHORT_INDEX
    rows = json.loads(index.read_text())['records']
    provenance = json.loads((index.parent / 'cohort-provenance.json').read_text())
    assert provenance['disjoint_from_training_by_seed'] and provenance['disjoint_from_training_by_input_field']
    models = {}
    for name, model in stated['models'].items():
        assert name.startswith(prefix + '-')
        per_case, per_time = [], []
        for position, row in enumerate(rows):
            with np.load(index.parent / row['path']) as case:
                target, supplied = case['target'].copy(), case['input'].copy()
            assert sha(index.parent / row['path']) == row['sha256']
            with np.load(folder / name / (row['case_id'] + '.prediction.npz')) as saved:
                prediction = saved['prediction'].copy()
            errors = check_prediction(prediction, target, supplied)
            assert np.allclose(errors, np.asarray(model['errors'][position]), rtol=1e-11, atol=1e-13), row['case_id']
            per_case.append(float(errors.max()))
            per_time.append(errors.tolist())
        assert sha(root / 'out' / name / 'best.pt') == model['checkpoint_sha256']
        models[name] = dict(fixed_initial=statistics(per_case), case_maximum_errors=per_case, per_time_errors=per_time,
                            checkpoint_sha256=model['checkpoint_sha256'], best_epoch=model['best_epoch'])
    return dict(present=True, cases=[r['case_id'] for r in rows], models=models, cohort_index_sha256=FNO_COHORT_INDEX,
                comparability='same eight cases and references as the Burgers lane ROM/FOM diagnosis and the FNO job; '
                              'accuracy comparable across jobs, timing not')


def audit_timing(root):
    folder = root / 'out/timing'
    if not (folder / 'timing.json').exists():
        return dict(present=False)
    stated = json.loads((folder / 'timing.json').read_text())
    arrays = np.load(folder / 'timing.npz')
    checked = {}
    for name, model in stated['models'].items():
        device, host = arrays[f'device_{name}'], arrays[f'host_{name}']
        assert device.shape == host.shape and device.shape[0] == len(stated['cases'])
        assert np.isfinite(device).all() and (device > 0).all()
        recomputed = float(np.median(device) * 1e3)
        assert abs(recomputed - model['device_query_pooled']['median_ms']) <= 1e-9 * max(1., recomputed)
        checked[name] = dict(repetitions=list(device.shape), device_pooled_median_ms=recomputed,
                             device_median_of_case_medians_ms=float(np.median(np.median(device, axis=1)) * 1e3),
                             host_pooled_median_ms=float(np.median(host) * 1e3),
                             device_plus_host_pooled_median_ms=float(np.median(device + host) * 1e3))
    return dict(present=True, gpu=stated['environment']['gpu'], models=checked,
                repetitions_per_case=stated['repetitions_per_case'], burn_in_per_block=stated['burn_in_per_block'],
                limitation='same-job, same-GPU timings of this lane\'s arms only; never divide by another job')


def main(attempt):
    run = LANE / 'runs' / attempt
    root = run / 'archive' / attempt
    provenance = json.loads((run / 'PROVENANCE.json').read_text())
    assert (root / 'COMMIT.txt').read_text().strip() == provenance['source_commit']
    spec = provenance['spec']
    prefix = spec['prefix']
    log = next(iter(sorted((root / 'logs').glob('*.out')))).read_text()
    job_id = re.search(r'job=(\d+)', log).group(1)
    assert 'jax_backend=gpu' in log and log.rstrip().endswith('ALL-DONE'), 'gate G1/G2'
    assert 'torch_backend=cuda' in (root / 'logs/precision.log').read_text()
    gpu = re.search(r'\n(NVIDIA [^,]+), ', log).group(1)
    data_verified = int(re.search(r'data_verified=(\d+)', log).group(1))
    pde = spec.get('pde', 'burgers')
    rows = json.loads((root / 'data/validation/index.json').read_text())['records']
    assert sha(root / 'data/validation/index.json') == (FNO_VALIDATION_INDEX if pde == 'burgers' else FNO_POISSON_VALIDATION_INDEX)
    arm_audit = audit_arm if pde == 'burgers' else audit_arm_poisson
    arms = {}
    for folder in sorted((root / 'out').glob(f'{prefix}-*')):
        arms[folder.name] = arm_audit(folder, rows, root / 'data/validation') if (folder / 'result.json').exists() \
            else dict(complete=False)
    selection = json.loads((root / 'out/capacity-selection.json').read_text())
    worker = json.loads((root / 'out/worker.json').read_text())
    precision = json.loads((root / 'out/precision.json').read_text())
    assert precision['passed']
    expected_arms = {f"{prefix}-{a['name']}" for a in spec['arms']}
    if spec.get('refine_learning_rate'):
        expected_arms.add(f'{prefix}-refine')
    assert set(arms) == expected_arms, (set(arms), expected_arms)
    assert not selection['stopped_by_signal']
    cohort = audit_cohort(root, prefix) if pde == 'burgers' else dict(present=False, note='no matched cohort for Poisson')
    assert pde == 'poisson' or (cohort['present'] and set(cohort['models']) == expected_arms), 'cohort must be scored for every arm'
    timing = audit_timing(root)
    assert timing['present'] and set(timing['models']) == expected_arms
    result = dict(attempt=attempt, job_id=job_id, gpu=gpu, source_commit=provenance['source_commit'], pde=pde,
                  spec=spec, data_files_verified=data_verified, jax_backend='gpu',
                  train_index_sha256=FNO_TRAIN_INDEX if pde == 'burgers' else FNO_POISSON_TRAIN_INDEX,
                  validation_index_sha256=FNO_VALIDATION_INDEX if pde == 'burgers' else FNO_POISSON_VALIDATION_INDEX,
                  identical_split_to_fno_job=True, validation_cases=len(rows),
                  arms=arms, capacity_selection=selection, worker_tasks=worker,
                  cohort=cohort, timing=timing,
                  metric='maximum over the six requested output times of the interior l2 discrepancy divided by '
                         'the interior l2 norm of the supplied initial field (Burgers lane metric)',
                  passed=bool(arms) and all(a.get('complete') for a in arms.values()),
                  limitation='single seed, one mesh, one Gaussian continuum family, bounded wall budget per arm; '
                             'no speed ratio, no cross-job timing')
    (run / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(job=job_id, gpu=gpu, passed=result['passed'],
                          arms={k: ((v['fixed_initial'] if pde == 'burgers' else v['physical_candidate'])['maximum'],
                                    v['epochs_completed'], v['stop_reason'])
                                for k, v in arms.items() if v.get('complete')},
                          cohort={k: v['fixed_initial']['maximum'] for k, v in result['cohort'].get('models', {}).items()})))


if __name__ == '__main__':
    main(sys.argv[1])
