"""Independent NumPy audit of one collected attempt. Imports neither torch nor jax.

    python audit.py <attempt>

Rewritten for this lane rather than inherited: both design audits found that
`ops-deeponet-b2d`'s `audit.py` would abort every job here for reasons unrelated to the
science — it asserts the FNO train-index hash for *every* arm (no extended-data arm can pass),
a frozen `batch_size`/`patience`/`learning_rate` whitelist (no tuned arm can pass), one history
entry per epoch (this lane validates on a wall fraction), a `capacity-selection.json` this lane
does not write, and a timing block this lane deliberately does not run. Keeping its kind of
checking and dropping its literals is the only way both can be true.

What it does:

  * recomputes every reported per-case, per-time fixed-initial error from the saved prediction
    fields and the archived validation cases, and checks the supplied initial state is returned
    bitwise, the boundary is exactly zero and every prediction is finite float64;
  * checks each arm against **its own declared data source and prefix** — the pinned index hash
    for pinned-mount arms, the generated index hash and the exact case count for extended-data
    arms — instead of one global literal;
  * rebuilds every arm's configuration from the base config plus the spec's declared override
    (plus the recorded schedule decision) and asserts the config the arm actually trained with
    is that, so "one declared override per arm" is verified, not asserted;
  * recomputes the schedule decision (S1), the composition (S2) and the selection (S3) from the
    arm scores, independently of the worker;
  * recomputes the persistence control on both cohorts, and the label-discrepancy ratio the
    T0 criterion needs;
  * records, per arm, its parameters, steps, evaluations, epochs, realised budget, what ended
    it, whether its patience could fire at all, and the training-versus-validation gap.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

LANE = Path(__file__).resolve().parent
# The pinned cache's two index hashes. Unlike the parent lane, these are not bare literals:
# `checks/pinned/` holds the two index files themselves and this asserts they hash to these.
PINNED_TRAIN_INDEX = '5333584b7162df622ec7bc2b08e68d8003036b1b05abbea520249ed408fc8d49'
PINNED_VALIDATION_INDEX = '468b9e70df3392c4b5bbb41381c9062d3d697ac4772b7236c4ee59d0ca73ebad'
FNO_COHORT_INDEX = json.loads((LANE.parent / 'neural-operator-audit/runs/fno_burgers02/field-audit.json')
                              .read_text())['diagnosis_cohort']['cohort_index_sha256']
STOP_REASONS = ('signal', 'early_stopping', 'wall_budget', 'epoch_cap')


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


def expected_config(root, spec, arm, schedule_decision, composition):
    """The configuration an arm should have trained with: base + the declared override."""
    base = json.loads((root / spec['base_config']).read_text())
    merged = dict(base)
    sweep = {a['name']: a for a in spec['arms']}
    index = [a['name'] for a in spec['arms']]
    if arm in sweep:
        rule = spec.get('schedule_decision')
        # The schedule decision applies to every arm AFTER the deciding arm.
        if rule and schedule_decision and index.index(arm) > index.index(rule['arm']):
            merged.update(schedule_decision)
        merged.update(sweep[arm].get('override', {}))
    elif spec.get('compose') and arm == spec['compose']['arm']['name']:
        merged.update(schedule_decision or {})
        merged.update(composition['override'])
    else:
        raise AssertionError(f'arm {arm} is not in the spec')
    return merged


def audit_arm(folder, rows, data_root, spec, expected, generated_index_sha):
    name = folder.name.split('-', 1)[1]
    result = json.loads((folder / 'result.json').read_text())
    provenance = json.loads((folder / 'provenance.json').read_text())
    assert provenance['validation_index_sha256'] == PINNED_VALIDATION_INDEX, folder
    assert provenance['final_cohort_opened'] is False
    arm = next(a for a in spec['arms'] if a['name'] == name) if any(a['name'] == name for a in spec['arms']) \
        else spec['compose']['arm']
    mount = spec['mounts'][arm['mount']]

    # Each arm against ITS OWN declared source and prefix, not one global literal.
    if arm['mount'] == 'pinned':
        assert provenance['train_index_sha256'] == PINNED_TRAIN_INDEX, folder
        assert provenance['pool'] is None
    else:
        assert generated_index_sha is not None, 'no generated index to check the extended arms against'
        assert provenance['train_index_sha256'] == generated_index_sha, folder
        assert provenance['pool'] and provenance['pool']['index_sha256'] == generated_index_sha
        assert provenance['pool']['limit'] == arm.get('pool_limit'), folder
    declared_cases = arm.get('pool_limit')
    if declared_cases is not None:
        assert provenance['training_cases'] == declared_cases == result['training_cases'], folder

    # One declared override per arm, verified against the config that actually trained.
    config = provenance['config']
    for key, value in expected.items():
        assert config.get(key) == value, (folder, key, config.get(key), value)
    extra = {k: v for k, v in config.items()
             if k not in expected and k not in ('arm', 'knob', 'derived_from', 'applied_schedule_decision')}
    assert not extra, (folder, 'configuration keys not accounted for by base + override', extra)

    history = json.loads((folder / 'history.json').read_text())
    assert len(history) == result['evaluations_completed']
    assert [h['step'] for h in history] == sorted(h['step'] for h in history)
    position = next(i for i, h in enumerate(history) if h['step'] == result['best_step'])
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
    # The selection score came from the batch-8 pass; the reported numbers from the batch-1 pass
    # whose fields are saved. Batched float32 kernels need not be batch-invariant, so the
    # pre-registered tolerance is 1e-5 RELATIVE for float32 and 1e-11 for float64.
    best = history[position]
    assert best['validation']['mean_case_max'] == result['validation']['batched_selection_score']
    tolerance = 1e-5 if result['parameter_dtype'] == 'torch.float32' else 1e-11
    selection_gap = abs(best['validation']['mean_case_max'] - result['validation']['mean_case_max'])
    assert selection_gap <= tolerance * result['validation']['mean_case_max'], (folder, selection_gap)
    reason = result['stop_reason']
    assert reason in STOP_REASONS and result[f'stopped_by_{reason}'] is True
    floor = config.get('min_learning_rate', 1e-5)
    evaluations_after_best = len(history) - 1 - position
    return dict(complete=True, arm=name, family=result['family'], mount=arm['mount'],
                parameter_dtype=result['parameter_dtype'], config=config,
                config_sha256=provenance['config_sha256'], seed=provenance['seed'],
                knob=arm.get('knob'), override=arm.get('override', {}),
                training_cases=result['training_cases'],
                train_index_sha256=provenance['train_index_sha256'],
                validation_index_sha256=provenance['validation_index_sha256'],
                real_parameter_count=result['real_parameter_count'],
                best_step=result['best_step'], best_epoch=result['best_epoch'],
                steps_completed=result['steps_completed'], epochs_completed=result['epochs_completed'],
                evaluations_completed=result['evaluations_completed'],
                examples_seen=result['steps_completed'] * config['batch_size'],
                schedule=result['schedule'], output_scale_mode=result['output_scale_mode'],
                stop_reason=reason, wall_budget_seconds=result['wall_budget_seconds'],
                budget_shortened=result['wall_budget_seconds'] < arm['seconds'] - 1,
                training_seconds=result['training_seconds'],
                patience_could_fire=evaluations_after_best >= config['patience_evaluations'] or reason == 'early_stopping',
                evaluations_after_best=evaluations_after_best,
                evaluations_at_minimum_learning_rate=sum(1 for h in history if h['learning_rate'] <= floor * 1.000001),
                final_learning_rate=history[-1]['learning_rate'],
                train_loss_at_best=result['train_loss_at_best'], train_loss_final=result['train_loss_final'],
                training_subset=result['training_subset'],
                generalisation_gap=result['validation']['mean_case_max'] / max(result['training_subset']['mean_case_max'], 1e-300),
                batched_selection_score=result['validation']['batched_selection_score'],
                batch1_vs_batch8_selection_gap=selection_gap,
                best_checkpoint_sha256=result['best_checkpoint_sha256'],
                peak_allocated_bytes=max(h['peak_allocated_bytes'] for h in history),
                fixed_initial=statistics(per_case), case_maximum_errors=per_case,
                per_time_errors=per_time, worst_per_time=np.asarray(per_time).max(axis=0).tolist(),
                mean_per_time=np.asarray(per_time).mean(axis=0).tolist(), declared_errors=declared)


def persistence_baseline(rows, data_root):
    """The trivial floor a reader needs to size any operator error: predict u(t) = u(0) at every
    requested output time, scored by the same metric on the same cases."""
    per_case = []
    for row in rows:
        with np.load(data_root / row['path']) as case:
            target, supplied = case['target'].copy(), case['input'].copy()
        prediction = np.repeat(supplied[None], target.shape[0], axis=0)
        per_case.append(float(fixed_initial_errors(prediction, target).max()))
    return dict(cases=len(rows), fixed_initial=statistics(per_case), case_maximum_errors=per_case,
                definition='hold the supplied initial state at every output time; no training, no parameters')


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
        models[name] = dict(fixed_initial=statistics(per_case), case_maximum_errors=per_case,
                            per_time_errors=per_time, checkpoint_sha256=model['checkpoint_sha256'])
    return dict(present=True, cases=[r['case_id'] for r in rows], models=models,
                cohort_index_sha256=FNO_COHORT_INDEX,
                built_against='the PINNED 128-case training index, so this index hash is the FNO job\'s; '
                              'disjointness from the EXTENDED training cases is checked separately',
                comparability='same eight cases and references as the Burgers lane ROM/FOM diagnosis and the '
                              'FNO job; accuracy comparable across jobs, timing not; these are reused '
                              'calibration cases, not an independent test set',
                selection_role='none: no selection in this lane uses the cohort (DESIGN 5.1)')


def recompute_decisions(spec, arms, worker_decision, composition):
    """S1, S2 and S3 of DESIGN 5.1, recomputed from the arm scores here."""
    score = {name.split('-', 1)[1]: a['fixed_initial']['mean'] for name, a in arms.items() if a.get('complete')}
    out = dict(scores=score)
    rule = spec.get('schedule_decision')
    if rule and rule['arm'] in score and rule['reference'] in score:
        won = score[rule['arm']] < score[rule['reference']]
        out['schedule_decision'] = dict(applied=dict(rule['override']) if won else {},
                                        challenger=score[rule['arm']], reference=score[rule['reference']],
                                        matches_worker=(dict(rule['override']) if won else {}) == (worker_decision or {}))
    if spec.get('compose') and composition:
        reference = spec['compose']['reference']
        threshold = spec['compose']['threshold']
        winners = {}
        for entry in spec['arms']:
            if 'knob' not in entry or entry['name'] not in score or entry['name'] == reference:
                continue
            relative = (score[reference] - score[entry['name']]) / score[reference]
            if relative >= threshold and (entry['knob'] not in winners
                                          or score[entry['name']] < score[winners[entry['knob']]]):
                winners[entry['knob']] = entry['name']
        override = {}
        for knob in sorted(winners):
            override.update(next(a for a in spec['arms'] if a['name'] == winners[knob])['override'])
        out['composition'] = dict(winners=winners, override=override,
                                  matches_worker=override == composition['override'])
    candidates = {name: value for name, value in score.items()
                  if any(a['name'] == name and 'knob' in a for a in spec['arms'])
                  or name in (spec.get('compose', {}).get('arm', {}).get('name'), spec.get('compose', {}).get('reference'))}
    if candidates:
        selected = min(candidates, key=lambda k: (candidates[k], arms[f"{spec['prefix']}-{k}"]['fixed_initial']['maximum']))
        best_worst = min(candidates, key=lambda k: arms[f"{spec['prefix']}-{k}"]['fixed_initial']['maximum'])
        out['selection'] = dict(candidates=sorted(candidates), selected=selected, selected_score=candidates[selected],
                                best_worst_case_arm=best_worst,
                                selected_worst=arms[f"{spec['prefix']}-{selected}"]['fixed_initial']['maximum'],
                                best_worst=arms[f"{spec['prefix']}-{best_worst}"]['fixed_initial']['maximum'],
                                rule='DESIGN 5.1: argmin validation mean case-maximum over the sweep arms and '
                                     'the composed arm; ladder rungs and the 128-case controls are not candidates')
    return out


def main(attempt):
    run = LANE / 'runs' / attempt
    root = run / 'archive' / attempt
    provenance = json.loads((run / 'PROVENANCE.json').read_text())
    assert (root / 'COMMIT.txt').read_text().strip() == provenance['source_commit']
    spec = provenance['spec']
    prefix = spec['prefix']
    log = next(iter(sorted((root / 'logs').glob('*.out')))).read_text()
    job_id = re.search(r'job=(\d+)', log).group(1)
    assert 'jax_backend=gpu' in log and log.rstrip().endswith('ALL-DONE'), 'gate 1'
    assert 'torch_backend=cuda' in (root / 'logs/precision.log').read_text()
    gpu = re.search(r'\n(NVIDIA [^,]+), ', log).group(1)
    verified = {m.group(1): int(m.group(2)) for m in re.finditer(r'data_verified_(\w+)=(\d+)', log)}
    assert 'pinned' in verified, 'gate 2: the pinned cache must verify through its mount'

    # The pinned indices, held in this lane so the two "literals" are checkable here.
    assert sha(LANE / 'checks/pinned/train-index.json') == PINNED_TRAIN_INDEX
    assert sha(LANE / 'checks/pinned/validation-index.json') == PINNED_VALIDATION_INDEX
    validation_index = root / 'data/pinned/validation/index.json'
    assert sha(validation_index) == PINNED_VALIDATION_INDEX, 'gate 3'
    rows = json.loads(validation_index.read_text())['records']

    generated = root / 'data/ext/train/index.json'
    generated_index_sha = sha(generated) if generated.exists() else None
    extended = dict(present=generated.exists())
    if generated.exists():
        index = json.loads(generated.read_text())
        assert index['complete'] and index['split'] == 'train'
        comparison = index['pinned_comparison']
        extended = dict(present=True, index_sha256=generated_index_sha, cases=index['count'],
                        reference_setting=index['reference_setting'], profile=index['profile'],
                        protocol_sha256=index['protocol_sha256'],
                        generating_job=index['provenance'].get('job_id'),
                        label_discrepancy=dict(cases=comparison['cases'], maximum=comparison['maximum'],
                                               median=comparison['median'], mean=comparison['mean'],
                                               per_time_mean=comparison['per_time_mean']),
                        note='training targets only; every evaluation cohort is pinned refined-anchor data')

    decision_path = root / 'out/schedule-decision.json'
    worker_decision = json.loads(decision_path.read_text())['applied'] if decision_path.exists() else None
    composition_path = root / 'out/composition.json'
    composition = json.loads(composition_path.read_text()) if composition_path.exists() else None

    arms = {}
    for folder in sorted((root / 'out').glob(f'{prefix}-*')):
        if not (folder / 'result.json').exists():
            arms[folder.name] = dict(complete=False)
            continue
        expected = expected_config(root / 'code', spec, folder.name.split('-', 1)[1],
                                   worker_decision, composition)
        arms[folder.name] = audit_arm(folder, rows, validation_index.parent, spec, expected, generated_index_sha)

    selection = json.loads((root / 'out/selection.json').read_text())
    worker = json.loads((root / 'out/worker.json').read_text())
    precision = json.loads((root / 'out/precision.json').read_text())
    assert precision['passed']
    planned = {f"{prefix}-{a['name']}" for a in spec['arms']}
    if spec.get('compose'):
        planned.add(f"{prefix}-{spec['compose']['arm']['name']}")
    skipped = {r['task']: r['skipped'] for r in worker if 'skipped' in r}
    missing = planned - set(arms) - {f'{prefix}-{name}' for name in skipped}
    assert not missing, f'arms neither present nor recorded as skipped: {missing}'
    baseline = dict(validation=persistence_baseline(rows, validation_index.parent))
    cohort_index = root / 'data/diagnosis-cohort/index.json'
    if cohort_index.exists():
        baseline['diagnosis-8'] = persistence_baseline(json.loads(cohort_index.read_text())['records'],
                                                       cohort_index.parent)
    cohort = audit_cohort(root, prefix)
    decisions = recompute_decisions(spec, arms, worker_decision, composition)

    # T0's ratios: the measured label perturbation against each arm's own error (DESIGN 5.2).
    # Three, because one number would hide which statistic is being compared with which:
    # mean-to-mean is the like-for-like reading, worst-to-worst the tail reading, and
    # worst-to-mean the deliberately conservative one.
    label = extended.get('label_discrepancy') or {}
    ratios = {}
    for name, a in arms.items():
        if not a.get('complete') or not label or a['mount'] == 'pinned':
            continue
        f = a['fixed_initial']
        ratios[name] = dict(mean_over_mean=label['mean'] / f['mean'],
                            worst_over_worst=label['maximum'] / f['maximum'],
                            worst_over_mean=label['maximum'] / f['mean'])
    complete = [a for a in arms.values() if a.get('complete')]
    result = dict(attempt=attempt, job_id=job_id, gpu=gpu, source_commit=provenance['source_commit'],
                  pde='burgers', spec=spec, data_verified=verified, jax_backend='gpu',
                  validation_index_sha256=PINNED_VALIDATION_INDEX, validation_cases=len(rows),
                  pinned_train_index_sha256=PINNED_TRAIN_INDEX, extended_training_data=extended,
                  arms=arms, skipped_arms=skipped, selection_record=selection, worker_tasks=worker,
                  decisions=decisions, cohort=cohort, persistence_baseline=baseline,
                  label_discrepancy_ratio=ratios,
                  label_protocol_negligible={k: bool(v['mean_over_mean'] <= 0.2 and v['worst_over_worst'] <= 0.2)
                                             for k, v in ratios.items()},
                  label_ratio_definition='rho = measured training-label discrepancy / arm validation error, '
                                         'in three pairings; negligible requires mean-over-mean AND '
                                         'worst-over-worst at or below 0.2 (DESIGN 5.2 T0). The '
                                         'perturbation is on TRAINING labels only; every evaluation '
                                         'cohort is pinned refined-anchor data.',
                  timing=dict(present=False, reason='no timing block runs in this lane; no speed '
                                                    'statement from it is admissible (DESIGN 7)'),
                  metric='maximum over the six requested output times of the interior l2 discrepancy divided by '
                         'the interior l2 norm of the supplied initial field (Burgers lane metric)',
                  passed=bool(complete) and all(a.get('complete') for a in arms.values()),
                  limitation='single seed, one mesh, one Gaussian continuum family, bounded wall budget per arm; '
                             'no speed ratio, no cross-job timing; the eight-case cohort is reused calibration '
                             'evidence, not an independent test set')
    (run / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(job=job_id, gpu=gpu, passed=result['passed'],
                          arms={k: (round(v['fixed_initial']['mean'] * 100, 4),
                                    round(v['fixed_initial']['maximum'] * 100, 4),
                                    v['training_cases'], v['stop_reason'])
                                for k, v in arms.items() if v.get('complete')},
                          decisions=decisions.get('selection'),
                          label_ratio={k: {n: round(x, 4) for n, x in v.items()}
                                       for k, v in ratios.items()}), indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
