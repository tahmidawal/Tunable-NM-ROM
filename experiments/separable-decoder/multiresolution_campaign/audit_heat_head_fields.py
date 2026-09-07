"""Independent NumPy audit of saved heat head-refinement fields and accounting.

No experiment module or device computation is imported. This checks preserved
outputs, frozen weights, declared membership and timing repetitions; it does not
prove a continuum reference bound or global nonlinear optimality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import statistics

import numpy as np


def restrict(a, source, target):
    assert source % target == 0
    stride = source // target
    return a[..., stride-1::stride, stride-1::stride]


def metrics(a, b, n):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    difference = np.linalg.norm(a-b, axis=1)
    norms = np.linalg.norm(b, axis=1)
    assert np.all(np.isfinite(a)) and np.all(norms > 0)
    return dict(relative_current=difference/norms, relative_initial=difference/norms[0],
                absolute_l2=difference/n, truth_norm_over_initial=norms/norms[0],
                energy=np.sum(a*a, axis=1)/(2*n*n))


def compare_tree(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare_tree(x, y)
    else:
        np.testing.assert_array_equal(a, b)


def audit(path):
    native = json.loads(path.read_text())
    cfg, settings = native['config'], native['settings']
    meta = native['metadata']
    assert native['complete'] and meta['backend'] == 'gpu' and meta['x64']
    assert meta['precision'] == 'highest' and not settings['final_cohort_opened']
    assert not native['cohorts']['validation_in_training_or_lookup']
    assert not native['cohorts']['final_cohort_opened']
    assert len(native['cohorts']['original_train']) == cfg['n_train']
    assert len(native['cohorts']['additional_train']) == settings['additional_training_trajectories']
    assert len(native['cohorts']['validation']) == cfg['n_validation']
    train = native['cohorts']['original_train'] + native['cohorts']['additional_train']
    assert all(v not in train for v in native['cohorts']['validation'])
    for label, seed, count in (('original_train', cfg['train_seed'], cfg['n_train']),
                               ('additional_train', settings['additional_training_seed'], settings['additional_training_trajectories']),
                               ('validation', cfg['validation_seed'], cfg['n_validation'])):
        rng = np.random.default_rng(seed)
        draws = np.column_stack((rng.uniform(*cfg['center_range'], (count, 2)),
                                 rng.uniform(*cfg['width_range'], count), rng.uniform(*cfg['amplitude_range'], count)))
        np.testing.assert_array_equal(draws, native['cohorts'][label])
    artifacts, loaded = {}, {}
    def field(record):
        name = record['path']
        if name not in loaded:
            p = path.parent/name
            with np.load(p) as saved:
                a = np.ascontiguousarray(saved['field'])
            digest = hashlib.sha256(str((a.shape, a.dtype.str)).encode()+a.tobytes()).hexdigest()
            assert digest == record['sha256_array']
            assert list(a.shape) == record['shape'] and a.dtype.str == record['dtype']
            assert a.dtype == np.float64 and np.all(np.isfinite(a))
            artifacts[name] = hashlib.sha256(p.read_bytes()).hexdigest()
            loaded[name] = a
        return loaded[name]
    worst = 0.
    def check(a, b, n, recorded):
        nonlocal worst
        observed = metrics(a, b, n)
        for key, value in observed.items():
            np.testing.assert_allclose(value, recorded[key], rtol=3e-12, atol=3e-14)
            worst = max(worst, float(np.max(np.abs(value-recorded[key]))))
        return observed
    expected_models = {'frozen'} | {f'{cohort}_seed{seed}' for cohort in ('original', 'expanded')
                                    for seed in settings['minibatch_seeds']}
    assert {m['name'] for m in native['models']} == expected_models
    models = {}
    for row in native['models']:
        checkpoint = path.parent/row['checkpoint_path']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row['checkpoint_sha256']
        with checkpoint.open('rb') as handle:
            models[row['name']] = pickle.load(handle)
        artifacts[row['checkpoint_path']] = row['checkpoint_sha256']
        if row['name'] != 'frozen':
            assert row['details']['sampled_snapshots'] == settings['updates']*settings['batch_size']
    origins = [parent/settings['checkpoint_path'] for parent in path.parents
               if (parent/settings['checkpoint_path']).is_file()]
    assert len(origins) == 1
    assert hashlib.sha256(origins[0].read_bytes()).hexdigest() == settings['checkpoint_sha256']
    with origins[0].open('rb') as handle:
        original = pickle.load(handle)
    compare_tree(original['params'], models['frozen']['params'])
    compare_tree(original['codes'], models['frozen']['codes'])
    for name, model in models.items():
        for key in ('B', 'g', 'out_scale'):
            compare_tree(models['frozen']['params'][key], model['params'][key])
        count = cfg['n_train'] + (settings['additional_training_trajectories'] if name.startswith('expanded') else 0)
        assert len(model['codes']) == count*len(cfg['times'])
    bank = native['bank_projection']
    truth, projected = field(bank['truth']), field(bank['reconstructed'])
    refs = {r['case']: (r['intervals'], field(r['field'])) for r in native['reference_fields']}
    assert set(refs) == set(range(cfg['n_validation']))
    assert {r['model'] for r in native['reconstruction']} == expected_models
    reconstructions, qualified = [], []
    for row in native['reconstruction']:
        predicted = field(row['reconstructed'])
        info = np.asarray(row['fits'])
        best = np.asarray(row['selected_starts'])
        np.testing.assert_array_equal(np.argmin(info[:, :, 3], axis=1), best)
        chosen = info[np.arange(len(best)), best]
        cases, initial_ok = [], True
        for case in row['cases']:
            c = case['case']
            observed = check(predicted[c], truth[c], row['intervals'], case['vs_same_grid'])
            check(projected[c], truth[c], row['intervals'], case['bank_projection'])
            fine, ref = refs[c]
            check(predicted[c], restrict(ref, fine, row['intervals']), row['intervals'], case['vs_physical'])
            initial_info = chosen[c*len(cfg['times'])]
            stationary = bool(initial_info[2] == 1 and initial_info[4] <= settings['fit_gradient_tolerance'])
            initial_ok &= stationary and observed['relative_current'][0] <= settings['rollout_gate']['initial_worst_max']
            cases.append(dict(case=c, initial_error=float(observed['relative_current'][0]),
                              later_worst_error=float(max(observed['relative_current'][1:])), initial_stationary=stationary))
        passed = bool(initial_ok and row['model'] != 'frozen')
        assert row['rollout_gate_passed'] == passed
        if passed:
            qualified.append(row['model'])
        reconstructions.append(dict(model=row['model'], cases=cases, rollout_gate_passed=passed,
            nonstationary_snapshot_fits=int(np.sum((chosen[:, 2] != 1) | (chosen[:, 4] > settings['fit_gradient_tolerance'])))))
    assert set(qualified) == set(native['qualified_refinements'])
    assert native['rollout_performed'] == bool(qualified)
    expected_methods = {'fom_dst_exact_time'} | {f'{name}_gtol{tol:g}' for name in ['frozen']+qualified
                                               for tol in settings['rollout_gradient_tolerances']}
    expected = {(n, c, m) for n in settings['rollout_intervals'] for c in range(cfg['n_validation'])
                for m in expected_methods} if qualified else set()
    assert {(r['intervals'], r['case'], r['method']) for r in native['rows']} == expected
    assert len(native['rows']) == len(expected)
    case_fields = {(r['intervals'], r['case']): r for r in native['case_fields']}
    invocations, rows = 0, []
    for row in native['rows']:
        n, case = row['intervals'], row['case']
        fine, ref = refs[case]
        saved = case_fields[n, case]
        discrete = field(saved['discrete'])
        np.testing.assert_array_equal(field(saved['physical']), restrict(ref, fine, n))
        assert {r['repetition'] for r in row['repetitions']} == set(range(settings['timing_repetitions']))
        assert len(row['repetitions']) == settings['timing_repetitions']
        errors, times, initial_nonstationary, step_nonstationary = [], [], 0, 0
        for rep in row['repetitions']:
            a = field(rep['field'])
            check(a, discrete, n, rep['vs_same_grid'])
            check(a, restrict(ref, fine, n), n, rep['vs_physical_per_grid'])
            common = settings['observation_intervals']
            measured = check(restrict(a, n, common), restrict(ref, fine, common), common, rep['vs_physical_common_grid'])
            phases = rep['phases']
            assert all(np.isfinite(v) and v >= 0 for v in phases.values())
            assert sum(v for k, v in phases.items() if k != 'query_seconds') <= phases['query_seconds']+1e-12
            errors.append(float(max(measured['relative_current'])))
            times.append(phases['query_seconds'])
            if row['model'] != 'fom':
                fits, steps = np.asarray(rep['solver']['initial_fits']), np.asarray(rep['solver']['steps'])
                initial = fits[np.argmin(fits[:, 3])]
                initial_nonstationary += int(initial[2] != 1 or initial[4] > row['gradient_tolerance'])
                step_nonstationary += int(np.sum((steps[:, 2] != 1) | (steps[:, 4] > row['gradient_tolerance'])))
            invocations += 1
        rows.append(dict(intervals=n, case=case, method=row['method'], model=row['model'],
                         query_median_seconds=statistics.median(times), worst_current_error=max(errors),
                         initial_nonstationary_repetitions=initial_nonstationary,
                         nonstationary_steps_all_repetitions=step_nonstationary))
    return dict(scope=__doc__, source_json=str(path), source_json_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_commit=native['source_manifest']['source_commit'], job_id=meta['job_id'],
                artifact_hashes=artifacts, maximum_metric_disagreement=worst, verified_timed_invocations=invocations,
                frozen_spatial_parameters_match=True, declared_endpoint_and_repetition_counts_match=True,
                reconstruction=reconstructions, rows=rows, rigorous_reference_bound=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('job_id', 'maximum_metric_disagreement', 'verified_timed_invocations')}))
