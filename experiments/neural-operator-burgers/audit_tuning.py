"""Independent NumPy audit of a fixed-checkpoint tuning run.

Recomputes every reported error and stopping record from the saved arrays, checks
the output contract, the quadrature rules, the arm specifications against the
immutable configuration, and repetition coverage.  Nothing here trusts the run's
own summary numbers.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import data as d
from refine import configure


def validate_fields(field, intervals):
    assert field.shape == (6, intervals + 1, intervals + 1) and field.dtype == np.float64
    assert np.isfinite(field).all()
    assert not np.any(field[:, [0, -1], :]) and not np.any(field[:, :, [0, -1]])


def independent_errors(field, truth):
    f, t = field[:, 1:-1, 1:-1], truth[:, 1:-1, 1:-1]
    return np.sqrt(np.sum((f - t) ** 2, axis=(1, 2))) / np.sqrt(np.sum(t[0] * t[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True,
                        help='calibration reference index, or the validation dataset index')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    configure()
    report = json.loads(args.index.read_text())
    reference = json.loads(args.reference.read_text())
    cfg = report['config']
    L = cfg['fixed']['intervals']
    assert report['provenance']['backend'] == 'gpu' and report['provenance']['f64']
    assert report['provenance']['matmul_precision'] == 'highest'
    assert d.sha(args.reference) in (cfg['reference_index_sha256'], cfg['validation_index_sha256'])
    assert report['checkpoint_sha256'] == cfg['checkpoint_sha256']

    # truth for every case that appears in the run
    truths = {}
    if report['phase'] == 'calibration':
        anchor = cfg['reference_anchor']
        for index in range(8):
            case_id = d.case_record('calibration', index)['case_id']
            row = next((r for r in reference['solves'] if r['case_id'] == case_id
                        and r['intervals'] == anchor['intervals'] and r['dt'] == anchor['dt']), None)
            if row is None:
                continue
            path = args.reference.parent / row['path']
            assert d.sha(path) == row['sha256']
            truths[case_id] = d.restrict(np.load(path)['fields'], L)
    else:
        for row in reference['records']:
            path = args.reference.parent / row['path']
            assert d.sha(path) == row['sha256']
            with np.load(path) as arrays:
                truths[row['case_id']] = np.ascontiguousarray(arrays['target'][:, 0], dtype=np.float64)

    checked_arrays = 0
    worst_error_difference = 0.
    worst_gradient_difference = 0.
    stopping_mismatches = []
    contract_failures = []
    repetition_hash_mismatches = []
    unexpected_cap_exits = []
    incomplete_repetitions = []
    seen = {}
    for row in report['invocations']:
        key = (row['pass_name'], row['case_id'], row['method'])
        seen.setdefault(key, []).append(row)
        if row.get('arm'):
            spec = row['arm']
            declared = cfg['settings'][spec['setting']]
            assert spec['step_budget'] == declared['step_budget']
            assert spec['evolution_gtol'] == declared['evolution_gtol']
            assert spec['evolution_residual_scale'] == declared['evolution_residual_scale']
            assert spec['ic_budget'] == cfg['fixed']['initial_fit']['ic_budget']
            assert spec['initial_gtol'] == cfg['fixed']['initial_fit']['gtol']
        if not row.get('path'):
            continue
        path = args.index.parent / row['path']
        assert d.sha(path) == row['sha256']
        truth = truths[row['case_id']]
        with np.load(path) as arrays:
            fields = arrays['fields']
            validate_fields(fields, L)
            if not np.array_equal(fields[0], truth[0]):
                contract_failures.append(row['path'])
            actual = independent_errors(fields, truth)
            expected = np.asarray(row['error']['per_time'])
            worst_error_difference = max(worst_error_difference, float(np.abs(actual - expected).max()))
            assert np.allclose(actual, expected, rtol=1e-10, atol=1e-14)
            if row.get('arm'):
                reasons = np.asarray(arrays['step_reasons']).astype(int)
                gradients = np.asarray(arrays['step_gradients'])
                iterations = np.asarray(arrays['iterations']).astype(int)
                recomputed = dict(step_iterations_total=int(iterations.sum()),
                                  steps_at_iteration_cap=int((reasons == 0).sum()),
                                  early_stopped=bool((reasons == 0).any()),
                                  gradient_stationary=bool(gradients.max() <= 1e-6
                                                           and float(arrays['initial_gradient']) <= 1e-6))
                for name, value in recomputed.items():
                    if row[name] != value:
                        stopping_mismatches.append(dict(path=row['path'], field=name,
                                                        recorded=row[name], recomputed=value))
                worst_gradient_difference = max(worst_gradient_difference,
                                                abs(float(gradients.max()) - row['worst_step_normalized_gradient']))
                assert int(iterations.sum()) <= spec['step_budget'] * len(reasons)
                if (reasons == 0).any() and spec['setting'] == 'native':
                    unexpected_cap_exits.append(row['path'])
            else:
                residuals = np.asarray(arrays['residuals'])
                assert np.isfinite(residuals).all()
                assert row['converged'] == bool(residuals.max() <= row['preset']['ntol'])
            checked_arrays += 1
    for key, rows in seen.items():
        if len(rows) != report['repetitions']:
            incomplete_repetitions.append(dict(key=list(key), repetitions=len(rows)))
        hashes = {r['fields_sha256'] for r in rows}
        if len(hashes) != 1:
            repetition_hash_mismatches.append(dict(key=list(key), hashes=sorted(hashes)))

    for rule in report['eq_rules']:
        if rule.get('eq_weights'):
            weights = np.asarray(rule['eq_weights'])
            assert np.all(weights >= 0) and len(weights) == rule['actual_m']
            assert len(set(rule['eq_indices'])) == rule['actual_m']

    record = dict(scope='independent recomputation of a fixed-checkpoint tuning run from its saved arrays',
                  passed=bool(not contract_failures and not stopping_mismatches
                              and not (report['complete'] and incomplete_repetitions)),
                  index_sha256=d.sha(args.index), phase=report['phase'], complete=report['complete'],
                  job_id=report['provenance']['job_id'], gpu=report['provenance']['gpu'],
                  source_commit=report['provenance']['source_commit'],
                  invocations=len(report['invocations']), arrays_recomputed=checked_arrays,
                  distinct_case_arm_pairs=len(seen), repetitions=report['repetitions'],
                  worst_absolute_error_metric_difference=worst_error_difference,
                  worst_absolute_gradient_difference=worst_gradient_difference,
                  supplied_initial_output_contract_failures=contract_failures,
                  stopping_record_mismatches=stopping_mismatches,
                  repetition_output_hash_mismatches=repetition_hash_mismatches,
                  incomplete_repetitions=incomplete_repetitions,
                  unexpected_native_setting_cap_exits=unexpected_cap_exits,
                  note=('a repetition hash mismatch is a determinism failure, not necessarily an error; '
                        'timings are expected to vary while outputs are not'))
    d.write_json(args.out, record)
    print(json.dumps(record, indent=1), flush=True)
    assert record['passed']


if __name__ == '__main__':
    main()
