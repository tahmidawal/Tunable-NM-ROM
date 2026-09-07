"""Independently recompute the heat pilot errors and shared-grid diagnostics.

Read archived arrays without importing experiment code or loading its checkpoint.
This checks saved outputs and accounting; it does not certify a continuum error
bound or the correctness of the learned dynamical equations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def metrics(a, b, intervals):
    a, b = np.asarray(a), np.asarray(b)
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    difference = np.linalg.norm(a-b, axis=1)
    norm = np.linalg.norm(b, axis=1)
    assert np.all(norm > 0), 'explicit vanishing-norm handling required'
    return dict(relative_current=difference/norm,
                relative_initial=difference/norm[0], absolute_l2=difference/intervals)


def restriction(a, source, target):
    assert source % target == 0
    stride = source//target
    return a[:, stride-1::stride, stride-1::stride]


def audit(path):
    native = json.loads(path.read_text())
    assert native['complete']
    cfg = native['config']
    common = min(cfg['evaluation_intervals'])
    source_paths = [path, path.parent/'physical_reference.npz']
    physical = np.load(source_paths[-1])
    fine = int(physical['intervals'])
    tolerance = cfg['gradient_tolerance']
    rows, maximum_metric_disagreement = [], 0.
    for n in cfg['evaluation_intervals']:
        source_paths.append(path.parent/f'fields_N{n}.npz')
        saved = np.load(source_paths[-1])
        for row in native['rows']:
            if row['intervals'] != n:
                continue
            case, method = row['case'], row['method']
            assert len(row['repetitions']) == cfg['repetitions']
            prefix = f'n{n}_case{case}'
            discrete = saved[f'{prefix}_discrete_reference']
            per_grid_physical = restriction(physical['fields'][case], fine, n)
            common_physical = restriction(physical['fields'][case], fine, common)
            np.testing.assert_array_equal(per_grid_physical, saved[f'{prefix}_physical_reference'])
            errors, seconds, initial_nonstationary, step_nonstationary = [], [], 0, 0
            for rep in row['repetitions']:
                index = rep['repetition']
                fields = saved[f'{prefix}_{method}_rep{index}']
                assert np.all(np.isfinite(fields))
                for reference, key in [(discrete, 'vs_same_grid'), (per_grid_physical, 'vs_physical_reference')]:
                    observed = metrics(fields, reference, n)
                    for metric, values in observed.items():
                        reported = rep[key][metric]
                        np.testing.assert_allclose(values, reported, rtol=2e-12, atol=2e-14)
                        maximum_metric_disagreement = max(maximum_metric_disagreement,
                                                         float(np.max(np.abs(values-reported))))
                shared = metrics(restriction(fields, n, common), common_physical, common)
                errors.append({key: float(np.max(values)) for key, values in shared.items()})
                phases = rep['phases']
                assert all(np.isfinite(value) and value >= 0 for value in phases.values())
                assert sum(value for key, value in phases.items() if key != 'query_seconds') <= phases['query_seconds']
                seconds.append(phases['query_seconds'])
                if method.startswith('rom'):
                    fits = np.asarray(rep['solver']['initial_fits'])
                    selected = fits[np.argmin(fits[:, 3])]
                    initial_nonstationary += int(selected[4] > tolerance or selected[2] != 1)
                    steps = np.asarray(rep['solver']['steps'])
                    step_nonstationary += int(np.sum((steps[:, 4] > tolerance) | (steps[:, 2] != 1)))
            rows.append(dict(intervals=n, case=case, method=method,
                             repetitions=len(seconds), query_median_seconds=statistics.median(seconds),
                             common_grid_time_max_errors={key: max(e[key] for e in errors) for key in errors[0]},
                             initial_nonstationary_repetitions=initial_nonstationary,
                             nonstationary_steps_all_repetitions=step_nonstationary))
    return dict(scope='Independent saved-field and query-accounting audit; development only.',
                source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                job_id=native['metadata']['job_id'], source_commit=native['source_manifest']['source_commit'],
                common_observation_intervals=common,
                saved_field_metric_check_passed=True,
                maximum_native_metric_disagreement=maximum_metric_disagreement,
                physical_reference_self_difference=max(native['verification']['reference_spectral_refinement_current_errors']),
                reference_limit='Spectral refinement difference is empirical convergence evidence, not a certified error bound.',
                rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('job_id', 'saved_field_metric_check_passed', 'maximum_native_metric_disagreement')}))
