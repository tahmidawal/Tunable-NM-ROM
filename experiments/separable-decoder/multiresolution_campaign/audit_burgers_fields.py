"""Recompute development Burgers physical errors from each saved timed output."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def audit(path):
    data = json.loads(path.read_text())
    assert data['complete'] and data['backend'] == 'gpu' and data['x64']
    assert data['matmul_precision'] == 'highest'
    cfg = data['config']
    references = [np.load(path.parent/f"ref_L{cfg['reference_mesh']}_dt{cfg['reference_dt']}_case{case}.npz")['fields']
                  for case in range(cfg['cases'])]
    groups = defaultdict(list)
    worst_disagreement = 0.
    for row in data['invocations']:
        groups[row['name']].append(row)
        field = np.load(path.parent/row['observation_artifact'])['fields']
        reference = references[row['case']]
        difference = np.linalg.norm((field-reference).reshape(len(field), -1), axis=1)
        norms = np.linalg.norm(reference.reshape(len(field), -1), axis=1)
        assert np.all(norms > 0)
        if row['finite']:
            actual = {'fixed_initial_per_time': difference/norms[0],
                      'current_relative_per_time': difference/norms}
            for key, values in actual.items():
                reported = row['physical_error'][key]
                np.testing.assert_allclose(values, reported, rtol=3e-12, atol=2e-14)
                worst_disagreement = max(worst_disagreement, float(np.max(np.abs(values-reported))))
            np.testing.assert_allclose(max(actual['fixed_initial_per_time']), row['physical_error']['fixed_initial_max'])
            np.testing.assert_allclose(max(actual['current_relative_per_time']), row['physical_error']['current_relative_max'])
    results = []
    for name, rows in groups.items():
        keys = {(row['case'], row['rep']) for row in rows}
        expected = {(case, rep) for case in range(cfg['cases']) for rep in range(cfg['reps'])}
        assert len(rows) == len(keys) and keys == expected
        cases = []
        for case in range(cfg['cases']):
            selected = [row for row in rows if row['case'] == case]
            method = selected[0]['method']
            finite = all(row['finite'] for row in selected)
            record = dict(case=case, query_median_seconds=statistics.median(row['seconds'] for row in selected),
                          finite=finite,
                          initial_normalized_max=max(row['physical_error']['fixed_initial_max'] for row in selected) if finite else None,
                          current_normalized_max=max(row['physical_error']['current_relative_max'] for row in selected) if finite else None)
            if method == 'rom':
                record['initial_fit_budget_or_failure_repetitions'] = sum(row['ic_reason'] in (0, 3) for row in selected)
                record['rollout_budget_or_failure_steps_all_repetitions'] = sum(sum(reason in (0, 3) for reason in row['stop_reasons']) for row in selected)
                record['rollout_stall_stops_all_repetitions'] = sum(row['stop_reasons'].count(2) for row in selected)
            else:
                record['fom_unmet_tolerance_repetitions'] = sum(not row['nonlinear_tolerance_satisfied'] for row in selected)
            cases.append(record)
        results.append(dict(configuration=name, cases=cases))
    return dict(scope='Saved common-observation fields and repetition integrity only; no continuum qualification.',
                source_json=str(path), source_json_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                job_id=data['job_id'], source_commit=data['commit'],
                verified_invocations=len(data['invocations']), maximum_metric_disagreement=worst_disagreement,
                reference_refinement_estimates=data['reference_uncertainty'],
                reference_limit='Recorded refinement estimates do not alone certify a continuum error bound.',
                configurations=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('job_id', 'verified_invocations', 'maximum_metric_disagreement')}))
