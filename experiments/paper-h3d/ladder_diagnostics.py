"""Describe the fixed correction ladder after independent numerical audits."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('attempt'); args = parser.parse_args()
    assert args.attempt.isalnum()
    run = Path(__file__).resolve().parent/'runs'/args.attempt
    collected = json.loads((run/'COLLECTED.json').read_text())
    assert collected['checksums_verified']
    audits = ['audit-local.json', 'audit-states-primary.json', 'audit-panel-primary.json',
              'audit-field-seedB.json', 'audit-states-seedB.json', 'audit-panel-seedB.json']
    assert all(json.loads((run/name).read_text())['passed'] for name in audits)
    panels = []; sources = {}
    for cohort, out in [('primary', run/'archive/out'), ('seedB', run/'archive/out/seedB')]:
        record = json.loads((out/'result.json').read_text())
        summaries = json.loads((out/'summary.json').read_text())['rows']
        sources[cohort] = hashlib.sha256((out/'result.json').read_bytes()).hexdigest()
        k, = record['config']['latent_dimensions']; qs = record['config']['q_ladder']
        for n in record['config']['evaluation_intervals']:
            points = []; errors = []
            for q in qs:
                name = f'nmrom_K{k}_q{q}_dense'
                summary = next(r for r in summaries if r['intervals'] == n and r['method'] == name)
                cases = []
                for case in range(summary['cases']):
                    repeated = [r for r in record['invocations'] if r['intervals'] == n and r['case'] == case and r['method'] == name]
                    assert len(repeated) == record['config']['repetitions']
                    assert len({r['same_grid']['current_evolved'] for r in repeated}) == 1
                    cases.append(repeated[0]['same_grid']['current_evolved'])
                errors.append(cases)
                points.append(dict(q=q, method=name, median_rel_l2=summary['same_grid_current_evolved_median'],
                    worst_rel_l2=summary['same_grid_current_evolved_worst'], device_ms=summary['device_ms_median'],
                    nonstationary_cases=summary['cases_with_nonstationary_solves']))
            values = np.asarray(errors)
            transitions = []
            for i in range(len(qs)-1):
                delta = values[i+1]-values[i]
                transitions.append(dict(q_from=qs[i], q_to=qs[i+1],
                    regressing_cases=np.flatnonzero(delta > 1e-12).tolist(),
                    improving_cases=int(np.count_nonzero(delta < -1e-12)),
                    largest_absolute_error_increase=float(np.max(delta)),
                    largest_error_ratio=float(np.max(values[i+1]/values[i]))))
            panels.append(dict(cohort=cohort, intervals=n, points=points, transitions=transitions,
                worst_error_monotone=bool(np.all(np.diff([r['worst_rel_l2'] for r in points]) <= 1e-12)),
                median_error_monotone=bool(np.all(np.diff([r['median_rel_l2'] for r in points]) <= 1e-12)),
                every_case_monotone=all(not r['regressing_cases'] for r in transitions),
                worst_error_improvement_factor=points[0]['worst_rel_l2']/points[-1]['worst_rel_l2'],
                median_error_improvement_factor=points[0]['median_rel_l2']/points[-1]['median_rel_l2'],
                device_cost_factor=points[-1]['device_ms']/points[0]['device_ms']))
    result = dict(schema='heat3d-fixed-ladder-diagnostics-v1', attempt=args.attempt,
        status='accepted' if collected.get('actual_git_blob_bytes_verified') and collected['removed'] else 'numerically_audited_pending_complete_git_retention',
        final_based_selection=False, source_result_sha256=sources, panels=panels,
        glossary=dict(q='Prospectively fixed number of bank correction directions.',
            median_rel_l2='Median across query cases of maximum evolved-time current-relative L2 error.',
            worst_rel_l2='Maximum across cases and evolved times of current-relative L2 error.',
            device_ms='Median synchronized same-allocation query time, in milliseconds.',
            regressing_cases='Zero-based case IDs whose evolved-time error increases by more than an absolute 1e-12 error fraction at this fixed q transition.',
            every_case_monotone='Every query case is nonincreasing at every declared adjacent q transition; this is stronger than a monotone median or worst.',
            device_cost_factor='Highest-q median query time divided by q0 median query time; a within-job comparison.',
            improvement_factor='q0 aggregate error divided by highest-q aggregate error; larger values mean more aggregate error reduction.'))
    (run/'LADDER-DIAGNOSTICS.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
