"""Read-only qualification of retained direct/global pilot results.

This audits actual returned diagnostics; it never changes solver output or
substitutes projection errors for physical trajectory errors.
"""
from pathlib import Path
import argparse, hashlib, json, math

H = Path(__file__).resolve().parent


def inspect(name):
    path = H / 'results' / name / 'output/result.json'
    retention = json.loads((H / 'manifests' / f'retention-{name}.json').read_text())
    assert retention['local_and_remote_verified'] and retention['terminal_state'] == 'COMPLETED'
    retained_hash = next(f['sha256'] for f in retention['files'] if f['file'] == 'output/result.json')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == retained_hash
    result = json.loads(path.read_text())
    assert result['status'] == 'complete_fixed_tree_diagnostic_pilot'
    station = result.get('stationarity_tolerance', 1e-8)
    update = result.get('update_tolerance', 1e-10)
    rows = []
    for case in result['cases']:
        for method, values in case['methods'].items():
            status = values['step_status']
            converged = [all(isinstance(x, (int, float)) and math.isfinite(x) for x in s)
                         and s[1] <= station
                         and (s[2] <= update or s[1] <= station / 100)
                         for s in status]
            expected_steps = round(.25 / result['dt'])
            trajectory_complete = (values['finite_trajectory'] and
                                   values['full_trajectory_returned'] and
                                   len(status) == expected_steps and
                                   len(values['physical_error']) == 6)
            physical = max(values['physical_error'][1:] + [values['actual4096_final_error']])
            accuracy_pass = (trajectory_complete and values['output_shape'] == [4096, 4096]
                             and physical + values['reference_allowance_max'] <= .01)
            change = values.get('previous_state_relative_change')
            rows.append(dict(case=case['case'], method=method,
                             steps=len(status), nonconverged_steps=sum(not x for x in converged),
                             full_solver_convergence=trajectory_complete and all(converged),
                             physical_accuracy_pass=accuracy_pass,
                             max_physical_error=physical,
                             reference_allowance=values['reference_allowance_max'],
                             initial_linear_residual=values['initial_linear_residual'],
                             max_field_change=max(change) if change is not None else None,
                             sensitivity_pass=max(change) <= .001 if change is not None else None,
                             common_tests=case['common_tests']))
    passed = all(v['full_solver_convergence'] and v['physical_accuracy_pass']
                 and v['sensitivity_pass'] is not False for v in rows)
    return dict(name=name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                previous_result_sha256=result.get('previous_result_sha256'),
                scope='Only the listed pilot trajectories and frozen representations',
                settings={k:result.get(k) for k in
                          ['L','dt','time_integrator','quadrature_intervals','global_control',
                           'global_control_sha256','stationarity_tolerance','update_tolerance',
                           'cg_relative_tolerance','cg_absolute_tolerance',
                           'initial_relative_tolerance','initial_absolute_tolerance']},
                passed=passed, rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    parser.add_argument('--record', required=True)
    args = parser.parse_args()
    output = H / 'manifests' / args.record
    assert output.parent == H / 'manifests' and output.suffix == '.json'
    assert not output.exists(), 'Use a fresh record name; preserve prior admissions and failures'
    records = [inspect(name) for name in args.names]
    output.write_text(json.dumps(dict(records=records, all_passed=all(r['passed'] for r in records)), indent=2)+'\n')
    for r in records:
        print(r['name'], 'PASS' if r['passed'] else 'FAIL',
              'worst physical', max(v['max_physical_error'] for v in r['rows']),
              'nonconverged', sum(v['nonconverged_steps'] for v in r['rows']),
              'max change', max((v['max_field_change'] or 0) for v in r['rows']))


if __name__ == '__main__':
    main()
