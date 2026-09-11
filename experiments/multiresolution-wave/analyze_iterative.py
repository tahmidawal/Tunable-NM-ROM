"""Generate same-grid results after independent CPU field and accounting checks."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from audit_iterative_extra import extra_audit
from audit_dynamics import recompute, compare_metrics, parameter_rows, initial_fields, NAMES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main(record):
    cluster = record/'cluster'; native = cluster/'out/pilot'; analysis = record/'analysis'
    analysis.mkdir(exist_ok=True)
    data = json.loads((native/'result.json').read_text()); cfg = data['config']
    meta = data['provenance']; cleanup = json.loads((record/'cleanup.json').read_text())
    submission = json.loads((record/'submission.json').read_text())
    assert data['complete'] and not data['final_test_opened']
    assert meta['jax_backend']=='gpu' and meta['x64'] and meta['matmul_precision']=='highest'
    assert cleanup['remote_deleted_and_absence_checked'] and cleanup['all_three_manifests_verified']
    assert (cluster/'EXIT_CODE').read_text().strip()=='0'
    logs = '\n'.join(p.read_text() for p in (cluster/'logs').glob('*'))
    assert 'jax_backend=gpu' in logs
    for bad in ('captured constant', 'Captured constant', 'out of memory', 'RESOURCE_EXHAUSTED',
                'Traceback', 'No space left', 'cuInit'):
        assert bad not in logs, bad
    for path, expected in submission['source_hashes'].items():
        assert sha(cluster/path)==expected
        rel = 'experiments/'+str(Path(path).relative_to('code'))
        blob = subprocess.check_output(['git', 'show', submission['source_commit']+':'+rel], cwd=Path(__file__).resolve().parents[2])
        assert hashlib.sha256(blob).hexdigest()==expected
    for path, expected in data['input_sha256'].items(): assert sha(cluster/'in'/path)==expected
    for path, expected in data['output_sha256'].items(): assert sha(native/path)==expected
    assert len(data['invocations'])==cfg['expected_timed_invocations']
    assert len(data['accuracy_controls'])==cfg['expected_accuracy_only_queries']
    extra = extra_audit(record, data)
    pars = parameter_rows(cfg['validation_seed'], max(cfg['validation_indices'])+1)
    groups = defaultdict(list)
    for row in data['invocations']+data['accuracy_controls']:
        groups[(row['boundary'], row['intervals'], row['case'], row['method'], row['measurement_role'])].append(row)
    audited = []
    for (bc, n, ci, method, role), rows in groups.items():
        with np.load(native/f'reference_{bc}_{n}_{ci}.npz') as ref:
            ut, vt, u0, v0 = ref['u'], ref['v'], ref['u0'], ref['v0']
            np.testing.assert_array_equal(ref['parameters'], pars[ci])
        iu, iv = initial_fields(pars[ci], n, bc)
        np.testing.assert_allclose(iu, u0, atol=1e-13, rtol=1e-12)
        np.testing.assert_allclose(iv, v0, atol=1e-13, rtol=1e-12)
        with np.load(native/rows[0]['field_artifact']) as f:
            u, v = f['u'], f['v']
            expected_hash = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in (('u', u), ('v', v))}
            if method.startswith('frozen_mlp'):
                with np.load(native/f'mesh_{bc}_{n}.npz') as mesh:
                    np.testing.assert_allclose(f['coefficients']@mesh['g'].T, u.reshape(49,-1), atol=1e-11, rtol=1e-10)
                    np.testing.assert_allclose(f['velocity_coefficients']@mesh['g'].T, v.reshape(49,-1), atol=1e-10, rtol=1e-9)
        independent = recompute(u, v, ut, vt, n, bc, pars[ci,5])
        for row in rows:
            assert row['output_sha256']==expected_hash
            assert row['output_bytes']==u.nbytes+v.nbytes
            assert compare_metrics(independent, row['same_grid_discrepancy'])<1e-10
            for name in NAMES:
                actual = independent[name]
                current = actual['reference_norm']; scale = actual['initial_scale']
                zero = current<=1e-14*scale
                saved = row['same_grid_discrepancy'][name]
                np.testing.assert_array_equal(zero, saved['reference_zero'])
                np.testing.assert_array_equal(current<=cfg['vanishing_fraction']*scale, saved['reference_vanishing'])
                expected = np.divide(actual['absolute'], current, out=np.full_like(current, np.nan), where=~zero)
                supplied = np.array([np.nan if x is None else x for x in saved['current_relative']])
                np.testing.assert_allclose(supplied, expected, atol=1e-10, rtol=1e-10, equal_nan=True)
            seconds = row['seconds']
            assert abs(sum(seconds[k] for k in ('initialization_and_parameter_projection','evolution','dense_device_output'))-seconds['complete_device_query'])<1e-12
            assert row['comparison_eligible']==(role=='timed_comparison')
        audited.append(dict(boundary=bc, intervals=n, case=ci, method=method, role=role,
                            repetitions=len(rows), full_field_hashes=expected_hash))
    summary = dict(config=cfg, provenance=meta, result_sha256=sha(native/'result.json'),
                   audited_invocations=sum(r['repetitions'] for r in audited), field_checks=audited,
                   gpu_assignment_proof=dict(path='gpu-assignment.txt',sha256=sha(record/'gpu-assignment.txt'),contents=(record/'gpu-assignment.txt').read_text()),
                   extra_independent_audit=extra, references=data['references'], time_refinement=data['time_refinement'], groups=[],
                   integrity_audit_passed=True,
                   all_reference_refinement_gates_passed=all(r['refinement_passed'] for r in data['references']),
                   reference_gate_scope='Both absorbing same-grid reference refinements are archived and independently compared. Reflective references are checked with independent SciPy sine transforms. No continuum bound.')
    panels = defaultdict(list)
    for key, rows in groups.items():
        bc,n,ci,method,role = key
        if role!='timed_comparison': continue
        med = float(np.median([r['seconds']['complete_device_query'] for r in rows]))
        errors = rows[0]['same_grid_discrepancy']
        panels[bc,n,method].append(dict(case=ci, query_median=med,
            raw_query_seconds=[r['seconds']['complete_device_query'] for r in rows],
            component_medians={k:float(np.median([r['seconds'][k] for r in rows])) for k in rows[0]['seconds']},
            timing_outliers_above_twice_case_median=sum(r['seconds']['complete_device_query']>2*med for r in rows),
            errors=errors, completed=all(r['completed'] for r in rows),
            fit_stationary=all(r.get('fit_stationary',True) for r in rows),
            cg_all_converged=all(r.get('cg_all_converged',True) for r in rows),
            maximum_cg_true_relative=max((max(r.get('cg_true_relative',[0])) for r in rows)),
            cg_total_iterations=max((sum(r.get('cg_iterations',[])) for r in rows)),
            cg_max_step_iterations=max((max(r.get('cg_iterations',[0])) for r in rows)),
            device_plus_output_transfer_median=float(np.median([r['device_plus_output_transfer_seconds'] for r in rows]))))
    for (bc,n,method), cases in panels.items():
        fm = 'cg_1e-06'
        ratios = [next(f['query_median'] for f in panels[bc,n,fm] if f['case']==c['case'])/c['query_median'] for c in cases]
        refinements = [r for r in data['time_refinement'] if (r['boundary'],r['intervals'],r['method'])==(bc,n,method)]
        references = [r for r in data['references'] if (r['boundary'],r['intervals'])==(bc,n)]
        qualification = {}
        for target in cfg['accuracy_targets']:
            passed = []
            for case in cases:
                ref_ok = next(r['refinement_passed'] for r in references if r['case']==case['case'])
                time_ok = all(r['passed'] for r in refinements if r['case']==case['case'])
                initial_ok = all(case['errors'][name]['max_initial_normalized']<=target for name in NAMES)
                current_ok = all(case['errors'][name]['max_current_relative']<=target for name in NAMES)
                passed.append(dict(case=case['case'],
                    numerical_gates_passed=bool(ref_ok and time_ok and case['completed'] and case['fit_stationary'] and case['cg_all_converged']),
                    initial_accuracy_passed=initial_ok, current_accuracy_passed=current_ok))
            qualification[str(target)] = dict(cases=passed,
                all_initial_qualified=all(c['numerical_gates_passed'] and c['initial_accuracy_passed'] for c in passed),
                all_current_qualified=all(c['numerical_gates_passed'] and c['current_accuracy_passed'] for c in passed))
        error_summary = {}
        for name in NAMES:
            initial = [c['errors'][name]['max_initial_normalized'] for c in cases]
            current = [c['errors'][name]['max_current_relative'] for c in cases]
            error_summary[name] = dict(initial_max_median=float(np.median(initial)), initial_max_worst=max(initial),
                current_max_median=float(np.median(current)), current_max_worst=max(current),
                outliers_by_initial_target={str(target):sum(e>target for e in initial) for target in cfg['accuracy_targets']},
                outliers_by_current_target={str(target):sum(e>target for e in current) for target in cfg['accuracy_targets']})
        component_gate_by_target={str(target): {name: dict(all_initial_passed=all(c['errors'][name]['max_initial_normalized']<=target for c in cases),all_current_passed=all(c['errors'][name]['max_current_relative']<=target for c in cases)) for name in NAMES} for target in cfg['accuracy_targets']}
        summary['groups'].append(dict(boundary=bc, intervals=n, method=method, cases=cases,
            query_median=float(np.median([c['query_median'] for c in cases])),
            device_plus_output_transfer_median=float(np.median([c['device_plus_output_transfer_median'] for c in cases])),
            cg_failed_cases=sum(not c['cg_all_converged'] for c in cases),
            cg_max_step_iterations=max(c['cg_max_step_iterations'] for c in cases),
            cg_total_iterations=sum(c['cg_total_iterations'] for c in cases),
            maximum_cg_true_relative=max(c['maximum_cg_true_relative'] for c in cases),
            paired_fom_over_method_median=float(np.median(ratios)), paired_ratios=ratios,
            errors=error_summary, failed_cases=sum(not c['completed'] for c in cases),
            nonstationary_cases=sum(not c['fit_stationary'] for c in cases),
            time_refinement_failed_cases=sum(not r['passed'] for r in refinements),
            reference_refinement_failed_cases=sum(not r['refinement_passed'] for r in references),
            component_accuracy_by_target=component_gate_by_target,
            accuracy_qualification_by_target=qualification,
            timing_outliers=sum(c['timing_outliers_above_twice_case_median'] for c in cases)))
    write(analysis/'summary.json', summary)
    lines = ['# Fresh-wave iterative FOM multiresolution comparison', '',
        'These are new bounded development measurements using verified post-reset wave operators and frozen heads. All values below are generated from the retained timed outputs; no final cohort or continuum-accuracy claim is included.', '',
        '| Boundary | Intervals | Method | Query median (ms) | CG 1e-6/query ratio | Worst initial-scaled displacement / velocity / energy | Worst current-relative displacement / velocity / energy | Failed / nonstationary / refinement |',
        '|---|---:|---|---:|---:|---|---|---|']
    for g in summary['groups']:
        initial=' / '.join(f'{g["errors"][k]["initial_max_worst"]:.6g}' for k in NAMES)
        current=' / '.join(f'{g["errors"][k]["current_max_worst"]:.6g}' for k in NAMES)
        lines.append(f'| {g["boundary"]} | {g["intervals"]} | {g["method"]} | {1000*g["query_median"]:.6g} | {g["paired_fom_over_method_median"]:.6g} | {initial} | {current} | {g["failed_cases"]} / {g["nonstationary_cases"]} / {g["time_refinement_failed_cases"]} |')
    lines += ['', 'Ratios compare complete device queries to the new same-grid implicit-midpoint CG FOM at tolerance 1e-6. This is a matched-time-step iterative control using fresh wave mathematics, not a literal historical wave algorithm. Cold initialization, all projection, speed preparation, evolution and full GPU fields are charged. Host transfers are outside this timer. Inspect the adjacent errors and failure counts before interpreting a ratio as useful acceleration.', '',
        f'All recorded same-grid reference refinement gates passed: **{summary["all_reference_refinement_gates_passed"]}**. Integrity checks and completed execution do not establish numerical-gate or target-accuracy acceptance. The JSON explicitly qualifies each target using reference refinement, ROM refinement, completed evolution, fit stationarity and all three error components.', '',
        'The summary JSON retains component timings, every repetition, per-case error curves, reference temporal refinement and separate nonlinear time refinement. A timing outlier is a repetition above twice its case median. Error-outlier counts are retained for each predeclared target. Reference refinement is empirical and does not establish a continuum bound.', '',
        '## Glossary', '',
        '- **Intervals / method:** spatial cells per axis / the named frozen reduced model or full solver.',
        '- **CG 1e-6/query ratio:** the median of paired per-case FOM time divided by method time; above one means the named method is faster under this timing contract.',
        '- **Initial-scaled / current-relative:** error divided by the fixed reference initial scale / the reference norm at the observation time.',
        '- **Energy:** the joint displacement-gradient and physical-velocity norm.',
        '- **Failed / nonstationary / refinement:** incomplete trajectories / initial fits missing their fixed stopping criteria / trajectories failing the smaller-step comparison.',
        '- **Median / worst / outlier:** the middle value / largest value / a timing or error exceeding the declared threshold.',
        '- **Device query / cold initialization:** GPU input through full GPU output / fitting the initial reduced coordinates from supplied fields.', '']
    (analysis/'results.md').write_text('\n'.join(lines))
    print(json.dumps(dict(audited_invocations=summary['audited_invocations'], groups=len(summary['groups']), result_sha256=summary['result_sha256'])))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('record',type=Path)
    main(ap.parse_args().record)
