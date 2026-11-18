"""Generate pilot audit, fixed-case decisions and measured plots from retained arrays."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path, nargs='+')
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    outs, dest = args.output, args.destination
    dest.mkdir(parents=True, exist_ok=True)
    out = outs[0]
    summary = dict(scope='same-grid same-step discrete target; exposed development cases', cases={}, source_provenance=[])
    source_for_case, banks_by_source, gram_errors = {}, {}, {}
    env = json.loads((out/'environment.json').read_text())
    cfg = env['config']
    ms = cfg['solve_rows']
    n = cfg['mesh_intervals']
    for source in outs:
        this_env = json.loads((source/'environment.json').read_text())
        for key in ['mesh_intervals','bank_width','solve_rows','observer_rows','dt','horizon','reference_arithmetic',
                    'solver_gtols','solver_max_iterations','reference_tolerances','reference_max_disagreement',
                    'tangent_relative_eigenvalue_threshold','timing_repetitions']:
            assert this_env['config'].get(key) == cfg.get(key), key
        assert this_env['checkpoint_sha256'] == env['checkpoint_sha256'] and this_env['rotation_sha256'] == env['rotation_sha256']
        this_summary = json.loads((source/'summary.json').read_text())
        summary['source_provenance'].append(dict(output=str(source.resolve()), job_id=this_env['job_id'],
            commit=this_env['commit'], summary_sha256=hashlib.sha256((source/'summary.json').read_bytes()).hexdigest(),
            environment_sha256=hashlib.sha256((source/'environment.json').read_bytes()).hexdigest(),
            case_ids=this_env['config']['case_ids'], checkpoint_sha256=this_env['checkpoint_sha256'],
            rotation_sha256=this_env['rotation_sha256']))
        assert set(this_summary['cases']) == set(map(str, this_env['config']['case_ids']))
        for case_id, case in this_summary['cases'].items():
            assert case_id not in summary['cases'], 'Do not pool duplicate pilot and repaired cases.'
            summary['cases'][case_id] = case
            source_for_case[case_id] = source
        banks = sorted(source.glob('bank-columns-*.npz'))
        G = np.concatenate([np.load(p)['bank'] for p in banks], axis=1)
        banks_by_source[source] = G
        operators = np.load(source/'operators.npz')
        gram_errors[str(source)] = float(np.linalg.norm((G.T @ G)/n**2-operators['gram'])/np.linalg.norm(operators['gram']))
    gram_error = max(gram_errors.values())
    summary['source_for_case'] = {k:str(v.resolve()) for k,v in source_for_case.items()}
    summary['frozen_scientific_configuration'] = {k:cfg[k] for k in ['mesh_intervals','bank_width','solve_rows','observer_rows','dt','horizon',
        'solver_gtols','solver_max_iterations','reference_tolerances','reference_max_disagreement','tangent_relative_eigenvalue_threshold']}
    (dest/'combined-source.json').write_text(json.dumps(summary, indent=2)+'\n')
    audit = dict(scope=summary['scope'], source_outputs=list(map(str, outs)), gram_relative_error=gram_error,
                 per_source_gram_errors=gram_errors, cases={}, transitions=[])
    times = np.arange(round(cfg['horizon']/cfg['dt'])+1)*cfg['dt']
    table = []
    for case_id, case in summary['cases'].items():
        folder = source_for_case[case_id]/f'case-{int(case_id):02d}'
        G = banks_by_source[source_for_case[case_id]]
        ref0 = np.load(folder/'reference-0.npz')['fields']
        ref = np.load(folder/'reference-1.npz')['fields']
        norm0 = np.linalg.norm(ref0[0])
        refdiff = np.linalg.norm(ref-ref0, axis=1).max()/norm0
        proj = np.load(folder/'projection.npz')
        projected = proj['coeff'] @ G.T
        projection_error = np.linalg.norm(projected-ref, axis=1)/norm0
        evidence = dict(reference_disagreement=float(refdiff), max_metric_disagreement=0.,
                        projection_metric_disagreement=float(np.max(np.abs(projection_error-proj['error']))), arms={})
        for m in ms:
            arm = case['arms'][str(m)]
            z = np.load(folder/f'm{m}-tighter-rep0.npz')
            field = z['coeff'] @ G.T
            error = np.linalg.norm(field-ref, axis=1)/norm0
            evolution_distance = np.linalg.norm(field-projected, axis=1)/norm0
            metric_delta = float(np.max(np.abs(error-z['metrics'][:, 0])))
            gap_delta = float(np.max(np.abs(evolution_distance-z['evolution_distance'])))
            identity = error**2-projection_error**2-evolution_distance**2
            tight_coeff = np.load(folder/f'm{m}-tight-rep0.npz')['coeff']
            tolerance_delta = np.linalg.norm((tight_coeff-z['coeff']) @ G.T, axis=1).max()/norm0
            rr = np.load(folder/f'm{m}-residuals.npz')
            shell = np.linalg.norm(rr['scaled'][:, max(ms):], axis=1)
            per = dict(error=error.tolist(), evolution_distance=evolution_distance.tolist(),
                       projection_error=projection_error.tolist(), independent_shell=shell.tolist(),
                       error_max=float(error.max()), evolution_distance_max=float(evolution_distance.max()),
                       initial_error=float(error[0]), evolved_error_max=float(error[1:].max()),
                       maximum_error_time=float(times[int(np.argmax(error))]),
                       metric_disagreement=metric_delta, evolution_metric_disagreement=gap_delta,
                       pythagorean_max_absolute=float(np.max(np.abs(identity))),
                       tight_tighter_difference=float(tolerance_delta), independent_scaled_shell_max=float(shell.max()),
                       numerically_eligible=arm['numerically_eligible'])
            evidence['arms'][str(m)] = per
            evidence['max_metric_disagreement'] = max(evidence['max_metric_disagreement'], metric_delta, gap_delta)
            table.append(dict(case=int(case_id), M=m, max_discrete_error=per['error_max'], evolution_distance_max=per['evolution_distance_max'],
                              initial_error=per['initial_error'], evolved_error_max=per['evolved_error_max'], maximum_error_time=per['maximum_error_time'],
                              projection_error_max=float(projection_error.max()), shell_max=float(shell.max()),
                              tight_tighter=float(tolerance_delta), iterations=arm['levels']['tighter']['total_iterations'],
                              rejected=arm['levels']['tighter']['rejected_steps'], eligible=arm['numerically_eligible'],
                              physical_sigma_min_final=arm['reached_spectra'][-1]['spectra'][str(m)]['physical_min']))
        audit['cases'][case_id] = evidence
    for small, large in zip(ms[:-1], ms[1:]):
        pair = []
        for case_id, case in audit['cases'].items():
            a, b = case['arms'][str(small)], case['arms'][str(large)]
            uncertainty = a['tight_tighter_difference']+b['tight_tighter_difference']
            pair.append(dict(case=int(case_id), eligible=a['numerically_eligible'] and b['numerically_eligible'],
                error_ratio=b['error_max']/a['error_max'], shell_ratio=b['independent_scaled_shell_max']/a['independent_scaled_shell_max'],
                resolved_error_improvement=a['error_max']-b['error_max'] > uncertainty,
                error_decrease=a['error_max']-b['error_max'], solver_uncertainty=uncertainty))
        all_eligible=all(v['eligible'] for v in pair)
        useful=all_eligible and all(v['error_ratio'] <= 1.02 for v in pair) and any(v['resolved_error_improvement'] and v['shell_ratio'] < 1 for v in pair)
        audit['transitions'].append(dict(smaller=small, larger=large, all_cases_eligible=all_eligible,
                                        predeclared_useful_region=useful, cases=pair))
    # Fixed-state row enrichment must obey the algebraic singular-value monotonicity.
    monotonic=[]
    for case in summary['cases'].values():
        for rec in case['fixed_state_spectra']:
            values = [rec['spectra'][str(m)]['physical_min'] for m in ms]
            monotonic.append(float(min(np.diff(values))))
    audit['fixed_state_minimum_sigma_increment']=min(monotonic)
    audit['saved_array_audit_passed']=bool(gram_error < 1e-10 and all(c['max_metric_disagreement'] < 1e-10 and c['projection_metric_disagreement'] < 1e-10 for c in audit['cases'].values()) and min(monotonic) >= -1e-10)
    audit['environmental_acceptance']='must be read from separate collector audit; numerical file alone is insufficient'
    audit['aggregate'] = {}
    for m in ms:
        rows = [r for r in table if r['M'] == m]
        audit['aggregate'][str(m)] = dict(case_count=len(rows), qualified_count=sum(r['eligible'] for r in rows),
            maximum_error_median=float(np.median([r['max_discrete_error'] for r in rows])),
            maximum_error_worst=float(max(r['max_discrete_error'] for r in rows)),
            initial_error_median=float(np.median([r['initial_error'] for r in rows])),
            evolved_error_worst=float(max(r['evolved_error_max'] for r in rows)),
            evolution_distance_worst=float(max(r['evolution_distance_max'] for r in rows)),
            initial_time_is_maximum_count=sum(r['maximum_error_time'] == 0 for r in rows),
            iterations_median=float(np.median([r['iterations'] for r in rows])),
            rejected_proposals_total=int(sum(r['rejected'] for r in rows)))
    (dest/'analysis.json').write_text(json.dumps(audit, indent=2)+'\n')
    (dest/'table.json').write_text(json.dumps(table, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    count = len(audit['cases'])
    curve_rows = (count+1)//2
    fig = plt.figure(figsize=(11, 3.7*(curve_rows+1)))
    grid = fig.add_gridspec(curve_rows+1, 2)
    curve_axes = [fig.add_subplot(grid[i//2, i%2]) for i in range(count)]
    sigma_axis, shell_axis = fig.add_subplot(grid[-1, 0]), fig.add_subplot(grid[-1, 1])
    colors = plt.cm.viridis(np.linspace(.08, .88, len(ms)))
    for case_index, (case_id, case) in enumerate(audit['cases'].items()):
        ax = curve_axes[case_index]
        for color, m in zip(colors, ms):
            arm = case['arms'][str(m)]
            suffix = '' if arm['numerically_eligible'] else ' (unqualified)'
            ax.plot(times, 100*np.asarray(arm['error']), color=color, label=f'M={m}{suffix}')
        firstarm=case['arms'][str(ms[0])]
        ax.plot(times, 100*np.asarray(firstarm['projection_error']), 'k--', label='best span projection')
        ax.set(title=f'Development case {case_id}', xlabel='Time', ylabel='Error / initial-field norm (%)')
        ax.legend(fontsize=8)
    for case_index, (case_id, case) in enumerate(summary['cases'].items()):
        rec=case['fixed_state_spectra'][-1]
        label = f'case {case_id}' + ('' if all(v['numerically_eligible'] for v in case['arms'].values()) else ' (unqualified)')
        sigma_axis.semilogy(ms, [rec['spectra'][str(m)]['physical_min'] for m in ms], marker='o', label=label+', same state')
        shell_axis.semilogy(ms, [audit['cases'][case_id]['arms'][str(m)]['independent_scaled_shell_max'] for m in ms], marker='o', label=label)
    sigma_axis.set(xlabel='Smooth test rows M', ylabel='Smallest physical tangent singular value', title='Fixed projected-reference state at final time')
    shell_axis.set(xlabel='Smooth test rows M', ylabel='Maximum independent shell residual', title='Next 1024 rows; maximum over all evolved steps')
    sigma_axis.legend(fontsize=8)
    shell_axis.legend(fontsize=8)
    fig.suptitle('MEASURED development study — same-grid/same-step reference; qualification shown per curve', fontsize=12)
    fig.tight_layout()
    fig.savefig(dest/'measured-visibility.png', dpi=180)
    fig.savefig(dest/'measured-visibility.svg')
    print(json.dumps({'saved_array_audit_passed':audit['saved_array_audit_passed'], 'transitions':audit['transitions']}))


if __name__ == '__main__':
    main()
