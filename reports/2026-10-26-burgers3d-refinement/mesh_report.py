"""Render the actual audited four-case mesh/operator dataset without population promotion."""
from __future__ import annotations

import collections
import json

import matplotlib.pyplot as plt
import numpy as np

CASES = [0, 5, 30, 43]


def render(doc, reference, table, save):
    assert doc['schema_version'] == 1 and doc['execution']['complete'] is True
    cohort, ref = doc['cohort'], doc['reference']
    assert cohort['name'] == 'val64' and cohort['seed'] == reference['cohorts']['val64']['seed']
    assert cohort['ordered_cases'] == CASES
    assert cohort['canonical_table_sha256'] == reference['cohorts']['val64']['table_sha256']
    assert (ref['dimension'], ref['kind'], ref['field']) == (3, 'ST', 'finest')
    assert ref['store_version'] == reference['store_version']
    assert ref['approved_status_sha256'] == reference['status_sha256']
    assert ref['output_times'] == reference['output_times']
    assert doc['execution']['job_id'] and doc['execution']['gpu'] and doc['execution']['source_commit']
    assert doc['execution']['raw_file_manifest']
    for qid, query in doc['queries'].items():
        frozen = doc['frozen_modes'][query['bank']]
        assert (query['r'], query['M'], query['mode_hash']) == (frozen['r'], frozen['M'], frozen['mode_hash'])
        assert sorted(int(c) for c in doc['case_results'][qid]) == sorted(query['cases'])
        assert set(query['cases']) <= set(CASES)
    rows = doc['timing_repetitions']
    invocations = {r['invocation_id']: r for r in rows}
    assert len(invocations) == len(rows)
    for row in rows:
        assert row['query'] in doc['queries'] and row['case'] in CASES
        sc = row['score']
        target = reference['cohorts']['val64']['cases'][str(row['case'])]
        assert sc['eligible'] is target['eligible']
        assert np.isclose(sc['epsilon'], target['eps_ref'], rtol=1e-10, atol=0)
        assert np.isfinite(sc['e_t']).all() and abs(max(sc['e_t'][1:])-sc['e']) <= 1e-12
        assert len(sc['e_t']) == len(ref['output_times'])
        for mask, field in [('timing_admitted', 'descriptor_seconds'), ('native_timing_admitted', 'given_native_seconds')]:
            assert isinstance(row[mask], bool)
            if row[mask]:
                assert np.isfinite(row[field]) and row[field] > 0
        assert row['input_prep_seconds'] >= 0 and np.isfinite(row['input_prep_seconds'])
        assert abs(row['descriptor_seconds']-row['given_native_seconds']-row['input_prep_seconds']) <= 1e-9
    summary, cost_rows, phase_rows, qualification_rows = {}, [], [], []
    def peak(values):
        return 'nonfinite / missing' if not values or any(v is None or not np.isfinite(v) for v in values) else f'{max(values):.7g}'
    phases = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in doc['phase_repetitions']:
        assert row['case'] in CASES and np.isfinite(row['seconds']) and row['seconds'] > 0
        assert row['split_fused_relative'] <= 1e-12
        phases[(row['query'], row['phase'])][row['case']].append(row['seconds'])
    for qid, source_costs in doc['cost_summaries'].items():
        query = doc['queries'][qid]
        assert query['bank'] == 'M2' and query['rule'] == 'gl32' and query['contract'] in ['mf', 'mesh']
        result = doc['case_results'][qid]
        own = [r for r in rows if r['query'] == qid and r['role'] == 'B']
        assert sorted({r['case'] for r in own}) == CASES
        for cid, record in result.items():
            row = invocations[record['invocation_id']]
            assert (row['query'], row['case'], row['role']) == (qid, int(cid), 'B')
            assert row['output_sha256'] == record['output_sha256']
            assert row['coefficient_sha256'] == record['coefficient_sha256']
            assert abs(row['score']['e']-record['score']['e']) <= 1e-12
        costs = {}
        for contract, field, mask in [('descriptor', 'descriptor_seconds', 'timing_admitted'),
                                       ('given_native', 'given_native_seconds', 'native_timing_admitted')]:
            percase = {cid: [r[field] for r in own if r['case'] == cid and r[mask]] for cid in CASES}
            complete = all(percase.values())
            median = float(np.median([np.median(v) for v in percase.values()])) if complete else None
            source = source_costs[contract]
            assert source['complete_case_coverage'] is complete
            if complete:
                assert np.isclose(source['median_seconds'], median, rtol=1e-12)
            costs[contract] = dict(complete=complete, median_seconds=median, case_repetitions=percase)
        work = {name: float(np.median([r['work'][name] for r in result.values()]))
                for name in ['nonlinear_iterations', 'jacobian_contractions']}
        e = [r['score']['e'] for r in result.values()]
        q1, q2 = query['Q1']['passed'], query['Q2']['passed']
        summary[qid] = dict(config=query, costs=costs, work=work, Q1=q1, Q2=q2,
                            median_error=float(np.median(e)), worst_error=max(e))
        cost_rows.append([qid, query['N'], query['contract'], str(query['input_shape']), str(query['output_shape']),
            'missing' if costs['descriptor']['median_seconds'] is None else f"{costs['descriptor']['median_seconds']*1e3:.7g}",
            'missing' if costs['given_native']['median_seconds'] is None else f"{costs['given_native']['median_seconds']*1e3:.7g}",
            f"{work['nonlinear_iterations']:.6g}/{work['jacobian_contractions']:.6g}", f'{q1}/{q2}',
            f'{100*np.median(e):.6g}/{100*max(e):.6g}'])
        q2_cases = query['Q2']['cases']
        qualification_rows.append([qid, f"{query['Q1']['nonstationary']}/{query['Q1']['steps']}",
            peak([r['common_distance_tight'] for r in result.values()]),
            peak([r['native_distance_tight'] for r in result.values()]) if query['output_nodes'] == 129 else 'fixed output',
            peak([r['common_tight_tighter_distance'] for r in result.values()]),
            peak([r['native_tight_tighter_distance'] for r in result.values()]) if query['output_nodes'] == 129 else 'fixed output',
            peak([r['distance'] for r in q2_cases]), peak([r['finer_distance'] for r in q2_cases]),
            peak([v for r in q2_cases for v in r['rho']])])
        parts = []
        for phase in ['initialization', 'evolve', 'decode']:
            percase = phases[(qid, phase)]
            parts.append('incomplete' if sorted(percase) != CASES else f'{np.median([np.median(v) for v in percase.values()])*1e3:.7g}')
        phase_rows.append([qid, *parts, f"{query['compile_and_warm_seconds']:.7g}"])
    assert len(summary) == 4
    fixed = [(v['config']['r'], v['config']['M'], v['config']['m'], v['config']['mode_hash'],
              v['config']['scheme'], v['config']['dt'], v['config']['refresh'], v['config']['trust']) for v in summary.values()]
    assert all(v == fixed[0] for v in fixed), 'reduced settings or modes changed across mesh cost curves'
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = {'mf': 'Fixed Gauss input / fixed output', 'mesh': 'Native mesh input / growing output'}
    colours = {'mf': '#27659a', 'mesh': '#a65c25'}
    for path in ['mf', 'mesh']:
        points = sorted([v for v in summary.values() if v['config']['contract'] == path], key=lambda v: v['config']['N'])
        for contract, style in [('descriptor', '-'), ('given_native', ':')]:
            good = [v for v in points if v['costs'][contract]['complete']]
            axes[0].plot([v['config']['N'] for v in good], [v['costs'][contract]['median_seconds']*1e3 for v in good],
                         style, color=colours[path], label=labels[path]+' · '+contract.replace('_', ' '))
            for v in good:
                axes[0].scatter(v['config']['N'], v['costs'][contract]['median_seconds']*1e3,
                    c=colours[path] if v['Q1'] and v['Q2'] else 'none', edgecolors=colours[path], s=38)
        for field, style in [('nonlinear_iterations', '-'), ('jacobian_contractions', ':')]:
            axes[1].plot([v['config']['N'] for v in points], [v['work'][field] for v in points],
                         style, color=colours[path], marker='o', label=labels[path]+' · '+field.replace('_', ' '))
    axes[0].set(yscale='log', ylabel='Case-balanced query median (ms)', title='Two explicit input/output contracts')
    axes[1].set(ylabel='Median logical count per trajectory', title='Nonlinear work at frozen reduced settings')
    for ax in axes:
        ax.set(xlabel='Mesh nodes per axis N', xticks=[65, 129]); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('One allocation · fixed r, M, m and modes · selected cases only\nQ2 uses the common scoring lattice', fontsize=11)
    save(fig, 'fixed_mode_mesh_cost_work')
    text = '''### Fixed-mode mesh and output-cost controls

These are selected-case diagnostics on the same GPU allocation, with identical bank, rank,
test count/order, quadrature and stepping settings. Filled cost markers pass the declared
subset Q1/Q2; open markers retain failed subset controls. The growing-output score restricts
the actual returned fields to the common scoring lattice. Native-field and common-lattice
tight distances remain separate. Q2 successor agreement uses the common lattice; these
checks do not supply a finer-native-grid E0 accuracy certificate. No exponent is fitted from two meshes.
This is not complete-validation qualification at the new mesh or a comparative POD advantage.

'''
    text += (f"Mesh job **{doc['execution']['job_id']}**, GPU `{doc['execution']['gpu']}`. "
             'Its wall-clock costs are compared only within this mesh allocation, separately from the complete-validation frontier job.\n\n')
    parameter_rows = []
    for qid, scope in [(next(iter(summary)), 'M2 timed mesh controls'),
                       ('W1024_CN_baseline_gl56', 'W1024 supplied-coefficient controls'),
                       ('W1024_BDF2_baseline_gl56', 'W1024 supplied-coefficient controls')]:
        q = doc['queries'][qid]
        policy = (f"QN refresh every {q['refresh']} steps" if 'refresh' in q else
                  f"QN refresh every {q['work_policy']} steps" if q['solver'] == 'qn' else
                  f"FS {q['work_policy']} fixed sweep(s) after startup")
        parameter_rows.append([scope, q['r'], q['M'], q['m'], q['scheme'], f"{q['dt']:.10g}", policy, q['trust']])
    text += table(['Bank / scope', 'Rank r', 'Test modes M', 'Quadrature points m', 'Scheme',
                   'Step size', 'Solver / work policy', 'Trust radius'], parameter_rows) + '\n\n'
    text += table(['Query', 'Nodes', 'Input/output contract', 'Input shape', 'Returned shape', 'Descriptor (ms)',
                   'Prepared native (ms)', 'Median corrections/Jacobian contractions', 'Subset Q1/Q2',
                   'Common-lattice median/worst E0 error (%)'], cost_rows) + '\n\n'
    text += '![Fixed-mode mesh cost and work](figs/fixed_mode_mesh_cost_work.png)\n\n'
    text += table(['Query', 'Fresh bad/steps', 'Max common production–tight distance',
                   'Max growing-native production–tight distance', 'Max common tight–tighter distance',
                   'Max growing-native tight–tighter distance',
                   'Common target→next distance', 'Common next→finest distance', 'Own-state rho'], qualification_rows) + '\n\n'
    text += ('The two contracts are different supplied-data/output tasks. Preparation and decode are part '
             'of total query cost. The separate phase calls below are indicative and are not summed to '
             'replace the measured fused query. All listed statistics use the same selected cases.\n\n')
    text += table(['Query', 'Initialisation (ms)', 'Evolution (ms)', 'Decode (ms)', 'Compile / warm-up (s)'], phase_rows) + '\n\n'
    operator_rows, case5_rows, operator_summary, case5_values = [], [], {}, []
    fig, ax = plt.subplots(figsize=(9, 4.5))
    interventions = ['baseline', 'composite', 'lambda', 'moments']
    for si, scheme in enumerate(['CN', 'BDF2']):
        values, epsilons, admitted = [], [], []
        for ii, intervention in enumerate(interventions):
            qid = f'W1024_{scheme}_{intervention}_gl56'
            query, result = doc['queries'][qid], doc['case_results'][qid]
            errors = [r['score']['e'] for r in result.values()]
            operator_summary[qid] = dict(cases=query['cases'], median_error=float(np.median(errors)),
                worst_error=max(errors), Q1=query['Q1']['passed'], Q2=query['Q2']['passed'],
                case_errors={cid: r['score']['e'] for cid, r in result.items()})
            operator_rows.append([qid, str(query['cases']), f'{100*np.median(errors):.7g}', f'{100*max(errors):.7g}',
                f"{query['Q1']['passed']}/{query['Q2']['passed']}", f"{max(r['common_distance_tight'] for r in result.values()):.7g}"])
            own = result['5']['score'];base = doc['case_results'][f'W1024_{scheme}_baseline_gl56']['5']['score']
            eps = own['epsilon'];assert np.isclose(eps, reference['cohorts']['val64']['cases']['5']['eps_ref'], rtol=1e-10, atol=0)
            resolved = own['e']+eps < max(base['e']-eps, 0)
            case5_values.append(dict(scheme=scheme, intervention=intervention, error=own['e'],
                baseline_error=base['e'], reference_uncertainty=eps, change=own['e']-base['e'],
                resolved_reference_error_improvement=resolved, Q1=query['Q1']['passed'], Q2=query['Q2']['passed']))
            case5_rows.append([scheme, intervention, f"{100*own['e']:.7g}", f"{100*(own['e']-base['e']):.7g}",
                               'baseline' if intervention == 'baseline' else resolved])
            values.append(own['e']*100);epsilons.append(eps*100)
            admitted.append(query['Q1']['passed'] and query['Q2']['passed'])
        xx = np.arange(len(interventions))+(si-.5)*.34
        bars = ax.bar(xx, values, .32, label=scheme, yerr=epsilons, capsize=3)
        for bar, good in zip(bars, admitted):
            if not good:
                bar.set_hatch('//');bar.set_alpha(.5)
    ax.set(xticks=range(len(interventions)), xticklabels=interventions,
           ylabel='Case 5 conditional E0 ST error (%)',
           title='Supplied-c0 operator interventions, fixed scoring lattice\nHatching: failed subset Q1/Q2')
    ax.grid(axis='y', alpha=.2);ax.legend();save(fig, 'operator_interventions')
    text += '''#### Separate operator interventions

The following controls hold the original coefficient projection and output judge fixed.
They are supplied-coefficient diagnostics, not growing-mesh queries, and have no measured
deployment cost in this panel. Composite controls use the selected diagnostic cases;
lambda and moment controls use the single declared difficult case. The lambda parameter
changes diffusion eigenvalues together with induced least-squares row weights. Changing
moments also recomputes the actual predictor QR. These factors do not isolate physical
diffusion bias or pure temporal error from every projection/weighting effect.

'''
    text += table(['Control', 'Exact case ids', 'Median error (%)', 'Worst error (%)', 'Subset Q1/Q2',
                   'Max common production–tight distance'], operator_rows) + '\n\n'
    text += table(['Scheme', 'Case 5 intervention', 'Error (%)', 'Change from baseline (percentage points)',
                   'Resolved relative to reference uncertainty only'], case5_rows) + '\n\n'
    text += '![Supplied-coefficient operator controls](figs/operator_interventions.png)\n\n'
    text += ('The bars show conditional numerical-reference intervals, not statistical confidence. '
             'Qualification masks and scope in the table apply to every interpretation; an apparently '
             'lower diagnostic error does not create an admitted production frontier point.\n\n')
    setup_rows = []
    for level in ['shared_setup', 'incremental_setup']:
        for record in doc[level]:
            before, after = record['before'], record['after']
            current = None if before['current_bytes'] is None or after['current_bytes'] is None else after['current_bytes']-before['current_bytes']
            peak = after['whole_job_cumulative_peak_bytes']
            setup_rows.append([level, record['bank'], record.get('n', 'both'), record.get('rule', 'shared'),
                record.get('intervention', 'none'), f"{record['seconds']:.6g}",
                'unavailable' if current is None else f'{current/1e9:.6g}',
                'unavailable' if peak is None else f'{peak/1e9:.6g}'])
    text += table(['Setup level', 'Bank', 'Base data nodes', 'Rule', 'Intervention', 'Seconds',
                   'Current-memory change (GB)', 'Whole-job peak so far (GB)'], setup_rows) + '\n\n'
    projection_rows = [[key, f"{v['common65_distance']:.7g}", v.get('baseline', v.get('held_projection', 'see contract'))]
                       for key, v in doc['initial_projection_differences'].items()]
    text += table(['Initial-projection comparison', 'Common-lattice field distance', 'Projection baseline/scope'], projection_rows) + '\n\n'
    text += ('Shared assembly prepares both meshes; incremental setup is additional rule/operator work. '
             'Base data nodes identify the runner data structure; a held-projection control can use finer-mesh matrices with the original base data. '
             'The assembly table excludes compilation and warm-up, which are listed separately for timed target queries above. '
             'The peak counter accumulates through the entire job, so it is not standalone method memory. '
             'Native-projection differences above are reported separately and are not inserted into '
             'the held-projection operator controls. Exact matrix/weight hashes, projection geometry '
             'and all raw repetitions remain in the frozen dataset.\n\n')
    return dict(text=text, cost_queries=summary, operator_case5=case5_values, operator_controls=operator_summary,
                execution=doc['execution'], frozen_modes=doc['frozen_modes'])
