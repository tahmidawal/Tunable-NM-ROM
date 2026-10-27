"""Render an actual, frozen Stage-F dataset; never substitute historical costs or scores."""
from __future__ import annotations

import collections
import json

import matplotlib.pyplot as plt
import numpy as np


def explicit_pass(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        for name in ['pass', 'passed', 'ok', 'all_pass']:
            if name in value and isinstance(value[name], bool):
                return value[name]
    return False


def timing_pass(row, contract):
    if contract == 'prepared_native' and 'native_timing_admitted' in row:
        assert isinstance(row['native_timing_admitted'], bool)
        return row['native_timing_admitted']
    value = row['timing_admitted']
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        for name in [contract, contract + '_seconds', 'given_native' if contract == 'prepared_native' else contract]:
            if name in value and isinstance(value[name], bool):
                return value[name]
    raise ValueError(f'unknown contract-specific timing admission schema: {value}')


def method_group(subject_id, subject):
    cfg = subject['config']
    text = ' '.join(str(v) for v in [subject_id, cfg.get('kind'), cfg.get('method'), cfg.get('family')]).lower()
    if 'pod' in text or 'ecsw' in text:
        return 'POD'
    if 'spectral' in text or 'ps|' in text or text.startswith('ps'):
        return 'Spectral FOM'
    if 'fd' in text or 'central' in text:
        return 'Finite-difference FOM'
    return 'Bank ROM'


def components(doc, table, save):
    phases = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in doc['phase_repetitions']:
        if row['seconds'] is None:
            continue
        assert np.isfinite(row['seconds']) and row['seconds'] > 0
        phases[(row['subject'], row['phase'])][int(row['case'])].append(row['seconds'])
    phase_rows, phase_values, work_rows, setup_rows = [], {}, [], []
    columns = [('nonlinear_iterations', 'Corrections'), ('jacobian_contractions', 'Jacobian contractions'),
               ('residual_jacobian_pairs', 'Residual/Jacobian pairs'), ('residual_evaluations', 'Residual calls'),
               ('nonlinear_evaluations', 'Nonlinear calls')]
    for sid, subject in doc['subjects'].items():
        result, coverage = {}, {}
        for phase in ['initialization', 'evolve', 'decode']:
            own = phases[(sid, phase)]
            assert set(own) <= {0, 5, 30, 43}, (sid, phase, 'unexpected phase case')
            coverage[phase] = sorted(own)
            if sorted(own) == [0, 5, 30, 43]:
                result[phase] = float(np.median([np.median(v) for v in own.values()]))
            else:
                result[phase] = None
        result['case_coverage'] = coverage
        phase_values[sid] = result
        phase_rows.append([sid] + [f'incomplete {len(coverage[p])}/4' if result[p] is None and coverage[p]
                                  else 'not applicable / unavailable' if result[p] is None else f'{result[p]*1e3:.6g}'
                                  for p in ['initialization', 'evolve', 'decode']])
        counts = []
        raw = doc['case_results'][sid]
        for key, _ in columns:
            vals = [v['work'].get(key) for v in raw.values()]
            if any(v is None for v in vals):
                counts.append('not exposed')
            else:
                assert np.isfinite(vals).all() and min(vals) >= 0
                counts.append(f'{np.median(vals):.6g} [{min(vals):.6g}, {max(vals):.6g}]')
        work_rows.append([sid] + counts)
        memory = subject['setup_memory']
        before, after = memory['before']['current_bytes'], memory['after']['current_bytes']
        setup_rows.append([sid, ', '.join(subject['shared_setup_keys']) or 'none',
            f"{subject['setup_seconds']:.6g}", f"{subject['compile_and_warm_seconds']:.6g}",
            'unavailable' if before is None or after is None else f'{(after-before)/1e9:.6g}',
            'unavailable' if subject['setup_peak_bytes'] is None else f"{subject['setup_peak_bytes']/1e9:.6g}"])
    fig, ax = plt.subplots(figsize=(9, max(4, .34 * len(phase_values))))
    matrix = np.array([[np.nan if v[p] is None else v[p]*1e3 for p in ['initialization', 'evolve', 'decode']]
                       for v in phase_values.values()])
    from matplotlib.colors import LogNorm
    im = ax.imshow(matrix, aspect='auto', norm=LogNorm(), cmap='Blues')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            x = matrix[i, j]
            ax.text(j, i, 'n/a' if np.isnan(x) else f'{x:.3g}', ha='center', va='center', fontsize=7,
                    color='white' if np.isfinite(x) and x > np.nanmedian(matrix)*2 else 'black')
    ax.set(xticks=range(3), xticklabels=['Initialisation', 'Reduced/full evolution', 'Output decode/synthesis'],
           yticks=range(len(phase_values)), yticklabels=list(phase_values),
           title='Separate phase calls: case-balanced median milliseconds\nDiagnostic cases only; phases do not sum to fused query cost')
    fig.colorbar(im, ax=ax, label='Milliseconds (log colour scale)'); save(fig, 'query_components')
    text = '''#### Query phases, work and offline setup

The separate phase calls below use only the fixed diagnostic cases, with the same output
checked against the fused query. Their costs are indicative components and are not added
to construct or replace the full-validation query timer. A full solver supplied its native
dynamic state has no separate initial coefficient projection.
Aggregates require all diagnostic cases; skipped invalid cases remain incomplete rather than
being dropped to create a faster phase estimate. Raw per-case phase calls remain in the dataset.

'''
    text += table(['Subject', 'Initialisation (ms)', 'Evolution (ms)', 'Decode/synthesis (ms)'], phase_rows) + '\n\n'
    text += '![Separately measured query phases](figs/query_components.png)\n\n'
    text += ('The following work counts are medians [minimum, maximum] over the complete validation '
             'population. They count declared logical operations, not GPU kernels. Missing counters '
             'remain missing; an iteration of one method is not assumed to cost the same as another.\n\n')
    text += table(['Subject'] + [label for _, label in columns], work_rows) + '\n\n'
    text += table(['Subject', 'Shared setup', 'Incremental setup (s)', 'Compile/warm (s)',
                   'Current-memory change (GB)', 'Whole-job peak so far (GB)'], setup_rows) + '\n\n'
    shared_rows = [[sid, f"{v['seconds']:.6g}", json.dumps({k: v[k] for k in ['memory_before', 'memory_after']})]
                   for sid, v in doc['shared_setup'].items()]
    text += table(['Shared bank/basis setup', 'Seconds', 'Memory counters'], shared_rows) + '\n\n'
    text += (f"Common provenance/descriptor setup takes {doc['common_setup']['seconds']:.6g} seconds. "
             'Shared setup is reported once, separate from each incremental configuration and from '
             'compilation/warm-up. It loads and assembles existing artifacts; historical training and '
             'support fitting are not rerun or included in those setup seconds. The new rejected fit '
             'has separate allocation accounting. Device-memory changes are current-allocation differences; the peak '
             'counter is cumulative for the entire job and cannot rank standalone method memory. '
             'Inputs and returned fields are on device at the query boundary; synchronization is included, '
             'while output transfer, scoring and storage are excluded equally. Descriptor preparation is '
             'included in the primary query contract.\n\n')
    return text, phase_values


def comparisons(doc, summaries, table):
    rows = []
    for sid, families in doc['matching']['subjects'].items():
        for family, contracts in families.items():
            own = contracts['descriptor']
            selected = own['selected']
            rows.append([sid, family, selected or 'unresolved / no tested joint match',
                'unavailable' if selected is None else f"{own['competitor_over_subject_cost_ratio']:.6g}"])
            if selected:
                assert summaries[sid]['registered_qualified'] and summaries[selected]['registered_qualified']
                assert own['checks'][selected]['joint_match'] is True
                assert (summaries[selected]['median_upper'] <= summaries[sid]['median_lower'] and
                        summaries[selected]['worst_upper'] <= summaries[sid]['worst_lower'])
                expected = summaries[selected]['costs']['descriptor']['median_seconds'] / summaries[sid]['costs']['descriptor']['median_seconds']
                assert np.isclose(own['competitor_over_subject_cost_ratio'], expected, rtol=1e-12)
    text = '''#### Tuned POD and full-solver comparisons

The primary descriptor-total comparison below requires a joint match: both the competitor's
median upper conditional error and worst upper conditional error must be no larger than the
bank/POD subject's corresponding lower errors, on exactly the same eligible cases. The selected
competitor is the fastest admitted member of the frozen tested set satisfying both conditions.
An unresolved overlap is not a win. A ratio below unity means the matched competitor is faster.

'''
    text += table(['Subject', 'Competitor family', 'Fastest tested joint match',
                   'Competitor / subject cost'], rows) + '\n\n'
    text += ('Prepared-native matches and the separate median-only/worst-only predicates are retained '
             'in the machine dataset. This is a finite developmental comparison, not a global optimum '
             'or sealed confirmation. ')
    hr = [sid for sid, v in summaries.items() if v['method'] == 'POD' and v['registered_qualified'] and
          v['qualification']['method_specific_extra_masks'].get('hyper_reduced') is True]
    text += ('Qualified hyper-reduced POD is included for the explicitly admitted improved-policy subject(s).\n\n'
             if hr else 'The qualified hyper-reduced POD gap remains separate from the measured unsampled POD comparison.\n\n')
    old = [sid for sid, v in doc['subjects'].items() if v['config'].get('archived_key')]
    new = [sid for sid, v in doc['subjects'].items() if v['config'].get('successors')]
    rows = []
    for sid in new:
        for baseline in old:
            a, b = summaries[sid], summaries[baseline]
            valid = a['registered_qualified'] and b['registered_qualified'] and all(
                v['complete_score_population'] and v['costs']['descriptor']['complete_population'] for v in [a, b])
            rows.append([sid, baseline,
                'unqualified / incomplete' if not valid else a['median_upper'] < b['median_lower'],
                'unqualified / incomplete' if not valid else a['worst_upper'] < b['worst_lower'],
                'unavailable' if not valid else f"{b['costs']['descriptor']['median_seconds']/a['costs']['descriptor']['median_seconds']:.6g}"])
    if rows:
        text += ('#### Improvement over earlier bank ROMs\n\n'
                 'The following same-allocation comparisons separate resolved median improvement, '
                 'resolved worst-case improvement and speed. A gain in one column does not imply '
                 'a gain in the others. True requires non-overlapping conditional reference intervals.\n\n')
        text += table(['New policy', 'Earlier bank setting', 'Median resolved better', 'Worst resolved better',
                       'Earlier / new query cost'], rows) + '\n\n'
    return text


def key_findings(doc, summaries, table):
    usable = {sid: v for sid, v in summaries.items() if v['registered_qualified'] and
              v['complete_score_population'] and v['costs']['descriptor']['complete_population']}
    banks = [sid for sid, v in usable.items() if v['method'] == 'Bank ROM']
    spectral = []
    for sid in banks:
        match = doc['matching']['subjects'][sid]['PS']['descriptor']
        if match['selected'] and match['competitor_over_subject_cost_ratio'] < 1:
            spectral.append(dict(subject=sid, competitor=match['selected'],
                subject_over_competitor_cost=1/match['competitor_over_subject_cost_ratio']))
    new = sorted([sid for sid in banks if doc['subjects'][sid]['config'].get('successors')],
                 key=lambda sid: doc['subjects'][sid]['config']['dt'], reverse=True)
    old = [sid for sid in banks if doc['subjects'][sid]['config'].get('archived_key')]
    simultaneous = []
    for sid in new:
        a = usable[sid]
        for baseline in old:
            b = usable[baseline]
            if a['median_upper'] < b['median_lower'] and a['worst_upper'] < b['worst_lower'] and a['costs']['descriptor']['median_seconds'] < b['costs']['descriptor']['median_seconds']:
                simultaneous.append([sid, baseline])
    text = '## What the bounded campaign establishes\n\n'
    text += (f"All {len(usable)} admitted subjects below retain complete error and descriptor-cost coverage. "
             'Admission certifies the registered numerical checks; it is not a physical-accuracy target.\n\n')
    if banks and len(spectral) == len(banks):
        ratios = [v['subject_over_competitor_cost'] for v in spectral]
        text += (f"**Every tested admitted bank setting has a faster jointly accuracy-matching spectral comparator.** "
                 f"Across these {len(banks)} bank settings, observed descriptor-query speed ratios range "
                 f"from {min(ratios):.4g} to {max(ratios):.4g}. Matching requires both competitor upper "
                 'conditional median/worst errors to lie below the corresponding bank lower errors. '
                 'This is a same-allocation result on the frozen box/cohort and finite tested set.\n\n')
    else:
        text += f"A faster jointly matching spectral comparator exists for {len(spectral)}/{len(banks)} admitted bank settings.\n\n"
    rows = []
    for sid in new:
        v = usable[sid]
        rows.append([sid, f"{v['median_error']*100:.7g}", f"{v['worst_error']*100:.7g}",
                     v['errors_above_1pct'], f"{v['costs']['descriptor']['median_seconds']*1e3:.7g}"])
    if rows:
        text += ('The refreshed-QN endpoints close their declared solver/quadrature checks. '
                 'Their complete-validation accuracy and cost are:\n\n')
        text += table(['Refreshed bank endpoint', 'Median error (%)', 'Worst error (%)', 'Cases above 1%',
                       'Descriptor query (ms)'], rows) + '\n\n'
        if not simultaneous:
            text += ('Neither tested endpoint simultaneously improves resolved median error, resolved worst error '
                     'and measured query cost over any tested earlier bank setting. Refinement improves the '
                     'median while worsening the worst case and increasing cost. The intermediate time setting '
                     'has selected-case evidence only and is not silently discarded.\n\n')
    pod = [sid for sid, v in usable.items() if v['method'] == 'POD']
    text += (f"The {len(pod)} admitted POD subjects evaluate the unsampled full residual. Their finite-schedule "
             'matches are distinct from earlier-bank improvements and from the faster full-solver matches. '
             'The unresolved hyper-reduced-POD gate is not evidence of a bank advantage. '
             'Detailed matching tables and projection/operator controls follow.\n\n')
    return text, dict(admitted_complete_subjects=len(usable), bank_subjects=banks,
        faster_joint_spectral_matches=spectral, new_endpoints=new, simultaneous_new_bank_wins=simultaneous,
        admitted_unsampled_POD=pod)


def projection_geometry(doc, table, save):
    budgets = doc['projection_budgets']
    rows, worst_rows, all_floors = [], [], []
    for span, budget in budgets.items():
        cases = budget['cases']
        assert set(int(k) for k in cases) <= set(range(64)), (span, 'unexpected projection population')
        floors = []
        for cid, record in cases.items():
            assert record['reference'] == f'3d/val64/ST/c{int(cid):03d}'
            assert len(record['floor_e_t']) == len(doc['reference']['output_times'])
            assert max(record['orthogonality_relative']) <= 1e-10
            assert np.isclose(record['floor_max'], max(record['floor_e_t'][1:]), rtol=1e-12)
            floors.append(record['floor_max'])
        all_floors.extend(floors)
        complete = sorted(int(k) for k in cases) == list(range(64))
        rows.append([span, budget['r'], f'{len(floors)}/64',
                     f'{100*np.median(floors):.7g}' if complete else 'incomplete',
                     f'{100*max(floors):.7g}' if complete else 'incomplete'])
    fig, ax = plt.subplots(figsize=(8, 5))
    for sid, subject in doc['subjects'].items():
        records = doc['case_results'][sid]
        pairs = []
        for cid, record in records.items():
            geometry = record.get('projection_budget')
            if geometry is None:
                continue
            floor = np.asarray(geometry['reference_projection_floor_e_t'])
            dyn = np.asarray(geometry['evolution_in_span_distance_e_t'])
            actual = np.asarray(record['score']['e_t'], dtype=float)
            if not np.isfinite(actual).all():
                continue
            defect = actual**2 - floor**2 - dyn**2
            assert np.max(np.abs(defect)) <= 1e-8 * max(float(np.max(actual**2)), 1e-16) + 1e-12
            pairs.append((max(floor[1:]), record['score']['e'], int(cid), actual, floor, dyn))
        if not pairs:
            continue
        if len(pairs) == 64:
            worst = max(pairs, key=lambda v: v[1])
            t = 1 + int(np.argmax(worst[3][1:]))
            worst_rows.append([sid, worst[2], doc['reference']['output_times'][t], f'{100*worst[3][t]:.7g}',
                               f'{100*worst[4][t]:.7g}', f'{100*worst[5][t]:.7g}'])
        else:
            worst_rows.append([sid, 'incomplete', 'incomplete', 'incomplete', 'incomplete', 'incomplete'])
        # Show the wide-bank diagnostics and the tuned POD representative; all spans remain in the table.
        cfg = subject['config']
        show = cfg.get('setting') == 'W1024' or ('POD' in sid and 'CN' in sid)
        if show:
            ax.scatter([v[0]*100 for v in pairs], [v[1]*100 for v in pairs], s=20, alpha=.65, label=sid)
    limits = [v for v in all_floors if v > 0]
    if limits:
        lo, hi = min(limits)*100, max(limits)*100
        ax.plot([lo, hi], [lo, hi], '--', color='#555', label='Rollout equals best field fit')
    ax.set(xscale='log', yscale='log', xlabel='Best reference-field fit: maximum evolved-time error (%)',
           ylabel='Actual rollout: maximum evolved-time error (%)', title='Reference-span geometry on the common scoring lattice')
    ax.grid(alpha=.2); ax.legend(fontsize=7); save(fig, 'reference_projection_budget')
    text = '''#### Representation floor and trajectory error

The best least-squares fit of each numerical reference field supplies a geometric control,
using the same scoring lattice and fixed reference initial norm. Coverage is explicit;
population medians and maxima require every validation case. The control is not a new bank
fit or a feasible PDE trajectory, and missing geometry from a failed rollout remains missing.
A floor below the conditional reference uncertainty is not a resolved physical lower bound.

'''
    text += table(['Frozen span', 'Rank', 'Cases', 'Median reference floor (%)', 'Worst reference floor (%)'], rows) + '\n\n'
    text += '![Reference projection versus actual trajectory error](figs/reference_projection_budget.png)\n\n'
    text += ('For each subject below, the case and time are selected by its actual largest evolved-output '
             'error. At that same case/time, squared total error equals squared projection residual plus '
             'squared distance within the span, up to the recorded orthogonality/round-off checks. '
             'This identity does not causally split time, spatial, quadrature and solver errors.\n\n')
    text += table(['Subject', 'Worst case', 'Time', 'Total error (%)', 'Reference floor at that time (%)',
                   'Distance within span (%)'], worst_rows) + '\n\n'
    return text


def render(doc, figdir, table, save, reference):
    required = ['schema_version', 'cohort', 'reference', 'contracts', 'subjects', 'case_results',
                'timing_repetitions', 'phase_repetitions', 'matching', 'frontiers', 'execution']
    assert all(k in doc for k in required) and doc['schema_version'] == 1
    cohort, ref = doc['cohort'], doc['reference']
    expected = reference['cohorts']['val64']
    assert cohort['name'] == 'val64' and cohort['seed'] == expected['seed']
    ordered = cohort.get('ordered_case_ids', cohort.get('ordered_cases'))
    assert ordered == list(range(64))
    if 'ordered_case_ids' in cohort and 'ordered_cases' in cohort:
        assert cohort['ordered_case_ids'] == cohort['ordered_cases']
    assert cohort['eligible_cases'] == sorted(int(k) for k, v in expected['cases'].items() if v['eligible'])
    assert cohort['canonical_table_sha256'] == expected['table_sha256']
    assert ref['dimension'] == 3 and ref['kind'] == 'ST' and ref['field'] == 'finest'
    assert ref['store_version'] == reference['store_version'] and ref['status_sha256'] == reference['status_sha256']
    assert np.allclose(ref['output_times'], reference['output_times'], rtol=0, atol=1e-15)
    execution = doc['execution']
    assert execution['complete'] is True, 'progress checkpoint is not a completed frontier'
    assert execution['job_id'] and execution['gpu'] and execution['source_commit'] and execution['config_sha256']
    assert execution['raw_file_manifest']
    assert all(v['sha256'] and v['bytes'] > 0 for v in execution['raw_file_manifest'])
    rows = doc['timing_repetitions']
    by_invocation = {r['invocation_id']: r for r in rows}
    assert len(by_invocation) == len(rows), 'duplicate timing invocation IDs'
    # Derive the exact observed full population from per-invocation evidence, then check the contract.
    ids = sorted({int(r['case']) for r in rows})
    assert ids == list(range(64)), f'not the full declared validation population: {ids}'
    invalid_subjects = set()
    for row in rows:
        assert row['subject'] in doc['subjects']
        score = row.get('score') or {}
        cid = str(int(row['case']))
        target = reference['cohorts']['val64']['cases'][cid]
        assert isinstance(score['epsilon'], (int, float)) and np.isfinite(score['epsilon'])
        assert np.isclose(score['epsilon'], target['eps_ref'], rtol=1e-10, atol=0)
        assert score['eligible'] is target['eligible']
        for contract, field in [('prepared_native', 'given_native_seconds'), ('descriptor', 'descriptor_seconds')]:
            if timing_pass(row, contract):
                assert isinstance(row[field], (int, float)) and np.isfinite(row[field]) and row[field] > 0
        assert np.isfinite(row['input_prep_seconds']) and row['input_prep_seconds'] >= 0
        assert abs(row['descriptor_seconds'] - row['given_native_seconds'] - row['input_prep_seconds']) <= 1e-9
        finite = isinstance(score.get('e'), (int, float)) and np.isfinite(score['e'])
        if not finite or score.get('eligible') is not True:
            # The driver records clock-drift admission separately from numerical validity.
            # Retain the failed invocation and its raw duration, but never plot its cost/error.
            invalid_subjects.add(row['subject'])
            continue
        assert len(score['e_t']) == len(reference['output_times']) and np.isfinite(score['e_t']).all()
        assert abs(max(score['e_t'][1:]) - score['e']) <= 1e-12
        assert row['output_sha256']
    summaries, table_rows = {}, []
    groups = collections.defaultdict(list)
    for row in rows:
        if row['role'] == 'B':
            groups[row['subject']].append(row)
    for subject_id, subject in doc['subjects'].items():
        own = groups.get(subject_id, [])
        raw_cases = doc['case_results'][subject_id]
        assert sorted(int(c) for c in raw_cases) == ids, (subject_id, 'missing raw case result')
        for cid, result in raw_cases.items():
            row = by_invocation[result['invocation_id']]
            assert (row['subject'], row['case'], row['role']) == (subject_id, int(cid), 'B')
            assert row['output_sha256'] == result['output_sha256'] and row['score'] == result['score']
            if result.get('coefficient_sha256') is not None:
                assert row['coefficient_sha256'] == result['coefficient_sha256']
        score_by_case = {}
        for row in own:
            if row['score'].get('e') is None or not np.isfinite(row['score']['e']) or row['score'].get('eligible') is not True:
                continue
            cid = int(row['case'])
            if cid in score_by_case:
                assert abs(score_by_case[cid] - row['score']['e']) <= 1e-12, (subject_id, cid)
            score_by_case[cid] = float(row['score']['e'])
        complete_scores = sorted(score_by_case) == ids and subject_id not in invalid_subjects
        q = subject['qualification']
        common = explicit_pass(q['common_rollout_check'])
        q1, q2 = explicit_pass(q['Q1']), explicit_pass(q['Q2'])
        assert q['scope'] and isinstance(q['method_specific_extra_masks'], dict)
        qualified = str(q['status']).lower() in ['qualified', 'pass', 'accepted']
        assert qualified == (q1 and q2 and common), (subject_id, 'status disagrees with registered admission masks')
        entry = dict(id=subject_id, method=method_group(subject_id, subject), qualification=q,
                     registered_qualified=qualified, common_rollout_pass=common,
                     complete_score_population=complete_scores,
                     missing_score_case_ids=[c for c in ids if c not in score_by_case],
                     numerically_invalid_cases=[int(c) for c, r in raw_cases.items() if not r['valid']],
                     case_ids=sorted(score_by_case), costs={}, source_config=subject['config'])
        if complete_scores:
            errors = [score_by_case[c] for c in ids]
            epsilon = np.array([reference['cohorts']['val64']['cases'][str(c)]['eps_ref'] for c in ids])
            lower, upper = np.maximum(np.array(errors)-epsilon, 0), np.array(errors)+epsilon
            entry.update(median_error=float(np.median(errors)), worst_error=max(errors),
                         median_lower=float(np.median(lower)), median_upper=float(np.median(upper)),
                         worst_lower=float(max(lower)), worst_upper=float(max(upper)),
                         errors_above_1pct=sum(e > .01 for e in errors))
        for contract, field in [('prepared_native', 'given_native_seconds'), ('descriptor', 'descriptor_seconds')]:
            good = [r for r in own if timing_pass(r, contract)]
            case_times = {c: [float(r[field]) for r in good if int(r['case']) == c] for c in ids}
            complete_timing = all(case_times[c] for c in ids)
            entry['costs'][contract] = dict(complete_population=complete_timing,
                admitted_repetitions=len(good), case_repetitions=case_times,
                median_seconds=float(np.median([np.median(case_times[c]) for c in ids])) if complete_timing else None)
        summaries[subject_id] = entry
        source_summary = doc['summaries'][subject_id]
        for contract in ['descriptor', 'prepared_native']:
            ours, theirs = entry['costs'][contract], source_summary['costs'][contract]
            assert ours['complete_population'] is theirs['coverage_complete']
            assert ours['admitted_repetitions'] == theirs['admitted_repetitions']
            if ours['complete_population']:
                assert np.isclose(ours['median_seconds'], theirs['median_seconds'], rtol=1e-12)
        if complete_scores:
            assert np.isclose(entry['median_error'], source_summary['median_error'], rtol=1e-12)
            assert np.isclose(entry['worst_error'], source_summary['worst_error'], rtol=1e-12)
        full_solver = entry['method'] in ['Spectral FOM', 'Finite-difference FOM']
        q2_label = 'n/a' if q['Q2'].get('applicability') else str(q2)
        check_label = f'stage-valid={q1}; Q2={q2_label}' if full_solver else f'Q1={q1}; Q2={q2_label}'
        table_rows.append([subject_id, entry['method'], q['status'], check_label, common,
            'missing' if not complete_scores else f"{100*entry['median_error']:.6g}",
            'missing' if not complete_scores else f"{100*entry['worst_error']:.6g}",
            entry.get('errors_above_1pct', 'missing'),
            'missing' if entry['costs']['descriptor']['median_seconds'] is None else f"{entry['costs']['descriptor']['median_seconds']*1e3:.6g}",
            'missing' if entry['costs']['prepared_native']['median_seconds'] is None else f"{entry['costs']['prepared_native']['median_seconds']*1e3:.6g}"])
    figures = []
    colours = {'Bank ROM': '#27659a', 'POD': '#9a5b27', 'Spectral FOM': '#3a7656', 'Finite-difference FOM': '#6c668b'}
    for contract in ['descriptor', 'prepared_native']:
        points = [r for r in summaries.values() if r['complete_score_population'] and
                  r['costs'][contract]['complete_population']]
        if not points:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, metric in zip(axes, ['median_error', 'worst_error']):
            seen = set()
            for point in points:
                q, group = point['qualification'], point['method']
                qualified = point['registered_qualified']
                common = explicit_pass(q['common_rollout_check'])
                label = group if group not in seen else None
                seen.add(group)
                xx = point['costs'][contract]['median_seconds'] * 1e3
                yy = point[metric] * 100
                bound_name = 'median' if metric == 'median_error' else 'worst'
                low, high = point[bound_name + '_lower'] * 100, point[bound_name + '_upper'] * 100
                lower_bar = yy-low if low > 0 else 0
                ax.errorbar(xx, yy, yerr=[[max(lower_bar, 0)], [max(high-yy, 0)]],
                            fmt='none', ecolor=colours[group], alpha=.35, linewidth=.7, capsize=2)
                ax.scatter(xx, yy, s=48, c=colours[group] if qualified else 'none',
                           edgecolors=colours[group], marker='o' if qualified or common else 's', label=label)
            ax.set(xscale='log', yscale='log', xlabel=f'{contract.replace("_", " ")} query: median over 64 cases (ms)',
                   ylabel=f'{metric.replace("_", " ").capitalize()} (%)')
            ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle(f"Same allocation job {execution['job_id']} · full validation · device outputs · conditional E0 ST error\n"
                     'Filled: admitted; open: retained · Whiskers: conditional reference intervals', fontsize=10)
        name = f'full64_{contract}_frontier'
        save(fig, name)
        figures.append((contract, name))
    text = '''### Same-allocation full-validation frontier

The following dataset uses the complete validation population and scores the fields returned
by the same measured invocations. Query costs are case-balanced medians; timing rejection
cannot silently remove a difficult case. Missing timing coverage remains missing.
Descriptor-total is the primary cross-method contract. Prepared-native costs use different
method-specific supplied data, so those values are shown as decomposition evidence.

'''
    text += f"Frontier job **{execution['job_id']}**, GPU `{execution['gpu']}`, source `{execution['source_commit']}`. Costs below are compared within this allocation.\n\n"
    text += table(['Subject', 'Method', 'Qualification status', 'Registered checks', 'Method common check', 'Median E0 error (%)', 'Worst E0 error (%)',
                   'Cases above 1%', 'Descriptor total (ms)', 'Prepared native (ms)'], table_rows) + '\n\n'
    parameter_rows = []
    for sid, subject in doc['subjects'].items():
        cfg, p = subject['config'], subject['effective_parameters']
        shape = subject['input_contract']['shape']
        if cfg['kind'] == 'bank':
            spatial = f"r={p['r']}; M={p['M']}; m={p['m']}"
            time = f"{cfg['scheme']}; dt={cfg['dt']:.10g}"
            policy = (f"QN refresh={p['refresh']}; g_in={p['g_in']}; max corrections={p['q_max']}" if cfg['solver'] == 'qn' else
                      f"FS sweeps={p['K']}; gtol={p['gtol']}; adaptive startup steps={p['FS_a_eff']}; startup budget={p['lm_budget']}")
            policy += f"; trust={p['trust']:.7g}"
        elif cfg['kind'] == 'POD':
            spatial = f"r={p['r']}; full residual at {int(np.prod(shape))} native points"
            time = f"{cfg['scheme']}; dt={cfg['dt']:.10g}"
            policy = f"gtol={p['production_gtol']}; tolf={p['production_tolf']}; trust={p['trust']:.7g}"
        elif cfg['kind'] == 'PS':
            spatial = f"sine cutoff={p['N']}; coefficients={int(np.prod(shape))}; oversampling intervals={p['nonlinear_oversampling_intervals']}"
            time = f"ETDRK4; dt={p['dt']:.10g}"
            policy = f"contour points={p['ETDRK4_contour_points']}"
        else:
            assert cfg['kind'] == 'FD'
            spatial = f"grid intervals={p['intervals']}; native points={int(np.prod(shape))}"
            time = f"central advection / CN with Rannacher startup; dt={p['dt']:.10g}"
            policy = f"Newton tolerance={p['ntol']}; linear tolerance={p['ltol']}; calibrated level={p['tolerance_level']}"
        parameter_rows.append([sid, spatial, time, policy, str(shape)])
    text += ('The actual adapter settings below make the finite comparison explicit. The PS identifier\'s '
             'N is its sine cutoff, retaining indices below N in each axis; it is not the mesh-study node count. '
             'Further frozen parameters and source hashes remain in the dataset.\n\n')
    text += table(['Subject', 'Spatial parameters', 'Time discretization', 'Solver/work parameters', 'Prepared-native input shape'], parameter_rows) + '\n\n'
    extra_rows = []
    for sid, subject in doc['subjects'].items():
        q = subject['qualification']
        extra_rows.append([sid, q['scope'], json.dumps(q['method_specific_extra_masks'], sort_keys=True)])
    text += table(['Subject', 'Qualification scope', 'Method-specific or supplementary masks (raw)'], extra_rows) + '\n\n'
    for contract, name in figures:
        text += f'![Full validation {contract.replace("_", " ")} cost versus accuracy](figs/{name}.png)\n\n'
    text += ('''Filled markers pass their registered admission gates. Open circles pass the common
rollout check but lack registered admission; open squares fail the common rollout check.
Common rollout-consistency checks and method-specific extra checks are separate masks.
The extra POD gradient threshold is not equivalent to the bank advection-rho threshold.
All preregistered verdicts remain unchanged. A POD point that passes the common checks but
fails its extra conservative gate remains visible as descriptive evidence; its exclusion
cannot be used to claim a bank advantage over tuned POD. Unqualified points are retained.
Full solvers use their declared stage-validity/convergence checks; they do not receive ROM
tight/tighter-anchor certification, and quadrature qualification is not applicable to them.
Unsampled POD restricts the trial state to the frozen POD span and minimises the full
central finite-difference residual norm; it does not first Galerkin-project that residual.
Whiskers show conditional reference-error intervals; lower endpoints at zero are omitted
on the logarithmic axis. They are numerical-reference uncertainty, not statistical intervals.

The descriptor includes the canonical initial-condition normalisation scalar. Cohort draw
and canonical descriptor construction are common offline work. All methods return the same
six fields on the fine scoring lattice; input preparation and output synthesis/decoding are
included in the descriptor-total timer. All errors and matches remain developmental and
conditional on the E0 reference uncertainty. Competitor searches cover a finite tested set.

''')
    component_text, phase_values = components(doc, table, save)
    text += comparisons(doc, summaries, table) + component_text + projection_geometry(doc, table, save)
    summary_text, findings = key_findings(doc, summaries, table)
    return dict(subjects=summaries, figures=figures, text=text, summary_text=summary_text, key_findings=findings,
                diagnostic_phase_seconds=phase_values,
                matching=doc['matching'], frontiers=doc['frontiers'], execution=doc['execution'])
