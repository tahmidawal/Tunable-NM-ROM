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


def render(doc, figdir, table, save, reference):
    required = ['schema_version', 'cohort', 'reference', 'contracts', 'subjects', 'case_results',
                'timing_repetitions', 'phase_repetitions', 'matching', 'frontiers', 'execution']
    assert all(k in doc for k in required) and doc['schema_version'] == 1
    cohort, ref = doc['cohort'], doc['reference']
    expected = reference['cohorts']['val64']
    assert cohort['name'] == 'val64' and cohort['seed'] == expected['seed']
    assert cohort['ordered_case_ids'] == list(range(64))
    assert cohort['canonical_table_sha256'] == expected['table_sha256']
    assert ref['dimension'] == 3 and ref['kind'] == 'ST' and ref['field'] == 'finest'
    assert ref['store_version'] == reference['store_version'] and ref['status_sha256'] == reference['status_sha256']
    assert np.allclose(ref['output_times'], reference['output_times'], rtol=0, atol=1e-15)
    rows = doc['timing_repetitions']
    # Derive the exact observed full population from per-invocation evidence, then check the contract.
    ids = sorted({int(r['case']) for r in rows})
    assert ids == list(range(64)), f'not the full declared validation population: {ids}'
    invalid_subjects = set()
    for row in rows:
        assert row['subject'] in doc['subjects']
        score = row.get('score') or {}
        cid = str(int(row['case']))
        target = reference['cohorts']['val64']['cases'][cid]
        finite = all(isinstance(score.get(k), (int, float)) and np.isfinite(score[k]) for k in ['e', 'epsilon'])
        if not finite or score.get('eligible') is not True:
            assert not any(timing_pass(row, c) for c in ['prepared_native', 'descriptor']), 'invalid result admitted for timing'
            invalid_subjects.add(row['subject'])
            continue
        assert np.isclose(score['epsilon'], target['eps_ref'], rtol=1e-10, atol=0)
        assert len(score['e_t']) == len(reference['output_times']) and np.isfinite(score['e_t']).all()
        assert abs(max(score['e_t'][1:]) - score['e']) <= 1e-12
        assert row['output_sha256']
    summaries, table_rows = {}, []
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['subject']].append(row)
    for subject_id, subject in doc['subjects'].items():
        own = groups.get(subject_id, [])
        score_by_case = {}
        for row in own:
            if row['subject'] in invalid_subjects:
                continue
            cid = int(row['case'])
            if cid in score_by_case:
                assert abs(score_by_case[cid] - row['score']['e']) <= 1e-12, (subject_id, cid)
            score_by_case[cid] = float(row['score']['e'])
        complete_scores = sorted(score_by_case) == ids and subject_id not in invalid_subjects
        q = subject['qualification']
        entry = dict(id=subject_id, method=method_group(subject_id, subject), qualification=q,
                     complete_score_population=complete_scores,
                     case_ids=sorted(score_by_case), costs={}, source_config=subject['config'])
        if complete_scores:
            errors = [score_by_case[c] for c in ids]
            entry.update(median_error=float(np.median(errors)), worst_error=max(errors),
                         errors_above_1pct=sum(e > .01 for e in errors))
        for contract, field in [('prepared_native', 'given_native_seconds'), ('descriptor', 'descriptor_seconds')]:
            good = [r for r in own if timing_pass(r, contract)]
            case_times = {c: [float(r[field]) for r in good if int(r['case']) == c] for c in ids}
            complete_timing = all(case_times[c] for c in ids)
            entry['costs'][contract] = dict(complete_population=complete_timing,
                admitted_repetitions=len(good), case_repetitions=case_times,
                median_seconds=float(np.median([np.median(case_times[c]) for c in ids])) if complete_timing else None)
        summaries[subject_id] = entry
        table_rows.append([subject_id, entry['method'], q['status'],
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
                qualified = str(q['status']).lower() in ['qualified', 'pass', 'accepted']
                common = explicit_pass(q['common_rollout_check'])
                label = group if group not in seen else None
                seen.add(group)
                xx = point['costs'][contract]['median_seconds'] * 1e3
                yy = point[metric] * 100
                ax.scatter(xx, yy, s=48, c=colours[group] if qualified else 'none',
                           edgecolors=colours[group], marker='o' if qualified or common else 's', label=label)
            ax.set(xscale='log', yscale='log', xlabel=f'{contract.replace("_", " ")} query: median over 64 cases (ms)',
                   ylabel=f'{metric.replace("_", " ").capitalize()} (%)')
            ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle('Same allocation · full validation · conditional E0 ST error\nFilled: registered qualified; open: other status retained', fontsize=10)
        name = f'full64_{contract}_frontier'
        save(fig, name)
        figures.append((contract, name))
    text = '''## Same-allocation full-validation frontier

The following dataset uses the complete validation population and scores the fields returned
by the same measured invocations. Query costs are case-balanced medians; timing rejection
cannot silently remove a difficult case. Missing timing coverage remains missing.
Descriptor-total is the primary cross-method contract. Prepared-native costs use different
method-specific supplied data, so those values are shown as decomposition evidence.

'''
    text += table(['Subject', 'Method', 'Qualification status', 'Median E0 error (%)', 'Worst E0 error (%)',
                   'Cases above 1%', 'Descriptor total (ms)', 'Prepared native (ms)'], table_rows) + '\n\n'
    for contract, name in figures:
        text += f'![Full validation {contract.replace("_", " ")} cost versus accuracy](figs/{name}.png)\n\n'
    text += ('''Common rollout-consistency checks and method-specific extra checks are separate masks.
The extra POD gradient threshold is not equivalent to the bank advection-rho threshold.
All preregistered verdicts remain unchanged. A POD point that passes the common checks but
fails its extra conservative gate remains visible as descriptive evidence; its exclusion
cannot be used to claim a bank advantage over tuned POD. Unqualified points are retained.

The descriptor includes the canonical initial-condition normalisation scalar. Cohort draw
and canonical descriptor construction are common offline work. All methods return the same
six fields on the fine scoring lattice; input preparation and output synthesis/decoding are
included in the descriptor-total timer. All errors and matches remain developmental and
conditional on the E0 reference uncertainty. Competitor searches cover a finite tested set.

''')
    return dict(subjects=summaries, figures=figures, text=text,
                matching=doc['matching'], frontiers=doc['frontiers'], execution=doc['execution'])
