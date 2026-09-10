"""Generate the modified CP pilot report from owner handoff JSONs.

Usage: /home/tahmid/Dev/.venv/bin/python reports/generate_modcp_comparison.py
       --input PATH [--input PATH ...] --date YYYY-MM-DD

The date is the date numbers were finalized. Model/configuration selection must
already be frozen using validation. This generator never selects on evaluation.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from modcp_audit import audit_field_archive, digest, summarize_rows, load_field_archive
from modcp_freeze import verify_evaluation_freeze

ROOT = Path(__file__).resolve().parents[1]
FULL_GRID_KINDS = ('self_contained_full_grid', 'full_grid_with_shared_truth')


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join('---' for _ in headers)+' |',
                      *['| '+' | '.join(map(str, row))+' |' for row in rows]])


def relative(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def provenance_check(provenance):
    required = ('commit', 'job_id', 'gpu', 'backend', 'x64', 'matmul_precision')
    missing = [key for key in required if not provenance.get(key)]
    if missing:
        raise ValueError(f'Missing scientific provenance: {missing}')
    if provenance['backend'] != 'gpu' or provenance['x64'] is not True or provenance['matmul_precision'] != 'highest':
        raise ValueError('Wrong scientific backend or precision')


def summarize_input(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    provenance_check(data['provenance'])
    config = data['config']
    if config.get('smoke') or config.get('smoke_only') or data.get('phase') == 'development' or data['status'] == 'complete_smoke':
        raise ValueError('Smoke-test outputs are not scientific pilot measurements')
    references, reference_artifacts = list(data.get('references', [])), []
    if data['case_name'] == 'burgers2d':
        for split in ('validation', 'evaluation'):
            for n in config.get('meshes', []):
                artifact = path.parent/'references'/f'{split}_L{n}_uncertainty.json'
                if artifact.exists():
                    references.extend(dict(split=split, kind='nested_finer_reference', **r)
                                      for r in json.loads(artifact.read_text()))
                    reference_artifacts.append(dict(path=relative(artifact), sha256=digest(artifact)))
    groups = defaultdict(list)
    for row in data.get('invocations', []):
        if 'provenance' in row:
            for field in ('job_id', 'gpu'):
                if str(row['provenance'].get(field)) != str(data['provenance'][field]):
                    raise ValueError(f'Mixed-allocation timing panel: {field}')
        if row.get('finite') and data['case_name'] != 'burgers2d' and not all(
                name in (row.get('errors') or {}) for name in ('displacement', 'velocity', 'energy_state')):
            raise ValueError('Wave invocation omits a required physical error component')
        key = row['split'], row['method'], row['configuration'], row['intervals']
        groups[key].append(row)
    summaries = []
    for (split, method, setting, intervals), rows in groups.items():
        membership = config[f'{split}_case_ids']
        repetitions = config[f'{split}_repetitions'] if f'{split}_repetitions' in config else config['repetitions']
        if split == 'evaluation':
            declarations = [s for s in data['selections'] if
                            (s['method'], s['configuration'], s['intervals']) == (method, setting, intervals)]
            if not declarations:
                raise ValueError(f'Evaluation configuration lacks validation-frozen selection: {setting}')
        else:
            declarations = [{'target': target} for target in config['targets']]
        for selection in declarations:
            summary = summarize_rows(rows, membership, repetitions,
                                     selection['target'] if selection['target'] is not None else float('inf'))
            if selection['target'] is None:
                summary['qualified'] = summary['qualified_and_converged'] = False
            summaries.append(dict(case_name=data['case_name'], split=split, method=method,
                                  configuration=setting, intervals=intervals,
                                  target=selection['target'], **summary))
    if data['status'] == 'complete':
        declared = {(s['method'], s['configuration'], s['intervals']) for s in data['selections']
                    if s.get('configuration') is not None}
        observed = {(s['method'], s['configuration'], s['intervals']) for s in summaries
                    if s['split'] == 'evaluation' and s['complete_coverage']}
        if not declared or declared != observed:
            raise ValueError('Completed campaign is missing a selected evaluation configuration or invocation')
    evaluation_freeze = verify_evaluation_freeze(path, data)
    audits, seen, field_groups, reference_groups = [], {}, {}, {}
    for row in data.get('invocations', []):
        artifact = row.get('field_artifact')
        if not artifact or row.get('field_artifact_kind') not in FULL_GRID_KINDS or not row.get('finite'):
            continue
        full = (path.parent / artifact).resolve()
        reference = ((path.parent/row['reference_artifact']).resolve()
                     if row['field_artifact_kind'] == 'full_grid_with_shared_truth' else None)
        reference_hash = row.get('reference_sha256') if reference else None
        n = row['intervals']
        nt = len(config['output_times']) if data['case_name'] == 'burgers2d' else int(round(config['end_time']/config['observation_dt']))+1
        side = n-1 if data['case_name'] == 'wave_reflective' else n+1
        # One saved deterministic result can serve several repetitions only when
        # the owner records that each actual repetition has the same field hash.
        key = str(full), str(reference), reference_hash, data['case_name'], n, nt, side, tuple((name, row['errors'][name]) for name in
                              ('displacement', 'velocity', 'energy_state') if name in row['errors'])
        if key not in seen:
            seen[key] = audit_field_archive(full, data['case_name'], row['errors'],
                                           expected_shape=(nt, side, side), expected_intervals=n,
                                           reference_path=reference, reference_sha256=reference_hash)
            audits.append(seen[key])
        reference_key = row['split'], row['intervals'], row['case']
        reference_identity = seen[key]['reference_sha256'], seen[key]['reference_metadata']
        if reference_key in reference_groups and reference_groups[reference_key] != reference_identity:
            raise ValueError('Methods or repetitions use different reference fields for the same case')
        reference_groups[reference_key] = reference_identity
        field_groups[row['split'], row['method'], row['configuration'], row['intervals'], row['case'], row['rep']] = seen[key]
    usable_fields = dict(field_groups)
    paired_fields = 0
    for row in data.get('invocations', []):
        if row['split'] != 'evaluation' or not row.get('finite'):
            continue
        key = row['split'], row['method'], row['configuration'], row['intervals'], row['case'], row['rep']
        audit = field_groups.get(key, field_groups.get((*key[:-1], 0)))
        if audit is None:
            raise ValueError('Finite evaluation invocation lacks an independently auditable full field')
        recorded = {'u': row.get('field_sha256')} if data['case_name'] == 'burgers2d' else row.get('output_sha256')
        if recorded != audit['output_sha256']:
            raise ValueError('Timed invocation output hash differs from its audited field')
        for component, error in audit['errors'].items():
            if not np.isclose(row['errors'][component], error, rtol=1e-8, atol=1e-10):
                raise ValueError('Timed invocation error differs from its audited field')
        usable_fields[key] = audit
        paired_fields += 1
    for summary in summaries:
        repetitions = config.get(f"{summary['split']}_repetitions", config['repetitions'])
        keys = [(summary['split'], summary['method'], summary['configuration'], summary['intervals'], case, rep)
                for case in config[f"{summary['split']}_case_ids"] for rep in range(repetitions)]
        group_audits = [usable_fields.get(key) for key in keys]
        complete = all(audit is not None for audit in group_audits)
        for label, index in (('initial', 0), ('final', -1)):
            summary[f'worst_{label}_error'] = max(
                values[index] for audit in group_audits for values in audit['error_series'].values()) if complete else None
    return dict(source=relative(path), sha256=digest(path), status=data['status'],
                case_name=data['case_name'], provenance=data['provenance'], config=config,
                selections=data.get('selections', []), summaries=summaries,
                evaluation_freeze=evaluation_freeze,
                field_audits=audits, owner_artifacts=data.get('artifacts', {}),
                audited_paired_evaluation_invocations=paired_fields,
                references=references, reference_artifacts=reference_artifacts,
                eq_audits=data.get('eq_audits', []),
                full_weak_audits=data.get('full_weak_audits', []),
                representation_diagnostics=data.get('representation_diagnostics', []),
                checkpoint_hashes=data.get('checkpoint_hashes', data.get('checkpoint_sha256', {})))


def frontier_plot(sources, stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'cp': '#0072B2', 'modcp': '#D55E00', 'film': '#009E73'}
    panels = [(source, n) for source in sources for n in sorted({s['intervals'] for s in source['summaries']})]
    if not panels:
        return []
    fig, axes = plt.subplots(len(sources), 2, figsize=(12, 4*len(sources)), squeeze=False,
                             constrained_layout=True)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, (source, n) in zip(axes.flat, panels):
        ax.set_visible(True)
        unique = {(r['split'], r['method'], r['configuration']): r for r in source['summaries'] if r['intervals'] == n}
        methods = sorted({r['method'] for r in unique.values()})
        for method in methods:
            for split in ('validation', 'evaluation'):
                rows = [r for r in unique.values() if r['method'] == method and r['split'] == split
                        and r['median_seconds'] is not None and r['worst_error'] is not None]
                if not rows:
                    continue
                color = colors.get(method, '#666666')
                ax.scatter([1e3*r['median_seconds'] for r in rows], [max(1e-12, 100*r['worst_error']) for r in rows],
                           marker='o' if split == 'validation' else '*', color=color,
                           s=25 if split == 'validation' else 95, alpha=.6 if split == 'validation' else 1,
                           label=f'{method} {split}')
                failed = [r for r in rows if r['failed_cases'] or not r['complete_coverage']]
                if failed:
                    ax.scatter([1e3*r['median_seconds'] for r in failed], [max(1e-12, 100*r['worst_error']) for r in failed],
                               marker='x', color='black', s=50)
        for target in source['config']['targets']:
            ax.axhline(100*target, color='#999999', linestyle=':', linewidth=.8)
        ax.set(xscale='log', yscale='log', xlabel='Complete query ms', ylabel='Worst fixed-initial error %',
               title=f"{source['case_name']} — {n} intervals — job {source['provenance']['job_id']}")
        ax.grid(True, alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle('Validation sweep and frozen evaluation points; × marks incomplete/failed configurations')
    output = []
    for extension in ('png', 'pdf'):
        path = stem.with_name(stem.name+'-frontiers').with_suffix('.'+extension)
        fig.savefig(path, dpi=180, metadata={'CreationDate': None, 'ModDate': None} if extension == 'pdf' else None)
        output.append(path.name)
    plt.close(fig)
    return output


def reference_table(sources):
    rows = []
    def largest(records, key):
        values = [r[key] for r in records if r.get(key) is not None]
        return f'{max(values):.6g}' if values else '—'
    for source in sources:
        groups = defaultdict(list)
        for record in source['references']:
            groups[record['split'], record['intervals'], record['kind']].append(record)
        for (split, n, kind), records in sorted(groups.items()):
            for r in records:
                if 'temporal_refinement' in r:
                    r['temporal_difference'] = r['temporal_refinement']['maximum']
            balance_key = 'max_relative_energy_drift' if kind == 'exact_semidiscrete_sine' else 'max_relative_energy_balance'
            rows.append([source['case_name'], split, n, kind,
                         largest(records, 'temporal_difference'),
                         largest(records, 'nested_space_time_difference'),
                         largest(records, balance_key), largest(records, 'max_invariant_drift')])
    return table(['Case', 'Cohort', 'Intervals/axis', 'Reference', 'Worst temporal difference',
                  'Worst nested space/time difference', 'Worst energy balance defect',
                  'Worst invariant drift'], rows) if rows else 'Reference diagnostics have not been collected yet.'


def representative_plots(paths, stem):
    """Fixed evaluation case zero; use only validation-selected configurations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = []
    for source_path in paths:
        source_path = Path(source_path).resolve()
        data = json.loads(source_path.read_text())
        rows = [r for r in data['invocations'] if r['split'] == 'evaluation' and r['case'] == 0
                and r['rep'] == 0 and r['method'] in ('cp', 'modcp', 'film')]
        if not rows:
            continue
        n = max(data['config']['meshes'])
        chosen = []
        for method in ('cp', 'modcp', 'film'):
            declared = [s for s in data['selections'] if s['method'] == method and s['intervals'] == n
                        and s.get('configuration') is not None]
            # Prefer the predeclared best-validation-accuracy diagnostic, else
            # the tightest validation target. Evaluation values never rank it.
            declared.sort(key=lambda s: -1 if s.get('target') is None else s['target'])
            row = next((r for r in rows if r['method'] == method and r['intervals'] == n
                        and declared and r['configuration'] == declared[0]['configuration']), None)
            chosen.append((method, row))
        fields, truth = [], None
        for method, row in chosen:
            if row is None or row.get('field_artifact_kind') not in FULL_GRID_KINDS:
                fields.append((method+'\nmissing full field', None))
                continue
            reference_path = source_path.parent/row['reference_artifact'] if row['field_artifact_kind'] == 'full_grid_with_shared_truth' else None
            archive = load_field_archive(source_path.parent/row['field_artifact'], reference_path, row.get('reference_sha256'))
            u, reference = archive['u'], archive['truth_u']
            if truth is not None and not np.array_equal(truth, reference):
                raise ValueError('Representative fields do not share the same reference')
            truth = reference
            if not np.isfinite(u).all():
                fields.append((method+'\nnonfinite rollout', None))
            else:
                fields.append((method+('' if row.get('completed') else '\nsolver failure'), u))
        if truth is None:
            # No reference can be drawn from an incomplete source. Never fill
            # the gap with an invented field or a smaller surviving mesh.
            continue
        indices = np.unique(np.linspace(0, len(truth)-1, min(4, len(truth)), dtype=int))
        fields = [('reference', truth), *fields]
        bound = max(float(np.max(np.abs(field[indices]))) for _, field in fields if field is not None)
        bound = max(bound, np.finfo(float).tiny)
        fig, axes = plt.subplots(len(indices), len(fields), squeeze=False,
                                 figsize=(3*len(fields), 2.5*len(indices)), constrained_layout=True)
        dt = data['config'].get('observation_dt', .05)
        for i, frame in enumerate(indices):
            for j, (method, field) in enumerate(fields):
                ax = axes[i, j]
                if field is None:
                    ax.text(.5, .5, method, ha='center', va='center', transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
                plot = ax.imshow(field[frame].T, origin='lower', extent=(0, 1, 0, 1),
                                 vmin=-bound, vmax=bound, cmap='RdBu_r', interpolation='nearest')
                ax.set_title(f'{method}, t={frame*dt:g}', fontsize=9)
                ax.set(xlabel='x', ylabel='y')
        fig.colorbar(plot, ax=axes.ravel().tolist(), shrink=.7, label='Displacement / scalar field')
        fig.suptitle(f"{data['case_name']} — evaluation case zero — {n} intervals\n"
                     'Validation-selected configurations; common field scale includes all predictions')
        for extension in ('png', 'pdf'):
            path = stem.with_name(stem.name+'-'+data['case_name']+'-fields').with_suffix('.'+extension)
            fig.savefig(path, dpi=160, metadata={'CreationDate': None, 'ModDate': None} if extension == 'pdf' else None)
            if extension == 'png':
                output.append((data['case_name'], path.name))
        plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', type=Path, required=True)
    parser.add_argument('--date', required=True)
    parser.add_argument('--training-audit', type=Path, default=ROOT/'reports'/'2026-09-10-modified-cp-training-audit.json')
    args = parser.parse_args()
    sources = [summarize_input(path) for path in args.input]
    seal_hashes = {source['evaluation_freeze']['sha256'] for source in sources if source['evaluation_freeze']}
    if len(seal_hashes) > 1:
        raise ValueError('Evaluation panels do not share the same global validation freeze')
    training = json.loads(args.training_audit.read_text())
    training_rows = []
    for checkpoint in training['checkpoints']:
        cfg = checkpoint['configuration']
        if digest(ROOT/checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Training checkpoint changed after its independent audit')
        for source in sources:
            if source['case_name'] == checkpoint['case_name'] and source['checkpoint_hashes'].get(cfg['architecture']) != checkpoint['sha256']:
                raise ValueError('Campaign decoder differs from the independently audited training checkpoint')
        training_rows.append([checkpoint['case_name'], cfg['architecture'], cfg['intervals'], cfg['k'],
                              cfg['rank'] if cfg['architecture'] != 'film' else '—',
                              checkpoint['parameter_count'], checkpoint['completed_updates']])
    diagnosis_path = ROOT/'worktrees'/'2026-09-10-modcp-burgers2d'/'experiments'/'modcp-eq'/'runs'/'diagnose01'/'out'/'diagnosis.json'
    span_path = ROOT/'reports'/'2026-09-10-modified-cp-span-audit.json'
    diagnosis, span = json.loads(diagnosis_path.read_text()), json.loads(span_path.read_text())
    provenance_check(diagnosis['provenance'])
    if not diagnosis['validation_only'] or not diagnosis['accuracy_only'] or diagnosis['checkpoints_modified'] or diagnosis['query_algorithms_modified']:
        raise ValueError('Unexpected scope of validation-only reconstruction diagnostic')
    diagnostic_rows = []
    for row in diagnosis['rows']:
        diagnostic_rows.append([row['case'], row['method'], row['intervals'],
                                *[f'{100*value:.6g}%' for value in (
                                    row['recorded_query_fine_error'][0], row['snapshots'][0]['best_oracle_error'],
                                    row['snapshots'][-1]['best_oracle_error'], row['recorded_query_fine_error'][-1])],
                                f"{100*row['unconstrained_CP_spatial_span']['errors'][0]:.6g}%"
                                if 'unconstrained_CP_spatial_span' in row else 'not established'])
    stem = ROOT/'reports'/f'{args.date}-modified-cp-eq-comparison'
    plots = frontier_plot(sources, stem)
    field_plots = representative_plots(args.input, stem)
    rows, missing_selection_keys = [], set()
    for source in sources:
        for row in source['summaries']:
            if row['split'] != 'evaluation':
                continue
            rows.append([row['case_name'], row['intervals'], row['method'],
                         f"{100*row['target']:g}%" if row['target'] is not None else 'diagnostic', row['configuration'],
                         f"{1e3*row['median_seconds']:.6g}" if row['median_seconds'] is not None else 'missing',
                         f"{100*row['worst_error']:.6g}%" if row['worst_error'] is not None else 'failed/nonfinite',
                         row['outlier_cases'] if row['target'] is not None else '—', row['failed_cases'],
                         row['nonstationary_cases'] if row['method'] in ('cp', 'modcp', 'film') else '—',
                         row['timing_outlier_invocations'],
                         'yes' if row['qualified'] else 'no'])
        for selection in source['selections']:
            if selection.get('configuration') is None:
                key = source['case_name'], selection['intervals'], selection['method'], selection['target']
                if key in missing_selection_keys:
                    continue
                missing_selection_keys.add(key)
                rows.append([source['case_name'], selection['intervals'], selection['method'],
                             f"{100*selection['target']:g}%", 'No validation-qualified setting',
                             '—', '—', '—', '—', '—', '—', 'no'])
    provenance_rows = [[s['case_name'], s['status'], s['provenance']['job_id'], s['provenance']['gpu'],
                        s['provenance']['commit'], len(s['field_audits'])] for s in sources]
    comparisons = []
    phase_errors, wave_components = [], []
    for source in sources:
        seen_phase = set()
        for row in source['summaries']:
            key = row['method'], row['configuration'], row['intervals']
            if row['split'] != 'evaluation' or key in seen_phase:
                continue
            seen_phase.add(key)
            phase_errors.append([source['case_name'], row['intervals'], row['method'], row['configuration'],
                                 *[f"{100*row[name]:.6g}%" if row.get(name) is not None else 'incomplete/nonfinite'
                                   for name in ('worst_initial_error', 'worst_final_error', 'worst_error')]])
            if source['case_name'] != 'burgers2d':
                wave_components.append([source['case_name'], row['intervals'], row['method'], row['configuration'],
                                        *[f"{100*row['worst_error_by_component'][name]:.6g}%"
                                          if row['worst_error_by_component'].get(name) is not None else 'incomplete/nonfinite'
                                          for name in ('displacement', 'velocity', 'energy_state')]])
        evaluated = [r for r in source['summaries'] if r['split'] == 'evaluation' and r['qualified']]
        for rom in [r for r in evaluated if r['method'] in ('cp', 'modcp', 'film')]:
            for fom in [r for r in evaluated if r['method'] not in ('cp', 'modcp', 'film')
                        and (r['intervals'], r['target']) == (rom['intervals'], rom['target'])]:
                comparisons.append([source['case_name'], rom['intervals'], f"{100*rom['target']:g}%",
                                    rom['method'], fom['method'],
                                    f"{fom['median_seconds']/rom['median_seconds']:.6g}×",
                                    rom['nonstationary_cases']])
    evaluated_cases = {s['case_name'] for s in sources if s['status'] == 'complete'
                       and any(r['split'] == 'evaluation' for r in s['summaries'])}
    expected_cases = {'burgers2d', 'wave_reflective', 'wave_absorbing'}
    incomplete = not expected_cases.issubset(evaluated_cases) or any(s['status'] not in ('complete', 'validation_frozen') or
                     s['case_name'] not in evaluated_cases for s in sources)
    status = 'Provisional: one or more owner campaigns is incomplete.' if incomplete else (
        'Completed single-seed pilot; accuracy and speed claims are limited to the declared families and cohorts.')
    lines = ['# Modified CP with empirical quadrature: Burgers and waves', '',
             'This report compares the original CP decoder, latent-modulated CP factors, and a FiLM coordinate decoder. '+status,
             '', '## Evaluation of validation-selected configurations', '',
             'Queries start with full GPU-resident initial fields and return full GPU-resident output trajectories. '
             'Timing includes initialization, evolution, and reconstruction. Compilation, offline setup, and host transfers are excluded.', '',
             table(['Case', 'Intervals/axis', 'Method', 'Target', 'Configuration', 'Median query ms',
                    'Worst error', 'Outlier cases', 'Failed cases', 'Nonstationary cases',
                    'Timing outliers', 'Target attained'], rows) if rows else
             'No validation-selected evaluation measurements are available yet.', '',
             'Worst error is the maximum over evaluation cases, stored times, and recorded repetitions; '
             'for waves it is also the maximum over displacement, velocity, and energy-state errors. '
             'All expected cases and repetitions must be present for a target to qualify. '
             'Failure counts retain numerical breakdowns and incomplete trajectories. '
             'Iteration-capped or small-step exits may attain a physical accuracy target, but are separately counted '
             'as nonstationary and never described as converged PDE solves. Stationarity concerns the weak '
             'least-squares objective; physical accuracy and full-residual diagnostics are assessed separately.', '',
             'These errors compare against the declared numerical reference. Any unresolved reference uncertainty '
             'keeps the corresponding continuum-accuracy interpretation provisional. '
             'No configuration is chosen using evaluation accuracy or timing.', '',
             f'![Validation and evaluation error versus query time]({plots[0]})' if plots else '', '',
             'Configurations with nonfinite errors have no finite position on the logarithmic axes; '
             'their failures remain in the summary tables and raw records.', '',
             '## Initial fitting and subsequent evolution', '',
             table(['Case', 'Intervals/axis', 'Method', 'Configuration', 'Worst initial error',
                    'Worst final error', 'Worst trajectory error'], phase_errors) if phase_errors else
             'Independent evaluation field decompositions are not available yet.', '',
             'Each column takes its own maximum over the complete evaluation cohort and physical components, '
             'so the maximizing case may differ between columns. Every time uses the same initial-reference '
             'normalization. These are measured field discrepancies; local snapshot-fitting diagnostics do '
             'not establish a mathematical best-approximation floor.', '',
             table(['Wave case', 'Intervals/axis', 'Method', 'Configuration', 'Worst displacement error',
                    'Worst velocity error', 'Worst energy-state error'], wave_components) if wave_components else '', '',
             '## Matched-accuracy full-solver comparisons', '',
             table(['Case', 'Intervals/axis', 'Target', 'ROM', 'Full solver', 'Median-time ratio FOM/ROM',
                    'ROM nonstationary cases'], comparisons) if comparisons else
             'No paired evaluation configurations currently qualify at a common declared target.', '',
             'Each ratio uses the same owner job and GPU and two validation-selected configurations '
             'that both attain the target on the untouched cohort. A ratio above unity means a smaller '
             'median ROM query time. Numerical completion and latent convergence remain separate.', '',
             '## Trained models and online work', '',
             table(['Case', 'Decoder', 'Training intervals/axis', 'Latent dimension', 'CP rank',
                    'Decoder parameters', 'Completed training updates'], training_rows), '',
             'Each PDE/boundary has separately trained weights, frozen for both evaluation meshes. '
             'CP and modified CP share the same initial CP training stage. FiLM uses the full update budget '
             'from its own initialization. Training update budgets match; parameter counts and training costs differ. '
             'This pilot compares the declared architectures without a parameter-matched or exhaustive tuning claim.', '',
             'For each output component, the CP family represents', '',
             r'$$\widetilde u(z;x,y)=m(x,y)\left[\beta+\sum_{r=1}^{R}c_r(z)a_r(x;z)b_r(y;z)\right].$$', '',
             r'Here $z$ is the solved latent state, $R$ is the CP rank, $c_r$ is the nonlinear coefficient head '
             r'with a linear skip, and $m$ enforces the boundary. Original CP uses factors independent of $z$. '
             'Modified CP adds a latent-conditioned nonlinear correction to each one-dimensional factor; '
             'its zero correction exactly recovers the shared CP initialization. FiLM instead conditions a '
             'two-dimensional coordinate network on the latent state and includes its own linear latent skip. '
             'Coordinate-only features are cached offline for all three decoders. Dirichlet mesh transfer '
             'uses a boundary strip that preserves training-node values and avoids interpolating untrained '
             'masked endpoint parameters into new interior nodes.', '',
             'Each reduced implicit step minimizes the scaled weak discrete residual,', '',
             r'$$z_{n+1}\approx\operatorname*{arg\,min}_z\frac12\left\|W_{\mathrm{EQ}}'
             r'\,\mathcal R_n(\widetilde u(z);\widetilde u(z_n),\mu)\right\|_2^2.$$', '',
             r'$\mathcal R_n$ is the fully discrete PDE residual, $\mu$ contains physical parameters, '
             r'and $W_{\mathrm{EQ}}$ applies the smooth test functions, fitted quadrature, and fixed scaling. '
             'The weak residual and its latent Jacobian use JAX automatic differentiation; damped '
             'Gauss–Newton solves small dense systems and warm-starts each time step from the preceding latent state. '
             'The initial latent fit uses the supplied field at sampled locations and stored starting codes. '
             'The full-solver CG comparison belongs to the SPD wave discretization; nonlinear Burgers uses '
             'Newton–BiCGStab.', '',
             'CP precontracts its fixed spatial factors with the selected EQ weights for weak linear terms. '
             'The quadrature approximation is preserved. Burgers still evaluates the nonlinear upwind term '
             'on its sampled stencil. Modified CP and FiLM retain state-dependent spatial evaluation. '
             'On waves, CP\'s contracted evolution dimensions stay fixed when EQ count changes; EQ count still '
             'affects approximation and sampled initialization. Solver iteration limits, tolerance, and time step '
             'continue to change work. Full-field output cost grows with the requested mesh for every decoder. '
             'This implementation uses EQ-fitted operators, and does not establish an exact quadrature-free operator claim.', '',
             'Burgers validation preceded the equivalent CP mass contraction and GPU scalar preloading. '
             'Its final evaluation keeps those validation-selected settings and uses the optimized runner, '
             'whose numerical parity was checked separately. The final timing ratios come entirely from '
             'that evaluation allocation. The older validation timings do not establish the fastest '
             'configuration for the optimized implementation.', '',
             '## Diagnosis of the Burgers validation failure', '',
             table(['Validation case', 'Decoder', 'Intervals/axis', 'Online initial error',
                    'Best tested initial fit', 'Best tested final-snapshot fit', 'Online final error',
                    'CP affine-image initial floor'], diagnostic_rows), '',
             f"For validation case {span['case']}, independent full-grid least squares over the "
             f"{span['rank']} learned CP spatial products gives an initial error floor of "
             f"{100*span['relative_projection_errors'][0]:.6g}%. Its factor matrix has condition number "
             f"{span['condition_number']:.6g}. This limits this frozen checkpoint even with freely chosen "
             'spatial coefficients; changing only quadrature or nonlinear iteration settings cannot overcome it. '
             'The same conclusion is not established for modified CP or FiLM. Their tested stationary '
             'snapshot fits are achieved reconstruction errors, which can exceed the unknown global minimum. '
             'Later-time snapshot fitting uses the reference solution and is excluded from online timings, '
             'initialization, and configuration selection. These cases were chosen to diagnose validation failures.', '',
             '![Independent CP reconstruction diagnostic](2026-09-10-modified-cp-span-audit.png)', '',
             '## Provenance and independent review', '',
             table(['Case', 'Campaign status', 'Job ID', 'GPU', 'Source commit', 'Full-field audits'], provenance_rows), '',
             'Raw repetition records, validation sweeps, selection declarations, source hashes, and field-audit results '
             f'are indexed in [{stem.name}.json]({stem.name}.json). '
             'Timing ratios must use the same job and GPU, and an FOM configuration meeting the same accuracy target. '
             'This report does not substitute timings from separate jobs. '
             'Every evaluation panel is bound to the same saved global validation freeze, including its '
             'unchanged cohort seed, checkpoint identities, selected settings, and quadrature file hashes. '
             'Reference arrays must agree across every method and repetition for each case.', '',
             'The [source audit](2026-09-10-modified-cp-source-audit.json) compares collected code with immutable Git objects. '
             'The [raw archive manifest](2026-09-10-modified-cp-raw-artifacts.json) identifies retained field archives '
             'and explains checksum verification and extraction. Large raw fields are retained outside Git history.', '',
             'Wave validation retains full-grid metrics and output hashes but only bounded observation fields. '
             'Its audit checks source, reference operators, selection records, and saved observations; '
             'the unsaved full-grid validation errors cannot be independently recomputed from those observations. '
             'Every finite final evaluation invocation is instead checked against full-grid fields, with '
             'independently recomputed errors and matching output hashes.', '',
             '## Numerical reference checks', '', reference_table(sources), '',
             r'The Burgers system is $\partial_t u+u(\partial_xu+\partial_yu)=\nu\Delta u$ on the unit square '
             'with homogeneous Dirichlet boundaries and localized Gaussian initial fields. '
             'The full solver uses backward Euler, sign-dependent upwinding, and Newton–BiCGStab. '
             r'The wave system is $\partial_t u=v$, $\partial_t v=c^2\Delta u$, with reflective $u=0$ '
             r'or absorbing $\partial_t u+c\partial_nu=0$ boundaries and the fresh localized Gaussian-core family. '
             'Its implicit comparison uses a symmetric positive-definite Crank–Nicolson elimination solved by CG. '
             'Direct and explicit wave controls are reported separately. Exact parameter generators and recorded '
             'cohort parameters remain in the source artifacts indexed above.', '',
             'Burgers is scored against a finer-grid trajectory restricted to the output grid; '
             'its nested difference also contains spatial discretization error. Wave errors use '
             'the same-grid discrete system: reflective propagation is exact for that system, '
             'and absorbing references have temporal refinement and energy/boundary balance checks. '
             'These checks do not establish a rigorous continuum error bound. Differences and energy '
             'balance defects are dimensionless; invariant drift is an absolute signed-moment magnitude.', '',
             'The new wave decoder represents displacement and velocity jointly. Its latent dimension is not '
             'the phase-state dimension of the earlier displacement-manifold experiments; changes relative to '
             'those earlier results do not isolate decoder architecture.', '',
             *[f'![{name}: reference and decoder fields]({path})\n' for name, path in field_plots],
             '## Glossary', '',
             '- **CP:** a sum of products of learned one-dimensional spatial factors.',
             '- **Modified CP:** CP factors with small nonlinear changes conditioned on the solved latent state.',
             '- **FiLM / INR:** feature-wise modulation of a neural coordinate-to-field decoder.',
             '- **EQ:** empirical quadrature, an offline-selected set of spatial samples and nonnegative integration weights.',
             '- **FOM:** the full-order numerical PDE solver used as a speed comparison.',
             '- **Weak residual:** the PDE mismatch integrated against smooth spatial test functions.',
             '- **Latent state:** the small vector of unknowns solved inside the decoder.',
             '- **CP rank:** the number of spatial product terms, separate from the latent dimension.',
             '- **Linear skip:** a direct linear dependence on latent coordinates added to the nonlinear decoder head.',
             '- **FiLM conditioning:** latent-dependent scales and shifts applied to hidden coordinate features.',
             '- **Jacobian / automatic differentiation:** respectively derivatives with respect to latent coordinates and their computation by differentiating the implemented numerical operations.',
             '- **Gauss–Newton / damping:** local least-squares iteration using that Jacobian, with regularization and acceptance checks to control the step.',
             '- **CG / SPD:** conjugate gradients and the symmetric positive-definite matrix property required by that solver.',
             '- **Newton–BiCGStab:** nonlinear Newton iteration with a Krylov linear solver that allows nonsymmetric systems.',
             '- **Crank–Nicolson / backward Euler:** implicit time-stepping rules used here for waves and Burgers respectively.',
             '- **Decoder parameters:** trained weights in the field decoder, excluding training-only snapshot codes.',
             '- **Training updates:** optimizer steps completed before validation and evaluation.',
             '- **Affine spatial image:** the fixed CP bias plus every linear combination of its learned spatial products.',
             '- **Reconstruction floor:** the smallest error in the specified fixed linear/affine space, giving a lower bound for a decoder restricted to that space.',
             '- **Snapshot fit:** a local latent optimization against one known reference field; its achieved error is not a proof of the best possible decoder error.',
             '- **Intervals/axis:** subdivisions of the unit domain; the number of stored nodes depends on boundary conditions.',
             '- **Validation-selected configuration:** solver and quadrature settings frozen before evaluation fields are examined.',
             '- **Target / target attained:** the declared error ceiling, and whether every expected invocation completes below it.',
             '- **Median query ms:** median across cases of each case\'s median recorded duration, in milliseconds.',
             '- **Worst error:** the largest fixed-initial-normalized error across the reported cases, times, state components, and repetitions.',
             '- **Outlier cases:** cases with any error above the target or invalid error values.',
             '- **Failed cases:** cases with any incomplete/nonfinite solve or missing trajectory.',
             '- **Nonstationary cases:** ROM cases with any latent fit or weak time step lacking the declared gradient condition; accurate capped rollouts remain labeled. Classical methods show a dash because they use their own completion checks.',
             '- **Timing outliers:** invocations taking more than twice their own case\'s repetition median; retained in all summaries.',
             '- **Temporal difference:** discrepancy after refining the reference time step, on fixed initial physical scales.',
             '- **Nested space/time difference:** discrepancy against a finer spatial grid and time step, restricted back to the reported grid.',
             '- **Energy balance defect:** relative failure of reference energy conservation, or energy plus outgoing boundary flux conservation.',
             '- **Invariant drift:** change in the absorbing reference\'s area integral of velocity plus speed times its boundary integral of displacement.',
             '- **Semidiscrete / continuum:** respectively the spatially discretized PDE and the original PDE before spatial discretization.',
             '- **Median-time ratio FOM/ROM:** the full solver\'s median query duration divided by the ROM\'s, for paired qualifying configurations.',
             '- **Energy-state error:** the physical energy norm of the displacement/velocity error, scaled by the initial reference energy.',
             '- **Full-field audit:** independent NumPy recomputation from a saved full-grid prediction and reference.',
             '- **Campaign status / job ID / GPU / source commit:** completion state and identifiers of the recorded scientific execution.',
             '- **Single-seed pilot:** an initial comparison using one training random seed, without a training-variance claim.', '']
    stem.with_suffix('.json').write_text(json.dumps({'sources': sources,
        'validation_diagnosis': {'source': relative(diagnosis_path), 'sha256': digest(diagnosis_path), 'data': diagnosis,
                                'independent_span_audit': relative(span_path), 'independent_span_sha256': digest(span_path)},
        'training_audit': {'path': relative(args.training_audit), 'sha256': digest(args.training_audit),
                           'checkpoints': training['checkpoints']}}, indent=2, allow_nan=False)+'\n')
    stem.with_suffix('.md').write_text('\n'.join(lines))
    print(relative(stem.with_suffix('.md')))


if __name__ == '__main__':
    main()
