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

from modcp_audit import audit_field_archive, digest, summarize_rows

ROOT = Path(__file__).resolve().parents[1]


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
    audits, seen = [], set()
    for row in data.get('invocations', []):
        artifact = row.get('field_artifact')
        if not artifact or row.get('field_artifact_kind') != 'self_contained_full_grid' or not row.get('finite'):
            continue
        full = (path.parent / artifact).resolve()
        # One saved deterministic result can serve several repetitions only when
        # the owner records that each actual repetition has the same field hash.
        key = str(full), tuple((name, row['errors'][name]) for name in
                              ('displacement', 'velocity', 'energy_state') if name in row['errors'])
        if key in seen:
            continue
        audits.append(audit_field_archive(full, data['case_name'], row['errors']))
        seen.add(key)
    return dict(source=relative(path), sha256=digest(path), status=data['status'],
                case_name=data['case_name'], provenance=data['provenance'], config=config,
                selections=data.get('selections', []), summaries=summaries,
                field_audits=audits, owner_artifacts=data.get('artifacts', {}))


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', type=Path, required=True)
    parser.add_argument('--date', required=True)
    args = parser.parse_args()
    sources = [summarize_input(path) for path in args.input]
    stem = ROOT/'reports'/f'{args.date}-modified-cp-eq-comparison'
    plots = frontier_plot(sources, stem)
    rows = []
    for source in sources:
        for row in source['summaries']:
            if row['split'] != 'evaluation':
                continue
            rows.append([row['case_name'], row['intervals'], row['method'],
                         f"{100*row['target']:g}%" if row['target'] is not None else 'diagnostic', row['configuration'],
                         f"{1e3*row['median_seconds']:.6g}" if row['median_seconds'] is not None else 'missing',
                         f"{100*row['worst_error']:.6g}%" if row['worst_error'] is not None else 'failed/nonfinite',
                         row['outlier_cases'], row['failed_cases'], row['nonstationary_cases'],
                         'yes' if row['qualified'] else 'no'])
        for selection in source['selections']:
            if selection.get('configuration') is None:
                rows.append([source['case_name'], selection['intervals'], selection['method'],
                             f"{100*selection['target']:g}%", 'No validation-qualified setting',
                             '—', '—', '—', '—', '—', 'no'])
    provenance_rows = [[s['case_name'], s['status'], s['provenance']['job_id'], s['provenance']['gpu'],
                        s['provenance']['commit'], len(s['field_audits'])] for s in sources]
    comparisons = []
    for source in sources:
        evaluated = [r for r in source['summaries'] if r['split'] == 'evaluation' and r['qualified']]
        for rom in [r for r in evaluated if r['method'] in ('cp', 'modcp', 'film')]:
            for fom in [r for r in evaluated if r['method'] not in ('cp', 'modcp', 'film')
                        and (r['intervals'], r['target']) == (rom['intervals'], rom['target'])]:
                comparisons.append([source['case_name'], rom['intervals'], f"{100*rom['target']:g}%",
                                    rom['method'], fom['method'],
                                    f"{fom['median_seconds']/rom['median_seconds']:.6g}×",
                                    rom['nonstationary_cases']])
    incomplete = any(s['status'] != 'complete' for s in sources)
    status = 'Provisional: one or more owner campaigns is incomplete.' if incomplete else (
        'Completed single-seed pilot; accuracy and speed claims are limited to the declared families and cohorts.')
    lines = ['# Modified CP with empirical quadrature: Burgers and waves', '',
             'This report compares the original CP decoder, latent-modulated CP factors, and a FiLM coordinate decoder. '+status,
             '', '## Evaluation of validation-selected configurations', '',
             'Queries start with full GPU-resident initial fields and return full GPU-resident output trajectories. '
             'Timing includes initialization, evolution, and reconstruction. Compilation, offline setup, and host transfers are excluded.', '',
             table(['Case', 'Intervals/axis', 'Method', 'Target', 'Configuration', 'Median query ms',
                    'Worst error', 'Outlier cases', 'Failed cases', 'Nonstationary cases', 'Target attained'], rows) if rows else
             'No validation-selected evaluation measurements are available yet.', '',
             'Worst error is the maximum over evaluation cases, stored times, and recorded repetitions; '
             'for waves it is also the maximum over displacement, velocity, and energy-state errors. '
             'All expected cases and repetitions must be present for a target to qualify. '
             'Failure counts retain numerical breakdowns and incomplete trajectories. '
             'Iteration-capped or small-step exits may attain a physical accuracy target, but are separately counted '
             'as nonstationary and never described as converged PDE solves.', '',
             'These errors compare against the declared numerical reference. Any unresolved reference uncertainty '
             'keeps the corresponding continuum-accuracy interpretation provisional. '
             'No configuration is chosen using evaluation accuracy or timing.', '',
             f'![Validation and evaluation error versus query time]({plots[0]})' if plots else '', '',
             '## Matched-accuracy full-solver comparisons', '',
             table(['Case', 'Intervals/axis', 'Target', 'ROM', 'Full solver', 'Median-time ratio FOM/ROM',
                    'ROM nonstationary cases'], comparisons) if comparisons else
             'No paired evaluation configurations currently qualify at a common declared target.', '',
             'Each ratio uses the same owner job and GPU and two validation-selected configurations '
             'that both attain the target on the untouched cohort. A ratio above unity means a smaller '
             'median ROM query time. Numerical completion and latent convergence remain separate.', '',
             '## Provenance and independent review', '',
             table(['Case', 'Campaign status', 'Job ID', 'GPU', 'Source commit', 'Full-field audits'], provenance_rows), '',
             'Raw repetition records, validation sweeps, selection declarations, source hashes, and field-audit results '
             f'are indexed in [{stem.name}.json]({stem.name}.json). '
             'Timing ratios must use the same job and GPU, and an FOM configuration meeting the same accuracy target. '
             'This report does not substitute timings from separate jobs.', '',
             'The new wave decoder represents displacement and velocity jointly. Its latent dimension is not '
             'the phase-state dimension of the earlier displacement-manifold experiments; changes relative to '
             'those earlier results do not isolate decoder architecture.', '',
             '## Glossary', '',
             '- **CP:** a sum of products of learned one-dimensional spatial factors.',
             '- **Modified CP:** CP factors with small nonlinear changes conditioned on the solved latent state.',
             '- **FiLM / INR:** feature-wise modulation of a neural coordinate-to-field decoder.',
             '- **EQ:** empirical quadrature, an offline-selected set of spatial samples and nonnegative integration weights.',
             '- **FOM:** the full-order numerical PDE solver used as a speed comparison.',
             '- **Weak residual:** the PDE mismatch integrated against smooth spatial test functions.',
             '- **Latent state:** the small vector of unknowns solved inside the decoder.',
             '- **Intervals/axis:** subdivisions of the unit domain; the number of stored nodes depends on boundary conditions.',
             '- **Validation-selected configuration:** solver and quadrature settings frozen before evaluation fields are examined.',
             '- **Target / target attained:** the declared error ceiling, and whether every expected invocation completes below it.',
             '- **Median query ms:** median across cases of each case\'s median recorded duration, in milliseconds.',
             '- **Worst error:** the largest fixed-initial-normalized error across the reported cases, times, state components, and repetitions.',
             '- **Outlier cases:** cases with any error above the target or invalid error values.',
             '- **Failed cases:** cases with any incomplete/nonfinite solve or missing trajectory.',
             '- **Nonstationary cases:** cases with any latent fit or time step lacking the declared convergence condition; accurate capped rollouts remain labeled.',
             '- **Median-time ratio FOM/ROM:** the full solver\'s median query duration divided by the ROM\'s, for paired qualifying configurations.',
             '- **Energy-state error:** the physical energy norm of the displacement/velocity error, scaled by the initial reference energy.',
             '- **Full-field audit:** independent NumPy recomputation from a saved full-grid prediction and reference.',
             '- **Campaign status / job ID / GPU / source commit:** completion state and identifiers of the recorded scientific execution.',
             '- **Single-seed pilot:** an initial comparison using one training random seed, without a training-variance claim.', '']
    stem.with_suffix('.json').write_text(json.dumps({'sources': sources}, indent=2, allow_nan=False)+'\n')
    stem.with_suffix('.md').write_text('\n'.join(lines))
    print(relative(stem.with_suffix('.md')))


if __name__ == '__main__':
    main()
