"""Generate the frozen-checkpoint mesh-ladder report, its JSON and its figure.

Every number in the report comes from this script reading the drivers' result
JSONs. Nothing is typed by hand. Run it after collecting both attempts:

    /home/tahmid/Dev/.venv/bin/python experiments/mesh-ladder/reports/generate_ladder.py
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CELL = HERE.parent
ROOT = CELL.parents[1]

# Validated categorical slots (blue, orange, aqua, violet): all-pairs CVD dE 9.2,
# normal-vision dE 16.3 on the light surface. Aqua sits below 3:1 contrast, so
# every series also carries a direct label and a table row.
SERIES = {
    'cached': ('#2a78d6', 'o', 'ROM cached reduced solve'),
    'complete': ('#eb6834', 's', 'ROM complete device query'),
    'setup': ('#1baf7a', '^', 'ROM offline per-mesh setup'),
    'fom': ('#4a3aa7', 'D', 'Efficient FOM device query'),
}
SURFACE, INK, MUTED = '#fcfcfb', '#0b0b0b', '#52514e'
TARGET = 0.05


def summarize(values):
    values = [float(v) for v in values]
    array = np.asarray(values)
    q1, q3 = np.percentile(array, [25, 75])
    fence = q3 + 1.5 * (q3 - q1)
    return dict(count=len(values), median=float(np.median(array)),
                minimum=float(array.min()), maximum=float(array.max()),
                upper_tukey_fence=float(fence), upper_outliers=int((array > fence).sum()),
                repetitions=values)


def scalar_error(value):
    """Both drivers' error fields, reduced to one number per invocation."""
    if value is None:
        return None
    if isinstance(value, dict):
        return float(value['fixed_initial_max'])
    return float(value)


def load(attempt):
    for candidate in (CELL / 'artifacts' / attempt / 'result.json',
                      CELL / 'runs' / attempt / 'out' / 'result.json'):
        if candidate.exists():
            return json.loads(candidate.read_text()), candidate
    raise SystemExit(f'no result.json for attempt {attempt}')


def aggregate(result):
    """Pooled statistics per (mesh, subject), plus medians of per-case medians."""
    meshes = sorted({row['intervals'] for row in result['invocations']})
    device = defaultdict(list)
    host = defaultdict(list)
    per_case = defaultdict(lambda: defaultdict(list))
    common = defaultdict(list)
    requested = defaultdict(list)
    same_grid = defaultdict(list)
    iterations = defaultdict(list)
    stopping = defaultdict(lambda: defaultdict(int))
    for row in result['invocations']:
        key = (row['intervals'], row['name'])
        device[key].append(row['complete_device_seconds'])
        host[key].append(row['host_to_host_seconds'])
        per_case[key][row['case']].append(row['complete_device_seconds'])
        common[key].append(scalar_error(row.get('physical_error_common_grid')))
        requested[key].append(scalar_error(row.get('physical_error_requested_grid')))
        if row.get('same_grid_discrepancy_requested') is not None:
            same_grid[key].append(scalar_error(row['same_grid_discrepancy_requested']))
        if isinstance(row.get('iterations'), list):
            iterations[key].append(int(np.sum(row['iterations'])))
        elif row.get('iterations') is not None:
            iterations[key].append(int(row['iterations']))
        if row.get('stationary') is not None:
            stopping[key]['stationary' if row['stationary'] else 'not_stationary'] += 1
        if row.get('nonlinear_tolerance_satisfied') is not None:
            stopping[key]['tolerance_met' if row['nonlinear_tolerance_satisfied']
                          else 'tolerance_missed'] += 1
        if row.get('cg_converged') is not None:
            stopping[key]['converged' if row['cg_converged'] else 'not_converged'] += 1
        if row.get('reason') is not None and row['method'] == 'rom':
            stopping[key][f'exit_reason_{row["reason"]}'] += 1

    cached = defaultdict(list)
    cached_iterations = defaultdict(list)
    cached_stationary = defaultdict(lambda: defaultdict(int))
    for row in result['cached_invocations']:
        cached[row['intervals']].append(row['cached_reduced_seconds'])
        cached_iterations[row['intervals']].append(
            int(np.sum(row['iterations'])) if isinstance(row.get('iterations'), list)
            else int(row.get('attempts', 0)))
        if row.get('stationary') is not None:
            cached_stationary[row['intervals']]['stationary' if row['stationary']
                                                else 'not_stationary'] += 1

    setup = {}
    for row in result['setup'] if 'setup' in result else result['mesh_setup']:
        setup[row['intervals']] = row

    subjects = sorted({row['name'] for row in result['invocations']})
    table = {}
    for mesh in meshes:
        for name in subjects:
            key = (mesh, name)
            if key not in device:
                continue
            case_medians = [statistics.median(v) for v in per_case[key].values()]
            table[f'{mesh}|{name}'] = dict(
                intervals=mesh, subject=name,
                device_seconds=summarize(device[key]),
                host_to_host_seconds=summarize(host[key]),
                median_of_case_medians=float(np.median(case_medians)),
                worst_error_common_grid=max(e for e in common[key] if e is not None),
                median_error_common_grid=float(np.median([e for e in common[key] if e is not None])),
                worst_error_requested_grid=max(e for e in requested[key] if e is not None),
                median_error_requested_grid=float(np.median([e for e in requested[key] if e is not None])),
                worst_same_grid_discrepancy=(max(same_grid[key]) if same_grid[key] else None),
                median_total_iterations=(float(np.median(iterations[key])) if iterations[key] else None),
                stopping=dict(stopping[key]),
                meets_target=bool(max(e for e in requested[key] if e is not None) <= TARGET))
    return dict(meshes=meshes, subjects=subjects, table=table,
                cached={str(m): dict(seconds=summarize(cached[m]),
                                     median_iterations=float(np.median(cached_iterations[m])),
                                     stopping=dict(cached_stationary[m])) for m in meshes},
                setup={str(m): setup[m] for m in meshes})


def rom_name(result):
    return next(row['name'] for row in result['invocations'] if row['method'] == 'rom')


def poisson_same_grid(result, summary, attempt):
    """ROM against the exact same-grid full-order solution, on the common grid.

    The direct sine transform diagonalises this discrete operator exactly, so the
    `dst` arm's own output *is* the converged same-grid full-order field. Taking
    the discrepancy from the archived observation fields keeps it a measured
    quantity rather than a second solve. It is omitted when the unpacked run
    directory is not beside the report.
    """
    fields = CELL / 'runs' / attempt / 'out' / 'fields'
    if not fields.is_dir():
        return False
    direct = {(row['intervals'], row['case']): row['field_sha256']
              for row in result['invocations'] if row['name'] == 'dst'}
    cache = {}

    def load_field(digest):
        if digest not in cache:
            cache[digest] = np.load(fields / f'{digest}.npz')['observation_field']
        return cache[digest]

    discrepancy = defaultdict(list)
    for row in result['invocations']:
        if row['method'] != 'rom':
            continue
        key = (row['intervals'], row['case'])
        if key not in direct:
            return False
        reduced, exact = load_field(row['field_sha256']), load_field(direct[key])
        discrepancy[row['intervals']].append(
            float(np.linalg.norm(reduced - exact) / np.linalg.norm(exact)))
    for mesh, values in discrepancy.items():
        entry = summary['table'][f'{mesh}|{summary["rom"]}']
        entry['worst_same_grid_discrepancy'] = max(values)
        entry['median_same_grid_discrepancy'] = float(np.median(values))
        entry['same_grid_note'] = ('against the exact same-grid direct-transform solution, '
                                   'measured on the common observation grid')
    return True


def efficient_fom(summary, mesh):
    """Cheapest full-order arm meeting the target; otherwise the cheapest arm."""
    arms = [v for k, v in summary['table'].items()
            if v['intervals'] == mesh and not k.endswith('|' + summary['rom'])]
    passing = [a for a in arms if a['meets_target']]
    pool = passing or arms
    best = min(pool, key=lambda a: a['device_seconds']['median'])
    return best, bool(passing)


def setup_total(row):
    return row.get('total_offline_setup_seconds')


def figure(summaries, out_png, out_pdf):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(summaries), figsize=(12.4, 5.4), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for axis, (title, summary) in zip(axes, summaries.items()):
        axis.set_facecolor(SURFACE)
        meshes = summary['meshes']
        series = {
            'cached': [1e3 * summary['cached'][str(m)]['seconds']['median'] for m in meshes],
            'complete': [1e3 * summary['table'][f'{m}|{summary["rom"]}']['device_seconds']['median']
                         for m in meshes],
            'setup': [1e3 * setup_total(summary['setup'][str(m)]) for m in meshes],
            'fom': [1e3 * efficient_fom(summary, m)[0]['device_seconds']['median'] for m in meshes],
        }
        errors = {
            'complete': [summary['table'][f'{m}|{summary["rom"]}']['worst_error_requested_grid']
                         for m in meshes],
            'fom': [efficient_fom(summary, m)[0]['worst_error_requested_grid'] for m in meshes],
        }
        for key, values in series.items():
            color, marker, label = SERIES[key]
            axis.plot(meshes, values, color=color, marker=marker, linewidth=2.0,
                      markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
                      label=label, zorder=3, clip_on=False)
        # Direct labels ride the right-hand end of each line. Series can finish at
        # nearly the same cost, so spread any that would overprint each other.
        ends = sorted(((values[-1], key) for key, values in series.items()))
        span = np.log10(max(v for v, _ in ends) / min(v for v, _ in ends)) or 1.0
        gap = 0.055 * span
        placed = []
        for value, key in ends:
            position = np.log10(value)
            if placed and position - placed[-1] < gap:
                position = placed[-1] + gap
            placed.append(position)
            axis.annotate(SERIES[key][2], (meshes[-1], 10 ** position),
                          textcoords='offset points', xytext=(11, 0), va='center',
                          ha='left', fontsize=8.5, color=MUTED, zorder=4)
        for key in ('complete', 'fom'):
            color = SERIES[key][0]
            for mesh, value, error in zip(meshes, series[key], errors[key]):
                axis.annotate(f'{100 * error:.2f}%', (mesh, value), textcoords='offset points',
                              xytext=(0, -15), ha='center', fontsize=7.5, color=color, zorder=4)
        axis.set_xscale('log', base=2)
        axis.set_yscale('log')
        axis.set_xticks(meshes)
        axis.set_xticklabels([str(m) for m in meshes], color=INK)
        axis.set_xlabel('intervals per axis', color=MUTED, fontsize=10)
        axis.set_ylabel('time (ms, log scale)', color=MUTED, fontsize=10)
        axis.set_title(title, color=INK, fontsize=12, loc='left', pad=10)
        axis.grid(True, which='major', color='#e6e5e1', linewidth=0.8, zorder=0)
        axis.grid(True, which='minor', color='#f2f1ee', linewidth=0.6, zorder=0)
        axis.tick_params(colors=MUTED, labelsize=9)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            axis.spines[side].set_color('#d8d7d2')
        axis.set_xlim(min(meshes) * 0.85, max(meshes) * 2.9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
               fontsize=9.5, labelcolor=MUTED, bbox_to_anchor=(0.5, -0.012))
    fig.suptitle('Cost against mesh size at one frozen checkpoint per PDE',
                 color=INK, fontsize=13.5, x=0.02, ha='left', y=0.99)
    fig.text(0.02, 0.925,
             'Percentages are the worst physical error over every development case and repetition, '
             'on that mesh, against the independently refined reference.',
             color=MUTED, fontsize=9, ha='left')
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    fig.savefig(out_pdf, facecolor=SURFACE)
    plt.close(fig)


def percent(value):
    return '—' if value is None else f'{100 * value:.4f}'


def milliseconds(value):
    return '—' if value is None else f'{1e3 * value:.3f}'


def cost_table(summary):
    lines = ['| Intervals | Interior unknowns | ROM cached (ms) | ROM complete query (ms) | '
             'ROM host-to-host (ms) | ROM offline setup (s) | Efficient FOM | FOM device (ms) | '
             'ROM worst err (%) | FOM worst err (%) | ROM vs same-grid FOM (%) |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for mesh in summary['meshes']:
        rom = summary['table'][f'{mesh}|{summary["rom"]}']
        fom, passing = efficient_fom(summary, mesh)
        lines.append(
            f'| {mesh} | {(mesh - 1) ** 2} | '
            f'{milliseconds(summary["cached"][str(mesh)]["seconds"]["median"])} | '
            f'{milliseconds(rom["device_seconds"]["median"])} | '
            f'{milliseconds(rom["host_to_host_seconds"]["median"])} | '
            f'{setup_total(summary["setup"][str(mesh)]):.3f} | '
            f'`{fom["subject"]}`{"" if passing else " (no arm meets the target)"} | '
            f'{milliseconds(fom["device_seconds"]["median"])} | '
            f'{percent(rom["worst_error_requested_grid"])} | '
            f'{percent(fom["worst_error_requested_grid"])} | '
            f'{percent(rom["worst_same_grid_discrepancy"])} |')
    return '\n'.join(lines)


def subject_table(summary):
    lines = ['| Intervals | Subject | Device (ms) | Host-to-host (ms) | Worst err, requested grid (%) | '
             'Worst err, common grid (%) | Median iterations | Outliers | Stopping |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for mesh in summary['meshes']:
        for name in summary['subjects']:
            row = summary['table'].get(f'{mesh}|{name}')
            if row is None:
                continue
            stopping = ', '.join(f'{k}={v}' for k, v in sorted(row['stopping'].items())) or '—'
            iterations = '—' if row['median_total_iterations'] is None else f'{row["median_total_iterations"]:.0f}'
            lines.append(
                f'| {mesh} | `{name}` | {milliseconds(row["device_seconds"]["median"])} | '
                f'{milliseconds(row["host_to_host_seconds"]["median"])} | '
                f'{percent(row["worst_error_requested_grid"])} | '
                f'{percent(row["worst_error_common_grid"])} | {iterations} | '
                f'{row["device_seconds"]["upper_outliers"]}/{row["device_seconds"]["count"]} | {stopping} |')
    return '\n'.join(lines)


def flatness(summary, key):
    if key == 'cached':
        values = [summary['cached'][str(m)]['seconds']['median'] for m in summary['meshes']]
    else:
        values = [summary['table'][f'{m}|{summary["rom"]}']['device_seconds']['median']
                  for m in summary['meshes']]
    return dict(values_ms=[1e3 * v for v in values],
                ratio_finest_over_coarsest=values[-1] / values[0],
                maximum_over_minimum=max(values) / min(values),
                unknown_growth=((summary['meshes'][-1] - 1) ** 2) / ((summary['meshes'][0] - 1) ** 2))


def crossover(summary):
    """Finest mesh at which the ROM complete query is still slower than the FOM."""
    rows = []
    for mesh in summary['meshes']:
        rom = summary['table'][f'{mesh}|{summary["rom"]}']['device_seconds']['median']
        fom, passing = efficient_fom(summary, mesh)
        rows.append(dict(intervals=mesh, rom_device_ms=1e3 * rom,
                         fom_subject=fom['subject'], fom_device_ms=1e3 * fom['device_seconds']['median'],
                         fom_meets_target=passing,
                         rom_meets_target=summary['table'][f'{mesh}|{summary["rom"]}']['meets_target'],
                         speedup_fom_over_rom=fom['device_seconds']['median'] / rom))
    faster = [r for r in rows if r['speedup_fom_over_rom'] > 1]
    return dict(rows=rows,
                first_mesh_where_rom_is_faster=(faster[0]['intervals'] if faster else None),
                rom_ever_faster=bool(faster),
                rom_faster_and_both_meet_target=[r['intervals'] for r in faster
                                                 if r['rom_meets_target']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--burgers', default='burgers01')
    parser.add_argument('--poisson', default='poisson01')
    parser.add_argument('--out', default=str(HERE / '2026-09-14-frozen-checkpoint-mesh-ladder.md'))
    args = parser.parse_args()

    summaries = {}
    provenance = {}
    for label, attempt in (('Burgers 2D', args.burgers), ('Poisson 2D', args.poisson)):
        result, path = load(attempt)
        summary = aggregate(result)
        summary['rom'] = rom_name(result)
        if result['pde'] == 'poisson':
            summary['same_grid_from_direct_solver'] = poisson_same_grid(result, summary, attempt)
        summary['flatness_cached'] = flatness(summary, 'cached')
        summary['flatness_complete'] = flatness(summary, 'complete')
        summary['crossover'] = crossover(summary)
        summaries[label] = summary
        provenance[label] = dict(
            attempt=attempt, source=str(path.relative_to(ROOT)), job_id=result.get('job_id'),
            commit=result.get('commit'), gpu=result.get('gpu'), node=result.get('slurm_node'),
            jax_backend=result.get('jax_backend'), jax=result.get('jax_version'),
            matmul_precision=result.get('matmul_precision'), x64=result.get('x64'),
            checkpoint=result.get('checkpoint'), checkpoint_sha256=result.get('checkpoint_sha256'),
            elapsed_seconds=result.get('elapsed_seconds'),
            cases=len(result.get('physical_cases', result.get('cohort', {}).get('parameters', []))),
            repetitions=result['config'].get('reps', result['config'].get('repetitions')),
            restriction=result['restriction'], restriction_checks=result['restriction_checks'],
            reference_metrics=result.get('reference_metrics'),
            reference_note=result.get('reference_note'),
            cost_contract=result['cost_contract'],
            verification=result.get('verification'),
            staged_parity=[dict(intervals=r['intervals'],
                                relative_difference=r.get('staged_vs_retained_relative_difference',
                                                          r.get('staged_vs_native_relative_difference')))
                           for r in (result.get('setup') or result.get('mesh_setup'))],
            component_invocations=result.get('component_invocations', [])[:1],
            frozen_weight_transfer=result['frozen_weight_transfer'])
        provenance[label]['reference_seconds'] = result.get('reference_seconds')

    out = Path(args.out)
    data = dict(generated_from=provenance, target_relative_error=TARGET, summaries=summaries)
    out.with_suffix('.json').write_text(json.dumps(data, indent=2) + '\n')
    figure(summaries, out.with_suffix('.png'), out.with_suffix('.pdf'))

    text = [render(summaries, provenance)]
    out.write_text('\n'.join(text))
    print(f'wrote {out}, {out.with_suffix(".json")}, {out.with_suffix(".png")}, {out.with_suffix(".pdf")}')


def render(summaries, provenance):
    from report_text import document
    return document(summaries, provenance, cost_table, subject_table, efficient_fom,
                    setup_total, percent, milliseconds, TARGET)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(HERE))
    main()
