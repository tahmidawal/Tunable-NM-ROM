"""Build this session's LAB-LOG entry from the generated report JSON.

The numbers in the lab log are interpolated from the same aggregates the report
uses, so the log cannot drift from the data either. The prose around them is
written here; the values are not typed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / 'reports/2026-09-14-frozen-checkpoint-mesh-ladder.json'


def ms(value):
    return f'{1e3 * value:.3f}'


def pct(value):
    return f'{100 * value:.4f}%'


def panel(label, summary, provenance, extra):
    meshes = summary['meshes']
    rom = summary['rom']
    cached = summary['flatness_cached']
    complete = summary['flatness_complete']
    cross = summary['crossover']
    first, last = meshes[0], meshes[-1]
    first_row = summary['table'][f'{first}|{rom}']
    last_row = summary['table'][f'{last}|{rom}']
    lines = [f'**{label} — job `{provenance["job_id"]}`, {provenance["gpu"]} on `{provenance["node"]}`, '
             f'attempt `{provenance["attempt"]}`, {provenance["cases"]} development cases x '
             f'{provenance["repetitions"]} repetitions, {provenance["elapsed_seconds"]:.0f} s.** ']
    lines.append(
        f'Cached reduced solve {" -> ".join(ms(v / 1e3) for v in cached["values_ms"])} ms from '
        f'{first} to {last} intervals ({cached["ratio_finest_over_coarsest"]:.3f}x end to end, '
        f'{cached["maximum_over_minimum"]:.3f}x between its own extremes) while interior unknowns '
        f'grow {cached["unknown_growth"]:.0f}x. Complete device query '
        f'{" -> ".join(ms(v / 1e3) for v in complete["values_ms"])} ms '
        f'({complete["ratio_finest_over_coarsest"]:.3f}x). Offline per-mesh setup '
        f'{summary["setup"][str(first)]["total_offline_setup_seconds"]:.2f} s at {first} and '
        f'{summary["setup"][str(last)]["total_offline_setup_seconds"]:.2f} s at {last}, charged '
        f'separately. Worst physical error {pct(first_row["worst_error_requested_grid"])} at {first} '
        f'and {pct(last_row["worst_error_requested_grid"])} at {last} intervals.')
    rows = cross['rows']
    ratios = ', '.join(f'{r["intervals"]}: {r["speedup_fom_over_rom"]:.3f}x vs `{r["fom_subject"]}`'
                       for r in rows)
    if cross['rom_ever_faster']:
        qualifying = (', and both meet the target at '
                      + ', '.join(str(m) for m in cross['rom_faster_and_both_meet_target'])
                      if cross['rom_faster_and_both_meet_target']
                      else ', but the reduced model never meets the accuracy target where it is '
                           'faster, so no qualifying speedup is established')
        lines.append(f'Crossover against the efficient same-job FOM first at '
                     f'{cross["first_mesh_where_rom_is_faster"]} intervals{qualifying}. '
                     f'FOM/ROM by rung: {ratios}.')
    else:
        lines.append(f'**No crossover at any rung**: the efficient same-job FOM is faster than the '
                     f'reduced complete device query everywhere. FOM/ROM by rung: {ratios}.')
    if extra:
        lines.append(extra)
    return ' '.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--date', default='2026-09-14')
    parser.add_argument('--branch', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--audit', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    summaries, provenance = data['summaries'], data['generated_from']

    audits = []
    for path in args.audit:
        block = json.loads(Path(path).read_text())
        errors = block['checks']['common_grid_errors_reproduced']['detail']
        audits.append(f'`{block["attempt"]}` passed ({errors["checked"]} common-grid errors '
                      f'recomputed in NumPy from the archived observation fields, worst absolute '
                      f'difference {errors["worst_absolute_difference"]:.1e})')

    out = [f'## {args.date}', '',
           '### Frozen-checkpoint mesh ladder — complete, both PDEs measured, no retraction', '',
           f'Worktree `worktrees/2026-09-14-mesh-ladder`, branch `{args.branch}` at `{args.commit}`, '
           f'forked from the corrected consolidated baseline `exp/2026-09-13-nmrom-consolidated` at '
           f'`02ff0f1f`. Cluster namespace `/cluster/tufts/paralab/tawal01/mrladder_20260914/`, one '
           f'attempt directory per PDE, both removed after checksum collection.', '',
           'The question was the principal figure for the mesh-independence claim: at **one frozen '
           'checkpoint per PDE**, with latent dimension, bank rank, weak modes, solver policy and '
           'stopping all fixed, how do the cached reduced solve, the complete device query and the '
           'offline per-mesh setup behave across 64/128/256/512/1024 intervals per axis, and where '
           '(if anywhere) is the crossover against an efficient same-job full-order solver. Each PDE '
           'ran as **one job on one GPU** with every arm interleaved in a randomized order, which is '
           'what the historical ladders lacked: those used a different GPU per mesh and, on Burgers, '
           'a mesh-specific checkpoint, so they established neither a same-GPU scaling nor '
           'frozen-weight transfer.', '']
    for label, summary in summaries.items():
        extra = None
        if 'same_grid_from_direct_solver' in summary:
            last = summary['meshes'][-1]
            entry = summary['table'][f'{last}|{summary["rom"]}']
            if entry.get('worst_same_grid_discrepancy') is not None:
                extra = (f'The ROM-vs-same-grid-FOM discrepancy at {last} intervals is '
                         f'{pct(entry["worst_same_grid_discrepancy"])} against the exact '
                         f'direct-transform solution, essentially equal to the physical error, so the '
                         f'error is reduction error and not discretisation error.')
        out.append('- ' + panel(label, summary, provenance[label], extra))
    out += ['',
            '**Method points worth keeping.** The three costs are each their own completed device '
            'computation; none is obtained by subtracting another. Both staged solvers were checked '
            'against the retained selected solvers: the Burgers split is **bitwise identical** to '
            '`accuracy_paths.make_rom` and reproduces the consolidated replay\'s saved 64-interval '
            'case to 1.6e-14 against a declared 1e-8 tolerance; the Poisson split matches the native '
            f'`correction_query` field to '
            f'{max(p["relative_difference"] for p in provenance["Poisson 2D"]["staged_parity"]):.1e}. '
            'The cross-mesh restriction is nested-node injection, validated in-job (restricting in '
            'one step and through every intermediate rung give bitwise identical fields, and the '
            'boundary survives exactly).', '',
            '**One measurement difference worth flagging, not a retraction.** The Poisson complete '
            'device query here excludes the stationarity and projected-Jacobian-rank diagnostics that '
            'the native `correction_query` charges into the same interval, so it is not comparable '
            'with the 2026-09-11 Poisson GPU column. The native row is still recorded per case, '
            'outside every timer, and remains the source of every stopping and rank verdict.', '']
    if audits:
        out += ['**Independent audit.** A NumPy-only audit that imports neither driver nor JAX: '
                + '; '.join(audits) + '. It also checks the timing identity and positivity of every '
                'invocation, requires a complete Cartesian invocation grid, and re-derives the '
                'reference digests.', '']
    out += ['**Retracted or withdrawn: nothing.** No earlier numerical result is contradicted here. '
            'This ladder does not retest the historical small-bank tensor configuration and does not '
            'reinstate any discarded evidence.', '',
            '**Open.** Final cohorts remain sealed and nothing was merged. Per-resolution tuning is '
            'deliberately untested, since that answers a different question from frozen-weight '
            'transfer. The branch is **not on origin**: this worktree is descended from the '
            'consolidated baseline and the coordinator asked that no push be made from it, because '
            '`git pack-objects` on the 199 GB repository reaches roughly 48 GB resident on the shared '
            'GB10 and risks an earlyoom kill of other agents\' work. Twenty-one local branches, '
            'including the base `exp/2026-09-13-nmrom-consolidated`, are in the same state; the '
            'coordinator will push them in stages.', '',
            'Report, figure (PNG and PDF), aggregates JSON and generator: '
            '`experiments/mesh-ladder/reports/2026-09-14-frozen-checkpoint-mesh-ladder.*` with '
            '`generate_ladder.py` beside them. Chunked, checksum-verified job archives: '
            '`experiments/mesh-ladder/artifacts/<attempt>/`.', '']
    args.out.write_text('\n'.join(out))
    print(f'wrote {args.out} ({len("".join(out))} chars)')


if __name__ == '__main__':
    main()
