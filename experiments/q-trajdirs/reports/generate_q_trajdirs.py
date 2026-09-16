"""Generate the q-trajdirs report and its figure.

Every number and every plotted point is read from the audited run JSON. Nothing in the
report is typed by hand: the generator takes the audit as its only numeric input, plus
the raw `result.json` for the direction-set metadata it passes through.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SETCOLOR = {'old': '#b03a3a', 'traj': '#3b6ea5', 'prac': '#4f8a3d', 'shared': '#777777'}
SETLABEL = {'old': 'old (static residual)', 'traj': 'traj (trajectory, 32)',
            'prac': 'prac (trajectory, 6)'}


def yn(x):
    return {True: 'yes', False: 'no', None: '—'}[x]


def f(x, d=4):
    if x is None:
        return '—'
    if isinstance(x, bool):
        return yn(x)
    if isinstance(x, float):
        return f'{x:.{d}f}'
    return str(x)


def sci(x, d=2):
    return '—' if x is None else f'{x:.{d}e}'


def pct(x, d=4):
    return '—' if x is None else f'{x:.{d}f}'


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |',
           '|' + '|'.join(['---'] * len(header)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(out) + '\n'


def gate_table(checks):
    rows = []
    for name, v in sorted(checks.items()):
        if not isinstance(v, dict):
            continue
        if v.get('passed') is None:
            continue
        kind = 'informational' if v.get('blocking') is False else 'blocking'
        rows.append([f'`{name}`', yn(v.get('passed')), kind])
    return table(['gate', 'passed', 'kind'], rows)


def figure(aud, out_png, out_pdf):
    """Evolved-times error against q for old and new directions, plus per-time panels."""
    lads = aud['ladders']
    times = aud['output_times']
    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.35, 1.35, 1.0], hspace=.75, wspace=.38)

    groups = [('primary', 'M = 4(K+q); solid = EQ per rung, dashed = dense twin',
               [('old', 'old_primary', 'old_dense'), ('traj', 'traj_primary', 'traj_dense'),
                ('prac', 'prac_primary', 'prac_dense')]),
              ('fixedM', 'fixed M = 256, EQ per rung',
               [('old', 'old_fixedM', None), ('traj', 'traj_fixedM', None)])]

    for col, (key, title) in enumerate((('evolved', 'worst over EVOLVED times ($t>0$)'),
                                        ('all_times', 'worst over ALL times ($t\\geq 0$)'))):
        for grow, (gid, gtitle, members) in enumerate(groups):
            ax = fig.add_subplot(gs[grow, col * 3:(col + 1) * 3])
            for setname, lid, dense_lid in members:
                lad = lads.get(lid)
                if not lad:
                    continue
                qs = [u['q'] for u in lad['rungs']]
                vs = [u[key] for u in lad['rungs']]
                cv = [u['converged'] for u in lad['rungs']]
                ax.plot(qs, vs, '-', color=SETCOLOR[setname], lw=1.8, zorder=2,
                        label=SETLABEL[setname])
                for q, v, c in zip(qs, vs, cv):
                    ax.plot([q], [v], 'o' if c else 'X', ms=8 if c else 10,
                            mfc=SETCOLOR[setname] if c else 'white',
                            mec=SETCOLOR[setname], mew=1.8, zorder=3)
                dl = lads.get(dense_lid) if dense_lid else None
                if dl:
                    ax.plot([u['q'] for u in dl['rungs']], [u[key] for u in dl['rungs']],
                            '--', color=SETCOLOR[setname], lw=1.1, alpha=.75, zorder=1)
                    for u in dl['rungs']:
                        ax.plot([u['q']], [u[key]], 'o' if u['converged'] else 'X', ms=5,
                                mfc='white', mec=SETCOLOR[setname], mew=1.1, zorder=1)
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda y, _: f'{y:g}'))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
                lambda y, _: f'{y:g}'))
            ax.tick_params(axis='y', which='minor', labelsize=7)
            ax.set_xscale('symlog', linthresh=16)
            ax.set_xticks(sorted({u['q'] for lad in lads.values() for u in lad['rungs']}))
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_xlabel('correction directions $q$')
            ax.set_ylabel('same-grid error (%)')
            ax.set_title(f'{title}\n{gtitle}', fontsize=10)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8, loc='best')

    # Per-output-time panels on the primary ladders.
    for i, t in enumerate(times):
        ax = fig.add_subplot(gs[2, i])
        for setname, lid, _dense in groups[0][2]:
            lad = lads.get(lid)
            if not lad:
                continue
            qs = [u['q'] for u in lad['rungs']]
            vs = [u['per_time'][i] for u in lad['rungs']]
            ax.plot(qs, vs, 'o-', ms=4, lw=1.4, color=SETCOLOR[setname],
                    label=SETLABEL[setname] if i == 0 else None)
        ax.set_xscale('symlog', linthresh=16)
        ax.set_xticks([0, 16, 64, 256])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.tick_params(labelsize=7)
        ax.set_title(f'$t = {t:g}$', fontsize=9)
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f'{y:g}'))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f'{y:g}'))
        ax.grid(alpha=.25)
        if i == 0:
            ax.set_ylabel('worst same-grid\nerror (%)', fontsize=8)
    fig.suptitle('Trajectory-fitted vs static correction directions — Burgers $256^2$, '
                 'frozen checkpoint\nfilled marker = converged, cross = not converged; '
                 'bottom row: worst-over-cases error at each output time, primary ladders',
                 fontsize=11)
    fig.savefig(out_png, dpi=170, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', required=True)
    p.add_argument('--result', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--figure', required=True)
    p.add_argument('--qdiag', default=None)
    a = p.parse_args()
    aud = json.loads(Path(a.audit).read_text())
    raw = json.loads(Path(a.result).read_text())
    cfg = raw['config']
    lads = aud['ladders']
    by_arm = {x['arm']: x for x in aud['arms']}
    v = aud['verdict']
    times = aud['output_times']

    figure(aud, Path(a.figure).with_suffix('.png'), Path(a.figure).with_suffix('.pdf'))

    prim = lads['traj_primary']
    oldfix = lads.get('old_fixedM', {})
    trajfix = lads.get('traj_fixedM', {})

    state = 'PASSED' if v['passes'] else 'did NOT pass'
    L = []
    L.append('# Trajectory-fitted correction directions on the Burgers ladder\n')
    L.append(
        'Does fitting the correction directions to the ROM\'s **trajectory** error, instead '
        'of the head\'s static reconstruction residual, make the Burgers $256^2$ fixed-weight '
        'correction ladder monotone on the worst-over-evolved-times metric while keeping it '
        'monotone on worst-over-all-times? One frozen checkpoint, one mesh, one job, six '
        f'opened development cases. The pre-registered pass **{state}**. These numbers are '
        'final for this cell: every one is recomputed in an independent NumPy audit from the '
        'saved output fields, and nothing here is typed by hand.\n')
    L.append(f'Job `{aud["job_id"]}`, source `{aud["commit"]}`, GPU `{aud["gpu"]}`, '
             f'elapsed {aud["elapsed_seconds"]:.1f} s, JAX backend gpu, float64, matmul '
             'precision `highest`. Predeclared protocol and every amendment: '
             '[`DESIGN.md`](../DESIGN.md).\n')

    # ------------------------------------------------------------- the verdict
    L.append('## The verdict\n')
    crit = [
        ['1. worst **evolved** error non-increasing in $q$, every rung converged',
         yn(v['criterion_1_evolved_monotone_every_rung_converged']),
         f"monotone {yn(v['monotone_evolved'])}, all rungs converged {yn(v['all_converged'])}"],
        ['2. worst **all-times** error non-increasing in $q$',
         yn(v['criterion_2_all_times_monotone']),
         f"monotone {yn(v['monotone_all_times'])}"],
        ['3. converged non-dominated set spans $\\ge 2\\times$ in evolved error',
         yn(v['criterion_3_passes']),
         (f"{v['criterion_3_error_span']['points']} points, error span "
          f"{f(v['criterion_3_error_span']['error_span'], 3)}x, cost span "
          f"{f(v['criterion_3_error_span']['cost_span'], 3)}x")]]
    L.append(table(['pre-registered criterion (traj primary ladder)', 'holds', 'measured'], crit))
    L.append(f'**Overall: the pre-registered pass {state}.**\n')

    if v['regressions_evolved']:
        L.append('\nRegressions on the evolved metric that survive on the `traj` primary '
                 'ladder:\n\n')
        L.append(table(['from $q$', 'to $q$', 'from %', 'to %'],
                       [[x['from_q'], x['to_q'], pct(x['from_percent']), pct(x['to_percent'])]
                        for x in v['regressions_evolved']]))

    L.append('\n### F1 — the $q=16$ regression, per ladder\n')
    L.append('The incumbent cell (`btq201`) showed the evolved-times metric getting **worse** '
             'from $q=0$ to $q=16$. This is that same comparison, in this job, for every '
             'ladder that has both rungs.\n\n')
    L.append(table(['ladder', 'directions', '$q=0$ evolved %', '$q=16$ evolved %',
                    'change (pp)', 'regression'],
                   [[f'`{lid}`', lads[lid]['dirset'], pct(x['q0']), pct(x['q16']),
                     f"{x['delta_percentage_points']:+.4f}", yn(x['regression'])]
                    for lid, x in sorted(v['q16_regression'].items())]))

    # --------------------------------------------------------- the directions
    L.append('\n## The three direction sets\n')
    ds = aud['direction_sets']
    L.append(table(['set', 'rule', 'residual rows', 'available rank', 'orthonormality dev',
                    'fit seconds', '`directions_sha256`'],
                   [[f'`{k}`',
                     ('static reconstruction residual' if k == 'old' else
                      f"trajectory error, {ds[k].get('cohort', {}).get('count', '—')} trajectories"),
                     ds[k]['residual_rows'], ds[k]['available_rank'],
                     sci(ds[k].get('orthonormality_deviation')),
                     f(ds[k].get('seconds'), 1), '`' + ds[k]['directions_sha256'][:16] + '…`']
                    for k in ('old', 'traj', 'prac') if k in ds]))

    L.append('\n### Residual energy captured, and cross-capture\n')
    L.append('$\\kappa_q(P, C)=\\|\\tilde P\\,\\tilde C_{:,1:q}\\|_F^2/\\|\\tilde P\\|_F^2$ — the '
             'fraction of residual matrix $P$\'s whitened energy that direction set $C$ can reach '
             'at rung $q$. The diagonal is each set\'s own POD energy; the off-diagonal is the '
             'diagnostic.\n\n')
    cc = aud['direction_comparison']['cross_capture']
    qs = [q for q in cfg['q_ladder'] if q > 0]
    rows = []
    for pn in ('old', 'traj', 'prac'):
        for cn in ('old', 'traj', 'prac'):
            if pn not in cc or cn not in cc[pn]:
                continue
            rows.append([f'`{pn}`', f'`{cn}`'] + [f"{cc[pn][cn][str(q)] * 100:.2f}" for q in qs])
    L.append(table(['residual $P$', 'directions $C$'] + [f'$q={q}$ %' for q in qs], rows))

    L.append('\n### Principal angles between the direction subspaces\n')
    L.append('Field-metric principal angles, in degrees, between the first $q$ columns of each '
             'pair. `overlap` is $\\frac1q\\sum_i\\cos^2\\theta_i$ and is 1 for identical '
             'subspaces, $q/R$ for a random pair.\n\n')
    pa = aud['direction_comparison']['principal_angles']
    rows = []
    for pair, d in sorted(pa.items()):
        for q in sorted(d, key=int):
            x = d[q]
            rows.append([f'`{pair}`', q, f(x['overlap'], 5), f(x['angle_min_degrees'], 2),
                         f(x['angle_median_degrees'], 2), f(x['angle_max_degrees'], 2)])
    L.append(table(['pair', '$q$', 'overlap', 'min angle°', 'median angle°', 'max angle°'], rows))

    coh = aud['direction_cohorts']
    L.append('\n### Cohorts\n')
    L.append(table(['cohort', 'trajectories', 'indices sha256', 'min distance to the six cases',
                    'disjoint'],
                   [[f'`{k}`', coh[k]['count'], '`' + coh[k]['indices_sha256'][:16] + '…`',
                     sci(coh[k]['min_distance_to_evaluation_cases']), yn(coh[k]['disjoint'])]
                    for k in ('traj', 'prac') if k in coh]))

    # ------------------------------------------------------------- the ladders
    L.append('\n## The ladders, both metrics\n')
    for lid in ('old_primary', 'traj_primary', 'prac_primary', 'old_fixedM', 'traj_fixedM',
                'old_dense', 'traj_dense', 'prac_dense'):
        lad = lads.get(lid)
        if not lad:
            continue
        L.append(f'\n**`{lid}`** — {lad["label"]}. Monotone on evolved: '
                 f'{yn(lad["monotone_evolved"])}; on all times: {yn(lad["monotone_all_times"])}; '
                 f'every rung converged: {yn(lad["all_converged"])}.\n\n')
        L.append(table(['$q$', 'arm', '$M$', '$m$', 'worst all times %', 'worst evolved %',
                        '$t=0$ compression %', 'best-found %', 'median GPU ms', 'budget exits',
                        'worst joint gradient', 'converged'],
                       [[u['q'], f"`{u['arm']}`", u['M'], u['m'] or '—', pct(u['all_times']),
                         pct(u['evolved']), pct(u['t0']), pct(u['best_found']),
                         f(u['gpu_ms'], 3), u['budget_exits'],
                         sci(u['max_joint_stationarity']), yn(u['converged'])]
                        for u in lad['rungs']]))

    # --------------------------------------------------------- per output time
    L.append('\n## Worst-over-cases error at each output time\n')
    L.append('The figure\'s bottom row, as a table. Same-grid error against the same-job '
             '`fft_tight` solve, worst over the six cases.\n\n')
    for lid in ('old_primary', 'traj_primary', 'prac_primary'):
        lad = lads.get(lid)
        if not lad:
            continue
        L.append(f'\n**`{lid}`**\n\n')
        L.append(table(['$q$'] + [f'$t={t:g}$ %' for t in times],
                       [[u['q']] + [pct(x) for x in u['per_time']] for u in lad['rungs']]))

    # ------------------------------------------------------------- the frontier
    L.append('\n## Non-dominated sets over (median GPU ms, error)\n')
    for key, label in (('all_subjects_all_times', 'every subject, all output times'),
                       ('all_subjects_evolved', 'every subject, evolved times only'),
                       ('converged_rom_all_times', 'converged ROM arms, all output times'),
                       ('converged_rom_evolved', 'converged ROM arms, evolved times only')):
        names = aud['frontier'][key]
        L.append(f'\n**{label}**\n\n')
        L.append(table(['arm', 'family', 'directions', '$q$', 'worst all times %',
                        'worst evolved %', 'median GPU ms', 'converged'],
                       [[f'`{n}`', by_arm[n]['family'], by_arm[n]['dirset'] or '—',
                         by_arm[n]['q'] if by_arm[n]['q'] is not None else '—',
                         pct(by_arm[n]['worst_all_times_percent']),
                         pct(by_arm[n]['worst_evolved_percent']),
                         f(by_arm[n]['median_gpu_ms'], 3), yn(by_arm[n]['converged'])]
                        for n in names]))

    # --------------------------------------------------------- the FOM controls
    L.append('\n## Same-job full-order controls\n')
    L.append(table(['control', 'ntol', 'ltol', '$\\Delta t$', 'worst all times %',
                    'worst evolved %', 'median GPU ms'],
                   [[f"`{s['name']}`", sci(s['ntol']), sci(s['ltol']), s['dt'],
                     pct(by_arm[s['name']]['worst_all_times_percent']),
                     pct(by_arm[s['name']]['worst_evolved_percent']),
                     f(by_arm[s['name']]['median_gpu_ms'], 3)]
                    for s in cfg['fom_settings'] if s['name'] in by_arm]))

    # ------------------------------------------------------------- every arm
    L.append('\n## Every arm\n')
    L.append(table(['arm', 'directions', '$q$', '$M$', '$m$', 'quad', 'all times %',
                    'evolved %', '$t_0$ %', 'reference %', 'median GPU ms',
                    'median iters/step', 'budget exits', 'EQ rel fit', 'EQ valid', 'converged'],
                   [[f"`{x['arm']}`", x['dirset'] or '—',
                     x['q'] if x['q'] is not None else '—', x['M'] or '—', x['m'] or '—',
                     x['quadrature'] or '—', pct(x['worst_all_times_percent']),
                     pct(x['worst_evolved_percent']), pct(x['worst_t0_compression_percent']),
                     pct(x['worst_reference_percent']), f(x['median_gpu_ms'], 3),
                     f(x['median_iterations'], 1), x['total_budget_exits']
                     if x['total_budget_exits'] is not None else '—',
                     sci(x['quadrature_fit_relative']), yn(x['eq_rule_valid']),
                     yn(x['converged'])]
                    for x in aud['arms']]))

    # -------------------------------------------------------------- fidelity
    L.append('\n## Cross-job fidelity\n')
    fid = aud['checks']['cross_job_fidelity']['detail']
    L.append(table(['arm', 'reproduces', 'job', 'declared tolerance', 'worst relative difference',
                    'ours %', 'theirs %', 'passed'],
                   [[f'`{k}`', f"`{d['detail']['comparator']}`", d['detail']['job'],
                     sci(d['detail']['declared_tolerance']),
                     sci(d['detail']['worst_relative_difference']),
                     pct(d['detail']['ours_worst_same_grid_percent']),
                     pct(d['detail']['theirs_worst_same_grid_percent']), yn(d['passed'])]
                    for k, d in sorted(fid.items()) if isinstance(d.get('detail'), dict)]))
    probe = aud['checks']['btq201_envelope_probe']['detail']
    if probe:
        L.append('\nAnd the same arms against the `b-ladder-top` envelope job, as a probe '
                 '(the EQ rules are refitted per job, so these are not expected to be bitwise):\n\n')
        L.append(table(['arm', 'btq201 arm', 'ours all %', 'theirs all %', 'ours evolved %',
                        'theirs evolved %', 'ours GPU ms', 'theirs GPU ms'],
                       [[f'`{k}`', f"`{d['comparator']}`", pct(d['ours_all']),
                         pct(d['theirs_all']), pct(d['ours_evolved']), pct(d['theirs_evolved']),
                         f(d['ours_gpu_ms'], 3), f(d['theirs_gpu_ms'], 3)]
                        for k, d in sorted(probe.items())]))

    # ------------------------------------------------------------------ gates
    L.append('\n## Gates\n')
    L.append(gate_table(aud['checks']))
    if aud['failed']:
        L.append('\nFailed: ' + ', '.join(f'`{x}`' for x in aud['failed']) + '.\n')
    else:
        L.append('\nNo blocking gate failed.\n')

    # ------------------------------------------------------------- q-diag lane
    L.append('\n## The parallel diagnosis lane\n')
    if a.qdiag and Path(a.qdiag).exists():
        L.append(f'The `q-diag` lane\'s verdict, cited as required by `DESIGN.md` section 7, '
                 f'is at `{a.qdiag}`.\n')
    else:
        L.append('The `q-diag` lane, which is measuring the cause of the $q=16$ regression from '
                 'the saved fields in parallel, had **not** published a report under '
                 '`worktrees/2026-09-16-q-diag/experiments/q-diag/reports/` when this report was '
                 'generated, so there is no verdict of its to cite. `DESIGN.md` section 7 '
                 'required this to be said either way.\n')

    # ---------------------------------------------------------------- figure
    L.append('\n## Figure\n')
    L.append(f'![evolved-times error against q, old vs new directions]'
             f'({Path(a.figure).with_suffix(".png").name})\n')

    # -------------------------------------------------------------- glossary
    L.append('\n## Glossary\n')
    L.append(
        'Written for a reader who knows none of this project\'s vocabulary.\n\n'
        '- **ROM / reduced-order model.** A cheap surrogate solver. Instead of solving the '
        'PDE for all $n=255^2$ interior grid values, it solves for a handful of coefficients '
        'and reconstructs the field from them.\n'
        '- **FOM / full-order model.** The real solver on the real grid. `fft_tight` is the '
        'converged one; `fft_loose` and `nt1e-2_dt01` are deliberately under-solved cheap ones.\n'
        '- **The bank $G$.** A fixed matrix with $R=512$ columns, each a field on the grid, '
        'produced once by the frozen neural checkpoint. Every state the ROM can represent is '
        'some combination $G\\eta$ of those 512 fields.\n'
        '- **The head $h_\\theta$.** The frozen neural map from a $K=16$-dimensional latent '
        'code $z$ to the 512 bank coefficients. It is never retrained anywhere in this cell.\n'
        '- **$q$, the correction directions.** Extra bank directions $C_q$ solved alongside '
        'the latent code: $\\eta = h_\\theta(z) + C_q y$. $q=0$ is the plain frozen model; '
        'larger $q$ gives the solver more freedom, and more work per step. The question of '
        'this cell is how to CHOOSE those directions.\n'
        '- **`old` directions.** The incumbent rule: decompose the *static* error the head '
        'makes when it is allowed to fit each training snapshot on its own.\n'
        '- **`traj` / `prac` directions.** The new rules: decompose the error the $q=0$ ROM '
        'actually accumulates along a *trajectory*, at every internal time step, against the '
        'converged full-order solve. `traj` uses 32 training trajectories, `prac` uses 6.\n'
        '- **POD (proper orthogonal decomposition).** A singular-value decomposition of a '
        'matrix of error vectors; its leading directions are the ones that explain the most '
        'error energy. "Nested in $q$" means the rank-16 set is the first 16 columns of the '
        'rank-256 set, so one matrix serves every rung.\n'
        '- **Field metric / whitening $R_G$.** The bank columns are not orthogonal, so the '
        'ordinary length of a coefficient vector is not the physical size of the field it '
        'makes. Multiplying by $R_G$ (from the QR factorisation of $G$) fixes that, and makes '
        '"least squares in coefficients" mean "least squares in the field".\n'
        '- **Rung.** One value of $q$ on a ladder. **Ladder.** One ordered sweep of $q$ at '
        'fixed everything else.\n'
        '- **$M$, the test count.** How many test functions the weak residual is measured '
        'against. `M = 4(K+q)` grows with $q$; the `fixedM` ladders hold it at 256 so the '
        'effect of $q$ is separated from the effect of $M$.\n'
        '- **$m$ / EQ / empirical quadrature.** The nonlinear term is exactly a sum over all '
        '$n$ grid points ("dense"); an empirical-quadrature rule replaces it with a weighted '
        'sum over $m$ chosen points, fitted offline by nonnegative least squares. It is the '
        'main source of the ROM\'s speed. A rule whose fitter ran out of time is `truncated` '
        'and disqualifies its arm.\n'
        '- **Dense twin.** The same rung run with the exact quadrature instead of the EQ rule, '
        'so any EQ artefact is visible.\n'
        '- **Same-grid error.** Distance from the same-job converged full-order solve on the '
        'same grid, divided by the norm of the supplied initial field. **Reference error** is '
        'the same thing against a 4096-interval solve, which also contains the discretisation '
        'error of the 256 grid and is therefore a much larger, less discriminating number.\n'
        '- **All times / evolved times / $t_0$ compression.** The six output times are '
        '$t=0,0.05,\\dots,0.25$. "All times" takes the worst over all six; "evolved times" '
        'drops $t=0$; "$t_0$ compression" is the $t=0$ term alone — the error the decoder '
        'makes just reproducing the field it was handed, before any time stepping. The all-times '
        'metric is usually pinned by that term, which is why both are always reported.\n'
        '- **Worst / median.** Worst is over the six evaluation cases. Median GPU ms is the '
        'median over all retained timed repetitions (three per case, all kept).\n'
        '- **Converged.** Every time step and the initial fit exited on a residual, tiny-step '
        'or gradient criterion (never on the iteration budget) AND the joint normalized '
        'gradient stayed at or below $10^{-6}$. An unconverged arm\'s error is still reported, '
        'labelled, and excluded from the pre-registered criterion.\n'
        '- **Budget exit.** A time step that ran out of its per-step iteration budget (600 '
        'here) before meeting the stopping rule.\n'
        '- **Joint normalized gradient.** $\\|J^\\top r\\|/(\\|J\\|\\,\\|r\\|)$ over both '
        'blocks of unknowns: the scale-free measure of "is this actually a stationary point".\n'
        '- **Best-found.** The smallest error any code on that rung\'s manifold can achieve on '
        'a supplied field, found offline by multistart optimisation. It is the accuracy floor '
        'the online solver is chasing. **Bank projection** is the floor below that, using all '
        '512 bank directions.\n'
        '- **Non-dominated set.** The arms that nothing else beats on both cost and error at '
        'once — the usable trade-off curve.\n'
        '- **Cross-capture $\\kappa_q$.** The fraction of one rule\'s error energy that the '
        'other rule\'s directions can represent at all. Low off-diagonal numbers mean the two '
        'rules are aiming at genuinely different things.\n'
        '- **Principal angles / overlap.** How far apart two subspaces are. Overlap 1 means '
        'identical; overlap $q/R$ means as unrelated as a random pair.\n'
        '- **Gate.** A check that had to pass before any number here was allowed to count. '
        'Blocking gates decide; informational ones are probes reported either way.\n'
        '- **`params_draw` landmine.** The case generator draws column by column, so asking '
        'for 32 cases from a seed does not give the first 32 of 128 from that seed. Every '
        'cohort here is a set of indices into ONE draw of 128.\n')

    text = ''.join(x if x.endswith('\n\n') else x + '\n' for x in L)
    Path(a.out).write_text(text)
    print(a.out, hashlib.sha256(Path(a.out).read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
