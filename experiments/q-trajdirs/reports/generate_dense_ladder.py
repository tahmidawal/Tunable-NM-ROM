"""Generate the dense-correction-ladder report and its figure.

Inputs, all machine-generated and all already audited:

  --audit / --result          the dense-ladder job (qtd02), the PRIMARY subject
  --archives                  `dense_from_archives.py` output, Part 1 (no GPU)
  --control-audit             the trajectory-directions job (qtd01), the CONTROL
  --qdiag                     the q-diag lane's report, cited if it exists

Nothing in the report is typed by hand.
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
RULECOLOR = {'m4': '#3b6ea5', 'mfix': '#b03a3a'}


def yn(x):
    return {True: 'yes', False: 'no', None: '—'}[x]


def f(x, d=4):
    if x is None:
        return '—'
    if isinstance(x, bool):
        return yn(x)
    return f'{x:.{d}f}' if isinstance(x, float) else str(x)


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
        if not isinstance(v, dict) or v.get('passed') is None:
            continue
        rows.append([f'`{name}`', yn(v.get('passed')),
                     'informational' if v.get('blocking') is False else 'blocking'])
    return table(['gate', 'passed', 'kind'], rows)


def logaxes(ax):
    ax.set_yscale('log')
    for which in ('major', 'minor'):
        getattr(ax.yaxis, f'set_{which}_formatter')(
            matplotlib.ticker.FuncFormatter(lambda y, _: f'{y:g}'))
    ax.tick_params(axis='y', which='minor', labelsize=7)
    ax.set_xscale('symlog', linthresh=16)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(alpha=.25)


def ladder_line(ax, lad, key, color, label, dashed=False):
    qs = [u['q'] for u in lad['rungs']]
    vs = [u[key] for u in lad['rungs']]
    cv = [u['converged'] for u in lad['rungs']]
    ax.plot(qs, vs, '--' if dashed else '-', color=color, lw=1.2 if dashed else 1.9,
            alpha=.8 if dashed else 1., zorder=2, label=label)
    for q, v, c in zip(qs, vs, cv):
        ax.plot([q], [v], 'o' if c else 'X', ms=7 if c else 9,
                mfc=color if c else 'white', mec=color, mew=1.6, zorder=3)


def figure(aud, control, out_png, out_pdf):
    times = aud['output_times']
    lads = aud['ladders']
    nrow = 3 if control else 2
    fig = plt.figure(figsize=(15, 4.6 * nrow))
    gs = fig.add_gridspec(nrow, 6, height_ratios=[1.35] * (nrow - 1) + [1.0],
                          hspace=.72, wspace=.42)

    for col, (key, title) in enumerate((('evolved', 'worst over EVOLVED times ($t>0$)'),
                                        ('all_times', 'worst over ALL times ($t\\geq 0$)'))):
        ax = fig.add_subplot(gs[0, col * 3:(col + 1) * 3])
        for lid, lad in sorted(lads.items()):
            ladder_line(ax, lad, key, RULECOLOR.get(lad['rule'], '#555555'),
                        f"{lid} ({lad['rule']})")
        logaxes(ax)
        ax.set_xticks(sorted({u['q'] for lad in lads.values() for u in lad['rungs']}))
        ax.set_xlabel('correction directions $q$')
        ax.set_ylabel('same-grid error (%)')
        ax.set_title(f'{title}\nDENSE ladder, budget 600, one allocation', fontsize=10)
        ax.legend(fontsize=8, loc='best')

    if control:
        clads = control['ladders']
        for col, (key, title) in enumerate((('evolved', 'worst over EVOLVED times ($t>0$)'),
                                            ('all_times', 'worst over ALL times ($t\\geq 0$)'))):
            ax = fig.add_subplot(gs[1, col * 3:(col + 1) * 3])
            for lid, lad in sorted(clads.items()):
                if lad['rule'] != 'm4':
                    continue
                ladder_line(ax, lad, key, SETCOLOR.get(lad['dirset'], '#555'),
                            f"{lad['dirset']} {lad['quadrature']}",
                            dashed=(lad['quadrature'] == 'dense'))
            logaxes(ax)
            ax.set_xticks(sorted({u['q'] for lad in clads.values() for u in lad['rungs']}))
            ax.set_xlabel('correction directions $q$')
            ax.set_ylabel('same-grid error (%)')
            ax.set_title(f'{title}\nCONTROL job: three direction rules, $M=4(K+q)$;\n'
                         'solid = empirical quadrature, dashed = dense', fontsize=10)
            ax.legend(fontsize=8, loc='best')

    for i, t in enumerate(times):
        ax = fig.add_subplot(gs[nrow - 1, i])
        for lid, lad in sorted(lads.items()):
            ax.plot([u['q'] for u in lad['rungs']], [u['per_time'][i] for u in lad['rungs']],
                    'o-', ms=4, lw=1.4, color=RULECOLOR.get(lad['rule'], '#555'),
                    label=lid if i == 0 else None)
        if control:
            cl = control['ladders'].get('traj_primary') or control['ladders'].get('old_primary')
            if cl:
                ax.plot([u['q'] for u in cl['rungs']], [u['per_time'][i] for u in cl['rungs']],
                        's--', ms=3.5, lw=1.1, color='#8a5a2a',
                        label='control, EQ' if i == 0 else None)
        logaxes(ax)
        ax.set_xticks([0, 16, 64, 256])
        ax.tick_params(labelsize=7)
        ax.set_title(f'$t = {t:g}$', fontsize=9)
        if i == 0:
            ax.set_ylabel('worst same-grid\nerror (%)', fontsize=8)
            ax.legend(fontsize=6, loc='best')
    fig.suptitle('The DENSE correction ladder on a frozen Burgers $256^2$ checkpoint\n'
                 'filled marker = converged, cross = not converged; '
                 'bottom row: worst-over-cases error at each output time', fontsize=11)
    fig.savefig(out_png, dpi=170, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)


def ladder_block(L, lads, order=None):
    for lid in (order or sorted(lads)):
        lad = lads.get(lid)
        if not lad:
            continue
        L.append(f'\n**`{lid}`** — {lad["label"]}. Monotone on evolved: '
                 f'{yn(lad["monotone_evolved"])}; on all times: '
                 f'{yn(lad["monotone_all_times"])}; every rung converged: '
                 f'{yn(lad["all_converged"])}.\n\n')
        L.append(table(['$q$', 'arm', '$M$', '$m$', 'worst all times %', 'worst evolved %',
                        '$t=0$ compression %', 'best-found %', 'median GPU ms',
                        'budget exits', 'worst joint gradient', 'converged'],
                       [[u['q'], f"`{u['arm']}`", u['M'], u['m'] or '—', pct(u['all_times']),
                         pct(u['evolved']), pct(u['t0']), pct(u['best_found']),
                         f(u['gpu_ms'], 3), u['budget_exits'],
                         sci(u['max_joint_stationarity']), yn(u['converged'])]
                        for u in lad['rungs']]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', required=True)
    p.add_argument('--result', required=True)
    p.add_argument('--archives', required=True)
    p.add_argument('--control-audit', default=None)
    p.add_argument('--qdiag', default=None)
    p.add_argument('--out', required=True)
    p.add_argument('--figure', required=True)
    a = p.parse_args()
    aud = json.loads(Path(a.audit).read_text())
    raw = json.loads(Path(a.result).read_text())
    arc = json.loads(Path(a.archives).read_text())
    control = json.loads(Path(a.control_audit).read_text()) if a.control_audit else None
    cfg = raw['config']
    lads = aud['ladders']
    by_arm = {x['arm']: x for x in aud['arms']}
    v = aud['verdict']
    times = aud['output_times']

    figure(aud, control, Path(a.figure).with_suffix('.png'), Path(a.figure).with_suffix('.pdf'))
    state = 'PASSES' if v['passes'] else 'does NOT pass'

    L = []
    L.append('# The dense correction ladder as a tunability knob\n')
    L.append(
        'Does the **dense** fixed-weight correction ladder — incumbent directions, per-step '
        'iteration budget 600, no empirical quadrature anywhere — satisfy the knob criterion on '
        '**both** error metrics, inside one allocation, on a frozen Burgers $256^2$ checkpoint? '
        f'**The redirected criterion {state}.** These numbers are final for this cell: every one '
        'is recomputed in an independent NumPy audit from saved output fields, and nothing in '
        'this report is typed by hand.\n')
    L.append(
        f'Primary job `{aud["job_id"]}` (`qtd02`), source `{aud["commit"]}`, GPU '
        f'`{aud["gpu"]}`, elapsed {aud["elapsed_seconds"]:.1f} s; JAX backend gpu, float64, '
        'matmul precision `highest`. Predeclared protocol, the redirect and every amendment: '
        '[`DESIGN.md`](../DESIGN.md).\n')

    # ------------------------------------------------------------ the redirect
    L.append('## Why this is the question\n')
    L.append(
        'This lane was opened to ask whether correction directions fitted to the ROM\'s '
        '**trajectory** error, rather than the head\'s **static** reconstruction residual, would '
        'remove the $q=16$ regression on the worst-over-evolved-times metric. While its job was '
        'running, the `q-diag` lane answered the underlying question from data that already '
        'existed: **the regression is the empirical quadrature, not the directions** — every '
        'dense ladder it measured is monotone on both metrics and only empirical-quadrature '
        'ladders regress. The lane was redirected to certify the arm that does not depend on the '
        'quadrature at all, and the trajectory-directions job was let finish as a **control** of '
        'that verdict. The full redirect is section 11 of `DESIGN.md`.\n')

    # ------------------------------------------------------------- the verdict
    L.append('## The verdict\n')
    L.append(table(['redirected criterion (`' + v['ladder'] + '`)', 'holds', 'measured'],
                   [['1. worst **evolved** error non-increasing in $q$, every rung converged',
                     yn(v['criterion_1_evolved_monotone_every_rung_converged']),
                     f"monotone {yn(v['monotone_evolved'])}, all rungs converged "
                     f"{yn(v['all_converged'])}"],
                    ['2. converged non-dominated set spans $\\ge2\\times$ in **error** and '
                     '$\\ge2\\times$ in **cost** on the evolved metric',
                     yn(v['criterion_3_passes']),
                     f"{v['criterion_3_error_span']['points']} points, error span "
                     f"{f(v['criterion_3_error_span']['error_span'], 3)}x, cost span "
                     f"{f(v['criterion_3_error_span']['cost_span'], 3)}x"],
                    ['(reported, not required) worst **all-times** error non-increasing in $q$',
                     yn(v['criterion_2_all_times_monotone']), '—']]))
    L.append(f'**Overall: the redirected criterion {state}.**\n')
    L.append('Converged non-dominated arms on the evolved metric: '
             + ', '.join(f'`{n}`' for n in v['converged_nondominated_evolved']) + '.\n')
    # What the criterion does NOT say, stated from the frontier rather than from prose.
    fams = {by_arm[n]['family'] for k in ('all_subjects_all_times', 'all_subjects_evolved')
            for n in aud['frontier'][k]}
    L.append(
        '\n**What this does not say.** The criterion is about the ladder: it certifies that '
        '$q$ is a real accuracy/cost dial on a frozen checkpoint. It is not a claim that the '
        'reduced solver beats the full-order one. Over EVERY subject in this job the '
        'non-dominated set on both metrics contains only '
        + ', '.join(sorted(f'`{x}`' for x in fams)) + ' arms'
        + (' — that is, only the same-job full-order controls, exactly as the campaign\'s '
           'standing conclusion says' if fams == {'fom'} else '')
        + '; the tables below give it in full.\n')
    if v['regressions_evolved']:
        L.append('\nMonotonicity violations on the evolved metric that survive:\n\n')
        L.append(table(['from $q$', 'to $q$', 'from %', 'to %'],
                       [[x['from_q'], x['to_q'], pct(x['from_percent']), pct(x['to_percent'])]
                        for x in v['regressions_evolved']]))

    # ------------------------------------------------- Part 1, existing data --
    L.append('\n## Part 1 — the dense ladder in data that already existed (no GPU)\n')
    L.append(
        'Four already-collected, checksum-verified, independently audited jobs were restored '
        'from their Git-tracked archive chunks and every same-grid error was recomputed in NumPy '
        'from the saved fields. This needed no new job and it is what turned the redirect from a '
        'hypothesis into a measurement.\n\n')
    L.append(table(['job', 'Slurm id', 'GPU', 'per-step budget',
                    'reference recomputation (relative)'],
                   [[f'`{k}`', d['job_id'], f"`{d['gpu']}`", d['step_budget'] or '—',
                     sci(d['reference_recomputation_relative'])]
                    for k, d in sorted(arc['jobs'].items())]))
    L.append('\nAll four jobs share one evaluation cohort (`all_jobs_share_one_cohort` '
             f"{yn(arc['checks']['all_jobs_share_one_cohort']['passed'])}).\n")
    L.append('\n**Every dense ladder in the archives, both metrics:**\n\n')
    L.append(table(['ladder', 'rungs', 'monotone evolved', 'monotone all times',
                    'every rung converged', 'converged non-dominated points',
                    'evolved error span', 'cost span', 'passes redirected criterion'],
                   [[f'`{lid}`', ', '.join(str(u['q']) for u in d['rungs']),
                     yn(d['monotone_evolved']), yn(d['monotone_all_times']),
                     yn(d['all_converged']), d['span_evolved']['points'],
                     f(d['span_evolved']['error_span'], 3), f(d['span_evolved']['cost_span'], 3),
                     yn(d['passes_redirected_criterion'])]
                    for lid, d in sorted(arc['ladders'].items())]))
    nmono = sum(1 for d in arc['ladders'].values()
                for k in ('monotone_evolved', 'monotone_all_times') if d[k])
    L.append(f'\n**{nmono} of {2 * len(arc["ladders"])}** dense ladder/metric combinations in the '
             'archives are monotone, independently reproducing the `q-diag` census from the '
             'fields rather than from its tables.\n')
    for lid, d in sorted(arc['ladders'].items()):
        L.append(f'\n**`{lid}`** — {d["label"]}.\n\n')
        L.append(table(['$q$', '$M$', 'worst all times %', 'worst evolved %', '$t_0$ %',
                        'median GPU ms', 'budget exits', 'worst joint gradient', 'converged'],
                       [[u['q'], u['M'], pct(u['all_times']), pct(u['evolved']), pct(u['t0']),
                         f(u['gpu_ms'], 2), u['budget_exits'],
                         sci(u['max_joint_gradient']), yn(u['converged'])] for u in d['rungs']]))
    if arc['budget600_dense_substitutes']:
        L.append('\nThe one dense rung in the archives that was rerun at per-step budget 600 '
                 '(`btq102`), which is why the redirected job runs every rung at 600:\n\n')
        L.append(table(['arm', '$q$', '$M$', 'worst all times %', 'worst evolved %',
                        'median GPU ms', 'budget exits', 'converged'],
                       [[f"`{x['arm']}`", x['q'], x['M'], pct(x['all_times']), pct(x['evolved']),
                         f(x['gpu_ms'], 2), x['budget_exits'], yn(x['converged'])]
                        for x in arc['budget600_dense_substitutes']]))

    # ---------------------------------------------------- Part 2, the new job -
    L.append('\n## Part 2 — one allocation, every dense rung at budget 600\n')
    L.append(f'Job `{aud["job_id"]}` on `{aud["gpu"]}`: two dense ladders, '
             f'$q\\in\\{{{", ".join(str(q) for q in cfg["q_ladder"])}\\}}$, the same six opened '
             'development cases, three timed repetitions with burn-in in a seeded randomized '
             'subject order, all repetitions retained. No empirical quadrature is fitted '
             'anywhere in this job.\n')
    ladder_block(L, lads, ['dense_m4', 'dense_fixedM'])

    L.append('\n### Worst-over-cases error at each output time\n')
    for lid in ('dense_m4', 'dense_fixedM'):
        lad = lads.get(lid)
        if not lad:
            continue
        L.append(f'\n**`{lid}`**\n\n')
        L.append(table(['$q$'] + [f'$t={t:g}$ %' for t in times],
                       [[u['q']] + [pct(x) for x in u['per_time']] for u in lad['rungs']]))

    L.append('\n### Same-job full-order controls\n')
    L.append(table(['control', 'ntol', 'ltol', '$\\Delta t$', 'worst all times %',
                    'worst evolved %', 'median GPU ms'],
                   [[f"`{s['name']}`", sci(s['ntol']), sci(s['ltol']), s['dt'],
                     pct(by_arm[s['name']]['worst_all_times_percent']),
                     pct(by_arm[s['name']]['worst_evolved_percent']),
                     f(by_arm[s['name']]['median_gpu_ms'], 3)]
                    for s in cfg['fom_settings'] if s['name'] in by_arm]))

    L.append('\n### Non-dominated sets over (median GPU ms, error)\n')
    for key, label in (('all_subjects_all_times', 'every subject, all output times'),
                       ('all_subjects_evolved', 'every subject, evolved times only'),
                       ('converged_rom_all_times', 'converged ROM arms, all output times'),
                       ('converged_rom_evolved', 'converged ROM arms, evolved times only')):
        L.append(f'\n**{label}**\n\n')
        L.append(table(['arm', 'family', '$q$', 'worst all times %', 'worst evolved %',
                        'median GPU ms', 'converged'],
                       [[f'`{n}`', by_arm[n]['family'],
                         by_arm[n]['q'] if by_arm[n]['q'] is not None else '—',
                         pct(by_arm[n]['worst_all_times_percent']),
                         pct(by_arm[n]['worst_evolved_percent']),
                         f(by_arm[n]['median_gpu_ms'], 3), yn(by_arm[n]['converged'])]
                        for n in aud['frontier'][key]]))

    L.append('\n### Every arm of the primary job\n')
    L.append(table(['arm', '$q$', '$M$', 'all times %', 'evolved %', '$t_0$ %', 'reference %',
                    'median GPU ms', 'median iters/step', 'budget exits', 'converged'],
                   [[f"`{x['arm']}`", x['q'] if x['q'] is not None else '—', x['M'] or '—',
                     pct(x['worst_all_times_percent']), pct(x['worst_evolved_percent']),
                     pct(x['worst_t0_compression_percent']), pct(x['worst_reference_percent']),
                     f(x['median_gpu_ms'], 3), f(x['median_iterations'], 1),
                     x['total_budget_exits'] if x['total_budget_exits'] is not None else '—',
                     yn(x['converged'])] for x in aud['arms']]))

    L.append('\n### Cross-job fidelity of the primary job\n')
    fid = aud['checks']['cross_job_fidelity']['detail']
    L.append(table(['arm', 'reproduces', 'job', 'declared tolerance',
                    'worst relative difference', 'ours %', 'theirs %', 'passed'],
                   [[f'`{k}`', f"`{d['detail']['comparator']}`", d['detail']['job'],
                     sci(d['detail']['declared_tolerance']),
                     sci(d['detail']['worst_relative_difference']),
                     pct(d['detail']['ours_worst_same_grid_percent']),
                     pct(d['detail']['theirs_worst_same_grid_percent']), yn(d['passed'])]
                    for k, d in sorted(fid.items()) if isinstance(d.get('detail'), dict)]))

    L.append('\n### Gates of the primary job\n')
    L.append(gate_table(aud['checks']))
    L.append('\n' + ('Failed: ' + ', '.join(f'`{x}`' for x in aud['failed']) + '.\n'
                     if aud['failed'] else 'No blocking gate failed.\n'))

    # ------------------------------------------------------------- the control
    if control:
        cl = control['ladders']
        cv = control['verdict']
        cds = control['direction_sets']
        L.append('\n## The control — correction directions fitted to the trajectory error\n')
        L.append(
            f'Job `{control["job_id"]}` (`qtd01`) on `{control["gpu"]}`, source '
            f'`{control["commit"]}`, elapsed {control["elapsed_seconds"]:.1f} s, submitted '
            '**before** the `q-diag` verdict arrived and let finish rather than cancelled. It '
            'builds three direction sets that differ only in which residual matrix is '
            'decomposed — the incumbent static reconstruction residual, and the $q=0$ ROM\'s own '
            'trajectory error against the same-mesh converged full-order solve over 32 and over '
            '6 training trajectories — and runs each over the ladder. Its own pre-registered '
            f'pass (section 6 of `DESIGN.md`) **{"passes" if cv["passes"] else "does not pass"}**, '
            'and it is reported here as a test of `q-diag`\'s verdict rather than as this cell\'s '
            'subject.\n')
        L.append('\n**The three direction sets.**\n\n')
        L.append(table(['set', 'residual decomposed', 'rows', 'rank', 'orthonormality dev',
                        'fit s', '`directions_sha256`'],
                       [[f'`{k}`',
                         ('static reconstruction residual, 1024 snapshots' if k == 'old'
                          else f"trajectory error, "
                               f"{cds[k].get('cohort', {}).get('count', '—')} trajectories"),
                         cds[k]['residual_rows'], cds[k]['available_rank'],
                         sci(cds[k].get('orthonormality_deviation')), f(cds[k].get('seconds'), 1),
                         '`' + cds[k]['directions_sha256'][:16] + '…`']
                        for k in ('old', 'traj', 'prac') if k in cds]))
        cc = control['direction_comparison']['cross_capture']
        qs = sorted({u['q'] for d in cl.values() for u in d['rungs'] if u['q'] > 0})
        L.append('\n**Cross-capture** $\\kappa_q(P,C)=\\|\\tilde P\\tilde C_{:,1:q}\\|_F^2/'
                 '\\|\\tilde P\\|_F^2$, per cent of residual matrix $P$\'s whitened energy that '
                 'direction set $C$ reaches at rung $q$. The diagonal is each set\'s own POD '
                 'energy; the off-diagonal says how differently the two rules aim.\n\n')
        L.append(table(['residual $P$', 'directions $C$'] + [f'$q={q}$' for q in qs],
                       [[f'`{pn}`', f'`{cn}`']
                        + [f'{cc[pn][cn][str(q)] * 100:.2f}' for q in qs]
                        for pn in ('old', 'traj', 'prac') if pn in cc
                        for cn in ('old', 'traj', 'prac') if cn in cc.get(pn, {})]))
        pa = control['direction_comparison']['principal_angles']
        L.append('\n**Principal angles** between the direction subspaces (field metric, '
                 'degrees); `overlap` is $\\frac1q\\sum_i\\cos^2\\theta_i$, 1 for identical '
                 'subspaces and $q/R$ for a random pair.\n\n')
        L.append(table(['pair', '$q$', 'overlap', 'min', 'median', 'max'],
                       [[f'`{pair}`', q, f(d[q]['overlap'], 5), f(d[q]['angle_min_degrees'], 2),
                         f(d[q]['angle_median_degrees'], 2), f(d[q]['angle_max_degrees'], 2)]
                        for pair, d in sorted(pa.items()) for q in sorted(d, key=int)]))
        L.append('\n**Its ladders, both metrics.**\n')
        ladder_block(L, cl, ['old_primary', 'traj_primary', 'prac_primary', 'old_dense',
                             'traj_dense', 'prac_dense', 'old_fixedM', 'traj_fixedM'])
        L.append('\n**The $q=16$ evolved-metric regression, per ladder of the control.** If '
                 '`q-diag` is right, the regression appears on empirical-quadrature ladders and '
                 'not on dense ones, whichever direction rule drives them.\n\n')
        L.append(table(['ladder', 'directions', 'quadrature', '$q=0$ evolved %',
                        '$q=16$ evolved %', 'change (pp)', 'regression'],
                       [[f'`{lid}`', cl[lid]['dirset'], cl[lid]['quadrature'], pct(x['q0']),
                         pct(x['q16']), f"{x['delta_percentage_points']:+.4f}",
                         yn(x['regression'])]
                        for lid, x in sorted(cv['q16_regression'].items())]))
        L.append('\n**Gates of the control job.**\n\n')
        L.append(gate_table(control['checks']))
        L.append('\n' + ('Failed: ' + ', '.join(f'`{x}`' for x in control['failed']) + '.\n'
                         if control['failed'] else 'No blocking gate failed.\n'))
    else:
        L.append('\n## The control\n')
        L.append('The trajectory-directions job did not produce a collectable result, so there '
                 'is no control section. That is recorded rather than omitted.\n')

    # ----------------------------------------------------------- the q-diag lane
    L.append('\n## The lane this question came from\n')
    if a.qdiag and Path(a.qdiag).exists():
        L.append(f'`q-diag`\'s diagnosis is at `{a.qdiag}`, SHA256 '
                 f'`{hashlib.sha256(Path(a.qdiag).read_bytes()).hexdigest()}`. Its verdict — that '
                 'the $q=16$ evolved-metric regression is the empirical quadrature and not the '
                 'directions — is what redirected this lane, and Part 1 above reproduces its '
                 'dense-ladder census independently from the saved fields.\n')
    else:
        L.append('The `q-diag` lane had published no report when this one was generated.\n')

    L.append('\n## Figure\n')
    L.append(f'![dense ladder]({Path(a.figure).with_suffix(".png").name})\n')

    # -------------------------------------------------------------- glossary
    L.append('\n## Glossary\n')
    L.append(
        'Written for a reader who knows none of this project\'s vocabulary.\n\n'
        '- **ROM / reduced-order model.** A cheap surrogate solver: instead of solving the PDE '
        'for all $n=255^2$ interior grid values it solves for a handful of coefficients and '
        'reconstructs the field from them.\n'
        '- **FOM / full-order model.** The real solver on the real grid. `fft_tight` is the '
        'converged one and is the yardstick every error here is measured against; `fft_loose` '
        'and `nt1e-2_dt01` are deliberately under-solved cheap ones, present so the ROM is '
        'never only compared against an over-solved competitor.\n'
        '- **The bank $G$.** A fixed matrix of $R=512$ spatial fields produced once by the '
        'frozen neural checkpoint. Everything the ROM can represent is a combination of them.\n'
        '- **The head $h_\\theta$.** The frozen neural map from a $K=16$-dimensional latent code '
        '$z$ to the 512 bank coefficients. It is never retrained anywhere in this cell.\n'
        '- **$q$, the correction directions.** Extra bank directions $C_q$ solved alongside the '
        'latent code: $\\eta = h_\\theta(z) + C_q y$. $q=0$ is the plain frozen model; larger $q$ '
        'gives the solver more freedom and costs more per step. **Rung** = one value of $q$; '
        '**ladder** = one ordered sweep of $q$ at fixed everything else.\n'
        '- **Knob / tunability.** The claim being certified: that turning one dial at inference '
        'time, with the trained weights untouched, buys a real range of accuracy for a real '
        'range of cost. Made precise here as the criterion in "The verdict".\n'
        '- **Dense vs empirical quadrature (EQ).** The nonlinear advection term is exactly a sum '
        'over all $n$ grid points ("dense"); an empirical-quadrature rule replaces it with a '
        'weighted sum over $m$ chosen points, fitted offline. EQ is where the ROM\'s speed comes '
        'from — and, per the `q-diag` lane, where the non-monotonicity came from. This report\'s '
        'primary job uses **no EQ at all**.\n'
        '- **$M$, the test count.** How many test functions the weak residual is measured '
        'against. `M = 4(K+q)` grows with $q$; the fixed-$M$ ladder holds it at 256 so the '
        'effect of $q$ is separated from the effect of $M$. $M>K+q$ is required, which is why '
        'the fixed-$M$ ladder stops before $q=256$.\n'
        '- **Same-grid error.** Distance from the same-job converged full-order solve on the '
        'same grid, divided by the norm of the supplied initial field. **Reference error** is '
        'the same against a 4096-interval solve, which also contains the 256-grid\'s own '
        'discretisation error and is therefore larger and less discriminating.\n'
        '- **All times / evolved times / $t_0$ compression.** The six output times are '
        '$t=0,0.05,\\dots,0.25$. "All times" is the worst over all six; "evolved times" drops '
        '$t=0$; "$t_0$ compression" is the $t=0$ term alone — the error the decoder makes just '
        'reproducing the field it was handed, before any time stepping. The all-times metric is '
        'usually pinned by that term, which is why both are always reported.\n'
        '- **Worst / median.** Worst is over the six evaluation cases. Median GPU ms is the '
        'median over all retained timed repetitions (three per case, every one kept).\n'
        '- **Converged.** Every time step and the initial fit exited on a residual, tiny-step or '
        'gradient criterion — never on the iteration budget — AND the joint normalized gradient '
        'stayed at or below $10^{-6}$. An unconverged arm\'s error is still reported and '
        'labelled, but cannot count towards the criterion.\n'
        '- **Budget exit.** A time step that ran out of its per-step iteration budget (600 here, '
        '180 in the older archives) before meeting the stopping rule. **Joint normalized '
        'gradient** $\\|J^\\top r\\|/(\\|J\\|\\,\\|r\\|)$ is the scale-free test for "is this '
        'actually a stationary point".\n'
        '- **Best-found.** The smallest error any code on that rung\'s manifold can reach on a '
        'supplied field, found offline by multistart optimisation: the accuracy floor the online '
        'solver is chasing.\n'
        '- **Non-dominated set.** The arms that nothing else beats on both cost and error at '
        'once — the usable trade-off curve. **Error span / cost span** are the ratios between '
        'its extreme points, and are what the criterion measures.\n'
        '- **Direction rules (control only).** `old` decomposes the error the head makes fitting '
        'each training snapshot on its own; `traj` and `prac` decompose the error the $q=0$ ROM '
        'actually accumulates along a trajectory, over 32 and over 6 training trajectories. '
        '**Cross-capture** is how much of one rule\'s error energy the other rule\'s directions '
        'can represent at all; **principal angles / overlap** measure how far apart the two '
        'subspaces are (overlap 1 = identical, $q/R$ = as unrelated as a random pair).\n'
        '- **Gate.** A check that had to pass before any number here was allowed to count. '
        'Blocking gates decide; informational ones are probes, reported either way.\n')

    text = ''.join(x if x.endswith('\n\n') else x + '\n' for x in L)
    Path(a.out).write_text(text)
    print(a.out, hashlib.sha256(Path(a.out).read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
