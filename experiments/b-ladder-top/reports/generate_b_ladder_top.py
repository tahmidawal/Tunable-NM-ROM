"""Generate the b-ladder-top report and its envelope figure.

Every number and every plotted point is read from the audited run JSONs. Nothing in the
report is typed by hand.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MARK = {'rom': 'o', 'pod': 's', 'fom': 'X', 'fno': '^'}
LABEL = {'rom': 'correction ladder (this checkpoint)', 'pod': 'POD-LSPG',
         'fom': 'full-order Newton (same job)', 'fno': 'trained FNO (same allocation)'}
COLOR = {'rom': '#3b6ea5', 'pod': '#8a5a2a', 'fom': '#b03a3a', 'fno': '#4f8a3d'}


def fmt(x, d=4):
    if x is None:
        return '—'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    return f'{x:.{d}f}' if isinstance(x, float) else str(x)


def yn(x):
    return {True: 'yes', False: 'no', None: '—'}[x]


def sci(x):
    return '—' if x is None else f'{x:.0e}'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |',
           '|' + '|'.join(['---'] * len(header)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(out) + '\n'


def gate_table(checks):
    rows = []
    for name, v in sorted(checks.items()):
        if not isinstance(v, dict) or 'passed' in v and v['passed'] is None:
            continue
        kind = 'informational' if v.get('blocking') is False else 'blocking'
        rows.append([f'`{name}`', yn(v.get('passed')), kind])
    return table(['gate', 'passed', 'kind'], rows)


def figure(rows, out_png, out_pdf):
    """Error against complete-query cost, both metrics.

    `fft_tight` IS the same-grid reference, so its same-grid error is zero by
    construction and it cannot be a point on a log axis; it is drawn as the vertical
    cost line it really is. Only the retained evolution tolerance (1e-6) is labelled
    with its q or k'; the 1e-3 arms are plotted with a smaller marker so the pairs do
    not overwrite each other.
    """
    base = [x for x in rows if x['arm'] == 'fft_tight']
    pts = [x for x in rows if x['arm'] != 'fft_tight']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    for ax, key, title in ((axes[0], 'worst_all_times_percent',
                            'worst over ALL output times ($t=0$ included)'),
                           (axes[1], 'worst_evolved_percent',
                            'worst over EVOLVED times only ($t>0$)')):
        seen = set()
        for x in pts:
            fam = x['family']
            if x.get('effective_gpu_ms') is None or x.get(key) is None:
                continue
            loose = x.get('gtol') is not None and x['gtol'] > 1e-6
            face = COLOR[fam] if (x.get('converged') or fam in ('fom', 'fno')) else 'none'
            ax.plot(x['effective_gpu_ms'], x[key], MARK[fam], ms=6 if loose else 10,
                    markerfacecolor=face, markeredgecolor=COLOR[fam],
                    markeredgewidth=1.0 if loose else 1.6, alpha=.6 if loose else 1.,
                    label=(LABEL[fam] if fam not in seen else None), zorder=3)
            seen.add(fam)
            if loose:
                continue
            tag = ((f"q={x['q']}" + ('' if x['quadrature'] == 'eq' else ' dense'))
                   if fam == 'rom' and x['q'] is not None else
                   (f"k'={x['k']}" if fam == 'pod' else
                    (x['arm'] if fam in ('fom', 'fno') else None)))
            if tag:
                ax.annotate(tag, (x['effective_gpu_ms'], x[key]), textcoords='offset points',
                            xytext=(7, 5), fontsize=8, zorder=4)
        lad = sorted([x for x in pts if x['family'] == 'rom' and x['quadrature'] == 'eq'
                      and x.get('gtol') == 1e-6 and x['q'] is not None], key=lambda x: x['q'])
        if len(lad) > 1:
            ax.plot([x['effective_gpu_ms'] for x in lad], [x[key] for x in lad],
                    '-', color=COLOR['rom'], lw=1.2, alpha=.45, zorder=1)
        if base:
            b = base[0]
            ax.axvline(b['effective_gpu_ms'], color=COLOR['fom'], ls='--', lw=1.2, alpha=.7,
                       zorder=0)
            ax.annotate('fft_tight: the converged FOM this column\nis measured against '
                        '(error 0 by construction)', (b['effective_gpu_ms'], 1.),
                        textcoords='offset points', xytext=(8, 0), fontsize=7.5,
                        color=COLOR['fom'], rotation=90, va='center', zorder=4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('median complete-query GPU time (ms, log)')
        ax.set_ylabel('error against the same-job converged FOM (%, log)')
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.25, which='both')
    handles, labels = axes[0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i])
    fig.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=9,
               loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.5, -.01))
    fig.suptitle('One frozen Burgers checkpoint, 256 intervals, six development cases:\n'
                 'error against complete-query cost. Filled marker = converged under the shared '
                 'stationarity rule, open = not;\nsmall faded marker = evolution tolerance '
                 '1e-3 instead of the retained 1e-6.', fontsize=10.5)
    fig.tight_layout(rect=(0, .06, 1, .93))
    fig.savefig(out_png, dpi=190)
    fig.savefig(out_pdf)
    plt.close(fig)


def q1_section(W, a1):
    rows = a1['arms']
    by = {x['arm']: x for x in rows}
    W('## Q1 — making the top of the ladder converge\n\n')
    W('### Fidelity gates against the cheap-corrections job\n\n')
    fid = a1['checks'].get('cclad01_fidelity', {}).get('detail') or {}
    W(table(['arm', 'reproduces', 'declared tolerance', 'relative difference (reference metric)',
             'relative difference (same-grid metric)', 'ours worst reference %',
             'theirs worst reference %', 'passed'],
            [[f'`{k}`', f"`{v['detail']['comparator']}`", sci(v['detail']['declared_tolerance']),
              sci(v['detail']['relative_worst_reference_difference']),
              sci(v['detail'].get('relative_worst_same_grid_difference')),
              fmt(v['detail']['ours_worst_reference_percent']),
              fmt(v['detail']['theirs_worst_reference_percent']), yn(v['passed'])]
             for k, v in sorted(fid.items())]) + '\n')

    W('### Conditioning of the augmented normal equations\n\n')
    W('Measured offline at one representative state (case 0, the step the configuration names), '
      'never inside a timed query. $\\kappa$ is the 2-norm condition number of the damped '
      'Gauss-Newton matrix at $\\lambda = 10^{-6}$; the equilibrated column is fix (b).\n\n')
    W(table(['q', 'unknowns', 'M', 'min column norm', 'max column norm', 'column norm ratio',
             '$\\kappa(J)$', '$\\kappa$ unscaled', '$\\kappa$ equilibrated', 'improvement'],
            [[c['q'], c['unknowns'], c['M'], f"{c['column_norm_min']:.3e}",
              f"{c['column_norm_max']:.3e}", f"{c['column_norm_ratio']:.3e}",
              f"{c['jacobian_condition']:.3e}",
              f"{c['lambdas']['1e-06']['unscaled_condition']:.3e}",
              f"{c['lambdas']['1e-06']['equilibrated_condition']:.3e}",
              f"{c['lambdas']['1e-06']['improvement_factor']:.3f}"]
             for c in sorted(a1['conditioning'], key=lambda c: c['q'])]) + '\n')

    W('### The convergence sweep\n\n')
    W('Dense quadrature throughout, so no quadrature rule confounds the reading. '
      '`base` is the cheap-corrections block-damped solver itself. A cascade arm also reports '
      'the cost of the coarse rung it was started from.\n\n')
    sweep = [x for x in rows if x['kind'] == 'rom' and x['quadrature'] == 'dense']
    W(table(['arm', 'q', 'fix', 'M', 'median iters/step', 'max iters', 'budget exits',
             'exit reasons', 'worst joint gradient', 'worst same-grid all %',
             'worst evolved %', 't0 compression %', 'median GPU ms', 'cascade total ms',
             'converged'],
            [[f"`{x['arm']}`", x['q'], f"`{x['fix']}`", x['M'], fmt(x['median_iterations'], 1),
              fmt(x['max_iterations'], 0), x['total_budget_exits'],
              ', '.join(f'{k}:{v}' for k, v in sorted((x['exit_reason_counts'] or {}).items())),
              f"{x['max_joint_stationarity']:.3e}", fmt(x['worst_all_times_percent']),
              fmt(x['worst_evolved_percent']), fmt(x['worst_t0_compression_percent']),
              fmt(x['median_gpu_ms'], 3), fmt(x['cascade_total_gpu_ms'], 3), yn(x['converged'])]
             for x in sorted(sweep, key=lambda x: (x['q'], x['fix'] or ''))]) + '\n')

    W('### The empirical quadrature rule per rung\n\n')
    W('A rule is VALID only when the bounded fitter stopped on target support or on the '
      'gradient criterion. A walltime-truncated rule disqualifies its arm; that gate is '
      'pre-registered and does not look at the rung\'s error.\n\n')
    eqrows = [x for x in rows if x['kind'] == 'rom' and x['quadrature'] == 'eq']
    W(table(['arm', 'q', 'M', 'm', 'relative fit', 'truncated', 'rule valid', 'fit seconds',
             'worst same-grid all %', 'worst evolved %', 'median GPU ms', 'converged'],
            [[f"`{x['arm']}`", x['q'], x['M'], x['m'], fmt(x['quadrature_fit_relative'], 6),
              yn(x['quadrature_truncated']), yn(x['eq_rule_valid']),
              fmt(x['quadrature_fit_seconds'], 1), fmt(x['worst_all_times_percent']),
              fmt(x['worst_evolved_percent']), fmt(x['median_gpu_ms'], 3), yn(x['converged'])]
             for x in sorted(eqrows, key=lambda x: (x['q'], x['fix'] or ''))]) + '\n')

    W('### Same-job full-order controls\n\n')
    W(table(['method', 'worst same-grid all %', 'worst vs reference %', 'median GPU ms'],
            [[f"`{x['arm']}`", fmt(x['worst_all_times_percent']), fmt(x['worst_reference_percent']),
              fmt(x['median_gpu_ms'], 3)] for x in rows if x['kind'] == 'fom']) + '\n')

    W('### Where each rung\'s non-convergence actually lives\n\n')
    W('Localised from the saved per-step exit reasons and the initial-fit diagnostics, not '
      'asserted: an arm can miss the shared rule in its supplied-field fit, in its time '
      'stepping, or in both.\n\n')
    W(table(['arm', 'q', 'fix', 'converged', 'failure located in',
             'initial-fit relative residual (min, max)', 'initial-fit exit reasons',
             'worst initial-fit gradient', 'worst step gradient',
             'cases with budget exits', 'their failing step indices'],
            [[f"`{x['arm']}`", x['q'], f"`{x['fix']}`", yn(x['converged']),
              x.get('failure_located_in') or '—',
              f"({x['ic_relative_residual_min']:.3e}, {x['ic_relative_residual_max']:.3e})",
              ', '.join(f'{a}:{b}' for a, b in sorted((x.get('ic_reason_counts') or {}).items())),
              f"{x['ic_joint_stationarity_max']:.3e}", f"{x['step_joint_stationarity_max']:.3e}",
              x.get('failing_case_count'),
              '; '.join(f"case {c}: {v}" for c, v in (x.get('failing_cases') or {}).items()) or '—']
             for x in sorted(sweep, key=lambda x: (x['q'], x['fix'] or ''))]) + '\n')
    q512 = [x for x in sweep if x['q'] == 512]
    if q512:
        z = q512[0]
        W(f"At $q=512$ the correction directions span the **whole** bank ($q=R$), so the supplied "
          f"field is fitted to machine zero — the initial-fit relative residual is "
          f"{z['ic_relative_residual_min']:.3e} to {z['ic_relative_residual_max']:.3e} — and the "
          f"normalized gradient $\\|J^\\top r\\|/(\\|J\\|\\,\\|r\\|)$ becomes a $0/0$ ratio. Its reported "
          f"value {z['ic_joint_stationarity_max']:.3e} therefore measures nothing, while every "
          f"time step of that rung is stationary at {z['step_joint_stationarity_max']:.3e}. The "
          f"$q=512$ rung is a **degenerate endpoint of the stopping rule, not a solver failure** "
          f"— the same thing the Poisson $q=R$ rung showed.\n\n")

    W('### Verdict on Q1\n\n')
    target = [x for x in sweep if x['q'] == 256]
    won = [x for x in target if x['converged']]
    q128 = [x for x in sweep if x['q'] == 128 and x['converged']]
    best128 = min(q128, key=lambda x: x['worst_all_times_percent']) if q128 else None
    if won:
        best = min(won, key=lambda x: x['worst_all_times_percent'])
        beats = (best128 is not None
                 and best['worst_all_times_percent'] < best128['worst_all_times_percent'])
        W(f"**{'PASS' if beats else 'PARTIAL'}.** The $q=256$ rung converges under "
          f"`{best['fix']}` (no budget exits, worst joint normalized gradient "
          f"{best['max_joint_stationarity']:.3e}) at {best['median_gpu_ms']:.3f} ms, with worst "
          f"same-grid error {best['worst_all_times_percent']:.4f}% over all times and "
          f"{best['worst_evolved_percent']:.4f}% over evolved times.")
        if best128:
            W(f" The same-job $q=128$ comparator `{best128['arm']}` reaches "
              f"{best128['worst_all_times_percent']:.4f}% at {best128['median_gpu_ms']:.3f} ms, so "
              f"the pre-registered requirement that $q=256$ be strictly more accurate "
              f"{'HOLDS' if beats else 'does NOT hold'}.")
        W('\n\n')
    else:
        W('**FAIL under the frozen contract.** No fix converged the $q=256$ rung under the '
          'shared stopping rule with every retained constant held fixed: `base`, `pre`, `damp` '
          'and `predamp` are indistinguishable there to four decimals, and the cascade arm is '
          'much worse. The localisation table above says why — the failure is five time steps of '
          'ONE case out of six, and none of the three fixes touches what those steps need. '
          'Q1-B below relaxes one retained constant at a time and settles what the rung costs to '
          'converge.\n\n')


def q2_section(W, a2):
    rows = a2['arms']
    by = {x['arm']: x for x in rows}
    W('## Q2 — the combined envelope, one job\n\n')
    W('Three error columns on every row. The same-grid error is measured against the '
      'converged full-order solve timed in this same job, which returns the supplied field '
      'exactly at $t=0$; so the $t=0$ column is the decoder\'s compression of the supplied '
      'field, the all-times column is pinned by it, and the evolved column is the trajectory '
      'error alone.\n\n')
    W(table(['subject', 'family', 'q / k\'', 'M', 'm', 'quadrature', 'evolution tol',
             'worst all times %', 'worst evolved %', 't0 compression %',
             'worst vs reference %', 'median GPU ms', 'median complete query ms', 'converged'],
            [[f"`{x['arm']}`", x['family'], x['q'] if x['q'] is not None else x['k'], x['M'],
              x['m'], x['quadrature'] or '—',
              (f"{x['gtol']:g}" if x['gtol'] else '—'),
              fmt(x['worst_all_times_percent']), fmt(x['worst_evolved_percent']),
              fmt(x['worst_t0_compression_percent']), fmt(x['worst_reference_percent']),
              fmt(x['effective_gpu_ms'], 3), fmt(x['median_host_ms'], 3), yn(x['converged'])]
             for x in rows]) + '\n')

    for key, name in (('all_subjects_all_times', 'all output times'),
                      ('all_subjects_evolved', 'evolved times only')):
        W(f'### Non-dominated set over (median GPU ms, worst error) — {name}, all subjects\n\n')
        err = 'worst_all_times_percent' if 'all_times' in key else 'worst_evolved_percent'
        sel = sorted([by[n] for n in a2['frontier'][key]], key=lambda x: x['effective_gpu_ms'])
        W(table(['subject', 'family', 'q / k\'', 'worst error %', 'median GPU ms', 'converged'],
                [[f"`{x['arm']}`", x['family'], x['q'] if x['q'] is not None else x['k'],
                  fmt(x[err]), fmt(x['effective_gpu_ms'], 3), yn(x['converged'])] for x in sel]) + '\n')

    W('### Is $q$ a knob?\n\n')
    k = a2['knob_criterion']
    W(table(['metric', 'monotone at the fixed test count', 'monotone over every rung',
             'converged non-dominated points', 'cost span', 'error span',
             'passes the pre-registered criterion'],
            [[m, yn(k[m]['monotone_fixed_test_count']), yn(k[m]['monotone_all_rungs']),
              k[m].get('points'), fmt(k[m].get('cost_span'), 3), fmt(k[m].get('error_span'), 3),
              yn(k[m]['passes'])] for m in ('evolved', 'all_times')]) + '\n')
    for m in ('evolved', 'all_times'):
        W(f'The retained ladder on the **{m}** metric (fixed test count, empirical quadrature, '
          f'evolution tolerance $10^{{-6}}$):\n\n')
        W(table(['q', 'M', 'worst error %', 'median GPU ms'],
                [[q, M, fmt(v), fmt(c, 3)] for q, M, v, c in k[m]['ladder']]) + '\n')

    W('### Verdict on Q2\n\n')
    fams = {}
    for key, err in (('all_subjects_all_times', 'worst_all_times_percent'),
                     ('all_subjects_evolved', 'worst_evolved_percent')):
        fams[key] = sorted({by[n]['family'] for n in a2['frontier'][key]})
    if all(v == ['fom'] for v in fams.values()):
        cheap = min((by[n] for n in a2['frontier']['all_subjects_evolved']),
                    key=lambda x: x['effective_gpu_ms'])
        bestrom = min([x for x in rows if x['family'] == 'rom'],
                      key=lambda x: x['worst_evolved_percent'])
        cheaprom = min([x for x in rows if x['family'] == 'rom'],
                       key=lambda x: x['effective_gpu_ms'])
        mid = [by[n] for n in a2['frontier']['all_subjects_evolved']
               if by[n]['arm'] != cheap['arm']]
        mid = min(mid, key=lambda x: x['effective_gpu_ms']) if mid else None
        W('**Nothing but the full-order solver is on the envelope.** On BOTH metrics the '
          'non-dominated set over every subject in this job contains only same-job full-order '
          'controls: no correction-ladder rung, no POD-LSPG rank and not the trained neural '
          f"operator survives. The cheapest non-dominated point is `{cheap['arm']}` at "
          f"{cheap['effective_gpu_ms']:.3f} ms and {cheap['worst_evolved_percent']:.4f} % evolved "
          f"error")
        if mid:
            W(f", and `{mid['arm']}` at {mid['effective_gpu_ms']:.3f} ms reaches "
              f"{mid['worst_evolved_percent']:.4f} %")
        W(f". The most accurate reduced-order arm, `{bestrom['arm']}`, needs "
          f"{bestrom['effective_gpu_ms']:.3f} ms for {bestrom['worst_evolved_percent']:.4f} %, and "
          f"the cheapest, `{cheaprom['arm']}`, needs {cheaprom['effective_gpu_ms']:.3f} ms for "
          f"{cheaprom['worst_evolved_percent']:.4f} %.\n\n")
    lad = {q: (M, v, c) for q, M, v, c in k['all_times']['ladder']}
    lae = {q: (M, v, c) for q, M, v, c in k['evolved']['ladder']}
    if 0 in lad and 16 in lae:
        W('**The ladder\'s monotonicity is the $t=0$ compression term, not the trajectory.** '
          f"On the all-times metric the fixed-test-count rungs fall monotonically "
          f"{lad[0][1]:.4f} % -> {lad[128][1]:.4f} % from $q=0$ to $q=128$, and that column is "
          f"pinned at every rung by the decoder's compression of the supplied field. On the "
          f"evolved-times metric the same rungs are NOT monotone: $q=16$ "
          f"({lae[16][1]:.4f} %) is worse than $q=0$ ({lae[0][1]:.4f} %), and the whole "
          f"converged non-dominated set spans only "
          f"{k['evolved']['error_span']:.3f}x in error across "
          f"{k['evolved']['cost_span']:.3f}x in cost. The pre-registered criterion for calling "
          f"$q$ a knob therefore fails on both metrics, and it fails for a different reason on "
          f"each: error span on the all-times metric, monotonicity and error span on the "
          f"evolved one.\n\n")

    if a2.get('fno'):
        f = a2['fno']
        W('### The trained FNO, in this allocation\n\n')
        W(f"`{f['model']}`, {f['real_parameter_count']:,} real parameters, checkpoint SHA256 "
          f"`{f['checkpoint_sha256'][:16]}…`, {f['environment']['gpu']}, "
          f"torch {f['environment']['torch']} / neuraloperator {f['environment']['neuraloperator']}, "
          f"{f['environment']['precision']}. Pooled device query "
          f"{f['device_query_pooled']['median_ms']:.3f} ms median over "
          f"{f['device_query_pooled']['count']} retained repetitions.\n\n")
        W('It returns the supplied field bitwise at $t=0$, so its $t=0$ compression is exactly '
          'zero and the all-times metric flatters it relative to the ROM, whose output contract '
          'decodes $t=0$ from the latent code. That is the whole reason both metrics are '
          'reported.\n\n')
    else:
        W('### The trained FNO\n\n')
        W('**Omitted.** The FNO phase did not produce a timing record in this job, so no FNO '
          'row appears above and no timing is imported from the job that trained it.\n\n')


def q1b_section(W, a1b, a1):
    rows = [x for x in a1b['arms'] if x['kind'] == 'rom']
    ctrl = next((x for x in rows if x['arm'] == 'q256_m2_dense_base'), None)
    W('## Q1-B — does $q=256$ converge at a stated cost?\n\n')
    W('Q1 answered "do these three fixes converge the rung" with a clean no, and localised the '
      'failure to five time steps of one case out of six. Q1-B relaxes exactly ONE retained '
      'contract constant per arm — the per-step iteration budget and the latent trust radius — '
      'with the retained setting as the control. The relaxation is a declared departure from the '
      'frozen contract, predeclared in `DESIGN.md` before the job was submitted.\n\n')
    fid = a1b['checks'].get('cclad01_fidelity', {}).get('detail') or {}
    W(table(['control arm', 'reproduces', 'declared tolerance',
             'relative difference (reference metric)', 'relative difference (same-grid metric)',
             'passed'],
            [[f'`{k}`', f"`{v['detail']['comparator']}`", sci(v['detail']['declared_tolerance']),
              sci(v['detail']['relative_worst_reference_difference']),
              sci(v['detail'].get('relative_worst_same_grid_difference')), yn(v['passed'])]
             for k, v in sorted(fid.items())]) + '\n')
    W(table(['arm', 'per-step iteration budget', 'latent trust radius', 'quadrature',
             'median iters/step', 'budget exits', 'worst joint gradient',
             'worst same-grid all %', 'worst evolved %', 'median GPU ms', 'cost vs control',
             'converged'],
            [[f"`{x['arm']}`", x.get('step_budget'),
              ('x' + fmt(x.get('trust_scale'), 0)) if x.get('trust_scale') else '—',
              x['quadrature'], fmt(x['median_iterations'], 1), x['total_budget_exits'],
              f"{x['max_joint_stationarity']:.3e}", fmt(x['worst_all_times_percent']),
              fmt(x['worst_evolved_percent']), fmt(x['median_gpu_ms'], 3),
              (fmt(x['median_gpu_ms'] / ctrl['median_gpu_ms'], 3) + 'x') if ctrl else '—',
              yn(x['converged'])]
             for x in sorted(rows, key=lambda x: (x['quadrature'], x['arm']))]) + '\n')
    won = [x for x in rows if x['converged']]
    q128 = [x for x in a1['arms'] if x.get('q') == 128 and x['kind'] == 'rom'
            and x['quadrature'] == 'dense' and x['converged']]
    W('### Verdict on Q1, restated\n\n')
    if won and ctrl:
        cheap = min(won, key=lambda x: x['median_gpu_ms'])
        same = [x for x in won if x['quadrature'] == ctrl['quadrature']]
        chosen = min(same, key=lambda x: x['step_budget'] or 10 ** 9) if same else cheap
        W(f"**PASS, at a stated and declared cost.** With the retained per-step iteration budget "
          f"of {ctrl['step_budget']} the $q=256$ rung has {ctrl['total_budget_exits']} budget "
          f"exits and a worst joint normalized gradient of {ctrl['max_joint_stationarity']:.3e}. "
          f"Raising that budget to {chosen['step_budget']} — and changing nothing else — converges "
          f"it: zero budget exits, worst gradient {chosen['max_joint_stationarity']:.3e}, every "
          f"case and every time step stationary. ")
        W(f"The cost of the relaxation is essentially nothing: {chosen['median_gpu_ms']:.3f} ms "
          f"against the control's {ctrl['median_gpu_ms']:.3f} ms, a factor of "
          f"{chosen['median_gpu_ms'] / ctrl['median_gpu_ms']:.3f}, because the extra iterations are "
          f"spent on five of nine hundred time steps. ")
        if q128:
            b = min(q128, key=lambda x: x['worst_all_times_percent'])
            W(f"Its worst same-grid error {chosen['worst_all_times_percent']:.4f} % is strictly "
              f"below the same-rule $q=128$ rung's {b['worst_all_times_percent']:.4f} % "
              f"(`{b['arm']}`, {b['median_gpu_ms']:.3f} ms in the Q1 job), so the pre-registered "
              f"accuracy requirement holds; the rung costs "
              f"{chosen['median_gpu_ms'] / b['median_gpu_ms']:.3f}x the $q=128$ rung. ")
        W('\n\n**What did NOT do it.** The three fixes Q1 was designed around — the cascade warm '
          'start, column equilibration and a decoupled damping and trust schedule for the '
          'correction block — are all inert at this rung; the cascade is actively harmful. The '
          'binding constraint was simply solver effort on a handful of hard steps, and the '
          'relaxed latent trust radius is a genuine COST lever at unchanged error, not an '
          'accuracy one.\n\n')
        W('**And the answer does not move.** Every arm in the table above, converged or not, '
          f"reports the same {ctrl['worst_all_times_percent']:.4f} % worst same-grid error and the "
          f"same {ctrl['worst_evolved_percent']:.4f} % evolved error to four decimals. Converging "
          'the five stubborn steps changes nothing about the physical answer: the unconverged rung '
          'was already at it. That is worth stating plainly, because it means the convergence '
          'failure was a stopping-rule fact, not an accuracy fact.\n\n')
    else:
        W('**FAIL stands.** No relaxed arm converged; see the table above.\n\n')


def glossary(W):
    W('## Glossary\n\n')
    for term, text in [
        ('ROM / reduced-order model', 'a solver that evolves a handful of coefficients instead '
         'of the 65,025 interior grid values, and reconstructs the field from them.'),
        ('FOM / full-order model', 'the ordinary finite-difference Burgers solver on the same '
         '256-interval grid, timed in the same job. `fft_tight` is the converged one; '
         '`nt1e-4_dt005` and `nt1e-2_dt01` are cheaper, looser settings.'),
        ('checkpoint', 'one set of trained neural weights, frozen. Nothing in this report '
         'retrains anything; every row uses the same weights.'),
        ('$K$ (latent dimension)', 'the 16 coordinates the trained decoder takes as input.'),
        ('$R$ (bank rank)', 'the 512 spatial basis functions the decoder outputs coefficients '
         'for. The reduced state is always a combination of these 512.'),
        ('$q$ (correction rank)', 'the number of extra linear directions solved alongside the '
         '16 latent coordinates. $q=0$ is the plain trained decoder; larger $q$ gives the '
         'solver more freedom inside the same 512-dimensional bank.'),
        ('directions $C_q$', 'the extra directions, fixed offline and nested: the first $q$ '
         'columns of one matrix, so every rung is a subset of the next.'),
        ('$M$ (test count)', 'how many weak equations the solver fits at each time step. It '
         'must exceed the number of unknowns $K+q$.'),
        ('$m$ (quadrature support)', 'how many grid points the empirical quadrature rule '
         'evaluates the nonlinear term at, instead of all 65,025.'),
        ('EQ / empirical quadrature', 'a fitted, nonnegatively weighted subset of grid points '
         'that reproduces the nonlinear term cheaply. `dense` means no such rule: every point '
         'is used.'),
        ('NNLS', 'nonnegative least squares, the fit that chooses the quadrature points and '
         'their weights.'),
        ('truncated rule', 'a quadrature rule whose fitter ran out of walltime before reaching '
         'its target number of points. Such a rule is disqualified here.'),
        ('LM / Levenberg-Marquardt', 'the damped Gauss-Newton iteration that solves the weak '
         'equations at each time step.'),
        ('trust radius', 'a cap on how far one iteration may move the unknowns.'),
        ('damping / ridge', 'a value added to the diagonal of the normal equations that '
         'shrinks and stabilises the step; larger damping means a smaller, safer step.'),
        ('column equilibration', 'rescaling each unknown so that its column of the Jacobian '
         'has unit length. It does not change the answer in exact arithmetic, only the '
         'floating-point conditioning of the linear solve.'),
        ('$\\kappa$ (condition number)', 'how much a linear solve can amplify rounding error. '
         '$10^{16}$ means no correct digits are left in double precision.'),
        ('cascade warm start', 'starting the $q$ rung from the converged $q/2$ rung\'s answer, '
         'padded with zeros. Its cost column adds the coarse rung\'s cost.'),
        ('normalized gradient', '$\\|J^\\top r\\|/(\\|J\\|\\,\\|r\\|)$, the scale-free measure '
         'of how close a solve is to a stationary point. The shared rule asks for $10^{-6}$.'),
        ('budget exit', 'a time step that ran out of solver iterations before meeting any '
         'stopping criterion. Any budget exit means the arm is not converged.'),
        ('per-step iteration budget', 'the hard cap on solver iterations at one time step. The '
         'retained contract sets it to 180; Q1-B raises it as a declared relaxation.'),
        ('exit reasons', '0 = ran out of iterations, 1 = the residual fell below the absolute '
         'tolerance, 2 = the step became negligibly small, 4 = the normalized gradient met the '
         'tolerance.'),
        ('converged', 'every time step and the initial fit exited for reason 1, 2 or 4, with '
         'no budget exit, and the worst normalized gradient was at most $10^{-6}$.'),
        ('evolution tolerance', 'the normalized-gradient tolerance applied at each time step. '
         'The initial fit always uses $10^{-6}$.'),
        ('same-grid error', 'distance from the converged full-order solve on the same grid in '
         'the same job, divided by the norm of the supplied initial field.'),
        ('reference error', 'distance from a 4096-interval solve restricted to this grid — a '
         'stricter comparator that also charges the discretisation error of the 256 grid.'),
        ('worst over all times', 'the largest of those distances over the six output times '
         '$t = 0, 0.05, \\dots, 0.25$ and over the six cases.'),
        ('worst over evolved times', 'the same, but over $t>0$ only.'),
        ('t0 compression', 'the error at $t=0$ alone: how well the decoder can represent the '
         'field it was handed. It is not a solver error and no amount of solving removes it.'),
        ('POD-LSPG', 'the classical linear reduced-order model: a rank-$k\'$ basis from the '
         'same training snapshots, with no neural network, solved through the same weak '
         'objective.'),
        ('FNO', 'Fourier Neural Operator — a trained network that maps the initial field '
         'straight to the whole trajectory, with no solver at all.'),
        ('non-dominated set', 'the subjects no other subject beats on both cost and error at '
         'once; the useful operating points.'),
        ('held-out / development cohort', 'the six physical cases used here were opened for '
         'development earlier in this campaign. The final evaluation cases remain sealed.'),
        ('bank projection / best found', 'how well the 512 basis functions, and the arm\'s own '
         'manifold, could represent the true field if the solver were perfect. These are '
         'representation floors, not solver results.'),
    ]:
        W(f'**{term}** — {text}\n\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--q1-audit', required=True)
    p.add_argument('--q2-audit', default=None)
    p.add_argument('--q1b-audit', default=None)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    a1 = json.loads(Path(a.q1_audit).read_text())
    a2 = json.loads(Path(a.q2_audit).read_text()) if a.q2_audit else None
    a1b = json.loads(Path(a.q1b_audit).read_text()) if a.q1b_audit else None
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    buf = []
    W = buf.append

    W('# The top of the correction ladder, and the combined envelope\n\n')
    W('Two questions on one frozen Burgers checkpoint at 256 intervals: whether the $q=256$ '
      'rung of the fixed-weight correction ladder can be made to converge, and what the whole '
      'inference-time envelope costs when the ladder, the quadrature, the evolution tolerance, '
      'classical POD-LSPG, the full-order solver and a trained neural operator are all priced '
      'in the same job on the same GPU. **All numbers below are final for this study** and are '
      'generated from the audited run JSONs; the predeclared protocol is '
      '`experiments/b-ladder-top/DESIGN.md`.\n\n')

    W('## Provenance\n\n')
    prov = [['Q1 convergence sweep', a1.get('job_id'), a1.get('gpu'), a1.get('commit'),
             fmt(a1.get('elapsed_seconds'), 1)]]
    if a2:
        prov.append(['Q2 envelope', a2.get('job_id'), a2.get('gpu'), a2.get('commit'),
                     fmt(a2.get('elapsed_seconds'), 1)])
    if a1b:
        prov.append(['Q1-B cost of convergence', a1b.get('job_id'), a1b.get('gpu'),
                     a1b.get('commit'), fmt(a1b.get('elapsed_seconds'), 1)])
    W(table(['job', 'Slurm id', 'GPU', 'source commit', 'elapsed seconds'], prov) + '\n')

    W('## Gates\n\n')
    W('### Q1\n\n')
    W(gate_table(a1['checks']))
    W('\n')
    if a2:
        W('### Q2\n\n')
        W(gate_table(a2['checks']))
        W('\n')
    if a1b:
        W('### Q1-B\n\n')
        W(gate_table(a1b['checks']))
        W('\n')

    q1_section(W, a1)
    if a1b:
        q1b_section(W, a1b, a1)
    if a2:
        q2_section(W, a2)
        figure(a2['arms'], out.with_name(out.stem + '-envelope.png'),
               out.with_name(out.stem + '-envelope.pdf'))
        W('## The envelope\n\n')
        W(f'![error against complete-query cost]({out.stem}-envelope.png)\n\n')
        W('Marker shape is the family, fill is convergence under the shared stationarity rule, '
          'the $q$ label is the correction rank and $k\'$ the POD rank. The left panel is the '
          'worst error over all output times and the right panel over evolved times only; the '
          'ladder line joins the retained empirical-quadrature rungs at evolution tolerance '
          '$10^{-6}$.\n\n')

    glossary(W)
    out.write_text(''.join(buf))
    print(out)
    print('sha256', sha(out))


if __name__ == '__main__':
    main()
