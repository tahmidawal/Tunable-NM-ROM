"""jcp-regimes: generate report.md, plots/*.png and runs/EXPORT.json from checks/analysis.json, checks/smoke-accept.json
and runs/GPU-LEDGER.json. Every number in the report comes from those files (CLAUDE.md: never hand-type a number).

    JAX_PLATFORMS=cpu python make_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AN = json.loads((HERE / 'checks' / 'analysis.json').read_text())
SM = json.loads((HERE / 'checks' / 'smoke-accept.json').read_text())
LED = json.loads((HERE / 'runs' / 'GPU-LEDGER.json').read_text())
REG = ('Ap', 'Am', 'C1', 'C2', 'B1', 'B2')
NAMES = dict(N0='N0 (fresh in-range)', N0s='N0s (stiff part of the training box)', Ap='A+ (ν above range)',
             Am='A− (ν below range)', C1='C1 (low ν, high s, paired with A−)', C2='C2 (lower ν, higher s)',
             B1='B1 (two same-sign bumps)', B2='B2 (signed pair)')
COND = 'conditional E0-style estimate'


def pct(x, d=3):
    return '—' if x is None else f'{100 * x:.{d}g} %'


def num(x, d=3):
    return '—' if x is None else f'{x:.{d}g}'


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = HERE / 'plots'
    out.mkdir(exist_ok=True)
    cfgs = [k for k, v in AN['configs'].items() if not v['diagnostic']]
    # per-regime median error per configuration
    fig, ax = plt.subplots(figsize=(9, 4.5))
    regs = ('N0', 'N0s') + REG
    for k in cfgs:
        c = AN['configs'][k]
        ys = [c['baseline']['median'] if g == 'N0' else c['cells'].get(g, {}).get('median') for g in regs]
        ax.plot(range(len(regs)), [np.nan if y is None else y for y in ys], marker='o', label=k)
    ax.set_xticks(range(len(regs)), [g if g != 'N0' else 'in-range I' for g in regs])
    ax.set_yscale('log'); ax.set_ylabel('median error (eligible cases)'); ax.legend(fontsize=7, ncol=2)
    ax.set_title('Median error by regime (E0-style, conditional)')
    fig.tight_layout(); fig.savefig(out / 'median_by_regime.png', dpi=140); plt.close(fig)
    # error vs distance for A+/A-, and vs s for A-/C1/C2
    import rcohorts as RG
    for key, cohs, xf, xl, fn in (('d_nu', ('Ap', 'Am'), lambda d: d['d_nu'] * (1 if d['nu'] > .1 else -1), 'signed decades outside [0.01, 0.1] (− = below)', 'error_vs_nu.png'),
                                  ('s', ('N0', 'N0s', 'Am', 'C1', 'C2'), lambda d: d['s'], 's = a/ν', 'error_vs_s.png')):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for k in ('R384|LSPG_CNR_1', 'R128|LSPG_BE_1'):
            c = AN['configs'][k]
            xs, ys = [], []
            for cid, x in c['per_case'].items():
                coh, j = cid.split('|')
                if coh in cohs and x['e'] and x['valid'] and x['eligible']:
                    xs.append(xf(RG.desc_2d(coh, RG.descriptors_2d(coh)[int(j)]))); ys.append(x['e'])
            ax.scatter(xs, ys, s=14, label=k)
        ax.set_yscale('log'); ax.set_xlabel(xl); ax.set_ylabel('error')
        if key == 's':
            ax.set_xscale('log'); ax.axvline(200, color='grey', ls='--', lw=.8)
        else:
            ax.axvline(0, color='grey', ls='--', lw=.8)
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / fn, dpi=140); plt.close(fig)
    # sigma per regime
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, g in enumerate(('N0', 'N0s') + REG):
        v = [x['sigma']['sigma'] for x in AN['sigma'].get(g, []) if x['sigma'] and x['sigma']['sigma']]
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-.15, .15, len(v)), v, s=10)
    ax.set_xticks(range(8), ('N0', 'N0s') + REG); ax.set_yscale('log'); ax.set_ylabel('σ_a (output-time interior gradient / a)')
    fig.tight_layout(); fig.savefig(out / 'sigma_by_regime.png', dpi=140); plt.close(fig)


def esc(x):
    """Markdown table cell: escape the pipe (configuration ids contain '|')."""
    return str(x).replace('|', '\\|')


def row(cells):
    return '| ' + ' | '.join(esc(c) for c in cells) + ' |'


def cell_text(x):
    """A primary / secondary cell with its qualifications next to the number (precedence label first)."""
    if not x:
        return '—'
    t = f"{x['label']}"
    if x.get('accuracy_label_unqualified'):
        t += f" [accuracy {x['accuracy_label_unqualified']}, unqualified]"
    t += f"; ratio {num(x['ratio_median'])} [{num(x['ci'][0])}, {num(x['ci'][1])}]; {x['n_used']}/{x['n']} used"
    t += f"; {x.get('unresolved_under_allowance') or 0} unresolved under the allowance"
    if x['not_usable']:
        t += '; not usable (median > 10 %)'
    if x.get('regime_label'):
        t += f"; {x['regime_label']}"
    return t


def main():
    plots()
    L = []
    w = L.append
    spent = LED['spent_gpu_hours']
    cal = AN['calibration']
    calnote = f" Every $\\epsilon_{{ref}}$ of this lane also carries: **{cal['label']}**." if cal['label'] else ''
    w('# jcp-regimes: where the frozen off-mesh ROM degrades outside its training distribution (viscous Burgers 2D)\n')
    w(f'**Status:** generated by `make_report.py` from `checks/analysis.json`; every accuracy number is a **{COND}** '
      '(reference uncertainty $\\epsilon_{ref}$ is a conditional Richardson estimate; E0 A10 and combined A10/A11/A15 labels '
      'carried). Labels are **exploratory development evidence** without family-wise error control (DESIGN A1.8, A2.4). '
      f'GPU-hours spent: {spent:g} of {LED["cap_gpu_hours"]:g}.{calnote}\n')
    if AN.get('deferred_cases'):
        w('**Deferred cases (A3.3):** ' + '; '.join(f'{g}: ' + ', '.join(f'c{c}' for c in v)
                                               for g, v in AN['deferred_cases'].items()) + ' (not run; not part of any label).\n')
    if AN.get('separation_note'):
        w(f"**Separation checks:** {AN['separation_note']}.\n")
    w('## Labels per configuration and regime (primary)\n')
    w('Rule (DESIGN §7, A1.8): FAILS (an invalid production run) ▸ NOT QUALIFIED (Q1 or Q2 fails on that regime\'s cases) ▸ '
      'IMPROVES / HOLDS / INCONCLUSIVE / DEGRADES from the 95 % bootstrap interval of the median-error ratio against the '
      'in-range baseline $I$ (factor 2), on reference-eligible valid cases.\n')
    w(row(['configuration'] + [NAMES[g] for g in REG]))
    w('|---|' + '---|' * len(REG))
    for k, c in AN['configs'].items():
        w(row([k + (' (diagnostic)' if c['diagnostic'] else '')] + [cell_text(c['cells'].get(g)) for g in REG]))
    w('')
    w('## Errors: medians, worst cases and ratios\n')
    w(row(['configuration', 'baseline I median', 'baseline I worst', 'in-range consistency N0 / (dev6+val32) [95 %]']
          + [f'{g} median / worst / worst ratio' for g in REG + ('N0s',)]))
    w('|---|---|---|---|' + '---|' * (len(REG) + 1))
    for k, c in AN['configs'].items():
        ic = c['in_range_consistency']
        w(row([k, pct(c['baseline']['median']), pct(c['baseline']['worst']), f"{num(ic['ratio'])} [{num(ic['lo'])}, {num(ic['hi'])}]"]
              + [(f"{pct(x['median'])} / {pct(x['worst'])} / {num(x['ratio_worst'])}" + (' (worst degrades)' if x['worst_degrades'] else ''))
                 if (x := c['cells'].get(g)) else '—' for g in REG + ('N0s',)]))
    w('')
    w('## N0s (stiff part of the training box, descriptive) and the N0-only sensitivity\n')
    w(row(['configuration', 'N0s vs I'] + [f'{g} vs N0 alone' for g in REG]))
    w('|---|---|' + '---|' * len(REG))
    for k, c in AN['configs'].items():
        w(row([k, cell_text(c['cells'].get('N0s'))] + [cell_text(c['cells_vs_N0'].get(g)) for g in REG]))
    w('')
    w('## Reference qualifications\n')
    w(f"- Calibration probe (A2.2, A3.1): smoke agreement {cal['smoke_agree']}; production probes {cal['production_probes']}; "
      + (f"label **{cal['label']}**." if cal['label'] else 'no extra label.'))
    for g, lab in AN['regime_labels'].items():
        sp = AN['separation'].get(g)
        w(f"- Separation check, {NAMES[g]}: " + (json.dumps(sp) if sp is not None else '—')
          + (f" → relabelled **{lab}**" if lab else '') + '.')
    for k, v in AN.get('ps', {}).items():
        w(f"- Pseudo-spectral cross-check {k}: {v['verdict']}" + (f" ({v.get('reason')})" if v.get('reason') else '')
          + (f"; x_ps {num(v.get('x_ps'))}, 2 eps_ref {num(2 * v['eps_ref']) if v.get('eps_ref') else '—'}" if v.get('x_ps') is not None else '')
          + (f"; {v['note']}" if v.get('note') else '') + ' (E0 A10-amended rule, label carried).')
    w('')
    w('## Paired A−/C1 contrast and distance trends\n')
    w('Paired: amplitude raised at matched geometry and viscosity (A1.1), not a pure stiffness mechanism; both cohorts\' '
      'separation labels apply. Trends are descriptive (Spearman, permutation p; OLS slope of log10 error per unit).\n')
    w(row(['configuration', 'pairs', 'median q = e_C1/e_A− [95 %]', 'Wilcoxon p', 'label', 'A+ vs d_ν', 'A− vs d_ν', 'A−∪C1∪C2 vs log10 s']))
    w('|---|---|---|---|---|---|---|---|')
    fmt = lambda t: '—' if t.get('rho') is None else f"ρ {num(t['rho'])}, p {num(t['p'])}, slope {num(t.get('ols_slope_log10e_per_unit'))}, n {t['n']}"
    for k, c in AN['configs'].items():
        pq, tr = c['paired_Am_C1'], c['trends']
        ci = pq.get('ci') or [None, None]
        lab = pq['label'] + ''.join(f"; {AN['regime_labels'][g]} ({g})" for g in ('Am', 'C1') if AN['regime_labels'].get(g))
        w(row([k, pq['n_pairs'], f"{num(pq.get('median_q'))} [{num(ci[0])}, {num(ci[1])}]", num(pq.get('wilcoxon_p')), lab,
               fmt(tr['Ap|d_nu']), fmt(tr['Am|d_nu']), fmt(tr['Am+C1+C2|log10_s'])]))
    w('')
    w('## Q1 / Q2 per configuration and regime (every sampled case)\n')
    w(row(['configuration'] + [NAMES[g] for g in REG + ('N0s',)]))
    w('|---|' + '---|' * (len(REG) + 1))
    for k, c in AN['configs'].items():
        q = c['qualification']
        def qtxt(g):
            if g not in q:
                return '—'
            q1, q2 = q[g]['Q1'], q[g]['Q2']
            bad = [kk for kk in ('all_valid', 'certificate', 'gcert', 'distance', 'tight_tighter', 'offline_agrees') if not q1.get(kk)]
            t = f"Q1 {'pass' if q1['pass'] else 'fail: ' + ', '.join(bad)} (non-stationary {q[g]['pooled_noncertified']}/{q[g]['pooled_steps']})"
            t += f"; Q2 {q2.get('label')}"
            if q2.get('worst_dist_qp') is not None:
                t += f" (successor distance {num(q2['worst_dist_qp'])}, ρ max {num(q2.get('rho_max'))})"
            return t
        w(row([k] + [qtxt(g) for g in REG + ('N0s',)]))
    w('')
    w('## Representation floor (lattice projection on the bank span, diagnostic)\n')
    w(row(['cohort | width', 'median', 'max', 'n']))
    w('|---|---|---|---|')
    for k, v in AN['projection_floor'].items():
        w(row([k, pct(v['median']), pct(v['max']), v['n']]))
    w('')
    w('## Coverage of the references\n')
    w(row(['cohort', 'sampled', 'eligible', 'refined', 'tight path used', 'unresolved under the tight allowance', 'outcomes']))
    w('|---|---|---|---|---|---|---|')
    for coh, v in AN['coverage'].items():
        w(row([coh, v['n'], v['eligible'], v.get('refined', '—'), v.get('tight_used', '—'), v.get('unresolved_under_allowance', '—'),
               json.dumps(v.get('outcomes', '—'))]))
    w('')
    sizes = {coh: v['n'] for coh, v in AN['coverage'].items()}
    w('## Glossary\n')
    for t, d in (('regime', 'a set of freshly drawn development cases differing from the training family in one declared way; '
                  'sizes here: ' + ', '.join(f'{g} {n}' for g, n in sizes.items())),
                 ('in-range baseline I', 'N0 (fresh in-range cases) plus dev6 and val32 (previously used development/validation cases)'),
                 ('ratio', 'median error in the regime divided by the median error on the baseline; 2 is the pre-registered practical margin'),
                 ('bootstrap interval', 'the 2.5–97.5 % range of the ratio over 10 000 resamplings of the cases'),
                 ('worst ratio', 'largest regime error divided by the largest baseline error (descriptive; sample sizes differ)'),
                 ('Q1 / Q2', "solver qualification / quadrature qualification on the regime's own cases (COMMON-BRIEF §3)"),
                 ('eligible', 'a case whose new reference passed the E0 order, Péclet, floor and direction checks'),
                 ('refined', 'a reference that needed the extra 12288² level because its first three levels failed the order check'),
                 ('tight allowance', 'the number of cases allowed to rerun at tighter solver tolerances (A+ 1, others 0); cases blocked by it are "unresolved under the allowance"'),
                 ('σ_a', 'largest interior gradient at the output times divided by the IC amplitude: a diagnostic of front steepness'),
                 ('s = a/ν', 'nominal steepness (inverse front width once a front forms); training range 5–200'),
                 ('d_ν', 'decades of viscosity outside the training range [0.01, 0.1]'),
                 ('paired contrast', 'C1 case j has the same position, width and viscosity as A− case j, only a larger amplitude'),
                 ('Wilcoxon p', 'two-sided signed-rank test that the paired log error ratio is centred on zero'),
                 ('Spearman ρ', 'rank correlation between error and distance; p from 10 000 permutations'),
                 ('representation floor', 'error of the best least-squares fit of the reference by the first R′ bank functions on the scoring lattice'),
                 ('diagnostic', 'a registered row (GAL-CN) reported but not part of the primary labels'),
                 ('not usable', 'regime median error above 10 %')):
        w(f'- **{t}**: {d}.')
    (HERE / 'report.md').write_text('\n'.join(L) + '\n')
    exp = dict(lane='jcp-regimes', label=COND, qualification='exploratory development evidence, no family-wise control',
               gpu_hours=dict(spent=spent, cap=LED['cap_gpu_hours']), deferred_cases=AN.get('deferred_cases'),
               coverage=AN['coverage'], separation=AN['separation'], separation_note=AN.get('separation_note'),
               regime_labels=AN['regime_labels'], calibration=cal, projection_floor=AN['projection_floor'], ps=AN.get('ps'),
               configs={k: dict(diagnostic=c['diagnostic'], baseline=c['baseline'], in_range_consistency=c['in_range_consistency'],
                                cells=c['cells'], sensitivity_vs_N0=c['cells_vs_N0'], qualification=c['qualification'],
                                trends=c['trends'], paired_Am_C1={kk: v for kk, v in c['paired_Am_C1'].items() if kk != 'pairs'})
                        for k, c in AN['configs'].items()})
    (HERE / 'runs' / 'EXPORT.json').write_text(json.dumps(exp, indent=1) + '\n')
    print('wrote report.md, plots/, runs/EXPORT.json')


if __name__ == '__main__':
    main()
