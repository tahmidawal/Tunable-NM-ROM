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


def finish(fig, ax, path, logx=False):
    """Log axes when there is finite positive data; otherwise linear axes with an explicit 'no usable data' note."""
    import matplotlib.pyplot as plt
    try:
        ax.set_yscale('log')
        if logx:
            ax.set_xscale('log')
        fig.tight_layout()
        fig.savefig(path, dpi=140)
    except ValueError:
        ax.set_yscale('linear')
        ax.set_xscale('linear')
        ax.text(.5, .5, 'no usable data', transform=ax.transAxes, ha='center', va='center')
        fig.savefig(path, dpi=140)
    plt.close(fig)


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
    ax.set_ylabel('median error (eligible cases)'); ax.legend(fontsize=7, ncol=2)
    ax.set_title('Median error by regime (E0-style, conditional)')
    finish(fig, ax, out / 'median_by_regime.png')
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
        ax.set_xlabel(xl); ax.set_ylabel('error')
        if key == 's':
            ax.axvline(200, color='grey', ls='--', lw=.8)
        else:
            ax.axvline(0, color='grey', ls='--', lw=.8)
        ax.legend(fontsize=8)
        finish(fig, ax, out / fn, logx=(key == 's'))
    # sigma per regime
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, g in enumerate(('N0', 'N0s') + REG):
        v = [x['sigma']['sigma'] for x in AN['sigma'].get(g, []) if x['sigma'] and x['sigma']['sigma']]
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-.15, .15, len(v)), v, s=10)
    ax.set_xticks(range(8), ('N0', 'N0s') + REG); ax.set_ylabel('σ_a (output-time interior gradient / a)')
    finish(fig, ax, out / 'sigma_by_regime.png')


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
    bi = x.get('base_info') or {}
    bua = bi.get('unresolved_under_allowance')
    t += (f"; ratio {num(x['ratio_median'])} [{num(x['ci'][0])}, {num(x['ci'][1])}]; {x['n_used']}/{x['n']} used vs "
          f"{x.get('n_base_used', '—')} baseline ({bi.get('excluded', '—')} baseline cases excluded, "
          f"{'—' if bua is None else bua} N0 unresolved under the allowance)")
    ua = x.get('unresolved_under_allowance')
    t += f"; {'—' if ua is None else ua} unresolved under the allowance"
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
    w('**Selection qualification (A3.2):** reference eligibility depends on a tight-path allowance that differs between '
      'cohorts (A+ 1, others 0); cases unresolved under the allowance are excluded, and the direction of the resulting '
      'bias on ROM errors is unknown. Their counts are printed next to every coverage entry.\n')
    if AN.get('separation_note'):
        w(f"**Separation checks:** {AN['separation_note']}.\n")
    # ---- generated summary (counts and ranges read from analysis.json)
    prim = {k: c for k, c in AN['configs'].items() if not c['diagnostic']}
    w('## Summary\n')
    for g in REG:
        labs = [c['cells'][g]['label'] for c in prim.values() if g in c['cells']]
        cnt = {l: labs.count(l) for l in sorted(set(labs))}
        meds = [c['cells'][g]['median'] for c in prim.values() if c['cells'].get(g, {}).get('median') is not None]
        rat = [c['cells'][g]['ratio_median'] for c in prim.values() if c['cells'].get(g, {}).get('ratio_median') is not None]
        nq = sum(1 for c in prim.values() if g in c['cells'] and not c['cells'][g]['qualified'])
        w(f"- **{NAMES[g]}** ({len(labs)} primary configurations): labels {cnt}; numerically qualified on this regime "
          f"{len(labs) - nq}/{len(labs)}; median error {pct(min(meds)) if meds else '—'}–{pct(max(meds)) if meds else '—'} "
          f"({COND}); median-error ratio to the in-range baseline {num(min(rat)) if rat else '—'}–{num(max(rat)) if rat else '—'}"
          + (f"; relabelled **{AN['regime_labels'][g]}**" if AN['regime_labels'].get(g) else '') + '.')
    labs = [c['cells']['N0s']['label'] for c in prim.values() if 'N0s' in c['cells']]
    rat = [c['cells']['N0s']['ratio_median'] for c in prim.values() if c['cells'].get('N0s', {}).get('ratio_median') is not None]
    w(f"- **{NAMES['N0s']}** (descriptive): labels { {l: labs.count(l) for l in sorted(set(labs))} }; ratio {num(min(rat))}–{num(max(rat))}: "
      'the stiff part of the training box is already far less accurate than the in-range mixture.')
    fl = AN['projection_floor']
    w('- **Representation floor** (best fit of the reference by the bank span, R384): ' +
      ', '.join(f"{g} {pct(fl[f'{g}|R384']['median'])}" for g in ('val32', 'N0', 'N0s', 'Ap', 'Am', 'C1', 'C2', 'B1', 'B2')
                if f'{g}|R384' in fl) + ' (medians).')
    ps_ok = sum(1 for v in AN.get('ps', {}).values() if v.get('verdict') == 'PASS')
    w(f"- **Pseudo-spectral cross-check:** {ps_ok} of {len(AN.get('ps', {}))} scheduled cases PASS (E0 A10-amended rule, label carried).")
    w('- **Deferred, not run:** the 3D regimes (DESIGN A1.12: they need their own amendment) and the FOM accuracy comparison '
      '(A3.3 step 1).\n')
    w('```mermaid')
    w('flowchart LR')
    w('  G[regime generator, fresh seeds]:::code --> R[new E0-method references, own store]:::job')
    w('  G --> Q[frozen ROM at 1024^2: 9 configurations, in-job Q1/Q2]:::job')
    w('  R --> S[CPU scoring: errors, sigma_a, floors, PS cross-check]:::cpu')
    w('  Q --> S')
    w('  S --> L[labels per configuration x regime]:::cpu')
    w('  classDef code fill:#e5e7eb,stroke:#374151;')
    w('  classDef job fill:#dbeafe,stroke:#1d4ed8;')
    w('  classDef cpu fill:#dcfce7,stroke:#15803d;')
    w('```\n')
    w('Plots: `plots/median_by_regime.png`, `plots/error_vs_nu.png`, `plots/error_vs_s.png`, `plots/sigma_by_regime.png`.\n')
    w('## What went wrong during the lane (disclosed)\n')
    w('- The first smoke (smk1) stopped at an over-specified identity check (a GPU-reduction fingerprint); every other '
      'field matched; DESIGN A7 split the smoke and dropped that field.')
    w('- The H200 queue was saturated; A9 moved the ordinary jobs to A100-80G (no timing result is reported; GPU type '
      'is recorded per job) and replaced a pending refC1 by refC1b.')
    w('- The first analysis run stopped at an exact-hash identity check: generated descriptors differ by one or two ulps '
      'between node CPU types; A10 compares them to round-off instead. No result had been read.')
    w('- **Correction after results audit 1 (A11):** the scoring had inherited a validity rule that allows solver budget '
      'exits, contrary to DESIGN §7.1; corrected, it changes only the GAL-CN diagnostic row on C2 (now FAILS); no primary '
      'cell changed.')
    w('- Prior related evidence (not re-scored here; see LAB-LOG 2026-09-17, lane b-lowvisc): an earlier low-viscosity '
      'cohort also showed much larger linear-bank representation floors; this lane\'s floors point the same way for the '
      'low-ν regimes.\n')
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
        w(row([k, pct(c['baseline']['median']), pct(c['baseline']['worst']),
               f"{num(ic['ratio'])} [{num(ic['lo'])}, {num(ic['hi'])}]; n {ic['n_N0']} vs {ic['n_devval']}; "
               f"N0 unresolved under the allowance {AN['coverage'].get('N0', {}).get('unresolved_under_allowance', '—')}"
               + ('' if ic['coverage_ok'] else '; inconclusive (coverage)')]
              + [(f"{pct(x['median'])} / {pct(x['worst'])} / {num(x['ratio_worst'])}" + (' (worst degrades)' if x['worst_degrades'] else '')
                  + f" [{x['label']}; {x['n_used']}/{x['n']} used; {'—' if x.get('unresolved_under_allowance') is None else x['unresolved_under_allowance']} unresolved"
                  + ('; not usable' if x['not_usable'] else '') + (f"; {x['regime_label']}" if x.get('regime_label') else '') + ']')
                 if (x := c['cells'].get(g)) else '—' for g in REG + ('N0s',)]))
    w('')
    w('## N0s (stiff part of the training box, descriptive) and the N0-only sensitivity\n')
    w(row(['configuration', 'N0s vs I'] + [f'{g} vs N0 alone' for g in REG]))
    w('|---|---|' + '---|' * len(REG))
    for k, c in AN['configs'].items():
        w(row([k, cell_text(c['cells'].get('N0s'))] + [cell_text(c['cells_vs_N0'].get(g)) for g in REG]))
    w('')
    w('## Reference qualifications\n')
    w(f"- Calibration probe (A2.2, A3.1): smoke agreement {cal['smoke_agree']}; production probes {cal['production_probes']} of "
      f"{cal.get('production_probes_expected')} expected{'' if cal.get('complete') else ' (INCOMPLETE)'}; "
      + (f"label **{cal['label']}**." if cal['label'] else 'no extra label.'))
    for g, lab in AN['regime_labels'].items():
        sp = AN['separation'].get(g)
        w(f"- Separation check, {NAMES[g]}: " + (json.dumps(sp) if sp is not None else '—')
          + (f" → relabelled **{lab}**" if lab else '') + '.')
    pst = AN.get('ps_status')
    if isinstance(pst, str):
        w(f'- Pseudo-spectral cross-check: {pst}.')
    elif isinstance(pst, dict):
        got_ = [k for k in pst['expected'] if k in AN.get('ps', {}) and AN['ps'][k].get('verdict') != 'MISSING']
        w(f"- Pseudo-spectral cross-check: {len(got_)} of {len(pst['expected'])} scheduled cases present"
          + (f"; missing configurations: {', '.join(pst['missing_configs'])}" if pst.get('missing_configs') else '')
          + (f"; missing outputs: {', '.join(k for k in pst['expected'] if k not in got_)}" if len(got_) < len(pst['expected']) else '') + '.')
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
        ctx = pq.get('context', {})
        lab = pq['label'] + ''.join(f"; {g}: {v['label']}, {v['invalid_runs']} invalid, {v['excluded']} excluded, "
                                    f"{'—' if v['unresolved_under_allowance'] is None else v['unresolved_under_allowance']} unresolved"
                                    + (f", {v['regime_label']}" if v.get('regime_label') else '') for g, v in ctx.items())
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
                t += (f" (successor distance {num(q2['worst_dist_qp'])}, ρ max {num(q2.get('rho_max'))}, check distance "
                      f"{num(q2.get('worst_dist_qp_qpp'))}, successor valid {q2.get('successor_valid')}, check valid {q2.get('check_valid')})")
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
                 ('not usable', 'regime median error above 10 %'),
                 ('OLS slope', 'least-squares slope of log10(error) against the distance variable'),
                 ('successor / check rule', 'the next finer quadrature rule (Q2 replay) and the one after it (convergence check)'),
                 ('non-stationary steps', 'time steps whose final stationarity ratio exceeds 1e-3 (LSPG) or whose residual criterion exceeds 1e-8 (GAL)'),
                 ('configuration id', 'R<width>|<form>_<scheme>_<step factor>: bank width R′, LSPG or GAL, backward Euler (BE), Crank–Nicolson with a Rannacher start (CNR) or Crank–Nicolson (CN), and the step as a multiple of Δt0 = 0.005'),
                 ('in-range consistency', 'median error on N0 divided by that on dev6+val32: should contain 1 if the fresh references and the E0 references agree')):
        w(f'- **{t}**: {d}.')
    (HERE / 'report.md').write_text('\n'.join(L) + '\n')
    exp = dict(lane='jcp-regimes', label=COND, qualification='exploratory development evidence, no family-wise control',
               gpu_hours=dict(spent=spent, cap=LED['cap_gpu_hours']), deferred_cases=AN.get('deferred_cases'),
               coverage=AN['coverage'], separation=AN['separation'], separation_note=AN.get('separation_note'),
               regime_labels=AN['regime_labels'], calibration=cal, projection_floor=AN['projection_floor'], ps=AN.get('ps'),
               ps_status=AN.get('ps_status'),
               selection_qualification='eligibility depends on cohort-specific tight-path allowances; bias direction unknown (A3.2)',
               configs={k: dict(diagnostic=c['diagnostic'], baseline=c['baseline'], in_range_consistency=c['in_range_consistency'],
                                cells=c['cells'], sensitivity_vs_N0=c['cells_vs_N0'], qualification=c['qualification'],
                                trends=c['trends'], paired_Am_C1={kk: v for kk, v in c['paired_Am_C1'].items() if kk != 'pairs'})
                        for k, c in AN['configs'].items()})
    (HERE / 'runs' / 'EXPORT.json').write_text(json.dumps(exp, indent=1) + '\n')
    print('wrote report.md, plots/, runs/EXPORT.json')


if __name__ == '__main__':
    main()
