"""jcp-wide-bank: generate report.md and plots/*.png from the run JSONs (no hand-typed numbers). DESIGN.md + A0-A5.

    python make_report.py            # uses every collected job found under runs/<attempt>/archive

Sources (each optional; a section is written only when its job exists and its audit accepted it):
  2D  J1   runs/j1/archive/output/result.json        audit checks/j1-audit.json
  3D  J2   runs/j2/archive/output/result.json        (3D trim, old bank)
  3D  J3   runs/j3b/archive/output/training.json     (wide bank training)
  3D  J4   runs/j4/archive/output/result.json        (3D dial)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = Path(__file__).resolve().parent
PLOTS = HERE / 'plots'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']        # dataviz reference categorical slots 1-4 (validated)
MARKERS = ['o', 's', '^', 'D']
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
PP = 0.05e-2                                                    # 0.05 percentage points as a fraction


def load(p):
    p = HERE / p
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, d=2):
    return '–' if x is None else f'{100 * x:.{d}f} %'


def sci(x):
    return '–' if x is None else f'{x:.1e}'


def gb(b):
    return '–' if b is None else (f'{b / 1e9:.2f} GB' if b >= 1e8 else f'{b / 1e6:.1f} MB')


def plain(ax, axis='y'):
    from matplotlib.ticker import FuncFormatter, LogLocator
    fmt = FuncFormatter(lambda v, _: f'{v:g}')
    for a_ in ((ax.yaxis,) if axis == 'y' else (ax.xaxis,) if axis == 'x' else (ax.xaxis, ax.yaxis)):
        a_.set_major_formatter(fmt)
        a_.set_minor_formatter(matplotlib.ticker.NullFormatter())


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, color=INK, fontsize=11, loc='left')
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, lw=0.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2)


def fig():
    f, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    f.patch.set_facecolor(SURF)
    return f, ax


def save(f, name):
    PLOTS.mkdir(exist_ok=True)
    f.tight_layout()
    f.savefig(PLOTS / name, facecolor=SURF)
    plt.close(f)
    return f'plots/{name}'


# ----------------------------------------------------------------------------------------------------- 2D (J1)
def summarise_2d(d):
    """Per setting: deployed arm and its cohort statistics, converged rollout statistics, floors, m*, costs."""
    out = {}
    rows = d['rows']
    for key, st in d['settings'].items():
        Rp, M = st['Rp'], st['M']
        R = [r for r in rows if r['setting'] == key]
        stat = lambda arm, tag: (lambda v: (max(v), float(np.median(v))) if v else (None, None))(
            [r[f'ref_{tag}_evolved'] for r in R if r['arm'] == arm])
        sel = d['selection'][key]
        dep = sel.get('deployed')
        fl = d['floors'].get(f'R{Rp}', {})
        flok = fl.get('check_passed', False)
        e = dict(key=key, Rp=Rp, M=M, kappa=st['kappa'], tensor_bytes=st['tensor_bytes'],
                 conv_ST=stat('gref', 'ST'), conv_S=stat('gref', 'S'),
                 floor_ST=(fl['ST']['worst'], fl['ST']['median']) if flok else (None, None),
                 floor_S=(fl['S']['worst'], fl['S']['median']) if flok else (None, None),
                 mstar={k: v for k, v in sel['entries'].items()}, deployed=dep,
                 gates=dict(conv=d['gates'][f'converged_{key}']['passed'],
                            target=d['gates'][f'continuum_target_{key}']['passed'],
                            controls=d['gates'][f'controls_{key}']['discriminating']),
                 timing_valid=d['timing'][key].get('valid'), A_condition=st['A_condition'],
                 jac=st.get('jacobian_reached', {}).get('states', {}), memory=st.get('memory', {}))
        if dep:
            e['dep_ST'], e['dep_S'] = stat(dep['arm'], 'ST'), stat(dep['arm'], 'S')
            e['dep_bytes'] = st['arms'][dep['arm']]['bytes']
            e['dep_ms_setting'] = d['timing'][key]['subjects'][dep['arm']]['median_ms']
            fk = f"{key}|{dep['family']}|{dep['arm']}"
            e['dep_ms_final'] = d['final_timing'].get('subjects', {}).get(fk, {}).get('median_ms')
            e['final_key'] = fk
            e['dep_cases'] = {(r['cohort'], r['case']): r['ref_ST_evolved'] for r in R if r['arm'] == dep['arm']}
        else:
            e['dep_ST'] = e['dep_S'] = (None, None)
        out[key] = e
    return out


def verdict(a, b, ratio, tag):
    """A2-7 precedence: unavailable > unresolved (< 0.05 pp) > meets (worst <= ratio x, median lower) > does not meet.
    a = the wider setting, b = the baseline; tag selects the reference."""
    wa, ma = a.get(f'dep_{tag}', (None, None))
    wb, mb = b.get(f'dep_{tag}', (None, None))
    if None in (wa, wb):
        return 'unavailable'
    if abs(wa - wb) < PP:
        return 'unresolved'
    if wa <= ratio * wb and ma < mb:
        return 'meets the registered bar'
    return 'does not meet the registered bar'


def paired_ratio(d, k_trim, k_base):
    inv = d['final_timing'].get('invocations', [])
    med = {}
    for r in inv:
        med.setdefault((r['key'], r['case'], r['phase']), []).append(r['seconds'])
    rat = [np.median(med[(k_trim, c, ph)]) / np.median(med[(k_base, c, ph)])
           for (k, c, ph) in med if k == k_trim and (k_base, c, ph) in med]
    return float(np.median(rat)) if rat else None


def section_2d(d, audit, L):
    S = summarise_2d(d)
    by = {(e['Rp'], round(e['kappa'])): e for e in S.values()}
    Rps = sorted({e['Rp'] for e in S.values()})
    tv = d.get('timing_valid_jobwide')
    L.append('## 2D: the dial in $R\'$ (1a) and the test-count trim (1d)\n')
    L.append(f"Job `{d['config'].get('attempt')}` {d.get('job_id')} on {d['gpu']}, commit `{(d.get('commit') or '')[:9]}`, "
             f"mesh ${d['mesh']}^2$, cohorts {' ∪ '.join(d['config']['cohorts'])} "
             f"({len({(r['cohort'], r['case']) for r in d['rows']})} cases), elapsed {d['elapsed_seconds'] / 3600:.2f} h. "
             f"Independent audit (`audit_w.py`): **{'accepted' if audit and audit.get('accepted') else 'NOT accepted'}**. "
             f"Job-wide timing validity (K-time): **{tv}**. All errors below are **PROVISIONAL**: both references are "
             f"first-order upwind solutions at $8192^2$ (ST: $\\Delta t/16$; S: the ROM's own $\\Delta t$).\n")
    L.append('### The dial at $M=4R\'$\n')
    L.append('| $R\'$ | $M$ | deployed rule ($m$) | worst / median vs ST | worst / median vs S | converged rollout vs ST | '
             'converged vs S | projection floor vs ST | floor vs S | query ms (final panel) | off-mesh bytes | tensor bytes |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for Rp in Rps:
        e = by.get((Rp, 4))
        if not e:
            continue
        dep = e['deployed']
        L.append(f"| {Rp} | {e['M']} | {dep['arm'] + ' (' + str(dep['m']) + ')' if dep else 'unavailable'} | "
                 f"{pct(e['dep_ST'][0])} / {pct(e['dep_ST'][1])} | {pct(e['dep_S'][0])} / {pct(e['dep_S'][1])} | "
                 f"{pct(e['conv_ST'][0])} / {pct(e['conv_ST'][1])} | {pct(e['conv_S'][0])} / {pct(e['conv_S'][1])} | "
                 f"{pct(e['floor_ST'][0], 3)} | {pct(e['floor_S'][0], 3)} | "
                 f"{'%.1f' % e['dep_ms_final'] if e.get('dep_ms_final') and tv else 'withdrawn' if e.get('dep_ms_final') else '–'} | "
                 f"{gb(e.get('dep_bytes'))} | {gb(e['tensor_bytes'])} |")
    a, b = by.get((512, 4)), by.get((384, 4))
    if a and b:
        vST, vS = verdict(a, b, 0.9, 'ST'), verdict(a, b, 0.9, 'S')
        dST = (a['dep_ST'][0] - b['dep_ST'][0]) if None not in (a['dep_ST'][0], b['dep_ST'][0]) else None
        dS = (a['dep_S'][0] - b['dep_S'][0]) if None not in (a['dep_S'][0], b['dep_S'][0]) else None
        both = vST if vST == vS else 'reference-dependent'
        imp = [b['dep_cases'][k] - a['dep_cases'][k] for k in a.get('dep_cases', {}) if k in b.get('dep_cases', {})]
        L.append(f"\n**H1 ($R'=512$ vs $384$, $M=4R'$), PROVISIONAL, benchmark-relative:** against ST *{vST}* "
                 f"(worst {pct(b['dep_ST'][0])} → {pct(a['dep_ST'][0])}, change {('%+.3f pp' % (100 * dST)) if dST is not None else '–'}); "
                 f"against S *{vS}* (worst {pct(b['dep_S'][0])} → {pct(a['dep_S'][0])}, change "
                 f"{('%+.3f pp' % (100 * dS)) if dS is not None else '–'}); combined verdict: **{both}**. "
                 + (f"Paired per case vs ST: median improvement {100 * float(np.median(imp)):+.3f} pp, "
                    f"{sum(x > 0 for x in imp)}/{len(imp)} cases improved." if imp else '') + '\n')
    L.append('\n### All twelve settings (trim grid)\n')
    L.append('| $R\'$ | $\\kappa$ | $M$ | $m^\\star$ Gauss / Fibonacci ($\\tau=2.5\\times10^{-4}$) | same at $\\tau=10^{-3}$ | '
             '$m_d$ / $m_\\rho$ (Gauss) | deployed worst vs ST / S | query ms (setting panel) | gates conv / target / controls | cond $A$ |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for Rp in Rps:
        for k in (4, 3, 2):
            e = by.get((Rp, k))
            if not e:
                continue
            ms = e['mstar']
            g = lambda t, f: ms.get(f'{t}|{f}', {}).get('m') or ('none' if ms.get(f'{t}|{f}', {}).get('m_raw') is None else 'n/a')
            L.append(f"| {Rp} | {k} | {e['M']} | {g('primary', 'gauss')} / {g('primary', 'fib')} | "
                     f"{g('secondary', 'gauss')} / {g('secondary', 'fib')} | "
                     f"{ms.get('primary|gauss', {}).get('m_d')} / {ms.get('rho_only|gauss', {}).get('m_rho')} | "
                     f"{pct(e['dep_ST'][0])} / {pct(e['dep_S'][0])} | "
                     f"{'%.1f' % e['dep_ms_setting'] if e.get('dep_ms_setting') else '–'} | "
                     f"{e['gates']['conv']} / {e['gates']['target']} / {e['gates']['controls']['primary']} | "
                     f"{e['A_condition']:.1f} |")
    L.append('\n**H3 (trim), PROVISIONAL:** a trim $\\kappa<4$ is *acceptable* if the deployed worst and median ST errors '
             'are within 0.05 pp of the $\\kappa=4$ deployed rollout, *useful* if also the paired query-time ratio is '
             '$\\le0.9$ in the final panel and in the per-setting panels (only when K-time is valid job-wide).\n')
    L.append('| $R\'$ | $\\kappa$ | worst ST change vs $\\kappa=4$ | median ST change | worst S change | acceptable | '
             'paired time ratio (final) | setting-panel ratio | useful |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    h3 = {}
    for Rp in Rps:
        b = by.get((Rp, 4))
        for k in (3, 2):
            a = by.get((Rp, k))
            if not (a and b) or None in (a['dep_ST'][0], b['dep_ST'][0]):
                if a and b:
                    L.append(f'| {Rp} | {k} | – | – | – | unavailable | – | – | – |')
                continue
            dw, dm = a['dep_ST'][0] - b['dep_ST'][0], a['dep_ST'][1] - b['dep_ST'][1]
            dS = a['dep_S'][0] - b['dep_S'][0]
            acc = dw <= PP and dm <= PP
            pr = paired_ratio(d, a['final_key'], b['final_key']) if tv else None
            sr = (a['dep_ms_setting'] / b['dep_ms_setting']) if tv else None
            use = bool(acc and pr is not None and sr is not None and pr <= 0.9 and sr <= 0.9)
            h3[(Rp, k)] = dict(acceptable=acc, useful=use)
            L.append(f"| {Rp} | {k} | {100 * dw:+.3f} pp | {100 * dm:+.3f} pp | {100 * dS:+.3f} pp | {acc} | "
                     f"{'%.2f' % pr if pr is not None else 'withdrawn'} | {'%.2f' % sr if sr is not None else 'withdrawn'} | {use} |")
    # ---- gates detail
    L.append('\n### Gates and controls (2D)\n')
    L.append('| setting | K-conv: worst distance Gauss $768^2$ vs $640^2$ (bar $2.5\\times10^{-5}$) | K-target: worst $\\rho$ check / flux (bar $10^{-5}$) | '
             'control Gauss $8^2$: worst distance / $\\rho_{\\max}$ | control Smolyak-8: worst distance / $\\rho_{\\max}$ | floor check | K-time drift / deterministic |')
    L.append('|---|---|---|---|---|---|---|')
    for key, e in S.items():
        gcv, gt = d['gates'][f'converged_{key}'], d['gates'][f'continuum_target_{key}']
        ca = d['gates'][f'controls_{key}']['arms']
        tg = d['timing'][key]['gates']
        L.append(f"| {key} | {sci(gcv['gref_check_worst_distance'])} ({gcv['passed']}) | {sci(gt['check_rho_max'])} / "
                 f"{sci(gt['flux_rho_max'])} ({gt['passed']}) | "
                 f"{sci(ca['ctrl_gauss8']['worst_distance'])} / {sci(ca['ctrl_gauss8']['rho_max'])} | "
                 f"{sci(ca['ctrl_smolyak8']['worst_distance'])} / {sci(ca['ctrl_smolyak8']['rho_max'])} | "
                 f"{d['floors'].get('R%d' % e['Rp'], {}).get('check_passed')} | {tg['drift_worst']:.3f} / {tg['deterministic']} |")
    # ---- what the numbers say (phrasing per audits/codex-results-j1.md)
    rows = d['rows']

    def per_case(key, arm, tag):
        return {(r['cohort'], r['case']): r[f'ref_{tag}_evolved'] for r in rows if r['setting'] == key and r['arm'] == arm}
    a, b = by.get((512, 4)), by.get((384, 4))
    if a and b and a['deployed'] and b['deployed']:
        sa, sb = per_case(a['key'], a['deployed']['arm'], 'S'), per_case(b['key'], b['deployed']['arm'], 'S')
        worse = sum(sa[k] > sb[k] for k in sa)
        wk = max(sa, key=sa.get)
        sa2 = [v for k, v in sa.items() if k != wk]
        sb2 = [v for k, v in sb.items() if k != wk]
        L.append('\n### What the 2D numbers say (PROVISIONAL)\n')
        L.append(f"- **The 2D dial stops at $R'=384$ on this bank.** "
                 f"**$R'=512$ is worse than $R'=384$ against S on {worse} of {len(sa)} cases**; without the worst case ({wk[0]} {wk[1]}) the worst S error is still "
                 f"{pct(max(sa2))} vs {pct(max(sb2))}. The span floor improves only slightly "
                 f"({pct(b['floor_S'][0], 3)} → {pct(a['floor_S'][0], 3)}), and the conditioning of $A$ grows "
                 f"({b['A_condition']:.1f} → {a['A_condition']:.1f}). The 512-column bank adds little representable content "
                 "and costs accuracy in the reduced dynamics. *This is a hypothesis, not a demonstrated cause:* no experiment "
                 "here isolates whether the conditioning growth or something else degrades the $R'=512$ rollouts.")
        L.append(f"- **The ST median is flat across $R'$** ({', '.join(pct(by[(r, 4)]['dep_ST'][1]) for r in Rps if by.get((r, 4)))} "
                 f"for $R'$ = {', '.join(str(r) for r in Rps)}), while the S median falls "
                 f"({', '.join(pct(by[(r, 4)]['dep_S'][1]) for r in Rps if by.get((r, 4)))}). This is consistent with the "
                 "backward-Euler time error at $\\Delta t=0.005$ masking the spatial gains; it is not causally isolated here "
                 "(lane C2 tests second-order time stepping).")
        L.append(f"- **Cost grows steeply with $R'$**: the final-panel query time is "
                 f"{', '.join(('%.0f ms' % by[(r, 4)]['dep_ms_final']) for r in Rps if by.get((r, 4)) and by[(r, 4)].get('dep_ms_final'))} "
                 f"for $R'$ = {', '.join(str(r) for r in Rps)}, because the points needed ($m^\\star$) grow with the setting.")
        L.append('- **The mechanism registered for 1d is contradicted.** DESIGN 1d predicted that a smaller $M$ (lower test '
                 'frequencies) would need fewer quadrature points. It does not: $m^\\star$ depends jointly on $R\'$ and $M$ and on '
                 'the tested ladder, and reducing $M$ sometimes *increases* it. In every setting the rollout distance, not $\\rho$, '
                 'decides $m^\\star$.')
        u = [k for k, v in h3.items() if v['useful']]
        L.append(f"- **Trim:** useful under the registered ST criterion only at {', '.join(f"$R'={k[0]}$, $\\kappa={k[1]}$" for k in u) or 'no setting'}; "
                 "every trim increases the worst error against S, so the acceptance is specific to the space+time reference.\n")
    plots_2d(S, by, Rps, tv)
    return S, by, h3


def plots_2d(S, by, Rps, tv):
    # error vs R' (kappa = 4): deployed ST and S, floors
    f, ax = fig()
    xs = [r for r in Rps if by.get((r, 4)) and by[(r, 4)]['dep_ST'][0] is not None]
    lines = [('deployed vs ST (space + time ref.)', [by[(r, 4)]['dep_ST'][0] for r in xs]),
             ('deployed vs S (space-only ref.)', [by[(r, 4)]['dep_S'][0] for r in xs]),
             ('projection floor vs S', [by[(r, 4)]['floor_S'][0] for r in xs])]
    for i, (lab, ys) in enumerate(lines):
        ys = [100 * y if y is not None else np.nan for y in ys]
        ax.plot(xs, ys, color=SERIES[i], lw=2, marker=MARKERS[i], ms=7, label=lab)
        ax.annotate(lab.split(' (')[0], (xs[-1], ys[-1]), textcoords='offset points', xytext=(6, 0), color=INK2, fontsize=8,
                    va='center')
    ax.set_yscale('log')
    ax.set_xticks(xs)
    plain(ax)
    style(ax, "2D dial (M = 4R'): worst error over 38 validation cases\nPROVISIONAL, first-order references", "R' (bank columns used)",
          'worst relative error (%)')
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    save(f, '2d_error_vs_Rp.png')
    # memory vs R'
    f, ax = fig()
    xs = [r for r in Rps if by.get((r, 4))]
    ax.plot(xs, [by[(r, 4)]['tensor_bytes'] / 1e9 for r in xs], color=SERIES[1], lw=2, marker=MARKERS[1], ms=7,
            label='precomputed tensor, $8MR\'^2$')
    xd = [r for r in xs if by[(r, 4)].get('dep_bytes')]
    ax.plot(xd, [by[(r, 4)]['dep_bytes'] / 1e9 for r in xd], color=SERIES[0], lw=2, marker=MARKERS[0], ms=7,
            label='off-mesh rule at $m^\\star$, $8m(2R\'+M)$')
    ax.set_yscale('log')
    ax.set_xticks(xs)
    plain(ax)
    style(ax, "2D advection memory vs R' (M = 4R')", "R'", 'GB (log scale)')
    ax.legend(frameon=False, fontsize=8)
    save(f, '2d_memory_vs_Rp.png')
    # m* vs M
    f, ax = fig()
    for i, r in enumerate(Rps):
        es = sorted([e for e in S.values() if e['Rp'] == r], key=lambda e: e['M'])
        xm = [e['M'] for e in es if e['mstar'].get('primary|gauss', {}).get('m')]
        ym = [e['mstar']['primary|gauss']['m'] for e in es if e['mstar'].get('primary|gauss', {}).get('m')]
        if xm:
            ax.plot(xm, ym, color=SERIES[i], lw=2, marker=MARKERS[i], ms=7, label=f"R' = {r}")
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    plain(ax, 'both')
    style(ax, 'Gauss points needed ($m^\\star$, $\\tau=2.5\\times10^{-4}$) vs test count M', 'M (sine tests)',
          '$m^\\star$ (Gauss points)')
    ax.legend(frameon=False, fontsize=8, title="bank width", title_fontsize=8)
    save(f, '2d_mstar_vs_M.png')
    # cost vs R'
    if tv:
        f, ax = fig()
        for i, k in enumerate((4, 3, 2)):
            xk = [r for r in Rps if by.get((r, k)) and by[(r, k)].get('dep_ms_final')]
            ax.plot(xk, [by[(r, k)]['dep_ms_final'] for r in xk], color=SERIES[i], lw=2, marker=MARKERS[i], ms=7,
                    label=f'M/R\' = {k}')
        ax.set_xticks(Rps)
        style(ax, "2D query time of the deployed rule vs R'\n(one A100, final A–B–A panel)", "R'", 'median ms per query')
        ax.legend(frameon=False, fontsize=8)
        save(f, '2d_cost_vs_Rp.png')


# ----------------------------------------------------------------------------------------------------- 3D training (J3)
def section_train(t, L):
    if t is None:
        return None
    L.append('## 3D: the wider bank (1b)\n')
    b, o = t.get('bank', {}), t.get('ordering', {})
    ff, cf = t.get('floors_full', {}), t.get('compare_bank_floors_full', {})
    L.append(f"Job {t.get('job_id')} on {t.get('gpu')}, commit `{(t.get('commit') or '')[:9]}`: single-seed, capacity-scaled "
             f"baseline-recipe bank, rank {t['config']['bank_rank']}, width {t['config']['bank_width']}, POD modes "
             f"{t['config']['modes']} (DESIGN A1/A2). Selected checkpoint step {b.get('selected_step')} of "
             f"{t['config']['bank_steps']}; worst bank-validation error {pct(b.get('validation_worst_over_groups'))}; "
             f"bank time {b.get('seconds', 0) / 3600:.2f} h; total {t.get('seconds', 0) / 3600:.2f} h.\n")
    L.append("Floors here are **span** floors on the native-grid bank-validation fields (seed 923751, 96 cases × six saved "
             "times including $t=0$, each snapshot normalised by its own norm), not errors against the refined reference, and not "
             "rollout errors: a lower floor shows more representable content, not a more accurate reduced model (that is 1c). "
             "The old/new comparison at equal deployed width does not isolate rank (width, POD truncation, seed and whitening "
             "weight differ).\n")
    rows = []
    for n in ('33', '65', '129'):
        for r in ('256', '512', '768', '1024'):
            rows.append((n, r, ff.get(n, {}).get(r, {}).get('worst'), cf.get(n, {}).get(r, {}).get('worst')))
    L.append('| mesh nodes | $R\'$ | new bank: worst full-grid floor | `model_M2`: same fields |')
    L.append('|---|---|---|---|')
    for n, r, a, c in rows:
        L.append(f'| {n} | {r} | {pct(a)} | {pct(c) if c is not None else "unavailable (bank has 512 columns)"} |')
    f129 = ff.get('129', {})
    c129 = cf.get('129', {}).get('512', {}).get('worst')
    w512, w768, w1024 = (f129.get(k, {}).get('worst') for k in ('512', '768', '1024'))
    B2 = (o.get('inverse_check') is not None and o['inverse_check'] <= 1e-8 and b.get('condition_65') is not None
          and b['condition_65'] < 1e10)
    B3 = None if None in (w512, c129) else bool(w512 <= 1.10 * c129)
    B4s = None if None in (w512, w768, w1024) else bool(w1024 < w768 < w512)
    B4m = None if None in (w512, w1024) else bool(w1024 <= 0.8 * w512)
    L.append(f"\nB2′ (inverse check {sci(o.get('inverse_check'))}, 65-grid condition {b.get('condition_65', float('nan')):.3g}): "
             f"**{B2}**. B3 (new $R'=512$ floor at 129 nodes ≤ 1.10 × `model_M2`'s, comparison flag): **{B3}** "
             f"({pct(w512)} vs {pct(c129)}). B4 strict decrease 512 → 768 → 1024: **{B4s}**; B4′ material gain "
             f"(1024 ≤ 0.8 × 512): **{B4m}** ({pct(w1024)} vs {pct(w512)}).\n")
    return dict(B2=B2, B3=B3, B4=B4s, B4m=B4m)


# ----------------------------------------------------------------------------------------------------- 3D panels (J2, J4)
def section_3d(d, title, L, tag, audit=None):
    if d is None:
        return None
    L.append(f'## {title}\n')
    L.append(f"Independent NumPy audit (`audit_w3.py`): **{'accepted' if audit and audit.get('accepted') else 'NOT accepted / not run'}**. "
             f"GPU: **{d['gpu']}** (timings are comparable only within this job).\n")
    if d.get('amendment'):
        L.append('**Selection is POST HOC (amendment A7).** This job ran with a per-rollout eligibility rule that deviated from '
                 'the pooled rule A0-3 cites; one non-stationary step in one validation case made several settings unavailable. '
                 'A7 was written *after* the first setting of this job had finished and its log (per-arm errors and distances) '
                 'had been read, but before any selection under the pooled rule was computed. The table shows the amended '
                 '(post hoc) deployment; the as-run outcome is given beside it. The job-time final cross-setting panel '
                 + 'covered only the as-run deployments; the H3 confirmation block (final-panel paired ratio) is missing for: '
                 + (', '.join(k for m_ in d['meshes'].values() for k in m_.get('final_panel_coverage', {}).get('missing', []))
                    or 'none') + '.\n')
        L.append('| mesh | bank | setting | as-run deployed (per-rollout rule) | amended deployed (A7, post hoc) |')
        L.append('|---|---|---|---|---|')
        for n, m_ in d['meshes'].items():
            for bn, b in m_['banks'].items():
                for key, S in b['settings'].items():
                    ar, am = S['as_run'].get('deployed'), S.get('deployed')
                    L.append(f"| {int(n) - 1}³ | {bn} | {key} | {ar['arm'] + ' (' + str(ar['m']) + ')' if ar else 'unavailable (' + str(S['as_run']['selection']['entries']['primary|lat'].get('reason')) + ')'} | "
                             f"{am['arm'] + ' (' + str(am['m']) + ')' if am else 'unavailable'} |")
        L.append('')
    L.append(f"Job {d.get('job_id')} on {d['gpu']}, commit `{(d.get('commit') or '')[:9]}`, validation cohort "
             f"{d['config']['cohort_seed']} × {len(d['config'].get('case_subset') or range(d['config']['cohort_count']))} cases; "
             f"job-wide timing validity: **{d.get('timing_valid_jobwide')}**. Errors against the 513-node first-order "
             f"reference on the $63^3$ lattice: **PROVISIONAL**.\n")
    L.append('| mesh | bank | $R\'$ | $\\kappa$ | $M$ | deployed ($m$) | worst / median refined | converged worst / median | '
             'floor worst | $m^\\star$ lattice / Gauss ($\\tau=2.5\\times10^{-4}$) | same at $\\tau=10^{-3}$ | query ms | Jacobian ms | off-mesh bytes | tensor bytes | gates conv / target / controls |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    recs = []
    for n, m_ in d['meshes'].items():
        for bn, b in m_['banks'].items():
            for key, s in b['settings'].items():
                dep = s.get('deployed')
                da = s['arms'].get(dep['arm']) if dep else None
                cv = s['arms']['conv']
                fl = b['floors'].get(str(s['Rp']), {})
                ent = s['selection']['entries']
                rec = dict(n=int(n), bank=bn, Rp=s['Rp'], kappa=s['kappa_nominal'], M=s['M'], dep=dep,
                           dep_worst=da['worst_refined'] if da else None, dep_med=da['median_refined'] if da else None,
                           conv_worst=cv['worst_refined'], conv_med=cv['median_refined'],
                           floor=fl.get('worst') if fl.get('check_passed') else None,
                           ms=s['timing']['subjects'].get(dep['arm'], {}).get('median_ms') if dep else None,
                           jac=s['microbench'].get(dep['arm'], {}).get('jacobian_ms_median') if dep else None,
                           bytes=da['bytes'] if da else None, tensor_bytes=s['tensor_bytes'],
                           mlat=ent.get('primary|lat', {}).get('m'), mgau=ent.get('primary|gauss', {}).get('m'),
                           mlat2=ent.get('secondary|lat', {}).get('m'), mgau2=ent.get('secondary|gauss', {}).get('m'),
                           tensor_worst=s['arms'].get('tensor', {}).get('worst_refined'),
                           gates=(s['gates']['converged']['passed'], s['gates']['target']['passed'],
                                  s['selection']['discriminating']['primary']))
                rec['final_key'] = f"{bn}|{key}|{dep['arm']}" if dep else None
                rec['tv'], rec['gpu'] = d.get('timing_valid_jobwide'), d['gpu']
                rec['dep_cases'] = {c['case']: c['worst_refined'] for c in da['cases']} if da else {}
                recs.append(rec)
                tv = d.get('timing_valid_jobwide')
                L.append(f"| {n} | {bn} | {s['Rp']} | {s['kappa_nominal']} | {s['M']} | "
                         f"{dep['arm'] + ' (' + str(dep['m']) + ')' if dep else 'unavailable'} | "
                         f"{pct(rec['dep_worst'])} / {pct(rec['dep_med'])} | {pct(rec['conv_worst'])} / {pct(rec['conv_med'])} | "
                         f"{pct(rec['floor'])} | {rec['mlat']} / {rec['mgau']} | {rec['mlat2']} / {rec['mgau2']} | "
                         f"{('%.1f' % rec['ms']) if rec['ms'] and tv else 'withdrawn' if rec['ms'] else '–'} | "
                         f"{('%.3f' % rec['jac']) if rec['jac'] else '–'} | {gb(rec['bytes'])} | {gb(rec['tensor_bytes'])} | "
                         f"{rec['gates'][0]} / {rec['gates'][1]} / {rec['gates'][2]} |")
    L.append('')
    tv = d.get('timing_valid_jobwide')
    by = {(r['n'], r['bank'], r['Rp'], r['kappa']): r for r in recs}
    # tensor comparison
    tr = [r for r in recs if r['tensor_worst'] is not None]
    if tr:
        L.append('Tensor rule (the incumbent, mesh backward-difference advection) on the same cases: ' +
                 '; '.join(f"{r['n'] - 1}³ {r['bank']} $R'={r['Rp']}$, $\\kappa={r['kappa']}$: worst {pct(r['tensor_worst'])}"
                           for r in tr) + '.\n')
    # H3 trim (3D)
    trims = [r for r in recs if r['kappa'] < 4 and (r['n'], r['bank'], r['Rp'], 4) in by]
    if trims:
        L.append('**H3 (3D trim), PROVISIONAL:** acceptable if the deployed worst and median refined errors are within 0.05 pp '
                 'of $\\kappa=4$; useful if also the paired time ratio is $\\le0.9$ in the final panel and the setting panels '
                 '(only with job-wide K-time validity).\n')
        L.append('| mesh | $R\'$ | $\\kappa$ | worst change | median change | acceptable | paired ratio (final) | setting ratio | useful |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        for r in trims:
            b4 = by[(r['n'], r['bank'], r['Rp'], 4)]
            if None in (r['dep_worst'], b4['dep_worst']):
                L.append(f"| {r['n'] - 1}³ | {r['Rp']} | {r['kappa']} | – | – | unavailable | – | – | – |")
                continue
            dw, dm = r['dep_worst'] - b4['dep_worst'], r['dep_med'] - b4['dep_med']
            acc = dw <= PP and dm <= PP
            ft = d['meshes'][str(r['n'])].get('final_timing') or {}
            pr = None
            if tv and ft:
                med = {}
                for x in ft['invocations']:
                    med.setdefault((x['name'], x['case'], x['phase']), []).append(x['seconds'])
                rat = [np.median(med[(r['final_key'], c, ph)]) / np.median(med[(b4['final_key'], c, ph)])
                       for (k, c, ph) in med if k == r['final_key'] and (b4['final_key'], c, ph) in med]
                pr = float(np.median(rat)) if rat else None
            sr = (r['ms'] / b4['ms']) if tv and r['ms'] and b4['ms'] else None
            use = bool(acc and pr is not None and sr is not None and pr <= 0.9 and sr <= 0.9)
            L.append(f"| {r['n'] - 1}³ | {r['Rp']} | {r['kappa']} | {100 * dw:+.3f} pp | {100 * dm:+.3f} pp | {acc} | "
                     f"{'%.2f' % pr if pr is not None else ('missing (not measured)' if tv else 'withdrawn (K-time)')} | "
                     f"{'%.2f' % sr if sr is not None else 'withdrawn (K-time)'} | {use} |")
        L.append('')
        # what the numbers say (phrasing per audits/codex-results-j2.md)
        ex = []
        for r in trims:
            b4 = by[(r['n'], r['bank'], r['Rp'], 4)]
            wc = max(r['dep_cases'], key=r['dep_cases'].get)
            others = [k for k in r['dep_cases'] if k != wc and k in b4['dep_cases']]
            ex.append(f"$R'={r['Rp']}$, $\\kappa={r['kappa']}$: {100 * (max(r['dep_cases'][k] for k in others) - max(b4['dep_cases'][k] for k in others)):+.3f} pp "
                      f"without case {wc}")
        same_m = all(r['mgau'] == by[(r['n'], r['bank'], r['Rp'], 4)]['mgau'] and r['mlat'] == by[(r['n'], r['bank'], r['Rp'], 4)]['mlat']
                     for r in trims)
        L.append('**What the 3D trim numbers say (PROVISIONAL, post-hoc selection):** no tested trim meets the H3 accuracy '
                 'requirement; the worst-error growth survives removing the worst case (' + '; '.join(ex) + '). Trimming $M$ '
                 'lowers the setting-panel query time but ' + ('does not change' if same_m else 'does not reliably lower') +
                 ' the primary $m^\\star$ at fixed $R\'$ — the mechanism registered for 1d is contradicted in 3D as well. The '
                 'off-mesh rule is far more accurate than the mesh tensor against the refined reference, reproducing the '
                 '2026-10-01 lane\'s validation pattern.\n')
    # H4 (3D dial on the new bank)
    for n in sorted({r['n'] for r in recs}):
        a_, b_ = by.get((n, 'W1024', 1024, 4)), by.get((n, 'W1024', 512, 4))
        if a_ and b_:
            if None in (a_['dep_worst'], b_['dep_worst']):
                v = 'unavailable'
            elif abs(a_['dep_worst'] - b_['dep_worst']) < PP:
                v = 'unresolved'
            elif a_['dep_worst'] <= 0.8 * b_['dep_worst'] and a_['dep_med'] < b_['dep_med']:
                v = 'meets the registered bar'
            else:
                v = 'does not meet the registered bar'
            imp = [b_['dep_cases'][k] - a_['dep_cases'][k] for k in a_['dep_cases'] if k in b_['dep_cases']]
            L.append(f"**H4 at {n - 1}³ ($R'=1024$ vs 512, new bank), PROVISIONAL, reference-limited:** *{v}* — worst "
                     f"{pct(b_['dep_worst'])} → {pct(a_['dep_worst'])} ({100 * (a_['dep_worst'] - b_['dep_worst']):+.3f} pp, "
                     f"{100 * (a_['dep_worst'] / b_['dep_worst'] - 1):+.1f} %), median {pct(b_['dep_med'])} → {pct(a_['dep_med'])}; "
                     f"{sum(x > 0 for x in imp)}/{len(imp)} cases improved; projection floor {pct(b_['floor'])} → {pct(a_['floor'])}.\n")
    if d.get('cross_mesh'):
        L.append('Cross-mesh lattice distance of the same arm between 64³ and 128³ (worst over cases, relative to $\\lVert u_0\\rVert$): ' +
                 '; '.join(f"{k.replace('|', ' ')} {sci(v['worst'])}" for k, v in sorted(d['cross_mesh'].items())
                           if k.split('|')[-1] in ('conv',) or any(r.get('dep') and r['dep']['arm'] == k.split('|')[-1] for r in recs)) + '.\n')
    return recs


def plots_3d(recs, tag):
    if not recs:
        return
    f, ax = fig()
    i = 0
    for bn in sorted({r['bank'] for r in recs}):
        for n in sorted({r['n'] for r in recs}):
            rs = sorted([r for r in recs if r['bank'] == bn and r['n'] == n and r['kappa'] == 4 and r['ms'] and r['tv']], key=lambda r: r['Rp'])
            if rs:
                ax.plot([r['Rp'] for r in rs], [r['ms'] for r in rs], color=SERIES[i % 4], lw=2, marker=MARKERS[i % 4], ms=7,
                        label=f'{bn}, {n - 1}³')
                i += 1
    ax.set_xticks(sorted({r['Rp'] for r in recs}))
    style(ax, f"3D query time of the deployed rule vs R'\n(per-setting A–B–A medians, {recs[0]['gpu']}; K-time-valid jobs only)", "R'",
          'median ms per query')
    ax.legend(frameon=False, fontsize=8)
    save(f, f'3d_{tag}_cost_vs_Rp.png')
    f, ax = fig()
    i = 0
    for fam, key in (('lattice', 'mlat'), ('Gauss', 'mgau')):
        for n in sorted({r['n'] for r in recs}):
            rs = sorted([r for r in recs if r['n'] == n and r[key]], key=lambda r: (r['Rp'], r['M']))
            if rs:
                ax.plot([r['M'] for r in rs], [r[key] for r in rs], color=SERIES[i % 4], lw=0, marker=MARKERS[i % 4], ms=8,
                        label=f'{fam}, {n - 1}³')
                i += 1
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    plain(ax, 'both')
    style(ax, '3D points needed ($m^\\star$, $\\tau=2.5\\times10^{-4}$) vs test count M', 'M', '$m^\\star$')
    ax.legend(frameon=False, fontsize=8)
    save(f, f'3d_{tag}_mstar_vs_M.png')
    banks = sorted({r['bank'] for r in recs})
    meshes = sorted({r['n'] for r in recs})
    f, ax = fig()
    i = 0
    for bn in banks:
        for n in meshes:
            rs = sorted([r for r in recs if r['bank'] == bn and r['n'] == n and r['kappa'] == 4 and r['dep_worst']],
                        key=lambda r: r['Rp'])
            if rs:
                ax.plot([r['Rp'] for r in rs], [100 * r['dep_worst'] for r in rs], color=SERIES[i % 4], lw=2,
                        marker=MARKERS[i % 4], ms=7, label=f'{bn}, {n - 1}³: deployed worst')
                i += 1
    ax.set_xticks(sorted({r['Rp'] for r in recs}))
    style(ax, f"3D dial: worst refined error vs R' (M ≈ 4R', PROVISIONAL)", "R'", 'worst relative error (%)')
    ax.legend(frameon=False, fontsize=8)
    save(f, f'3d_{tag}_error_vs_Rp.png')
    f, ax = fig()
    xs = np.array([256, 384, 512, 640, 768, 896, 1024])
    ax.plot(xs, 8 * 4 * xs ** 3 / 1e9, color=SERIES[1], lw=2, marker=MARKERS[1], ms=7, label='tensor, $8\\cdot4R\'\\cdot R\'^2$')
    rs = sorted([r for r in recs if r['kappa'] == 4 and r['bytes']], key=lambda r: r['Rp'])
    if rs:
        ax.plot([r['Rp'] for r in rs], [r['bytes'] / 1e9 for r in rs], color=SERIES[0], lw=0, marker=MARKERS[0], ms=8,
                label='off-mesh rule at $m^\\star$ (measured settings)')
    ax.axhline(141, color=INK2, lw=1, ls='--')
    ax.annotate('H200 memory (141 GB)', (xs[0], 141), textcoords='offset points', xytext=(0, 4), color=INK2, fontsize=8)
    ax.set_yscale('log')
    plain(ax)
    style(ax, "3D advection memory vs R': tensor vs off-mesh", "R'", 'GB (log scale)')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    save(f, f'3d_{tag}_memory_vs_Rp.png')


# ----------------------------------------------------------------------------------------------------- main
def main():
    L = ['# C1 wide bank: the accuracy dial with wider banks under off-mesh quadrature\n',
         'Lane C1 of the JCP campaign: how far the accuracy dial extends when the ordered coordinate-network bank is '
         'widened, now that the off-mesh rule stores $m(2R\'+M)$ numbers instead of the $MR\'^2$ tensor. '
         '**Status: PROVISIONAL** — every accuracy number is scored against first-order references (the second-order '
         'references lane has not run); verdicts are benchmark-relative. Selection used validation cohorts only; no test '
         'cohort was opened. Every number below is generated by `make_report.py` from the run JSONs; the pre-registration '
         'is `DESIGN.md` (amendments A0–A5).\n']
    d1 = load('runs/j1/archive/output/result.json')
    a1 = load('checks/j1-audit.json')
    res = {}
    if d1:
        res['2d'] = section_2d(d1, a1, L)
    t = load('runs/j3b/archive/output/training.json')
    res['train'] = section_train(t, L)
    d2 = load('runs/j2/archive/output/result_A7.json') or load('runs/j2/archive/output/result.json')
    a2 = load('checks/j2-audit.json')
    r2 = section_3d(d2, '3D: test-count trim on the old bank (1d)', L, 'j2', audit=a2)
    plots_3d(r2, 'j2')
    d4 = load('runs/j4/archive/output/result.json')
    r4 = section_3d(d4, '3D: the scaling law on the wider bank (1c)', L, 'j4', audit=load('checks/j4-audit.json'))
    plots_3d(r4, 'j4')
    L.append(GLOSSARY)
    (HERE / 'report.md').write_text('\n'.join(L) + '\n')
    print('report.md written;', 'plots:', sorted(p.name for p in PLOTS.glob('*.png')) if PLOTS.exists() else [])


GLOSSARY = r"""## Glossary

- **bank, $R'$**: the frozen coordinate network $\hat G(x)$ whose columns are ordered by importance; a solve uses the first $R'$ columns, $u(x)=\hat G_{R'}(x)c$. **Wider bank**: more columns available.
- **$M$, $\kappa=M/R'$**: number of sine test functions in the reduced residual and its ratio to $R'$; $\kappa=4$ is the existing convention. **Trim**: using $\kappa=2$ or 3.
- **off-mesh rule, $m$**: a fixed quadrature (Gauss $p^d$ or a rank-1 lattice / Fibonacci lattice) with $m$ points placed independently of the mesh, evaluating the advection term with the bank's exact gradient. **tensor**: the precomputed quadratic form it replaces, $MR'^2$ numbers.
- **converged rollout**: the same solve with a very fine rule (2D Gauss $640^2$, 3D Gauss $48^3$/$56^3$); its check uses a finer or coarser rule (2D $768^2$, 3D $40^3$/$48^3$).
- **$m^\star$**: the smallest rule in a family's ladder whose rollouts are eligible on every validation case, stay within $\tau$ of the converged rollout, and whose $\rho$ is at most 0.116. **$\tau$**: $2.5\times10^{-4}$ (primary) or $10^{-3}$ (secondary), relative to $\lVert u_0\rVert$. **$m_d$ / $m_\rho$**: the sizes chosen by only the distance or only the $\rho$ criterion (diagnostics).
- **deployed rule**: the $m^\star$ rule (Gauss or lattice) with the lower measured query time in its setting's own timing panel.
- **$\rho$**: relative error of a rule's tested advection against the continuum target on the converged rollout's reached states.
- **ST / S reference**: first-order references at $8192^2$: ST with time step $\Delta t/16$ (space + time), S with the ROM's $\Delta t$ (space only). In 3D: the 513-node, $\Delta t/4$ reference.
- **worst / median**: over validation cases, of the largest relative error over the five evolved output times. **pp**: percentage points.
- **projection floor**: the best possible error of the span against the reference (least squares at the shared nodes), with no time stepping, tests or quadrature.
- **eligible**: 2D, per rollout: finite, no damping-limit exit, at most 1 % of its steps on the iteration budget. 3D (A7), per arm and cohort: every rollout finite, no reason-3 exit, and non-stationary steps (reasons 0 and 2) at most 1 % of all pooled steps.
- **adaptive_first**: number of initial time steps solved by full adaptive Levenberg–Marquardt before the solver switches to one cached-Jacobian sweep per step (3 in J2; 6 in J4 by amendment A8).
- **K-conv, K-target, controls**: gates: the converged rollout agrees with its check; the continuum target agrees with its check; the under-resolved control rules (Gauss $8^2$, Smolyak-8; 3D `lat256`, `smol8`) must not pass both selection criteria.
- **A–B–A timing, K-time**: timed reduced solves, then a fixed baseline, then the solves again, on one GPU in one job; K-time requires drift within 10 % and outputs identical to the untimed run. **paired ratio**: per case and phase, trimmed-setting time divided by the $\kappa=4$ setting's time.
- **span floor (3D bank table)**: least-squares projection error of native-grid bank-validation snapshots onto the first $R'$ bank columns, worst over 96 cases × six times, each relative to its own norm.
- **B1–B4′**: acceptance checks of the new 3D bank (finite training; ordering inverse and conditioning; equal-width comparison with the old bank; span gain from 512 to 1024).
- **PROVISIONAL**: scored against first-order references; not a physical-accuracy claim.
"""

if __name__ == '__main__':
    main()
