"""Generate report.md and plots/*.png of jcp-combined from the committed analysis JSONs (no number is typed by hand).

Inputs (whatever exists; missing parts are rendered as 'pending'):
  checks/analysis-c2grid-2d_first_order.json   (analyze2d.py)       checks/scores-c2grid-2d_first_order.json (score.py)
  checks/SELECTION-3D.json, checks/SELECTION-3D-POSTHOC.json     (select3d.py; pre-registered and A10/A11 post hoc)
  checks/cnfail3d-c3desc*.json                                   (cnfail3d.py)
  checks/analysis3d-c3grid.json                                  (analyze3d.py)
  checks/budget2d-c2bud*-2d_first_order.json, checks/budget3d-c3bud-3d_first_order.json  (budget2d.py / budget3d.py)
  runs/JOB-LEDGER.json, audits/codex-*.md, runs/c3desc*/archive/output/result.json (per-case escalation distances)

    python make_report.py            -> report.md, plots/
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
CK = HERE / 'checks'
PL = HERE / 'plots'
DT0_2D = .005
BAR = 2.5e-5
TAU = 2.5e-4


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def pc(x, nd=3):
    """percent of ||u0|| (accuracy numbers are PROVISIONAL)."""
    return '—' if x is None else f'{100 * x:.{nd}g} %'


def sci(x, nd=2):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}e}'


def fx(x, nd=3):
    return '—' if x is None else f'{x:.{nd}g}'


def armname(k):
    """'R256|LSPG|CNR|0.005|prod' -> 'LSPG-CN-R, Δt = 1 Δt0'."""
    p = k.split('|')
    off = 1 if len(p) == 5 else 0
    form, sc, dt = p[off], p[off + 1], float(p[off + 2])
    scn = {'CNR': 'CN-R'}.get(sc, sc)
    f = dt / DT0_2D
    fs = f'{f:.4g}' if abs(f - round(f)) > 1e-9 else f'{int(round(f))}'
    if abs(f - 5 / 3) < 1e-6:
        fs = '5/3'
    return f'{form}-{scn} at {fs}Δt₀'


class R:
    def __init__(self):
        self.lines = []

    def __call__(self, *s):
        self.lines.extend(s)

    def table(self, head, rows):
        self.lines.append('| ' + ' | '.join(head) + ' |')
        self.lines.append('|' + '|'.join('---' for _ in head) + '|')
        for r in rows:
            self.lines.append('| ' + ' | '.join(str(x) for x in r) + ' |')
        self.lines.append('')


# --------------------------------------------------------------------------------------------------------- plots
def plot_pareto2d(an, tag):
    arms = an['arms']
    fig, ax = plt.subplots(figsize=(7.5, 5))
    cols = dict(fast='#9ca3af', R128='#60a5fa', R256='#2563eb', acc='#a78bfa', R384='#7c3aed', R512='#db2777')
    for k, x in arms.items():
        s, form, sc, dt, lev = k.split('|')
        if lev != 'prod' or not x.get('eligible') or not x.get('ratio_common') or sc == 'CN' and form == 'LSPG':
            continue
        be = sc == 'BE'
        ax.scatter(x['ratio_common']['median'], 100 * x[f'median_{tag}'], s=28 if be else 20,
                   marker='o' if be else '^', facecolors='none' if be else cols[s], edgecolors=cols[s], linewidths=1.2)
    for s, c in cols.items():
        ax.scatter([], [], marker='^', color=c, label=s)
    ax.scatter([], [], marker='o', facecolors='none', edgecolors='k', label='LSPG-BE (open)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('median paired time ratio vs common baseline (acc LSPG-BE Δt₀)')
    ax.set_ylabel(f'median error vs {tag} (% of ‖u₀‖, PROVISIONAL)')
    ax.set_title(f'2D error–cost: BE (open) vs second order (filled), by bank width — {tag}')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.3, which='both')
    fig.tight_layout()
    fig.savefig(PL / f'pareto2d_{tag}.png', dpi=130)
    plt.close(fig)


def plot_dial2d(an):
    arms = an['arms']
    dial = ['R128', 'R256', 'R384', 'R512']
    rp = [128, 256, 384, 512]
    g = an['selection']['global']['scheme']
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for tag, ls in (('TX', '-'), ('ST', '--')):
        be = [arms[f'{s}|LSPG|BE|0.005|prod'][f'median_{tag}'] * 100 for s in dial]
        sel = [an['selection']['per_scheme_setting'][f'{s}|{g}']['accurate_equal_cost'] for s in dial]
        so = [arms[f"{s}|{x['key']}"][f'median_{tag}'] * 100 for s, x in zip(dial, sel)]
        ax.plot(rp, be, 'o' + ls, color='#6b7280', label=f'LSPG-BE Δt₀ vs {tag}')
        ax.plot(rp, so, '^' + ls, color='#db2777', label=f'{g} accurate-equal-cost vs {tag}')
    ax.set_xlabel("bank width R′")
    ax.set_ylabel('median error (% of ‖u₀‖, PROVISIONAL)')
    ax.set_yscale('log')
    ax.set_xticks(rp)
    ax.set_title('2D dial under BE and second order')
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, which='both')
    fig.tight_layout()
    fig.savefig(PL / 'dial2d.png', dpi=130)
    plt.close(fig)


def plot_j4(esc_rows):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    order = ['gl48|gl56', 'gl56|gl64', 'gl64|gl80', 'gl80|gl96']
    cols = dict(M2_512='#9ca3af', W256='#60a5fa', W512='#2563eb', W768='#7c3aed', W1024='#db2777')
    for s, pairs in esc_rows.items():
        xs = [i for i, p in enumerate(order) if p in pairs]
        ax.plot(xs, [pairs[order[i]]['be'] for i in xs], 'o-', color=cols.get(s, 'k'), label=f'{s} fixed-sweep BE')
        ax.plot(xs, [pairs[order[i]]['adcn'] for i in xs], '^--', color=cols.get(s, 'k'), label=f'{s} adaptive CN')
    ax.axhline(BAR, color='k', lw=1, ls=':', label='bar 2.5e-5')
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([o.replace('gl', 'Gauss ').replace('|', ' vs ') for o in order], fontsize=8)
    ax.set_yscale('log')
    ax.set_ylabel('worst-case successive-rule distance (field, /‖u₀‖)')
    ax.set_title('3D J4 escalation: successive Gauss rules (64 cases)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=.3, which='both')
    fig.tight_layout()
    fig.savefig(PL / 'j4_escalation.png', dpi=130)
    plt.close(fig)


def plot_fom(an):
    m = an['hypotheses']['H5']['ST']['matches']
    best = {}
    for k, v in m.items():
        s, form, sc = k.split('|')[:3]
        g = 'LSPG-BE' if sc == 'BE' else f'{form}-{sc}'
        best[(s, g)] = max(best.get((s, g), 0.), v['speedup'])
    sets = ['fast', 'R128', 'R256', 'acc', 'R384', 'R512']
    groups = sorted({g for _, g in best})
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = .8 / len(groups)
    for i, g in enumerate(groups):
        ax.bar(np.arange(len(sets)) + i * w, [best.get((s, g), 0.) for s in sets], w, label=g)
    ax.axhline(1, color='k', lw=.8)
    ax.set_xticks(np.arange(len(sets)) + .4 - w / 2)
    ax.set_xticklabels(sets)
    ax.set_ylabel('best matched speed-up vs FOM (same job, same GPU)')
    ax.set_title('2D: ROM vs second-order FOM at matched worst error (ST), PROVISIONAL')
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(PL / 'fom2d.png', dpi=130)
    plt.close(fig)


def plot_speedup3d(an3):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cols = dict(M2_512='#9ca3af', W256='#60a5fa', W512='#2563eb', W768='#7c3aed', W1024='#db2777')
    mk = dict(CN='^', CNR='s', BDF2='D', BE='o')
    for s, S in an3['settings'].items():
        comp = S['arms']['fs|BE|0.01|K1']
        for k, x in S['arms'].items():
            if not k.startswith('fs|') or x.get('ratio') is None:
                continue
            sc = k.split('|')[1]
            el = x.get('eligible_H4', k == 'fs|BE|0.01|K1')
            ax.scatter(1 / x['ratio'], x['anchor_median_all'] / max(comp['anchor_median_all'], 1e-300), marker=mk.get(sc, 'x'),
                       facecolors=cols.get(s, 'k') if el else 'none', edgecolors=cols.get(s, 'k'), s=30)
    for s, c in cols.items():
        ax.scatter([], [], marker='o', color=c, label=s)
    for sc, m in mk.items():
        ax.scatter([], [], marker=m, color='k', label=sc)
    ax.axvline(1, color='k', lw=.6)
    ax.axhline(1, color='k', lw=.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('speed-up vs deployed fixed-sweep BE Δt₀ (1 / paired time ratio)')
    ax.set_ylabel('median anchor discrepancy / BE comparator')
    ax.set_title('3D fixed-sweep second order: speed vs time accuracy (open = not H4-eligible)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=.3, which='both')
    fig.tight_layout()
    fig.savefig(PL / 'speedup3d.png', dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------- main
def main():
    PL.mkdir(exist_ok=True)
    r = R()
    an = load(CK / 'analysis-c2grid-2d_first_order.json')
    sc2 = load(CK / 'scores-c2grid-2d_first_order.json')
    sel_pre = load(CK / 'SELECTION-3D.json')
    sel_ph = load(CK / 'SELECTION-3D-POSTHOC.json')
    an3 = load(CK / 'analysis3d-c3grid.json')
    b2 = [load(p) for p in sorted(glob.glob(str(CK / 'budget2d-c2bud*-2d_first_order.json')))]
    b3 = load(CK / 'budget3d-c3bud-3d_first_order.json')
    ledger = load(HERE / 'runs' / 'JOB-LEDGER.json')
    done = dict(grid2d=an is not None, sel3d=sel_ph is not None, grid3d=an3 is not None, bud2d=bool(b2), bud3d=b3 is not None)
    pending = [k for k, v in done.items() if not v]

    r('# jcp-combined: second-order stepping with a wide bank and a fast fixed-sweep solver, with an error budget', '',
      '**Status:** ' + ('all stages complete. ' if not pending else f"interim; pending: {', '.join(pending)}. ") +
      'Every accuracy number is **PROVISIONAL** (scored against first-order references: 2D ST/S/TX, 3D the BE '
      'reference at Δt₀/4) and is marked so where it appears. The 3D rule selection is **POST HOC** (DESIGN A10, A11): '
      'the pre-registered selection was unavailable. Lane `exp/2026-10-13-jcp-combined`, generated by `make_report.py` '
      'from the committed analysis files.', '')

    # ------------------------------------------------------------------------------------------------ pipeline
    r('## How the work was done', '',
      '```mermaid', 'flowchart TB',
      '  D[DESIGN + amendments A1-A12, Codex-audited]:::doc --> SM[cluster smokes: smk2d, smk3esc, smk3grid, smk2bud, smk3bud, smk3ext2]:::job',
      '  SM --> G2[c2grid: 2D grid, anchors, FOM, timing]:::job',
      '  SM --> E3[c3descW1024, c3descW768, c3descB: 3D escalation + rule ladder]:::job',
      '  E3 --> X3[c3extW1024: W1024 escalation extension, POST HOC A11]:::posthoc',
      '  G2 --> S2[score + analyze2d + freeze 2D]:::doc',
      '  E3 --> S3[select3d: pre-registered = unavailable; POST HOC A10 selection]:::posthoc',
      '  X3 --> S3',
      '  S3 --> C3[c3grid: 3D second-order grid + timing]:::job',
      '  S2 --> B2[c2budA/B/C: 2D error budget]:::job',
      '  C3 --> B3[c3bud: 3D error budget]:::job',
      '  B2 --> REP[make_report.py]:::doc',
      '  B3 --> REP',
      '  classDef doc fill:#e5e7eb,stroke:#374151;',
      '  classDef job fill:#dbeafe,stroke:#1d4ed8;',
      '  classDef posthoc fill:#fde68a,stroke:#b45309;',
      '```', '')

    # ------------------------------------------------------------------------------------------------ 2D
    r('## 2D (Burgers 2D, dev6 + val32 = 38 cases)', '')
    if an is None:
        r('Pending.', '')
    else:
        g = an['selection']['global']['scheme']
        gl = an['selection']['global']
        if sc2:
            gt = sc2['gate']
            r(f"**Scoring gate.** `score.py` re-decoded every saved trajectory on CPU and reproduced {gt['compared']} of "
              f"{gt['expected']} job values (worst difference {sci(gt['worst'])}); every negative control broke agreement "
              f"(case swap {sci(sc2['controls']['case_swap'])}, time shift {sci(sc2['controls']['time_shift'])}, ST↔S swap "
              f"{sci(sc2['controls']['ST_S_swap'])}).", '')
        r(f"**Tie thresholds** (heuristic, from ST−TX): medians {pc(an['tie']['median'])}, worst values "
          f"{pc(an['tie']['worst'])}; paired per-case ties at $2\\varepsilon_{{\\rm ref}}$.", '')
        r('### Which second-order scheme', '',
          f"Scheme wins over the four dial settings: " + ', '.join(f"{k} {v}" for k, v in gl['points'].items()) +
          f"; tie broken by the smaller sum of the four medians (" +
          ', '.join(f"{k} {pc(v)}" for k, v in gl['sums'].items()) + f"). **Selected scheme: {g}.**", '')
        rows = []
        for s in ['fast', 'R128', 'R256', 'acc', 'R384', 'R512']:
            be = an['arms'][f'{s}|LSPG|BE|0.005|prod']
            ps = an['selection']['per_scheme_setting'][f'{s}|{g}']
            for kind, lab in (('accurate_equal_cost', 'accurate, equal cost'), ('fast_equal_accuracy', 'fast, equal accuracy')):
                x = ps.get(kind)
                if not x:
                    rows.append([s, lab, 'none', '—', '—', '—', '—', '—'])
                    continue
                a = an['arms'][f"{s}|{x['key']}"]
                rows.append([s, lab, armname(x['key']), fx(x['cost']), pc(a['median_TX']), pc(a['worst_TX']),
                             pc(a['median_ST']), pc(a['median_S'])])
            rows.append([s, 'baseline', 'LSPG-BE at 1Δt₀', '1', pc(be['median_TX']), pc(be['worst_TX']), pc(be['median_ST']),
                         pc(be['median_S'])])
        r('Per setting, the selected scheme\'s two arms against that setting\'s deployed LSPG-BE (errors PROVISIONAL):', '')
        r.table(['setting', 'role', 'arm', 'cost / own BE', 'median TX', 'worst TX', 'median ST', 'median S'], rows)

        H = an['hypotheses']
        r('### H1: does the wide bank pay off once time error is removed?', '')
        rows = []
        for step, v in H['H1'].items():
            for tag, k in (('TX', 'TX'), ('ST', 'ST_sensitivity')):
                x = v[k]
                rows.append([step, tag, v['verdict'] if tag == 'TX' else '', pc(x['median_from']), pc(x['median_to']),
                             f"{100 * x['median_change']:+.1f} %", f"{x['improved']}/{x['worsened']}/{x['ties']}",
                             pc(x['worst_from']), pc(x['worst_to'])])
            for tag in ('TX', 'ST'):
                x = v['BE_dt0'][tag]
                rows.append([step, f'{tag} (LSPG-BE Δt₀)', '', pc(x['median_from']), pc(x['median_to']),
                             f"{100 * x['median_change']:+.1f} %", f"{x['improved']}/{x['worsened']}/{x['ties']}",
                             pc(x['worst_from']), pc(x['worst_to'])])
        r.table(['dial step', 'reference', 'verdict', 'median from', 'median to', 'change', 'improved/worsened/tied',
                 'worst from', 'worst to'], rows)
        r('A step is **reference-limited** when fewer than half of the 38 cases are resolved (paired difference above '
          '$2\\varepsilon_{\\rm ref}$): the first-order references cannot rank those arms. Verdicts: ' +
          '; '.join(f"{k} **{v['verdict']}**" for k, v in H['H1'].items()) + '.', '')
        r('![2D dial](plots/dial2d.png)', '')

        h2 = H['H2']
        r('### H2: does tunability survive?', '',
          f"(a) Every eligible LSPG-BE arm is covered by a second-order arm with lower or equal cost and error: worst-TX "
          f"{h2['a']['worst_TX']['covered']}/{h2['a']['worst_TX']['total']}, median-TX {h2['a']['median_TX']['covered']}/"
          f"{h2['a']['median_TX']['total']} → **{'pass' if h2['a_pass'] else 'fail'}**. (b) Along the dial, the "
          f"accurate-equal-cost arms' common-baseline cost and median TX: " +
          '; '.join(f"{d['setting']} {fx(d['cost_common'])}× / {pc(d['median_TX'])}" for d in h2['dial']) +
          f" → **{'pass' if h2['b'] else 'fail'}**.", '',
          '![2D Pareto TX](plots/pareto2d_TX.png)', '', '![2D Pareto ST](plots/pareto2d_ST.png)', '')

        h3 = H['H3']
        r('### H3: why did the fast setting\'s worst case get worse under CN?', '')
        rows = []
        for tag in ('ST', 'TX'):
            p = h3[tag]['primary']
            rows.append([tag, h3[tag]['worst_case'], p['t_star'], pc(p['e_BE']), pc(p['e_CN']), pc(p['e_inf']), fx(p['cos']),
                         ' '.join('yes' if x else 'no' for x in p['guards']), p['verdict']])
        r.table(['reference', 'worst case', 'output index t*', 'e(BE Δt₀)', 'e(CN Δt₀)', 'e(anchor) = e∞', 'cos(δ_BE, ε∞)',
                 'criteria a b c', 'verdict'], rows)
        wc = h3['ST']['worst_case']
        comp = []
        for s in ('fast', 'R128', 'R256'):
            x = an['selection']['per_scheme_setting'][f'{s}|{g}']['accurate_equal_cost']
            comp.append(f"{s} {g} accurate arm worst ST {pc(an['arms'][f'{s}|' + x['key']]['worst_ST'])}")
        p_ = h3['ST']['primary']
        failed = [n for n, ok_ in zip('abc', p_['guards']) if not ok_]
        r(f"Pre-registered explanation (error cancellation) at the worst case {wc}: **{p_['verdict']}** (ST; TX: "
          f"{h3['TX']['primary']['verdict']})" + (f"; failing criteria: {', '.join(failed)}" if failed else '') +
          f". Criterion (a) needs |e(CN) − e∞| ≤ 0.2 |e(BE) − e∞|; here |e(CN) − e∞| = {pc(abs(p_['e_CN'] - p_['e_inf']))} "
          f"and |e(BE) − e∞| = {pc(abs(p_['e_BE'] - p_['e_inf']))}. The fast setting's LSPG-BE worst ST is "
          f"{pc(an['arms']['fast|LSPG|BE|0.005|prod']['worst_ST'])}; " + '; '.join(comp) + '. The budget names the '
          'component that limits that case.', '')

        h5 = H['H5']
        cnr = {k: v for k, v in h5['ST']['matches'].items() if f'|{g.replace("-", "|")}|' in f'|{k}'}
        bc = max(cnr.items(), key=lambda kv: kv[1]['speedup']) if cnr else None
        r('### H5: speed against the second-order full-order model (same job, same GPU)', '',
          f"Best matched speed-up of any eligible ROM arm: **{fx(h5['ST']['best_combined'][0])}×** "
          f"({h5['ST']['best_combined'][1]}, matched on worst ST); the C2 `acc` arms re-measured in this job: "
          f"{fx(h5['ST']['best_comparator'][0])}× ({h5['ST']['best_comparator'][1]}). Verdict: **{h5['ST']['verdict']}** "
          f"(TX: {h5['TX']['verdict']}). Scope: the winner is an LSPG-CN arm (not second order, excluded from scheme "
          f"selection); the best matched speed-up of the selected scheme {g} is "
          + (f"{fx(bc[1]['speedup'])}× ({bc[0]})." if bc else 'not available.') +
          ' The FOM is second order in time only; its advection stays sign-upwind (first order in space).', '',
          '![FOM comparison](plots/fom2d.png)', '')
        plot_pareto2d(an, 'TX')
        plot_pareto2d(an, 'ST')
        plot_dial2d(an)
        plot_fom(an)

    # ------------------------------------------------------------------------------------------------ 3D escalation
    r('## 3D (Burgers 3D, 65³, validation cohort 923801, 64 cases)', '')
    esc_rows = {}
    per_case_note = []
    for run in sorted(glob.glob(str(HERE / 'runs' / 'c3desc*' / 'archive' / 'output' / 'result.json'))):
        res = load(run)
        for s, v in res['settings'].items():
            pr = v['escalation']['pairs']
            esc_rows[s] = {p: dict(be=x['selection']['fs|BE|0.01|K1']['worst'],
                                   adcn=x['diagnostic']['ad|LSPG|CN|0.025|tight']['worst'],
                                   sel=x['selection_pair_status'], ad=x['adaptive_diagnostic_status'],
                                   n_over=int(np.sum(np.array(x['selection']['fs|BE|0.01|K1']['per_case']) > BAR)),
                                   argmax=int(np.argmax(x['selection']['fs|BE|0.01|K1']['per_case'])))
                           for p, x in pr.items()}
    for run in sorted(glob.glob(str(HERE / 'runs' / 'c3ext*' / 'archive' / 'output' / 'result.json'))):
        res = load(run)
        for s, v in res['settings'].items():
            for p, x in v['extension']['pairs'].items():
                esc_rows.setdefault(s, {})[p] = dict(be=x['posthoc']['fs|BE|0.01|K1']['worst'],
                                                     adcn=x['posthoc']['ad|LSPG|CN|0.025|tight']['worst'],
                                                     sel=x['selection_pair_status'], ad=x['adaptive_diagnostic_status'],
                                                     n_over=int(np.sum(np.array(x['posthoc']['fs|BE|0.01|K1']['per_case']) > BAR)),
                                                     argmax=int(np.argmax(x['posthoc']['fs|BE|0.01|K1']['per_case'])))
    if esc_rows:
        r('### The converged-rule gate (J4) and why it failed at R′ ≥ 768', '')
        rows = []
        for s in ['M2_512', 'W256', 'W512', 'W768', 'W1024']:
            for p, x in esc_rows.get(s, {}).items():
                rows.append([s, p.replace('gl', 'Gauss ').replace('|', ' vs '), sci(x['be']), sci(x['adcn']), x['n_over'],
                             f"case {x['argmax']}", x['sel'], x['ad']])
        r.table(['setting', 'rule pair', 'worst dist. fixed BE', 'worst dist. adaptive CN', 'cases > bar (fixed BE)',
                 'worst case', 'pre-registered axis', 'adaptive axis'], rows)
        nover = {s: max((x['n_over'] for x in v.values()), default=0) for s, v in esc_rows.items()}
        r('The pre-registered selection axis is "unresolved" wherever its second arm (fixed-sweep LSPG-CN, $K=2$, '
          '$\\tfrac52\\Delta t_0$) is ineligible (next subsection). Largest number of cases above the bar at any pair '
          '(fixed-sweep BE): ' + ', '.join(f'{s} {n}' for s, n in nover.items()) + '.', '',
          '![J4 escalation](plots/j4_escalation.png)', '')
        plot_j4(esc_rows)

    cnf = {Path(p).stem.replace('cnfail3d-', ''): load(p) for p in sorted(glob.glob(str(CK / 'cnfail3d-c3desc*.json')))}
    if cnf:
        r('### The fixed-sweep CN arm fails on a fixed subset of cases', '')
        rows = []
        for job, d in cnf.items():
            for s, v in d['settings'].items():
                nc = {k: x for k, x in v['rules'].items() if k != 'lat256'}
                med = np.median([x['median_worst'] for x in nc.values()])
                mx = max(x['max_worst'] for x in nc.values())
                rows.append([s, len(nc), ', '.join(map(str, v['failing_union_non_control'])),
                             'yes' if v['failing_set_identical_across_non_control_rules'] else
                             ('no' if v['failing_set_identical_across_non_control_rules'] is False else 'n/a'),
                             sci(med), sci(mx)])
        r.table(['setting', 'rules', 'failing cases (distance > 2.5e-4)', 'same set at every rule', 'median of per-rule median distance',
                 'largest distance'], rows)
        r('On the failing cases the two LM sweeps per step leave most steps, from the first step after the adaptive start '
          'on, non-stationary (gradient norm far above the 10⁻³ stationarity test), and the distance to the adaptive solve '
          'then grows over the rollout; on every other case the '
          'fixed-sweep CN trajectory agrees with the verified adaptive solve. The set does not depend on the rule, so it '
          'is a solver failure, not a quadrature effect. Per-case evidence: `checks/cnfail3d-*.json`.', '')

    def sel_table(sel, title):
        rows = []
        for s in ['M2_512', 'W256', 'W512', 'W768', 'W1024']:
            v = sel['settings'].get(s)
            if v is None:
                continue
            conv = v.get('posthoc_converged', v.get('converged'))
            rows.append([s, conv or 'none', v['status'], (v.get('mstar') or {}).get('lattice') or '—',
                         (v.get('mstar') or {}).get('gauss') or '—', v.get('deployed') or '—',
                         ', '.join(v.get('fallbacks') or []) or '—'])
        r(title, '')
        r.table(['setting', 'converged rule', 'status', 'm* lattice', 'm* Gauss', 'deployed', 'fallbacks'], rows)
    if sel_pre:
        sel_table(sel_pre, '### Rule selection, pre-registered (DESIGN 3.3 / A6 / A7)')
    if sel_ph:
        sel_table(sel_ph, f"### Rule selection, {sel_ph['label']}")
        w = sel_ph['settings'].get('W1024', {}).get('extension')
        if w:
            r(f"W1024 extension (A11): pair statuses recomputed {w['pair_status_recomputed']} "
              f"(agree with the job: {w['statuses_agree']}); 80³ reproduction across jobs "
              + ', '.join(f"{k} {sci(v)}" for k, v in w['reproduction_gl80'].items()) + '.', '')

    r('### 3D second-order grid (H4, H6) and speed-up', '')
    if an3 is None:
        r('Pending (`c3grid`).', '')
    else:
        r(f"Rule selection label carried by the grid: **{an3.get('rule_selection_label')}**. Comparator: the deployed "
          'fixed-sweep BE at Δt₀ (K = 1) on the same bank, R′ and rule. Anchor discrepancy = distance to GAL-BDF2 at '
          'Δt₀/8 (time-converged, reference-free); reference errors are reference-limited (PROVISIONAL).', '')
        rows = []
        for s, h in an3['hypotheses']['H4'].items():
            sel = an3['selection'].get(s, {})
            A = an3['settings'][s]['arms']
            comp = A['fs|BE|0.01|K1']
            row = [s, an3['settings'][s]['rule'], f"{h.get('population', '—')}/{h.get('cases', '—')}",
                   h.get('verdict') or h.get('H4a', {}).get('verdict', '—'), h.get('H4b', {}).get('verdict', '—')]
            for kind in ('fast_equal_accuracy', 'accurate_equal_cost'):
                k = sel.get(kind)
                if k:
                    x = A[k]
                    row += [k, fx(x['ratio']), sci(x['anchor_median_all']), pc(x['median_ref'])]
                else:
                    row += ['none', '—', '—', '—']
            row += [sci(comp['anchor_median_all']), pc(comp['median_ref'])]
            rows.append(row)
        r.table(['setting', 'rule', 'anchor-resolved cases', 'H4a', 'H4b', 'fast-equal-accuracy arm', 'time ratio',
                 'median anchor disc.', 'median ref. error', 'accurate-equal-cost arm', 'time ratio', 'median anchor disc.',
                 'median ref. error', 'BE median anchor disc.', 'BE median ref. error'], rows)
        h6 = an3['hypotheses'].get('H6', {})
        if h6:
            r('H6 (3D dial, descriptive, reference-limited): ' + '; '.join(
                f"{k}: median change {('%+.1f %%' % (100 * v['median_change'])) if 'median_change' in v else '—'}, "
                f"improved {fx(v.get('improved_fraction'))}, bars met {v.get('bars_met', '—')}" for k, v in h6.items()) + '.', '')
        plot_speedup3d(an3)
        r('![3D speed-up](plots/speedup3d.png)', '')
    r('## Error budget', '')
    if not b2 and b3 is None:
        r('Pending (`c2budA/B/C`, `c3bud`).', '')

    # ------------------------------------------------------------------------------------------------ retractions
    r('## Corrections and retractions', '',
      '- **Pre-registered 3D selection unavailable.** The second selection arm (fixed-sweep LSPG-CN, $K=2$) was ineligible on '
      'every rule in every setting, so the pre-registered selection could not be made; the 3D rules used downstream come from '
      'the post-hoc A10 selection (adaptive CN arm), labelled as such.',
      '- **The two-case smokes did not contain any failing case**, so the A6 choice of $K=2$ looked safe on the smoke and '
      'was not.',
      '- **C1\'s J4 failure was not only an under-resolved 48³ check**: at R′ = 1024 the successive-rule distance of one case '
      'stays above the bar up to 64³ vs 80³ (A11); the extension to 96³ is post hoc.',
      '- **Gauss 112³ at R′ = 1024 is infeasible** on a 141 GB H200 with the vendored table builder (smoke `smk3ext` failed '
      'out of memory, A11.3); the extension stops at 96³.',
      '- **B0 real-data mismatch control was non-discriminating** as an absolute test on the 2D smoke; it is now relative '
      '(A12).',
      '- **Offline scoring on the local GB10 GPU corrupted R′ = 1024 decodes** (caught by the G-score gate); all offline '
      'analysis runs on CPU (A8.1).', '')

    # ------------------------------------------------------------------------------------------------ provenance
    if ledger:
        r('## Jobs', '')
        r.table(['attempt', 'Slurm id', 'GPU', 'commit', 'role', 'status'],
                [[e['attempt'], e.get('slurm', '—'), e.get('gpu', '—'), e.get('commit', '—'), e.get('role', ''),
                  e.get('status', '')] for e in ledger['jobs']])
        r.table(['failed attempt', 'Slurm id', 'note'],
                [[e['attempt'], e.get('slurm', '—'), e.get('note', '')] for e in ledger.get('failed_attempts', [])])
    aud = []
    for p in sorted(glob.glob(str(HERE / 'audits' / 'codex-*.md')), key=lambda q: Path(q).stat().st_mtime):
        first = next((ln for ln in Path(p).read_text().splitlines() if ln.strip()), '')
        first = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', first).replace('**', '').replace('|', '/').strip()
        aud.append([f'`audits/{Path(p).name}`', first[:220] + ('…' if len(first) > 220 else '')])
    r('## Codex audits', '')
    r('Chronological; the opening statement of each audit (its verdict line). Every WRONG / HOLD was fixed and re-audited '
      'before the next step; the full findings are in the files.', '')
    r.table(['audit', 'opening statement'], aud)

    # ------------------------------------------------------------------------------------------------ glossary
    r('## Glossary', '',
      '- **bank, R′, dial**: the frozen ordered coordinate network; a solve uses its first R′ columns; the dial is the '
      'sequence of R′ settings (R128 → R256 → R384 → R512 in 2D; W256 … W1024 in 3D). `fast` and `acc` are the 2D settings '
      'deployed before this lane; `M2_512` the deployed 3D setting.',
      '- **rule, m, m\\***: an off-mesh quadrature for the advection term (Gauss pᵈ tensor, Fibonacci or rank-1 lattice) '
      'with m points; m\\* is the smallest rule of a family whose rollouts stay within τ = 2.5×10⁻⁴ of the converged rule\'s.',
      '- **converged rule, J4, bar**: a rule fine enough that the next finer rule changes the rollout by at most the bar '
      '2.5×10⁻⁵ (worst case over 64 cases, field distance relative to ‖u₀‖); J4 is that check.',
      '- **BE, CN, CN-R, BDF2**: backward Euler; Crank–Nicolson; CN with two BE start steps; second-order backward '
      'differentiation with one BE start step. **LSPG / GAL**: least-squares and Galerkin solves of each step.',
      '- **fixed-sweep solver, K**: after a few adaptive steps, exactly K Levenberg–Marquardt sweeps per time step with an '
      'extrapolated predictor (the deployed 3D BE solver is K = 1). **Adaptive (tight)**: iterate each step until its '
      'stopping test passes ("verified").',
      '- **solver distance**: field distance between a fixed-sweep run and the verified adaptive run of the same scheme and '
      'step; eligible if ≤ 2.5×10⁻⁴ on every case.',
      '- **ST / S / TX**: 2D references at 8192² with Δt₀/16 / Δt₀ / their Richardson combination, all first-order '
      'accurate in time; the 3D reference is BE at Δt₀/4 on 513³ nodes. **PROVISIONAL**: scored against these first-order '
      'references; not a physical-accuracy claim.',
      '- **tie, ε_ref, resolved**: a paired difference below 2ε_ref (ε_ref = per-case ST–TX distance) is a tie; a case is '
      'resolved when it is not a tie; a verdict with fewer than half the cases resolved is **reference-limited**.',
      '- **accurate-equal-cost / fast-equal-accuracy**: the lowest-median-error candidate arm costing no more than the '
      "setting's LSPG-BE at Δt₀; the cheapest candidate arm whose worst and median errors are no larger than that BE arm's.",
      '- **cost / own BE, common-baseline cost**: median paired A–B–A time ratio against the setting\'s own LSPG-BE at Δt₀, or '
      'against one common baseline (acc LSPG-BE Δt₀ in 2D).',
      '- **anchor, e∞**: GAL-BDF2 at Δt₀/8, tight tolerance (time-converged reduced solution) and its error.',
      '- **matched speed-up**: FOM time / ROM time for the cheapest calibrated FOM configuration whose worst error is no '
      'larger than the ROM arm\'s; same job, same GPU.',
      '- **pre-registered / POST HOC**: decided before results were read / decided after (labelled; exploratory).',
      '- **dev6, val32, 923801**: development and validation cohorts (selection allowed); test64 / 923901 are replication-only '
      'and were not used.',
      '- **pp, % of ‖u₀‖**: errors are reported relative to the initial-condition norm.', '')
    (HERE / 'report.md').write_text('\n'.join(r.lines) + '\n')
    print(HERE / 'report.md', 'pending:', pending)


if __name__ == '__main__':
    main()
