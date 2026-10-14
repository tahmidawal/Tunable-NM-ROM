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
            self.lines.append('| ' + ' | '.join(str(x).replace('|', ' / ') for x in r) + ' |')
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
    ax.set_ylabel(f"median error vs {'E0 R_ST' if an['refs'] == '2d_e0' else tag} (% of ‖u₀‖{'' if an['refs'] == '2d_e0' else ', PROVISIONAL'})")
    ax.set_title(f'2D error–cost: BE (open) vs second order (filled), by bank width — {tag}')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.3, which='both')
    fig.tight_layout()
    fig.savefig(PL / f'pareto2d_{tag}.png', dpi=130)
    plt.close(fig)


def plot_dial2d(an, an_sel):
    arms = an['arms']
    dial = ['R128', 'R256', 'R384', 'R512']
    rp = [128, 256, 384, 512]
    g = an_sel['selection']['global']['scheme']
    E0 = an['refs'] == '2d_e0'
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for tag, ls in ((('ST', '-'),) if E0 else (('TX', '-'), ('ST', '--'))):
        be = [arms[f'{s}|LSPG|BE|0.005|prod'][f'median_{tag}'] * 100 for s in dial]
        sel = [an_sel['selection']['per_scheme_setting'][f'{s}|{g}']['accurate_equal_cost'] for s in dial]
        so = [arms[f"{s}|{x['key']}"][f'median_{tag}'] * 100 for s, x in zip(dial, sel)]
        lab = 'E0 R_ST' if E0 else tag
        ax.plot(rp, be, 'o' + ls, color='#6b7280', label=f'LSPG-BE Δt₀ vs {lab}')
        ax.plot(rp, so, '^' + ls, color='#db2777', label=f'{g} accurate-equal-cost vs {lab}')
    ax.set_xlabel("bank width R′")
    ax.set_ylabel('median error (% of ‖u₀‖)' + ('' if E0 else ', PROVISIONAL'))
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
    ax.set_title('2D: ROM vs second-order-in-time FOM at matched worst error' + (' (E0 R_ST)' if an['refs'] == '2d_e0' else ' (ST, PROVISIONAL)'))
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(PL / 'fom2d.png', dpi=130)
    plt.close(fig)


def plot_budget(settings, name, dim, eP, edyn):
    labs, vals = [], []
    comps = [('delta_quad', 'δ_quad'), ('delta_solve', 'δ_solve'), ('delta_form', 'δ_form'), ('delta_time', 'δ_time'),
             (eP, 'e_P (floor)'), (edyn, 'e_dyn')]
    for sn, S in settings.items():
        for ak, A in S['arms'].items():
            labs.append(f'{sn}\n{ak}')
            vals.append([(A['summary'].get(k) or {}).get('median') or 0. for k, _ in comps])
    v = np.array(vals) * 100
    fig, ax = plt.subplots(figsize=(max(8, .55 * len(labs)), 4.8))
    bottom = np.zeros(len(labs))
    for i, (_, lab) in enumerate(comps):
        ax.bar(range(len(labs)), v[:, i], bottom=bottom, label=lab)
        bottom += v[:, i]
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=90, fontsize=6)
    ax.set_ylabel('median component (% of ‖u₀‖), stacked for display only')
    ax.set_title(f'{dim} error budget per setting and arm (components are separate ablations; e_P, e_dyn PROVISIONAL)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PL / name, dpi=130)
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
    an_old = load(CK / 'analysis-c2grid-2d_first_order.json')
    an = load(CK / 'analysis-c2grid-2d_e0.json') or an_old            # A13: E0 is the primary scoring when available
    sc2 = load(CK / 'scores-c2grid-2d_first_order.json')
    sel_pre = load(CK / 'SELECTION-3D.json')
    sel_ph = load(CK / 'SELECTION-3D-POSTHOC.json')
    an3o = load(CK / 'analysis3d-c3grid.json')
    an3 = load(CK / 'analysis3d-c3grid-3d_e0.json') or an3o          # A13: E0 primary for reference errors
    b2o = [load(p) for p in sorted(glob.glob(str(CK / 'budget2d-c2bud[ABC]-2d_first_order.json')))]
    b2e = [load(p) for p in sorted(glob.glob(str(CK / 'budget2d-c2bud[ABC]-2d_e0.json')))]
    b2 = b2e if len(b2e) == len(b2o) and b2e else b2o                 # A13: E0 primary when complete
    b3o = load(CK / 'budget3d-c3bud-3d_first_order.json')
    b3 = load(CK / 'budget3d-c3bud-3d_e0.json') or b3o
    ledger = load(HERE / 'runs' / 'JOB-LEDGER.json')
    done = dict(grid2d=an is not None, sel3d=sel_ph is not None, grid3d=an3 is not None, bud2d=bool(b2), bud3d=b3 is not None)
    pending = [k for k, v in done.items() if not v]

    r('# jcp-combined: second-order stepping with a wide bank and a fast fixed-sweep solver, with an error budget', '',
      '**Status:** ' + ('all stages complete. ' if not pending else f"interim; pending: {', '.join(pending)}. ") +
      'Every accuracy number is **PROVISIONAL** (scored against first-order references: 2D ST/S/TX, 3D the BE '
      'reference at Δt₀/4) and is marked so where it appears. The 3D rule selection is **POST HOC** (DESIGN A10, A11): '
      'the pre-registered selection was unavailable. Lane `exp/2026-10-13-jcp-combined`, generated by `make_report.py` '
      'from the committed analysis files.', '')

    SUMMARY_AT = len(r.lines)                        # the generated summary is inserted here at the end
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
        E0 = an['refs'] == '2d_e0'
        P = 'ST' if E0 else 'TX'                     # primary reference column (A13: under E0 TX is an alias of ST)
        PL_ = 'E0 R_ST' if E0 else 'TX (PROVISIONAL)'
        g = an['selection']['global']['scheme'] if not E0 else an_old['selection']['global']['scheme']
        r(f"**Primary scoring: {'the E0 second-order references' if E0 else 'the first-order references'}** — "
          f"{an['provisional']}." + (f" Non-rankable case-kinds (scored, never ranked): {', '.join(an.get('not_rankable', [])) or 'none'}." if E0 else ''), '')
        if sc2:
            gt = sc2['gate']
            r(f"**Scoring gate (first-order, certifies every rescoring).** `score.py` re-decoded every saved trajectory on CPU "
              f"and reproduced {gt['compared']} of {gt['expected']} job values (worst difference {sci(gt['worst'])}); every "
              f"negative control broke agreement (case swap {sci(sc2['controls']['case_swap'])}, time shift "
              f"{sci(sc2['controls']['time_shift'])}, ST↔S swap {sci(sc2['controls']['ST_S_swap'])}).", '')
        if E0:
            ev = list(an['eps_ref'].values())
            r(f"**Ties (E0).** Per case $\\eta=\\epsilon_{{\\rm ref}}(R_{{ST}})$ (median {pc(float(np.median(ev)))}, largest "
              f"{pc(max(ev))}); two arms are ranked on a case only if their errors differ by more than $2\\eta$; medians and "
              'worst values compare by interval overlap with endpoint propagation.', '')
        else:
            r(f"**Tie thresholds** (heuristic, from ST−TX): medians {pc(an['tie']['median'])}, worst values "
              f"{pc(an['tie']['worst'])}.", '')
        gl = an_old['selection']['global']
        r('### Which second-order scheme (selected against the first-order references, frozen)', '',
          f"Scheme wins over the four dial settings: " + ', '.join(f"{k} {v}" for k, v in gl['points'].items()) +
          f"; tie broken by the smaller sum of the four medians (" +
          ', '.join(f"{k} {pc(v)}" for k, v in gl['sums'].items()) + f", PROVISIONAL). **Selected scheme: {g}.**", '')
        rows = []
        fr = an_old['selection']['per_scheme_setting']
        for s in ['fast', 'R128', 'R256', 'acc', 'R384', 'R512']:
            be = an['arms'][f'{s}|LSPG|BE|0.005|prod']
            for kind, lab in (('accurate_equal_cost', 'accurate, equal cost'), ('fast_equal_accuracy', 'fast, equal accuracy')):
                x = fr[f'{s}|{g}'].get(kind)
                if not x:
                    rows.append([s, lab, 'none'] + ['—'] * 6)
                    continue
                a = an['arms'][f"{s}|{x['key']}"]
                ao = an_old['arms'][f"{s}|{x['key']}"]
                rows.append([s, lab, armname(x['key']), fx(x['cost']), pc(a[f'median_{P}']), pc(a[f'worst_{P}']),
                             'unmatched' if E0 else pc(a['median_S']), pc(ao['median_TX']), pc(ao['median_ST'])])
            bo = an_old['arms'][f'{s}|LSPG|BE|0.005|prod']
            rows.append([s, 'baseline', 'LSPG-BE at 1Δt₀', '1', pc(be[f'median_{P}']), pc(be[f'worst_{P}']), pc(be['median_S']),
                         pc(bo['median_TX']), pc(bo['median_ST'])])
        r(f"Per setting, the selected scheme's frozen arms against that setting's deployed LSPG-BE. Primary columns against "
          f"{PL_}; S is the space-only reference time-matched to LSPG-BE at Δt₀ only; the last two columns are the "
          'first-order scores (PROVISIONAL, continuity):', '')
        r.table(['setting', 'role', 'arm', 'cost / own BE', f'median {PL_}', f'worst {PL_}', 'median S (BE-matched)',
                 'old median TX', 'old median ST'], rows)

        H = an['hypotheses']
        r('### H1: does the wide bank pay off once time error is removed?', '')
        rows = []
        tags = (('ST', 'TX'),) if E0 else (('TX', 'TX'), ('ST', 'ST_sensitivity'))
        for step, v in H['H1'].items():
            for tag, k in tags:
                x = v[k]
                rows.append([step, 'E0 R_ST' if E0 else tag, v['verdict'], pc(x['median_from']), pc(x['median_to']),
                             f"{100 * x['median_change']:+.1f} %", f"{x['improved']}/{x['worsened']}/{x['ties']}",
                             pc(x['worst_from']), pc(x['worst_to'])])
            x = v['BE_dt0']['TX' if E0 else 'TX']
            rows.append([step, ('E0 R_ST' if E0 else 'TX') + ' (LSPG-BE Δt₀)', '', pc(x['median_from']), pc(x['median_to']),
                         f"{100 * x['median_change']:+.1f} %", f"{x['improved']}/{x['worsened']}/{x['ties']}",
                         pc(x['worst_from']), pc(x['worst_to'])])
        r.table(['dial step', 'reference', 'verdict', 'median from', 'median to', 'change', 'improved/worsened/tied',
                 'worst from', 'worst to'], rows)
        if E0 and an_old:
            r('First-order verdicts (PROVISIONAL, continuity): ' + '; '.join(
                f"{k} {v['verdict']}" for k, v in an_old['hypotheses']['H1'].items()) + '.', '')
        r('A step is **reference-limited** when fewer than half of the 38 cases are resolved (paired difference above the '
          'tie threshold). Verdicts: ' + '; '.join(f"{k} **{v['verdict']}**" for k, v in H['H1'].items()) + '.', '')
        r('![2D dial](plots/dial2d.png)', '')

        h2 = H['H2']
        r('### H2: does tunability survive?', '',
          f"(a) Every eligible LSPG-BE arm is covered by a second-order arm with lower or equal cost and error: worst "
          f"{h2['a']['worst_TX']['covered']}/{h2['a']['worst_TX']['total']}, median {h2['a']['median_TX']['covered']}/"
          f"{h2['a']['median_TX']['total']} → **{'pass' if h2['a_pass'] else 'fail'}**. (b) Along the dial, the "
          f"accurate-equal-cost arms' common-baseline cost and median error ({PL_}): " +
          '; '.join(f"{d['setting']} {fx(d['cost_common'])}× / {pc(d['median_TX'])}" for d in h2['dial']) +
          f" → **{'pass' if h2['b'] else 'fail'}**.", '',
          '![2D Pareto](plots/pareto2d_ST.png)', '')

        h3 = H['H3']
        r('### H3: why did the fast setting\'s worst case get worse under CN?', '')
        rows = []
        for tag in (('ST',) if E0 else ('ST', 'TX')):
            p_ = h3[tag]['primary']
            rows.append(['E0 R_ST' if E0 else tag, h3[tag]['worst_case'], p_['t_star'], pc(p_['e_BE']), pc(p_['e_CN']),
                         pc(p_['e_inf']), fx(p_['cos']), ' '.join('yes' if x else 'no' for x in p_['guards']), p_['verdict']])
        r.table(['reference', 'worst case', 'output index t*', 'e(BE Δt₀)', 'e(CN Δt₀)', 'e(anchor) = e∞', 'cos(δ_BE, ε∞)',
                 'guards a b c', 'verdict'], rows)
        wc = h3['ST']['worst_case']
        comp = []
        for s in ('fast', 'R128', 'R256'):
            x = fr[f'{s}|{g}']['accurate_equal_cost']
            comp.append(f"{s} {g} accurate arm worst {pc(an['arms'][f'{s}|' + x['key']]['worst_ST'])}")
        p_ = h3['ST']['primary']
        failed = [n for n, ok_ in zip('abc', p_['guards']) if not ok_]
        r(f"Pre-registered explanation (error cancellation) at the worst case {wc}: **{p_['verdict']}**" +
          (f"; failing guards: {', '.join(failed)}" if failed else '') +
          f". Criterion (a) needs |e(CN) − e∞| ≤ 0.2 |e(BE) − e∞|; here |e(CN) − e∞| = {pc(abs(p_['e_CN'] - p_['e_inf']))} "
          f"and |e(BE) − e∞| = {pc(abs(p_['e_BE'] - p_['e_inf']))}. The fast setting's LSPG-BE worst error is "
          f"{pc(an['arms']['fast|LSPG|BE|0.005|prod']['worst_ST'])}; " + '; '.join(comp) + f" (against {PL_}). "
          + h3['ST'].get('note', ''), '')

        h5 = H['H5']
        cnr = {k: v for k, v in h5['ST']['matches'].items() if f'|{g.replace("-", "|")}|' in f'|{k}'}
        bc = max(cnr.items(), key=lambda kv: kv[1]['speedup']) if cnr else None
        r('### H5: speed against the second-order full-order model (same job, same GPU)', '',
          f"Matched on worst error against {PL_}. Best matched speed-up of any eligible ROM arm: "
          f"**{fx(h5['ST']['best_combined'][0])}×** ({h5['ST']['best_combined'][1]}); the C2 `acc` arms re-measured in "
          f"this job: {fx(h5['ST']['best_comparator'][0])}× ({h5['ST']['best_comparator'][1]}). Verdict: "
          f"**{h5['ST']['verdict']}**. Scope: the overall winner may be an LSPG-CN arm (not second order, excluded from "
          f"scheme selection); the best matched speed-up of the selected scheme {g} is "
          + (f"{fx(bc[1]['speedup'])}× ({bc[0]})." if bc else 'not available.') +
          ' The FOM is second order in time only; its advection stays sign-upwind (first order in space).', '',
          '![FOM comparison](plots/fom2d.png)', '')
        if E0 and an_old:
            ho = an_old['hypotheses']['H5']['ST']
            r(f"First-order matching (PROVISIONAL, continuity): best {fx(ho['best_combined'][0])}× ({ho['best_combined'][1]}), "
              f"comparator {fx(ho['best_comparator'][0])}×, verdict {ho['verdict']}.", '')
        plot_pareto2d(an, 'ST')
        plot_dial2d(an, an_old)
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
          'Δt₀/8 (time-converged, reference-free; H4 and the selections use only it). Reference errors: '
          f"{an3['provisional']}.", '')
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
            dg = (an3o or an3)['hypotheses']['H4'][s].get('cheapest_equal_accuracy_diagnostic')
            row += [sci(comp['anchor_median_all']), pc(comp['median_ref']),
                    f"{dg['key']} ({fx(dg['ratio'])})" if dg else '—']
            rows.append(row)
        r.table(['setting', 'rule', 'anchor-resolved cases', 'H4a', 'H4b', 'fast-equal-accuracy arm', 'time ratio',
                 'median anchor disc.', 'median ref. error', 'accurate-equal-cost arm', 'time ratio', 'median anchor disc.',
                 'median ref. error', 'BE median anchor disc.', 'BE median ref. error',
                 'cheapest equal-accuracy arm (diagnostic, ratio)'], rows)
        r('The registered fast-equal-accuracy arm is H4a\'s cheapest (Δt ≥ 1.25 Δt₀, anchor discrepancy no worse than '
          'the comparator, paired ratio ≤ 0.6); where H4a fails it is "none". The last column is a diagnostic only '
          '(A15: an earlier frozen selection used it by mistake and was retracted before any budget job).', '')
        h6 = an3['hypotheses'].get('H6', {})
        if h6 and an3['refs'] == '3d_e0':
            sel_ = an3['selection']
            wst = {s_: an3['settings'][s_]['arms'][sel_[s_]['accurate_equal_cost']] for s_ in sel_ if sel_[s_].get('accurate_equal_cost')}
            worse = [k for k, v in h6.items() if all(x in wst for x in k.split('->')) and
                     wst[k.split('->')[1]]['worst_ref'] > wst[k.split('->')[0]]['worst_ref']]
            r('H6 compares the **selected accurate-equal-cost configurations** along the dial (scheme, step and K change '
              'with R′ and the rule), so it is a statement about the deployed dial, not an isolated bank-width effect. '
              'Median / worst error of the selected arm per rung (E0): ' + '; '.join(
                  f"{k} {pc(v['median_ref'])} / {pc(v['worst_ref'])}" for k, v in wst.items()) + '.'
              + (f" The worst error rises at {', '.join(worse)} although the median bars are met." if worse else ''), '')
            r('H6 (3D dial, against E0 with per-case ties at 2 ε_ref): ' + '; '.join(
                f"{k}: **{v['verdict']}** (median change {('%+.1f %%' % (100 * v['median_change']))}, resolved {v.get('resolved')}, "
                f"improved/worsened {v.get('improved_resolved')}/{v.get('worsened_resolved')}, ties {v.get('ties')})"
                for k, v in h6.items()) + '.', '')
            h6 = (an3o or {}).get('hypotheses', {}).get('H6', {})
            if h6:
                r('First-order H6 (PROVISIONAL, descriptive): ' + '; '.join(
                    f"{k}: median change {('%+.1f %%' % (100 * v['median_change'])) if 'median_change' in v else '—'}, "
                    f"bars met {v.get('bars_met', '—')}" for k, v in h6.items()) + '.', '')
        elif h6:
            r('H6 (3D dial, descriptive, reference-limited): ' + '; '.join(
                f"{k}: median change {('%+.1f %%' % (100 * v['median_change'])) if 'median_change' in v else '—'}, "
                f"improved {fx(v.get('improved_fraction'))}, bars met {v.get('bars_met', '—')}" for k, v in h6.items()) + '.', '')
        plot_speedup3d(an3)
        r('![3D speed-up](plots/speedup3d.png)', '')
    r('## Error budget (each component its own ablation; none is a remainder)', '',
      '$e_{\\rm tot}=\\lVert u_a-u_{\\rm ref}\\rVert$; floor $e_P=\\lVert(I-\\Pi_{R\'})u_{\\rm ref}\\rVert$; reduced dynamics '
      '$e_{\\rm dyn}=\\lVert u_\\star-\\Pi_{R\'}u_{\\rm ref}\\rVert$ (ideal model $\\star$ = converged rule, GAL, BDF2, $\\Delta t_0/8$, '
      'tight); $\\delta_{\\rm quad},\\delta_{\\rm solve},\\delta_{\\rm form},\\delta_{\\rm time}$ = distance of the arm to the same arm '
      'with only the rule / solver tolerance / test space / (scheme, step) changed to the ideal; $D_a=\\lVert u_a-u_\\star\\rVert$ '
      'measured directly. All are maxima over the five evolved output times relative to $\\lVert u_0\\rVert$, medians over the '
      'budget population (2D: 38 cases; 3D: first 32). $e_{\\rm tot}$, $e_P$, $e_{\\rm dyn}$ are PROVISIONAL (first-order '
      'references); the $\\delta$\'s and $D_a$ are reference-free.', '')
    comps = ['delta_quad', 'delta_solve', 'delta_form', 'delta_time', 'D_a']

    def budget_rows(settings, eP, edyn, etot):
        rows = []
        for sn, S in settings.items():
            for ak, A in S['arms'].items():
                sm_ = A['summary']
                med = lambda k: sm_.get(k, {}).get('median')
                lg = A.get('largest') or {}
                lg = lg.get('ST', lg) if isinstance(lg, dict) else lg
                rows.append([sn, ak, A['arm'].get('role', '')] + [sci(med(k)) for k in comps] +
                            [pc(med(etot)), pc(med(eP)), pc(med(edyn)),
                             ', '.join(f'{k} {v}' for k, v in (lg.items() if isinstance(lg, dict) else [])) or '—',
                             'yes' if (A.get('finalist') or {}).get('qualified') else ('no' if A.get('finalist') else '—')])
        return rows
    head = ['setting', 'arm', 'role', 'δ_quad', 'δ_solve', 'δ_form', 'δ_time', 'D_a', 'e_tot', 'e_P', 'e_dyn',
            'largest component (cases)', 'finalist qualified']
    if b2:
        e0b = b2[0]['refs'] == '2d_e0'
        r(f"### 2D budget (median over 38 cases; e_tot, e_P, e_dyn against {'the E0 R_ST reference' if e0b else 'ST (PROVISIONAL)'})", '',
          f"Label: {b2[0]['provisional']}." + (f" Not rankable (scored, never ranked): {', '.join(b2[0].get('not_rankable', []))}." if e0b else ''), '')
        st2 = {}
        for b in b2:
            st2.update(b['settings'])
        r.table(head, budget_rows(st2, 'e_P_ST', 'e_dyn_ST', 'e_tot_ST'))
        gl = [f"{Path(b['run']).parents[1].name}: " + ', '.join(f"{k} {'pass' if (v.get('passed') if isinstance(v, dict) else v) is True else (v.get('passed') if isinstance(v, dict) else v)}"
                                                               for k, v in b['gates'].items()) for b in b2]
        r('Budget gates: ' + '; '.join(gl) + '.', '')
        wh = {sn: S.get('withheld') for sn, S in st2.items() if S.get('withheld')}
        if wh:
            r('Withheld attributions (non-discriminating control): ' + '; '.join(f'{k}: {v}' for k, v in wh.items()) + '.', '')
        plot_budget(st2, 'budget2d.png', '2D', 'e_P_ST', 'e_dyn_ST')
        r('![2D budget](plots/budget2d.png)', '')
        # H3 follow-up: the budget of the worst case named by H3, in the fast setting (A1.9: the budget names what dominates)
        if an is not None and 'fast' in st2:
            wc = an['hypotheses']['H3']['ST']['worst_case']
            tg = 'ST'
            rows = []
            for ak, A in st2['fast']['arms'].items():
                pcs = [x for x in A['per_case'] if x['case'] == wc]
                if not pcs:
                    continue
                x = pcs[0]
                mx = lambda k: (max(v for v in x[k][1:] if v is not None) if isinstance(x.get(k), list) and any(v is not None for v in x[k][1:]) else None)
                rows.append([armname(ak.replace('|1', '|0.005').replace('|2', '|0.01')) if ak.count('|') == 2 else ak,
                             pc(mx(f'e_tot_{tg}')), pc(mx(f'e_star_{tg}')), pc(mx(f'e_P_{tg}')), pc(mx(f'e_dyn_{tg}')),
                             pc(mx('delta_time')), pc(mx('delta_form')), pc(mx('delta_quad'))])
            if rows:
                r(f"**The H3 worst case ({wc}, `fast` setting) through the budget** (max over output times; e_tot, e_⋆, e_P, e_dyn "
                  f"against {'E0 R_ST' if e0b else 'ST (PROVISIONAL)'}): e_⋆ is the error of the ideal model (converged rule, GAL, "
                  'BDF2 at Δt₀/8, tight), e_P the bank\'s projection floor, e_dyn the ideal model\'s distance to the projected '
                  'reference.', '')
                r.table(['arm', 'e_tot', 'e_⋆ (ideal)', 'e_P (floor)', 'e_dyn', 'δ_time', 'δ_form', 'δ_quad'], rows)
    else:
        r('2D budget pending (`c2budA/B/C`).', '')
    fb2 = load(CK / 'fallback2d.json')
    if fb2:
        r('### 2D finalist failures and their fallbacks (A14, pre-registered conditional step)', '',
          'The selected arms whose deployed rule fails the finalist quadrature check (δ_quad ≤ τ on every case) keep their '
          'selection verdict, labelled "rule inadequate for this arm"; their first fallback rule is qualified separately '
          '(accuracy, solver, quadrature, timing, and the arm\'s original criterion against the original deployed BE). '
          'Accuracy in this table: first-order (the selection reference, PROVISIONAL) and E0 R_ST beside it.', '')
        rows = []
        for s_, e in fb2['settings'].items():
            for k, a in e['arms'].items():
                c_ = a['criterion']
                rows.append([s_, k.replace('_', ' '), armname(a['key']), f"{e['fallback_setting']} ({e['fallback_rule']})",
                             'yes' if a['finalist'].get('qualified') else 'no', sci(a['finalist'].get('delta_quad_worst')),
                             pc(a['first_order']['TX']['median']), pc(a['e0_ST']['median']),
                             fx(c_.get('value', c_.get('cost_ratio'))), 'yes' if c_['ok'] else 'no',
                             '**yes**' if a['qualified'] else '**no**'])
        r.table(['setting', 'arm', 'step', 'fallback (rule)', 'finalist pass', 'worst δ_quad', 'median TX (old)',
                 'median E0 R_ST', 'cost vs original BE', 'original criterion met', 'fallback qualified'], rows)
        r('Cost method: ' + '; '.join(sorted({a['timing_method'] for e in fb2['settings'].values() for a in e['arms'].values()})) + '.', '')
    if b3:
        r('### 3D budget (median over the first 32 cases)', '')
        r.table(head, budget_rows(b3['settings'], 'e_P', 'e_dyn', 'e_tot'))
        r('Budget gates: ' + ', '.join(f"{k} {(v.get('passed') if isinstance(v, dict) else v)}" for k, v in b3['gates'].items()) + '.', '')
        plot_budget(b3['settings'], 'budget3d.png', '3D', 'e_P', 'e_dyn')
        r('![3D budget](plots/budget3d.png)', '')
    else:
        r('3D budget pending (`c3bud`).', '')

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
    # ------------------------------------------------------------------------------------------------ summary
    S = []
    if an is not None:
        H = an['hypotheses']
        ref = 'E0 R_ST' if an['refs'] == '2d_e0' else 'first-order TX (PROVISIONAL)'
        g = an_old['selection']['global']['scheme']
        S.append(f"- **2D scheme.** {g} (selected against the first-order references, frozen). Against {ref}: "
                 + '; '.join(f"H1 {k} **{v['verdict']}**" for k, v in H['H1'].items())
                 + f"; H2 (tunability) **{'pass' if H['H2']['a_pass'] and H['H2']['b'] is True else 'fail'}**; H3 at "
                 f"{H['H3']['ST']['worst_case']}: **{H['H3']['ST']['primary']['verdict']}**; H5 best matched speed-up "
                 f"{fx(H['H5']['ST']['best_combined'][0])}× vs re-measured C2 `acc` {fx(H['H5']['ST']['best_comparator'][0])}× "
                 f"(**{H['H5']['ST']['verdict']}**).")
    if fb2:
        fails = [s_ for s_ in fb2['settings']]
        qual = [f"{s_} ({k.replace('_', ' ')} arm → {e['fallback_rule']})" for s_, e in fb2['settings'].items()
                for k, a in e['arms'].items() if a['qualified']]
        notq = [f"{s_} ({k.replace('_', ' ')} arm: cost {fx(a['criterion'].get('value'))}× the deployed BE)" if
                k == 'accurate_equal_cost' else f"{s_} ({k.replace('_', ' ')} arm)"
                for s_, e in fb2['settings'].items() for k, a in e['arms'].items() if not a['qualified']]
        S.append(f"- **2D finalist quadrature check.** The selected CN-R arms fail it (δ_quad > τ) at {', '.join(fails)}; "
                 f"first fallbacks qualified: {', '.join(qual) or 'none'}; not qualified: {', '.join(notq) or 'none'}.")
    if esc_rows:
        w = esc_rows.get('W1024', {})
        over = {s_: max((x['n_over'] for x in v.values()), default=0) for s_, v in esc_rows.items()}
        S.append('- **3D converged-rule gate (J4).** Cases above the bar at any successive pair (fixed-sweep BE): '
                 + ', '.join(f'{k} {v}' for k, v in over.items()) + '; worst case at every pair: case '
                 + ', '.join(sorted({str(x['argmax']) for v in esc_rows.values() for x in v.values()}))
                 + (('. POST HOC converged rules: ' + ', '.join(f"{k} {v.get('posthoc_converged')}" for k, v in sel_ph['settings'].items()))
                    if sel_ph else '') + '.')
    if cnf:
        S.append('- **3D fixed-sweep CN (K = 2, 2.5 Δt₀)** departs from the verified adaptive solve on a fixed set of cases '
                 '(the same at every rule), so the pre-registered rule selection was unavailable; the POST HOC selection uses '
                 'the adaptive CN arm.')
    if an3 is not None:
        h4 = an3['hypotheses']['H4']
        S.append('- **3D fast solver (H4).** H4a (speed at equal accuracy, ratio ≤ 0.6): '
                 + ', '.join(f"{k} {v.get('H4a', {}).get('verdict', v.get('verdict'))}" for k, v in h4.items())
                 + '; H4b (accuracy at equal cost): '
                 + ', '.join(f"{k} {v.get('H4b', {}).get('verdict', '—')}" for k, v in h4.items())
                 + ('. Read together: the second-order fixed-sweep solver buys accuracy at equal cost, not speed at equal '
                    'accuracy.' if all(v.get('H4a', {}).get('verdict') == 'fail' and v.get('H4b', {}).get('verdict') == 'pass'
                                       for v in h4.values()) else '.'))
    if b2:
        cnt = {}
        for b in b2:
            for s_, S_ in b['settings'].items():
                for ak, A in S_['arms'].items():
                    lg = (A.get('largest') or {}).get('ST', {})
                    if lg:
                        top = max(lg.items(), key=lambda kv: kv[1])[0]
                        cnt[top] = cnt.get(top, 0) + 1
        S.append('- **2D budget: what limits next** (largest component per setting-arm, by number of setting-arms): '
                 + ', '.join(f'{k} {v}' for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])) + '.')
    r.lines[SUMMARY_AT:SUMMARY_AT] = ['## Answers in brief (generated)', ''] + S + ['']
    (HERE / 'report.md').write_text('\n'.join(r.lines) + '\n')
    print(HERE / 'report.md', 'pending:', pending)


if __name__ == '__main__':
    main()
