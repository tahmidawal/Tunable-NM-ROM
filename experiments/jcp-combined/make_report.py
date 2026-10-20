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


def ref_label(refs):
    """Accuracy label per reference registry entry (A13.4), dimension-specific."""
    return {'2d_e0': 'E0 R_ST (order-verified second-order reference; rankings conditional on ε_ref, a Richardson estimate, '
                     'not a rigorous bound)',
            '3d_e0': 'E0 R_ST (order-verified second-order reference; rankings conditional on ε_ref, a Richardson estimate, '
                     'not a rigorous bound)',
            '2d_first_order': 'first-order references (PROVISIONAL)',
            '3d_first_order': 'first-order BE reference (PROVISIONAL, reference-limited)'}.get(refs, refs)


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
    ax.set_title(f"2D error–cost: LSPG-BE (open) vs CN / CN-R / BDF2 / GAL-CN arms (filled), by bank width — {'E0 R_ST' if an['refs'] == '2d_e0' else tag}")
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
    ax.set_title('2D dial: LSPG-BE vs the selected LSPG-CN-R arm')
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
    ax.set_title('2D: ROM vs the CN/BDF2-family-in-time FOM at matched worst error' + (' (E0 R_ST)' if an['refs'] == '2d_e0' else ' (ST, PROVISIONAL)'))
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(PL / 'fom2d.png', dpi=130)
    plt.close(fig)


def finer_invalid(setting):
    """Number of unverified runs at the finer rule of a 3D budget setting (from the budget job's rows)."""
    for a_ in ('c3budA', 'c3budW1024r'):
        res = load(HERE / 'runs' / a_ / 'archive' / 'output' / 'result.json')
        if not res:
            continue
        cfg_s = next((x for x in res['config']['settings'] if x['name'] == setting), None)
        if cfg_s:
            return sum(1 for r_ in res['rows'] if r_['setting'] == setting and r_['rule'] == cfg_s['finer']
                       and r_['valid'] is not True)
    return None


def tstats(panel, subj):
    """Per-subject timing statistics from an A-B-A panel: median ratio, IQR, MAD-outlier count, median absolute times."""
    inv = [i for i in panel['invocations'] if i['B'] == subj]
    if not inv:
        return None
    x = np.array([i['ratio'] for i in inv])
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    out = int(np.sum(np.abs(x - med) > 3 * 1.4826 * mad)) if mad > 0 else 0
    return dict(n=len(x), median=med, q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)), outliers=out,
                tB=float(np.median([i['tB'] for i in inv])), tA=float(np.median([.5 * (i['tA1'] + i['tA2']) for i in inv])))


def plot_budget(settings, name, dim, eP, edyn, refs):
    """Grouped (not stacked: the components are separate, non-additive measurements, A1.4) median components per arm."""
    labs, vals = [], []
    comps = [('delta_quad', 'δ_quad'), ('delta_solve', 'δ_solve'), ('delta_form', 'δ_form'), ('delta_time', 'δ_time'),
             (eP, 'e_P (floor)'), (edyn, 'e_dyn')]
    for sn, S in settings.items():
        for ak, A in S['arms'].items():
            labs.append(f'{sn}\n{ak}')
            vals.append([(A['summary'].get(k) or {}).get('median') or np.nan for k, _ in comps])
    v = np.array(vals) * 100
    fig, ax = plt.subplots(figsize=(max(8, .7 * len(labs)), 4.8))
    w = .8 / len(comps)
    for i, (_, lab) in enumerate(comps):
        ax.bar(np.arange(len(labs)) + (i - len(comps) / 2 + .5) * w, v[:, i], w, label=lab)
    ax.set_yscale('log')
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=90, fontsize=6)
    ax.set_ylabel('median over the budget population (% of ‖u₀‖)')
    ax.set_title(f'{dim} error budget: separate one-ingredient measurements (grouped); e_P, e_dyn vs {ref_label(refs).split(" (")[0]}',
                 fontsize=9)
    ax.legend(fontsize=7, ncol=3)
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
    ax.set_title('3D fixed-sweep CN / CN-R / BDF2 vs deployed BE: speed vs time accuracy (open = not H4-eligible)')
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
    def merge3(tag):
        fs_ = [load(CK / f'budget3d-{a_}-{tag}.json') for a_ in ('c3budA', 'c3budW1024r')]
        fs_ = [f_ for f_ in fs_ if f_]
        if not fs_:
            return None
        m_ = dict(fs_[0], settings={}, gates={}, parts=[f_['run'] for f_ in fs_], complete=len(fs_) == 2)
        for f_ in fs_:
            m_['settings'].update(f_['settings'])
            for k_, v_ in f_['gates'].items():
                m_['gates'][f"{Path(f_['run']).parents[1].name}:{k_}"] = v_
        return m_
    b3o = merge3('3d_first_order')
    b3e = merge3('3d_e0')
    b3 = b3e if (b3e and b3o and b3e['complete'] == b3o['complete']) else b3o
    ledger = load(HERE / 'runs' / 'JOB-LEDGER.json')
    done = dict(grid2d=an is not None, sel3d=sel_ph is not None, grid3d=an3 is not None, bud2d=bool(b2),
                bud3d=bool(b3 and b3.get('complete')))
    pending = [k for k, v in done.items() if not v]

    r('# jcp-combined: CN/BDF2-family reduced stepping with a wide bank, a fixed-sweep 3D solver and an error budget', '',
      '**Status:** ' + ('all stages complete. ' if not pending else f"interim; pending: {', '.join(pending)}. ") +
      'Primary accuracy numbers are scored against the campaign\'s **E0 second-order references** (DESIGN A13): '
      'measured against an order-verified reference, with rankings conditional on ε_ref (a Richardson estimate, not a '
      'rigorous bound). The original first-order scores are kept beside them for continuity and are **PROVISIONAL**. '
      'All selections were made against the first-order references and frozen. The 3D rule selection is **POST HOC** '
      '(DESIGN A10, A11): '
      'the pre-registered selection was unavailable. Lane `exp/2026-10-13-jcp-combined`, generated by `make_report.py` '
      'from the committed analysis files. Every measured number is generated from those files; numbers in the prose that '
      'are not generated are pre-registered design constants (thresholds, bars, step factors) defined in DESIGN.md.', '')

    SUMMARY_AT = len(r.lines)                        # the generated summary is inserted here at the end
    # ------------------------------------------------------------------------------------------------ pipeline
    r('## How the work was done', '',
      '```mermaid', 'flowchart TB',
      '  D[DESIGN + amendments A1-A16, Codex-audited]:::doc --> SM[cluster smokes: smk2d, smk3esc, smk3grid, smk2bud, smk3bud, smk3ext2]:::job',
      '  SM --> G2[c2grid: 2D grid, anchors, FOM, timing]:::job',
      '  SM --> E3[c3descW1024, c3descW768, c3descB: 3D escalation + rule ladder]:::job',
      '  E3 --> X3[c3extW1024: W1024 escalation extension, POST HOC A11]:::posthoc',
      '  G2 --> S2[score + analyze2d + freeze 2D]:::doc',
      '  E3 --> S3[select3d: pre-registered = unavailable; POST HOC A10 selection]:::posthoc',
      '  X3 --> S3',
      '  S3 --> C3[c3grid: 3D CN/BDF2-family fixed-sweep grid + timing]:::job',
      '  S2 --> B2[c2budA/B/C: 2D error budget]:::job',
      '  B2 --> FB[A14: finalist failures -> c2fbgrid + c2fbbud fallback qualification]:::posthoc',
      '  C3 --> F3[analyze3d + freeze 3D solver; A15 retraction and re-freeze]:::doc',
      '  F3 --> B3[c3budA + c3budW1024 (OOM, A16) -> c3budW1024r: 3D error budget]:::job',
      '  E0[(E0 second-order references, A13)]:::ref --> RS[rescoring of every grid and budget, certified by the first-order G-score]:::doc',
      '  B2 --> RS',
      '  B3 --> RS',
      '  FB --> REP[make_report.py]:::doc',
      '  RS --> REP',
      '  classDef doc fill:#e5e7eb,stroke:#374151;',
      '  classDef job fill:#dbeafe,stroke:#1d4ed8;',
      '  classDef posthoc fill:#fde68a,stroke:#b45309;',
      '  classDef ref fill:#dcfce7,stroke:#15803d;',
      '```', '')

    # ------------------------------------------------------------------------------------------------ 2D
    if an is None:
        r('## 2D', '', 'Pending.', '')
    else:
        n2 = an['arms']['fast|LSPG|BE|0.005|prod']['cases']
        r(f"## 2D (Burgers 2D, development + validation cohorts dev6 and val32, {n2} cases)", '')
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
        fz_sha = (an.get('selection') or {}).get('frozen_sha256')
        r('### Which CN/BDF2-family scheme (selected against the first-order references, frozen)', '',
          (f"The frozen selection used by this analysis is `checks/frozen-versions/FROZEN-SELECTION-{fz_sha[:8]}.json` "
           f"(sha256 {fz_sha[:12]}…; its 2D section is unchanged in every later version)." if fz_sha else ''), '',
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
        res2 = load(HERE / 'runs' / 'c2grid' / 'archive' / 'output' / 'result.json')
        trow = []
        for s_ in ['fast', 'R128', 'R256', 'acc', 'R384', 'R512']:
            for kind in ('accurate_equal_cost', 'fast_equal_accuracy'):
                x = fr[f'{s_}|{g}'].get(kind)
                if not x:
                    continue
                t_ = tstats(res2['timing'].get(f'setting|{s_}', {}), f"{s_}|{x['key']}")
                if t_:
                    trow.append([s_, kind.replace('_', ' '), armname(x['key']), t_['n'], fx(t_['median']),
                                 f"{fx(t_['q25'])}–{fx(t_['q75'])}", t_['outliers'], f"{t_['tB']:.3g} s", f"{t_['tA']:.3g} s"])
        r(f"**2D timing detail** (A–B–A against the setting's LSPG-BE at Δt₀, one {res2.get('gpu')}, warm JIT):", '')
        r.table(['setting', 'role', 'arm', 'invocations', 'median ratio', 'IQR', 'MAD outliers', 'median time B',
                 'median time A (BE)'], trow)

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
          f"(a) Every eligible LSPG-BE arm is covered by a CN/CN-R/BDF2/GAL-CN arm with lower or equal cost and error: worst "
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
          f". Criterion (a) needs $|e_{{\\rm CN}}-e_\\infty|\\le0.2\\,|e_{{\\rm BE}}-e_\\infty|$; here $|e_{{\\rm CN}}-e_\\infty|$ = {pc(abs(p_['e_CN'] - p_['e_inf']))} "
          f"and $|e_{{\\rm BE}}-e_\\infty|$ = {pc(abs(p_['e_BE'] - p_['e_inf']))}. The fast setting's LSPG-BE worst error is "
          f"{pc(an['arms']['fast|LSPG|BE|0.005|prod']['worst_ST'])}; " + '; '.join(comp) + f" (against {PL_}). "
          + h3['ST'].get('note', ''), '')

        h5 = H['H5']
        cnr = {k: v for k, v in h5['ST']['matches'].items() if f'|{g.replace("-", "|")}|' in f'|{k}'}
        bc = max(cnr.items(), key=lambda kv: kv[1]['speedup']) if cnr else None
        r('### H5: speed against the full-order model with CN/BDF2-family time stepping (same job, same GPU)', '',
          f"Matched on worst error against {PL_}. Best matched speed-up of any eligible ROM arm: "
          f"**{fx(h5['ST']['best_combined'][0])}×** ({h5['ST']['best_combined'][1]}); the C2 `acc` arms re-measured in "
          f"this job: {fx(h5['ST']['best_comparator'][0])}× ({h5['ST']['best_comparator'][1]}). Verdict: "
          f"**{h5['ST']['verdict']}**. The overall winner is {armname(h5['ST']['best_combined'][1])} in the "
          f"`{h5['ST']['best_combined'][1].split('|')[0]}` setting"
          + (' — an LSPG-CN arm, which is not in the scheme selection (DESIGN §6)' if '|LSPG|CN|' in h5['ST']['best_combined'][1] else '')
          + f"; the best matched speed-up of the selected scheme {g} is "
          + (f"{fx(bc[1]['speedup'])}× ({bc[0]})." if bc else 'not available.') +
          ' Scope: the FOM uses the same mesh as the ROM, CN/BDF2-family time stepping with sign-upwind advection, and '
          'tolerances calibrated on dev6; speed-ups are matched on the development + validation cohorts only. Under E0 the '
          'matching is recomputed against the new reference (reference-dependent rematching of the same timings, no new '
          'timing measurement).', '',
          '![FOM comparison](plots/fom2d.png)', '')
        if E0 and an_old:
            ho = an_old['hypotheses']['H5']['ST']
            r(f"First-order matching (PROVISIONAL, continuity): best {fx(ho['best_combined'][0])}× ({ho['best_combined'][1]}), "
              f"comparator {fx(ho['best_comparator'][0])}×, verdict {ho['verdict']}.", '')
        plot_pareto2d(an, 'ST')
        plot_dial2d(an, an_old)
        plot_fom(an)

    # ------------------------------------------------------------------------------------------------ 3D escalation
    n3 = next(iter((an3 or {}).get('hypotheses', {}).get('H4', {}).values()), {}).get('cases', '—') if an3 else '—'
    r(f"## 3D (Burgers 3D, validation cohort 923801, {n3} cases)", '')
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
    extra = []
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
        sets_all = [set(x['distance_failing_cases']) for d in cnf.values() for v in d['settings'].values()
                    for k, x in v['rules'].items() if k != 'lat256']
        common = sorted(set.intersection(*sets_all)) if sets_all else []
        extra = sorted({(sn, k, c) for d in cnf.values() for sn, v in d['settings'].items() for k, x in v['rules'].items()
                        if k != 'lat256' for c in x['distance_failing_cases']
                        if c not in set.intersection(*[set(y['distance_failing_cases']) for kk, y in v['rules'].items() if kk != 'lat256'])})
        r(f"Cases failing in every setting at every non-control rule: {', '.join(map(str, common)) or 'none'} (M2_512 fails on further, setting-specific cases, listed in the table, at every one of its rules). "
          + (('Rule-dependent exceptions: ' + '; '.join(f'{sn} case {c} at {k}' for sn, k, c in extra) + '. ') if extra else '')
          + 'The control `lat256` (a deliberately inadequate rule) fails on every case and is excluded. On the failing '
          'cases the two LM sweeps per step leave most steps non-stationary (gradient norm far above the 10⁻³ stationarity '
          'test) and the distance to the verified adaptive solve grows over the rollout. Because essentially the same '
          'cases fail at every adequate rule, this is a persistent solver failure rather than a quadrature effect '
          '(not a proof of rule independence). Per-case evidence: `checks/cnfail3d-*.json`.', '')

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

    r('### 3D CN / CN-R / BDF2 fixed-sweep grid (H4, H6) and speed', '')
    if an3 is None:
        r('Pending (`c3grid`).', '')
    else:
        r(f"Rule selection label carried by the grid: **{an3.get('rule_selection_label')}**. Comparator: the deployed "
          'fixed-sweep BE at Δt₀ (K = 1) on the same bank, R′ and rule. Anchor discrepancy = distance to GAL-BDF2 at '
          'Δt₀/8 (time-converged, reference-free; H4 and the selections use only it). Reference errors: '
          f"{ref_label(an3['refs'])}; the first-order value of the selected arm is shown beside it (PROVISIONAL).", '')
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
            ao = an3o['settings'][s]['arms'] if an3o else None
            acc_k = sel.get('accurate_equal_cost')
            row += [sci(comp['anchor_median_all']), pc(comp['median_ref']),
                    pc(ao[acc_k]['median_ref']) if (ao and acc_k) else '—',
                    pc(ao['fs|BE|0.01|K1']['median_ref']) if ao else '—',
                    f"{dg['key']} ({fx(dg['ratio'])})" if dg else '—']
            rows.append(row)
        r.table(['setting', 'rule', 'anchor-resolved cases', 'H4a', 'H4b', 'fast-equal-accuracy arm', 'time ratio',
                 'median anchor disc.', 'median ref. error', 'accurate-equal-cost arm', 'time ratio', 'median anchor disc.',
                 'median ref. error', 'BE median anchor disc.', 'BE median ref. error',
                 'accurate arm median ref. error, first-order (PROVISIONAL)', 'BE median ref. error, first-order (PROVISIONAL)',
                 'cheapest equal-accuracy arm (diagnostic, ratio)'], rows)
        dgr = [d_['ratio'] for d_ in ((an3o or an3)['hypotheses']['H4'][s_].get('cheapest_equal_accuracy_diagnostic')
                                     for s_ in (an3o or an3)['hypotheses']['H4']) if d_]
        r('H4a (speed at equal accuracy) has a registered target: a paired time ratio ≤ 0.6. It is met nowhere, so the '
          'registered fast-equal-accuracy arm is **unavailable** in every setting. Smaller speed gains at equal accuracy do '
          + (f'exist: the cheapest equal-accuracy arms (diagnostic) have ratios {fx(min(dgr))}–{fx(max(dgr))}. ' if dgr else 'not exist. ')
          + 'H4b (accuracy at equal cost) is a witness test at Δt₀ allowing a ratio ≤ 1.1; the accurate-equal-cost '
          'selection is a different rule (smallest median anchor discrepancy among eligible arms at any step with ratio ≤ 1), '
          'so the two can name different arms.', '')
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
                r('First-order H6 (PROVISIONAL, reference-limited (descriptive)): ' + '; '.join(
                    f"{k}: median change {('%+.1f %%' % (100 * v['median_change'])) if 'median_change' in v else '—'}, "
                    f"bars met {v.get('bars_met', '—')}" for k, v in h6.items()) + '.', '')
        elif h6:
            r('H6 (3D dial, descriptive, reference-limited): ' + '; '.join(
                f"{k}: median change {('%+.1f %%' % (100 * v['median_change'])) if 'median_change' in v else '—'}, "
                f"improved {fx(v.get('improved_fraction'))}, bars met {v.get('bars_met', '—')}" for k, v in h6.items()) + '.', '')
        plot_speedup3d(an3)
        r('![3D speed-up](plots/speedup3d.png)', '')
        res3 = load(HERE / 'runs' / 'c3grid' / 'archive' / 'output' / 'result.json')
        rows = []
        for s_, sel_ in an3['selection'].items():
            pan = res3['timing'].get(f'setting|{s_}', {})
            rule = an3['settings'][s_]['rule']
            for role, k in [('accurate equal cost', sel_.get('accurate_equal_cost'))] + \
                    [('cheapest equal accuracy (diagnostic)', ((an3o or an3)['hypotheses']['H4'][s_].get('cheapest_equal_accuracy_diagnostic') or {}).get('key'))]:
                if not k:
                    continue
                t_ = tstats(pan, f'{s_}|{rule}|{k}')
                cm = an3['settings'][s_]['arms'][k].get('ratio_common')
                if t_:
                    rows.append([s_, role, k, t_['n'], fx(t_['median']), f"{fx(t_['q25'])}–{fx(t_['q75'])}", t_['outliers'],
                                 f"{t_['tB']:.3g} s", f"{t_['tA']:.3g} s", fx(cm)])
        r(f"**3D timing detail** (A–B–A against the deployed fixed-sweep BE at Δt₀; one {res3.get('gpu')}; absolute times "
          'are medians per rollout of one case):', '')
        r.table(['setting', 'role', 'arm', 'invocations', 'median ratio', 'IQR', 'MAD outliers', 'median time B',
                 'median time A (BE)', 'common-panel ratio'], rows)
        r(f"Common-panel baseline: `{res3['timing'].get('common', {}).get('A')}`.", '')
    r('## Error budget (each component its own ablation; none is a remainder)', '',
      '$e_{\\rm tot}=\\lVert u_a-u_{\\rm ref}\\rVert$; floor $e_P=\\lVert(I-\\Pi_{R\'})u_{\\rm ref}\\rVert$; reduced dynamics '
      '$e_{\\rm dyn}=\\lVert u_\\star-\\Pi_{R\'}u_{\\rm ref}\\rVert$ with the ideal model $\\star$ (converged rule, Galerkin, BDF2 at '
      '$\\Delta t_0/8$, tight solver). The ablations: $\\delta_{\\rm quad}=\\lVert u_a-u_{a[r\\to r_\\infty]}\\rVert$; '
      '$\\delta_{\\rm solve}=\\lVert u_a-u_{a[\\sigma\\to{\\rm tight}]}\\rVert$; '
      '$\\delta_{\\rm form}=\\lVert u_{a[\\sigma\\to{\\rm tight}]}-u_{a[\\sigma\\to{\\rm tight},\\,f\\to{\\rm GAL}]}\\rVert$ (LSPG vs Galerkin '
      'at the tight tolerance; the test functions themselves are unchanged); $\\delta_{\\rm time}=\\lVert u_a-u_{a[(s,h)\\to({\\rm BDF2},'
      '\\Delta t_0/8)]}\\rVert$, a **package**: scheme, step, startup steps, LSPG weights and (3D) the adaptive fraction change '
      'together; $D_a=\\lVert u_a-u_\\star\\rVert$ measured directly. Every quantity is a maximum over the five evolved output '
      'times relative to $\\lVert u_0\\rVert$; tables show medians over the budget population. The identity '
      '$e_\\star^2=e_P^2+e_{\\rm dyn}^2$ holds **at each output time** (gate B0), not between separately maximised or aggregated '
      'values. The δ\'s and $D_a$ are reference-free; $e_{\\rm tot}$, $e_P$, $e_{\\rm dyn}$ carry the label of the reference '
      'named in each table. Components smaller than $3U_\\star$ (three times the ideal model\'s own uncertainty, from the '
      '$\\Delta t_0/8$ vs $\\Delta t_0/16$ and tight vs tighter replays) '
      'are counted as unresolved in the diagnostics tables.', '')
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

    def diag_rows(settings):
        rows = []
        for sn, S in settings.items():
            for ak, A in S['arms'].items():
                sm_ = A['summary']
                wv = lambda k: sci((sm_.get(k) or {}).get('worst'))
                un = A.get('unresolved_vs_Ustar') or {}
                ucount = lambda k: f"{(un.get(k) or {}).get('unresolved', '—')}/{(sm_.get(k) or {}).get('cases', '—')}"
                cl = [c_['D_over_rss'] for c_ in (A.get('closure') or []) if c_ and c_.get('D_over_rss') is not None]
                ctx = A.get('context') or {}
                ctxs = ', '.join(f"{k}: {v.get('flag')}" for k, v in ctx.items()) if ctx else \
                    ', '.join(f'{k} {v}' for k, v in (A.get('context_sensitive_cases') or {}).items())
                mdv = lambda k: sci((sm_.get(k) or {}).get('median'))
                unav = sum((v_ or {}).get('unavailable', 0) for v_ in un.values())
                rows.append([sn, ak, wv('delta_quad'), wv('delta_solve'), wv('delta_form'), wv('delta_time'),
                             mdv('refine'), mdv('I_time_quad'), mdv('I_time_solve'),
                             ucount('delta_quad'), ucount('delta_solve'), ucount('delta_form'), ucount('delta_time'),
                             unav, sci((sm_.get('U_star') or {}).get('median')),
                             fx(float(np.median(cl))) if cl else '—', ctxs or '—'])
        return rows
    dhead = ['setting', 'arm', 'worst δ_quad', 'worst δ_solve', 'worst δ_form', 'worst δ_time', 'median refine',
             'median I_time,quad', 'median I_time,solve', 'δ_quad unresolved (<3U⋆)', 'δ_solve unresolved',
             'δ_form unresolved', 'δ_time unresolved', 'unavailable (all components)', 'median U⋆',
             'median closure D_a/√Σδ²', 'context (ideal-end vs deployed-end)']

    def pair_rows(settings, key, eP, edyn, etot):
        rows = []
        for sn, S in settings.items():
            for ak, A in S['arms'].items():
                pairs = [x.get(key) for x in A.get('per_case', []) if x.get(key)]
                if not pairs:
                    continue
                md = lambda f: float(np.median([q[f] for q in pairs]))
                wsum = A['summary']
                rows.append([sn, ak, len(pairs), pc(md('e_star')), pc(md('e_P')), pc(md('e_dyn')),
                             pc((wsum.get(etot) or {}).get('worst')), pc((wsum.get(eP) or {}).get('worst')),
                             pc((wsum.get(edyn) or {}).get('worst'))])
        return rows
    phead = ['setting', 'arm', 'cases', 'median e_⋆ at t*', 'median e_P at t*', 'median e_dyn at t*', 'worst e_tot',
             'worst e_P', 'worst e_dyn']
    if b2:
        e0b = b2[0]['refs'] == '2d_e0'
        st2 = {}
        for b in b2:
            st2.update(b['settings'])
        nb2 = max((A['summary'].get('delta_time') or {}).get('cases', 0) for S_ in st2.values() for A in S_['arms'].values())
        r(f"### 2D budget (median over {nb2} cases; e_tot, e_P, e_dyn against {ref_label(b2[0]['refs'])})", '',
          (f"Not rankable under E0 (scored, never ranked): {', '.join(b2[0].get('not_rankable', []))}. " if e0b else '')
          + 'The general converged-rule check (converged vs check rule) covers a pre-registered subset of cases (dev6 plus the first val32 cases; '
          + ', '.join(f"{k} {v['conv_check'].get('cases')} cases, worst {sci(v['conv_check'].get('worst'))}" for k, v in st2.items())
          + '); the finalist check of the selected arms covers every case.', '')
        r.table(head, budget_rows(st2, 'e_P_ST', 'e_dyn_ST', 'e_tot_ST'))
        if b2o and e0b:
            st2o = {}
            for b in b2o:
                st2o.update(b['settings'])
            r('First-order e_tot / e_P / e_dyn against ST (PROVISIONAL, continuity), medians per setting-arm: ' + '; '.join(
                f"{sn} {ak} {pc((A['summary'].get('e_tot_ST') or {}).get('median'))} / {pc((A['summary'].get('e_P_ST') or {}).get('median'))}"
                f" / {pc((A['summary'].get('e_dyn_ST') or {}).get('median'))}" for sn, S_ in st2o.items()
                for ak, A in S_['arms'].items()) + '.', '')
        r('**Budget diagnostics** (worst values; counts of cases where a component is not resolved against the ideal '
          'model\'s own uncertainty; closure; context sensitivity):', '')
        r.table(dhead, diag_rows(st2))
        r('**Orthogonal pair at the common time $t^\\ast$** (where $e_\\star^2=e_P^2+e_{\\rm dyn}^2$ holds) and worst values:', '')
        r.table(phead, pair_rows(st2, 'orthogonal_pair_at_tstar_ST', 'e_P_ST', 'e_dyn_ST', 'e_tot_ST'))
        r('Controls: ' + '; '.join(f"{k}: B2 {'discriminating' if (v.get('B2') or {}).get('discriminating') else 'NOT discriminating'}, "
                                   f"B3 {(v.get('B3') or {}).get('exceeds')}/{(v.get('B3') or {}).get('cases')} cases"
                                   for k, v in st2.items()) + '.', '')
        gl = [f"{Path(b['run']).parents[1].name}: " + ', '.join(f"{k} {'pass' if (v.get('passed') if isinstance(v, dict) else v) is True else (v.get('passed') if isinstance(v, dict) else v)}"
                                                               for k, v in b['gates'].items()) for b in b2]
        r('Budget gates: ' + '; '.join(gl) + '.', '')
        wh = {sn: S.get('withheld') for sn, S in st2.items() if S.get('withheld')}
        if wh:
            r('Withheld attributions (non-discriminating control): ' + '; '.join(f'{k}: {v}' for k, v in wh.items()) + '.', '')
        plot_budget(st2, 'budget2d.png', '2D', 'e_P_ST', 'e_dyn_ST', b2[0]['refs'])
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
        nb3 = max((A['summary'].get('delta_time') or {}).get('cases', 0) for S_ in b3['settings'].values() for A in S_['arms'].values())
        r(f"### 3D budget (median over the first {nb3} cases; e_tot, e_P, e_dyn against {ref_label(b3['refs'])})", '',
          ('' if b3.get('complete') else 'Partial: W1024 pending.'), '')
        r.table(head, budget_rows(b3['settings'], 'e_P', 'e_dyn', 'e_tot'))
        if b3o and b3['refs'] == '3d_e0':
            r('First-order e_tot / e_P / e_dyn (PROVISIONAL, reference-limited; continuity), medians per setting-arm: ' + '; '.join(
                f"{sn} {ak} {pc((A['summary'].get('e_tot') or {}).get('median'))} / {pc((A['summary'].get('e_P') or {}).get('median'))}"
                f" / {pc((A['summary'].get('e_dyn') or {}).get('median'))}" for sn, S_ in b3o['settings'].items()
                for ak, A in S_['arms'].items()) + '.', '')
        r('**Budget diagnostics:**', '')
        r.table(dhead, diag_rows(b3['settings']))
        r('**Orthogonal pair at the common time $t^\\ast$** and worst values:', '')
        r.table(phead, pair_rows(b3['settings'], 'orthogonal_pair_at_tstar', 'e_P', 'e_dyn', 'e_tot'))
        r('Budget gates: ' + ', '.join(f"{k} {(v.get('passed') if isinstance(v, dict) else v)}" for k, v in b3['gates'].items()) + '.', '')
        wh3 = {sn: S_.get('withheld') for sn, S_ in b3['settings'].items() if S_.get('withheld')}
        cc3 = {sn: S_.get('conv_check', {}) for sn, S_ in b3['settings'].items()}
        r('Converged-rule check per setting (r_inf vs the next finer rule, every r_inf-dependent run type, 32 cases): ' +
          '; '.join(f"{k} worst {sci(v.get('worst'))}, all runs valid {v.get('all_valid')}, passed {v.get('passed')}"
                    for k, v in cc3.items()) + '.'
          + (' Withheld attributions: ' + '; '.join(f'{k}: {v}' for k, v in wh3.items()) + '. The check fails closed '
             'because ' + ', '.join(f'{k}: {finer_invalid(k)}' for k in wh3) + ' run(s) at the finer rule are unverified '
             '(convergence unverified, not shown inadequate); the selected '
             'arms\' own finalist checks can still pass because they compare only those arms\' trajectories at the '
             'deployed, converged and finer rules, all verified.' if wh3 else '')
          + ' Controls: ' + '; '.join(f"{k}: B2 {'discriminating' if (v.get('B2') or {}).get('discriminating') else 'NOT discriminating'}, "
                                      f"B3 {(v.get('B3') or {}).get('exceeds')}/{(v.get('B3') or {}).get('cases')} cases"
                                      for k, v in b3['settings'].items()) + '.', '')
        plot_budget(b3['settings'], 'budget3d.png', '3D', 'e_P', 'e_dyn', b3['refs'])
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
      f"- **C1's J4 failure was not only an under-resolved 48³ check**: at R′ = 1024 the successive-rule distance of "
      f"{max((x['n_over'] for p_, x in esc_rows.get('W1024', {}).items() if p_ == 'gl64|gl80'), default='—')} case(s) "
      'stays above the bar up to 64³ vs 80³ (A11); the extension to 96³ is post hoc.',
      '- **Gauss 112³ at R′ = 1024 is infeasible** on one H200 with the vendored table builder (smoke `smk3ext` failed '
      'out of memory, A11.3); the extension stops at 96³.',
      '- **B0 real-data mismatch control was non-discriminating** as an absolute test on the 2D smoke; it is now relative '
      '(A12).',
      '- **Offline scoring on the local GB10 GPU corrupted R′ = 1024 decodes** (caught by the G-score gate); all offline '
      'analysis runs on CPU (A8.1).',
      '- **Retracted: the first frozen 3D fast-equal-accuracy selection** (A15). It ignored H4a\'s step and 0.6-ratio '
      'constraints; under the registered rule no 3D fast-equal-accuracy arm exists. Caught by the Codex results audit before '
      'any 3D budget job ran; the retracted frozen file is kept in `checks/frozen-versions/`.',
      '- **Deployed 2D rules are too coarse for the selected LSPG-CN-R arms at `fast`, `acc` and R256** (finalist '
      'quadrature check, A14); the first-order selection verdicts stand, labelled "rule inadequate for this arm", and the '
      'fallbacks are reported beside them.',
      f"- **First-order 2D/3D conclusions that change under E0** (A13): 2D H1 steps "
      f"{', '.join(k for k, v in (an_old or {}).get('hypotheses', {}).get('H1', {}).items() if v['verdict'] != an['hypotheses']['H1'][k]['verdict']) or 'none'} "
      'change verdict (see H1); of the 3D H6 transitions, '
      f"{sum(1 for k, v in (an3o or {}).get('hypotheses', {}).get('H6', {}).items() if 'median_change' in v and k in an3['hypotheses']['H6'] and (v['median_change'] < 0) != (an3['hypotheses']['H6'][k]['median_change'] < 0))}"
      f" of {len((an3o or {}).get('hypotheses', {}).get('H6', {}))} reverse the direction of the median change. The first-order "
      'numbers are kept beside the E0 ones and are superseded by them; the frozen selections are unchanged.',
      '- **The in-job 3D cohort-table hash is not stable across H200 nodes** (A11.4): the extension job is bound to the '
      'escalation job by the initial coefficients instead; offline scoring checks the CPU table hash.',
      '- **A leftover first-order reference access in `budget3d.py`** (G-score controls) was missed by the A13 code audit '
      'and caught when the smoke budget analysis was re-run; fixed before any production 3D budget analysis.',
      '- **The first W1024 3D budget job (`c3budW1024`) ran out of memory** because the budget loop kept the '
      'converged rule\'s tables alive while preparing the finer rule (A16); fixed and rerun as `c3budW1024r`.',
      '- **No formal-order claims for the reduced arms** (DESIGN A1.1): the report speaks of CN / CN-R / BDF2-family '
      'stepping and measured accuracy and cost; the E0 references are second order, the ROM arms are not claimed to be.',
      '- **Earlier drafts of this report** stacked non-additive budget components, stated the B0 identity without its '
      'per-time scope, misdefined $\\delta_{\\rm form}$ in the glossary and used second-order wording for the reduced arms; '
      'corrected after the report audits (`codex-report-1.md`, `codex-report-2.md`).', '')

    # ------------------------------------------------------------------------------------------------ provenance
    if ledger:
        r('## Jobs', '')
        r.table(['attempt', 'Slurm id', 'GPU', 'commit', 'role', 'status'],
                [[e['attempt'], e.get('slurm', '—'), e.get('gpu', '—'), e.get('commit', '—'),
                  e.get('role', '').replace('second-order', 'CN/BDF2-family'),
                  e.get('status', '')] for e in ledger['jobs']])
        r.table(['failed attempt', 'Slurm id', 'note'],
                [[e['attempt'], e.get('slurm', '—'), e.get('note', '')] for e in ledger.get('failed_attempts', [])])
    aud = []
    for p in sorted(glob.glob(str(HERE / 'audits' / 'codex-*.md')), key=lambda q: Path(q).stat().st_mtime):
        first = next((ln for ln in Path(p).read_text().splitlines() if ln.strip()), '')
        first = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', first).replace('**', '').replace('|', '/').strip()
        aud.append([f'`audits/{Path(p).name}`', first[:220] + ('…' if len(first) > 220 else '')])
    r('## Codex audits', '')
    r('Chronological; the opening statement of each audit (its verdict line). Every HOLD / NOT READY on a job or a '
      'selection was resolved and re-audited before that job ran or that selection was used. Reporting findings are '
      'addressed report version by report version; the last report audit (`codex-report-*.md`) states what remains. The '
      'full findings are in the files.', '')
    r.table(['audit', 'opening statement'], aud)

    # ------------------------------------------------------------------------------------------------ glossary
    G = [
        ('bank, $R\'$, dial', 'the frozen ordered coordinate network; a solve uses its first $R\'$ columns; the dial is the '
         'sequence of $R\'$ settings (R128 → R256 → R384 → R512 in 2D; W256 … W1024 in 3D). `fast` and `acc` are the 2D '
         'settings deployed before this lane; `M2_512` the deployed 3D setting.'),
        ('rule, $m$, $m^\\star$', 'an off-mesh quadrature for the advection term (Gauss tensor, Fibonacci or rank-1 lattice) '
         'with $m$ points; $m^\\star$ is the smallest rule of a family whose rollouts stay within $\\tau$ of the converged rule\'s.'),
        ('converged rule $r_\\infty$, J4, bar', 'a rule fine enough that the next finer rule changes the rollout by at most '
         'the bar (worst case over the cohort, field distance relative to $\\lVert u_0\\rVert$); J4 is that check.'),
        ('BE, CN, CN-R, BDF2', 'backward Euler; Crank–Nicolson; CN with two BE start steps; the two-step backward '
         'differentiation formula with one BE start step. The report calls CN / CN-R / BDF2 the "CN/BDF2 family" and makes '
         'no formal-order claim for the reduced arms (DESIGN A1.1).'),
        ('LSPG / GAL', 'least-squares (Petrov–Galerkin) and Galerkin solves of each time step with the same test functions.'),
        ('fixed-sweep solver, $K$; `fs` / `ad`; LM', 'after a few adaptive steps, exactly $K$ Levenberg–Marquardt (LM) '
         'sweeps per step with an extrapolated predictor (`fs`); `ad` = adaptive solver run to its stopping test ("verified").'),
        ('solver distance', 'field distance between a fixed-sweep run and the verified adaptive run of the same scheme and '
         'step; eligibility requires it to be at most $\\tau$ (per-case H4 bar: at most $\\min(\\tau, 0.25\\times$ own anchor '
         'discrepancy$)$).'),
        ('ST / S / TX (first-order references, PROVISIONAL)', '2D first-order references at fine resolution: ST with a small '
         'time step, S at $\\Delta t_0$, TX their Richardson combination; the 3D first-order reference is backward Euler at a '
         'reduced step (reference-limited).'),
        ('E0, $R_{ST}$, $R_S$, $\\epsilon_{\\rm ref}$ ($\\eta$)', 'the campaign\'s order-verified second-order references '
         '(central differences + Crank–Nicolson) and their per-case Richardson uncertainty (a conditional estimate, not a '
         'bound). $R_S$ (backward Euler at $\\Delta t_0$) exists in 2D only and is time-matched only to the 2D LSPG-BE arms at '
         '$\\Delta t_0$; every other S value is an unmatched diagnostic. 3D uses $R_{ST}$ only.'),
        ('tie, resolved, reference-limited', '2D first-order references: a paired difference below a heuristic threshold '
         'from the ST–TX distance is a tie (no tie threshold exists for the 3D first-order reference, whose comparisons are '
         'descriptive and reference-limited); E0: two arms are ranked on a case only if their errors differ by more than $2\\eta$ '
         'and the case-kind is eligible, medians and worst values by interval overlap. A case is resolved when it is not a '
         'tie; a verdict with fewer than half the cases resolved is reference-limited.'),
        ('selection rules', '2D accurate-equal-cost = smallest median error among a scheme\'s eligible steps with paired '
         'time ratio $\\le 1$ against the setting\'s LSPG-BE at $\\Delta t_0$; 2D fast-equal-accuracy = cheapest eligible step '
         'whose worst and median ST and TX errors are no worse than that BE\'s (tie policy). 3D accurate-equal-cost = smallest '
         'median anchor discrepancy with ratio $\\le 1$; 3D fast-equal-accuracy must meet H4a ($\\Delta t\\ge1.25\\Delta t_0$, '
         'anchor discrepancy no worse than the comparator, ratio $\\le 0.6$).'),
        ('H4b witness', 'an eligible 3D fixed-sweep arm at $\\Delta t_0$ with median anchor discrepancy $\\le 0.1\\times$ the '
         'comparator\'s and ratio $\\le 1.1$; it proves H4b and is distinct from the accurate-equal-cost selection.'),
        ('H1–H6', 'the pre-registered hypotheses of DESIGN §6: the wide bank pays off; tunability survives; CN worst-case '
         'cancellation; 3D fixed-sweep speed (H4a) and accuracy (H4b); speed against the FOM; the 3D dial.'),
        ('$t^\\ast$, guards a / b / c (H3)', '$t^\\ast$ is the output time of the largest anchor error; (a) CN lands near the '
         'time-converged error relative to BE, (b) the BE time error points against the anchor error (cosine $\\le -0.3$), (c) '
         'the cancelling cross term is large; the guards repeat (a)–(c) with the anchor (and, under E0, reference) '
         'uncertainty added.'),
        ('cost / own BE, common-baseline cost, paired time ratio', 'median over A–B–A invocations of candidate time divided '
         'by the mean of a baseline run just before and after it, on one GPU; against the setting\'s own LSPG-BE at $\\Delta t_0$ '
         '(2D) or deployed fixed-sweep BE (3D), or against one common baseline. IQR = interquartile range of the per-invocation '
         'ratios; outliers = invocations more than $3\\times1.4826\\times$ MAD from the median.'),
        ('anchor, anchor discrepancy, anchor-resolved', 'GAL-BDF2 at $\\Delta t_0/8$, tight tolerance (time-converged reduced '
         'solution) and the distance of a rollout from it; a case is anchor-resolved when the comparator\'s discrepancy '
         'exceeds $3\\times$ the anchor uncertainty.'),
        ('FOM, matched speed-up', 'the full-order finite-difference model on the same mesh with CN/BDF2-family time stepping '
         'and calibrated Newton tolerances; matched speed-up = FOM time / ROM time for the cheapest calibrated FOM '
         'configuration whose worst error is no larger than the ROM arm\'s.'),
        ('arm keys', 'grid keys carry absolute steps: `form|scheme|Δt|level` (2D; `prod` = production tolerance) and '
         '`fs|scheme|Δt|K` / `ad|form|scheme|Δt|level` (3D); 2D budget keys `form|scheme|f` carry the step factor '
         '$f=\\Delta t/\\Delta t_0$; $\\Delta t_0$ = 0.005 (2D) and 0.01 (3D).'),
        ('ideal model $\\star$', 'the reduced model with every ingredient at its ideal: converged rule, Galerkin, BDF2 at '
         '$\\Delta t_0/8$, tight solver.'),
        ('$e_{\\rm tot}$, $e_\\star$, $e_P$, $e_{\\rm dyn}$', 'an arm\'s error, the ideal model\'s error, the projection floor (the '
         'best error in the span) and the ideal model\'s distance to the projected reference; $e_\\star^2=e_P^2+e_{\\rm dyn}^2$ '
         'holds at each output time (gate B0), not between separately maximised or aggregated values. The orthogonal pair '
         'is reported at the common time $t^\\ast$.'),
        ('$\\delta_{\\rm quad}$, $\\delta_{\\rm solve}$, $\\delta_{\\rm form}$, $\\delta_{\\rm time}$, $D_a$', 'rule changed to $r_\\infty$; '
         'solver tolerance changed to tight; LSPG vs Galerkin both at the tight tolerance (same test functions); the time '
         'package (scheme, step, startup steps, LSPG weights, 3D adaptive fraction) changed to BDF2 at $\\Delta t_0/8$; $D_a$ the '
         'direct distance to the ideal model. Each is a separate measurement; none is a remainder.'),
        ('refine, mixed differences', 'refine = change when the arm\'s own step is halved (tight solver); mixed differences '
         '$I_{\\rm time,quad}$ and $I_{\\rm time,solve}$ = how much the time ablation changes when the rule or the solver is '
         'also changed (interaction indicators).'),
        ('$U_\\star$, unresolved, unavailable', '$U_\\star$ = the ideal model\'s own uncertainty (time-step and tolerance replays); '
         'a component is unresolved on a case if it is smaller than $3U_\\star$; unavailable if a needed trajectory is unverified.'),
        ('closure, context', 'closure = $D_a/\\sqrt{\\sum\\delta^2}$, a descriptive ratio of separately maximised quantities (it '
         'does not establish orthogonality or an additive decomposition); context = whether a component measured from the '
         'ideal end differs by more than $2\\times$ from the deployed end.'),
        ('largest component (cases)', 'per case, which measured component is biggest, counted over cases; a sensitivity '
         'ranking, not a causal share.'),
        ('finalist check, $\\tau$, fallback', 'a selected arm must stay within $\\tau=2.5\\times10^{-4}$ of itself run with the '
         'converged rule on every case (and the converged-vs-finer check must pass); a failing arm keeps its selection '
         'labelled "rule inadequate"; its frozen fallback (next finer rule of the family) is qualified separately.'),
        ('B0–B3, G-score, controls', 'B0 the per-time Pythagorean identity; B1 a bitwise re-run; B2 a deliberately bad rule '
         'that must be detected; B3 a time ablation that must be detectable; G-score the independent re-decoding of every '
         'saved trajectory; negative controls (case swap, time shift, ...) that must break agreement.'),
        ('pre-registered / POST HOC', 'decided before results were read / decided after (labelled; exploratory).'),
        ('dev6, val32, 923801', 'development and validation cohorts (selection allowed); test64 / 923901 are replication-only '
         'and were not used.'),
        ('% of $\\lVert u_0\\rVert$, pp', 'errors relative to the initial-condition norm, in percent; pp = percentage points (a '
         'difference of two such percentages).'),
    ]
    r('## Glossary', '')
    for term, d_ in G:
        r(f'- **{term}**: {d_}')
    r('')
    # ------------------------------------------------------------------------------------------------ summary
    S = []
    if an is not None:
        H = an['hypotheses']
        ref = 'E0 R_ST' if an['refs'] == '2d_e0' else 'first-order TX (PROVISIONAL)'
        g = an_old['selection']['global']['scheme']
        S.append(f"- **2D scheme.** {g} (a CN-family scheme; selected against the first-order references, frozen). Against {ref}: "
                 + '; '.join(f"H1 {k} **{v['verdict']}**" for k, v in H['H1'].items())
                 + f"; H2 (tunability) **{'pass' if H['H2']['a_pass'] and H['H2']['b'] is True else 'fail'}**; H3 at "
                 f"{H['H3']['ST']['worst_case']}: **{H['H3']['ST']['primary']['verdict']}**; H5 best matched speed-up "
                 f"{fx(H['H5']['ST']['best_combined'][0])}× (an {armname(H['H5']['ST']['best_combined'][1])} arm in "
                 f"`{H['H5']['ST']['best_combined'][1].split('|')[0]}`, not the selected scheme) vs re-measured C2 `acc` "
                 f"{fx(H['H5']['ST']['best_comparator'][0])}× "
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
        S.append('- **3D fixed-sweep CN (K = 2, 2.5 Δt₀)** departs from the verified adaptive solve on a persistent set of '
                 f"cases (the same at every adequate rule, with {len(extra)} rule-dependent exception(s)), so the pre-registered rule selection "
                 'was unavailable; the POST HOC selection uses the adaptive CN arm.')
    if an3 is not None:
        h4 = an3['hypotheses']['H4']
        S.append('- **3D fast solver (H4).** H4a (speed at equal accuracy, ratio ≤ 0.6): '
                 + ', '.join(f"{k} {v.get('H4a', {}).get('verdict', v.get('verdict'))}" for k, v in h4.items())
                 + '; H4b (accuracy at equal cost): '
                 + ', '.join(f"{k} {v.get('H4b', {}).get('verdict', '—')}" for k, v in h4.items())
                 + ('. Read together: the fixed-sweep CN/BDF2-family solver buys accuracy at equal cost; the registered '
                    'speed-at-equal-accuracy target (ratio ≤ 0.6) is unavailable, although smaller speed gains exist.' if all(v.get('H4a', {}).get('verdict') == 'fail' and v.get('H4b', {}).get('verdict') == 'pass'
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
        S.append('- **2D budget: largest measured component** (per setting-arm; a one-ingredient sensitivity, not a causal share or a guaranteed gain): '
                 + ', '.join(f'{k} {v}' for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])) + '.')
    if b3:
        cnt3 = {}
        for s_, S_ in b3['settings'].items():
            for ak, A in S_['arms'].items():
                lg = A.get('largest') or {}
                if lg:
                    top = max(lg.items(), key=lambda kv: kv[1])[0]
                    cnt3[top] = cnt3.get(top, 0) + 1
        wh = [k for k, v in b3['settings'].items() if v.get('withheld')]
        S.append('- **3D budget: largest measured component** (per setting-arm; a sensitivity, not a causal share): '
                 + ', '.join(f'{k} {v}' for k, v in sorted(cnt3.items(), key=lambda kv: -kv[1]))
                 + ('; quadrature attribution withheld at ' + ', '.join(f'{k} ({finer_invalid(k)} unverified run(s) at the finer rule; convergence unverified, not shown inadequate)' for k in wh) if wh else '')
                 + '.')
    r.lines[SUMMARY_AT:SUMMARY_AT] = ['## Answers in brief (generated)', ''] + S + ['']
    (HERE / 'report.md').write_text('\n'.join(r.lines) + '\n')
    print(HERE / 'report.md', 'pending:', pending)


if __name__ == '__main__':
    main()
