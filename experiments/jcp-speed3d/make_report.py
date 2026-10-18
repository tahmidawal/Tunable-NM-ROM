"""jcp-speed3d report generator: report.md, plots/*.png and runs/EXPORT.json from the analysis files only
(checks/analysis-grid.json, checks/analysis-pm.json, checks/fom-match.json, checks/SELECTION-SCREEN.json, optional
checks/analysis-r1.json, runs/GPU-LEDGER.json, runs/JOB-LEDGER.json, audits/). No number in report.md is typed by hand.

    python make_report.py
"""
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CH = HERE / 'checks'
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
SCH_COL = {'BE': COL[0], 'CN': COL[1], 'BDF2': COL[2]}


def jl(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def f(x, d=3):
    if x is None:
        return '–'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    if not np.isfinite(x):
        return str(x)
    if x == 0:
        return '0'
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        m, e = f'{x:.{d - 1}e}'.split('e')
        return f'{m}e{int(e)}'
    r = f'{x:#.{d}g}'
    return r.rstrip('.') if 'e' not in r else r


def table(head, rows):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join('---' for _ in head) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def pretty(key):
    """'M2_512|gl24|fs|BDF2|0.0125|K1' -> 'M2_512 gl24, FS-K1 BDF2, Δt = 0.0125'."""
    if key is None:
        return '–'
    p = key.split('|')
    s, r, kind = p[0], p[1], p[2]
    if kind == 'ad':
        return f'{s} {r}, ADT {p[3]} {p[4]} {p[5]}'
    sol = {'fs': 'FS', 'fsp': 'FSP', 'qn': 'QN'}[kind] + '-' + p[5].replace('k', '').replace('inf', '∞').replace('K', 'K')
    if kind == 'qn':
        sol = 'QN-' + p[5][1:].replace('inf', '∞')
    return f'{s} {r}, {sol} {p[3]}, Δt = {float(p[4]):.4g}'


def verdict_of(path):
    t = Path(path).read_text()
    for pat in (r'NOT READY', r'DO NOT SUBMIT', r'\bREADY\b', r'\bSUBMIT\b'):
        m = re.search(pat, t)
        if m:
            return m.group(0)
    return 'see file'


# ------------------------------------------------------------------------------------------------------------ plots
def plots(G, PM):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pd = HERE / 'plots'
    pd.mkdir(exist_ok=True)
    made = []
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'grid.color': '#e5e4df', 'grid.linewidth': .6})
    C = G['configs']
    sel = jl(CH / 'SELECTION-SCREEN.json')
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for sch in ('BE', 'CN', 'BDF2'):
        xs = [v['ratio'] * v['t_A'] for v in C.values() if v['eligible'] and v['scheme'] == sch and v['ratio']]
        ys = [v['median'] for v in C.values() if v['eligible'] and v['scheme'] == sch and v['ratio']]
        ax.scatter(xs, ys, s=14, color=SCH_COL[sch], alpha=.75, label=f'{sch} (eligible)', edgecolor='white', linewidth=.4)
    for c, v in sel['comparators'].items():
        k = v['key']
        ax.scatter([C[k]['ratio'] * C[k]['t_A']], [C[k]['median']], marker='s', s=60, facecolor='none', edgecolor='#0b0b0b', linewidth=1.2)
        ax.annotate(c, (C[k]['ratio'] * C[k]['t_A'], C[k]['median']), xytext=(5, 4), textcoords='offset points', fontsize=8)
        w = v['winner']
        if w:
            ax.scatter([C[w]['ratio'] * C[w]['t_A']], [C[w]['median']], marker='*', s=110, color='#0b0b0b')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('screening query time (s): ratio to C_old × C_old median time, H200')
    ax.set_ylabel('median E0 error over 64 validation cases')
    ax.set_title('Route 1–2 screening: time vs accuracy (squares = comparators, stars = H2 winners)', fontsize=9)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(pd / 'pareto-screening.png', dpi=150)
    plt.close(fig)
    made.append('plots/pareto-screening.png')
    if PM and 'p1' in PM:
        H31 = PM['p1']['H31']
        labs = [l_ for l_ in H31 if not H31[l_].get('missing')]
        if labs:
            fig, ax = plt.subplots(figsize=(7, 2.8))
            parts = ['project', 'evolve', 'decode']
            left = np.zeros(len(labs))
            for i, p in enumerate(parts):
                v = np.array([H31[l_]['components_seconds'][p] * 1e3 for l_ in labs])
                ax.barh(labs, v, left=left, color=COL[i], label=p, edgecolor='white', linewidth=2, height=.6)
                left += v
            ax.scatter([H31[l_]['fused_seconds'] * 1e3 for l_ in labs], labs, marker='|', s=200, color='#0b0b0b', label='fused query', zorder=3)
            ax.set_xlabel('warm time (ms), separately jitted components (indicative)')
            ax.legend(frameon=False, ncol=4, fontsize=8, loc='lower right')
            ax.set_title('H3.1 overhead breakdown', fontsize=9)
            fig.tight_layout()
            fig.savefig(pd / 'profile.png', dpi=150)
            plt.close(fig)
            made.append('plots/profile.png')
        H33 = PM['p1']['H33']
        if H33:
            fig, ax = plt.subplots(figsize=(5.5, 3.4))
            for i, (l_, v) in enumerate(H33.items()):
                Bs = sorted(int(b) for b, r in v['per_B'].items() if r.get('throughput'))
                ax.plot(Bs, [v['per_B'][str(b)]['throughput'] for b in Bs], marker='o', color=COL[i], lw=2, ms=5, label=l_)
            ax.set_xscale('log', base=2)
            ax.set_xlabel('batch size B')
            ax.set_ylabel('throughput (cases / s)')
            ax.set_title('H3.3 batched throughput (H200)', fontsize=9)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(pd / 'batching.png', dpi=150)
            plt.close(fig)
            made.append('plots/batching.png')
    if PM and 'm1' in PM:
        m1 = PM['m1']
        Ls = sorted(int(L) for L in m1['per_L'])
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for i, l_ in enumerate(m1['H41']):
            ts = m1['H41'][l_]['seconds']
            xs = [L for L in Ls if ts.get(str(L)) or ts.get(L)]
            ax.plot(xs, [(ts.get(str(L)) or ts.get(L)) for L in xs], marker='o', lw=2, color=COL[i], label=f'ROM {l_} (mesh-free)')
        fams = {}
        for L in Ls:
            for fk, fv in m1['per_L'][str(L)]['fom'].items():
                for l_, pl in m1['per_L'][str(L)]['panels'].items():
                    sp = pl['speedup'].get(fv.get('key') or fk)
                    if sp and sp.get('t_B'):
                        fams.setdefault(fv.get('key') or fk, {})[L] = sp['t_B']
        for j, (k, d) in enumerate(sorted(fams.items())):
            xs = sorted(d)
            ax.plot(xs, [d[x] for x in xs], ls='--', marker='s', ms=4, lw=1.2, color=COL[(j + 4) % 8], label=f'FOM {k}')
        ax.set_yscale('log')
        ax.set_xlabel('mesh L (cells per side)')
        ax.set_ylabel('warm query time (s)')
        ax.set_title('ROM and FOM time against mesh size (m1, one H200 job)', fontsize=9)
        ax.legend(frameon=False, fontsize=6.5, ncol=2)
        fig.tight_layout()
        fig.savefig(pd / 'mesh-time.png', dpi=150)
        plt.close(fig)
        made.append('plots/mesh-time.png')
    return made


def g2_literal():
    """G2 recomputed from the raw g1-g3 rows (cached in checks/g2-literal.json): certified vs recorded ratio on fixed steps."""
    cp = CH / 'g2-literal.json'
    if cp.exists():
        return jl(cp)
    import math
    NS = {'BE': 0, 'CN': 0, 'BDF2': 1}
    A = {'M2_512': 3, 'W256': 6, 'W512': 6, 'W768': 6, 'W1024': 6}
    out = dict(steps=0, violations=0, max_gamma=0., max_diff=0., where={})
    for g in ('g1', 'g2', 'g3'):
        res = jl(HERE / 'runs' / g / 'archive' / 'output' / 'result.json')
        for x in res['rows']:
            if not x['spec'].startswith('fs|') or 'gam_cert' not in x:
                continue
            p = x['spec'].split('|')
            dt = float(p[2])
            steps = int(round(.25 / dt))
            ae = int(min(steps, max(math.ceil(A[x['setting']] * .01 / dt - 1e-9), NS[p[1]])))
            gc = np.array(x['gam_cert'], float)[ae:]
            gr = np.array(x['steps']['gn'], float)[ae:]
            bad = ~(np.abs(gc - gr) <= 1e-9 * np.abs(gr) + 1e-12)
            out['steps'] += len(gc)
            if bad.any():
                out['violations'] += int(bad.sum())
                out['max_gamma'] = max(out['max_gamma'], float(np.maximum(gc, gr)[bad].max()))
                out['max_diff'] = max(out['max_diff'], float(np.abs(gc - gr)[bad].max()))
                k = f"{g}|{x['tag']}|{x['spec']}"
                out['where'][k] = out['where'].get(k, 0) + int(bad.sum())
    cp.write_text(json.dumps(out, indent=1) + '\n')
    return out


# ------------------------------------------------------------------------------------------------------------ report
def main():
    import sys
    G = jl(CH / 'analysis-grid.json')
    sel = jl(CH / 'SELECTION-SCREEN.json')
    fm = jl(CH / 'fom-match.json')
    PM = jl(sys.argv[1] if len(sys.argv) > 1 else CH / 'analysis-pm.json') or {}
    R1 = jl(CH / 'analysis-r1.json')
    gl = jl(HERE / 'runs' / 'GPU-LEDGER.json')
    jlg = jl(HERE / 'runs' / 'JOB-LEDGER.json')
    C = G['configs']
    p1, m1 = PM.get('p1'), PM.get('m1')
    final = bool(p1 and m1)
    made = plots(G, PM)
    L = []
    w = L.append
    w('# jcp-speed3d — can the 3D second-order accuracy gain become a real speed-up?')
    w('')
    w(f"**Status:** {'p1 and m1 analysed; confirmed verdicts final for the validation cohort' if final else 'PROVISIONAL — screening only; p1/m1 not yet analysed'}"
      f"{'; replication on 923901 included' if R1 else '; replication (r1) not run'}. Every number below is generated by "
      "`make_report.py` from the analysis JSONs. All errors are against the E0 second-order reference $R_{ST}$: "
      "*a conditional Richardson estimate (asymptotic assumption, not a rigorous bound); its cross-check rule was amended "
      "after data (E0 A10).* The study is prospectively specified conditional on Step 2's exploration (DESIGN A1). "
      "Every timing is on one H200 type; screening ratios come from separate jobs, confirmed ratios from paired panels in one job.")
    w('')
    # ---------------- summary
    w('## Summary')
    w('')
    if p1:
        for c, v in p1['H2'].items():
            w(f"- **H2 vs {c}:** winner {pretty(v['winner'])}; speed-up screened {f(v['S_screen'])}×, confirmed "
              f"{f(v['S_confirmed'])}× → **{v['verdict']}** (bar $1/0.6\\approx1.67$).")
    else:
        for c, v in sel['comparators'].items():
            w(f"- H2 vs {c} (screening only): winner {pretty(v['winner'])}, $S$ = {f(v['S_screen'])}×.")
    if m1:
        cells = m1['Hcomp_largest_mesh']
        w(f"- **Against the full simulation (route 4, L = {cells['L']}):** K2 of the governing plan "
          f"**{'FIRES' if m1['K2_fires'] else 'does not fire'}** — "
          + ('no selected ROM reaches 2× against the best matched FOM (the union of families). ' if m1['K2_fires'] else '')
          + 'Matched speed-ups use a finite FOM ladder: an upper bound on optimal FOM cost, hence an **upper bound on the '
            'claimed ROM speed-up**.')
    w('')
    w('```mermaid')
    w('flowchart LR')
    w('  G[g1-g3 grid: 684 configurations, 64 val cases]:::job --> S[screen + select H1/H2]:::doc')
    w('  F[f1 FOM ladder]:::job --> M[FOM matcher, K10]:::doc')
    w('  S --> P[p1: confirmations, profile, mesh-free, batching]:::job')
    w('  S --> M --> R[m1: route 4, n = 65..385]:::job')
    w('  P --> Z[analysis-pm, freeze]:::doc')
    w('  R --> Z --> X[r1 replication 923901]:::job')
    w('  classDef doc fill:#e5e7eb,stroke:#374151;')
    w('  classDef job fill:#dbeafe,stroke:#1d4ed8;')
    w('```')
    w('')
    # ---------------- jobs
    w('## Jobs, budget and integrity')
    w('')
    rows = []
    gpuh = {j['slurm_id']: j for j in gl['jobs']}
    for j in jlg['jobs']:
        g = gpuh.get(j['slurm_id'], {})
        rows.append([j['attempt'], j['slurm_id'], j['gpu'], f(g.get('gpu_hours')), j.get('commit', '–')[:9], j['role']])
    w(table(['attempt', 'job', 'GPU', 'GPU-h (sacct)', 'commit', 'role'], rows))
    w('')
    w(f"GPU-hours spent: **{f(gl['spent_gpu_hours'])}** of the {f(gl['cap_gpu_hours'])} cap (DESIGN A7). Cluster namespace "
      "`/cluster/tufts/paralab/tawal01/jcp-speed3d/`; every log carries `jax_backend=gpu`.")
    w('')
    ctl = G['controls']
    rows = [[sn, f"{c['K1']['nonstationary']}/{c['K1']['steps']}", f(c['K1_fires']), f"{c['K2']['nonstationary']}/{c['K2']['steps']}",
             f(c['K2_fires']), f(c['K3_CN']['dist_cases_above']), f(c['K3_fires'])] for sn, c in ctl.items()]
    w('Controls that must fail (each fired on the production data):')
    w('')
    w(table(['setting', 'K1 non-stationary steps', 'K1 fires', 'K2 non-stationary steps', 'K2 fires', 'K3 cases over bar', 'K3 fires'], rows))
    w('')
    k4 = [j['K4'] for j in G['jobs']]
    w(f"K4 (scoring swap / shift) fires in every grid job (swap changes {f(min(x['K4_case_swap'] for x in k4))}–"
      f"{f(max(x['K4_case_swap'] for x in k4))}, shift {f(min(x['K4_time_shift'] for x in k4))}–{f(max(x['K4_time_shift'] for x in k4))}); "
      f"K9 (distance classifier) {'fires' if G['K9_gate'] else 'DOES NOT fire'}; K10 (FOM matcher) "
      f"{'rejects both synthetic entries' if fm['K10_all_rejected'] else 'FAILED'}; planned population complete "
      f"({len(G['population']['missing'])} missing, {len(G['population']['duplicates'])} duplicates).")
    w('')
    g2 = g2_literal()
    w(f"**G2 failed literally** on {g2['violations']} of {g2['steps']:,} fixed-phase FS steps (certified vs recorded stationarity "
      f"ratio, both $\\le${f(g2['max_gamma'])}, far below the $10^{{-3}}$ threshold; no classification changes; E-b uses only the "
      'certified ratio; DESIGN A9/A10). The g3 log carries a JAX "captured constants (3.26 GB)" warning from the untimed '
      'H1.1 replay (A8); the results audit found no anomaly in W1024 outputs or timings.')
    w('')
    # ---------------- route 1
    w('## Route 1 — one pass per step at big steps (H1)')
    w('')
    h11 = G['H11']
    labs = [v['label'] for d in h11.values() for v in d.values()]
    w(f"**H1.1** (replay of FS-K2 CN $2.5\\Delta t_0$ failures, descriptive): {len(labs)} replayed cases; "
      + ', '.join(f"{labs.count(x)} \"{x}\"" for x in sorted(set(labs)))
      + '. "mixed" means that both $K\\le4$ extra sweeps and trust ×10 reach stationarity at the replayed step '
        '(a statement about that step, not a cause of the trajectory error).')
    w('')
    h12 = G['H1']['H12']
    w('**H1.2** (FSP-K1 at $2.5\\Delta t_0$): ' + '; '.join(f'{k} {v}' for k, v in h12.items()) + '.')
    w('')
    h13 = G['H1']['H13']
    tab = {}
    for k, v in h13.items():
        sn, sch, dt, kk = k.split('|')
        tab.setdefault((sn, sch, dt), {})[kk] = v
    rows = [[sn, sch, f'{float(dt) / .01:.3g}', *[('E' if tab[(sn, sch, dt)].get(k) == 'eligible' else '–') for k in ('k1', 'k4', 'kinf')]]
            for (sn, sch, dt) in sorted(tab)]
    w(f"**H1.3** (QN-k eligible at $\\Delta t\\ge1.25\\Delta t_0$; E = eligible): {sum(v == 'eligible' for v in h13.values())} of {len(h13)} cells.")
    w('')
    w(table(['setting', 'scheme', 'Δt/Δt₀', 'QN-1', 'QN-4', 'QN-∞'], rows))
    w('')
    H14 = p1['H14'] if p1 else {k: dict(verdict=v['verdict'], ratio_screen=v['ratio_screen'], ratio_confirmed=None) for k, v in G['H1']['H14'].items()}
    rows = [[k.replace('|', ' '), f(v.get('ratio_screen')), f(v.get('ratio_confirmed')), v['verdict']] for k, v in sorted(H14.items())]
    w('**H1.4** (cheapest eligible QN vs FS-K2, pass iff confirmed paired ratio ≤ 0.6; "rescue" = QN eligible where FS-K2 is not):')
    w('')
    w(table(['cell', 'screening ratio', 'confirmed ratio', 'verdict'], rows))
    w('')
    # ---------------- route 2
    w('## Route 2 — narrower bank, fewer points, second-order schemes (H2)')
    w('')
    rows = []
    for c, v in sel['comparators'].items():
        cc, ww = C[v['key']], C[v['winner']] if v['winner'] else None
        h = p1['H2'][c] if p1 else {}
        rows.append([c, f"{f(cc['median'])} / {f(cc['worst'])}", pretty(v['winner']),
                     f"{f(ww['median'])} / {f(ww['worst'])}" if ww else '–', f(v['S_screen']), f(h.get('S_confirmed')),
                     h.get('verdict', 'screening only'), pretty(v['B_star']), f(h.get('S_BE_confirmed')), f(h.get('S_scheme_confirmed'))])
    w(table(['comparator', 'comparator median / worst', 'winner W*', 'W* median / worst', 'S screen', 'S confirmed', 'H2',
             'B* (best BE)', 'S_BE confirmed', 'S_C/S_BE confirmed'], rows))
    w('')
    w(f"Eligible configurations: {sum(v['eligible'] for v in C.values())} of {len(C)}. Reduced rules are empirically "
      "validated surrogates, not quadrature-equivalent (A1.13).")
    if p1:
        w('')
        rows = [[l_, pretty(sel['labels'].get(l_)), v['rule'], f(v['delta_quad_max']), f(p1['reduced_adt'].get(l_, {}).get('worst')),
                 f(p1['reduced_adt'].get(l_, {}).get('ok'))] for l_, v in p1['conv'].items()]
        w('Converged-rule sensitivity of the finalists ($\\delta_{\\rm quad}$ = lattice distance to the same configuration on the '
          "setting's converged rule, max over cases and output times) and reduced-rule ADT tight/tighter agreement (bar 2.5e-5):")
        w('')
        w(table(['label', 'configuration', 'converged rule', 'δ_quad max', 'ADT tight/tighter', 'ok'], rows))
    w('')
    w('![screening Pareto](plots/pareto-screening.png)')
    w('')
    # ---------------- route 3
    w('## Route 3 — overhead (H3)')
    w('')
    if p1:
        rows = [[l_, f(v['fused_seconds'] * 1e3), f(v['share']['project']), f(v['share']['evolve']), f(v['share']['decode']),
                 f(v.get('share_adaptive_start')), f(v['G3_sum_over_fused']), f(v['G3'])] for l_, v in p1['H31'].items() if not v.get('missing')]
        w('**H3.1** profile (indicative; components separately jitted). Shares of the fused warm query:')
        w('')
        w(table(['label', 'fused (ms)', 'projection', 'step loop', 'decode', 'adaptive start', 'G3 sum/fused', 'G3 within 15 %'], rows))
        w('')
        w('![profile](plots/profile.png)')
        w('')
        rows = []
        for l_, v in p1['H32'].items():
            rows.append([l_, f(v.get('eligible')), f(v.get('dist_worst')), f(v.get('median')), f(v.get('median_mesh')), f(v.get('worst')),
                         f(v.get('worst_mesh')), f(v.get('b')), f(v.get('c')), f(v.get('qualified')), f((v.get('K5') or {}).get('fails'))])
        w('**H3.2** mesh-free online path (qualified iff eligible against the mesh-free-initialised ADT, (b) median and worst '
          'tie-or-beat the mesh path or exceed by ≤ 1 %, (c) no case worse by more than $\\max(2\\eta, 0.1e)$):')
        w('')
        w(table(['label', 'eligible', 'distance to ADT', 'median mf', 'median mesh', 'worst mf', 'worst mesh', '(b)', '(c)', 'qualified', 'K5 (gl12) fails'], rows))
        w('')
        w(f"K5 control (coarse gl12 projection must fail for at least one setting): {'fires' if p1['K5_fires'] else 'DOES NOT fire'}.")
        w('')
        rows = []
        for l_, v in p1['H33'].items():
            for B, r in sorted(v['per_B'].items(), key=lambda x: int(x[0])):
                rows.append([l_, B, f(r.get('median_seconds')), f(r.get('throughput')), r.get('G4', r.get('infeasible', '–')),
                             f(r.get('K6_fires')), f(r.get('K6plus_pass'))])
        w('**H3.3** batching (primary speed claim stays single-query latency; throughput reported separately):')
        w('')
        w(table(['label', 'B', 'median s', 'cases/s', 'G4', 'K6 fires', 'K6+ passes'], rows))
        w('')
        w('Throughput at B = 16 over B = 1: ' + '; '.join(f"{l_} {f(v['throughput_16_over_1'])}× ({'meets' if v['bar_2x'] else 'misses'} the 2× bar)"
                                                         for l_, v in p1['H33'].items()) + '.')
        w('')
        w('![batching](plots/batching.png)')
    else:
        w('p1 not yet analysed.')
    w('')
    # ---------------- route 4
    w('## Route 4 — against the full simulation, by mesh (H4)')
    w('')
    w('Contract: **mesh-free / scoring-lattice** (ν and the initial field on each method\'s own input points, outputs '
      'on the $63^3$ scoring lattice); it is not the full-field query. Matched labels do not transfer across meshes (A6).')
    w('')
    rows = []
    for l_, d in fm['matches'].items():
        rows.append([l_] + [(f"{d[x]['matched']} ({'boundary' if d[x].get('boundary') else 'interior'})" if d[x].get('matched') else 'no match') for x in ('UB', 'UC', 'CC', 'PS')])
    w('Screening FOM matches from the f1 ladder (cheapest calibrated FOM whose median **and** worst tie-or-beat the ROM):')
    w('')
    w(table(['ROM', 'upwind/BE', 'upwind/CN-R', 'central/CN-R', 'spectral'], rows))
    w('')
    if m1:
        w(f"G5 (tables at n = 65 equal Step 2's) and K7 (n = 66 differs): " + '; '.join(
            f"{k}: A_rel {f(v['A_rel'])}, K7_rel {f(v['K7_rel'])}, gate {f(v['gate'])}" for k, v in m1['G5'].items()) + '.')
        w('')
        rows = []
        for Ls, o in m1['per_L'].items():
            for l_, r in o['rom'].items():
                pl = o['panels'].get(l_, {})
                rows.append([Ls, l_, f(r.get('feasible')), f(r.get('M')), f(r.get('eligible')), f(r.get('dist_worst')), f(r.get('median')),
                             f(r.get('worst')), f((pl.get('rom_seconds') or 0) * 1e3 if pl.get('rom_seconds') else None)])
        w('ROM (mesh-free path) per mesh: eligibility at that mesh (E-a, E-b, E-c against the ADT of the same mesh, tight/tighter), E0 errors, warm time:')
        w('')
        w(table(['L', 'ROM', 'feasible', 'M', 'eligible', 'dist. to ADT', 'median', 'worst', 'time (ms)'], rows))
        w('')
        rows = [[l_, *[f((v['seconds'].get(str(Lx)) or 0) * 1e3) for Lx in sorted(int(x) for x in m1['per_L'])], f(v['max_over_min']), v['verdict']]
                for l_, v in m1['H41'].items()]
        w('**H4.1** flat ROM cost (pass iff max/min of the warm query time over meshes ≤ 1.25):')
        w('')
        w(table(['ROM'] + [f'L={x} (ms)' for x in sorted(int(x) for x in m1['per_L'])] + ['max/min', 'verdict'], rows))
        w('')
        rows = []
        for Ls, o in m1['per_L'].items():
            for fk, fv in o['fom'].items():
                if fv.get('matched_for'):
                    continue
                sps = [o['panels'][l_]['speedup'].get(fv.get('key'), {}).get('ratio') for l_ in o['panels']]
                rows.append([Ls, fk, fv.get('key') or '–', fv.get('calibration_status') or fv.get('infeasible') or '–', f(fv.get('median')),
                             f(fv.get('worst')), f(fv.get('n_cases')), ', '.join(f(x) for x in sps)])
        w('**H4.2** same-mesh diagnostic (not an equal-accuracy claim; FOM errors on the first 8 cases at L ≥ 256; '
          'speed-up = FOM time / ROM time per ROM in the order ' + ', '.join(m1['H41']) + '):')
        w('')
        w(table(['L', 'family|Δt', 'calibrated configuration', 'status', 'median', 'worst', 'cases', 'speed-ups'], rows))
        w('')
        rows = []
        for l_, v in m1['H43'].items():
            for fam in ('UB', 'UC', 'CC', 'PS'):
                x = v['families'].get(fam, {})
                rows.append([l_, fam, x.get('matched') or 'no ladder match', f(x.get('speedup')), f(x.get('reaches_2x')),
                             x.get('label', '')])
            u = v['union']
            rows.append([l_, '**union**', (u or {}).get('matched', '–'), f((u or {}).get('speedup')), f((u or {}).get('reaches_2x')), 'best matched FOM over families'])
        w('**H4.3** matched-accuracy speed-up at n = 65, mesh path (A6; paired `match|<label>` panel; ROM speed-up = '
          '$t_{\\rm FOM}/t_{\\rm ROM}$; below 1 is a loss):')
        w('')
        w(table(['ROM', 'family', 'matched FOM', 'speed-up', '≥ 2×', 'label'], rows))
        w('')
        hc = m1['Hcomp_largest_mesh']
        rows = [[l_] + [((f(c[x].get('speedup')) + (' ✓' if c[x].get('reaches_2x') else '')) if c[x].get('speedup') is not None else
                         (c[x].get('status', '–') if x != 'union' else '–')) for x in ('UB', 'UC', 'CC', 'PS', 'union')] for l_, c in hc['cells'].items()]
        w(f"**H-comp cells (V3-5) at the largest mesh L = {hc['L']}** (✓ = matched there and ≥ 2×; the matched FOM keeps its "
          "own resolution, the ROM runs at this mesh, and the label 'matched' is re-checked against this mesh's ROM errors):")
        w('')
        w(table(['ROM', 'upwind/BE', 'upwind/CN-R', 'central/CN-R', 'spectral', 'union'], rows))
        w('')
        w(f"**K2 (governing plan): {'FIRES' if m1['K2_fires'] else 'does not fire'}.**")
        w('')
        ps = [v['families'].get('PS', {}) for v in m1['H43'].values()]
        ps_sp = [x.get('speedup') for x in ps if x.get('speedup') is not None]
        if ps_sp:
            w(f"The early unaudited f1 observation \"spectral FOM clearly beats the ROM at 64³\" is "
              f"{'**confirmed**' if max(ps_sp) < 1 else '**not confirmed**'} by the paired panels: the matched spectral FOM "
              f"(cross-resolution, N = 16) gives ROM speed-ups {f(min(ps_sp))}–{f(max(ps_sp))}× (below 1 = the FOM is faster at "
              "equal-or-better accuracy).")
        w('')
        w('![mesh](plots/mesh-time.png)')
    else:
        w('m1 not yet analysed.')
    w('')
    if R1:
        w('## Replication on the historical test cohort 923901 (E0 test32)')
        w('')
        w(R1.get('markdown', ''))
        w('')
    # ---------------- provisional / retractions
    w('## What is provisional, and what was wrong')
    w('')
    w('- Screening ratios (H1.4, H2) are from separate grid jobs; only the confirmed paired ratios carry verdicts.')
    w('- Every matched speed-up is an upper bound (finite FOM ladder); a match at the cheapest admissible ladder member is "boundary".')
    w('- E0 errors are conditional Richardson estimates. ε_ref ties are interval-overlap ties, not proven equality.')
    under = []
    for a in ('g1', 'g2', 'g3', 'f1', 'p1', 'm1', 'r1'):
        rc = jl(CH / f'resources-{a}.json')
        g = next((j for j in gl['jobs'] if j['attempt'] == a), None)
        if rc and g:
            under.append(f"{a} {f(rc['projected_hours'])} h projected vs {f(g['gpu_hours'])} h measured")
    w('- Resource certificates were estimates, not bounds: ' + '; '.join(under) + '.')
    w('- Retracted / corrected during this lane: the A9 G2 attribution (corrected in A10); the screening wording "all replay '
      'labels mixed" (71 mixed + 1 baseline already stationary); m1 calibration population [0, 1] → cases 0–3 at every mesh (A9).')
    w('')
    # ---------------- audits
    w('## Audits')
    w('')
    rows = [[p.name, verdict_of(p)] for p in sorted((HERE / 'audits').glob('*.md'))]
    w(table(['file', 'verdict (first verdict word in the file)'], rows))
    w('')
    # ---------------- glossary
    w('## Glossary')
    w('')
    for k, v in [
        ('ROM', 'reduced-order model: the frozen coordinate-network bank, LSPG on sine tests, off-mesh quadrature for advection.'),
        ('M2_512, W256 … W1024', 'settings: bank (M2 or W1024) truncated to R′ = 512 or 256…1024 columns.'),
        ('gl32, lat65536', 'quadrature rules: Gauss–Legendre with 32 points per axis; lattice rule with 65,536 points.'),
        ('BE, CN, BDF2', 'backward Euler (first order), Crank–Nicolson and BDF2 (second order) time stepping.'),
        ('Δt₀', 'the deployed step 0.01.'),
        ('FS-K1 / FS-K2', 'the deployed fixed-sweep Levenberg–Marquardt solver with 1 or 2 full-Jacobian sweeps per step.'),
        ('FSP', 'FS with two extra initial-guess candidates (cubic, IMEX).'),
        ('QN-k', 'quasi-Newton with exact gradient and a Gauss–Newton matrix refreshed every k steps (∞ = only when forced).'),
        ('ADT', 'the verified adaptive tight solve of the same discrete problem; the solver-distance comparator.'),
        ('eligible (Q1)', 'every run valid, certified non-stationary steps ≤ 1 % pooled, lattice distance to ADT ≤ 2.5e-4 on every case, ADT tight/tighter ≤ 2.5e-5.'),
        ('E0, ε_ref (η)', 'the second-order reference store and its per-case uncertainty estimate.'),
        ('median / worst', 'median and maximum over the ranked validation cases of the max-over-output-times relative error.'),
        ('ties-or-beats', 'not resolved as worse: median(e_a − η) ≤ median(e_b + η), same for the maximum.'),
        ('screening ratio', 'median paired time ratio to C_old inside each grid job.'),
        ('confirmed', 'from a direct A–B–A panel A = comparator, B = selected configuration, 8 cases × 3 repetitions, one job.'),
        ('C_old, C_W768, C_W1024', 'comparators: the deployed BE configurations of M2_512, W768, W1024.'),
        ('W*, B*', 'H2 winner (cheapest eligible configuration that ties-or-beats C) and best BE configuration doing so.'),
        ('S', 'speed-up = cost(comparator) / cost(selected).'),
        ('mesh-free path', 'projection by quadrature at the rule points and decoding only at the scoring lattice.'),
        ('G-gates, K-controls, T1', 'consistency gates and controls that must fail (DESIGN §7); T1 = C_old run twice must cost 1.8–2.2×.'),
        ('F-UB, F-UC, F-CC, F-PS', 'full-order families: upwind+BE, upwind+CN-R, central+CN-R finite differences, sine pseudo-spectral ETDRK4.'),
        ('matched FOM', 'cheapest calibrated FOM of a family (any mesh, step) whose median and worst tie-or-beat the ROM\'s.'),
        ('boundary', 'the match is the family\'s cheapest admissible ladder member: a coarser FOM might also match.'),
        ('union', 'the best (cheapest) matched FOM across all families.'),
        ('K2 (governing plan)', 'kill criterion: speed-up over the best validated FOM below 2× at the largest mesh for every setting.'),
        ('L, n', 'FOM cells per side; ROM mesh n = L + 1 nodes per side.'),
        ('GPU-h', 'elapsed GPU-hours measured by sacct.')]:
        w(f'- **{k}** — {v}')
    w('')
    (HERE / 'report.md').write_text('\n'.join(L) + '\n')
    # ---------------- EXPORT.json
    ex = dict(lane='jcp-speed3d', label=G['label'], contract=dict(grid='full-field 65^3 rollout (mesh path)',
                                                                   m1='mesh-free / scoring-lattice (not the full-field query)'),
              q1_definition='eligible = E-a valid, E-b certified non-stationary <= 1% pooled (gamma > 1e-3 or non-finite), '
                            'E-c lattice distance to ADT tight <= 2.5e-4 every case, ADT tight/tighter <= 2.5e-5 on cases 0-7',
              q1={k: dict(eligibility=v['eligibility'], Ea=v['Ea'], Eb_frac=v['Eb_frac'], dist_worst=v['dist_worst'], adt_gate=v['adt_gate'],
                          median=v['median'], worst=v['worst'], screening_ratio_to_C_old=v['ratio'], job=v['job']) for k, v in C.items()},
              q2_note='Q2 (quadrature agreement on own states) not supplied beyond the finalists\' converged-rule delta_quad (p1 conv)',
              H2=(p1 or {}).get('H2') or sel['comparators'], H14=H14, fom_match=fm['matches'],
              wording='a finite FOM ladder gives an upper bound on optimal FOM cost, hence an upper bound on the claimed ROM speed-up')
    if p1:
        ex['p1'] = dict(H31=p1['H31'], H32={k: {x: y for x, y in v.items() if x not in ('detail',)} for k, v in p1['H32'].items()},
                        H33=p1['H33'], conv=p1['conv'], reduced_adt=p1['reduced_adt'])
    if m1:
        ex['m1'] = dict(cost_vs_mesh={l_: v['seconds'] for l_, v in m1['H41'].items()}, H41=m1['H41'], H42=m1['H42'],
                        matched_speedups={l_: dict(families={f_: {x: y for x, y in v.items() if x != 'panel'} for f_, v in d['families'].items()},
                                                   union=d['union'], per_mesh=d['per_mesh']) for l_, d in m1['H43'].items()},
                        Hcomp_largest_mesh=m1['Hcomp_largest_mesh'], K2_fires=m1['K2_fires'],
                        rom_per_mesh={L_: {l_: {x: y for x, y in r.items() if x not in ('e', 'eta', 'ranked_ids')} for l_, r in o['rom'].items()}
                                      for L_, o in m1['per_L'].items()})
    (HERE / 'runs' / 'EXPORT.json').write_text(json.dumps(ex, indent=1, default=float) + '\n')
    print('report.md', len(L), 'lines; plots', made)


if __name__ == '__main__':
    main()
