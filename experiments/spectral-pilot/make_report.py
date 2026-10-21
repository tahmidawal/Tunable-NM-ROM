"""spectral-pilot: generate report.md and plots from the run JSONs and runs/ANALYSIS.json. Every measured number in
report.md comes from a JSON through this script. PILOT: small, not a result.

  python analyze.py && python make_report.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import common as C
import score as SC

H = Path(__file__).resolve().parent
R = H / 'runs'
PL = H / 'plots'
PL.mkdir(exist_ok=True)
A = json.loads((R / 'ANALYSIS.json').read_text())
LV = [lv for lv in ('A1', 'A2', 'A3', 'A4') if A['A'].get(lv, {}).get('status') == 'run']
LVX = LV + (['AB'] if A['A'].get('AB', {}).get('status') == 'run' else [])
TAUS = [str(t) for t in SC.TARGETS]
PILOT = '**PILOT: small, not a result.**'


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def g(x, f='%.3g'):
    if x is None:
        return '–'
    if isinstance(x, float) and not math.isfinite(x):
        return '∞'
    return f % x if isinstance(x, (int, float)) else str(x)


def ms(x):
    return '–' if x is None else '%.3g ms' % (1e3 * x)


def tau_s(t):
    return '%g' % float(t)


# ------------------------------------------------------------------------------------------------- plots -----

def plot_floors():
    fig, axs = plt.subplots(1, len(LVX), figsize=(4.2 * len(LVX), 3.6), sharey=True)
    for ax, lv in zip(np.atleast_1d(axs), LVX):
        rom = load(R / 'A' / lv / 'rom.json')
        fl = rom['floors']
        r = np.arange(1, len(fl['pod']['worst']) + 1)
        ax.loglog(r, fl['pod']['worst'], label='POD span (96 traj.)')
        ax.loglog(np.arange(1, len(fl['pod48']['worst']) + 1), fl['pod48']['worst'], '--', label='POD span (48 traj.)')
        rs = np.arange(1, len(fl['sine']['worst']) + 1)
        ax.loglog(rs, fl['sine']['worst'], ':', label='first r sines (Laplacian order)')
        Ns = sorted(int(k) for k in fl['spectral'])
        ax.loglog([(N - 1) ** 2 for N in Ns], [fl['spectral'][str(N)]['worst'] for N in Ns], 'k.-', ms=3,
                  label='sine block $(N-1)^2$ (spectral FOM space)')
        ax.set_title(f'{lv}: median s = {A["A"][lv]["s_median"]:.0f}')
        ax.set_xlabel('number of unknowns')
        ax.set_xlim(4, 3e4)
        ax.set_ylim(1e-6, 1)
        ax.grid(alpha=.3)
    np.atleast_1d(axs)[0].set_ylabel('worst projection floor (12 test cases)')
    np.atleast_1d(axs)[0].legend(fontsize=7)
    fig.suptitle('Representation floors (PILOT: small, not a result)', fontsize=10)
    fig.tight_layout()
    fig.savefig(PL / 'floors.png', dpi=130)
    plt.close(fig)


def plot_pareto():
    fig, axs = plt.subplots(1, len(LVX) + 1, figsize=(4.2 * (len(LVX) + 1), 3.6))
    for ax, lv in zip(axs, LVX):
        ts = load(R / 'A' / lv / 'time_screen.json')
        if not ts or 'targets' not in ts:
            continue
        F, Rr = {}, {}
        for t, e in ts['targets'].items():
            for x in e['fom_timed']:
                F[(x['N'], x['n'])] = (x['t'], x['worst'])
            for x in e['rom_timed']:
                Rr[(x['form'], x['r'], x['Mq'], x['n'], x['Ninit'])] = (x['t'], x['worst'])
        if F:
            ax.loglog([v[0] * 1e3 for v in F.values()], [v[1] for v in F.values()], 'o', label='spectral FOM')
        if Rr:
            ax.loglog([v[0] * 1e3 for v in Rr.values()], [v[1] for v in Rr.values()], 's', mfc='none',
                      label='POD-Galerkin ROM')
        ax.set_title(lv)
        ax.set_xlabel('query time, 1 core [ms]')
        ax.grid(alpha=.3)
    axs[0].set_ylabel('worst error (12 cases)')
    axs[0].legend(fontsize=7)
    ax = axs[-1]
    ts = load(R / 'B' / 'time_screen.json')
    if ts and 'targets' in ts:
        F, Rr = {}, {}
        for t, e in ts['targets'].items():
            for k, x in e['fom'].items():
                F[k] = (x['t'], x['worst'], x['fam'])
            for k, x in e['rom'].items():
                Rr[k] = (x['t'], x['worst'])
        for fam, mk in (('IMEX', 'o'), ('RK4', '^')):
            v = [x for x in F.values() if x[2] == fam]
            if v:
                ax.loglog([x[0] * 1e3 for x in v], [x[1] for x in v], mk, label=f'FD {fam}')
        if Rr:
            ax.loglog([v[0] * 1e3 for v in Rr.values()], [v[1] for v in Rr.values()], 's', mfc='none',
                      label='POD-Galerkin ROM')
        ax.legend(fontsize=7)
    ax.set_title('B (L-shape)')
    ax.set_xlabel('query time, 1 core [ms]')
    ax.grid(alpha=.3)
    fig.suptitle('Timed candidates that meet at least one target (PILOT: small, not a result)', fontsize=10)
    fig.tight_layout()
    fig.savefig(PL / 'pareto.png', dpi=130)
    plt.close(fig)


def plot_sizes():
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    s = [A['A'][lv]['s_median'] for lv in LV]
    for t, mk in zip(TAUS, 'osd^'):
        Nm = [A['A'][lv]['targets'][t]['N_min'] for lv in LV]
        rm = [A['A'][lv]['targets'][t]['r_min'] for lv in LV]
        axs[0].loglog(s, [(x - 1) ** 2 if x else np.nan for x in Nm], mk + '-', label=f'FOM $(N-1)^2$, τ={tau_s(t)}')
        axs[0].loglog(s, [x if x else np.nan for x in rm], mk + '--', mfc='none', label=f'ROM r, τ={tau_s(t)}')
        S = [A['A'][lv]['targets'][t].get('S') for lv in LV]
        axs[1].loglog(s, [x if x else np.nan for x in S], mk + '-', label=f'τ={tau_s(t)}')
    axs[0].set_xlabel('median steepness s = a/ν')
    axs[0].set_ylabel('smallest feasible number of unknowns')
    axs[0].legend(fontsize=6)
    axs[0].grid(alpha=.3)
    axs[1].axhline(1, color='k', lw=.8)
    axs[1].axhline(2, color='k', lw=.5, ls=':')
    axs[1].set_xlabel('median steepness s = a/ν')
    axs[1].set_ylabel('S = t_FOM / t_ROM (matched)')
    axs[1].legend(fontsize=7)
    axs[1].grid(alpha=.3)
    fig.suptitle('Part A sweep (PILOT: small, not a result)', fontsize=10)
    fig.tight_layout()
    fig.savefig(PL / 'sweep.png', dpi=130)
    plt.close(fig)


def plot_cost():
    fig, ax = plt.subplots(figsize=(5, 3.6))
    lv = LV[-1]
    cm = A['A'][lv].get('cost_model')
    if cm:
        N = sorted(int(k) for k in cm['fom'])
        ax.loglog([(n - 1) ** 2 for n in N], [cm['fom'][str(n)] * 1e3 for n in N], 'o-',
                  label='spectral FOM: one nonlinear evaluation vs $(N-1)^2$')
        r = sorted(int(k) for k in cm['rom_t'])
        ax.loglog(r, [cm['rom_t'][str(x)] * 1e3 for x in r], 's--', label='ROM packed tensor vs r')
    ax.set_xlabel('number of unknowns')
    ax.set_ylabel('time per nonlinear evaluation, 1 core [ms]')
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    ax.set_title(f'Cost per evaluation ({lv} cost model; PILOT)', fontsize=9)
    fig.tight_layout()
    fig.savefig(PL / 'cost.png', dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ report -----

def main():
    for f in (plot_floors, plot_pareto, plot_sizes, plot_cost):
        try:
            f()
        except Exception as ex:  # noqa: BLE001
            print('plot failed', f.__name__, ex)
    L = []
    w = L.append
    w('# Spectral pilot: can the reduced-order model beat a spectral or best-practice solver?\n')
    w(f'**Status (2026-10-21): {PILOT[2:-2]}** Small local CPU pilot (lane `exp/2026-10-20-spectral-pilot`), run '
      'to find direction and feasibility for a later cluster lane. Every number below is generated by '
      '`make_report.py` from the run JSONs in `runs/` and labelled PILOT. The reduced model tested is a '
      '**POD–Galerkin feasibility comparator** (POD span + exact quadratic tensor or grid rule + ETDRK4), not the '
      'coordinate-network bank: a loss means this implementation loses, a win only licenses the full experiment '
      '(DESIGN A1.1). Errors are against numerical references (conditional allowances, DESIGN A2.5/A3.4). '
      'Design: `DESIGN.md` (amended exploratory design, amendments A1–A6; A6 = post-hoc control AB); audits in `audits/`.\n')

    # ---- in brief (filled after analysis)
    w('## 1. In brief\n')
    brief = brief_lines()
    for b in brief:
        w(f'- {b}')
    w('')

    # ---- what ran
    w('## 2. What ran\n')
    w('| level | ν band | test steepness s (median, range) | $N_{\\rm ref}$ | allowance U | front sharpness t>0 (t=0) | remaining energy at t=0.25 | gates |')
    w('|---|---|---|---|---|---|---|---|')
    for lv in LVX:
        e = A['A'][lv]
        ref = load(R / 'A' / lv / 'ref.json')
        dg = e['diag'] or {}
        gates = 'all pass' if (e['U'] is not None and e['controls']['G4'] and e['controls']['SW4'] and e['controls']['G7']) else 'FAILED (unresolved)'
        w(f"| {lv} | [{g(ref['nu'][0], '%.4g')}, {g(ref['nu'][1], '%.4g')}] | {g(e['s_median'], '%.0f')} "
          f"({g(e['s_range'][0], '%.0f')}–{g(e['s_range'][1], '%.0f')}) | {e['Nref']} | {g(e['U'], '%.1e')} | "
          f"{g(dg.get('sharp'), '%.1f')} ({g(dg.get('sharp_t0'), '%.1f')}) | {g(dg.get('energy'), '%.2f')} | {gates} |")
    w(f"\nTable: PILOT. A4 (ν band [0.0022, 0.0044]) was **not run (time cap)**: rule A2.6 allows A4 only if all of "
      "A3's stages finished; A3's first ROM job hit its 30-minute unit cap (its r = 512 part was re-run as a "
      "separate unit). A3's reference generation took "
      f"{g(load(R / 'A' / 'A3' / 'ref.json')['seconds'] / 60, '%.0f')} minutes on the shared CPU. AB is the post-hoc "
      "control of A6. Gates per level: reference space/time/padding differences, SW1a/SW1b (matcher), SW2 "
      "(reference gate), G4 (tensor ROM = direct-evaluation ROM), SW4 (tensor-slice corruption), G7 (IC rule), D2 "
      "(no run below its floor). Front sharpness is $\\max|\\nabla u|/\\max|u|$, median over test cases of the max "
      "over $t_1..t_5$.\n")
    b = A.get('B')
    if b:
        o = b['order']['lat64']
        w(f"Part B (L-shape, ν ∈ [0.02, 0.05]): reference = SBDF2 + SuperLU at L = 1024; observed orders "
          f"p₁ = {g(o['p1'], '%.2f')}, p₂ = {g(o['p2'], '%.2f')} (gate {'passed' if b['order_gate'] else 'failed'}); "
          f"Richardson reference-error estimate {g(b['e_ref'], '%.1e')}, time-gate difference {g(b['delta_t'], '%.1e')}, "
          f"allowance U = {g(b['U'], '%.1e')} (conditional). The ROM's own operator (FD at L = 512) differs from the "
          f"reference by {g(b['fd512']['worst'], '%.1e')} (worst), a measured discretisation discrepancy that may "
          "limit the ROM's accuracy (PILOT).\n")

    # ---- Part A table
    w('## 3. Part A: the box, paired viscosity sweep\n')
    w('Table (PILOT): smallest observed feasible sizes (any step count meets τ under the three-way rule), '
      'representation floors (smallest size whose projection floor ≤ τ), and the screen-selected fastest among '
      'evaluated candidates, re-timed in a fresh process (one-repetition pruning, A5). "cens." = not reached inside '
      'the ladder (censored); "–" = not applicable / not timed. ROM matched lists form, r, $M_q$ (grid size, ROM-Q), '
      'steps and $N_{\\rm init}$.\n')
    w('| level | τ (worst) | FOM $N_{\\min}$ [$(N-1)^2$] | ROM $r_{\\min}$ | sine-ROM $r_{\\min}$ | floor: block $N$ | floor: POD r (96 / 48 traj.) | FOM matched (N, steps, time) | ROM matched (form, r, $M_q$, steps, $N_{\\rm init}$, time) | S | S native |')
    w('|---|---|---|---|---|---|---|---|---|---|---|')
    for lv in LVX:
        for t in TAUS:
            x = A['A'][lv]['targets'][t]
            Nm = x['N_min']
            fo, ro = x.get('fom'), x.get('rom')
            cz = lambda y: 'cens.' if y is None else g(y)
            w(f"| {lv} | {tau_s(t)} | {cz(Nm)} [{g((Nm - 1) ** 2 if Nm else None)}] | {cz(x['r_min'])} | "
              f"{cz(x.get('r_min_sine'))} | {cz(x['N_floor'])} | {cz(x['r_floor'])} / {cz(x['r_floor48'])} | "
              f"{(str(fo['N']) + ', ' + str(fo['n']) + ', ' + ms(fo['t'])) if fo else '–'} | "
              f"{(ro['form'] + ', ' + str(ro['r']) + ', ' + g(ro['Mq']) + ', ' + str(ro['n']) + ', ' + str(ro['Ninit']) + ', ' + ms(ro['t'])) if ro else '–'} | "
              f"{g(x.get('S'), '%.3g')} | {g(x.get('native_S'), '%.3g')} |")
    w('\n![floors](plots/floors.png)\n\n![sweep](plots/sweep.png)\n\n![pareto](plots/pareto.png)\n')

    w('### 3.1 Verdicts per target (DESIGN A1.3, A2.1, A3.1; PILOT)\n')
    w('| τ | eligible levels | H-A1 α (N ∝ s^α) | H-A2: $r_{\\min}/(N_{\\min}-1)^2$ | H-A2′: $r_{\\min}/r_\\times(N_{\\min})$ (cheaper form; tensor only) | H-A3 category; S per level | slope of log S |')
    w('|---|---|---|---|---|---|---|')
    for t in TAUS:
        v = A['verdicts'][t]
        if 'H-A1' not in v:
            w(f"| {tau_s(t)} | {', '.join(v['eligible']) or 'none'} | {v.get('verdict', '')} | | | | |")
            continue
        a2p = v['H-A2p']
        w(f"| {tau_s(t)} | {', '.join(v['eligible'])} | {g(v['H-A1']['alpha'], '%.2f')} ({v['H-A1']['verdict']}) | "
          f"{', '.join(g(q, '%.2g') for q in v['H-A2']['ratio'])} ({v['H-A2']['verdict']}) | "
          f"{(', '.join(g(q, '%.2g') for q in a2p['ratio']) + ' (' + a2p['verdict'] + '); ' + ', '.join(g(q, '%.2g') for q in v.get('H-A2p_tensor_only', []))) if 'ratio' in a2p else a2p['verdict']} | "
          f"{v['H-A3']['category']}; {', '.join(g(q, '%.3g') for q in v['H-A3']['S'])} | {g(v['H-A3']['slope'], '%.2f')} |")
    w('\nA target is *eligible* at a level only if both methods reached it and the timing was valid; a verdict needs '
      'three eligible levels. Censored cells (a method never reached τ) are not evidence for either side.\n')

    # cost model
    w('### 3.2 Cost per nonlinear evaluation (one core; PILOT)\n')
    cm = A['A'][LV[-1]].get('cost_model')
    if cm:
        w(f'From the {LV[-1]} screen (median of 20 calls):\n')
        w('| spectral FOM N | time | | ROM tensor r | time |')
        w('|---|---|---|---|---|')
        Ns = sorted(int(k) for k in cm['fom'])
        rs = sorted(int(k) for k in cm['rom_t'])
        for i in range(max(len(Ns), len(rs))):
            a_ = f"{Ns[i]} | {ms(cm['fom'][str(Ns[i])])}" if i < len(Ns) else ' | '
            b_ = f"{rs[i]} | {ms(cm['rom_t'][str(rs[i])])}" if i < len(rs) else ' | '
            w(f'| {a_} | | {b_} |')
        w('\n![cost](plots/cost.png)\n')
        rx = [(lv, t, A['A'][lv]['targets'][t].get('r_cross')) for lv in LV for t in TAUS]
        rx = [(lv, t, x) for lv, t, x in rx if x and x.get('status') == 'ok']
        if rx:
            w('Equal-cost reduced size $r_\\times(N_{\\min})$ (the cheaper measured ROM evaluation form per r, '
              'unrestricted by its accuracy, as expensive as one FOM evaluation at the FOM\'s smallest feasible N; '
              'PILOT): ' +
              '; '.join(f"{lv} τ={tau_s(t)}: N={A['A'][lv]['targets'][t]['N_min']} → r× ≈ {g(x['value'], '%.0f')}"
                        for lv, t, x in rx) + '.\n')

    # ---- Part B
    w('## 4. Part B: L-shaped domain (PILOT)\n')
    if b:
        w('| τ (worst) | F-IMEX $L_{\\min}$ | F-RK4 $L_{\\min}$ | ROM $r_{\\min}$ | FD matched (family, L, steps, time) | ROM matched (r, steps, L_init, time) | S | reading |')
        w('|---|---|---|---|---|---|---|---|')
        for t in TAUS:
            x = b['targets'][t]
            fo, ro = x.get('fom'), x.get('rom')
            w(f"| {tau_s(t)} | {g(x.get('L_min_IMEX'))} | {g(x.get('L_min_RK4'))} | {g(x.get('r_min'))} | "
              f"{(fo['fam'] + ', ' + str(fo['L']) + ', ' + str(fo['n']) + ', ' + ms(fo['t'])) if fo else '–'} | "
              f"{(str(ro['r']) + ', ' + str(ro['n']) + ', ' + str(ro['Linit']) + ', ' + ms(ro['t'])) if ro else '–'} | "
              f"{g(x.get('S'), '%.3g')} | {x.get('verdict', 'censored or not timed')} |")
        w('\nThe claim is scoped to these two FD implementations (A1.8): multigrid, stabilised explicit and '
          'high-order/spectral-element competitors were not run. The B basis qualification is "unestablished" '
          '(A3.4).\n')

    # ---- controls
    w('## 5. Controls and timing quality (PILOT)\n')
    w('| check | ' + ' | '.join(LVX) + ' | B |')
    w('|---|' + '---|' * (len(LVX) + 1))
    rows = [('SW1a error metric', lambda e: e['SW1a'], lambda: b and b['SW1']),
            ('SW1b three-way matcher', lambda e: e['SW1b'], lambda: b and b['SW1']),
            ('SW2 reference gate rejects a perturbed reference', lambda e: e['SW2'],
             lambda: b and (b.get('allowance') or {}).get('SW2_fired')),
            ('G4 tensor ROM = direct ROM (r=32, 128)', lambda e: e['controls']['G4'], lambda: b and b['controls_ok']),
            ('SW4 corrupted tensor slice detected', lambda e: e['controls']['SW4'], lambda: b and b['controls_ok']),
            ('G7 IC rule', lambda e: e['controls']['G7'], lambda: 'n/a (nodal IC)'),
            ('timing valid (screen + final sentinel)', lambda e: e['timing_valid'], lambda: b and b['timing_valid'])]
    for name, fa, fb in rows:
        w(f"| {name} | " + ' | '.join(str(fa(A['A'][lv])) for lv in LVX) + f" | {fb()} |")
    w('\nPhysical diagnostics (not asserted): coarse "references" ' + ', '.join(
        f"{lv}: N_ref/4 differs by {g(load(R / 'A' / lv / 'ref.json')['checks']['diag_coarse']['max'], '%.1e')}"
        for lv in LV) + '; random 128-dimensional span floors ' + ', '.join(
        f"{lv}: {g(load(R / 'A' / lv / 'rom.json')['floors']['random']['worst'][127], '%.2f')}" for lv in LV) + '.\n')

    w('| level | τ | FOM spread (max/min) | FOM outliers | ROM spread | ROM outliers |')
    w('|---|---|---|---|---|---|')
    for lv in LVX:
        tf = load(R / 'A' / lv / 'time_final.json')
        for t, x in (tf or {}).get('targets', {}).items():
            if x.get('S') is not None:
                w(f"| {lv} | {tau_s(t)} | {g(x['fom']['spread'], '%.2f')} | {x['fom']['outliers']} | "
                  f"{g(x['rom']['spread'], '%.2f')} | {x['rom']['outliers']} |")
    tf = load(R / 'B' / 'time_final.json')
    for t, x in (tf or {}).get('targets', {}).items():
        if x.get('S') is not None:
            w(f"| B | {tau_s(t)} | {g(x['fom']['spread'], '%.2f')} | {x['fom']['outliers']} | "
              f"{g(x['rom']['spread'], '%.2f')} | {x['rom']['outliers']} |")
    w('\nSpread = max/min over the 15 repetitions (3 cases × 5); outliers = repetitions more than 20 % above the '
      'median.\n')
    tail(w)
    (H / 'report.md').write_text('\n'.join(L) + '\n')
    print('report written')


def brief_lines():
    out = []
    for t in TAUS:
        v = A['verdicts'][t]
        if 'H-A3' in v:
            out.append(f"Part A (box, wide family, this POD–Galerkin ROM), τ = {tau_s(t)}: {v['H-A3']['category']} (S = "
                       f"{', '.join(g(q, '%.2g') for q in v['H-A3']['S'])} at {', '.join(v['eligible'])}; PILOT).")
    for lv in LV:
        cens = [tau_s(t) for t in TAUS if A['A'][lv]['targets'][t]['r_min'] is None]
        if cens:
            out.append(f"{lv}: the ROM never reached τ ∈ {{{', '.join(cens)}}} with r ≤ "
                       f"{max(int(k) for k in load(R / 'A' / lv / 'rom.json')['rom_t'])} (censored; PILOT).")
    b = A.get('B')
    if b:
        for t in TAUS:
            x = b['targets'][t]
            if x.get('S') is not None:
                out.append(f"Part B (L-shape vs these second-order FD solvers), τ = {tau_s(t)}: {x['verdict']} (S = {g(x['S'], '%.2g')}; PILOT).")
    ab = A['A'].get('AB')
    if ab and ab.get('status') == 'run':
        out.append('Post-hoc control AB (Part-B cases on the box vs the spectral FOM): S = ' + ', '.join(
            f"{g(ab['targets'][t].get('S'), '%.2g')} (τ={tau_s(t)})" for t in TAUS) + '; PILOT.')
    return out


from report_text import tail  # noqa: E402


if __name__ == '__main__':
    main()
