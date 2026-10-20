"""Generate report.md, plots and runs/EXPORT.json from the frozen verdict files (never hand-typed numbers).

Inputs (configs/report.json): per cell {cell, d, L, tune_attempt, stageb_attempt[, tune_json]}, plus the text-only
records of cancelled attempts, FRONT-2D membership and deferred scope; reads the tuning JSON of each cell,
checks/VERDICT-<cell>.json (+ .sha256), runs/GPU-LEDGER.json, inputs/ours_V.json and inputs/ours_T.json.

Dimension-level budget wording (A3.1, A4.3, A19.3): baseline V = sum over the dimension's cells of the cumulative V
(every attempt, A44.2); ratio = baseline V / ours' V. The lower ratio uses the recorded-phase lower bound, the upper
ratio the allocation upper bound. Both in [0.8, 1.25] -> "within the registered factor-1.25 band"; both on one side
outside -> "unequal-budget, descriptive"; otherwise "unresolved". The resulting qualification is attached to every
pair label of that dimension in the report and the export (A3.1).

Usage: python make_report.py
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DT0 = {2: 0.005, 3: 0.01}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fmt(x, nd=3):
    if x is None:
        return 'n/a'
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return f'{int(x):,}'
    x = float(x)
    if not np.isfinite(x):
        return 'non-finite'
    return f'{x:.{nd}g}'


def cid(s):
    """A configuration id inside a Markdown table cell: '|' escaped (GFM splits cells on it even inside backticks)."""
    return '`' + str(s).replace('|', '\\|') + '`'


def ge(flag):
    return '≥ ' if flag else '= '


def band(lo, hi):
    """A4.3 wording on the baseline/ours V ratio bounds [lo, hi]."""
    if lo is None:
        return 'V ratio unavailable', 'unavailable'
    rng = f'ratio in [{lo:.3g}, {hi:.3g}]' if hi is not None else f'ratio ≥ {lo:.3g}'
    if hi is not None and 0.8 <= lo and hi <= 1.25:
        return f'within the registered factor-1.25 band ({rng})', 'within band'
    if (hi is not None and hi < 0.8) or lo > 1.25:
        return f'unequal-budget, descriptive ({rng})', 'unequal-budget, descriptive'
    return f'unresolved against the factor-1.25 band ({rng})', 'budget band unresolved'


def dt_text(dt, d):
    k = round(0.05 / dt)
    r = DT0[d] / dt
    rr = {1.0: '$\\Delta t_0$', 0.8: '$\\Delta t_0\\cdot 5/4$', 0.6: '$\\Delta t_0\\cdot 5/3$', 0.4: '$\\Delta t_0\\cdot 5/2$'}
    return f'$\\Delta t = 1/{20 * k}$' + (f' ({rr[round(r, 6)]})' if round(r, 6) in rr else '')


def parse_b(key):
    kind, r, hrp, scheme, dt, solver, gen = key.split('|')
    return dict(kind=kind, r=int(r), hrp=hrp, scheme=scheme, dt=0.05 / round(0.05 / float(dt)), solver=solver, gen=gen)


def parse_x(key, d):
    p = key.split('|')
    if d == 2:
        return dict(scheme=p[2], dt=DT0[2])
    return dict(scheme=p[3], dt=0.05 / round(0.05 / float(p[4])))


# ------------------------------------------------------------------------------------------------------------ plots
def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


COL = {'none': '#6b7280', 'ecsw': '#2563eb', 'qdeim': '#d97706'}


def front_plot(c, T, V, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=150)
    for kind in ('none', 'ecsw', 'qdeim'):
        ok = [(v['t'], v['W']) for v in T['stageA'].values() if v['kind'] == kind and v['timing_ok']
              and np.isfinite(v['W']) and v['q1']['q1']]
        bad = [(v['t'], v['W']) for v in T['stageA'].values() if v['kind'] == kind and v['timing_ok']
               and np.isfinite(v['W']) and not v['q1']['q1']]
        if ok:
            ax.scatter(*zip(*ok), s=10, alpha=.7, color=COL[kind], label=f'{kind}: Q1 pass')
        if bad:
            ax.scatter(*zip(*bad), s=14, alpha=.6, marker='x', color=COL[kind], linewidths=.7,
                       label=f'{kind}: Q1 FAIL (not ranked)')
    for i, (key, xr) in enumerate((T.get('ours_pass') or {}).items()):
        if xr.get('status') != 'complete':
            continue
        t = float(np.median([cc['seconds_warm'] for cc in xr['cases'].values()]))
        w = max(cc['e0']['e'] for cc in xr['cases'].values())
        diag = V['X'].get(key, {}).get('diagnostic_only')
        ax.scatter([t], [w], marker='*', s=110, zorder=5, color='#f59e0b' if diag else '#dc2626',
                   edgecolors='k', linewidths=.4)
        ax.annotate(f'X{i + 1}', (t, w), fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.scatter([], [], marker='*', s=90, color='#dc2626', edgecolors='k', linewidths=.4,
               label='ours, qualified' + (' (batch 2)' if c['d'] == 3 else ''))
    if c['d'] == 3:
        ax.scatter([], [], marker='*', s=90, color='#f59e0b', edgecolors='k', linewidths=.4,
                   label='ours, batch 1 (DIAGNOSTIC ONLY)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('median warm full-query time, 6 timing cases (s, tuning job)')
    ax.set_ylabel('worst E0 error, eligible screening cases')
    ax.set_title(f"{c['cell']}: Stage-A SCREENING only (not the verdict); stars X1.. keyed in the table below",
                 fontsize=8)
    ax.legend(fontsize=6, loc='best')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pod_plot(c, T, path):
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150)
    for lab, s, col in (('primary (central)', T['basis']['sigma'], '#2563eb'),
                        ('probe (upwind)', (T.get('probe_basis') or {}).get('info', {}).get('sigma'), '#d97706')):
        if s:
            s = np.asarray(s, float)
            a1.semilogy(np.arange(1, len(s) + 1), s / s[0], color=col, lw=1, label=lab)
    a1.set_xlabel('index i')
    a1.set_ylabel(r'$\sigma_i/\sigma_1$')
    a1.legend(fontsize=7)
    a1.set_title('POD singular values', fontsize=8)
    for lab, P, col in (('primary', T['basis']['proj'], '#2563eb'),
                        ('probe', (T.get('probe_basis') or {}).get('info', {}).get('proj'), '#d97706')):
        if P:
            rs = sorted(int(r) for r in P)
            a2.loglog(rs, [P[str(r)]['frob_rel'] for r in rs], 'o-', color=col, ms=3, label=lab)
    a2.set_xlabel('rank r')
    a2.set_ylabel('rel. projection error (own snapshots)')
    a2.legend(fontsize=7)
    a2.set_title('projection floors (each basis on its own snapshot set)', fontsize=8)
    fig.suptitle(c['cell'], fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def samples_plot(c, T, path):
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150, sharey=True)
    ranks = sorted({v['r'] for v in T['stageA'].values()})
    cmap = plt.get_cmap('viridis')
    for ax, kind in zip(axs, ('ecsw', 'qdeim')):
        for i, r in enumerate(ranks):
            p = [(v['hr_points'], v['W']) for v in T['stageA'].values() if v['kind'] == kind and v['r'] == r
                 and v['q1']['q1'] and np.isfinite(v['W']) and v['hr_points']]
            if p:
                ax.scatter(*zip(*p), s=9, color=cmap(i / max(1, len(ranks) - 1)), label=f'r={r}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('sample nodes (ECSW support / Q-DEIM points)')
        ax.set_title(f'{kind}: Q1-passing screening runs', fontsize=8)
    axs[0].set_ylabel('worst E0 error, screening cases')
    axs[1].legend(fontsize=6)
    fig.suptitle(f"{c['cell']}: hyper-reduction sample count vs error (all schemes, steps and solvers)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def solver_pairs(T):
    out = []
    for k, v in T['stageA'].items():
        if '|native|' not in k:
            continue
        w = T['stageA'].get(k.replace('|native|', '|common|'))
        if w and v['q1']['q1'] and w['q1']['q1'] and v['timing_ok'] and w['timing_ok']:
            out.append((k, v, w))
    return out


def solver_plot(c, pairs, path):
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150)
    for kind in ('none', 'ecsw', 'qdeim'):
        p = [(v['t'], w['t'], v['W'], w['W']) for k, v, w in pairs if v['kind'] == kind]
        if p:
            tn, tc, wn, wc = zip(*p)
            a1.scatter(tc, tn, s=9, color=COL[kind], label=kind)
            a2.scatter(wc, wn, s=9, color=COL[kind], label=kind)
    for a, lab in ((a1, 'full-query time (s)'), (a2, 'worst E0 error')):
        a.set_xscale('log')
        a.set_yscale('log')
        lo, hi = a.get_xlim()
        a.plot([lo, hi], [lo, hi], 'k:', lw=.7)
        a.set_xlabel(f'common solver: {lab}')
        a.set_ylabel(f'native solver: {lab}')
        a.legend(fontsize=6)
    fig.suptitle(f"{c['cell']}: native vs common solver, same configuration, both Q1-passing (screening)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def offline_plot(cells, path):
    plt = _plt()
    keys = sorted({k for c, V, T in cells for k in T['timers']})
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    wdt = .8 / len(cells)
    for i, (c, V, T) in enumerate(cells):
        ax.bar(np.arange(len(keys)) + i * wdt, [T['timers'].get(k, 0) for k in keys], wdt, label=c['cell'])
    ax.set_xticks(np.arange(len(keys)) + .4 - wdt / 2)
    ax.set_xticklabels(keys, rotation=40, ha='right', fontsize=6)
    ax.set_yscale('log')
    ax.set_ylabel('GPU-seconds (in-job timers)')
    ax.set_title('baseline offline construction cost by component', fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def qualify_x(x, bq, front):
    """Export copy of a verdict X record (pairs dropped) with the budget qualification appended to every label (A3.1)."""
    import copy
    o = copy.deepcopy({k: v for k, v in x.items() if k != 'pairs'})
    tag = f' [{bq}]'
    for k in ('label', 'cell_line'):
        if o.get(k):
            o[k] += tag
    for grp in ('roles', 'exploratory'):
        for r in (o.get(grp) or {}).values():
            if isinstance(r, dict) and r.get('label'):
                r['label'] += tag
    o['budget_qualification'] = bq
    o['front2d_member'] = front
    return o


# ------------------------------------------------------------------------------------------------------------- main
def main():
    R = json.loads((HERE / 'configs' / 'report.json').read_text())
    led = json.loads((HERE / 'runs' / 'GPU-LEDGER.json').read_text())
    ov = json.loads((HERE / 'inputs' / 'ours_V.json').read_text())
    oT = json.loads((HERE / 'inputs' / 'ours_T.json').read_text())
    (HERE / 'plots').mkdir(exist_ok=True)
    cells, export = [], dict(lane='jcp-hr-baselines', label='E0 R_ST: conditional Richardson estimate', cells={})
    for c in R['cells']:
        V = json.loads((HERE / 'checks' / f"VERDICT-{c['cell']}.json").read_text())
        assert (HERE / 'checks' / f"VERDICT-{c['cell']}.sha256").read_text().split()[0] == \
            sha(HERE / 'checks' / f"VERDICT-{c['cell']}.json"), 'verdict file differs from its pinned hash'
        tpath = HERE / (c.get('tune_json') or f"runs/{c['tune_attempt']}/archive/output/tune.json")
        T = json.loads(tpath.read_text())
        assert sha(tpath) == V['tune_sha256'], ('tuning JSON differs from the verdict', tpath)
        cells.append((c, V, T))
    front2d = set(R.get('front2d_members', []))

    # ---- budget per dimension (computed first: its qualification is attached to every pair label) ----
    budget = {}
    for dim, keys in ((2, ('V2',)), (3, ('V3_batch2', 'V3_batch1'))):
        cs = [V for c, V, T in cells if c['d'] == dim]
        acc = [V['flags']['accounting'] for V in cs]
        bv = sum(a['V_hours_cumulative'] for a in acc)
        lb = any(a['V_cumulative_is_lower_bound'] for a in acc)
        ub = (None if any(a['V_upper_bound_cumulative'] is None for a in acc)
              else sum(a['V_upper_bound_cumulative'] for a in acc))
        tb = sum(a['T_hours_cumulative'] for a in acc)
        tlb = any(a['T_cumulative_is_lower_bound'] for a in acc)
        for k in keys:
            ours = ov[k]['total']
            lo = bv / ours
            hi = (ub / ours if ub is not None else None) if lb else lo
            stt, short = band(lo, hi)
            budget[(dim, k)] = dict(baseline_V=bv, lower_bound=lb, upper_bound=ub, ours_V=ours, ours_V_key=k,
                                    ratio=[lo, hi], statement=stt, short=short, baseline_T=tb, T_lower_bound=tlb)

    def blabel(d, xr):
        if d == 2:
            return budget[(2, 'V2')]['short']
        return budget[(3, 'V3_batch1' if xr.get('diagnostic_only') else 'V3_batch2')]['short']

    lines = []
    w = lines.append
    nX = {c['cell']: len(V['X']) for c, V, T in cells}
    d3 = [V for c, V, T in cells if c['d'] == 3][0]
    n3b1 = sum(1 for x in d3['X'].values() if x.get('diagnostic_only'))
    ncanc = len(R['cancelled'])
    nfail3 = len(d3['flags'].get('tune_construction_failed') or [])
    fr3 = sorted({k.split('|')[0] + ' rank ' + f"{int(k.split('|')[1]):,}" for k in d3['flags'].get('tune_construction_failed') or []})
    w('# C3-minimal: POD-LSPG baselines with ECSW and Q-DEIM hyper-reduction against our configurations')
    w('')
    w('Viscous Burgers, 2D $256^2$ and $1024^2$ and 3D $64^3$ (64 intervals, 65 nodes, $63^3$ interior unknowns): '
      'tuned POD-LSPG baselines (no hyper-reduction, ECSW, Q-DEIM; native and common solvers) against our '
      'configurations, timed in the same job on the same GPU. In 2D every X is qualified (Q1 + Q2, jcp-p1-scaling; '
      f'{sum(1 for c, V, T in cells if c["d"] == 2 for k in V["X"] if k in front2d)} are FRONT-2D members). In 3D, {len(d3["X"]) - n3b1} X are qualified (batch 2, jcp-qualify Q2, a qualified '
      f'subset) and {n3b1} are **diagnostic only** (batch 1, speed3d deployed rules, which fail jcp-qualify Q2); '
      'batch-1 lines never support a conclusion about our qualified method. Numbers are final for this lane: the '
      'final attempt of every cell completed and passed its results audit; '
      f'{ncanc} earlier attempts were cancelled '
      '(their reasons and cost are listed under "Cancelled attempts" and their hours are counted in the budget), and '
      f'{nfail3:,} 3D configurations ({", ".join(fr3)}) could not be constructed (device memory).')
    w('')
    w('Generated by `make_report.py` from the frozen verdict files; every number below is read from run JSONs. '
      'All comparisons are **developmental, in-sample, unadjusted** (A1.6): both methods were selected on the same '
      'cases, with unequal selection histories. The baseline roles (B_acc, B_cost) are chosen **by the registered role '
      'rule on the Stage-B data** (A26.5); "registered pick" means "selected by the registered rule", not an '
      'independently frozen winner or a held-out confirmation (A44.3). Errors are E0 R_ST errors against references '
      'whose uncertainty $\\varepsilon_{\\rm ref}$ is a conditional Richardson estimate. Q1 distances of the Stage-B members: results '
      'audits 2 (3D) and 3 (2D $256^2$) reconstructed them from the saved coefficients through the recorded '
      'basis-orthonormality certificate (the bases themselves are in the cluster archive); results audit 1 (2D '
      '$1024^2$) checked them against the recorded values. The 2D $256^2$ restriction "saturation check failed" means '
      'the 12-step basis failed the A1.3 check and the registered remedy — the all-step basis — was installed and used '
      'for that cell (its own adequacy passes); the snapshot limitation of the failed 12-step set applies to the 2D '
      '$1024^2$ cell. Under A35.1 a restriction withholds favourable or neutral conclusions; observed baseline '
      'dominations remain reportable.')
    w('')
    w('## Headline per cell')
    w('')
    for c, V, T in cells:
        xs = V['X']
        rs = '; '.join(V['flags']['restrictions']) or 'none'
        grp = {}
        for k, x in xs.items():
            b = ('batch 1, DIAGNOSTIC ONLY' if x.get('diagnostic_only') else
                 ('batch 2, qualified subset' if c['d'] == 3 else 'qualified (P1 Q1 + Q2)'))
            g = grp.setdefault(b, Counter())
            g['n'] += 1
            g['dom'] += 'baseline dominates' in (x.get('cell_line') or '')
            g['wh'] += 'withheld' in (x.get('cell_line') or '')
            g['inc'] += any(s in (x.get('label') or '') for s in ('incomplete', 'not assessed'))
        bq = blabel(c['d'], {}) if c['d'] == 2 else None
        for b, g in grp.items():
            bql = bq or blabel(3, dict(diagnostic_only=b.startswith('batch 1')))
            w(f"- **{c['cell']}, {b}**: {g['dom']} of {g['n']} X dominated by a baseline selected by the registered "
              f"role rule (lower full-query time and lower worst-case error); {g['wh']} X with a withheld neutral "
              f"conclusion; {g['inc']} X not assessed or incomplete. Restrictions: {rs}. Budget: {bql}. "
              f"Developmental, in-sample, unadjusted.")
    w('')
    w('## Budget (validation-informed GPU-hours, A3.1/A4.3)')
    w('')
    w('Baseline V sums every attempt of every cell of the dimension (A44.2): the lower bound counts the recorded '
      'validation-informed phases, the upper bound the measured allocation of those jobs. Ours V counts every job that '
      'selected or certified an X; where the owning lane\'s logs do not separate phases, the **whole job counts as V** '
      '(inseparable comparator jobs, A3.1). The interval of ratios $V_{\\rm base}/V_{\\rm ours}$ is compared with the '
      'registered band $[0.8, 1.25]$.')
    w('')
    w('| dimension | ours V used | baseline V (h) | baseline V upper bound (h) | ours V (h) | statement |')
    w('|---|---|---|---|---|---|')
    for (dim, k), b in budget.items():
        w(f"| {dim}D | `{k}` | {ge(b['lower_bound'])}{fmt(b['baseline_V'])} | {fmt(b['upper_bound'])} | "
          f"{fmt(b['ours_V'])} | {b['statement']} |")
        export['cells'][f'budget_{dim}D_{k}'] = {kk: vv for kk, vv in b.items()}
    w('')
    b2, b3 = budget[(2, 'V2')], budget[(3, 'V3_batch2')]
    w(f"In 2D the ratio interval [{b2['ratio'][0]:.3g}, {b2['ratio'][1]:.3g}] crosses the band's upper edge 1.25, so "
      f'whether the budgets are comparable is **unresolved**: the baseline spent between {b2["ratio"][0]:.3g}× and '
      f'{b2["ratio"][1]:.3g}× ours\' validation-informed time; every 2D pair label carries "budget band unresolved". '
      f"In 3D the qualified comparison uses ours' `V3_batch2` as the denominator; the baseline spent **less** "
      f"validation-informed time than ours (ratio at most {b3['ratio'][1]:.3g}), so every 3D pair label carries "
      '"unequal-budget, descriptive" (batch 1 is compared with `V3_batch1` and is likewise unequal).')
    w('')
    w('### Training-only construction (T)')
    w('')
    w('| side | dimension | T (h) | items | not available |')
    w('|---|---|---|---|---|')
    for dim in (2, 3):
        b = budget[(dim, 'V2' if dim == 2 else 'V3_batch2')]
        w(f"| baseline | {dim}D | {ge(b['T_lower_bound'])}{fmt(b['baseline_T'])} | snapshots, POD, advection POD, "
          'Q-DEIM pivoting, ECSW training runs and NNLS, every attempt | - |')
        o = oT[f'{dim}D']
        w(f"| ours | {dim}D | {fmt(sum(o['items'].values()))} | " + '; '.join(o['items']) + f" ({o['covers']}) | "
          + '; '.join(o['not_available']).replace('|', '\\|') + ' |')
    w('')
    # ---- comparators ----
    w('## The compared configurations (ours)')
    w('')
    w("- **2D (P1)**: jcp-p1-scaling configurations queried with that lane's byte-identical code "
      '(`p1common.Setting`, `p1core.make_split_query` fused query). Query contract: the full-field initial condition '
      'is sampled at the cold quadrature points and projected onto the reduced space, LSPG stepping at '
      '$\\Delta t_0 = 0.005$, decode of the six full-field outputs on the mesh. FRONT-2D members (jcp-qualify v1, '
      f"A27): {', '.join('`' + k + '`' for k in sorted(front2d))}. {R['front2d_excluded']}.")
    w("- **3D (speed3d / jcp-qualify)**: speed3d's own query, `s3common.make_query(spec, setting, path='mesh')`, "
      'with byte-identical `s3common` / `qn3` / `fs3p`. Query contract: least-squares projection of the full-field '
      'initial condition on the mesh (`TB.project`), FS or QN stepping, decode of the six outputs on the $63^3$ '
      'interior lattice. This is the full-field mesh-path contract (A21.2); the mesh-free m1 contract is not assessed.')
    w('- **Baselines**: initialisation $\\Phi_r^\\top u_0$, stepping, decode $\\Phi_r c$ on the mesh. For every query, '
      'timing covers initialisation + stepping + decode with `block_until_ready`; one-off setup (bank and table '
      'loading, rule data, basis and HR data transfer, compilation) is excluded on both sides.')
    w('')
    # ---- cancelled attempts and coverage ----
    w('## Cancelled attempts and coverage')
    w('')
    w('| attempt | Slurm | measured GPU-h | reason |')
    w('|---|---|---|---|')
    jl = {j['attempt']: j for j in led['jobs']}
    for a, why in R['cancelled'].items():
        w(f"| {a} | {jl[a].get('slurm')} | {fmt(jl[a].get('measured_gpu_hours'))} | {why} |")
    w('')
    for c, V, T in cells:
        ic = T.get('inventory_counts') or {}
        cf = V['flags'].get('tune_construction_failed') or []
        reasons = sorted({str(v).split(':')[0] for k, v in (T.get('construction_failures') or {}).items()
                           if k.endswith('|*')})
        w(f"- **{c['cell']}** tuning inventory: {fmt(ic.get('complete'))} configurations completed, "
          f"{fmt(len(cf))} not constructible" + (f" ({', '.join(sorted({k.split('|')[0] + ' $r=' + k.split('|')[1] + '$' for k in cf}))}; "
                                                f"first failure: {'; '.join(reasons)}, i.e. device memory exhausted at the first NNLS append)"
                                                if cf else '') +
          f", {fmt(ic.get('run_failed'))} run failures, {fmt(ic.get('deferred'))} deferred.")
    w(f"- Deferred scope (not attempted, not failures): {R['deferred_scope']}.")
    w('')
    # ---- per cell ----
    for c, V, T in cells:
        F = V['flags']
        d = c['d']
        w(f"## {c['cell']} ({d}D, $L = {c['L']}$ intervals)")
        w('')
        w(f"Tuning attempt `{c['tune_attempt']}`, Stage-B attempt `{c['stageb_attempt']}`. Basis used: "
          f"{str(F.get('basis_used')).replace('1024^2', '$1024^2$')} ({fmt(T['basis'].get('n_snap'))} snapshots). In-cell adequacy: "
          + ', '.join(f"$r={r}$: {'pass' if v['adequate'] else 'FAIL'} (ratio {v['ratio']:.3g})"
                      for r, v in T['adequacy'].items() if isinstance(v, dict) and 'ratio' in v)
          + (f"; saturation ($256^2$): {'pass' if F.get('saturation_256') else 'FAIL'}" if d == 2 and c['L'] == 256 else '')
          + '.')
        w('')
        w('Restrictions carried by every conclusion of this cell: ' +
          ('; '.join(F['restrictions']) if F['restrictions'] else 'none') + f'; budget: {blabel(d, {})}'
          + (' (batch 2) / ' + blabel(3, dict(diagnostic_only=True)) + ' (batch 1)' if d == 3 else '') + '.')
        w('')
        a = F['accounting']
        w(f"Construction and validation hours. Selected attempts: $T$ = {fmt(a['T_hours'])} h; $V$ = {fmt(a['V_hours'])} h "
          f"(tuning {fmt(a['V_hours_tuning'])}, Stage B {fmt(a['V_hours_stageB'])}); limiting comparator "
          f"{fmt(a['limiting_comparator_hours'])} h. Cumulative over every attempt (A44.2, A45.1): "
          f"$T$ {ge(a['T_cumulative_is_lower_bound'])}{fmt(a['T_hours_cumulative'])} h, "
          f"$V$ {ge(a['V_cumulative_is_lower_bound'])}{fmt(a['V_hours_cumulative'])} h (allocation upper bound "
          f"{fmt(a['V_upper_bound_cumulative'])} h); extra attempts: "
          f"{', '.join(e['attempt'] for e in a['extra_attempts']) or 'none'}.")
        w('')
        front_plot(c, T, V, HERE / 'plots' / f"front-{c['cell']}.png")
        w(f"![screening front](plots/front-{c['cell']}.png)")
        w('')
        w('*Screening plot (Stage A on the six screening cases, tuning job), not the verdict front: Q1 failures are '
          'drawn as crosses and are never ranked; the verdict uses Stage-B runs on the full verdict cohort. Star labels '
          'X1, X2, ... are keyed below.*')
        w('')
        w('| star | X (ours) | batch / qualification |')
        w('|---|---|---|')
        for i, key in enumerate((T.get('ours_pass') or {})):
            xr = V['X'].get(key, {})
            q = xr.get('qualification', '').replace('; FRONT-2D v1 member if listed in A27', '')
            if d == 2:
                q += '; FRONT-2D member' if key in front2d else '; not a FRONT-2D member'
            w(f"| X{i + 1} | {cid(key)} | {xr.get('batch')}: {q} |")
        w('')
        w('Per X: worst and median E0 error of X and of each role pick on the eligible verdict cases; $\\rho$ = '
          'baseline/X full-query time ratio (median over the six timing cases of the per-case median A–B–A ratio) with '
          'its 95 % case-bootstrap interval; $\\rho < 1$ means the baseline is faster. "Dominates" refers to query time and '
          'worst-case error only; the median error is printed beside it (A44.3, A45.2). Picks may use a different '
          'scheme or time step from X: the comparison is between tuned configurations, not at a fixed scheme or step.')
        w('')
        w('| X (ours) | batch | X worst / median | role | pick | pick worst / median | $\\rho$ [95 %] | label |')
        w('|---|---|---|---|---|---|---|---|')
        reversals, picks_dom, domX = [], Counter(), {}
        for key, xr in V['X'].items():
            ro = xr.get('roles', {})
            bl = f" [{blabel(d, xr)}]"
            fm = ' (FRONT-2D)' if d == 2 and key in front2d else ''
            if not ro:
                w(f"| {cid(key)}{fm} | {xr.get('batch')} | {fmt(xr.get('W_X'))} | - | - | - | - | "
                  f"{xr.get('label', '')}{bl} |")
                continue
            for role in ('B_acc', 'B_cost'):
                b = ro.get(role, {})
                ci = (b.get('time') or {}).get('ci95') or [None, None]
                lab = b.get('label', '')
                w(f"| {cid(key)}{fm} | {xr.get('batch')} | {fmt(xr.get('W_X'))} / {fmt(xr.get('med_X'))} | {role} | "
                  f"{cid(b.get('id', '-'))} | {fmt(b.get('W'))} / {fmt(b.get('med'))} | {fmt(b.get('T'))} "
                  f"[{fmt(ci[0])}, {fmt(ci[1])}] | {lab}{bl} |")
                if 'baseline dominates' in lab:
                    picks_dom[b['id']] += 1
                    domX.setdefault(key, set()).add(parse_b(b['id'])['r'])
                    if b.get('med') is not None and xr.get('med_X') and b['med'] > xr['med_X']:
                        reversals.append((key, role, b['id'], xr['W_X'], b['W'], xr['med_X'], b['med'],
                                          xr.get('diagnostic_only')))
            w(f"| | | | cell | | | | {xr.get('cell_line', '')}{bl} (pairs assessed: {xr.get('n_assessed_pairs', 0)}) |")
        w('')
        bx = {}
        for key, xr in V['X'].items():
            k = 'batch 1 (DIAGNOSTIC ONLY)' if xr.get('diagnostic_only') else (
                'batch 2 (qualified subset)' if d == 3 else 'batch 1 (qualified, P1)')
            g = bx.setdefault(k, Counter())
            g['n'] += 1
            g['dom'] += 'baseline dominates X' in (xr.get('cell_line') or '')
            g['wh'] += 'withheld' in (xr.get('cell_line') or '')
        w('By batch: ' + '; '.join(f"{k}: {v['dom']} of {v['n']} X dominated, {v['wh']} withheld" for k, v in bx.items())
          + f". Restrictions: {'; '.join(F['restrictions']) or 'none'}; budget: "
          + (blabel(2, {}) if d == 2 else f"{blabel(3, {})} (batch 2), {blabel(3, dict(diagnostic_only=True))} (batch 1)")
          + '; developmental, in-sample, unadjusted.')
        w('')
        if reversals:
            w('**Median-error reversals.** In these dominated pairs the pick has the lower worst-case error but a '
              '*higher* median error than X; the domination statement concerns worst-case error and time only, and the '
              'pick is not "more accurate" in general:')
            w('')
            w('| X | batch | role | pick | X worst → pick worst | X median → pick median | median ratio |')
            w('|---|---|---|---|---|---|---|')
            for key, role, pid, wx, wb, mx, mb, diag in sorted(reversals, key=lambda r: -r[6] / r[5]):
                w(f"| {cid(key)} | {'1 (diagnostic)' if diag else V['X'][key].get('batch')} | {role} | {cid(pid)} | "
                  f"{fmt(wx)} → {fmt(wb)} | {fmt(mx)} → {fmt(mb)} | {mb / mx:.3f}{' (substantial)' if mb / mx > 1.5 else ''} |")
            w('')
        if picks_dom:
            nmax = picks_dom.most_common(1)[0][1]
            tops = sorted(k for k, v in picks_dom.items() if v == nmax)
            for top in tops:
                n = nmax
                pb = parse_b(top)
                xs_opt = {(parse_x(k, d)['scheme'], round(parse_x(k, d)['dt'], 12)) for k in V['X']}
                same = (pb['scheme'], round(pb['dt'], 12)) in xs_opt
                adq = T['adequacy'].get(str(pb['r']), {})
                fails = [r for r, v in T['adequacy'].items() if isinstance(v, dict) and 'adequate' in v and not v['adequate']]
                dom_ranks = sorted({parse_b(p)['r'] for p in picks_dom})
                hp = (f"support target $\\kappa = {pb['hrp'][1:]}$" if pb['kind'] == 'ecsw' else
                      f"$p = {pb['hrp']}$ interpolation points" if pb['kind'] == 'qdeim' else 'no hyper-reduction')
                adr = {r: v for r, v in T['adequacy'].items() if isinstance(v, dict) and 'adequate' in v}
                def adtxt(r):
                    return ('not checked (rank outside the registered adequacy ranks)' if str(r) not in adr else
                            'pass' if adr[str(r)]['adequate'] else 'FAIL')
                w(f"**Most frequent dominating pick{' (tied, ' + str(len(tops)) + ' co-leaders)' if len(tops) > 1 else ''}**: {cid(top)} ({n} dominating role pairs): {pb['kind'].upper()} rank "
                  f"{pb['r']}, {hp}, {pb['scheme']}, {dt_text(pb['dt'], d)}, {pb['solver']} solver. Its scheme/step "
                  f"combination {'occurs' if same else 'differs from that of every X'}"
                  f"{' among the X' if same else ''}. Ranks of all dominating picks: "
                  f"{', '.join(f'{r} (adequacy {adtxt(r)})' for r in dom_ranks)}"
                  + (f"; adequacy fails at rank(s) {', '.join(fails)}, which carry the cell's restriction" if fails else '')
                  + f". Of the {len(domX)} dominated X, {sum(1 for v in domX.values() if any(str(r) in adr and adr[str(r)]['adequate'] for r in v))} "
                  f"have a dominating pick at a rank whose adequacy check passes"
                  + '.' + (' The comparisons set FS/QN stepping of the off-mesh bank against POD-LSPG and therefore do not '
                           'isolate the basis, the hyper-reduction or the solver.' if d == 3 else
                           ' The comparisons set the bank LSPG against POD-LSPG; they do not isolate the basis from the '
                           'hyper-reduction.'))
                w('')
        # ---- per-cell supporting tables ----
        sA = T['stageA']
        w('### Supporting evidence (screening, tuning job)')
        w('')
        w('Q1 outcome of the Stage-A configurations by arm (failures are reported and never ranked):')
        w('')
        w('| arm | solver | Q1 pass | Q1 fail |')
        w('|---|---|---|---|')
        for kind in ('none', 'ecsw', 'qdeim'):
            for sv in ('native', 'common'):
                g = [v for k, v in sA.items() if v['kind'] == kind and f'|{sv}|' in k]
                if g:
                    w(f"| {kind} | {sv} | {sum(v['q1']['q1'] for v in g)} | {sum(not v['q1']['q1'] for v in g)} |")
        w('')
        fl = T.get('fom_limiting') or {}
        if fl:
            w('Limiting comparator — the un-reduced central FOM at the baseline mesh, scheme and step, scored against E0 '
              '(the error floor any POD-LSPG of this FOM approaches as $r\\to n$; §3.1):')
            w('')
            w('| scheme | step | cases scored | eligible | worst E0 | median E0 |')
            w('|---|---|---|---|---|---|')
            opt = {}
            for k, v in fl.items():
                p = k.split('|')
                opt.setdefault((p[2], float(p[3])), []).append(v)
            for (sc, dt), vs in sorted(opt.items()):
                el = [v['e0']['e'] for v in vs if v.get('status') == 'complete' and v['e0'].get('eligible')]
                w(f"| {sc} | {dt_text(dt, d)} | {len(vs)} | {len(el)} | {fmt(max(el) if el else None)} | "
                  f"{fmt(float(np.median(el)) if el else None)} |")
            w('')
        rows = []
        for k, v in sA.items():
            if v['kind'] == 'none' or not v['q1']['q1']:
                continue
            p = k.split('|')
            nk = '|'.join(['none', p[1], '-'] + p[3:])
            n = sA.get(nk)
            if n and n['q1']['q1']:
                cs = [j for j in v['e0'] if v['e0'][j].get('eligible') and j in n['e0']]
                dd = max(abs(v['e0'][j]['e'] - n['e0'][j]['e']) for j in cs) if cs else None
                rows.append((v['kind'], v['r'], v['W'] / n['W'] if n['W'] else None, dd))
        if rows:
            w('Hyper-reduction versus `none` at the same rank, scheme, step and solver (the HR analogue of Q2, reported, '
              'not gated). Only the **error-level** discrepancy is available: the field distance between the HR run and '
              'the `none` run was not computed in-job and cannot be recovered from the pulled evidence.')
            w('')
            w('| arm | rank | configurations | median worst-error ratio HR/none | max per-case abs. error difference |')
            w('|---|---|---|---|---|')
            for kind in ('ecsw', 'qdeim'):
                for r in sorted({x[1] for x in rows if x[0] == kind}):
                    g = [x for x in rows if x[0] == kind and x[1] == r]
                    w(f"| {kind} | {r} | {len(g)} | {fmt(float(np.median([x[2] for x in g if x[2] is not None])))} | "
                      f"{fmt(max(x[3] for x in g if x[3] is not None))} |")
            w('')
        sp = solver_pairs(T)
        if sp:
            tr = [v['t'] / x['t'] for k, v, x in sp]
            wr = [v['W'] / x['W'] for k, v, x in sp]
            w(f"Native versus common solver: {len(sp)} configurations Q1-pass with both; median time ratio native/common "
              f"{fmt(float(np.median(tr)))} (range {fmt(min(tr))}–{fmt(max(tr))}); median worst-error ratio "
              f"{fmt(float(np.median(wr)))} (range {fmt(min(wr))}–{fmt(max(wr))}).")
            w('')
            solver_plot(c, sp, HERE / 'plots' / f"solver-{c['cell']}.png")
            w(f"![native vs common](plots/solver-{c['cell']}.png)")
            w('')
        P, PP = T['basis']['proj'], (T.get('probe_basis') or {}).get('info', {}).get('proj') or {}
        w(f"POD (primary basis, {fmt(T['basis'].get('n_snap'))} snapshots) and projection floors; the probe basis "
          f"(upwind generator, {fmt((T.get('probe_basis') or {}).get('info', {}).get('n_snap'))} snapshots) beside it:")
        w('')
        w('| rank | primary: projection error | probe: projection error | held-out / in-set ratio (adequacy) |')
        w('|---|---|---|---|')
        for r in sorted(P, key=int):
            ad = T['adequacy'].get(r, {})
            w(f"| {r} | {fmt(P[r]['frob_rel'])} | {fmt(PP.get(r, {}).get('frob_rel'))} | "
              + (f"{fmt(ad.get('ratio'))} ({'pass' if ad.get('adequate') else 'FAIL'})" if 'ratio' in ad else 'not checked')
              + ' |')
        w('')
        pod_plot(c, T, HERE / 'plots' / f"pod-{c['cell']}.png")
        samples_plot(c, T, HERE / 'plots' / f"hr-samples-{c['cell']}.png")
        w(f"![POD spectra](plots/pod-{c['cell']}.png)")
        w('')
        w(f"![HR sample count vs error](plots/hr-samples-{c['cell']}.png)")
        w('')
        pr = T.get('probe_runs') or {}
        if pr:
            st = T['basis']['snapshots'].get('steps') or []
            pst = (T.get('probe_basis') or {}).get('info', {}).get('snapshots', {}).get('steps') or []
            w('Generator probe (A2.3, ratified by the coordinator in A19.2: the primary snapshots come from the central '
              'FOM on the training parameter draws, which gives the baseline its best data; the probe regenerates the '
              "snapshots with the bank's sign-upwind BE generator on the same draws). `none` arm, native solver, worst "
              'E0 on the screening cases, beside the central basis at the same rank and option'
              + (f'. **Confound in this cell**: the primary basis uses {len(st)} saved steps per trajectory '
                 f"({fmt(T['basis'].get('n_snap'))} snapshots, the all-step remedy) while the probe uses the "
                 f"{len(pst)}-step set, so the difference is not solely a generator change."
                 if len(st) != len(pst) else '; both bases use the same saved steps.'))
            w('')
            w('| configuration | probe worst E0 | central worst E0 | status |')
            w('|---|---|---|---|')
            for k, v in pr.items():
                ce = sA.get(k.replace('|upwind', '|central'), {}).get('W')
                pe = (max(x['e0']['e'] for x in v['cases'].values()) if v.get('status') == 'complete' else None)
                w(f"| {cid(k)} | {fmt(pe)} | {fmt(ce)} | {v.get('status')} |")
            w('')
        w('Offline construction, in-job GPU-seconds (selected tuning attempt): ' +
          ', '.join(f'{k} {v:.4g}' for k, v in T['timers'].items()) + '. Compilation time is not separately recorded (it falls inside these phase timers and the per-run '
          'warm-up, and is excluded from every query time). Break-even query counts (§11) are not computed: in the '
          'dominated pairs the baseline is both Q1-qualified and faster per query, and in every other assessed pair the cell '
          'restriction withholds any favourable or neutral conclusion. A break-even count needs comparable offline '
          'costs of both methods; ours\' per-mesh setup cost (and, in 3D, the M2 bank cost) is not available (see the T '
          'table), so break-even counts are reported as unavailable.')
        w('')
        export['cells'][c['cell']] = dict(
            flags=F, budget_qualification={'batch2' if d == 3 else 'all': blabel(d, {}),
                                           **({'batch1': blabel(3, dict(diagnostic_only=True))} if d == 3 else {})},
            X={k: qualify_x(x, blabel(d, x), (k in front2d) if d == 2 else None) for k, x in V['X'].items()},
            median_reversals=[dict(X=r[0], role=r[1], pick=r[2], W_X=r[3], W_pick=r[4], med_X=r[5], med_pick=r[6])
                              for r in reversals],
            verdict_sha256=sha(HERE / 'checks' / f"VERDICT-{c['cell']}.json"))
    offline_plot(cells, HERE / 'plots' / 'offline-cost.png')
    w('## Offline cost overview')
    w('')
    w('![offline cost](plots/offline-cost.png)')
    w('')
    # ---- ledger ----
    tot = sum(j.get('measured_gpu_hours') or 0 for j in led['jobs'])
    phase = sum(V['flags']['accounting']['T_hours_cumulative'] + V['flags']['accounting']['V_hours_cumulative']
                for c, V, T in cells)
    smk = sum(j.get('measured_gpu_hours') or 0 for j in led['jobs'] if j['attempt'].startswith('smk'))
    w('## Ledger')
    w('')
    w(f"Measured allocation over all jobs: **{tot:.4g} GPU-h** of the {led['cap_gpu_hours']} GPU-h cap. Of this, "
      f"{smk:.3g} h were the two phase-I smokes and {phase:.4g} h are accounted to recorded T and V phases of the "
      f"production attempts; the remaining {tot - smk - phase:.3g} h are allocation overhead outside the phase timers "
      '(job start-up, environment and asset checks, compilation, collection) and the unfinished phases of cancelled '
      'attempts, which count only in the V upper bounds.')
    w('')
    w('| attempt | Slurm | GPU | role | requested h | measured h | status |')
    w('|---|---|---|---|---|---|---|')
    for j in led['jobs']:
        w(f"| {j['attempt']} | {j.get('slurm')} | {j.get('gpu', '')} | {j.get('role', '')} | "
          f"{fmt(j.get('requested_hours'))} | {fmt(j.get('measured_gpu_hours'))} | {j.get('status', '')} |")
    w('')
    export['ledger'] = dict(measured_total=tot, cap=led['cap_gpu_hours'], smokes=smk, phase_accounted=phase)
    # ---- glossary ----
    w('## Glossary')
    w('')
    for k, v in (
            ('POD', 'proper orthogonal decomposition: the basis $\\Phi_r$ of the $r$ leading left singular vectors of '
                    'the snapshot matrix (training trajectories only)'),
            ('LSPG', 'least-squares Petrov–Galerkin: at each step the reduced coefficients $c$ minimise the norm of the '
                     'full-order residual $\\|r(\\Phi_r c)\\|$ (or its hyper-reduced version)'),
            ('none / ECSW / Q-DEIM', 'the three baseline arms: no hyper-reduction (full residual); energy-conserving '
                                     'sampling and weighting (residual evaluated on a sparse weighted node set found '
                                     'by non-negative least squares, NNLS); Q-DEIM (the nonlinear advection term '
                                     'interpolated from $p$ rows chosen by pivoted QR of an advection POD basis, linear '
                                     'terms exact)'),
            ('configuration id (baseline)', '`arm|r|hr-parameter|scheme|Δt|solver|generator`: rank $r$; hr-parameter '
                                            '= ECSW support target $s=\\kappa r$ (`k4`: the first NNLS path state with at '
                                            'least $4r$ nodes, A15.1, or the KKT-certified NNLS optimum when the path '
                                            'ends there before reaching the target, A23; the target is not the actual '
                                            'support, which is the sample-node count in the HR plots) or the Q-DEIM point count $p$ (`-` for none); generator `central` (primary) or '
                                            '`upwind` (probe)'),
            ('configuration id (ours)', '2D `<L>|R<r>[nom]|<scheme>`: bank width $R\'=r$ of the ordered bank, '
                                        '`nom` = nominal mode policy; 3D `<bank setting>|<rule>|<fs or qn>|<scheme>|'
                                        '<Δt>|<K or k>`'),
            ('BE / CN / CNR / BDF2', 'backward Euler; Crank–Nicolson; CN-R = Crank–Nicolson after two backward-Euler '
                                     'start-up steps; second-order backward differentiation (BE first step)'),
            ('FS / QN', "our 3D steppers: FS = fixed-sweep LSPG (adaptive Levenberg–Marquardt on the first steps, then K "
                        "cached-predictor LM sweeps per step, `K1` = one); QN-k = iterations with the exact gradient "
                        "and a frozen Gauss–Newton matrix refreshed per the k setting (speed3d `qn3`)"),
            ('native / common solver', 'Gauss–Newton with Householder QR (the LSPG solver of Carlberg et al.) / our '
                                       'Levenberg–Marquardt solver (`t2core.make_lm`, 3-candidate predictor), applied '
                                       'to every arm (§4)'),
            ('mesh', '$L$ counts intervals per axis: 2D $L=256$ has $255^2$ unknowns; 3D $L=64$ has 65 nodes per axis '
                     'and $63^3$ interior unknowns'),
            ('E0 error', '$e=\\max_{k\\ge1}\\|F_k-R_k\\|/\\|R_0\\|$ over the outputs $t_k=0.05k$, $F$ the method\'s '
                         'field and $R$ the E0 reference on the scoring lattice; "worst" = maximum over the eligible '
                         'cases, "median" = median over them'),
            ('$\\varepsilon_{\\rm ref}$', 'reference uncertainty: the conditional Richardson estimate of the E0 reference error; pair labels '
                      'treat error differences within it as unresolved'),
            ('eligible', 'a case whose reference passes the store\'s status for its case-kind; ineligible cases are '
                         'scored and shown, never ranked'),
            ('screening / verdict cohort', 'Stage A: six cases (2D `dev6`, 3D cases 0–5) used for tuning; Stage B / '
                                           'verdict: 2D `dev6`+`val32` (38 cases), 3D all 64 cases'),
            ('Q1', 'numerical qualification: every run valid; at most 1 % non-stationary steps ($\\gamma > 10^{-3}$, '
                   '$\\gamma = \\|J^\\top \\hat r\\|/(\\|J\\|_F\\|\\hat r\\|)$); distance to a tight solve '
                   '$\\le 2.5\\times10^{-4}$ on every case; tight vs tighter $\\le 2.5\\times10^{-5}$ on two cases'),
            ('Q2', 'quadrature qualification of our configurations (owning lanes); 3D batch 2 = jcp-qualify Q2'),
            ('adequacy / saturation', 'adequacy: held-out over in-set projection error ratio of the basis at rank $r$ '
                                      '(fails above the registered bound); saturation ($256^2$ only): whether more '
                                      'saved steps per trajectory change the basis (the 12-step set failed, so the '
                                      'all-step basis was used)'),
            ('X', 'one of our configurations'),
            ('B_acc', 'fastest baseline on the X-specific front whose worst error is at most X\'s'),
            ('B_cost', 'most accurate baseline on the X-specific front whose median time ratio to X is at most 1'),
            ('$\\rho$ (time ratio)', 'full-query time of the baseline divided by X\'s, measured A–B–A (X, baseline, X) on six '
                               'timing cases × three repeats; per case the median ratio, then the median over cases; '
                               '95 % interval from 10,000 case-level bootstrap resamples'),
            ('full query', 'initialisation from the full-field initial condition + time stepping to $t=0.25$ + decode '
                           'of the six outputs, timed warm with `block_until_ready`'),
            ('dominates', 'the 95 % interval of $\\rho$ lies below 1 and the $\\varepsilon_{\\rm ref}$-envelope of the baseline\'s worst error lies '
                          'below that of X'),
            ('withheld', 'a favourable or neutral conclusion suppressed because a coverage or adequacy restriction applies'),
            ('batch 1 / batch 2', '3D: speed3d deployed rules (DIAGNOSTIC ONLY) / jcp-qualify Q2-qualified subset; '
                                  '2D batch 1 = the P1-qualified configurations'),
            ('FRONT-2D member', 'a 2D X that is on jcp-qualify\'s frozen 2D front (v1)'),
            ('T / V', 'training-only construction hours / validation-informed hours (every run on a development case '
                      'that selected or certified a configuration); the budget compares $V_{\\rm base}/V_{\\rm ours}$ '
                      'with $[0.8, 1.25]$'),
            ('lower / upper bound (accounting)', 'lower: recorded phase timers of completed phases; upper: measured '
                                                 'allocation of the same jobs; "≥" marks a lower bound'),
            ('ledger fields', 'requested h = wall time reserved; measured h = elapsed allocation × 1 GPU (sacct)'),
            ('limiting comparator', 'the un-reduced FOM with the baseline\'s operator, mesh, scheme and step'),
            ('generator probe', 'the same `none` runs with a basis built from upwind-generated snapshots (A2.3)'),
            ('G-ours', 'replay of our exported errors to $10^{-10}$ relative before any timing'),
    ):
        w(f'- **{k}**: {v}.')
    (HERE / 'report.md').write_text('\n'.join(lines) + '\n')
    export['report_sha256'] = sha(HERE / 'report.md')
    (HERE / 'runs' / 'EXPORT.json').write_text(json.dumps(export, indent=1, default=float) + '\n')
    print('report.md, runs/EXPORT.json, plots/ written')


if __name__ == '__main__':
    main()
