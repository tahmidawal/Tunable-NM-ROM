"""Generate report.md, plots and runs/EXPORT.json from the frozen verdict files (never hand-typed numbers).

Inputs (configs/report.json): per cell {cell, d, L, tune_attempt, stageb_attempt}; reads
runs/<tune_attempt>/archive/output/tune.json, runs/<stageb_attempt>/archive/output/stageb.json,
checks/VERDICT-<cell>.json (+ .sha256), checks/FROZEN-STAGEB-<cell>.json, runs/GPU-LEDGER.json, inputs/ours_V.json.

Dimension-level budget wording (A3.1, A4.3, A19.3): baseline V = sum of the cells' V (tuning + Stage B) of that
dimension; ratio = baseline V / ours' V; in [0.8, 1.25] -> "within the registered factor-1.25 band (ratio = x)",
otherwise "unequal-budget, descriptive (ratio = x)"; the ratio is printed in every case; when any V is a lower bound
(a truncated job) the ratio is printed as a bound and the band statement is "unresolved (truncated accounting)".

Usage: python make_report.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fmt(x, nd=3):
    if x is None:
        return 'n/a'
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    if not np.isfinite(x):
        return 'non-finite'
    return f'{x:.{nd}g}'


def band(lo, hi):
    """A4.3 wording on the baseline/ours V ratio, with the accounting bounds: lo = ratio from the recorded-phase lower
    bound, hi = ratio from the allocation upper bound (None if unknown). Both inside [0.8, 1.25] -> within the band;
    both on the same side outside it -> unequal-budget, descriptive; otherwise unresolved. The ratio bounds are printed."""
    if lo is None:
        return 'V ratio unavailable'
    rng = f'ratio in [{lo:.3g}, {hi:.3g}]' if hi is not None else f'ratio >= {lo:.3g}'
    if hi is not None and 0.8 <= lo and hi <= 1.25:
        return f'within the registered factor-1.25 band ({rng})'
    if (hi is not None and hi < 0.8) or lo > 1.25:
        return f'unequal-budget, descriptive ({rng})'
    return f'unresolved against the factor-1.25 band ({rng})'


def front_plot(cell, T, V, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    colors = {'none': '#6b7280', 'ecsw': '#2563eb', 'qdeim': '#d97706'}
    for kind in ('none', 'ecsw', 'qdeim'):
        pts = [(v['t'], v['W']) for v in T['stageA'].values()
               if v['kind'] == kind and v['q1']['q1'] and v['timing_ok'] and np.isfinite(v['W'])]
        if pts:
            x, y = zip(*pts)
            ax.scatter(x, y, s=12, alpha=.7, color=colors[kind], label=f'{kind} (Q1-qualified, screening)')
    for key, xr in (T.get('ours_pass') or {}).items():
        if xr.get('status') != 'complete':
            continue
        t = float(np.median([c['seconds_warm'] for c in xr['cases'].values()]))
        w = max(c['e0']['e'] for c in xr['cases'].values())
        ax.scatter([t], [w], marker='*', s=90, color='#dc2626', zorder=5)
        ax.annotate(key.split('|')[0] + '|' + key.split('|')[1], (t, w), fontsize=6, xytext=(3, 3),
                    textcoords='offset points')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('median warm query time on the 6 timing cases (s, same job, same GPU)')
    ax.set_ylabel('worst E0 error on the eligible screening cases')
    ax.set_title(f'{cell}: Stage-A screening (dots) and ours (stars)', fontsize=9)
    ax.legend(fontsize=7, loc='best')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    R = json.loads((HERE / 'configs' / 'report.json').read_text())
    led = json.loads((HERE / 'runs' / 'GPU-LEDGER.json').read_text())
    ov = json.loads((HERE / 'inputs' / 'ours_V.json').read_text())
    (HERE / 'plots').mkdir(exist_ok=True)
    cells, export = [], dict(lane='jcp-hr-baselines', label='E0 R_ST: conditional Richardson estimate', cells={})
    deferred = []
    for c in R['cells']:
        if c.get('deferred'):                          # A30.1 / A43.3: tuning reported, no Stage B / verdict
            deferred.append(c)
            continue
        V = json.loads((HERE / 'checks' / f"VERDICT-{c['cell']}.json").read_text())
        assert (HERE / 'checks' / f"VERDICT-{c['cell']}.sha256").read_text().split()[0] == \
            sha(HERE / 'checks' / f"VERDICT-{c['cell']}.json"), 'verdict file differs from its pinned hash'
        tpath = HERE / (c.get('tune_json') or f"runs/{c['tune_attempt']}/archive/output/tune.json")
        T = json.loads(tpath.read_text())
        assert sha(tpath) == V['tune_sha256'], ('tuning JSON differs from the verdict', tpath)
        front_plot(c['cell'], T, V, HERE / 'plots' / f"front-{c['cell']}.png")
        cells.append((c, V, T))
    lines = []
    w = lines.append
    w('# C3-minimal: POD-LSPG baselines with ECSW and Q-DEIM hyper-reduction against our configurations')
    w('')
    w('Viscous Burgers, 2D 256² and 1024² and 3D 65³: tuned POD-LSPG baselines (no hyper-reduction, ECSW, Q-DEIM; '
      'native and common solvers) against our qualified configurations, timed in the same job on the same GPU. '
      'Numbers are final for this lane (all production jobs complete and results-audited); the wording is subject to '
      'the report audit.')
    w('')
    w('Generated by `make_report.py` from the frozen verdict files; every number below is read from run JSONs. '
      'All comparisons are **developmental, in-sample, unadjusted** (A1.6): both methods were selected on the same '
      'cases, with unequal selection histories. The baseline roles (B_acc, B_cost) are chosen **by the registered role '
      'rule on the Stage-B data** (A26.5); "registered pick" means "selected by the registered rule", not an '
      'independently frozen winner or a held-out confirmation (A44.3). Errors are E0 R_ST errors, conditional '
      'Richardson estimates. 3D batch-1 lines are diagnostic only (A22.0); 3D batch-2 lines are a qualified subset, not '
      'a completeness claim (A29). Q1 distances of the Stage-B members were checked by the results audit against the '
      'recorded values, not reconstructed (the bases are in the cluster archive).')
    w('')
    w('## Headline per cell')
    w('')
    for c, V, T in cells:
        xs = V['X']
        dom = [k for k, x in xs.items() if 'baseline dominates' in (x.get('cell_line') or '')]
        inc = [k for k, x in xs.items() if 'incomplete' in (x.get('label') or '') or 'not assessed' in (x.get('label') or '')]
        wh = [k for k, x in xs.items() if 'withheld' in (x.get('cell_line') or '')]
        bat = {}
        for k, x in xs.items():
            b = 'batch 1 (diagnostic only)' if x.get('diagnostic_only') else f"batch {x.get('batch')}"
            bat.setdefault(b, [0, 0])
            bat[b][0] += 1
            bat[b][1] += k in dom
        bs = '; '.join(f'{b}: {v[1]} of {v[0]}' for b, v in bat.items()) if c['d'] == 3 else ''
        w(f"- **{c['cell']}**: {len(dom)} of {len(xs)} X dominated by a baseline selected by the registered role rule "
          f"(lower full-query time and lower worst-case error){' (' + bs + ')' if bs else ''}; {len(wh)} X with a "
          f"withheld neutral conclusion; {len(inc)} X not assessed or incomplete. "
          f"Restrictions: {'; '.join(V['flags']['restrictions']) or 'none'}.")
    for c in deferred:
        w(f"- **{c['cell']}**: Stage B deferred ({c['deferred']}); tuning results only.")
    w('')
    # ---- budget per dimension ----
    w('## Budget (validation-informed GPU-hours, A3.1/A4.3)')
    w('')
    w('| dimension | baseline V (h) | lower bound only | upper bound (h) | ours V (h) | statement |')
    w('|---|---|---|---|---|---|')
    for dim, ovk in ((2, 'V2'), (3, 'V3_batch2')):
        cs = [(c, V) for c, V, T in cells if c['d'] == dim]
        if not cs:
            continue
        acc = [V['flags']['accounting'] for c, V in cs]
        bv = sum(a['V_hours_cumulative'] for a in acc)                     # A44.2: every attempt of every cell
        lb = any(a['V_cumulative_is_lower_bound'] for a in acc)
        ub = (None if any(a['V_upper_bound_cumulative'] is None for a in acc)
              else sum(a['V_upper_bound_cumulative'] for a in acc))
        ours = ov[ovk]['total']
        lo = bv / ours if ours else None
        hi = (ub / ours if ours else None) if lb else lo
        hi = hi if ub is not None or not lb else None
        stt = band(lo, hi)
        w(f'| {dim}D | {fmt(bv)} | {lb} | {fmt(ub)} | {fmt(ours)} | {stt} |')
        export['cells'][f'budget_{dim}D'] = dict(baseline_V=bv, lower_bound=lb, upper_bound=ub, ours_V=ours,
                                                 statement=stt)
    w('')
    w('Training-only construction (T) is reported per cell below and is never omitted (A3.1).')
    w('')
    # ---- per cell ----
    for c, V, T in cells:
        F = V['flags']
        w(f"## {c['cell']} ({c['d']}D, L = {c['L']})")
        w('')
        w(f"Tuning attempt `{c['tune_attempt']}`, Stage-B attempt `{c['stageb_attempt']}`. Basis used: "
          f"{F.get('basis_used')}. Adequacy: {F.get('adequate')}; saturation (256²): {F.get('saturation_256')}.")
        w('')
        w('Restrictions carried by every conclusion of this cell: ' +
          ('; '.join(F['restrictions']) if F['restrictions'] else 'none') + '.')
        w('')
        a = F['accounting']
        w(f"Selected attempts: T = {fmt(a['T_hours'])} h; V = {fmt(a['V_hours'])} h (tuning {fmt(a['V_hours_tuning'])}, "
          f"Stage B {fmt(a['V_hours_stageB'])}); limiting comparator {fmt(a['limiting_comparator_hours'])} h. "
          f"Cumulative over every attempt (A44.2, A45.1): T >= {fmt(a['T_hours_cumulative'])} h, V >= "
          f"{fmt(a['V_hours_cumulative'])} h (allocation upper bound {fmt(a['V_upper_bound_cumulative'])} h); "
          f"extra attempts: {', '.join(e['attempt'] for e in a['extra_attempts']) or 'none'}.")
        w('')
        w(f"![screening front](plots/front-{c['cell']}.png)")
        w('')
        w('Per X (ours): worst and median E0 error of X and of each role pick on the eligible verdict cases; T = '
          'baseline/X full-query time (median over the six timing cases of the per-case median A-B-A ratio) with its '
          '95 % bootstrap interval. "Dominates" refers to query time and worst-case error only; the median error is '
          'shown beside it (A44.3, A45.2). Picks may use a different scheme or time step from X (registered option '
          'ladder): the comparison is between tuned configurations, not at a fixed scheme or step.')
        w('')
        w('| X (ours) | batch | X worst / median | role | pick | pick worst / median | T [95 %] | label |')
        w('|---|---|---|---|---|---|---|---|')
        for key, xr in V['X'].items():
            ro = xr.get('roles', {})
            if not ro:
                w(f"| `{key}` | {xr.get('batch')} | {fmt(xr.get('W_X'))} | - | - | - | - | {xr.get('label', '')} |")
                continue
            for role in ('B_acc', 'B_cost'):
                b = ro.get(role, {})
                ci = (b.get('time') or {}).get('ci95') or [None, None]
                w(f"| `{key}` | {xr.get('batch')} | {fmt(xr.get('W_X'))} / {fmt(xr.get('med_X'))} | {role} | "
                  f"`{b.get('id', '-')}` | {fmt(b.get('W'))} / {fmt(b.get('med'))} | {fmt(b.get('T'))} "
                  f"[{fmt(ci[0])}, {fmt(ci[1])}] | {b.get('label', '')} |")
            w(f"| | | | cell | | | | {xr.get('cell_line', '')} (pairs assessed: {xr.get('n_assessed_pairs', 0)}) |")
        w('')
        bx = {}
        for key, xr in V['X'].items():
            k = 'batch 1 (diagnostic)' if xr.get('diagnostic_only') else f"batch {xr.get('batch')}"
            dom = 'baseline dominates X' in (xr.get('cell_line') or '')
            wh = 'withheld' in (xr.get('cell_line') or '')
            bx.setdefault(k, [0, 0, 0])
            bx[k][0] += 1
            bx[k][1] += dom
            bx[k][2] += wh
        w('By batch: ' + '; '.join(f"{k}: {v[1]} of {v[0]} X dominated, {v[2]} withheld" for k, v in bx.items()) + '.')
        w('')
        pr = T.get('probe_runs') or {}
        if pr:
            w('Generator probe (A2.3; `none`, native solver, upwind-generated basis; worst E0 on the screening cases) '
              'beside the central-snapshot basis at the same rank and option:')
            w('')
            w('| configuration | probe worst E0 | central worst E0 | status |')
            w('|---|---|---|---|')
            for cid, v in pr.items():
                ce = T['stageA'].get(cid.replace('|upwind', '|central'), {}).get('W')
                pe = (max(x['e0']['e'] for x in v['cases'].values()) if v.get('status') == 'complete' else None)
                w(f"| `{cid}` | {fmt(pe)} | {fmt(ce)} | {v.get('status')} |")
            w('')
        export['cells'][c['cell']] = dict(flags=F, X={k: {kk: vv for kk, vv in x.items() if kk != 'pairs'}
                                                       for k, x in V['X'].items()},
                                          verdict_sha256=sha(HERE / 'checks' / f"VERDICT-{c['cell']}.json"))
    w('## Ledger')
    w('')
    w('| attempt | Slurm | role | requested h | measured h | status |')
    w('|---|---|---|---|---|---|')
    for j in led['jobs']:
        w(f"| {j['attempt']} | {j.get('slurm')} | {j.get('role', '')} | {fmt(j.get('requested_hours'))} | "
          f"{fmt(j.get('measured_gpu_hours'))} | {j.get('status', '')} |")
    w('')
    w('## Glossary')
    w('')
    for k, v in (
            ('X', 'one of our configurations (P1 in 2D; speed3d / jcp-qualify in 3D), timed in the same job as the baseline'),
            ('B_acc', 'fastest baseline on the X-specific front whose worst error is at most X\'s'),
            ('B_cost', 'most accurate baseline on the X-specific front whose median time ratio to X is at most 1'),
            ('worst E0', 'largest E0 R_ST error over the eligible verdict cases (a conditional Richardson estimate)'),
            ('T / V', 'training-only construction hours / validation-informed hours (selection, certification)'),
            ('Q1', 'numerical qualification: stationarity of every step, distance to a tighter-tolerance anchor'),
            ('G-ours', 'replay of our exported errors to 1e-10 relative before any timing'),
            ('batch 1 / batch 2', '3D comparators: deployed rules (diagnostic only) / jcp-qualify Q2-qualified'),
            ('withheld', 'a favourable or neutral conclusion suppressed because a coverage or adequacy check failed'),
    ):
        w(f'- **{k}**: {v}.')
    (HERE / 'report.md').write_text('\n'.join(lines) + '\n')
    export['report_sha256'] = sha(HERE / 'report.md')
    (HERE / 'runs' / 'EXPORT.json').write_text(json.dumps(export, indent=1, default=float) + '\n')
    print('report.md, runs/EXPORT.json, plots/ written')


if __name__ == '__main__':
    main()
