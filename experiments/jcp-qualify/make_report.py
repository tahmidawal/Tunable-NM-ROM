"""Generate report.md and plots/*.png from checks/analysis.json, checks/smoke-accept.json, runs/GPU-LEDGER.json and
runs/FRONT-2D.json (no hand-typed numbers; design constants such as bars and step factors are the registered values).

    python make_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AN = json.loads((HERE / 'checks' / 'analysis.json').read_text())
SMK = json.loads((HERE / 'checks' / 'smoke-accept.json').read_text()) if (HERE / 'checks' / 'smoke-accept.json').exists() else {}
LED = json.loads((HERE / 'runs' / 'GPU-LEDGER.json').read_text())
FR = json.loads((HERE / 'runs' / 'FRONT-2D.json').read_text()) if (HERE / 'runs' / 'FRONT-2D.json').exists() else None
FRSHA = (HERE / 'runs' / 'FRONT-2D.sha256').read_text().split()[0] if (HERE / 'runs' / 'FRONT-2D.sha256').exists() else None
MESHES = ('256', '1024', '4096')
SCH = {'LSPG|BE': 'LSPG-BE', 'LSPG|CNR': 'LSPG-CN-R', 'GAL|CN': 'GAL-CN'}
COL = {'LSPG|BE': '#4269d0', 'LSPG|CNR': '#efb118', 'GAL|CN': '#3ca951'}
LBL = 'E0 $R_{ST}$ (conditional Richardson estimate)'


def pct(x, d=3):
    return '—' if x is None else f'{100 * x:.{d}g} %'


def g(x, d=3):
    return '—' if x is None else (f'{x:.{d}g}' if isinstance(x, (int, float)) else str(x))


def esc(x):
    return str(x).replace('|', ' \\| ')


def rng(x):
    return '—' if not x else f'{x[0]:.3g}–{x[1]:.3g}'


def parse(k):
    p = k.split('|')
    return dict(mesh=p[0], r=int(p[1][1:]), form=p[2], scheme=p[3], factor=float(p[4]), fb=len(p) > 5)


def name(k):
    c = parse(k)
    return f"R{c['r']} {SCH[c['form'] + '|' + c['scheme']]} {g(c['factor'])}Δt₀" + (' (fallback rule)' if c['fb'] else '')


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pd = HERE / 'plots'
    pd.mkdir(exist_ok=True)
    for old in pd.glob('*.png'):
        old.unlink()
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                         'grid.color': '#e5e4df', 'grid.linewidth': .6})
    made = []
    F = AN.get('fronts', {})
    for stat in ('median', 'worst'):
        fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
        for ax, m in zip(axs, MESHES):
            f = F.get(m, {})
            cand = f.get('candidates', {})
            front = set(f.get('fronts', {}).get(f'{stat}|c', []))
            for k, v in cand.items():
                c = parse(k)
                key = c['form'] + '|' + c['scheme']
                on = k in front
                ax.scatter(v['c'], v[stat], s=26 + 8 * (c['r'] // 128), color=COL[key], alpha=1 if on else .35,
                           edgecolor='#0b0b0b' if on else 'none', linewidth=.8, zorder=3 if on else 2)
                if on:
                    ax.annotate(f"R{c['r']}·{g(c['factor'])}", (v['c'], v[stat]), xytext=(4, 3), textcoords='offset points', fontsize=7)
            pts = sorted((cand[k]['c'], cand[k][stat]) for k in front)
            if pts:
                ax.plot(*zip(*pts), color='#0b0b0b', lw=.8, zorder=1)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(f'{m}² — {len(front)} front members of {len(cand)} qualified', fontsize=9)
            ax.set_xlabel('online cost: fused query / reference A (same H200 job)')
        axs[0].set_ylabel(f'{stat} error over 38 cases\n{LBL}')
        for key, col in COL.items():
            axs[-1].scatter([], [], color=col, label=SCH[key])
        axs[-1].legend(frameon=False, fontsize=8, title='marker size ∝ r; outlined = front')
        fig.tight_layout()
        fig.savefig(pd / f'front2d_{stat}.png', dpi=150)
        plt.close(fig)
        made.append(f'plots/front2d_{stat}.png')
    # unqualified panel
    fig, ax = plt.subplots(figsize=(7, 4))
    rows = [(m, k, v) for m in MESHES for k, v in F.get(m, {}).get('unqualified', {}).items() if isinstance(v, dict)]
    for m, k, v in rows:
        c = parse(k)
        if v.get('c') and v.get('median'):
            ax.scatter(v['c'], v['median'], color=COL[c['form'] + '|' + c['scheme']], marker='x')
            ax.annotate(f"{m}² R{c['r']}·{g(c['factor'])}", (v['c'], v['median']), xytext=(3, 3), textcoords='offset points', fontsize=7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'Unqualified or unusable configurations ({len(rows)}), never on a headline front', fontsize=9)
    ax.set_xlabel('online cost / reference A')
    ax.set_ylabel('median error (E0)')
    fig.tight_layout()
    fig.savefig(pd / 'unqualified.png', dpi=150)
    plt.close(fig)
    made.append('plots/unqualified.png')
    # B1 error vs r per family and mesh
    C = AN['configurations']
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, m in zip(axs, MESHES):
        for fam in sorted({(parse(k)['form'], parse(k)['scheme'], parse(k)['factor']) for k in C if not parse(k)['fb']}):
            ks = sorted([k for k in C if k.startswith(m + '|') and not parse(k)['fb'] and
                         (parse(k)['form'], parse(k)['scheme'], parse(k)['factor']) == fam], key=lambda k: parse(k)['r'])
            xs = [parse(k)['r'] for k in ks if C[k]['numerically_qualified']]
            ys = [C[k]['stats']['median'] for k in ks if C[k]['numerically_qualified']]
            if xs:
                ax.plot(xs, ys, marker='o', color=COL[fam[0] + '|' + fam[1]], alpha=1 if fam[2] == 1 else .55,
                        ls='-' if fam[2] == 1 else ('--' if fam[2] == 2 else ':'), label=f"{SCH[fam[0] + '|' + fam[1]]} {g(fam[2])}Δt₀")
        ax.set_yscale('log')
        ax.set_xticks([128, 256, 384, 512])
        ax.set_title(f'{m}²', fontsize=9)
        ax.set_xlabel('bank width r')
    axs[0].set_ylabel(f'median error, 38 cases\n{LBL}')
    axs[-1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(pd / 'error_vs_r.png', dpi=150)
    plt.close(fig)
    made.append('plots/error_vs_r.png')
    # B4
    B4 = AN.get('B4', {})
    keys = [k for k in B4 if k != 'TH06_control' and isinstance(B4[k], dict) and 'cases' in B4[k]]
    if keys:
        fig, ax = plt.subplots(figsize=(8, 3.8))
        for i, k in enumerate(keys):
            ps = [c['p'] for c in B4[k]['cases'] if c['p'] is not None]
            ok = [c['interpretable'] for c in B4[k]['cases'] if c['p'] is not None]
            ax.scatter([i] * len(ps), ps, c=['#0b0b0b' if o else '#d0d0d0' for o in ok], s=18)
        ax.axhspan(.8, 1.2, color='#4269d0', alpha=.08)
        ax.axhspan(1.8, 2.2, color='#3ca951', alpha=.08)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([k.replace('_', ' ') for k in keys], rotation=30, ha='right', fontsize=7)
        ax.set_ylabel('observed order p̂ (per dev6 case)')
        ax.set_title('B4: observed temporal order (black = interpretable case; bands = integer-order intervals)', fontsize=9)
        fig.tight_layout()
        fig.savefig(pd / 'order_b4.png', dpi=150)
        plt.close(fig)
        made.append('plots/order_b4.png')
    # 3D
    T3 = AN.get('threeD') or {}
    if T3.get('tested'):
        import front3d as F3
        Cd = F3.candidates()
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        xs = [v['ratio'] * v['t_A'] for v in Cd.values()]
        ys = [v['median'] for v in Cd.values()]
        ax.scatter(xs, ys, s=8, color='#c8c8c8', label='speed3d eligible (not Q2-tested)')
        for k, v in T3['tested'].items():
            if k not in Cd:
                continue
            ok = v['label'] == 'pass'
            ax.scatter(Cd[k]['ratio'] * Cd[k]['t_A'], Cd[k]['median'], s=40, marker='o' if ok else 'x',
                       color='#3ca951' if ok else '#d03030')
        front = T3['qualified_subset_front']['median']
        pts = sorted((Cd[k]['ratio'] * Cd[k]['t_A'], Cd[k]['median']) for k in front)
        if pts:
            ax.plot(*zip(*pts), color='#0b0b0b', lw=.8)
        ax.scatter([], [], marker='o', color='#3ca951', label='Q2-3D pass')
        ax.scatter([], [], marker='x', color='#d03030', label='Q2-3D not pass')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('speed3d screening time (s), H200')
        ax.set_ylabel(f'median error, 64 cases\n{LBL}')
        ax.set_title('3D: qualified subset of the tested candidates (line) — never a completeness claim', fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(pd / 'front3d.png', dpi=150)
        plt.close(fig)
        made.append('plots/front3d.png')
    return made


def main():
    made = plots()
    C = AN['configurations']
    L = []
    w = L.append
    nq = sum(v['numerically_qualified'] for v in C.values())
    w('# jcp-qualify: numerical qualification, accuracy against r and N, temporal order and the qualified error–cost front (2D/3D viscous Burgers, off-mesh ROM)')
    w('')
    w(f"**Status:** final for the priority-2 Q/B1/B4/B5 tranche, lane `jcp-qualify`. Every accuracy number is scored against the E0 second-order reference $R_{{ST}}$, whose per-case uncertainty $\\varepsilon_{{\\rm ref}}$ is a **conditional Richardson estimate** (E0 A10 cross-check amended post data; combined A10, A11 and the retracted A15 labels carry forward): all errors are **conditional E0 accuracy estimates**, not certified errors. Every number is generated by `make_report.py` from `checks/analysis.json` (`analyze.py`), `checks/smoke-accept.json` and `runs/GPU-LEDGER.json`; registered design constants (bars, step factors, rule names) are quoted from `DESIGN.md`.")
    w('')
    w(f"GPU-hours spent: {g(LED['spent_gpu_hours'])} of the {g(LED['cap_gpu_hours'])} cap ({len(LED['jobs'])} jobs; `runs/GPU-LEDGER.json`).")
    w('')
    w('## Summary')
    w('')
    w(f"- **Q1/Q2 (2D):** {nq} of {len(C)} configurations numerically qualified (Q1 solver + Q2 quadrature; P1's certified configurations are cited).")
    for m in MESHES:
        f = AN.get('fronts', {}).get(m, {})
        if 'members' in f:
            w(f"- **Front at {m}²:** {len(f['fronts']['median|c'])} median-front and {len(f['fronts']['worst|c'])} worst-front members among {len(f['candidates'])} qualified candidates; {len(f['unqualified'])} configurations unqualified or without usable timing.")
    th = AN.get('B4', {}).get('TH06_control', {})
    w(f"- **B4 TH06 control:** label {th.get('label')}, interpretable {th.get('interpretable')}, agrees with smoke {th.get('agrees_with_smoke')} → {'passed' if th.get('passed') else 'FAILED (B4 labels blocked)'}.")
    if AN.get('threeD'):
        T3 = AN['threeD']
        npass = sum(v['label'] == 'pass' for v in T3['tested'].values())
        w(f"- **3D Q2:** {npass} of {len(T3['tested'])} tested speed3d configurations pass their own successor-rule check; control fired: {T3.get('control_fired')}; {len(T3['open_candidates'])} untested candidates remain open (result label: {T3['label']}).")
    if FR:
        w(f"- **Deliverable:** `runs/FRONT-2D.json` (sha256 `{FRSHA[:16]}…`), frozen after the results audit.")
    w('')
    w('```mermaid')
    w('flowchart LR')
    w('  D[DESIGN + A1-A5, 5 Codex design audits]:::doc --> K[code + 7 Codex code/job audits]:::doc --> S[real-size smokes smkA smkB smk3]:::job')
    w('  S --> A[accuracy jobs acc256 acc1024 acc4096: Q1 Q2 Q3 per configuration]:::job --> T[per-mesh timing jobs time256 time1024 time4096]:::job')
    w('  S --> O[order job: B4]:::job')
    w('  S --> Q[q3d: Q2-3D of speed3d front members]:::job')
    w('  A --> B1[B1: H-acc-N, H-acc-r]:::doc')
    w('  T --> F[B5 fronts + FRONT-2D.json]:::doc')
    w('  classDef doc fill:#e5e7eb,stroke:#374151;')
    w('  classDef job fill:#dbeafe,stroke:#1d4ed8;')
    w('```')
    w('')
    w('## Jobs and budget')
    w('')
    w('| attempt | job | state | elapsed GPU-h | time limit (h) |')
    w('|---|---|---|---|---|')
    for j in LED['jobs']:
        w(f"| {j['attempt']} | {j.get('job_id')} | {j['state']} | {g(j.get('elapsed_hours'))} | {g(j['time_limit_hours'])} |")
    w('')
    w('## Q1–Q3: numerical qualification of every 2D configuration')
    w('')
    w(f'Errors over the 38 dev6 + val32 cases ({LBL}). Q1 = solver qualification (LSPG: certified stationarity; GAL: certified Galerkin residual, DESIGN A1.3); Q2 = successor-rule quadrature check on 16 cases; "repro" = registered reproduction gate against combined\'s archive ($1024^2$, R128/R384/R512).')
    w('')
    w('| mesh | configuration | source | Q1 | Q2 | repro | qualified | n | median e | worst e | Q3 median cond(J) |')
    w('|---|---|---|---|---|---|---|---|---|---|---|')
    rg = AN.get('reproduction_combined', {})
    for k in sorted(C, key=lambda k: (int(parse(k)['mesh']), parse(k)['form'], parse(k)['scheme'], parse(k)['factor'], parse(k)['r'])):
        v = C[k]
        q3 = (v.get('q3') or {}).get('J_cond_median')
        rep = rg.get(k, {}).get('passed') if isinstance(rg.get(k), dict) else None
        w(f"| {parse(k)['mesh']}² | {name(k)} | {'P1 (cited)' if 'P1' in v['source'] else 'lane'} | {'pass' if v['Q1'].get('pass') else 'fail'} | {v['Q2'].get('label')} | {'—' if rep is None else ('pass' if rep else 'FAIL')} | {'yes' if v['numerically_qualified'] else 'no'} | {v['stats']['n']} | {pct(v['stats']['median'])} | {pct(v['stats']['worst'])} | {g(q3)} |")
    w('')
    w('## B1: error against r and N')
    w('')
    w('H-acc-N (AM-4 + P1 A1.9; P256 on 38 cases, P4096 on the registered 16-case subset, trend on that subset):')
    w('')
    w('| configuration family | outcome | P4096 on 38 (sensitivity) |')
    w('|---|---|---|')
    for k, v in sorted(AN['B1']['h_acc_n'].items()):
        sen = (v.get('sensitivity_P4096_38') or {}).get('outcome')
        w(f"| {esc(k)} | {v.get('outcome')} | {g(sen)} |")
    w('')
    w('H-acc-r per mesh and family (majority improvement on resolved cases; ties within $2\\varepsilon_{\\rm ref}$; never "every case"):')
    w('')
    w('| mesh / family | 128→256 | 256→384 | 384→512 | range 128–384 | range 128–512 |')
    w('|---|---|---|---|---|---|')
    for k, v in sorted(AN['B1']['h_acc_r'].items()):
        cell = lambda t: (f"{v[t].get('outcome')} ({v[t].get('improve', '—')}/{v[t].get('worsen', '—')}/{v[t].get('tie', '—')}/{v[t].get('unqualified', '—')})" if isinstance(v.get(t), dict) else '—')
        w(f"| {esc(k)} | {cell('128->256')} | {cell('256->384')} | {cell('384->512')} | {v.get('range_128_384')} | {v.get('range_128_512')} |")
    w('')
    w('Counts in brackets: improve / worsen / tie / unqualified cases. P1\'s own verdicts for its two Δt₀ families are cited in `checks/analysis.json` (`B1.p1_cited`; its $4096^2$ transitions are on the 16-case subset).')
    w('')
    w('## B4: observed temporal order')
    w('')
    w('| scheme, r | label | interpretable cases | median p̂ | range |')
    w('|---|---|---|---|---|')
    for k, v in AN.get('B4', {}).items():
        if isinstance(v, dict) and 'label' in v and isinstance(v['label'], dict):
            lab = v['label']
            w(f"| {esc(k.replace('_', ' '))} | {lab['label']} | {lab['interpretable']} | {g(lab.get('p_median'))} | {rng(lab.get('p_range'))} |")
    w('')
    w('An "order 2" label is a **finite-range observed order** over $\\Delta t_0\\dots\\Delta t_0/4$ against a same-form anchor at $\\Delta t_0/64$; for the LSPG CN family the jcp-time2 analysis predicts asymptotic order 1, which this measurement does not contradict or confirm.')
    w('')
    w('## B5: the qualified error–cost front per mesh')
    w('')
    w('Cost = median over valid fused triplets of (full online query: input sampling, evolution, full-field decode) / mean of the two flanking runs of the reference A (r = 384, $1024^2$, LSPG-CN-R, $\\Delta t_0$), one H200 job per mesh. A candidate needs both numerical labels (accuracy job and timing job), usable timing and all 38 cases. Dominance as DESIGN §6 (5 % cost band, error margin $\\max(2\\varepsilon, 2\\%)$).')
    w('')
    for m in MESHES:
        f = AN.get('fronts', {}).get(m, {})
        if 'members' not in f:
            w(f'- {m}²: {f.get("status")}')
            continue
        w(f'**{m}²**')
        w('')
        w('| configuration | front | cost / A | evolution / A | median e | worst e | role |')
        w('|---|---|---|---|---|---|---|')
        for k in sorted(f['candidates'], key=lambda k: f['candidates'][k]['c']):
            v = f['candidates'][k]
            fr = ', '.join(x.split('|')[0] for x in ('median|c', 'worst|c') if k in f['fronts'][x]) or '—'
            w(f"| {name(k)} | {fr} | {g(v['c'])} | {g(v['c_ev'])} | {pct(v['median'])} | {pct(v['worst'])} | {AN.get('roles', {}).get(k, {}).get('role', '—')} |")
        w('')
        if f['unqualified']:
            w('Not on any headline front (separate panel): ' + '; '.join(
                f"{name(k)} ({'no accuracy entry' if not isinstance(v, dict) else ', '.join(x for x, ok in (('accuracy label', v['acc_label']), ('timing label', v['timing_label']), ('usable timing', v['usable']), ('38 cases', v['complete38'])) if not ok)})"
                for k, v in f['unqualified'].items()))
            w('')
    w('## 3D: Q2 of speed3d\'s front members and winners')
    w('')
    T3 = AN.get('threeD') or {}
    if T3:
        w(f"Contract: {T3['contract']}. Label: **{T3['label']}**. Q1 is speed3d's eligibility (cited).")
        w('')
        w('| configuration | Q2-3D | worst q vs q⁺ distance | ρ max | q⁺ vs q⁺⁺ | reproduction |')
        w('|---|---|---|---|---|---|')
        for k, v in T3['tested'].items():
            rp = v.get('reproduction') or {}
            w(f"| {esc(k)} | {v['label']} | {g(v['worst_dist_qp'])} | {g(v['rho_max'])} | {g(v['worst_dist_qp_qpp'])} | {g(rp.get('worst'))} |")
        w('')
        c = T3.get('control', {})
        w(f"Under-resolved control (W512 lat8192 → lat16384): label {c.get('label')}, worst distance {g(c.get('worst_dist_qp'))}, fired {c.get('passed')}.")
        w('')
        w(f"Qualified subset front (median): {', '.join('`' + x + '`' for x in T3['qualified_subset_front']['median']) or 'none'}; (worst): {', '.join('`' + x + '`' for x in T3['qualified_subset_front']['worst']) or 'none'}. Open candidates: {len(T3['open_candidates'])}.")
        w('')
    w('## Controls')
    w('')
    req = {k: v for k, v in SMK.items() if v.get('required')}
    w(f"Real-size smoke acceptance (`checks/smoke-accept.json`): {sum(v['passed'] for v in req.values())} of {len(req)} required controls behave as registered.")
    w('')
    w('## Deviations and retractions')
    w('')
    w('- **GAL certificate (DESIGN A1.3, A2.5):** for GAL-CN the brief\'s least-squares stationarity ratio was replaced by a registered relative Galerkin residual bar ($10^{-8}$), because the ratio is not a convergence measure for a square root problem; the literal ratio is reported, without a verdict.')
    w('- **3D front:** the result is a qualified subset of the tested candidates, never a complete qualified front (A2.1).')
    w('')
    w('## Plots')
    w('')
    for p in made:
        w(f'![{p}]({p})')
    w('')
    w('## Glossary')
    w('')
    for t, d in [
        ('r (bank width)', 'number of ordered bank coefficients solved for (R′); 128, 256, 384, 512 here.'),
        ('N, mesh', 'points per axis of the full-order grid on which the input is sampled and the output decoded (256², 1024², 4096²).'),
        ('Δt₀, factor', 'Δt₀ = 0.005; a configuration runs at factor × Δt₀ (1, 2, and the registered larger step: 2.5 for LSPG-BE and LSPG-CN-R, 5 for GAL-CN).'),
        ('LSPG-BE, LSPG-CN-R, GAL-CN, TH06', 'least-squares Petrov–Galerkin backward Euler; LSPG Crank–Nicolson with two backward-Euler start-up steps; Galerkin Crank–Nicolson; θ-method θ = 0.6 (a first-order control).'),
        ('E0 R_ST, ε_ref', 'second-order space–time reference (primary); its per-case uncertainty, a conditional Richardson estimate, not a bound.'),
        ('e, median e, worst e', 'e = max over the five evolved outputs of ‖u − R‖ / ‖R(t₀)‖ on the 257² scoring lattice; median / worst over the 38 cases.'),
        ('Q1', 'solver qualification: every run valid, ≤ 1 % of steps not certified (LSPG stationarity ratio ≤ 10⁻³; GAL relative Galerkin residual ≤ 10⁻⁸), distance to a tight solve ≤ 2.5·10⁻⁴ on every case, tight/tighter ≤ 2.5·10⁻⁵ on dev6.'),
        ('Q2', 'quadrature qualification on 16 cases: replay with the next finer rule within 2.5·10⁻⁴, own-state ρ ≤ 0.116, and the finer rule converged against the next one.'),
        ('Q3, cond(J)', 'conditioning: condition number of the scheme\'s step Jacobian at outputs 1, 3, 5 of the dev6 trajectories.'),
        ('ρ', 'relative change of the tested advection vector between a rule and its successor at the configuration\'s own output states.'),
        ('numerically qualified', 'Q1 and Q2 pass (and the reproduction gate where one applies).'),
        ('repro', 'registered gate: the lane\'s run equals combined\'s archived coefficients to ≤ 10⁻¹⁰ after numerical-manifest equality.'),
        ('cost / A, evolution / A', 'paired timing ratios against the reference configuration A in the same job and GPU (full query; evolution only).'),
        ('front (median, worst)', 'qualified candidates not dominated by any other at the same mesh under the DESIGN §6 rule, for the median or the worst error.'),
        ('role', 'pre-declared descriptive label against the same-r LSPG-BE at Δt₀: fast-equal-accuracy (cost ≤ 0.6, ties or beats both errors) or accurate-equal-cost (cost ≤ 1.1, lower median).'),
        ('H-acc-N, H-acc-r', 'mesh-independence (equivalence test of per-case differences) and rank-improvement (majority of resolved cases) hypotheses of P1, applied to each configuration family.'),
        ('improve / worsen / tie / unqualified', 'per case in an r transition: lower / higher error beyond 2ε_ref, within 2ε_ref, or not qualified on both rungs.'),
        ('p̂, interpretable', 'least-squares slope of log error against log step over Δt₀, Δt₀/2, Δt₀/4 versus a same-form anchor at Δt₀/64; a case is interpretable if the asymptotic, anchor, solver and quadrature checks pass.'),
        ('q, q⁺, q⁺⁺', 'a configuration\'s quadrature rule, its successor and the check rule (next entries of the rule ladder).'),
        ('open candidates (3D)', 'untested 3D configurations not dominated (median and worst) by any Q2-passed configuration.'),
        ('fallback rule', 'the registered one-level replacement of a rule that failed Q2 by its successor.'),
    ]:
        w(f'- **{t}**: {d}')
    (HERE / 'report.md').write_text('\n'.join(L) + '\n')
    print('report.md', len(L), 'lines;', len(made), 'plots')


if __name__ == '__main__':
    main()
