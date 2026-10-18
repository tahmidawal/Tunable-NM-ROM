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


def olab(x):
    return str(x).replace('dt0..dt0/4', '$\\Delta t_0\\dots\\Delta t_0/4$')


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
                    i_ = sorted(front, key=lambda kk: cand[kk]['c']).index(k)
                    ax.annotate(f"R{c['r']}·{g(c['factor'])}", (v['c'], v[stat]), xytext=(8, 7 if i_ % 2 == 0 else -12) if i_ % 3 != 2 else (-34, 4),
                                textcoords='offset points', fontsize=6.5,
                                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=.8))
            pts = sorted((cand[k]['c'], cand[k][stat]) for k in front)
            if pts:
                ax.plot(*zip(*pts), color='#0b0b0b', lw=.8, zorder=1)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.margins(.12)
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
            ax.scatter(v['c'], v['median'], color=COL[c['form'] + '|' + c['scheme']], marker={'256': 'x', '1024': '+', '4096': '1'}[m], s=40)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'Unqualified or unusable configurations ({len(rows)}), never on a headline front', fontsize=9)
    ax.set_xlabel('online cost / reference A')
    ax.set_ylabel(f'median error, 38 cases\n{LBL}')
    for m, mk in (('256', 'x'), ('1024', '+'), ('4096', '1')):
        ax.scatter([], [], marker=mk, color='#555555', label=f'{m}²')
    for key, col in COL.items():
        ax.scatter([], [], marker='s', color=col, label=SCH[key])
    ax.legend(frameon=False, fontsize=7, ncol=2)
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
    # error vs N per r (LSPG-CN-R and GAL-CN at dt0, BE at dt0), qualified only
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (form, sc) in zip(axs, (('LSPG', 'BE'), ('LSPG', 'CNR'), ('GAL', 'CN'))):
        for r, col in zip((128, 256, 384, 512), ('#4269d0', '#efb118', '#3ca951', '#ff725c')):
            for f, ls in ((1., '-'), (2., '--')):
                xs, ys = [], []
                for m in MESHES:
                    v = C.get(f'{m}|R{r}|{form}|{sc}|{f:g}')
                    if v and v['numerically_qualified']:
                        xs.append(int(m))
                        ys.append(v['stats']['median'])
                if xs:
                    ax.plot(xs, ys, marker='o', ls=ls, color=col, label=f'R{r} {g(f)}Δt₀')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks([256, 1024, 4096])
        ax.set_xticklabels(['256²', '1024²', '4096²'])
        ax.set_title(SCH[form + '|' + sc], fontsize=9)
        ax.set_xlabel('mesh N')
    axs[0].set_ylabel(f'median error, 38 cases\n{LBL}')
    axs[-1].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(pd / 'error_vs_N.png', dpi=150)
    plt.close(fig)
    made.append('plots/error_vs_N.png')
    # conditioning vs r (Q3, 1024^2)
    fig, ax = plt.subplots(figsize=(7, 4))
    for (form, sc), mk in ((('LSPG', 'BE'), 'o'), (('LSPG', 'CNR'), 's'), (('GAL', 'CN'), '^')):
        for f, ls in ((1., '-'), (2., '--')):
            xs, ys = [], []
            for r in (128, 256, 384, 512):
                v = C.get(f'1024|R{r}|{form}|{sc}|{f:g}')
                if v and v.get('q3'):
                    xs.append(r)
                    ys.append(v['q3']['J_cond_median'])
            if xs:
                ax.plot(xs, ys, marker=mk, ls=ls, color=COL[form + '|' + sc], label=f"{SCH[form + '|' + sc]} {g(f)}Δt₀")
    ax.set_yscale('log')
    ax.set_xticks([128, 256, 384, 512])
    ax.set_xlabel('bank width r')
    ax.set_ylabel('median cond(step Jacobian), dev6, outputs 1, 3, 5')
    ax.set_title('Q3 conditioning at 1024² (lane configurations; P1 Δt₀ LSPG values in P1 report)', fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(pd / 'conditioning_vs_r.png', dpi=150)
    plt.close(fig)
    made.append('plots/conditioning_vs_r.png')
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
            cnt = lambda st: ', '.join(f"{SCH[x]} {sum(1 for k in f['fronts'][st] if parse(k)['form'] + '|' + parse(k)['scheme'] == x)}" for x in SCH)
            lo = min(f['fronts']['median|c'], key=lambda k: f['candidates'][k]['median'])
            w(f"  (Errors: conditional E0 estimates.) Median front by scheme: {cnt('median|c')}; worst front: {cnt('worst|c')}. Most accurate median-front member: {name(lo)} at {pct(f['candidates'][lo]['median'])} median, cost {g(f['candidates'][lo]['c'])}× A.")
    for k, v in AN.get('B4', {}).items():
        if k != 'TH06_control' and isinstance(v, dict) and isinstance(v.get('label'), dict):
            w(f"- **B4 {k.replace('|', ' ').replace('_', ' ').replace('CNR', 'CN-R').replace('LSPG ', 'LSPG-').replace('GAL ', 'GAL-')}:** {olab(v['label']['label'])} (median p̂ {g(v['label'].get('p_median'))}, {v['label']['interpretable']} interpretable cases).")
    from collections import Counter
    hn = Counter(v.get('outcome') for v in AN['B1']['h_acc_n'].values())
    w(f"- **B1 H-acc-N** over {len(AN['B1']['h_acc_n'])} (r, family) tests: " + ', '.join(f'{k} {n}' for k, n in sorted(hn.items(), key=lambda x: str(x[0]))) + '.')
    r1 = Counter(v.get('range_128_384') for v in AN['B1']['h_acc_r'].values())
    r2 = Counter(v.get('range_128_512') for v in AN['B1']['h_acc_r'].values())
    w(f"- **B1 H-acc-r** over {len(AN['B1']['h_acc_r'])} (mesh, family) cells: range 128–384 " + ', '.join(f'{k} {n}' for k, n in sorted(r1.items(), key=lambda x: str(x[0]))) +
      '; range 128–512 ' + ', '.join(f'{k} {n}' for k, n in sorted(r2.items(), key=lambda x: str(x[0]))) + '.')
    rl = AN.get('roles', {})
    w(f"- **Roles (descriptive):** {sum('fast' in v['role'] for v in rl.values())} fast-equal-accuracy and {sum('accurate' in v['role'] for v in rl.values())} accurate-equal-cost labels among {len(rl)} qualified configurations with usable timing.")
    th = AN.get('B4', {}).get('TH06_control', {})
    w(f"- **B4 TH06 control:** label {th.get('label')}, interpretable {th.get('interpretable')}, agrees with smoke {th.get('agrees_with_smoke')} → {'passed' if th.get('passed') else 'FAILED (B4 labels blocked)'}.")
    if AN.get('threeD'):
        T3 = AN['threeD']
        npass = sum(v['label'] == 'pass' for v in T3['tested'].values())
        w(f"- **3D Q2:** {npass} of {len(T3['tested'])} tested speed3d configurations pass their own successor-rule check over {len(T3['rounds'])} rounds; control fired: {T3.get('control_fired')}; round limit reached: {T3.get('round_limit_reached')} ({len(T3.get('exposed_untested_after_last_round') or [])} untested candidates still exposed on the next front); {len(T3['open_candidates'])} untested candidates remain open (result label: {T3['label']}).")
    if FR:
        w(f"- **Deliverable:** `runs/FRONT-2D.json` (sha256 `{FRSHA[:16]}…`), frozen after the results audit.")
    w('')
    nd = len(list((HERE / 'audits').glob('codex-design-*.md')))
    nc = len(list((HERE / 'audits').glob('codex-code-*.md'))) + len(list((HERE / 'audits').glob('codex-job-*.md')))
    nr = len(list((HERE / 'audits').glob('codex-results-*.md')))
    w('```mermaid')
    w('flowchart LR')
    w(f'  D[DESIGN + A1-A6, {nd} Codex design audits]:::doc --> K[code + {nc} Codex code/job audits]:::doc --> S[real-size smokes smkA smkB smk3]:::job')
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
    w('| configuration family | outcome | P4096 on 38 (exploratory sensitivity, not a verdict) |')
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
            w(f"| {esc(k.replace('_', ' '))} | {olab(lab['label'])} | {lab['interpretable']} | {g(lab.get('p_median'))} | {rng(lab.get('p_range'))} |")
    w('')
    w('An "order 2" label is a **finite-range observed order** over $\\Delta t_0\\dots\\Delta t_0/4$ against a same-form anchor at $\\Delta t_0/64$; for the LSPG CN family the jcp-time2 analysis predicts asymptotic order 1, which this measurement does not contradict or confirm.')
    w('')
    w('## B5: the qualified error–cost front per mesh')
    w('')
    w('Cost = median over valid fused triplets of (full online query: input sampling, evolution, full-field decode) / mean of the two flanking runs of the reference A (r = 384, $1024^2$, LSPG-CN-R, $\\Delta t_0$), one H200 job per mesh. A candidate needs both numerical labels (accuracy job and timing job), usable timing and all 38 cases. Dominance as DESIGN §6: $X\\succ Y$ iff $c_X\\le1.05\\,c_Y$, $e_X\\le e_Y+\\tau_{XY}$ and ($c_X<c_Y/1.05$ or $e_X<e_Y-\\tau_{XY}$), with $\\tau_{XY}=\\max\\big(2\\max(\\bar\\varepsilon_X,\\bar\\varepsilon_Y),\\,0.02\\max(e_X,e_Y)\\big)$. Errors are conditional E0 estimates.')
    w('')
    for m in MESHES:
        f = AN.get('fronts', {}).get(m, {})
        if 'members' not in f:
            w(f'- {m}²: {f.get("status")}')
            continue
        w(f'**{m}²** (errors: conditional E0 $R_{{ST}}$ estimates)')
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
        w(f"Round limit reached: {T3.get('round_limit_reached')}; untested candidates exposed on the next front after the last round: {len(T3.get('exposed_untested_after_last_round') or [])} (listed in `checks/analysis.json`, `threeD.exposed_untested_after_last_round`).")
        w('')
        w(f"Qualified subset front (median): {', '.join('`' + x + '`' for x in T3['qualified_subset_front']['median']) or 'none'}; (worst): {', '.join('`' + x + '`' for x in T3['qualified_subset_front']['worst']) or 'none'}. Open candidates: {len(T3['open_candidates'])}.")
        w('')
    w('## Controls')
    w('')
    req = {k: v for k, v in SMK.items() if v.get('required')}
    w(f"Real-size smoke acceptance (`checks/smoke-accept.json`): {sum(v['passed'] for v in req.values())} of {len(req)} required controls behave as registered.")
    w('')
    w('## Corrections to earlier reports (stated plainly)')
    w('')
    for m in MESHES:
        for r in (128, 256):
            k = f'{m}|R{r}|LSPG|CNR|2'
            v = C.get(k)
            if v and not v['Q1'].get('pass'):
                w(f"- **{m}², R{r} LSPG-CN-R at 2Δt₀** fails Q1 here: worst distance to the tight solve {g(v.get('worst_dist_tight'))} on {v.get('worst_dist_tight_case')} (bar 2.5·10⁻⁴), although its Q2 label is {v['Q2'].get('label')}. jcp-combined's finalist table (`worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/report.md`, finalist and fallback tables) calls the R128 and R256-fallback \"fast equal accuracy\" arms of this scheme and step qualified under its own finalist criterion; under the every-case Q1 contract of V4-2 they are **not** numerically qualified. Combined's measured errors and timings are not retracted; its qualification label for these arms is.")
    T3 = AN.get('threeD') or {}
    for k in ('M2_512|gl24|fs|BDF2|0.0125|K1', 'M2_512|gl24|qn|CN|0.016666667|k4'):
        v = (T3.get('tested') or {}).get(k)
        if v:
            w(f"- **speed3d H2 winner `{k}`**: Q2-3D {v['label']} (successor-rule distance {g(v['worst_dist_qp'])} > 2.5·10⁻⁴, ρ {g(v['rho_max'])}). speed3d already labels reduced rules \"empirically validated surrogates, not quadrature-equivalent\" (`worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/report.md`); its speed-ups stand as measured, but the winners must not be described as quadrature-qualified.")
    w('- **P1:** no numerical retraction; its $4096^2$ R128/R256 results are on its 16-case subset (extended here to 38 cases for the front).')
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
        ('$r$ (bank width)', "number of ordered bank coefficients solved for ($R'$); 128, 256, 384, 512 here."),
        ('$N$, mesh', 'points per axis of the full-order grid on which the input is sampled and the output decoded ($256^2$, $1024^2$, $4096^2$).'),
        (r'$\Delta t_0$, factor', r'$\Delta t_0=0.005$; a configuration runs at factor $\times\,\Delta t_0$ (1, 2, and the registered larger step: 2.5 for LSPG-BE and LSPG-CN-R, 5 for GAL-CN).'),
        ('LSPG-BE, LSPG-CN-R, GAL-CN, TH06, BDF2', r'least-squares Petrov–Galerkin backward Euler; LSPG Crank–Nicolson with two backward-Euler start-up steps; Galerkin Crank–Nicolson; $\theta$-method with $\theta=0.6$ (a first-order control); second-order backward differentiation (3D only).'),
        (r'E0 $R_{ST}$, $\varepsilon_{\rm ref}$, $\bar\varepsilon_X$', r'second-order space–time reference (primary); its per-case uncertainty, a conditional Richardson estimate, not a bound; $\bar\varepsilon_X$ = median over cases of $\varepsilon_{\rm ref}$ for configuration $X$.'),
        ('$e$, median $e$, worst $e$', r'$e=\max_{k=1..5}\lVert u(t_k)-R(t_k)\rVert/\lVert R(t_0)\rVert$ on the $257^2$ scoring lattice (2D); median / worst over the 38 cases.'),
        ('Q1', r'solver qualification: every run valid, at most 1 % of steps not certified (LSPG stationarity ratio $\gamma\le10^{-3}$; GAL relative Galerkin residual $\eta\le10^{-8}$), distance to a tight solve $\le2.5\times10^{-4}$ on every case, tight/tighter $\le2.5\times10^{-5}$ on dev6.'),
        ('Q2 (2D)', r'quadrature qualification on 16 cases: replay with the next finer rule within $2.5\times10^{-4}$, own-state $\rho\le0.116$, and the finer rule converged against the next one.'),
        (r'Q3, $\mathrm{cond}(J)$', "conditioning: condition number of the scheme's step Jacobian at outputs 1, 3, 5 of the dev6 trajectories."),
        (r'$\rho$', "relative change of the tested advection vector between a rule and its successor at the configuration's own output states."),
        ('numerically qualified', 'Q1 and Q2 pass (and the reproduction gate where one applies).'),
        ('repro', r"registered gates: 2D, the lane's run equals combined's archived coefficients to $\le10^{-10}$ after numerical-manifest equality; 3D, the rerun with the configuration's own rule equals speed3d's archive to $\le10^{-10}$."),
        ('cost / A, evolution / A', 'paired timing ratios against the reference configuration A in the same job and GPU (full query; evolution only).'),
        ('front (median, worst)', 'qualified candidates not dominated by any other at the same mesh under the DESIGN §6 rule, for the median or the worst error.'),
        ('role', r'pre-declared descriptive label against the same-$r$ LSPG-BE at $\Delta t_0$: fast-equal-accuracy (cost ratio $\le0.6$, ties or beats both errors) or accurate-equal-cost (cost ratio $\le1.1$, lower median).'),
        ('H-acc-N, H-acc-r', 'mesh-independence (equivalence test of per-case differences) and rank-improvement (majority of resolved cases) hypotheses of P1, applied to each configuration family.'),
        ('improve / worsen / tie / unqualified', r'per case in an $r$ transition: lower / higher error beyond $2\varepsilon_{\rm ref}$, within $2\varepsilon_{\rm ref}$, or not qualified on both rungs.'),
        (r'$\hat p$, interpretable', r'least-squares slope of $\log e$ against $\log\Delta t$ over $\Delta t_0,\Delta t_0/2,\Delta t_0/4$ versus a same-form anchor at $\Delta t_0/64$; a case is interpretable if the asymptotic, anchor, solver and quadrature checks pass.'),
        (r'$q$, $q^+$, $q^{++}$', "a configuration's quadrature rule, its successor and the check rule (next entries of the rule ladder)."),
        ('open candidates (3D)', 'untested 3D configurations not dominated (on both the median and the worst panel) by any Q2-passed configuration.'),
        ('fallback rule', 'the registered one-level replacement of a rule that failed Q2 by its successor.'),
        ('n', 'number of E0-eligible cases in a statistic (38 here for every configuration).'),
        ('dev6, val32, 16-case subset', 'the 6 development and 32 validation cases of the 2D cohorts; the subset is dev6 plus val32 cases 0–9.'),
        ('P256, P4096, sensitivity (P4096 on 38)', r'H-acc-N case pairs: $256^2$ vs $1024^2$ on 38 cases; $4096^2$ vs $1024^2$ on the registered 16-case subset. The sensitivity column re-runs the same procedure with P4096 on all 38 cases: its outcomes are exploratory, not registered verdicts.'),
        ('3D settings (M2_512, W256 … W1024), $C_{\\rm old}$', r'speed3d bank / width settings: the M2 bank at $r=512$, and prefixes of width 256–1024 of the W1024 bank, at $65^3$; $C_{\rm old}$ = speed3d\'s reference configuration (M2_512, gl32, fixed-sweep BE, $\Delta t=0.01$) against which its screening times are ratios.'),
        ('3D rules (glN, latN)', r'Gauss tensor rules with $N^3$ points; rank-1 lattice rules with $N$ points.'),
        ('3D solver specs (fs, qn, K, k)', 'fixed-sweep Gauss–Newton (fs) with K full sweeps per step; quasi-Newton (qn) with Jacobian refresh every k iterations; followed by scheme (BE, CN, BDF2) and step.'),
        ('3D error and distance', r'max over the five evolved outputs of the relative $\ell^2$ difference on the $63^3$ scoring lattice, normalised by the initial field; errors against E0 $R_{ST}$ over the 64 cases of cohort 923801.'),
        ('Q2-3D', r'successor-rule replay within $2.5\times10^{-4}$ on all 64 cases, own-state $\rho\le0.116$, and the successor converged against the next rule on cases 0–7.'),
        ('round limit, exposed', 'Q2-3D ran in at most three rounds; "exposed" candidates are untested configurations on the front of the untested-plus-passed set after the last round.'),
    ]:
        w(f'- **{t}**: {d}')
    (HERE / 'report.md').write_text('\n'.join(L) + '\n')
    print('report.md', len(L), 'lines;', len(made), 'plots')


if __name__ == '__main__':
    main()
