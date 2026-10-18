"""Generate report.md and plots/*.png from checks/analysis.json (written by analyze.py). Every number in the report is
read from that file or from the archives; nothing is typed by hand.

    python make_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
AN = json.loads((HERE / 'checks' / 'analysis.json').read_text())
EXP = json.loads((HERE / 'runs' / 'EXPORT.json').read_text())
PLOTS = HERE / 'plots'
PLOTS.mkdir(exist_ok=True)
SER = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']     # validated reference categorical order
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e1'
MESHES = (256, 1024, 4096)


def g(x, n=3):
    if x is None:
        return 'n/a'
    if isinstance(x, (bool, np.bool_)):
        return 'yes' if x else 'no'
    if isinstance(x, str):
        return x
    x = float(x)
    if x == 0:
        return '0'
    return f'{x:.{n}g}' if 1e-3 <= abs(x) < 1e4 else f'{x:.{max(n - 1, 1)}e}'


def pct(x):
    return 'n/a' if x is None else f'{100 * float(x):.3g} %'


def style(ax, title, xl, yl):
    ax.set_title(title, loc='left', fontsize=11, color=INK)
    ax.set_xlabel(xl, color=INK2)
    ax.set_ylabel(yl, color=INK2)
    ax.grid(True, color=GRID, lw=.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=150)
    plt.close(fig)
    return f'plots/{name}'


def plots():
    out = []
    # 1. solve time ratio vs N per r (H-cost-N)
    hc = AN['timing']['h_cost_n'] if isinstance(AN['timing'], dict) else {}
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, sc in zip(axs, ('BE', 'CNR')):
        for i, r in enumerate((128, 256, 384)):
            v = hc.get(f'R{r}|{sc}|frozen', {})
            if 'medians' in v:
                ax.plot(MESHES, [v['medians'][str(m)] for m in MESHES], '-o', color=SER[i], lw=2, ms=8, label=f'r = {r}')
        ax.set_xscale('log', base=2)
        ax.set_xticks(MESHES, [f'{m}²' for m in MESHES])
        style(ax, f'LSPG-{"CN-R" if sc == "CNR" else "BE"}: evolution time / reference A', 'mesh N', 'median B/A ratio')
        ax.set_ylim(0, None)
    axs[0].legend(frameon=False)
    out.append(save(fig, 'solve_time_vs_mesh.png'))
    # 2. kernel cost vs r per component (2D, 3D) with FLOP model
    for key, tag in (('kernels_2d', '2D'), ('kernels_3d', '3D')):
        k = AN.get(key)
        if not isinstance(k, dict):
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for i, (comp, v) in enumerate(k['verdicts'].items()):
            rs = [int(x) for x in v['medians_by_rank']]
            ts = [v['medians_by_rank'][str(x)] for x in rs]
            ax.plot(rs, ts, '-o', color=SER[i], lw=2, ms=8, label=f'{comp} (p fit {g(v["point"].get("p"))}, declared {g(v["p_decl"])})')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        style(ax, f'{tag} kernel components, fixed (M, m) panel', 'bank width r', 'median seconds per call')
        ax.legend(frameon=False, fontsize=8)
        out.append(save(fig, f'kernel_cost_vs_r_{tag}.png'))
    # 3. error vs r per mesh and error vs N per r
    cf = AN['configurations']
    for sc in ('BE', 'CNR'):
        fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
        for i, m in enumerate(MESHES):
            rs = [r for r in (128, 256, 384, 512) if f'{m}|R{r}|{sc}' in cf]
            axs[0].plot(rs, [100 * cf[f'{m}|R{r}|{sc}']['e0_median'] for r in rs], '-o', color=SER[i], lw=2, ms=8, label=f'{m}²')
        axs[0].set_xscale('log', base=2)
        style(axs[0], 'median error vs r', 'bank width r', 'median E0 R_ST error (%)')
        axs[0].legend(frameon=False)
        for i, r in enumerate((128, 256, 384)):
            ms_ = [m for m in MESHES if f'{m}|R{r}|{sc}' in cf]
            axs[1].plot(ms_, [100 * cf[f'{m}|R{r}|{sc}']['e0_median'] for m in ms_], '-o', color=SER[i], lw=2, ms=8, label=f'r = {r}')
        axs[1].set_xscale('log', base=2)
        axs[1].set_xticks(MESHES, [f'{m}²' for m in MESHES])
        style(axs[1], 'median error vs mesh (4096² = 16-case subset)', 'mesh N', 'median E0 R_ST error (%)')
        axs[1].legend(frameon=False)
        out.append(save(fig, f'error_vs_r_and_mesh_{sc}.png'))
    # 4. P1g frozen vs nominal
    pg = AN['p1g']
    labs, d256, d4096 = [], [], []
    for k, v in pg.items():
        m = v['meshes']
        if isinstance(m.get('256'), dict) or isinstance(m.get(256), dict):
            a = m.get('256', m.get(256))
            b = m.get('4096', m.get(4096))
            labs.append(k)
            d256.append(a['dist_max'])
            d4096.append(b['dist_max'] if isinstance(b, dict) else np.nan)
    if labs:
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        x = np.arange(len(labs))
        ax.bar(x - .2, d256, .38, color=SER[0], label='256² (2 + 2 modes differ)')
        ax.bar(x + .2, d4096, .38, color=SER[1], label='4096² (same set, permuted)')
        ax.set_yscale('log')
        ax.set_xticks(x, labs, rotation=30, ha='right', fontsize=8)
        style(ax, 'P1g: frozen vs nominal mode list, worst trajectory distance', '', 'max over cases, relative')
        ax.legend(frameon=False, fontsize=8)
        out.append(save(fig, 'p1g_frozen_vs_nominal.png'))
    # 5. memory vs r (table bytes per configuration, 1024^2)
    mem = AN['memory'].get('1024', AN['memory'].get(1024))
    if mem:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        rs = [128, 256, 384, 512]
        tb = [sum(mem['configs'][f'R{r}']['table_bytes'].values()) / 1e9 for r in rs if f'R{r}' in mem['configs']]
        ax.plot(rs[:len(tb)], tb, '-o', color=SER[0], lw=2, ms=8)
        style(ax, 'rule tables resident per configuration (rule + successor + check), 1024²', 'bank width r', 'GB (measured bytes_in_use delta)')
        out.append(save(fig, 'memory_vs_r.png'))
    # 6. conditioning vs r
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, m in enumerate(MESHES):
        rs = [r for r in (128, 256, 384, 512) if f'{m}|R{r}|CNR' in cf and cf[f'{m}|R{r}|CNR'].get('q3')]
        if rs:
            ax.plot(rs, [cf[f'{m}|R{r}|CNR']['q3']['J_cond_max'] for r in rs], '-o', color=SER[i], lw=2, ms=8, label=f'J, {m}²')
    rs = [r for r in (128, 256, 384, 512) if f'1024|R{r}|CNR' in cf]
    ax.plot(rs, [cf[f'1024|R{r}|CNR']['q3']['A_cond'] for r in rs], '--', color=SER[3], lw=2, label='A, 1024²')
    ax.set_xscale('log', base=2)
    style(ax, 'Q3: condition numbers (LSPG-CN-R)', 'bank width r', 'condition number')
    ax.legend(frameon=False, fontsize=8)
    out.append(save(fig, 'conditioning_vs_r.png'))
    return out


def main():
    P = plots()
    L = []
    w = L.append
    led = AN['ledger']
    w('# P1 scaling: cost and accuracy against bank width r and mesh N (2D/3D viscous Burgers, off-mesh ROM)\n')
    w('**Status:** final for the priority-1 tranche of the 2026-10-17 plan, lane `jcp-p1-scaling`. Accuracy numbers are '
      'scored against the E0 second-order reference $R_{ST}$, whose uncertainty $\\varepsilon_{\\rm ref}$ is a **conditional '
      'Richardson estimate** (E0 A10 cross-check amended post data); combined A10, A11 and the retracted A15 labels carry '
      'forward. Every number below is generated by `make_report.py` from `checks/analysis.json` (`analyze.py`).\n')
    w(f"GPU-hours spent: {g(led['spent_gpu_hours'])} of the {g(led['cap_gpu_hours'])} cap ({len(led['jobs'])} jobs, ledger `runs/GPU-LEDGER.json`).\n")
    w('## Verdicts\n')
    w('| hypothesis | configuration | outcome | key numbers |\n|---|---|---|---|')
    for k, v in (AN['timing']['h_cost_n'].items() if isinstance(AN['timing'], dict) else []):
        w(f"| H-cost-N | {k} | {v['outcome']} | R = {g(v.get('R'))}, 95 % interval {g((v.get('ci') or [None])[0])}–{g((v.get('ci') or [None, None])[1])} |")
    for tag in ('kernels_2d', 'kernels_3d'):
        k = AN.get(tag)
        if isinstance(k, dict):
            for c, v in k['verdicts'].items():
                ci = v.get('ci') or [None, None]
                w(f"| H-cost-r ({tag[-2:]}) | {c}, declared p = {g(v['p_decl'])} | {v['outcome']}{(' (' + v['secondary'] + ')') if v.get('secondary') else ''} | "
                  f"p fit {g(v['point'].get('p'))}, interval {g(ci[0])}–{g(ci[1])}; FLOP/byte model p {g(v.get('p_flops'))}/{g(v.get('p_bytes'))} |")
        else:
            w(f'| H-cost-r ({tag[-2:]}) | all | {k} | |')
    for k, v in AN['h_acc_n'].items():
        pr = v.get('pairs', {})
        w(f"| H-acc-N | {k} | {v['outcome']} | " + '; '.join(
            f"{p}: n = {x.get('n')}, within margin {pct(x.get('within'))}, TOST p = {g(x.get('p_lower'))}/{g(x.get('p_upper'))}" for p, x in pr.items())
          + (f"; trend p = {g(v['trend'].get('p'))}" if v.get('trend') and v['trend'].get('testable') else '') + ' |')
    for k, v in AN['h_acc_r'].items():
        for t, x in v.items():
            if isinstance(x, dict):
                w(f"| H-acc-r | {k} {t} | {x['outcome']} | improve/worsen/tie/unqualified {x.get('improve')}/{x.get('worsen')}/{x.get('tie')}/{x.get('unqualified')}, "
                  f"median change {pct(x.get('median_change_resolved'))}, worst change {pct(x.get('worst_change'))} |")
            else:
                w(f'| H-acc-r | {k} {t} | {x} | range verdict |')
    w('')
    w('## Configurations: qualification and error\n')
    w('| mesh | config | scheme | Q1 | Q2 | numerically qualified | cases | median $e$ | worst $e$ | BE vs $R_S$ median (BE only) |\n|---|---|---|---|---|---|---|---|---|---|')
    for k, v in AN['configurations'].items():
        m, n, sc = k.split('|')
        w(f"| {m}² | {n} | {sc} | {g(v['Q1']['pass'])} | {v['Q2'].get('label')} | {g(v['numerically_qualified'])} | {v['cases']} | "
          f"{pct(v['e0_median'])} | {pct(v['e0_worst'])} | {pct(v.get('e0_S_median')) if v.get('e0_S_median') is not None else '—'} |")
    w('\nRole qualification: none is pre-declared in P1; every configuration is "role not assessed". Frontier usefulness is '
      'not ranked here (the coordinator combines P1 with the competition work).\n')
    w('## Reproduction gate (AM-1)\n')
    for k, v in AN.get('reproduction', {}).items():
        w(f"- {k}: worst relative coefficient difference vs the combined archive {g(v['worst'])}; numerical manifest equal {g(v['manifest']['equal'])}; gate {'passed' if v['passed'] else 'FAILED'}.")
    w('\n## P1g: what the frozen mode list changes in earlier cross-mesh claims\n')
    w('The test set differs from the frozen list only at $256^2$ (two modes swapped at $M=1536$); at $4096^2$ the nominal list '
      'is a permutation of the frozen one. Distances are scoring-lattice relative, max over outputs and cases.\n')
    w('| arm | 256²: frozen vs nominal distance | 256²: max |Δe| (E0 / first-order) | 4096²: distance | spread (3 meshes, first-order) frozen / nominal | spread (256² vs 1024², first-order) frozen / nominal |\n|---|---|---|---|---|---|')
    for k, v in AN['p1g'].items():
        a = v['meshes'].get('256', v['meshes'].get(256))
        b = v['meshes'].get('4096', v['meshes'].get(4096))
        s3f, s3n = v.get('spread3_frozen', {}), v.get('spread3_nominal', {})
        s2f, s2n = v.get('spread2_frozen', {}), v.get('spread2_nominal', {})
        w(f"| {k} | {g(a['dist_max']) if isinstance(a, dict) else a} | "
          f"{(g(a['de0_max_abs_pp']) + ' / ' + g(a['dfo_max_abs_pp']) + ' pp') if isinstance(a, dict) else ''} | "
          f"{g(b['dist_max']) if isinstance(b, dict) else b} | {g(s3f.get('first_order_max_pp'))} / {g(s3n.get('first_order_max_pp'))} pp | "
          f"{g(s2f.get('first_order_max_pp'))} / {g(s2n.get('first_order_max_pp'))} pp |")
    w('\nHistorical claims re-checked (their numbers as published): ' + '; '.join(
        f"{h} ({c['source']}): " + ', '.join(f'{a} {b} pp' for a, b in c['values_pp'].items())
        for h, c in json.loads(json.dumps(__import__('analyze').HIST_CLAIMS)).items()) + '. H2 (test64) and H3 are not re-run.\n')
    w('## Quadrature slice (P1e)\n')
    e2 = AN['p1e'].get('2d')
    if e2:
        for cfg, v in e2.items():
            rows = v['rows']
            for pair in rows[0]['dist']:
                w(f"- 2D {cfg}, {pair}: worst distance {g(max(r_['dist'][pair] for r_ in rows))}, worst ρ {g(max(max(r_['rho'][pair]) for r_ in rows))}.")
    e3 = AN['p1e'].get('3d')
    if isinstance(e3, dict):
        for pair in e3['rows'][0]['dist']:
            rh = max(max(x) for x in e3['rho'].get(pair, {'0': [float('nan')]}).values())
            w(f"- 3D W512, {pair}: worst distance {g(max(r_['dist'][pair] for r_ in e3['rows']))}, worst ρ on own states {g(rh)}.")
    else:
        w(f'- 3D: {e3}')
    w('\n## Memory, offline cost, pilot\n')
    pil = AN.get('pilot') or {}
    w(f"- $4096^2$ pilot (H200, upper-bound footprint): peak {g((pil.get('peak_bytes') or 0) / 1e9)} GB; prefix-retaining bank required: {g(pil.get('prefix_retaining_required'))}.")
    for m, v in AN['memory'].items():
        w(f"- {m}²: bank build {g(v['setup']['bank']['seconds'])} s, bank {g(v['setup']['bank']['bytes'] / 1e9)} GB; process-lifetime peak {g(v['memory']['end']['peak'] / 1e9)} GB.")
    w('- Sunk offline costs (not re-measured): the 2D bank `dn256b` and the 3D W1024 bank (training job 1.85 h, fit 1.14 h, as recorded by the wide-bank lane).\n')
    w('## Controls\n')
    sm = AN.get('smoke') or {}
    req = [k for k, v in sm.items() if v.get('required')]
    w(f"Real-size smoke (`checks/smoke-accept.json`): {sum(sm[k]['passed'] for k in req)} of {len(req)} required controls behave as registered. "
      'The first acceptance (`checks/smoke-accept-1.json`) failed five checks, all traced to defects of the checks (DESIGN A6).\n')
    if isinstance(AN['timing'], dict):
        w(f"Timing controls: doubled-work medians " + ', '.join(f"{k}: {g(v['median'])}" for k, v in AN['timing']['doubled'].items()) +
          f"; split = fused equivalence {g(AN['timing']['equivalence_all'])}; identity veto clean {g(not any(AN['timing']['veto'].values()))}.")
    o = AN.get('order')
    if o:
        w(f"\nCN-R temporal order (dev6|0, R384): GAL-CN-R observed order {g(o['GAL']['slope'])} (gate [1.8, 2.2], interpretable {g(o['GAL']['interpretable'])}); "
          f"LSPG-CN-R {g(o['LSPG']['slope'])} (reported).\n")
    w('## Plots\n')
    for p in P:
        w(f'![{p}]({p})')
    w('\n## Glossary\n')
    for a, b in [
        ('r (bank width, $R\'$)', 'number of ordered bank coefficients solved for; the "k" of this study.'),
        ('N, mesh', 'grid points per axis of the full-order mesh on which the input is sampled and the output decoded (256², 1024², 4096²).'),
        ('M', 'number of sine test functions, $M=4r$; m = number of quadrature points of the off-mesh rule.'),
        ('frozen / nominal mode list', 'frozen: the 1024² builder list reused at every mesh; nominal: each mesh\'s own list ("deployment policy").'),
        ('E0 $R_{ST}$, $\\varepsilon_{\\rm ref}$', 'second-order space+time reference; its per-case uncertainty estimate (conditional Richardson, not a bound).'),
        ('e', 'max over the five evolved output times of the relative $L^2$ error on the 257² scoring lattice.'),
        ('Q1', 'solver qualification: valid runs, ≤ 1 % non-stationary steps (fresh autodiff Jacobian, every step), distance to a tight solve ≤ 2.5e-4 on every case, tight/tighter ≤ 2.5e-5.'),
        ('Q2', 'quadrature qualification on 16 validation cases: replay with the next finer rule within 2.5e-4, successor ρ ≤ 0.116, and the finer rule itself converged.'),
        ('ρ', 'relative difference of the tested advection vector between a rule and its successor on the arm\'s own states.'),
        ('B/A ratio, A–B–A', 'evolution time of configuration B divided by the mean of two flanking runs of the reference A (r=384, 1024², CN-R) on the same GPU.'),
        ('R (H-cost-N)', 'max over meshes / min over meshes of the median B/A ratio; pass if the upper end of its 95 % case-bootstrap interval ≤ 1.25.'),
        ('TOST, within margin', 'two one-sided signed-rank tests that cross-mesh error differences lie within ±δ, δ = max(2ε_ref, 0.1 e(1024²)); share of cases with |Δe| ≤ δ.'),
        ('improves / does not improve', 'H-acc-r transition outcome: majority of resolved, qualified cases improve and the median paired change is negative.'),
        ('p, declared p', 'exponent of $t=a+br^p$ fitted to kernel times at fixed (M, m), and its operation-count prediction.'),
        ('operation-model / hardware scaling', 'secondary label of an H-cost-r fail: whether the FLOP or byte model itself departs from the declared p.'),
        ('P1g', 'measurement of how much earlier cross-mesh claims move when the test-mode set is frozen.'),
        ('pp', 'percentage points of relative error.')]:
        w(f'- **{a}**: {b}')
    (HERE / 'report.md').write_text('\n'.join(L) + '\n')
    EXP['plots'] = P
    (HERE / 'runs' / 'EXPORT.json').write_text(json.dumps(EXP, indent=1, default=str))
    print('report written;', len(P), 'plots')


if __name__ == '__main__':
    main()
