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


def lab(k):
    return str(k).replace('|', ' / ')


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
        for i, m in enumerate(MESHES):              # common population per mesh: the 16-case subset at every mesh
            rs = [r for r in (128, 256, 384, 512) if f'{m}|R{r}|{sc}' in cf]
            axs[0].plot(rs, [100 * cf[f'{m}|R{r}|{sc}']['e0_median_sub16'] for r in rs], '-o', color=SER[i], lw=2, ms=8, label=f'{m}²')
        axs[0].set_xscale('log', base=2)
        style(axs[0], 'median error vs r (16-case subset)', 'bank width r', 'median conditional E0 error (%)')
        axs[0].legend(frameon=False)
        for i, r in enumerate((128, 256, 384)):
            ms_ = [m for m in MESHES if f'{m}|R{r}|{sc}' in cf]
            axs[1].plot(ms_, [100 * cf[f'{m}|R{r}|{sc}']['e0_median_sub16'] for m in ms_], '-o', color=SER[i], lw=2, ms=8, label=f'r = {r}')
        axs[1].set_xscale('log', base=2)
        axs[1].set_xticks(MESHES, [f'{m}²' for m in MESHES])
        style(axs[1], 'median error vs mesh (16-case subset)', 'mesh N', 'median conditional E0 error (%)')
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
        tb = [sum(mem['configs'][f'R{r}']['table_bytes'].get(k, 0) for k in ('main', 'qp', 'qpp')) / 1e9 for r in rs if f'R{r}' in mem['configs']]
        ax.plot(rs[:len(tb)], tb, '-o', color=SER[0], lw=2, ms=8)
        style(ax, 'Rule tables (rule + successor + check), 1024²', 'bank width r', 'GB, measured bytes_in_use delta')
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
      'forward. All accuracy numbers are therefore **conditional E0 accuracy estimates**, not certified errors. Every number below is '
      'generated by `make_report.py` from `checks/analysis.json` (`analyze.py`), which computes them from the job archives; the exceptions are '
      'quoted published values (historical claims, sunk offline cost), which `analyze.py` reads from the cited source files, and registered '
      'design constants (thresholds, ranks, rule names).\n')
    w(f"GPU-hours spent: {g(led['spent_gpu_hours'])} of the {g(led['cap_gpu_hours'])} cap ({len(led['jobs'])} jobs, ledger `runs/GPU-LEDGER.json`).\n")
    w('## Verdicts\n')
    w('| hypothesis | configuration | outcome | key numbers |\n|---|---|---|---|')
    for k, v in (AN['timing']['h_cost_n'].items() if isinstance(AN['timing'], dict) else []):
        w(f"| H-cost-N | {lab(k)} | {v['outcome']} | R = {g(v.get('R'))}, 95 % interval {g((v.get('ci') or [None])[0])}–{g((v.get('ci') or [None, None])[1])} |")
    for tag in ('kernels_2d', 'kernels_3d'):
        k = AN.get(tag)
        if isinstance(k, dict):
            for c, v in k['verdicts'].items():
                ci = v.get('ci') or [None, None]
                w(f"| H-cost-r ({tag[-2:]}, component kernel) | {c}, declared p = {g(v['p_decl'])} | {v['outcome']}{(' (' + v['secondary'] + ' by the A1.9 rule)') if v.get('secondary') else ''} | "
                  f"p fit {g(v['point'].get('p'))}, interval {g(ci[0])}–{g(ci[1])}; FLOP/byte model p {g(v.get('p_flops'))}/{g(v.get('p_bytes'))} |")
        else:
            w(f'| H-cost-r ({tag[-2:]}) | all | {k} | |')
    for k, v in AN['h_acc_n'].items():
        pr = v.get('pairs', {})
        w(f"| H-acc-N | {lab(k)} | {v['outcome']} | " + '; '.join(
            f"{p}: n = {x.get('n')}, within margin {pct(x.get('within'))}, TOST p = {g(x.get('p_lower'))}/{g(x.get('p_upper'))}" for p, x in pr.items())
          + (f"; trend p = {g(v['trend'].get('p'))}" if v.get('trend') and v['trend'].get('testable') else '') + ' |')
    for k, v in AN['h_acc_r'].items():
        for t, x in v.items():
            if isinstance(x, dict):
                w(f"| H-acc-r | {lab(k)}, {t} | {x['outcome']} | improve/worsen/tie/unqualified {x.get('improve')}/{x.get('worsen')}/{x.get('tie')}/{x.get('unqualified')}, "
                  f"median change {pct(x.get('median_change_resolved'))}, worst change {pct(x.get('worst_change'))} |")
            else:
                w(f'| H-acc-r | {lab(k)}, {t} | {x} | range verdict: "monotone" = every transition in the range passes the majority rule (not every case improves) |')
    w('')
    w('## Configurations: qualification and error\n')
    w('Conditional E0 accuracy estimates. Errors are over each configuration\'s own case set (38, or the 16-case subset at 4096² for R128/R256); the column "median, 16-case subset" '
      'is on the common population and is the one to compare across meshes. $R_S$ (BE only, unmatched for CN-R) is summarised over the '
      'S-eligible cases only (dev6|1, val32|21, val32|24 are not S-eligible).\n')
    w('| mesh | config | scheme | Q1 | Q2 | numerically qualified | cases | median $e$ | median $e$, 16-case subset | worst $e$ | BE vs $R_S$ median, S-eligible |\n|---|---|---|---|---|---|---|---|---|---|---|')
    for k, v in AN['configurations'].items():
        m, n, sc = k.split('|')
        w(f"| {m}² | {n} | {sc} | {g(v['Q1']['pass'])} | {v['Q2'].get('label')} | {g(v['numerically_qualified'])} | {v['cases']} | "
          f"{pct(v['e0_median'])} | {pct(v['e0_median_sub16'])} | {pct(v['e0_worst'])} | {pct(v.get('e0_S_median_eligible')) if v.get('e0_S_median_eligible') is not None else '—'} |")
    w('\nRole qualification: none is pre-declared in P1; every configuration is "role not assessed". Frontier usefulness is '
      'not ranked here (the coordinator combines P1 with the competition work).\n')
    w('## Reproduction gate (AM-1)\n')
    for k, v in AN.get('reproduction', {}).items():
        w(f"- {lab(k)}: worst relative coefficient difference vs the combined archive {g(v['worst'])}; numerical manifest equal {g(v['manifest']['equal'])}; gate {'passed' if v['passed'] else 'FAILED'}.")
    w('\n## P1g: what the frozen mode list changes in earlier cross-mesh claims\n')
    w('The test set differs from the frozen list only at $256^2$ (two modes swapped at $M=1536$); at $4096^2$ the nominal list '
      'is a permutation of the frozen one. Distances are scoring-lattice relative, max over outputs and cases.\n')
    w('| arm | 256²: frozen vs nominal distance | 256²: largest error change, E0 / first-order | 4096²: distance | spread (3 meshes, first-order) frozen / nominal | spread (256² vs 1024², first-order) frozen / nominal | nominal vs historical job, largest first-order error difference (256² / 1024² / 4096²) |\n|---|---|---|---|---|---|---|')
    for k, v in AN['p1g'].items():
        a = v['meshes'].get('256', v['meshes'].get(256))
        b = v['meshes'].get('4096', v['meshes'].get(4096))
        s3f, s3n = v.get('spread3_frozen', {}), v.get('spread3_nominal', {})
        s2f, s2n = v.get('spread2_frozen', {}), v.get('spread2_nominal', {})
        w(f"| {lab(k)} | {g(a['dist_max']) if isinstance(a, dict) else a} | "
          f"{(g(a['de0_max_abs_pp']) + ' / ' + g(a['dfo_max_abs_pp']) + ' pp') if isinstance(a, dict) else ''} | "
          f"{g(b['dist_max']) if isinstance(b, dict) else b} | {g(s3f.get('first_order_max_pp'))} / {g(s3n.get('first_order_max_pp'))} pp | "
          f"{g(s2f.get('first_order_max_pp'))} / {g(s2n.get('first_order_max_pp'))} pp | " +
          (' / '.join(g((v.get('historical_reproduction') or {}).get(m, (v.get('historical_reproduction') or {}).get(str(m), {})).get('max_abs'))
                      for m in MESHES) if v.get('historical_reproduction') else '—') + ' |')
    w('\nReading: the nominal-list runs reproduce the historical job rows to round-off (last column), so the published numbers stand '
      'as computed; but they compared slightly different test spaces at 256². Frozen vs nominal is the total effect of the list change '
      'under each rule (quadrature and test space change together), not a list-only effect; it moves trajectories by up to the 256² '
      'distances above (above the 2.5e-4 Q1 bar) and errors by at most the listed hundredths of a percentage point.\n')
    w('\nHistorical claims re-checked (their numbers as published): ' + '; '.join(
        f"{h} ({c['source']}): " + ', '.join(f'{a} {b} pp' for a, b in c['values_pp'].items())
        for h, c in AN['historical_claims'].items() if h != 'note') + '. H2 (test64) and H3 are not re-run.\n')
    w('## Quadrature slice (P1e, descriptive)\n')
    w('| slice | rule pair (coarser vs finer) | worst rollout distance | rollout bar 2.5e-4 | worst own-state ρ | ρ bar 0.116 | all runs valid |\n|---|---|---|---|---|---|---|')
    e2 = AN['p1e'].get('2d') or {}
    for cfg, v in (AN['p1e'].get('2d_judged') or {}).items():
        names = e2[cfg]['rules']
        alias = dict(fail='fail', fail_succ='fail_succ', main='deployed', qp='finer', qpp='check')
        word = dict(fail='registered failing', fail_succ='its successor', main='deployed', qp='finer', qpp='check')
        nm = lambda role: f"{names.get(alias[role])} ({word[role]})"
        for pr, x in v.items():
            a_, b_ = pr.split('|')
            w(f"| 2D {cfg}, 1024², BE, dev6 | {nm(a_)} vs {nm(b_)} | {g(x['worst_distance'])} | {g(x['rollout_bar'])} | {g(x['worst_rho'])} | {g(x['rho_bar'])} | {g(x['all_valid'])} |")
    for pr, x in (AN['p1e'].get('3d_judged') or {}).items():
        if pr == 'solver':
            continue
        w(f"| 3D W512, 65³, FS-BE, 923801 cases 0–5 | {pr.replace('|', ' vs ')} | {g(x['worst_distance'])} | {g(x['rollout_bar'])} | {g(x['worst_rho'])} | {g(x['rho_bar'])} | {g(x['all_valid'])} |")
    for cfg, v in (AN['p1e'].get('2d_judged') or {}).items():
        x = v.get('fail|fail_succ', {})
        if x.get('rollout_bar') and x.get('rho_bar'):
            w(f"\nThe 2D rule registered as failing ({e2[cfg]['rules']['fail']}) **passes** both bars against its successor "
              f"(distance {g(x['worst_distance'])}, ρ {g(x['worst_rho'])}): the pre-data expectation that it fails was wrong; it is reported, not re-labelled.")
    sv = (AN['p1e'].get('3d_judged') or {}).get('solver')
    if sv:
        w(f"\n3D solver evidence: {sv['nonstationary']} of {sv['steps']} steps non-stationary, {sv['saturated']} saturated, {sv['reason3']} non-finite; "
          f"fixed-sweep vs vendored deployed solver, worst relative coefficient difference {g(sv['vendor_bridge_worst'])}.")
    w('\n## Conditioning (Q3)\n')
    w('| mesh | config | scheme | cond(A) | rank(A) | median cond(J) | max cond(J) | max cond(JᵀJ) |\n|---|---|---|---|---|---|---|---|')
    for k, v in AN['configurations'].items():
        if v.get('q3') and not k.split('|')[1].startswith('H'):
            m, n, sc = k.split('|')
            q = v['q3']
            w(f"| {m}² | {n} | {sc} | {g(q['A_cond'])} | {q['A_rank']} | {g(q['J_cond_median'])} | {g(q['J_cond_max'])} | {g(q['JTJ_cond_max'])} |")
    if isinstance(AN['timing'], dict):
        w('\n## Timing diagnostics (not part of the H-cost-N verdict)\n')
        w('Seconds per call, median over dev6 × 3 repeats on one H200 (full-field contract: input sampled at quadrature points, full-field decode). '
          'Initialisation and decode are expected to grow with N. Scope of H-cost-N (A3.2): the timed program is the solver **with internal-state '
          f"recording** (one row write per step; smoke non-recording / recording time ratio median {g((AN.get('recording_overhead') or {}).get('ratio_norec_over_rec_median'))}, "
          f"range {g((AN.get('recording_overhead') or {}).get('ratio_min'))}–{g((AN.get('recording_overhead') or {}).get('ratio_max'))}), and the verdict is about **evolution** time, "
          'not the full query. Identity audit limitation: every timed execution was compared in-job with its scoring run (W, every internal '
          'state, exit reasons, iterations) and all comparisons passed, but the archive keeps the comparison outcomes, not each execution\'s own hash.\n')
        w('| configuration | init 256² / 1024² / 4096² | evolve 256² / 1024² / 4096² | decode 256² / 1024² / 4096² | fused query / A, 256² / 1024² / 4096² |\n|---|---|---|---|---|')
        for k, v in AN['timing']['h_cost_n'].items():
            ds = v.get('descriptive_seconds')
            if not ds:
                continue
            f3 = lambda d: ' / '.join(g(d[str(m)]) for m in MESHES)
            w(f"| {lab(k)} | {f3(ds['t_init'])} | {f3(ds['t_evolve'])} | {f3(ds['t_decode'])} | {f3(v['fused_ratio_median'])} |")
        comp = AN['timing'].get('components') or []
        if comp:
            w('\nComponent-sum diagnostic (A2.2, no gate): ' + '; '.join(f"{lab(c['subject'])}: model / evolution = {g(c['ratio'])}" for c in comp) + '.')
        tq = AN['timing'].get('timing_qualification') or {}
        w(f"\nTiming-job qualification (A2.3, dev6 trajectories of the timed program): {sum(bool(x['timing_job']) for x in tq.values())} of {len(tq)} subjects "
          f"numerically qualified; both labels (timing job and accuracy job) pass for {sum(bool(x['both']) for x in tq.values())} of {len(tq)}.")
    tr = [k for k, v in AN['h_acc_n'].items() if v.get('trend') and v['trend'].get('rejects')]
    if tr:
        w('\nNote on H-acc-N: the three-mesh trend test rejects "no trend" for ' + ', '.join(lab(k) for k in tr) +
          ' while both equivalence tests pass with every case within its margin; by the registered precedence the outcome is pass. '
          'This is practical equivalence within the registered margin, not evidence of zero mesh dependence: a small systematic trend is detectable. '
          'Median per-case slopes: ' + ', '.join(f"{lab(k)} {g(AN['h_acc_n'][k]['trend']['median_slope'])} per doubling of N" for k in tr) + '.')
    w('\n## Memory, offline cost, pilot\n')
    pil = AN.get('pilot') or {}
    w(f"- $4096^2$ pilot (H200, upper-bound footprint): peak {g((pil.get('peak_bytes') or 0) / 1e9)} GB; prefix-retaining bank required: {g(pil.get('prefix_retaining_required'))}.")
    for m, v in AN['memory'].items():
        w(f"- {m}²: bank build {g(v['setup']['bank']['seconds'])} s, bank {g(v['setup']['bank']['bytes'] / 1e9)} GB; process-lifetime peak {g(v['memory']['end']['peak'] / 1e9)} GB.")
    os_ = AN.get('offline_sunk') or {}
    w(f"- Sunk offline costs (not re-measured): the 2D bank `dn256b`; the 3D W1024 bank (job total {g(os_.get('w1024_job_total_hours'))} H200-h, "
      f"bank fit {g(os_.get('w1024_bank_fit_hours'))} h; source `{os_.get('source')}`).\n")
    w('## Controls\n')
    sm = AN.get('smoke') or {}
    req = [k for k, v in sm.items() if v.get('required')]
    w(f"Real-size smoke (`checks/smoke-accept.json`): {sum(sm[k]['passed'] for k in req)} of {len(req)} required controls behave as registered. "
      f"The first acceptance (`checks/smoke-accept-1.json`) failed {len((AN.get('smoke_first') or {}).get('failed', []))} checks, all traced to defects of the checks (DESIGN A6).\n")
    if isinstance(AN['timing'], dict):
        w(f"Timing controls: doubled-work medians " + ', '.join(f"{k}: {g(v['median'])}" for k, v in AN['timing']['doubled'].items()) +
          f"; split = fused equivalence {g(AN['timing']['equivalence_all'])}; identity veto clean {g(not any(AN['timing']['veto'].values()))}.")
    o = AN.get('order')
    if o:
        w(f"\nCN-R temporal order (dev6|0, R384): GAL-CN-R observed order {g(o['GAL']['slope'])} (gate [1.8, 2.2], interpretable {g(o['GAL']['interpretable'])}); "
          f"LSPG-CN-R {g(o['LSPG']['slope'])} (reported).\n")
    w('\n## Not done here (by design) and provenance\n')
    w('- 3D cost against mesh (H-cost-N in 3D) and 3D Q1 for the speed3d arms come from `jcp-speed3d` (m1, g1–g3) under its mesh-free / '
      'scoring-lattice contract; the coordinator combines them. This lane ran none of them (AM-6).')
    w('- Deferred items: none (the defer order was not needed; spending stayed far below the cap).')
    w('- References: E0 store version ' + str(AN['configurations'] and json.loads((HERE / 'runs' / 'MANIFEST-dev.json').read_text())['e0_store_version'][:16]) + '.')
    w('')
    w('## Plots\n')
    w('Accuracy plots use the 16-case subset at every mesh (conditional E0 estimates). A median curve can move slightly in the opposite '
      'direction of a paired H-acc-r verdict (e.g. CN-R at R512): the verdict counts resolved paired cases, the curve shows one population median.\n')
    for p in P:
        w(f'![{p}]({p})')
    w('\n## Glossary\n')
    for a, b in [
        ('r (bank width, $R\'$)', 'number of ordered bank coefficients solved for; the "k" of this study.'),
        ('N, mesh', 'grid points per axis of the full-order mesh on which the input is sampled and the output decoded (256², 1024², 4096²; 65³ in 3D).'),
        ('M, m', '$M$ = number of sine test functions ($M=4r$ in the accuracy and timing runs and the nominal kernel panel; fixed at 1536 (2D) or 2052 (3D) in the fixed-$(M,m)$ kernel panel); $m$ = number of quadrature points.'),
        ('A, J, N, JᵀJ, chol', '$A$: projection of the bank onto the tests ($M\\times r$); $N(c)$: tested advection vector; $J$: Jacobian of the LSPG residual; $J^{\\mathsf T}J$: normal matrix; chol: its damped Cholesky factorisation and solve. Kernel components time these separately.'),
        ('BE, CNR (CN-R), LSPG, GAL, FS-BE', 'backward Euler; Crank–Nicolson with two BE start-up steps; least-squares Petrov–Galerkin (minimises the weighted residual); Galerkin (used only in the order check); fixed-sweep BE (the deployed 3D solver).'),
        ('frozen / nominal mode list', 'frozen: the 1024² builder list reused at every mesh; nominal: each mesh\'s own list ("deployment policy").'),
        ('R128 … R512, H64, H96, HF6765, H640, "nom"', 'configurations: Rr = bank width r with its frozen rule (R128 Fibonacci 4181, R256/R384 Gauss 128², R512 Gauss 192²); H… = historical rules at r = 384 (Gauss 64², 96², Fibonacci 6765, Gauss 640²); suffix nom = nominal list.'),
        ('E0 $R_{ST}$, $R_S$, $\\varepsilon_{\\rm ref}$, eligibility', 'second-order space+time reference (primary); space-only BE reference (matched to BE only); per-case uncertainty estimate (conditional Richardson, not a bound); a case-kind is eligible if the E0 store accepts it (all 38 ST case-kinds; S is not eligible for dev6|1, val32|21, val32|24).'),
        ('e', '$e=\\max_{k=1..5}\\lVert u(t_k)-R(t_k)\\rVert/\\lVert R(t_0)\\rVert$ on the 257² scoring lattice (normalised by the initial reference norm).'),
        ('cases, 16-case subset', 'dev6 (6 development cases) + val32 (32 validation cases) = 38; the 16-case subset is dev6 + val32[0:10].'),
        ('Q1', 'solver qualification: valid runs, at most 1 % non-stationary steps (stationarity ratio by a fresh autodiff Jacobian at every step), distance to a tight solve at most $2.5\\times10^{-4}$ on every case, tight/tighter at most $2.5\\times10^{-5}$.'),
        ('Q2', 'quadrature qualification on the 16-case subset (6 development + 10 validation cases): replay with the next finer rule within $2.5\\times10^{-4}$, successor $\\rho\\le0.116$, and the finer rule converged against the next one.'),
        ('Q3, cond, rank', 'conditioning: cond = ratio of largest to smallest singular value; rank = number of singular values above $10^{-13}$ times the largest.'),
        ('$\\rho$', 'relative difference of the tested advection vector between a rule and its successor on the arm\'s own output states.'),
        ('B/A ratio, A–B–A', 'evolution time of configuration B divided by the mean of two flanking runs of the reference A (r = 384, 1024², CN-R) on the same GPU.'),
        ('R (H-cost-N)', 'max over meshes / min over meshes of the median B/A ratio; pass if the upper end of its 95 % case-bootstrap interval is at most 1.25.'),
        ('TOST, within margin, trend p', 'two one-sided signed-rank tests that cross-mesh error differences lie within $\\pm\\delta_i$, $\\delta_i=\\max(2\\varepsilon_{\\rm ref,i},\\,0.1\\,e_i(1024^2))$; share of cases with $|\\Delta e_i|\\le\\delta_i$; Wilcoxon p-value of the per-case slopes of $e$ against $\\log_2 N$ (a test p-value, unrelated to the exponent p).'),
        ('resolved, tie, improve / worsen', 'a case is resolved when both rungs are qualified and $|\\Delta e|>2\\varepsilon_{\\rm ref}$; otherwise a tie; improve = error decreases.'),
        ('improves / does not improve, monotone', 'H-acc-r transition: more than half of resolved cases improve and the median paired change is negative; a range is monotone if every transition improves in that sense.'),
        ('p, declared p', 'exponent of $t=a+b\\,r^p$ fitted to component-kernel times at fixed $(M,m)$, and its operation-count prediction.'),
        ('operation-model / hardware scaling', 'secondary label of an H-cost-r fail by the registered rule (A1.9): "operation-model" if the FLOP or byte count model itself departs from the declared p by more than 0.25; it is a classification, not a causal explanation.'),
        ('P1g', 'measurement of how much earlier cross-mesh claims move when the test-mode set is frozen (total effect under each rule).'),
        ('pp', 'percentage points of relative error.')]:
        w(f'- **{a}**: {b}')
    (HERE / 'report.md').write_text('\n'.join(L) + '\n')
    EXP['plots'] = P
    (HERE / 'runs' / 'EXPORT.json').write_text(json.dumps(EXP, indent=1, default=str))
    print('report written;', len(P), 'plots')


if __name__ == '__main__':
    main()
