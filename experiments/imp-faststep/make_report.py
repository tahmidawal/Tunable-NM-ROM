"""imp-faststep report generator: report.md, EXPORT.json and plots from the production run JSONs and their offline
checks (checks/<job>.json). Every number in the report is read here from those files; nothing is typed by hand.

    python make_report.py --jobs f1024 f4096 [--smoke smk1 smk4096]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NEW = ('G-CN', 'G-E4a', 'G-E4b', 'G-R1')
NAMES = {'L-CNR': 'arm 1, LSPG CN-R (incumbent)', 'G-CN': 'arm 2, Galerkin CN (precontracted)',
         'G-E4a': 'arm 3a, Galerkin ETDRK4 dt 0.0125', 'G-E4b': 'arm 3b, Galerkin ETDRK4 dt 0.025',
         'G-R1': 'arm 4, one-pass linearised Galerkin CN', 'L-CNR-M1152': 'arm 6, LSPG CN-R with M = 1152'}


def g(x, d=3):
    if x is None:
        return 'n/a'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    if not np.isfinite(x):
        return 'inf'
    return f'{x:.{d}g}'


def pct(x):
    return 'n/a' if x is None or not np.isfinite(x) else f'{100 * x:.3g} %'


def load(job):
    d = HERE / 'runs' / job / 'archive' / 'output' / f'fsrun_{job}'
    return json.loads((d / 'result.json').read_text()), json.loads((HERE / 'checks' / f'{job}.json').read_text()), d


def timing_parts(R):
    """Median (over dev6 cases and reps) of the B-stage times from the E and F triplets, per subject."""
    out = {}
    recs = (R.get('timing') or {}).get('records', {})
    for t in recs.get('E', []) + recs.get('F', []):
        if not t.get('valid'):
            continue
        o = out.setdefault(t['B'], dict(init=[], evolve=[], decode=[], fused=[]))
        for k in o:
            if f't_{k}' in t:
                o[k].append(t[f't_{k}'])
    return {k: {kk: (float(np.median(vv)) if vv else None) for kk, vv in v.items()} for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', nargs='+', required=True)
    ap.add_argument('--smoke', nargs='*', default=[])
    a = ap.parse_args()
    L, EX = [], dict(lane='imp-faststep', experiment='E5', jobs={}, verdicts={}, sources={})
    led = json.loads((HERE / 'runs' / 'GPU-LEDGER.json').read_text())
    data = {j: load(j) for j in a.jobs}
    smk = json.loads((HERE / 'runs/smk1/archive/output/fsrun_smk1/result.json').read_text())
    SL = {rp: b['controls']['C3m']['slope'] for rp, b in smk['blocks'].items() if b.get('controls')}
    L.append('# E5: a faster reduced time step for the off-mesh bank ROM (Burgers 2D)')
    L.append('')
    status_bits = []
    for j, (R, C, _) in data.items():
        status_bits.append(f"{j}: mesh {R['mesh']}², job {R['job_id']}, {R['gpu']}, offline acceptance {g(C['accepted'])}")
    L.append('**Status:** final for this lane unless stated. ' + '; '.join(status_bits) + '. All errors are conditional E0 ST '
             'accuracy estimates ($\\varepsilon_{\\rm ref}$ is a conditional Richardson estimate; inherited caveats carry forward: the E0 '
             'reference cross-check rule was amended after its data (E0 A10), and the combined lane\'s post-hoc labels A10, A11 and '
             'the retracted A15 apply to the incumbent configuration). The $4096^2$ job did not rerun the must-fire controls; it '
             'inherits the accepted real-size smoke `smk1` evidence by sha256 (DESIGN A5.10). POD comparisons are developmental, in-sample and unadjusted, on a '
             'snapshot-limited basis, with the 2D tuning-budget band unresolved. GPU-hours: '
             f"{g(led['spent_gpu_hours'])} of 4 (ledger `runs/GPU-LEDGER.json`).")
    L.append('')
    L.append('The question (DESIGN.md §1): does a Galerkin step with offline precontraction give an r = 384 evolution at least 3× '
             'faster than the incumbent LSPG CN-R, with no invalid or unstable run and no larger worst-case error? The stop rule '
             '(brief, verbatim): "Stop if any run is unstable or invalid, its worst error is above the current one (arm 1), or '
             'the r = 384 evolution is < 3× faster."')
    L.append('')
    L.append('```mermaid')
    L.append('flowchart LR')
    L.append('  A["offline: QR of A, eig of QᵀΛQ, tables Bz Dz Pz"]:::off --> B["online z-ODE: 3 (m×r) products per evaluation"]:::on')
    L.append('  B --> C1["G-CN: Newton, r×r LU"]:::on')
    L.append('  B --> C2["G-E4: ETDRK4, 4 evaluations, no Jacobian"]:::on')
    L.append('  B --> C3["G-R1: one r×r LU per step"]:::on')
    L.append('  classDef off fill:#e0e7ff,stroke:#3730a3;')
    L.append('  classDef on fill:#dcfce7,stroke:#166534;')
    L.append('```')
    L.append('')
    # ---------------- stop rule table
    L.append('## Stop-rule outcome per new arm')
    L.append('')
    L.append('| mesh | arm | invalid/unstable | worst error above arm 1 (all 38 / eligible) | r = 384 evolve speed-up | fires | numerically qualified | passes |')
    L.append('|---|---|---|---|---|---|---|---|')
    for j, (R, C, _) in data.items():
        for arm in NEW:
            s = C['stop'].get(arm)
            if not s:
                continue
            wa = ', '.join(f"r{rp}: {pct(v['X'])} vs {pct(v['arm1'])}" for rp, v in s['worst_all38'].items())
            we = 'fired' if any(v['fired'] for v in s['worst_eligible'].values()) else 'not fired'
            sp = s['speed']['statistic']
            L.append(f"| {R['mesh']}² | {NAMES[arm]} | {g(s['fired']['stability_validity'])} | {wa} ({'fired' if any(v['fired'] for v in s['worst_all38'].values()) else 'not fired'} / {we}) "
                     f"| {g(sp.get('value'))}× | {', '.join(k for k, v in s['fired'].items() if v) or 'none'} | {g(s['qualified'])} | {g(s['passes'])} |")
            EX['verdicts'][f"{j}|{arm}"] = dict(fired=s['fired'], passes=s['passes'], qualified=s['qualified'],
                                                 speedup_evolve_r384=sp.get('value'),
                                                 worst={str(k): v['X'] for k, v in s['worst_all38'].items()},
                                                 worst_arm1={str(k): v['arm1'] for k, v in s['worst_all38'].items()})
        EX['verdicts'][f'{j}|E5_fixed_on_completed_scope'] = C['E5_fixed_on_completed_scope']
    L.append('')
    L.append('Per-case comparison with arm 1 (cases better / worse / tied by $2\\varepsilon_{\\rm ref}$, all cases E0-eligible):')
    L.append('')
    for j, (R, C, _) in data.items():
        for arm in NEW:
            s = C['stop'].get(arm)
            if s:
                L.append(f"- {R['mesh']}², {arm}: " + '; '.join(f"r = {rp}: {v['ranking_vs_arm1']['better']} / {v['ranking_vs_arm1']['worse']} / {v['ranking_vs_arm1']['tie']}"
                                                          for rp, v in s['worst_all38'].items()))
    L.append('')
    L.append('Reading: the Galerkin arms are more accurate than the incumbent on most cases, but on the hardest case their error is '
             'larger by more than $2\\varepsilon_{\\rm ref}$, so the registered worst-error condition fires for every new arm at every '
             'mesh and rank. The ETDRK4 speed-ups are **evolution-only** speed-ups against the incumbent, not full-query gains '
             'against POD. "Offline acceptance" means the evidence passed every integrity check, not that E5 succeeded. The '
             f"ETDRK4 finite-range slope observed in the smoke control fixture (dev6[0], $1024^2$) is {g(SL.get('384'))} at $r = 384$ (stiff order reduction, DESIGN A6) and {g(SL.get('128'))} at $r = 128$; it was not measured on every production case.")
    L.append('')
    fixed = [C['E5_fixed_on_completed_scope'] for _, C, _ in data.values()]
    per = ', '.join(f"{R['mesh']}²: {g(C['E5_fixed_on_completed_scope'])}" for R, C, _ in data.values())
    L.append(f"**E5 line:** fixed on the completed scope at every mesh run: {g(all(bool(x) for x in fixed))} (per mesh: {per}).")
    L.append('')
    # ---------------- per-arm labels, errors and costs
    L.append('## Accuracy, qualification and cost per arm')
    L.append('')
    L.append('| mesh | arm | r | worst E0 error | median E0 error | Q1 | Q2 | init (ms) | evolve (ms) | per step (ms) | decode (ms) | full query (ms) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for j, (R, C, _) in data.items():
        tp = timing_parts(R)
        rows = {}
        for r_ in R['runs']:
            rows.setdefault((r_['arm'], r_['Rp']), []).append(r_)
        EX['jobs'][j] = dict(mesh=R['mesh'], job_id=R['job_id'], gpu=R['gpu'], arms={})
        for (arm, rp), rr in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            es = [x['e0']['e'] for x in rr]
            lab = C['labels'].get(f'{arm}|{rp}', {})
            t = tp.get(f'{arm}|{rp}', {})
            steps = rr[0].get('steps') or 50
            per = t.get('evolve') / steps if t.get('evolve') else None
            ms = lambda x: g(1e3 * x) if x else 'n/a'
            L.append(f"| {R['mesh']}² | {arm} | {rp} | {pct(max(es))} | {pct(float(np.median(es)))} | {g(lab.get('Q1', {}).get('pass'))} "
                     f"| {lab.get('Q2', {}).get('label', 'n/a')} | {ms(t.get('init'))} | {ms(t.get('evolve'))} | {ms(per)} | {ms(t.get('decode'))} | {ms(t.get('fused'))} |")
            EX['jobs'][j]['arms'][f'{arm}|{rp}'] = dict(worst=max(es), median=float(np.median(es)), Q1=lab.get('Q1', {}).get('pass'),
                                                       Q2=lab.get('Q2', {}).get('label'), seconds=dict(t, per_step=per, steps=steps))
    L.append('')
    L.append('Times are medians over the six dev6 timing cases × 3 repetitions of the timed stages inside paired triplets on one GPU '
             'per job; never compare them across jobs or GPU types. The evolve column includes the ETDRK4 coefficient construction '
             'and the coordinate maps. "Per step" is the evolve time divided by the step count (setup and maps amortised), not a '
             'separately timed kernel. "Full query" is a separately timed fused invocation and is not the sum of the stage medians.')
    L.append('')
    # ---------------- F5
    L.append('## F5 (arm 4) secondary criteria')
    L.append('')
    L.append('| mesh | r | median observed slope | reference converged | order | worst error vs arm 2 | end-to-end gain over arm 2 | F5 outcome |')
    L.append('|---|---|---|---|---|---|---|---|')
    for j, (R, C, _) in data.items():
        for rp, f5 in C['f5'].items():
            L.append(f"| {R['mesh']}² | {rp} | {g(f5['median_slope'])} | {g(f5['reference_converged'])} | {f5['order']} "
                     f"| {pct(f5['worst_vs_arm2']['arm4'])} vs {pct(f5['worst_vs_arm2']['arm2'])} | {g(f5['end_to_end_gain'])}× "
                     f"| {'rejected' if (f5['order'] != 'pass' or f5['worst_vs_arm2']['fired'] or f5['gain_fired']) else 'accepted'} |")
            EX['verdicts'][f'{j}|F5|{rp}'] = f5
    L.append('')
    # ---------------- H-base
    L.append('## H-base reading: matched POD competitors (1024² only)')
    L.append('')
    L.append('| our arm | matched competitor B* (dev6) | B* labels (Q1 / Q2-POD) | match dev6 / val32 / all 38 | B* full-query time / ours (direct triplets) |')
    L.append('|---|---|---|---|---|')
    for j, (R, C, _) in data.items():
        for X, m in (C.get('selection') or {}).get('matches', {}).items():
            st = m['status']
            pl = C['pod'].get(m['B'], {})
            lab = f"{g(pl.get('Q1', {}).get('pass'))} / {pl.get('Q2', {}).get('label', 'native (descriptive)')}"
            esc = lambda s_: s_.replace('|', '\\|')
            L.append(f"| {esc(X)} | {esc(m['B'])} | {lab} | {g(st['dev6'])} / {g(st['val32'])} / {g(st['all38'])} | {g((m['direct'] or {}).get('value'))} |")
        EX['verdicts'][f'{j}|matches'] = (C.get('selection') or {}).get('matches')
        EX['verdicts'][f'{j}|best_stepper'] = ((C.get('selection') or {}).get('recomputed') or {}).get('best_stepper')
    L.append('')
    for j, (R, C, _) in data.items():
        bs = ((C.get('selection') or {}).get('recomputed') or {}).get('best_stepper')
        if bs:
            fx = C['stop'][bs.split('|')[0]]['worst_all38']['384']
            L.append(f"Frozen dev6 best-stepper selection at {R['mesh']}²: `{bs}`; on the 38 cases its worst error {pct(fx['X'])} "
                     f"exceeds the incumbent's {pct(fx['arm1'])}, so it is reported as a **failed frozen selection** (A2.9), not as a best stepper.")
        else:
            L.append(f"Frozen dev6 best-stepper selection at {R['mesh']}²: none qualified on dev6.")
        L.append('')
    labs = {C['pod'].get(m['B'], {}).get('Q2', {}).get('label') for _, C, _ in data.values() for m in (C.get('selection') or {}).get('matches', {}).values()}
    L.append(f"Q2-POD labels of the selected competitors: {', '.join(sorted(str(x) for x in labs))} (HR-unqualified = the successor-rule agreement fails the Q2-POD bar). "
             'Rows not matched on val32 or all 38 are "dev-matched only": no validation-population upper-bound claim is made for them.')
    L.append('A ratio above 1 means the competitor is slower than our arm on the full query. The POD ladder is finite, so every '
             'such speed-up is an upper bound, "against the strongest tested competitor". Unmatched rows mean no tested '
             'competitor reached our dev6 worst error.')
    L.append('')
    for j, (R, C, _) in data.items():
        if R['config'].get('deferred'):
            L.append(f"Deferred at {R['mesh']}² (registered defer order, GPU-hour cap): " + '; '.join(R['config']['deferred']) + '. The 4096² conclusion is therefore on the completed scope (r = 384, arms 1–4) only.')
            L.append('')
    L.append('## Glossary')
    L.append('')
    for k, v in [('arm', 'one time-stepping configuration of the reduced model; arm 1 is the deployed one'),
                 ('LSPG / Galerkin', 'two ways to reduce the equations: least-squares residual minimisation versus projection onto the tests'),
                 ('CN-R', 'Crank–Nicolson with two backward-Euler start-up steps (Rannacher)'),
                 ('ETDRK4', 'a fourth-order exponential integrator that treats diffusion exactly and advection explicitly'),
                 ('precontraction', 'multiplying the fixed matrices together offline so each step needs only three small products'),
                 ('r, M, m', 'number of unknowns (bank columns), number of test functions, number of quadrature points'),
                 ('E0 ST', 'the second-order reference solutions the errors are measured against'),
                 ('worst / median E0 error', 'the largest / median, over the 38 dev6 + val32 cases, of the maximum over output times of the relative field error'),
                 ('eligible', 'cases whose reference uncertainty allows ranking two arms'),
                 ('Q1', 'numerical qualification: every run valid and certified step by step by independent code'),
                 ('Q2', 'quadrature qualification: a finer rule changes neither the advection nor the trajectory beyond the bars'),
                 ('evolve speed-up', 'incumbent evolution time divided by the arm\'s, paired on the same GPU'),
                 ('B*', 'the fastest tested POD competitor whose dev6 worst error is no worse than ours'),
                 ('dev6 / val32', 'development (selection) and validation cohorts; no test cohort was used'),
                 ('POD, ECSW', 'the competitor: a mesh basis from snapshots (POD) with a reduced set of weighted mesh nodes for the nonlinear term (ECSW); k4 / k8 = about 4 or 8 nodes per unknown'),
                 ('Q2-POD, HR-unqualified', "agreement of a POD member with the next larger node set; 'HR-unqualified' means that agreement fails the bar"),
                 ('F5', 'secondary criteria for arm 4: observed order >= 1.8, worst error not above arm 2, end-to-end gain >= 1.3x'),
                 ('H-base', 'the comparison against tuned POD + ECSW baselines'),
                 ('Richardson uncertainty', 'the reference error estimate from two reference resolutions, conditional on asymptotic behaviour'),
                 ('observed slope, reference converged', 'log-log slope of error against step size over the tested steps; the fine reference is accepted when its own step-halving change is 1000x smaller than the errors'),
                 ('init / evolve / decode / full query', 'fitting the initial state, time stepping, producing the six output fields, and the whole query in one call')]:
        L.append(f'- **{k}**: {v}.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (HERE / 'plots').mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, len(data), figsize=(5.5 * len(data), 4), squeeze=False)
    for ax, (j, (R, C, _)) in zip(axs[0], data.items()):
        tp = timing_parts(R)
        for (sid, t) in tp.items():
            if sid.split('|')[0] not in NAMES or not t.get('evolve'):
                continue
            arm, rp = sid.split('|')
            w = EX['jobs'][j]['arms'][sid]['worst']
            ax.scatter(1e3 * t['evolve'], 100 * w, marker='o' if rp == '384' else 's', color='C%d' % list(NAMES).index(arm))
            ax.annotate(f'{arm} r{rp}', (1e3 * t['evolve'], 100 * w), fontsize=6, xytext=(3, 3 + 7 * (list(NAMES).index(arm) % 3)),
                        textcoords='offset points', annotation_clip=True)
        ax.set_xscale('log'); ax.set_xlabel('evolution time per query (ms, median dev6)'); ax.set_ylabel('worst E0 error (%)')
        ax.set_title(f"{R['mesh']}², {R['gpu']}")
    fig.tight_layout(); fig.savefig(HERE / 'plots' / 'worst_error_vs_evolve_time.png', dpi=150)
    gi = L.index('## Glossary')
    L[gi:gi] = ['![worst error vs evolution time](plots/worst_error_vs_evolve_time.png)', '',
                '*Figure: worst conditional E0 error over the 38 cases against the median dev6 evolution time, per arm and rank, '
                'on the completed scope of each job (squares $r = 128$, circles $r = 384$).*', '']
    (HERE / 'report.md').write_text('\n'.join(L) + '\n')
    for j in a.jobs:
        EX['sources'][j] = dict(result=f'runs/{j}/archive/output/fsrun_{j}/result.json', checks=f'checks/{j}.json')
    EX['gpu_hours'] = led['spent_gpu_hours']
    (HERE / 'EXPORT.json').write_text(json.dumps(EX, indent=1, default=float) + '\n')
    print('report.md, EXPORT.json written')


if __name__ == '__main__':
    main()
