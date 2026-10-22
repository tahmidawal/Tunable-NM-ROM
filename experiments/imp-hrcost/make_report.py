"""imp-hrcost report generator: every number in report.md and EXPORT.json is read from the run JSONs.

    python make_report.py [attempt=prd1]
Inputs: checks/analysis-<attempt>.json (hscore.py), runs/<attempt>/output/hjob/result.json, runs/GPU-LEDGER.json,
checks/smoke-accept-smk1.json, checks/resources-<attempt>.json. Outputs: report.md, EXPORT.json, plots/*.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ATT = sys.argv[1] if len(sys.argv) > 1 else 'prd1'
AN = HERE / 'checks' / f'analysis-{ATT}.json'
RES = HERE / 'runs' / ATT / 'output' / 'hjob' / 'result.json'
A = json.loads(AN.read_text())
R = json.loads(RES.read_text())
LED = json.loads((HERE / 'runs' / 'GPU-LEDGER.json').read_text())
DIAG = json.loads((HERE / 'checks' / 'diag-mass-convergence-r384-gb10.json').read_text())
SMK = json.loads((HERE / 'checks' / 'smoke-accept-smk1.json').read_text())
ARMS = A['arms']
ORDER = ['R'] + [n for n in ARMS if n.startswith('F2_')] + [n for n in ARMS if n.startswith(('F3a', 'F3b', 'F2F3'))]


def pct(x):
    return '∞' if x is None or not np.isfinite(x) else f'{100 * x:.3f} %'


def f2(x, d=2):
    return '—' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'


def arm_row(n):
    a = ARMS[n]
    t = a.get('timing', {})
    v = a.get('vs_R', {})
    e = a['errors'] or {}
    ci = t.get('e2e_ci')
    return (f"| {n} | {a['proj']} | {a['rule']} | {a['m']} | {a['q1']} | {a['q2']} | {a['q2_projected']} | "
            f"{pct(e.get('worst'))} | {pct(e.get('median'))} | {f2(t.get('e2e_median'))} "
            f"[{f2(ci[0]) if ci else '—'}, {f2(ci[1]) if ci else '—'}] | {f2(t.get('evo_median'))} | "
            f"{v.get('no_worse', '—')} | {v.get('faster', '—')} | {v.get('gain15', '—')} |")


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (HERE / 'plots').mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in ORDER:
        a = ARMS[n]
        if not a['errors'] or n == 'R':
            continue
        t = a.get('timing', {})
        if 'e2e_median' not in t:
            continue
        mk = 'o' if n.startswith('F2_') else ('s' if n.startswith('F3a') else ('^' if n.startswith('F3b') else 'D'))
        ax.scatter(t['e2e_median'], 100 * a['errors']['worst'], marker=mk, s=40,
                   facecolors='C0' if a['qualified'] else 'none', edgecolors='C0')
        ax.annotate(n, (t['e2e_median'], 100 * a['errors']['worst']), fontsize=6, xytext=(4, -10 + 8 * (ORDER.index(n) % 4) + (14 if n.startswith('F3b') else 0)), textcoords='offset points')
    ax.axvline(1, color='0.5', lw=.8)
    ax.axvline(1.5, color='C3', lw=.8, ls='--')
    ax.axhline(100 * ARMS['R']['errors']['worst'], color='0.5', lw=.8, ls=':')
    ax.set_yscale('log')
    ax.margins(x=.15, y=.25)
    ax.set_xlabel('end-to-end speed-up vs R (median paired ratio)')
    ax.set_ylabel('worst E0 error over 38 cases (%)')
    ax.set_title('arms vs R; errors: conditional E0 estimates; speed-ups exploratory (filled = qualified)', fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / 'plots' / 'error-vs-speedup.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for fam, mk in (('gauss', 'o'), ('fib', 's')):
        pts = sorted((v['m'], v['detail'].get('rollout_max'), v['detail'].get('rho_max'))
                     for k, v in R['q2'].items() if k.startswith(f'F2_{fam}'))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, label=f'F2 {fam}: rollout distance')
            ax.plot([p[0] for p in pts], [p[2] for p in pts], marker=mk, ls='--', label=f'F2 {fam}: tested-vector ρ')
    ax.axhline(2.5e-4, color='C3', lw=.8)
    ax.axhline(.116, color='C3', lw=.8, ls='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('rule size m')
    ax.set_title('F2 ladder: Q2 rollout distance and ρ against the successor rule', fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / 'plots' / 'f2-ladder-q2.png', dpi=150)
    plt.close(fig)


def main():
    plots()
    jb = dict(R['jac_bench'])
    jb.setdefault('ratio_quad', float('nan'))
    gpu = R['gpu'][0] if R['gpu'] else 'unknown'
    tot = LED['total_gpu_hours']
    stop, stopp = A['stop'], A['stop_projected']
    F2 = R.get('F2', {})
    fit = R['fits']
    rref = ARMS['R']
    L = []
    f2_ok, f3_ok = stop['F2'].startswith('not fired'), stop['F3'].startswith('not fired')
    verdict = ('both levers pass their stop rules' if f2_ok and f3_ok else 'the fitted-weight lever passes, bank tests do not'
               if f3_ok else 'bank tests pass, fitted weights do not' if f2_ok else 'neither lever passes its stop rule')
    L.append(f'# imp-hrcost (E6): a cheaper nonlinear term by bank tests (F2) or fitted weights (F3) — {verdict}\n')
    L.append(f"**Status:** final for the registered production job `{ATT}` (one H200 job, all arms paired in it); "
             f"production population {'yes' if A['production'] else 'NO — not a verdict'}. Every error is a **conditional "
             "E0 estimate** (second-order reference $R_{ST}$, conditional Richardson $\\varepsilon_{\\rm ref}$; E0 A10, "
             "combined A10/A11 and the retracted A15 labels carry forward). Speed-ups are paired, same-job, same-GPU "
             f"ratios against the lane's reference arm R (E5's precontracted Galerkin CN), on `{gpu}`; they are an "
             "internal ablation, **not** a comparison with any competitor. Every measured number below is generated by "
             "`make_report.py` from `checks/analysis-" + ATT + ".json`, the job's `result.json`, the ledger, the smoke "
             "acceptance `checks/smoke-accept-smk1.json` and the construction diagnostic named where used; bars, rule names "
             "and step sizes are registered design constants from `DESIGN.md`.\n")
    L.append(f"GPU-hours: {tot:.3f} of the 3.0 cap ({len(LED['jobs'])} jobs; `runs/GPU-LEDGER.json`).\n")
    L.append('## Summary\n')
    L.append(f"- **F2 stop rule (bank-tested Galerkin, classical rules):** {stop['F2']}. Non-binding Q2-projected "
             f"reading: {stopp['F2']}.")
    L.append(f"- **F3 stop rule (sine-tested, fitted ECSW weights, $\\kappa\\le16$):** {stop['F3']}. Non-binding reading: "
             f"{stopp['F3']}.")
    f3d = {n: R['q2'][n]['detail'].get('rollout_max') for n in R['q2'] if n.startswith(('F3a', 'F3b'))}
    L.append('- F3 Q2 rollout distances to the parent Gauss 128² rule (bar 2.5e-4, 16 cases): ' +
             ', '.join(f'{n} {v:.2e}' for n, v in f3d.items()) + '.')
    L.append(f"- Reference R (sine tests, Gauss 128², m = 16384): Q1 {rref['q1']}, Q2 {rref['q2']}; worst error "
             f"{pct(rref['errors']['worst'])}, median {pct(rref['errors']['median'])} (conditional E0 estimates).")
    best = max((n for n in ARMS if n != 'R' and ARMS[n]['qualified'] and 'timing' in ARMS[n]),
               key=lambda n: ARMS[n]['timing']['e2e_median'], default=None)
    if best:
        b = ARMS[best]
        L.append(f"- Fastest numerically qualified arm: **{best}** (m = {b['m']}), end-to-end {f2(b['timing']['e2e_median'])}× "
                 f"(95 % case-bootstrap {f2(b['timing']['e2e_ci'][0])}–{f2(b['timing']['e2e_ci'][1])}; exploratory, "
                 f"in-sample, selection-unadjusted), evolution only {f2(b['timing']['evo_median'])}×, worst error "
                 f"{pct(b['errors']['worst'])}.")
    L.append(f"- Gains are small: the point-work microbenchmark (step-Jacobian assembly, $4mr^2$ flops) costs "
             f"{f2(jb['dup'][0] / jb['R'][0])}× for 2× the points and {f2(jb['ratio_quad'])}× for 4× (C5′ "
             f"{'fired' if jb['ratio_quad'] >= 1.5 else 'did NOT fire: speed predicates are incomplete evidence'}); "
             "a 4× point reduction therefore buys only a modest query gain; a latency or launch-overhead floor is a possible "
             "explanation that these measurements do not establish (it would need profiling).\n")
    L.append('## What was compared\n')
    L.append('Every arm is the precontracted Galerkin ODE $\\dot c=-P^{\\mathsf T}[(G_qc)\\odot(G_sc)]+\\nu\\mathcal L_0c$ '
             'integrated by the same E5 Crank–Nicolson Newton stepper ($\\Delta t_0=0.005$, 50 steps); only the tables differ.\n')
    L.append('```mermaid\nflowchart LR\n  B["frozen bank G (r = 384)"]:::frozen --> S["sine tests, A = QR"]:::frozen\n'
             '  B --> K["bank tests, Mass / Stiff (Gauss 1024²)"]:::frozen\n'
             '  S --> R["R: Gauss 128² (m = 16384)"]:::ref\n  S --> F3["F3a / F3b: NNLS weights on the 128² points, κr points"]:::fit\n'
             '  K --> F2["F2: classical Gauss / Fibonacci ladder"]:::cls\n  K --> FF["F2+F3: NNLS weights, bank tests"]:::fit\n'
             '  R & F3 & F2 & FF --> E["E5 CN stepper (same code)"]:::frozen\n'
             '  classDef frozen fill:#e5e7eb,stroke:#374151;\n  classDef ref fill:#dbeafe,stroke:#1e40af;\n'
             '  classDef fit fill:#fef3c7,stroke:#92400e;\n  classDef cls fill:#dcfce7,stroke:#166534;\n```\n')
    L.append('## Arms (38 cases, 1024²; errors are conditional E0 estimates; speed-ups exploratory and unadjusted)\n')
    L.append('| arm | tests | rule | m | Q1 | Q2 (binding) | Q2-projected | worst error | median error | end-to-end × [95 % CI] | evolution × | no worse than R | faster | ≥ 1.5× |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for n in ORDER:
        L.append(arm_row(n))
    L.append('\n![error vs speed-up](plots/error-vs-speedup.png)\n\n*Figure: worst error (conditional E0 estimate) against the median paired end-to-end speed-up vs R (exploratory, in-sample, selection-unadjusted); filled = numerically qualified.*\n')
    L.append('## F2: the classical ladder for bank tests\n')
    L.append(f"Candidates ($m\\le16384$, increasing): {', '.join(F2.get('candidates', []))}. First qualified (binding): "
             f"{F2.get('first_qualified')}; $m^*$ identified: {F2.get('mstar_identified')}; first qualified under "
             f"Q2-projected: {F2.get('first_qualified_projected')}.\n")
    L.append('| rule | m | Q2 | rollout distance to successor | tested-vector ρ max | Q2-projected | Q1 |')
    L.append('|---|---|---|---|---|---|---|')
    for q in F2.get('candidates', []):
        v = R['q2'][f'F2_{q}']
        st = F2['status'][q]
        L.append(f"| {q} | {v['m']} | {A['q2'][f'F2_{q}']['label']} | {v['detail'].get('rollout_max', float('nan')):.2e} | "
                 f"{v['detail'].get('rho_max', float('nan')):.3g} | {A['q2'][f'F2_{q}']['label_projected']} | {st['q1']} |")
    L.append('\n![F2 ladder](plots/f2-ladder-q2.png)\n')
    bo = R['setup'].get('bank_ops', {})
    L.append(f"Bank linear terms (P3, A5): {bo.get('rule')} against the check rule, relative differences "
             f"{bo.get('dM', float('nan')):.1e} (mass) and {bo.get('dK', float('nan')):.1e} (stiffness); cond(Mass) "
             f"{bo.get('cond_mass', float('nan')):.3g}. The bank needs Gauss 1024² for its stiffness to converge to "
             f"2.5e-6 (Gauss 384² and 256² differ by {100 * DIAG['384']['dK_prev']:.2f} % in the construction diagnostic `checks/diag-mass-convergence-r384-gb10.json`, a GB10 run): the bank carries "
             "high-wavenumber content, which also enters the bank-tested advection integrand.\n")
    L.append('## F3: fitted weights on our off-mesh points\n')
    L.append('Construction seconds = training rollouts + ECSW system + one NNLS path (all κ checkpoints come from that one path) + support checks.\n')
    L.append('| family | κ | support | training-fit residual | status | total family construction seconds |')
    L.append('|---|---|---|---|---|---|')
    for fam, v in fit.items():
        for k, rr in v.get('rules', {}).items():
            L.append(f"| {fam} | {k} | {rr.get('size', '—')} | {rr.get('relres', float('nan')):.2e} | {rr['status']} | "
                     f"{v.get('seconds', float('nan')):.0f} |")
    L.append('\nThe training-fit residual is never used to certify a rule (lesson of 2026-08-25); Q2 on the arm\'s own '
             'dev/validation states decides.\n')
    L.append('## Controls\n')
    L.append(f"Smoke `smk1` acceptance: {SMK['accepted']} (blocked claims: {', '.join(SMK['blocked_claims']) or 'none'}). "
             f"C5′ in this job: quadruplicated-rule Jacobian ratio {f2(jb['ratio_quad'])} (bar 1.5). A–A timing "
             f"(C6, this job): {f2(A.get('C6', {}).get('e2e_median'))}.\n")
    L.append('| control | result |')
    L.append('|---|---|')
    for k, v in SMK['items'].items():
        L.append(f"| {k} | {'fired / passed' if v.get('fired', v.get('passed')) else 'did not fire'} |")
    L.append('\n## What is provisional, and what was corrected\n')
    L.append('- All errors are conditional E0 estimates; all speed-ups are exploratory, in-sample, selection-unadjusted '
             'and against R only (one GPU type).')
    L.append('- Retracted during the lane: the registered mass/stiffness bar (1e-10 at Gauss 384²) was unattainable and '
             'was replaced before any verdict data (A5); the first justification for a projected ρ gate was withdrawn and the inherited tested-vector gate restored as binding (A2.1); the mesh-sum P5 fixtures were defective and replaced (A6); the '
             'doubled-rule timer control did not fire at real size and was replaced by C5′ (A9).')
    L.append('- The F2 arm changes both the test space and the diffusion discretisation (continuum Mass/Stiff instead of '
             'mesh eigenvalues); its accuracy difference is the combined effect.\n')
    L.append('## Glossary\n')
    L.append('- **R**: the reference arm — E5\'s precontracted sine-tested Galerkin Crank–Nicolson step with the deployed '
             'Gauss 128² rule (m = 16384 points).')
    L.append('- **F2**: Galerkin with the bank functions themselves as test functions, classical (Gauss tensor or '
             'Fibonacci lattice) rules. **F3a / F3b**: R\'s projection with sparse non-negative (NNLS) weights fitted on the '
             'Gauss 128² points, target = projected (F3a) or full tested (F3b) advection. **F2+F3**: bank tests with fitted weights.')
    L.append('- **m**: number of quadrature points ($m^*$: the smallest qualified registered candidate); **κ**: fitted points per unknown (support $=\\kappa r$); **r**: number of bank '
             'coefficients (384); **M**: number of sine tests (1536).')
    L.append('- **Q1**: solver qualification (valid runs, at most 1 % non-converged steps pooled, distance to tight / tighter solves). **Q2 (binding)**: '
             'quadrature qualification on the arm\'s own states — rollout distance to the next rule ≤ 2.5e-4 on 16 cases, '
             'tested-vector $\\rho\\le0.116$, and the next rule converged against the one after it on dev6 (check rule). **Q2-projected**: the same with ρ on the Galerkin-projected advection '
             '(non-binding). **ρ**: relative difference of the tested advection under two rules.')
    L.append('- **worst / median error**: max / median over the 38 dev6 + val32 cases of the max-over-time relative error '
             'against the E0 reference on the 257² lattice. **no worse than R**: worst error not above R\'s by more than '
             '$2\\varepsilon^*$, with $\\varepsilon^*$ the larger reference uncertainty at the two arms\' worst cases. '
             '**ε_ref**: conditional Richardson estimate of the reference error (from comparing references at two resolutions; '
             'valid only if the reference converges as assumed).')
    L.append('- **end-to-end ×**: median over cases of the paired A–B–A time ratio (R time / arm time) of the whole '
             'query (initial fit + 50 steps + 1024² decode); **evolution ×**: the same for the 50 steps only. '
             '**faster**: median > 1 and bootstrap lower bound > 1. **≥ 1.5×**: median ≥ 1.5 and lower bound > 1.')
    L.append('- **stop rule fired (tested failure / incomplete evidence)**: the pre-registered kill criterion triggered on '
             'fully evaluated evidence / because some registered evidence was missing.')
    L.append('- **bank**: the frozen learned basis functions $G$ of the reduced model; **test functions**: the functions the '
             'residual is made orthogonal to (sines or the bank itself); **quadrature**: a weighted point rule approximating the '
             'integrals; **rollout distance**: relative difference of two reduced solutions on the scoring lattice.')
    L.append('- **ECSW**: energy-conserving sampling and weighting, a hyper-reduction that picks few points and fits non-negative '
             'weights so a reduced nonlinear term is reproduced on training snapshots. **NNLS**: non-negative least squares, '
             'the fit used for those weights. **training-fit residual**: the relative error of that fit on the training snapshots.')
    L.append('- **dev6 / val32**: the 6-case development and 32-case validation cohorts (selection allowed). **CI / bootstrap**: '
             '95 % interval of the median paired ratio from 2000 resamples of the 38 cases. **cond(Mass)**: ratio of the '
             'largest to smallest eigenvalue of the bank mass matrix.')
    L.append('- **precontracted Galerkin**: the reduced equation with every test-space contraction done offline, so each step '
             'needs only $m\\times r$ products; $P$, $G_q$, $G_s$: the per-point contraction, bank value and bank derivative '
             'tables; $\\mathcal L_0$: the reduced diffusion operator; $\\nu$: viscosity; $c$: bank coefficients.')
    L.append('- **controls**: C1 coarse rule must fail Q2; C2 permuted fitted weights must fail Q2; C3 corrupted states must be '
             'flagged; C4 a NaN table must be rejected; C5′ 4× points must cost ≥ 1.5× in the Jacobian benchmark; C6 R against '
             'itself must time 0.9–1.1; C7 a dev case in the fit set must be refused; C8 a swapped case id must be refused; '
             'P1–P6 positive equivalence / convergence / identity checks (DESIGN §6, A2.6, A6, E5R).')
    L.append('- **C1–C8, P1–P6, C5′**: must-fail controls and positive checks of DESIGN §6 and amendments.')
    sc_ = dict(phases_seconds=R['phases'], fit_seconds={k: v.get('seconds') for k, v in fit.items()},
               bank_ops_seconds=bo.get('seconds'))
    kstar = next((n for n in ('F3a_k4', 'F3b_k4', 'F3a_k8', 'F3b_k8', 'F3a_k16', 'F3b_k16') if ARMS.get(n, {}).get('qualified')), None)
    gi = L.index('## Glossary\n')
    sec = ['## Setup cost (offline, not in the query time)\n', '| item | seconds |', '|---|---|',
           f"| bank Mass / Stiff (Gauss 1024² + check) | {bo.get('seconds', float('nan')):.0f} |"]
    sec += [f"| {k}: training rollouts + system + NNLS | {v.get('seconds', float('nan')):.0f} |" for k, v in fit.items()]
    sec += [f"| table build {n} ({ARMS[n]['rule']}) | {R['q1'][n]['meta'].get('build_seconds', float('nan')):.2f} |"
            for n in ORDER if not n.startswith(('F3a', 'F3b', 'F2F3'))]
    sec += ['| fitted-table builds (row subsets of the parent Gauss 128² tables) | not measured separately; the parent build is listed above |']
    L[gi:gi] = sec + ['', f"Smallest numerically qualified fitted support among F3a/F3b ($\\kappa^*$): {kstar or 'none'}; F2+F3 "
                         f"qualified supports: {', '.join(n for n in ARMS if n.startswith('F2F3') and ARMS[n]['qualified']) or 'none'}.\n"]
    (HERE / 'report.md').write_text('\n'.join(L) + '\n')
    kstar = next((n for n in ('F3a_k4', 'F3b_k4', 'F3a_k8', 'F3b_k8', 'F3a_k16', 'F3b_k16') if ARMS.get(n, {}).get('qualified')), None)
    exp = dict(lane='imp-hrcost', kappa_star=kstar, setup_cost=sc_, experiment='E6', attempt=ATT, production=A['production'], gpu=gpu,
               gpu_hours_total=tot, stop=stop, stop_projected=stopp, F2=dict(candidates=F2.get('candidates'),
               first_qualified=F2.get('first_qualified'), mstar_identified=F2.get('mstar_identified')),
               jac_bench=dict(dup_ratio=jb['dup'][0] / jb['R'][0], quad_ratio=jb['ratio_quad']),
               arms={n: dict(m=a['m'], q1=a['q1'], q2=a['q2'], q2_projected=a['q2_projected'], qualified=a['qualified'],
                             worst=(a['errors'] or {}).get('worst'), median=(a['errors'] or {}).get('median'),
                             e2e=a.get('timing', {}).get('e2e_median'), e2e_ci=a.get('timing', {}).get('e2e_ci'),
                             evo=a.get('timing', {}).get('evo_median'), vs_R=a.get('vs_R')) for n, a in ARMS.items()},
               labels='errors: conditional E0 estimates; speed-ups: exploratory, in-sample, unadjusted, internal vs R',
               sources=dict(analysis=str(AN.relative_to(HERE)), result=str(RES.relative_to(HERE)),
                            ledger='runs/GPU-LEDGER.json', report='report.md'))
    (HERE / 'EXPORT.json').write_text(json.dumps(exp, indent=1, default=str))
    print('report.md, EXPORT.json written')


if __name__ == '__main__':
    main()
