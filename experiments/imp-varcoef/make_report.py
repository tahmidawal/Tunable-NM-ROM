"""Generate report.md, EXPORT.json and plots for lane imp-varcoef from the run JSONs (no hand-typed numbers).

    python make_report.py --ladder runs/<L2>/output/vladder_<cfg>/result.json --l1 runs/<L1>/output/vladder_<cfg>/result.json
                          --timing runs/<time>/output/vtime_time/result.json --gate checks/gate.json
                          [--test runs/<test>/output/vtest_test-run/result.json]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def J(p):
    return json.loads(Path(p).read_text())


def pct(x, d=3):
    return '–' if x is None else f'{100 * x:.{d}g} %'


def ms(x):
    return '–' if x is None else f'{1e3 * x:.3g} ms'


def g3(x):
    return '–' if x is None else f'{x:.3g}'


def kf(k):
    return '–' if not k or k == '–' else '`' + str(k).replace('|', '\\|') + '`'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plots(L2, T, gate, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(exist_ok=True)
    pooled = L2['pooled'].get('100', {})
    scr = L2['screening']
    cols = {'PSI': '#2f6fb0', 'ETD': '#c2562b', 'FDMG': '#3d8a4a', 'CCV': '#7a4fa0'}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for f, c in cols.items():
        xs = [(scr[f'{k}@C100']['cost'], v['worst']) for k, v in pooled.items() if k.split('|')[0].upper() == f and f'{k}@C100' in scr]
        if xs:
            a = np.array(xs)
            ax.scatter(a[:, 1] * 100, a[:, 0] * 1e3, s=14, color=c, label=f, alpha=.75)
    rb = J(HERE / 'checks/rom-box.json')
    for tau, r in rb['cheapest_by_target'].items():
        if r:
            ax.scatter([r['worst'] * 100], [r['seconds'] * 1e3], marker='*', s=160, color='k')
            ax.annotate(f"ROM R{r['Rp']} (constant κ)", (r['worst'] * 100, r['seconds'] * 1e3), fontsize=7,
                        xytext=(4, 4), textcoords='offset points')
    for tau in (1, 3, 10):
        ax.axvline(tau, color='#999', lw=.6, ls=':')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('worst error over eligible dev+val cases, C = 100 (%)')
    ax.set_ylabel('screening cost, six 4096² fields (ms, H200)')
    ax.set_title('Variable-coefficient Burgers FOMs (C = 100) vs the box ROM')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / 'fom_cost_vs_error.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ts, Gs, lo, hi = [], [], [], []
    for k, v in gate['targets'].items():
        if v.get('G') is not None and v['C'] == 100.0:
            ts.append(f"{100 * v['tau']:g} %"); Gs.append(v['G']); lo.append(v['G'] - v['G_ci95'][0]); hi.append(v['G_ci95'][1] - v['G'])
    if Gs:
        ax.errorbar(range(len(Gs)), Gs, yerr=[lo, hi], fmt='o', color='#2f6fb0', capsize=4)
        ax.set_xticks(range(len(Gs))); ax.set_xticklabels(ts)
    ax.axhline(2, color='#c2562b', ls='--', lw=1, label='gate: 2×')
    ax.axhline(1, color='#999', lw=.6)
    ax.set_yscale('log'); ax.set_ylabel('G = cost(fastest FOM) / cost(ROM)'); ax.set_xlabel('worst-error target (C = 100)')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / 'gate.png', dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ladder', required=True)
    ap.add_argument('--l1', required=True)
    ap.add_argument('--timing', required=True)
    ap.add_argument('--gate', required=True)
    ap.add_argument('--test')
    a = ap.parse_args()
    L2, L1, T, G = J(a.ladder), J(a.l1), J(a.timing), J(a.gate)
    TE = J(a.test) if a.test else None
    El = J(HERE / 'checks/eligibility.json')
    RB = J(HERE / 'checks/rom-box.json')
    led = J(HERE / 'runs/GPU-LEDGER.json')
    sm = J(HERE / 'checks/smoke-accept.json')
    plots(L2, T, G, HERE / 'plots')
    E = {}
    rows = []
    for k, v in G['targets'].items():
        if v['C'] != 100.0:
            continue
        fom = v.get('fom') or {}
        rom = v.get('rom') or {}
        rows.append(f"| {pct(v['tau'], 2)} | {rom.get('label', '–').split('|')[1] if rom.get('label') else 'none qualified'} "
                    f"{pct(rom.get('worst'))} | {g3(rom.get('cost'))} | {kf(fom.get('key', '–'))} | {g3(fom.get('cost'))} | "
                    f"{g3(v.get('G'))} [{g3((v.get('G_ci95') or [None, None])[0])}, {g3((v.get('G_ci95') or [None, None])[1])}] | "
                    f"{G['gate']['target_outcomes'].get(f'{v['tau']:g}', 'not decisive (no ROM)')} |")
        E[f'tau_{v["tau"]:g}'] = dict(G=v.get('G'), G_ci95=v.get('G_ci95'), fom=fom.get('key'), fom_cost=fom.get('cost'),
                                       rom=rom.get('label'), rom_cost=rom.get('cost'), rom_in_job=rom.get('in_job'),
                                       fom_seconds=v.get('fom_seconds'), rom_seconds=v.get('rom_seconds'),
                                       status=v.get('status'), unresolved_families=v.get('unresolved_families'))
    # per-family fastest matched configuration (screening cost, ladder) at C = 100
    fam_rows = []
    pooled = L2['pooled'].get('100', {})
    qual = L2['qualification']
    for f in ('PSI', 'ETD', 'FDMG', 'CCV'):
        for tau in (0.01, 0.03, 0.1):
            gq = L2['candidates']['100'].get(f'{f}|{tau:g}|C100', {})
            c = gq.get('candidates') or []
            if c:
                k = c[0]
                p = pooled[k]
                fam_rows.append(f"| {f} | {pct(tau, 2)} | {kf(k)} | {pct(p['median'])} | {pct(p['worst'])} | "
                                f"{ms(L2['screening'][f'{k}@C100']['cost'])} |")
            else:
                other = [k for t2 in (0.01, 0.03, 0.1) for k in (L2['candidates']['100'].get(f'{f}|{t2:g}|C100', {}).get('candidates') or [])
                         if pooled[k]['worst'] <= tau]
                note = gq.get('status', '–') + ' within the 8-attempt capped search' + (
                    f" (qualified {kf(other[0])} from another target also meets it)" if other else '')
                fam_rows.append(f"| {f} | {pct(tau, 2)} | {note} | – | – | – |")
    el100, el10 = El['contrasts']['100'], El['contrasts']['10']
    eps100 = [v['eps_ref'] for v in el100['cases'].values() if v['eligible']]
    tk = L1['controls'].get('T-kappa@C100', {})
    tk10 = L1['controls'].get('T-kappa@C10', {})
    nC10 = len([r for r in L1['runs'] if float(r['C']) == 10.0])
    temporal = L2['temporal'].get('100', {})
    sp = sum(j.get('gpu_hours') or 0 for j in led['jobs'])
    gate = G['gate']['outcome']
    test_rows = []
    if TE:
        for k, v in TE['keys'].items():
            tau_k = sorted({float(g.split('|')[1]) for g, x in T['selection'].items() if x.get('selected') == k
                            and g.split('|')[2] == 'C100' and g.split('|')[1] in ('0.03', '0.1')})
            tt = min(tau_k) if tau_k else None
            ok_t = None if (tt is None or v['worst'] is None) else v['worst'] <= tt
            test_rows.append(f"| {kf(k)} | {pct(tt, 2) if tt else '–'} | {'meets' if ok_t else 'misses'} | {v['n_eligible']} | {pct(v['median'])} | {pct(v['worst'])} | {'yes' if v['all_valid'] else 'no'} |")
    test_ref_line = 'not run'
    tr = HERE / 'runs/test/output/vref_ref-test/result.json'
    if tr.exists():
        import collections
        rr = J(tr)['cases']
        oc = collections.Counter(c['judge']['outcome'] for c in rr.values())
        test_ref_line = ', '.join(f'{v} {k}' for k, v in sorted(oc.items())) + f' of {len(rr)} cases'
    audits = sorted(p.name for p in (HERE / 'audits').glob('codex-*.md'))
    jobs = '\n'.join(f"| {j['job']} | {j['jobid']} | {j.get('node', '–')} | {j.get('state')} | {g3(j.get('gpu_hours'))} |"
                     for j in led['jobs'])
    cons = [v.get('rom', {}).get('vs_inherited') for v in G['targets'].values() if v['C'] == 100.0 and v.get('rom')]
    dec = {k: (v.get('fom') or {}).get('decomposition') for k, v in G['targets'].items() if v['C'] == 100.0}
    psi_c1 = {k: v.get('psi_C100_over_C1') for k, v in G['targets'].items() if v['C'] == 100.0}
    md = f"""# Variable-coefficient Burgers: does the fastest full-order solver become 2× slower than the box ROM?

**Status:** final for Phase 1 (FOM-only benchmark, lane `imp-varcoef`, experiment E3, 2026-10-21). Gate outcome: **{gate}**
(no ROM trained). Accuracy numbers are **conditional** estimates against this lane's second-order references (E0 recipe,
Richardson $\\epsilon_{{\\rm ref}}$, conditional); every speed ratio is **against the strongest tested competitor** and a
**finite-ladder upper bound on the ROM speed-up**; the ROM costs are those of the **constant-viscosity** bank (inherited
qualification, prior evidence; E0 A10 and combined A10/A11/A15 labels carry forward). All measured results are generated
by `make_report.py` from the run JSONs (registered design constants such as cohort sizes, levels and caps are literals).

## 1. In brief

- **Gate ($C=100$, worst-error targets 3 % and 10 %; references under the post-hoc A10 cross-check rule):** {gate}
  (under the registered rule: {G.get('gate_registered_rule', {}).get('outcome', '–')}). Per target:

| target (worst) | ROM (constant κ), worst | ROM cost (×A) | fastest FOM (stage-1 union winner) | FOM cost (×A) | $G$ [95 % slot bootstrap] | outcome |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

  Costs are anchor-normalised (ratio to the C2 reference ROM A, H200, same job). $G<1$ means the FOM is faster than the ROM.
- **Reading:** the coefficient matters (a constant-ν solver is off by {pct(tk.get('median'))} median / {pct(tk.get('worst'))} worst on the
  dev cases at $C=100$, T-kappa diagnostic), yet the fastest tuned FOM at both decisive targets is the exponential integrator
  with a constant-coefficient split (union winners {', '.join(sorted({(v.get('fom') or {}).get('key', '–') for v in G['targets'].values() if v['C'] == 100.0 and v.get('fom')}))}), and it is
  not slower than the box ROM. Under the registered cross-check rule the gate is **{G.get('gate_registered_rule', {}).get('outcome', '–')}**
  (references not decidable, A10); the outcome above uses the post-hoc A10 rule.
- **References:** {len(eps100)} of {el100['n_cases']} pooled cases eligible at $C=100$ (coverage rule ≥ 24: {'met' if el100['decidable'] else 'NOT met'}),
  {el10['coverage']} of {el10['n_cases']} at $C=10$; $\\epsilon_{{\\rm ref}}$ (conditional) {g3(min(eps100)) if eps100 else '–'}–{g3(max(eps100)) if eps100 else '–'} at $C=100$;
  cross-checks passed at $C=100$: {'yes' if not el100['crosscheck_failed'] else 'no'}, at $C=10$: {'yes' if not el10['crosscheck_failed'] else 'no'}.
- **Compute:** {g3(sp)} GPU-h of the 8 GPU-h Phase-1 cap ({len(led['jobs'])} jobs, ledger below).

## 2. Problem, competitors and the gate

$$u_t+u(u_x+u_y)=\\nabla\\cdot(\\kappa(x)\\nabla u),\\quad (0,1)^2,\\ u=0\\ \\text{{on the walls}},\\ t\\in[0,0.25],$$

$\\kappa=\\nu e^{{s\\hat g(x)}}$, $\\hat g\\in[-\\tfrac12,\\tfrac12]$ a normalised random cosine field (15 modes), contrast $C=e^s\\in\\{{10,100\\}}$,
$\\nu\\in[0.02,0.1]$ log-uniform, the campaign's Gaussian initial data; six outputs at $t=0.05k$. Error: E0 metric (max over
$k\\ge1$ of the lattice $L^2$ error relative to $\\lVert R(t_0)\\rVert$), worst and median over eligible cases. Design:
`DESIGN.md` (amendments A1–A9).

```mermaid
flowchart LR
  R[references E0 recipe<br/>512/1024/2048, Richardson]:::ref --> M[frozen eligibility mask]:::ref
  M --> L1[ladder L1: dev + val<br/>{len(L2['config']['keys'])} base configurations]:::fom
  L1 --> L2[ladder L2: full-field screening<br/>+ qualification]:::fom
  L2 --> T[paired timing, H200<br/>ROM A anchor, same job]:::time
  B[ROM box cost<br/>constant κ, inherited]:::rom --> T
  T --> G{{G ≥ 2 ?}}:::gate
  classDef ref fill:#e8f0fb,stroke:#2f6fb0;
  classDef fom fill:#fdf0e6,stroke:#c2562b;
  classDef time fill:#eef7ee,stroke:#3d8a4a;
  classDef rom fill:#f3eefa,stroke:#7a4fa0;
  classDef gate fill:#fff7d6,stroke:#9a7b00;
```

Competitors (each tuned over its knobs; DESIGN §6): **PSI** sine–Galerkin IMEX (SBDF2/CNAB2) with warm-started CG
preconditioned by the constant-coefficient inverse; **ETD** ETDRK4 with a constant-coefficient split; **FDMG** second-order FD,
SBDF2, CG with a geometric-multigrid (or DST) preconditioner; **CCv** the E0 scheme (CN-R, Newton–BiCGStab). Not built: Krylov/Leja
full-operator exponentials, reused factorisations, algebraic multigrid, POD-ECSW. Deferred (§11 order): $C=10$ timing, $C=10$
validation ladder, CCv at $L=512$.

## 3. Fastest qualified configuration per family at $C=100$ (ladder, pooled eligible dev+val)

Errors: conditional estimates against this lane's references; screening cost ranks only (not a paired timing).

| family | target | cheapest qualified (screening) | median | worst | screening cost |
|---|---|---|---|---|---|
{chr(10).join(fam_rows)}

![FOM cost vs error](plots/fom_cost_vs_error.png)

*Figure 1 (illustrative; not a paired comparison).* Screening cost (full-field 4096² contract, single warm calls, ranking only) against pooled worst error for every
configuration matching some target at $C=100$; stars: the inherited constant-κ ROM denominators (C2 seconds).

Temporal checks (A4.10/A5.10) at $C=100$: {', '.join(f"{f}: {v.get('final')}" for f, v in temporal.items())}.
$C=10$ (dev-only, deferred selection): {nC10} lattice runs; T-kappa at $C=10$: {pct(tk10.get('median'))} median / {pct(tk10.get('worst'))} worst.

## 4. Paired timing and the gate

Timing job: {T.get('gpu')}, job {T.get('job_id')}, gates {T['gates']}, controls fired: {', '.join(f"{k} {'yes' if v.get('fires') else 'no'}" for k, v in T['controls'].items())}.
In-job ROM cost vs inherited C2 cost (ratio): {', '.join(g3(c) if isinstance(c, float) else str(c) for c in cons)} (consistency band ±20 %).
Decomposition of the union winners (kernel with prepared setup / setup only, ×A): {json.dumps(dec)}.
Paired PSI reading (cost at $C=100$ / same configuration at $C=1$): {json.dumps({k: (v or {}).get('ratio') for k, v in psi_c1.items()})}.

![gate](plots/gate.png)

*Figure 2 (paired, same H200 job; conditional references; strongest tested competitor, finite-ladder upper bound; inherited
constant-κ ROM).* Anchor-normalised cost ratio $G$ per decisive target with the slot-bootstrap interval; the gate line is 2.

## 5. Test-once (Phase-1 frozen winners, 16 test cases, not used for selection)

| configuration | target it won | test worst vs target | eligible test cases | median | worst | all valid |
|---|---|---|---|---|---|---|
{chr(10).join(test_rows) if test_rows else '| not run | – | – | – | – | – | – |'}

Test references: {test_ref_line}. The A8.3 test fallback for the non-accepted cases was not run (time/usage budget; recorded as
deferred); test errors are over the eligible test cases only. A target met on dev + val need not be met on test: compare each
winner's test worst error with its target.

## 6. What is provisional, what was changed, what is retracted

- Accuracy: conditional on the lane's references ($\\epsilon_{{\\rm ref}}$ conditional Richardson) and on the eligible population.
- Speed: against the strongest *tested* competitor; Krylov/Leja exponentials, factorisations, AMG and POD-ECSW were not built,
  so a stronger FOM can only lower $G$ further.
- The ROM denominator is the inherited constant-κ configuration (not requalified here; lane brief). It is an assumed,
  optimistic comparator: the cost and accuracy of a variable-κ ROM remain unmeasured.
- Post-smoke amendments before production (A7–A9): shared compiled programs, a qualification cap (search truncation),
  reference levels one step coarser (with fallback), two-job ladder, deferrals. These are recorded in `DESIGN.md`.
- Timing identity: G-FR is {T['gates'].get('G-FR')} — subjects whose timed lattice error did not reproduce the ladder error within
  1e-9: {', '.join(kf(k) for k, v in T['subjects'].items() if v.get('kind') == 'fom' and not v.get('repro_ok')) or 'none'}
  (none is a decisive union winner; the gate uses only subjects whose scoring reproduced, `gate.py`).
- Ladder lineage: L2 recomputes some lattice runs of L1 (fields needed for qualification); candidate summaries keep the
  pre-qualification pooled statistics, which can differ from the recomputed runs in the last digits (`audits/codex-results-1.md`).
- Retractions: none within this lane. The where-we-win report's synthetic warm-start column for this setting (no configuration
  ≥ 2×) is qualitatively consistent with the measured outcome; this does not validate that synthetic model.

## 7. Jobs and audits

| job | Slurm id | node | state | GPU-h |
|---|---|---|---|---|
{jobs}

Smoke acceptance: {'accepted' if sm['accepted'] else 'NOT accepted'} (`checks/smoke-accept.json`). Audits: {', '.join(audits)}.

## 8. Glossary

- **FOM:** full-order model, a conventional solver on a grid or a spectral basis.
- **ROM / bank ROM:** the reduced model of the campaign (coordinate-network bank + off-mesh quadrature); here only its cost on
  the constant-viscosity problem is used.
- **Contrast $C$:** $\\kappa_{{\\max}}/\\kappa_{{\\min}}$ of the diffusion coefficient.
- **PSI / ETD / FDMG / CCv:** the four competitor families of §2.
- **Worst / median error:** maximum / median over eligible cases of the E0 error metric.
- **Target $\\tau$:** a worst-error level (1 %, 3 %, 10 %) at which FOM and ROM are matched.
- **Eligible case:** its reference passed every E0 acceptance check with $\\epsilon_{{\\rm ref}}\\le10^{{-3}}$ and the cross-check.
- **$\\epsilon_{{\\rm ref}}$:** conditional Richardson estimate of the reference error.
- **Screening cost:** single warm call latency used only to rank candidates; **anchor-normalised cost:** median over timing slots
  of $t_B/\\tfrac12(t_{{A1}}+t_{{A2}})$ with A the reference ROM run in the same job.
- **$G$:** anchor-normalised cost of the fastest FOM divided by that of the ROM at the same target; the gate passes at $G\\ge2$.
- **Qualified (FOM):** valid runs, tight-solve distance ≤ 2.5e-4, tight/tighter ≤ 2.5e-5, and (spectral) quadrature checks.
- **Slot:** one of six timing pairs (a FOM case and the ROM anchor case) resampled by the bootstrap.
- **dev / val / test:** development, validation (selection) and test-once cohorts, seeded separately.
- **T-kappa:** diagnostic: the constant-coefficient solver scored against the variable-coefficient reference.
- **Configuration keys:** `family|N or L|Δt|…`: N = spectral modes per axis, L = grid intervals per axis, Δt = time step; PSI adds the
  scheme (SBDF2/CNAB2: second-order implicit–explicit schemes) and CG tolerance; ETD adds `mid`/`max` (the constant
  coefficient κ̄ = mid-range or maximum of κ); FDMG adds MG/DST (multigrid or fast-sine-transform preconditioner) and tolerance;
  CCv adds `t1…t4` (Newton/linear tolerance levels, t4 loosest); `@C100` = contrast; `R256 gauss128` etc. name ROM bank width and quadrature.
- **A:** the reference ROM (R384, LSPG-CN-R) timed between every pair of measurements; **C2:** the earlier jcp-fom2d timing job whose
  ROM costs are inherited; **E0:** the campaign's second-order reference recipe.
- **G-RR / G-FR / A-valid:** timing-job checks that ROM and FOM errors reproduce their earlier values and that A's runs are valid;
  **T-dbl / T-fault / T-id:** timing controls (doubled work is detected, clock drift is rejected, a wrong identity is vetoed).
- **Slot bootstrap:** resampling the six timing slots to give a 95 % interval for $G$; **all valid:** every test run finished with finite
  output and converged solves.
"""
    (HERE / 'report.md').write_text(md)
    E.update(gate=gate, gate_rule='post-hoc A10 cross-check rule', gate_registered_rule=G.get('gate_registered_rule'),
             labels=['accuracy: conditional (lane references, conditional Richardson eps_ref)',
                     'speed: against the strongest tested competitor; finite-ladder upper bound on the ROM speed-up',
                     'ROM: inherited constant-kappa configuration and qualification (prior evidence); variable-kappa ROM unmeasured'],
             gate_detail=G['gate'], gpu_hours=sp, references=dict(eligible_C100=len(eps100), eligible_C10=el10['coverage'],
             eps_ref_C100=[min(eps100), max(eps100)] if eps100 else None),
             tkappa_C100=dict(median=tk.get('median'), worst=tk.get('worst')),
             sources={n: dict(path=str(p), sha256=sha(p)) for n, p in dict(ladder=a.ladder, l1=a.l1, timing=a.timing, gate=a.gate,
                                                                            eligibility=HERE / 'checks/eligibility.json',
                                                                            rombox=HERE / 'checks/rom-box.json').items()},
             test={k: dict(median=v['median'], worst=v['worst']) for k, v in (TE or {}).get('keys', {}).items()},
             plots=['plots/fom_cost_vs_error.png', 'plots/gate.png'])
    (HERE / 'EXPORT.json').write_text(json.dumps(E, indent=1, default=float) + '\n')
    print('report.md, EXPORT.json written; gate', gate)


if __name__ == '__main__':
    main()
