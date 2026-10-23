"""imp-geometry report generator: every number in report.md and EXPORT.json is read from the run JSONs here.
  python make_report.py"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import decide                       # noqa: E402  (audited decision code)

RUNS = HERE / 'runs'
CELLS = {'L': RUNS / 'decide' / 'L|wide|S1', 'H': RUNS / 'decide' / 'H|wide|S1'}
TAUS = decide.TAUS
FAM_NAME = dict(sem='spectral elements (C-SEM)', fdmg='FD + multigrid (C-FDMG)', rk4='FD + RK4 (C-RK4)',
                ecsw='POD + ECSW (C-ECSW)', podg='POD-Galerkin (R-PODG)', spec='sine spectral (C-SPEC)')


def j(p):
    return json.loads(Path(p).read_text())


def g(x, d=3):
    return '–' if x is None else (f'{x:.{d}g}' if isinstance(x, float) else str(x))


def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (HERE / 'plots').mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    fams = ('sem', 'fdmg', 'rk4', 'ecsw', 'podg')
    for ax, geom in zip(axes, ('L', 'H')):
        fin = data[geom][1]
        for f in fams:
            ys = [((fin['summary'].get(f'{t}|worst') or {}).get('per_family') or {}).get(f) for t in TAUS]
            ax.plot([str(t) for t in TAUS], [y if y else float('nan') for y in ys], marker='o', label=FAM_NAME[f])
        ax.axhline(2, ls='--', c='k', lw=1)
        ax.axhline(1, ls=':', c='grey', lw=1)
        ax.set_yscale('log')
        ax.set_title(f'{geom} | wide | S1 (validation, worst error)' + ('' if geom == 'H' else '\ndescriptive: references ineligible'))
        ax.set_xlabel('target tau')
    axes[0].set_ylabel('S = t_competitor / t_bank')
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / 'plots' / 'S_per_family.png', dpi=130)
    plt.close(fig)


def main():
    led = j(RUNS / 'GPU-LEDGER.json')
    s1 = j(RUNS / 'decision_S1.json')
    data, exp = {}, dict(lane='imp-geometry', experiment='E2', cells={}, A10=s1['A10'],
                         geometry_labels={g: s1[g].get('geometry_label') for g in ('L', 'H')},
                         gpu_hours_spent=led['total_gpu_hours'], cap=62)
    for geom, cd in CELLS.items():
        ref, fin, test, inter = (j(cd / f) for f in ('ref.json', 'final.json', 'test.json', 'inter.json'))
        labels = {tau: decide.cell_label(cd, tau) for tau in TAUS}
        data[geom] = (ref, fin, test, inter, labels)
        exp['cells'][f'{geom}|wide|S1'] = dict(
            allowance=ref['allowance'], D_bnd_median_val=inter['val']['D_bnd_median'],
            D_bnd_median_test=(test.get('inter_test') or {}).get('D_bnd_median'),
            S_val={k: (v or {}).get('S') for k, v in fin['summary'].items()},
            S_test={k: (v or {}).get('S') for k, v in test['summary'].items()},
            best_competitor={k: (v or {}).get('best_competitor') for k, v in fin['summary'].items()},
            per_family_val={k: (v or {}).get('per_family') for k, v in fin['summary'].items()},
            labels={str(t): v for t, v in labels.items()},
            source=str(cd.relative_to(HERE)))
    L = []
    L.append('# Geometry without a fast transform (E2, S1 wide family): on the holed box the bank is slower than the strongest tested competitor (L-shape: descriptive only)\n')
    L.append(f"**Status (2026-10-22): final for the two executed S1 wide-family cells (H qualified; L descriptive only, its "
             f"validation references fail the eligibility gate); S2 and the box controls were not run.** "
             f"Lane `exp/2026-10-21-imp-geometry`. Pre-registered A10 outcome: **{s1['A10']}** "
             f"(geometry L: {s1['L']['geometry_label']}; geometry H: {s1['H']['geometry_label']}). Every measured number below is read by `make_report.py` from the run JSONs; protocol constants (cap, thresholds, cohort sizes) are those registered in DESIGN.md; references are "
             f"conditional Richardson estimates. Every S is an upper bound on our speed-up relative to more thoroughly "
             f"tuned competitors (finite ladders), with the selected bank fixed.\n")
    L.append('## 1. In brief\n')
    for geom in CELLS:
        ref, fin, test, inter, labels = data[geom]
        Sv = [v.get('S') for v in fin['summary'].values() if v and v.get('S') is not None]
        qual = '' if geom == 'H' else ' (descriptive; reference-ineligible)'
        L.append(f"- **{geom} | wide | S1{qual}:** S (bank vs best competitor, validation) ranges "
                 f"{g(min(Sv))}–{g(max(Sv))} over the three targets and two statistics; the best competitor is a POD "
                 f"reduced model ({', '.join(sorted({v['best_competitor'] for v in fin['summary'].values() if v and v.get('best_competitor')}))}). "
                 f"Decision labels per target (worst error): "
                 + '; '.join(f"τ={t}: {labels[t]['label']}" for t in TAUS) + '.')
    L.append('- Against the high-order spectral-element competitor alone the bank is also slower at every target '
             '(per-family S in §3); it is faster than FD + multigrid only at the loosest targets.')
    L.append(f"- GPU-hours spent: {g(led['total_gpu_hours'], 4)} of the 62 cap.\n")
    L.append('## 2. What ran\n')
    L.append('| job | name | GPU | state | GPU-h |\n|---|---|---|---|---|')
    for x in led['jobs']:
        L.append(f"| {x['job']} | {x['name']} | {x['gpu']} | {x['state']} | {g(x['gpu_hours'], 3)} |")
    L.append('\nTable: Slurm ledger (`runs/GPU-LEDGER.json`, elapsed × GPUs from `sacct`).\n')
    L.append('| cell | eligible refs (E0 / corner / ineligible) | U | median D_bnd (val / test) |\n|---|---|---|---|')
    for geom in CELLS:
        ref, fin, test, inter, labels = data[geom]
        a = ref['allowance']
        L.append(f"| {geom} | {a['n_E0']} / {a['n_corner']} / {a['n_ineligible']} | {g(a['U'])} | "
                 f"{g(inter['val']['D_bnd_median'])} / {g((test.get('inter_test') or {}).get('D_bnd_median'))} |")
    L.append('\nTable: references (val32) and the boundary-interaction diagnostic (A3.2; gate 0.05).\n')
    L.append('## 3. Speed ratios\n')
    L.append('| cell | τ | statistic | S val | S test | best competitor | ' + ' | '.join(FAM_NAME[f] for f in ('sem', 'fdmg', 'rk4', 'ecsw', 'podg')) + ' |')
    L.append('|' + '---|' * 11)
    for geom in CELLS:
        ref, fin, test, inter, labels = data[geom]
        for tau in TAUS:
            for stat in ('worst', 'median'):
                v = fin['summary'].get(f'{tau}|{stat}') or {}
                t = test['summary'].get(f'{tau}|{stat}') or {}
                pf = v.get('per_family') or {}
                L.append(f"| {geom}{'' if geom == 'H' else ' (descriptive)'} | {tau} | {stat} | {g(v.get('S'))} | {g(t.get('S'))} | "
                         f"{FAM_NAME.get(v.get('best_competitor'), '–')} | "
                         + ' | '.join(g(pf.get(f)) for f in ('sem', 'fdmg', 'rk4', 'ecsw', 'podg')) + ' |')
    L.append('\nTable: S = median paired ratio t_competitor / t_bank (S < 1: the bank is slower); "S val" is the max–min '
             'over finalists on val[0:8], "S test" the same on test[0:8] with test-gated finalists; per-family columns '
             'are validation S against each family.\n')
    plot(data)
    L.append('![per-family S](plots/S_per_family.png)\n')
    L.append('Figure: validation $S$ of each competitor family against the bank (worst-error matching; log scale; '
             'dashed line $S=2$ is the win bar, $S<1$ means the bank is slower).\n')
    L.append('## 4. Decision (pre-registered rules)\n')
    L.append(f"- A10 S1 readout: **{s1['A10']}**.")
    for geom in CELLS:
        L.append(f"- Geometry {geom}: **{s1[geom]['geometry_label']}** (A4.2: a WIN needs both wide levels to win; a level "
                 "with a measured loss withdraws the geometry; the box controls only decide attribution of a WIN).")
    L.append('- Test-once finalists that missed τ or failed Q1/Q2 on test32 are excluded from the test S (A4.2); '
             'several bank test finalists failed Q2 or the tight-solve bar, which is why some test S are empty.')
    L.append('- Limitation: the test stage stores only family-level T-dbl pass flags, not the individual ratios.')
    aL = data['L'][0]['allowance']
    L.append(f'- L labels are "no evidence (reference)" because {aL["n_ineligible"]} of 32 validation references fail '
             'the Richardson order gate (A3.3 requires all 32); the measured S values are nevertheless below 2 at every '
             'target.\n')
    L.append('## 5. Deviations, retractions, limitations\n')
    L.append('- The first geometry (side-2 domains) gave no interaction between bumps and features; it was redesigned '
             'by the interaction diagnostic (DESIGN A2). **Retraction (A3.1):** A2 claimed no solver or ROM result existed '
             'at selection time; smoke ladder rows had in fact been produced and partly inspected, so the redesign was '
             'diagnostic-guided problem development, not outcome-blind.')
    to = next(x for x in led['jobs'] if x['state'] == 'TIMEOUT')
    L.append(f"- The first L cell job ({to['job']}) timed out in the ladder ({g(to['gpu_hours'], 3)} GPU-h); the "
             "registered reductions were applied (A9: reduced-arm rank ladder r in {16,32,64,128,256,384}; implementation "
             "variants screened only for the 3 fastest candidates; spectral-element grading off/0.5) and the cell was "
             "resumed; the narrow family, the S2 cells and the "
             "box controls were deferred (A8, A9).")
    sm = sum(x['gpu_hours'] for x in led['jobs'] if x['name'].startswith('smoke'))
    L.append(f"- The pre-release smoke allowance of 1 GPU-h was exceeded by {g(sm - 1, 2)} GPU-h (recorded).")
    L.append('- Results are scoped to this family, the field output contract, these implementations and one A100-80GB.')
    L.append('- Competitors are admitted on validity and accuracy; their Q1 is reported, not gated (A4.2); this can only '
             'lower S.')
    L.append('- Static condensation and multi-domain Chebyshev were not built; C-SEM uses Jacobi-PCG or an exact '
             'eigendecomposition solve.\n')
    L.append('## 6. Glossary\n')
    for k, v in [('S', 'speed ratio: competitor time over bank time at matched error; S ≥ 2 is the win bar; S < 1 means the bank is slower'),
                 ('τ, worst / median', 'error target; the worst or median relative L2 error over the 32 validation cases'),
                 ('best competitor', 'the competitor family whose matched configuration gives the smallest S'),
                 ('bank', 'the coordinate-network ordered span with a distance envelope (our method), CN-R LSPG stepper'),
                 ('POD-Galerkin / POD + ECSW', 'linear reduced models built from the same training data on the FD grid, the second with empirical cubature'),
                 ('U, eligible', 'reference allowance (2 × max Richardson estimate + time check); a target is usable only if all 32 references are eligible and U ≤ τ/5'),
                 ('D_bnd', 'difference between the domain solution and a box evolution of the same initial condition: how much the feature matters'),
                 ('test-once', 'a held-out cohort scored once with the frozen validation-selected configurations'),
                 ('A4.2, A10', 'the registered decision table and the coordinator S1 conditional (DESIGN.md)'),
                 ('L, H', 'the L-shaped domain (unit square minus a corner quadrant) and the unit square with a circular hole'),
                 ('S1, S2', 'the two viscosity (steepness) levels; only S1 was run'),
                 ('wide family', 'initial bumps with centres spread over a 0.4-wide square (minus a small exclusion zone near the features)'),
                 ('E0 / corner-limited', 'reference cases whose observed Richardson order is in [1.8, 2.2] / [1.3, 2.5]; others are ineligible'),
                 ('Richardson estimate', 'reference-error estimate from three grid levels; a conditional estimate, not a bound'),
                 ('Q1 / Q2', 'numerical qualification of a reduced configuration: solver certificates (Q1) and quadrature-rule convergence (Q2)'),
                 ('T-dbl, T-id, sentinel', 'timing controls: serial two-case call vs the sum of single calls; timed outputs equal the official outputs; a reference timing repeated before and after'),
                 ('FD, RK4, SEM', 'finite differences (with multigrid or explicit Runge-Kutta 4) and nodal spectral elements'),
                 ('CN-R LSPG', 'Crank-Nicolson with two backward-Euler start steps, least-squares Petrov-Galerkin residual minimisation'),
                 ('finalists, max-min', 'up to two fastest qualified configurations per family; S = max over bank finalists of min over competitor finalists'),
                 ('S test', 'the same statistic on the held-out test cases, using only finalists that meet the target and qualify on test')]:
        L.append(f'- **{k}:** {v}')
    (HERE / 'report.md').write_text('\n'.join(L) + '\n')
    (HERE / 'EXPORT.json').write_text(json.dumps(exp, indent=1, default=str))
    print('report.md and EXPORT.json written')


if __name__ == '__main__':
    main()
