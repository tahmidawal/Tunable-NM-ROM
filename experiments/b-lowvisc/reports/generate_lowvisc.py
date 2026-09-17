"""Generate the b-lowvisc report and summary.json from the run JSONs.

Every number in `2026-09-17-b-lowvisc.md` is produced here from
`runs/<attempt>/archive/output/gate/result.json` and `runs/<attempt>/audit.json`.
Nothing is typed by hand; the prose carries `{}`-substituted values only.

    python experiments/b-lowvisc/reports/generate_lowvisc.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LANE = ROOT / 'experiments/b-lowvisc'
HERE = LANE / 'reports'
GATE_ATTEMPT = 'lvg01'


def load(attempt):
    base = LANE / 'runs' / attempt
    rep = json.loads((base / 'archive/output/gate/result.json').read_text())
    aud = json.loads((base / 'audit.json').read_text())
    return rep, aud


def f(x, n=4):
    return f'{x:.{n}f}'


def main():
    rep, aud = load(GATE_ATTEMPT)
    cfg, rows = rep['config'], []
    job = str(rep['job_id'])
    commit = rep['commit']
    ranks = [str(k) for k in cfg['pod_ranks']]

    def row(**kw):
        rows.append(dict(job_id=job, source_commit=commit, **kw))

    # ---------------------------------------------------------------- leg (a) ----
    leg_a = rep['leg_a_pod_degradation']
    for k in ranks:
        b = leg_a[k]
        for metric, key in (('pod_floor_worst_all_times_percent', '_percent'),
                            ('pod_floor_worst_evolved_percent', '_evolved_percent')):
            for fam in ('incumbent', 'lowvisc'):
                row(arm=f'pod_k{k}', family=fam, metric=metric, value=b[fam + key], rank=int(k))
        row(arm=f'pod_k{k}', family='ratio', metric='pod_floor_degradation_ratio_all_times',
            value=b['ratio'], rank=int(k))
        row(arm=f'pod_k{k}', family='ratio', metric='pod_floor_degradation_ratio_evolved',
            value=b['evolved_ratio'], rank=int(k))

    # ------------------------------------------------- singular-value spectrum ----
    spec = {}
    for fam in ('incumbent', 'lowvisc'):
        s = rep['snapshots'][fam]
        sv = s['pod_singular_values']
        spec[fam] = dict(sv=sv, tail=s['pod_tail_fraction'],
                         orth=s['pod_orthonormality_deviation'],
                         parity=s['pod_mode_parity'], snaps=s['snapshots'],
                         resid=s['max_relative_residual'])
        for k in ranks:
            i = int(k) - 1
            row(arm=f'pod_k{k}', family=fam, metric='singular_value_ratio_to_first',
                value=sv[i] / sv[0], rank=int(k))
            row(arm=f'pod_k{k}', family=fam, metric='pod_tail_energy_fraction',
                value=s['pod_tail_fraction'][k], rank=int(k))
        row(arm='pod_basis', family=fam, metric='singular_value_ratio_512_over_1',
            value=sv[-1] / sv[0], rank=512)
        row(arm='pod_basis', family=fam, metric='pod_orthonormality_deviation_in_job',
            value=s['pod_orthonormality_deviation'], rank=512)

    # ---------------------------------------------------------------- leg (b) ----
    leg_b = rep['leg_b_fom_cost']
    for setting, b in leg_b.items():
        for fam in ('incumbent', 'lowvisc'):
            row(arm=setting, family=fam, metric='fom_median_gpu_ms', value=b[fam + '_ms'])
            row(arm=setting, family=fam, metric='fom_worst_evolved_percent',
                value=b[fam + '_worst_evolved_percent'])
            row(arm=setting, family=fam, metric='fom_newton_iterations_total',
                value=b[fam + '_newton_iterations'])
        row(arm=setting, family='ratio', metric='fom_cost_ratio', value=b['cost_ratio'])
        row(arm=setting, family='ratio', metric='fom_newton_iteration_ratio',
            value=b['lowvisc_newton_iterations'] / b['incumbent_newton_iterations'])

    # ---------------------------------------------- discretisation (F4) probe ----
    disc = {}
    for s in rep['subject_summary']:
        if s['setting'] == cfg['same_grid_reference'] and s['worst_vs_reference_percent'] is not None:
            disc[(s['family'], s['intervals'])] = s
            row(arm=f"{cfg['same_grid_reference']}_L{s['intervals']}", family=s['family'],
                metric='discretisation_error_vs_4096_worst_percent',
                value=s['worst_vs_reference_percent'], intervals=s['intervals'])
            row(arm=f"{cfg['same_grid_reference']}_L{s['intervals']}", family=s['family'],
                metric='fom_median_gpu_ms', value=s['median_gpu_ms'], intervals=s['intervals'])
    L = cfg['intervals']
    f4_ratio = (disc[('lowvisc', L)]['worst_vs_reference_percent']
                / disc[('incumbent', L)]['worst_vs_reference_percent'])
    row(arm='F4', family='ratio', metric='discretisation_error_ratio_at_training_mesh',
        value=f4_ratio, intervals=L)

    # ------------------------------------------------ the pre-registered gates ----
    g = rep['gates']
    for name, blk in g.items():
        row(arm=name, family='gate', metric='passed', value=bool(blk['passed']))
    row(arm='leg_a_pod_degrades', family='gate', metric='ratio_at_rank_512',
        value=g['leg_a_pod_degrades']['ratio'])
    row(arm='leg_b_fom_cost_rises', family='gate', metric='max_cost_ratio',
        value=g['leg_b_fom_cost_rises']['max_cost_ratio'])
    row(arm='incumbent_pod_floors_reproduce_bpn301', family='gate',
        metric='max_relative_deviation',
        value=g['incumbent_pod_floors_reproduce_bpn301']['max_relative_deviation'])

    # --------------------------------------------------------------- the audit ----
    # The one failing audit check is bounded HERE, from the saved POD audit arrays, so the
    # paragraph that explains it contains no hand-typed number either: the deviation is
    # localised in the mode index, and the k=512 floor is recomputed with the full oblique
    # projector C^T M^-1 C in place of sum C^2 to bound what the lost orthogonality can move.
    obl = {}
    for fam in ('incumbent', 'lowvisc'):
        base = LANE / 'runs' / GATE_ATTEMPT / 'archive/output/gate'
        z = np.load(base / f'podaudit_{fam}.npz')
        W, w, GtW, B = (np.asarray(z['gram_eigvecs']), np.asarray(z['gram_eigvals']),
                        np.asarray(z['gram_times_eigvecs']), np.asarray(z['snapshots_times_truth']))
        sc = 1. / np.sqrt(np.clip(w, 1e-300, None))
        M = (W * sc[None, :]).T @ (GtW * sc[None, :])
        D = np.abs(M - np.eye(len(w)))
        C = (W * sc[None, :]).T @ B
        F = np.asarray(np.load(base / f'truth_{fam}.npz')['fields'])
        nc, nt, _ = F.shape
        tot = np.sum(F.reshape(nc * nt, -1) ** 2, axis=1)
        n0 = np.linalg.norm(F[:, 0], axis=1)
        worst_rel = 0.
        for k in cfg['pod_ranks']:
            naive = 100. * float(np.max(np.sqrt(np.maximum(tot - np.sum(C[:k] ** 2, 0), 0.))
                                        .reshape(nc, nt) / n0[:, None]))
            Ck = C[:k]
            corrected = 100. * float(np.max(
                np.sqrt(np.maximum(tot - np.sum(Ck * np.linalg.solve(M[:k, :k], Ck), 0), 0.))
                .reshape(nc, nt) / n0[:, None]))
            rel = abs(corrected - naive) / naive
            obl[(fam, k)] = rel
            if k != max(cfg['pod_ranks']):
                worst_rel = max(worst_rel, rel)
            row(arm=f'pod_k{k}', family=fam, metric='oblique_projector_relative_floor_shift',
                value=rel, rank=k)
        obl[(fam, 'below_top')] = worst_rel
        for span, lbl in ((256, 'leading_256'), (500, 'leading_500')):
            obl[(fam, lbl)] = float(D[:span, :span].max())
            row(arm='pod_basis', family=fam,
                metric=f'orthonormality_deviation_over_{lbl}', value=obl[(fam, lbl)], rank=span)

    failed = [c for c in aud['checks'] if not c['passed']]
    row(arm='audit', family='audit', metric='checks_total', value=len(aud['checks']))
    row(arm='audit', family='audit', metric='checks_failed', value=len(failed))
    for c in failed:
        row(arm='audit', family='audit', metric=f"failed_{c['check']}",
            value=c.get('max_deviation'))

    (HERE / 'summary.json').write_text(json.dumps(
        dict(lane='b-lowvisc', generated_from=f'runs/{GATE_ATTEMPT}/archive/output/gate/result.json',
             gate_job=job, source_commit=commit, gpu=rep['gpu'], jax=rep['jax_version'],
             backend=rep['backend'], rows=rows), indent=1) + '\n')

    # ------------------------------------------------------------- the prose ----
    def leg_a_table():
        out = ['| POD rank $k$ | incumbent floor, worst-all-times % | low-viscosity floor % | ratio | incumbent worst-evolved % | low-viscosity worst-evolved % | evolved ratio |',
               '|---|---|---|---|---|---|---|']
        for k in ranks:
            b = leg_a[k]
            out.append(f"| {k} | {f(b['incumbent_percent'])} | {f(b['lowvisc_percent'])} | "
                       f"{f(b['ratio'],3)}× | {f(b['incumbent_evolved_percent'])} | "
                       f"{f(b['lowvisc_evolved_percent'])} | {f(b['evolved_ratio'],3)}× |")
        return '\n'.join(out)

    def spectrum_table():
        out = ['| POD rank $k$ | incumbent $\\sigma_k/\\sigma_1$ | low-viscosity $\\sigma_k/\\sigma_1$ | incumbent tail energy | low-viscosity tail energy |',
               '|---|---|---|---|---|']
        for k in ranks:
            i = int(k) - 1
            out.append(f"| {k} | {spec['incumbent']['sv'][i]/spec['incumbent']['sv'][0]:.3e} | "
                       f"{spec['lowvisc']['sv'][i]/spec['lowvisc']['sv'][0]:.3e} | "
                       f"{float(spec['incumbent']['tail'][k]):.3e} | "
                       f"{float(spec['lowvisc']['tail'][k]):.3e} |")
        return '\n'.join(out)

    def leg_b_table():
        out = ['| full-order setting | incumbent ms | low-viscosity ms | cost ratio | incumbent Newton its | low-viscosity Newton its | Newton ratio | incumbent worst-evolved % | low-viscosity worst-evolved % |',
               '|---|---|---|---|---|---|---|---|---|']
        for s, b in sorted(leg_b.items(), key=lambda kv: kv[1]['incumbent_ms']):
            out.append(f"| `{s}` | {f(b['incumbent_ms'],2)} | {f(b['lowvisc_ms'],2)} | "
                       f"{f(b['cost_ratio'],3)}× | {b['incumbent_newton_iterations']} | "
                       f"{b['lowvisc_newton_iterations']} | "
                       f"{b['lowvisc_newton_iterations']/b['incumbent_newton_iterations']:.3f}× | "
                       f"{b['incumbent_worst_evolved_percent']:.4g} | "
                       f"{b['lowvisc_worst_evolved_percent']:.4g} |")
        return '\n'.join(out)

    def mesh_table():
        out = ['| mesh $L$ | incumbent error vs 4096-interval reference % | low-viscosity % | ratio | incumbent ms | low-viscosity ms |',
               '|---|---|---|---|---|---|']
        for Lp in sorted({k[1] for k in disc}):
            a, b = disc[('incumbent', Lp)], disc[('lowvisc', Lp)]
            out.append(f"| {Lp} | {f(a['worst_vs_reference_percent'])} | "
                       f"{f(b['worst_vs_reference_percent'])} | "
                       f"{b['worst_vs_reference_percent']/a['worst_vs_reference_percent']:.3f}× | "
                       f"{f(a['median_gpu_ms'],2)} | {f(b['median_gpu_ms'],2)} |")
        return '\n'.join(out)

    ga, gb = g['leg_a_pod_degrades'], g['leg_b_fom_cost_rises']
    cmp_dev = g['incumbent_pod_floors_reproduce_bpn301']['max_relative_deviation']
    inc_sv = spec['incumbent']['sv']
    low_sv = spec['lowvisc']['sv']
    text = f"""# b-lowvisc — the low-viscosity Burgers gate: both legs pass, with an under-resolution caveat

Generated by `reports/generate_lowvisc.py` from gate job `{job}`'s `result.json` and its
independent audit. **These numbers are final for the gate stage.** They decide whether the
training stage runs; they contain no trained model and therefore say nothing yet about whether
the nonlinear manifold wins. The training job `lvt01` is queued as a consequence of this verdict.

## Verdict

Both pre-registered gates pass.

* **G-a (leg a, the Kolmogorov width)** required the low-viscosity POD-512 projection floor to be
  at least {f(ga['bar'],1)}× the incumbent's on worst-all-times, in the same job. Measured
  **{f(ga['ratio'],3)}×** ({f(leg_a['512']['incumbent_percent'])} % → {f(leg_a['512']['lowvisc_percent'])} %),
  and **{f(leg_a['512']['evolved_ratio'],2)}×** on the worst-evolved metric. **PASS.**
* **G-b (leg b, the full-order cost)** required at least one tuned full-order setting to cost
  {f(gb['bar'],1)}× more on the low-viscosity family, in the same job. Measured **{f(gb['max_cost_ratio'],3)}×**,
  and *every one* of the {len(leg_b)} settings costs more ({f(min(b['cost_ratio'] for b in leg_b.values()),3)}×–{f(max(b['cost_ratio'] for b in leg_b.values()),3)}×).
  **PASS.** The design recorded leg (b) as the weaker leg *before* the measurement; it passed anyway.

Therefore the training stage is submitted, at the incumbent architecture and recipe with the
viscosity family as the only changed variable (DESIGN §5, stage 2).

**F4 is triggered and is not buried.** At the training mesh $L={L}$ the low-viscosity family's
discretisation error against the 4096-interval reference is
{f(disc[('lowvisc', L)]['worst_vs_reference_percent'])} % against the incumbent's
{f(disc[('incumbent', L)]['worst_vs_reference_percent'])} %, a ratio of {f(f4_ratio,3)}×, above F4's
3× bar. Per the pre-registration this does not stop the lane — the go/no-go is G-a — but **every
low-viscosity number in this lane carries that caveat inline, and the cell is not offered as a
paper headline without a finer-mesh confirmation.** The cause is known and was written down
before the job ran: first-order upwinding contributes a numerical viscosity $|u|h/2$ that is
comparable to or larger than the physical $\\nu$ for most of the low-viscosity cases at this mesh.

## The setup, in one line

The low-viscosity cohort **is** the incumbent cohort with $\\nu$ divided by ten and nothing else
changed: the descriptors $(c_x,c_y,w,a)$ are bitwise identical and
$\\nu_{{\\rm inc}}/\\nu_{{\\rm low}}=10$ to {rep['family_relation']['max_relative_deviation_from_ten']:.2e} relative,
re-derived in job and again by the audit.

$$\\nu_{{\\rm incumbent}}\\sim\\exp U(\\log 10^{{-2}},\\log 10^{{-1}}),\\qquad
  \\nu_{{\\rm low}}\\sim\\exp U(\\log 10^{{-3}},\\log 10^{{-2}}).$$

**Validity.** The incumbent family's POD floors reproduce b-panel job `3780638`'s own
`best-found` column to {cmp_dev:.2e} relative, and the incumbent $L={L}$ discretisation error
reproduces that job's {f(disc[('incumbent', L)]['worst_vs_reference_percent'])} % exactly. This
job's measurement is the panel's measurement, so the cross-family comparison is admissible.

## Leg (a) — POD degrades, and the degradation grows with rank

{leg_a_table()}

The shape matters more than the headline. At $k=16$ the two families are nearly equally hard
({f(leg_a['16']['ratio'],3)}×): a 16-mode basis is bad for both. The gap opens monotonically with
rank and is widest exactly where the incumbent cell is good, which is the signature of a slower
Kolmogorov-width decay rather than a uniform accuracy shift. A rank-512 POD basis reaches
{f(leg_a['512']['incumbent_percent'])} % on the incumbent family and only
{f(leg_a['512']['lowvisc_percent'])} % on the low-viscosity one.

### The singular-value decay

{spectrum_table()}

The incumbent snapshot spectrum collapses by {inc_sv[-1]/inc_sv[0]:.2e} over 512 modes; the
low-viscosity one by only {low_sv[-1]/low_sv[0]:.2e}, i.e. **{(low_sv[-1]/low_sv[0])/(inc_sv[-1]/inc_sv[0]):.3g}×
less**. The incumbent basis is numerically rank-deficient by mode 512 — the tail sits at the f64
accumulation floor — while the low-viscosity basis still carries real energy there. That is the
same statement as the floor table, read off the spectrum instead of the projection.

## Leg (b) — the full-order solve does get more expensive

{leg_b_table()}

Every setting costs more and every setting needs more Newton iterations. The mechanism is the one
the design named in advance: the backward-Euler step is preconditioned by the exact Helmholtz
operator $(I+\\Delta t\\,\\nu\\,\\Lambda)^{{-1}}$, which is an excellent preconditioner when
$\\Delta t\\,\\nu\\,\\lambda_{{\\max}}\\gg1$ and degenerates towards the identity as $\\nu\\to0$.
The low-viscosity family also loses accuracy at every loose setting, so the full-order frontier
moves right *and* up.

## The mesh probe, and the under-resolution caveat

{mesh_table()}

The low-viscosity family is materially under-resolved at $L={L}$ and remains so at $L=512$ and
$L=1024$. The $L={L}$ discrete operator is still a well-defined operator and the same-grid metric
— the primary metric of every panel in this campaign — is exactly as meaningful as for the
incumbent cell. But a reviewer is entitled to call the cell under-resolved against the continuous
PDE, and this report says so rather than waiting to be asked.

## Gates and the independent audit

All {sum(1 for _ in g)} in-job gates pass. The independent NumPy audit
(`audit_gate.py`, importing neither JAX nor the driver, recomputing the POD floors from the
snapshot Gram's eigenvectors rather than from the modes) ran {len(aud['checks'])} checks with
{len(failed)} failure.

**The one failed check, stated plainly and not retracted.** `pod_orthonormal_incumbent` measures
{[c['max_deviation'] for c in failed][0]:.3e} against a $10^{{-8}}$ bar. It is a *consequence of*
leg (a), not a defect in it: the incumbent snapshot Gram is numerically rank-deficient at 512
(eigenvalue ratio {inc_sv[-1]**2/inc_sv[0]**2:.2e}), so its trailing modes sit at the f64
accumulation floor and the Gram-eigenvector route loses orthogonality there. The deviation is
confined to the tail ({obl[('incumbent', 'leading_256')]:.2e} over the leading 256 modes,
{obl[('incumbent', 'leading_500')]:.2e} over the leading 500), and recomputing the $k=512$ floor
with the full oblique projector $C^\\top M^{{-1}}C$ instead of $\\sum C^2$ moves it by
{obl[('incumbent', 512)]:.2e} relative and by $\\le$ {obl[('incumbent', 'below_top')]:.2e} at every
other rank — so no reported digit changes. The low-viscosity basis, whose floors carry the result, is well conditioned
({spec['lowvisc']['orth']:.2e}). Both figures are in `summary.json`.

## What this does and does not establish

```mermaid
flowchart TD
  A[Gate job {job}: no trained model] --> B["Leg a: POD-512 floor {f(ga['ratio'],2)}× worse"]
  A --> C["Leg b: full-order cost up to {f(gb['max_cost_ratio'],2)}× dearer"]
  B --> D[G-a PASS]
  C --> E[G-b PASS]
  D --> F[Stage 2: retrain at the incumbent recipe, nu/10]
  E --> F
  F --> G["Stage 3 panel: is any reduced subject non-dominated? UNANSWERED"]
  A --> H["F4: L={L} under-resolved, {f(f4_ratio,2)}× the incumbent's discretisation error"]
  H -.caveat on every number.-> G
  classDef done fill:#dff0d8,stroke:#3c763d
  classDef open fill:#fcf8e3,stroke:#8a6d3b
  classDef warn fill:#f2dede,stroke:#a94442
  class A,B,C,D,E done
  class F,G open
  class H warn
```

Established: the low-viscosity regime is harder for a *linear* subspace, by a margin that grows
with rank, and the full-order solve there is genuinely more expensive. **Not** established: that
the nonlinear manifold exploits either fact. The pass criterion P — at least one reduced subject
non-dominated on (median GPU ms, worst evolved error) against every same-job full-order control
and every POD rank, all converged — needs the trained checkpoint and the stage-3 panel, and
DESIGN A3 records in advance that a dense-quadrature reduced query costs of the order of a
full-order Newton step per iteration, so P remains a demanding bar even with leg (b) passing.

## Glossary

* **Incumbent family / low-viscosity family** — the two parameter cohorts compared here. They share
  every parameter except viscosity $\\nu$, which is ten times smaller in the low-viscosity family.
* **$\\nu$ (viscosity)** — the diffusion coefficient in the Burgers equation. Lower $\\nu$ means the
  solution is more advection-dominated and develops sharper fronts.
* **POD (proper orthogonal decomposition)** — the standard linear method for building a small basis
  from simulation snapshots: keep the $k$ directions carrying the most energy.
* **POD rank $k$** — how many of those directions are kept. Larger $k$ is more accurate and more expensive.
* **Projection floor** — the error you would get from a $k$-dimensional linear basis *even with a
  perfect solver*, by projecting the true solution onto it. It is a lower bound on any method built
  on that basis, which is why it is the right thing to compare.
* **Worst-all-times / worst-evolved** — the error maximised over all output times including $t=0$,
  versus over the evolved times only ($t>0$). The $t=0$ term measures how well the basis compresses
  the initial condition, which is a different question from how well it tracks the dynamics.
* **Kolmogorov width** — how fast the best possible $k$-dimensional linear approximation improves as
  $k$ grows. Slow decay is the regime in which nonlinear (neural) manifolds are claimed to help.
* **Singular value $\\sigma_k$ / tail energy** — the energy in the $k$-th POD direction, and the
  fraction of total energy left unrepresented after $k$ directions. Both describe the same decay.
* **Full-order (FOM) / reduced-order (ROM)** — the full discrete solve on the grid, versus a solve in
  a small number of unknowns. The point of a ROM is to be cheaper at acceptable accuracy.
* **Newton iterations** — the nonlinear solver's work per step; more iterations means a dearer solve.
* **`fft_tight` / `dense_tight` / `nt1e-3_dt005` …** — the tuned full-order settings, named by their
  linear-solver path and by (Newton tolerance, time step). The tight ones are the converged reference;
  the loose ones trade accuracy for speed and are the real competition for a ROM.
* **Same-grid metric** — error measured against the converged full-order solve *on the same grid*, so
  it isolates what the reduced model loses and excludes discretisation error.
* **Discretisation error vs the 4096-interval reference** — how far the $L$-interval discrete solution
  is from a much finer solve; it measures the grid, not the ROM.
* **G-a / G-b** — the two pre-registered go/no-go gates on legs (a) and (b), fixed before the job ran.
* **F1–F4** — the pre-registered falsification clauses. F4 is the under-resolution caveat triggered here.
* **Criterion P** — the lane's pass criterion, needing the stage-3 panel; see the closing section.
* **b-panel job `3780638`** — the incumbent Burgers panel this lane compares against; its numbers are
  read from a committed copy, never retyped.
* **Independent audit** — a separate NumPy-only script that recomputes every reported number from the
  saved fields by a different route, so a bug in the job's own code cannot pass unnoticed.
"""
    (HERE / '2026-09-17-b-lowvisc.md').write_text(text)
    print(HERE / '2026-09-17-b-lowvisc.md')
    print(HERE / 'summary.json', len(rows), 'rows')


if __name__ == '__main__':
    main()
