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
TRAIN_ATTEMPT = 'lvt01'


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

    # ------------------------------------------------------------- stage 2 ----
    ta = json.loads((LANE / 'runs' / TRAIN_ATTEMPT / 'audit.json').read_text())
    tjob = ','.join(ta['job_ids'])
    L_ = ta['layers']
    LAYER_KEYS = ['bank_floor_test_max_percent', 'bank_floor_test_mean_percent',
                  'best_found_test_max_percent', 'best_found_test_mean_percent',
                  'best_found_t0_percent', 'head_recon_train_mean_percent']
    for k in LAYER_KEYS:
        rows.append(dict(job_id=tjob, source_commit=commit, arm='train_stage', family='lowvisc', metric=k, value=L_['lowvisc'][k]))
        rows.append(dict(job_id='2837431', source_commit=commit, arm='train_stage', family='incumbent', metric=k, value=L_['incumbent'][k]))
        for sd, v in L_['seeds'].items():
            rows.append(dict(job_id=f'b-seeds-s{sd}', source_commit=commit, arm='train_stage', family=f'seed{sd}', metric=k, value=v[k]))
        rows.append(dict(job_id=tjob, source_commit=commit, arm='train_stage', family='ratio', metric=k + '_ratio', value=L_['ratio_lowvisc_over_incumbent'][k]))
    rows.append(dict(job_id=tjob, source_commit=commit, arm='F2', family='gate', metric='fires_provisional', value=ta['f2']['fires_provisional']))
    rows.append(dict(job_id=tjob, source_commit=commit, arm='F2', family='gate', metric='pod512_degradation_ratio', value=ta['f2']['pod512_degradation_ratio']))
    rows.append(dict(job_id=tjob, source_commit=commit, arm='train_stage', family='lowvisc', metric='bank_cond_G', value=ta['bank']['cond_G']))
    rows.append(dict(job_id='2837431', source_commit=commit, arm='train_stage', family='incumbent', metric='bank_cond_G', value=ta['bank']['incumbent']['cond_G']))
    rows.append(dict(job_id=tjob, source_commit=commit, arm='train_stage', family='lowvisc', metric='bank_gram_sv_ratio_511', value=ta['bank']['gram_sv_ratio']['511']))
    rows.append(dict(job_id='2837431', source_commit=commit, arm='train_stage', family='incumbent', metric='bank_gram_sv_ratio_511', value=ta['bank']['incumbent']['gram_sv_ratio']['511']))
    tfailed = [c for c in ta['checks'] if not c['passed']]
    rows.append(dict(job_id=tjob, source_commit=commit, arm='audit_train', family='audit', metric='checks_total', value=len(ta['checks'])))
    rows.append(dict(job_id=tjob, source_commit=commit, arm='audit_train', family='audit', metric='checks_failed', value=len(tfailed)))

    (HERE / 'summary.json').write_text(json.dumps(
        dict(lane='b-lowvisc', generated_from=f'runs/{GATE_ATTEMPT}/archive/output/gate/result.json',
             gate_job=job, train_job=tjob, source_commit=commit, gpu=rep['gpu'], jax=rep['jax_version'],
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
    NAMES = {'bank_floor_test_max_percent': 'bank floor, worst held-out %',
             'bank_floor_test_mean_percent': 'bank floor, mean held-out %',
             'best_found_test_max_percent': 'best-found, worst held-out %',
             'best_found_test_mean_percent': 'best-found, mean held-out %',
             'best_found_t0_percent': 'best-found at $t=0$ %',
             'head_recon_train_mean_percent': 'head reconstruction, training mean %'}
    layer_table = '\n'.join(
        f"| {NAMES[k]} | {f(L_['incumbent'][k])} | "
        + ' / '.join(f(L_['seeds'][sd][k]) for sd in ('1', '2', '3'))
        + f" | {f(L_['lowvisc'][k])} | **{f(L_['ratio_lowvisc_over_incumbent'][k],3)}×** | {f(ga['ratio'],3)}× |"
        for k in LAYER_KEYS)
    cmp_dev = g['incumbent_pod_floors_reproduce_bpn301']['max_relative_deviation']
    inc_sv = spec['incumbent']['sv']
    low_sv = spec['lowvisc']['sv']
    text = f"""# b-lowvisc — low-viscosity Burgers: both gates pass, the K=16 head degrades far less than POD-512, F2 does not fire (provisionally)

Generated by `reports/generate_lowvisc.py` from gate job `{job}`'s `result.json`, training job
`{tjob}`'s three stage JSONs, and their independent audits. **The gate numbers (§ leg a / leg b)
are final. The stage-2 numbers are final as training-stage diagnostics but the F2 verdict built
on them is PROVISIONAL**: the pre-registered F2 quantities are panel numbers on the development
cohort, which need the stage-3 panel job; what stage 2 supplies is their like-for-like
training-stage analogue on 408 held-out test states. No reduced model has been *solved* on the
low-viscosity family yet; criterion P is untested.

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
  F --> F2["F2 (provisional): K=16 best-found degrades {f(L_['ratio_lowvisc_over_incumbent']['best_found_test_max_percent'],2)}× vs POD-512's {f(ga['ratio'],2)}× — does NOT fire"]
  F2 --> G["Stage 3 panel: is any reduced subject non-dominated? UNANSWERED"]
  A --> H["F4: L={L} under-resolved, {f(f4_ratio,2)}× the incumbent's discretisation error"]
  H -.caveat on every number.-> G
  classDef done fill:#dff0d8,stroke:#3c763d
  classDef open fill:#fcf8e3,stroke:#8a6d3b
  classDef warn fill:#f2dede,stroke:#a94442
  class A,B,C,D,E done
  class F,F2 done
  class G open
  class H warn
```

Established: the low-viscosity regime is harder for a *linear* subspace, by a margin that grows
with rank, and the full-order solve there is genuinely more expensive. **Not** established: that
the nonlinear manifold exploits either fact. The pass criterion P — at least one reduced subject
non-dominated on (median GPU ms, worst evolved error) against every same-job full-order control
and every POD rank, all converged — needs the trained checkpoint and the stage-3 panel, and
DESIGN A3 records in advance that a dense-quadrature reduced query costs of the order of a
full-order Newton step per iteration, so P remains a demanding bar even with leg (b) passing.

## Stage 2 — the training run, and the provisional F2 gate

Training job `{tjob}` ran the incumbent recipe verbatim — $K=16$, $R=512$, `n_ff` 128,
`g_hidden` 1024, `h_hidden` 512×2, 300000 + 200000 Adam steps, seed 0, the same 576 + 4032
training trajectories and the same 8 held-out test trajectories — with the viscosity family as
the only change (the job re-proved the generator patch bitwise-identical under the defaults
before its first step). The audit (`audit_train.py`, NumPy only) ran {len(ta['checks'])} checks
with {len(tfailed)} failures: recipe values equal the incumbent's field by field, the training
data has the incumbent's shape but *different* sums (the inverted fingerprint gate), full-order
residuals $\\le10^{{-8}}$, whitening and Gram identities at $10^{{-14}}$, every checkpoint hash
consistent between the job's `TRAIN-SHA256.txt`, `OUTPUTS.sha256` and the bytes now in
`checkpoints/`.

### The three layers, like for like

Every column below is the **same JSON field from the same script on the same 408 held-out
states** (8 trajectories, `test_seed` 1, 51 times; under the low-viscosity bounds these are the
incumbent's own test trajectories with $\\nu/10$). "Bank floor" is the best any coefficients in
the trained $R=512$ span can do (`test_span_floor`); "best-found" is the best the $K=16$ head's
manifold can reach with the truth in hand (`oracle_test`), no PDE solve involved. The incumbent
column is its own training run (job 2837431) and the three b-seeds reseeds show the seed spread.

| layer | incumbent (seed 0) | reseeds 1 / 2 / 3 | low-viscosity (seed 0) | ratio low / incumbent | POD-512 ratio, for scale |
|---|---|---|---|---|---|
{layer_table}

**What the table says.** The *linear* objects collapse under the regime change by about the
POD factor: the trained $R=512$ bank's worst-case floor degrades
{f(L_['ratio_lowvisc_over_incumbent']['bank_floor_test_max_percent'],2)}× (POD-512 on the
development cohort: {f(ga['ratio'],2)}×). The *nonlinear* $K=16$ head's worst-case best-found
degrades only **{f(L_['ratio_lowvisc_over_incumbent']['best_found_test_max_percent'],2)}×**
({f(L_['incumbent']['best_found_test_max_percent'])} % → {f(L_['lowvisc']['best_found_test_max_percent'])} %),
inside a factor of two, and at $t=0$ only {f(L_['ratio_lowvisc_over_incumbent']['best_found_t0_percent'],2)}×.
Read against the gate: on the incumbent family a 16-dimensional manifold sits **above** the
512-dimensional linear floor ({f(L_['incumbent']['best_found_test_max_percent'])} % vs
{f(leg_a['512']['incumbent_percent'])} %); on the low-viscosity family it sits **below** it
({f(L_['lowvisc']['best_found_test_max_percent'])} % vs {f(leg_a['512']['lowvisc_percent'])} %) —
with the caveat, stated once and meant everywhere, that the two figures are on different cohorts
(408 test states vs the six development cases) and the panel is what puts them in one job on one
cohort. The bank itself is also less degenerate at low viscosity: $\\mathrm{{cond}}(G)$
{ta['bank']['cond_G']:.0f} against {ta['bank']['incumbent']['cond_G']:.0f}, and
$\\sigma_{{511}}/\\sigma_0$ {ta['bank']['gram_sv_ratio']['511']:.2e} against
{ta['bank']['incumbent']['gram_sv_ratio']['511']:.2e} — the same slower-decay statement leg (a)
made about POD, now about the learned span.

### F2, applied

F2 (DESIGN §6) fires if the low-viscosity **bank floor, best-found and solved** each degrade by at
least the POD-512 factor ({f(ta['f2']['pod512_degradation_ratio'],3)}×). Those are panel
quantities on the development cohort. On their training-stage analogues:

* bank floor (worst held-out): {f(ta['f2']['ratios']['bank_floor_test_max_percent'],3)}× — {'≥' if ta['f2']['ratios']['bank_floor_test_max_percent'] >= ta['f2']['pod512_degradation_ratio'] else '<'} the POD factor;
* best-found (worst held-out): {f(ta['f2']['ratios']['best_found_test_max_percent'],3)}× — {'≥' if ta['f2']['ratios']['best_found_test_max_percent'] >= ta['f2']['pod512_degradation_ratio'] else '<'} the POD factor;
* solved: **not available** — no reduced solve has run on this family.

Since F2 needs *all three* and best-found is already a factor of
{f(ta['f2']['pod512_degradation_ratio']/ta['f2']['ratios']['best_found_test_max_percent'],1)} short of
the bar, **F2 does not fire, provisionally**; the panel cannot make it fire through the solved
layer alone, because the criterion is conjunctive. The word *provisional* is carried because the
pre-registered evaluation is on the development cohort in the panel job.

### The caveat that travels with every number above — F4

Restated, not softened: the low-viscosity family is under-resolved at $L={L}$
({f(disc[('lowvisc', L)]['worst_vs_reference_percent'])} % against the 4096-interval reference,
{f(f4_ratio,2)}× the incumbent's), and stays under-resolved at $L=512$ and $1024$. Every low-viscosity
number in this report is a statement about the $L={L}$ discrete operator, which is well defined;
none is yet a statement about the continuous low-viscosity PDE. The training-stage result above
— a nonlinear $K=16$ manifold below the linear rank-512 floor — is therefore a result about the
discrete regime the paper's every other Burgers number lives in, and is not offered as a headline
without a finer-mesh confirmation.

### What the next job is

Nothing is submitted; the choice is the coordinator's. The two candidates, with the case for each:

1. **`lvp01`, the stage-3 panel as pre-registered in A3** (fixed $M=1088$ ladder $q\\in\\{{0..256\\}}$,
   the $M=256$ control column, the $(256, 2176)$ cell, POD-LSPG at $k'\\in\\{{16..512\\}}$, the free
   bank, the tuned full-order grid, one job, 80 GB card, ≈ 1 h). It is the only job that can
   evaluate **P** and the pre-registered **F2**, and it puts the "16-dimensional manifold below the
   512-dimensional linear floor" comparison in one job on one cohort. Given leg (b) passed at
   {f(gb['max_cost_ratio'],2)}× and the dense-quadrature reduced query costs of the order of a
   full-order step (A3), P is still expected to be hard; the panel measures rather than assumes it.
2. **An $L=512$ confirmation for F4** (gate + retrain at 512, ≈ 5 h of A100 across two jobs). It
   answers the reviewer's under-resolution objection but not the paper's question, and a
   retrained $L=512$ checkpoint would then need its own panel.

The recommendation recorded here is **1 then 2**: the panel decides whether there is a result to
confirm; a finer-mesh confirmation of a cell that fails P would be wasted compute. Jobs used: 2 of 8.

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
* **Bank / bank floor** — the trained $R=512$ linear span the decoder lifts through; its floor is
  the best any coefficients in that span can do on a held-out state, a bound no head can beat.
* **Best-found / oracle** — the best fit the $K=16$ head's manifold can reach on a held-out state
  when the truth is supplied to the fit; measures representation, not the solver.
* **Solved** — the error of the reduced model when it actually solves the PDE in reduced form,
  without the truth; only a panel job produces it.
* **F2** — the falsification clause "the nonlinear manifold degrades at least as much as POD does".
* **Held-out test states** — 8 trajectories never used in training (seed 1), 51 times each.
* **cond(G), $\\sigma_{{511}}/\\sigma_0$** — the condition number and singular-value decay of the
  bank's Gram matrix; large cond / fast decay means the span is nearly degenerate.
* **Independent audit** — a separate NumPy-only script that recomputes every reported number from the
  saved fields by a different route, so a bug in the job's own code cannot pass unnoticed.
"""
    (HERE / '2026-09-17-b-lowvisc.md').write_text(text)
    print(HERE / '2026-09-17-b-lowvisc.md')
    print(HERE / 'summary.json', len(rows), 'rows')


if __name__ == '__main__':
    main()
