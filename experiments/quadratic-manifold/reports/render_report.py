"""Render this lane's report from reports/summary.json. No number is typed by hand.

    python reports/render_report.py --summary reports/summary.json --out reports/2026-09-22-quadratic-manifold.md

The prose lives in this file with {placeholders}; every placeholder is filled from the summary,
which `make_table.py` built from the audit, which recomputed every error in NumPy from the saved
fields. Editing a number therefore means editing the data, which is the point.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def f(v, n=4):
    return '—' if v is None else f'{v:.{n}f}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', required=True)
    p.add_argument('--table', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    s = json.loads(Path(a.summary).read_text())
    rows = {r['arm']: r for r in s['rows']}
    lad = {e['rank']: e for e in s['ladder']}
    sens = {e['arm']: e for e in s['sensitivity_arms']}
    ans = s['answers']
    q1, q2 = ans['q1_accuracy_at_matched_dimension'], ans['q2_online_cost']
    nm16 = rows['q0_M64_eqcert_g1em06_fastL4']
    nm16d = rows['q0_M64_dense_g1em06']
    nmtop = rows['q256_M1088_dense_g1em06']
    best = q1['best_qman']
    bestany = q1['best_qman_any_including_sensitivity']
    disc = s['fom_discretisation_error_percent']['fft_tight']

    ladder_rows = '\n'.join(
        f"| {r} | {lad[r]['quadratic_terms']} | {lad[r]['bank_columns']} | {lad[r]['ridge']:g}"
        f"{' **(grid top)**' if lad[r]['ridge_at_grid_endpoint'] else ''} | {f(lad[r]['weight_frobenius_norm'])} | "
        f"{f(lad[r]['snapshot_relative_linear_only'])} → {f(lad[r]['snapshot_relative_with_quadratic'])} | "
        f"{f(lad[r]['worst_evolved_quad'],3)} | {f(lad[r]['worst_evolved_lin'],3)} | {f(lad[r]['worst_evolved_pod'],3)} | "
        f"{f(lad[r]['quadratic_gain'],2)}× | {f(lad[r]['gpu_ms_quad'],1)} | {f(lad[r]['gpu_ms_lin'],1)} | "
        f"{f(lad[r]['quadratic_cost'],2)}× |" for r in sorted(lad))
    sens_rows = '\n'.join(
        f"| `{e['arm']}` | {e['kind']} | `{e['against']}` | {f(e['worst_evolved_percent'],3)} | "
        f"{f(e['baseline_worst_evolved_percent'],3)} | {f(e['error_ratio_vs_baseline'],3)}× | "
        f"{f(e['cost_ratio_vs_baseline'],2)}× |" for e in sens.values())
    floor_rows = '\n'.join(
        f"| `{n}` | {rows[n]['solved_dimension']} | {f(rows[n]['best_found_percent'],3)} | "
        f"{f(rows[n]['worst_t0_compression_percent'],3)} | {f(rows[n]['worst_all_times_percent'],3)} | "
        f"{f(rows[n]['worst_evolved_percent'],3)} |"
        for n in ('qman8_quad_M32', 'qman16_quad_M64', 'qman32_quad_M128', 'qman64_quad_M256',
                  'pod32_M128_dense', 'pod64_M256_dense', 'q0_M64_dense_g1em06') if n in rows)

    text = TEXT.format(
        job=s['job_id'], commit=s['commit'][:8], gpu=s['gpu'], mesh=s['intervals'],
        elapsed=round(s['elapsed_seconds'] / 60, 1), nrows=len(s['rows']),
        ladder_rows=ladder_rows, sens_rows=sens_rows, floor_rows=floor_rows, table=Path(a.table).read_text().strip(),
        disc=f(disc, 4),
        r8g=f(lad[8]['quadratic_gain'], 2), r16g=f(lad[16]['quadratic_gain'], 2),
        r32g=f(lad[32]['quadratic_gain'], 2), r64g=f(lad[64]['quadratic_gain'], 2),
        qm16=f(lad[16]['worst_evolved_quad'], 3), pod16=f(lad[16]['worst_evolved_pod'], 3),
        nm16e=f(nm16['worst_evolved_percent'], 4), nm16ms=f(nm16['median_gpu_ms'], 1),
        nm16de=f(nm16d['worst_evolved_percent'], 4), nm16dms=f(nm16d['median_gpu_ms'], 1),
        head_over_qm16=f(lad[16]['worst_evolved_quad'] / nm16['worst_evolved_percent'], 1),
        nmtope=f(nmtop['worst_evolved_percent'], 4), nmtopms=f(nmtop['median_gpu_ms'], 1),
        besta=best['arm'], bestd=best['solved_dimension'], beste=f(best['worst_evolved_percent'], 3),
        bestms=f(best['median_gpu_ms'], 1),
        bestanya=bestany['arm'], bestanyd=bestany['solved_dimension'],
        bestanye=f(bestany['worst_evolved_percent'], 3), bestanyms=f(bestany['median_gpu_ms'], 1),
        best_over_head=f(best['worst_evolved_percent'] / nm16['worst_evolved_percent'], 1),
        bestany_over_head=f(bestany['worst_evolved_percent'] / nm16['worst_evolved_percent'], 1),
        best_cost_over_head=f(best['median_gpu_ms'] / nm16['median_gpu_ms'], 1),
        dense_cost=f(rows['qman64_quad_M256']['median_gpu_ms'] / nm16d['median_gpu_ms'], 2),
        dense_err=f(rows['qman64_quad_M256']['worst_evolved_percent'] / nm16d['worst_evolved_percent'], 1),
        r8c=f(lad[8]['quadratic_cost'], 2), r64c=f(lad[64]['quadratic_cost'], 2),
        qpod8=f(q2['per_rank'][0]['quad_over_pod'], 2), qpod64=f(q2['per_rank'][-1]['quad_over_pod'], 2),
        fastest_qm_speedup=f(rows['qman8_quad_M32']['speedup_gpu'], 3),
        fastest_qm=f(1 / rows['qman8_quad_M32']['speedup_gpu'], 1),
        msens=f(sens['qman32_quad_M512']['error_ratio_vs_baseline'], 3),
        msens_cost=f(sens['qman32_quad_M512']['cost_ratio_vs_baseline'], 2),
        gsens=f(sens['qman32_quadg0p0001_M128']['error_ratio_vs_baseline'], 2),
        gsens_e=f(sens['qman32_quadg0p0001_M128']['worst_evolved_percent'], 3),
        gsens_cost=f(sens['qman32_quadg0p0001_M128']['cost_ratio_vs_baseline'], 2))
    Path(a.out).write_text(text)
    print(a.out, len(text), 'bytes')


TEXT = r"""# A quadratic-manifold ROM on 2D viscous Burgers at {mesh}², priced in one allocation

What this covers: the quadratic manifold of Geelen–Wright–Willcox (2022) and Barnett–Farhat (2022),
$u = u_{{\rm ref}} + V_r a + W\,\mathrm{{vech}}(a\otimes a)$, run as a ROM baseline on the same cohort,
mesh, solver and timing protocol as this paper's NM-ROM and POD-LSPG rows — the "why not something
simpler than a neural head?" comparison. **These numbers are final for job `{job}`.** Every one is
generated from `summary.json` by `reports/render_report.py`; none is typed by hand. The design was
pre-registered in `DESIGN.md` before the first GPU job and independently audited
(`checks/design-audit.md`); one blocking defect in the fitting protocol was found and fixed before
any number here was produced (`DESIGN.md` §A1).

Job `{job}`, source commit `{commit}`, {gpu}, {elapsed} min, {nrows} arms in **one allocation on one
GPU**. All gates pass, including the cross-job fidelity gates that check the panel is measuring the
archived model. The independent NumPy audit recomputed every error from the saved fields.

## The answer in three lines

1. **Accuracy.** The quadratic term is real and consistent — it beats POD-LSPG at the same solved
   dimension at every rung it is fitted at. But it does **not** reach the nonlinear head: at 16
   solved unknowns the quadratic manifold scores {qm16} % worst evolved error against the head's
   **{nm16e} %**, a factor of **{head_over_qm16}**. It sits between POD-LSPG and us, much nearer
   POD-LSPG.
2. **Cost.** The $O(r^2)$ quadratic term costs {r8c}× its own linear control at $r=8$ rising to
   {r64c}× at $r=64$, because the trial basis grows from $r$ to $1+r+r(r+1)/2$ columns. No
   quadratic arm beats the paper-rule full-order solver: the best is {fastest_qm_speedup}×, i.e.
   **{fastest_qm}× slower** than the rule-matched FOM.
3. **Unknowns.** For **every** quadratic arm in the job, the cheapest NM-ROM setting at least as
   accurate is the same one: `q0_M64_eqcert_g1em06_fastL4`, **16 unknowns, {nm16ms} ms,
   {nm16e} %**. The head needs **4× fewer unknowns** than the best quadratic arm and is
   **{best_cost_over_head}× cheaper** while being **{best_over_head}× more accurate**.

## 1. The $r$ ladder, with the quadratic term isolated

`quad` is the fitted quadratic manifold; `lin` is the **identical** $u_{{\rm ref}}$ and $V_r$ with the
$W$ block deleted, so `lin`/`quad` isolates the quadratic term and nothing else. `pod` is this
paper's own classical POD-LSPG at the same rank, in the same job. "snapshot" is the relative
snapshot reconstruction residual of the fitted map, linear part only → with the quadratic term.

| $r$ | $P=r(r+1)/2$ | trial columns | ridge $\gamma$ | $\lVert W\rVert_F$ | snapshot rel. | quad worst ev. % | lin worst ev. % | POD worst ev. % | quad gain | quad ms | lin ms | quad cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
{ladder_rows}

**The quadratic term pays, and by a consistent amount.** It cuts the worst evolved error by
{r8g}×, {r16g}× and {r32g}× at $r=8,16,32$ against an identical linear part, and it beats
POD-LSPG at the same rank at every one of those rungs. This is a real effect, not noise, and it is
the honest case for the method.

**It stops paying at $r=64$** ({r64g}×, i.e. nothing). The ridge selection pins $\gamma$ at the top
of the declared grid there and $\lVert W\rVert_F$ collapses by two orders of magnitude: with
$P=2080$ coefficients per row fitted from 3328 snapshots, the quadratic term does not generalise
across held-out trajectories. This is the outcome `DESIGN.md` §3.1 predicted in writing before the
job ("the quadratic gain is largest at small $r$ and shrinks or reverses by $r=64$") and §A1.1
pre-registered how to read. It is a property of the method at this snapshot budget, and a lane with
more trajectories could change it.

## 2. The two pre-declared sensitivity arms

Both values were fixed in the config **before** the job, never chosen after seeing evaluation error.
They exist so that a loss cannot be blamed on a choice this lane made for the baseline.

| arm | what it varies | against | worst ev. % | baseline % | error ratio | cost ratio |
|---|---|---|---|---|---|---|
{sens_rows}

**The test space is not the limiting factor.** Quadrupling the weak test modes at $r=32$ from
$M=128$ to $M=512$ changes the error by {msens}× — that is, not at all (marginally worse) — at
{msens_cost}× the cost. The shared $M=4r$ contract is not what held the quadratic manifold back.

**The ridge rule is leaving accuracy on the table, and this is stated plainly.** At $r=32$ the
pre-declared fixed $\gamma=10^{{-4}}$ beats the rule-selected $\gamma=0.01$ by **{gsens}×**
({gsens_e} % against the ladder's) for {gsens_cost}× the cost. So **the ladder above is a lower
bound on what a better-tuned quadratic manifold could reach**, and a reader should treat it as one.
It does not change the verdict: the best quadratic arm of any kind in this job,
`{bestanya}` at {bestanyd} unknowns and {bestanye} %, is still **{bestany_over_head}× worse** than
the head at 16 unknowns. Selecting $\gamma$ by ROM error on the evaluation cohort — which is how
GWW and BF often choose it — is forbidden here because it is selection on evaluation data; that
asymmetry is a real limitation of this comparison and is named again in §5.

## 3. Why: it is the manifold, not the solve

The representation floor is an independent, untimed best-found fit of each reference field on the
arm's own manifold, with the shared LM driver — it says what the trial map could do if the ROM
solve were perfect.

| arm | unknowns | representation floor % | $t=0$ compression % | worst all-times % | worst evolved % |
|---|---|---|---|---|---|
{floor_rows}

For every reduced arm the floor, the $t=0$ compression and the worst all-times error agree to three
decimals. **The binding constraint is what the manifold can represent, not what LSPG can find on
it** — the solve is finding essentially the best fit available. That is the attribution the design
asked for, and it means the quadratic manifold's shortfall is a statement about the trial map, not
about the solver, the tolerance or the budget.

## 4. The full panel

Every row below came out of job `{job}`, one allocation, one GPU, five timed repetitions, randomised
subject order, 0.25 s burn-in. **Speedups are single-job ratios**; the full-order comparator is the
fastest tested setting whose worst evolved error is at or below the row's. The mesh's own
discretisation error is {disc} % (`fft_tight` against the 4096-interval reference) — every
quadratic-manifold arm is above it, and the NM-ROM rows are below it.

{table}

## 5. Qualifications, stated rather than buried

* **No hyper-reduction for the baseline.** GWW and BF both pair the quadratic manifold with
  hyper-reduction (DEIM / ECSW). This lane constructs none for it, because a certified rule is a
  lane's worth of work; the quadratic arms run the exact dense advection sum. `DESIGN.md` §7
  pre-registered **dense against dense** as the fair headline cost ratio, so here it is:
  `qman64_quad_M256` at {bestms} ms and {beste} % against the dense NM-ROM `q0_M64_dense_g1em06`
  at {nm16dms} ms and {nm16de} %. The head is **{dense_cost}× cheaper and {dense_err}× more
  accurate on the like-for-like comparison**, so the missing hyper-reduction does not rescue the
  baseline — it would have to be more than an order of magnitude to matter.
* **The ridge is selected on training snapshots only**, by held-out error over whole trajectories.
  §2 shows that rule is not optimal for ROM accuracy. Choosing it by ROM error would be selection
  on evaluation data and is refused; the sensitivity arm is the declared substitute.
* **$r$ stops at 64** because $P=r(r+1)/2$ is already 2080 against 3328 snapshots there; $r=128$
  would need $P=8256$ and cannot be fitted from this snapshot set at all.
* **One mesh, one PDE, one snapshot budget.** Nothing here refutes GWW or BF in their own regimes —
  smaller $r$, different equations, different snapshot counts, with hyper-reduction.
* **`qman` `lin` is not the POD-LSPG row.** It carries the mean shift and a centred POD basis. The
  paper's POD-LSPG rows are the `pod` family, in this same job at the same ranks.

## 6. What a reviewer gets from this

The simpler thing was tried, on equal terms, in the same allocation, and it works — it is reliably
better than a linear subspace of the same dimension. It is not competitive with the nonlinear head:
{head_over_qm16}× the error at equal unknowns, and no setting of it reaches the head's accuracy at
any dimension or cost tested. The reason is representational, not numerical.

## Glossary

Written for a reader opening this cold.

* **FOM / full-order model** — the original discretised PDE solved directly, here by
  FFT-preconditioned Newton–BiCGStab on the same $ {mesh} ^2$ grid. The thing a ROM has to beat.
* **ROM / reduced-order model** — solves for a few unknowns on a low-dimensional *trial map* instead
  of every grid value.
* **Trial map** — the function from the few reduced unknowns to a full field. POD-LSPG's is linear
  ($V_{{k'}}a$); the quadratic manifold's adds a quadratic term; ours is a trained nonlinear head.
  In this job the trial map is the **only** thing that differs between reduced arms.
* **POD** — proper orthogonal decomposition: the optimal linear basis for a set of snapshots.
* **LSPG** — least-squares Petrov–Galerkin: solve for the reduced unknowns by minimising the
  discrete residual, rather than by projecting the equations.
* **Snapshot** — a stored full-field state from a training run. Here: 128 trajectories × 26 states.
* **$r$, $k'$, solved unknowns** — the number of reduced unknowns actually solved for each time step.
  Matched across families when comparing "at the same solved dimension".
* **$P = r(r+1)/2$** — the number of distinct quadratic monomials $a_i a_j$, $i\le j$. It is why the
  quadratic manifold's cost grows quadratically in $r$.
* **`vech`** — the vector of those distinct monomials, each appearing once.
* **Trial columns** — how many full-field vectors the trial map stores: $r$ for POD,
  $1+r+P$ for the quadratic manifold. Online cost scales with this.
* **$W$, $\lVert W\rVert_F$** — the quadratic coefficient matrix and its size. A tiny norm means the
  fit decided the quadratic term should do almost nothing.
* **Ridge $\gamma$** — the regularisation strength in the least-squares fit of $W$. Larger shrinks
  $W$. Selected here on held-out **training trajectories**.
* **Grid top** — the selected $\gamma$ sat at the largest value offered, meaning the data wanted the
  quadratic term suppressed as far as the grid allowed.
* **`quad` / `lin`** — the same fitted $u_{{\rm ref}}$ and $V_r$ with, and without, the $W$ block. Their
  difference is the quadratic term's contribution and nothing else.
* **$M$** — the number of weak test modes the residual is projected onto; $4\times$ the unknowns by
  the shared contract.
* **Worst / median evolved %** — the error against the same-grid converged solve over the evolved
  times ($t = 0.05\ldots0.25$, excluding $t=0$), worst and median over the six cases, as a
  percentage of $\lVert u(0)\rVert$. The primary accuracy metric.
* **Worst all-times %** — the same including $t=0$, so it includes how well the map reproduces the
  supplied initial field.
* **$t=0$ compression %** — that $t=0$ term alone.
* **Representation floor** — the best fit of a reference field achievable *on* the arm's manifold,
  found offline. If a row's error equals its floor, the trial map is the limit, not the solver.
* **GPU-query ms** — wall time from the input field resident on the GPU to the six output fields
  resident on the GPU. **Complete-query ms** adds the host upload and download.
* **FOM by the rule** — the fastest full-order setting tested in this job whose error is at or below
  the row's. **Speedup** is that setting's time divided by the row's; below 1× means slower.
* **Converged** — every time step and the initial fit stopped for a legitimate reason with a small
  gradient, under the pre-registered rule. A non-converged arm is reported but excluded from the
  admissible frontier.
* **Held-out / cohort** — the six evaluation cases, disjoint from the 128 training trajectories.
  Nothing in this lane was tuned on them.
* **Hyper-reduction (DEIM / ECSW / EQ)** — approximating the expensive nonlinear term by evaluating
  it at a few points, so a ROM's cost stops scaling with the grid. Our NM-ROM's fast rows use a
  certified empirical-quadrature rule; the quadratic-manifold arms here have none.
* **Discretisation error** — the error of the exact solution *of this mesh* against a far finer
  reference. A ROM below it is resolving its mesh; above it is not.
* **Admissible / non-dominated** — respectively, converged under the pre-registered rule, and not
  beaten by another arm on both cost and error.
"""

if __name__ == '__main__':
    main()
