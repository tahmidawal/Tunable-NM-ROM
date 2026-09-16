# Trajectory-fitted correction directions — predeclared design

One question, on the frozen Burgers checkpoint
`sep_hfit_dense_mid_N256_dense.pkl`
(SHA256 `18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589`, $K=16$,
$R=512$), at 256 intervals, $\Delta t = 0.005$, on the same six opened development cases
the head ablation (`abl01`), the audited ladder (`qlad01`), the cheap-corrections cell
(`cclad01`) and the top-of-ladder cell (`btq101` / `btq102` / `btq201`) used
(`eval_seed 7090702` $\times 4$, `eval_fresh_seed 911702` $\times 2$).

> **Does fitting the correction directions to the ROM's TRAJECTORY error, instead of the
> head's static reconstruction residual, make the Burgers $256^2$ ladder monotone on the
> worst-over-evolved-times metric ($t>0$) while keeping it monotone on
> worst-over-all-times?**

Nothing else changes: the checkpoint, the bank, the head, the weak objective, the
initializer policy, the stopping rule (per-step iteration budget **600**, as
`b-ladder-top` Q1-B), the block-damped variable-projection solver, the output contract,
the mesh, the time step and the cohort are the `b-ladder-top` Q1-B contract verbatim.
**Only the matrix $C_q$ changes.**

---

## 1. The inherited contract

With $\Phi\in\mathbb R^{n\times M}$ the $M$ lowest discrete sine test modes
($\Phi^\top\Phi=I$), $\lambda$ their Laplacian eigenvalues, $G\in\mathbb R^{n\times R}$
the frozen separable bank, $A=\Phi^\top G$, backward Euler at step $\Delta t$ and the
FOM's own sign-upwind advection $\mathcal N$, the reduced state is $u=G\eta$ with

$$\eta(z,y)=h_\theta(z)+C_q\,y,\qquad z\in\mathbb R^{K},\ y\in\mathbb R^{q},\ K=16,$$

$$r(w)=\frac{A\eta-p+\Delta t\big(\Phi^\top\mathcal N(G\eta)+\nu\,\lambda\odot A\eta\big)}
{1+\Delta t\,\nu\lambda},\qquad p=A\,\eta^{\rm prev},\qquad w=(z,y).$$

Writing $G=Q_G R_G$ for the thin QR (`arms.whiten`), coefficient least squares in the
$R_G$ metric *is* field-metric least squares, and a set of field-orthonormal directions
$\tilde C$ gives $C = R_G^{-1}\tilde C$ with $G C = Q_G \tilde C$ orthonormal.

The solver is `varpro.make_block_lm` (`fix = 'base'` in `topfix.ARMS`): one Jacobian per
iteration, Levenberg damping and the $q=0$ trust radius on the $z$ block, a fixed
$\varepsilon_0=10^{-10}$ ridge and no radius on the $y$ block. A solve is **converged**
only if every time step and the initial fit exit for a reason in $\{1,2,4\}$ with no
iteration-budget exit *and* the joint normalized gradient

$$g=\frac{\sqrt{\|J_z^{\top}r\|^2+\|J_y^{\top}r\|^2}}
{\sqrt{\|J_z\|_F^2+\|J_y\|_F^2}\;\|r\|}\ \le\ 10^{-6}.$$

## 2. The incumbent direction rule (`old`), restated

`cheap-corrections/directions.py::audited` = `ladder.residual_directions`. For each of
$S=1024$ seeded bank-coefficient snapshots $\eta_i$ (from the 128 training
trajectories, state stride 2, `residual_seed 20260915`), the **best-found static head
code** is found with the shared LM rule in the field metric from 4 multistarts at
budget 200,

$$z^\*_i=\arg\min_{z}\ \|R_G\,(h_\theta(z)-\eta_i)\|_2 ,\qquad
\rho^{\rm old}_i=\eta_i-h_\theta(z^\*_i)\in\mathbb R^{R},$$

and $\tilde C^{\rm old}$ is the right-singular basis of the stacked whitened residuals
$\tilde P^{\rm old}=\big[R_G\rho^{\rm old}_i\big]_{i=1}^{S}$, nested in $q$.

**This is a *static* rule.** It asks what the head cannot represent when it is allowed
to fit each snapshot on its own. It never asks what the head's *trajectory* does.

## 3. The new direction rules

### 3.1 Variant A — `traj` (the pre-registered subject)

Let $T\subset\{0,\dots,127\}$ be a seeded subset of the incumbent training draw.

> **`params_draw` landmine (b-speed D3).** `engines.params_draw(seed, count)` draws
> **column by column**, so `params_draw(0, 32)` is *not* a prefix of
> `params_draw(0, 128)`. The full incumbent draw `params_draw(0, 128)` is therefore
> taken **once**, and every direction-fitting cohort is a set of *indices into that one
> array*. This guarantees the direction-fitting trajectories are genuine members of the
> incumbent training family.

For each $j\in T$:

1. **FOM.** Run the same-mesh full-order solver at the **`fft_tight` setting**
   ($\texttt{ntol}=10^{-6}$, $\texttt{ltol}=10^{-8}$, $\Delta t = 0.005$) with output
   spacing $\Delta t$, giving the converged state at **every internal step**
   $u^{\rm FOM}_{j,t}$, $t=0,\dots,50$ (`engines.make_fom(L, dt, None, .25, dt)`).
2. **Bank projection.** The field-metric bank projection of that state,
   $$c^\*_{j,t}=\arg\min_{c}\big\|G c-u^{\rm FOM}_{j,t}\big\|_2=R_G^{-1}Q_G^{\top}u^{\rm FOM}_{j,t}.$$
3. **ROM.** Run the **$q=0$ ROM** — dense quadrature, $M=4K=64$, the retained
   contract, per-step budget 600 — on the same supplied initial field, and take its
   internal latents $z_{j,t}$, $t=0,\dots,50$ (index 7 of the query tuple; $z_{j,0}$ is
   the initial fit to the supplied field).
4. **Trajectory residual.**
   $$\rho^{\rm traj}_{j,t}=c^\*_{j,t}-h_\theta(z_{j,t})\in\mathbb R^{R}.$$

Stack every $(j,t)$ pair, whiten and take the field-metric POD:

$$\tilde P^{\rm traj}=\big[R_G\,\rho^{\rm traj}_{j,t}\big],\qquad
\tilde P^{\rm traj}=U\Sigma V^{\top},\qquad
\tilde C^{\rm traj}=V_{:,1:\text{rank}},\qquad C^{\rm traj}=R_G^{-1}\tilde C^{\rm traj},$$

nested in $q$ by construction. **The POD arithmetic is literally the same function** as
the incumbent rule's (`trajdirs.pod_from_residual`, verified in the local smoke to
reproduce `directions.audited` bitwise when handed the incumbent's own $\rho^{\rm old}$),
so the only difference between `old` and `traj` is *which residual matrix is decomposed*.

**Cohort.** $|T|=32$, `direction_traj_seed = 20260916`, drawn without replacement from
$\{0,\dots,127\}$. $32\times 51=1632 \ge R=512$ residual vectors, so the available rank
covers the whole ladder.

### 3.2 Variant B — `prac` (the practitioner's rule)

The same quantity on a **six-trajectory** cohort: the training-family analogues of the
six evaluation cases. $|T'|=6$, `direction_prac_seed = 20260917`, drawn without
replacement from $\{0,\dots,127\}\setminus T$, so `traj` and `prac` are independent
draws from the same training family rather than one being a subset of the other.

> **Recorded ambiguity and its resolution.** The task's phrasing — "directions from the
> residual of the $q=0$ ROM's OWN rollout against the FOM … on the six-case cohort's
> TRAINING-family analogues" — says "the same quantity", so `prac` differs from `traj`
> **only in the cohort**: 6 training trajectories instead of 32. That is the reading
> implemented. $6\times51=306$ residual vectors, so the available rank is 306, which
> covers the ladder to $q=256$ but would not cover $q=512$; the ladder stops at 256.

**Never the evaluation cases.** Both cohorts are asserted, in the job, to be exactly
disjoint from the six evaluation cases by parameter vector, and the minimum descriptor
distance is recorded.

### 3.3 What is reported about the directions themselves

* $\texttt{directions\_sha256}$ of each of the three matrices, and each matrix saved as
  an artifact `.npz` inside the job output.
* **Residual energy captured per $q$**, self: $\sum_{i\le q}\sigma_i^2/\sum_i\sigma_i^2$
  for each set on its own residual matrix.
* **Cross-capture**, which is the sharp diagnostic: for every ordered pair of residual
  matrix $\tilde P$ and direction set $\tilde C$,
  $$\kappa_q(\tilde P,\tilde C)=\frac{\big\|\tilde P\,\tilde C_{:,1:q}\big\|_F^2}
  {\big\|\tilde P\big\|_F^2}.$$
  $\kappa_q(\tilde P^{\rm traj},\tilde C^{\rm old})$ says how much of the *trajectory*
  error the incumbent directions can reach at all.
* **Principal angles** between $\operatorname{col}\tilde C^{\rm new}_{:,1:q}$ and
  $\operatorname{col}\tilde C^{\rm old}_{:,1:q}$ at $q\in\{16,64,256\}$, from the
  singular values $\sigma_i=\cos\theta_i$ of
  $(\tilde C^{\rm new}_{:,1:q})^{\top}\tilde C^{\rm old}_{:,1:q}$ (both bases are
  field-orthonormal, so these are the field-metric principal angles). Reported as
  $\theta_{\min},\theta_{\rm median},\theta_{\max}$ in degrees plus the subspace overlap
  $\frac1q\sum_i\sigma_i^2$.

## 4. The ladder

$q\in\{0,16,32,64,128,256\}$. Empirical quadrature (EQ) is refitted **per rung** on the
**enriched decoder outputs** of that rung's own direction set —
$w_{j,t}=(z_{j,t},\,y_{j,t})$ with $y_{j,t}=\tilde C_{:,1:q}^{\top}R_G\rho_{j,t}$
(`varpro.enriched_codes`, unchanged) — with the `b-ladder-top` fitter settings:
bounded block-greedy NNLS, target support $m=\min(4M,2048)$, refit block $m/16$,
walltime bound 2400 s, `eq_seed 20259`, 8192 candidate points, $\le 8192$ design rows.
A rule with `truncated == true` is **invalid** and disqualifies its arm.

| block | direction set | $M$ | quadrature | $q$ |
| --- | --- | --- | --- | --- |
| primary | `traj` | $4(K+q)$ | eq | 0, 16, 32, 64, 128, 256 |
| primary dense twins | `traj` | $4(K+q)$ | dense | 0, 16, 64, 256 |
| fixed-$M$ control | `traj` | 256 | eq | 0, 16, 32, 64, 128 |
| primary | `prac` | $4(K+q)$ | eq | 0, 16, 32, 64, 128, 256 |
| primary dense twins | `prac` | $4(K+q)$ | dense | 0, 16, 64, 256 |
| primary control | `old` | $4(K+q)$ | eq, dense | 0, 16, 64 |
| fixed-$M$ control | `old` | 256 | eq | 0, 16, 64 |
| fidelity | — | 64 (retained fitter), 256 | eq, dense | 0 |

At $q=0$ the correction block is empty, so **every direction set gives the identical
arm**; the $q=0$ rows are built once and shared by all three ladders, which is itself
an in-job invariant rather than three near-duplicate runs.

$M=256$ is illegal above $q=239$ ($M>K+q$ is required), so the fixed-$M$ control ladder
stops at $q=128$; that is the same restriction `b-ladder-top` Q2 met.

**Full-order controls, same job, same GPU, same randomized order:** `fft_tight`
($\texttt{ntol}\,10^{-6}$, $\texttt{ltol}\,10^{-8}$, $\Delta t\,0.005$), `fft_loose`
($10^{-2}$, $0.5$, $0.005$), `nt1e-2_dt01` ($10^{-2}$, $0.5$, $0.01$).

**Timing.** Three timed repetitions per (subject, case) with a 0.25 s burn-in before
every invocation, a compile warm-up pass first, and a seeded randomized subject order
within each (rep, case). **Every repetition is retained** in `result.json`; medians are
taken in the audit, never in the job.

## 5. Both metrics, on every row

Against the same-job converged `fft_tight` solve $f^{\rm FOM}$ on the same grid, with
$u_0$ the supplied initial field:

$$\text{t0 compression}=\frac{\|f_0-u_0\|}{\|u_0\|},\qquad
\text{worst all times}=\max_{t\in\{0,0.05,\dots,0.25\}}\frac{\|f_t-f^{\rm FOM}_t\|}{\|u_0\|},$$
$$\text{worst evolved times}=\max_{t\in\{0.05,\dots,0.25\}}\frac{\|f_t-f^{\rm FOM}_t\|}{\|u_0\|}.$$

Worst is over the six cases. The error against the 4096-interval reference is also
recorded for every invocation. Every one of the three is recomputed in the NumPy audit
from the saved fields; nothing is taken on the job's word.

The per-output-time, per-case error curve is retained for every invocation so the
figure's per-time panels are generated from data, not from a summary.

## 6. Pre-registered pass

With the **`traj`** directions, on the primary ($M=4(K+q)$, EQ) ladder:

1. worst **evolved**-times error non-increasing in $q$ from 0 to 256, **with every rung
   converged**; and
2. the worst **all**-times metric also non-increasing in $q$; and
3. the converged non-dominated set on the (median GPU ms, worst evolved error) plane
   spans $\ge 2\times$ in error.

All three must hold. ($q=256$ at $0.76\,\%$ against $q=0$ at $1.90\,\%$ would already
satisfy (3).) Whether the fixed-$M$ ladder and the `prac` variant also pass is reported
beside it but does not decide the verdict.

## 7. Falsification, stated before the run

* **F1.** If the `traj` directions do not remove the $q=16$ regression on the evolved
  metric, the answer is **no** and the report says so in those words. The incumbent's
  regression is `btq201`'s fixed-$M$ EQ ladder: evolved
  $1.3186 \to \mathbf{1.8066} \to 1.3048 \to \mathbf{1.5738} \to 1.3517 \to 0.7580$ at
  $q=0,16,32,64,128,256$ — non-monotone at $q=16$ **and** at $q=64$ **and** at $q=128$.
* **F2.** If they remove it only at the cost of the $t=0$ term — the t0 compression
  column rising, or the all-times metric losing monotonicity — the report says so, with
  both columns beside each other.
* **F3.** If the new rungs stop converging where the old ones converged, the ladder is
  not comparable and the verdict is "not established", not "passed".
* The **`q-diag`** lane is measuring the cause of the $q=16$ regression in parallel from
  the saved fields. If its verdict lands before this report, this report **cites it**
  (read-only, `worktrees/2026-09-16-q-diag/experiments/q-diag/reports/`); if it does not,
  the report says that it did not.

## 8. Gates before any verdict

| # | gate | tolerance |
| --- | --- | --- |
| i | `backend_gpu`, `x64`, `precision_highest`, checkpoint SHA256 unchanged before/after | all |
| ii | evaluation cohort **bitwise** equals `abl01`'s six physical cases | exact |
| iii | every direction-fitting trajectory exactly disjoint from the six evaluation cases | exact |
| iv | each direction matrix hashed and saved as a job artifact | present |
| v | each direction set's available rank $\ge \max q$ | all |
| vi | $C_q$ is the first $q$ columns of $C$ for every $q$ (nesting) | bitwise |
| vii | $q=0$ dense arms reproduce `btq101 q0_m4_dense_base` / `cclad01 q0_m4_dense_block` and `btq201 q0_M256_dense_g1em06` / `cclad01 q0_m256_dense_block` | $\le10^{-9}$ |
| viii | $q=0$ EQ arm with the **retained** fitter reproduces `cclad01 q0_m4_eq_varpro` | $\le10^{-9}$ |
| ix | `old` $q=16$ / $q=64$ dense arms reproduce `cclad01 q16_m4_dense_block` / `q64_m4_dense_block` | $\le10^{-9}$; a $10^{-3}$ second tier is reported separately when the regenerated `directions_sha256` is not bitwise (the inherited cross-node behaviour), and a pass at the second tier only is reported as a **fail at $10^{-9}$**, never as a pass |
| x | independent NumPy audit recomputes every reported error from the saved fields | $<10^{-9}$ |
| xi | every solve carries exit reason, budget-exit count and all stationarity norms | present |
| xii | every EQ rule carries `truncated`, `support`, `relative_fit`, `fit_seconds`; truncated $\Rightarrow$ arm invalid | present |
| xiii | every ROM invocation satisfies $M>K+q$ | all |
| xiv | every (subject, case) has all 3 repetitions and identical output across them | exact |
| xv | reference solves' max relative residual $<2\times10^{-11}$ | all |
| xvi | same-grid baseline (`fft_tight`) present for all six cases | all |

## 9. Routine calls made without asking

Recorded here rather than folded in silently.

1. **One cluster job**, `--gpu a100`, excluding `pax007` (failed `cuInit` in this
   campaign), `gpu` partition, matching the GPU class of every comparator job so the
   cross-job fidelity gates have their best chance. The 3-job cap leaves two spare.
2. **Per-step iteration budget 600 everywhere**, as `b-ladder-top` Q1-B, including the
   $q=0$ direction-generating rollout. At $q=0,16,64$ the comparator jobs recorded zero
   budget exits at budget 180, so raising the budget cannot change those trajectories
   and the fidelity gates stay exact; the job records the exit histogram to prove it.
3. **`fix = 'base'` on every arm.** `b-ladder-top` measured `pre`, `damp`, `predamp`
   and `casc` to be inert or harmful; re-running them would buy a known answer.
4. **The direction-generating $q=0$ ROM uses the dense quadrature at $M=64$**, so the
   directions never depend on an NNLS rule that itself depends on the directions.
5. **The direction-target FOM uses the `fft_tight` setting**, not the tighter
   `snapshot_ntol` $10^{-9}$ used to build the bank-projection snapshots. The metric the
   ladder is graded on is the same-grid error against `fft_tight`, so the directions aim
   at exactly the error being measured. Both tolerances are recorded.
6. **The reconstruction (best-found-on-manifold) diagnostic is computed for every
   (direction set, $q$) pair on the ladder**, because the static best-found floor is the
   confounder that would otherwise explain a trajectory-metric change.
7. **Three repetitions, all retained**; medians taken only in the audit.
8. **The branch is not pushed** and nothing is merged.

## 10. Amendments

*(Appended as they are made; nothing above is edited after the first commit.)*

### After the local smoke, before submission

* **A1 — the self-comparison tolerance on principal angles.** The first smoke asserted
  that a direction set compared against itself gives principal angles below $10^{-6}$
  degrees. It does not: $\arccos$ is $\sqrt{\cdot}$-ill-conditioned at $\sigma=1$, so a
  bitwise-identical subspace lands at $2.8\times10^{-5}$ degrees. The gate is now on the
  well-conditioned quantity, the overlap $\frac1q\sum\sigma_i^2$, required within
  $10^{-12}$ of 1, with the angle bound loosened to $10^{-3}$ degrees and both reported.
  The angles themselves are still reported because they are the readable summary.
* **A2 — the local smoke ran 134 s, not sub-minute.** Two sibling agents share the GB10.
  Recorded as a deviation from `CLAUDE.md`; the five gates each need a real q = 0 query,
  a real direction fit and a real NNLS rule, and splitting them into five sub-minute
  processes would have paid five JAX start-ups instead.
* **A3 — the GB10 is not bit-reproducible across processes.** Two runs of the identical
  smoke produced `directions_sha256` `61c11777…` and `9c99805b…`, and the $q=0$ field
  reproduced the saved case at $2.00\times10^{-14}$ and $1.50\times10^{-14}$. Within one
  process everything is bitwise. This is the `b-speed` D1 reassociation class showing up
  again and is the reason the cross-job `directions_sha256` probe is informational, never
  a gate.
* **A4 — what the smoke already shows, on a tiny 64-interval cohort.** The incumbent
  directions capture **3.9 % / 6.1 %** of the trajectory residual's energy at $q=4/8$,
  and the trajectory directions capture **3.3 % / 5.9 %** of the incumbent residual's
  energy. The two subspaces are close to disjoint even at the bottom of the ladder. The
  smoke's `old` rule is deliberately under-sampled (16 snapshots), so this is a signal
  that the experiment can separate the two rules, not a result.

---

## 11. Amendment, 2026-09-16 — REDIRECT after the `q-diag` lane reported

The `q-diag` lane published its diagnosis
(`worktrees/2026-09-16-q-diag/experiments/q-diag/reports/2026-09-16-q16-regression-diagnosis.md`,
SHA256 `c0bf57626d9e0a422965446b39e79e423c9a39f6886b7ca1893b8a03cd803c6e`) **while this
lane's job `qtd01` was already running**. Its verdict:

> The $q=16$ evolved-metric regression is the **empirical quadrature**, not the
> directions. Across four jobs every dense ladder is monotone on both metrics (12 of 12
> dense ladder/metric combinations, 0 violations) and only EQ ladders regress. Cause (1),
> trajectory-blind directions, is **refuted** on both of its pre-registered criteria: the
> incumbent directions' per-interval step map *improves* with $q$
> (0.95 / 0.87 / 0.73 / 0.65).

The coordinator therefore redirected this lane. Both parts of the redirect are recorded
here; **nothing above this line is edited**, and everything already built and committed
is kept.

### A. The trajectory-directions job becomes a control

`qtd01` (job 3756800) was already running when the redirect arrived, so it is **let
finish and reported as a control**, not as the subject. Its pre-registered pass of
section 6 is still evaluated and reported exactly as written, because a pre-registration
that is abandoned once its hypothesis looks dead is worth nothing. It is now a *test of
`q-diag`'s verdict*: if `q-diag` is right that the directions are innocent, then

* the three direction sets should give ladders that agree closely at matched $q$, $M$ and
  quadrature; and
* the $q=16$ evolved regression should appear on the EQ ladders of **all three** direction
  sets and on the dense ladders of **none**.

**No second job is spent on new directions.**

### B. The new primary question — certify the DENSE ladder

> Does the **dense** correction ladder — incumbent directions, per-step budget 600 —
> satisfy the knob criterion on **both** metrics, inside one allocation?

This is the tunability certification on the arm that does not depend on the quadrature at
all; the `q-ridge` lane is separately redirected to fix the EQ rules themselves.

**Part 1, no GPU.** From the existing audited archives (`cclad01`, `btq101`, `btq102`,
`btq201`), **dense arms only**, compute the converged non-dominated set on both metrics,
monotonicity, cost span and error span. `experiments/q-trajdirs/dense_from_archives.py`.

**Part 2, one job (`qtd02`).** Dense ladder, $q\in\{0,16,32,64,128,256\}$, at **fixed
$M=256$** and at **$M=4(K+q)$**, per-step budget 600, the same six opened development
cases, same-job `fft_tight` / `fft_loose` / `nt1e-2_dt01` controls, three timed
repetitions with burn-in, both metrics per output time per case, exit reasons. It fits
**no empirical quadrature anywhere**, so no number in it can depend on a quadrature rule.
$M=256$ is illegal above $q=239$, so the fixed-$M$ ladder stops at $q=128$; the
$M=4(K+q)$ ladder carries $q=256$ and is the pass ladder.

**Redirected pass (`config-dense.json`, `pass_criteria`).** On the `dense_m4` ladder:

1. worst **evolved**-times error non-increasing in $q$ from 0 to 256, **with every rung
   converged**; and
2. the converged non-dominated set on the (median GPU ms, worst evolved error) plane
   spans $\ge 2\times$ in **error** **and** $\ge 2\times$ in **cost**.

The all-times metric is reported beside it on every row. Criterion 2 is strictly harder
than section 6's, which required only the error span.

**Gates unchanged**, plus the coordinator's: $q=0$ and $q=128$ dense reproduce the
`b-ladder-top` rows to $\le 10^{-9}$; every error recomputed from the saved fields; the
evaluation cohort bitwise `abl01`'s six cases. The two jobs run **concurrently in their
own attempt directories** — `qtd01` and `qtd02` — which is two of the three-job cap.

### C. What this lane can no longer claim

The trajectory-direction hypothesis was **refuted by another lane before this lane's own
job produced a number**. Whatever `qtd01` reports, this cell does not get to claim it
discovered that the directions are innocent; `q-diag` established that from data that
already existed, and this report cites it. What `qtd01` adds is an independent,
same-allocation check of that verdict under a direction rule `q-diag` never had.

### D. Observed during the runs, before any verdict

* **D1 — the trajectory direction rule is two orders of magnitude cheaper to fit than the
  incumbent one.** In `qtd01`, on one A100, the incumbent static rule took **1546.9 s**
  (almost all of it one XLA slow-compile alarm on its doubly vectorised multistart
  Levenberg-Marquardt fit over 1024 snapshots), while `traj` took **27.0 s** over 32
  training trajectories and `prac` **8.9 s** over 6. The trajectory rule needs no
  multistart optimisation at all: the ROM's own internal latents already are the codes.
  That is a real, incidental advantage of the rule the redirect just made irrelevant, and
  it is recorded because it would otherwise be lost.
* **D2 — the `sbatch` config placeholder bug.** Parameterising the staging script by
  config file introduced a bare `CONFIG` token that also matched inside `MPLCONFIGDIR`,
  so the generated script exported a corrupted matplotlib cache path. Caught by reading
  the staged script before submission; **no job ran with it**. The placeholder is now
  `__CONFIG__`.
* **D3 — the session scratchpad is shared.** A synthetic fixture written under the
  scratchpad was overwritten by a sibling agent between two uses of it. Nothing that
  matters lives there — every input to every reported number is a committed file or a
  checksum-verified archive — but fixtures are now written to a lane-specific subdirectory.

### E. Outcome, and what was retracted

* **E1 — the redirected primary PASSES.** `qtd02` (job 3757505, `NVIDIA A100-PCIE-40GB`,
  3152.1 s): on the `dense_m4` ladder the worst evolved-times error falls monotonically
  $1.8890 \to 1.3985 \to 1.2336 \to 1.0843 \to 0.8930 \to 0.5194\,\%$ over
  $q=0,16,32,64,128,256$ with **every rung converged** — zero budget exits anywhere, worst
  joint gradient $9.99\times10^{-7}$ — and the worst all-times error falls monotonically
  $2.5629 \to 0.9053\,\%$. Six converged non-dominated points spanning **3.637×** in error
  and **13.070×** in cost. Both legs of the redirected criterion clear 2×. The fixed-$M=256$
  dense ladder is also monotone on both metrics and fully converged but spans only 1.22× in
  evolved error, so $M$ growing with $q$ carries part of the cost span and part of the error
  span; both ladders are reported.
* **E2 — the control job DIED and was not resubmitted.** `qtd01` (job 3756800) built all 31
  reduced arms and all 21 empirical-quadrature rules, ran every offline diagnostic, and then
  hit `CUDA_ERROR_OUT_OF_MEMORY` in the compile warm-up of its 21st subject. It was sized
  for an 80 GB A100 and landed on the 40 GB part; 31 simultaneous jitted queries holding
  their own dense test-mode matrices and quadrature stencils do not fit in 40 GB. Its
  offline output is collected, checksum-verified, audited in a `--partial` mode and reported;
  **its timed ladders do not exist.** The redirect said not to spend another job on new
  directions, so it was not rerun, and the third job of the cap is unused.
* **E3 — the trajectory-direction hypothesis is withdrawn, and this cell does not get to
  claim it tested it.** `q-diag` refuted it from data that already existed, before this
  lane's own job produced a number. What survives from `qtd01` is a measurement of how
  differently the two rules aim — the incumbent directions reach only **11.7 %** of the
  trajectory residual's energy at $q=16$ and **39.0 %** at $q=64$, with a field-metric
  subspace overlap of 0.176 and 0.441 against the trajectory directions — together with the
  fact that aiming differently was never what the ladder needed.
* **E4 — sizing rule for this family of jobs, worth carrying.** A job that builds $N$
  jitted reduced queries holds all $N$ compiled executables plus all $N$ operator sets on
  the device at once. `qtd02`, with 14 subjects and no quadrature stencils, fits 40 GB
  comfortably; `qtd01`, with 34, does not. Either request the 80 GB part explicitly, or
  split the sweep across attempt directories.
