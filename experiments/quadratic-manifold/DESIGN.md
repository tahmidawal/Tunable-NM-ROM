# quadratic-manifold — "why not something simpler than a neural head?", priced in ONE allocation

Written and committed before any cluster job of this lane. Amendments are appended as §A1, §A2,
… and nothing above them is edited.

## 0. Why this lane exists

`GeelenWrightWillcox2022` and `BarnettFarhat2022` are already in the manuscript's bibliography.
Both propose the obvious simpler thing: keep the linear reduced basis and add a **quadratic form
in the same reduced coordinates**, fitted offline by least squares. No network, no training run.
A reviewer will ask why our nonlinear head is needed if that suffices, and their absence from the
results table is conspicuous precisely because we cite them.

This lane answers the question with a measurement, in the same allocation, on the same cohort,
against the same reference, through the same solver, as every other row of the paper's $256^2$
Burgers panel.

## 1. The question

At $256^2$ on the six opened development cases, with the trial map the **only** thing that
changes between subjects:

1. **Accuracy.** At matched solved dimension, does the quadratic manifold reach the accuracy of
   our nonlinear head, or does it sit between classical POD-LSPG and us?
2. **Cost.** What does it cost online, given that the quadratic term's Jacobian assembly is
   $O(r^2)$ in the number of reduced unknowns and inflates the trial basis from $r$ columns to
   $1 + r + r(r+1)/2$?
3. **Unknowns.** Does it need more or fewer unknowns than our $k+q$ for the same error?

This lane is a **measurement**, not a hypothesis test. The pre-registered content is the subject
list, the ladder of $r$, the fitting and regularisation protocol, the gates that make a number
admissible, the metric definitions and the pass/fail and stop rules below. **If the quadratic
manifold beats us, this lane reports that**; §7 says so explicitly.

## 2. The method, as implemented

The quadratic manifold writes the state as

$$u(a) \;=\; u_{\rm ref} \;+\; V_r\,a \;+\; W\,\mathrm{vech}(a\,a^\top),\qquad a\in\mathbb R^{r},$$

with $u_{\rm ref}\in\mathbb R^{n}$ a fixed shift, $V_r\in\mathbb R^{n\times r}$ a POD basis and
$W\in\mathbb R^{n\times P}$, $P=r(r+1)/2$, the quadratic coefficient matrix. $\mathrm{vech}$ is
the vector of distinct monomials $a_i a_j$, $i\le j$, so each monomial appears once and $W$ absorbs
the factor-of-two on the off-diagonals.

**Offline, in-job, from the snapshots the panel already generates.** The snapshot matrix
$S\in\mathbb R^{n\times N_s}$ is `ladder.generate_snapshots`' own output — 128 training
trajectories, every second state, $N_s=3328$ — bitwise the matrix the panel's classical POD-LSPG
arms are built from, and disjoint from the evaluation cases by the driver's existing assertion.
Then:

* $u_{\rm ref}$ = the snapshot mean (GWW's shift);
* $V_r$ = the leading $r$ POD modes of the **centred** snapshots $S_c=S-u_{\rm ref}\mathbf 1^\top$,
  by the same `ablation.pod_basis` method-of-snapshots routine the POD arms use;
* $A = V_r^\top S_c$ the reduced coordinates, $\Pi$ the $P\times N_s$ matrix of monomials
  $\mathrm{vech}(a\,a^\top)$ of its columns, and $E = S_c - V_r A$ the residual the linear term
  leaves (Barnett–Farhat's POD complement);
* $W$ solves the ridge problem $\min_W \lVert W\Pi - E\rVert_F^2 + \gamma\,s\,\lVert W\rVert_F^2$,
  i.e. $W = E\Pi^\top(\Pi\Pi^\top+\gamma s I)^{-1}$ with $s=\mathrm{tr}(\Pi\Pi^\top)/P$ the scale
  that makes $\gamma$ dimensionless. **One linear solve. There is no training run in this lane.**

**Regularisation, and how $\gamma$ is chosen.** $\gamma$ is selected from the fixed pre-registered
grid $\{0,10^{-10},10^{-8},10^{-6},10^{-4},10^{-2},1\}$ by **held-out relative error on a seeded
20 % split of the snapshots** (seed 20260922): $W$ is fitted on the 80 % and scored by
$\lVert W\Pi_{\rm te}-E_{\rm te}\rVert_F/\lVert S_{c,\rm te}\rVert_F$, and the winner is then
**refitted on all snapshots** at that $\gamma$. The whole grid's held-out errors, the chosen
$\gamma$, the Gram condition number and the in-sample residual are written into `result.json`
per rung, so the selection is auditable from the JSON alone. **The selection touches only training
snapshots. It never sees the evaluation cohort, and this lane opens no sealed cohort.**

**Online, nothing new is written.** The columns $B=[\,u_{\rm ref}\mid V_r\mid W\,]$ become an
`arms.GridBank` and the coefficient map is
$\eta(a) = [\,1,\;a,\;\mathrm{vech}(a\,a^\top)\,]$, so $B\,\eta(a)$ **is** the trial map above.
That pair is handed to `arms.make_query` unchanged, which means the discrete Burgers residual,
the weak test projection onto the $M$ lowest sine modes, the fixed-Gauss initializer with the
nearest-code start, the damped Levenberg–Marquardt driver, the budgets, the tolerance, the
two-way extrapolation guard and the "dense field on the GPU → six dense fields on the GPU" output
contract are **literally the same code paths** POD-LSPG and the NM-ROM arms run. The Jacobian
$V_r + 2W(a\otimes\,\cdot\,)$ comes out of `jax.jacfwd` of $\eta$; `qman.demo()` checks it against
the analytic expression to $10^{-13}$, so no solver, residual or quadrature code is duplicated.

## 3. The contract every reduced subject shares, and the arms

With $\Phi\in\mathbb R^{n\times M}$ the $M$ lowest discrete sine test modes, $\lambda$ their
Laplacian eigenvalues, $A_\Phi=\Phi^\top B$, backward Euler at $\Delta t=0.005$ and the full-order
model's own sign-upwind advection $\mathcal N$, every reduced subject writes $u=B\,\eta(w)$ and
minimises

$$r(w)=\frac{A_\Phi\eta-p+\Delta t\big(\Phi^\top\mathcal N(B\eta)+\nu\,\lambda\odot A_\Phi\eta\big)}
{1+\Delta t\,\nu\lambda},\qquad p=A_\Phi\,\eta^{\rm prev},$$

with per-step budget **600**, initial-fit budget 400, evolution tolerance `gtol` $10^{-6}$. The
subjects differ only in $(B,\eta,M)$ and the advection quadrature:

| family | $B$ | $\eta(w)$ | unknowns | $M$ | quadrature |
|---|---|---|---|---|---|
| `qman` quad | $[\,u_{\rm ref}\mid V_r\mid W\,]$ | $[1,\,a,\,\mathrm{vech}(aa^\top)]$ | $r$ | $4r$ | exact (dense) |
| `qman` lin | $[\,u_{\rm ref}\mid V_r\,]$ | $[1,\,a]$ | $r$ | $4r$ | exact (dense) |
| `pod` | $V_{k'}$ (classical **uncentred** POD of the same snapshots) | identity | $k'$ | $4k'$ | exact |
| `rom` dense | frozen bank $G$ | $h_\theta(z)+C_q y$ | $K+q$ | $4(K+q)$ | exact |
| `rom` certified EQ | $G$ | $h_\theta(z)+C_q y$ | $K+q$ | $4(K+q)$ | qrg304's certified rule |
| `fast` | $G$ | $h_\theta(z)$ | 16 | 64 | the certified $q=0$ rule, through b-speed's `L4` kernel |

### 3.1 Subjects, $256^2$ (job 1, `qmn101`)

Pre-registered, in build-priority order after the full-order controls:

* **NM-ROM endpoints**: `q0_M64_dense`, `q256_M1088_dense`, `q0_M64_eqcert`, `q256_M1088_eqcert`,
  all at `gtol` $10^{-6}$, and the **fast** arm `q0_M64_eqcert_fastL4` — the paper's fast and
  accurate settings — 5 arms;
* **quadratic manifold**, $r\in\{8,16,32,64\}$, `quad` then `lin` — 8 arms;
* **POD-LSPG**, $k'\in\{8,16,32,64\}$ (matched to the $r$ ladder) and $\{256,512\}$ (the paper's
  own POD rows) — 6 arms;
* **full-order controls**, all FFT-preconditioned Newton–BiCGStab on the same grid: the Newton
  grid $n_{\rm tol}\in\{10^{-2},10^{-3},10^{-4}\}\times\Delta t\in\{0.01,0.005\}$, plus
  `fft_tight` ($10^{-6}/10^{-8}$) — the converged same-grid solve every same-grid error is
  measured against — and `dense_tight`, its dense-preconditioner twin, gated to reproduce
  `fft_tight` to $10^{-9}$ — 8 settings, 3 compiled programs.

19 compiled reduced queries plus 3 full-order programs, against the 39 + 3 that fitted on an
80 GB A100 in `bpn301`. **5 timed repetitions** (b-panel's 3, raised to the 5 the sibling
`ops-timing-panel` used), 6 cases, randomised subject order per (repetition, case), 0.25 s GPU
burn-in before every invocation. No neural-operator phase: those arms are `ops-timing-panel`'s.

**Why the ladder stops at $r=64$.** $P=r(r+1)/2$: the trial basis is $1+r+P$ columns, which is
45, 153, 561 and **2145** at $r=8,16,32,64$. At $r=64$ that is already two thirds of $N_s=3328$
snapshots, so $\Pi\Pi^\top$ is near-rank-deficient and the ridge must do real work; $r=128$ would
need $P=8256>N_s$ columns and could not be fitted from this snapshot set at all. **The prediction
recorded before the job: the quadratic gain is largest at small $r$ and shrinks or reverses by
$r=64$** — that is a property of the method at this snapshot budget, not a defect of the run, and
it is reported as such either way.

### 3.2 One declared deviation: the initializer's quadrature

The shared initializer fits the supplied field on the subject's own manifold at a fixed
$48\times48=2304$-point Gauss rule. That rule is over-determined in the bank's columns for every
incumbent family ($D\le512$) but **not** for `qman` at $r=64$ ($D=2145$). The `qman` family
therefore raises the rule to $96\times96=9216$ points (`qman_cold_axis_points`) **only on a rung
that needs it**; the driver keeps the incumbent 48-point rule wherever $\text{points}>2D$ still
holds, which is every rung through $r=32$ ($2\times561<2304$), and asserts the condition either way.
Only the $r=64$ pair is raised, so `qman8/16/32` stay cost-comparable with `pod8/16/32` at the same
rank. The raised rule is **not** free at run time — `ui = e.sample_field(u0, xy, L)*w` and
$y=Q^\top u_i$ run inside the timed query — but it is order microseconds and the kernel count is
unchanged; the arm's `cold_axis_points` is reported in its row so the reader can see which rungs
carry it. Both `quad` and `lin` arms at a rank share the rule, so the with/without-$W$ comparison is
unaffected; the `pod` arms keep the incumbent rule and a `qman` `lin` row is **not** bitwise a `pod`
row and is not claimed to be (§3.3).

### 3.3 What the `lin` arm is, and what it is not

`qman` `lin` is the **isolation control**: the identical $u_{\rm ref}$ and $V_r$ with the $W$
block deleted, so `quad` − `lin` at fixed $r$ is the quadratic term's contribution and nothing
else. It is *not* the paper's POD-LSPG row: it carries the mean shift, its POD is of the centred
snapshots, and its initializer rule differs (§3.2). The paper's POD-LSPG rows are the `pod`
family, in this same job at the same ranks, and both appear in the table.

## 4. Metrics, all recomputed by the independent NumPy audit from saved fields

For every subject and case, with $u_{\rm ref}$ the converged same-grid `fft_tight` solve **of this
job** and $u_{4096}$ the 4096-interval, $\Delta t=3.125\times10^{-4}$ reference restricted to the
grid, normalised by $\lVert u(0)\rVert$:

* **worst / median evolved %**: $\max_t\lVert u(t)-u_{\rm ref}(t)\rVert/\lVert u(0)\rVert$ over
  $t\in\{0.05,\dots,0.25\}$, worst and median over the six cases — the primary error metric;
* **worst / median all-times %**: the same including $t=0$;
* **$t=0$ compression %**: the $t=0$ term alone — the manifold's reproduction of the supplied
  field through its own initial fit;
* **worst vs reference %**: against $u_{4096}$, which contains the mesh's own discretisation error;
* **median GPU ms**: device-synchronised, supplied field resident on the GPU → six output fields
  resident on the GPU, median over 6 cases × 5 repetitions; every repetition retained;
* **median complete-query ms**: the same including the host→device upload and device→host copy;
* **converged** (§5); median and maximum LM iterations per step; for `qman`, the chosen $\gamma$,
  the held-out error, the Gram condition number, the bank column count and $P$.

**Ratios are formed only inside this one job.** The paper's FOM rule — *the fastest tested
full-order setting whose error is at or below the row's error* — is applied within the job to
every row, so every speedup is a single-job ratio. **Cross-job timing is forbidden** and no
number from another job enters a table except through the §6 fidelity gates.

## 5. Convergence — the pre-registered definition

Inherited verbatim from b-panel §5 / §A13 and applied identically to every reduced subject,
`qman` included. A query is **converged** iff (i) every time step and the initial fit exit for a
reason in {residual, tiny step, gradient} — no budget exit, no rejected step; (ii) every time step
has normalised joint gradient $g\le10^{-6}$ **or** exited on the residual rule; (iii) the initial
fit has $g\le10^{-6}$ **or** relative residual $\le10^{-10}$. The threshold is the fixed $10^{-6}$
of DESIGN §5 as written (`converged_design5`), which defines `admissible`; the own-`gtol` variant
and the stricter every-gradient variant are reported beside it. Every row reports its budget
exits, exit-reason counts, worst step gradient, initial-fit gradient and initial-fit relative
residual.

A `qman` arm that does not converge is **reported with its numbers and excluded from the
admissible non-dominated set**, exactly as any other family would be. Non-convergence of the
quadratic manifold is a finding about the method at that $r$, not a licence to retune it.

**Carve-out (§A1).** Non-convergence that is attributable to *this harness* rather than to the
method is not reported as a property of the method. Three such causes are named in advance, each
with the evidence that would identify it and the response, and no other cause may be added after
the data is seen:

| cause | how it is identified from the recorded columns | response |
|---|---|---|
| an over-regularised or under-regularised $W$ | the selected $\gamma$ sits at an endpoint of the declared grid, and the pre-declared ridge-sensitivity arm at the same rank (§3.1) behaves differently | both rows are printed; the finding is about the *selection rule*, and the method's row is the better-behaved of the two, labelled |
| the unpivoted Gauss-Jordan solve at $\le64$ unknowns | `qman64_quad` shows rejected or budget exits that `pod64` at the same rank does not | reported as a solver artifact of the shared contract, not as a property of the quadratic manifold |
| the initial fit stalling on a nonlinear-in-$a$ head | the initial-fit gradient and relative residual columns fail while every time step passes | reported as an initializer artifact; the representation floor (§4) says whether the manifold could have held the field at all |

In every case the number is still printed and the cause is stated beside it. The carve-out permits
an **attribution**, never a rerun to improve a number — §8's stop rules (a) and (d) stand.

## 6. Gates (all must pass for a number to enter the report)

In-job: `jax_backend=gpu`, `JAX_ENABLE_X64`, matmul precision `highest`; checkpoint SHA256
identical before and after; the directions file and every rule file SHA256 and prefix hash equal
to `inputs/PROVENANCE.json`; the six cases bitwise the abl01 cohort (SHA256 `108f12dc…`); the
training draw disjoint from the cases (driver assertion); reference residuals $<2\times10^{-11}$;
`dense_tight` reproduces `fft_tight` to $10^{-9}$; every reduced subject $M>$ unknowns; every
invocation finite and paired (cost with error); repetitions of one (subject, case) bitwise
identical in output; `fast` parity within $10^{-12}$ with identical integer iteration and
exit-reason vectors.

`qman`-specific, asserted in the driver or the smoke: the trial basis has exactly $1+r+r(r+1)/2$
columns for `quad` and $1+r$ for `lin`; the chosen $\gamma$ lies in the declared grid; the
initializer rule satisfies points $>2D$; and at every rank the `quad` and `lin` output fields
**differ** (a silently-zero $W$ would otherwise reproduce the linear arm and look like "the
quadratic term does not help").

Cross-job fidelity, at $256^2$, on worst all-times, worst evolved and worst-vs-reference
percentages: `q0_M64_dense` and `q0_M64_eqcert` against qtd02 / qrg304 at $10^{-9}$;
`q256_M1088_dense` against qtd02 at $10^{-9}$ first tier and $10^{-3}$ second tier;
`q256_M1088_eqcert` against qrg304 at $10^{-3}$. These check that the panel is measuring the
archived model; they are the only place a number from another job appears.

Independent audit: `audit_panel.py` imports neither JAX nor the driver; it recomputes every error
from the saved fields, evaluates every gate, computes the non-dominated sets and writes the audit
JSON the report generator reads. A local smoke (`smoke_panel.py`, 64 intervals) runs the real
driver over every family including `qman`, reproduces the consolidated saved Burgers case to
$\le10^{-12}$, and runs the audit and report generator, before the first job is staged.

## 7. Pass / fail, and what each outcome means

This lane **succeeds** if the panel lands with its gates passing and the three questions of §1 get
numbers. It does not succeed by the quadratic manifold losing.

Pre-registered readings, stated before the data exists:

* **The quadratic manifold reaches our accuracy at matched solved dimension and costs no more.**
  Then the nonlinear head is not necessary on this problem and **the paper must say so.** The row
  goes in the main table, not an appendix, and the positioning claim is rewritten. This outcome is
  reported in full, with the numbers, in the lab log and the final report.
* **It sits between POD-LSPG and us** — better than the linear subspace at matched $r$, short of
  the head. Then it is the honest "simpler alternative" row the reviewer is asking for, and the
  gap to the head at matched unknowns is the quantitative answer.
* **It does not beat POD-LSPG at matched $r$.** Then the claim is only that the quadratic
  correction did not pay on *this* snapshot budget and *this* problem, reported with the ridge
  trace and the Gram conditioning that explain why, and explicitly **not** as a refutation of
  GWW/BF, whose regimes (smaller $r$, different PDEs, different snapshot budgets) this job does
  not cover.
* **Cost.** Whatever the accuracy, the online cost of the $O(r^2)$ Jacobian is reported as
  measured, not inferred, and the FOM-rule speedup of every `qman` row is given beside the
  NM-ROM's from the same job. **The fair headline comparison is dense against dense** (§A1): the
  `qman` arms run the exact advection sum, while the NM-ROM's headline rows run a certified
  empirical-quadrature rule and, for the fast setting, b-speed's optimised kernel. Geelen–Wright–
  Willcox and Barnett–Farhat both pair the quadratic manifold with hyper-reduction (DEIM / ECSW);
  this lane constructs none for it, because a certified rule is a lane's worth of work. A cost
  ratio quoted against the EQ or fast NM-ROM rows therefore measures, in part, the absence of
  hyper-reduction for the baseline, and **every such ratio carries that sentence**. The dense
  NM-ROM arms `q0_M64_dense` and `q256_M1088_dense` are in this job precisely so the like-for-like
  ratio exists.

**The panel is unusable and this lane reports failure if**: any $10^{-9}$ fidelity gate fails (the
panel would not be measuring the archived model), or `fft_tight` does not converge on every case
(there would be no same-grid reference), or fewer than the four `quad` arms, the four `lin` arms,
POD-8/16/32/64 and the full-order controls survive the OOM rule (the comparison would again be
assembled across jobs).

## 8. Stop rules and job accounting

Budget: **≤ 1 running job, ≤ 4 total** (account cap 6 running, shared with three sibling lanes;
`squeue -u tawal01` before **and** after every submit). Namespace
`/cluster/tufts/paralab/tawal01/qman_20260922/`, **one directory per job**, `gpu` partition only,
preflight must print `jax_backend=gpu`, f64, `JAX_DEFAULT_MATMUL_PRECISION=highest`, all output
under paralab.

| attempt | mesh | purpose | budget |
|---|---|---|---|
| `qmn101` | $256^2$ | the panel | job 1 |
| — | — | one resubmit for a GPU-type escalation or an infrastructure failure | job 2 |
| — | — | held for a declared amendment the data forces (e.g. an OOM'd rung rerun) | jobs 3–4 |

**Stop rules.** (a) If `qmn101` lands with its gates passing, the lane is **done** — no rung is
added, no $\gamma$ grid is widened and no second job is run to improve a number, because a second
job's timings could not be compared with the first's. (b) A job that dies on infrastructure
(pending > 3 h, node `cuInit` failure, disk-full, OOM at build) is resubmitted once, on another
GPU type if the failure was the GPU, with the **science unchanged**. (c) A code fix found after a
job means the job's numbers are retracted in the lab log before any new number is quoted. (d) The
job budget is never spent to re-pick a hyperparameter after seeing an error.

After the job: independent NumPy re-computation of every error from the saved fields,
checksum-verified pull, remote job directory deleted, code / configs / logs / small summaries
committed (nothing > 50 MB), `HANDOFF.md` current, dated lab-log entry appended including
anything wrong or retracted.

## 9. What this lane does not do

It does not train anything, does not refit $C$, does not construct new EQ rules, does not open the
sealed cohorts, does not run the neural-operator arms, does not touch the root `paper/`, `main`,
another worktree or another namespace, and does not push or merge. One checkpoint, one training
seed, six opened development cases, one mesh.

## A1 — 2026-09-22, before any job: the independent design audit, and the amendments it forced

The protocol's pre-job independent audit was run against this document and the lane's code before
the first submit. The full report is `checks/design-audit.md`; every disposition is below.

**Codex could not run.** `codex exec -m gpt-6-astra -s read-only` was launched first, as the
protocol asks, and its sandbox failed to start on this box — `bwrap: loopback: Failed RTM_NEWADDR:
Operation not permitted`. It returned an explicit audit-access blocker having read nothing, rather
than an opinion; that is the correct behaviour and it is recorded, not worked around. An
independent subagent auditor was commissioned with the identical eight-question brief.

### A1.1 The blocking finding, and the measurement that confirmed it

The ridge selection of §2 held out a random 20 % of snapshot **columns**. The snapshot matrix
concatenates trajectories at 26 states each, so consecutive columns are states $\Delta t=0.01$
apart in a smooth viscous flow — near-duplicates. A uniform column holdout leaves almost every
held-out column's own temporal neighbours in the training half, so the criterion cannot see
overfitting and rewards interpolation. **This systematically hands the baseline its weakest $W$**,
which is the worst possible defect in a lane whose purpose is to give a competing method a fair
row.

It was confirmed by measurement before anything was changed, not argued. The 64-interval smoke's
$r=8$ rung was refitted with the ridge **forced** to $10^{-4}$ instead of the $0.0$ the column
split selected, everything else identical (`checks/config-probe64.json`, evidence
`checks/probe64.json`):

| | column split, $\gamma=0$ | forced $\gamma=10^{-4}$ |
|---|---|---|
| converged (§5) | **no** | **yes** |
| max LM iterations | 520 | 12 |
| worst step stationarity | $8.7\times10^{-4}$ | $\le10^{-6}$ |
| worst evolved % | 83.27 | **64.92** |
| vs its own `lin` control (66.38 %) | 25 % **worse** | **better** |
| vs POD-8 (66.61 %) | worse | **better** |
| median GPU ms | 143.0 | **21.7** |

The rule, not the method, produced the failure. **§2 is amended: the ridge holdout is by
TRAJECTORY**, ~26 of the 128 held out at $256^2$, seeded as before, still touching only training
data and never the evaluation cohort. `qman.fit` takes `states_per_trajectory` from the snapshot
generator and asserts the column count divides by it.

**The opposite tail is now visible too, and is not tuned away.** With the trajectory split the
64-interval smoke — which has only 4 training trajectories, so 3 fit and 1 held out — selects the
grid's **top** value $\gamma=1$, which shrinks $W$ to $\lVert W\rVert_F=0.0026$ and collapses the
`quad` arm onto its `lin` control (66.382 % against 66.376 %). That is a small-sample artifact of a
4-trajectory smoke, not a prediction for 128 trajectories, and the response is **not** to retune
the criterion — that would be selecting the baseline's regularisation for a better-looking answer.
The response is to make the outcome legible: see A1.2's ridge-sensitivity arm, the recorded
$\gamma$ trace and $\lVert W\rVert_F$ per rung, and §5's carve-out.

**Reading a boundary selection, pre-registered here.** If the criterion selects an endpoint of the
declared grid at $256^2$, the row says so (`ridge_at_grid_endpoint` in the summary). At the top
endpoint it means the quadratic term does not generalise across held-out trajectories on this
problem — widening the grid upward would only shrink $W$ further, so the endpoint is
self-consistent and the finding stands. At the bottom endpoint ($\gamma=0$) it means the criterion
saw no overfitting penalty at all, and the ridge-sensitivity arm is the control that says whether
that cost anything.

### A1.2 Every finding and its disposition

| # | severity | finding | disposition |
|---|---|---|---|
| 5a | **BLOCKING** | leaky column-level ridge holdout | **Fixed**, after the confirming measurement above. `qman.fit` splits by trajectory; §2 amended; the smoke asserts `split == 'by trajectory'` and that trajectories × states = snapshots |
| 5b | should-fix (high) | a non-finite score in the $\gamma$ loop locks in $\gamma=0$ with a NaN $W$, killing the job mid-timing | **Fixed.** A non-finite score can never win; a grid of only non-finite scores raises; `assert isfinite(W)` after the refit; `demo()` checks that a non-finite fit is rejected |
| 5c | note | `scale` from the 80 % Gram applied to the full-Gram refit; `gram_condition` can come out negative; $\lVert W\rVert_F$ recorded nowhere | **Fixed.** The refit uses its own Gram's scale; the condition number is floored and the negative-eigenvalue case flagged; $\lVert W\rVert_F$ is recorded per rung and surfaced into the table |
| 6 | should-fix | no representation floor for `qman`, so a large error cannot be attributed to the manifold vs the solve | **Fixed.** An untimed, OOM-tolerant best-found fit per `qman` arm, started from the exact linear projection $a_0=V_r^\top(u-u_{\rm ref})$ plus its nearest training coordinates; wired through the audit's `best_found_percent` |
| 4 | should-fix | §3.2's claim that the raised initializer rule costs nothing at run time is false | **Fixed both ways.** §3.2 corrected, and the rule is now raised **per rank** — 48 through $r=32$, 96 only at $r=64$ — so only the rung that needs it deviates |
| 8.3 | note | the `qman` gates ran only in the smoke, never at $r=64$ | **Fixed.** Bank-column count and $\gamma$-in-grid are asserted in the driver at build; the quad-vs-lin field-difference gate is an in-job gate the independent audit re-checks |
| 2 | should-fix | $M=4r$ may under-resolve a 2145-column trial space; a loss would not be separable from test-space truncation | **Accepted.** One $M$-sensitivity arm added, `qman32_quad` at $M=512$ against the ladder's $M=128$ |
| — | new, from the smoke | the selection rule can sit at either grid endpoint | **Accepted.** One **pre-declared** ridge-sensitivity arm added, `qman32_quad` at a fixed $\gamma=10^{-4}$. The value is in the config before the job and is never chosen after seeing evaluation error |
| 8.1 | should-fix | §7's cost bullet omits that `qman` has no hyper-reduction while the NM-ROM headline rows do | **Fixed.** §7 now pre-registers dense-vs-dense as the fair headline ratio and requires the qualification on any ratio against the EQ or fast rows |
| 8.2 | should-fix | §5 has no carve-out for harness-attributable non-convergence | **Fixed.** §5 carries a carve-out naming the three admissible causes in advance, each with its identifying evidence; it permits attribution, never a rerun |
| 3 (note) | note | §A1 cited a `checks/design-audit.md` that did not exist | **Fixed.** The report is committed at that path; this section is written from it |
| 2 (GJ), 7 | note | the unpivoted Gauss-Jordan solve is a bigger risk for `qman64` than `pod64`; memory fits with ~10 GB steady state | **Noted, no change.** The GJ risk is the second row of §5's carve-out table; the memory finding confirms the plan and needs nothing |

Two arms were added, so the $256^2$ job now declares **29** subjects (21 reduced, 8 full-order
settings in 3 programs) against the 27 of §3.1 as first written; the $r$ ladder, the variants, the
tolerance, the metrics, the gates and the pass/fail readings are unchanged. Both new arms are named
in `config-256-qman.json`'s `priority_override` and are built after the eight ladder arms, so the
OOM rule still takes `pod256`/`pod512` first.

### A1.3 What the audit did not find

No evaluation-data selection anywhere, in the design or the code. The trial map is the papers'
method, and where it deviates it deviates in the baseline's **favour** — $W$ is unrestricted in
$\mathbb R^{n\times P}$, of which Barnett–Farhat's $W=\bar V\bar W$ is a special case. The timing
contract is comparable with the POD, NM-ROM and full-order rows. The wiring — the bank/head algebra,
the `jacfwd` Jacobian, the initial-fit path, `GridBank`'s zero padding under homogeneous Dirichlet
boundaries — is correct. The job fits on an 80 GB A100 with roughly 10 GB steady state against a
72 GB budget, and `priority_override` is right if it does not.
