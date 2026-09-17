# b-panel — the paper's central comparison, priced in ONE allocation per mesh: predeclared design

Written and committed before any cluster job of this lane. Amendments are appended as §A1, §A2,
… and nothing above them is edited.

## 0. Why this lane exists

The NeurIPS submission died on cost comparisons assembled across different jobs and against an
over-solved full-order solver. Today the dense correction ladder (`qtd02`, job 3757505), the
POD-LSPG ranks and the FNO (`btq201`, job 3747245, where POD-32/64/128 were flagged "not
converged"), the certified empirical-quadrature rules (`qrg304`, job 3768168) and the optimised
kernel (`b-speed`) live in different allocations on different GPUs. **Every cost number in the
paper's table T5 and headline figure must come from one allocation on one GPU**, against
full-order controls timed in that same allocation with the same reference. This lane runs that
panel at $256^2$ (job 1), at $1024^2$ with the frozen model transferred (job 2), and at $512^2$
if the job cap and time allow (job 3).

## 1. The question

For the frozen Burgers checkpoint `sep_hfit_dense_mid_N256_dense.pkl` (SHA256 `18f0266a…`,
$K=16$, $R=512$), on the six opened development cases, what is the **same-job non-dominated set**
over (cost, error) when the correction ladder (dense and certified-EQ, two evolution
tolerances), the classical POD-LSPG ladder, the unrestricted-bank endpoint, the trained FNO and
a grid of full-order Newton solvers are all timed in one allocation on one GPU?

This lane is a **measurement**, not a hypothesis test: the pre-registered content is the subject
list, the gates that make a number admissible, the metric definitions, and the non-dominance
definition. The falsification clause (§7) states what would make the panel unusable.

## 2. The frozen model and what is transferred

The model is the triple $(G, h_\theta, C)$: the frozen separable bank $G$, the frozen head
$h_\theta$, and the nested direction matrix $C\in\mathbb R^{512\times512}$. Everything in this
lane uses:

* the checkpoint above, byte-checked in-job before and after;
* **$C$ = qtd02's archived `directions_old.npz`** (SHA256 `79d79458…`, prefix hashes recorded in
  `inputs/PROVENANCE.json`). It is loaded, not refitted, in both meshes, so the ladder's
  correction subspace is bitwise the one the dense ladder that passed the knob bar used. The
  earlier lanes refit $C$ in every job and paid a $10^{-3}$-level cross-GPU drift at $q\ge64$;
  loading it removes that drift from this panel and makes job 2's model literally job 1's;
* **the certified EQ rules = qrg304's archived `rule_q{q}_reachable_m{m}.npz`** for the rule
  qrg304 chose per rung: $m=1024$, primary-certified ($\rho_{\max}\le0.116$) at $q\le64$;
  $m=2048$, secondary-certified ($\rho_{95}\le0.116$, $\rho_{\max}$ 0.19 / 0.17) at
  $q=128,256$. Each rule is the pair (nodes, weights); the stencil bank $G_5$ and the weighted
  test matrix $P_q$ are rebuilt from the frozen bank exactly as `eqcert.fit_rule` builds them.
  The $q=128$ and $q=256$ arms are **labelled secondary-certified** everywhere they appear.

## 3. The contract every reduced subject shares

With $\Phi\in\mathbb R^{n\times M}$ the $M$ lowest discrete sine test modes ($\Phi^\top\Phi=I$),
$\lambda$ their Laplacian eigenvalues, $A=\Phi^\top B$ for the subject's bank $B$, backward Euler
at $\Delta t=0.005$ and the full-order model's own sign-upwind advection $\mathcal N$, every
reduced subject writes $u=B\,\eta(w)$ and minimises

$$r(w)=\frac{A\eta-p+\Delta t\big(\Phi^\top\mathcal N(B\eta)+\nu\,\lambda\odot A\eta\big)}
{1+\Delta t\,\nu\lambda},\qquad p=A\,\eta^{\rm prev},$$

with the retained initializer (fixed Gauss state fit, nearest-code start), the retained
block-damped Levenberg–Marquardt solver, per-step budget **600**, initial-fit budget 400,
evolution tolerance `gtol` $10^{-6}$ (and $10^{-3}$ where declared), the retained two-way
extrapolation guard, and the output contract "supplied dense field on the GPU $\to$ six dense
output fields on the GPU". The subjects differ only in $(B,\eta,M)$ and in the quadrature of the
advection term:

| family | $B$ | $\eta(w)$ | unknowns | $M$ | quadrature |
|---|---|---|---|---|---|
| `rom` dense | $G$ | $h_\theta(z)+C_q y$ | $K+q$ | $4(K+q)$ | exact (dense) |
| `rom` certified EQ | $G$ | $h_\theta(z)+C_q y$ | $K+q$ | $4(K+q)$ | qrg304's certified $m$-point rule |
| `pod` | $V_{k'}$ (classical POD of the same 128-trajectory truth snapshots, rebuilt at each mesh) | identity | $k'$ | $4k'$ | exact |
| `free` | $G$ | identity on $\mathbb R^{512}$ | 512 | 1024 | exact |
| `fast` | $G$ | $h_\theta(z)$ | 16 | 64 | the certified $q=0$ rule, through b-speed's `L4` kernel |

`fast` is the b-speed lane's parity-passing optimised kernel (fused residual+Jacobian, hoisted
constants, shared probe, scan unroll 5, head output layer folded into the stencil bank), vendored
byte for byte from commit `b3f9ecf9`. It is admitted to the panel **only if** it passes the
in-job parity gate against the incumbent `q0_M64_eqcert_g1em06` arm: output fields within
$10^{-12}$ relative on every case and identical integer iteration-count and exit-reason vectors.
If it fails, it is reported and excluded from the non-dominated sets.

### 3.1 Subjects, $256^2$ (job 1)

* dense ladder: $q\in\{0,16,32,64,128,256\}$, `gtol` $10^{-6}$ — 6 arms;
* certified-EQ ladder: the same $q$, `gtol` $\in\{10^{-6},10^{-3}\}$ — 12 arms;
* fidelity arm `q0_M256_dense_g1em06` (fixed $M=256$, btq201's row) — 1 arm;
* POD-LSPG $k'\in\{16,32,64,128,256,512\}$, `gtol` $10^{-6}$ — 6 arms;
* free bank, $R=512$, $M=1024$ — 1 arm;
* `fast` — 1 arm;
* full-order controls, all FFT-preconditioned Newton–BiCGStab on the same grid: the Newton grid
  $n_{\rm tol}\in\{10^{-2},10^{-3},10^{-4}\}\times\Delta t\in\{0.01,0.005\}$ with linear tolerance
  $0.5$ at $n_{\rm tol}=10^{-2}$ (the retained loose setting) and $10^{-2}n_{\rm tol}$ otherwise —
  6 settings; `fft_tight` ($10^{-6}/10^{-8}$, $\Delta t=0.005$), the converged same-grid solve
  every same-grid error is measured against; `dense_tight`, the same solve with the dense
  sine-transform preconditioner (the "direct" control), which must reproduce `fft_tight`'s
  fields to $10^{-9}$ as an in-job gate — 8 settings, 3 compiled programs;
* the trained FNO `fno-large` (validation-selected, 17.9 M real parameters, checkpoint SHA256
  `208d9002…`), timed by b-ladder-top's `fno_panel.py` in a second process **in the same Slurm
  allocation on the same GPU** after the JAX process exits (the declared deviation btq201 made
  and reported), on the identical six cases against the identical references.

27 compiled reduced queries plus 3 full-order programs. That is above the 20 that btq201 held on
a 40 GB A100 and above the 21st that OOM'd in qtd01, so job 1 requests `--constraint=a100-80G`
(the same node class qrg304 and btq101 ran on) and sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`.
**If a subject still fails to build or to warm up with `RESOURCE_EXHAUSTED`, the driver records
it as `dropped_oom` and continues**; subjects are built in the priority order FOM controls, dense
ladder, certified EQ at $10^{-6}$, POD, certified EQ at $10^{-3}$, the fidelity arm, free bank,
`fast`, so what is lost first is what matters least. Any drop is reported, never silently
absorbed, and a dropped subject may be rerun in a later job with its own FOM controls, labelled as
such.

### 3.2 Subjects, $1024^2$ (job 2)

The same panel with $(G,h_\theta,C)$ transferred unchanged and the POD basis rebuilt at the mesh.
The memory arithmetic that decides the GPU: $G$ alone is $1{,}046{,}529\times512\times8$ B
$=4.3$ GB; the dense test matrices $\Phi$ at $M=64\ldots1088$ total 19.8 GB; POD-16/128 add
4.8 GB; the 128-trajectory snapshot matrix is 27.9 GB transient. A 40 GB A100 cannot hold this
(qtd01 OOM'd at $256^2$ with 31 arms), an 80 GB A100 is marginal, so job 2 goes to an **H200
(141 GB) with `--mem 240G`**, the fallback the coordinator named.

Priority order at $1024^2$: FOM (Newton $n_{\rm tol}\in\{10^{-2},10^{-4}\}\times\Delta t$, 4
settings, plus `fft_tight`); dense $q\in\{0,64,256\}$; certified-EQ $q\in\{0,64,256\}$ at
$10^{-6}$; POD-16, POD-128; then dense $q\in\{16,32,128\}$; certified-EQ $q\in\{16,32,128\}$ at
$10^{-6}$; certified-EQ at $10^{-3}$ for all $q$; POD-32/64/256; free bank; FNO if it runs. The
first group is the coordinator's reduced set and is what the job must deliver; everything after
it is best effort under the OOM-drop rule.

**Certified EQ at $1024^2$.** A rule is a set of grid nodes and weights and is therefore
mesh-bound; qrg304's rules live on the $256^2$ grid. Refitting and certifying rules from scratch at
$1024^2$ (reachable-state collection with the unrolled 12-iterate collector, GPU NNLS to
$m=2048$, held-out $\rho$) is what qrg304 spent 10 500 s on at $256^2$ and is out of scope. The
$1024^2$ EQ arms therefore use **transferred rules**: the certified rule's node set mapped to the
same physical points ($ij\mapsto4\,ij$), the weights **refitted** by nonnegative least squares on
that fixed support against reachable states at $1024^2$ collected exactly as qrg304 collected
them (`eqcert.make_collect_query`, dense quadrature, 12 iterates per step) on 8 training-family
trajectories disjoint from the evaluation cases, and **certified by held-out $\rho$** on 4 further
disjoint trajectories against the same bar $\rho^\star=0.116$. Each transferred rule is reported
with its $\rho_{\max}$, $\rho_{95}$ and basis (primary / secondary / uncertified); an uncertified
rule's arm is still timed and reported, labelled uncertified, and excluded from the converged
non-dominated set. This is a new construction and is named `eqxfer`, never `eqcert`.

### 3.3 Job 3, $512^2$ (if the cap allows)

The $256^2$ subject list minus the free bank, on an 80 GB A100, for the crossover curve. Rules
transferred as in §3.2.

## 4. Metrics, all recomputed by the independent NumPy audit from saved fields

For every subject and every case, with $u_{\rm ref}$ the converged same-grid `fft_tight` solve
of the same job and $u_{4096}$ the 4096-interval, $\Delta t=3.125\times10^{-4}$ reference
restricted to the grid, normalised by $\|u(0)\|$:

* **worst all-times %**: $\max_t\|u(t)-u_{\rm ref}(t)\|/\|u(0)\|$ over the six output times,
  worst over cases;
* **worst evolved %**: the same over $t\in\{0.05,\ldots,0.25\}$ only;
* **$t=0$ compression %**: the $t=0$ term alone (the decoder's reproduction of the supplied
  field; the FNO returns the supplied field exactly, so its value is 0);
* **worst vs reference %**: $\max_t\|u(t)-u_{4096}(t)\|/\|u(0)\|$, worst over cases, the
  column btq201 carried; it contains the mesh's own discretisation error, which the `fft_tight`
  row makes explicit;
* medians of the above over cases are reported beside every worst;
* **median GPU ms**: device-synchronised time from the supplied field resident on the GPU to
  the six output fields resident on the GPU, median over 6 cases × 3 timed repetitions with a
  0.25 s GPU burn-in before every invocation and a randomised subject order per (repetition,
  case); every repetition is retained;
* **median complete-query ms**: the same invocation including the host→device upload of the
  supplied field and the device→host copy of the six outputs;
* **converged**: §5; **iterations**: median LM iterations per time step, and the maximum.

Ratios are formed only inside one job. No number from another job enters a table except in the
fidelity gates of §6.

## 5. Convergence — the pre-registered definition, and why btq201's POD flags change

btq201 marked POD-32 (one case), POD-64 and POD-128 "not converged". Its own rows show why:
every one of their time steps exits on the gradient rule with $g\le10^{-6}$ and zero budget
exits; only the **initial fit** exits on the tiny-step rule with a normalised gradient of
$0.09$–$0.18$, at a relative residual of $10^{-18}$–$10^{-20}$. The POD initial fit
$R\,z=Q^\top u_{\rm in}$ is a square linear system: it is solved exactly, its residual is
round-off, and $\|J^\top r\|/(\|J\|\,\|r\|)$ is a $0/0$ ratio, the degeneracy `arms.py` already
documents for attainable fits and the same one the $q=R$ rung showed. No iteration budget can
change a $0/0$. Raising the budget is therefore not the remedy; a stationarity test that
recognises an attained fit is. Pre-registered rule, applied identically to every reduced
subject:

A query is **converged** iff (i) every time step and the initial fit exit for a reason in
{residual, tiny step, gradient} — no budget exit, no rejected step; (ii) every time step has
normalised joint gradient $g\le10^{-6}$ **or** exited on the residual rule
($\|r\|\le10^{-9}\cdot$scale); (iii) the initial fit has $g\le10^{-6}$ **or** relative residual
$\|r_{\rm ic}\|/\|u_{\rm in}\|\le10^{-10}$.

btq201's stricter rule (every gradient $\le10^{-6}$ regardless of residual) is also evaluated and
reported as `converged_strict`, so the reader sees exactly which subjects flip and why. Every
reduced row reports its budget exits, exit-reason counts, worst step gradient, initial-fit
gradient and initial-fit relative residual, so "document exactly why one cannot" is a column, not
a footnote. The per-step budget is 600 for every reduced subject including POD and the free bank
(btq201 used 180 for POD); if any POD rank shows a budget exit at 600 the report says so and the
rank is not converged.

## 6. Gates (all must pass for a number to enter the report)

In-job: `jax_backend=gpu`, `JAX_ENABLE_X64`, matmul precision `highest`; checkpoint SHA256
identical before and after; $C$ file SHA256 and every prefix hash equal to `PROVENANCE.json`;
every rule file SHA256 equal to `PROVENANCE.json`; six cases bitwise the abl01 cohort (SHA256
`108f12dc…`); training draw disjoint from the cases; reference residuals $<2\times10^{-11}$;
`dense_tight` reproduces `fft_tight` to $10^{-9}$; every reduced subject $M>$ unknowns; every
invocation finite and paired (cost with error); repetitions of one (subject, case) bitwise
identical in output; `fast` parity as in §3.

Cross-job fidelity, at $256^2$, on worst all-times, worst evolved and worst-vs-reference
percentages, relative difference:

| this job | reproduces | source | tolerance |
|---|---|---|---|
| `q0_M64_dense_g1em06` | `q0_M64_dense` | qtd02 | $10^{-9}$ |
| `q0_M256_dense_g1em06` | `q0_M256_dense_g1em06` | btq201 | $10^{-9}$ |
| `q0_M64_eqcert_g1em06` | `q0_m4_eqcert` | qrg304 | $10^{-9}$ |
| `q{16,32,64,128,256}_M*_dense_g1em06` | `old_q*_M*_dense` | qtd02 | $10^{-9}$ first tier, $10^{-3}$ second tier (recorded which) |
| `q64_M320_eqcert_g1em06`, `q256_M1088_eqcert_g1em06` | `q64_m4_eqcert`, `q256_m4_eqcert` | qrg304 | $10^{-3}$ (qrg304 refit its own $C$; this lane loads qtd02's) |

A first-tier miss on a $q>0$ dense rung is a finding about GPU-dependence of the ladder, not a
reason to withhold the row; a miss at the second tier withholds the row and is a retraction.

Independent audit: `audit_panel.py` imports neither JAX nor the driver; it recomputes every
error from the saved fields, evaluates every gate, computes the non-dominated sets, and writes
the audit JSON the report generator reads. A local smoke (`smoke_panel.py`, 64 intervals)
reproduces the consolidated saved Burgers case to $\le10^{-12}$ through the panel's own subject
builders, checks the rule-rebuild path bitwise against `build_operators`, and runs every family
once, before the first job is staged.

## 7. Non-dominance, and the falsification clause

A subject $s$ is **dominated** if another subject $s'$ in the same job has cost $\le$ and error
$\le$ with at least one strict. The **non-dominated set** is reported for each of the four
metric pairs (GPU ms | complete-query ms) × (worst all-times | worst evolved), twice: over all
subjects, and over admissible subjects only (reduced subjects converged under §5, FNO and FOM
always admissible, `fast` only if parity passed, uncertified `eqxfer` excluded). The primary pair
for the paper is (median GPU ms, worst evolved %) over admissible subjects; the others are
beside it. The figure plots every subject with the family colour-coded and the non-dominated set
outlined, generated by `reports/generate_panel.py` from the audit JSON with the plotted numbers
written to a JSON beside it.

**The panel is unusable and this lane reports failure if**: any of the $10^{-9}$ fidelity gates
fails (the panel would not be measuring the archived model), or `fft_tight` does not converge on
every case (there would be no same-grid reference), or fewer than the dense ladder, the
$10^{-6}$ certified-EQ ladder, POD-16/128 and the FOM controls survive the OOM rule in job 1
(the central comparison would again be assembled across jobs). A `fast` parity failure, an FNO
phase failure, or a dropped low-priority subject are reported findings, not failures of the lane.

## 8. Job accounting

Cap 8. Planned: job 1 ($256^2$, A100-80G), job 2 ($1024^2$, H200 240 GB), job 3 ($512^2$) if
time allows; one attempt directory per job under `/cluster/tufts/paralab/tawal01/b_panel_20260917/`;
`squeue` before and after every submit; A100 → H100 → H200 → L40S escalation after 3 h pending
with the science unchanged; exact remote directory deleted after checksum collection and audit.

## 9. What this lane does not do

It does not train, does not refit $C$, does not construct new rules at $256^2$, does not open the
sealed cohorts, does not batch the full-order solver, and does not compare against any number
from any other job except through the §6 gates. One checkpoint, one training seed, six opened
development cases.

## A1 — 2026-09-17 00:55, before any job: the Codex audit could not run

The protocol's pre-job Codex audit was launched (`codex exec -s read-only`, prompt in the
scratchpad, output to `reports/codex-design-audit.md`). It read the design and the parent code
for ~108 k tokens and then died on `You've hit your usage limit … try again at Sep 19th, 2026
11:33 AM`; a probe with another model returned `not supported … ChatGPT account`. The Codex
quota is shared by all nine lanes and is exhausted until after this campaign's useful window.
The lane proceeds without an independent pre-job audit and records that fact here; a self-audit
against the eight questions in the prompt is in `reports/self-audit-design.md`, and Codex is
retried on the final report if the quota returns before the deadline. No number is affected.

## A2 — 2026-09-17 00:55, before any job: smoke deviations declared

The local smoke runs the real driver twice (64 intervals with rules on the 64-grid, then 128
intervals with those rules TRANSFERRED from 64) on two cases at one repetition, then the audit
and the report generator on both outputs. Its baseline gate reproduces the consolidated saved
Burgers case (`consolidated/fixtures/burgers/expected.npz`, the panel's own case 0) to
$\le10^{-12}$ through the panel's rule-rebuild path from the archived nodes and weights. Two
driver runs with ~14 and ~8 compiled subjects exceed the sub-minute rule; the excess is a
recorded deviation, as in the parent lanes.

## A3 — 2026-09-17 06:50, before job 2, after the smoke: the reachable population for transferred rules

The smoke's transfer run certified both rules with sixteen *identical* held-out $\rho$ values
and a refit residual of $10^{-15}$: qrg304's fixed-iterate collector (`eqcert.make_collect_query`,
12 unrolled LM iterates per step, retained for fidelity to that lane) returned one frozen state.
A GPU diagnostic at 64 intervals showed why: from the same initial fit the production solver's
first time step needs 29 iterations with several damping rejections, so a 4- or 12-iterate unroll
accepts nothing, and the carried state never moves. It worked in qrg304 only because at $256^2$
the first steps converged inside the unroll; at $1024^2$ that cannot be assumed. **The transferred
rules' fit and certification populations are therefore the production dense query's own converged
per-step states** (51 per trajectory, the same executable the timed dense arm uses), on the same
8 fit / 4 certification trajectories, disjoint from the cases. This is a smaller population than
qrg304's (converged states only, no intermediate iterates) and is labelled so; the $\rho$ bar,
the tiers and the admissibility rule are unchanged. The driver asserts every rollout it collects
actually moves. Job 1 ($256^2$) does not use the transfer path and is unaffected.

## A4 — 2026-09-17 09:40, AFTER job 1 landed: a third, post-hoc non-dominated set

Job 1's pre-registered sets (all subjects; admissible subjects) both contain only full-order
controls and the FNO, so they answer "is anything reduced worth running here" (no) but not "does
the nonlinear manifold beat the classical linear one", which is the question the paper's head
ablation asks and which this panel can now answer at ranks the ablation never reached
($k'=256,512$). A third set, **`reduced_only`** — the frontier among admissible `rom`, `fast`,
`pod` and `free` subjects — is therefore reported beside the other two. It is computed by the
same function with the same domination rule; it changes no pre-registered criterion, no
admissibility flag and no reported number. It was added after seeing the data and is labelled
post-hoc everywhere it appears, here and in the report.

## A5 — 2026-09-17 10:20: the remaining job budget, the reserved EQ re-run, and a risk recorded before it can be rationalised

Two jobs spent (`bpn101` 256² complete; `bpn201` 1024² in flight), six of the cap of eight free.
Planned, in priority order:

| attempt | mesh | purpose | status |
|---|---|---|---|
| `bpn101` | 256² | the panel | complete, job 3780638 |
| `bpn201` | 1024² | the panel, model and rules transferred | submitted, job 3783817 |
| `bpn301` | 256² | **reserved**: the panel re-run on b-eqtop's improved rules | on the coordinator's signal, not before |
| `bpn401` | 512² | the middle mesh, so the trend is three points not two | after `bpn301` |

Four spare for GPU-type escalation or a resubmit.

### A5.1 The reserved re-run must be a FULL panel, not a cheap-arm patch

The request was to re-run "the 256² cheap arm" with better rules. Taken literally that would
reintroduce the exact defect this lane exists to remove: new empirical-quadrature costs measured in
`bpn301` would have to be compared against `bpn101`'s full-order controls, i.e. **across two
allocations on two GPUs**, which is the NeurIPS failure mode. `bpn101` cost 28m50s, so a complete
re-run is cheap. `bpn301` therefore re-runs the **whole** 256² panel — every FOM control, POD rank,
the dense ladder and the FNO — in one allocation, and its table replaces `bpn101`'s wholesale rather
than being spliced into it.

It additionally carries **both rule sets as arms in that one job**: qrg304's rules (the superseded
ones, `eqcert`) and b-eqtop's (`eqtop`), at matched $q$, $M$ and tolerance. The improvement is then
measured in-allocation against its own predecessor instead of across jobs, and if the ladder becomes
monotone the evidence for *why* is in the same table. `bpn101` remains the record of what the
superseded rules gave; nothing about it is withdrawn.

### A5.2 A risk to `bpn201`, recorded now because it will be tempting to explain away later

b-eqtop's finding is that the binding constraint on an EQ rule is the number of **fit states**, not
the node count $m$: at $q=256$ a 64-state rule certifies on the primary bar at $m=2048$ where the
8-state rule gives $\rho_{\max}=0.1678$. That 0.1678 is precisely the qrg304 rule this lane
transfers at $q=256$.

`bpn201`'s transferred-rule refit chooses its fit-state count by the incumbent convention
$\mathrm{clip}(8192/M,\,8,\,64)$, which yields **64, 64, 42, 25, 14 and 8** states at
$q=0,16,32,64,128,256$. Its top two rungs are therefore fitted on 14 and 8 states and are at
elevated risk of returning secondary-certified or uncertified at $1024^2$ — for the reason b-eqtop
identified, not for anything about the mesh.

**This is predicted here, before the job returns.** If those rungs come back uncertified, that is a
confirmation of b-eqtop's diagnosis and not a surprise to be reinterpreted afterwards; if they
certify, the prediction was wrong and will be recorded as such.

**Why the job is not being cancelled and resubmitted with a larger fit-state count**, although it is
still PENDING with zero GPU time and could be. Raising the count makes the fixed-support refit a
nonnegative least-squares problem of up to $69632\times2048$, a size the local smoke never
exercised; shipping an unsmoked change into the most expensive job of the lane is the worse risk.
`bpn201`'s dense ladder, POD ranks, full-order controls and FNO — which carry the crossover evidence
the 1024² job exists for — do not depend on any EQ rule at all. The fit-state count is therefore
fixed in `bpn301`, where it can be smoked first, and `bpn201` reports its rules' certification
status honestly under the existing admissibility rule.

## A6 — 2026-09-17 11:15, after `bpn201` failed: a config-parsing crash, retracted and resubmitted as `bpn202`

`bpn201` (job 3783817, H200 on `pax008`, 240 GB, staged from `e330cca4`) exited 1 after 20m23s.
Disk was checked first: paralab at 92 % with 396 GB free at job start and now, both log files
non-empty, so not disk-full. Not memory either: the batch step's peak host RSS was 76 GB of 240,
and the crash is a Python exception, verbatim from `logs/3783817.err`:

```
  File ".../experiments/b-panel/panel.py", line 445, in main
    order = {s['name']: i for i, s in enumerate(cfg.get('priority_override', []))}
             ~^^^^^^^^
TypeError: string indices must be integers, not 'str'
```

`config-1024.json` (and `config-512.json`) hold `priority_override` — §3.2's build order — as a
list of subject-name strings; the driver indexed each entry as a dict. Neither smoke config
carried the key, so the local smoke never exercised the path, and `config-256.json` has no
override, so `bpn101` was unaffected. The job had by then completed the preflight
(`jax_backend=gpu`), the six references (~740 s), the 1024² snapshots (3328 states, 78 s) and all
six rule transfers (~740 s), and reached the subject declaration, the first line after them.

**What changed (commit after this amendment), and why it does not alter the science.**
`priority_override` is read as a list of names; every name must be a declared subject, asserted;
the subject declaration is factored into `declare_subjects()` and runs at the top of `main()`,
before any expensive work, so a malformed config now fails in the first second. The declared
list, its names, priorities and sort key are otherwise byte-identical to the block that was
inline. `config-smoke128.json` gains a `priority_override` so the smoke exercises the path, and
the smoke asserts (i) the declared build order honours it and (ii) `config-256/512/1024.json`'s
override names are declared subjects. No arm, tolerance, rule, seed, population or metric changed.

**The resubmission is `bpn202`**: the same `config-1024.json`, H200, `--mem 240G`, staged from
the commit carrying this amendment. Per protocol rule 4 the science is unchanged between the
attempt and its resubmission; in particular the transferred-rule fit-state count stays at the
§A5.2 convention and is **not** raised here, for §A5.2's reason (unsmoked size in the lane's most
expensive job). `bpn201` counts against the cap (retracted jobs count): three jobs used.

**What the failed attempt showed before it died, recorded because it bears on §A5.2's
prediction.** The six transfers ran and certified: $q=0,16,32$ primary ($\rho_{\max}$ 0.018,
0.059, 0.031 on 64/64/42 fit states); $q=64$ **uncertified**, $\rho_{\max}=0.1316$ on 25 states;
$q=128$ uncertified, $\rho_{\max}=1.03$ on 14; $q=256$ uncertified, $\rho_{\max}=0.41$ on 8. The
prediction named the top two rungs; the third from the top also missed, by a small margin. These
values come from a retracted attempt, are archived in `artifacts/bpn201-retracted/` and enter no
table; `bpn202` recomputes them and its values are the record. The refits themselves were cheap
(3–22 s each; design rows $\le 8704$, support $\le 2048$), which bounds the cost of raising the
fit-state cap in `bpn301` once that size has been smoked.

The `runs/bpn201/` staging copy is deleted locally with the remote directory; the retracted
archive is `artifacts/bpn201-retracted/` (`FAILURE.json`, verbatim logs, scheduler record,
partial `result.json`, the six transferred rule files, remote hashes).
