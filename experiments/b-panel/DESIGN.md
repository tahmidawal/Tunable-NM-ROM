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
non-empty, so not disk-full. Not memory either: the batch step's peak host RSS was 76 023 968 K = 72.5 GiB of 240 GB,
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

**Submitted.** `bpn202` = job **3787247** (H200, `--mem 240G`, staged from `494c3f48`, remote
`b_panel_20260917/bpn202`, manifest verified remotely, one `bpn_` job in the queue before and
after). The remote `bpn201` directory and its local staging copy are deleted; the retracted
record is `artifacts/bpn201-retracted/`. The smoke that gated the fix: `checks/smoke-panel-a6.json`
(baseline $1.4\times10^{-14}$, GATE5 override honoured, GATE6 the three cluster configs declare).

## A7 — 2026-09-17 12:20, after `bpn202` failed: an out-of-memory in the untimed reconstruction diagnostic; fixed and smoked, NOT resubmitted by this session

`bpn202` (job 3787247, H200 on `pax011`, 240 GB, staged from `494c3f48`) exited 1 after 28m37s.
Not disk (paralab 92 %, 400 GB free at start; both logs non-empty) and not host memory (batch
peak RSS 86 GiB of 240). The GPU preflight, the six 4096-interval references, the 1024²
snapshots, **all six rule transfers, all 29 subject builds and the q = 0 and q = 64
reconstruction diagnostics completed**; the crash is in the *untimed* best-found reconstruction
at q = 256, verbatim from `logs/3787247.err`:

```
  File ".../experiments/b-panel/panel.py", line 630, in main
    z, rn, it, reason = host(recon(starts, target, G))
jax.errors.JaxRuntimeError: INTERNAL: Failed to get configs for: 2 out of 73 instructions.
  ... Autotuning failed for HLO: %input_reduce_fusion.19 = f64[1046529,8]{1,0} ...
  Failed to profile configs: Out of memory while trying to allocate 16.98GiB.
```

`arms.make_reconstruction` vmaps one Levenberg–Marquardt solve per start; with eight starts at
$q=256$ on the $1024^2$ grid the batched Jacobian is `f64[8, 2, 523266, 272]` (18.2 GB) and an
`[8, n, 512]` intermediate is 34.3 GB, on top of 29 resident subjects — of which the twelve EQ arms
each held their **own** copy of their test matrix $\Phi_M$ (≈ 40 GB at $1024^2$). No timed subject
had been invoked; no number exists. The record is `artifacts/bpn202-failed/` (`FAILURE.json` with the
verbatim traceback and what completed, logs, the partial `result.json`, the six transferred rules,
remote hashes); the remote directory and the local `runs/bpn202/` staging copy are deleted.

**What the attempt showed before it died.** The six transfers recomputed exactly the retracted
`bpn201`'s values: $q=0,16,32$ primary ($\rho_{\max}$ 0.0180, 0.0594, 0.0313 on 64/64/42 fit states);
$q=64,128,256$ **uncertified** (0.1316, 1.0295, 0.4144 on 25/14/8 states). These are the values of
record for the `clip(8192/M, 8, 64)` convention and confirm §A5.2's prediction (and its q = 64 miss).

**What changed (this commit), and why it does not alter the timed science.**

1. *Reconstruction diagnostic.* It now runs **before the subjects are built**, when the GPU holds
   only the bank, the test matrices and the snapshot basis (≈ 30 GB at $1024^2$ against the
   34 + 18 GB transient, on 141 GB), over the declared rom $q$'s and POD ranks instead of the
   built ones; the programs and their inputs are unchanged, so the values are. It is also
   OOM-tolerant: an OOM records `dropped[name=reconstruction_q{q}, phase=reconstruction]` and the
   job continues to the timed subjects. **Chunking the eight starts was tried first and
   dropped.** The smoke's chunk gate failed — 0.6 % and 9.5 % *relative* differences in the
   best-found residual — but a probe at budgets 100 / 400 / 1000 (`checks/recon-chunk-probe.json`)
   shows those are relative comparisons of round-off-level residuals ($\approx10^{-13}$ on the
   attainable fixture fit; the found codes agree to $8\times10^{-14}$ and the whole batch is
   bitwise stable across reruns), so chunking is equivalent to round-off and the gate was badly
   posed, not the transformation. It was still dropped: moving the diagnostic keeps the
   pre-registered vmapped program byte-identical, which a chunked variant cannot claim.
2. *Memory.* Every EQ arm now reuses the dense arm's resident $\Phi_M$ and $A=\Phi_M^\top G$ for
   its $M$ (the same device inputs through the same program) instead of holding its own copy — 12
   copies (≈ 40 GB) at $1024^2$, 24 copies (≈ 4.9 GB) at $256^2$. Covered by the smoke's
   $10^{-12}$ baseline gate, the in-job matched-rule bitwise gate (§A8) and the cross-job
   fidelity gates.
3. *Fit-state count* — the fix §A5.2 deferred to `bpn301`, now made because the transfer path
   is being touched anyway. `nfit = fit_states` (64) at every rung; `max_fit_rows` is retired
   and the driver asserts it is absent. The size it produces — 64 × 1088 = 69 632 rows on a
   2048-point support — is 87 s of host NNLS on the GB10 (`checks/nnls-size.json`: 12.4 / 45.3 /
   86.8 s at 8 704 / 34 816 / 69 632 rows), so the cap the convention existed for is not needed.
   This **is** a science change for any future $1024^2$ / $512^2$ transfer, pre-announced in §A5.2.

**Not resubmitted here.** The coordinator's instruction for this session was `bpn301` first and
`bpn202` left as it was; `bpn202` has landed, failed, and the fix is committed and smoked. A
resubmission (`bpn203`: same `config-1024.json`, H200, 240 GB) is one `stage.py` + `sbatch` away and
waits for the coordinator's word. Jobs used: three (`bpn101`, `bpn201` retracted, `bpn202` failed).

## A8 — 2026-09-17 12:45, the local smoke that gates `bpn301`: two rule sets, the uncapped
fit-state fix, and the diagnostic-before-build reorder, all in one driver — every gate passes

`bpn301` carries both `eqcert` (qrg304's rules) and `eqtop` (b-eqtop's final exported set,
§A5/§config-256.json) as arms at matched $q$, $M$ and tolerance, on top of §A7's uncapped
fit-state count and reordered untimed reconstruction diagnostic. Before staging, the smoke
(`smoke_panel.py`, `checks/smoke-panel-a8.json`) was run to gate all of it, including a new
in-job gate this multi-rule-set change requires: **two rule sets whose files are byte-identical
must reproduce each other's fields bitwise.** The smoke drives this with a synthetic second set
(`eqdup`) that points at the *same* `.npz` files as `eqcert` — the cheapest way to exercise the
multi-set code path without the real `eqtop` files, which differ from `eqcert`'s at $q\ge64$ by
construction and are not expected to reproduce them.

**Gate values (`checks/smoke-panel-a8.json`):**

| gate | result |
|---|---|
| GATE1 `baseline_saved_case` — reproduces the saved audited case | relative $\ell_2$ $1.44468\times10^{-14}$, latent $8.2477\times10^{-13}$, converged, 0 budget exits — **pass**, matches §A6's $1.4\times10^{-14}$ |
| GATE4 `duplicate_rule_set_bitwise` — a second rule *set* over the same files reproduces the first | 4 arm pairs, 8 invocations, **all bitwise** — **pass** |
| GATE4b `matched_rule_files_bitwise` (in-job driver gate, new this commit) | 4 pairs ($q\in\{0,4\}\times g\in\{10^{-6},10^{-3}\}$), each pair one file SHA256, `same_fields=true`, `same_iterations=true` for all 4 — **pass** |
| GATE4b audit recomputation (`audit_panel.py`, independent of the driver) | `saved_fields_identical=true` for all 4 pairs, recomputed from the saved field arrays, not from the driver's own verdict — **pass** |
| GATE5 `priority_override_honoured` | declared build order matches the override for both a partial and a full override list — **pass** |
| GATE7 `fit_states_uncapped` (§A7's fix) | configured 16 fit states; used $[16,16]$ at $q=\{0,4\}$; design rows $[1024,1280]=16\times M$ exactly, no cap applied — **pass** |
| `transfer_fit_cert_disjoint` (smoke128) | **pass** |
| `fft_tight_converged_everywhere` (both smoke64 and smoke128) | **pass** |
| `repetition_output_identical` (both) | **pass** |
| `directions_file_sha256` / `directions_prefix_hashes` (both) | **pass** |
| `direct_reproduces_fft_tight` / `fast_parity` (smoke64) | **pass** |
| `cluster_config_declarations` | `config-256.json`: 47 subjects declared, override list of 38 names honoured, includes both `eqcert` and `eqtop` arms at all six $q$ and both tolerances (§config-256.json's `extra_rule_sets`); `config-512.json` / `config-1024.json`: 35 / 29 subjects, override honoured (these two meshes are not part of `bpn301` and still carry only the single-set `eqxfer` arms; unaffected by this change) — **pass** |
| `audit_failed` (both driver runs) | `[]` — **pass** |

No gate failed. The `q0`/`q4` transfer certifications printed in the smoke
($\rho_{\max}=0.0761$ and $0.0632$ on the tiny 64-interval fixture) are sanity numbers for the
mechanism, not the real ladder's rules — those are qrg304's and b-eqtop's archived values,
carried by SHA256 and `construction_status`, not refitted here.

**Deviation, already priced under §A2's precedent.** `total_seconds = 167.86` for the combined
smoke64 + smoke128 driver runs, over the nominal sub-minute local budget; recorded, not hidden,
same category as the deviation §A2 already logged for this lane's smoke.

**What changed in the driver to make these gates possible** (already committed by the previous
session in this worktree, verified here, not re-derived):

1. `panel.py`: the untimed reconstruction diagnostic now runs before subjects are built (§A7),
   `max_fit_rows` is retired and `nfit = fit_states` unconditionally (§A7), every EQ arm reuses
   the dense arm's resident $\Phi_M$/$A$ instead of holding its own copy (§A7), and the new
   `matched_rule_files_bitwise` gate compares fields and iteration counts for any two rule-set
   arms that share a file SHA256 at the same $q$ and `gtol` (this commit, exercised by GATE4b).
2. `cluster/stage.py`: stages every file under `config['extra_rule_sets'][*]['rules']`, keyed by
   that set's `subdir` (`rules-eqtop/` for `eqtop`), in addition to the existing `rules/`.
3. `config-256.json` (`attempt: bpn301`): `extra_rule_sets` adds the `eqtop` set at
   `inputs/rules-eqtop/`, one file per rung (`q=0,16,32`: b-eqtop's own copies of qrg304's
   $m=1024$ files, confirmed constructions 3/3, 3/3, 2/2; `q=64`: bet201's $m=2048$ `std` rule,
   confirmed 2/2; `q=128,256`: single-draw rules above the marginal $m=2048$, status
   `certified in one draw`); `priority_override` lists all six FOM controls, both rule sets at
   both tolerances, the six POD ranks, the fidelity arm, the free bank and the `fast` kernel in
   that order, with a note that the loose-tolerance ($10^{-3}$) EQ arms of both sets are the
   first to drop under an OOM, ahead of anything not already duplicated in the table; `inputs/`
   carries `rules-eqtop/` (the six `.npz` verified against `SHA256SUMS.b-eqtop` and
   `PROVENANCE.json`'s own re-hash) and `PROVENANCE.json` records, per file, `source_lane`,
   `construction_status`, `export_basis` and the `eqtop_status_note` quoted from b-eqtop's
   provenance so the caveat travels with the data, not only with this design document.

**Not re-run.** This smoke was produced by the previous session in this worktree before it was
killed by the model usage limit; per the lane protocol's local-smoke rule and this task's
instruction, it is read and recorded here rather than re-executed, since every gate above
already passed. Jobs used: still three (`bpn101` complete, `bpn201` retracted, `bpn202`
failed); no cluster job has run since this amendment. `bpn301` is staged and submitted next.

**Submitted.** `bpn301` = job **3789570**, `NVIDIA A100` with `--constraint=a100-80G`, `--mem 180G`,
`gpu` partition, `--exclude=pax007`, staged from commit `f3c5fbed`, remote
`b_panel_20260917/bpn301`, manifest verified remotely (`sha256sum -c MANIFEST.sha256 --quiet`).
`squeue -u tawal01` before submission showed no `bpn_` job; immediately after, exactly one,
`bpn_bpn301` (3789570), `PD`. Not waited on.

## A9 — 2026-09-17 ~13:00, coordinator follow-up on `bpn202`: cause confirmed, not the refit;
resubmitted as `bpn203` with the same §A7/§A8 fix, no arms dropped

The coordinator asked, after `bpn301` was submitted, for `bpn202`'s cause to be pulled from its
remote log, recorded, archived and cleaned up, and — conditionally, if the transferred-rule
refit turned out to be the problem — for the quadrature arms to be dropped from the $1024^2$ job
rather than fought. All of the recording/archiving/cleanup the coordinator asked for had already
been done by the session that hit the usage limit (§A7, before this session started): the
verbatim cause is `artifacts/bpn202-failed/FAILURE.json` (checked again here, not re-derived),
the logs are archived alongside it, and the remote attempt directory is confirmed deleted —
`ssh tufts-login "ls /cluster/tufts/paralab/tawal01/b_panel_20260917/"` returns empty. Disk was
not the cause: 92% full, 400 GB free at job start, matching §A6/§A7's check, and both logs were
non-empty (28m37s of real work, not a preamble death).

**The conditional does not fire.** The cause is *not* the transferred-rule refit: all six rule
transfers for `bpn202`'s config completed (37–390 s each; the values are in `FAILURE.json`'s
`completed_before_crash.rule_transfers`, identical to `bpn201`'s retracted numbers and to
§A5.2's prediction — $q=64,128,256$ uncertified for fit-state-starvation reasons, exactly as
predicted, but *not crashed*). The crash is an XLA autotuning OOM inside the **untimed best-found
reconstruction diagnostic** at $q=256$, after all 29 timed subjects had already been built and
were resident (`panel.py:630`, `INTERNAL: Failed to get configs`, allocation requests
16.97/16.98/31.97 GiB) — this is exactly what §A7 already diagnosed and fixed for the same
reason `bpn202` itself hit it. Dropping the quadrature arms would not have prevented this crash:
the diagnostic runs over every declared ROM $q$ regardless of which quadrature built it, and the
29 resident subjects that starved the autotuner's memory headroom are dominated by the *dense*
ladder's own test matrices, not by the EQ arms' (each EQ arm's $\Phi_M$/$A$ is now shared with
its dense twin, per §A7). The correct fix is therefore the one already committed — reorder the
diagnostic before the builds and share the resident test matrices — not removing arms that carry
crossover-relevant numbers the $1024^2$ job exists to produce.

**Smoked before resubmission.** The fix is the same code path `checks/smoke-panel-a8.json`
already gates (§A8): `GATE7` confirms the uncapped fit-state count: `cluster_config_declarations`
confirms `config-1024.json`'s 29 subjects and its `priority_override` are declared and honoured
under the current driver, unchanged in shape from what `bpn202` ran. No new local smoke was run
for this amendment; the mechanism smoke already covers the diagnostic reorder and the memory
share, and nothing else in `config-1024.json` changed. `config-1024.json`'s `attempt` field is
now `bpn203`; nothing else in the file differs from what `bpn202` staged (`git diff` against
`bpn202`'s content is one line).

**Resubmitted once, per the coordinator's instruction: `bpn203`.** Job id, GPU and namespace are
recorded in the lab log entry this session appends. Not waited on.

Job count after this amendment: `bpn101` (complete), `bpn201` (retracted), `bpn202` (failed),
`bpn301` (submitted this session), `bpn203` (submitted this session) — five of the cap of eight.

**Submitted.** `bpn203` = job **3789572**, H200, `--mem 240G`, `gpu` partition,
`--exclude=pax007`, staged from commit `f3c5fbed`, remote `b_panel_20260917/bpn203`, manifest
verified remotely. `squeue -u tawal01` immediately before this submission showed exactly one
`bpn_` job (`bpn_bpn301`, 3789570, from this session's other submission above); immediately
after, exactly two (`bpn_bpn301` 3789570, `bpn_bpn203` 3789572), each the only job in its own
attempt directory. Not waited on. This is the resubmission's only attempt, per the
coordinator's "do not resubmit more than once."

## A10 — 2026-09-17 ~18:20, after `bpn301` and `bpn203` both landed: what the two jobs say,
the §A5.2 prediction scored, and why `bpn401` (512²) is now worth its job

Both jobs completed with exit 0 (`sacct`: 3789570 `COMPLETED 0:0` 29m58s on `pax049`; 3789572
`COMPLETED 0:0` 1h33m43s on `pax010`), both printed `jax_backend=gpu x64=True precision=highest`,
both ended `ALL-DONE`, neither dropped a subject. Each was checksum-collected (`OUTPUTS.sha256`
and `MANIFEST.sha256` verified remotely before the archive was made and again locally after it),
independently NumPy-audited (`audit_panel.py`, which imports neither the driver nor JAX), re-derived
a second time by `checks/recheck_headline.py` (a different code path to the same headline
quantities: worst relative difference against the audit **0.0** for both jobs), archived as bounded
Git chunks whose concatenation reproduces the recorded whole-archive SHA256, and only then was its
exact remote attempt directory deleted. `ls` of the namespace is now empty.

**`bpn301` (256²).** 47 timed subjects, 43 gates, none failed. Its table replaces `bpn101`'s
wholesale per §A5.1; `bpn101` is not withdrawn and stays in `artifacts/bpn101/`. The pre-registered
verdict is unchanged: **0 of the 39 admissible reduced subjects are non-dominated** on (median GPU
ms, worst evolved %) — the admissible frontier is the FNO plus full-order Newton. What the second
rule set changes is inside the reduced set: the `eqtop` ladder is **monotone on evolved times at
both tolerances** where the `eqcert` ladder is not, and the break is exactly at the rungs b-eqtop
replaced. At q = 256 the `eqtop` rule halves the evolved error (1.0361 → 0.5129 % at gtol 1e-6) for
7 % more time, and reaches its dense twin's all-times error (0.9053 %) at 5.3× less time; at q = 64
it is 1.2275 → 1.0840 % for 24 % more time. At q = 0, 16, 32 the two sets are the same files and the
in-job `matched_rule_files_bitwise` gate confirms they produced bitwise identical fields and
iteration counts.

**Every `eqtop` row and caption carries its construction status**, which is the point of §A8: at
q ≤ 64 the constructions are confirmed (3/3, 3/3, 2/2, 2/2); at q = 128 and 256 the exported rules
are **single-draw** (`certified in one draw`) above an m where the same construction was measured
marginal. The q = 256 improvement above therefore rests on one draw, and the report says
"single-draw at q ≥ 64" rather than "certified" wherever those rungs appear.

**`bpn203` (1024²).** 29 timed subjects, none dropped, 29 gates passed and 2 not applicable
(`matched_rule_files_bitwise`: one rule set at this mesh; `cross_job_fidelity`: no comparator at
this mesh). Here the answer flips: **5 of the 20 admissible reduced subjects are non-dominated** on
(median GPU ms, worst evolved %) — `q0`, `q16`, `q32` transferred-EQ arms sit on the frontier
beside `nt1e-2_dt01`, `nt1e-4_dt005` and `fft_tight`. On the all-times metric no reduced arm is
non-dominated (the t = 0 compression term dominates it). The within-job reduced/full-order cost
ratios move the way a crossover should: the cheapest admissible reduced query is 4.480× the cheapest
same-job full-order setting at 256² and 1.818× at 1024², and 0.446× → 0.153× the same-job converged
`fft_tight`. Those are ratios formed inside one allocation each; the raw milliseconds of the two
jobs are never compared (A100 against H200).

**§A5.2's prediction, scored.** The prediction named the top two transferred rungs (q = 128, 256) and
said they would come back uncertified because the then-current `clip(8192/M, 8, 64)` convention would
fit them on 14 and 8 states. Its **outcome held**: under `bpn203` neither rung is primary-certified
(q = 128 ρ_max 0.1702, secondary; q = 256 ρ_max 0.3239, uncertified; q = 64 also missed at 0.1275).
Its **mechanism did not survive**, and this is the finding to carry: §A7 removed the cap, so every
rung in `bpn203` was fitted on **64 states, uncapped**, and the same rungs still miss the 0.116 bar.
Fit-state starvation is therefore not a sufficient explanation for the transferred top rungs at
1024². The capped values (q = 128 ρ_max 1.0295, q = 256 0.4144, from the retracted `bpn201`/`bpn202`)
are context only — the capped and uncapped refits are not a controlled A/B, so the comparison is
reported with that caveat rather than as a clean before/after. What is left as the likely cause is
the transfer itself: the mapped support keeps only 918–1713 of the 1024/2048 offered nodes at
positive weight, and the 1024² reachable population differs from the one the rule was built on.
The uncertified arms are timed and reported and excluded from the admissible frontier, as
pre-registered.

**Codex.** Still unavailable (quota until 2026-09-19 11:33; coordinator notice). The substitute
required by that notice is `reports/self-audit-2026-09-17-bpn301-bpn203.md`, generated by
`checks/self_audit.py` from the audit JSONs — one row per claim with the JSON field it rests on and
the check run against it, no number typed by hand.

**Is `bpn401` (512²) still worth its job? Yes, and more than before.** Before these two jobs the 512²
run was "a third point on a trend"; now it is the only measurement that can locate the crossover.
256² puts zero reduced subjects on the frontier and 1024² puts five, so the mesh at which the
reduced set becomes non-dominated lies between them and is currently unbounded. 512² would also
separate two explanations of the 1024² frontier that this panel cannot presently tell apart: a
genuine cost crossover, versus the H200's different balance of full-order and reduced work. The one
thing it should change relative to §3.3: run it on the **same GPU class as one of the two existing
jobs** (A100-80G, matching `bpn301`) so that at least two meshes share hardware. Job count after
this amendment: `bpn101`, `bpn201` (retracted), `bpn202` (failed), `bpn301`, `bpn203` — **five of the
cap of eight**, three free.

## A11 — 2026-09-17 ~18:30, before `bpn401`: the 512² panel, pre-registered

**Approved by the coordinator after §A10.** `bpn401` runs the 512² panel on an **A100-80G** so that it
shares hardware with `bpn301` (256²). Nothing about the science changes from `bpn301`/`bpn203`: the
same subject set, both rule sets where rules exist, the same-job full-order Newton grid plus
`dense_tight` and `fft_tight`, three timed repetitions, the §5 convergence rule, the §7 domination rule,
uncapped fit states per §A7.

**The question, pre-registered.** (i) Is the count of non-dominated *admissible* reduced subjects on
(median GPU ms, worst evolved %) at 512² **0 or > 0**? 256² gave 0 (`bpn301`), 1024² gave 5 (`bpn203`);
512² brackets the crossover. (ii) The same-job cheapest-admissible-reduced / cheapest-FOM cost ratio
(and the same against the same-job `fft_tight`), to sit between 4.480× (256²) and 1.818× (1024²) or not.
**The 512² and 256² ratios will be compared as ratios on shared hardware (A100-80G, both); 1024² stays on
the H200 and enters only as a ratio.** No raw millisecond is compared across jobs. Anything else the
job shows (which rungs certify on transfer, the eqtop-versus-eqcert ordering at the middle mesh) is
reported as observed, not as a pre-registered claim.

**What `config-512.json` declares** (`checks/config512-declare-a11.json`, produced by the driver's own
`declare_subjects` on the CPU, no GPU work): **47 subjects** — 8 FOM, 31 `rom` (6 dense, 6 + 6
transferred-EQ at 1e-6, 6 + 6 at 1e-3, the M = 256 fidelity arm), 6 POD ranks, the free bank, the fast
kernel. The two rule sets resolve to `eqxfer` (qrg304's rules, the primary set) and `eqtopxfer`
(b-eqtop's export as an extra set, added to the config for this attempt); both are 256²-grid rules and
both are transferred by the §3.2/§A3 path with `fit_states = 64` uncapped. All 12 rule files are present
under `inputs/rules/` and `inputs/rules-eqtop/` with SHA256 matching `PROVENANCE.json`. The
`priority_override` is the coordinator's reduced set (as `bpn203`); the default order then puts the
loose-tolerance EQ arms of both sets ahead of the fidelity arm, the free bank and the fast kernel under
the OOM-drop rule. Unlike §3.3's original plan the free bank is kept (it was in `config-512.json` already
and `bpn301` carried it at 47 subjects on the same GPU class); it is last but one in the build order and
is dropped first if memory binds.

**Memory arithmetic, recorded before the job.** 47 subjects on an 80 GB A100 held at 256² (`bpn301`, no
drop). At 512² the bank is 1.07 GB and the resident test matrices (shared between each EQ arm and its
dense twin per §A7) total ≈ 16 GB; the untimed reconstruction diagnostic runs before any subject is
built (§A7). If the OOM rule fires, the drops are recorded and reported, per §3.1; a drop of the free
bank or the fast kernel does not affect the pre-registered question. Every `eqtopxfer` row carries its
source rule's construction status from `PROVENANCE.json`, and its *transferred* certification (held-out
ρ at 512²) is what decides admissibility, exactly as at 1024².

**Mechanics.** A100-80G (`--constraint=a100-80G`), `--mem 180G`, `gpu` partition, `--exclude=pax007`, one
job in `b_panel_20260917/bpn401`, `squeue` before and after, not waited on. Expected wall time: between
`bpn301`'s 30 min (47 subjects at 256²) and `bpn203`'s 94 min (29 subjects at 1024² with six transfers);
twelve transfers and 47 subjects at 512² put the estimate at **1.5–2.5 h**; the allocation asks for 20 h.
Job count after submission: **six of the cap of eight** (`bpn101`, `bpn201` retracted, `bpn202` failed,
`bpn301`, `bpn203`, `bpn401`).

**Submitted.** `bpn401` = job **3805065**, `--gres=gpu:a100:1 --constraint=a100-80G`, `--mem 180G`, `gpu`
partition, `--exclude=pax007`, `--time 20:00:00`, staged from commit `7a57f02e`, pushed by `rsync` to
`b_panel_20260917/bpn401` and `sha256sum -c MANIFEST.sha256 --quiet` verified remotely before submission.
`squeue -u tawal01` immediately before showed no `bpn_` job; immediately after, exactly one,
`bpn_bpn401` (3805065), `PENDING (Priority)` — the only job in its attempt directory. The queue held 26
pending `gpu` jobs at submission; per protocol rule 4 it is escalated to H100 → H200 → L40S only if still
pending after 3 h, science unchanged (an escalation would forfeit the shared-hardware comparison with
`bpn301`, and is to be recorded here if it happens). Not waited on. Job count: **six of eight**.
