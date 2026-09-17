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
