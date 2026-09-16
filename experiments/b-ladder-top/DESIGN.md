# The top of the correction ladder, and the combined envelope — predeclared design

Two questions, in order, on the frozen Burgers checkpoint
`sep_hfit_dense_mid_N256_dense.pkl`
(SHA256 `18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589`, $K=16$,
$R=512$), at 256 intervals, $\Delta t=0.005$, on the same six opened development cases
the head ablation, the audited ladder and the cheap-corrections cell used
(`eval_seed 7090702` $\times 4$, `eval_fresh_seed 911702` $\times 2$).

**Q1.** Make the $q=256$ rung converge. The cheap-corrections block-damped solver
converges through $q=128$ at a median three iterations per step, but $q=256$ exits on
the iteration budget and $q=512$ never reaches the shared stationarity rule. Three
candidate fixes, each isolated and each reported whether it works or not.

**Q2.** One job that prices the whole envelope: the $q$ ladder crossed with the
quadrature and the evolution tolerance, against same-job full-order controls,
POD-LSPG, and the trained FNO if it can be run in the same allocation.

Nothing about the model changes: the checkpoint, the nested direction rule and its
seed, the weak objective, the initializer policy, the output contract, the mesh, the
time step and the cohort are all the cheap-corrections contract verbatim. Only the
linear algebra inside the solver, the starting iterate, and the offline quadrature
budget change.

---

## The inherited contract

With $\Phi\in\mathbb R^{n\times M}$ the $M$ lowest discrete sine test modes
($\Phi^\top\Phi=I$), $\lambda$ their Laplacian eigenvalues, $G\in\mathbb R^{n\times R}$
the frozen separable bank, $A=\Phi^\top G$, backward Euler at step $\Delta t$ and the
FOM's own sign-upwind advection $\mathcal N$, the reduced state is $u=G\eta$ with

$$\eta(z,y)=h_\theta(z)+C_q\,y,\qquad z\in\mathbb R^{K},\ y\in\mathbb R^{q},\ K=16,$$

$$r(w)=\frac{A\eta-p+\Delta t\big(\Phi^\top\mathcal N(G\eta)+\nu\,\lambda\odot A\eta\big)}
{1+\Delta t\,\nu\lambda},\qquad p=A\,\eta^{\rm prev},\qquad w=(z,y).$$

$C_q$ is the first $q$ columns of the audited nested direction matrix (field-metric POD
of the decoder-output residual $\eta-h_\theta(z^\*)$, 1024 seeded snapshots, 4
multistarts, budget 200, seed 20260915). Writing $G=Q_GR_G$ for the thin QR and
$\tilde C_q$ for the field-orthonormal directions, $C_q=R_G^{-1}\tilde C_q$, so

$$G\,C_q=Q_G\tilde C_q\quad\text{has orthonormal columns:}\quad \|G C_q\,\delta_y\|_2=\|\delta_y\|_2 .$$

That identity is used below: a step in $y$ *is* its own field-norm increment, so a trust
radius on $y$ is a physical radius and needs no mesh-dependent scaling.

The retained solver (`block` in the cheap-corrections cell) takes one Jacobian
$J=\partial r/\partial w$ per iteration and solves

$$\big(J^\top J+\lambda D_z+\varepsilon_0 D_y\big)\,\delta=-J^\top r,
\qquad D=\operatorname{diag}\!\big(\operatorname{diag}(J^\top J)+10^{-30}\big),$$

with $D_z$ keeping only the latent diagonal, $D_y$ only the correction diagonal,
$\varepsilon_0=10^{-10}$ **fixed**, and acceptance iff
$\|r(w+\delta)\|<\|r(w)\|$ and $\|\delta_z\|\le\Delta$ ($\Delta$ the $q=0$ trust
radius). On acceptance $\lambda\leftarrow\max(\lambda/3,10^{-12})$, otherwise
$\lambda\leftarrow\min(10\lambda,10^{14})$. A solve is **converged** only if every time
step and the initial fit exit for a reason in $\{1,2,4\}$ with no iteration-budget exit
*and* the joint normalized gradient

$$g=\frac{\sqrt{\|J_z^{\top}r\|^2+\|J_y^{\top}r\|^2}}
{\sqrt{\|J_z\|_F^2+\|J_y\|_F^2}\;\|r\|}\ \le\ 10^{-6}.$$

## What is actually wrong at the top of the ladder

Read off the audited cheap-corrections rows, the two failing rungs fail for *different*
reasons, and the design separates them:

* **$q=256$** (`q256_m2_dense_block`): 15 iteration-budget exits over 18 invocations,
  worst $g=1.19\times10^{-3}$, median 4.75 iterations per step. This is a genuine
  solver failure — the iteration runs out of budget before it is stationary.
* **$q=512$** (`q512_m2_dense_block`): **zero** budget exits, median 2 iterations per
  step, worst $g=1.38\times10^{-1}$. It exits early on the *residual* criterion
  ($\|r\|\le10^{-9}\cdot\text{scale}$, exit reason 1) with a normalized gradient that is
  still $O(10^{-1})$. With 512 correction directions the enriched manifold can drive the
  weak residual to the absolute tolerance, at which point
  $\|J^\top r\|/(\|J\|\,\|r\|)$ is a $0/0$ ratio and stops being a stationarity measure
  — the failure mode `arms.py` already documents for attainable reduced fits. So the
  $q=512$ rung is not "not converged" in the sense the $q=256$ rung is.

Both readings are hypotheses stated before the run. Q1 records, for every arm and every
time step, the exit-reason histogram, $\|r\|/(\text{tol})$ at exit, $g$, and the
per-block gradients $\|J_z^\top r\|/(\|J_z\|\|r\|)$ and $\|J_y^\top r\|/(\|J_y\|\|r\|)$,
so the two diagnoses are confirmed or refuted from the data rather than asserted.

---

## Fix (a) — coarse-to-fine (cascade) warm start

For rung $q$, take the converged rung $q'=q/2$ and pad its solution with zeros:

$$w^{(0)}_{t}=\big(z^{(q')}_{t},\ y^{(q')}_{t},\ 0_{q-q'}\big)\in\mathbb R^{K+q},
\qquad t=0,\dots,T,$$

which is legitimate because the directions are **nested**: the first $q'$ columns of
$C_q$ *are* $C_{q'}$, so the padded vector names exactly the state the coarse rung
converged to. At time step $t$ the initial iterate is chosen as the best of

$$\big\{\,w^{(0)}_{t+1}\ (\text{padded coarse}),\quad w_t\ (\text{current}),\quad
2w_t-w_{t-1}\ (\text{the retained extrapolation})\,\big\}$$

by residual norm, so the cascade can never start from a worse iterate than the retained
rule. At the initial fit only the $z$ start changes ($z^{(q')}_0$ in place of the
nearest-code pick), because the initial fit is **linear** in $y$ and the retained
`block` path already eliminates $y$ there exactly by an orthogonal projection onto the
complement of $\operatorname{col}(R\,C_q)$; there is nothing left to warm-start in $y$.

**Cost accounting, declared here.** A cascade rung is not free: it needs its coarse
rung. Every cascade arm reports **two** costs — `median_gpu_ms` (its own timed query,
what a deployment that already holds the coarse trajectory pays) and
`cascade_total_gpu_ms` (its own median plus the median of the coarse arm it was started
from). Non-domination and the Q2 envelope use `cascade_total_gpu_ms` for cascade arms;
both numbers appear in every table.

## Fix (b) — column equilibration of the augmented Jacobian

Let $s_i=\|J e_i\|_2$, floored at $10^{-16}\max_j s_j$, and $S=\operatorname{diag}(1/s_i)$.
Solve

$$\big(\tilde J^\top\tilde J+\lambda \tilde D_z+\varepsilon\tilde D_y\big)\tilde\delta
=-\tilde J^\top r,\qquad \tilde J=JS,\qquad \delta=S\tilde\delta ,$$

where $\tilde J^\top\tilde J$ has unit diagonal, so $\tilde D_z,\tilde D_y$ are plain
indicator diagonals.

**This is a no-op in exact arithmetic.** With $D=\operatorname{diag}(J^\top J)$ the
retained damping is already Marquardt-scaled: $S(J^\top J+\lambda D)S=\tilde J^\top\tilde
J+\lambda I$ exactly. The fix is therefore purely a floating-point conditioning fix, and
it is expected to matter only because the trailing residual-POD directions have field
singular values many orders below the leading ones, so the columns of $J_y$ span many
orders of magnitude and the dense LU solve of the unscaled normal equations loses digits.
Its effect is measured, not assumed:

* offline, at a fixed representative state (case 0, step 25), the 2-norm condition
  numbers $\kappa(J^\top J+\lambda D)$ and $\kappa(\tilde J^\top\tilde J+\lambda I)$ at
  $\lambda\in\{0,10^{-6}\}$, together with $\min_i s_i$, $\max_i s_i$ and
  $\kappa(J)=\sigma_{\max}/\sigma_{\min}$, for $q\in\{0,128,256,512\}$;
* online, the achieved $g$, the iteration counts and the budget exits.

If the condition numbers are benign and the arm changes nothing, that is the reported
answer.

## Fix (c) — a damping and trust schedule for the $y$ block, decoupled from $z$

Replace the fixed $\varepsilon_0=10^{-10}$ by its own Levenberg schedule and give the
$y$ block its own trust radius:

$$\big(J^\top J+\lambda D_z+\varepsilon D_y\big)\delta=-J^\top r,
\qquad \|\delta_z\|\le\Delta_z,\quad \|\delta_y\|\le\Delta_y,$$

$$\varepsilon_0=10^{-6},\qquad
\varepsilon\leftarrow\max(\varepsilon/3,10^{-12})\ \text{on acceptance},\qquad
\varepsilon\leftarrow\min(10\,\varepsilon,10^{14})\ \text{on rejection},$$

with $\Delta_z=\Delta$ the retained $q=0$ radius and
$\Delta_y=\tau_y\,\|R_G\,\eta^{(0)}\|_2$, i.e. a fraction $\tau_y$ of the field norm of
the step's own starting state — well defined and mesh-independent because
$\|\delta_y\|_2$ is exactly the field norm of the correction increment (see above).
$\tau_y=0.1$ is declared here and not tuned.

The rationale is the retained solver's blind spot: with $\varepsilon$ frozen at
$10^{-10}$, a rejected step re-damps the latent block only, so the same near-unbounded
$y$ step is proposed again and again; $\lambda$ ramps to $10^{14}$ (or the budget runs
out) while the $y$ block is never regularised. Fix (c) makes rejection damp both blocks,
while acceptance still lets the $y$ block take the (essentially undamped) exact
Gauss–Newton step that made the cheap-corrections cell converge at $q\le128$.

## The arms

Five solver arms, so each fix is isolated and the two interesting combinations are
measured:

| arm | (a) cascade | (b) equilibration | (c) decoupled $y$ damping/trust |
| --- | --- | --- | --- |
| `base` | — | — | — |
| `pre` | — | yes | — |
| `damp` | — | — | yes |
| `predamp` | — | yes | yes |
| `casc` | yes | yes | yes |

`base` at a given $(q,M,\text{quadrature})$ must reproduce the cheap-corrections
`block` arm. At $q=0$ every arm is special-cased to the retained joint path with an
empty $y$, so all five must be bitwise identical there; that is checked in-job on field
SHA256.

---

## Q1 — the sweep

Burgers, 256 intervals, $\Delta t=0.005$, the six cases, 3 timed repetitions with a GPU
burn-in before every invocation, all repetitions retained, randomised subject order,
complete device-query timing (supplied dense initial field on GPU $\to$ six dense GPU
output fields), with the same-job full-order controls interleaved.

*Fidelity gates* (must reproduce the cheap-corrections rows):

| arm | reproduces | declared tolerance |
| --- | --- | --- |
| `q0_m4_dense_base` | `q0_m4_dense_block` | $\le10^{-9}$ relative on every reported error |
| `q0_m4_eq_base` | `q0_m4_eq_varpro` | $\le10^{-9}$ relative |
| `q128_m2_dense_base` | `q128_m2_dense_block` | $\le10^{-9}$ if the directions reproduce bitwise, else $\le10^{-3}$ |

The split tolerance on the $q=128$ gate is declared **before** the run for a reason the
cheap-corrections cell already measured: the regenerated direction matrix did not
reproduce bitwise across jobs when the GPU model differed (A100-PCIE-40GB vs A100 80GB
PCIe), while the fields agreed to $9.4\times10^{-7}$. $q=0$ does not use the directions
at all, so its gate is hard at $10^{-9}$. Whichever way it falls is reported as measured;
the gate is never relaxed after the fact.

*Convergence sweep*: $q\in\{128,256,512\}$ $\times$ five arms, rule `m2`
($M=2(K+q)$: 288, 544, 1056), dense quadrature, so the quadrature never confounds the
convergence reading. The `casc` arm at $q=256$ starts from $q=128$; at $q=512$ from
$q=256$; at $q=128$ from $q=64$ (run untimed for that purpose only).

*Quadrature*: one empirical rule per rung is fitted for $q\in\{128,256,512\}$ with a
3000 s walltime budget each and a block size of $m/20$, targeting
$m=\min(4M,2048)$. **A rule is VALID only if the fitter stopped on target support or on
the gradient criterion, never on walltime** (`truncated == false`). This is
pre-registered and non-circular: it does not look at the rung's error. An invalid rule
means the rung is reported as **dense**, and the broken `q512_m2_eq_block` row from the
cheap-corrections cell (27.8 % same-grid against 0.60 % for its dense twin) is the
reason the gate exists. Each rung's converged arm, if any, is also timed with its rule.

*Reported per arm*: median and maximum iterations per step, budget exits, exit-reason
histogram, worst joint normalized gradient $g$ and its two block components, worst
same-grid error, worst error against the 4096-interval reference, median GPU ms, and —
for cascade arms — the cascade total.

### Pre-registered pass for Q1

> $q=256$ **converged** — no budget exits, $g\le10^{-6}$ on every case and every time
> step under the shared rule — with worst same-grid error **strictly below** the
> $q=128$ rung, at a stated cost.

The $q=128$ comparator is the same-job `q128_m2_dense_base` arm, not the cross-job
cheap-corrections number.

### What would falsify the premise

No fix removes the budget exits; or the fixes remove the budget exits but the converged
$q=256$ answer is *worse* than $q=128$, showing the audited ladder's monotonicity came
from over-fitting an unconverged iteration; or every fix converges only by exiting on
the residual criterion with $g$ still $O(1)$, showing the stopping rule, not the solver,
is what the top of the ladder breaks.

---

## Q2 — the combined envelope, in one job

Grid, all in one Slurm job on one GPU, one randomised order, 3 timed repetitions with
burn-in, every repetition retained:

* **ROM**: $q\in\{0,16,32,64,128\}$ and $q=256$ *if Q1 converged it*, each with its
  empirical quadrature rule, $\times$ evolution tolerance $\in\{10^{-3},10^{-6}\}$.
  Test count $M=256$ (fixed) for $q\le128$ — the rule the cheap-corrections
  non-dominated frontier used — and $M=2(K+q)=544$ for $q=256$, which is the smallest
  legal fixed count there; the change of rule at the top rung is declared, not hidden.
* **Dense control at $q=0$**: `q0_m256_dense` at both tolerances, so the quadrature's
  cost factor is separable from $q$.
* **Full-order controls, same job**: `fft_tight` (Newton $10^{-6}$, linear $10^{-8}$,
  $\Delta t=0.005$), `nt1e-2_dt01` (Newton $10^{-2}$, $\Delta t=0.01$) and
  `nt1e-4_dt005` (Newton $10^{-4}$, $\Delta t=0.005$) — the two the fixed-checkpoint
  tuning study identified as the real bars, plus the converged reference solve.
* **POD-LSPG**: head-ablation arm (e) verbatim — classical POD of the same truth
  snapshots, identity head, $k'\in\{16,32,64,128\}$, $M=4k'$, dense quadrature,
  evolution tolerance $10^{-6}$.
* **FNO**: the trained `fno-large` checkpoint from the `2026-09-14-no-audit` lane, run
  **inside this job's allocation** on the same GPU, on the same six cases, with that
  lane's `timing.py` protocol replicated (timed region = supplied on-device field and
  viscosity to on-device complete trajectory; `torch.cuda.synchronize()` brackets every
  repetition; its own burn-in; host transfer timed separately; every repetition
  retained). Routine calls recorded below. If it cannot be run, it is omitted and the
  omission is stated — no timing is imported from the other job.

### Both metrics on every row

The same-grid error of a ROM query is measured against the same-job converged full-order
solve `fft_tight`, which returns the supplied field exactly at $t=0$. Three columns are
reported for every subject:

$$\text{t0 compression}=\frac{\|f_0-u_0\|}{\|u_0\|},\qquad
\text{worst all times}=\max_{t\in\{0,\dots,0.25\}}\frac{\|f_t-f^{\rm FOM}_t\|}{\|u_0\|},$$
$$\text{worst evolved times}=\max_{t\in\{0.05,\dots,0.25\}}\frac{\|f_t-f^{\rm FOM}_t\|}{\|u_0\|}.$$

This matters and is the reason the task asks for both: the decoder reproduces the
*supplied* field only to its own compression error, which pins the all-times metric,
while the FNO returns the supplied field bitwise and has t0 compression exactly zero.
Reporting only the all-times metric flatters the FNO; reporting only the evolved metric
hides a real cost the ROM's output contract pays. Both are reported, neither is
preferred, and the non-dominated set is given on each.

### Pre-registered criterion for "$q$ is a knob"

On the **evolved-times** metric:

1. worst evolved error monotone non-increasing in $q$ within the retained configuration
   (fixed $M$, EQ, tolerance $10^{-6}$);
2. at least **3 converged non-dominated points** on the (median GPU ms, worst evolved
   error) plane, spanning $\ge2\times$ in cost **and** $\ge2\times$ in error.

Whether the all-times metric also passes is reported, and is expected not to, because
the t0 compression floors it.

---

## Gates before any verdict

| # | gate | tolerance |
| --- | --- | --- |
| i | local smoke: every new arm at $q=0$ is bitwise identical to the retained path | field SHA256 equal |
| ii | local smoke: `pre` reproduces `base` on the solved field at small $q$ | $\le10^{-9}$ relative |
| iii | in-job: $q=0$ arms reproduce the cheap-corrections `q0_m4_dense_block` / `q0_m4_eq_varpro` errors | $\le10^{-9}$ relative |
| iv | in-job: `q128_m2_dense_base` reproduces `q128_m2_dense_block` | $\le10^{-9}$ bitwise-directions, else $\le10^{-3}$ |
| v | regenerated `directions_sha256` equals `f270e5bf…` | bitwise (informational, as in the cheap-corrections cell) |
| vi | independent NumPy audit recomputes every reported error from the saved fields | $<10^{-9}$ |
| vii | every solve carries its exit reason, budget-exit count and all five stationarity norms | present for all |
| viii | every EQ rule carries `truncated`, `support`, `relative_fit`, `fit_seconds`; a truncated rule disqualifies its arm | present for all |
| ix | every ROM invocation satisfies $M>K+q$ | all |
| x | Q2: the FNO's six inputs are disjoint from its training set by generation descriptor and by input-field hash | exact |
| xi | Q2: the FNO returns the supplied field at $t=0$ exactly | bitwise |
| xii | backend gpu, x64, matmul precision highest, checkpoint unchanged before/after | all |

---

## Routine calls made without asking

Recorded here rather than folded in silently.

1. **`--gpu a100` for both jobs**, excluding `pax007` (it has failed `cuInit` in this
   campaign), matching the GPU class of the cheap-corrections job so the cross-job
   fidelity gates have their best chance.
2. **Q1 uses rule `m2` ($M=2(K+q)$) across $q\in\{128,256,512\}$.** It is the smallest
   legal fixed rule at the top rungs and the one the cheap-corrections cell ran there,
   so the gate comparators exist. `m256` is illegal above $q=239$.
3. **Q1's quadrature is dense except for the per-rung EQ arms**, so the convergence
   reading is never confounded by a quadrature rule.
4. **The EQ walltime budget is raised from 300 s to 3000 s** and the block size from
   $m/64$ to $m/20$; the target support is $\min(4M,2048)$. The validity gate is
   `truncated == false`.
5. **`run_flat_directions` is off.** The cheap-corrections cell already measured that
   the flattened fit is bitwise identical and ~124 s *slower*; re-running it would cost
   ~1100 s for a known answer.
6. **The `casc` arm's coarse rung is run untimed in the setup phase** when it is not
   already a timed subject, and its setup time is recorded in the arm's setup block.
7. **Evolution tolerance is split from the initial-fit tolerance.** `ic_gtol` stays at
   $10^{-6}$ in every arm; only the per-step `gtol` is swept in Q2. When both are
   $10^{-6}$ the code path is the retained one exactly.
8. **An arm run at evolution tolerance $10^{-3}$ records two flags**: `converged` at its
   own tolerance and `stationary_1e-6` under the shared rule. The Q2 frontier's
   "converged" fill uses the shared rule.
9. **The FNO runs as a second process inside the same Slurm allocation**, after the JAX
   phase exits, rather than inside the JAX process. JAX preallocates most of the device
   and PyTorch would be starved; a second process on the same GPU in the same job keeps
   the "one allocation, one GPU, no cross-job ratio" property that matters, at the cost
   of the "one process" wording in the other lane's protocol. The deviation is stated in
   the report. If the FNO phase fails for any reason the JAX results are already written
   and the job still completes.
10. **The FNO checkpoint is staged out of band.** `best.pt` (429 MB) lives under a
    `.gitignore`d `runs/` tree in the other lane, so it cannot be staged from a Git
    object like every other file; it is copied directly and its SHA256 is recorded in
    the attempt's provenance instead.
11. **Q1 is submitted first and Q2 after it lands**, because Q2's top rung is
    conditional on Q1's verdict. That is a true dependency, not a sequencing of
    independent work.

## Amendments

*(Appended as they are made; nothing above is edited after the first commit.)*

### After the local smoke, before submission

1. **`base` at $q>0$ is not bitwise identical to the cheap-corrections `block` arm.**
   The new solver records two extra per-step block gradients inside the same `scan`,
   which changes XLA fusion, exactly as the cheap-corrections wrapper did against
   `accuracy_paths.make_rom`. The smoke measures the difference at small $q$ and the
   in-job gate iv carries the cross-job comparison. At $q=0$ bitwise identity is
   preserved because the retained path is taken verbatim.
2. **Q1's `casc` arm at $q=128$ starts from $q=64$**, which is a converged rung in the
   cheap-corrections data, so the cascade is tested at a rung where the baseline already
   converges and any change is attributable to the warm start alone.
3. **The local smoke runs at 64 intervals with $q\in\{0,8,16\}$** and is kept under a
   minute, per the shared-box rule; the cheap-corrections cell recorded a full-mesh
   local cost probe as a deviation and this cell does not repeat it. The real fidelity
   reproduction is the in-job gate, which is also the only place the numbers are
   comparable.

### After the runs, before the report

*(filled in below at close)*
