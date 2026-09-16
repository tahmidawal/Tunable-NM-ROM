# b-speed — how much faster can the frozen Burgers 256² query get at bit-level parity, and where does the time go?

Pre-registered before any run. Worktree `worktrees/2026-09-16-b-speed`, branch
`exp/2026-09-16-b-speed`, forked from `exp/2026-09-15-cheap-corrections` at `3ea369e0`.
Cluster namespace `/cluster/tufts/paralab/tawal01/b_speed_20260916/`. Hard cap: **3 cluster
jobs**.

## 1. The question

The incumbent Burgers query — arm `a_neural_eq` of the head ablation, audited at
**47.649 ms median GPU** on an `NVIDIA A100 80GB PCIe` at 256 intervals
(`experiments/head-ablation/artifacts/abl01/result.json`, job `3711424`) — is the single
frozen configuration the whole campaign prices everything against:

| item | value |
|---|---|
| checkpoint | `experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl`, sha256 `18f0266a…` |
| latent dimension | $K = 16$ |
| bank rank | $R = 512$ |
| weak test modes | $M = 4K = 64$ |
| empirical quadrature | $m = 4M = 256$ points, `eq_seed` 20259, candidate cap 8192, 64 fit states |
| time step | $\Delta t = 0.005$, 50 steps to $t = 0.25$ |
| outputs | six dense fields at $t = 0, .05, \dots, .25$ |
| solver | damped Levenberg–Marquardt, `gj` step solve, `ic_budget` 400, `step_budget` 180, `gtol` 1e-6 |

**Question.** How much faster can this exact query be made *at bit-level parity* — identical
output fields to $\le 10^{-12}$ relative, identical per-step iteration counts and identical
exit reasons — and where does the 47.6 ms actually go?

**Pre-registered target.** $\ge 2\times$ end-to-end single-query latency at parity. The
achieved factor is reported either way. This is the 1D result's target: the 2026-08-28 b1d
study reached 2.20–2.25× end-to-end at $\le 5\times10^{-9}$ parity by killing a cuSOLVER
call, fusing residual+Jacobian, hoisting per-trajectory constants and removing cuBLAS calls
on tiny shapes, and that report lists porting them to 2D as the obvious open lever.

**Falsification.** The target fails if no parity-passing arm reaches 2×. Then the deliverable
is the profile plus the measured reason, and the honest factor. An arm that is faster but
fails any parity gate is reported as **rejected**, not as a speedup: this cell may not change
what is computed, only how.

## 2. What runs today (the incumbent, read from source)

`experiments/head-ablation/arms.py::make_query` with `quadrature='eq'`, `linear='gj'`.
With $h = h_\theta(z) \in \mathbb{R}^{R}$, $G_5 \in \mathbb{R}^{m\times 5\times R}$ the bank on
the quadrature stencil, $A = \Phi^\top G \in \mathbb{R}^{M\times R}$ the exact linear test
projection, $P_q \in \mathbb{R}^{m\times M}$ the weighted quadrature rows, $\Lambda$ the $M$
sine eigenvalues and $p$ the previous step's projection, the weak residual is

$$
us_{s,i} = \sum_r (G_5)_{s,i,r}\,h_r,\qquad
\mathrm{adv}_s = c_s L\Big[\mathbb{1}_{c_s>0}(c_s - x^-_s) + \mathbb{1}_{c_s\le 0}(x^+_s - c_s)
                    + \mathbb{1}_{c_s>0}(c_s - y^-_s) + \mathbb{1}_{c_s\le 0}(y^+_s - c_s)\Big],
$$

$$
r(z) \;=\; \frac{A h \;-\; p \;+\; \Delta t\,\big(P_q^\top \mathrm{adv} + \nu\Lambda\odot A h\big)}
                {1 + \Delta t\,\nu\Lambda}\;\in\;\mathbb{R}^{M}.
$$

One LM iteration currently costs, per source reading:

1. `rn2 = ‖fun(zn)‖` — a **full primal residual evaluation** used only for the accept test;
2. `lax.cond(accept, evaluate(zn), keep)` where `evaluate` = `fun` **plus** `jax.jacfwd(fun)`
   — a **second primal** and the Jacobian;
3. `grad(r2,J2)` recomputing $J^\top r$, which the *next* iteration's `g = J.T @ r`
   recomputes again;
4. `H = J^\top J`, `g = J^\top r`, and `e.gj_solve` on the damped $16\times16$ normal system,
   which is a Python-level `for k in range(16)` — **16 strictly sequential elimination
   stages**, each 3–4 array ops.

Per time step there are additionally two probe residual norms ($\|r(z)\|$, $\|r(z_e)\|$) and
one $p = A h_\theta(z)$. The measured anatomy of this program class is already on record:
`experiments/separable-decoder/PROFILE.md` fits **295 µs fixed + 181 µs per LM iteration** at
$N=256$ on an A100, with the compiled residual+Jacobian moving 16 MB and doing 31 MFLOP in
123 µs — 130 GB/s and 254 GFLOP/s, two orders below the device — across 24 fusions. The
incumbent runs ≈ 200 evolution iterations plus 9–90 initial-fit iterations per query, so
47.6 ms ≈ 300 iterations × ≈ 160 µs. **The query is kernel-count bound.** Every optimisation
below removes kernels or sequential stages; none removes arithmetic that matters.

## 3. The optimisations, each an arm, each with its algebra

Two parity classes are declared in advance:

* **bitwise** — the transform provably evaluates the identical floating-point expressions.
  Gate: the output field sha256 equals the incumbent's, exactly.
* **reassociation** — algebraically exact, different rounding. Gate: fields agree to
  $10^{-12}$ relative, and iteration counts and exit reasons are *identical integers*.

### O1 `fuse` — one-pass residual+Jacobian, no `lax.cond`, one $J^\top r$ (bitwise)

`r, jvp = jax.linearize(fun, z)`; `J = vmap(jvp)(I_K)`. The linearisation primal is
`fun(z)` and the JVPs are `jacfwd`'s columns, so values are unchanged. This deletes the
separate accept-test primal (1 of 2 primals), deletes the conditional (replaced by
`jnp.where` on already-computed arrays), and reuses `g = J^\top r` for both the stationarity
ratio and the next step's gradient. Expected: −1 primal and −1 branch per iteration.

### O2 `hoist` — per-trajectory constants out of the LM loop (bitwise)

$\nu$ is fixed for a whole query, so $\nu\Lambda$ and $1 + \Delta t\,\nu\Lambda$ are constants
computed once instead of inside every residual evaluation (2–3× per LM iteration, ≈ 900× per
query). Identical expressions, identical values.

### O3 `share` — one evaluation for the step projection and the probe residual (bitwise)

Each time step computes $p = A h_\theta(z)$ and then, immediately, $\|r(z)\|$ at the **same**
$z$ — and $r(z)$ recomputes $h_\theta(z)$, the stencil contraction and $Ah$ from scratch.
Splitting the residual at $(us, Ah)$ lets one evaluation serve both: $p$ is the $Ah$ block of
that evaluation. This removes one head evaluation, one $5m\times R$ contraction and one
$M\times R$ matvec per time step (50 per query), evaluating the identical expressions.

### O4 `unroll` — `lax.scan(..., unroll=5)` on the 50-step loop (bitwise)

Loop control only; the emitted arithmetic is the identical sequence. Worth ≈ 4 % in 1D.

### O5 `probe` — the two extrapolation probes as one batched evaluation (reassociation)

$\|r(z)\|$ and $\|r(z_e)\|$ become one `vmap` over a 2-row stack: half the launches for that
part of each time step.

### O6 `lean` — fold the head's output layer into the frozen operators (reassociation)

The head is $h(z) = a_2(z)W_3 + b_3 + zW_{\mathrm{lin}}$. Set
$f(z) = [\,a_2(z)\,;\,z\,] \in \mathbb{R}^{d}$, $d = 128+K = 144$, and
$W_m = \begin{bmatrix}W_3\\ W_{\mathrm{lin}}\end{bmatrix}\in\mathbb{R}^{d\times R}$, so
$h = W_m^\top f + b_3$. Then the stencil contraction, the test projection and the head's last
layer collapse into **one** matmul per residual evaluation:

$$
\begin{bmatrix}\mathrm{vec}(us)\\ Ah\end{bmatrix}
= \underbrace{\begin{bmatrix}G_5^{(2)}\\ A\end{bmatrix}W_m^{\top}}_{S \in \mathbb{R}^{(5m+M)\times d}} f(z)
\;+\; \underbrace{\begin{bmatrix}G_5^{(2)}\\ A\end{bmatrix} b_3}_{s_0},
$$

with $G_5^{(2)}\in\mathbb{R}^{5m\times R}$ the reshaped stencil bank. FLOPs per evaluation
fall from $Rd + 5mR + MR = 7.9\times10^{5}$ to $(5m+M)d = 1.9\times10^{5}$, and three matmuls
become one. $S$ and $s_0$ are functions of the frozen weights and the frozen operators only,
so they are built once in setup and **charged to setup, never to a query**.

`lean` also rescales the residual. With $\iota = (1+\Delta t\,\nu\Lambda)^{-1}$ (constant per
query) and $\tilde p = \iota\odot p$ (constant per time step),

$$
r(z) \;=\; Ah \;-\; \tilde p \;+\; \tilde P_q^{\top}\mathrm{adv},
\qquad \tilde P_q = P_q\,\mathrm{diag}(\Delta t\,\iota),
$$

removing a division and three elementwise passes over $\mathbb{R}^M$ from every residual
evaluation. The same fold applies to the initial-condition residual
$R_{\mathrm{cold}}h - y = (R_{\mathrm{cold}}W_m^\top)f + R_{\mathrm{cold}}b_3 - y$.

### O7 `block` — block Gauss–Jordan for the damped $16\times16$ solve (reassociation)

`e.gj_solve` eliminates one unknown per stage: **16 sequential stages** on a $(16,17)$
augmented matrix, and the 1D study measured the analogous $8\times8$ solve at 28 µs/iteration
even after the cuSOLVER call was removed. Block size $b$ eliminates $b$ unknowns per stage:

$$
\mathrm{row}_k \leftarrow P_k^{-1}\,\mathrm{row}_k, \qquad
\mathrm{mat} \leftarrow \mathrm{mat} - \mathrm{mat}[:,\mathcal{B}_k]\,\mathrm{row}_k \ \ (\text{rows } \mathcal{B}_k \text{ masked}),
$$

with $P_k$ the $b\times b$ pivot block, inverted in closed form for $b=2$ and by one $2\times2$
block-inverse formula for $b=4$. Sequential depth drops from 16 to $16/b$. The matrix is the
SPD damped normal matrix the incumbent already eliminates without pivoting, so this changes
no algorithmic assumption. **Gate: $b=1$ must reproduce `e.gj_solve` bitwise.**

### O8 `nodot` — cuBLAS-free small matvecs (reassociation)

$J^\top J$, $J^\top r$, $\tilde P_q^\top\mathrm{adv}$ and the head's two hidden layers become
broadcast-reduce expressions, which fuse instead of dispatching a cuBLAS call per tiny shape.
The $S f$ matmul stays a dot: at $1344\times144$ it is no longer a tiny shape.

### O9 `leandec` — decode as one folded matmul (reassociation) — deliverable S2

The six dense outputs are $u_t = G\,h(z_t)$ with $G\in\mathbb{R}^{n\times R}$,
$n = (L-1)^2$. Folding as in O6, $G h = (GW_m^\top)f + G b_3$ with
$GW_m^\top\in\mathbb{R}^{n\times d}$: the decode streams $d/R = 144/512 = 0.28$ of the bytes
and does $0.28$ of the FLOPs, and the query no longer touches $G$ at all. At $L=1024$ that is
1.2 GB instead of 4.3 GB. The batched form $G[h(z_1)\dots h(z_6)]$ is checked against the
`vmap` the incumbent writes — the profile records which one XLA already emits.

### Non-parity variant (labelled, excluded from every parity claim) — deliverable S2

`f32out`: the six output fields cast to float32 at the end of the decode. The introduced
error is measured and stated beside it; it is **not** eligible for the headline factor.

## 4. Arms

| arm | composition | class |
|---|---|---|
| `incumbent` | `arms.make_query` verbatim, imported unmodified | reference |
| `o_fuse`, `o_hoist`, `o_share`, `o_unroll`, `o_probe`, `o_lean`, `o_block4`, `o_nodot`, `o_leandec` | each optimisation alone on the incumbent | isolated |
| `L1` = `fuse` | | cumulative |
| `L2` = `L1+hoist+share` | | cumulative |
| `L3` = `L2+unroll` | | cumulative |
| `L4` = `L3+lean` | | cumulative |
| `L5` = `L4+block4` | | cumulative |
| `L6` = `L5+probe` | | cumulative |
| `L7` = `L6+nodot+leandec` | the full port | cumulative |
| `f32out` | `L7` with float32 outputs | **non-parity, labelled** |
| `thru8` | `L7` (or the best parity arm) `vmap`ped over 8 queries | **throughput, labelled** |

Isolated arms attribute the gain; cumulative arms give the achieved factor. `o_block1` is a
correctness arm, not a speed arm: it asserts the block solve at $b=1$ is bitwise `gj_solve`.

## 5. FOM controls — same job, never a cross-job ratio

Interleaved with every ROM arm in the same randomized order, on the same GPU, in the same
process:

| control | preconditioner | ntol | ltol | dt |
|---|---|---|---|---|
| `fft_loose` | fft | 1e-2 | 0.5 | 0.005 |
| `fft_loose_dt01` | fft | 1e-2 | 0.5 | 0.01 |
| `fft_tight` | fft | 1e-6 | 1e-8 | 0.005 |

`fft_loose` and `fft_tight` are exactly the head ablation's two settings (15.356 ms and
88.729 ms at 256 in `abl01`); `fft_loose_dt01` is the `same_nt1e-2_dt01` equivalent. The
single-query ROM/FOM latency ratio is reported against each, honestly, with the FOM **not**
batched in the throughput arm.

## 6. S0 — the profile

Delivered before the ladder, from the same job:

1. **Phase timers.** `initial fit`, `evolution`, `decode` compiled and timed as separate
   programs with `block_until_ready` around each, plus the fused whole query and the host
   transfer. Both the phases and the whole query are reported; the phases are **not**
   obtained by subtracting one from another, and their sum is not claimed to equal the whole.
2. **Per-iteration marginal cost.** The evolution run with an unconditional fixed LM budget
   $\in \{1,2,4,8\}$, least-squares fit to `fixed + slope × iterations`, giving the per-step
   fixed overhead and the marginal LM iteration cost. Diagnostic only — a capped-budget arm
   is an algorithmic change and is never a production number.
3. **In-loop component microbenchmarks.** A `fori_loop` of $T$ iterations whose body is only:
   (a) a scalar add (loop-control floor); (b) the head $h(z)$; (c) the EQ gather+contraction
   $G_5 h$; (d) the quadrature matvec $P_q^\top\mathrm{adv}$; (e) the full residual;
   (f) residual+Jacobian; (g) `gj_solve` on a representative damped normal matrix;
   (h) `gj_block` at $b=2,4$. µs/iteration each, medians over retained repetitions.
4. **XLA cost analysis.** `jit(...).lower(...).compile().cost_analysis()` FLOPs and bytes for
   every phase and for the LM body, plus the fusion and custom-call census from the compiled
   HLO text.
5. **Launch-bound fraction.** Reported as: measured µs/iteration against
   (fusion count × the calibrated in-loop per-kernel cost from (3a)), and against the
   arithmetic and bandwidth roofline implied by the cost analysis. The claim made will be of
   the form "$X$ of the $Y$ µs per iteration is accounted for by kernel count", with the
   calibration shown.

## 7. Parity gates

For every arm, against the **in-job incumbent**, on every case and every repetition:

| gate | bar |
|---|---|
| `field_relative` | $\max_t \|u^{\mathrm{arm}}_t - u^{\mathrm{inc}}_t\| / \|u^{\mathrm{inc}}_t\| \le 10^{-12}$ |
| `latent_relative` | same bar over all 51 internal latents |
| `iterations_identical` | the 50-vector of per-step LM iterations is equal as integers |
| `ic_iterations_identical` | equal as integers |
| `reasons_identical` | the 50-vector of exit reasons and the IC reason are equal as integers |
| `bitwise` (declared-bitwise arms only) | output field sha256 equal |

Plus, once per mesh:

| gate | bar |
|---|---|
| `incumbent_reproduces_abl01` | the in-job incumbent reproduces `abl01`'s saved `a_neural_eq` fields at 256 to $10^{-12}$ relative |
| `block1_is_gj` | `gj_block(·,·,1)` equals `e.gj_solve` bitwise on 512 random damped SPD systems |
| `checkpoint_unchanged` | sha256 before == after == `18f0266a…` |
| `backend_gpu`, `x64`, `precision_highest` | asserted in-job, printed to the log |

An arm failing `field_relative`, `iterations_identical` or `reasons_identical` is **rejected**
and reported in the rejected table with its numbers; it cannot contribute to the headline.
The audit recomputes `field_relative` and `latent_relative` in **NumPy only**, from the saved
`.npz` fields, importing neither JAX nor the drivers.

## 8. Timing protocol

* One GPU, one job, every arm and every FOM control interleaved in a per-(rep, case)
  randomized order drawn from a recorded seed.
* `e.burn(0.25)` before every timed block.
* The input field is `device_put` and `block_until_ready`-ed **before** the GPU timer starts;
  `gpu_seconds` closes with `block_until_ready` on the outputs; `host_seconds` spans the
  device put through the host copy of the six fields. This is `abl01`'s contract verbatim.
* Medians over repetitions; **every repetition is retained** in `result.json`.
* No ratio is ever taken across jobs. The 47.649 ms `abl01` figure is used only as a
  provenance check, never as a denominator.
* 8 cases × 5 repetitions per mesh.

## 9. Cohort

`e.params_draw(7090702, 4)` followed by `e.params_draw(911702, 4)` — eight opened development
cases whose **first six are exactly `abl01`'s six**, so the `abl01` field gate is available
and the throughput arm has a genuine eight-case cohort. No final cohort is opened. No
training, no refitting: the checkpoint, the bank, the head and the EQ rule construction are
the retained ones.

## 10. Jobs (hard cap 3)

| job | attempt | content | GPU |
|---|---|---|---|
| 1 | `spd01` | 256²: S0 profile, full ladder, decode study, throughput, three FOM controls | `a100-80G`, exclude `pax007` |
| 2 | `fine01` | 1024²: incumbent vs the ladder's cumulative arms, decode study, throughput, three FOM controls | `a100-80G`, exclude `pax007` |
| 3 | `comp01` | 256² **and** 1024²: the composed arms `C1`–`C3` (the ladder without `block`), plus the incumbent, the `abl01` gate, three FOM controls and throughput — see deviation D4 | `a100-80G`, exclude `pax007` |

Jobs 1 and 2 are independent and are submitted **simultaneously from their own attempt
directories**; `squeue` is checked before and after each submission. 1024² is where the story
would live if there is one: the mesh ladder measured cached cost flat at 44 ms while interior
unknowns grow 264×, so all mesh growth is in decode and host transfer — exactly what O9
attacks.

S4 is explicitly optional and is only attempted if both jobs land clean and the third slot is
otherwise unused; its offline direction fit and per-rung NNLS cost ≈ 25 min inside the job.

## 11. Deviations register

Deviations from this document are recorded here as they happen, with their reason, before the
report is written.

### D1 (2026-09-16, before any cluster run) — the whole-query `bitwise` class is withdrawn

Section 3 declared `fuse`, `hoist`, `share`, `unroll` and `block` at $b=1$ **bitwise at the
whole-query level**, gated on output-field sha256 equality. The local smoke shows that bar is
not attainable and the declaration was wrong, for a reason that has nothing to do with the
optimisations: **the null arm already fails it.** `o_none` — the optimisation harness with
every switch off, emitting the same expressions as `arms.make_query` — reproduces the
incumbent to `4.883e-14` relative in the field and `3.947e-13` in the latent path, not to the
bit. Re-expressing the same arithmetic in a differently *structured* program changes XLA's
fusion decisions, and fused versus unfused f64 accumulation rounds differently. No amount of
care inside these arms removes that, because the arms are by construction differently
structured programs.

What replaces it:

* Every whole-query arm, `o_none` included, is in the **reassociation** class and is gated at
  $10^{-12}$ relative on fields and latents with **identical integer** iteration counts and
  exit reasons. That is exactly the bar the task states.
* `o_none`'s deviation is reported beside every arm as the **harness floor**: an arm at
  `5e-14` has changed nothing that the harness itself does not already change.
* `bitwise` survives where it is real and is verified at **component** level, inside one
  compiled program:
  * `gj_block(·,·,1)` equals `engines.gj_solve` **exactly** on 256 random damped SPD systems
    (`checks/smoke-solve.json`, 256/256 bitwise; $b=2,4,8$ agree to $7.4\times10^{-16}$, all
    tighter than `gj_solve`'s own $4.5\times10^{-16}$ deviation from `numpy.linalg.solve`).
  * `vmap(linearize-jvp)` equals `jacfwd` **exactly**, and the linearisation primal equals
    `fun(z)` **exactly**, on the real head inside one jit (`checks/smoke-jac.json`, 16/16).

One consequence is recorded in advance: a *measured* parity of order $5\times10^{-14}$ cannot
distinguish an arm that reassociates from one that does not, so the per-arm class in
`ladders.py` is documentation of intent, not a claim about the measurement.

### D4 (2026-09-16, after `spd01`'s profile, before its ladder was read) — the third job goes to composed arms, not to S4

Section 10 left job 3 as a reserve for a repair or for S4 (the cheap-corrections $q=16$ EQ
arm). It went to neither. `spd01`'s in-loop microbenchmarks, read as soon as the profile was
written, measure the block Gauss–Jordan solve as a **loss** on this device: 26.4 µs per
in-loop iteration at $b=1$ against 64.7, 93.3 and 103.2 µs at $b=2,4,8$, because the block
form trades 16 sequential elimination stages for 41–67 compiled fusions in a program whose
cost tracks fusion count. The pre-registered cumulative ladder puts `block4` at `L5`, so
`L5`, `L6` and `L7` all inherit that loss and the declared "full port" arm is not the fastest
composition — through no fault of the optimisations that come after it.

Job 3 (`comp01`) therefore runs `C1` = `L4`+`probe`, `C2` = `C1`+`nodot`, `C3` = `C2`+folded
decode — the same ladder with `block` left out — at **both** 256 and 1024 intervals, with its
own incumbent, its own three full-order controls, its own `abl01` gate and its own throughput
arms, so nothing in it is a cross-job ratio. These arms are **composed after seeing the
isolated measurements** and are labelled as post-hoc in every table; they are never folded
into the pre-registered ladder, and the pre-registered ladder's regression is reported as
measured.

S4 is **not run**. Porting the optimisations into `varpro.py`'s block-damped $q=16$ solver is
a substantial change to a solver this cell did not write, its offline direction fit and
per-rung NNLS cost about 25 minutes inside the job, and doing that unattended with the last
job slot risks the deliverable for a marginal extension. It stays open.

### D3 (2026-09-16, during the run) — the abl01 field gate covers four cases, not six

Section 9 claimed the cohort's first six cases are "exactly `abl01`'s six". They are not,
and the job's own gate printed `2.33` relative, which is the finding rather than a failure
of the solver. `engines.params_draw` draws **column by column**:
`r.uniform(.15,.85,count)` for every case's first parameter, then the second, and so on. So
drawing 4 cases from a seed does **not** extend a draw of 2 from the same seed — only the
first column agrees. `abl01`'s two "fresh development" cases came from
`params_draw(911702, 2)`; this cohort's four came from `params_draw(911702, 4)`, so they are
different physical cases and comparing them against `abl01`'s saved fields is comparing
different problems.

Cases 0–3 come from `params_draw(7090702, 4)` in both and are bit-identical parameter
vectors, so the gate is **those four complete trajectories at 256 intervals against the
retained `abl01` `a_neural_eq` fields**, at the same $10^{-12}$ bar. The audit restricts the
gate to them and reports the excluded cases explicitly; the job's own six-case number is
recorded but is not the gate. The cohort itself is unaffected — eight development cases from
the two campaign seeds, no final cohort opened.

This is worth carrying beyond this cell: any experiment that reuses a `params_draw` seed with
a different `count` gets a different cohort, silently.

### D2 (2026-09-16, before any cluster run) — matvec orientation fixed in the harness

The first harness draft wrote the quadrature contraction as `adv @ Pq` where the incumbent
writes `Pq.T @ adv`. Mathematically identical, but XLA emits a different contraction, which
inflated the harness floor. The dot path now writes `W.T @ x` exactly as the incumbent does.
No arm's definition changed.

## 12. Deliverable

`reports/2026-09-16-b-speed.md`, generated from `result.json` by a script beside it — **no
number is hand-typed**. It carries the profile table, the per-optimisation gain at parity, the
cumulative ladder, the rejected arms, the ratios against each FOM control, the throughput
numbers, a mermaid diagram of the query with `classDef` colouring for hoisted / fused /
unchanged stages, the gate table, and a plain-language glossary.
