# Per-query head refinement as an inference-time knob — predeclared design

One question: **is per-query refinement of the frozen head's weights a usable inference-time
accuracy/cost knob?** Everything except the refinement is the head-ablation arm (a) contract:
the frozen bank $G$, the latent dimension $K=16$, the weak objective, the test-mode family,
the time discretization, the initializer policy, the Levenberg–Marquardt stopping rule and the
output contract are untouched. The only online knob is the number $n$ of gradient steps taken
on the head parameters $\theta$ at query time, with an anchor weight $\mu$ pinning $\theta$ to
the trained weights $\theta_0$.

This is the sibling of the fixed-weight correction ladder (`experiments/head-ablation/ladder.py`,
job `3713867`), which added $q$ *fixed linear bank directions* and found the error monotone in
$q$ but the upper rungs not converged and no rung dominating an efficient full-order solve.
Here the extra capacity is not new directions but a **moved manifold**: the same $K$ unknowns,
a different $h$.

## The parameterisation and the parameter metric

The separable decoder is $u(x;z) = g(x)^\top h_\theta(z)$ with

$$h_\theta(z) = \mathrm{MLP}_{h}(z) + z^\top W_{\rm lin}.$$

$\theta$ is **exactly** the head's own weights — the three $\mathrm{MLP}_h$ weight/bias pairs and
$W_{\rm lin}$ — flattened by `jax.flatten_util.ravel_pytree` into one vector of
$P$ real numbers. The spatial track $(B, g, \texttt{out\_scale})$, and therefore the bank $G$,
the empirical-quadrature rule and the cold-start candidate table, are **never** touched: the
bank stays frozen exactly as the ablation requires, and the quadrature rule and initializer
remain the offline artifacts fitted at $\theta_0$.

**The parameter metric is the plain Euclidean norm on that flat vector**, made dimensionless by
$\|\theta_0\|_2$:

$$\Omega(\theta) = \frac{\|\theta-\theta_0\|_2^2}{\|\theta_0\|_2^2}.$$

We report the *relative drift* $\sqrt{\Omega(\theta)}$ for every case so a reader can see how
far the anchor actually let the head move.

## V1 — INITIAL-ONLY

Refine once against the supplied initial field $u_0$, then evolve with the refined head and the
**unchanged** solver. The objective is a field fit restricted to the same sampled node set the
initializer uses — the fixed $48\times48$ Gauss product rule, with the same weights $w$ — so the
refinement sees exactly what the cold start sees and no more:

$$F_1(z,\theta) = \frac{\big\|\,\mathrm{R}\,h_\theta(z) - y\,\big\|_2^2}{\|u_0^{(w)}\|_2^2}
  \;+\; \mu\,\Omega(\theta),$$

where $G^{(w)} = QR$ is the thin QR of the weighted bank at the Gauss nodes,
$u_0^{(w)}$ is the weighted sampled initial field and $y = Q^\top u_0^{(w)}$. Because
$\|G^{(w)}\delta\|_2 = \|\mathrm R\,\delta\|_2$, the first term is the weighted field-fit
residual on those nodes up to the constant orthogonal part — it is the initializer's own
objective, which is why this is the right thing to attack: `abl01` recorded a **1.87 % worst
initial-compression loss**, comparable to the whole trajectory error.

The algorithm is alternating minimisation, $n$ rounds:

1. $z \leftarrow \mathrm{LM}_{\rm ic}(z;\theta)$ — the retained initializer solve (nearest
   candidate, `ic_budget` 400, gtol $10^{-6}$), run once before the loop exactly as arm (a).
2. repeat $n$ times: one Adam step on $\nabla_\theta F_1(z,\theta)$; then re-solve
   $z \leftarrow \mathrm{LM}_{\rm ic}(z;\theta)$ warm-started at the current $z$, under the
   identical rule.

Then the trajectory is evolved with $h_\theta$ held fixed at the refined $\theta$.

## V2 — PER-STEP

Refine after every time step, against that step's own weak residual. With
$r_w(z;\theta,p)$ the arm (a) weak residual and $p = A\,h(z^{\rm prev})$ the projection of the
**previously accepted field** (so $p$ is carried forward with the $\theta$ that produced it, not
recomputed under a newer $\theta$):

$$F_2(\theta) = \frac{\big\|\,r_w(z;\theta,p)\,\big\|_2^2}{\|p\|_2^2} + \mu\,\Omega(\theta).$$

Per time step: the arm (a) extrapolated start, the arm (a) LM solve for $z$, then $n$ Adam steps
on $\nabla_\theta F_2$, then **one re-solve of $z$ under the identical LM rule**. $\theta$ is
carried forward along the trajectory and is always anchored to $\theta_0$ (not to the previous
step's $\theta$). Both the first-solve and the re-solve iteration counts and exit reasons are
recorded for every step.

## $n = 0$

At $n=0$ the refinement block is omitted at the Python level, so there is no extra LM re-solve to
perturb a converged point, and V1 and V2 and both $\mu$ coincide: **one** arm, `n0`. It differs
from head-ablation arm (a) only in that $\theta$ is a traced runtime operand of the compiled
query instead of a compile-time constant. Gates (i)–(iii) measure exactly that difference.

## Optimizer and step size

Plain **Adam**, implemented inline (no optax dependency in the timed path):
$\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$, bias correction, moments initialised at
zero at the start of **each query**. $\mu$ and the step size $\alpha$ are runtime operands, not
compile-time constants, so one compiled query serves both anchor weights.

**$\alpha$ is chosen once, in-job, on ONE calibration case that is not an evaluation case:**
the first draw of the training family (`params_draw(train_seed)[0]` on Burgers,
`source_params(train_seed)[0]` on Poisson). The rule, fixed before the run:

> Sweep $\alpha \in \{10^{-8}, 3\cdot10^{-8}, 10^{-7}, 3\cdot10^{-7}, 10^{-6}, 3\cdot10^{-6}, 10^{-5}, 3\cdot10^{-5}, 10^{-4}\}$,
> run V1 with $n=8$ at $\mu=\mu_{\rm loose}$ on the calibration case, and take the $\alpha$ with
> the smallest **data term** (the relative field-fit residual on the Gauss nodes; on Poisson, the
> relative weak residual). Ties within $10^{-3}$ relative are broken by the smaller $\alpha$.

The selected $\alpha$ is then **frozen** and used for both variants, both anchor weights, every
$n$ and every evaluation case. It is never re-tuned per case. Each PDE gets its own $\alpha$
because the objectives have different units; that is one calibration per PDE, recorded.

## Anchor weights

Pre-registered, fixed before the run: $\mu_{\rm loose} = 10^{2}$, $\mu_{\rm tight} = 10^{5}$.
A separate **diagnostic** sweep of the drift at $n=32$ over
$\mu \in \{1,10^{2},10^{3},10^{4},10^{5},10^{6}\}$ is run on the calibration case only, to show
where the anchor starts to bind. It is diagnostic: it does **not** change the two swept values.
If both swept $\mu$ turn out to lie on the same side of the binding point, that is reported as a
limitation, not corrected after the fact.

## The sweep

**Burgers 2D**, frozen checkpoint `sep_hfit_dense_mid_N256_dense.pkl`
(SHA256 `18f0266ae6f0…`), $K=16$, $R=512$, 256 intervals, $\Delta t = 0.005$, $M=4K=64$ weak
modes, the retained nonnegative empirical-quadrature rule at $m=4M=256$ points fitted from the
$\theta_0$ decoder outputs with the arm (a) seed and the arm (a) full code table, the same six
opened development cases `abl01` used, and the same refined reference ($L=4096$,
$\Delta t=3.125\times10^{-4}$, restricted to the working mesh).

| arm | variant | $n$ | $\mu$ | quadrature |
| --- | --- | ---: | ---: | --- |
| `n0` | baseline | 0 | — | eq |
| `v1_n{n}_mu{loose,tight}` | initial-only | 1,2,4,8,16,32 | $10^{-3}$, $10$ | eq |
| `v2_n{n}_mu{loose,tight}` | per-step | 1,2,4,8,16,32 | $10^{-3}$, $10$ | eq |
| `n0_dense`, `v1_n8_mutight_dense` | dense control pair | 0, 8 | — , $10$ | dense |

27 reduced arms. Efficient full-order controls `fft_loose` and `fft_tight` are interleaved in the
same job. The dense control is a **pair** at the two ends of the V1 ladder rather than a single
arm, because one dense row alone cannot be differenced against anything; that is a recorded
deviation from the single-control instruction and it costs two extra arms.

**Poisson 2D**, the `pabl01` frozen bank/head (`r128_joint.pkl`, $K=16$, $R=128$), 1024
intervals, the same twelve development sources. The Poisson query is a **single static solve**,
so V2 has no per-step structure to exploit and collapses onto V1; only V1 is run and this is
stated in the report. The Poisson V1 objective is the weak residual itself against the
source-projected data, $F_1(z,\theta) = \|B h_\theta(z) - f_m\|^2/\|f_m\|^2 + \mu\Omega(\theta)$,
which is simultaneously the initial fit and the solve. Arms: `n0` plus
`v1_n{1,2,4,8,16,32}_mu{loose,tight}` = 13, with `dst_direct` interleaved.

**Timing.** Three timed repetitions, GPU burn-in before every timed block, randomised subject
order, all repetitions retained, the complete device query as `abl01` defined it — supplied dense
initial field on the GPU to the dense output fields — **with the refinement steps inside the
timed interval**. Final cohorts stay sealed; no new case is opened.

## Reported per arm (the three layers)

1. **Bank projection floor** — the best any coefficients at all could do in the frozen bank, per
   output time; unchanged by refinement, so it is a floor for every arm.
2. **Best-found reconstruction with the REFINED head** — the best that arm's own *moved* manifold
   can do on the reference field, with no PDE involved: a seeded multistart LM fit on
   $\{G h_\theta(z)\}$ using the $\theta$ that arm actually produced on that case, at the $\theta$
   in force at each output time for V2.
3. **Solved worst same-grid error** — the real online solve against the converged same-mesh
   full-order solve (`fft_tight` on Burgers, `dst_direct` on Poisson), which isolates reduction
   error from this mesh's discretisation error. The refined-reference error is also reported.

Plus: exit reason and iteration count for **every** latent solve (initial, every V1 re-solve,
every V2 first-solve and re-solve), the relative drift $\sqrt{\Omega}$, the ratio of anchor to
data gradient norm at each refinement step, and the error **per output time** so that
overfitting the initial field can be seen directly.

## Gates before any verdict

- **(i) local smoke.** $n=0$ through the new code path reproduces the consolidated saved Burgers
  case and the incumbent `accuracy_paths.make_rom` to $\le 10^{-12}$ relative, reusing the
  head-ablation fixtures that `smoke_arms.py` uses.
- **(ii) in-job Burgers.** `n0` reproduces `abl01`'s `a_neural_eq` on all six cases to
  $\le 10^{-9}$ relative.
- **(iii) in-job Poisson.** `n0` reproduces `pabl01`'s `a_neural` at 1024 intervals to
  $\le 10^{-6}$ relative — the declared cross-job tolerance, matching the precedent set by the
  correction ladder's cross-job gate, because the two jobs may land on different A100 models
  where XLA can select different kernels. The achieved value and the bitwise-identical count are
  both reported.
- **(iv) independent audit.** A NumPy-only audit that imports neither driver nor JAX recomputes
  every reported error from the saved fields.
- **(v) re-decode.** The refined weights are saved per (arm, case), together with a fixed
  4096-node sample of the frozen bank, so the audit can recompute $G h_\theta(z)$ in pure NumPy
  at those nodes and compare it against the saved output field.

## Pre-registered acceptance — when is $n$ "a knob"?

All three must hold, per variant:

1. **Monotone**: the worst same-grid error is non-increasing along $n = 0,1,2,4,8,16,32$.
2. **A real frontier**: at least three non-dominated points spanning $\ge 2\times$ in median GPU
   cost and $\ge 2\times$ in worst same-grid error.
3. **All converged**: none of the points on that frontier is early-stopped — no iteration-budget
   exit and no rejected-step exit in any latent solve, initial or per-step, first-solve or
   re-solve. An early-stopped arm keeps that status and is never relabelled stationary.

## What would falsify it

Any of: the error not monotone in $n$; the frontier collapsing to fewer than three non-dominated
points or spanning less than a factor two on either axis; the refinement driving latent solves
into budget exits; the error at $t=0$ falling while the error at $t=0.25$ rises (V1 overfitting
the initial field — the explicit held-out generalisation risk); or the whole ladder sitting above
the bank projection floor by a margin that refinement does not move, which would mean the head is
not what binds.

A negative outcome is reported as the result. This cell does not attempt to rescue it.

## Routine calls made without asking, recorded here

- The dense control is a pair rather than a single arm (see above).
- The calibration case is drawn from the **training** family, so no new evaluation case is
  opened and no evaluation case is used for tuning.
- Adam is hand-rolled rather than taken from optax, so the timed path has no version-dependent
  behaviour; the update rule is the standard one and is stated above.
- $\mu$ and $\alpha$ are runtime operands, so a single compiled query serves both anchor weights.
- Snapshot generation is omitted: nothing in this cell needs training snapshots — the bank
  projection floor comes from the reference fields, and there is no basis to fit.
- The Burgers reference, cohort, EQ rule, trust radius, budgets and tolerance are copied verbatim
  from the `abl01` arm (a) recipe so gate (ii) is meaningful.

## Amendment, 2026-09-15, before any cluster submission

Two numbers in this document were changed after the local fidelity smoke
(`smoke_refine.py`, record `checks/smoke-refine.json`) and before any evaluation run.
Both changes are **scale** corrections; no accuracy number, no error against any
reference and no evaluation-case outcome entered either choice. The smoke reports only
the fidelity gate, the drift, the objective terms and the gradient norms.

1. **Anchor weights** $\mu_{\rm loose}, \mu_{\rm tight}$ moved from $\{10^{-3}, 10\}$ to
   $\{10^{2}, 10^{5}\}$. The smoke measured the anchor gradient norm at
   $\|\nabla_\theta\Omega\| \approx 3\times10^{-6}$ against a data-term gradient norm of
   $6\times10^{-2}$ to $1$: with $\|\theta_0\|_2 = 209.2$ and $P = 542{,}208$, the anchor
   gradient is $2\sqrt\Omega/\|\theta_0\|$, so any $\mu \lesssim 10^{3}$ is numerically
   indistinguishable from no anchor at all. At $\mu=10$ the measured drift differed from
   $\mu=10^{-3}$ by 0.2 %, which would have made the "loose versus tight" contrast vacuous.
   The new pair brackets the binding point: $\mu=10^{2}$ leaves the anchor gradient roughly
   $10^{-3}$ of the data gradient, $\mu=10^{5}$ makes them comparable.
2. **The step-size grid** was widened downward, from $[10^{-6}, 10^{-3}]$ to
   $[10^{-8}, 10^{-4}]$. Adam's update has magnitude $\approx\alpha$ per coordinate, so the
   drift per step is $\alpha\sqrt P/\|\theta_0\| \approx 3.5\alpha$; the smoke confirmed that
   at $\alpha = 10^{-4}$ a single step *increases* the field-fit objective (from
   $\approx 4\times10^{-4}$ to $8\times10^{-3}$), i.e. it steps outside the region where the
   linearisation holds. The original grid's lower end was therefore its only usable part.

The selection rule, the sweep, the gates and the acceptance criteria are unchanged.
