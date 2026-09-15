# The prior dial — predeclared design

One question: **on one frozen checkpoint, is "trust in the neural prior" a usable
inference-time accuracy/cost knob?**

The head-ablation cell established that the nonlinear head earns its place at matched
latent dimension, and the correction ladder established that *bank enrichment* buys
accuracy monotonically but only at a large cost and without converging above $q=16$.
Both of those change the *parameterization*. This cell changes nothing about the
parameterization: it keeps the full bank coefficient vector free at every setting and
moves only **how hard the solver is pulled back towards the head's own prediction**.

Everything else is the head-ablation arm (a) contract: every network weight, the bank
$G$, the latent dimension $K=16$, the initializer policy, $\Delta t$, the stopping rule
and the output contract.

## The objective

Arm (a) solves the weak residual for $z$ alone, with the bank coefficients pinned to the
head:

$$\min_{z\in\mathbb R^{K}}\ \big\|r_w\big(h_\theta(z)\big)\big\|_2^2 .$$

This cell solves for the **full** bank coefficient vector $c\in\mathbb R^{R}$ and adds a
penalty pulling it towards the head:

$$\boxed{\ \min_{z\in\mathbb R^{K},\,c\in\mathbb R^{R}}\ \big\|r_w(c)\big\|_2^2\;+\;\lambda\,\big\|R_G\,(c-h_\theta(z))\big\|_2^2\ }$$

$r_w$ is the **same** weak residual arm (a) uses, evaluated at $u=Gc$: exact linear terms
through $A=\Phi^\top G$, and the advection term through either the retained nonnegative
empirical-quadrature (EQ) rule or the exact dense grid sum, exactly as `arms.py` does for
the free-bank arm (d).

$G=Q_GR_G$ is the thin QR of the bank already used in `arms.py`, so
$\|G\delta\|_2=\|R_G\delta\|_2$ and the penalty is the **squared field-norm distance**
between the solved state and the state the head would have produced. The penalty is a
field metric, not a coefficient metric.

### Limits, by construction

- $\lambda\to\infty$ forces $c=h_\theta(z)$ and reproduces **arm (a)** exactly.
- $\lambda\to 0$ with $M\ge R$ reproduces the **free-bank arm (d)** exactly.
- $\lambda\to 0$ with $M<R$ is **underdetermined and is not the free bank.** On Burgers
  $R=512$, so the $M=64$ and $M=256$ settings have fewer weak equations than bank
  coefficients and the small-$\lambda$ end of those two sweeps is a *regularized
  underdetermined* solve whose answer is set jointly by $\lambda$ and by the
  Levenberg–Marquardt damping. This is stated at every such row of the report and is
  **not** presented as arm (d). Only $M=1024>R$ on Burgers, and $M=257>R=128$ on Poisson,
  reach arm (d) as $\lambda\to 0$.

### Parameterization actually solved

Write $y=R_G\,(c-h_\theta(z))$, so that

$$u = Gc = G\,h_\theta(z) + Q_G\,y,\qquad \|R_G(c-h_\theta(z))\|_2=\|y\|_2 .$$

Every online product is then formed in the split form ($G h_\theta(z)+Q_Gy$,
$A h_\theta(z)+(AR_G^{-1})y$, and the same for the 5-point EQ stencil), which is the
well-conditioned way to write it: $R_G^{-1}$ is never applied to a solved vector. The
solved vector is $w=(z,y)\in\mathbb R^{K+R}$ and the residual handed to the solver is

$$F(w) = \begin{bmatrix} r_w\big(h_\theta(z)+R_G^{-1}y\big) \\[2pt] \sqrt{\lambda}\;y \end{bmatrix}\in\mathbb R^{M+R}.$$

At $\lambda=\infty$ the $y$ block has **zero width**: $F$ is then literally arm (a)'s
residual, in the same function, which is what makes gate (i) a bitwise statement rather
than a tolerance.

### The scaling of $\lambda$ — declared

$\lambda$ carries units (residual$^2$ per field-norm$^2$), so a bare number is
meaningless. The exact linear part of $\partial r_w/\partial y$ is **state independent**:
the $1/(1+\Delta t\,\nu\lambda_j)$ row scaling of the weak residual cancels the
$\Delta t\,\nu\lambda_j$ diffusion term exactly, leaving $A R_G^{-1}=\Phi^\top Q_G$.
So we set

$$\lambda \;=\; \lambda_{\rm rel}\,\sigma^2,\qquad
\sigma \;=\; \big\|A\,R_G^{-1}\big\|_2 \;=\; \big\|\Phi^\top Q_G\big\|_2 ,$$

the largest singular value of the exact linear residual response to a unit field-norm
departure from the head. On Burgers, $\Phi$ and $Q_G$ both have orthonormal columns, so
$\sigma$ is the **largest principal cosine between the test-mode span and the bank span**
and $\sigma\le 1$; it is a pure geometric property of the frozen bank and the test family
and is recorded per arm. On Poisson the same definition is used with $B$ (the retained
scaled sine-product reduced operator) in place of $A$.

$\lambda_{\rm rel}=1$ therefore means "one unit of field-norm departure from the head is
penalized as strongly as the strongest linear response of the weak residual to it".
$\sigma$ is reported for every arm so that any reader can convert $\lambda_{\rm rel}$ back
to $\lambda$.

### Poisson: variable projection with a closed form

The Poisson weak residual is exactly $r(c)=Bc-f_m$, linear in $c$. For fixed $z$ the
minimizer in $y$ is closed form,

$$\big(B_y^\top B_y+\lambda I\big)\,y \;=\; B_y^\top\big(f_m-B\,h_\theta(z)\big),\qquad B_y=B\,R_G^{-1},$$

equivalently $(B^\top B+\lambda R_G^\top R_G)c = B^\top f_m+\lambda R_G^\top R_G h_\theta(z)$
in the stated $c$ form. It is evaluated from one precomputed eigendecomposition of
$B_y^\top B_y$ ($R\times R$, $R=128$) built at setup, so the whole $\lambda$ sweep shares
one factorization and the inner solve is an $O(R^2)$ matvec chain inside the timed query.
The outer Levenberg–Marquardt runs in $z$ only and differentiates **through** the closed
form (Golub–Pereyra), so the reported solved dimension on Poisson is $K$, not $K+R$. On
Burgers the advection term makes $r_w$ quadratic in $c$, $y$ cannot be eliminated, and the
whole $w=(z,y)$ vector is solved by the same damped LM.

## Sweep grid

$\lambda_{\rm rel}\in\{\infty,\,10^{3},\,10^{2},\,10^{1},\,10^{0},\,10^{-1},\,10^{-2},\,10^{-3}\}$ —
eight settings, seven of them finite and log-spaced over six decades, plus the exact
$\lambda=\infty$ elimination. $\lambda_{\rm rel}=10^{3}$ is the numerically
"effectively infinite" end of the *finite* code path and exists to show that path
converging to arm (a).

**Pre-registered amendment rule.** If a local development probe on a coarse mesh shows the
entire transition lying outside this grid, the grid may be shifted or widened **once**, and
the probe, the old grid and the new grid are recorded in this file before the cluster job
is submitted. Nothing else about the protocol may be changed after submission.

### Burgers, 256 intervals, frozen `sep_hfit_dense_mid_N256_dense.pkl` (SHA256 `18f0266ae6f0…`)

| block | $M$ | quadrature | $\lambda_{\rm rel}$ values | why |
| --- | ---: | --- | --- | --- |
| primary | 64 | EQ, $m=4M=256$ | all 8 | $M=4K=64$ and the EQ rule are arm (a)'s own setting |
| test count | 256 | dense | all 8 | still $M<R$: underdetermined at small $\lambda$ |
| test count | 1024 | dense | all 8 | $M>R$: the only Burgers block that reaches arm (d) |
| quadrature control | 64 | dense | $\infty,10^{1},10^{-1},10^{-3}$ | paired with the primary block: isolates the EQ rule's validity off the head manifold |
| trust control | 64 | EQ | $10^{0},10^{-3}$ | correction block released from the latent trust radius (below) |

The six **opened development cases** are exactly the ones arm (a) and the ladder used
(`params_draw(7090702, 4)` then `params_draw(911702, 2)`). No new case is opened. The
refined reference is the campaign's own $L=4096$, $\Delta t=3.125\times10^{-4}$ solve.

Same-job efficient FOM controls, interleaved in the same randomized order:
`fft_loose` ($n_{\rm tol}=10^{-2}$, $l_{\rm tol}=0.5$) and `fft_tight`
($10^{-6}$, $10^{-8}$), the head-ablation names.

### Poisson, frozen `pabl01` bank/head ($K=16$, $R=128$, retained $M=257$)

$\lambda_{\rm rel}$ over the same eight values at **1024 intervals** and, because a Poisson
query costs ~2.5 ms, also at **64 intervals**. The twelve development sources are exactly
pabl01's (`source_params(7090703, 6)` then `source_params(7090732, 6)`). Same-job FOM
control: `dst_direct`.

### Timing protocol

Three timed repetitions, GPU burn-in before **every** timed block, all repetition arrays
retained, randomized subject order, complete device-query timing exactly as `abl01`
defined it (supplied dense field on the GPU to the dense output fields; the host column
adds the same-invocation transfers).

## Recorded deviations, decided in advance

1. **The trust radius applies to the latent block only.** Arm (a)'s trust radius is
   $0.01\times$ the radius of the training code cloud, $0.05256511659045252$ — a *latent
   space* quantity. The correction $y$ is in **field-norm** units, where a departure of
   order one is exactly what a percent-level correction needs, so carrying the same
   number over to $y$ would cap the field correction at $0.0526$ and would silently
   regularize the small-$\lambda$ end with something that is not $\lambda$. The primary
   sweep therefore bounds $\|\delta z\|\le$ trust and leaves $\|\delta y\|$ free, and the
   **trust control** rows repeat two $\lambda$ values with $\|\delta y\|\le$ trust as
   well, so the size of this choice is measured rather than assumed. At $\lambda=\infty$
   there is no $y$ block and the two are identical.
2. **The residual tolerance now includes the penalty.** The LM stops on $\|F\|$, and $F$
   carries the $\sqrt\lambda\,y$ block, so the absolute residual exit ($10^{-9}\times$
   scale) is harder to reach at large $\lambda$ than it is for arm (a) at the same $z$.
   The stationarity and small-step exits are unaffected. Every row reports its exit-reason
   histogram.
3. **Arms above 64 unknowns use a pivoted dense step solve** instead of the incumbent
   unrolled Gauss–Jordan, exactly as the ladder did. More accurate, not weaker.
4. **EQ is fitted only at $M=64$** ($m=256$), which is arm (a)'s own rule, refit with the
   identical seed, candidate cap, fit-state count and code table so it is bitwise the same
   rule. At $M=256$ and $M=1024$ an $m=4M$ nonnegative-least-squares fit is not
   constructible inside the job budget, so those blocks use the exact dense grid sum. The
   EQ rule is fitted on **decoder-output** advection snapshots and is therefore
   extrapolating once $c$ leaves the head manifold; the paired dense block at $M=64$ is
   what measures that.
5. **Layers 1 and 2 of the error decomposition coincide at every finite $\lambda$.** The
   reachable set is the whole bank for any finite $\lambda$ — the penalty is a solver-side
   prior, not a restriction of the manifold — so the best-found reconstruction of a finite
   $\lambda$ arm *is* the bank projection floor. The head-manifold best-found is reported
   as the $\lambda=\infty$ row. Consequently, whatever $\lambda$ does, it does entirely in
   the reduction/solver layer. Both layers are still printed for every setting.

## Gates before any verdict

- **(i) local smoke.** $\lambda=\infty$ **through the new code path** reproduces the
  consolidated saved Burgers case and the incumbent `accuracy_paths.make_rom` to
  $\le 10^{-12}$ relative, reusing `smoke_arms.py`'s fixture and comparison.
  Additionally the *finite* path at $\lambda_{\rm rel}=10^{6}$ is compared with the
  $\lambda=\infty$ path, and $\sigma$, $\operatorname{cond}(R_G)$ and the field-norm scale
  are recorded.
- **(ii) in job, Burgers.** $\lambda=\infty$ at $M=64$ with EQ reproduces `abl01`'s
  `a_neural_eq` on all 6 cases to $\le 10^{-9}$ relative.
- **(iii) in job, Poisson.** $\lambda=\infty$ reproduces `pabl01`'s `a_neural` on all 12
  sources at each retained mesh to $\le 10^{-9}$ relative, and $\lambda_{\rm rel}=10^{-3}$
  is compared with `pabl01`'s `d_freebank` as a soft (reported, not required) check of the
  other limit.
- **(iv) independent audit.** A NumPy-only audit (no JAX, no GPU) recomputes every
  reported error from the saved output fields, recomputes the same-grid discrepancy
  against the same-job converged FOM, and re-derives every table row.

## Acceptance criteria, pre-registered

$\lambda$ is called **a knob** only if all three hold on the primary Burgers block:

1. **Monotone.** Worst same-grid error is monotone non-increasing as $\lambda$ decreases.
2. **A real frontier.** At least **3 non-dominated** points (Pareto-optimal in
   worst same-grid error vs median complete-query GPU ms) spanning $\ge 2\times$ in cost
   **and** $\ge 2\times$ in error.
3. **Honest.** **None** of those points is early-stopped: every one completes under the
   shared stopping rule, with no iteration-budget exit and no rejected-step exit anywhere.

### What would falsify it

- Error not monotone in $\lambda$ (the weak objective is minimized, the field error is
  not).
- The frontier collapsing to one or two useful points — e.g. cost flat in $\lambda$ (then
  it is an accuracy lever, not a knob, and the cheapest good $\lambda$ simply wins), or
  error flat in $\lambda$ (then the prior was never binding).
- The accuracy only appearing at $\lambda$ values whose solves do not converge, which is
  how the correction ladder failed.
- The whole span being inside the gap between `fft_loose` and `fft_tight`, i.e. no point
  that is not already dominated by the same-job efficient FOM. The FOM rows are context,
  not competitors, and are reported as such.

Whichever way it falls is reported.

## Reported per setting

Bank projection floor, best-found reconstruction, solved worst and median same-grid error,
worst reference error, the prior departure $\|y\|/\|u\|$ actually realised, LM iteration
counts, exit-reason histograms, stopping status, $\sigma$, and same-job complete-query
GPU/host cost with burn-in and every repetition retained. Development cases only; the
final cohort stays sealed.

---

## Amendments (append only)

*(2026-09-15, before submission — see "pre-registered amendment rule" above.)*

**Amendment 1 — routine calls made without asking, recorded here as required.**

- Burgers uses one attempt directory `pdial01` and Poisson one attempt directory
  `ppdial01`, in the namespace `/cluster/tufts/paralab/tawal01/prior_dial_20260915/`,
  submitted as two independent jobs so they run concurrently on separate nodes.
- The correction-ladder job's 128-trajectory FOM snapshot generation and its offline
  residual-direction POD are **both dropped**: the correction directions here are the
  whole bank in the field metric ($C=R_G^{-1}$), which is fixed by the checkpoint and
  needs no data. This removes the ladder's dominant setup cost (1757.6 s) and no reported
  quantity depends on it.
- A100 requested on the `gpu` partition, `--time 04:00:00`, `--mem 150G` for Burgers and
  `--mem 100G` for Poisson.

**Amendment 2 — the $\lambda_{\rm rel}$ grid, after the local probe.**

Recorded after the probe ran; see `checks/smoke-prior.json` for the measured quantities.
