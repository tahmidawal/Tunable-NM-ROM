# Training the Burgers head — predeclared design

**One question.** Can *training alone*, at the same separable architecture, move the Burgers
head's best-found reconstruction from the incumbent's **2.5447 %** worst toward the frozen
bank's **0.3918 %** projection floor at 256 intervals — and if it does, does the online solve
follow it down?

Nothing about the architecture changes. The bank is $g(x)$, a random-Fourier-feature MLP; the
head is $h_\theta(z)$, an MLP with a linear skip; the codes are auto-decoder codes. No
mixtures, no FiLM, no new feature families. What is allowed to move is the **training data
amount**, the **training objective**, the **latent dimension $K$** and the **bank rank $R$**.

Everything downstream of the checkpoint is frozen and unchanged: every trained checkpoint is
hashed, frozen, and evaluated through the *unchanged* head-ablation arm (a) machinery
(`experiments/head-ablation/arms.py`), with the empirical-quadrature rule refit on that
checkpoint's own decoder-output advection snapshots, the same fixed-Gauss cold initializer,
the same trust radius rule, the same Levenberg–Marquardt stopping rule and the same output
contract.

---

## 1. The model, and the exact identity the training uses

At $L=256$ intervals the interior state is $u\in\mathbb R^{n}$, $n=(L-1)^2=65025$. The
decoder is separable,

$$u(x;z) = \mathrm{bc}(x)\,g(x)^\top h_\theta(z),\qquad
g:\mathbb R^2\to\mathbb R^{R},\quad h_\theta:\mathbb R^{K}\to\mathbb R^{R},$$

so on any fixed point set the bank collapses to a cached matrix $G\in\mathbb R^{n\times R}$ and
the decoder is $u = G\,h_\theta(z)$.

With $G$ **frozen**, write the Gram $\Gamma=G^\top G=\Lambda\Lambda^\top$ (Cholesky,
$\Lambda$ lower triangular), and for a snapshot $u_i$ let

$$c^\ast_i=\arg\min_{c}\|Gc-u_i\|_2,\qquad f_i=\|Gc^\ast_i-u_i\|_2 .$$

Then for **every** $h$,

$$\|G h-u_i\|_2^2=(h-c^\ast_i)^\top\Gamma(h-c^\ast_i)+f_i^2
=\big\|\Lambda^\top h-a_i\big\|_2^2+f_i^2,\qquad a_i=\Lambda^\top c^\ast_i. \tag{$\ast$}$$

Identity $(\ast)$ — established and gated in `experiments/separable-decoder/sep_hfit.py` — means
that fitting $h$ against full fields and fitting the whitened map $q_\theta=\Lambda^\top
h_\theta$ against the precomputed span coefficients are the **same** optimisation problem, at
$O(S R^2)$ instead of $O(S n R)$. Every frozen-bank arm below optimises the whitened problem
and reports field-space errors through $(\ast)$. The whitening round-trip
$h\mapsto q\mapsto h$ is asserted exact to $10^{-10}$ before any arm runs.

$f_i$ is the **span floor**: the best any coefficients could do. It is a constant of the
frozen bank and is the same for every frozen-bank arm.

---

## 2. Data

The evaluation lane's own generator is used throughout: `engines.params_draw` for the
5-parameter family $(c_x,c_y,w,a,\nu)$ and `engines.make_fom` for the implicit Burgers solve at
$L=256$, $\Delta t=0.005$, horizon $0.25$ (51 states per trajectory, **state stride 1**, the
same stride for every density), Newton tolerance $10^{-9}$ / linear tolerance $10^{-7}$.

### 2.1 The ladder is nested, and its top rung is the incumbent's own draw

The incumbent checkpoint `sep_hfit_dense_mid_N256_dense.pkl` was trained on
**4608** trajectories: the canonical `sample_params(seed=0)` draw of 576, plus 4032 appended
from seed 1000 (job `2837431`, `runs/dn256b`). `burgers2d_film.sample_params` and
`engines.params_draw` are the *same* sequential RNG draw over the *same* ranges, so that data
set is reproducible in this lane as

$$\mathcal T_{\rm all}=\big[\texttt{params\_draw}(0,576);\ \texttt{params\_draw}(1000,4032)\big],$$

and the density ladder is the **nested prefix** $\mathcal T_{\rm all}[:N_{\rm traj}]$ for

$$N_{\rm traj}\in\{128,\;512,\;2048,\;4608\}.$$

> **Recorded deviation from the session brief.** The brief called 128 trajectories "the
> incumbent". 128 is the *head-ablation configuration's snapshot draw*
> (`config-ablation.json: train_trajectories = 128`), used there for POD bases, linear/quadratic
> map fitting and the empirical-quadrature rule — **not** for training the head. The head's own
> training set is the 4608-trajectory draw above. Nesting the ladder inside that draw makes
> density the only variable and puts the incumbent's exact data at the top rung, so one arm is a
> true like-for-like retrain of the incumbent inside this pipeline. The 4608 rung is an addition
> to the brief's three; 128/512/2048 are run as instructed.

### 2.2 Disjointness

The six evaluation cases are `params_draw(7090702, 4)` (opened development) and
`params_draw(911702, 2)` (fresh development) — exactly abl01's cohort. The driver asserts, as
`ablation.py` does, that no training trajectory coincides with any evaluation case; it asserts
the same for the held-out generalisation cohort. The final cohort stays sealed.

### 2.3 Held-out generalisation cohort (never fitted, used for in-job selection)

$\mathcal T_{\rm hold}=\texttt{params\_draw}(20260916, 64)$, disjoint from $\mathcal T_{\rm all}$
and from the evaluation cases. On its states the **representation oracle** is computed in
coefficient space by multistart Levenberg–Marquardt on $\|q_\theta(z)-a\|$ (the
`sep_hfit.oracle` protocol: initialisations are the mean training code and a training-only
encoder $a\mapsto z$). This is the only selection statistic, it is declared here before any run,
and it never touches an evaluation case.

---

## 3. The three levers and their exact losses

Write $\mathcal B$ for a minibatch of snapshot indices, $z_i$ the code of snapshot $i$,
$q_\theta=\Lambda^\top h_\theta$.

### L1 — data density (§2.1), and bank+head joint training

Arms `d128k16rec`, `d512k16rec`, `d2048k16rec`, `d4608k16rec`: bank frozen, only $h_\theta$ and
the codes $Z$ train. This isolates **head generalisation**, because the reachable set (the bank
span) and therefore the span floor are identical across the four arms.

Then, at the density selected by §2.3, arm `jointbest` trains **bank and head together** from the
incumbent initialisation, against the field-space loss on a fixed seeded subset of
$P$ interior grid points (§6, deviation D2).

### L2 — objective

Let $\Phi\in\mathbb R^{n\times M}$ be the $M=4K$ lowest discrete sine test modes
($\Phi^\top\Phi=I$) with Laplacian eigenvalues $\lambda$, $A=\Phi^\top G$, and let
$\mathcal N$ be the FOM's own sign-upwind advection. The ROM's weak residual — the exact-linear
form the online solve minimises — is

$$r_w(c;p,\nu)=\frac{A c-p+\Delta t\big(\Phi^\top\mathcal N(Gc)+\nu\,\lambda\odot A c\big)}
{1+\Delta t\,\nu\lambda}. \tag{W}$$

**Reconstruction term** (all arms; `norm='snap'`, i.e. the mean per-snapshot relative MSE):

$$\mathcal L_{\rm rec}(\theta,Z)=\frac{1}{|\mathcal B|}\sum_{i\in\mathcal B}
\frac{\|q_\theta(z_i)-a_i\|_2^2}{\|u_i\|_2^2}.$$

**W — anchored weak residual** (`d128k16w`). On a residual sub-batch
$\mathcal B_r\subset\mathcal B$ of declared size $B_r$, with the previous state taken from
*truth*:

$$\mathcal L_{\rm W}(\theta,Z)=\frac{1}{|\mathcal B_r|}\sum_{i\in\mathcal B_r}
\frac{\big\|r_w\big(h_\theta(z_i);\,A c^\ast_{i-1},\,\nu_i\big)\big\|_2^2}{\|A c^\ast_i\|_2^2},$$

where $i-1$ is the preceding state of the same trajectory ($t_i>0$).

**T — trajectory / two-code** (`d128k16t`). Decode at two consecutive codes and penalise the
backward-Euler weak residual between them:

$$\mathcal L_{\rm T}(\theta,Z)=\frac{1}{|\mathcal B_r|}\sum_{i\in\mathcal B_r}
\frac{\big\|r_w\big(h_\theta(z_i);\,A h_\theta(z_{i-1}),\,\nu_i\big)\big\|_2^2}{\|A c^\ast_i\|_2^2}.$$

**Z — code smoothness** (`d128k16z`), the optional third term: for each $i$ let $\pi(i)$ be the
same time index on the nearest trajectory in standardised parameter space,

$$\mathcal L_{\rm Z}(Z)=\frac{1}{|\mathcal B|}\sum_{i\in\mathcal B}
\frac{\|z_i-z_{i-1}\|_2^2+\|z_i-z_{\pi(i)}\|_2^2}{\sigma_z^2},\qquad
\sigma_z^2=\tfrac1K\mathbb E\|z-\bar z\|_2^2 .$$

**Total.**
$$\mathcal L=\mathcal L_{\rm rec}+\beta_W\mathcal L_{\rm W}+\beta_T\mathcal L_{\rm T}
+\gamma\,\mathcal L_{\rm Z}.$$

The precedent is heat's initial-plus-tail training (LAB-LOG 2026-09-11): adding a term for the
states the query actually has to produce lowered worst current-relative error at 1024 intervals
from 7.5953 % to 4.7625 %. Here the analogue is a term for the equation the query actually has
to satisfy.

$\beta_W,\beta_T,\gamma$ are chosen **once**, before any evaluation, by a declared scale rule:
each weight is set so that at the incumbent head the added term contributes a fraction
$\rho=0.1$ of $\mathcal L_{\rm rec}$'s value on a fixed calibration batch drawn from the
*training* cohort. The realised weights are recorded. No accuracy number is consulted.

### L3 — capacity

$K\in\{16,32\}$ (arms `d128k32rec`, `d2048k32rec` or the selected density). $K$ changes the
online solved dimension and therefore $M=4K$, $m=4M$ and the cost, which is why the cost gate
in §5 is part of the success criterion rather than a footnote.

$R\in\{512,1024\}$ is **conditional** and runs only if L1 shows the gap is capacity- and not
data-limited, by the pre-registered test in §5. If it triggers, `jointr1024` keeps the
incumbent's 512 bank features and adds 512 freshly initialised ones in the *same* feature
family (the $g$-MLP's output width doubles; nothing else changes), trained jointly with the
head at the selected density.

---

## 4. Evaluation — unchanged arm (a) machinery

Each frozen checkpoint is evaluated exactly as `ablation.py` evaluates `a_neural`:

* bank $=$ that checkpoint's own `CoordBank`; head $=$ `arms.neural_head` on its own params;
* $M=4K$ test modes, empirical quadrature at $m=4M$ refit from **that checkpoint's own**
  decoder-output advection snapshots with the same seed, candidate cap and fit-state count;
* cold initializer: the same fixed $48\times48$ Gauss state fit with nearest-candidate
  multistart, candidates $=$ that checkpoint's own training codes subsampled to $\le 8192$;
* trust radius $=0.01\times$ the radius of that checkpoint's own code cloud (`ablation.radius`);
* the same stationarity-aware LM (`ic_budget` 400, `step_budget` 180, gradient tolerance
  $10^{-6}$), the same Gauss–Jordan step solve;
* the same output contract: one supplied dense initial field on the GPU to six dense output
  fields at $t=0,0.05,\dots,0.25$.

The **incumbent checkpoint is arm (a)** and is re-run in the same job, interleaved, as the
control. The same-job full-order controls `fft_tight` and `fft_loose` run in the same
randomised order with GPU burn-in before every timed block; all repetitions are retained.

### Reported per checkpoint — the three layers

| layer | what it is |
|---|---|
| bank projection floor | worst over the six cases and six output times of $\|Q_GQ_G^\top u-u\|/\|u_0\|$ — the best any coefficients could do |
| best-found reconstruction | worst over cases and times of the seeded-multistart best fit on that checkpoint's own manifold, no PDE involved |
| solved worst same-grid (all times) | worst discrepancy against the same-job converged full-order solve `fft_tight` on the same mesh |
| solved worst same-grid (evolved times) | the same maximum restricted to $t>0$ |
| $t=0$ compression | the same at $t=0$: the model's own compression of the supplied initial field |

plus worst error against the refined reference ($L=4096$, $\Delta t=3.125\times10^{-4}$),
median iterations per step, budget exits, stationarity, completion status, and median GPU and
host milliseconds.

The $t=0$ row is reported separately because the 2026-09-15 `prior-dial` and `head-refine`
cells both found the incumbent's worst-over-all-times number **pinned** at its $t=0$
compression: no inference-time knob can move it. Training *can*, which is precisely why it is
the interesting row here.

---

## 5. Pre-registered criteria

**Success** (all four, on the six development cases at 256 intervals):

1. best-found reconstruction worst $< 1.0\,\%$;
2. solved worst same-grid (all times) $< 1.3\,\%$;
3. every solve stationary — no budget exit, no rejected-step exit, on every case and repetition;
4. median GPU query cost $\le 1.5\times$ the incumbent's, in the same job, same quadrature.

**Data-vs-capacity diagnostic**, on the held-out oracle worst at fixed $K=16$ along
$N_{\rm traj}=128,512,2048,4608$:

* *data-limited* if the sequence is monotone decreasing and the $512\to4608$ improvement
  exceeds 10 % relative;
* *capacity/objective-limited* if it has saturated by 512, i.e. the $512\to2048$ relative
  improvement is $<10\,\%$.

The $R=1024$ arm runs **only** in the capacity/objective-limited case. Whichever way it falls is
reported.

**Falsification.** The hypothesis "training alone can close the head's generalisation gap" is
falsified if, at the largest density and the larger $K$, the best-found reconstruction worst
stays above $2.0\,\%$ — i.e. within 25 % of the incumbent — while the span floor stays at
$0.39\,\%$. It is also falsified in a second, weaker sense if best-found improves but the solved
same-grid error does not follow it: that would relocate the binding layer from representation to
the reduction/solver layer, and would be reported as such rather than as a null.

**Training cost** (GPU hours per checkpoint) is reported for every arm, because a training-side
win that costs an order of magnitude more offline is a different claim from one that does not.

---

## 6. Necessary, recorded deviations

* **D1 — the incumbent's density.** See §2.1. The ladder is nested inside the incumbent's own
  4608-trajectory draw rather than three independent `params_draw(0,n)` draws, so that density is
  the only variable and the top rung is the incumbent's data.
* **D2 — the joint bank+head arm's loss.** With the bank moving, identity $(\ast)$ no longer
  applies and the field-space loss needs the fields. Holding every snapshot's full interior field
  is 27 GB at the top density, so the joint arms train against a **fixed, seeded subset of $P$
  interior grid points** (default $P=8192$ of 65025), which is an unbiased subsample of the same
  loss. Frozen-bank arms are unaffected and remain exact. The joint arms' reported errors, like
  every other arm's, are recomputed on the full grid at evaluation time.
* **D3 — the incumbent checkpoint is not reproducible bit-for-bit.** It was produced by a
  different driver (`sep_coeff_extract.py` + `sep_hfit_run.py`) with a dense pick of 131072 of the
  235008 available states and its own step budget. The `d4608k16rec` arm is a like-for-like
  *retrain* on the same trajectories through this lane's pipeline, not a reproduction, and is
  labelled as such. The incumbent itself is carried through evaluation unchanged as the control.
* **D4 — the weak-residual training terms use the exact dense advection**, not the empirical
  quadrature rule, because the EQ rule is fitted at a fixed head and would drift as $\theta$
  moves. The residual sub-batch size $B_r$ is declared and bounds the cost.
* **D5 — training precision.** Training runs in float64 with
  `JAX_DEFAULT_MATMUL_PRECISION=highest`, the same as evaluation. The recipe's own
  `sep_hfit.fit` is used unchanged for the frozen-bank reconstruction arms.

## 7. Gates before any verdict

1. **Local smoke.** The evaluation path, with the incumbent checkpoint and the archived $L=64$
   operators, reproduces the consolidated saved Burgers case to $\le 10^{-8}$ relative and agrees
   with the incumbent `accuracy_paths.make_rom` to $\le 10^{-12}$. The training path's whitening
   round-trip and identity $(\ast)$ are checked to $\le 10^{-10}$ and $\le 10^{-9}$ on a tiny mesh.
2. **In-job.** `incumbent_eq` must reproduce abl01's `a_neural_eq` errors on all six cases to
   $\le 10^{-9}$ relative. Without it there is no report.
3. **Audit.** An independent NumPy recomputation — importing neither driver nor JAX — of every
   reported error from the saved output fields, of every same-grid discrepancy, and of every
   checkpoint SHA-256 before and after evaluation.
4. `jax_backend=gpu`, float64, matmul precision `highest`, one job per attempt directory,
   `squeue` checked before and after every submission.

---

## 8. Amendments, dated, all made before any evaluation run

**A1 (2026-09-16, before the first training job produced any number).** The joint bank+head arms
were amended in two ways, and the first pending submission was cancelled before it started rather
than run with the defect.

1. *Loss scale.* The sampled-point sum over $P$ points estimates $(P/n)$ of the full-grid squared
   error, so it is rescaled by $n/P$ and the joint arms' data term reads as the same per-snapshot
   relative MSE the frozen-bank arms minimise. Without the rescaling the data term sat a factor
   $n/P \approx 16$ below its intended value and every fixed regulariser weight sat that factor
   too high against it.
2. *Orthonormality regulariser off.* `sep_common.train_autodecoder` adds
   $\lambda_{\rm orth}\,\overline{(C_G-I)^2}$ with $\lambda_{\rm orth}=10^{-4}$ and
   $C_G = G^\top G / (P\,s^2)$. That penalty exists to condition a **freshly initialised** bank.
   These arms warm start from the incumbent's already-trained bank, whose Gram is far from the
   identity (its condition number is $\approx 2.6\times10^{4}$), so at the warm start the penalty
   is orders of magnitude larger than the data term and a joint run under it would be an
   orthonormalisation rather than a refinement. $\lambda_{\rm orth}$ is therefore set to $0$ and
   the realised feature-Gram deviation is **reported** — before and after training — together with
   the resulting bank's Gram condition number, rather than imposed. No accuracy number was
   consulted in making either change.

**A2 (2026-09-16, same point).** The joint arms **continue from the selected frozen-bank arm's
head and codes** rather than cold-starting, because the question the brief asks is what unfreezing
the bank adds *on top of* the best head-only fit, not whether a bounded joint run can rediscover
it. The warm-start arm is named in the result JSON. Their density is additionally capped
(`joint_density_cap`) because the joint arms must hold raw field values and that block grows with
both states and points.

**A3 (2026-09-16, same point).** Frozen-bank arms are graded by the held-out oracle with two
initialisations (mean training code and a training-only encoder); the joint arms have no whitened
training block in their own bank, so they are graded with the mean-code initialisation only. Both
columns are reported for every arm and the **mean-code-only column is the like-for-like one**
across the two families. Selection happens among frozen-bank arms only, where the two-init column
is available for all of them.
