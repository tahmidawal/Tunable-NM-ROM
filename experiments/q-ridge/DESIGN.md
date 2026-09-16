# q-ridge — is the Burgers correction ladder's evolved-times regression test-space overfitting?

Pre-registered before any job was submitted. Every amendment is appended to
[§9 Amendments](#9-amendments) with its reason and the date, never edited in place.

Worktree `worktrees/2026-09-16-q-ridge`, branch `exp/2026-09-16-q-ridge`, forked from
`exp/2026-09-16-b-ladder-top` at `b8efd5b4`. Cluster namespace
`/cluster/tufts/paralab/tawal01/q_ridge_20260916/`. Nothing is merged; nothing is pushed.

---

## 1. The question, and what the archive already says about it

The `b-ladder-top` cell (2026-09-16) left this open:

> The $q=16$ regression on the evolved-times metric is unexplained and reproducible across
> both quadratures and both evolution tolerances; nothing here says why one extra correction
> direction makes the trajectory worse while making the supplied-field fit better.

The coordinator's hypothesis under test:

> **H(overfit).** The extra unknowns overfit the $M$ test equations — the solve lowers the
> projected weak residual by moving along directions the $M$ tests barely observe, which
> raises the field error.

### 1.1 What the retained archives already show (measured here, before submitting anything)

The regression's location was *not* known when this lane was commissioned, so the first
thing done was to recompute the two reported metrics for **every** arm of the audited
`cheap-corrections` job `cclad01` (job 3734098) from its own saved fields — the archive is
Git-tracked in this worktree as bounded chunks, so this needed no GPU and no new job. The
same-grid metric (against `cclad01`'s own converged `fft_tight` full-order solve, the
metric `b-ladder-top` reports) gives, worst over the six cases:

| rule | quadrature | $q=0$ evolved % | $q=16$ evolved % | $q=32$ | $q=64$ | $q=128$ | monotone? |
|---|---|---|---|---|---|---|---|
| $M=4(K+q)$ | dense | 1.8890 | 1.3985 | 1.2336 | 1.0843 | 0.8930 | **yes** |
| $M=256$ fixed | dense | 1.2710 | 1.2305 | 1.1869 | 1.1255 | 1.0418 | **yes** |
| $M=256$ fixed | EQ | 1.3113 | 1.4956 | 1.5231 | 1.5557 | 1.6141 | **no** |
| $M=4(K+q)$ | EQ | 1.9002 | 2.9937 | — | — | — | **no** |

and `b-ladder-top`'s own Q2 envelope reproduces the third row at a different evolution
tolerance (EQ, $M=256$: 1.3186, 1.8066, 1.3048, 1.5738, 1.3517, 0.7580 for
$q = 0,16,32,64,128,256$).

**The regression is carried by the empirical quadrature, not by the test count.** The dense
$M=256$ arms solve *the identical $M$ test equations against the identical $K+q$ unknowns*
as the EQ $M=256$ arms; the only difference is that the advection integral
$\Phi^\top n(u)$ is replaced by an $m$-point rule. The dense arms are monotone; the EQ arms
are not. That is already strong evidence against H(overfit) — but it is a *post hoc* reading
of somebody else's job, it does not measure the held-out residual H(overfit) actually
predicts, and it does not test either remedy. This lane runs the pre-registered test.

### 1.2 What this changes about the pre-registered design

The remedies must be applied where the regression exists. The **target configuration** is
therefore

> **EQ quadrature, $M = 4(K+q)$ tests, $m = 4M$ quadrature points** (the incumbent rule),

which shows the regression at $q=16$ on **both** reported metrics ($1.9002 \to 2.9937$
evolved, and $2.5629 \to 2.9937$ all-times, because at $q=16$ the evolved error overtakes
the $t=0$ compression). Every arm is run in the **dense** quadrature as well, at the same
$q$, the same $M$ and the same $\lambda$, as the exact-quadrature control: dense is where
the regression is absent, so a remedy that only helps in EQ localises the cause to the rule.

This is recorded here, before submission, precisely because it makes the pre-registered pass
criterion *non-vacuous in EQ and vacuous in dense*. Both are reported; the verdict is stated
per quadrature.

---

## 2. What is frozen

Everything except the three declared manipulations. Inherited verbatim from
`b-ladder-top` / `cheap-corrections`:

| item | value |
|---|---|
| checkpoint | `experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl`, SHA256 `18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589` |
| latent dimension | $K = 16$; bank dimension $R = 512$ |
| mesh / time step | $L = 256$ intervals, $\Delta t = 0.005$, 50 steps to $t = 0.25$, output every 10 steps |
| reference | FOM at 4096 intervals, $\Delta t = 3.125\times10^{-4}$, restricted to the query grid |
| evaluation cohort | **bitwise `abl01`'s six cases**: `params_draw(7090702, 4)` then `params_draw(911702, 2)`; reference field SHA256s are declared in the config and gated |
| directions $C_q$ | `directions.audited` = `ladder.residual_directions` verbatim — **the OLD rule**, seed 20260915, 1024 residual snapshots, 4 starts, budget 200; nested in $q$ |
| head | $u(z,y) = G\,(h_\theta(z) + C_q y)$ |
| solver | `varpro.make_block_lm`, the budget-600 block-damped variable-projection solver (`b-ladder-top` Q1-B's converged contract) |
| stopping rule | `gtol` $=10^{-6}$ on the normalized joint gradient; residual tolerance $10^{-9}\|u_{\text{gauss}}\|\sqrt{n_g}$; per-step budget **600**; IC budget 400 |
| initializer | fixed 48-point Gauss state fit + two-way extrapolation guard, verbatim |
| output contract | supplied dense initial field on GPU $\to$ six dense GPU output fields |
| timing | 3 repetitions, randomised subject order (`order_seed` 911716), 0.25 s burn-in before each, all retained |
| numerics | float64, `JAX_DEFAULT_MATMUL_PRECISION=highest`, `jax_backend=gpu` asserted |

Same-job controls, in the same allocation on the same GPU: `fft_loose`, `nt1e-2_dt01`,
`fft_tight`. `fft_tight` *is* the same-grid reference, so its same-grid error is zero by
construction.

Two error metrics are reported for every arm, per time, per case, as `b-ladder-top` does:

- **worst over all output times** — includes $t=0$, which is the decoder's compression of
  the supplied field and is charged to the ROM;
- **worst over evolved times only** ($t>0$) — the trajectory error proper.

The $t=0$ compression is reported separately. Both are measured against the same-job
`fft_tight` field (the same-grid metric) and, additionally, against the 4096-interval
reference.

---

## 3. R1 — a ridge on the corrections

### 3.1 The objective

The retained per-step solve minimises the weak residual over $w = (z,y) \in
\mathbb{R}^{K+q}$:

$$\min_{z,\,y}\; \bigl\| r_w(z, y) \bigr\|^2 , \qquad
r_w \in \mathbb{R}^{M},$$

with $r_w$ the row-scaled implicit-Euler weak residual of `arms.weak_dense` /
`arms.weak_eq`. R1 replaces it by

$$\boxed{\;\min_{z,\,y}\; \bigl\| r_w(z, y) \bigr\|^2 \;+\; \lambda\,\|y\|_W^2\;}$$

where $W$ is the **field metric**: $\|y\|_W^2 = \|G\,C_q\,y\|_2^2$, the squared $L^2$-ish
grid norm of the correction field that $y$ adds to the head's own state.

**$W$ is the identity in these coordinates, and that is not an assumption.** The directions
are built as $C_q = R_G^{-1}\,\tilde C_{:, :q}$ with $G = Q_G R_G$ the thin QR of the bank
(`arms.whiten`) and $\tilde C$ orthonormal — the same $R_G$ whitening `prior-dial` used. Then
$G C_q = Q_G \tilde C_{:, :q}$ has orthonormal columns, so

$$\|G C_q y\|_2 = \|y\|_2 .$$

The correction coordinates are *already* field-whitened. The ridge is therefore written
$\lambda\|y\|_2^2$ in code, and the statement "$W$ = field metric" is exact, not a
convenience.

### 3.2 The scaling of $\lambda$ — why it is dimensionless

$\|r_w\|^2$ carries the units of the test-mode coefficients; $\|y\|^2$ carries field units.
The exact linear part of $\partial r_w / \partial y$ converts between them. Writing the weak
residual out,

$$r_w = \frac{A h - p + \Delta t\,(\Phi^\top n(u) + \nu\,\Lambda\,A h)}{1 + \Delta t\,\nu\,\Lambda},
\qquad A = \Phi^\top G,$$

the terms linear in $h$ collapse exactly — $(A h + \Delta t \nu \Lambda A h)/(1 + \Delta t
\nu \Lambda) = A h$ — so the **state-independent** linear response of the residual to $y$ is
$A C_q = \Phi^\top G C_q$. Define

$$\sigma_q \;=\; \bigl\| \Phi^\top G\,C_q \bigr\|_2 \quad\text{(largest singular value)},
\qquad \lambda \;=\; \lambda_{\mathrm{rel}}\;\sigma_q^{\,2}.$$

$\lambda_{\mathrm{rel}}$ is dimensionless and comparable across $q$, across $M$ and across
quadratures. $\lambda_{\mathrm{rel}} = 1$ balances the penalty against the *strongest* linear
residual response the tests can see. This is `prior-dial`'s convention
($\lambda = \lambda_{\mathrm{rel}}\sigma^2$, $\sigma = \|A R_G^{-1}\|_2 = \|\Phi^\top
Q_G\|_2$) transported from the full bank to the $q$-dimensional correction block; at $q = R$
the two coincide. $\sigma_q$ is computed once per (arm, quadrature) at build time and
recorded.

### 3.3 How $\lambda$ enters the solve

Exactly as an augmented residual, so the elimination of $y$, the Jacobian, the damping, the
trust radius and the stopping rule are all the retained ones applied to the ridged objective:

$$F(z,y) = \begin{bmatrix} r_w(z,y) \\ \sqrt{\lambda}\, y \end{bmatrix} \in \mathbb{R}^{M+q},
\qquad
J_F = \begin{bmatrix} J_z & J_y \\ 0 & \sqrt{\lambda}\,I_q \end{bmatrix}.$$

The block-damped normal equations the retained solver already forms,

$$\bigl(J^\top J + \operatorname{diag}(\mu\,d\odot m_z + \varepsilon\,d\odot m_y)\bigr)\,
\delta w = -\,J^\top r ,$$

become, with $J \to J_F$ and $r \to F$, the **inner normal equations of the ridged problem**:

$$\bigl(J^\top J + \lambda P_y + \operatorname{diag}(\mu\,d\odot m_z + \varepsilon\,d\odot m_y)\bigr)
\,\delta w = -\,\bigl(J^\top r + \lambda\,[\,0_K;\, y\,]\bigr),
\qquad P_y = \operatorname{diag}(0_K, I_q).$$

$\mu$ is the retained Levenberg parameter on the $z$ block and $\varepsilon = 10^{-10}$ the
retained tiny ridge on the $y$ block (a numerical safeguard that does not change the
objective's stationary point, kept unchanged so the $\lambda = 0$ arm is the incumbent). No
line of the solver is rewritten: `varpro.make_block_lm` is called on $F$ rather than on
$r_w$. Acceptance, the residual-tolerance exit and the normalized-gradient exit are therefore
all taken on $F$ and $J_F$ — the same stopping rule, applied to the objective actually being
minimised.

**$\lambda = 0$ takes a Python branch to the retained path** (`varpro.make_block_lm` on
$r_w$ itself, and at $q=0$ `varpro.make_query` itself), not to the augmented residual with
$q$ zero rows appended. The reproduction gate is then exact rather than
floating-point-approximate.

### 3.4 What is *not* ridged, and why

The **initial-condition fit** at $t=0$ is left exactly as retained: $y$ is eliminated there
by an orthogonal projection onto the complement of $\operatorname{col}(R_{\text{cold}} C_q)$,
with no penalty. Two reasons: the initializer policy is frozen by the contract, and the IC
fit is a *field* fit in the 512-dimensional cold-start Gauss space, not a fit to the $M$ weak
test equations that H(overfit) is about. A measurable consequence, used as an in-job gate:

> **the $t=0$ output field, and therefore the $t=0$ compression, must be bitwise identical
> across every $\lambda$ at fixed $(q, M, \text{quadrature})$.**

### 3.5 The R1 grid

$\lambda_{\mathrm{rel}} \in \{0,\,10^{-4},\,10^{-3},\,10^{-2},\,10^{-1},\,1\}$ at
$q \in \{16, 64, 256\}$, in **both** quadratures, at the incumbent rule $M = 4(K+q)$
(so $M = 128,\,320,\,1088$), with $q = 0$ at $\lambda_{\mathrm{rel}} = 0$ as the ladder's
base point ($q=0$ has no $y$, so $\lambda$ is vacuous there and only one arm is run).
$38$ ROM arms, plus gate arms and the full-order controls.

---

## 4. R2 — more tests at fixed $q$

The same hypothesis from the other side. If the extra unknowns are overfitting $M$ test
equations, raising $M$ at fixed $q$ must remove the regression with no penalty at all.

$$M \in \{4(K+q),\; 8(K+q),\; 16(K+q)\} \quad\text{at}\quad q \in \{0, 16, 64\},$$

i.e. $M = 64/128/256$ at $q=0$, $128/256/512$ at $q=16$, $320/640/1280$ at $q=64$, each in
**both** quadratures, all at $\lambda_{\mathrm{rel}} = 0$.

For the EQ arms the rule is **refit per $(q, M)$** at $m = 4M$ target support, using the
enriched codes of that rung and the walltime-bounded block-greedy NNLS fitter with the
`b-ladder-top` refit block ($m/16$), which reached full support on every rule in that lane.
Where $4M$ exceeds the declared support cap the row states the achieved $m$, the target $4M$,
the ratio $m/M$ and the fitter's `truncated` flag; **a rule whose fitter reports `truncated`
is disqualified** and its row is reported as invalid rather than compared. The cap is
declared in the config as `quadrature_cap = 2048`, the value `b-ladder-top` used, so at
$M = 640$ and $M = 1280$ the EQ rows are $m/M = 3.2$ and $1.6$ rather than $4$, and the dense
twin at the same $(q, M)$ is the exact-quadrature statement for those cells. Every row
reports which.

---

## 5. R3 — the control: the weak residual on held-out test modes

The direct signature H(overfit) predicts. If the solve is buying a lower projected residual
on the $M$ modes it is graded on by moving along directions those modes barely see, then the
**same** residual, exactly integrated and tested against modes that were *not* in the solve,
must be **higher** at $q=16$ than at $q=0$.

### 5.1 Definition

Let $\Psi$ be sine modes, ordered as `engines.modes` orders them (ascending discrete
Laplacian eigenvalue, stable sort). For an arm with test set $\Phi = $ modes $1\ldots M$,
the **held-out set** is modes $M+1 \ldots M+4M$ (the next $4M$), with eigenvalues
$\Lambda_\Psi$. For every solved step $n = 1 \ldots 50$ of every saved case, and the exact
(dense) advection operator,

$$r^{\text{held}}_n = \frac{\Psi^\top u_n - \Psi^\top u_{n-1}
+ \Delta t\,\bigl(\Psi^\top n(u_n) + \nu\,\Lambda_\Psi \Psi^\top u_n\bigr)}
{1 + \Delta t\,\nu\,\Lambda_\Psi},$$

and the in-space counterpart $r^{\text{in}}_n$ with $\Phi$ and $\Lambda_\Phi$ in place of
$\Psi$ and $\Lambda_\Psi$. Both are **exactly integrated** (dense $n(\cdot)$), for EQ arms as
well as dense arms; for a converged dense arm $r^{\text{in}}$ is the quantity the solver drove
to zero, for an EQ arm it is not, and that difference is itself part of the diagnosis.

Reported per arm, worst over cases and steps, as per-mode RMS
$\|r\|/\sqrt{\#\text{modes}}$ normalised by $\|u_{n-1}\|_2/\sqrt{n_g}$ so that sets of
different size are comparable, together with the raw norms and the ratio
$r^{\text{held}}/r^{\text{in}}$.

Because the per-arm held-out sets differ when $M$ differs, the **comparison used for the
falsification test** is on a *common* held-out set: the $4M_{\max}$ modes beyond
$M_{\max} = \max M$ over the arms being compared, so $q=0$ and $q=16$ are graded on
identical tests. Both the per-arm and the common-set numbers are reported.

### 5.2 How it is computed — post hoc, in NumPy, no JAX and no GPU

The job saves, for every arm and every case, the **bank coefficient vector**
$c_n = h_\theta(z_n) + C_q y_n \in \mathbb{R}^{512}$ at **every** one of the 51 steps
($51 \times 512$ doubles $\approx$ 209 kB per case), and saves the frozen bank
$G \in \mathbb{R}^{65025 \times 512}$ **once** per job. The audit reconstructs
$u_n = G c_n$ on the grid, applies the five-point upwind advection of `engines.spatial`
re-implemented in NumPy, and forms $\Phi$, $\Psi$ and the eigenvalues with
`engines.modes`' arithmetic re-implemented in NumPy. No JAX is imported by the audit.

Two consequences, both used as gates: the reconstructed $u_n$ at the six output steps must
reproduce the independently saved output fields to $10^{-12}$ relative, and the recomputed
errors must reproduce the job's own to $10^{-9}$.

`bank_G.npz` is 266 MB. It is collected by checksum and kept in the gitignored `runs/` tree,
and **excluded from the Git-tracked chunked archive** with its SHA256 recorded in the
manifest — the same treatment `b-ladder-top`'s collector gives the 429 MB FNO checkpoint, and
for the same reason: it is a deterministic function of the frozen checkpoint, not a result.

---

## 6. Jobs, and what each one carries

Hard cap three cluster jobs. Two are planned; the third is reserve.

| job | attempt | contents | GPU |
|---|---|---|---|
| 1 | `qrg101` | **R1**: $\lambda$ sweep, $q \in \{0,16,64,256\}$ at $M=4(K+q)$, dense and EQ, plus the reproduction gate arms | A100, `--exclude pax007` |
| 2 | `qrg201` | **R2**: $M \in \{4,8,16\}(K+q)$ at $q \in \{0,16,64\}$, dense and EQ, $\lambda=0$ | A100, `--exclude pax007` |
| 3 | — | reserve (a failed submission, or one predeclared follow-up) | — |

Each attempt gets its own submit directory under
`experiments/q-ridge/runs/<attempt>/` and its own remote directory
`/cluster/tufts/paralab/tawal01/q_ridge_20260916/<attempt>/`; `squeue` is checked before and
after every submission; exactly one job per directory. `gpu` partition only. R3 is computed
in the audit from both jobs' saved coefficients.

---

## 7. Gates

Blocking unless marked informational.

1. `complete`, `backend_gpu`, `x64`, `precision_highest`, `bank_frozen`,
   `checkpoint_unchanged`, `final_cohort_unopened`, `artifacts_present`.
2. `evaluation_cohort_bitwise_abl01` — the six physical cases, and the six restricted
   4096-interval reference fields, match the SHA256s declared in the config.
3. `reference_residuals` — every reference solve below $2\times10^{-11}$ relative.
4. `recorded_errors_recomputed_from_saved_fields` — every reported error recomputed in
   NumPy from the retained fields, agreeing to $10^{-9}$ relative.
5. `reproduces_q0_m4_dense_l0` — the $\lambda = 0$, $M = 4(K+q)$, dense, $q=0$ arm
   reproduces `b-ladder-top` `btq101`'s `q0_m4_dense_base` to $\le 10^{-9}$ relative on
   both metrics. ($q=0$ carries no directions, so this gate is unconditional.)
6. `reproduces_q0_m4_eq_l0_retained` — the $q=0$, $M=64$, EQ arm built with the
   **retained** NNLS fitter reproduces `btq101`'s `q0_m4_eq_base` to $\le 10^{-9}$.
7. `reproduces_q256_m2_dense_l0` (job 1) — the $\lambda=0$, $M = 2(K+q)$, dense, $q=256$,
   budget-600 arm reproduces `b-ladder-top` `btq102`'s `q256_m2_dense_base_b600` to
   $\le 10^{-9}$ when the directions hash matches, $\le 10^{-3}$ otherwise (declared in the
   config as `tolerance` / `tolerance_if_directions_bitwise`, `b-ladder-top`'s own
   convention; the directions matrix is GPU-model dependent across jobs and is reported as a
   probe, not a requirement).
8. `reproduces_q16_m4_dense_l0`, `reproduces_q64_m4_dense_l0` — against `cclad01`'s
   `q16_m4_dense_block` and `q64_m4_dense_block`, same two-tier tolerance.
9. `t0_field_invariant_in_lambda` — at fixed $(q, M, \text{quadrature})$ the $t=0$ output
   field is **bitwise** identical across every $\lambda$ (§3.4).
10. `overdetermined_weak_system` — $M > K+q$ for every arm.
11. `every_eq_rule_reports_validity` and `eq_rules_untruncated` — every EQ rule reports
    `truncated`, and every EQ rule used in a compared row has `truncated == false`.
12. `every_rom_carries_exit_and_stationarity`, `every_subject_case_has_all_reps`,
    `repetition_output_identical`, `every_invocation_paired`.
13. `same_grid_baseline_present` — `fft_tight` present for all six cases.
14. `decoded_fields_match_saved_outputs` — $u_n = G c_n$ reconstructed in the audit
    reproduces the independently saved output fields to $10^{-12}$ relative (§5.2).
15. `bank_sha256_consistent` — the job's recorded `bank_G` SHA256 equals the collected
    file's.
16. Informational probes: `directions_hash_matches_cclad01`,
    `reference_fields_bitwise_match_cclad01`.

---

## 8. Pre-registered verdict

Stated **per quadrature**, because §1.1 establishes that the regression exists in EQ and not
in dense, and because the coordinator's criterion is about removing a regression.

**PASS** (for a remedy, in a quadrature) iff, in that quadrature:

- the worst-over-**evolved**-times error is **non-increasing in $q$** over
  $\{0, 16, 64, 256\}$ for R1 at some single $\lambda_{\mathrm{rel}}$ (one $\lambda$ for the
  whole ladder, not a per-rung choice), or over $\{0, 16, 64\}$ for R2 at some single $M$
  rule; **and**
- every rung of that ladder is **converged** under the shared stationarity rule (gradient
  $\le 10^{-6}$, zero budget exits, IC converged); **and**
- the median complete-query GPU time of each rung is $\le 1.5\times$ that rung's
  $\lambda = 0$ / $M = 4(K+q)$ cost; **and**
- the worst-over-**all**-times error is not raised at any rung relative to
  $\lambda = 0$ / $M = 4(K+q)$ (tolerance: no rung worse by more than $10^{-4}$ relative).

**FALSIFICATION of H(overfit)**, reported as the lane's answer whether or not a remedy
passes:

- if R3 shows the held-out weak residual is **not** higher at $q=16$ than at $q=0$ on the
  common held-out set, test-space overfitting is **not** the cause, and the lane says so;
- if R1 or R2 removes the regression **only** by removing the accuracy gain at $q=256$ (R1)
  or at the largest $q$ measured (R2) — i.e. the top rung's evolved error rises to within
  $10\%$ of the $q=0$ rung's — the lane says so explicitly and does not call it a pass.

The dense arms are expected to pass vacuously (no regression to remove at $\lambda=0$); that
is stated in advance here so it cannot be read afterwards as a result.

---

## 9. Amendments

*(none yet)*
