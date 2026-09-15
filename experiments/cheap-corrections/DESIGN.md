# Making the correction ladder cheap — predeclared design

One question: **can the fixed-weight correction ladder be made to converge at a cost that
keeps $q$ a usable knob, without changing the checkpoint, the direction rule or the
reachable set?**

The audited ladder (job `3713867`, `experiments/head-ablation/artifacts/qlad01`) established
that solved error on Burgers is monotone in $q$ — worst same-grid $2.5629\% \to 0.6027\%$ from
$q=0$ to $q=512$ — but that only $q\in\{0,16\}$ converge under the shared stopping rule and
that cost grows $28\times$ over the ladder. Three things caused that, and this cell changes
exactly those three, each isolated so its effect is measurable:

1. all $K+q$ unknowns were solved by one Levenberg–Marquardt iteration **at the $q=0$ trust
   radius**, so a fixed trust radius throttled an ever larger step;
2. the test count was forced to grow as $M=4(K+q)$, and the $q=0$ control at $M=2112$ showed
   this alone is a $4.04\times$ cost factor;
3. there was **no empirical-quadrature rule above $q=16$**, and empirical quadrature is a
   $6$–$7\times$ cost lever at zero accuracy change.

Everything else is frozen: the checkpoint `sep_hfit_dense_mid_N256_dense.pkl`
(SHA256 `18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589`), the nested
direction rule and its seed, the weak objective, the initializer policy, the stopping
tolerance, the iteration budgets, the output contract, the mesh, the time step and the six
opened development cases.

---

## The common contract (unchanged from the ladder)

With $\Phi\in\mathbb R^{n\times M}$ the $M$ lowest discrete sine test modes
($\Phi^\top\Phi=I$), $\lambda$ their Laplacian eigenvalues, $G$ the frozen separable bank,
$A=\Phi^\top G$, backward Euler at step $\Delta t$ and the FOM's own sign-upwind advection
$\mathcal N$, the reduced state is $u=G\eta$ with enriched coefficients

$$\eta(z,y)=h_\theta(z)+C_q\,y,\qquad z\in\mathbb R^{K},\ y\in\mathbb R^{q},\ K=16,$$

and the weak residual is

$$r(z,y)=\frac{A\eta-p+\Delta t\big(\Phi^\top\mathcal N(G\eta)+\nu\,\lambda\odot A\eta\big)}
{1+\Delta t\,\nu\lambda},\qquad p=A\,\eta^{\rm prev}.$$

$q=0$ is head-ablation arm (a) verbatim. $C_q$ is the first $q$ columns of the **same**
nested direction matrix the audited ladder used (field-metric POD of the decoder-output
residual $\eta-h_\theta(z^\*)$, 1024 seeded snapshots, 4 multistarts, budget 200, seed
20260915, `directions_sha256 = f270e5bf682ad220f2164ced82dde0f3e0374e84e2fa002f3101e0728062d399`).
The reachable set at every $q$ is therefore identical to the audited ladder's.

---

## Change 1 — variable projection: eliminate $y$, iterate on $z$ alone

$r$ is quadratic in $y$ (through the product $u\,\partial u$ in $\mathcal N$, piecewise in the
upwind sign) and non-convex only in $z$. So define

$$y^\*(z)=\arg\min_{y}\ \|r(z,y)\|_2^2,\qquad F(z)=\tfrac12\big\|r\big(z,y^\*(z)\big)\big\|_2^2 ,$$

and run the retained stationarity-aware damped LM on $z\in\mathbb R^{16}$ only, at the
**$q=0$ trust radius**, with $y$ never subject to that radius.

**Inner solve.** $y^\*$ is approximated by $N_{\rm in}$ damped Gauss–Newton steps from a warm
start $y^{\rm seed}$ (the previous time step's coefficients, a fixed parameter of the outer
solve, so $y^\*(\cdot)$ is a well-defined deterministic map of $z$):

$$y^{(i+1)}=y^{(i)}+\delta^{(i)},\qquad
\big(J_y^{\top}J_y+\mu\,\mathrm{diag}(J_y^{\top}J_y)\big)\,\delta^{(i)}=-J_y^{\top}r^{(i)},
\qquad J_y=\frac{\partial r}{\partial y}\Big|_{(z,y^{(i)})},$$

with a three-point safeguard: the accepted iterate is the best of
$\{y^{(i)},\,y^{(i)}+\tfrac12\delta^{(i)},\,y^{(i)}+\delta^{(i)}\}$ by residual norm, so the
inner solve can never increase $\|r\|$. Because $r$ is quadratic in $y$ this converges in a
few steps from any reasonable warm start; $N_{\rm in}$ is recorded and its inner stationarity
is measured, not assumed.

**Outer Jacobian (Kaufman).** The outer LM uses $J_z=\partial r/\partial z$ evaluated at
$y=y^\*(z)$, with $y^\*$ treated as constant (`stop_gradient`). This is exact for the
*gradient*, which is what the stopping rule tests:

$$\nabla F(z)=J_z^{\top}r+\Big(\tfrac{\partial y^\*}{\partial z}\Big)^{\!\top} J_y^{\top}r,
\qquad J_y^{\top}r=0\ \text{at inner optimality},$$

so only the Gauss–Newton Hessian model is approximated, never the stationarity test. The
alternative — differentiating through the inner iteration — needs $16\times q$ forward
tangents on $n=65025$ intermediates and is rejected on memory grounds; that rejection is a
recorded design decision, not a measured result.

**Why this is cheap.** The joint solver builds a Jacobian with $K+q$ forward tangents at every
LM iteration ($528$ at $q=512$). Variable projection builds $16$ tangents for the outer step
and $q$ tangents for the inner step, but the inner step is a *linear* least-squares solve that
needs no line search on $z$ and no trust region, and its Jacobian is the cheap block: $J_y$
never passes through the network.

**Alternating variant (also tested).** Block coordinate descent: per time step, $N_{\rm rd}$
rounds of

$$z^{(j+1)}=\mathrm{LM}_z\big(z^{(j)};\,y^{(j)}\big),\qquad
y^{(j+1)}=\mathrm{GN}_y\big(y^{(j)};\,z^{(j+1)}\big),$$

with the outer iteration budget split across rounds. Cheaper per round (no inner solve inside
the outer function evaluation) but with no guarantee the $z$-step sees the effect of $y$.
One of the two variants is kept for the main sweep, chosen on the $q\in\{16,64\}$ comparison
by (converged, then cost); the other's numbers are still reported.

**Stationarity accounting.** Every solve records $\|J_z^{\top}r\|$, $\|J_y^{\top}r\|$,
$\|J_z\|_F$, $\|J_y\|_F$ and $\|r\|$, so the *joint* normalized gradient

$$g=\frac{\sqrt{\|J_z^{\top}r\|^2+\|J_y^{\top}r\|^2}}
{\sqrt{\|J_z\|_F^2+\|J_y\|_F^2}\;\|r\|}$$

is reconstructible exactly and is compared with the retained tolerance $10^{-6}$. At $q=0$
this is the retained rule verbatim. A solve is **converged** only if every time step and the
initial fit exit for a reason in $\{1,2,4\}$ with no iteration-budget exit *and* $g\le10^{-6}$.

**$q=0$ is special-cased** to the joint path exactly (no inner solve, empty $y$), so the
variable-projection and alternating arms at $q=0$ must produce **bitwise identical** output to
the joint arm. That is checked in-job on field SHA256.

---

## Change 2 — test count

The weak objective needs $M>K+q$. Three rules are compared at every $q$ where they are legal:

| rule | $M$ | legal for |
| --- | --- | --- |
| `m4` | $4(K+q)$ — the retained rule | every $q$ |
| `m2` | $2(K+q)$ | every $q$ |
| `m256` | $256$, fixed | $q\le 128$ (needs $256>16+q$) |

`m256` is the interesting one: it decouples the test count from $q$ entirely, which is also
what makes a *fixed-size* empirical-quadrature rule constructible at every rung.

---

## Change 3 — one empirical-quadrature rule per rung

The retained EQ rule is a nonnegative least-squares (NNLS) weighting of $m$ grid points that
reproduces $\Phi^\top\mathcal N(G\eta)$ on decoder-output snapshots. Above $q=16$ the audited
ladder could not construct it inside the job budget, so those rungs paid the full grid sum.

**Enriched snapshots.** With $G=Q_GR_G$ thin QR and $\tilde C_q$ the field-orthonormal
residual directions ($C_q=R_G^{-1}\tilde C_q$), the enriched-manifold best fit of a truth
snapshot with bank coefficients $\eta_i$ is

$$z_i^\*=\text{best-found head code},\qquad
\rho_i=\eta_i-h_\theta(z_i^\*),\qquad
y_i=\tilde C_q^{\top}R_G\,\rho_i,$$

because $\tilde C_q$ is orthonormal in the field metric. So the enriched codes
$w_i=(z_i^\*,y_i)$ come free from the direction fit that already runs, and the quadrature is
fitted on exactly the states the $q$-rung can reach. At $q=0$ this reduces to the retained
rule's own code table, so the retained rules are reproduced exactly.

**Bounded fitter.** The retained greedy NNLS refits the full support after every single added
point, which is why $m=512$ cost 250 s. The bounded fitter adds points in blocks of
$\lceil m/64\rceil$, refits once per block, prunes zero weights, and stops at whichever of
{target $m$, walltime budget} binds. It records `support`, `fit_relative_residual`,
`fit_seconds`, `blocks`, `truncated`. The retained exact fitter is still used wherever the
audited ladder used it ($q=0$ at $m=256$, $q=16$ at $m=512$), so those arms reproduce
`qlad01` bitwise and the two fitters are compared head to head at $q=16$.

Offline quadrature-fitting cost is reported in its own column and is **never** inside a query
timing.

---

## The sweep

Burgers, 256 intervals, $\Delta t=0.005$, frozen checkpoint above, the same six opened
development cases (`eval_seed 7090702` ×4, `eval_fresh_seed 911702` ×2), 3 timed repetitions
with GPU burn-in before every block, all repetitions retained, complete device-query timing as
`qlad01` defined it (supplied dense initial field on GPU → six dense GPU output fields), with
the same-job FOM controls `fft_tight` and `nt1e-2` interleaved in the randomised order.

*Variant selection* (dense, `m4`): $q\in\{0,16,64\}\times\{$joint, varpro, alternating$\}$,
plus the retained EQ arms at $q\in\{0,16\}$.

*Main ladder* (best variant): $q\in\{0,16,32,64,128,256,512\}$ × {`m4`, `m2`, `m256` where
legal} × {dense, eq where constructible}.

Poisson, 1024 intervals, checkpoint `r128_joint.pkl`
(SHA256 `a128e7635c318faae215c3bc3d9885ac479616caeaf81f02936e73f871583b3c`), the same twelve
development sources `pabl01` used (`eval_seed 7090703` ×6, `fresh_seed 7090732` ×6), with
`dst_direct` interleaved. The Poisson weak residual is exactly $Bh(z)-f_m$ — **linear** in the
coefficients — so the `pabl01` $q=32$ path already eliminates $y$ exactly; that path *is*
variable projection with a one-step exact inner solve, and it is extended to
$q\in\{0,8,16,32,64,128\}$. There is no quadrature on Poisson, so change 3 does not apply; the
test count is varied instead.

**Poisson direction extension.** `basis.npz` stores only 32 nested directions. Columns
$1..32$ are used verbatim, so $q\le32$ reproduces the retained arm exactly; columns $33..128$
extend them by the same rule (right singular vectors of the normalised training residual in
the QR physical metric) applied to the component orthogonal to the retained 32, computed
in-job from a regenerated training set. The construction is recorded, and the $q=32$ arm is
gated against `pabl01`'s `a_neural_q32`. At $q=128=R$ the corrections span the whole bank, the
eliminated operator $B_\perp$ is numerically zero and the nonlinear code is irrelevant: that
rung is the degenerate bank-floor endpoint and is reported as such, not as a solver result.

---

## Gates before any verdict

| # | gate | tolerance |
| --- | --- | --- |
| i | local smoke: $q=0$ through the new solver reproduces the consolidated saved Burgers case and the incumbent `accuracy_paths.make_rom` | $\le10^{-12}$ relative |
| ii | in-job: $q=0$ reproduces `qlad01` `q0_eq` / `abl01` `a_neural_eq` on all 6 cases | $\le10^{-9}$ relative |
| iii | $q=16$ variable projection agrees with `qlad01` `q16_dense` on the solved field | $\le10^{-3}$ relative, declared here |
| iv | regenerated `directions_sha256` equals `f270e5bf…` | bitwise |
| v | independent NumPy audit recomputes every reported error from saved fields | $<10^{-9}$ |
| vi | every solve carries its exit reason, budget-exit count and the five stationarity norms | present for all |
| vii | in-job: varpro and alternating at $q=0$ are bitwise identical to the joint arm | field SHA256 equal |
| viii | Poisson $q=32$ reproduces `pabl01` `a_neural_q32` at 1024 intervals | $\le10^{-9}$ relative |

Gate iv is bitwise across jobs and GPUs and may legitimately fail on kernel selection alone;
the audited ladder already recorded that no cross-job Burgers field was bitwise identical. The
**substantive** check on the directions is gate iii, which fails if the regenerated $C_q$
differs materially. If gate iv fails it is reported as a bitwise non-reproduction with gate
iii carrying the scientific weight — it is not silently downgraded. The reference FOM fields
are also hashed against `qlad01`'s recorded `field_sha256`, as a free cross-job bitwise probe.

---

## Pre-registered acceptance

**Target (pass/fail).** $q=64$ converged — no budget exits, $g\le10^{-6}$ on every case and
time step — at $\le 3\times$ the $q=0$ median GPU cost on Burgers, where the $q=0$ baseline is
the ladder's own retained $q=0$ rung (`m4`, `eq`), the cheapest converged $q=0$ arm.

**"$q$ is a knob"** requires both:

- worst same-grid error monotone non-increasing in $q$ within the retained configuration;
- at least **3 non-dominated CONVERGED points** on the (worst same-grid error, median GPU ms)
  plane, spanning $\ge2\times$ in cost and $\ge2\times$ in error.

**What would falsify the premise.** Variable projection does not reduce cost (the inner solves
dominate, or the outer LM needs many more iterations because the Kaufman Hessian is poor); the
error stops being monotone once the solves actually converge, showing the audited monotonicity
was an artefact of unconverged solves; fewer than 3 non-dominated converged points; or the
cheap rungs converge but land at a worse error than the audited unconverged ones, showing the
ladder's accuracy came from over-solving rather than from capacity.

The outcome is reported whichever way it falls. No new cases are opened; the final cohorts
stay sealed.

## Routine calls made without asking

- `--gpu a100` requested for both jobs, matching `qlad01`/`pabl01`, to give gate iv and the
  reference-hash probe their best chance.
- EQ is fitted for the `m256` rule at every legal $q$ and for `m2` at $q\in\{256,512\}$ with
  $m=\min(4M,m_{\rm cap})$; the `m4` rule keeps EQ only at $q\in\{0,16\}$, where the retained
  exact fitter reproduces `qlad01`. This bounds offline fitting to roughly 30 minutes.
- The reported reconstruction (three-layer) diagnostics are computed once per $q$ and shared
  across the test-count and quadrature arms at that $q$, because they do not depend on $M$.
- The direction fit is run twice: once with the audited doubly-vectorised code path (whose
  $C$ is the one used downstream, so gate iv is on the exact retained computation) and once
  with the flattened single-`vmap` path, to measure the compilation saving the audited ladder
  flagged and to test whether the two are bitwise equal.
