# ns2d — 2D incompressible Navier–Stokes (vorticity–streamfunction, periodic torus): the second nonlinear test of the correction-rank trade

Lane `ns2d` of the 2026-09-17 ICLR campaign. Branch `exp/2026-09-17-ns2d`, forked from
`exp/2026-09-13-nmrom-consolidated` at `02ff0f1f`. Cluster namespace
`/cluster/tufts/paralab/tawal01/ns_20260917/<attempt>/`. Written and committed **before** the
first cluster job; amended only by appending `§A1, §A2, …`.

## The question

Reviewers GwrW and 5mgh asked for "Navier–Stokes, Euler, … anything nonlinear". The paper's
structural claim is that the correction-rank trade (a monotone error-versus-cost ladder in the
number $q$ of linear correction directions solved jointly with the latent code) exists when the
projected residual is **nonlinear in the bank coefficients**. Burgers is the paper's nonlinear
hero. This lane asks whether the same trade appears on 2D incompressible NS, whose projected
advection term is an exact **degree-2** polynomial in the coefficients (the roadmap principle of
2026-08-30: degree 1 and 2 are sample-free).

The lane is **phased** so that whatever phase is reached is a gated, reportable result:

1. **Phase 1 — FOM + data.** A certified full-order model and a dataset with recorded seeds.
2. **Phase 2 — bank + head.** A separable periodic bank, an auto-decoder head, the bank floor,
   the head oracle, and the classical POD floors on held-out trajectories.
3. **Phase 3 — ROM.** The weak least-squares NM-ROM with the precomputed advection tensor, the
   correction ladder $q\in\{0,16,64,256\}$, POD-LSPG at matched online dimension, and the FOM
   timed in the same job.

**Iterate within a phase until its gates pass; never advance on a failed gate.** If Phase 3
cannot start by 2026-09-22 the deliverable is Phases 1–2.

## Pre-registered pass/fail criterion (Phase 3, the paper question)

With the **exact** tensor (no empirical quadrature, so no quadrature confound), on the eight
opened development cases at the headline mesh:

- **PASS** if the worst-over-cases worst-evolved-time error is **monotone non-increasing** in
  $q$ along $\{0,16,64,256\}$ **and** the top rung reduces it by **at least 2×** relative to
  $q=0$, with the cost ratio of the top rung to $q=0$ reported (Burgers: 3.6× error, 13× cost;
  no cost bar is set, the non-dominated set is the deliverable).
- **FAIL** if the ladder is non-monotone or the top rung improves by less than 2×.
- The verdict is stated for each head separately ($K=16,R=256$ and $K=32,R=512$).

**Falsification clause.** The claim "the trade exists when the projected residual is nonlinear
in the coefficients" is falsified on NS if, with an exact degree-2 residual, the ladder is not
monotone. It is *not* rescued by pointing at a quadrature rule (there is none), at an unconverged
solve (the per-step stopping reasons are recorded and a rung with budget exits is reported as
unconverged), or at the head (the POD-LSPG arm at matched online dimension $k'=K+q$ is run
beside every rung; if POD-LSPG at $k'$ beats the neural rung at $q=k'-K$ everywhere, the head
adds nothing on NS and the report says so).

## Formulation (frozen)

Unit torus $\Omega=[0,1)^2$, periodic. Vorticity $\omega$, streamfunction $\psi$:

$$\partial_t\omega + \mathbf{u}\cdot\nabla\omega = \nu\,\Delta\omega + f,\qquad
\mathbf{u}=\nabla^\perp\psi=(\partial_y\psi,\,-\partial_x\psi),\qquad \Delta\psi=-\omega .$$

With $J(\psi,\omega)=\psi_x\omega_y-\psi_y\omega_x$ one has $\mathbf{u}\cdot\nabla\omega=-J(\psi,\omega)$, so

$$\partial_t\omega = J(\psi,\omega) + \nu\Delta\omega + f .$$

Divergence-freeness is automatic in this form; the Stokes cell's "stream-function bank" becomes,
here, the precomputed stream-function bank $\Psi=-\Delta_h^{-1}G$ of a scalar vorticity bank $G$.
$\omega$ has zero mean; $\psi$ is fixed in the mean-zero gauge.

### Discretisation (frozen)

- Grid: $N\times N$ nodes $x_i=ih$, $h=1/N$, $N\in\{64,128,256\}$ (32 for local smokes,
  512 only inside the refinement gate). State $\omega\in\mathbb{R}^{N^2}$.
- Laplacian $\Delta_h$: 5-point, periodic. Exactly diagonalised by the 2D DFT with
  $\lambda_{\mathbf{k}}=\frac{2}{h^2}\big(2-\cos(2\pi k_x/N)-\cos(2\pi k_y/N)\big)$, so
  $\psi=-\Delta_h^{-1}\omega$ is an **exact** FFT solve (mean mode zero).
- Velocity (reporting, CFL, energy): centred differences $u=\delta_y\psi$, $v=-\delta_x\psi$.
- Advection: the **Arakawa (1966) Jacobian** $J_A=\tfrac13(J^{++}+J^{+\times}+J^{\times+})$,
  second order, exactly bilinear, and conserving $\sum J_A=\sum\omega J_A=\sum\psi J_A=0$
  identically on the periodic grid.
- Time: **implicit midpoint** (Crank–Nicolson on every term, advection included),
  $$\frac{\omega^{n+1}-\omega^n}{\Delta t}=J_A(\psi_m,\omega_m)+\nu\Delta_h\omega_m+f(t_{n+1/2}),
  \qquad \omega_m=\tfrac12(\omega^n+\omega^{n+1}),\ \psi_m=-\Delta_h^{-1}\omega_m,$$
  solved by tolerance-terminated Newton with matrix-free BiCGStab preconditioned by the exact
  FFT Helmholtz inverse $(I-\tfrac{\Delta t\,\nu}{2}\Delta_h)^{-1}$ — the `engines.make_fom`
  pattern of the Burgers hero. Newton stops at $\|F\|\le \mathrm{ntol}\,\|\omega^n\|$.
- **Why fully implicit and not IMEX (a deliberate deviation from the pre-registration's
  "explicit RK/AB2 for advection").** An IMEX residual is *linear* in $\omega^{n+1}$, so the
  projected residual would be linear in the new coefficients and the correction block $y$
  could be eliminated analytically — exactly the Poisson/heat situation in which the top rung is
  a linear ROM and the lane would not test the claim it exists to test. Implicit midpoint keeps
  the residual quadratic in $h^{n+1}$, matches the Burgers hero's implicit pattern, and at
  $\nu=0$ conserves energy and enstrophy to Newton tolerance, which gives a machine-precision
  budget gate.
- $\Delta t=2\times10^{-3}$, horizon $T=1$ (500 steps), outputs at $t\in\{0,0.2,\dots,1.0\}$,
  training snapshots every $0.04$ (26 states per trajectory). The scheme is A-stable; the
  measured CFL $\max|\mathbf u|\Delta t/h$ is recorded per trajectory and temporal convergence
  is gated (below), not assumed.

### Data family (frozen)

Decaying flow from a **band-limited random initial vorticity**:
$$\omega_0(\mathbf x)=\sum_{\mathbf k\in\mathcal K}\big(a_{\mathbf k}\cos 2\pi\mathbf k\cdot\mathbf x+b_{\mathbf k}\sin 2\pi\mathbf k\cdot\mathbf x\big),\qquad
\mathcal K=\{(1,0),(0,1),(1,1),(1,-1),(2,0),(0,2)\},$$
$a,b\sim\mathcal N(0,1)$ i.i.d. (12 real parameters), then rescaled so that the root-mean-square
velocity $U_{\rm rms}=1$, so the Reynolds number is $\mathrm{Re}=U L/\nu=1/\nu$ with
$\nu\sim\log\mathcal U[10^{-3},10^{-2}]$, i.e. $\mathrm{Re}\in[100,1000]$. $f=0$. Intrinsic
dimension $11+1$ (+time), so $K=16$ is not vacuous and $K=32$ has room; a many-mode "turbulence"
initial condition would be unlearnable at these $K$ and is not used. Kolmogorov forcing is an
optional stretch arm, not run unless Phase 3 completes.

Cohorts, all drawn column-by-column from `numpy.random.default_rng(seed)` (the `params_draw`
landmine): **train** 512 trajectories (seed 20260917), **dev** 64 (seed 20260918), **sealed** 64
(seed 20260919, opened only after every gate and only if time remains). The ROM's timed cases are
the first 8 dev trajectories. Data are **regenerated from the seed inside every job**; Phase 1
records the SHA256 of every cohort at every mesh and every later job asserts it.

### Bank, head, test space, ROM (frozen)

- **Bank** $G(\mathbf x)\in\mathbb R^{n\times R}$: the separable coordinate network of
  `sep_common` with **integer** Fourier features (periodic by construction), no boundary mask,
  each column made mean-zero on the grid. $R\in\{256,512\}$.
- **Head** $h:\mathbb R^K\to\mathbb R^R$, MLP + linear skip, $K\in\{16,32\}$; **auto-decoder**
  training (joint Adam over $g,h$ and per-snapshot codes), the Burgers recipe. Two arms:
  $(K,R)=(16,256)$ and $(32,512)$, trained on the 256² training cohort.
- **Query-time latent fit**: least squares of the supplied $\omega_0$ on the grid in the whitened
  bank metric (thin QR of $G$), LM from the best-scoring training codes.
- **Test space**: the $M$ lowest real Fourier modes $\phi_m$ ordered by $|\mathbf k|^2$, which are
  exact eigenvectors of $\Delta_h$, so every linear term is exact and diagonal.
  $M=4(K+q)$ for the neural rungs, $M=4k'$ for POD-LSPG.
- **Residual** (implicit midpoint, projected, scaled by the Helmholtz diagonal):
  $$r(h)=\frac{Ah-Ah^n-\Delta t\big[T(h_m,h_m)-\nu\lambda\odot Ah_m\big]}{1+\Delta t\,\nu\lambda/2},
  \qquad A=\Phi^\top G,\quad T_{mjk}=\sum_{\mathbf x}\phi_m(\mathbf x)\,J_A(\psi_j,g_k)(\mathbf x),$$
  with $\Psi=-\Delta_h^{-1}G$ and $T$ built **once, blocked over $\mathbf x$**, gated by two
  chunkings (TB) and against the dense full-grid oracle at held-out latent states (TQ). Because
  $J_A$ is bilinear, $T$ is **exact**, not a positivity-restricted approximation.
- **Correction ladder**: $h(w)=h_\theta(z)+C_q y$, $w=(z,y)\in\mathbb R^{K+q}$, $C_q$ the nested
  field-metric POD of the decoder-output residual on training snapshots (`ladder.residual_directions`).
- **Controls**: POD-LSPG on the classical POD basis of the training snapshots at
  $k'\in\{16,64,256\}$ and at $k'=K+q$ for every rung; the FOM Newton–Krylov tolerance ladder
  timed in the same job; the same-grid converged FOM (ntol $10^{-11}$) as the error reference.
- **Errors** (Burgers convention): $\|\omega_{\rm rom}(t)-\omega_{\rm fom}(t)\|_2/\|\omega_{\rm fom}(0)\|_2$,
  reported as worst over evolved times ($t>0$), worst over all times, and the $t=0$ compression
  separately; worst and median over cases.
- **Timing**: three timed repetitions after burn-in, device sync, AB/BA balanced order, medians
  reported with every repetition retained; no ratio across jobs or GPUs.

## Gates

Every gate is a number with a threshold and, where it can be built, a negative control.
Thresholds are relative; nothing absolute on a mesh-scaling quantity.

### Phase 1 (job 1)

| gate | what | pass rule | negative control |
|---|---|---|---|
| S0 | backend, f64, matmul precision | `jax_backend=gpu`, x64, `highest` asserted | — |
| F-LAP | $\Delta_h$ FFT inverse is exact | $\|\Delta_h(-\Delta_h^{-1}\omega)+\omega\|/\|\omega\|\le10^{-12}$ on random mean-zero $\omega$ at every $N$ | — |
| F-JAC | Arakawa conservation | $|\sum J_A|,|\sum\omega J_A|,|\sum\psi J_A|$ normalised by $\sum|\omega||J_A|$ etc. $\le10^{-13}$; order 2 against the analytic $J$ of a smooth pair | a plain centred-difference Jacobian must **fail** the $\omega J$ identity at $\ge10^{-3}$ |
| F-TG | Taylor–Green, closed-form **discrete** solution $\omega_h(t)=\omega_0 e^{-\nu\lambda_h t}$ (Arakawa gives $J_A(\psi,c\psi)=0$ exactly) | FOM vs discrete-exact $\le10^{-8}$ relative at $N=128$, $T=1$, $\nu=10^{-2}$ (time error only; $\Delta t$-refined value recorded); FOM vs **continuum** TG at $N=128$ $\le10^{-6}$ with $\Delta t=2\times10^{-3}$ and observed spatial order $2.00\pm0.05$ over $64\to128\to256$ | flipping the Laplacian sign, or a 4/3-scaled $\lambda$, must fail |
| F-MMS | manufactured solution with $J\ne0$: $\omega_{\rm ex}=\sin2\pi x\cos2\pi y\,(1+\tfrac12\sin 2\pi t)+\tfrac12\cos 4\pi x\sin 2\pi y$, source $f=\partial_t\omega-J(\psi,\omega)-\nu\Delta\omega$ from the analytic $\psi=-\Delta^{-1}\omega$ | spatial order $2.00\pm0.05$ over $64\to128\to256$ at fixed small $\Delta t$; temporal order $2.0\pm0.1$ at $N=128$ over $\Delta t,\Delta t/2,\Delta t/4$ | **flipping the sign of $J_A$ must fail at $O(1)$** (the one gate that fixes the advection sign) |
| F-BUDGET | discrete enstrophy identity $Z^{n+1}-Z^n=\Delta t\,h^2\sum\omega_m(\nu\Delta_h\omega_m+f_m)$ and energy identity, per step | $\le10^{-10}$ relative on a dev trajectory at every $N$; at $\nu=0,f=0$: $|Z^n-Z^0|/Z^0$ and $|E^n-E^0|/E^0\le10^{-9}$ over 100 steps | backward Euler on the same problem must **not** conserve ($\ge10^{-4}$) |
| F-MESH | refinement of the family | observed order $2.0\pm0.1$ on a dev trajectory at $\nu=10^{-2}$ over $64\to128\to256\to512$ (error at $t=T$ against 512²); the $\nu=10^{-3}$ order is **reported** (64² may be under-resolved at Re 1000; that is a finding, not a failure, if 128→256→512 gives $2.0\pm0.15$) | — |
| F-INDEP | an **independent NumPy/SciPy** implementation (own stencils, own FFT solve, own Newton with a sparse-LU linear solve) at 64² on one dev trajectory | $\max_t\|\omega_{\rm jax}-\omega_{\rm np}\|/\|\omega_{\rm np}\|\le10^{-10}$ at ntol $10^{-13}$ | — |
| F-CFL | $\max|\mathbf u|\Delta t/h$ per trajectory | recorded; must be $\le 2$ at 256² for every trajectory; temporal order gate above is the accuracy guarantee | — |
| F-NEWTON | Newton/BiCGStab convergence on every generated step | zero budget exits; worst relative residual $\le$ ntol | — |
| F-DATA | cohort hashes at 64², 128², 256² | recorded; train/dev/sealed disjoint (no two draws `allclose`) | — |

### Phase 2 (jobs 2–3, one per head)

| gate | pass rule |
|---|---|
| B-DATA | regenerated cohort SHA256 equals Phase 1's |
| B-ORTH | thin-QR whitening of $G$ on the grid succeeds with $\mathrm{rank}=R$; $\kappa(R_b)$ recorded |
| B-FLOOR | bank projection floor on **held-out dev** trajectories (worst over times, relative) $\le 2\times$ the classical POD-$R$ truncation floor of the training snapshots evaluated on the same held-out trajectories |
| H-ORACLE | best-found multi-start LM fit on held-out fields: median $\le \tfrac12\times$ the POD-$K$ (linear, matched online dimension) held-out error, and $\ge$ the bank floor (an oracle below the floor is a bug) |
| H-SOLVED | query-time initialiser fit at $t=0$ within $1.5\times$ the oracle median |
| H-TRAIN | training reconstruction (mean and max relative $L_2$) recorded; no NaN |

### Phase 3 (jobs 4–5, then 6–8 if time)

| gate | pass rule |
|---|---|
| R-TB | tensor by two chunkings agree $\le10^{-12}$ relative |
| R-TQ | $T(h,h)$ vs dense $\Phi^\top J_A(\Psi h,Gh)$ at 32 held-out latent states $\le10^{-10}$ relative |
| R-LIN | exact linear terms vs dense $\Phi^\top\Delta_h G$ $\le10^{-12}$ |
| R-FOM | FOM ladder: every rung's same-grid discrepancy against the converged FOM recorded; the converged FOM has zero budget exits |
| R-CONV | per-step LM stopping reasons per rung; a rung with budget exits is flagged unconverged |
| R-LADDER | the pre-registered criterion above |

## Arms and job plan (cap 8)

| job | attempt | phase | content | GPU |
|---|---|---|---|---|
| 1 | `ns101` | 1 | FOM gates at 64/128/256(+512 in F-MESH), cohorts at 64/128/256 with hashes, F-INDEP | a100 |
| 2 | `ns201` | 2 | head $(16,256)$ on 256² train; floors and oracles on dev at 64/128/256 | a100 |
| 3 | `ns202` | 2 | head $(32,512)$, same | a100 |
| 4 | `ns301` | 3 | ROM ladder, head $(16,256)$, 256² | a100 |
| 5 | `ns302` | 3 | ROM ladder, head $(32,512)$, 256² | a100 |
| 6–8 | spare | 1–3 | reruns after a failed gate; then 128², 64² for the better head | a100 |

Local (32²) smokes only, through the slot helper. Queue policy per the lane protocol.

## What this lane does not claim

No speed-up is promised: on Burgers the honest ratio is ~1.8×, and 500 implicit steps with a
per-step LM will not be cheaper than an FFT-preconditioned Newton–Krylov FOM at 256². The
deliverable is the ladder (error versus cost versus $q$) and the non-dominated set against
POD-LSPG and the FOM tolerance ladder. If Phase 3 is not reached, the deliverable is a certified
FOM, a hashed dataset, and the bank/head floors — an appendix and a foundation.

## Glossary

- **FOM / ROM**: full-order model (the grid solve) / reduced-order model.
- **Bank / head / latent code**: learned spatial features $G$ / neural map $h$ from the code $z$ to bank coefficients / the $K$ numbers solved online.
- **Correction rank $q$**: number of extra linear directions solved jointly with $z$.
- **Test modes $M$**: Fourier modes the residual is projected onto (weak least squares).
- **Tensor**: the precomputed $M\times R\times R$ array giving the projected advection exactly.
- **POD-LSPG**: classical linear reduced model minimising the same weak residual on a POD basis.
- **Oracle / best-found**: best latent fit to a known field by multi-start LM; an upper bound on ROM accuracy.
- **Bank floor / POD floor**: error of projecting a field onto the bank span / onto the POD span.
- **Dev / sealed cohort**: trajectories used for evaluation now / held unopened.
- **Arakawa Jacobian**: the energy- and enstrophy-conserving finite-difference form of $J(\psi,\omega)$.
- **Implicit midpoint**: Crank–Nicolson with the nonlinear term at the midpoint state.

## §A1 (2026-09-17, before job 1) — the Taylor–Green gate, corrected for a second-order scheme

The F-TG row above copied the coordinator's "matches the analytic solution to $\le10^{-6}$ at
128²", which is a pseudo-spectral number. For the frozen second-order scheme the discrete
Laplacian eigenvalue of the TG mode is $\lambda_h=4N^2(1-\cos 2\pi/N)$ versus $8\pi^2$, so at
$N=128$, $\nu=10^{-2}$, $T=1$ the *continuum* discrepancy is $\nu T(8\pi^2-\lambda_h)\approx1.6\times10^{-5}$
by construction and cannot be $10^{-6}$. The gate is therefore restated, sharper:

- **F-TG-exact**: with Arakawa $J_A(\psi,c\psi)=0$, TG has the closed-form fully discrete
  solution $\omega^n=\big(\tfrac{1-a}{1+a}\big)^n\omega_0$, $a=\tfrac{\Delta t\,\nu\lambda_h}{2}$.
  The FOM must match it to $\le10^{-10}$ relative at $N\in\{64,128,256\}$ (this is a Newton-tolerance
  identity, and it tests the Laplacian, the Poisson solve, the Jacobian's antisymmetry and the
  time scheme together). Negative controls: a wrong $\lambda$ (scaled by 4/3) and the backward
  Euler scheme must both fail at $\ge10^{-4}$.
- **F-TG-semi**: FOM versus the semi-discrete $\omega_0e^{-\nu\lambda_h t}$ at $\le10^{-6}$
  (the time-discretisation error alone, predicted $\approx1.6\times10^{-7}$ at $\Delta t=2\times10^{-3}$).
- **F-TG-cont**: FOM versus continuum TG: observed spatial order $2.00\pm0.05$ over
  $64\to128\to256$, **and** the measured error at each $N$ must agree with the closed-form
  prediction $|e^{-\nu T\lambda_h}-e^{-8\pi^2\nu T}|/e^{-8\pi^2\nu T}$ to within 2 % — a
  prediction-matching gate, not an absolute threshold.

The coordinator's $10^{-6}$ is met by F-TG-semi (and exceeded by F-TG-exact); the continuum
comparison is a convergence-order gate, as it must be for a finite-difference scheme.

## §A2 (2026-09-17, during job 3780151, before Phase 2) — the F-JAC negative control was an absolute threshold on a mesh-scaling quantity

F-JAC required the plain centred-difference Jacobian to violate the $\sum\omega J$ identity by
$\ge10^{-3}$. That control's leak is itself $O(h^2)$ for a smooth pair: job 3780151 measured
2.0e-2, 4.2e-3, (64²: see report), 2.4e-4, 5.9e-5 at $N=16,32,\dots,256$, so the gate "failed"
at 128² and 256² while the Arakawa identities read 2–4e-18 — the control still discriminates
by thirteen orders of magnitude. This is the recurring failure mode the design itself names
(absolute tolerances for quantities that scale with the mesh). Restated as a **ratio**:

- F-JAC passes when every Arakawa identity and the antisymmetry defect are $\le10^{-13}$ **and**
  the centred control's $\sum\omega J$ violation is $\ge10^{6}\times$ the Arakawa
  $\sum\omega J$ violation on the same pair.

The stored numbers of job 3780151 are re-evaluated under this rule by `audit_phase1.py`
(recorded as `passed_A2` beside the original verdict); the driver uses the ratio rule from
commit §A2 onward. The scheme is unchanged; only the gate arithmetic is amended, and the
original verdict is kept in the JSON.
