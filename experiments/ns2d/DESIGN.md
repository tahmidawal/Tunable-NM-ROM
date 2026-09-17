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

## §A3 (2026-09-17, before Phase 3) — the advection tensor is projected with FFTs, and the pre-registered ladder is therefore affordable

The residual's test modes are, by the frozen contract, the lowest real Fourier modes, which
are exact eigenvectors of $\Delta_h$. Projecting any field on them is therefore an **FFT**, not
a dense $(M,n)$ matmul. The direct build costs $24\,n\,M\,R^2$ FLOPs — $4.7\times10^{14}$ at
$n=256^2$, $M=1152$, $R=512$, which is hours of A100 time and would have forced the top rung
to be cut. The FFT route costs $O(24\,n\,R^2 + R^2 n\log n)$ and measures **2.3 s at
$N=256$, $M=256$, $R=128$ on the (slow, shared) local GB10**, so the full $q=256$, $R=512$
tensor is minutes on an A100.

This changes **no science**: the two builds are the same exact quantity. `build_T` (direct) is
retained as the independent reference, and the new gate

- **R-TFFT**: the FFT build against the direct $(M,n)$-matmul build on a $32\times16\times16$
  sub-block, $\le10^{-12}$ relative,

is asserted in-job beside R-TB (two block orders) and R-TQ (against the dense full-grid
oracle). Verified locally before the amendment: direct vs FFT $6.6\times10^{-16}$ at
$N=16$ and $1.1\times10^{-15}$ at $N=32$; tensor vs dense oracle $8.6\times10^{-16}$ and
$1.8\times10^{-15}$.

**The pre-registered ladder $q\in\{0,16,64,256\}$ and both heads stand unchanged.** No rung is
cut for cost.

## §A4 (2026-09-17, after job 3783796 `ns201`, before any rerun) — the bank was structurally rank-capped at `g_hidden`; B-DATA becomes a hash-or-value gate; oracle errors are evaluated in field space

**What happened.** `ns201` ($K=16$, $R=256$) completed (1 h 32 m, A100 `pax049`, exit 0) and
failed three families of gates:

1. **B-ORTH at every mesh: numerical rank 128 of $R=256$**, $\kappa(R_b)\approx 2\times10^{15}$.
   The cause is structural, not numerical: the bank is
   $G=\text{out\_scale}\cdot\mathrm{MLP}_g(\text{features})$ and the last layer of
   $\mathrm{MLP}_g$ is **linear** over `g_hidden`$=128$ hidden units, so
   $\operatorname{rank}G\le\min(R,\texttt{g\_hidden})=128$ regardless of $R$.
   The parent lane paid for exactly this landmine on 2026-08 (`sep_burgers_r3.py`: "numerical
   rank was capped at `g_hidden + 1 = 257` because the g-track's last layer is linear; the fix
   is `G_HIDDEN >= 2R`") and its later drivers default `G_HIDDEN = 2R`. This lane's decoder
   inherited the old default 128. Both Phase-2 heads (`ns201`, and `ns202` still running with
   $R=512$) are rank-128 banks. The independent audit (`audit_phase2.py`) recomputes rank 128
   from the pickled weights at $N=64,128,256$.
2. **H-ORACLE at every mesh**: oracle median 0.203 vs POD-16 median 0.240 (ratio 1.19, bar 2.0);
   bank floor median 0.038. The head, not the span, is the limit — at 30 000 full-batch steps
   with a $128\times2$ head, which the parent lane's `HFIT.md` identifies as the short/narrow
   corner (25k-step arms were ~2× worse than 120k-step arms; latent Fourier features made
   things 10–50× *worse* and are **not** adopted).
3. **B-DATA at 256²** (train and dev) by hash, while 64² and 128² matched. `ns202` on `pax106`
   matched all Phase-1 hashes (from `pax105`) with the same commit; `ns201` on `pax049` did not.
   The 256² cohort hash is therefore **node-dependent at the bit level** (the campaign's
   "value gates, not hash gates, across machines" landmine), not a code change.

**A fourth finding from the audit.** The driver reported the oracle error as
$\sqrt{r_n^2+\|u-P u\|^2}/n_0$ with $r_n=\|R_b(h(z)-c)\|$ and $c=R_b^{-1}Q_b^\top u$. Through a
singular $R_b$, $c$ carries $\sim10^{12}$ garbage in the null directions and $R_b(h-c)$ cancels
catastrophically: the recomputed exact $\|Gh(z)-u\|/n_0$ differs from the reported value by up
to **1.1e-2 relative at 64², 7.6e-3 at 128², 3.0e-3 at 256²** on the 48 archived states
(the rank-128 projection formula agrees with the exact norm to all digits). No verdict changes
(0.203 vs the 0.120 needed), but the `ns201` oracle numbers carry that contamination and the
LM minimised a cancellation-noisy objective. The audit's six `e_oracle`/`e_single` checks
are recorded as MISMATCH for `ns201`; that is the correct record.

**Amendments (the science is unchanged; the model's implementation and two gates are fixed).**

- **B-RANKCAP** (new, asserted before any GPU time is spent): `g_hidden >= R`; the driver's
  default becomes `G_HIDDEN = 2R`, the parent lane's verified fix.
- **Phase-2 training recipe for the reruns**: `G_HIDDEN = 2R`, head $512\times3$, 100 000
  full-batch steps (same optimiser, schedule, seed, cohorts, `LAM_ORTH`). Nothing else changes.
  Cost estimate from `ns201`/`ns202` step times: ≈3.5 h ($K=16$) and ≈6.5 h ($K=32$) of A100.
- **B-DATA / R-DATA become hash-or-value gates.** Hash equality passes as before. On a
  mismatch, the dev cohort passes iff every one of the first 8 dev trajectories at the six
  evaluation times agrees with Phase 1's archived fields (`configs/dev8_eval_ref.npz`, cut from
  job 3780151's `dev8_N*.npz`) to **≤ 1e-8 relative per state**; the train cohort (no archived
  fields) passes iff the dev cohort at the same mesh passed on the same node, with the
  mismatch recorded in the JSON (`mode`, `hash_mismatch`, `value_worst_rel`). Dev is generated
  first so that the node is certified before the train cohort's gate is read. Phase 3 gates its
  8-case converged reference by value directly (`R-DATA_reference`).
- **Oracle, single-start and manifold errors are evaluated in field space**,
  $\|Gh(z)-u\|_2/\|u_{\rm case}(0)\|_2$, and the whitened-formula value is stored beside them
  (`formula_vs_field_worst_rel`), so the report never depends on $R_b^{-1}$.
- **`ns201` is closed as a failed Phase-2 attempt**, archived (`artifacts/ns201`), audited, and
  not used for Phase 3. `ns202` will be gated when it lands; its B-ORTH fails by construction.
- **Job plan**: jobs 4–5 become the Phase-2 reruns `ns203` ($K=16,R=256$) and `ns204`
  ($K=32,R=512$); Phase 3 moves to jobs 6–7; one spare. The 2026-09-22 stop rule stands.

**Pre-registered expectation for the reruns, stated before they run.** Full rank $R$ at every
mesh (B-ORTH passes by construction); the bank floor should drop below `ns201`'s 0.038
median; H-ORACLE is the open question — the parent lane's evidence says longer training is
worth ~2×, which would put the ratio near the 2.0 bar. If H-ORACLE fails again with a full-rank
bank and 100k steps, the finding is that this auto-decoder head cannot beat linear POD-$K$ by 2×
on decaying 2D NS at $\mathrm{Re}\in[100,1000]$, and the lane reports Phases 1–2 with that
negative — it does not lower the bar.

## §A5 (2026-09-17) — Codex unavailable: written self-audits substitute for the independent-model audits

`codex exec` is quota-blocked until 2026-09-19 11:33 (coordinator notice). In its place:
`reports/self-audit-fom-verification.md` (Phase 1) and `reports/self-audit-phase2.md` (Phase 2)
list each claim, the JSON field it rests on, and the check run, and are labelled as
self-audits. The independent NumPy audits (`audit_phase1.py`, `audit_phase2.py`,
`audit_phase3.py`) remain the mechanical check. The Codex report audit is to be run after
2026-09-19 11:33 if the lane is still open.

## §A6 (2026-09-17 18:3x EDT, after job 3787319 `ns203`) — H-ORACLE fails with a full-rank bank: the pre-registered negative finding for the $K=16$ head; Phase 3 is not submitted for it

**What happened.** `ns203` ($K=16$, $R=256$, `G_HIDDEN=512`, head $512\times3$, 100 000
full-batch steps; A100 `pax050`, 4 h 40 m, exit 0, `jax_backend=gpu`, `ALL-DONE`) passed
every Phase-2 gate that §A4 fixed — B-RANKCAP, B-DATA by hash at all three meshes on this node,
B-ORTH with numerical rank 256 and $\kappa(R_b)=44$ at $N\in\{64,128,256\}$, B-FLOOR with the
bank floor now *below* the POD-256 floor per state (median 0.012 vs 0.033) and within 1.08× of
it on the worst evolved state, H-SOLVED, H-TRAIN — and **failed H-ORACLE at every mesh with the
same ratio as the rank-capped `ns201`: POD-16 median / oracle median = 1.19** (0.240 / 0.202;
bar 2.0). The independent NumPy audit (`artifacts/ns203/audit.json`) matches all 67 checks;
the field-space and whitened-formula oracle values agree to $5\times10^{-14}$, so the §A4
contamination is gone and the number is real.

**Per §A4, stated before the run: this is the negative finding, and the bar is not lowered.**
This auto-decoder head (separable periodic bank + MLP head, Burgers recipe) cannot beat linear
POD-$K$ by 2× on decaying 2D NS at $\mathrm{Re}\in[100,1000]$ with $K=16$. Phase 3 is **not
submitted** for the $(16,256)$ head. The $(32,512)$ head's verdict is stated separately when
`ns204` lands (its rank-capped predecessor `ns202` had ratio 1.18).

**Diagnostic, from the stored per-state arrays (generated into the report, not typed).** The
failure is *generalisation*, not capacity:

- Training reconstruction median fell from 0.206 (`ns201`, 30k steps, $128\times2$ head) to
  **0.050** (`ns203`); the held-out oracle median did not move (0.2028 → 0.2024). The
  held-out / training ratio is **4.0**.
- By output time: at $t=0$ the oracle beats POD-16 by **1.63×** (0.026 vs 0.042); on every
  evolved time it is 1.1–1.4× (e.g. $t=0.4$: 0.279 vs 0.305). The initial condition is a
  12-parameter band-limited field that $K=16$ codes cover; the evolved states depend on
  (12 amplitudes, $\nu$, $t$), a 14-dimensional input, and **512 training trajectories do not
  cover it** ($512^{1/14}\approx1.6$ points per axis). The head memorises the training
  trajectories (0.050) and has no neighbours for a held-out draw.
- The bank is not the limit (floor median 0.012 at 256²), and neither is the fit: oracle LM
  median 39 iterations, no budget exits; single-start median within 1.10× of the oracle.

**What this means for the paper question.** The lane's Phase-3 test (does the correction-rank
trade exist on a degree-2 residual) is not reached for $K=16$, because its precondition — a
head whose manifold is closer to the held-out data than linear POD-$K$ — does not hold for
this family with this data budget. That is a finding about the data family (12 i.i.d. Gaussian
amplitudes is a high-dimensional family for 512 trajectories), not a rescue.

**Not done, and why.** Lowering the H-ORACLE bar, re-scoring on training trajectories, or
opening the sealed cohort would each turn a pre-registered fail into a pass by changing the
rule after the number was seen; none is done. Retraining with more trajectories or a
lower-dimensional family (e.g. 3 modes, 6 amplitudes) would be a *new* Phase 2 with a new
pre-registration and 2–3 more jobs; with 5 of 8 jobs used and the 2026-09-22 stop rule, that is
the coordinator's call, not this session's.

**Job plan after §A6.** Jobs used: `ns101`, `ns201`, `ns202`, `ns203`, `ns204` (5 of 8).
`ns204` pending. No Phase-3 job is staged.

## §A7 (2026-09-17 evening, after job 3787320 `ns204`) — the $K=32$ head also fails H-ORACLE with a full-rank bank; Phase 2 is closed for both heads and Phase 3 is not submitted

**What happened.** `ns204` ($K=32$, $R=512$, `G_HIDDEN=1024`, head $512\times3$, 100 000
full-batch steps; A100 `pax049`, 7 h 30 m, exit 0, `jax_backend=gpu`, `ALL-DONE`, commit
`beffbb1b`) passed B-RANKCAP, B-DATA by hash at every mesh, **B-ORTH with numerical rank
512 = R and $\kappa(R_b)=141$**, B-FLOOR (bank worst-evolved 0.0756 vs POD-512 0.0600, ratio
1.26; bank floor median 0.0040, *below* the POD-512 median 0.0115), H-SOLVED (single-start
0.1270 within 1.05× of the oracle) and H-TRAIN (reconstruction median 0.0376) — and **failed
H-ORACLE at every mesh: oracle median 0.1209 vs POD-32 median 0.1395, ratio 1.15** (64²:
0.1218/0.1407; 128²: 0.1211/0.1396; bar 2.0). The independent NumPy audit
(`artifacts/ns204/audit.json`) matches all 67 checks. Oracle LM median 65 iterations of a
300 budget; stopping-reason codes {0: 18, 4: 366} states (the driver's code map is in
`ns2d_phase2.py`).

**Per §A4 this is the negative finding for the second head; the bar is not lowered and no
Phase-3 job is staged.** Both pre-registered heads now carry the same verdict: with a
full-rank bank and the long recipe, the auto-decoder head cannot beat linear POD-$K$ by 2× on
held-out decaying 2D NS at $\mathrm{Re}\in[100,1000]$.

**Diagnostic (generated into the report from the per-state arrays).** Same mechanism as §A6,
and sharper: held-out / training = **3.2** (0.1209 vs 0.0376); on evolved times POD-32 / oracle
= 1.16; **at $t=0$ POD-32 is better than the oracle (0.0189 vs 0.0230, ratio 0.82)** — at $K=32$
even the initial condition is fit better by 32 linear modes than by the 32-code neural
manifold, because the 12-parameter initial family is inside the span of 32 POD modes of the
training snapshots. The bank floor (median 0.004) is not the limit.

**Consequence for the lane.** Phase 2 is closed as a gated negative for both $(16,256)$ and
$(32,512)$. The Phase-3 question (correction-rank trade on a degree-2 residual) is not
reached; the deliverable is Phases 1–2, the certified FOM and dataset, the floors, and this
negative. Any further Phase 2 (lower-dimensional family, more trajectories) is a new
pre-registration; jobs used 5 of 8; the 2026-09-22 stop rule stands.

## §A8 (2026-09-17, correction to §A6/§A7) — 3–5 % of oracle fits are LM budget exits, not "none"

§A6 wrote "no budget exits" for `ns203`; that was read off the `oracle_reasons` dict without
consulting the code map. `ns2d_decoder.make_lm_fit` defines reason 0 = budget (300 iterations),
4 = stationary (gradient ≤ 1e-6). Stored counts over 384 held-out states: `ns203` {0: 12, 4: 372}
at 256² (11–12 at the other meshes); `ns204` {0: 18, 4: 366} at 256² (12, 16 at 64², 128²). So
3.1 % / 4.7 % of states hit the cap. The oracle values on those states are upper bounds on the
best fit. The medians over 384 states and both verdicts stand (ratio 1.19 / 1.15 vs 2.0; even
placing every budget-exit state at the bank floor would move the median by at most 18 ranks).
The report table now carries the count per mesh (`oracle.budget_exits_of_states` in
`summary.json`). Recorded as a correction, not silently edited.

## §A9 (2026-09-17 ~19:30 EDT, user decision "keep pushing", cap raised 8 → 12) — a NEW Phase-2 cell: diagnose the generalisation failure and test its two candidate causes; one exploratory ladder on the best manifold

**Status of what came before.** The §A4/§A6/§A7 negative stands as recorded: the pre-registered
heads $(16,256)$ and $(32,512)$, trained by the Burgers auto-decoder recipe on 512 trajectories of
the 12-amplitude family, do not beat POD-$K$ by 2× on held-out states, and Phase 3 was not run
for them. Nothing below re-scores, re-gates or rescues `ns203`/`ns204`. This is a new cell with
its own question, pre-registered before any of its jobs is submitted.

**The diagnosis so far (§A6/§A7, generated numbers).** The held-out/training gap is 3–4×; the
head beats POD-16 at $t=0$ (1.63×) but not on evolved states (1.19×); the bank floor is 10–30×
below the oracle. Two candidate causes, not exclusive: (D) **data** — 512 trajectories do not
cover the 14-dimensional (12 amplitudes, $\nu$, $t$) input; (H) **head** — the MLP head (this
function class / this recipe) cannot interpolate the manifold even when the data does cover it.

### Arms (four jobs, submitted in parallel, one attempt directory each)

| attempt | driver | what | question |
|---|---|---|---|
| `ns301` | `ns2d_headfit.py` | head-only on the **frozen `ns203` bank** ($K=16$, $R=256$): nested subsets of the gated 512-trajectory cohort, $n\in\{128,256,512\}$ (the first $n$), × regimes {`plain`: the ns203 head 512×3, Adam, warmup-cosine, 100k steps, last iterate; `reg`: same head, AdamW weight decay $10^{-4}$ on the head weights, dev-select early stopping every 5000 steps; `reg_small`: 128×2 head, otherwise `reg`}; 9 arms in one job | does the held-out oracle improve with trajectory count, at what log-log slope, and does regularisation move it |
| `ns302` | `ns2d_phase2.py` | **4× data**: 512 base trajectories (seed 20260917, gated by hash) + 1536 from seed 20260920 (hash recorded), same family, $K=16$, $R=256$, `G_HIDDEN=512`, head 512×3, 100k steps; per-step batch 13 312 rows (= ns203's full batch) because 53 248 snapshots × 65 536 points do not fit three times on an 80 GB device | H-ORACLE vs POD-16 of the same 2048 trajectories, bar 2.0; the ratio at 1.5 recorded (`passes_at_1p5`) so the slope is visible |
| `ns303` | `ns2d_phase2.py` | **lower-dimensional family**: 3 modes $\{(1,0),(0,1),(1,1)\}$, 6 amplitudes, same $\nu$ range, 512 trajectories, otherwise the ns203 recipe | does the head pass H-ORACLE when the input is 8-dimensional (6 amplitudes, $\nu$, $t$) |
| `ns304` | `ns2d_phase3.py` | **exploratory ladder on the `ns204` manifold** ($K=32$, $R=512$, the best trained): $q\in\{0,32,64,128,256,512\}$ at one fixed test space $M=2176=4(K+q_{\max})$ for every subject; POD-LSPG at $k'\in\{32,512\}$ and at every matched $K+q$; the FOM tolerance ladder timed in the same job; three timed reps; three-layer decomposition | does the correction rank close the gap between the $q=0$ manifold (oracle 0.12) and the bank floor (0.012) — the paper's mechanism — independently of whether $q=0$ beats POD-32 |

### Pre-registered readings (stated before the jobs run)

- **ns301.** For each regime the dev-report oracle median at $n=128,256,512$ and the least-squares
  log-log slope. Reading rule: slope $\le-0.25$ under any regime (2× data → ≥19 % better) =
  "needs data" (supports D); slope $\ge-0.10$ under every regime = "head-limited" (supports H);
  between = ambiguous. Regularisation "moves it" if the `reg` or `reg_small` dev-report median at
  $n=512$ is ≥10 % below `plain`'s. Selection uses dev cases 32–63 only; every reported number is
  on dev cases 0–31; the oracle on 64 training trajectories is recorded beside it (the gap per
  arm). Caveat, stated now: the frozen bank was trained on all 512 trajectories, so the subset
  arms measure the head's data dependence with a bank that has seen everything; that biases the
  small-$n$ arms *optimistically*, so a flat slope is the stronger reading.
- **ns302 / ns303.** The Phase-2 gates of §A4 unchanged (B-RANKCAP, B-DATA, B-ORTH, B-FLOOR,
  H-TRAIN, H-SOLVED, H-ORACLE at the 2.0 bar), at the training mesh only (`EVAL_NS=256`: the
  three meshes were flat to three digits in every previous run). ns303's cohorts are a new
  family: their hashes are recorded (`mode=recorded-new-family`), not compared with Phase 1;
  ns302's base cohort keeps the Phase-1 hash gate and the extra cohort's hash is recorded.
  **Whichever of ns302/ns303 passes H-ORACLE gets the reserved Phase-3 job** with the §A3 driver
  at the pre-registered rungs. If both fail, the cell's finding is that neither 4× data nor an
  8-dimensional input rescues this head, and ns301's slope says which of D/H to believe.
- **ns304** is labelled *exploratory after a failed gate*. Its `R-LADDER` verdict is reported but
  is not a Phase-3 result of the paper's pre-registration (that requires a head that passed
  H-ORACLE). What it can establish: monotonicity of worst-evolved error in $q$, the same-job cost
  per rung, the non-dominated set against POD-LSPG at matched dimension, and whether the top rung
  ($q=R$, the full bank span) reaches the bank floor.

### Mechanics fixed by this amendment

- `ns2d_fom.params_draw(seed, count, nmodes=6)` and `initial` are mode-count aware; the default
  is byte-identical to Phase 1 (the smoke reproduces the frozen family's draw path).
- `ns2d_phase2.py`: `NMODES`, `TRAIN_EXTRA`/`SEED_TRAIN_EXTRA` (appended after the gated base
  cohort), `RECORD_HASHES` (new-family cohorts pass by recording), `POD_BIG` (a blocked method
  of snapshots for $S>20\,000$, verified against the GPU path to $7\times10^{-15}$ on the projector),
  `passes_at_1p5` recorded in H-ORACLE.
- `ns2d_phase3.py`: `M_FIXED` (one test space for every subject; one device copy of the tensor
  shared by all rungs).
- `ns2d_headfit.py`: new driver (documented in its header); trains in the whitened coefficient
  space, which is exactly the field-space loss up to the fixed bank-perpendicular term, and
  reports every error in field space (§A4).
- Checkpoints `checkpoints/ckpt_K16_R256.pkl` (ns203, sha `a8aebfc2…`) and
  `checkpoints/ckpt_K32_R512.pkl` (ns204, sha `1d03ee4e…`) are committed for staging.
- Local smokes (2026-09-17 19:2x EDT, 32², K=4, R=16): phase-2 with `NMODES=3`, `TRAIN_EXTRA=4`,
  `POD_BIG=100`, `RECORD_HASHES=1` — all mechanics gates pass; head-fit smoke — 6 arms, summary
  and slopes produced; phase-3 smoke with `M_FIXED=32` — R-TB 0, R-TFFT 6.7e-16, R-TQ 1.0e-15,
  R-LIN 1.2e-15; weight decay verified to act on the head weights only.
- Wall-time allocations: ns301 5 h, ns302 12 h, ns303 8 h, ns304 24 h (A100, 180 GB). Jobs
  6–9 of 12. One reserved for Phase 3 on a passing head; two spare.
