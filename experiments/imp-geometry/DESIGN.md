# imp-geometry (E2, pilot Part B′): does the reduced model win where no fast transform exists, against strong solvers? (pre-registration)

**Status (2026-10-21): design, pre-registered before any code runs. No result exists yet.** Amendments are appended at
the bottom, dated; the text above them is never rewritten. Every number this lane produces will come from
`make_report.py` reading run JSONs.

Lane `exp/2026-10-21-imp-geometry` (worktree `worktrees/2026-10-21-imp-geometry`, fork `93f399faa`; mirror
`origin/codeonly/exp/2026-10-21-imp-geometry`), lane dir `experiments/imp-geometry/`, cluster namespace
`/cluster/tufts/paralab/tawal01/imp-geometry/`. **GPU-hour cap 62; lane concurrent-job cap 3; campaign cap 8 via the
shared lock.** Phasing (brief): design, audits, code, smoke (≤ 1 GPU-h cluster smoke/memory pilots) now; production only
after the coordinator's "release production".

## 0. Why this experiment (what is already known)

- The closed Burgers campaign found that on the Dirichlet **box** a coarse sine pseudo-spectral FOM beats every
  qualified reduced configuration (K2 fires, `reports/2026-10-17-full-experiment-plan/final/report.md`).
- The spectral pilot (`worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/report.md`, PILOT) found, on a
  unit L-shape with a **narrow** family, a POD–Galerkin ROM 17–115× faster than two **second-order FD** solvers on one
  CPU core, but (i) FD is a weak competitor for smooth solutions, (ii) its post-hoc box control AB (same narrow cases
  on the box) gave the same ROM S = 0.6–2.9 against the spectral FOM, so family width, not geometry, may explain much of
  the win. The pilot's §7 Part B′ proposes this experiment: strong, tuned, high-order competitors, a wide family, a box
  control per cell, on the GPU.

**Question.** On two domains without a fast transform (an L-shape and a box with a circular hole), with a wide
family, is the coordinate-network bank with a distance envelope at least 2× faster than the best tuned competitor
(high-order spectral elements, FD-IMEX with multigrid, FD-RK4, POD + ECSW) at matched error — and is that advantage
absent on the box against the spectral FOM (so that it can be attributed to geometry)?

## 1. Problem and geometry (pinned here; the brief does not fix them)

Viscous Burgers with advection along $(1,1)$, as in the campaign:
$$u_t + u\,(u_x+u_y) = \nu\,\Delta u \ \text{ in } \Omega,\qquad u=0 \text{ on } \partial\Omega,\qquad t\in[0,0.25],$$
outputs at $t_k=0.05k$, $k=0,\dots,5$; initial condition $u_0=a\exp(-|x-c|^2/(2w^2))$, set to zero on $\partial\Omega$.

**Domains** (all inside $(0,2)^2$; the scale is 2 so that the wide family fits with a compatible initial condition,
see below):

| id | domain | non-box feature |
|---|---|---|
| L | $\Omega_L=(0,2)^2\setminus[1,2]^2$ | re-entrant corner at $(1,1)$; leading singular mode $\rho^{2/3}\sin(2\theta/3)$ |
| H | $\Omega_H=(0,2)^2\setminus\overline{B}((1,1),0.2)$ | curved interior boundary (circle of radius 0.2 centred at $(1,1)$) |
| B | $\Omega_B=(0,2)^2$ | box control only |

Both non-box features sit at $(1,1)$, on the advection path of every bump (bumps start in $[0.3,0.7]^2$ and move
along $(1,1)$ by about $aT\approx0.25$–$0.5$).

**Families** (bump width as A′, $w\sim U[0.035,0.05]$; $a\sim U[1,2]$; centres $c_x,c_y\sim U[c_0,c_0+W]$):

| family | $W$ | $c_0$ | note |
|---|---|---|---|
| wide | 0.40 | 0.30 | A′'s $W=0.4$; clearance to the outer walls $\ge0.30=6w_{\max}$ (wall value $\le1.6\times10^{-8}a$), to the hole $\ge0.224$ (value $\le4.6\times10^{-5}a$), to the re-entrant corner $\ge0.42$ |
| narrow | 0.08 | 0.46 | the pilot's $W=0.08$ narrow family, re-centred on the scaled domain and given A′'s bump widths so that only $W$ differs from wide |

**Steepness levels** (the brief says "two steepness levels" without values; pinned to A′'s levels A2 and A3 so the two
lanes can be compared): **S1** $\log\nu\sim U[\log0.02,\log0.04]$ ($s=a/\nu$ 25–100); **S2**
$\log\nu\sim U[\log(0.02/3),\log(0.04/3)]$ ($s$ 75–300).

**Cells:** 2 geometries × 2 families × 2 levels = 8 cells, plus 4 box-control cells (B × 2 families × 2 levels).

**Departures from the pilot, stated:** domain scale 2 (the pilot's unit L-shape cannot hold a $W=0.4$ family with a
compatible initial condition, since its arms are 0.5 wide); narrow family re-centred and given A′'s widths.

## 2. Cohorts (split by seed before any run)

Descriptors $(c_x,c_y,w,a,\nu)$ depend on (family, level, split) only, **not on the geometry**: the L, H and B cells
of one (family, level) use identical cases. This pairs the geometries and makes one box control serve both. Row $j$ of a
split comes from `np.random.default_rng([seed, j])`. Seeds: $7\,100\,000 + 1000\,f + 100\,\ell + \sigma$ with $f\in\{0
\ (\text{narrow}),1\ (\text{wide})\}$, $\ell\in\{1,2\}$, split $\sigma$: train 1, dev 2, val 3, test-once 4. Disjoint
from every campaign seed (asserted against `jcp-references/cohorts.py` and the pilot's seeds); descriptor rows asserted
disjoint across splits.

| split | size | use |
|---|---|---|
| train | 192 trajectories | POD bases, ECSW weights, bank training (identical data for every reduced arm) |
| dev | 8 | development, smoke, Q1 tight/tighter subset, Q2 population with val[0:8] |
| val | 32 | **all selection and matching** (the "32 test cases" of the brief) |
| test-once | 32 | scored **once**, at the end, on the frozen matched configurations (confirmation) |

No campaign cohort (dev6/val32/test64/923801/923901, sealed, reserved-OOD) is used; the E0 store is not touched (it has
no entry for these domains).

## 3. References (E0 recipe transplanted)

- **Solver:** second-order FD (5-point Laplacian, central advection; Shortley–Weller for the hole), SBDF2 in time
  (IMEX Euler first step), Helmholtz solves by the lane's multigrid to relative residual $10^{-12}$.
- **Ladder:** per-unit spacing $h\in\{1/256,1/512,1/1024\}$ (S1) and $\{1/512,1/1024,1/2048\}$ (S2), $\Delta t\propto h$
  ($\Delta t=0.25/n$, $n=2000,4000,8000$ for S1; $4000,8000,16000$ for S2). Fields compared on the scoring lattice.
- **$\varepsilon_{\rm ref}$** (E0 formula): $p(t_k)=\log_2(d_{12}/d_{23})$; $\varepsilon_{\rm ref}(t_k)=d_{23}/(n_0(2^p-1))$
  with $n_0=\lVert u_{\rm ref}(0)\rVert$; $\varepsilon_{\rm ref}=\max_k$. Per-case **eligibility**: $p(t_k)\in[1.8,2.2]$
  at every $k$, finite, solver converged. Reference field = the finest level. Every $\varepsilon_{\rm ref}$ is
  labelled **conditional Richardson estimate**.
- **Fallback:** if fewer than 24 of a cell's 32 val cases are eligible, the cell is scored two-way against the finest
  field with $U=d_{23}/n_0$ and every statement about it carries "conditional on an unqualified reference".
- **Independent cross-check (diagnostic, reported):** the SEM competitor at its finest setting against the reference on
  the 8 dev cases. Disagreement above $\max(2\varepsilon_{\rm ref},10^{-3})$ is reported as a reference problem.

## 4. Metrics and matching

- **Error of a run:** $e=\max_{k=1..5}\lVert u(t_k)-u_{\rm ref}(t_k)\rVert_{\rm lat}/\lVert u_{\rm ref}(0)\rVert_{\rm lat}$,
  discrete $\ell^2$ on the scoring lattice $\{x=i/64\}$ restricted to the open domain (L: 127² minus the removed
  quadrant and its edges; H: minus points with $|x-(1,1)|\le0.2$). Every method outputs the six fields at these points
  (FD: sub-sampling, $1/h$ a multiple of 64; SEM: precomputed element-local Lagrange interpolation; reduced arms:
  precomputed lattice rows of the basis).
- **Statistics over val32:** worst and median. Non-finite or invalid runs: $e=\infty$.
- **Targets** $\tau\in\{0.01,0.03,0.10\}$ (brief), on worst (primary) and median (secondary).
- **Three-way rule:** with $U=2\max_{\rm cases}\varepsilon_{\rm ref}$ (fallback: $U=\max d_{23}/n_0$), a configuration
  *meets* $\tau$ if statistic $+U\le\tau$, *misses* if statistic $-U>\tau$, else *unresolved* (never counted as a pass).
- **Matched configuration** of a method at $\tau$: the fastest (by the paired same-job timing, §6) configuration that
  meets $\tau$ and is Q1 (and, where applicable, Q2) qualified. $S(\tau)=t^*_{\rm competitor}/t^*_{\rm bank}$.
- **Confirmation on test-once:** the frozen matched configurations are run once on test-once. A $\tau$ at which either
  side's matched configuration does not meet $\tau$ on test-once is "not confirmed" for that cell.
- **Censoring:** a method that never meets $\tau$ in its ladder is censored at $\tau$ (no S; not evidence).
- **Every S is an upper bound on our speed-up over the tested competitor ladders** (finite ladders) and is reported
  as "against the strongest tested competitor".

## 5. Arms

All arms in JAX, f64, `JAX_DEFAULT_MATMUL_PRECISION=highest`, run and timed on one cluster GPU per job.

### 5.1 Competitors (each tuned over its own knobs; staged search of §5.4)

| id | method | knobs |
|---|---|---|
| C-SEM | **nodal spectral elements** (GLL order $p$, conforming quadrilateral macro-block mesh; exact circular arcs on the hole via Gordon–Hall blending; optional geometric grading toward the re-entrant corner); GLL collocation with lumped mass, optional 3/2 over-integration of the advection term; IMEX SBDF2 or SBDF3 (explicit advection, implicit diffusion); Helmholtz solves by matrix-free Jacobi-preconditioned CG with extrapolated warm start | $p\in\{4,6,8,10,12\}$, elements per macro edge $k\in\{1,2,3,4,6,8\}$, grading $\in\{\text{off},0.5\}$ (L), dealias $\in\{\text{off},\text{on}\}$, order $\in\{2,3\}$, $n$ steps, CG tolerance $\in\{10^{-4},10^{-6},10^{-8},10^{-10}\}$ |
| C-FDMG | second-order FD (as §3), SBDF2, Helmholtz by geometric multigrid V(2,2) cycles (damped Jacobi smoothing, full weighting, bilinear prolongation, rediscretised coarse levels) iterated to a relative-residual tolerance | $1/h\in\{64,96,128,192,256,384,512,768\}$ per unit, $n$, MG tolerance $\in\{10^{-3},10^{-4},10^{-6},10^{-8}\}$ |
| C-RK4 | the same space discretisation, classical explicit RK4 (the pilot's F-RK4) | $1/h$, $n$ (stability-limited) |
| C-ECSW | POD + ECSW (POD of the FD training data on the data grid; ECSW node weights by greedy NNLS on the tested nonlinear term, jcp-hr-baselines `nnls.py`), stepper as §5.2 | $r$, support $\kappa r$ with $\kappa\in\{2,4,8\}$, stepper, $n$ |
| C-SPEC (box only) | sine pseudo-spectral ETDRK4 on $(0,2)^2$ (the K2 winner of jcp-fom2d, rescaled to side 2), 3/2 padding | $N$, $n$ |

**Static condensation is not implemented.** The SEM Helmholtz operator $\tfrac{\beta}{\Delta t}B+\nu K$ is
mass-dominated at the steps used, so Jacobi-PCG converges in $O(10)$ iterations; the matrix-free solver is the standard
GPU choice (Nek5000-style). If, in the smoke, PCG iterations per step exceed 50 at a matched configuration, this is
recorded as a weakness of our C-SEM and S is labelled "against the strongest tested competitor". Multi-domain
Chebyshev is not built (C-SEM covers both geometries).

### 5.2 Reduced arms

All three share one reduced engine for a linear span $\Psi_r$ with Dirichlet-exact functions, and one **stepper
parameter**: default = the incumbent **CN-R LSPG** (Crank–Nicolson with two backward-Euler start steps; residual tested
on the nested prefix $\Theta=\Psi_{\min(4r,R_{\max})}$ of the same ordered span, Levenberg–Marquardt, incumbent
tolerances prod $(g_{\rm tol},t_f)=(10^{-3},10^{-9})$, tight $(10^{-7},0)$, tighter $(10^{-8},0)$). Alternatives,
implemented for E5's verdict: Galerkin CN (Newton), Galerkin ETDRK4 (reduced Laplacian diagonalised per prefix), one-pass
linearised Galerkin CN. On "release production" the coordinator names the stepper; it is recorded in an amendment.
The test space is a transplant of the incumbent's $M=4r$ sine tests: sines do not satisfy the boundary condition on
$\Omega_L$/$\Omega_H$, the span's own longer prefix does.

| id | span | operators | knobs |
|---|---|---|---|
| R-BANK | per-(geometry, family, level) retrained bank (dn256b Stage-A architecture: Fourier features, $g$-MLP, $K=16$ head, $R=512$ features; the box envelope `bc_poly` replaced by the R-function envelope $\varphi_\Omega$ below); ordered span from the training codes | continuum weak form on a composite Gauss rule on $\Omega$ (§5.3): mass, stiffness $\int\nabla\psi_i\cdot\nabla\psi_j$, nonlinear term in point form ($6mr$) or exact precontracted tensor ($r^3$), the cheaper charged | $r\in\{16,24,32,48,64,96,128,192,256,384\}$, rule $q$, stepper, $n$ |
| R-PODG | POD of the training data on the FD data grid | FD operators projected (grid inner product), exact quadratic tensor | $r$, stepper, $n$ |
| C-ECSW | as §5.1 (it is both a competitor and a reduced arm) | | |

**Envelopes** (scaled coordinates $X=x/2$): box $\varphi_B=16X(1-X)Y(1-Y)$ (the incumbent's `bc_poly`); L:
$\varphi_L=\varphi_B\cdot\big(f_1+f_2+\sqrt{f_1^2+f_2^2}\big)$, $f_1=1-x$, $f_2=1-y$ (R-disjunction, zero on both
re-entrant edges); H: $\varphi_H=\varphi_B\cdot\big(|x-(1,1)|^2-0.2^2\big)/0.4$ (normalised distance, positive outside
the disk). Caveat: $\varphi_L\sim\rho$ at the re-entrant corner while the solution's singular part is $\rho^{2/3}$, so
the quotient is singular there; the bank's floor near the corner is a measured quantity.

**Bank recipe (deviations from dn256b, stated):** (1) envelope $\varphi_\Omega$; (2) training data = this cell's 192
training trajectories (21 snapshots each, $t=0,0.0125,\dots,0.25$; 4032 states) sampled on the per-unit $1/256$ (S1) or
$1/384$ (S2) sub-lattice of the FD data grid, restricted to $\Omega$; (3) Stage A only (300k Adam steps, recipe
hyper-parameters unchanged: lr $10^{-3}$, $p_{\rm sub}=4096$, wd $10^{-5}$, EMA 0.999, full-batch last 10k,
$\lambda_{\rm orth}=10^{-4}$, $n_{ff}=128$, $g$ 1024 wide, $h$ 256 wide); Stage B (code densification) is dropped;
(4) ordering: $G=Q R_G$ by QR in the $L^2(\Omega)$ inner product of the composite Gauss rule (not the box grid), rows
$R_Gc_i/\lVert R_Gc_i\rVert$ with $c_i=h_\theta(z_i)$ of the 4032 training codes, SVD, $T=R_G^{-1}V_s$; width $r$ keeps
$T_{:, :r}$. One training seed per bank (retrains are not bitwise reproducible; seeds and hashes recorded).

**Training data** come from C-FDMG at the reference integrator (SBDF2, MG tol $10^{-12}$) at per-unit $h=1/512$ (S1) or
$1/768$ (S2), $n=2000$ (S1) / $4000$ (S2). Their accuracy against the reference is measured on 4 training-like dev cases
and reported (training data error must be $\le\tau_{\min}/10=10^{-3}$; otherwise labelled).

### 5.3 Quadrature on $\Omega$ (bank arm)

Composite Gauss–Legendre: the same macro-block mesh as C-SEM (L: three unit squares; H: an outer Cartesian block layer
plus four Gordon–Hall blended curved quads between the square $[0.7,1.3]^2$ and the circle), each element subdivided
$k_q\times k_q$, $q\times q$ Gauss points per element, weights times the map Jacobian. Rule ladder $q\in\{4,6,8,10,12\}$ at
$k_q$ fixed per cell by the smoke (smallest $k_q$ whose $q=12$ rule integrates the training snapshots' squared norm to
$10^{-8}$ relative). Successor of $q$ is $q+2$ (Q2).

### 5.4 Staged ladder search (all on val32, both sides symmetric)

1. **Space** at a fine step and tight solver tolerance: every size (SEM $(p,k,\text{grading},\text{dealias})$; FD $1/h$;
   reduced $r$; spectral $N$). A size is *spatially feasible* at $\tau$ if its worst error meets $\tau$.
2. **Step**, per feasible size: $n=5\cdot2^j$ ascending until the trajectory distance $\delta_n=\max_{\rm cases}\max_k
   \lVert u_n(t_k)-u_{n/2}(t_k)\rVert/n_0\le10^{-4}$ (both valid) or $n=20480$; time order (SEM 2/3) is a knob here.
3. **Solver tolerance**, at each step-feasible configuration: the loosest tolerance that still meets $\tau$ and
   Q1.
4. **Timing screen** of the surviving Pareto set (one repetition), then **final paired timing** of the screen's fastest
   per method and $\tau$ (§6). The screen-to-final pruning rule: keep every configuration within 1.5× of the screen's
   fastest.

## 6. Cost and timing contract

- **Query ("field" contract):** case descriptor → the six output fields on the scoring lattice. Charged: initial
  condition (FD/SEM: nodal Gaussian; spectral: sine coefficients by the shared quadrature table; reduced: projection
  $z_0=\Psi^\top W u_0(x_q)$ on the arm's rule, or on the data grid for POD arms), every $\nu$-dependent setup (ETDRK4
  coefficients, MG level operators, Jacobi diagonals), all steps and solves, output synthesis. Case-independent setup
  (meshes, geometric factors, bases, tensors, interpolation matrices) is offline, reported separately.
- **Device:** one GPU per timing job; every competitor and reduced arm of a cell timed **in the same job on the same
  GPU**; never across GPU types. Batch size 1. f64.
- **Protocol:** warm JIT (compile and one warm call excluded), burn-in of 3 calls, then paired A–B–A triplets
  (competitor, bank, competitor) on the 3 timing cases (val[0:3]) × 5 repetitions; $t^*$ = median; spread and outliers
  reported. Every timed call ends with `block_until_ready`. A sentinel configuration timed at the start and end; drift
  > 10 % re-runs the job once, a second drift marks it invalid.
- **Controls of the harness:** T-dbl (doubled query) must time at 1.8–2.2×.

## 7. Qualification

- **Q1** (COMMON-BRIEF §3), for every reduced configuration that is matched or reported:
  - LSPG / Newton steppers: in-job at every step with a fresh Jacobian, $\gamma=\lVert J^\top r\rVert/(\lVert J\rVert_F\lVert
    r\rVert+10^{-300})$; non-stationary steps ($\gamma>10^{-3}$ or non-finite) ≤ 1 % pooled; every run valid (finite, no
    damping saturation); distance to the tight solve ≤ $2.5\times10^{-4}$ on every val case; tight/tighter ≤
    $2.5\times10^{-5}$ on dev8.
  - Explicit ETDRK4 (no algebraic solve): validity, plus the step-halving distance $\le2.5\times10^{-4}$ on every val case
    (the analogue of "distance to the tight solve"; audited).
  - **Competitors with iterative solvers (C-SEM PCG, C-FDMG)** get the same tight-solve rule: distance of the production
    tolerance to tolerance $10^{-12}$ ≤ $2.5\times10^{-4}$ on every val case; non-converged solves (iteration cap hit)
    count as invalid steps (≤ 1 % pooled, else the configuration is invalid).
- **Q2** (bank only; POD arms use the exact discrete sums of their grid, ECSW's hyper-reduction is the arm itself): rule
  $q$ against successor $q+2$ on the arm's own reached states (6 outputs × dev8 + val[0:8]): $\rho_{\max}=\max\lVert
  N_q-N_{q+2}\rVert/\lVert N_{q+2}\rVert\le0.116$ on the tested nonlinear vector, and rollout distance ≤ $2.5\times10^{-4}$
  on every one of the 16 cases.
- Three labels kept separate: numerical (Q1+Q2), role (none pre-declared here beyond "matched at τ"), frontier (not
  dominated in the error–time plane).

## 8. Must-fail controls and gates (each must be shown to fire in the smoke, on real data)

| id | control | must |
|---|---|---|
| MF-REF | reference order gate on a corrupted ladder (finest level replaced by the middle level plus a $10^{-4}$ perturbation) | **fail** (case ineligible) |
| MF-MATCH | matcher on an under-resolved FD configuration ($1/h=64$, S2, coarsest $n$) | be **rejected** at τ = 0.01 |
| MF-TIME | doubled query | time at 1.8–2.2× single |
| MF-ENV | boundary trace of the bank functions on ∂Ω (500 points per edge/arc): $\max|\psi|\le10^{-12}$; the box envelope $\varphi_B$ used on Ω_L / Ω_H | **fail** (trace > $10^{-3}$) |
| MF-QUAD | composite rule integrates area and $\int_\Omega\varphi_\Omega^2$ against closed forms / a 4× finer rule to $10^{-12}$; chordal (straight) elements around the hole | **fail** ($> 10^{-6}$) |
| MF-SEM | Helmholtz manufactured solution (vanishing on ∂Ω) on each geometry: error falls ≥ 4 decades from $p=4$ to $p=12$ on curved elements; chordal elements on the hole | **plateau** (≤ 2 decades) |
| MF-FD | same manufactured solution with FD: observed order ∈ [1.8, 2.2] (Shortley–Weller); staircase boundary on the hole | **fail** (order < 1.5) |
| MF-MG | MG V-cycle contraction ≤ 0.2 per cycle on the Helmholtz operator at the finest reference level; smoother replaced by identity | **fail** (no contraction to tolerance in 50 cycles) |
| MF-TENS | tensor-form ROM = point-form ROM (2 cases, $r\in\{32,128\}$, rel. $10^{-9}$); one tensor slice zeroed | **differ** by ≥ $10^{-6}$ |
| MF-Q1 | LSPG with one LM iteration per step | **fail** the γ bar |
| MF-Q2 | bank with rule $q=2$ against $q=4$ | **fail** the Q2 bars |
| MF-RAND | reduced arm with a random orthonormal span (in the arm's inner product), $r\le128$ | **fail** τ = 0.10 |
| MF-ECSW | ECSW with uniform weights on a random support of the same size | **fail** the ECSW training fit (rel. > 10 %) or τ = 0.10 |
| G-COH | descriptor disjointness across splits and from campaign seeds | assert |
| G-FLOOR | every reduced run's error ≥ its span floor (least-squares projection of the reference on the sampled lattice basis) − $10^{-9}$ | assert (a run below its floor is a bug) |

A control that cannot fire on the real data is a bug in the control, fixed before any reading.

## 9. Win / withdraw rule (verbatim from the brief, then the operational reading)

> Win: $S \ge 2$ against the best competitor on the wide family. Otherwise withdraw the geometry claim. Also
> withdraw any geometry attribution if the box control gives the ROM the same S against the spectral FOM.
> If a high-order competitor cannot be built in budget: say so, use the strongest available, and label S "against
> the strongest tested competitor". The E2 verdict decides whether E4 (3D cube with small spherical holes) runs.

**Operational reading (pinned now):**
- "The ROM" = **R-BANK** (our method). "Best competitor" at $\tau$ = the fastest matched configuration among C-SEM,
  C-FDMG, C-RK4 and C-ECSW. R-PODG is reported beside it (S of R-PODG and of the best reduced arm are secondary
  readouts; if R-PODG beats R-BANK, the report says so).
- **Statistic:** worst error over val32 (primary); median is secondary and cannot create a win.
- **Cell win:** a wide-family cell (geometry, level) wins at $\tau$ if $S_{\rm worst}(\tau)\ge2$, both matched
  configurations are Q1 (and the bank Q2) qualified, and $\tau$ is confirmed on test-once.
- **Geometry win:** geometry G wins if **both** of its wide-family levels win at ≥ 1 target each.
- **E2 = WIN** if at least one geometry wins; otherwise **WITHDRAW** the geometry claim. Reported per geometry.
- **Attribution:** for each winning cell and each winning $\tau$, the box control of the same (family, level) is run with
  the same reduced arm against C-SPEC on $\Omega_B$ (box bank retrained; same protocol). The control gives "the same S"
  if $S_{\rm box}(\tau)\ge S_{\rm geom}(\tau)/2$ (within a factor 2, or larger); then the geometry attribution of that
  cell is withdrawn (the win stands as a result about that family, not about geometry). A geometry's attribution
  survives if at least one winning cell per level keeps it.
- **E4 recommendation:** run E4 only if geometry H (the curved-hole analogue of E4's spherical holes) wins with its
  attribution intact. The coordinator decides.
- **High-order competitor:** if C-SEM fails its gates (MF-SEM, reference cross-check) or cannot be built in budget,
  this is stated and every S is labelled "against the strongest tested competitor".
- Narrow-family cells are a control for family width (pilot AB): reported, never part of the win.

## 10. Readouts

R1 per cell and $\tau$ (worst, median): matched configurations, times, S, confirmation; censored cells. R2 Pareto
plots (error vs time, all methods) per cell. R3 query-cost decomposition per arm (IC, setup, steps, output). R4 SEM
and MG solver statistics (iterations per step). R5 bank vs POD floors per $r$ (held-out val). R6 reference statistics
(eligible cases, $\varepsilon_{\rm ref}$, observed orders), SEM cross-check. R7 box controls: $S_{\rm box}$ vs
$S_{\rm geom}$. R8 offline costs and break-even query counts (labelled: offline wall on GPU vs per-query GPU time).

## 11. Compute plan and GPU-hour estimate (planning; replaced by the smoke cost certificate before release)

| item | jobs | estimate (GPU-h) |
|---|---|---|
| references: 12 (domain, family, level) combos × 72 cases × 3 levels | 4–6 | 8 |
| training data: 12 combos × 192 trajectories | (same jobs) | 3 |
| banks: 8 geometry cells + 4 box cells (box banks only if a geometry cell wins; else defer) | 12 | 12 × 0.6 ≈ 7 |
| competitor ladders (C-SEM, C-FDMG, C-RK4; C-SPEC on box) on val32 | 12 | 15 |
| reduced ladders (R-BANK, R-PODG, C-ECSW) incl. Q1/Q2 | 12 | 8 |
| final paired timing + test-once | 12 | 6 |
| contingency | | 10 |
| **total** | | **≈ 57 (cap 62)** |

**Defer order if the cap binds:** (1) box banks for non-winning cells (never needed); (2) the narrow-family S2 cells;
(3) SEM dealias-on and SBDF3 branches; (4) the narrow-family S1 cells (the AB analogue then rests on POD arms only);
the wide-family cells and their box controls are never deferred. Every deferral is recorded in an amendment.

## 12. Deliverables

`DESIGN.md`; code in this folder; `runs/` (JSON, logs, small arrays; >100 MB arrays in
`/cluster/tufts/paralab/tawal01/burgers-archive/imp-geometry/` with a committed sha256 listing); `runs/GPU-LEDGER.json`;
`audits/`; `make_report.py` → `report.md` (generated numbers, provisional labels inline, LaTeX, mermaid, glossary);
`EXPORT.json`; PNG plots.

## 13. Stop rules (lane)

- The GPU-hour cap (62) is never exceeded: a job is submitted only if spent + certified projection ≤ 62.
- If any must-fail control cannot be made to fire, no production job runs.
- If the reference fallback (§3) applies to every wide-family cell, the verdict is reported as conditional.

---

## Amendment A1 (2026-10-21, after Codex design audit 1 = NOT-READY; before any production run)

Audit: `audits/codex-design-1.md` (items 1, 2, 4–11 WRONG, 3 NEEDS-RESTATEMENT). Every item is addressed below; A1
governs over the text above where they differ. Local CPU checks run while amending are cited by file, never typed in.

### A1.1 Win / withdraw rule (fixes audit 1)
The brief's rule, verbatim (the §9 quotation inserted an "and"):

> Win: $S \ge 2$ against the best competitor on the wide family. Otherwise withdraw the geometry claim. Also
> withdraw any geometry attribution if the box control gives the ROM the same S against the spectral FOM.
> If a high-order competitor cannot be built in budget: say so, use the strongest available, label S "against the
> strongest tested competitor". The E2 verdict decides whether E4 (3D cube with small spherical holes) runs.

Operational reading (replaces §9's):
- **Competitor set** = C-SEM, C-FDMG, C-RK4, C-ECSW **and R-PODG** (every tested non-bank method). "The ROM" = R-BANK.
- **Cell win at τ** (wide-family cell = geometry × level): on val32, $S_{\rm worst}(\tau)\ge2$ with both matched
  configurations numerically qualified; then **confirmed on test-once**: the frozen finalists (committed manifest,
  sha256) meet τ on test32 under the three-way rule, pass Q1 (and the bank Q2) in-job on test32, and their paired
  re-timing on test[0:8] in the test job gives $S_{\rm test}(\tau)\ge2$. Any failure = "not confirmed" at that τ. The
  validation-selected finalists are never replaced after any test number is produced.
- **Geometry win:** both wide-family levels of the geometry have ≥ 1 confirmed winning τ.
- **Attribution, per (cell, τ):** the box control of the same (family, level) at the same τ, same reduced arm (box
  bank retrained), against C-SPEC. "Same S" is a **lane choice, not in the brief**: $S_{\rm box}(\tau)\ge
  S_{\rm geom}(\tau)/2$ → attribution withdrawn at that τ; $S_{\rm box}<S_{\rm geom}/2$ → kept; box bank censored
  while C-SPEC meets τ → $S_{\rm box}=0$ → kept; C-SPEC censored, box control failed or missing → **unestablished**.
  A geometry win is **geometry-attributed** if each of its two levels has a winning τ whose attribution is kept and the
  interaction diagnostic (A1.2) passes for that cell; otherwise it is reported as a **family speed win (attribution
  withdrawn / unestablished)**.
- **Outcomes** (per geometry, then E2): WIN-attributed; WIN-unattributed; WITHDRAW (demonstrated loss: at every τ
  reached by both sides $S<2$, or the bank censored where a competitor meets τ); UNRESOLVED (missing evidence,
  budget, reference fallback of A1.4, or an unqualified matched configuration). E2 = the best geometry outcome in that
  order; E4 is recommended only for WIN-attributed on H.
- Every S is an upper bound **relative to more thoroughly tuned competitors, with the selected bank fixed** (finite
  ladders); it is not a statistical bound.

### A1.2 Problem redesign: the solution must interact with the feature (fixes audit 2)
Local checks (`runs/local/interaction_side2.txt`; a first redesign at side 1.75 with closer features, `runs/local/interaction_side175.txt`, also T = 0.5; coarse FD, n = 128) confirmed the audit: with the §1 geometry the bumps decay
and spread before reaching the features (the L/H reference fields equal the box fields to better than the scoring
precision). The geometry is therefore redesigned on the **unit square**, with the features next to the family and an
initial condition that is compatible with every boundary **by construction** rather than by clearance:

| id | domain |
|---|---|
| L | $\Omega_L=(0,1)^2\setminus[0.625,1]^2$ (re-entrant corner $(0.625,0.625)$, a grid line of every FD mesh) |
| H | $\Omega_H=(0,1)^2\setminus\overline B((0.7,0.7),0.1)$ |
| B | $(0,1)^2$ (box control) |

- **IC:** $u_0=a\,e^{-|x-c|^2/(2w^2)}\,\chi_\Omega(x)$ with $\chi_\Omega=\psi(x)\psi(1-x)\psi(y)\psi(1-y)\cdot
  \{\psi(\varphi_Q/2)\ (L),\ \psi(\varphi_h)\ (H),\ 1\ (B)\}$, $\psi(s)=1-e^{-(s/\delta)^4}$ ($s\le0\mapsto0$),
  $\delta=0.05$, $\varphi_Q=f_1+f_2+\sqrt{f_1^2+f_2^2}$ ($f=0.625-x,\,0.625-y$), $\varphi_h=(|x-(0.7,0.7)|^2-0.01)/0.2$.
  $u_0$ vanishes to fourth order on $\partial\Omega$ (so $u_0=\Delta u_0=0$ there), is $C^\infty$ except at the
  re-entrant corner point ($C^3$). Every method evaluates this same function (FD/SEM nodes, rule points, spectral DST).
- **Families** (bump widths and amplitudes unchanged): wide $W=0.4$, $c\in[0.15,0.55]^2$; narrow $W=0.08$,
  $c\in[0.47,0.55]^2$ (the downstream end of the wide range, next to the features, as the pilot's narrow family sat
  next to its corner). Bump centres keep ≥ 0.106 from the L quadrant and ≥ 0.112 from the hole.
- Levels S1/S2, $T=0.25$, outputs, splits and seeds unchanged (§1–2). Scoring lattice $x=i/64$: 63² box points restricted
  to Ω.
- **Interaction diagnostic (pre-registered, from the production references):** per case
  $D_{\rm geo}=\max_k\lVert u_\Omega(t_k)-u_B(t_k)\rVert_{{\rm lat}\cap\Omega}/\lVert u_\Omega(0)\rVert$ (box
  reference of the same descriptor; note the ICs differ near the feature through $\chi$, so $k\ge1$ only), and the
  feature-window energy fraction (lattice points within 0.2 of the corner / hole centre). A cell's geometry attribution
  requires median $D_{\rm geo}\ge0.05$ over val32; else it is "unestablished (weak interaction)". The smoke reports
  both on the redesigned problem before release (`runs/local/interaction_unit.txt`); if the wide cells fail the bar
  there, the problem is revised again by amendment before any production job.

### A1.3 Cohorts (audit 3)
Levels use independent descriptors: S1 vs S2 is a distributional, not a paired, comparison. Seeds are checked against
the campaign sources named in `geom.CAMPAIGN_SEEDS` only. Descriptor hashes, bank seeds and the finalist manifest are
committed before the test job; test reference eligibility, test errors and test timings are never used to choose or
replace any configuration. The Q2 population (dev8 + val[0:8]) is the stated scope of quadrature qualification.

### A1.4 References (audit 4)
- Ladders (unit square): S1 $n\in\{128,256,512\}$ per unit with $n_t=1000,2000,4000$ steps; S2 $n\in\{256,512,1024\}$
  with $2000,4000,8000$; MG tolerance $10^{-10}$; every solve must converge (else the case is ineligible).
- Order checks at **positive times only** ($k\ge1$); a zero or non-finite difference makes the case ineligible.
- **Labels per case:** E0 if $p_k\in[1.8,2.2]$ for all $k\ge1$; **corner-limited** if all $p_k\in[1.3,2.5]$
  (the $\rho^{2/3}$ corner and Shortley–Weller boundary can lower or perturb the observed order), $\varepsilon$ from the
  observed $p$; otherwise **ineligible**, $\varepsilon:=\max_k d_{23}$ (conservative stand-in, labelled). Ineligible
  cases stay in the cohort (worst over val32 is never over a shrunken set).
- **Independent time check:** at the finest $h$, $n_t$ vs $2n_t$ on dev[0:4] + val[0:4]; $\delta_t$ = max normalised
  difference.
- $U=2\max_{\rm cases}\varepsilon+\delta_t$ (one scalar per cell, conditional).
- **A WIN (A1.1) at τ needs ≥ 24 of 32 val cases E0 or corner-limited and $U\le\tau/5$**; otherwise that τ is
  UNRESOLVED (numerical-reference diagnostics only).
- **SEM cross-check:** the finest SEM setting vs the reference on dev8. A disagreement above $\max(2\varepsilon,10^{-3})$
  flags both; it is resolved (one more FD level and a higher SEM order on two cases) before any verdict, and no fault is
  assigned before that.
- Box references use the same FD recipe (the C-SPEC finest setting is a second cross-check on B).

### A1.5 Competitors fully specified (audit 5)
- **C-SEM solver options** (both built, both tuned, the faster wins): `pcg` (Jacobi-PCG, warm start, iteration cap 500)
  and `eig` — exact Helmholtz solves through the offline generalised eigendecomposition $K v=\lambda Bv$ on the
  interior nodes ($V^\top BV=I$, ν-independent, offline): $(\beta/\Delta t\,B+\nu K)^{-1}b=V\,{\rm diag}(1/(\beta/\Delta t+\nu
  \lambda))V^\top b$, two dense mat-vecs per step, allowed when the interior count $\le 20000$. Static condensation and
  multilevel Schwarz are not built; the smoke reports PCG iterations per step and the `eig`/`pcg` time ratio. If, at a
  matched configuration, PCG needs more than 50 iterations per step and `eig` is not admissible, S is labelled "against
  the strongest tested competitor".
- **Grading (L):** the $k$ subdivisions of every macro edge adjacent to the re-entrant corner are geometric with ratio
  $g\in\{0.5,0.7\}$ toward the corner (`geom.graded_breaks`), conforming across the three macro squares; affine element maps.
- **Over-integration:** $M=\lceil3(p+1)/2\rceil$ Gauss points per direction, exact-map metric terms at the Gauss points,
  Galerkin projection on the GLL Lagrange tests, then the lumped-mass inverse.
- **SBDF3 start:** SBDF1, SBDF2, then SBDF3, same Δt.
- **C-FDMG:** hierarchy $n\to n/2$ while $n$ is a multiple of 16 and $n/2\ge8$; damped Jacobi ω = 0.8, V(2,2); coarsest
  40 sweeps; full weighting; bilinear prolongation; coarse operators rediscretised (masks and Shortley–Weller fractions at
  the coarse $h$); cap 40 cycles per solve; any non-converged solve invalidates the run. FD ladder
  $n\in\{64,128,192,256,320,384,448,512,640,768\}$ (multiples of 64, so every scoring point is a node).
- **C-SPEC:** $N\in\{16,24,32,48,64,96,128,192\}$, $n=5\cdot2^j$, IC by DST of the IC sampled on $n_{\rm init}\in\{64,128,256\}$.
- **C-ECSW:** POD basis as R-PODG; fit population 12 training trajectories × 11 snapshots ($t=0.025k$) projected on
  $\Phi_r$; target = the tested nonlinear vector (Mt rows), each state block normalised by its exact norm;
  candidates = all nodes if ≤ 32768, else 16384 uniform + 16384 importance-sampled (∝ contribution norm), seed 7199001;
  greedy NNLS (vendor/nnls.py) recorded at supports $\kappa r$, $\kappa\in\{2,4,8\}$; online, only the selected rows of
  $\Phi$ and $D_h\Phi$ are used (precomputed; no neighbour values are needed).
- **Steppers:** R-PODG and C-ECSW are tuned over all four steppers; R-BANK uses the incumbent CN-R LSPG, plus E5's winner
  if it passes (brief). (This can only favour the competitors.)
- C-RK4 starts its step ladder at the stability estimate `fd.rk4_stable_n`.

### A1.6 Reduced-arm mathematics (audit 6)
- $R_{\max}=512$ for every arm; LSPG tests $M_t=\min(4r,512)$: the full $4r$ only for $r\le128$ (saturation at $r>128$
  stated next to every such row). **Independent residual diagnostic** (reported, not gated): the CN residual of the
  reached states tested on the *other* arm's 512-wide span (bank ↔ POD).
- **Bank basis frozen once:** $T$ from the reference rule $q_{\rm ref}=16$ at the cell's $k_q$; every rule $q$ forms its own
  $M_q=\Psi_q^\top W_q\Psi_q$, $K_q$, tests and IC projection $z_0=M_q^{-1}\Psi_q^\top W_qu_0$ with that same $T$ (Q2 changes
  only the rule). QR: thin QR of $W^{1/2}G$; SVD of the row-normalised $A=[R_Gc_i/\lVert R_Gc_i\rVert]$, right singular
  vectors $V_s$; rank cut $\sigma_j>10^{-10}\sigma_1$; codes with $\lVert R_Gc_i\rVert\le10^{-14}\max$ dropped (count
  reported); $T=R_G^{-1}V_s$.
- **POD on H:** the projected Shortley–Weller Laplacian is non-symmetric (asymmetry reported). ETDRK4 uses the
  eigendecomposition of $M^{-1}K$ (complex if needed) with ${\rm cond}(V)$ recorded; ETDRK4 is unavailable for an
  (arm, r) with ${\rm cond}(V)>10^8$.
- "Exact tensor" = exact contraction of the arm's discrete rule (not continuum integration).
- **Bank training sampling:** per-unit $1/128$ (S1) and $1/256$ (S2) sub-lattices of the FD data grid (data grid per-unit
  $1/512$ S1, $1/768$ S2); the full-batch phase uses a fixed 65,536-point subset when there are more points.

### A1.7 Search, scoring and timing (audit 7)
- **Search:** sizes in increasing order; per size the step rule of `ladder.py` (n = 5·2^j from the family's first rung;
  stop when $\delta_n\le10^{-4}$, after two consecutive invalid rungs following a valid one, or at $n=20480$); the first
  rung has no δ. The step distance is a **stopping heuristic**: every visited (size, n, tolerance) configuration is
  scored and is a candidate for **both** the worst and the median matching (separate searches). A size ladder stops
  after two consecutive sizes whose every visited configuration already meets τ = 0.01 on both statistics (larger sizes
  can only cost more per step at equal steps; the cheaper-step side is covered by the step rule). Tolerance ladders run
  at every step-feasible configuration of the size's Pareto front.
- **Timing:** final paired timing on **val[0:8]**: per finalist pair (competitor A, bank B) triplets A–B–A, 5 repetitions
  per case, ratio $t_A/t_B$ with $t_A$ = mean of the two A runs; $S$ = median ratio over the 40 triplets; every raw time
  persisted. The outputs of the timed calls are stored and compared with the ladder outputs (T-id, ≤ $10^{-12}$
  relative). Inputs are on the device before a timed call, outputs stay on the device (`block_until_ready`).
- **Certificates are not charged**: they run in an untimed replay with identical inputs whose outputs must equal the
  timed outputs (T-id).

### A1.8 Qualification (audit 8)
- **ETDRK4 (explicit):** in-job, every step: all four stage evaluations finite and $\lVert z_{n+1}\rVert\le10\max(\lVert
  z_0\rVert,\lVert z_n\rVert)$ (no blow-up); there is no algebraic solve to certify, so this is the analogue; temporal
  accuracy is left to the error metric (no step-halving requirement inside Q1).
- **Linearised CN:** distance to the fully converged Newton CN (tight) ≤ $2.5\times10^{-4}$ on every val case; Newton
  tight vs tighter ≤ $2.5\times10^{-5}$ on dev8.
- **Iterative competitors (C-SEM `pcg`, C-FDMG):** every solve converged (cap hit = invalid run); production tolerance vs
  $10^{-10}$ distance ≤ $2.5\times10^{-4}$ on every val case; $10^{-10}$ vs $10^{-12}$ ≤ $2.5\times10^{-5}$ on dev8.
- **Bank Q2:** rules $q\in\{4,6,8,10,12\}$ (successor $q+2$; $q=14$ exists for checking $q=12$); ρ skips states with
  $\lVert N_{q+2}\rVert\le10^{-12}\max$ (count reported); additionally $\lVert M_q-M_{q+2}\rVert/\lVert M\rVert$ and
  $\lVert K_q-K_{q+2}\rVert/\lVert K\rVert\le10^{-6}$ and the IC projection difference ≤ $2.5\times10^{-4}\lVert z_0\rVert$.
- **C-ECSW:** narrower contract, stated: Q1 (its stepper's algebraic certificate) only; hyper-reduction fidelity (distance
  to R-PODG at equal r, stepper, n) reported, not gated. This favours the competitor.

### A1.9 Controls (audit 9): predetermined, detectable faults
Kept (each is a deterministic computation, shown in the smoke): MF-SEM (curved vs chordal manufactured solution),
MF-FD (Shortley–Weller vs staircase orders), MF-MG (an identity smoother makes the V-cycle the identity in this code:
no coarse correction can act), MF-QUAD (area on curved vs chordal rules), MF-ENV (box envelope on L/H has a non-zero
trace). Replaced by fault injection on real computed states:
- **MF-REF:** fixtures (clean second-order ladder → E0; finest := middle → ineligible; first-order ladder → ineligible)
  plus the real-data version (finest replaced by the middle level).
- **MF-Q1:** the certificate evaluated on an iterate perturbed by $10^{-3}\lVert z\rVert$ must flag every step.
- **MF-Q2:** the reference rule with the weights of the elements carrying the largest share of $\lVert N\rVert$ zeroed
  must exceed the ρ bar.
- **MF-TENS:** the tensor slice with the largest measured contribution on a reached state zeroed must change the rollout
  by ≥ $10^{-6}$.
- **MF-INTERP:** SEM lattice interpolation of a degree-$p$ polynomial exact to $10^{-12}$; a wrong element index fails.
- **MF-MATCH:** SW1a/SW1b fixtures (`score.selftest`) plus the real under-resolved FD configuration rejected at τ = 0.01.
- **MF-LEAK:** the test runner refuses to start unless the finalist manifest exists, is committed, and its sha256 matches.
- **T-dbl:** two *different* cases run back to back in one timed call (no compiler elimination), 1.8–2.2×.
- **T-setup:** the timed calls use ν values different from the warm-up call, and the JAX compile count during timing must
  be 0 (no hidden per-query compilation; ν-dependent setup is inside the timed function by construction).
MF-RAND and MF-ECSW-random become **diagnostics** (failure is not mathematically ensured).

### A1.10 Budget (audit 10)
No production release without the smoke's certified cost projection (per stage, per cell, from measured cluster
seconds on the production GPU type, compile and qualification included, × 1.5 headroom), written to
`runs/cost/projection.json` and audited. Defer order (replaces §11's): (1) box banks of non-winning cells; (2) narrow-family
S2 cells; (3) narrow-family S1 cells. Competitor-strength branches (SEM `eig`, dealias, SBDF3, grading) are never deferred
while a strong-competitor claim is made. An incomplete wide-family panel is UNRESOLVED, not a negative result.

### A1.11 Diagnostics and scope (audit 11)
At every matched configuration: error on the finer lattice $x=i/128$ (where the method provides it: FD $n$ multiple of
128, SEM interpolation, reduced lattice rows) and in the feature window (points within 0.2 of the corner / hole centre);
decay $\lVert u(t_5)\rVert/\lVert u_0\rVert$ and the **zero-field baseline** $e_0=\max_k\lVert u_{\rm ref}(t_k)\rVert/
\lVert u_{\rm ref}(0)\rVert$; a τ that the zero field meets (worst) is excluded. Every conclusion is scoped to this
family, the field output contract, these implementations, and the stated qualification coverage.
