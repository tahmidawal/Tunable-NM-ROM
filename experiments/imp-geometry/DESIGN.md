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

---

## Amendment A2 (2026-10-21, after cluster smokes 1–3; before any ladder is read; no ROM or competitor result exists)

**Why.** A1.2's interaction diagnostic, run in the cluster smokes on the A1.2 geometry (`runs/smoke/smoke2-interaction/
interaction_v0_T0.25.json`, also $T=0.5$), passed for the narrow family but **failed for the wide family** (median
$D_{\rm geo}\approx0$): with $W=0.4$ most bumps start 0.2–0.4 from a feature placed downstream of the family square and
decay before reaching it. Variants were therefore compared on the diagnostic alone (FD, $n=256$, val[0:16], every
family × level; files `runs/smoke/smoke{2,3}-interaction/interaction_v{0,2,3,4}_T*.json`). No solver, ROM or timing
result entered this choice.

**Adopted: variant v4, $T=0.25$ unchanged** (`geom.VARIANTS['v4']`; it governs over A1.2's geometry and families):

| item | v4 |
|---|---|
| L | $\Omega_L=(0,1)^2\setminus[0.34375,1]^2$ (corner $(0.34375,0.34375)=(22/64,22/64)$; the diagnostic was run at 0.35, see below) |
| H | $\Omega_H=(0,1)^2\setminus\overline B((0.45,0.45),0.1)$; SEM / quadrature block around the hole $[0.3,0.6]^2$ |
| families | wide $c\in[0.15,0.55]^2$, narrow $c\in[0.22,0.30]^2$; centres falling in the **shared exclusion zone** (the L quadrant $[0.32,1]^2$ or the disk of radius $0.13$ about $(0.45,0.45)$, i.e. both features grown by 0.03) are redrawn from the same row generator, so L, H and B use identical descriptors |
| IC | A1.2's $u_0=a\,e^{-|x-c|^2/(2w^2)}\chi_\Omega$, with $\chi$ built from v4's features |

Diagnostic values for v4 are in `runs/smoke/smoke3-interaction/interaction_v4_T0.25.json`; every one of the eight
(geometry, family, level) cells meets A1.2's bar (median $D_{\rm geo}\ge0.05$) there. The production references recompute
it on val32 and the bar is applied to those (A1.2).

**Grid alignment of the L corner.** With the corner at 0.35, FD grid lines fall on $x=0.35$ only when $0.35n$ is an
integer. The FD ladders are therefore restricted to $n$ that are multiples of both 64 (scoring lattice) and 20, i.e.
$n\in\{320,640,960,1280\}$ — too coarse a ladder. Instead the L corner is moved to the nearest lattice-aligned value
$0.34375=22/64$ (so every $n\in64\mathbb N$ has grid lines there); the exclusion zone becomes $[0.31375,1]^2$. The hole is not
grid-aligned by design (Shortley–Weller).

The cohort seeds, splits, levels, metrics, rules and controls of A1 are unchanged.

---

## Amendment A3 (2026-10-21, after Codex design audit 2 = NOT-READY; before any production run)

Audit `audits/codex-design-2.md` (required fixes 1–8). A3 governs over everything above where they differ.

### A3.1 Chronology and the frozen problem (fix 1)
**Correction of A2's opening sentence.** A2 said "no ROM or competitor result exists". That is false. Before A2 was written,
cluster smokes 1–2 (geometry variant v2, 4 val cases, smoke-size references, 12 training trajectories, a 2000-step bank)
had produced FD, RK4, SEM, POD-Galerkin, ECSW and bank ladder rows (`runs/smoke/smoke2-cellL/ladder.jsonl`), bank
reconstruction numbers and cell-control values (`runs/smoke/smoke2-cellL/controls_L.json`), and the agent **printed the
three lowest-error rows per family** of that smoke ladder while debugging. The variant was chosen by the interaction
diagnostic only (A2's table), but the selection is **diagnostic-guided problem development**, not an outcome-blind
registration. Variants tried: v0 ($T=0.25,0.5$), v2 ($T=0.25,0.5$), v3, v4, v4a (all files in `runs/smoke/`).
**Frozen problem = `geom.VARIANTS['v4a']`** (A2's table, with the L corner at $22/64$; A2 wrongly named it `v4`): exclusion
zone = $\{c_x>LC-0.03\ \wedge\ c_y>LC-0.03\}\cup\{|c-(0.45,0.45)|<0.13\}$ with $LC=22/64$ (so the threshold is $0.31375$,
not A2's 0.32); a rejected centre is redrawn from the same row generator (`geom.draw`). The descriptor arrays of every
(family, level, split) and their sha256 are committed in `runs/descriptors.json` (`descriptors.py`), with acceptance
rates. No further redesign happens after this amendment.

### A3.2 Interaction: boundary effect separated from the initial condition (fix 2)
- The wide family is a **conditional** distribution on the non-rectangular support "square minus exclusion zone"; the
  narrow family is also moved (to $[0.22,0.30]^2$), so narrow vs wide differs in width **and** location (stated in the
  report).
- **Boundary diagnostic (primary, replaces A1.2's $D_{\rm geo}$ bar):** $D_{\rm bnd}=\max_{k\ge1}\lVert u_\Omega(t_k)-
  \tilde u_B(t_k)\rVert_{{\rm lat}\cap\Omega}/\lVert u_\Omega(0)\rVert$, where $\tilde u_B$ is the **box** evolution started
  from the non-box IC $u_{0,\Omega}$ extended by zero (same FD solver, $n=256$, $n_t=2000$ for S1 / $n=512$, 4000 for S2).
  $D_{\rm geo}$ (box IC) is kept as the combined problem contrast; the IC mismatch $\lVert u_{0,\Omega}-u_{0,B}\rVert/
  \lVert u_{0,\Omega}\rVert$ is reported.
- **Gate:** a cell's geometry attribution requires median $D_{\rm bnd}\ge0.05$ on val32 **and** on test32 (test32 is
  untouched by the variant selection, which used val[0:16]); computed in the cell job (val) and in the test job (test).
- The SEM finalists report the IC interpolation error on the lattice (cutoff-resolution diagnostic).
- v4a evidence so far: `runs/smoke/smoke4/interaction/interaction_v4a_T0.25.json` ($D_{\rm geo}$ only) and the v4a
  geometry controls `runs/smoke/smoke4/geom/controls_*.json`.

### A3.3 References for physical claims (fix 3)
A target τ of a cell can support a WIN (validation) or a confirmation (test) **only if every one of the 32 cases** of that
cohort is E0 or corner-limited (A1.4) and $U\le\tau/5$; otherwise τ is UNRESOLVED for that cohort. The "last difference"
stand-in remains only for diagnostics.

### A3.4 Decision table (fix 4) — governs over A1.1's outcome list
Per wide cell and τ, on **val32** (worst error; median never creates a win):

| val evidence at τ | label |
|---|---|
| reference not qualified (A3.3) | unresolved |
| both bank and best competitor matched, $S\ge2$ | candidate win |
| both matched, $S<2$ | loss |
| bank censored, some competitor matched | loss |
| bank matched, every competitor censored | candidate win ($S=\infty$) |
| both censored | no evidence |

A candidate win is **confirmed** iff on **test32** (test reference qualified, A3.3) the frozen bank and competitor finalists
meet τ, the bank passes Q1/Q2 in-job, and the paired test re-timing gives $S_{\rm test}\ge2$; otherwise **not confirmed**.
Cell outcome: **win** if ≥ 1 confirmed τ; else **loss** if ≥ 1 τ is loss or not confirmed; else **unresolved**.
Geometry: **WIN** iff both levels win; **WITHDRAW** iff some level is loss; otherwise **UNRESOLVED**.
**Attribution** (only for a geometry WIN), per level at its confirmed τ: the box control's finalists are selected on box
val32 by the same rules and confirmed on box test32. *Kept* iff $S_{\rm box,val}<S_{\rm geom,val}/2$ **and**
$S_{\rm box,test}<S_{\rm geom,test}/2$ **and** the cell's $D_{\rm bnd}$ gate passes (A3.2); *withdrawn* iff
$S_{\rm box,val}\ge S_{\rm geom,val}/2$; otherwise *unestablished* (box bank or C-SPEC censored, box reference unqualified,
box confirmation failed, $D_{\rm bnd}$ gate failed, box missing). A censored box bank is reported as a category, never as
$S_{\rm box}=0$ (A1.1's rule is withdrawn). WIN-attributed iff both levels have a kept τ; else WIN-unattributed.
**E2** = the best geometry outcome in the order WIN-attributed > WIN-unattributed > UNRESOLVED > WITHDRAW, every geometry
reported.

**Selection and the speed statistic (governs over §4 and A1.7):** per family and (τ, statistic), the candidates meeting τ
are screened (warm one-case median of 3 on val[0]); the **two** screen-fastest that meet τ in the unbatched official re-run
(and, for the bank, pass Q1/Q2) enter the paired timing. For each competitor finalist A and the bank finalist B,
$S_A$ = median over the 40 A–B–A triplets (val[0:8] × 5) of $\tfrac12(t_{A,1}+t_{A,2})/t_B$; the family's S is the smaller of
its two finalists; **S = min over competitor families** (the best competitor = the family attaining it; families within
5 % are both reported). The bank side uses its faster finalist by the same paired comparison against the best
competitor.

**Competitor gating (deliberate asymmetry, favouring competitors):** a competitor configuration is eligible when it is valid
(finite, no solver cap hit) and meets τ; its Q1 items (tight-solve distance etc.) are measured and **reported, not
gated**, and C-ECSW's support-successor Q2 ($\kappa\to2\kappa$, ρ and rollout bars) is reported, not gated. Only the bank
is gated on Q1/Q2. Excluding a fast but loosely solved competitor would inflate S; this choice is flagged for the
coordinator as a narrowing of COMMON-BRIEF §3 for competitors.

### A3.5 Q1 / Q2 for every selectable stepper (fix 5)
- **ETDRK4 (bank or POD arms):** in-job (i) a per-query **coefficient certificate**: the contour-integral coefficients $E,
  E_2,Q,f_1,f_2,f_3$ against an independent closed form (`expm1`-based for $|h\lambda|>0.2$, 8-term Taylor series below)
  agree to $10^{-10}$ relative on every mode; (ii) every step: all stage values finite; (iii) every step: $\lVert z_{n+1}\rVert
  \le10\max(\lVert z_0\rVert,\lVert z_n\rVert)$. Any failure in (i) invalidates the query; (ii)–(iii) failures count as
  non-stationary steps (≤ 1 % pooled, else not qualified).
- **Linearised CN:** every step, the linear-solve residual $\lVert J\delta+R\rVert/\lVert R\rVert\le10^{-10}$ (certificate),
  plus the tight-solve rule (distance to fully converged Newton CN ≤ $2.5\times10^{-4}$ on every case; Newton tight vs
  tighter ≤ $2.5\times10^{-5}$ on dev8).
- **Galerkin CN (Newton):** every step $\lVert R(z)\rVert/(\lVert M(z-z_n)\rVert/\Delta t+10^{-300})\le10^{-8}$ (≤ 1 %
  pooled), plus the tight/tighter rules.
- **LSPG CN-R:** as A1.8 (γ with fresh Jacobian).
- **Bank Q2:** rollout bar on **every** one of dev8 + val32 (40 cases) on validation and on test32 in confirmation; ρ on
  the 6 output states of the same cases; states with $\lVert N_{q+2}\rVert\le10^{-12}\max$ skipped, and Q2 fails if more
  than 5 % are skipped; M/K/IC bars as A1.8.

### A3.6 Controls (fix 6)
Convergence **demonstrations** (MF-SEM, MF-FD, MF-MG, MF-QUAD, MF-ENV) are reported as such. The v4a geometry controls all
fire, including MF-FD on L (`runs/smoke/smoke4/geom/controls_L.json`); the smoke2 MF-FD failure was on variant v2,
whose corner 0.45 is not a grid line of $n\in64\mathbb N$, i.e. the control correctly detected a staircase boundary.
**Fault detections** now establish the violation independently first:
- **MF-MATCH-real:** the best valid FD ladder output perturbed by exactly $2\tau$ (relative, every scored time) → must be
  *misses* at τ = 0.01; the unperturbed recomputation must reproduce the recorded error (replaces the contingent version
  that reported *meets* in smoke2).
- **MF-Q1:** the perturbed iterate's γ is first computed independently (NumPy, dense Jacobian); the in-job certificate
  must flag exactly the steps whose independent γ exceeds $10^{-3}$, and at least one such step must exist.
- **MF-TENS:** the slice fault is implemented independently in point form (the k-th test function removed); tensor-fault
  and point-fault rollouts must agree to $10^{-9}$, and differ from the clean rollout by ≥ $10^{-6}$.
- **MF-Q2:** the zeroed-element rule's ρ is computed directly; the Q2 checker must report failure when it exceeds the bar.
All controls are rerun on v4a in the final smoke.

### A3.7 Search coverage and the finer-lattice diagnostic (fix 7)
The A1.7 size-stop rule is **removed**: every registered size of every family is visited (its step ladder runs). The
finer-lattice diagnostic ($x=i/128$, val[0:8]) is computed for **every** matched configuration: reduced arms and C-SEM
natively, C-SPEC by synthesis, FD by bilinear interpolation of the grid solution (labelled; interpolation error is $O(h^2)$);
the reference uses its finest level ($n\ge512$, a multiple of 128).

### A3.8 Cost certificate (fix 8)
Release requires `runs/cost/projection.json` from `resources.py`, built from the production-size probe on the production
GPU type (`runs/smoke/smoke5/probe/*.json`) and the registered ladder sizes (row counts computed from the registered
ladders, not from the smoke), including compilation, Q1/Q2, timing, test and box jobs, × 1.5 headroom; audited before release.

---

## Amendment A4 (2026-10-21, after Codex design audit 3 = NOT-READY; before any production run)

Audit `audits/codex-design-3.md`. A4 governs over A3 where they differ.

**A4.1 (fix to A3.1).** `runs/descriptors.json` holds, per (family, level, split), the seed, size, **sha256 of the
descriptor array** and steepness range — not the arrays (they are regenerated deterministically by `geom.draw`). Its
acceptance rates are **estimates** from a separate 20,000-centre probe per cohort, not counts observed while drawing.

**A4.2 Decision table (replaces A3.4's).** Matching uses only *meets*; a configuration that is *unresolved* never matches.
Per method side (bank; best competitor over all competitor families) at τ on a cohort: **matched**; **censored** (every
valid configuration misses); **unresolved-match** (no config meets, at least one is unresolved); **invalid** (no valid
configuration). Per wide cell and τ on val32:

| reference | bank | competitor side | label |
|---|---|---|---|
| unqualified (A3.3) | any | any | no evidence (reference) |
| qualified | matched | matched, $S\ge2$ | candidate win |
| qualified | matched | matched, $S<2$ | loss |
| qualified | censored | matched | loss |
| qualified | matched | censored / unresolved-match / invalid | no evidence (no ratio; a speed win needs a matched competitor) |
| qualified | unresolved-match / invalid | any | no evidence |
| qualified | censored | not matched | no evidence |

Confirmation of a candidate win on test32: test reference unqualified → **unconfirmable** (no evidence); else confirmed iff
both frozen finalists meet τ, the bank passes Q1/Q2 in-job, and $S_{\rm test}\ge2$ (same max–min rule); a measured failure
of any of these → **not confirmed**. Cell: **win** iff ≥ 1 confirmed τ; else **loss** iff ≥ 1 τ is loss or not confirmed;
else **no evidence**. Geometry: **WIN** iff both wide levels win; **otherwise the geometry claim is WITHDRAWN** (the brief's
"Otherwise withdraw"), sub-labelled *withdrawn (measured loss)* if some level is loss, else *withdrawn (insufficient
evidence)*. **Attribution** (WIN only), per level at a confirmed τ: *withdrawn* if the box control gives
$S_{\rm box}\ge S_{\rm geom}/2$ on **either** cohort where both are measured (val or test); *kept* iff both cohorts are
measured with $S_{\rm box}<S_{\rm geom}/2$ and the $D_{\rm bnd}$ gate passes on both; otherwise *unestablished*.
WIN-attributed iff each level has a kept τ; else WIN-unattributed. E2 = best over geometries of WIN-attributed >
WIN-unattributed > WITHDRAWN.
**Speed statistic (deterministic):** with competitor finalists $\mathcal A$ (≤ 2 per family, all families) and bank
finalists $\mathcal B$ (≤ 2), $S=\max_{b\in\mathcal B}\min_{a\in\mathcal A}S(a,b)$, $S(a,b)$ = median paired ratio (A3.4);
ties broken by the lexicographically smaller configuration key; single finalists are the degenerate case. Identical
finalist pairs are timed once and reused across (τ, statistic).
**Competitor eligibility** is benchmark eligibility (valid, converged, meets τ), distinct from numerical qualification;
competitors' Q1/Q2 are reported (A3.4 wording "narrowing" withdrawn: COMMON-BRIEF §3 defines qualification, not
benchmark admission).

**A4.3 Certificates (replaces the ETDRK4 and CN parts of A3.5).**
- *ETDRK4:* every step, (i) the update is recomputed by an **independent transcription** of the Cox–Matthews step with
  the closed-form/series coefficients; per-step relative defect ≤ $10^{-9}$; (ii) the per-query coefficient certificate
  (A3.5) ≤ $10^{-10}$; (iii) **any** non-finite stage or growth-bound violation invalidates the run (no 1 % allowance).
  There is no algebraic tolerance in an explicit step; (i)–(ii) are its stated analogue of the tight/tighter check.
- *Galerkin CN:* $\lVert R(z)\rVert/(\lVert M(z-z_n)\rVert/\Delta t+10^{-14}\lVert Mz\rVert/\Delta t+10^{-300})\le10^{-8}$.
- *Linearised CN:* $\lVert J\delta+R\rVert/(\lVert R\rVert+10^{-14}\lVert Mz\rVert/\Delta t+10^{-300})\le10^{-10}$; a zero
  residual passes.
- *Q2 denominator:* the maximum is over the full population (all 40 cases × 6 output states) of $\lVert N_{q+2}\rVert$;
  states below $10^{-12}$ of it are skipped; more than 5 % skipped fails Q2.

**A4.4 Controls (fix to A3.6).** MF-MATCH-real fires only if an independent loop recomputation of the faulted worst error
satisfies $e_{\rm fault}-U>\tau$ **and** the matcher returns *misses*. MF-Q2 fires only if the directly computed
$\rho_{\rm fault}>0.116$ **and** the shared Q2 predicate (`finalize.rho_ok`) flags it. Both report
`violation_established`. MF-Q1 and MF-TENS as A3.6. The final v4a cell-control evidence (local CPU smoke) is a release
prerequisite and is listed in the ready note.

**A4.5 Cost certificate status.** `runs/cost/projection.json` (`resources.py`) exists and is **provisional**: its listed
unmeasured units (bank steady-state step time, ECSW NNLS fit time, the selection/timing stage) are estimates. The cluster
smoke allowance (1 GPU-h) is spent (1.07 GPU-h, 0.07 over, recorded in the ledger), so the bank-step and ECSW probe
(`cluster/bodies/smoke6.sh`, ≤ 0.15 GPU-h) is proposed as the first released job; production cells are submitted only
after the projection is re-certified with it. Peak memories are taken from the probe JSONs per stage (all below 6 GB
except POD/ECSW, bounded by the chunking of pod.py and A4's 66-state ECSW population).

---

## Amendment A5 (2026-10-21, after Codex design audit 4 = NOT-READY on certificates and cost; before production)

Audit `audits/codex-design-4.md` (A4.2 and A4.4 CORRECT; A4.3 WRONG; A4.5 NEEDS-RESTATEMENT).

**A5.1 ETDRK4 stage certificate (replaces A4.3's first bullet).** Every step, each of the three intermediate stages
$s_2,s_3,s_4$ and the update are recomputed by the independent transcription with the closed-form/series coefficients;
the per-step defect is $\max$ over the four vectors of $\lVert x-x'\rVert/(\lVert x'\rVert+10^{-14}\lVert y_n\rVert+10^{-300})$,
bar $10^{-9}$ (frozen). **Q2 populations:** validation = dev8 + val32 (40 cases × 6 output states); confirmation = test32
(32 × 6). States with $\lVert N_{q+2}\rVert=0$ or $\le10^{-12}$ of the population maximum are skipped; Q2 fails if more than
5 % are skipped (so an all-zero population fails).

**A5.2 Selected production panel and cost (replaces A4.5).** `runs/cost/projection.json` now uses **conservative bounds**
for every unmeasured unit (bank 3.0 h per training, SEM screen 30 s per row, selection/Q1/Q2/timing stage = 1.0 × the
ladder, S1 ladders = S2 ladders) and quantitative memory bounds (host training set 10.8 GB; device POD chunk 1.1 GB,
POD QR 2.7 GB, ECSW matrix 8.9 GB, tensor chunk 1.8 GB, bank data 1.2 GB; all within an A100-80GB, job memory 120 GB).
**Selected panel:** the four wide-family geometry cells, the two wide-family box cells, their test jobs and up to two
conditional box banks; **all narrow-family cells are deferred** (defer items 2–3 of A1.10). Under the conservative bounds
this panel projects above the cap (value in the JSON, `panel_selected_with_headroom_h`), so **production is not
releasable on the current certificate**. Release sequence: (1) the measurement job `cluster/bodies/smoke6.sh` (bank
steady-state step time, ECSW NNLS fit time; ≤ 0.15 GPU-h) replaces the two largest bounds by measurements; (2) the
projection is re-certified; (3) if spent + projection still exceeds 62 GPU-h, the following pre-registered reductions are
applied in order until it fits, each recorded in an amendment: (a) reduced-arm $r$ ladder $\{16,32,64,128,256,384\}$;
(b) selection screen $K=3$; (c) C-SEM grading $g\in\{\text{off},0.5\}$; (d) the wide S2 H cell deferred (the H geometry
verdict then rests on S1 and is UNRESOLVED at S2). Competitor families, their solver options and the paired timing protocol
are never reduced.

---

## Amendment A6 (2026-10-21, after Codex code audit 1 = NEEDS-FIX; implementation record, before production)

Audit `audits/codex-code-1.md`. Changes that touch the design (A6 governs where different):
- **Search coverage.** C-SEM: the spatial screen at fine steps is the size ladder; step ladders (orders 2 and 3) and the
  over-integration variant run on **every** spatial configuration whose fine-step error meets τ = 0.1 on either statistic
  (a configuration that misses the loosest target with a converged step cannot meet it with fewer steps); the `eig`
  solver and the bank's tensor form give the same outputs as `pcg` / point form to solver tolerance, so they are
  **variants screened in selection** (for every candidate whose screen time is within 10× of the family's fastest) and
  re-run officially if selected. C-ECSW runs all four steppers (Galerkin steppers share the $M_t=r$ fit). R-BANK runs the
  step ladder at **every** rule $q$.
- **Selection** orders candidates by a **warm single-query screen time** measured for every valid ladder row (median of
  3 after compilation), not by batched compile-inclusive seconds; all candidates are considered in that order.
- **Validity** everywhere (ladder, selection, confirmation) includes the solver statistics (`ladder.stats_valid`): no
  unconverged MG/PCG solve; reduced Newton/LSPG: no non-finite step, no damping saturation, unconverged LM steps ≤ 1 %
  (LSPG) / none (Galerkin); ETDRK4: no non-finite stage, no growth violation, $\mathrm{cond}(V)\le10^8$ enforced (else the
  row is invalid). Q1 for Galerkin and linearised CN requires the certificate at **every** step.
- **Certificates are untimed.** The ETDRK4 coefficient and stage certificates run only in the certificate replay, whose
  outputs must equal the production outputs (T-id). Implementation of the independent coefficients: Cox–Matthews
  $\varphi_1,\varphi_2,\varphi_3$, series (24 terms) for $|z|<1$, closed forms otherwise (replaces A3.5's "expm1 / 0.2 /
  8-term" wording; same purpose, frozen as implemented).
- **Q2** additionally checks the tested $M_t\times r$ mass and stiffness blocks and the IC projection (A1.8 bar).
- **Timing integrity.** The outputs of the timed calls are stored and compared with the official outputs (T-id ≤
  $10^{-12}$); a pair with T-id failure or any compilation during timing is excluded from S. T-dbl is serial (two blocked
  calls in one timed region). A sentinel (first competitor finalist) is timed before and after; drift > 10 % marks the
  timing invalid. Identical finalist pairs are timed once (validation and test).
- **Test-once guard.** `stage_time` writes `frozen.json` (sha256 of manifest, bank, POD basis, ECSW weights, references,
  descriptors; code commit). The test job runs only if the **committed** copy (staged with the code from `git archive
  HEAD`) equals the job's `frozen.json` byte for byte and every artefact hash still matches; test reference labels are
  computed only in the test stage.
- **POD IC** is the full data-grid projection (A1.5 / A2.3 registered form; the 128-grid sub-lattice is removed).
- **C-RK4** starts its ladder at the stability estimate with the correct Shortley–Weller fraction.
- **ECSW candidates:** 16,384 uniform plus 16,384 importance-sampled from the remaining nodes, without truncation.
- **Controls:** MF-MG bar 0.2 per cycle, measured at the **finest reference level** inside each production reference
  stage (`ref.json:MF_MG_finest`); MF-QUAD adds $\int\varphi^2$ with bar $10^{-6}$ (the L envelope has a kink at the corner,
  observed $3.6\times10^{-9}$ at the production rule); MF-Q1 compares flagged **step identities**; MF-MATCH perturbs the FD
  field itself.
- **Bank rule subdivision** $k_q=4$ is fixed (the §5.3 selection rule is withdrawn; Q2 qualifies the rule ladder at that
  $k_q$). The **cross-span residual diagnostic** of A1.6 is dropped (not implemented). The training-data accuracy, the
  SEM reference cross-check and the G-FLOOR check run in a `diag` stage on dev cases (reported).
- **Cluster:** staging is `git archive HEAD` (an uncommitted lane refuses to submit); one campaign-lock section reserves
  the job directory under both caps (lane 3, account 8) and another submits; the batch script fails when any stage fails;
  an execution lock prevents two runs in one directory; every stage is a separate process.
- **Decision table** implemented in `decide.py` (A4.2), using `final.json:side_status` for matched / censored /
  unresolved-match / invalid.
- **Release protocol (cost).** Because the certificate is provisional (A5.2), the first released production job is one
  full wide cell (L | wide | S1) together with `smoke6`; its measured GPU-hours re-certify the remaining panel before any
  further submission.

---

## Amendment A7 (2026-10-21, after Codex code audit 2 and the local CPU cell smoke; before production)

- **Q2 gates = the brief's bars** (COMMON-BRIEF §3): ρ ≤ 0.116 on the arm's own reached states and rollout distance ≤
  $2.5\times10^{-4}$ on every case, plus valid successor solves and the ≤ 5 % skip rule. The extra mass / stiffness / IC
  difference bars of A1.8 / A3.5 become **reported** (the local smoke showed them failing at $10^{-6}$ while carrying no
  information beyond the rollout bar).
- **Bank rule ladder** extended (A6's fixed $k_q=4$ kept for $q\in\{4,6,8,10,12\}$) with a finer tier on $2k_q$ elements:
  `f8`, `f12` (successors 14 → f8 → f12 → f16), so a rule-converged bank exists if the features need more points.
- **T-dbl (replaces A1.9 / A6):** $t(d_1;d_2)/(t(d_1)+t(d_2))\in[0.9,1.1]$ (two different cases back to back vs the sum of the single calls); the 1.8–2.2× form is confounded by case-dependent iteration counts (local smoke: C-SEM 1.71).
- **Smoke only:** `--smoke` adds τ = 0.2 and relaxes the bank's Q2 gate (recorded per item as `Q2_gate_relaxed_smoke`) so
  the selection/timing path of an untrained smoke bank is exercised; production never relaxes it.
- Code-audit-2 fixes (recorded in the code and commits): Q1 gates LSPG unconverged steps; Q2 replays must be valid
  solves; every ETDRK4 stage checked finite; the max–min statistic requires a **complete** valid timing matrix for a bank
  finalist (else that finalist is excluded; no S if none remains); T-id over every timed output; sentinel drift or a
  failed T-dbl voids S; side status distinguishes *unqualified* (no evidence) from *censored*; the test stage recomputes
  each (τ, statistic) verdict, gates its pairs on test accuracy / validity / bank Q1+Q2, counts compilations and runs
  T-dbl; the freeze check requires the staged committed copy `code/runs/frozen/<cell>.json`, unchanged descriptors and
  artefacts (incl. `ref.json`, dev references); submission archives the recorded commit and fails closed on queue errors.
- **Local CPU smoke evidence (v4a, L | wide | S1, smoke sizes):** `runs/smoke/local-cellL/` — cell controls all fire
  (`controls_L.json`), MF-MG at the finest smoke reference level and MF-REF on real data fire (`ref.json`).

---

## Amendment A8 (2026-10-21, coordinator decision; pre-data, before any production job)

- **Registered production scope:** the four wide-family geometry cells (L, H × S1, S2), their two box controls, and
  conditional box banks (only for cells whose geometry wins). **All narrow-family cells are deferred** (the win/withdraw
  rule is judged on the wide family). Cap: 62 GPU-h in total, including the 1.07 GPU-h of smokes already spent.
- **Order if the re-certified wide panel exceeds the remaining cap:** apply the A5.2 reductions (a) r ladder
  {16,32,64,128,256,384}, (b) selection screen K = 3, (c) C-SEM grading {off, 0.5}; then defer the wide S2 H cell. Each
  application is recorded in a dated amendment before the affected jobs run.
- **First released step** (≤ 12 GPU-h): the measurement job `meas6` (isolated bank steady-state step timing) and the full
  cell L | wide | S1 with the incumbent CN-R LSPG stepper. ECSW fit cost is measured inside that cell (per-fit wall times in
  its ladder/stage records), not by a synthetic probe. Re-certification (`resources.py --measured`) then uses the cell's
  measured stage seconds and peak memories. If E5's winner later passes, it is added as an extra bank arm only if the
  budget allows.

---

## Amendment A9 (2026-10-22, after the first cell timed out; before any result is read)

- **What happened:** job 6175736 (L | wide | S1) hit its 11.5 h limit inside the ladder stage (4974 rows; C-SEM 4.7 h,
  C-ECSW 2.5 h, R-BANK 2.0 h at about 70 % of its ladder, R-PODG 1.0 h; ref / data / bank 0.86 h together, peak device
  5.0 GB, host 14.8 GB). Everything is archived (`burgers-archive/imp-geometry/L_wide_S1`, SHA256SUMS) and collected
  (`runs/cells/L_wide_S1/`). No selection, timing or test stage ran; no result was read.
- **Reductions applied in the registered order (A5.2 / A8):** (a) reduced-arm $r\in\{16,32,64,128,256,384\}$;
  (b) implementation variants screened only for the 3 fastest candidate rows; (c) C-SEM grading $\{\text{off},0.5\}$.
  Ladder rows recorded before the reduction outside these sets never enter selection (`run_cell.registered`).
- **Implementation (cost only, same numbers):** the bank ladder runs in tensor form from now on (identical outputs to
  $10^{-9}$, MF-TENS; the point form is the screened variant); the ladder resumes from recorded rows (decisions replayed
  from the stored rows; ECSW fits reused).
- **Re-certification** (`runs/cost/projection_A9.json`, measured family costs + listed assumptions; remaining cap
  62 − spent): the full wide panel and the panel minus the wide S2 hole cell do **not** fit with the 1.5 headroom; the
  geometry cells L S1 (resume), H S1, L S2 do. **The wide S2 hole cell is deferred** (registered order). The box controls do
  not fit in either case; whether to drop or make them conditional is the coordinator's decision (without them no
  geometry attribution can be established, and with H S2 deferred geometry H cannot be a WIN under A4.2).
- The resume job restores the archived cell and runs ladder → inter → time → diag → freeze.
- **A9 addendum (after `audits/codex-job5-resume.md`):** the SEM / FD expansion frontiers use only base step-ladder
  rows (base tolerance, `pcg`, registered grades), so a resumed run expands exactly the frontier an uninterrupted run under
  the A9 rules would (on the archived checkpoint this adds 20 rows); selection pre-ranks by stored screens (top 12) and
  re-screens those rows and their variants in the current job, ordering by the fresh screens. The resume job needs a
  coordinator authorization beyond A8's 12 GPU-h step cap (11.62 h spent in step 1).
- **A9 addendum 2 (after `audits/codex-job6-resume.md`):** selection re-screens, in the current job, every candidate whose
  stored screen is within 3× of the family's best stored screen (6× if the fresh/stored ratios vary by more than 1.5×;
  the drift is recorded); variants for the 3 fastest by fresh screen; ordering by fresh screens only (replaces the
  "top 12" rule of the first addendum).
- **A9 addendum 3 (after `audits/codex-job7-resume.md`):** selection re-screens **every** eligible candidate in the
  current job (no window, no stored-screen ordering), with a per-job screen cache shared across targets and statistics;
  measured screening proxy on the checkpoint ≈ 2.4 GPU-h, so the selection/timing stage is now taken as 4.0 h (S1).
  Re-certified (`runs/cost/projection_A9.json`): only **one wide cell per geometry** (L S1 resumed + H S1) fits the
  remaining cap with the 1.5 headroom; L S2 does not fit in addition. Under A4.2 a geometry WIN needs both levels, so with
  this panel neither geometry can be a WIN (outcome at best WITHDRAWN (insufficient evidence)) unless the coordinator
  changes the cap or the rule. This is stated plainly to the coordinator; no further cell is submitted without a decision.

---

## Amendment A10 (2026-10-22, coordinator decision; recorded BEFORE any S1 selection, timing or test data exists)

- **Step 2 (authorized):** resume L | wide | S1, then H | wide | S1 (H job audited before submission); hard stop at
  30 GPU-h spent in step 2. The win/withdraw rule (A1.1 / A4.2) is **unchanged**.
- **S1 readout:** after both S1 cells are scored (selection, timing, test-once), S against the best competitor per target
  and geometry is computed with the audited `decide.py`. If neither a WIN nor a withdraw is reached, it is reported as
  **"S1-only, wide family; WIN not decidable without S2 and box controls"**.
- **Pre-registered conditional:** if $S\ge2$ against the best competitor at some target (worst error, val32, both matched
  configurations qualified and the target confirmed on test-once, as A4.2's "candidate win confirmed") on at least one
  geometry at S1, the coordinator funds L | wide | S2 and the box controls for the S1 cells from the E4 reserve,
  re-certified from step-2 measurements. If $S<2$ at every target on both geometries at S1 (A4.2 "loss" or "no evidence"
  at every τ), the lane stops and reports the geometry claim as **"not supported at S1; S2 untested"**.
