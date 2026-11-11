# Anisotropic diffusion on stretched grids: does a regime exist where the ROM beats strong solvers?

Protocol fixed 2026-11-11 before any solver outcome was seen. Later changes are appended under
"Amendments" with the reason. Lane `exp/2026-11-11-aniso-stretched`, cluster namespace
`/cluster/tufts/paralab/tawal01/rebuttal-aniso/`.

## Question

Reviewer iMp8: the paper's benchmarks are solved in milliseconds by FFT / multigrid / GPU direct
solvers. On constant-coefficient boxes that is true. Is there a parametrised diffusion family on
which an accuracy-matched strong solver is clearly slower than a ROM query? Prior: sceptical
(variable-coefficient and non-box campaigns found no ROM advantage).

## PDE families

$-\nabla\cdot(K(x;\mu)\nabla u) = f(x;\mu)$ on $(0,1)^2$, $u=0$ on the boundary.
$K = R(\theta)\,\mathrm{diag}(1,\varepsilon)\,R(\theta)^T$ (unit diffusion along the direction at
angle $\theta$ to the $x$ axis, $\varepsilon$ across it).
Source $f = 1 + 10\exp(-|x-c|^2/(2\sigma^2))$, $\sigma = 0.1$.

- **Family A (rotated, constant tensor).** $\theta$ constant in space.
  $\mu = (\theta, \log_{10}\varepsilon, c_x, c_y)$, $\theta\in[15^\circ,75^\circ]$,
  $\log_{10}\varepsilon\in[-4,-2]$, $c\in[0.3,0.7]^2$. No fast transform applies (cross term);
  no wall layers expected (field lines cross the walls).
- **Family B (bent field lines, wall layers).** $\theta(x,y) = \theta_m\sin(\pi x)\sin(\pi y)$.
  $\mu = (\theta_m, \log_{10}\varepsilon, c_x, c_y)$, $\theta_m\in[10^\circ,40^\circ]$, other ranges
  as in A. Field lines are tangent to the walls $y=0,1$, giving layers of width
  $\sim\sqrt{\varepsilon}$ there; the coefficient is variable and non-separable.

Parameters are drawn uniformly with fixed seeds: dev cohort (16 cases, seed 1101, used for every
choice of mesh/tolerance/setting) and held-out cohort (16 cases, seed 1102, used once per arm for
the reported numbers). Training parameters for any ROM use seed 1103.

## Discretisation

Bilinear (Q1) finite elements on a tensor-product grid with $N$ intervals per direction, $K$
sampled at element centres, consistent mass for the load. Node map
$x = \tfrac12\,(1+\tanh(\beta(2\xi-1))/\tanh\beta)$ per direction ($\beta=0$: uniform). The matrix
is symmetric positive definite for every $\mu$.

## Reference and metric

Reference: same discretisation on the evaluation grid $N=2048$, $\beta_x=\beta_y=2.5$, sparse
direct solve. Its own error is estimated from the $N=512,1024,2048$ sequence (Richardson) on four
dev cases incl. the smallest $\varepsilon$; it must be $\le 0.25\times$ the target. If it is not at
the smallest $\varepsilon$, the $\varepsilon$ range is truncated to where it is, as an amendment.

Primary metric: relative $L^2(\Omega)$ error on the evaluation grid (lumped-mass weights), solver
output interpolated bilinearly. Secondary: unweighted nodal relative $\ell^2$ on the evaluation
grid (over-weights the layers) and relative max-norm error. Targets: mean error over the cohort
$\le 1\%$ (primary), also $3\%$ and $0.3\%$; worst case is reported next to the mean.

## Comparators (each chooses its own mesh, tolerance and settings on the dev cohort)

Mesh choices: $N\in\{32,48,64,96,128,192,256,384,512,768,1024\}$,
$(\beta_x,\beta_y)\in\{0,1.5,2.5\}^2$.

- (a) AMG-preconditioned CG: pyamg smoothed aggregation (symmetric and evolution strength) and
  Ruge–Stüben (classical strength), V-cycle; relative-residual tolerance from
  $\{10^{-2},\dots,10^{-8}\}$. Timed as setup + solve (K depends on $\mu$), with solve-only and a
  frozen-hierarchy variant (hierarchy built once at the centre parameter) reported as well.
- (b) Sparse direct (SuperLU on CPU; cuDSS / GPU direct if it can be imported on the cluster):
  refactorise per query since K depends on $\mu$. Factor and solve times reported separately.
  Extra strong variant: CG preconditioned by a factor frozen at the centre parameter.
- (c) Unpreconditioned CG (the paper's named baseline) and Jacobi-CG.
- (d) Coarse-grid control: every solver above on every mesh choice, interpolated to the evaluation
  grid. The reported comparator time at a target is the minimum over (solver, mesh, tolerance) of
  the median warm time among configurations whose dev-cohort mean error meets the target; the same
  configuration is then timed and scored once on the held-out cohort.
- (e) If (a)–(d) say "go": a higher-order control (Q2 elements, direct solve) is added, because
  the grid-misalignment error of Q1 is a property of the discretisation, not of the problem.

Time = wall time per query, warm (first call discarded), median of $\ge 5$ repeats with min/max,
matrix assembly excluded for all solvers and reported separately (the ROM does not assemble).
CPU numbers are from the local 20-core Arm box (scipy/pyamg are single-threaded in the hot loop);
GPU numbers, where an implementation exists, from the same GPU type as the ROM.

## ROM arms (only if the decision rule says go)

Coordinate bank + linear span (and head) trained on Q1 solutions of the family with the repo's
Poisson pipeline, restricted bank widths as the tuning knob; POD-Galerkin on the same snapshots as
the classical control. Operator precomputation: Family A is exactly affine with three terms
($K_{xx},K_{xy},K_{yy}$ are scalars in $\mu$); Family B is not affine in $\theta_m$ and needs an
empirical-interpolation expansion of $\cos 2\theta,\sin 2\theta$ in $\theta_m$ (or empirical
quadrature); whichever is used is stated with its truncation error. Same metric, same held-out
cohort, time per query warm with spread, training + setup cost, break-even count
$= \text{offline cost}/(t_\text{solver}-t_\text{ROM})$.

## Decision rule

Measure comparators first, CPU only. For each family and the 1 % target:
- **No-go:** the best accuracy-matched comparator costs a few milliseconds (< 20 ms). Report the
  negative and stop; no GPU is spent on ROM training.
- **Go:** it costs $\ge 20$ ms on CPU *and* nothing indicates a GPU direct/AMG solver would close
  the gap to below a plausible ROM query (the paper's Poisson span costs 0.25–11 ms at
  $256^2$–$2048^2$). Then train the ROM and report the head-to-head, including (e).

## Amendments

(none yet)
