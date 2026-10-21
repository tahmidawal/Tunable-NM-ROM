# Spectral pilot: can a reduced-order model beat the spectral solver, and what decides it? (pre-registration)

**Status (2026-10-20): design, pre-registered before any pilot code runs.** Every number this lane produces is
labelled **PILOT: small, not a result.** The pilot looks for direction and feasibility, so that a later cluster lane
can be sized. It does not replace any campaign verdict. Amendments are appended at the bottom, dated; the text above
them is never rewritten.

Lane: `exp/2026-10-20-spectral-pilot` (worktree `worktrees/2026-10-20-spectral-pilot`, forked from
`exp/2026-10-20-jcp-consolidated` @ `93f399faa`; mirror `origin/codeonly/exp/2026-10-20-spectral-pilot`). The
worktree is sparse: only the top-level files and this folder are checked out. Other lanes are read from
`worktrees/2026-10-20-jcp-consolidated`. No cluster jobs. Local jobs: at most 2 at once, each at most 30 minutes.

## 0. Background (what is already known)

- The closed Burgers campaign (`reports/2026-10-17-full-experiment-plan/final/report.md`) found that **K2 fires** in 2D
  and 3D: on the Dirichlet box a coarse sine pseudo-spectral FOM (ETDRK4, $N=24$–$48$) matches the ROM's accuracy
  faster than every qualified ROM configuration.
- The ideation report (`reports/2026-10-20-improvement-ideation/report.md`) found that on the median in-range case a
  Laplacian-ordered sine basis is at least as compressive as the learned bank. The bank wins only on the worst case
  (about 3.4× fewer unknowns). The spectral FOM turns that small gap into a win through cheap unknowns: fast
  transforms, an exact linear part, 10–20 explicit steps, and a cheap output synthesis.
- jcp-regimes found that the incumbent bank cannot represent steeper fronts than it was trained on (r = 384 floor
  3.5 % on C1, 9.9 % on C2). It was never trained on such data.

This pilot asks two things the campaign did not.

- **(A)** Does the comparison flip as fronts sharpen, *when the reduced basis is trained on data of that steepness*?
- **(B)** Does the ROM win on a domain where no fast sine transform exists?

## 1. Problem

Viscous Burgers with advection along $(1,1)$, as in the campaign:
$$u_t + u\,(u_x+u_y) = \nu\,\Delta u \quad \text{in } \Omega,\qquad u=0 \text{ on } \partial\Omega,\qquad t\in[0,0.25],$$
with outputs at $t_k = 0.05k$, $k=0,\dots,5$. The initial condition is one Gaussian bump
$u_0 = a\exp\!\big(-|x-c|^2/(2w^2)\big)$, wall values set to zero, as in the campaign family. The nominal steepness is
$s=a/\nu$. A viscous Burgers front of jump $\Delta u\approx a$ has width of order $4\nu/a = 4/s$.

## 2. Hypotheses and falsifiers

### Part A: the box $\Omega=(0,1)^2$

Let $\tau$ be an error target and let $N^*(s,\tau)$ be the smallest spectral resolution, and $r^*(s,\tau)$ the
smallest reduced dimension, that reach worst-case error $\le\tau$ on a steepness level's test cases (§5).

- **H-A1 (spectral resolution grows with steepness).** Over the levels, $N^*$ grows like $\bar s^{\alpha}$ with
  $\alpha \ge 0.5$ at each $\tau$, where $\bar s$ is the level's median steepness.
  *Falsified* if the fitted $\alpha<0.5$ at every $\tau$ that is reached at three or more levels.
- **H-A2 (the trained basis needs fewer coefficients).** $r^*(s,\tau)$ grows more slowly in $\bar s$ than the
  number of spectral unknowns $(N^*-1)^2$. *Falsified* if $r^*/(N^*-1)^2$ does not decrease from the lowest to the
  highest level at every $\tau$.
- **H-A2′ (the cost-relevant version).** The measured cost model (§6) makes the ROM's work per step scale like
  $r^3$ (exact quadratic tensor) or $6mr$ (quadrature), while the FOM's scales like $N^2\log N$. A crossover therefore
  needs $r^*$ to fall below roughly $N^*$ (per axis), not below $(N^*)^2$. H-A2′ says $r^*/N^*$ **decreases** with
  steepness. *Falsified* if $r^*/N^*$ at the highest level is not below its value at the lowest level for any
  $\tau$.
- **H-A3 (crossover).** The matched speed ratio $S(s,\tau)=t^*_{\rm FOM}/t^*_{\rm ROM}$ rises with steepness and
  reaches 1 inside, or within a short extrapolation of, the pilot range. Pilot reading rules, fixed now:
  - *crossover reachable*: for some $\tau$, $S\ge 0.5$ at the highest feasible level and the least-squares slope of
    $\log S$ against $\log\bar s$ over the levels is positive, with the extrapolated $S=1$ at $\bar s\le 2000$;
  - *not reachable in this family*: $S<0.25$ at the highest feasible level for every $\tau$ **and** the slope is
    $\le 0$ for every $\tau$;
  - *undecided*: anything else.
  - The campaign's K2 bar is $S\ge 2$. It is reported alongside, not used for the reading.

The alternative this design must be able to detect: the comparison flips **the other way** (the ROM wins on smooth
data and loses on steep data). The reading rules are symmetric in the sign of the slope, and the report must say so
if that is what the pilot shows.

### Part B: an L-shaped domain

$\Omega_L = (0,1)^2 \setminus [\tfrac12,1]\times[\tfrac12,1]$, zero Dirichlet data on all six edges, including the two
re-entrant edges. No sine or cosine transform diagonalises the Laplacian on $\Omega_L$, so the box spectral FOM does not
apply.

- **H-B (ROM wins without a fast transform).** At some $\tau\in\{10^{-2},\,3\times10^{-3}\}$ the matched speed ratio
  of the optimistic ROM (§4.2) over the best finite-difference FOM of the ladder (§4.3) is $S_B\ge 2$.
  Pilot reading: *promising* if $S_B\ge2$ at some $\tau$; *marginal* if $1\le S_B<2$ at best; *ROM loses even
  optimistically* if $S_B<1$ at every reached $\tau$.

## 3. Cohorts

All cohorts are new, public, row-wise draws: row $j$ comes from `np.random.default_rng([seed, j])`. Seeds are fixed in
the code and are disjoint from every campaign seed. No sealed cohort is created or opened. Training and test
descriptors are asserted to be disjoint.

**Part A levels** (box; $a\sim U[1,2]$, $c_x,c_y\sim U[0.15,0.85]$, $w\sim U[0.05,0.20]$ as in the campaign family;
$\log\nu$ uniform in the band):

| level | $\nu$ band | $s$ range | role |
|---|---|---|---|
| A1 | $[0.05, 0.10]$ | 10–40 | smooth end |
| A2 | $[0.015, 0.03]$ | 33–133 | campaign range |
| A3 | $[0.005, 0.01]$ | 100–400 | overlaps jcp-regimes C1 |
| A4 | $[0.0025, 0.005]$ | 200–800 | overlaps C2; **conditional** on its reference passing the gate inside the time cap |

Per level: 96 training trajectories (seed $10^6+\ell$), 12 test trajectories (seed $2\times10^6+\ell$). A
data-sufficiency check retrains the basis on the first 48 training trajectories at every level (§7, D1).

**Part B** (L-shape, one level): $\nu$ log-uniform in $[0.02, 0.05]$ ($s$ 20–100), $a\sim U[1,2]$,
$c_x,c_y\sim U[0.20,0.30]$, $w\sim U[0.035,0.05]$. The centres keep the IC at most $3.4\times10^{-4}a$ on the
boundary (zeroed there); the flow then carries the bump into the re-entrant corner region. 96 training and 12 test
trajectories (seeds $3\times10^6$, $4\times10^6$). A second, steeper Part-B level ($\nu\in[0.007,0.02]$) runs only if
time allows; it is labelled exploratory.

## 4. Arms

### 4.1 Part A FOM: sine pseudo-spectral ETDRK4

The E0 cross-check solver `vendor/pscore.py` (byte-identical copy; sine coefficients $1\le k_a\le N-1$, exact linear
part, 3/2-padded products, ETDRK4 with Kassam–Trefethen contour coefficients). This is the solver family that
won K2 in jcp-fom2d. The lane wraps it with a flexible output schedule; its numerical kernels (`nonlinear`,
`etdrk4_coefs`, `ic_coefs_2d`) are imported unchanged.

- Ladder: $N\in\{16,24,32,48,64,96,128,192,256,384,512\}$ (capped at $N_{\rm ref}\cdot 3/4$), steps
  $n\in\{5,10,20,40,80,160,320,640,1280,2560\}$ ($\Delta t = 0.25/n$). Per $N$, only the $n$ with
  $0.25\cdot\Delta t^{-1}$ within a factor 8 of the advective estimate $n_{\rm cfl}=0.25\,u_{\max}N/0.6$ are run,
  ($u_{\max}=2$); larger steps that blow up are recorded as invalid.
- Initial condition: exact sine coefficients of the continuous $u_0$ (`pscore.ic_coefs_2d`).

### 4.2 Part A ROM: an *optimistic* linear-span ROM

The pilot does **not** retrain the coordinate-network bank (about 3 A100-hours per bank and a re-qualified rule per
bank). It uses a proxy that is optimistic for the ROM in each part of its cost and accuracy, so that a *negative*
pilot result transfers to the real bank and a *positive* one does not (it only licenses the full experiment):

- **Span.** POD of the level's training snapshots (96 trajectories × 21 times, $t = 0, 0.0125,\dots,0.25$), computed
  in sine-coefficient space at $N_{\rm ref}$ (Parseval: the Euclidean coefficient inner product is the $L^2$ one up
  to a constant). POD is the $L^2$-optimal span for its training data. The bank is also a linear span (Dirichlet-exact
  through its envelope), so its online cost at equal $r$ cannot be below this ROM's cost model, and its floor on data
  like the training data is not expected to be lower. Held-out floors of both are compared where data exist
  (§8 R5).
- **Projection.** Galerkin in that inner product.
- **Nonlinear term.** Burgers is exactly quadratic, so the Galerkin term is a third-order tensor
  $T_{kij} = -\langle\psi_k,\ \psi_i\,(\partial_x+\partial_y)\psi_j\rangle$, assembled offline by the FOM's own
  3/2-padded quadrature (exact for these band-limited products) and stored symmetrised in $(i,j)$, i.e. $r^2(r+1)/2$
  entries. Online cost: about $r^3$ flops per evaluation, independent of the mesh. The quadrature alternative
  (our method's form, $6mr$ per evaluation with $m$ Gauss points) is costed with the same harness and the cheaper of
  the two is charged to the ROM (§6).
- **Time stepping.** The F1b idea of the ideation report: rotate the basis so the reduced Laplacian
  $\Phi^\top\Lambda\Phi=VDV^\top$ is diagonal ($\nu$-independent, offline), then ETDRK4 with exact linear part. No
  Newton solve, no Jacobian. Step ladder as in §4.1.
- **Initial condition.** $z_0=\Psi^\top b_0$, with $b_0$ the exact sine coefficients truncated to
  $N_{\rm init}\in\{64,128,256,N_{\rm ref}\}$ (a tuning knob, priced in the query).
- Ladder: $r\in\{8,12,16,24,32,48,64,96,128,192,256,384,512\}$, capped at the snapshot count.
- **Controls inside the arm:** a *sine ROM* using the first $r$ Laplacian-ordered sines as the span (same tensor and
  stepper), and the *span floor* (orthogonal projection of the reference onto $\Psi_r$) and the *spectral floor*
  (truncation of the reference to $N$), which lower-bound each method's error at its size.

### 4.3 Part B FOMs: finite differences on the L-shape

Uniform grid $h=1/L$, $L$ even so the re-entrant edges are grid lines; 5-point Laplacian and central advection
$u\,(D_x+D_y)u$ (2nd order), the same discretisation as the E0 references in the box.

- **F-RK4:** explicit classical RK4 (JAX, compiled loop).
- **F-IMEX:** SBDF2 (BDF2 implicit diffusion with extrapolated explicit advection; first step IMEX Euler), the
  constant matrix $(\tfrac32 I-\Delta t\,\nu\Delta_h)$ factorised **once per query** by SuperLU (scipy `splu`;
  factorisation charged to the query), one sparse triangular solve per step.
- Ladder: $L\in\{64,128,192,256,320,384,512\}$ × steps $n$ (as in §4.1, restricted to the stable range for F-RK4;
  F-IMEX $n\in\{10,\dots,1280\}$).
- Not run (stated as a gap): high-order FE or spectral-element/Chebyshev multi-domain solvers. These are the
  strongest competitors on this domain and must be in the full experiment (§10).

### 4.4 Part B ROM

The same optimistic construction on the FD operators: POD of $L=512$ training trajectories (F-RK4), Galerkin in the
grid inner product, tensor $T_{kij}=-\phi_k^\top(\phi_i\odot(D_x+D_y)\phi_j)$ of the discrete central advection, the
reduced 5-point Laplacian diagonalised offline, ETDRK4. Initial-condition projection on a sub-lattice
$L_{\rm init}\in\{64,128,256,512\}$ (tuning knob). This ROM lives on the $L=512$ grid; a mesh-free bank with a
distance-function envelope (R-function $\phi=x(1-x)y(1-y)\cdot\big(f_1+f_2+\sqrt{f_1^2+f_2^2}\big)$,
$f_1=\tfrac12-x$, $f_2=\tfrac12-y$) is checked for *representability only*: the floor of held-out snapshots in
$\operatorname{span}\{\phi\,P_i(x)P_j(y)\}$ for Legendre degree $p\in\{8,16,32,48\}$ (D3, diagnostic, CPU least
squares).

## 5. Metrics

- **Error of one run** (as the campaign's $e$): $e=\max_{k=1..5}\lVert u_h(t_k)-u_{\rm ref}(t_k)\rVert/\lVert u_{\rm
  ref}(0)\rVert$ on the scoring lattice. Part A: the 257² box lattice $x_i=i/256$ (walls zero). Part B: the
  lattice $x_i=i/64$ restricted to $\Omega_L$ (shared by every ladder member, since every $L$ is a multiple of 64).
- **Worst** = max over the level's 12 test cases; **median** likewise. Non-finite or blown-up runs: invalid, error
  $=\infty$.
- **Targets** $\tau\in\{3\times10^{-2},10^{-2},3\times10^{-3},10^{-3}\}$ on the worst error (primary) and the same
  values on the median (secondary).
- **Matched configuration** for a method and $\tau$: the cheapest (by §6 time) ladder member with worst error
  $\le\tau$; $N^*$, $r^*$, $L^*$ are its sizes. $S=t^*_{\rm FOM}/t^*_{\rm ROM}$. Matching is in-sample and symmetric:
  both methods are tuned on the same 12 cases (labelled; a pilot has no held-out confirmation).
- **Reference acceptance.**
  - A: $N_{\rm ref}$ per level (A1 128, A2 192, A3 384, A4 768) and $\Delta t_{\rm ref}$ chosen from the advective
    estimate; accepted if $\lVert u_{N_{\rm ref}}-u_{3N_{\rm ref}/4}\rVert\le 10^{-4}\lVert u_0\rVert$ and
    $\lVert u_{\Delta t}-u_{\Delta t/2}\rVert\le10^{-5}\lVert u_0\rVert$ (both at $N_{\rm ref}$ size or the coarser
    one, max over times, every test case). A level whose reference fails is dropped from the reading (and reported).
  - B: F-RK4 at $L=1024$ and $L=512$; reference = $L=1024$; its error is estimated by
    $\lVert u_{512}-u_{1024}\rVert/3$ (2nd order), the observed order from $(256,512,1024)$ is reported, and a target
    $\tau$ is usable only if $\tau\ge10\times$ the worst estimated reference error.

## 6. Cost and fairness rules

- **Same device, same thread.** Every timed query runs on one CPU core of the GB10 (`taskset -c <k>`), with
  `OMP/OPENBLAS/MKL_NUM_THREADS=1` and `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
  intra_op_parallelism_threads=1`, f64, `JAX_PLATFORMS=cpu`. A GPU replication of the Part-A matched pairs is
  optional and labelled; the GB10 GPU is known to have corrupted results, so no GPU number enters a reading.
- **Warm timing.** Compilation and one warm-up call excluded; median of 5 repeats on each of 3 fixed timing cases
  (the first three test cases), then the median over cases. A sentinel configuration is timed at the start and end of
  each timing job; a drift above 10 % flags the job.
- **What a query includes (primary contract, "field").** From the case parameters to the six output fields on the
  scoring lattice: initial condition (FOM: coefficients or nodal values; ROM: projection at $N_{\rm init}$ /
  $L_{\rm init}$), all time steps, and output synthesis (FOM A: sine synthesis to the 257² lattice by DST via the FFT
  at the smallest multiple of 256 ≥ $N$; ROM: a precomputed lattice basis times $z$). F-IMEX includes its
  factorisation. A secondary "evolve" contract (initial condition + steps, native output) is reported too.
- **Tuned competitors.** Each method is matched over its whole ladder (size, step, init resolution); the FOM
  ladders include every family listed in §4.
- **ROM cost charge.** The ROM is timed with its real tensor wherever $r\le192$ (A) / $r\le256$ (B). For larger $r$
  the tensor contraction is timed on an array of the right shape and the matched step count (cost is shape-only for
  dense contractions); such entries are labelled "shape-timed". The quadrature form ($6mr$) is timed at the Gauss
  size $m^*$ at which the quadrature ROM's rollout differs from the tensor ROM's by $\le 0.1\tau$ on every test case;
  the ROM is charged the cheaper of the two forms.
- **Offline costs** (data, POD, tensor, rotation) are measured and reported separately, with the break-even query
  count $n_{\rm be}=t_{\rm offline}/(t^*_{\rm FOM}-t^*_{\rm ROM})$ where the ROM is faster.

## 7. Controls (must-fail and gates)

| id | control | must |
|---|---|---|
| MF1 | reference gate on a deliberately coarse "reference" ($N_{\rm ref}/4$ in A; $L=128$ in B) | **fail** the acceptance check |
| MF2 | matcher on an under-resolved FOM ($N=16$ at A3/A4; $L=64$ with the largest step in B) | be **rejected** at $\tau=10^{-2}$ |
| MF3 | timing harness on a doubled query (the query run twice inside one timed call) | time at 1.8–2.2× the single query |
| G4 / MF4 | tensor ROM vs exact-evaluation ROM (A: `pscore.nonlinear` at $N_{\rm ref}$ projected on $\Psi$; B: direct FD evaluation), 2 cases, $r\in\{32,128\}$ | agree to $\le10^{-9}$ relative (gate); a tensor with one $k$-slice zeroed must **differ** by $\ge10^{-6}$ |
| MF5 | ROM with a random orthonormal span (Gaussian, in the same coefficient space) at every $r\le256$ | **fail** $\tau=3\times10^{-2}$ |
| G6 | training/test descriptor disjointness; seeds disjoint from the campaign's | assert |
| D1 | data sufficiency: $r^*$ from 48 vs 96 training trajectories | if 96 lowers $r^*$ by more than one ladder rung at any $\tau$, the level's $r^*$ is labelled data-limited (pessimistic for the ROM) |
| D2 | span and spectral floors | every matched error must be $\ge$ its floor (a run below its floor is a bug) |
| D3 | (B) envelope-basis representability floor | diagnostic only |

A control that cannot fire on the real data is a bug in the control, and is fixed before any reading.

## 8. Readouts

- R1 per level: $N^*,r^*,L^*$, steps, times (field and evolve contracts), $S$, at each $\tau$ (worst primary, median
  secondary).
- R2 floors: $r_{\rm floor}(\tau)$ and $N_{\rm floor}(\tau)$ (smallest size whose floor is $\le\tau$) per level.
- R3 fits: $\alpha$ for $N^*$ (H-A1), the trend of $r^*/(N^*-1)^2$ (H-A2) and $r^*/N^*$ (H-A2′), the slope of
  $\log S$ (H-A3).
- R4 cost model: measured single-thread time per evaluation for the FOM nonlinear term vs $N$, the packed tensor vs
  $r$, and the quadrature form vs $(m,r)$, from which the break-even size relation $r_\times(N)$ (equal cost per
  step) is read.
- R5 (A2 only, if the data are readable without new jobs): the incumbent bank's floor vs the POD proxy's floor at
  equal $r$ on the jcp-regimes stored references (C1). Otherwise stated as not done.
- Plots: error vs time for both methods per level (Pareto), $N^*$ and $r^*$ vs $\bar s$, $S$ vs $\bar s$, Part B
  Pareto.

## 9. Compute plan

| step | where | estimate |
|---|---|---|
| A references + training data (4 levels) | local CPU, all cores, batched | 15–60 min (A4 may be dropped) |
| A POD, tensors, ladders | local CPU | 30–60 min |
| A timing | 1 core | 20 min |
| B references (L = 1024, 12 cases) + training ($L=512$, 96) | local CPU | 30–60 min |
| B ladders, ROM, timing | local CPU | 30 min |

Every job ≤ 30 min, ≤ 2 at once. No GPU-hours.

## 10. Deliverables

`DESIGN.md` (this file), code in this folder, `runs/` (JSON results, small arrays; files > 50 MB listed by SHA256),
`audits/` (Codex design, code, results audits), `make_report.py` → `report.md` with generated numbers, plots and a
glossary, and in the report a full-scale proposal for a cluster lane (arms, cohorts, GPU-hour budget, kill criteria).
A short pointer README goes on `main` in `reports/2026-10-20-spectral-investigation/experimentalist/`.

---

## Amendment A1 (2026-10-20, after Codex design audit 1, before any pilot run)

Audit `audits/codex-design-1.md` returned NOT-READY. Every item was checked against the files; the following replaces
the corresponding text above. Where A1 and the original text disagree, A1 governs.

### A1.1 Claims withdrawn

- **Withdrawn: "a negative pilot result transfers to the real bank."** POD minimises mean-square *training*
  reconstruction error; it does not bound held-out worst-case error, rollout error or the cost of other evaluation
  structures. The ROM arm is renamed the **POD–Galerkin feasibility comparator**. A loss means this registered
  implementation loses; a win licenses the full experiment. The report separates representation (floors), evolution
  error and measured cost, and any statement about other spans is scoped to that span and that cost model.
- **Withdrawn: "3/2 padding is exact".** `pscore.nonlinear` is oversampled collocation (its own docstring). The
  tensor reproduces *that discrete right-hand side* projected on the span (so the tensor ROM is the Galerkin ROM of
  the vendored FOM, to round-off), not the continuous Galerkin tensor. A padding-convergence check is added (A1.6 G9).
- **Withdrawn: "crossover needs $r$ below about $N$".** Equal per-evaluation cost gives, up to constants,
  $r_\times\sim(N^2\log N)^{1/3}$ for the tensor form. H-A2′ is restated with the *measured* $r_\times(N)$ (A1.3).
- **Withdrawn:** the front-width and CFL formulas as eligibility rules (they are rough scales only); the claim that
  the reading rules were symmetric; "not reachable" as a pilot conclusion.

### A1.2 Cohorts (paired across levels; compatible initial condition)

- **Paired design.** One draw of $(c_x,c_y,w,a,u_\nu)$ per row, $u_\nu\sim U[0,1]$, shared by every level; level
  $\ell$ uses $\nu=\nu_{\ell}^{\rm lo}\,2^{u_\nu}$. So between levels only $\nu$ changes, by a factor 3 per level.
  Training rows (seed 1 000 001) and test rows (seed 2 000 001) are drawn once.

  | level | $\nu$ band | $s=a/\nu$ range | $N_{\rm ref}$ | training trajectories |
  |---|---|---|---|---|
  | A1 | $[0.06, 0.12]$ | 8–33 | 128 | 96 |
  | A2 | $[0.02, 0.04]$ | 25–100 | 192 | 96 |
  | A3 | $[0.00667, 0.01333]$ | 75–300 | 384 | 96 |
  | A4 | $[0.00222, 0.00444]$ | 225–900 | 768 | 48 (memory), conditional (A1.7) |

- **Compatible IC.** $a\sim U[1,2]$, $c_x,c_y\sim U[0.3,0.7]$, $w\sim U[0.035,0.05]$, so $u_0\le 2e^{-18}\approx3\times10^{-8}$
  on the walls. The campaign family ($c\in[0.15,0.85]$, $w$ up to 0.2) has wall values up to about 0.75a; its sine
  series then converges only algebraically and, at small $\nu$, the wall-incompatibility layer (not the front) would
  set the spectral resolution. Changing the family isolates front steepness. This departs from the campaign family
  on purpose and is stated in the report.
- **Part B** keeps its own family (§3) with $c_x,c_y\sim U[0.22,0.30]$, $w\sim U[0.03,0.045]$ (wall values
  $\le 2e^{-12}$), $\nu$ log-uniform in $[0.02,0.05]$; 96 training, 12 test.

### A1.3 Quantities and hypotheses, per target

For a method and target $\tau$ (worst-case error over the 12 test cases):
- **minimum feasible size** $N_{\min}(\tau)$, $r_{\min}(\tau)$, $L_{\min}(\tau)$: the smallest size with any step count
  reaching $\tau$;
- **cost-optimal configuration**: the cheapest member (by the A1.5 contract) reaching $\tau$; its time is $t^*$.
  $S(\tau)=t^*_{\rm FOM}/t^*_{\rm ROM}$.
- If a method does not reach $\tau$ inside its ladder at a level, that cell is **censored** (reported as a bound,
  e.g. $S>t_{\rm FOM}^*/t_{\rm ROM}^{\max}$ is not formed; the cell reads "ROM did not reach $\tau$"). A level whose
  reference fails its gate is **unresolved**, never evidence for either side.
- Fits and trends use only levels that are eligible for that $\tau$ (reference eligible and both methods reached it),
  and need **at least three** such levels; otherwise the verdict for that $\tau$ is "too few levels".

Per-target verdicts (each $\tau$ separately; no aggregation across targets):
- **H-A1:** fitted $\alpha$ of $N_{\min}\propto\bar s^{\alpha}$ ($\bar s$ = level median of $s$). "Consistent" if
  $\alpha\ge0.5$, "inconsistent" otherwise.
- **H-A2:** "consistent" if $r_{\min}/(N_{\min}-1)^2$ decreases from the lowest to the highest eligible level.
- **H-A2′ (restated):** let $r_\times(N)$ be the reduced size at which the ROM's measured time per right-hand-side
  evaluation equals the FOM's at resolution $N$ (from the R4 cost model, cheaper of the two ROM forms). "Consistent"
  if $r_{\min}/r_\times(N_{\min})$ decreases from the lowest to the highest eligible level; the value $<1$ means the
  ROM's per-step work is below the FOM's.
- **H-A3 (descriptive categories):** *ROM faster throughout* ($S\ge1$ at every eligible level), *FOM faster
  throughout* ($S<1$ at every eligible level), *crossover toward the ROM* ($S<1$ at the lower and $S\ge1$ at the higher
  levels, one sign change), *reverse crossover* ($S\ge1$ at the lower, $S<1$ at the higher levels), *mixed*
  (more than one sign change). The least-squares slope of $\log S$ on $\log\bar s$ is reported with its sign, and,
  when it is positive and $S<1$, the $\bar s$ at which the fitted line reaches 1 is reported as **an extrapolation
  under this fit**, not as evidence. The K2 bar ($S\ge2$) is reported alongside.
- **H-B:** per $\tau$, *ROM ≥ 2× faster*, *ROM faster but < 2×*, or *FD faster*, against the faster of the FD
  families of §4.3. The claim is scoped to "these FD implementations"; it is not a statement about the best possible
  solver on the L-shape (A1.8).

### A1.4 Error metric and floors

- **Part A error is the continuous $L^2$ norm, computed exactly in coefficient space** (Parseval): FOM coefficients
  are zero-padded to $N_{\rm ref}$; ROM states are $\Psi z$ in the same space. $e=\max_{k=1..5}\lVert
  b(t_k)-b_{\rm ref}(t_k)\rVert/\lVert b_{\rm ref}(0)\rVert$. This removes the 257² lattice's aliasing (mode 256 is
  invisible there) from the scoring. The 257² lattice is used only as the output-synthesis *cost* contract.
- **Floors (A):** the span floor (orthogonal projection on $\Psi_r$) and the spectral floor (truncation to $N$) are
  the $L^2$-optimal errors in those spaces at each time, so for the max-over-times metric every run's error is
  $\ge$ its floor. D2 is kept as a software check in this norm.
- **Part B error** is the discrete $L^2$ norm on the lattice $x_i=i/64$ restricted to $\Omega_L$ (common to every
  ladder grid). Diagnostics: the same norm on the window $\{\rho<0.1\}$ around the re-entrant corner, and, for the
  matched configurations only, the norm on the $i/128$ lattice where the grid contains it. Part-B floors are computed
  by least squares with the *sampled* basis (lattice rows of $\Phi_r$, rank-revealing), so D2 holds in the scoring
  norm. Conclusions are restricted to the six output fields.

### A1.5 Query and timing contract (replaces §6 where different)

- **Query (primary, "field"):** case parameters $\to$ six output fields. Charged to both: initial condition from a
  shared, precomputed 1024-point composite Gauss–Legendre table (64 panels × 16; checked against `pscore`'s
  8192-point rule, gate G7); the $\nu$-dependent ETDRK4 coefficients (FOM: $(N-1)^2$ modes; ROM: $r$ modes);
  all steps; output synthesis to the 257² lattice (FOM: DST via FFT; ROM: precomputed lattice basis times $z$).
  Secondary contract "native": the same without output synthesis. F-IMEX (B) is charged **both** factorisations
  (IMEX Euler start and SBDF2).
- **Synchronisation:** every timed call ends with `block_until_ready` on all outputs (JAX) or returns materialised
  arrays (scipy).
- **Protocol:** one CPU core (`taskset`), single-thread flags as in §6, no other local job running. For each timing
  case, FOM and ROM calls are interleaved (F, R, F, R, …), 5 repetitions each after a warm-up; every repetition is
  stored. Sentinel (a fixed FOM configuration) at the start and end; drift $>10\%$ → the timing job is re-run once;
  a second drift marks it invalid. The cost-optimal FOM and ROM configurations at each level and $\tau$ are re-timed
  in a fresh process, and those times are the reported $t^*$.
- **Ladders (replace the CFL window):** for every size, step counts $n=5\cdot2^k$ are run in increasing order,
  from $n=5$, until the error is time-converged (relative change $<2\%$ between $n/2$ and $n$) or $n=2560$ (A,
  F-IMEX) / $n=40960$ (F-RK4). Blown-up runs are invalid. A: $N\in\{8,12,16,24,32,48,64,96,128,192,256,384,512\}$
  up to $\tfrac34N_{\rm ref}$; ROM $r\in\{4,6,8,12,16,24,32,48,64,96,128,192,256,384,512\}$ up to
  $r_{\max}$ (A1 256, A2 384, A3/A4 512). B: $L\in\{32,64,128,192,256,320,384,448,512\}$ (32 scored on the $i/32$
  sub-lattice only as a diagnostic). A size whose ladder top is reached without meeting $\tau$ gives a censored cell.
- **Two ROM forms, each scored and timed on its own:** ROM-T (packed tensor) and ROM-Q (grid-trapezoid rule of
  $(M_q-1)^2$ points, $M_q$ in a ladder, values and derivatives synthesised exactly; $6mr$ flops per evaluation). The
  ROM's cost-optimal configuration is the cheaper over both forms. **No shape-only timing:** every timed ROM uses
  its real operators.
- **Sine-ROM control:** the same Galerkin/ETDRK4 machinery on the first $r$ Laplacian-ordered sines (reported as a
  named arm, not as the ROM).

### A1.6 Controls (replaces §7)

*Deterministic software checks (must fire; a non-firing check is a code bug):*

| id | check | must |
|---|---|---|
| SW1 | matcher on real FOM fields with injected errors of known size: $b+\epsilon\,\lVert b_0\rVert\,\hat e$ with $\epsilon=2\tau$ and $\epsilon=\tau/2$ | reject $2\tau$, accept $\tau/2$ |
| SW2 | reference gate on a real reference perturbed by $5\times$ its threshold | reject |
| SW3 | timing on a call that runs two *different* cases and returns both outputs | 1.8–2.2× a single call |
| SW4 | tensor corruption: zero the slice $k^\star$ with the largest $\lvert T(z,z)_k\rvert$ along a real trajectory | rollout differs by $\ge10^{-6}$ |
| G4 | ROM-T vs the exact-evaluation ROM ($\Psi^\top$`pscore.nonlinear`$(\Psi z)$; B: direct FD evaluation), 2 cases, $r\in\{32,128\}$ | agree to $\le10^{-9}$ relative |
| G6 | training/test disjointness; seeds disjoint from campaign seeds | assert |
| G7 | 1024-point IC rule vs `pscore.ic_coefs_2d` (8192 points) | $\le10^{-12}$ relative |
| G9 | tensor at padding $M=3N_{\rm ref}/2$ vs $2N_{\rm ref}$, $r=64$ | report the rollout difference; must be $\le\tau_{\min}/10$ for targets to be eligible |
| D2 | every run's error $\ge$ its floor (A: continuous norm; B: sampled-basis floor) | assert |

*Physical diagnostics (reported, not asserted):* a coarse "reference" ($N_{\rm ref}/4$; B: $L=128$) against the
gate; $N=16$ and $L=64$ against $\tau=10^{-2}$; a random orthonormal span at $r\in\{32,128\}$; data sensitivity D1
(48 vs 96 training trajectories: a lower $r_{\min}$ with 96 is evidence of data sensitivity, not a quantified bound).

### A1.7 References (replaces §5's acceptance)

- **A:** continuous-norm gates on every test case: space $\lVert u_{N_{\rm ref}}-u_{3N_{\rm ref}/4}\rVert\le2\times10^{-5}$,
  time $\lVert u_{n}-u_{2n}\rVert\le2\times10^{-6}$ (relative to $\lVert u_0\rVert$, max over output times),
  $n=$ `ref_cfl_steps(N_ref, 2)`. A target is eligible only if $\tau\ge20\times$ the larger observed gate
  difference at that level. Four training trajectories per level are checked at $3N_{\rm ref}/4$ against
  $10^{-4}$. A failed gate makes the level unresolved (it may be re-run once at a larger $N_{\rm ref}$ by a dated
  amendment, before any ROM or FOM ladder is read).
- **B:** the reference is F-IMEX (SBDF2 + SuperLU) at $L=1024$, $n=2000$. Space: runs at $L\in\{256,512,1024\}$,
  same $n$; observed orders $p_{1}$ (256→512) and $p_{2}$ (512→1024) on the $i/64$ lattice and on the $i/256$
  lattice; gate $p_2\ge1$ and $\lvert p_1-p_2\rvert\le0.5$; reference error estimate
  $\hat e_{\rm ref}=\lVert u_{512}-u_{1024}\rVert/(2^{p_2}-1)$. Time: $n$ vs $2n$ at $L=512$ on 3 cases
  $\le10^{-1}\hat e_{\rm ref}$. Eligible targets $\tau\ge10\max\hat e_{\rm ref}$. The corner singularity
  $\rho^{2/3}\sin(2\theta/3)$ is why the order is measured, not assumed.
- **Feasibility rule (registered):** levels run in the order A1, A2, A3, A4; each stage is a job of at most 30 minutes.
  A3 runs only if A2's generation finished in under 15 minutes; A4 only if A3's finished in under 20 minutes and its
  snapshot set fits in 8 GB. Anything not run is "not run (time cap)", not a negative.

### A1.8 Competitors (Part B)

The two FD families stay (F-RK4 with the step ladder extended to $n=40960$, since RK4 diffusion stability needs
$n\gtrsim0.25\cdot8\nu L^2/2.785$; F-IMEX with SuperLU). Sparse direct solves in 2D at $n\le2\times10^5$ are a strong
practical choice, but multigrid, stabilised explicit (RKC) and high-order/spectral-element/multi-domain solvers are
**not** in the pilot; the H-B wording is restricted accordingly, and the full proposal must include them.

### A1.9 Compute and memory

Snapshots stream to disk per chunk; POD by the method of snapshots (Gram matrix accumulated in chunks). Memory caps:
A4 snapshot set 4.7 GB (48 trajectories), packed tensor at $r=512$ 0.54 GB, Part-B snapshot set 3.2 GB. Every
stage writes a JSON with its wall time; the report states which stages ran.

---

## Amendment A2 (2026-10-20, after Codex design audit 2, before any ladder is read)

Audit `audits/codex-design-2.md`: NOT-READY (items 2, 6, 7 CORRECT). Changes below govern over A1 and the original.
At this point only the A1 and A2 reference/training generation had run (gates printed; no FOM or ROM ladder run).

### A2.1 Interpretation

- Part A is a **paired viscosity sweep**, not an isolation of front steepness: lowering $\nu$ also changes diffusion
  and decay. Two diagnostics are reported per level from the references: the front sharpness
  $\max_x|\nabla u(t_k)|/\max_x|u(t_k)|$ (on the $N_{\rm ref}$ padded grid) and the remaining energy
  $\lVert u(t_5)\rVert/\lVert u_0\rVert$.
- If A4 runs (48 training trajectories), the trend A1–A4 is read with the **48-trajectory POD** at every level (a
  common training budget); the 96-trajectory trend covers A1–A3 and is reported separately.
- **H-A2′** concerns the *nonlinear-evaluation* cost only: $r_\times(N)$ is the $r$ at which the measured ROM
  nonlinear-evaluation time equals the FOM's at $N$, found by log–log interpolation between the two measured $r$
  values that bracket it; "unresolved" if no bracketing pair exists. The speed verdict (H-A3) uses measured whole-query
  times only.
- A's IC has a **negligible wall mismatch at pilot tolerances** ($\le3\times10^{-8}$), not exact compatibility.
- D3's R-function envelope behaves like $\rho$ at the re-entrant corner, while the leading corner solution can behave
  like $\rho^{2/3}$, so the quotient can be singular; D3 measures that polynomial-envelope space only, not
  neural-bank representability.

### A2.2 Step ladder: trajectory-distance rule (replaces the 2 % error rule)

For each size, $n=5\cdot2^k$ ascending; stop increasing $n$ once the trajectory distance
$\delta_n=\max_{\rm cases}\max_k\lVert u_n(t_k)-u_{n/2}(t_k)\rVert/\lVert u_0\rVert\le10^{-5}$ (both runs valid), or at
the ladder top. With a 4th-order integrator the remaining change for any larger $n$ is then about $\delta_n/15$; a
size's pass/fail at $\tau$ is **unresolved** if $|e_n-\tau|\le\delta_n$ for its last $n$. Minima are "smallest
observed feasible size"; a size stopped at the ladder top without meeting $\tau$ is censored for that $\tau$.

### A2.3 ROM-Q ladder and accounting

- ROM-Q grid sizes $M_q\in\{16,24,32,48,64,96,128,192,256,384,576,1152\}$ with $M_q<3N_{\rm ref}$, each run only
  where $6(M_q-1)^2r\le2r^3$ (where the grid form could beat the tensor by its flop count). Part B has no ROM-Q
  (deferred; a node-sampling operator would need its own definition).
- **Each selected POD prefix is diagonalised separately**; $\Psi_r=\Phi_rV_r$, and the tensor, the grid matrices,
  the initialiser and the decoder of size $r$ are all built from $\Psi_r$ (never by truncating a larger rotated basis).
- **Part A accounting:** IC coefficients from the shared 1024-point table (FOM at its $N$; ROM at $N_{\rm init}$),
  ETDRK4 coefficients for $(N-1)^2$ resp. $r$ modes, all steps, synthesis to the 257² lattice.
  **Part B accounting:** IC by nodal sampling (FOM: its grid; ROM: $z_0=\Phi^\top u_0$ on the $L_{\rm init}$
  sub-lattice of the $L=512$ grid, scaled by $(512/L_{\rm init})^2$), all steps (F-IMEX: both factorisations),
  output = the values on the $i/64$ lattice of $\Omega_L$ (FOM: subsampling; ROM: lattice rows of $\Phi$ times $z$).
- $N_{\rm init}$/$L_{\rm init}$ rule: the smallest value whose IC truncation (A: $\lVert b_0-{\rm trunc}(b_0)\rVert$;
  B: the $z_0$ difference from the full-grid projection, in the ROM norm) is $\le\tau/20$ on every test case; the
  matched configuration is then re-scored with it and must still meet $\tau$ (else the next value).
- Timing records every repetition, the sentinel values, the spread (max/min) and the count of repetitions more than
  20 % above the median. If the fresh finalist timing changes the ordering, the report gives the fresh times of the
  selected configurations and says so; it does not re-select.

### A2.4 Controls (replace A1.6's SW1, SW3, SW4)

- **SW1 (deterministic):** feed the matcher a "run" equal to the reference plus a perturbation of exact relative size
  $\epsilon$ at every scored time ($\epsilon=2\tau$ and $\tau/2$, same norm and denominator): it must reject and accept
  respectively.
- **SW3 → diagnostic H1:** a combined two-case call against the sum of the two single calls (ratio reported; 0.8–1.25
  expected).
- **SW4 (deterministic):** on a frozen reached state $z$, zeroing tensor slice $k^\star$ must change the right-hand
  side by exactly that slice's contribution (to round-off) and that contribution must be $\ge10^{-6}\lVert N(z)\rVert$;
  the rollout change is reported as a diagnostic.
- **G9** (reduced padding) is a diagnostic. The reference's own padding error is covered by the A space gate
  ($N_{\rm ref}$ vs $3N_{\rm ref}/4$ changes modes and padding together).
- **Training gates:** A: 4 trajectories at $3N_{\rm ref}/4$, $\le10^{-4}$; B: 4 training trajectories ($L=512$) against
  $L=1024$, difference reported. A failure labels the level's basis "unqualified" (the ROM is still run and
  labelled).

### A2.5 References and eligibility (replace A1.7's B rule and margins)

- **B space:** F-IMEX at $L\in\{128,256,512,1024\}$, same $n=2000$, compared on the common $i/64$ lattice (and the $i/128$
  lattice as a second readout). Two observed orders $p_1=\log_2(d_{128,256}/d_{256,512})$ and
  $p_2=\log_2(d_{256,512}/d_{512,1024})$ (max over cases and times of the normalised differences). Gate: $p_2\ge1$ and
  $|p_1-p_2|\le0.5$; $\hat e_{\rm ref}=d_{512,1024}/(2^{p_2}-1)$. If the gate fails, B targets are scored against the
  numerical $L=1024$ field and every B statement is labelled "conditional on an unqualified reference".
- **B time:** $n=2000$ vs $4000$ at $L=1024$ on the 4 test cases with the largest $s$, and at $L=512$ on all 12;
  the larger difference $\delta_t$ enters the allowance. Conclusions apply to that check's coverage.
- **B training:** SBDF2 + SuperLU at $L=512$, $n=2000$ (the reference integrator), snapshots every 0.0125.
- **Uncertainty allowance and three-way scoring.** $U_A=\delta_{\rm space}+\delta_{\rm time}$ (the two A gate
  differences); $U_B=\hat e_{\rm ref}+\delta_t$. A configuration *meets* $\tau$ if worst error $+U\le\tau$, *misses*
  if worst error $-U>\tau$, and is *unresolved* otherwise; unresolved cells are reported, never counted as passes.
  All errors are errors against the numerical references.

### A2.6 Feasibility (replaces A1.7's rule)

- Before each expensive stage (references, training, POD+tensor, each ladder, timing), a one-unit benchmark (one case,
  or one $r$) is timed and the stage's projected time and peak memory (arrays held at once) are written to the stage
  JSON. A stage whose projection exceeds 30 minutes is split into checkpointable units (per $r$, per $N$, per case
  chunk; each unit writes its results before the next starts) or not run.
- A timeout stops the stage; completed units are kept and the remainder is reported "not run (time cap)".
- Total local budget for the pilot: **8 hours of wall time**; A4 runs only if A3's stages all finished and its
  48-trajectory snapshot set (4.7 GB) plus a peak of 3× that in POD workspace fits in the 36 GB `jaxrun` ceiling.
- Stopping after A1–A2 (or A1–A3) is acceptable; "too few levels" is not a negative finding.

---

## Amendment A3 (2026-10-20, after Codex design audit 3; final pre-ladder amendment)

Audit `audits/codex-design-3.md`: NOT-READY (items 1, 2, 7, 10–12, 14 CORRECT or CORRECT with qualifications).
State at this amendment: A1–A3 references and training generated; an A1 ROM ladder ran as a code smoke (its output
was used to debug, not read for any verdict). This amendment governs over A2, A1 and the original.

### A3.1 Scoring and the step search

- **Every valid observed configuration** (size, step count, and for the ROM the form and $M_q$) is scored by the
  three-way rule of A2.5. A size is *observed feasible* at $\tau$ if any of its observed configurations *meets*
  $\tau$. The trajectory-distance rule of A2.2 is a **search-stopping heuristic only**: it adds no uncertainty to an
  observed configuration, step counts above the stopping point are reported as "not searched", and no order-based
  remaining-error estimate is used in any verdict. At the first rung, or after an invalid run, no distance exists and
  the search continues.
- $r_\times(N)$ (H-A2′): the first crossing from below in increasing $r$ (the smallest measured $r$ whose ROM
  nonlinear-evaluation time exceeds the FOM's at $N$, interpolated log–log with the preceding $r$); if the measured
  curve crosses more than once, all crossings are reported and the verdict is "unresolved".
- ROM-Q pruning ($6(M_q-1)^2r\le 2r^3$) is a **workload heuristic**: the packed tensor costs about $r^3$ flops
  ($r\cdot r(r+1)/2$ multiply–adds) and the grid form $6mr$; configurations it excludes are "not explored", and
  winners are "cheapest among evaluated configurations".

### A3.2 Part B accounting (corrects A2.3)

The B POD basis $\Phi$ is orthonormal in the Euclidean inner product of the $L=512$ interior nodes. For each $r$,
$\Psi_r=\Phi_rV_r$ (rotated, as in A); the initialiser is $z_0=(512/L_{\rm init})^2\,\Psi_{r,\rm sub}^\top u_{0,\rm sub}$
(the sub-lattice sum approximating the full-node sum; exact for $L_{\rm init}=512$) and the decoder is
$\Psi_{r,\rm lat}z$ on the $i/64$ lattice. The ROM is charged its $\nu$-dependent ETDRK4 coefficients ($r$ modes). IC
discrepancies for the $N_{\rm init}$/$L_{\rm init}$ rule are normalised by $\lVert u_0\rVert$ exactly as the scoring
metric; the chosen initialiser is re-scored with the full $e\pm U$ rule and is the one priced in the finalist timing.

### A3.3 Controls (replace A2.4's SW1 and qualify SW4)

- **SW1a (deterministic):** with $U=0$, a run equal to the reference plus a perturbation of exact relative size
  $\epsilon\in\{10^{-3},10^{-2}\}$ at every scored time must be scored as $\epsilon$ (relative agreement $10^{-10}$).
- **SW1b (deterministic):** the three-way matcher on fixtures $(e,U,\tau)$: $(0.5,0.1,1)$ meets; $(1.5,0.1,1)$
  misses; $(0.95,0.1,1)$ unresolved; $(0.9,0.1,1)$ meets (equality $e+U=\tau$); $(1.1,0.1,1)$ unresolved (equality
  $e-U=\tau$ is not a miss).
- **SW4** needs a finite, non-zero $N(z)$ on a reached state, with the largest-component slice; if no such state
  exists the fixture is "unavailable" (not a solver bug). The rollout change stays diagnostic.

### A3.4 References and the allowance (replace A2.5's fallback and A2.4's padding statement)

- **A padding gate (new):** at $N_{\rm ref}$, the FOM with padded grid $M=2N_{\rm ref}$ against the default
  $M=3N_{\rm ref}/2$ on the two test cases with the largest $s$ (same $n$); the difference $\delta_{\rm pad}$ joins the
  allowance, $U_A=\delta_{\rm space}+\delta_{\rm time}+\delta_{\rm pad}$, each the max over cases and times of the
  normalised difference. If the gate exceeds $2\times10^{-5}$ the level is unresolved. Its coverage (two cases) is
  stated next to $U_A$.
- **B:** $U_B=\hat e_{\rm ref}+\delta_t$ is used only if the order gate on the $i/64$ lattice passes and the three
  differences are finite and strictly decreasing (the $i/128$ orders are diagnostic). Otherwise $U_B$ is
  **unavailable**: B is scored two-way against the numerical $L=1024$ field ($U=\delta_t$) and every B verdict is
  labelled "against the numerical reference; physical accuracy unqualified". The twelve-case $\delta_t$ is
  labelled an extrapolation from the fine-grid check on 4 cases and the $L=512$ check on 12.
- B training comparison (4 trajectories, $L=512$ vs $L=1024$) is diagnostic; the B basis qualification is
  "unestablished" in the report whatever its value.
- **$U$ is a conditional allowance, not an error bound.** It is one scalar per level (or per part), the sum of the
  gate maxima above.

### A3.5 Memory and budget

Every unit must fit the 36 GB `jaxrun` ceiling (CPU jobs also run under `jaxrun`) including temporaries and
factorisation storage, by a projection written before it runs; a unit that does not fit is split or skipped and
reported. The 8-hour wall budget includes benchmarks, retries and the work done before this amendment; unit timeouts
(30 min) and the total budget both apply.

---

## Amendment A4 (2026-10-20, after Codex design audit 4: OVERALL READY, no blocking item)

Non-blocking restatements adopted from `audits/codex-design-4.md`:
- An exact measured equality counts as a crossing for $r_\times$.
- Each ROM candidate's initialiser ($N_{\rm init}$/$L_{\rm init}$) is selected and the candidate re-scored **before**
  finalist selection, and that initialiser is what is timed; if this order is broken the report says so.
- B fallback: on order-gate failure, raw errors against the numerical $L=1024$ field are reported and the three-way
  rule is applied with $U=\delta_t$ for that numerical-reference comparison only; physical-target qualification is
  unavailable.
- A missing or non-finite gate makes $U$ unavailable (never an implicit zero); the level or part is then unresolved.
- D2 is asserted with a documented tolerance (relative $10^{-9}$).
- Provenance: this is an **amended exploratory design**: the reference/training generation (A1–A3) and an A1 ROM
  ladder used for debugging ran before A3 was final; the report keeps that chronology and reads no verdict from the
  debugging run.
