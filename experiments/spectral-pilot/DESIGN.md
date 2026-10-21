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
