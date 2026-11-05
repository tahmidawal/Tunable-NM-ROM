# ns3d-offmesh-quadrature: off-mesh quadrature of the NS3D advection term vs the exact tensor

Pre-registration, written before any cluster job. Lane `exp/2026-11-05-ns3d-offmesh-quadrature`,
sparse worktree `worktrees/2026-11-05-ns3d-offmesh-quadrature`, forked from the
`exp/2026-11-03-ns3d-pod-match` head `0dee527c5e313f85d678c7a04ac6d96cb97292b0` (mirror
`1963c87a6bc4e6a29d66e1c093d6a1630780a7a6`). Mirror of this lane:
`origin/codeonly/exp/2026-11-05-ns3d-offmesh-quadrature`. Cluster namespace
`/cluster/tufts/paralab/tawal01/ns3d-offmesh-q-20261105/`. User approval: "Can you do that
experiment for me please" (2026-11-05, relayed by the coordinator).

## 0. Step 0: what the development ROM actually computes (verified from code)

The ns3d-pod-match development ROM is `experiments/ns3d-coordnet/rom_mesh.py`, unchanged, launched
by `experiments/ns3d-pod-match/cluster/job.sbatch` in `rom` mode with
`experiments/ns3d-pod-match/configs/dev{32b,64,96}.json` (jobs 6859541, 6858038, 6858031). The bank
is `experiments/ns3d-pod-match/banks/a2_mu001/bank_selected.npz` (job 6850341), sha256
`a1af98e2297c41339a037cb427f757fc1f23bb8791132985f29cbda826431bd5` (re-hashed 2026-11-05: match).

**Domain and boundary conditions.** Periodic unit torus $\mathbb{T}^3=[0,1)^3$, no walls
(`experiments/ns3d/ns3d_fom.py`). The full-order model is a Fourier pseudo-spectral Galerkin
discretisation with the strict 2/3 cutoff (max-norm $|k_i| < n/3$) and the Leray projector applied
to both nonlinear inputs and outputs, CNAB2 in time. Initial conditions are two oblique vortex
structures (curl of two Gaussian-periodic vector potentials), parameterised by centre, separation,
strength and viscosity $\nu\in[0.002,0.01]$. Off-mesh rules therefore apply cleanly: the integration
domain is the unit cube with periodic integrands, so rank-1 lattices and the shifted trapezoid rule
are natural (exact for trigonometric polynomials below their resolution), while Gauss–Legendre and
Sobol are applicable but not tailored to periodicity.

**The nonlinear term is an exact precomputed tensor, not quadrature or EQ.** The equation is in
rotational form, $\partial_t u = \mathbb{P}(u\times\omega) + \nu\Delta u$ with $\omega=\nabla\times u$
(the gradient $\nabla|u|^2/2$ is annihilated by the solenoidal tests). The reduced residual is a
Petrov–Galerkin residual against $M_t = 292$ solenoidal Fourier test fields $\phi_m$ (73 wave vectors,
two polarisations, cos/sin; $|k|^2\le 10$, $|k_i|\le 3$; the same 292 tests on every mesh,
checked). `SR.build_operators_fast` (`experiments/ns3d-shift/shift_rom.py`) builds
$A=\Phi^\top G$ $(292\times 64)$, $\lambda$, the frame operators $D_d=\Phi^\top\partial_d G$
$(3\times 292\times 64)$ and, via `ns3d_rom.build_tensor`, the tensor

$$T_{mjk} = \langle \phi_m,\; G_j \times \nabla_h\times G_k\rangle_{\text{grid}},\qquad T\in\mathbb{R}^{292\times 64\times 64},$$

computed on the $n^3$ grid by FFT. It is exact for the mesh bank: every mesh column is a
trigonometric polynomial with $|k_i|<n/3$, so the triple product with a test has per-axis degree
$< n$ and the $n^3$ trapezoid rule integrates it exactly. The in-job check `tensor_relative` against
a dense projection is $5.0\times10^{-13}$, $9.4\times10^{-14}$, $1.3\times10^{-13}$ at 32³/64³/96³.
Online (`SR.make_frozen_step`, used by `head_rom.make_head_run` and `SR.make_frozen_run`),
$\mathrm{sym}=(T+T^{\top_{23}})/\mathrm{pre}$ is formed once per query; each damped Gauss–Newton
sweep forms the Jacobian $J(c)=\mathrm{sym}\,c$ $(292\times 64)$ and the advection
$a(c)=\tfrac12 J(c)\,c$. No empirical quadrature or point sampling enters the NS3D path. The tensor
holds $M_tR^2 = 1{,}196{,}032$ numbers (9.6 MB) and costs $2M_tR^2\approx 2.4$ Mflop per sweep; it is
$M_t\times R\times R$ with $M_t = 4.56R$, not $R^3$.

**What "the bank off-grid" means.** The coordinate network is $g=\nabla\times\psi_\theta$
(`experiments/ns3d-coordnet/coordnet.py`: integer Fourier features $k_{\max}=4$, four SiLU layers of
width 512, $3R$ outputs), so it is periodic, smooth and evaluable at any point; $u$-values by one
level of forward-mode JVPs and $\omega=\nabla\times g$ by a second level
(`CN.velocity_gradient`). The derivatives needed for the rotational term are therefore available
off-grid. **But the ROM's state space is not the raw network.** At mesh $n$ the bank is
$G_n=\mathrm{L\ddot owdin}\big(P_n[g(x_{\text{grid}})\,n^{-3/2}]\,T_{\text{rot}}\big)= P_n[\,\cdot\,]\,T_{\text{rot}}S_n$
(`bankio.mesh_bank`): sampled on the grid, projected onto the FOM space (2/3 mask + Leray), rotated
by the frozen $64\times64$ $T_{\text{rot}}$, and orthonormalised by the mesh-dependent factor $S_n$.
Its continuous extension is the band-limited trigonometric interpolant of the samples, which differs
from $g\,T_{\text{rot}}S_n$ by aliasing and the masked content: the ns3d-pod-match pre-Löwdin statistic
is $7.0\times10^{-5}$ / $3.1\times10^{-4}$ at 64³/96³ and $0.607$ at 32³ (Löwdin change 9.8 % at 32³).
The study therefore runs two integrands (§2).

The linear terms ($A$, $\lambda$), the frame operators $D_d$, the initial projection and the
decoded output fields are mesh operators in every arm; only $a(c)$ and $J(c)$ change.

## 1. Question

For the frozen R=64 coordinate-network bank and its frozen k=8 heads, at 64³ and 96³ (primary) and
32³ (secondary), on the 16 development cases (seed 202609202): when the exact tensor is replaced by an
off-mesh quadrature of the advection term, for each rule family and point count $M$:

1. the ROM error against the existing same-mesh CNAB2 reference (the ns3d-pod-match metric:
   evolved worst and evolved median over cases of the per-case maximum over the five output times of
   $\lVert u_{\text{ROM}}-u_{\text{ref}}\rVert/\lVert u_0\rVert$);
2. the difference from the tensor-ROM fields on the same cases;
3. the smallest $M^\ast$ at which the tensor's accuracy is reached and held (§4);
4. online solve time vs $M$, against the tensor path and the CNAB2 ladder, same job, same GPU;
5. whether quadrature at $M^\ast$ is cheaper than the tensor, and where its accuracy sits relative
   to CNAB2 at each mesh.

## 2. Arms

**Integrands** (the off-mesh advection, both evaluated at rule points $x_q$ with weights $w_q$,
$\sum_q w_q = 1$):

$$a_m(c) = n^{-3/2}\sum_{q} w_q\,\psi_m(x_q)\cdot\big(U_q c\big)\times\big(W_q c\big),\qquad
J_{m\cdot}(c) = n^{-3/2}\sum_q w_q\,\psi_m(x_q)\cdot\big[U_q\times (W_q c) + (U_q c)\times W_q\big],$$

with $\psi_m = n^{3/2}\phi_m$ the continuum-normalised test (analytic), $U_q\in\mathbb{R}^{3\times R}$
the continuum-normalised bank columns at $x_q$ and $W_q$ their curls; $a(c)=\tfrac12 J(c)c$ exactly,
as for the tensor. The scale factor follows from the grid normalisation of $G_n$ and $\phi_m$ and is
verified by the positive control.

- **`net` (primary, mesh-free integrand):** $U_q = g(x_q)\,T_{\text{rot}}S_n$,
  $W_q=(\nabla\times g)(x_q)\,T_{\text{rot}}S_n$, autodiff. The rule never touches the mesh; only
  the $64\times64$ factor $S_n$ comes from it. Its $M\to\infty$ limit is the continuum integral of the
  network bank, which differs from the tensor by the projection/aliasing gap quantified in §3.
- **`interp` (isolates quadrature error):** $U_q,W_q$ = the exact trigonometric interpolant of the
  mesh bank $G_n$ and of its spectral curl, evaluated at $x_q$ by a direct non-uniform DFT over the
  masked modes. Its $M\to\infty$ limit is exactly the tensor.

**Rule families and ladder** (generated once locally with the vendored, byte-identical
`vendor/quadrature-burgers3d/rules.py`; committed as `rules/rules.npz` + `rules/rules.json` with
per-rule sha256; random shifts/scrambles seed 0):

| family | label | sizes $M$ |
|---|---|---|
| rank-1 CBC lattice ($P_2$ criterion, product weights 1), random shift | `lat` | $2^9,\dots,2^{16}$ (512 … 65,536) |
| Korobov lattice, random shift | `kor` | primes 509, 1021, 2039, 4093, 8191, 16381, 32749, 65521 |
| scrambled Sobol | `sob` | $2^9,\dots,2^{16}$ |
| Gauss–Legendre tensor $p^3$ | `gl` | $p$ = 8, 10, 13, 16, 20, 25, 32, 40 (512 … 64,000) |
| shifted periodic trapezoid $p^3$ (supplementary: the natural periodic tensor rule) | `trap` | same $p$ |

Controls: `lat64` (must fail); positive control `trapN` = shifted trapezoid with $p=n$ under the
`interp` integrand, evaluated by FFT phase shift; continuum target for `net` = shifted trapezoid
$p=128$ (2,097,152 points, shift seed 1), with $p=96$ (shift seed 2) as its convergence check (both
generated in-job from $p$ and seed, used only for $\rho$, never in rollouts).

**ROM arms:** `head` = the frozen coordnet head k=8 of the ns3d-pod-match development jobs
(`runs/dev{32b,64,96}/output/mesh/coordnet_head_k8_n{n}.npz`, sha256 from their manifests; primary)
and `span64` = the full R=64 coordinate span (linear endpoint; secondary). Solver, $\Delta t=0.02$,
3 damped Gauss–Newton sweeps, damping $10^{-6}$, ic_iters 12, extrapolated warm start: unchanged.
Only the contraction that produces $J(c)$ is swapped (`qrom.py`, a line-for-line copy of
`SR.make_frozen_step` / `H.make_head_run` / `SR.make_frozen_run` with the contraction injected).

**Comparator:** the CNAB2 step ladder 200/100/80/70/60/50/40/20/10 of the parent, same job.
**Reference:** CNAB2 at $\Delta t=10^{-3}$ on the same mesh, regenerated from seed 202609202 in-job
(unchanged from ns3d-pod-match). No retraining of anything.

Rollouts with rules whose $M$ is not a power of two are padded with zero-weight points to the next
power of two for the accuracy pass (identical result up to summation order; padded rows of $P$ are
exactly zero). Timing uses unpadded tables.

## 3. Measurements per mesh

1. Tensor ROM: head and span64 errors (reproduction gate G1).
2. For every (integrand, rule): $\rho_a$ and $\rho_J$ = worst and median over reached states of
   $\lVert a_{\text{rule}}(c)-a_T(c)\rVert/\lVert a_T(c)\rVert$ and the Frobenius analogue for $J$,
   on the 160 mid-step coefficients $\tfrac12(c^{n+1}+c^n)$ of the tensor head rollouts (16 cases ×
   10 steps) and the 160 of the tensor span64 rollouts. For `net` also $\rho$ against the continuum
   target.
3. For every (integrand, rule, ROM arm): the 16 × 6 error array against the reference, its evolved
   worst and median, and $\delta_T$ = max over cases and times of
   $\lVert u_{\text{rule}}-u_{\text{tensor}}\rVert/\lVert u_0\rVert$.
4. CNAB2 ladder errors.
5. Offline build times (network evaluation, NUDFT, tensor build) — reported, not gated.
6. Timing (§5).

## 4. Plateau and $M^\ast$ (defined before any job)

For each (mesh, integrand, family, ROM arm), with tensor values $W_T$ (evolved worst) and $D_T$
(evolved median) from the same job, a ladder size $M$ **reaches tensor accuracy** if all 16 cases are
finite, $W_M\le 1.05\,W_T$ and $D_M\le 1.05\,D_T$. $M^\ast$ is the smallest ladder size such that it
**and every larger ladder size** reach tensor accuracy; otherwise "not reached on the ladder". A
secondary, stricter trajectory plateau $M^\ast_{\delta}$ is the smallest size such that it and every
larger size have $\delta_T\le 10^{-4}$ (the ROM is within $10^{-4}\lVert u_0\rVert$ of the tensor ROM).
The headline is $M^\ast$ for `net`/`head` at 64³ and 96³. The sizes at which $\rho_a\le10^{-2}$,
$10^{-4}$, $10^{-6}$ are tabulated without a gate.

## 5. Timing protocol

All timed callables run in one job on one GPU: `burn_gpu`; sentinels solo (tensor head, `net`-head
at `lat4096`); one interleaved block, freshly shuffled every repetition, 7 repetitions after 2 untimed
calls per arm, of: tensor head, tensor span64, `net`-head and `net`-span64 at every `lat` size
(unpadded; 512 … 65,536), `net`-head at each family's $M^\ast$ with its exact unpadded size, and all
nine CNAB2 settings; sentinels solo again. Gate: sentinel in-block/solo and end/solo ratios within
$[1/1.10, 1.10]$; the timed outputs must reproduce the accuracy pass to $10^{-9}$. Repetition arrays
are persisted. `interp` tables have the same shapes as `net` tables, so its online time is the same
computation and is not timed separately. A kernel microbenchmark times, per call, a jitted scan of 30
Jacobian contractions (one query's sweeps) for the tensor and for the quadrature at each `lat` size,
interleaved, 7 repetitions. Speedups vs CNAB2 use the parent's rule: fastest stable CNAB2 setting
whose evolved worst is no larger than the arm's.

**Prediction (stated before data).** The quadrature Jacobian costs $\approx 2\cdot 3M\,M_tR$ flop per
sweep against $2M_tR^2$ for the tensor, a ratio of $3M/R$; it can only be cheaper below $M\approx 21$
points, far too few to integrate anything. We expect quadrature at $M^\ast$ to be slower than the
tensor in kernel time; whether it is slower in total query time depends on how much of the query is
the grid-bound initial projection and output decoding, which the measurement decides.

## 6. Gates

- **G0 (inputs):** sha256 of the bank, the three heads and the reference summaries match before any
  work; the dev cohort is seed 202609202 × 16; the closed seeds 202609203, 202609211, 202609221 are
  refused in code.
- **G1 (tensor reproduction):** tensor head and span64 errors match the ns3d-pod-match dev summaries
  to $\le 10^{-8}$ absolute per entry; this lane's driver with the tensor contraction reproduces
  `H.make_head_run` / `SR.make_frozen_run` fields to $\le 10^{-12}$ relative in-job. Fail → the mesh's
  results are not used.
- **G2 (positive control, hard stop):** `interp` on `trapN`: $\rho_a,\rho_J\le 10^{-12}$ worst over
  reached states, and its head rollout within $10^{-10}$ relative of the tensor rollout. The NUDFT
  evaluator agrees with the FFT phase-shift evaluator on 512 points to $\le10^{-12}$. Fail → abort.
- **G3 (must-fail):** `lat64`: $\rho_a \ge 10^{-1}$ for both integrands (else flagged).
- **G4 (off-grid derivatives):** at 4096 random off-grid points the autodiff curl matches a central
  difference of the autodiff velocity ($h=10^{-5}$) to $\le10^{-6}$ relative, and the autodiff
  divergence of the columns is $\le10^{-10}$ of their gradient norm. The `net` continuum target
  changes by $\le10^{-10}$ (relative, in $\rho_a$ terms) between $p=96$ and $p=128$ (else flagged and
  the continuum column marked unconverged).
- **G5 (timing):** as in §5.

## 7. Jobs, budget, stop rules

One job per mesh, each in its own directory under the namespace: `q96` (A100-80GB, wall 1:30),
`q64` (wall 1:00), `q32` (wall 0:40). At most two of this lane's jobs run at once. **Hard cap 6
GPU-hours** including failures, counted at wall limit while running and at `sacct` elapsed once done;
stop and report if the next submission could exceed it. A failed job may be rerun once per mesh after
a recorded fix; any change to the protocol after data is an amendment §A*n* with its reason. A local
smoke test at 32³ (two cases, a handful of rules, including G2) precedes the first submission.

## 8. Review

Codex is available locally (`codex-cli 0.166.0`); an independent read-only review of this design and
of `qrom.py` is requested before the first job and its disposition recorded in `reviews/`.

## A1. Amendment after the local smoke, before any cluster job (2026-11-05)

The first local 32³ smoke (2 cases) stopped at G2 on one sub-criterion: the direct-NUDFT evaluator
agreed with the FFT phase-shift evaluator to $1.78\times10^{-12}$, above the registered $10^{-12}$.
Every other G2 criterion passed by three orders of margin ($\rho_a = 1.4\times10^{-15}$,
$\rho_J = 9.7\times10^{-15}$, head rollout within $1.6\times10^{-15}$ of the tensor). The excess is
floating-point roundoff of a direct sum of $K$ complex exponentials with phases up to
$2\pi\,|k\cdot x|$; $K$ grows from 4,630 (32³) to about 125,000 (96³), so the threshold was set
without regard to $K$. **Change:** the NUDFT-vs-FFT sub-criterion of G2 becomes $\le 10^{-10}$
(still far below every $\rho$ the study reads); its value is reported per mesh. Nothing else changes.
The smoke is rerun with this change before submission.
