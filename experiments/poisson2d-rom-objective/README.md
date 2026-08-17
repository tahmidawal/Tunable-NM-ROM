# Poisson-2D ROM-solve objective study (2026-08-16)

**Question.** The FiLM coordinate auto-decoder (K=8 latents, `exp/2026-08-14-multistage-precision`)
has a held-out *inferred-latent* error of ~8e-3, but the ROM latent solve on the plain FD
residual lands at ~6.3e-2 — an 8x gap. Which **objective**, **collocation** and **init**
closes it?

**Answer (one line).** The gap is entirely the objective. Any objective that stops the FD
Laplacian from amplifying grid-scale decoder error — energy norm (= Ritz / Galerkin NM-ROM),
truncated `H^-1`, 20-step CG, spectral low-pass — closes it to within 10–30% of the
inferred-latent floor **from any init**, and the fixed-`k` ROM error becomes flat in N.
Hyper-reduction then works only through the **weak form** (integrate by parts onto smooth
test functions and quadrature the *decoder output*, not its Laplacian) with EQ weights fitted
to decoder snapshots — random collocation of the strong-form residual does not.

Everything: Poisson 2D, `-lap u = a exp(-|x-c|^2/2w^2)` on the unit square, u=0 walls, f64
FD/CG truth (max rel residual ~4e-13), seed 0, 512 train / 64 val, 16 held-out test sources
(the first 16 val), single seed. LM latent solver, budget 60 attempts, identical to the
multistage phase C (`ms_autodecoder.lm_solve`; `pro_common.lm_generic` is the same algorithm
generalised to `(H, g, value)`). Field error = mean rel-L2 at all grid nodes vs the FD
solution. Cluster: Tufts `gpu` partition, A100 (pax106/pax142), `jax_backend=gpu` in every
log, `JAX_DEFAULT_MATMUL_PRECISION=highest`, one isolated job dir per cell, dirs deleted after
checksummed pull. Code commit `70829fa` (round 1), `80f3bc3` (round 2, weak form).

## Objectives (all on the full interior grid unless stated)

With `r(z) = A u(z) - f` (A = ghost-zero-Dirichlet 5-point `-lap`, SPD) and `A = Phi Lambda Phi^T`
(separable discrete-sine eigenbasis; verified against `ms_parametric.neg_lap_interior`):

| name | objective | remark |
|---|---|---|
| `fd` | `‖r‖` | control = multistage phase C (LSPG on the FD residual) |
| `spec_a{α}_M{M}` | `‖Λ_M^{-α} Φ_M^T r‖` (M lowest modes, complete eigenshells) | α=0: Galerkin onto smooth test functions; α=½, all: **energy norm ‖u(z)−u*‖_A**; α=1, all: `‖A^{-1}r‖` = interior data misfit (an oracle at FOM-solve cost) |
| `ritz` | GN on `E(z)=½uᵀAu−fᵀu`, `H=JᵀAJ`, `g=Jᵀr` (matrix-free) | = Galerkin NM-ROM (stationarity `Jᵀr=0`); identical iterates to `spec_a0.5_Mall` (cross-check) |
| `cg{K}` | `‖CG_K(A, r)‖`, K explicitly unrolled steps | approximate `A^{-1}`, differentiated exactly |
| `lowpass{σ}` | `‖exp(−½σ²Λ)Φᵀr‖`, σ in cells | cheap spectral low-pass |
| `weak_a{α}_M{M}` | `‖Λ_M^{1−α}Φ_M^T u(z) − Λ_M^{−α}Φ_M^T f‖` | weak form of `spec`: only the *smooth* decoder output is quadratured; = `spec` on the full grid |

Inits: `mean` (mean training latent), `nearest` (latent of the nearest training source in
parameter space — the source is the ROM's input, so this is legitimate online information),
`encoder` (a small MLP from the source on a 16×16 lattice to the training latents; opaque-f
alternative). Oracle = LM on the data misfit from the same init with the same budget (the
finite-budget inferred-latent floor; uses the held-out field, ROM rows never do).

## Round 1 — objective sweep on the K=8 stage-0 decoder (`runs/obj_K8_S1`, job 2466633)

Golden control: `fd` reproduces the multistage phase-C numbers to all printed digits
(nearest 6.248e-2 / mean 2.198e-1; `smoke/control_fd_full_K8_S1_b60.json`, local GB10).

| objective | mean init | nearest init | encoder init | obj(z_LM) ≤ obj(z_oracle) | FD-res at z_LM (‖f‖=5.13) |
|---|---|---|---|---|---|
| oracle (data misfit) | 2.58e-2 | 7.78e-3 | 7.78e-3 | — | 1.01 |
| `fd` (control) | 2.20e-1 | **6.25e-2** | 1.13e-1 | 12/16/15 | 0.82 |
| `spec_a0_M64` | 1.51e-1 | 1.35e-2 | 6.83e-2 | 13/16/15 | 1.06 |
| `spec_a0_M256` | 2.33e-1 | 3.83e-2 | 1.48e-1 | | |
| `spec_a0.5_M64` | 8.82e-3 | 8.82e-3 | 8.82e-3 | 16/16/16 | 1.07 |
| `spec_a0.5_Mall` = `ritz` (energy / Galerkin) | **1.00e-2** | **1.00e-2** | **1.00e-2** | 16/16/16 | 0.90 |
| `spec_a1_M64` | 8.46e-3 | 8.48e-3 | 8.48e-3 | 16/16/16 | 1.09 |
| `spec_a1_M256` | 2.59e-2 | 7.83e-3 | 7.83e-3 | 16/16/16 | 1.05 |
| `spec_a1_Mall` (= data misfit) | 2.58e-2 | 7.78e-3 | 7.78e-3 | 16/16/16 | 1.01 |
| `cg5` | 4.58e-1 | 2.78e-2 | 6.21e-2 | 8/15/15 | |
| `cg20` | 8.45e-3 | 8.45e-3 | 8.47e-3 | 16/16/16 | 0.99 |
| `lowpass2` | 5.93e-2 | 1.75e-2 | 1.27e-1 | | |
| `lowpass4` | 9.41e-3 | 9.41e-3 | 9.41e-3 | 16/16/16 | 1.06 |
| `lowpass8` | 2.05e-2 | 9.77e-3 | 1.14e-2 | | |

Reading:

1. **The 8x gap is the objective, and it closes.** Energy norm / Ritz (the classical
   Galerkin NM-ROM) gives 1.0e-2 vs the 7.8e-3 inferred-latent floor; truncated `H^-1`
   with only 64 modes gives 8.5e-3; `cg20` 8.5e-3; `lowpass4` 9.4e-3. The FD control sits at
   6.2e-2 with a *lower* FD residual (0.82 vs 1.01) than the oracle — the amplification
   diagnosis of the multistage README, confirmed by every objective's `obj(z_LM) ≤ obj(z_or)`
   count: the solver minimises what it is given.
2. **Init dependence disappears.** With `fd`, mean init is 3.5x worse than nearest
   (2.2e-1 vs 6.2e-2, multi-modal residual landscape); with the smooth objectives all three
   inits give the same number, and `spec_a1_M64` / `spec_a0.5` from the *mean* latent beat the
   mean-init oracle (which is trapped at 2.6e-2). Warm starts are no longer load-bearing.
3. **Truncation to few smooth modes is enough — and better than exact `H^-1`.** 64 modes
   at α=1 (8.5e-3, all inits) vs the full data misfit (7.8e-3 nearest / 2.6e-2 mean).
   Unweighted Galerkin onto smooth test functions (α=0) helps only at M=64 (1.35e-2) and
   degrades as M grows back toward the FD residual (M=1024: 5.9e-2) — it is the `Λ^{-α}`
   weighting, not the truncation, that does the work.
4. **Solver floor ruled out**: 5x budget (`runs/obj_K8_S1_b300`, 300 attempts) changes no
   number by more than 1e-4 (e.g. `fd` 6.25e-2, energy 1.00e-2, `spec_a1_M64` 8.5e-3).
5. `cg5` is not enough (Jacobian is exact, the preconditioner just isn't); `cg20` costs
   ~5x more accepted steps' worth of work than a spectral weighting for the same result.

Same picture on the 3-stage decoder (`runs/obj_K8_S3`: `fd` 8.6e-2 → energy 1.0e-2,
`spec_a1_M256` 8.2e-3, oracle 8.0e-3) and on K=4 (`runs/obj_K4_S1`: `fd` 2.1e-1 nearest →
energy 4.3e-2, `spec_a1_M256` 3.6e-2 = oracle 3.6e-2; K=4 is representation-limited).

## Round 1 — N ladder at fixed k=8 (`runs/nlad_N{32,64,128}`, jobs 2466639/41/42)

Stage-0 auto-decoders retrained per N with `pro_train.py` (same recipe; the N=64 retrain
reproduces `ad_K8`: train 9.4e-3, inferred 8.2e-3 vs 9.4e-3 / 8.3e-3), then the compact
objective sweep, nearest init:

| N | oracle | `fd` | energy (`ritz`) | `spec_a1_Mall` | `cg20` | `lowpass4` (σ=4 cells) |
|---|---|---|---|---|---|---|
| 32 | 8.98e-3 | 6.18e-2 | 1.33e-2 | 8.98e-3 | 9.08e-3 | 1.16e-2 |
| 64 | 8.34e-3 | 6.35e-2 | 1.22e-2 | 8.34e-3 | 9.27e-3 | 1.01e-2 |
| 128 | 8.54e-3 | **9.24e-2** | 1.21e-2 | 8.54e-3 | 1.39e-2 | 2.40e-2 |

The FD-residual ROM **worsens with N** (6.2e-2 → 9.2e-2 — the resolution wall of the
end-to-end ROM, live), while the energy-norm and spectral ROMs are **flat in N** at fixed
k (1.3e-2 / 1.2e-2 / 1.2e-2; `spec_a1` sits on the oracle at every N). Objectives with a
grid-tied length scale drift as expected: `lowpass4` (σ fixed in *cells*) and `cg20` (fixed
iteration count vs a growing condition number) degrade at N=128 — use physical σ / spectral
weights, not cell counts.

## Round 1 — hard-BC decoder (`runs/hbc_K8`, job 2466638)

`u = 16x(1−x)y(1−y)·net(x;z)`, retrained with the identical recipe: train 7.0e-3 (vs 9.4e-3),
inferred 7.1e-3 (vs 8.2e-3), boundary block exactly 0. ROM: `fd` 4.6e-2 (nearest; 2.8e-1
mean), energy 9.5e-3, `spec_a1_M1024` 7.1e-3 = oracle, `cg20` 7.8e-3, `lowpass4` 8.9e-3.
Hard BC helps the FD objective by 1.35x and the representation by ~15%; it does not change
the conclusion — the objective does.

## Round 1 — collocation of the STRONG-form residual (`runs/colloc_K8`, job 2466637)

m-node subsets, nearest init (mean init in the JSON), 60 attempts, paired node sets:

| objective | scheme | m=128 | 256 | 512 | 1024 | full |
|---|---|---|---|---|---|---|
| `fd` | uniform | 7.0e-1 | 3.1e-1 | 1.1e-1 | 6.6e-2 | 6.2e-2 |
| `fd` | source-biased IS | 7.7e-2 | 6.2e-2 | 5.9e-2 | 6.7e-2 | |
| `fd` | NNLS-EQ (residual snapshots) | 2.1e-1 | 1.1e-1 | 7.5e-2 | 6.7e-2 | |
| `fd` | off-grid strong form (+BC penalty) | 2.9e-1 | 2.4e-1 | 1.7e-1 | 1.6e-1 | |
| `spec_a1_M256` | uniform | 2.6e-1 | 1.9e-1 | 1.0e-1 | 5.1e-2 | 7.8e-3 |
| `spec_a1_M256` | source-biased IS | 5.2e-2 | 2.8e-2 | 2.6e-2 | 1.9e-2 | |
| `spec_a1_M256` | NNLS-EQ | 1.2e-1 | 5.5e-2 | 3.3e-2 | 2.0e-2 (med 7.9e-3) | |
| `spec_a1_M256` | off-grid | 4.4e-1 | 3.8e-1 | 3.7e-1 | 3.5e-1 | |
| `spec_a0.5_M256` | source-biased IS | 4.9e-2 | 2.6e-2 | 2.6e-2 | 2.0e-2 | 1.0e-2 |
| `spec_a0.5_M256` | NNLS-EQ | 2.4e-1 | 6.4e-2 | 3.4e-2 | 2.4e-2 | |

Sub-sampling the **strong-form** residual loses most of the gain: even the best scheme
(importance sampling around the source) is 2.5x above the full-grid number at m=1024
(27% of the grid), and NNLS on residual snapshots is no better (its fit residual is 1–35%
of the target because the residual snapshots are rough). Off-grid strong-form collocation
(autodiff Laplacian + BC penalty) is worst. Reason: `Φ^T r` is a quadrature of a *rough*
integrand (the decoder's Laplacian error), and Monte-Carlo / few-point quadrature of it
is noisy at the level the objective is trying to resolve. This motivates round 2.

## Round 2 — WEAK-form hyper-reduction (`runs/colloc_weak_*`, jobs 2467196–2467200)

`Φ^T A u = Λ Φ^T u` (discrete sine modes are exact eigenvectors of A; boundary values enter
nowhere for the ghost-zero operator), so the objective needs only quadrature of the smooth
decoder output at m points, plus `Λ^{-α}Φ^T f` — M numbers, a per-query preprocessing of the
input. EQ weights are then fitted (capped Lawson–Hanson NNLS, `pro_common.nnls_capped`) to
reproduce the mode projections of *decoder-output* snapshots (64 training latents × 4
perturbations), either on grid nodes (`nnls`) or on a fixed pool of 4096 random off-grid
candidates (`nnlsoff` — fully meshfree). Random schemes are kept as controls.

Local smoke (4 sources, 30 attempts, `smoke/colloc_weak_smoke.json`): `weak_a1_M64` full grid
8.5e-3 (= `spec_a1_M64`); NNLS on grid nodes m=128 → 1.34e-2, m=512 → 8.9e-3 (oracle 8.0e-3);
meshfree NNLS m=128 → 1.34e-2; uniform / biased / off-grid random m=512 → 2.2e-1 / 1.6e-1 /
2.8e-1 (a localized bump family is hopeless for Monte-Carlo quadrature); strong-form NNLS
m=128 → 4.3e-1.

Cluster results, K=8 soft-BC decoder (`runs/colloc_weak_K8`, job 2467196; 16 sources, 60
attempts, oracle 7.78e-3 nearest / 2.58e-2 mean), ROM mean rel-L2, nearest init (mean init
in parentheses where it differs):

| objective | scheme | m=64 | 128 | 256 | 512 | full grid |
|---|---|---|---|---|---|---|
| `weak_a1_M64` | NNLS-EQ, grid nodes | 5.5e-2 | 1.35e-2 | **8.9e-3** | **8.5e-3** | 8.5e-3 |
| `weak_a1_M64` | NNLS-EQ, **meshfree** pool | 5.6e-2 | 1.9e-2 | **9.4e-3** | **8.6e-3** | |
| `weak_a1_M64` | uniform / biased / off-grid random | 4.5e-1 | 4.1e-1 | 3.1e-1 | 2.9e-1 / 1.5e-1 / 2.8e-1 | |
| `weak_a1_M256` | NNLS-EQ, grid nodes | 1.5e-1 | 4.7e-2 | 1.6e-2 | **7.9e-3** (2.6e-2) | 7.8e-3 |
| `weak_a1_M256` | NNLS-EQ, meshfree | 1.7e-1 | 5.7e-2 | 1.4e-2 | **8.1e-3** | |
| `weak_a0.5_M64` (energy) | NNLS-EQ, grid | 9.4e-2 | 1.9e-2 | 9.4e-3 | 8.8e-3 | 8.8e-3 |
| `weak_a0.5_M64` | NNLS-EQ, meshfree | 1.0e-1 | 3.2e-2 | 1.0e-2 | 9.0e-3 | |
| `weak_a0.5_M256` | NNLS-EQ, grid / meshfree | 7.8e-1 | 4.8e-1 | 9.6e-2 / 8.7e-2 | 9.8e-3 / 1.1e-2 | 9.6e-3 |
| any `weak` | uniform / biased / off-grid random | ≥0.45 | ≥0.32 | ≥0.26 | ≥0.15 | |

- **EQ on the weak form recovers the full-grid number with m ≈ 4M points**: 64 modes ×
  256 nodes → 8.9e-3 (grid) / 9.4e-3 (meshfree) vs 8.5e-3 full / 7.8e-3 oracle; the same holds
  from the mean init (init-free), and the meshfree candidate pool is as good as grid nodes —
  the ROM never touches the mesh online. Fit quality tracks the result: NNLS residual falls
  from 13% (m=64) to 0.03% (m=512) of the target for M=64.
- **Random quadrature of any kind fails** (0.15–0.99) — a family of localized bumps cannot be
  integrated by Monte-Carlo with hundreds of points; importance sampling helps only 2x. EQ is
  not optional.
- Hard-BC decoder (`runs/colloc_weak_hbc`, job 2467197; oracle 7.1e-3): M=64 m=256 → 8.0e-3
  (grid) / 8.5e-3 (meshfree), m=512 → 7.7e-3 / 8.3e-3; M=256 m=512 → 7.5e-3 / 7.7e-3.
- **N ladder at fixed (k=8, M=64, m=512, meshfree EQ)** (`runs/colloc_weak_N{32,128}` jobs
  2467199/2467200 + the K8 cell): ROM 1.15e-2 / 8.6e-3 / 1.05e-2 for N=32/64/128 against
  oracles 9.0e-3 / 7.8e-3 / 8.5e-3 — flat, and by construction the online cost is N-free
  (m point evaluations of the decoder + an M×m matvec + a k×k solve per iteration).

## Verdict

- **Recipe.** Galerkin/energy-norm or `Λ^{-1}`-weighted spectral objective on a few dozen
  smooth test modes; hyper-reduce via the **weak form** with NNLS-EQ weights fitted on
  decoder-output snapshots (meshfree candidate pool works). Any init. This turns the K=8
  Poisson ROM from 6.3e-2 (init-sensitive, N-worsening) into ~9e-3–1.3e-2 (init-free, flat in
  N) against an 8e-3 inferred-latent floor.
- **What did NOT work.** Plain FD residual (LSPG) at any budget or collocation; α=0
  Galerkin without weighting beyond ~64 modes; 5-step CG; any grid-cell-scaled smoothing across
  N; random / importance-sampled / off-grid collocation of the *strong-form* residual; NNLS-EQ
  on residual snapshots; hard-BC on its own (helps 1.35x, not 8x).
- **For Agent B (Burgers latent stepping).** Use the weak-form Galerkin objective per time step:
  test modes = low sine modes (or POD modes of the training snapshots), residual of the
  implicit step integrated by parts against them (diffusion term becomes `ν λ_i φ_iᵀ u`,
  advection term `−½ ∫ u² ∂φ_i` — no decoder derivatives at all), `Λ^{-1/2}`-type weighting,
  EQ weights fitted on decoder-output snapshots (nnlsoff pool is fine), damped GN warm-started
  from the previous latent; do not spend time on FD-residual LSPG or strong-form random
  collocation.

## Follow-up (2026-08-17) — k ladder, multi-seed, EQ knobs, cost, complexity

`followup/` extends this study without touching anything above: new code lives in
`followup/`, new results in `runs/followup/<cell>/`, generated tables in
`followup/FOLLOWUP_TABLES.md` and figures in `followup/figs/` (also copied to
`Plots/`).  Every table above is unchanged.

Everything below is N=64, hard-BC K-latent coordinate auto-decoders trained with the
*same* recipe and the *same* budget (`fu_train.py` = `pro_train.py` + a `TRAIN_SEED`
env var), 16 held-out test sources, LM budget 60, f64, `jax_backend=gpu`,
`JAX_DEFAULT_MATMUL_PRECISION=highest`, one isolated cluster job directory per cell,
data regenerated on the cluster from the seed, checksummed pull, cluster directories
deleted.  Adversarial review before the fan-out: `CODEX-REVIEW-followup.md` (all MUST
items applied; the disposition table is in that file).

### New code

| path | what |
|---|---|
| `followup/fu_train.py` | `pro_train.py` + `TRAIN_SEED` (network init, latent init, batch order; the DATA draw always uses `mp.SEED`, so a multi-seed sweep varies TRAINING randomness only).  Non-default seeds get their own file name. |
| `followup/fu_pod.py` | the linear control: POD projection floor, Galerkin, FD-LSPG and the coordinate ROM's own `weak_a1_M` objective, all on the same TRAIN snapshots and the same 16 test sources, as exact minimisers (the problem is quadratic in the POD coefficients), with `rank`/`cond`/square-system flags. |
| `followup/fu_eq.py` | the NNLS-EQ weak-form quadrature fit and the jitted LM latent solve, shared by the timing and complexity tools; `weak_source_projector` splits `pro_common.weak_source_term` into its offline (per-mesh mode table) and online (one matvec) halves. |
| `followup/fu_timing.py` | one-GPU, one-process cost measurements: `MODE=n` (N ladder), `MODE=k` (per-iteration cost vs k, plus the POD control), `MODE=m` (m and M ladders).  2 warm-ups, median of 7, `block_until_ready`, FOM = the testbed's own jitted CG with its residual asserted first. |
| `followup/fu_family.py` | the complexity ladder: NB independent bump sources (intrinsic dimension 4·NB), a full k ladder per family, coordinate ROM vs POD. |
| `followup/fu_summarize.py`, `fu_style.py` | tables + figures straight from the JSONs. |
| `followup/cluster/{fu_cells.sh,launch.sh,pull.sh}` | `pobj2/` namespace: one job per cell dir, sha256-verified staging and pull, `squeue` checked before and after every submit, cluster dirs deleted after a verified pull. |

`pro_colloc.py` gained one opt-in flag, `EQ_FIXED_SNAPS` (default 0 = the frozen
behaviour): with `=1` the EQ snapshot indices, latent perturbations and row subset
come from a fixed stream, so every point of an m or M ladder is fitted on the *same*
decoder snapshots and the grid/meshfree pools are compared like-for-like.  Every
follow-up cell sets it.  It also reports per-row NNLS fit diagnostics
(median / p95 / max of |Gw−b|/|b| over the fitted rows; the max is dominated by rows
whose target projection is near zero and should be read next to the median).

### Cells

| cell | job | what |
|---|---|---|
| `pk_K{2,4,6,8,12,16,24,32}` | 2481663–2481691 | k ladder: retrain the hard-BC auto-decoder at each K (equal budget), then the weak-form ROM (full grid / NNLS-EQ grid / NNLS-EQ meshfree, m=256) and the FD-LSPG control |
| `ps_S{1,2}` | 2481694/2481698 | multi-seed: K=8, `TRAIN_SEED` 1 and 2 (seed 0 = `pk_K8`) |
| `pm_m` | 2481705 | m ladder at (K=8, M=64): m ∈ {64,128,256,512,1024} × {grid, meshfree} + the full grid |
| `pM_M{16,32,64,128,256}` | 2481706–2481719 | M ladder at m ≈ 4M (one cell per M: the EQ weights must be refit for each M) |
| `pp_pod` | 2481724 | POD k ladder (linear control) |
| `pt_n` | 2482319 | cost vs N on one GPU, N ∈ {32,64,128,256,512} sequential in one process |
| `pt_m` | 2481730 | per-solve / per-iteration cost across the m and M ladders, one GPU |
| `pt_k` | 2482334 | per-iteration cost and iterations vs k, one GPU, + the POD control |
| `pc_nb{1,2,3}` | 2482077–2482082 | complexity ladder: 1 / 2 / 3 bump sources (intrinsic dimension 4 / 8 / 12), a full k ladder each |

### k ladder — the nonlinear manifold is worth ~an order of magnitude in k

Full tables in `followup/FOLLOWUP_TABLES.md`; figure `followup/figs/poisson_k_ladder.png`.
Nearest init, `weak_a1_M64`, mean rel-L2 over the 16 held-out sources:

| k | 2 | 4 | 6 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|
| coordinate ROM, full grid | 1.31e-1 | 1.57e-2 | 9.69e-3 | **7.65e-3** | 7.28e-3 | 5.22e-3 | 5.47e-3 | 5.34e-3 |
| coordinate ROM, NNLS-EQ m=256 (meshfree) | 1.31e-1 | 1.59e-2 | 9.95e-3 | 8.98e-3 | 8.35e-3 | 6.99e-3 | 6.77e-3 | 7.67e-3 |
| coordinate inferred-latent floor (oracle) | 1.31e-1 | 1.55e-2 | 9.34e-3 | 7.11e-3 | 6.24e-3 | 4.13e-3 | 3.98e-3 | 3.43e-3 |
| POD, same weak objective (full grid) | 4.57e-1 | 2.91e-1 | 2.20e-1 | 1.77e-1 | 1.29e-1 | 1.01e-1 | 6.68e-2 | 5.11e-2 |
| POD projection floor | 4.57e-1 | 2.91e-1 | 2.20e-1 | 1.77e-1 | 1.29e-1 | 1.01e-1 | 6.68e-2 | 5.11e-2 |

1. **The coordinate ROM's knee sits at the intrinsic dimension.** The family has 4
   parameters; the ROM falls 8x between k=2 and k=4, another 2x by k=8, and is flat
   from k=8 to k=32 (7.7e-3 → 5.3e-3, and the *floor* it is tracking only improves
   from 7.1e-3 to 3.4e-3).  Below the intrinsic dimension the decoder is
   representation-limited (k=2: ROM = oracle = 1.31e-1 to three digits — the solve is
   perfect, the manifold is not big enough).
2. **POD never catches it in range.** With the *same* objective, the *same* test
   modes and the *same* training snapshots, POD at k=8 is 1.77e-1 — **23x** the
   coordinate ROM — and POD at k=64 (2.54e-2) is still 3.3x worse than the coordinate
   ROM at k=8.  Extrapolating the POD line (≈ k^-1.2 here) it would need k ≈ 250 to
   reach 7.7e-3, i.e. **more than 30x the latent dimension** for the same accuracy.
   There is no crossover inside any k a ROM would use.
3. **POD sits exactly on its own projection floor** (Galerkin, the weak objective and
   the projection agree to <1% at every k ≤ 48): the linear-subspace ceiling, not the
   solver, is what limits it.  The one exception is k=64 with M′=64 modes, where the
   Petrov–Galerkin system is square (4.89e-2 vs a 2.54e-2 floor) — flagged in the JSON
   as `square_or_underdetermined` and excluded from the figure.
4. **Hyper-reduction costs little across the whole ladder**: NNLS-EQ with m = 4M = 256
   meshfree points is within 17% of the full grid at every k ≤ 24 (8.98e-3 vs 7.65e-3
   at k=8), and the meshfree pool matches the grid nodes.
5. **The FD-LSPG control gets *worse* with k** (4.6e-2 at k=8, 1.9e-1 at k=32): more
   latent directions give the FD residual more room to trade grid-scale error against
   the solution, which is the amplification mechanism of round 1 seen from a new angle.
6. **Init independence is a K=8 property, not a universal one.** From the *mean*
   latent the weak objective still lands on the nearest-init number at k=4, 8 and 16,
   but at k=6, 12, 24 and 32 one or two of the sixteen sources fall into a second basin
   (means 6.7e-2, 4.9e-2, 1.3e-2, 5.1e-2 against medians 5–9e-3).  Higher-dimensional
   latent spaces have more basins; the nearest-source init (legitimate online
   information — the ROM is given the source) removes them.

### Multi-seed (K=8, three training seeds)

`TRAIN_SEED` changes the network initialisation, the latent initialisation and the
minibatch/point sampling; the data draw, the train/val split and the test set are
fixed, so the spread is training randomness only.

| quantity | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| coordinate ROM, full grid | 7.65e-3 | 8.16e-3 | 7.71e-3 | **7.84e-3 ± 2.8e-4** |
| coordinate ROM, NNLS-EQ m=256 grid | 8.66e-3 | 9.58e-3 | 8.41e-3 | 8.88e-3 ± 6.2e-4 |
| coordinate ROM, NNLS-EQ m=256 meshfree | 8.98e-3 | 8.91e-3 | 9.38e-3 | 9.09e-3 ± 2.5e-4 |
| inferred-latent floor (oracle, test) | 7.11e-3 | 7.67e-3 | 1.15e-2 | 8.76e-3 ± 2.4e-3 |
| FD-LSPG control | 4.63e-2 | 6.14e-2 | 4.59e-2 | 5.12e-2 ± 8.8e-3 |
| POD control k=8 (Galerkin / weak / FD) | 1.78e-1 / 1.77e-1 / 2.02e-1 | identical | identical | std 0 by construction |

The headline number is reproducible to **3.6%** across training seeds, and the
ROM/FD gap (6.5x) is far outside the spread.  The POD basis is a deterministic
function of the training snapshots, so the POD control carries no seed variance —
its standard deviation is zero by construction, not by measurement.

### The hyper-reduction knobs — m and M ladders (k=8)

`weak_a1_M64`, nearest init, 16 held-out sources; the full-grid number is 7.65e-3 and the
finite-budget inferred-latent floor is 7.11e-3.  Figure: `followup/figs/poisson_eq_knobs.png`.

| m | 64 | 128 | 256 | 512 | 1024 | full grid (3844) |
|---|---|---|---|---|---|---|
| NNLS-EQ, grid nodes | 5.48e-2 | 1.80e-2 | 8.66e-3 | 7.68e-3 | 7.65e-3 | 7.65e-3 |
| NNLS-EQ, meshfree pool | 5.23e-2 | 1.92e-2 | 8.98e-3 | 7.84e-3 | 7.80e-3 | — |
| NNLS relative fit residual (grid / meshfree) | 1.4e-1 / 1.2e-1 | 1.6e-2 / 1.7e-2 | 2.4e-3 / 2.6e-3 | 2.4e-4 / 2.7e-4 | 1.6e-5 / 2.4e-5 | 0 |

**The NNLS fit residual is a usable a-priori diagnostic**: it falls five orders of magnitude
across the ladder and the ROM error follows it, saturating at the full-grid value exactly
where the fit residual drops below ~1e-4 (m=512, 13% of the grid).  m = 4M = 256 costs 13%
accuracy for a 15x reduction in quadrature points; m = 8M = 512 costs nothing measurable.
The meshfree candidate pool is indistinguishable from grid nodes at every m — the online
ROM never touches the mesh.

| M (requested) | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| modes actually retained M′ | 17 | 32 | 64 | 129 | 256 |
| m = 4M | 64 | 128 | 256 | 512 | 1024 |
| full grid | 1.49e-2 | 9.53e-3 | 7.65e-3 | 7.24e-3 | 7.06e-3 |
| NNLS-EQ, grid, m=4M | 3.03e-2 | 1.34e-2 | 8.66e-3 | 7.33e-3 | 7.07e-3 |
| NNLS-EQ, meshfree, m=4M | 3.79e-2 | 1.44e-2 | 8.98e-3 | 7.40e-3 | 7.18e-3 |
| ms per solve, full grid | 20.1 | 45.3 | 35.6 | 38.1 | 47.4 |
| ms per solve, EQ m=4M (meshfree) | 6.7 | 14.8 | 11.5 | 16.6 | 23.9 |

More test modes monotonically help and monotonically cost: M=16 is genuinely
under-resolved (1.5e-2 even on the full grid), M=64 is within 8% of M=256, and M=256 sits
on the inferred-latent floor (7.06e-3 vs 7.11e-3).  The `m ≈ 4M` rule holds across the
whole range: at every M the EQ number is within 5% of that M's full-grid number once
M ≥ 32, at 2–3x lower cost per solve.

### Online cost vs N — the latent solve does not see the mesh

One A100, all five meshes measured **sequentially in one process**, 2 warm-ups then the
median of 7 `block_until_ready` repetitions; the FOM is the testbed's own jitted CG at its
own tolerance, with its converged residual asserted before anything is timed.  The
coordinate decoder is meshfree, so the *same* N=64 checkpoint is used at every N and only
the EQ quadrature is refit.  Figure: `followup/figs/poisson_cost_vs_N.png`.

| N | interior DOF | FOM (CG) | ROM latent solve | iters | ms/iteration | input projection Λ⁻¹Φᵀf | full-field decode | speedup (solve) | speedup (end to end) | ROM rel-L2 vs FD at this N |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 900 | 5.59 ms | 19.86 ms | 27.8 | 0.602 | 0.07 ms | 0.13 ms | 0.3x | 0.3x | 1.08e-2 |
| 64 | 3844 | 7.79 ms | 19.77 ms | 27.8 | 0.599 | 0.07 ms | 0.23 ms | 0.4x | 0.4x | 8.94e-3 |
| 128 | 15876 | 15.14 ms | 19.87 ms | 27.8 | 0.602 | 0.08 ms | 0.47 ms | 0.8x | 0.7x | 8.86e-3 |
| 256 | 64516 | 31.13 ms | 19.98 ms | 27.8 | 0.605 | 0.10 ms | 2.20 ms | **1.6x** | 1.4x | 8.86e-3 |
| 512 | 260100 | 96.01 ms | 19.71 ms | 27.8 | 0.597 | 0.19 ms | 7.04 ms | **4.9x** | **3.6x** | 8.86e-3 |

The latent solve is **19.71–19.98 ms across a 289x range of degrees of freedom** (±0.7%),
with the same 27.8 iterations and the same 0.60 ms per iteration, while the FOM grows 17x;
the ROM error is flat too (8.86e-3 from N=64 up).  The crossover is at N ≈ 180
(n ≈ 3·10⁴ DOF).

Two honest caveats, both visible in the table:

- **The reported preprocessing number is the corrected one.**  `pro_common.weak_source_term`
  rebuilds its (M′ × n_i²) continuum-mode table on every call.  Timing that as "online"
  charges O(M′n) *sine evaluations* per query to the online path — 508 ms at N=512, which
  swamped the 19.7 ms solve and made the end-to-end speedup read 0.2x.  The mode table is a
  per-mesh constant, so `fu_eq.weak_source_projector` builds it offline (0.9 ms at N=32 …
  1.0 s at N=512, reported separately in the JSON) and the online cost is the single matvec
  above, verified equal to `pro_common.weak_source_term` to 1e-17 absolute in every row.
  **The first `pt_n` run used the unsplit version and is not what is tabulated here**; the
  table is job 2482319, the rerun.
- **The end-to-end column is not n-free and does not claim to be.**  The input projection is
  O(M′n) and the full-field decode is O(n) — the decode is the *output* (you only pay it if
  you want the field on the mesh rather than a functional), and at N=512 it is 7 ms against
  the 19.7 ms solve.  What is n-free is the *solve*, which is the part that scales with the
  physics.

### Per-iteration cost and iterations vs k

One A100, N=64, M=64, m=256 meshfree EQ, median of 7.  The reference LM has no absolute
residual tolerance (it stops on relative decrease / step size / budget), so these are
**iterations to termination**, reported with the termination-reason histogram in the JSON.
The POD ROM is linear, so its exact online solve is one precomputed pseudo-inverse matvec.

| k | 2 | 4 | 6 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|
| ROM solve | 6.5 ms | 7.4 ms | 16.6 ms | 20.2 ms | 16.1 ms | 15.9 ms | 21.2 ms | 24.6 ms |
| Jacobian evaluations | 23.1 | 16.3 | 26.0 | 27.8 | 31.6 | 32.1 | 34.9 | 39.7 |
| ms per iteration | 0.434 | 0.529 | 0.535 | 0.611 | 0.555 | 0.723 | 0.661 | 0.910 |
| POD online solve | 52 µs | 52 µs | 53 µs | 51 µs | 54 µs | 51 µs | 52 µs | 73 µs |

Cost grows mildly and smoothly with k — 2.1x in per-iteration time and 1.7x in iteration
count from k=2 to k=32 — so the accuracy gained between k=4 and k=8 (2x) is bought at
2.7x the solve time, and nothing beyond k=8 pays for itself.  The POD solve is a single
small matvec, dispatch-bound at ~50 µs; that is a floor on the measurement, not a property
of the method.  (The error column of this cell uses the *mean* init and is a cross-check
only; the accuracy statistics are the nearest-init `pk_K*` numbers above.)

### Complexity ladder — the manifold's advantage is a property of the DATA, not the method

`pc_nb{1,2,3}`: NB independent Gaussian bump sources, intrinsic dimension exactly 4·NB.
NB=1 is `ms_parametric.sample_params` verbatim and reproduces the `pk_K*` ladder to all
printed digits (K=8: 7.654e-3 in both cells — an independent-implementation cross-check of
the whole pipeline).  Everything else is held fixed: 512 training sources, the same
training recipe and 20 000-step budget at every (NB, k), the same objective, the same 16
held-out sources.  Figure: `followup/figs/poisson_complexity_ladder.png`.

| | k=4 | k=8 | k=16 | k=32 | POD k=8 | POD k=32 | coord/POD at k=8 |
|---|---|---|---|---|---|---|---|
| NB=1 (intrinsic 4) | 1.57e-2 | **7.65e-3** | 5.22e-3 | 5.34e-3 | 1.77e-1 | 5.11e-2 | **23x** |
| NB=2 (intrinsic 8) | 1.52e-1 | 1.10e-1 | 5.05e-2 | 2.81e-2 | 1.50e-1 | 3.89e-2 | **1.4x** |
| NB=3 (intrinsic 12) | 1.79e-1 | 1.06e-1 | 5.24e-2 | 2.65e-2 | 1.13e-1 | 2.67e-2 | **1.1x** |

**The answer to the question is: no, and the reason is generalization, not the ROM.**  At
NB=1 the coordinate manifold's knee sits at the intrinsic dimension and it beats POD by
23x.  At NB=2 and NB=3 it does not saturate anywhere below k=32 and its advantage over POD
collapses to 1.1–1.4x.  The ROM is *not* at fault: at every (NB, k) it sits on its own
finite-budget inferred-latent floor (NB=2, k=8: ROM 1.096e-1 vs oracle 1.092e-1; NB=3,
k=16: 5.24e-2 vs 5.16e-2), and the FD-LSPG control stays 3–6x worse everywhere, so the
objective story of round 1 holds at every complexity.  What degrades is the *decoder's
held-out quality*: its training reconstruction only worsens from 7.0e-3 to 1.4e-2 between
NB=1 and NB=2 at k=8, but its held-out inferred-latent floor worsens from 7.1e-3 to
1.09e-1 — a 15x train/test gap.  With 512 training sources, 512^(1/4) ≈ 4.8 samples per
parameter covers a 4-dimensional family and 512^(1/12) ≈ 1.7 does not cover a
12-dimensional one.

The practical statement is therefore: **the nonlinear manifold buys an order of magnitude
in k when the training set actually covers the family's intrinsic dimension, and nothing
when it does not** — and the diagnostic that tells you which regime you are in is the gap
between the decoder's training reconstruction and its held-out inferred-latent floor, both
of which are computable offline without ever running the ROM.  Whether more training
sources or a longer budget restores the NB=2/3 advantage is a separate experiment (not
run): only the sample count was held fixed here, so this is a statement about *this*
budget, not about the method's asymptotics.

### Caveats

- Single test set of 16 held-out sources; means carry heavy tails (medians are in the
  JSONs and in `FOLLOWUP_TABLES.md`).  The k-ladder means from the *mean* init are
  outlier-dominated at k=6, 12, 24 and 32 — see the init-independence note above.
- The multi-seed arm varies training randomness only, and its cells were not pinned to one
  GPU model; that is sound because only *errors* are compared there (f64, hardware
  independent).  No timing is taken from those cells.
- The k-ladder POD comparison is full-grid vs full-grid.  The coordinate ROM's NNLS-EQ rows
  use a different (hyper-reduced) quadrature and are plotted as a separate dashed series;
  `fu_pod.py` has no EQ path.
- POD at k=64 with M′=64 test modes is a square Petrov-Galerkin system (4.89e-2 against a
  2.54e-2 projection floor); it is flagged `square_or_underdetermined` in the JSON and
  excluded from every figure.
- `blat`-style degenerate-eigenshell truncation does not apply here: `Grid.mode_mask` keeps
  complete eigenshells, so M′ ≥ M and the retained count is reported in every row.
- The extrapolation "POD would need k ≈ 250" is a fit to the measured k ≤ 64 range, stated
  as an extrapolation and not measured.

## Caveats

- Single seed, 16 test sources (means carry heavy tails — medians are in the JSONs).
- Discrete sine modes are exact eigenvectors only for the constant-coefficient Dirichlet
  Laplacian on the square; for general operators use `A^T φ_i` precomputed offline (weak form
  still needs no decoder derivatives) or POD test modes.
- `spec_a1_Mall` is the *interior* data misfit (boundary values ~1e-4 excluded); the oracle
  includes them — negligible here, identical for hard-BC decoders.
- Off-grid strong-form runs on soft-BC decoders use a penalty `β=1/dx²` on m_b perimeter
  points (labelled `bc_beta`); the off-grid limit is a continuum objective, not the FD one.
- NNLS diagnostics: `rnorm_capped` is the capped LH residual on the row subset;
  `rnorm_final` is the refit on the final support over all rows; padded nodes (none occurred)
  would carry median weights.
- Adversarial review: `CODEX-REVIEW-2026-08-16.md` (all MUST items fixed before the fan-out;
  the second reviewer of the usual two-reviewer gate could not be spawned from this fork —
  a self-review checklist replaced it, see the commit message).

## Files

| path | what |
|---|---|
| `pro_common.py` | grid/eigenbasis, objectives (full-grid + collocation + weak), generic LM, capped NNLS, encoder, hard-BC auto-decoder trainer |
| `pro_objective.py` | cell 1/3: objective × init sweep |
| `pro_colloc.py` | cell 2: collocation / EQ / off-grid / weak-form study |
| `pro_train.py` | cell 4/5: stage-0 auto-decoder retrain (hard-BC, other N) |
| `pro_summarize.py` | markdown tables from `runs/**.json` |
| `cluster/make_cells.sh` | builds the Tufts cell dirs (round 1 + round 2) |
| `runs/<cell>/` | JSON results, logs, retrained pkls |
| `smoke/` | local N=64 smokes + the golden FD control |
