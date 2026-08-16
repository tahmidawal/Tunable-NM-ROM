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
