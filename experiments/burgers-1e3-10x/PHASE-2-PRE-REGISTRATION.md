# Burgers pure NM-ROM at 1e-3 and 10x — adaptive Phase-2 preregistration

Status: locked prospectively after the preregistered Phase-1 transported-Hermite
representation hard stop and before any Phase-2 scientific job.  Phase 2 is a new,
adaptive preregistration.  Its new cap is 12 scientific GPU cells.  Saying that
Phase 1 used three cells and that the cumulative accounting remains at most 15 is
bookkeeping, not retroactive authorization to change the Phase-1 search.

The seed-0 indices `512:576` have already been exposed and are development/selection
data.  The seed-0 model-validation indices `576:640`, reserve indices `640:704`, and
seed-20261031 confirmation draw remain unopened.  Every split, PDE, accuracy target,
timing rule, exclusion, and information boundary in `PRE-REGISTRATION.md` remains in
force unless this document explicitly tightens or replaces it.

## Fixed Phase-2 hypothesis

The sole candidate family is a transported compact-support tensor-product cubic
B-spline hyperdecoder.  It is genuinely nonlinear and has no POD basis, linear
corrector, full-grid FOM correction/fallback, learned coefficient basis, or direct
linear latent-to-coefficient skip.  Its parameter count depends on a fixed aligned
control grid but not on deployment mesh size `N`.

The online state is

```
z = (mu_x, mu_y, ell_11, ell_21, ell_22, q),   dim(z) = k,
L = [[exp(ell_11), 0], [ell_21, exp(ell_22)]],
xi = solve(L, x - mu).
```

Thus five variables describe translation and a full-covariance lower-triangular
transport, and `q` has dimension `k-5`.  Once per time step, a two-hidden-layer
width-32 Swish hypernetwork maps `q` to `R^2` spline coefficients.  The final layer
is affine, but there is no affine `q -> coefficients` skip and no fixed spatial
basis added to its output.  Coefficients are in physical solution units: there is
no per-control-point centering or scaling.  The only conditioning convention is
that every snapshot field loss is divided by its own squared L2 norm and the
autolatent raw variables are mapped to `q in (-1,1)^(k-5)` with `tanh`.

The direct predictor is a separate width-32, two-hidden-layer Swish MLP from the
seven locked deployable features `(recovered cx, cy, width, amplitude, viscosity,
time, target spacing)` to all `k` state variables.  Initial-field recovery reads the
fixed deterministic at-most-64-by-64 target-grid sample.  The direct decoded arm is
a charged surrogate control only; it is never headline-eligible as an NM-ROM.

## Spline mathematics locked before S0

The aligned domain is exactly `D=[-4.5,4.5]^2`.  In each coordinate the spline has
degree `p=3`, `R` control points, and the open uniform clamped knot vector

```
[-4.5]*4,
-4.5 + j*h for j=1,...,R-4,
[4.5]*4,                    h = 9/(R-3).
```

The Cox-de Boor span convention is right-open except that `+4.5` belongs to the
last span.  A point in `D` evaluates exactly the four active one-dimensional basis
functions in each direction and therefore exactly 16 tensor-product coefficients.
A point outside `D` returns exactly zero; coordinates are not extrapolated or
clipped into a nonzero span.

On a target grid, the final value is the spline value times the exact binary mask
`b_N(i,j)=1` only for `1 <= i,j <= N-2`, and zero otherwise.  The mask is applied
after aligned-domain support, so all four physical boundary rows/columns are
bitwise zero.  The transport is evaluated first using the convention above; no
physical-wall smoothing, boundary envelope, or post-S0 domain/knot adjustment is
allowed.

For the free spatial oracle, transport is not optimized.  It is the deterministic
positive-field L2 moment alignment

```
w_i = max(u_i,0)^2,
mu = sum_i w_i*x_i / sum_i w_i,
C = 2*sum_i w_i*(x_i-mu)*(x_i-mu)^T/sum_i w_i + 1e-8*I,
L = chol(C).
```

For each snapshot, coefficients minimize the binary-masked target-grid least
squares objective with Tikhonov ridge

```
||Phi c-u||_2^2 + lambda ||c||_2^2,
lambda = 1e-12 * max(mean(diag(Phi^T Phi)), 1e-300).
```

The implementation uses the exact 16-entry sparse rows and deterministic LSMR
with `atol=btol=1e-12`, `conlim=1e12`, and at most 500 iterations.  A fit is healthy
only when all values are finite and the explicit relative normal-equation residual
`||Phi^T(Phi c-u)+lambda*c|| / max(||Phi^T u||,1e-300)` is at most `1e-8`.
No failed fit is omitted.  The free oracle is nondeployable and can license only
the learned small-`k` coefficient-manifold stage.

The exact initial family becomes a unit Gaussian under the locked moment alignment.
Its relative L2 mass outside `D` is bounded by
`sqrt(1-erf(4.5)^2)=1.983e-5`.  For one aligned Gaussian coordinate,
`||g''''||_2/||g||_2=sqrt(105/16)`.  The conservative complete-cubic-spline bound
used for the pre-S0 screen is

```
delta_R = (5/384) * h^4 * sqrt(105/16),
two_dimensional_bound = sqrt(1-erf(4.5)^2) + 2*delta_R + delta_R^2.
```

It is approximately `2.27e-3`, `6.39e-4`, and `1.27e-4` for `R=24,32,48`.
Only `R=48` is analytically below the `2e-4` mean gate; the two smaller grids are
finite lower-resolution brackets, while `R=48` prevents an artificially cheap
all-fail bracket.  This bound screens the analytic initial family, not evolved
solutions.  S0's measured all-time free projection oracle on the locked exposed
selection trajectories is the falsification test for the evolved family.

## Finite candidate bracket and analytic cost screen

The ordered candidates are fixed and no other `R`, `k`, hypernetwork width, knot,
or domain is permitted:

| arm | R | k | q=k-5 | hyperdecoder parameters | base M | base m |
|---|---:|---:|---:|---:|---:|---:|
| A | 24 | 12 | 7 | 20,320 | 64 | 256 |
| B | 32 | 16 | 11 | 35,232 | 64 | 256 |
| C | 48 | 24 | 19 | 77,728 | 96 | 384 |

The hyperdecoder counts include all biases.  The direct predictor adds
`1,708`, `1,840`, or `2,104` parameters respectively.  Every arm satisfies
`M >= max(64,4k)` and `m=4M`; `M` follows the online latent dimension, not `R^2`.

Sparse query work is fixed at 16 coefficient gathers and tensor weights per point.
The conservative screen charges 128 equivalent f64 scalar operations per decoded
point, so a full 51-by-1024-squared decode is at most 6.845 equivalent GFLOP and
must write 0.428 GB of output.  Arm C's 51 coefficient generations are below
4.0 MFLOP, coefficient storage is 18,432 bytes per time slice, and the direct
predictor is below 0.1 MFLOP.  A candidate is killed before training if measured
peak device memory exceeds 20 GB, measured charged online construction exceeds
23.804863682 ms, its full 51-slice decode cannot complete, hyperdecoder parameters
exceed 100,000, or the implementation violates the 16-support rule.  The fixed
analytic screen kills any unlisted extension above 20 equivalent GFLOP per full
decode or above 2 GB of output/scratch that cannot be fused away.

S0 charges, end to end, the at-most-64-squared initial feature recovery, construction
of all 51 predictor features, direct state prediction, all 51 coefficient-grid
generations, and full 51-by-`N^2` decoding.  It records first use and amortized
setup.  At N=1024 it is timed in balanced order against a live outer-`3e-3`,
inner-`1e-1` cubic-history/exact-Helmholtz FOM on the same GPU.  The FOM accuracy,
work, residuals, flags, and cost come from those same invocations.  S0 is a cost
lower bound because it does not yet include the mandatory weak objective.

## Stage order, cells, and promotion gates

At most 12 Phase-2 scientific GPU job directories may be run:

1. `S0` (one cell): all three free spatial projection oracles on seed-0 selection
   `512:576`, plus the same-GPU N=1024 charged construction/FOM panel and device
   memory records.
2. Seed-11 small-`k` coefficient-manifold candidates A, then B, then C, sequentially
   (at most three cells).  A larger arm runs only after the smaller arm fails the
   representation gate.  Training uses seed-0 indices `0:512` and fixed time indices
   `[0,1,2,4,8,12,16,24,32,40,50]`; evaluation uses every one of the 51 times on
   selection.  Each cell jointly optimizes a newly initialized hyperdecoder and all
   of its training autolatents, then trains a newly initialized direct predictor.
3. The one locked finalist is retrained from scratch at seeds 29 and 47 (two cells).
   Each cell retrains the entire hyperdecoder, all training autolatents, and direct
   predictor; holding a seed-11 decoder fixed is forbidden.
4. At most one conditional all-seed loss revision (one cell, all three independent
   retrainings in that directory).  It is licensed only when every seed's learned
   coefficient-manifold oracle passes, and at least one seed's direct decoder misses
   the `3e-4` mean or `1e-3` worst gate by no more than 2x.  The only revision is to
   add fixed relative gradient, weak-PDE, and one-step rollout weights
   `(0.1,0.1,0.1)` to the existing relative-field loss; architecture and optimizer
   are unchanged.
5. One joint all-seed model-validation/full-weak/base-EQ cell after the architecture,
   loss, and seed policy are locked.  This is the first opening of indices `576:640`.
6. At most one conditional next-`M` EQ cell.  Base values are the table values.
   It is licensed exactly when every seed's full weak result passes, base EQ has zero
   failures, and base EQ misses `mean<=1e-3` or `EQ/full<=1.05`, while still having
   mean `<=1.2e-3` and degradation `<=1.20` for every seed.  The sole retry is
   `M_next=2*M_base`, `m_next=4*M_next`, with newly fitted NNLS weights.  A worse miss
   or failure is a hard stop.
7. One joint N=256/512 development scaling cell.
8. One N=1024 same-GPU development/preconfirmation cell.
9. Exactly one untouched confirmation cell, only if every earlier gate passes.

This allocation is `1+3+2+1+1+1+1+1+1=12` cells at maximum.  Infrastructure-only
attempts with zero scientific output may be resubmitted once and remain recorded.
Local jobs are execution smokes only and never scientific evidence.

Every complete retraining uses the same fixed optimizer schedule.  The manifold
stage uses AdamW (`beta1=0.9`, `beta2=0.999`, `eps=1e-8`, weight decay `1e-6`),
global gradient clipping at 1.0, and 30,000 updates with cosine learning rate
`1e-3 -> 1e-5`.  A batch contains 32 of the locked trajectory/time snapshots and
512 target-grid points drawn by the training seed without replacement within each
snapshot; epoch permutations and point draws are stored.  The optimized variables
are a newly initialized hyperdecoder and one newly initialized raw autolatent for
every locked training snapshot.  The five affine variables stay at their locked
moment values during this manifold stage.  The loss is snapshot-relative field
MSE on the sampled points.  The direct-predictor stage freezes the hyperdecoder and
uses newly initialized predictor weights, the same optimizer/clip/weight decay,
20,000 updates, and cosine rate `5e-4 -> 5e-6`; it minimizes the same decoded-field
loss.  The five affine targets are standardized with training-only mean/scale,
while predictor content outputs pass through `tanh`.  Selection evaluation always
uses all target-grid points and all 51 times, never the training point subsample.

S0 promotes an `R` only if its free projection oracle over all 64 selection
trajectories and 51 times has trajectory mean `<=2e-4`, trajectory worst `<=7e-4`,
zero unhealthy fits, exact binary boundaries, exact 16-support, and a charged
construction median below 23.804863682 ms.  If several pass, choose the smallest
`R`; if none pass, Phase 2 stops without training.

The learned coefficient-manifold oracle optimizes per-snapshot `q` with the
transport fixed by the locked moment rule.  It must independently achieve selection
trajectory mean `<=2e-4`, worst `<=7e-4`, zero nonfinite values, and exact boundaries.
The seed-specific direct predictor must achieve mean `<=3e-4`, worst `<=1e-3`, and
direct/oracle mean degradation `<=1.5`.  All three complete retrainings must pass;
seed averaging cannot rescue a seed.  Before model validation, the deployable seed
policy is fixed prospectively as the arithmetic mean of all three coefficient grids
and affine states, with the cost of all three predictors/hyperdecoders charged.

The direct arm is not an eligible ROM.  On every online step the headline candidate
must evaluate the weak objective at the predicted state.  Define the preconditioned
weak residual `r_w` exactly as in the audited sine-test weak form with exact target-
grid FOM upwind, and define `rho=||r_w||_2/max(||Phi^T u_previous||_2,1e-12)`.
Zero accepted correction is allowed only when finite `rho<=1e-3`; otherwise exactly
one damped Gauss-Newton trust-region attempt is required, with normalized-state
radius `0.25`, LM damping `1e-6`, and deterministic trial factors
`[1,0.5,0.25,0]`.  The best finite trial is accepted only if it reduces `rho`; a
no-op after a required attempt is recorded as a rejected attempt, not a bypass.
Every step persists residual before/after, Jacobian count, residual evaluation count,
attempt flag, accepted factor, step norm, and stopping reason.

The remaining gates are unchanged: independently tighter-reference numerical error
`<=1e-4`; every seed model-validation reconstruction mean `<=3e-4`, worst `<=1e-3`;
full weak-ROM mean `<=7e-4`; EQ mean `<=1e-3`; `EQ/full<=1.05`; zero failures; and
development/confirmation mean `<=1e-3`, worst `<=3e-3`.  N=1024 additionally needs
median same-GPU speedup `>=10x` and trajectory-clustered 95% lower bound `>=8x`
against the fastest eligible like-for-like FOM.  Failure of a hard gate stops all
downstream cells; thresholds are never weakened.

## Reference, weak form, EQ, and artifact locks

Locked training and exposed selection truth may use the legacy fixed-eight-Newton
generator only when every returned and independent residual is finite and at most
`1e-8`.  Accepted model-validation, scaling, N=1024, and confirmation results use
the audited cubic/exact-Helmholtz reference at outer/inner `1e-12/1e-7`, independently
checked against `3e-13/3e-8`; both chains must be finite with zero flags/breakdowns,
meet their returned-residual tolerance, and differ by at most `1e-4` in solution.

Weak tests are the deterministic lowest-eigenvalue sine modes.  Full weak residuals
use every interior target-grid stencil center in lexicographic `(i,j)` order.  EQ
candidate pools are the deterministic lexicographically sorted target-grid stencil
centers from the fixed uniform at-most-64-by-64 lattice rule; no off-grid or random
points are allowed.  Both full and EQ advection use the exact discrete FOM first-order
upwind sign/stencil convention.  NNLS EQ weights are fitted only on decoder-output
snapshots, including their exact upwind images, and refitted from scratch whenever
`N` or `M` changes.  Hyperreduction includes cold-start recovery and all weak calls.

Every scientific artifact stores the source commit and hashes, complete config,
seeds/draws/indices, `N`, viscosity/family, 51-point time grid, `R/k/M/m`, optimizer
state/config, GPU/job/backend, x64/highest flags, reference health, per-case errors,
failures/censoring, work, residual/Jacobian/acceptance arrays, first-use/setup,
balanced timing orders, all repetition arrays, per-trajectory medians/outliers, and
trajectory-clustered confidence intervals.  Timed methods compile, then burn the GPU,
and are compared only within one job/GPU.  Accuracy and cost come from the same solver
invocation.  Cluster staging, checksum pull, cleanup, reports, and generated prose
follow `AGENTS.md` and the original artifact contract.
