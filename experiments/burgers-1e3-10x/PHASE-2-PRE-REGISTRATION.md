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

The online state `z` is itself normalized and bounded in `(-1,1)^k`.  Its first
five components map to physical transport variables by

```
mu_x = 0.5 + 0.75*z_0,             mu_y = 0.5 + 0.75*z_1,
L_11 = exp(log(0.015) + (z_2+1)/2 * log(1/0.015)),
L_22 = exp(log(0.015) + (z_4+1)/2 * log(1/0.015)),
L_21 = 2*z_3*sqrt(L_11*L_22),      q = z_5:,
L = [[L_11, 0], [L_21, L_22]],
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

Every predictor output and every autolatent raw variable passes through `tanh`
before becoming `z`; weak trial states are clipped to `[-1+1e-8,1-1e-8]`.  Moment
transport targets are mapped through the exact inverse ranges above, and any target
outside the closed ranges is a hard failure rather than silently clipped.  This is
the decode-safe affine normalization.  Consequently the later normalized-state
trust radius 0.25 means Euclidean norm in this exact dimensionless `z` coordinate;
there is no learned or post-selection state scaling.

The direct predictor is a separate width-32, two-hidden-layer Swish MLP from the
seven locked deployable features `(recovered cx, cy, width, amplitude, viscosity,
time, target spacing)` to all `k` state variables.  Initial-field recovery reads the
fixed deterministic at-most-64-by-64 target-grid sample without constructing an
`N^2` coordinate array, locates its interior maximum, and performs the exact
log-quadratic fit only on the positive 3-by-3 sampled patch around that maximum.
The direct decoded arm is a charged surrogate control only; it is never headline-
eligible as an NM-ROM.

All seven predictor features are standardized by means and standard deviations
computed from the locked training mix only (a standard deviation below `1e-12` is
replaced by one).  Selection and training use that identical transformation.  For
deployment, standardization is folded exactly into the first affine layer,
`W_fold=W/scale[:,None]` and `b_fold=b-(mean/scale)@W`; the raw-feature folded and
standardized-feature predictions must agree to `1e-12` in f64.  Both parameterizations,
the train-only mean/scale, and the identity error are persisted.  Thus S0's raw-input
MLP has the exact deployed operation shape and no uncharged normalization kernel.

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
The SciPy call sets `damp=sqrt(lambda)`, which is exactly the written ridge objective.
No failed fit is omitted.  The free oracle is nondeployable and can license only
the learned small-`k` coefficient-manifold stage.

S0 executes all 17,136 locked fits: 5,712 per arm, consisting per arm of 3,264 at
N=64, 1,632 at N=128, and 816 at N=256.  It uses a deterministic ordered
eight-worker thread map with every BLAS backend fixed to one thread; output order
remains snapshot order and each fit still uses the exact independent objective and
stopping test above.  No cohort may be truncated for walltime.

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
peak device memory exceeds 20 GB, its full 51-slice decode cannot complete,
hyperdecoder parameters exceed 100,000, or the implementation violates the
16-support rule.  The fixed
analytic screen kills any unlisted extension above 20 equivalent GFLOP per full
decode or above 2 GB of output/scratch that cannot be fused away.

S0 charges, end to end, the at-most-64-squared initial feature recovery, construction
of all 51 predictor features, direct state prediction, all 51 coefficient-grid
generations, 50 base-`m` weak-objective evaluations, and final full 51-by-`N^2`
decoding.  A separate direct-only control omits weak work.  A separate shape-faithful
maximum-one-update path additionally evaluates one `M`-by-`k` Jacobian and the four
fixed trial residuals at each of 50 steps; it is a cost oracle, not a weak-accuracy
claim.  All weak kernels use the corrected state from the preceding step.  S0
records first use and amortized setup.  At N=1024 all arms are timed in balanced
order against a live outer-`3e-3`, inner-`1e-1` cubic-history/exact-Helmholtz FOM on
seed-20260822, draw-count 32, indices `0:4`, on the same GPU.  The FOM accuracy,
work, residuals, flags, and cost come from those same invocations.

The earlier 238.048636820 ms A100 FOM and 23.804863682 ms planning budget remain
context only and can never be a wall-clock gate across jobs.  S0 eligibility uses
the paired live same-job selected-FOM median divided by ten and, from paired case
medians, a trajectory-clustered speedup lower bound of at least 8.  The mandatory
zero-update lower bound includes all 50 weak-objective evaluations; if a later weak
oracle requires updates, its measured mixture including all Jacobian/trial work must
also pass the paired same-job gate before promotion.

S0's untrained shape oracle is deterministic.  Arm A/B/C predictor and hyperdecoder
weights use f64 Xavier-normal initialization with JAX keys 20262012, 20262016, and
20262024 respectively.  Its four online inputs are the live FOM cohort's recovered
initial features.  Base-`m` weak points are the lexicographically sorted tensor
lattice obtained by rounding an at-most-`ceil(sqrt(m))` uniform interior-axis grid
and taking the first `m` unique centers; structural weights are the constant
`(N-2)^2/m`.  They are explicitly not fitted EQ weights and cannot make an accuracy
claim.  Test modes and FOM-upwind stencils are the locked production versions.
Static peak memory for eligibility is the sum of argument, output, temporary, and
non-aliased compiled-executable bytes returned by JAX `memory_analysis()` for each
complete direct/mandatory-weak/max-one-update kernel.  Runtime allocator statistics
before/after first use are also persisted as contextual health evidence, but are not
compared across sequentially compiled arms.

## Stage order, cells, and promotion gates

At most 12 Phase-2 scientific GPU job directories may be run:

1. `S0` (one cell): all three free spatial projection oracles on the exposed mixed-
   resolution seed-0 selection cohorts locked below, plus the same-GPU N=1024
   charged direct/weak/FOM panel and device-memory records.
2. Seed-11 small-`k` coefficient-manifold candidates A, then B, then C, sequentially
   (at most three cells).  A larger arm runs only after the smaller arm fails the
   representation gate.  Training and evaluation use every one of the 51 times in
   the locked mixed-resolution cohorts.  Each cell jointly optimizes a newly
   initialized hyperdecoder and all of its training autolatents, then trains a newly
   initialized direct predictor.
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
   loss, and seed policy are locked.  This is the first opening of the exact mixed-
   resolution model-validation cohorts below.
6. At most one conditional next-`M` EQ cell.  Base values are the table values.
   It is licensed exactly when every seed's full weak result passes, base EQ has zero
   failures, and base EQ misses `mean<=1e-3` or `EQ/full<=1.05`, while still having
   mean `<=1.2e-3` and degradation `<=1.20` for every seed.  The sole retry is
   `M_next=2*M_base`, `m_next=4*M_next`, with newly fitted NNLS weights.  A worse miss
   or failure is a hard stop.
7. One joint N=256/512 development scaling cell.
8. One N=1024 same-GPU development/preconfirmation cell.
9. Exactly one untouched confirmation cell, only if every earlier gate passes.

Candidate training is licensed only by a checksummed artifact chain, never by a
manual override.  Seed 11 may run the S0-promoted smallest arm directly.  A larger
promoted arm may run only when every smaller promoted arm has a completed,
checksummed seed-11 artifact whose learned coefficient-manifold oracle fails; a
direct-predictor failure alone does not license escalation.  Seeds 29 and 47 require
the completed, checksummed, passing seed-11 finalist artifact for the same arm.
Every input artifact and decision, including JSON/NPZ/checkpoint hashes, is persisted.

This allocation is `1+3+2+1+1+1+1+1+1=12` cells at maximum.  Infrastructure-only
attempts with zero scientific output may be resubmitted once and remain recorded.
Local jobs are execution smokes only and never scientific evidence.

Mesh mixtures are fixed before S0 and every named trajectory contributes all 51
time slices.  Training is N=64 seed-0 indices `0:512`, N=128 indices `0:128`, and
N=256 indices `0:64`.  Exposed selection is N=64 indices `512:576`, N=128 indices
`512:544`, and N=256 indices `512:528`.  S0 and learned representation gates must
pass separately at every listed `N`, as well as on the pooled trajectories; a dense
N=64 cohort cannot hide a high-N failure.  Model-validation reconstruction is
N=64 indices `576:640`, N=128 indices `576:608`, and N=256 indices `576:592`.
The joint full/EQ model-validation gate uses the N=256 indices `576:592` for all
three complete retrainings.  N=256 EQ weights use decoder outputs from seed-0 train
indices `0:64`, with 256 `(trajectory,time)` pairs selected once by seed 20260824
and persisted.  Scaling and N=1024 development use seed-20260821 draw-count 32,
indices `0:8`; confirmation remains seed-20261031 draw-count 32, indices `0:8`.
Target spacing is therefore varied explicitly during training rather than constant.

Every complete retraining uses the same fixed optimizer schedule.  The manifold
stage uses AdamW (`beta1=0.9`, `beta2=0.999`, `eps=1e-8`, weight decay `1e-6`),
global gradient clipping at 1.0, and 30,000 updates with cosine learning rate
`1e-3 -> 1e-5`.  A batch contains 32 of the locked trajectory/time snapshots and
512 target-grid points drawn by the training seed without replacement within each
snapshot; epoch permutations and point draws are stored.  The optimized variables
are a newly initialized hyperdecoder and one newly initialized raw autolatent for
every locked training snapshot.  The five affine variables stay at their locked
moment values during this manifold stage.  The loss is snapshot-relative field
MSE on the sampled points plus `1e-6*mean(raw_q^2)`; raw autolatents are initialized
independently as `Normal(0,0.01^2)` from the named training seed.  The direct-
predictor stage freezes the hyperdecoder and
uses newly initialized predictor weights, the same optimizer/clip/weight decay,
20,000 updates, and cosine rate `5e-4 -> 5e-6`; it minimizes the same decoded-field
loss plus `0.1` times normalized-state MSE to the fixed moment/autolatent targets.
All predictor outputs pass through the fixed `tanh` state map above.  Selection
evaluation always uses all target-grid points and all 51 times, never the training
point subsample.

For the learned coefficient-manifold oracle, the trained hyperdecoder is frozen and
all exposed-selection `q` values are newly initialized together from three fixed
starts: zero, `Normal(0,0.25^2)` at seed 20260825, and the sign-reversed second
start.  Each start uses Adam (same beta/eps), 10,000 updates, batches of 64 snapshots
and 512 seeded without-replacement points, cosine rate `5e-2 -> 1e-3`, and
`1e-8*mean(raw_q^2)`.  At updates 4000, 6000, 8000, and 10000, full-grid/all-time
metrics are evaluated; a start may stop at the first checkpoint where every
per-N and pooled oracle gate passes.  The lowest final full-field loss among the
three stored starts is the oracle result.  The direct predictor is then evaluated
without selection-time latent optimization.  The optional loss-revision cell may
change only the already listed gradient/weak/rollout weights; it may not change any
optimizer, initialization, sampling, mesh, or oracle rule.
The loss-revision license is a joint all-seed decision made only after completed
seed-11, seed-29, and seed-47 artifacts exist for the same finalist: all three learned
oracles must pass and at least one direct arm must miss its gate by no more than 2x.
No single-seed training cell may claim that license; it records only its local
near-miss condition.

S0 promotes an `R` only if its free projection oracle at every locked selection N
and pooled over those trajectories has trajectory mean `<=2e-4`, trajectory worst
`<=7e-4`, zero unhealthy fits, exact binary boundaries, exact 16-support, and its
mandatory weak-work lower bound passes the paired live same-job 10x point and 8x
clustered-lower-bound speed gates.  If several pass, choose the smallest `R`; if
none pass, Phase 2 stops without training.

The learned coefficient-manifold oracle optimizes per-snapshot `q` with the
transport fixed by the locked moment rule.  It must independently achieve selection
trajectory mean `<=2e-4`, worst `<=7e-4`, zero nonfinite values, and exact boundaries.
The seed-specific direct predictor must achieve mean `<=3e-4`, worst `<=1e-3`, and
direct/oracle mean degradation `<=1.5`.  All three complete retrainings must pass;
seed averaging cannot rescue a seed.  Before model validation, the deployable seed
policy is fixed prospectively as seed 11.  Seeds 29 and 47 are complete robustness
retrainings that must pass every supporting gate, but are not averaged: averaging
would not be parameterized by one `k`-state and would invalidate the locked `M`.

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
attempt flag, accepted factor, step norm, and stopping reason.  The accepted (or
explicitly rejected/no-op) state becomes the previous state for the next step;
rescanning from the uncorrected direct trajectory is forbidden.  End-to-end timing
charges the final full-grid decode of every returned state in addition to predictor,
weak, EQ, guard, and update work.

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
