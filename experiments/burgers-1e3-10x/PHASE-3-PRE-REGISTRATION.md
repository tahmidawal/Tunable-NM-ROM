# Burgers 1e-3 / 10x adaptive Phase-3 preregistration

Status: prospectively locked after the audited Phase-2 S0 hard stop and before
any Phase-3 scientific job.  Phase 3 is a bounded numerical implementation
repair, not a new decoder search.  Seed-0 selection data and S0 timing are now
exposed development evidence.  The model-validation cohort (indices 576:640)
and seed-20261031 confirmation draw remain untouched.

## Fixed scientific object and unchanged gates

The only object retained is Phase-2 arm C: `R=48`, `k=24`, `M=96`, `m=384`,
H32, fixed transported domain `[-4.5,4.5]^2`, open-uniform clamped cubic knots,
exact 16-entry tensor-product support, zero outside the aligned domain, and the
exact binary physical boundary mask.  Transport, normalization, coefficient
normalization, hyperdecoder/direct-predictor shapes, weak test modes, target-grid
stencil centers, FOM-exact upwind, exact-Helmholtz weak preconditioner, and all
data/time grids remain exactly as locked in `PHASE-2-PRE-REGISTRATION.md`.
There is no POD, linear corrector, FOM correction/fallback, or grid-growing
learned parameter count.

The scientific gates do not change.  The free-spline oracle must have mean
`<=2e-4` and worst `<=7e-4` on every N=64/128/256 selection cohort and pooled,
all fits finite with relative ridge-normal residual `<=1e-8`, and exact boundary.
The N=1024 mandatory online path includes raw-feature direct prediction,
coefficient generation, 50 weak-objective evaluations, and the final full
51-by-N-squared decode.  Against a healthy and accuracy-eligible live FOM from
the same invocation, median speedup must be `>=10`, the trajectory-clustered
95% speedup lower bound `>=8`, compiled device memory `<=20 GB`, and zero
failures.  No absolute time from another job is a pass/fail threshold.

## Exact coefficient-solver bracket

Both candidates minimize the unchanged ridge objective

`||A c-u||_2^2 + lambda ||c||_2^2`,

where `lambda=1e-12*mean(diag(A^T A))`, on the identical sparse decoder design.

1. `S1`: augmented column-preconditioned LSMR.  Let
   `d_j=sqrt(||A[:,j]||_2^2+lambda)` and solve
   `[A D^-1; sqrt(lambda) D^-1] y=[u;0]`, `c=D^-1 y`, with
   `atol=btol=1e-12`, `conlim=1e12`, and `maxiter=4*48^2=9216`.
2. `S2`: sparse normal-equation SuperLU.  Form
   `H=A^T A+lambda I` in CSC and solve `Hc=A^T u` using fixed COLAMD ordering,
   pivot threshold 1, and equilibration disabled.

P3-D regenerates seed 0/draw 704 and uses only the following already-exposed
selection subsets at times `{0,1,10,20,30,40,50}`: N64
`{512,533,554,575}`, N128 `{512,523,530,543}`, and N256
`{512,517,522,527}`.  It additionally solves all 51 times for N128 case 530,
the already-observed S0 worst trajectory.  Duplicate `(530,time)` fits are
deduplicated.  Time 1 and N128 case 523 are included because the immutable S0
artifact identifies that pair as the exposed global snapshot-worst; case 530 is
the exposed trajectory-worst and also has its maximum snapshot error at time 1.
These targeted rows are diagnostic, never a cohort promotion claim.

Selection truth is the Phase-2 locked legacy fixed-eight rollout only when both
its returned and independently recomputed residual maxima are `<=1e-8`; this is
allowed solely for exposed selection projection.  The immutable S0 JSON, NPZ,
and independent AUDIT are staged by exact path and hash, and P3-D records all
three hashes before reading the old per-snapshot errors.  The paired N=1024 cost
truth instead uses the audited tight reference and independently tighter chain
with the unchanged `<=1e-4` cross-chain gate.

Every diagnostic fit must be finite, have normal residual `<=1e-8`, and retain
the exact boundary.  Relative field error at every sampled snapshot may exceed
the immutable S0 value by at most `1e-5`; N128 case 530 trajectory error may
exceed its S0 value by at most `1e-5` and must also be `<=7e-4`.  This numeric
rule prevents an apparently healthy algebraic solve from degrading the spatial
projection.  If one solver passes, select it.  If both pass, select smaller
worst normal residual; an exact tie selects S1.  Offline fit time is reported
but never selects the method or enters online cost.

## Mathematically identical kernel bracket

All candidates use the same C states, coefficients, fields, weak residual, and
charged work.  K0 is the Phase-2 Cox--de Boor sequential-time control.

1. `K1`: per-span cubic coefficients derived once by the defining Cox--de Boor
   recurrence, evaluated with Horner arithmetic; all 51 time slices remain
   sequential.
2. `K2`: the identical Horner evaluator with fixed time chunks of three for the
   51-slice final decode; the 50 weak evaluations remain sequential.
3. `K3`: one f64 Pallas/Triton block-local grid with spatial block 128 for each
   batched query.  It fuses time/point traversal for the 50 current-stencil,
   50 previous-center, and 51 full-grid decodes while materializing no
   `51*N^2*16` index/weight tensor.  Padded transport-state lanes are zero and
   not read.  If the cluster JAX/Triton backend cannot compile this exact f64
   route, K3 is recorded as infrastructure-infeasible; K1/K2 remain the complete
   noncustom bracket.

Before timing, each route must reproduce K0's support indices exactly and its
f64 weights/fields within relative L2 `<=2e-14` on all unique knots, their two
nextafter neighbors, fixed random states/coefficients, and all four live cost
cases.  Untimed identity probes persist and compare the 50 current five-point
weak-stencil decodes, 50 previous-center decodes, weak residual/rho arrays, and
full N=1024 outputs.  Support is exactly 16 and the boundary is bitwise zero.

P3-D regenerates the same locked seed-20260822 cases 0:4 and tight reference,
then compares K0/K1/K2/K3 and the selected cubic/exact-Helmholtz FOM on one H200.
The FOM must be finite, have zero flags/breakdowns, returned residual within its
recorded tolerance, mean error `<=1e-3`, and worst `<=3e-3`.  Accuracy and cost
come from the same invocation.  After explicit lower/compile recording, each
method gets a first execution, common warmup, GPU burn, and 20 repetitions per
trajectory.  A five-method cyclic/reversed schedule gives every method exactly
four observations in every clock position; if K3 is unavailable, the four-method
schedule gives exactly five observations in every position.  All arrays,
per-trajectory medians/outliers, work, residuals, compile/setup/memory, and the
10,000-resample trajectory-clustered intervals are persisted.  The fastest
kernel passing identity, memory, live-FOM, `>=10x`, and lower-bound `>=8x` is
selected; exact median ties use K1, K2, K3 order.  K0 is a control and cannot be
called a repair.

## Finite cells, promotion, and hard stops

Phase 3 authorizes at most two new scientific cells.  Earlier Phase-1/2 cells
are bookkeeping, not retroactive authorization.

1. `P3-D` (one H200 cell): run the fixed solver subset and paired end-to-end
   kernel/FOM diagnostic above.
2. `P3-F` (at most one H200 cell, conditional): only if P3-D selects both a
   healthy solver and an eligible K1/K2/K3 kernel, rerun the full arm-C exposed
   cohorts N64 512:576, N128 512:544, N256 512:528 over all 51 times with that
   solver.  In the same job, re-confirm the selected N=1024 mandatory kernel
   against a new live eligible FOM with the identical timing protocol.

P3-F passes only if every full-cohort fit has normal residual `<=1e-8`, every
mesh and pooled oracle has mean/worst `<=2e-4` / `<=7e-4`, boundary/support/
finiteness pass, and the repeated paired speed gates pass.  A P3-F pass replaces
only the failed Phase-2 S0 arm-C license and may reopen the already-preregistered
seed-11 C hyperdecoder/autolatent/direct-predictor cell after its artifact-chain
checker is updated.  It does not authorize model validation or confirmation.

If P3-D lacks either a solver or a kernel, P3-F is not run.  If P3-F misses any
gate, Phase 3 and the transported-spline line stop negative: no training,
next-R, next-k, looser identity, solver tolerance, chunk size, block size, or
additional fusion sweep is allowed.  The active numerical/accuracy/speed floors
and best measured same-job Pareto point are then reported.
