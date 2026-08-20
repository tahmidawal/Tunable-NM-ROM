# Burgers 1e-3 / 10x adaptive Phase-4 preregistration

Status: prospectively locked after the audited Phase-3 hard stop and before any
Phase-4 implementation or scientific job.  Phase 4 is a finite hierarchical
local-support representation test.  It is not another uniform resolution, latent
dimension, width, knot, or decoder-family sweep.  Phase-1/2/3 selection evidence is
exposed development evidence.  Model-validation cohorts and the seed-20261031
confirmation draw remain untouched.

## Scientific reason and fixed hypothesis

Phase 3 proved two narrow facts about the fixed transported R48 spline.  First, exact
ridge-preserving coefficient solvers remove the numerical-health failure but leave
the targeted N128 draw-530 trajectory at approximately `7.888e-4`, above the locked
`7e-4` gate.  Second, the exact K3 f64 Pallas route has large measured same-job cost
headroom.  Phase 3 did **not** establish that the remaining error is spatially
localized, and Phase 4 makes no such claim.

The only new hypothesis is structural: this is a transported localized family, so a
fixed central refinement in aligned coordinates moves with the solution and may add
useful resolution without refining the global tail or increasing local support
everywhere.  The hypothesis is falsified directly on the complete already-exposed
selection cohorts.  A uniform R64/R80 bracket, another unconstrained k sweep, an
exact-IC lift, and adaptive per-case patch placement are not authorized.

Every candidate remains a pure nonlinear coordinate decoder.  There is no POD basis,
linear decoder path, independent fine-patch state, full-grid FOM correction/fallback,
or learned parameter count that grows with deployment N.  Both coefficient branches
are generated jointly from one small nonlinear latent state.

## Exact hierarchical decoder

The global branch is the Phase-2/3 arm-C branch without modification: five affine
transport variables, aligned domain `[-4.5,4.5]^2`, R48 open-uniform clamped cubic
knots with spacing `9/(48-3)=0.2`, zero outside the aligned domain, and 16-entry
tensor-product support.  Let its decoded field before the physical mask be `S_c`.

The fine branch is an open-uniform clamped cubic tensor-product spline on the fixed
aligned domain `[-2,2]^2`.  Its one-dimensional spacing is `4/(P-3)`.  It is zero
outside that domain and has exactly 16 entries of local tensor-product support.  No
patch center, extent, knot, spacing, or taper is learned or selected per case.

Define the one-dimensional C2 taper using `r=(2-|s|)/0.5` by

```
q(s) = 1,                              |s| <= 1.5,
q(s) = 6*r^5 - 15*r^4 + 10*r^3,       1.5 < |s| < 2,
q(s) = 0,                              |s| >= 2.
```

The tensor window is exactly `w(xi,eta)=q(xi)*q(eta)`.  The hierarchical field is

```
(1-w)*S_c + w*(S_c+S_f) = S_c + w*S_f.
```

Thus the two partition weights sum to one identically, the fine branch is a local
correction rather than a linear bypass, and a query uses exactly 16 coarse entries
plus either zero or 16 fine entries.  The implementation persists a numerical POU
check, exact coarse/fine support indices at knots and nextafter neighbors, and a
maximum-support count of 32.

Evaluation order is locked.  The affine transport is applied first.  Coarse and fine
clamped rows and the tensor window are then evaluated in aligned coordinates.  Their
sum is formed, and the exact binary physical boundary mask is multiplied **last**.
No smoothing or postprocessing follows the binary mask; boundary values must be
bitwise zero.

A width-32 Swish hypernetwork maps the single latent's non-affine component `q` to
the concatenated coarse and fine coefficient grids once per time step.  The direct
predictor remains `7 -> 32 -> 32 -> k`, with all biases and the unchanged train-only
feature standardization/folded-first-layer identity.  The two fixed spatial arms are:

| arm | global / fine controls | fine spacing | k / q | M / m | hyper / predictor parameters |
|---|---|---:|---:|---:|---:|
| H1 | 48x48 + 32x32 | `4/29` | 24 / 19 | 96 / 384 | 111,520 / 2,104 |
| H2 | 48x48 + 48x48 | `4/45` | 24 / 19 | 96 / 384 | 153,760 / 2,104 |

The parameter counts include every bias.  If and only if the conditional latent
license below fires, H3 is the **same P4-D-selected H1 or H2 spatial arm**, completely
retrained at `k=32`, `q=27`, `M=128`, `m=512`.  H3 has 111,776 hypernetwork
parameters for H1 or 154,016 for H2, plus a 2,368-parameter predictor.  The H1
count is `32*27+32 + 32*32+32 + 33*(48^2+32^2)=111,776`; no rounded count may
enter an artifact.  No other k is authorized.

## Comparable free projection oracle

P4-D uses exactly the Phase-2 projection truth and health definition so its results
remain comparable to immutable S0.  Seed-0 draw-count 704 legacy fixed-eight-Newton
truth is allowed only for the locked training/exposed-selection projection cohorts,
and only when both the returned and independently recomputed residual maxima are
finite and at most `1e-8`.  P4-D does not replace that truth with a tight chain.
Accepted model validation, scaling, N1024, and confirmation continue to require the
tight and independently tighter chains defined below.

For a transported snapshot, let `Phi_c` be the exact sparse global design and
`Phi_f` the exact sparse fine design before the window.  After the physical boundary
mask `B`, the combined sparse design is

```
A = diag(B) [ Phi_c, diag(w) Phi_f ].
```

The single joint coefficient vector minimizes

```
||A c-u||_2^2 + lambda ||c||_2^2,
lambda = 1e-12 * max(mean(diag(A^T A)), 1e-300).
```

The solver is the Phase-3 S2 ridge-preserving sparse normal-equation SuperLU: form
`A^T A + lambda I` in CSC, use fixed COLAMD ordering, pivot threshold one, and
disable equilibration.  A fit is healthy only if coefficients, predictions, and
diagnostics are finite; the boundary is bitwise zero; and

```
||A^T(Ac-u)+lambda*c||_2 / max(||A^T u||_2,1e-300) <= 1e-8.
```

Every fit is retained.  Offline coefficient-fit time is reported but does not enter
online cost or select a candidate.

P4-D fits every time of every exposed selection trajectory: N64 indices `512:576`,
N128 `512:544`, and N256 `512:528`, all 51 times.  H1/H2 must pass separately at
each N and pooled: trajectory mean `<=2e-4`, worst `<=7e-4`, zero unhealthy fits,
exact boundary/POU/support, and finite outputs.  The smallest arm passing these and
the cost gates below is selected; an exact tie selects H1.

Two prospective no-regression checks are additional integrity guards, not substitute
promotion claims:

1. On every comparable full-selection `(N,case,time)` snapshot, hierarchical relative
   field error may exceed the immutable S0 arm-C snapshot error by at most `1e-5`.
2. On the exact fixed 128-fit Phase-3 subset, relative field error may exceed the
   immutable P3 S2 per-snapshot value by at most `1e-5`.  N128 draw 530's complete
   trajectory must also be `<=7e-4` and may exceed its immutable P3 S2 trajectory by
   at most `1e-5`.

S0 JSON/NPZ/AUDIT and P3-D JSON/NPZ/AUDIT are staged and bound by exact paths and
hashes.  The P3 subset guard is never described as full-cohort evidence.

## Analytic K3 cost and memory screen

The immutable P3 K3 control used 51-by-1024-squared output, 855,638,016 local
coefficient contributions, and 454,923,832 compiled eligibility bytes.  H2 keeps the
same output and adds at most one 16-support branch, so coefficient-contribution work
is bounded by two times the K3 control.  Window, branch, and index arithmetic give a
conservative complete operation envelope of `2.15` times K3.  H2's additional final
hypernetwork layer work is exactly `51*32*2304=3,760,128` multiply-accumulates, less
than 0.44% of the control's support contributions.

H2 adds 940,032 bytes for the 51 fine coefficient grids, 608,256 bytes for the fine
output-layer weights/biases, and at most one copy of K3's 947,472-byte temporary.
Allowing another 5 MB for k32/M128/m512 work arrays gives a static planning bound
below 0.47 GB.  The scientific compiled-device-memory gate remains 20 GB.

Applying the 2.15 operation factor to P3's measured K3 point and interval would give
planning-only margins above 10x/8x.  Those cross-job extrapolations never license a
candidate.  P4-D must compare both hierarchical arms with a fresh live eligible FOM
on the same H200 invocation.

## P4-D cost protocol and correction eligibility

P4-D regenerates seed-20260822 draw-count 32 indices `0:4` at N1024.  Cost truth is
the audited cubic-history/exact-Helmholtz chain at outer/inner `1e-12/1e-7`, checked
against `3e-13/3e-8`.  Both chains must be finite, have zero flags/breakdowns, meet
their returned-residual tolerances, and differ by at most `1e-4`.  The live selected
FOM uses outer/inner `3e-3/1e-1` and must be healthy with trajectory mean `<=1e-3`
and worst `<=3e-3` from the same invocation.

For H1 and H2, P4-D compiles and times two complete routes:

- `mandatory`: hyperreduced cold recovery, all raw features, direct prediction,
  both coefficient-grid generations, exactly 50 weak-objective/rho evaluations,
  and the final 51 full-grid hierarchical outputs;
- `max-one`: the mandatory route plus one M-by-k Jacobian and all fixed trial
  residuals at every step, with corrected state carried to the next step.

The five timed methods are FOM, H1-mandatory, H1-max-one, H2-mandatory, and
H2-max-one.  After explicit lower/compile/first-use recording, common warmup, and a
GPU burn, 20 repetitions per trajectory use a cyclic/reversed schedule with exactly
four observations in every clock position.  All repetition arrays, per-trajectory
medians and within-trajectory outliers, work/rho/Jacobian/trial records, first-use,
compiled memory, and 10,000-resample trajectory-clustered intervals are persisted.

A P4-D arm passes cost only when **both** mandatory and max-one have zero failures,
finite canonical work, compiled memory `<=20 GB`, paired median speedup `>=10`, and
clustered 95% lower bound `>=8` against the live eligible FOM.  Mandatory speed is a
structural lower bound and can never by itself license a deployable claim.

The final online rule remains unchanged.  Every step evaluates the exact-upwind weak
objective and
`rho=||r_w||_2/max(||Phi^T u_previous||_2,1e-12)`.  Zero attempt is allowed only for
finite `rho<=1e-3`; otherwise exactly one normalized-state radius-0.25, LM-`1e-6`
attempt with factors `[1,0.5,0.25,0]` is charged.  The accepted or explicitly
rejected corrected state becomes the next previous state.  If any accepted
development/validation/confirmation trajectory requires an attempt, the actual
charged corrected-rollout policy from that same solver invocation—not a mandatory
surrogate—must itself pass median `>=10x` and clustered lower bound `>=8x`.

## Locked data splits, training, and validation

The Phase-2 splits remain exact and every listed trajectory contributes all 51 times:

- train truth: seed 0, N64 `0:512`, N128 `0:128`, N256 `0:64`;
- exposed selection: seed 0, N64 `512:576`, N128 `512:544`, N256 `512:528`;
- untouched model validation: seed 0, N64 `576:640`, N128 `576:608`, N256
  `576:592`;
- joint full-weak/EQ model validation: N256 `576:592`;
- EQ decoder-output fit pool: seed-0 N256 train indices `0:64`, with 256
  `(trajectory,time)` pairs fixed by seed 20260824 and persisted;
- N256/N512 scaling and N1024 development: seed 20260821, draw-count 32, indices
  `0:8`;
- untouched confirmation: seed 20261031, draw-count 32, indices `0:8`.

Model-validation data generation is forbidden until three independently audited
passing complete retrainings exist.  Confirmation is opened exactly once and only
after every earlier gate passes.

The Phase-2 optimizer, sampling, normalization, state map, and loss schedules are
unchanged.  Every candidate uses all mixed-N/all-51 training snapshots.  Manifold
training is 30,000 AdamW updates with cosine `1e-3 -> 1e-5`, weight decay `1e-6`,
clip 1.0, batches of 32 snapshots and 512 seeded points, and independently initialized
hypernetwork/autolatents.  Direct-predictor training is a fresh 20,000-update AdamW
run with cosine `5e-4 -> 5e-6`; it freezes the hypernetwork and uses the unchanged
field plus `0.1` normalized-state loss.  Every full seed retrains hypernetwork,
autolatents, and predictor from scratch.  Seed 11 is the prospective deployable
policy; seeds 29 and 47 are independent robustness retrainings and may not be
averaged.

The learned coefficient-manifold oracle freezes a trained hypernetwork and optimizes
all exposed-selection q states from the three Phase-2 starts, using the same
10,000-step/checkpoint/full-grid protocol and pooled mean snapshot-relative-L2-squared
start selection.  It must pass at every N and pooled: trajectory mean `<=2e-4`,
worst `<=7e-4`, zero nonfinite values, and exact boundary.

The k32/H3 cell is licensed only when the selected H1/H2 seed-11 k24 learned oracle
fails at least one original gate but satisfies **all** of these exact near-miss gates
at every N and pooled: mean `<=4e-4`, worst `<=1.4e-3`, zero nonfinite values, and
exact boundary.  Direct-predictor failure alone never licenses k32.  H3 completely
restarts the same selected spatial arm at seed 11; it does not reuse k24 weights or
latents.  A worse learned-oracle miss is a hard stop.

The seed-specific direct predictor must have mean `<=3e-4`, worst `<=1e-3`, and
direct/oracle mean degradation `<=1.5` at every N and pooled.  All three full seeds
must pass every learned-oracle and direct gate; averaging cannot rescue a seed.

One conditional joint all-seed loss revision is allowed only after all three learned
oracles pass and at least one direct arm satisfies every exact 2x near-miss bound at
every N and pooled: mean `<=6e-4`, worst `<=2e-3`, and direct/oracle degradation
`<=3.0`, with finite exact-boundary output.  That single cell completely retrains all
three seeds and may add only the Phase-2 fixed gradient/weak-PDE/one-step-rollout
weights `(0.1,0.1,0.1)`; architecture, optimizer, sampling, and other weights do not
change.

## Weak, EQ, headline, and supporting gates

Weak tests remain the deterministic lowest-eigenvalue sine modes.  Full weak
residuals use every interior target-grid stencil center in lexicographic order.
Full and EQ advection use the exact discrete FOM first-order upwind convention.
Hyperreduction includes cold recovery and every weak call.  NNLS EQ weights are fit
only on decoder-output snapshots and exact-upwind images, and refit whenever N or M
changes.  Base `M/m` is the selected candidate's table value.

The conditional next-M EQ cell is licensed exactly when every seed's full weak result
passes, base EQ has zero failures, and base EQ misses mean `<=1e-3` or
`EQ/full<=1.05`, while every seed still has EQ mean `<=1.2e-3` and degradation
`<=1.20`.  The only retry is `M_next=2*M_base`, `m_next=4*M_next`, with fresh
decoder-output NNLS weights.  Thus k24 retries at M192/m768 and k32 at M256/m1024.
A worse miss or any failure hard-stops.

Supporting gates are unchanged: independently tighter-reference numerical error
`<=1e-4`; every seed's untouched-validation reconstruction mean `<=3e-4`, worst
`<=1e-3`; full weak-ROM mean `<=7e-4`; EQ mean `<=1e-3`; `EQ/full<=1.05`; zero
failure/censoring; and development/confirmation mean `<=1e-3`, worst `<=3e-3`.
N1024 additionally requires median same-GPU speedup `>=10x` and trajectory-clustered
95% lower bound `>=8x` against the fastest healthy like-for-like FOM satisfying the
same accuracy requirement.  Accuracy, cost, work, and failures come from the same
solver invocation.  N256 and N512 scaling are reported.  No threshold is weakened.

Accepted model-validation, scaling, N1024, and confirmation truth uses the audited
cubic/exact-Helmholtz outer/inner `1e-12/1e-7` chain and independent
`3e-13/3e-8` chain.  Both must be finite with zero flags/breakdowns, meet their
returned residual tolerances, and differ by at most `1e-4` in solution.

## Finite cell cap and hard stops

Phase 4 authorizes at most 11 scientific cells.  Earlier phases are bookkeeping,
not retroactive authorization.

1. P4-D hierarchical free-oracle plus paired mandatory/max-one cost diagnostic: 1.
2. Selected spatial arm, complete seed-11 k24 training: 1.
3. Conditional same-spatial-arm seed-11 k32 retraining: at most 1.
4. Locked finalist complete seed-29 and seed-47 retrainings: 2.
5. Conditional joint all-seed loss revision: at most 1.
6. Joint all-seed model-validation/full-weak/base-EQ: 1.
7. Conditional next-M EQ: at most 1.
8. N256/N512 scaling: 1.
9. N1024 same-GPU development/preconfirmation: 1.
10. Exactly one untouched confirmation: 1.

If k32 is used, the maximum is `1+1+1+2+1+1+1+1+1+1=11`.  If k24 passes, the
maximum is 10.  Infrastructure-only zero-output attempts may be resubmitted once and
remain recorded.  Local smokes never count as scientific evidence.

Failure of any promotion gate stops every downstream cell.  P4-D failure stops
training.  A worse-than-2x seed-11 k24 manifold miss stops rather than opening k32.
A failed seed, loss revision, weak/EQ gate, scaling gate, corrected-rollout cost gate,
or N1024 gate prevents confirmation.  No extra patch, window, P, R, k, M, hypernetwork
width, solver tolerance, block size, or fusion route is authorized.  The final report
will state success only if every headline and supporting gate passes on untouched
confirmation; otherwise it will report the active floor and best measured Pareto
point as a rigorous negative.

Every scientific artifact stores commit/source hashes, every input artifact hash,
seed/draw/index/N/viscosity/time grid, architecture/k/M/m, full optimizer state/config,
GPU/job/backend, f64/highest, reference health, per-case accuracy/work/residual/
Jacobian/trial/acceptance, all timing repetitions/orders/outliers, setup/first use,
memory, failures/censoring, and clustered intervals.  Cluster staging, preflight,
checksummed pull, exact remote cleanup, generated report, and canonical LAB-LOG updates
follow `AGENTS.md`.
