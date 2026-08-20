# Burgers pure NM-ROM at 1e-3 and 10x — preregistration

Status: locked before any new diagnostic, training, solver, or timing result was
produced.  The inherited results named below were already public in the canonical
lab log and are not selection data for this search.

## Fixed question and exclusions

The PDE, parameter family, and trajectory are unchanged: scalar viscous
Burgers-2D on `[0,1]^2`, homogeneous Dirichlet values, the existing Gaussian-blob
family, backward Euler with `dt=0.005` for 50 steps, FOM-exact first-order upwind
advection, and centered diffusion.  All arithmetic is f64 with
`JAX_DEFAULT_MATMUL_PRECISION=highest`.

The candidate must be a genuinely pure nonlinear manifold with a parameter count
that does not grow with deployment resolution.  POD plus a nonlinear correction,
a grid-sized learned output basis, a full-grid FOM correction, and an uncharged
guard/fallback are ineligible.  The only permitted online PDE correction is zero
or one bounded weak trust-region/LSPG update.  Every parameter estimator, latent
predictor, chart gate, decoder, weak residual, interpolation, and guard is charged.

The H160x4/group-2 and neighboring width/group brackets are closed.  This search
does not reopen them.  Heat and Poisson are out of scope.  AmgX is not a planned
cell: on this periodic/structured scalar problem the exact separable Helmholtz
preconditioner is the stronger credible control.  AmgX can enter only if the FOM
calibration shows the Helmholtz application itself is at least 70% of eligible
FOM online time and a source-free installed implementation is already available;
otherwise it is excluded without consuming a cell.

## Previously known evidence

The selected N=64,k=16 H160x4/group-2 decoder has test IC-fit mean `1.824e-2`,
oracle inferred-latent trajectory mean `7.688e-3`, full weak-ROM mean
`1.039e-2`, EQ256 mean `1.555e-2`, and EQ512 mean `1.164e-2`.  Its oracle
per-time error falls from `1.824e-2` at the initial condition to `5.615e-3` at
step 50.  Thus representation/boundary alignment is the first active floor;
EQ and rollout are downstream floors.  At N=1024 and outer tolerance `1e-6`,
the audited cubic-history exact-Helmholtz FOM is `390.409 ms`; the guarded K8
FiLM hybrid is `535.287 ms` and accepts a median 4/50 candidates.  These numbers
are brackets, not denominators for the new headline comparison.

## Locked draws and information boundaries

`sample_params` samples arrays sequentially, so every draw count is part of the
split definition and must be stored in every artifact.

* Training/selection/model-validation population: seed `0`, draw count `704`.
  Training indices are `0:512`; architecture/loss selection indices are
  `512:576`; the model-validation indices `576:640` remain unopened until one
  architecture and its three training seeds are locked.  Indices `640:704` are
  reserve-only and may be used solely for a predeclared tie break; if opened,
  they become selection data and can never be called validation.
* Development trajectory population: seed `20260821`, draw count `32`, fixed
  indices `0:8`.  It is used for weak-solver/EQ/timing selection and may never be
  called confirmation.
* FOM tolerance calibration population: seed `20260822`, draw count `32`, fixed
  indices `0:4`.  It is used only to lock the fastest eligible FOM.
* Untouched confirmation population: seed `20261031`, draw count `32`, fixed
  indices `0:8`.  No field, parameter, scalar metric, compilation, or timing from
  this draw may be inspected before the final candidate, decoder seed policy,
  N-specific EQ rules, stopping rule, and FOM denominator are committed.  It is
  opened once in a single confirmation job.
* Training seeds are `11`, `29`, and `47`.  A selected architecture must pass
  every gate for every seed; seed averaging cannot rescue a failure.  The final
  deployable policy is either one seed fixed before model validation or the
  charged arithmetic mean of all three decoders.  Post-validation seed picking
  is forbidden.

Training truth may use the fixed training population at N=64/128/256, with a
recorded resolution mixture and mesh spacing as an input.  Model validation and
confirmation are regenerated at their reported N.  Tight-reference solves use
the same discrete operator and a residual tolerance at least 10x tighter than the
tightest compared solver; solution error against that reference, not residual
alone, calibrates deployment tolerances.

## Candidate class

The only new representation family is a per-point-evaluable transported analytic
coordinate decoder.  A low-dimensional state controls translation, anisotropic
scale, optional shear/skew, and coefficients of fixed Hermite-Gaussian functions
in aligned coordinates.  A hard binary boundary mask enforces exactly the same
Dirichlet grid values as the FOM without multiplying the interior by the old
polynomial boundary envelope.  The functions are analytic and fixed, not a POD
basis; translation/scale makes the image a nonlinear manifold.  A small MLP maps
deployable initial-field features, viscosity, time, and mesh spacing to the
state.  Initial features are computed from at most a fixed 64x64 endpoint sample,
so neither parameter count nor feature cost grows with N.

The three seed-11 concepts, in order, are finite:

1. `HG4`: moment transport plus total Hermite degree 4.
2. `HG5`: moment transport plus total Hermite degree 5.
3. `HG5S`: degree 5 plus bounded learned shear/skew and a two-way
   interior/wall partition-of-unity chart.

No additional degree, width, or group-size sweep is allowed.  Concept 2 is run
only if HG4's aligned representation oracle has mean <=`2e-4` and worst
<=`7e-4` but its trained predictor misses a decoder gate.  Concept 3 is run only
if a wall-distance stratification attributes at least half of HG4/HG5 validation
squared error to the nearest-wall quartile.  At most one finalist is trained at
seeds 29 and 47.  One and only one loss/curriculum revision is allowed, and only
when an already passing representation oracle misses the prediction decoder gate
by no more than 2x.  The revision may add relative field, gradient/Sobolev,
weak-PDE, one-step rollout, and predictor-Jacobian conditioning terms; it may not
change the architecture class.

The latent dimension `k` is the explicit transported state dimension and is
recorded per concept.  It may not exceed 32.  Weak tests use `M >= max(64,4k)`;
all promoted EQ tests use `m=4M` exactly unless a larger predeclared diagnostic
is used to demonstrate quadrature insufficiency.  NNLS weights are fitted only
on decoder-output snapshots, separately at every `(N,M)`.  Candidate points are
target-grid stencil centers; the Burgers weak advection is the exact FOM upwind
operator.  Cold-start parameter estimation is hyper-reduced to the fixed 64x64
sample too.

## Finite cell budget

A cell is one real cluster GPU job directory.  Failed infrastructure attempts
count and remain recorded, but may be resubmitted once without consuming a new
scientific choice if no scientific output was produced.

* Diagnostics: at most 2 cells.  D0 reproduces/decomposes the inherited floor
  into IC/boundary, transport alignment, representation, latent prediction,
  full weak solve, EQ, and accumulation.  D1 measures HG4/HG5 representation
  oracles, wall strata, and a batched decoder/Jacobian cost oracle.
* Architecture/training: at most 6 cells: three seed-11 concepts, seeds 29/47
  for at most one promoted finalist, and at most one justified loss revision.
* Predictor/weak-solver/EQ/FOM calibration: at most 4 cells.  One locks the FOM
  tolerance ladder; one compares direct prediction with zero versus one bounded
  weak update; one brackets `M` at the minimum and next allowed value while
  keeping `m=4M`; one performs the same-GPU cost/degradation lock.  There is no
  second solver/globalization sweep.
* Scaling: at most 2 cells.  One joint N=256/512 development panel and one
  N=1024 pre-confirmation audit using only development cases.
* Untouched confirmation: exactly 1 cell if promoted.

The total scientific budget is therefore at most 15 GPU cells.  Local jobs are
execution smokes only, under one minute, and cannot supply a scientific number.

## Promotion gates and hard stops

Diagnostics first kill a concept if its nondeployable aligned representation
oracle cannot reach decoder mean `2e-4` and worst `7e-4`; prediction/training
cannot repair a representation failure.  A predictor is killed if its latent
oracle-to-predicted degradation is above 1.5 or if its measured N=1024
construction-cost lower bound exceeds one tenth of the calibrated FOM before a
weak update.  A weak-update arm is killed if its oracle update cannot bring mean
error to `7e-4`, or if a single update exceeds the remaining one-tenth-FOM budget.

Before untouched confirmation, every selected training seed must have:

* tight-reference/numerical solution error <=`1e-4`;
* decoder reconstruction mean <=`3e-4`, worst <=`1e-3` on unopened model
  validation once it is released;
* full weak-ROM mean <=`7e-4`, zero failures;
* EQ weak-ROM mean <=`1e-3`, zero failures, and `EQ/full <=1.05`;
* development end-to-end mean <=`1e-3`, worst <=`3e-3`, no divergent or
  censored trajectory;
* a same-GPU point estimate of at least 10x versus the locked fastest eligible
  like-for-like FOM at N=1024 and a trajectory-clustered 95% lower bound at
  least 8x.

Failure of the all-seed decoder gate, the full/EQ gate, the speed oracle, or the
N=1024 development gate stops the search without touching confirmation.  Passing
all gates opens the fixed confirmation draw exactly once.  Success requires, on
all eight confirmation trajectories simultaneously: mean trajectory relative
L2 <=`1e-3`, worst <=`3e-3`, zero divergence/censoring, median end-to-end speedup
>=10x against the fastest calibrated FOM meeting the same accuracy requirement,
and a trajectory-clustered 95% speedup lower bound >=8x.  N=256 and N=512 are
reported as scaling results, not substituted for the N=1024 gate.

The search ends as a rigorous negative result when the finite budget is consumed
or any hard stop makes all remaining preregistered concepts ineligible.  The
thresholds are not weakened and a favorable case is never promoted alone.

## Measurement and artifact contract

Every real job runs in its own directory under
`/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/` on `gpu`, with mandatory
GPU preflight.  Data are regenerated from the locked seeds.  The source commit
and content hashes, SLURM job id, node/GPU model, N, viscosity/family, time grid,
k/M/m, decoder/predictor configuration, reference and solver tolerances, and
f64/highest flags are stored in JSON.

Timed blocks compile first, burn the GPU, then use balanced AB/BA or rotated
orders on the same GPU.  Accuracy, cost, work, residuals, failures, and censoring
come from the same invocation.  All repetitions are persisted.  Reports use
per-trajectory medians, within-trajectory outliers, paired work, and a
trajectory-cluster bootstrap.  Setup-bearing methods report first-use and
amortized setup separately.  Pulled artifacts retain remote/local checksum
manifests and logs; the remote job directory is deleted only after checksums
pass.  Generated scripts own every table, figure, and prose number in the final
dated report.
