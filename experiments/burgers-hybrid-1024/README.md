# Burgers-2D FOM-exact hybrid optimisation through N=1024

Status: final untouched-seed confirmation complete. The selected-cohort panel
remains explicitly provisional; headline results come only from
`runs/confirm2/out/confirm2.json` and its generated summary.

This cell redesigns the Burgers warm-start path after the audited FiLM NM-ROM
rollout lost every wall-clock comparison through `N=256`. The delivered field is
always finished by the same full-order backward-Euler/Newton/BiCGStab solver to a
named tolerance. The fixed problem is the original testbed: `dt=0.005`, 50 steps,
f64, the exact upwind/centred finite-difference operator.

## Pre-registered gates

The audited `N=256`, `tau_FOM=1e-6` row spent 479.569 ms constructing the ROM
guess and saved only 15.149 ms in the FOM stage. Therefore:

1. A replacement must cost less than 15 ms per 50-step rollout at `N=256` if it
   preserves only the old FOM-stage saving.
2. It must beat **linear extrapolation**, not merely the previous-state start, in
   total time and BiCGStab work.
3. Cost and accuracy come from the same invocation; timing repetitions are saved.
4. Cross-resolution wall clock comes from one sequential job on one GPU only.
5. Every published arm must finish every step to the named Newton tolerance with
   finite arithmetic. A failed cheap solve is not a speedup.

## Sequence

1. `bh_oracle.py`: controlled warm-start quality curves relative to both free
   baselines. It interpolates between linear extrapolation and the exact next
   FOM state to measure how much guess error must be removed, and how much FOM
   time that removal can buy. Oracle rows are explicitly non-deployable.
2. Cheap deployable predictors: a charged history-only **classical control**,
   followed by a learned cached low-rank correction of linear extrapolation.
   The history-line arm is not called an NM-ROM. Candidate selection is based
   on total time and BiCGStab work, not field L2 alone.
3. Audited consolidation through `N={32,64,128,256,512,1024}` at
   `tau_FOM={1e-6,1e-8,1e-10}`.

## Completed gates

The first two timing gates are diagnostic rather than headline results: they
used a four-trajectory seed-1 population and the inherited over-tight inner
linear tolerance.  They established the direction of travel, but the final
confirmation will use the canonical draw-16 cohort, a dynamic oracle, and the
locked calibrated FOM configuration.

* The oracle at `N=256`, `tau_FOM=1e-6` found that removing 75--90% of the
  linear-extrapolation error can save only roughly 2--4 ms per 50-step chain.
  Seven paired repetitions were too noisy to order nearby qualities, so the
  narrowed oracle confirmation uses at least 21 paired repetitions.
* Zero-cost quadratic and cubic history predictors reduced guess-field error
  but increased BiCGStab work.  They are retained as strong charged classical
  controls, not called learned or NM-ROM methods.
* The disjoint seed-2 calibration chose inner tolerances `1e-4`, `1e-4`, and
  `1e-6` for outer tolerances `1e-6`, `1e-8`, and `1e-10`, respectively.  The
  selected rows had maximum returned outer residuals `9.997496e-7`,
  `9.988694e-9`, and `9.345625e-11`, with no solver-health failure.  This gate
  selected settings only; confirmation timings grade the returned timed solve
  instead of making a separate grading invocation.
* A train/validation-only correction-spectrum gate rejected the cached global
  linear basis.  At `N=64`, a rank-32 POD basis left 52.85% of validation
  correction norm, already above the 25% oracle target before coefficient
  prediction.  Both generator-parameter and deployable initial-field-moment
  coefficient maps left about 99.5% and had zero validation cases meeting the
  target.  Further work therefore requires a shift-aware nonlinear manifold;
  raising rank or tuning the same RFF map is not justified.
* A follow-up transported-basis gate aligned every correction by centroid and
  width measured from the currently available field.  This improved the
  rank-32 projection substantially, but still left 29.05% of global validation
  correction norm (the target was at most 25%); only 56.25% of individual
  snapshots met the target.  The best deployable time-sliced coefficient map
  left 62.03% globally and only 31.25% of snapshots met the target.  Because
  this already needs rank 32 plus a spatial warp, it cannot plausibly fit the
  oracle's few-millisecond budget.  The transported linear-basis surrogate is
  therefore stopped too, rather than being relabelled as a successful NM-ROM.
* The corrected live-history oracle used the locked Helmholtz FOM, canonical
  seed-1 draw-16/select-0:4 cohort, and 21 repetitions per case.  Removing 50%,
  75%, 90%, and 99% of dynamic extrapolation error saved paired medians 1.23,
  4.55, 14.25, and 17.49 ms at N=256.  This supersedes the preliminary noisy
  few-millisecond curve: a genuinely q90 predictor has room, but the deployable
  learned correction retained 62% rather than 10% of the error and cannot
  access that budget.
* A semi-implicit physics predictor (exact discrete upwind advection plus a
  Dirichlet diffusion solve by DST-I) reduced field-guess error in local smoke,
  but failed the solver-work objective in the charged cluster gate.  With the
  calibrated Helmholtz FOM at `tau=1e-6`, IMEX Euler lost 1.46 ms at N=64 and
  2.52 ms at N=256 to linear extrapolation; AB2 was statistically tied at N=64
  and lost 0.92 ms at N=256.  In contrast, the zero-construction cubic history
  control saved 6.26 ms and 8.93 ms, with paired 95% median intervals excluding
  zero.  This is direct evidence that field L2 alone is the wrong warm-start
  objective: the apparently more accurate physics guess did not minimise total
  finishing work.
* Inexact-Newton calibration showed that the inherited inner tolerance `1e-10`
  was substantial over-solving.  A Dirichlet diffusion/Helmholtz left
  preconditioner reduced BiCGStab work by roughly an order of magnitude in its
  own calibration job.  Because wall clock cannot be selected across jobs, a
  narrowed joint none-versus-Helmholtz lock job is required before any timing
  claim or final solver setting is accepted.  That joint job selected the
  Helmholtz preconditioner at every outer tolerance: inner tolerances `1e-2`,
  `1e-4`, and `1e-5` for outer `1e-6`, `1e-8`, and `1e-10`.  On the selected
  cubic calibration arm, N=256 required only 76.5, 144.75, and 223.5 mean
  BiCGStab iterations over all 50 steps; maximum returned outer residuals were
  9.65e-7, 9.82e-9, and 9.33e-11.  These settings are now locked identically
  across final arms.
* A bounded zero-cost history blend sweep did not improve the cubic endpoint.
  The closest competitor, 75% of the third backward difference, was tied with
  cubic: cubic's paired median saving versus that blend was 0.286 ms at N=64
  (95% bootstrap interval crossing zero) and 0.016 ms at N=256 (also crossing
  zero).  Cubic had the slightly lower pooled mesh median and is the simpler
  pre-existing endpoint, so it remains locked; no fourth-order extrapolator is
  justified by this plateau.

All exact values and complete arrays for these gates are in `runs/`.  No gate
number is promoted without its JSON, batch log, and matching remote/local
checksums.

`runs/film_smoke` is a superseded N=32 harness smoke from before the fixed-64
decode/prolongation optimization.  It validates execution only and is excluded
from every result table; the final FiLM job uses the optimized coarse path.

Before that final job, the fixed-coarse FiLM control also gates its live latent
initialisation on development data.  The existing method already carries the
previous accepted latent into the next weak LM solve and performs an immediate
weak-objective tolerance test.  The bounded alternative starts steps after the
first from `2*z_n-z_(n-1)`, retaining the same trust-region solver and early
exit.  `LATENT_HISTORY_MODES=previous,extrapolation` runs both variants against
the same cubic control in one rotated timing block; the locked winner alone
enters confirmation.  This gate changes neither the trained checkpoint nor the
FOM tolerance.

The held-out latent-history gate selected linear latent extrapolation: all 36
paired end-to-end records were faster than the previous-latent start, while the
finished FOM Newton/BiCGStab work was identical pairwise.  The unlimited weak
LM remained far slower than cubic, so it is not the final optimized FiLM arm.
The last bounded gate caps the weak solve at one or two Jacobian evaluations
per step.  Each resulting candidate is compared online against the live cubic
candidate using both exact full FOM residuals; Newton receives the lower-
residual candidate and the charged guard acceptance fraction is retained.
This makes a low-budget NM-ROM candidate safe without hiding a degraded guess
behind a separate untimed fallback.

The bounded gate selected the two-Jacobian arm for the final genuine hybrid.
The one-Jacobian arm performs only the initial weak evaluation, never passes a
candidate through the guard on the held-out cohort, and is therefore retained
only as the zero-update safety-floor diagnostic.  Two Jacobians permit one
actual LM update and pass a median one of 50 candidate steps through the exact
guard.  That extremely low acceptance is the mechanism-level reason the FiLM
work cannot amortize its construction cost; it is not hidden by reporting only
the finished field.

`runs/film_latent_gate_h100_redundant` is the already-queued duplicate of the
latent-history mechanism gate.  It completed after the A100 selection result;
it is retained for provenance but its wall clock is not pooled with or compared
against the A100 job.

The EQ refit is also bounded for the 1024 mesh.  A literal full-candidate
matrix would contain 8192 rows by about one million columns and is not a viable
offline algorithm.  The scalable refit uses a deterministic uniform tensor
pool of at most 4096 **target-grid** nodes, while retaining exact full-grid
projection targets from decoder-output snapshots and exact five-point FOM
upwind stencils at every candidate; NNLS still selects `m=256=4M` nodes anew at
each N.  This is grid EQ, not random or off-grid strong collocation.  Candidate
strategy, pool size, fit diagnostics, indices, and weights are persisted.

The six meshes and three locked outer/inner tolerance pairs run in one final
process and on one GPU.  Each mesh fits EQ only once, then all three tolerance
blocks reuse the identical FiLM constructor.  Every tolerance block separately
burns the GPU and rotates all arms, so reported paired differences never cross
jobs or solver settings.

Job 2663384 uses seed 1 indices 0:4 and is **provisional only**: although those
cases were disjoint from the latent-history and budget selection cases, the
corrected oracle had already inspected them.  The untouched confirmation uses
seed 20260819, draws 16 cases once, and fixes indices 0:4 without inspection.
Its uncertainty calculation first takes the paired median over seven repetitions
within each trajectory and then bootstraps the four trajectory clusters; it
does not treat 28 correlated timings as IID.  Tukey outliers are likewise
counted within trajectory.  Finally, deployable initializer moments gather at
most a fixed 64x64 endpoint sample from `u0` (and charge it), while the cold LM
fit remains on the 256 EQ nodes.  Thus no cold-start component scans N squared
values online at the large meshes.

## Final audit contract

Every confirmation arm is invoked in a joint post-burn block with rotated
order.  The timed call returns the field, residuals, Newton counts, BiCGStab
counts, flags, and breakdowns; grading happens only after that return.  JSON
retains per-case/per-repetition arrays, paired deltas and confidence intervals,
medians, outlier counts, and every selected trajectory.  The final reference
gate is at least ten times tighter than the tightest reported FOM tolerance,
the oracle is constructed from the live dynamic extrapolator, and the counting
solver is checked against both the testbed operator and JAX BiCGStab at every
reported mesh/tolerance.  The same locked inner solver is used by every
warm-start arm.

## Provenance and cluster layout

All jobs live under `/cluster/tufts/paralab/tawal01/hybb1024/`, one directory per
job, on the `gpu` partition. Batch scripts assert `jax_backend=gpu`, set
`JAX_DEFAULT_MATMUL_PRECISION=highest`, and regenerate trajectories from the
recorded seed. Pulled runs retain logs and remote/local checksums.

## Final untouched-seed result

Job 2664725 completed the pre-registered six-mesh, three-tolerance panel in one
process on one A100. It used the untouched seed-20260819 canonical draw-16
cohort, fixed indices 0:4, and the locked Helmholtz/inexact-Newton settings for
all arms. The checksummed source JSON, logs, and per-trajectory/per-repetition
arrays live in `runs/confirm2/`; the cluster job directory was removed only
after both checksum manifests passed locally.

`bh_summarize_confirm2.py` validates the complete condition grid, solver health,
precision, provenance, and repetition shape before generating
`runs/confirm2/out/final_summary.json` and
`runs/confirm2/out/final_table.md`. The generated result is unambiguous: the
cubic classical predictor beats linear throughout the full panel, while the
two-Jacobian genuine weak FiLM NM-ROM with its charged exact-residual guard and
live cubic fallback loses to cubic throughout. The NM-ROM arm is therefore a
mechanism-audited negative control, not the recommended Burgers warm start.
