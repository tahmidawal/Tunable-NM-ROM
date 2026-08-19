# Burgers-2D FOM-exact hybrid optimisation through N=1024

Status: active experiment. Numbers in this cell are provisional until the final
single-GPU consolidation and independent verification pass.

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
  narrowed confirmation uses at least 21 paired repetitions.
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
