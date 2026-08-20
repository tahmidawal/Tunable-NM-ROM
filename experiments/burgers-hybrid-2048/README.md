# Burgers-2D hybrid extension at N=2048

Status: final and independently audited. Raw primary results are under
`runs/final3/`; all primary and learned numbers are generated from run JSONs in
`FINAL-RESULT.generated.md` by `bh_2048_render_final.py`.

## Scope and classifications

This is a narrow resolution extension, not a new architecture search.  The
authoritative comparison is between:

- the optimized live cubic-history FOM with exact Dirichlet Helmholtz
  preconditioning; and
- the selected **classical**, nonlearned, non-NMROM warm start: live cubic plus
  one target-grid exact-upwind residual and one target-grid exact Helmholtz
  inverse at every one of the 50 time steps (`coarse_n=N`, relaxation 1).

Both arms use the same backward-Euler operator, calibrated outer/inner
tolerance pairs, Newton stopping test, and counting BiCGStab implementation.
The candidate's Newton/BiCGStab counters cover only the finishing solve.  Its
wall time additionally charges exactly 50 full-grid residual evaluations and
50 exact Helmholtz inverses per trajectory; those counters are returned and
persisted explicitly.

## N=2048 execution/memory smoke

Before fresh data are touched, one non-scientific H200 execution smoke uses
development seed 20260818, canonical draw 16, only index 12, N=2048, outer/inner
tolerances 1e-6/1e-2, one AB and reburned BA block, and a 0.1-second immediate
burn before each pair.  It must compile both complete dependent invocations,
return same-invocation accuracy/work/residual telemetry, pass solver/reference
equivalence, and record device peak/limit bytes.  It licenses the final panel
only with zero flags/breakdowns, all returned residuals within tolerance,
reference residual at most 1e-11, finite fields, and peak device allocation at
most 90% of the H200 limit.  Smoke timings are execution diagnostics and are
excluded from scientific claims.

The excluded smoke is complete as Slurm job `2670117` from preregistration
commit `3f95afa2d178dbe6ef93483e425680cdf48a500e`.  Its H200, f64/highest,
solver-health, reference, returned-residual, equivalence, raw-grid, and
remote/local checksum gates all pass.  Peak allocation was below both frozen
memory thresholds, so the authoritative classical panel is licensed and the
bounded learned sensitivity is mandatory.  The raw smoke bundle and generated
audit are retained under `runs/smoke1/`; its execution-only timing is not a
scientific result.

## Development reference failures and diagnostic

The original cohort was locked before execution to seed **20260827**, canonical
draw 16, indices 0:4.  Only N=2048 is run, at the inherited outer/inner pairs
(1e-6,1e-2), (1e-8,1e-4), and (1e-10,1e-5).  The policy is fixed at
`coarse_n=N`, relaxation 1; neither final timing nor accuracy changes it.

The first fresh-seed attempt, job `2672535`, stopped before compilation of an
online arm or creation of any timing row: the inherited fixed-eight-Newton
testbed truth rollout had worst returned reference residual `1.215e-2` at
N=2048 and failed the unchanged `1e-11` gate.  It is retained under
`runs/final1/` as an excluded pre-science reference-generation failure.  The
prospectively reviewed 25-iteration offline-only repair was run as job
`2674703`, but it reproduced the same residual exactly and likewise produced
zero timing rows.  Both failures are excluded from speed and accuracy claims.
Increasing a fixed iteration count is therefore retracted as a repair.

Before any further reference change, a bounded diagnostic-only H200 job is
locked to the same seed 20260827, draw 16, indices 0:4, and N=2048.  It first
persists every trajectory's 50 final step residuals from the exact public-JAX
fixed-25 path.  The trajectory with the largest nonfinite-or-finite maximum
(lowest index breaks a tie), and then its first nonfinite or largest-residual
step, are selected deterministically.  Only that single step is rerun with
every Newton correction recording the achieved true linear residual
`||J du+r||/||r||`, finite/update status, and a line-for-line instrumented JAX
0.10.2 BiCGStab exit code; the public `info` value is documented as always
`None`.  The selected trajectory's complete 50-step chain is compared under
three frozen tolerance-stopped counting routes:
unpreconditioned (outer `1e-12`, inner `1e-10`), exact Helmholtz (outer
`1e-12`, inner `1e-8`), and an independent stricter exact-Helmholtz ladder
(outer `1e-13`, inner `1e-10`).  It persists the complete residual, iteration,
flag, field-norm, and pairwise field-difference histories plus each route's
terminal field and checksum.  The diagnostic records no method timing, cannot
promote a replacement, and licenses no final rerun without a separate
post-diagnostic preregistration and audit.

The instrumented and public JAX updates must agree to relative `1e-12`.  A
future replacement preregistration may be proposed only if both Helmholtz
routes have zero flags/breakdowns, satisfy the unchanged `1e-11` nonlinear
reference gate at every step, and agree to `1e-10` in both maximum per-step and
whole-trajectory relative field difference.  Passing those diagnostic gates
does not itself replace the reference or license timing.

Diagnostic job `2676167` completed its phase-1 public rollout but is excluded
as a diagnostic-instrumentation failure.  The phase-1 result localizes the
failure to trajectory 3: steps 1--40 finish below `1e-12`, while steps 41--50
remain at `1.2154764671954222e-2`.  The detailed replay then failed the fixed
`1e-12` reproduction gate before any counting route ran.  It had replayed one
unbatched vector solve, whereas the public rollout applies an exact batch-one
`jax.vmap` transformation; this can change JAX BiCGStab's breakdown path at
N=2048.  The job contains no method timing and licenses neither a replacement
reference nor a primary rerun.  Its Slurm exit status is also excluded as a
completion signal: an `A && B` shell list allowed failed producer `A` to skip
checker `B` and continue to the final marker under Bash `set -e`.

A single bounded `diagnostic2` repair is prospectively locked before execution.
It changes only the detailed replay topology: a jitted wrapper invokes
`jax.vmap(detailed_step)` on `(u_prev[None], nu[None])`, waits for the batched
device result, and removes axis zero only afterward.  Snapshot indexing,
phase-1 cohort, deterministic trajectory/step selection, three counting
routes, tolerances, gates, and the prohibition on timing/self-promotion remain
unchanged.  The producer and checker are separate shell commands; a nonempty
producer-check artifact and `PRODUCER-CHECK-DONE` marker are mandatory.  If
this exact batch-one replay still misses either reproduction gate, the
diagnostic hard-stops with no further topology relaxation.

The repaired diagnostic is complete as H200 job `2677878` and is a **final
negative reference diagnostic**.  The exact batch-one replay reproduces the
public-JAX field and residual bit-for-bit, identifying 25 nonfinite public
updates at the frozen step.  The Helmholtz candidate is healthy at maximum
returned residual `9.429357347688513e-13` with zero flags/breakdowns.  It also
agrees with the strict ladder to maximum per-step/whole-trajectory relative
field differences `1.3674113431649328e-12` / `6.971405683089061e-13`.
Nevertheless, the strict ladder reaches the 25-Newton cap and returns a
nonzero flag at all 50 steps, while the public/instrumented update-match value
is null because every active public update is nonfinite.  The frozen
update-match and two-tight-route-health gates therefore fail.  This diagnostic
licenses no reference replacement and no timing result.

## Authoritative fresh-seed panel

The only licensed primary follow-up is a prospectively frozen, informed
reference repair on a wholly untouched cohort: seed **20260830**, canonical
draw 16, indices 0:4.  Seed 20260827 is permanently development-only and will
never produce a timing claim.  Before compiling or timing an online arm, each
new trajectory must pass two independently executed exact-Helmholtz counting
routes: candidate outer/inner tolerances `(1e-12,1e-8)` and inner-strict
`(1e-12,1e-10)`, both with at most 25 Newton iterations.  The shared outer
tolerance is above the observed N=2048 arithmetic floor; independence is in
the tenfold stricter inner solve.  Both routes must be finite, have zero
flags/breakdowns, and return actual outer residual at most `1e-11` at every
step.  Their maximum per-step and whole-trajectory relative field differences
must both be at most `1e-10`.  Every route's actual residual and work history
is persisted.  Candidate fields become the reference only if all gates pass
on all four trajectories; otherwise the job stops before any timing row.

If the reference gate passes, the primary methods, outer/inner tolerance
panel, dynamic correction, AB/BA ordering, six blocks, twelve repetitions per
arm/trajectory, three-second immediate burns, estimators, and accuracy/work
telemetry are exactly the originally registered design.  The fixed public-JAX
rollout and its equivalence path are not used as truth.  Final producer and
audit commands run on separate fail-closed shell lines, and both
`FINAL-AUDIT-DONE` and `ALL-DONE` markers are required.

Seed 20260829 is also excluded before final execution.  During implementation,
an N=32-only local structural smoke accidentally used its first four parameter
draws; it exercised only the new reference/equivalence plumbing and no N=2048
field, primary method, or timing.  It reported reference pass, maximum actual
outer residual `9.500501846359613e-13`, maximum route field difference
`4.781917940670456e-14`, row-equivalence maximum residual
`9.938508215690312e-7`, and maximum field difference
`3.5494211868205905e-6`.  The cohort was discarded immediately.  A repository
and run/log-artifact search found no prior occurrence of seed 20260830 before
it was locked here.

For every tolerance and trajectory, the process runs six exact AB then
immediately reburned BA blocks.  This gives twelve repetitions per arm and
trajectory, with a three-second GPU burn immediately before every pair.  Every
timed solver call returns the field, finish Newton/BiCGStab counts,
breakdowns/flags, returned residual, and explicit correction-work counts;
accuracy is graded from that same return.  All raw repetitions and burn records
are retained.  Timing is reduced to paired medians within trajectory and then a
20,000-draw bootstrap over the four trajectory clusters.  Tukey 1.5-IQR
outliers are counted within trajectory and retained.  A speedup is called
supported only when the trajectory-cluster interval for paired time saving is
strictly positive.  All three cells are reported regardless of sign.

The timing scope is warmed, compiled online trajectory latency.  It includes
the complete warm-start construction and FOM finish, but excludes compilation,
module/checkpoint loading, test/reference generation, and first-query latency.
All three tolerance blocks run sequentially in one job on one H200, so no wall
clock is compared across jobs.  f64/x64, highest matmul precision, exact-upwind
residuals, source hashes, staged manifest, Slurm/GPU provenance, solver
equivalence, raw repetition arrays, and remote/local checksums are mandatory.

## Bounded learned sensitivity

The authoritative paired block above is never expanded or reordered for a
learned arm.  The learned sensitivity decision is fixed before any primary
outcome is opened: it **shall run** if and only if the execution smoke passes
every health gate and its recorded peak allocation is at most 75% of the H200
device limit.  A smoke peak above 75%, missing memory telemetry, or any smoke
health failure is a hard stop for the learned sensitivity and cannot be
reconsidered after inspecting the primary result.  If licensed, it runs only
after a separate implementation audit and scientific-preregistration commit,
and it remains independent of the primary timing values.  It reuses the
already audited genuine K=8 weak FiLM NM-ROM unchanged (M=64,m=256, exact
upwind, two latent Jacobians per step, fixed N=64 decode/prolongation, exact
residual guard) on separate seed
20260828.  Any such job must be separately committed before data are opened and
must use exact balanced AB/BA pairs against both cubic and the classical dynamic
arm with an immediate burn before every pair.  It is a learned sensitivity,
not a candidate selected by the N=2048 primary data.

The smoke licensed this block before the primary result was opened.  Its
prospective execution is now fixed to seed **20260828**, canonical draw 16,
indices 0:4, N=2048, and the same three tolerance/linear-tolerance pairs as the
primary panel.  For each tolerance and trajectory it runs six blocks of the
four-pair schedule cubic/FiLM, FiLM/cubic, dynamic/FiLM, FiLM/dynamic, with a
three-second immediate burn before every pair.  Thus each comparison has
twelve balanced repetitions per arm and trajectory.  The FiLM output is the
unchanged K=8 extrapolated-latent arm with at most two latent Jacobians per
step, fixed-N=64 charged decode/prolongation, and an exact full-grid residual
guard.  All construction, guard, finish, accuracy, work, and returned-residual
telemetry comes from the same dependent invocation.  EQ is refit offline at
N=2048 on decoder-output snapshots with M=64 and m=256; offline fit, compile,
checkpoint loading, and reference generation are excluded from warmed online
latency.  Both comparisons are reported regardless of sign, using the same
within-trajectory paired median and four-trajectory cluster bootstrap as the
primary panel.

If the learned block's inherited fixed-eight reference attempt also stops
before any timing row, its only licensed rerun uses the same 25-iteration
offline reference/equivalence repair.  This is not a change to the weak
NM-ROM, its guard, or either online FOM finish.

## Isolation

Real jobs use only `/cluster/tufts/paralab/tawal01/hybb2048/`, one directory per
job, with `ctol_*` job names.  Results are pulled with checksums and the exact
remote directory is deleted only after validation.  This branch does not edit
`main`, the canonical `LAB-LOG.md`, or the N<=1024 immutable artifacts.
