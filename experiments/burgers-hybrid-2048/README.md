# Burgers-2D hybrid extension at N=2048

Status: prospectively registered from audited Burgers commit `559583c`; no
N=2048 scientific outcome has been opened at this commit.

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

## Authoritative fresh-seed panel

The untouched cohort is locked before execution to seed **20260827**, canonical
draw 16, indices 0:4.  Only N=2048 is run, at the inherited outer/inner pairs
(1e-6,1e-2), (1e-8,1e-4), and (1e-10,1e-5).  The policy is fixed at
`coarse_n=N`, relaxation 1; neither final timing nor accuracy changes it.

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

## Isolation

Real jobs use only `/cluster/tufts/paralab/tawal01/hybb2048/`, one directory per
job, with `ctol_*` job names.  Results are pulled with checksums and the exact
remote directory is deleted only after validation.  This branch does not edit
`main`, the canonical `LAB-LOG.md`, or the N<=1024 immutable artifacts.
