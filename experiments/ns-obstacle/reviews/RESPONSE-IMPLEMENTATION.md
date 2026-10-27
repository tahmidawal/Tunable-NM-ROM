# Response to independent Claude numerical-implementation consultation

Status: consultation returned, bounded implementation repairs in progress before any
numerical job. Raw exact-source packet and response are retained. No scientific threshold
has been changed and no reference/training gate has passed.

Accepted: fail-fast tiny sparse-factor and mapped VOL/BND API tests before costly cases;
valid exact-pressure-probe checks with unresolved failures; sorted boundary ordering; retain
partial summaries on failure; independent normal/sign checks; manufactured force equality
and analytic/reaction diagnostics; explicit non-gating DFG source deviations; signal/deadline
handling; label 3D physical balances as raw uncorrected continuum diagnostics while retaining
the corrected gradient-form identities; show tiny-system DOF/memory screen early.

Not automatically adopted:
- The proposed relaxation of linear relative residual from1e−10 to1e−6 is speculative.
  Keep the registered strict gate and record a failure if it is triggered. Any later change
  requires saved evidence and a dated numerical amendment, not consultant reassurance.
- Current pinned PyPI dependency metadata names ngsolve-openblas, not MKL. Do not add
  unneeded MKL/intel-openmp packages or change library paths without observed evidence.
  The tiny inverse preflight will establish actual backend availability.
- Suggested ElementId/jacobi.NumPy API substitutions are uncertain; verify documented
  behavior with the cheap mapped-point probe rather than changing to another guessed API.
- Pressure nudges may be named diagnostics if later useful, but never substitute for exact
  benchmark probe evaluation or enter a passed source-agreement claim unnoticed.
- Requested thread settings do not prove effective parallelism. First smoke carries no
  comparative cost claim; actual concurrency remains a later performance gate.

The first packages remain staged but unsubmitted. Revised sources receive new manifests,
independent focused diff review and fresh attempt directories before launch. 3D remains
conditional on reviewed2Dsmoke and its memory gate. No iterative/GPU full-solver substitution.
