# Independent Codex review — smoke06 results and prospective A7

Read-only reviewer `numerical_recovery`, 2026-10-28. Verdict: **partial API and empty-channel
controls accepted; full 2D G1 remains failed. A7 ready for the exact bounded 3D diagnostic
after a dated override and fresh remote verification.** No reference, training, ROM or
performance claim is admitted. The reviewer edited no files and ran no native FE job.

## Provenance audit

The reviewer recomputed all archive chunk hashes and the concatenated archive SHA
`29b507647409347756caee62e505c1ef5826bb148f12f92bf52bc63597701738`, verified every
one of the 794 extracted-file checksums, and compared all 16 staged inputs byte-for-byte
with source `9d431a0ccdf3f9477370e032a95d3f0c1a0b4323`. All matched. Runtime records
confirm the pinned NGSolve, required JAX GPU backend, x64 and highest precision.
Job 6567998 ran CPU finite elements under its H200 allocation; accounting matches
the saved Slurm evidence and central ledger.

Controlling evidence is in the 2D lane at
`experiments/ns-obstacle-2d/artifacts/g1-smoke06/`, particularly `RETENTION.json`,
`results/dependency_runtime.json`, `results/preflight.json`, `results/channel/result.json`,
`results/channel/newton_history.json` and `results/channel/jacobian_checks.json`.

## Numerical audit

The API preflight passes the recorded scalar UMFPACK residual, identity-gradient,
divergence and per-edge traction checks. The wrapped divergence actually returns a
finite float64 tuple of shape `(1,)`, confirming strict extraction for this observed case.

The reviewer independently recomputed velocity, pressure, gradient and divergence errors
from all saved volume points and wall force, flux and trace from all boundary points.
Results agree with saved JSON at roundoff. Every NPZ array is finite float64; all volume
weights are positive. Boundary divergence-theorem integration agrees with twice the
fluid area. The recorded channel residual, geometry, analytic velocity/pressure, sampled
trace, wall force, mass and divergence gates pass; physical momentum and energy balance
diagnostics are near roundoff. The sampled trace is not a continuous supremum certificate.

All three perturbed-state Jacobian checks pass with resolved signals. **The channel took
zero Newton updates:** its initial Stokes solution already solves Poiseuille flow. This
validates the mixed linear/control path and derivative check, not nonlinear obstacle
convergence. No mesh, polynomial-order, tolerance or domain convergence series is admitted.

The manufactured case failed before solve at the raw mapped-normal cosine threshold.
The actual cosine, vector, point and obstacle mesh were not saved. Reversed normals are
therefore a hypothesis rather than an observed diagnosis. DFG did not run. Diagnose the
unchanged geometry and integration path before any correction.

Both thread-setting warnings are real. The channel elapsed time includes diagnostic/export
work and effective concurrency is unmeasured; retain it only as a diagnostic duration.

## Prospective A7 review

All nine local 3D staging inputs match candidate
`1c2fd8fa141ded0d1dcdc15c850a5a8682572b00` and its receipt. The source independently
repeats cube/API/linear checks, including a nonsymmetric affine gradient fixture. CSG
pipe/sphere normal checks precede case solves; an unverified convention can fail there
and stop execution. Existing DOF, element, memory and outer wall-time guards are intact.
Only geometry, empty-pipe and manufactured cases are selected. Thread count is set before
the TaskManager, avoiding the known 2D warning.

No demonstrated shared blocker requires this diagnostic to await circle repair. The dated
release must explicitly supersede the old full-2D-G1 prerequisite for this exact candidate,
retain the 2D failed state, bind its manifest and verify the remote bytes. The reviewer
checked local staging only. The coordinator's independent remote receipt is stored in
`state/3D-SMOKE06-ROOT-RECHECK-2026-10-28.txt`; operational release is recorded separately
in `OPERATIONAL-A7-3D-G1-RELEASE.md`. All later scientific gates remain closed.

## Terms

G1: bounded runtime controls. CSG: constructive solid geometry. DOF: finite-element unknown.
NPZ: saved numerical arrays. Stokes: the viscous/pressure problem without nonlinear advection.
