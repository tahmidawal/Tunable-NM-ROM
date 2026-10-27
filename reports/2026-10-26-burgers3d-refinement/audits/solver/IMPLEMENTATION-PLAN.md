# Burgers 3D solver / quadrature diagnostic lane

Status: implementation plan only; numerical implementation waits for the independent campaign design gate.

Base: `93f399faad67f94ea2b56fe25f13003143e64694`. Namespace: `/cluster/tufts/paralab/tawal01/b3d-solver-20261026/`.
Budget: eight GPU-hours including failures, first pilot at most one GPU-hour; no retraining.

## D-S execution

Use the frozen W1024 bank prefixes at widths 768 and 1024, default shell-completed tests,
Gauss 40 cubed, BDF2 with step 0.025, seed 923801 and cases 0, 5, 30, 43. Regenerate physical
inputs from the recorded seed. Preserve the baseline production arithmetic byte-identically.
Reproduce six archived output coefficients and per-step records before interpreting changes.
Save every internal coefficient state, output-state hash, fresh nonlinear certificate and
physical error against the unchanged approved E0 ST lattice references.

First pilot: width 1024, case 30, baseline and separate full-trajectory interventions for
iteration budget, matrix refresh, trust radius and internal stationarity tolerance. Full
trajectory interventions preserve all other settings. They measure their own cost and error.
At failing baseline steps, compare the returned state with fresh-Jacobian adaptive tight and
tighter solutions of the identical step residual, holding the saved coefficient history fixed.
These are fresh-restart controls, not exact replays of the original cached solver carry:
`qn3` resets its cached matrices and damping at the initial replay step. Do not attribute a
restart improvement to refresh frequency. A full-trajectory arm alone tests refresh policy.

The same-history LM records residual norm, fresh stationarity, Jacobian singular values and
condition, accepted/rejected work, clipping and damping. Require independently recomputed
weak residual and Jacobian agreement, tight/tighter agreement and correct BDF2 history.
Persist actual residual/Jacobian evaluation counts where instrumented; mark unavailable
counts explicitly rather than infer them from the iteration cap.

## D-Q execution

Use W1024, default tests, fixed-sweep CN step 0.01 and one sweep. Existing dims gl56/gl64
trajectories cover all validation cases and must be reused with content hashes. The missing
farther successor is gl80. Pilot one case for memory/runtime, then first-eight checks and
archived maximum-successor-discrepancy cases if admitted by the measured budget. Keep
population scope explicit; subset evidence cannot qualify all validation cases. Calculate
own-state advection discrepancies and trajectory distances with identical initialisation.

## Archived E0 decode identity control

Where available, decode archived M2_512 gl32 QN-BDF2, W768 gl40 FS-CN, and W512 with twice the
default tests lat16384 FS-CN output coefficients on cases 0, 5, 30, 43. Use the original bank
and reference identities, verify output times and normalisation, and compare directly with
archived E0 per-case scores. Copy only the necessary unsealed approved reference artifacts;
never inspect or open sealed cohorts. Missing controls remain explicit gaps.

## Audits and execution gates

Independent design acceptance precedes numerical implementation. Independent code/small
control acceptance and exact staged-job/configuration acceptance precede every submission.
Every batch has GPU preflight, f64, highest precision, source/config/bank/reference hashes,
its own directory and a hard limit. Check queue before/after submission, checkpoint each case,
checksum complete collection, then remove only the verified remote duplicate. Burn in and
synchronise every timed block, save repetitions, and separate certification from query cost.

Commit and code-only mirror after each change. No merges or unique-data deletion. The
coordinator handles canonical LAB-LOG and Research Update changes.

## Glossary

- Width: retained frozen bank coefficients.
- Default tests: shell-completed low sine modes from the historical method.
- QN: exact nonlinear gradient with reused Gauss–Newton matrix.
- Fresh restart: a newly initialised solve of a step with frozen coefficient histories.
- CN / BDF2: Crank–Nicolson / second-order backward differentiation formulas.
- E0 ST: approved space/time-refined reference with conditional uncertainty.
- Own-state discrepancy: compare two integration rules at exactly the same coefficients.
