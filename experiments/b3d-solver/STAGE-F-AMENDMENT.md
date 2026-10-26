# Conditional time refinement and same-allocation validation frontier

Status: proposed amendment awaiting independent design review; no implementation of this
extension is released by the earlier D-S/D-Q pilot approvals. Total lane cap remains eight
GPU-hours, inclusive of failures and earlier jobs. Sealed/test cohorts remain closed.

## D-T: isolate time error after nonlinear and quadrature convergence

Run only after the audited D-S and D-Q diagnostics. At W1024/default tests use BDF2 with
steps 0.025, 0.0125 and 0.00625, rules gl40/gl48/gl56, and cases 0, 5, 30, 43 from seed923801.
Reuse every exact existing tight trajectory; never rerun it just to recreate a table.
The same-family verified adaptive LM uses original weak residuals/Jacobians, a fresh matrix
each iteration, the frozen trust radius, stationarity tolerances 1e-7 and 1e-8, and the
existing finite/budget/stationarity checks. Persist all states and work. The existing
recording p1core implementation will be copied byte-identically and content-pinned.

For every step and rule retain fresh certificates; a time-error interpretation requires
tight/tighter agreement at most2.5e-5, trajectory quadrature differences at most2.5e-4,
own-state advection rho at most0.116, and an independently checked successor. If an ingredient
fails, report that failure and do not attribute the remaining difference to time alone.
Compare E0 error at each output, including which output attains the maximum, so repairing
a late nonlinear step cannot be presented as lowering an unchanged early maximum.

This is a fixed bounded diagnostic set, not a new training/tuning grid. One hour maximum
for its first job; review actual elapsed/memory before extending. Stop cell on any integrity
failure, nonfinite state, exhausted verified solve or failed reference identity.

## Freeze the final validation manifest before running it

Original qualified anchors:

- M2_512, gl32, QN-BDF2, step0.05/3, refresh4.
- W512, shell-completed tests at nominal8r, lat16384, FS-CN, step0.01, one sweep.
- W768, default tests, gl40, FS-CN, step0.01, one sweep.

Conditionally include W1024 gl56 FS-CN only if the joined historical full64 gl56/gl64 and
new first-eight gl80 evidence meets the unchanged Q1/Q2 predicates. Include at most two
new repaired/time-refined configurations after the D-S/D-T results audit. Choose the
least-cost diagnostic policy passing every difficult-case tight-distance/validity check,
with pooled stationarity at most1%; do not select trust clipping by anecdote. A cost tie
within5% uses refresh-every-step, then extra-budget, then strict, then trust as a frozen
priority. Time candidates are the two finer predefined steps if their ingredient checks
pass and an E0 improvement is resolved relative to reference uncertainty; retain rejected
candidates with their exact failure rather than removing them from the inventory.

POD's qualified frozen winner comes from the separate POD lane and must run in this same
allocation before comparative cost claims. Give it the same output schedule and input
requirements. If qualification does not close, show its diagnostic status and forbid a
qualified-POD superiority claim.

## Frozen full solvers; no broad retuning

Use the already tested spectral selections N16/step0.025, N24/step0.01 and N32/step0.005,
plus central finite differences L64/CN-R at step0.0125/tolerance-level1 and
step0.005/tolerance-level0. `inputs/frozen-fom.json` preserves their original complete
case records, calibration, source/job and content hash. Copy the original kernels
byte-identically. Reject incomplete/nonfinite/uncalibrated finite-difference runs.
These predefined levels span the relevant accuracy tiers; matching uses the fastest
tested matching arm in this finite set and retains the historical broader ladder's
scope. A missing match is an explicit outcome. Only a specific missing accuracy tier
can justify a separately reviewed extension, never a new unrestricted grid.

## Population, input and cost contract

All64 unsealed validation cases, seed923801, approved E0 ST finest references, output times
0:0.05:0.25 on the fixed63-cubed interior lattice associated with the65-node mesh.
Retain every per-case eligibility/uncertainty mask. This is further development/validation
on an exposed cohort; it is not new held-out or sealed confirmation.

The common query input is the analytic family's known physical-parameter descriptor.
Report native-input preparation separately and include it in a second end-to-end timer:
ROM/POD prepare the same65-node input field then project; finite differences prepare their
native sampled field; spectral prepares its continuous sine coefficients. Also retain
prepared-native-input timings under that explicit narrower label. No comparison silently
omits a competitor's initialisation. Cohort generation, frozen bank/basis training and
reusable operator tables are offline; record their setup/memory separately.

Within a single H200 job, warm every subject, randomise A-B-A paired blocks with the M2
anchor, burn in before every invocation, and save all repetitions. Use at least three
repetitions per case/subject; reject performance inference when an anchor bracket drifts
outside[1/1.1,1.1]. Persist fields or reusable coefficients, output hashes and the E0 score
for each timed invocation. Cost and accuracy therefore refer to the same solve. Record
initialisation, evolution, decoding, full native query, parameter-to-output time, work
counts and memory; separately timed components are indicative and need not sum to a fused
query. Certification/diagnostic work is excluded from the deployable-query timer.

Keep full64 errors and full64 timing observations joined by case/invocation id. Median,
worst and outlier counts all use the identical eligible population. Apply uncertainty
interval matching before any speed ratio. No claim of an optimal unrestricted FOM or
improvement over an unqualified historical ROM/POD configuration follows from a match.

## Remaining scaling scope

Historical fixed-reduced-setting mesh evidence remains labelled with its original test-mode,
initialisation and fixed-output contracts. Any repaired finalist receives a bounded
64/128/256-interval check if the measured cost fits the remaining cap. Separate fixed-output
and growing-full-field output. A deferred scaling replay remains an open item and is never
filled using cross-job wall-clock ratios.

## Glossary

- r / M / m: retained coefficients, smooth weak tests, integration points.
- QN / FS: reused-Jacobian iteration / fixed number of nonlinear sweeps.
- CN-R / BDF2: Crank-Nicolson with damped startup / second-order backward differentiation.
- E0 ST: approved space/time-refined reference with conditional uncertainty.
- Native input: the sampled field or sine coefficients the solver consumes.
- A-B-A: reference, candidate, reference timing bracket on one device.
- Qualified: every applicable nonlinear, quadrature, validity and reference check passes.
- Exposed cohort: cases already inspected during development; no new confirmation claim.
