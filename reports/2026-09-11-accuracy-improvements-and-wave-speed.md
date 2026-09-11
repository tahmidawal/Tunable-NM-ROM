# Accuracy improvements and reflective-wave speed

This report records the controlled accuracy-training and wave-acceleration campaign. The included numbers have passed independent development audits; the overall campaign and publication validation are still incomplete.

Absorbing waves are excluded. Final paper cohorts remain unopened, and the existing experiment branches remain separate. Tables are generated from saved invocation records; no wall-clock ratio crosses jobs or GPUs.

## Heat: targeted training improves accuracy

The spatial bank, latent dimension and online solver are unchanged. Uniform continuation, extra initial-field sampling, and initial-field sampling plus a difficult-example loss are matched training arms. Initial-plus-tail was declared primary before evaluation.

At 1024 intervals, worst current-relative error on all 16 development cases falls from 7.595346% to 4.762515%. GPU query time changes from 11.970928 to 11.623313 ms; the primary is 4.633307× faster than the fastest tested passing CG setting in this job.

The new cohort contains harder cases. Compare old and new heads on the same cohort: the larger original-head error in the expanded table is not a retraction of the earlier table. Host-inclusive time and direct-DST comparisons are separate below.

| Cohort (cases) | Model | Median / worst error % | GPU ms | Host ms | Nonstationary initial / steps | 5% and solve criteria |
| --- | --- | --- | --- | --- | --- | --- |
| Earlier cases (12) | Original head | 3.4725 / 4.5555 | 11.9513 | 26.8356 | 0 / 0 | Pass |
| Earlier cases (12) | Uniform continuation | 2.4042 / 3.5112 | 12.0387 | 27.4988 | 3 / 0 | Fail |
| Earlier cases (12) | Initial emphasis | 2.3918 / 3.3066 | 11.6032 | 27.3826 | 0 / 0 | Pass |
| Earlier cases (12) | Initial + tail (primary) | 2.3536 / 3.5982 | 11.8375 | 27.3882 | 0 / 0 | Pass |
| New development (4) | Original head | 3.6055 / 7.5953 | 11.9716 | 26.5524 | 0 / 0 | Fail |
| New development (4) | Uniform continuation | 2.5656 / 5.3353 | 11.3747 | 27.0480 | 0 / 0 | Fail |
| New development (4) | Initial emphasis | 2.2820 / 4.8674 | 11.1090 | 26.8881 | 0 / 0 | Pass |
| New development (4) | Initial + tail (primary) | 2.4085 / 4.7625 | 11.4850 | 27.1927 | 0 / 0 | Pass |
| Combined (16) | Original head | 3.4725 / 7.5953 | 11.9709 | 26.8356 | 0 / 0 | Fail |
| Combined (16) | Uniform continuation | 2.4042 / 5.3353 | 11.7244 | 27.1555 | 3 / 0 | Fail |
| Combined (16) | Initial emphasis | 2.3918 / 4.8674 | 11.3712 | 27.1035 | 0 / 0 | Pass |
| Combined (16) | Initial + tail (primary) | 2.3536 / 4.7625 | 11.6233 | 27.3221 | 0 / 0 | Pass |

| Intervals per axis | Method | GPU ms | Host ms | Worst error % | GPU / host outliers |
| --- | --- | --- | --- | --- | --- |
| 64 | Original head | 11.942480 | 12.624216 | 7.595341 | 0 / 0 |
| 64 | Initial + tail (primary) | 11.127344 | 11.532207 | 4.762500 | 0 / 0 |
| 64 | CG, tolerance 1e-6 | 5.193358 | 5.540019 | 0.069093 | 0 / 0 |
| 64 | CG, tolerance 1e-2 | 2.301383 | 2.632502 | 1.586031 | 0 / 0 |
| 64 | Direct DST FOM | 0.173806 | 0.489892 | 0.093826 | 16 / 16 |
| 64 | Free linear bank (different ROM) | 0.112989 | 0.445599 | 1.675754 | 1 / 0 |
| 256 | Original head | 11.594060 | 12.821892 | 7.595346 | 0 / 0 |
| 256 | Initial + tail (primary) | 10.847778 | 11.804435 | 4.762515 | 0 / 0 |
| 256 | CG, tolerance 1e-6 | 20.153173 | 21.050890 | 0.035160 | 0 / 0 |
| 256 | CG, tolerance 1e-2 | 6.554194 | 7.389063 | 0.931950 | 0 / 0 |
| 256 | Direct DST FOM | 0.208149 | 1.010324 | 0.005856 | 13 / 10 |
| 256 | Free linear bank (different ROM) | 0.130083 | 0.979336 | 1.675830 | 0 / 2 |
| 1024 | Original head | 11.970928 | 26.835650 | 7.595346 | 0 / 0 |
| 1024 | Initial + tail (primary) | 11.623313 | 27.322126 | 4.762515 | 0 / 0 |
| 1024 | CG, tolerance 1e-6 | 191.350069 | 207.142028 | 0.039098 | 0 / 0 |
| 1024 | CG, tolerance 1e-2 | 53.854375 | 69.577017 | 0.790111 | 0 / 0 |
| 1024 | Direct DST FOM | 0.895605 | 16.483799 | 0.000366 | 0 / 0 |
| 1024 | Free linear bank (different ROM) | 0.528061 | 16.094075 | 1.675830 | 0 / 0 |

The small GPU runtime improvement does not imply a host-inclusive improvement. The direct DST FOM remains faster than the nonlinear ROM. The free linear-bank control uses unrestricted bank coefficients and is a different reduced model.

## Reflective waves: geometry and time-step screens

Shared analytic decoder derivatives and guarded Cholesky solves remove repeated work in latent evolution. At the original step, these preserve the mathematical trajectory to audited floating-point parity. Larger steps are a separate integration change and require refinement checks.

On the opened 64-interval screen, the unchanged-step implementation is 6.929744× faster than the original ROM. Including the retained larger step gives 24.993360×. These are ROM implementation ratios; an accuracy-qualified FOM advantage is not established by this screen.

| Method | Step / CG tolerance | GPU ms | Worst u / v / energy-state error % | Timing outliers | All-state 5% |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.0025 | 4466.021625 | 1.812375 / 4.167162 / 6.223986 | 0 | Fail |
| shared_svd_r | 0.0025 | 2898.490385 | 1.812375 / 4.167162 / 6.223986 | 0 | Fail |
| qr_guard | 0.0025 | 1119.013391 | 1.812375 / 4.167162 / 6.223986 | 0 | Fail |
| chol_guard | 0.0025 | 644.471400 | 1.812375 / 4.167162 / 6.223986 | 0 | Fail |
| chol_guard | 0.005 | 333.705406 | 1.812284 / 4.167207 / 6.223948 | 0 | Fail |
| chol_guard | 0.01 | 178.688326 | 1.810916 / 4.167481 / 6.223099 | 0 | Fail |
| cg_1e-06 | 1e-06 | 110.470355 | 0.199690 / 0.412971 / 0.574495 | 0 | Pass |
| cg_0.01 | 0.01 | 82.725549 | 0.303316 / 0.358723 / 0.485925 | 0 | Pass |
| dst | 0.0 | 3.851750 | 0.000000 / 0.000000 / 0.000000 | 0 | Pass |

Wave errors in this table use fixed initial physical scales. The energy-state error measures the error in displacement gradients and velocity; it is not energy-conservation drift. Current-relative displacement and velocity errors are separately retained in the JSON. DST is the same-grid semidiscrete reference, so its zero discrepancy is not zero continuum error.

The follow-up also tested unrestricted bank evolution and nonlinear output projection. Those methods evolve a larger linear state and are labeled separately from the original nonlinear latent dynamics. Both projected-output variants missed the all-state target; the unrestricted linear control passed on the opened cases.

| Follow-up method | Step / CG tolerance | GPU ms | Worst u / v / energy-state error % | All-state 5% |
| --- | --- | --- | --- | --- |
| baseline | 0.0025 | 4506.736031 | 1.812375 / 4.167162 / 6.223986 | Fail |
| chol_guard | 0.01 | 182.219969 | 1.810916 / 4.167481 / 6.223099 | Fail |
| chol_guard | 0.025 | 86.577930 | 1.804828 / 4.201964 / 6.260739 | Fail |
| linear_bank64 | 0.0 | 1.447262 | 0.782627 / 2.169768 / 3.211301 | Pass |
| projected_l2 | 0.0 | 377.396711 | 1.487063 / 3.771709 / 5.538662 | Fail |
| projected_h1 | 0.0 | 413.628283 | 1.507638 / 3.839339 / 5.566894 | Fail |
| cg_1e-06 | 1e-06 | 110.081740 | 0.199690 / 0.412971 / 0.574495 | Pass |
| cg_0.01 | 0.01 | 82.229586 | 0.303316 / 0.358723 / 0.485925 | Pass |
| cgdt_0.005_tol_1e-06 | 1e-06 | 69.395180 | 0.794879 / 1.548112 / 2.162953 | Pass |
| cgdt_0.005_tol_0.01 | 0.01 | 41.904818 | 1.213408 / 1.418882 / 1.923893 | Pass |
| cgdt_0.01_tol_1e-06 | 1e-06 | 45.235283 | 3.144053 / 5.874126 / 8.153122 | Fail |
| cgdt_0.01_tol_0.01 | 0.01 | 22.983202 | 4.694635 / 5.865361 / 7.941749 | Fail |
| cgdt_0.025_tol_1e-06 | 1e-06 | 39.998864 | 18.437037 / 33.157363 / 45.789823 | Fail |
| cgdt_0.025_tol_0.01 | 0.01 | 16.095903 | 22.146597 / 31.840654 / 43.556564 | Fail |
| dst | 0.0 | 3.737003 | 0.000000 / 0.000000 / 0.000000 | Pass |

Rejected time step: `chol_guard` at 0.025 on `opened_1` has a 1.143733% half-step discrepancy. Its faster timing is diagnostic only.

The CG control was also allowed to use larger time steps; any FOM speed ratio must use a tested setting that passes its physical and solve criteria. The next wave stage trains matched field-only and field/energy/tangent-velocity heads, then confirms frozen settings across meshes and new development cases.

## Work still in progress

Poisson staged bank/head training and Burgers initial-field training plus stationarity-aware solving are not yet accepted in this report. Their completed audits will be added here, including unsuccessful arms. Wave head training and multiresolution confirmation are also outstanding.

## Reproduction and evidence

Heat source `3563072` is the paired GPU job; scientific source and checkpoint hashes are in the linked owner panel. Independent coordinator checks cover each model's worst saved trajectory on every mesh.

- [Heat complete panel](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/accuracy10/analysis/summary.md)
- [Wave geometry audit](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/accel06/audit.json)
- [Wave follow-up audit](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/accel07/audit.json)
- [Normalized values, repetition arrays and source hashes](2026-09-11-accuracy-improvements-and-wave-speed.json)

Run `reports/generate_accuracy_campaign.py` with the repository Python environment to rebuild. All source hashes and generator identity are embedded in the adjacent JSON. These are single-training-seed development studies on the recorded families; they do not establish broad PDE generalization or final paper performance.

## Plain-language glossary

- **Intervals / mesh:** subdivisions along each spatial axis; larger values request more output points.
- **Bank / head / latent:** learned spatial functions / network choosing their coefficients / compressed coordinates solved online.
- **FOM / ROM / NMROM:** full-grid solver / reduced solver / reduced solver constrained to a nonlinear decoder.
- **GPU / host ms:** blocked complete GPU input-to-output query time / the same heat invocation including input and output transfers.
- **Relative error:** error magnitude divided by the specified reference magnitude. Heat uses the current true field at each time; the displayed wave screen uses initial physical scales.
- **Median / worst:** middle case error / largest case error, with each case scored at its worst saved output time. Runtime uses the median of all retained repetitions.
- **CG / DST:** iterative conjugate-gradient solver / direct discrete sine-transform solver. A CG tolerance is its stopping threshold, not its measured field error.
- **Stationarity:** sufficiently small gradient of the reduced solve objective. This does not itself guarantee physical accuracy.
- **Tail emphasis:** training loss that assigns more influence to large reconstruction errors within a training batch.
- **All-state / energy-state:** checking displacement, velocity and their combined energy norm / the norm combining velocity and spatial-gradient error.
- **Guard / parity / refinement:** a numerical check with a more robust fallback / agreement with unchanged equations / agreement after reducing the integration step.
- **Outlier:** heat repetition above one-and-a-half times its case median; wave repetition above twice its panel median. Counts and every duration are retained.
- **Development / sealed final:** cases used in diagnosis and method selection / untouched cases reserved for the paper's later final evaluation.
