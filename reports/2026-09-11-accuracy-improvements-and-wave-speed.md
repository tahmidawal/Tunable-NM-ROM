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

## Poisson: staged training at unchanged capacity did not solve the problem

The learned spatial bank is trained first with free training coefficients, then the nonlinear head is fitted in the full field metric, followed by joint refinement. Ordinary joint continuation is matched to the staged optimizer time. All procedures keep the original bank size, latent dimension, source-input contract and exact weak solver.

| Development cohort (cases) | Intervals | Model | Worst physical error % | Bank projection error % | GPU ms | GPU outliers | Invalid solves | 5% target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all (42) | 1024 | Original | 7.280248 | 6.489096 | 2.613091 | 1 | 0 | Fail |
| all (42) | 1024 | Matched joint | 7.571199 | 6.936344 | 2.545561 | 0 | 0 | Fail |
| all (42) | 1024 | Staged bank + head | 7.757472 | 6.981179 | 2.597750 | 0 | 0 | Fail |
| all (42) | 1024 | Staged + joint | 7.660328 | 6.966506 | 2.565345 | 0 | 0 | Fail |
| existing_development (30) | 1024 | Original | 6.801556 | 5.500597 | 2.638046 | 1 | 0 | Fail |
| existing_development (30) | 1024 | Matched joint | 6.605687 | 5.647629 | 2.531182 | 0 | 0 | Fail |
| existing_development (30) | 1024 | Staged bank + head | 7.353073 | 6.177772 | 2.622753 | 0 | 0 | Fail |
| existing_development (30) | 1024 | Staged + joint | 6.842717 | 5.948326 | 2.613186 | 0 | 0 | Fail |
| new_development (12) | 1024 | Original | 7.280248 | 6.489096 | 2.572861 | 0 | 0 | Fail |
| new_development (12) | 1024 | Matched joint | 7.571199 | 6.936344 | 2.565197 | 0 | 0 | Fail |
| new_development (12) | 1024 | Staged bank + head | 7.757472 | 6.981179 | 2.535608 | 0 | 0 | Fail |
| new_development (12) | 1024 | Staged + joint | 7.660328 | 6.966506 | 2.498484 | 0 | 0 | Fail |

Matched joint continuation slightly improves the earlier cases but worsens the later development cases. Neither staged endpoint improves the expanded-cohort worst error. The nonlinear solves are stationary; the remaining error is not resolved by simply allowing more online iterations.

Bank projection uses the full same-grid field norm; the online physical error uses a refined-grid reference. These columns are related diagnostics, not an additive error decomposition. A larger learned-bank experiment is now separate from this unsuccessful fixed-capacity comparison. Normalized training-snapshot POD projections motivate that experiment but do not prove a worst-case lower bound for every possible bank.

## Burgers: stricter solves work; initial-field retraining regressed

The fixed-bank comparison crosses the original and refined heads with the earlier stall-based optimizer and explicit stationarity stopping. The refined head trains on regenerated initial fields in the fine-grid physical norm, while replay preserves original decoded outputs at old training codes. Replay targets are not new PDE trajectories.

The combined cohort has 6 development cases: 4 previously opened and 2 introduced in this comparison. Every method has 3 timed repetitions per case and mesh.

| Intervals | Method | Median / worst fixed-initial error % | Worst initial error % | GPU ms | Host ms | Stationary queries | GPU outliers | Physical and numerical criteria |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | frozen_accepted | 4.174735 / 10.854648 | 3.956363 | 32.615563 | 34.503586 | 0/18 | 0 | Fail |
| 64 | frozen_stationary | 4.175852 / 10.856979 | 3.956378 | 48.257689 | 49.823298 | 18/18 | 0 | Fail |
| 64 | trained_accepted | 4.661121 / 11.038040 | 4.868049 | 29.028991 | 30.542819 | 0/18 | 0 | Fail |
| 64 | trained_stationary | 4.661048 / 11.045568 | 4.868142 | 42.288515 | 43.959457 | 18/18 | 0 | Fail |
| 64 | fft_loose | 3.272685 / 10.391989 | 0.000000 | 13.101714 | 14.625311 | — | 0 | Fail |
| 64 | fft_tight | 4.025783 / 10.989131 | 0.000000 | 64.784372 | 66.214041 | — | 0 | Fail |
| 256 | frozen_accepted | 2.196491 / 4.550700 | 2.562871 | 34.461494 | 36.046911 | 0/18 | 0 | Fail |
| 256 | frozen_stationary | 2.195423 / 4.554611 | 2.562872 | 47.846650 | 50.113181 | 18/18 | 0 | Pass |
| 256 | trained_accepted | 3.868826 / 5.788215 | 4.544334 | 29.376801 | 31.230879 | 0/18 | 0 | Fail |
| 256 | trained_stationary | 3.856137 / 5.794182 | 4.544334 | 43.434900 | 45.245599 | 18/18 | 0 | Fail |
| 256 | fft_loose | 1.002811 / 2.473687 | 0.000000 | 15.454815 | 17.543245 | — | 0 | Pass |
| 256 | fft_tight | 1.361525 / 4.026515 | 0.000000 | 89.796797 | 91.786073 | — | 0 | Pass |
| 1024 | frozen_accepted | 2.053097 / 3.907620 | 3.856220 | 37.532220 | 50.446109 | 0/18 | 0 | Fail |
| 1024 | frozen_stationary | 2.053097 / 3.884680 | 3.856220 | 49.119185 | 66.591242 | 18/18 | 0 | Pass |
| 1024 | trained_accepted | 4.218738 / 5.469473 | 4.592753 | 32.584959 | 49.033042 | 0/18 | 0 | Fail |
| 1024 | trained_stationary | 4.218156 / 5.459554 | 4.592752 | 45.627233 | 62.048669 | 18/18 | 0 | Fail |
| 1024 | fft_loose | 1.284695 / 2.389937 | 0.000000 | 59.825965 | 76.933701 | — | 0 | Pass |
| 1024 | fft_tight | 1.209609 / 2.141611 | 0.000000 | 443.099669 | 460.223361 | — | 0 | Pass |

On the largest mesh, strict stopping changes worst fixed-initial error from 3.907620% to 3.884680%, with GPU time 37.532220 → 49.119185 ms. It is 1.217976× faster than the same-job loose iterative FOM, and both meet the declared criteria. The trained strict head reaches 5.459554% error and is rejected as an accuracy improvement.

The old stopping contract accepted small-progress exits; those saved results are preserved. The added gradient audit reveals that they were not stationary under the stricter criterion. Stricter solving makes convergence explicit but barely changes the physical error. The retraining improves its training initial fields while worsening development trajectories, motivating a separate broader-training-coverage comparison.

An earlier attempt failed an overly tight diagnostic equality between two floating-point gradient evaluations. Its training checkpoint and references were preserved with audited lineage, then reused unchanged in this replay. All reported costs and predictions were regenerated together in the replay job. The stationarity threshold was unchanged; charged, posthoc and independent CPU gradients must all pass, and no threshold classification disagreement occurred.

Burgers errors are divided by the reference initial-field norm and maximized over saved times. Its physical criterion includes the empirical reference-refinement allowance; the coarse mesh remains unqualified. The FFT-labeled FOM is an iterative Newton–BiCGStab solve using an FFT preconditioner, not a direct nonlinear solution.

## Reflective waves: geometry and time-step screens

These screens use the same 2 opened reflective Dirichlet cases at 64 intervals, with 3 timed repetitions per case. They are development screens, not a large independent test set.

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

Wave displacement error is divided by the initial displacement norm. Velocity and energy-state errors are divided by $\sqrt{2E_0}$, where $E_0$ is initial physical energy; initial velocity can be zero, and its norm is not the denominator. The energy-state error measures displacement-gradient and velocity error, not energy-conservation drift. Current-relative displacement and velocity errors are separately retained in the JSON and exclude the recorded zero/vanishing reference times. DST is the same-grid semidiscrete reference, so its zero discrepancy is not zero continuum error.

The follow-up also tested unrestricted bank evolution and nonlinear output projection. Those methods evolve a larger linear state and are labeled separately from the original nonlinear latent dynamics. Both projected-output variants missed the all-state target; the unrestricted linear control passed on the opened cases.

| Follow-up method | Step / CG tolerance | GPU ms | Worst u / v / energy-state error % | GPU outliers | All-state 5% |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.0025 | 4506.736031 | 1.812375 / 4.167162 / 6.223986 | 0 | Fail |
| chol_guard | 0.01 | 182.219969 | 1.810916 / 4.167481 / 6.223099 | 0 | Fail |
| chol_guard | 0.025 | 86.577930 | 1.804828 / 4.201964 / 6.260739 | 0 | Fail |
| linear_bank64 | 0.0 | 1.447262 | 0.782627 / 2.169768 / 3.211301 | 0 | Pass |
| projected_l2 | 0.0 | 377.396711 | 1.487063 / 3.771709 / 5.538662 | 0 | Fail |
| projected_h1 | 0.0 | 413.628283 | 1.507638 / 3.839339 / 5.566894 | 0 | Fail |
| cg_1e-06 | 1e-06 | 110.081740 | 0.199690 / 0.412971 / 0.574495 | 0 | Pass |
| cg_0.01 | 0.01 | 82.229586 | 0.303316 / 0.358723 / 0.485925 | 0 | Pass |
| cgdt_0.005_tol_1e-06 | 1e-06 | 69.395180 | 0.794879 / 1.548112 / 2.162953 | 0 | Pass |
| cgdt_0.005_tol_0.01 | 0.01 | 41.904818 | 1.213408 / 1.418882 / 1.923893 | 0 | Pass |
| cgdt_0.01_tol_1e-06 | 1e-06 | 45.235283 | 3.144053 / 5.874126 / 8.153122 | 0 | Fail |
| cgdt_0.01_tol_0.01 | 0.01 | 22.983202 | 4.694635 / 5.865361 / 7.941749 | 0 | Fail |
| cgdt_0.025_tol_1e-06 | 1e-06 | 39.998864 | 18.437037 / 33.157363 / 45.789823 | 0 | Fail |
| cgdt_0.025_tol_0.01 | 0.01 | 16.095903 | 22.146597 / 31.840654 / 43.556564 | 0 | Fail |
| dst | 0.0 | 3.737003 | 0.000000 / 0.000000 / 0.000000 | 0 | Pass |

Rejected time step: `chol_guard` at 0.025 on `opened_1` has a 1.143733% half-step discrepancy. Its faster timing is diagnostic only.

The CG control was also allowed to use larger time steps; any FOM speed ratio must use a tested setting that passes its physical and solve criteria.

## Reflective waves: training the original latent dimension

Two matched training arms use a fixed encoder with consistent latent velocities. One trains displacement reconstruction; the other adds displacement-energy and tangent-velocity losses. Their comparison isolates those extra losses. The original head used independently optimized snapshot codes, so comparison with that checkpoint also changes the training-code procedure.

| Training-screen method | Step / CG tolerance | GPU ms | Worst u / v / energy-state error % | GPU outliers | All-state 5% |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.0025 | 4519.760249 | 1.812375 / 4.167162 / 6.223986 | 0 | Fail |
| chol_guard | 0.01 | 182.911717 | 1.810916 / 4.167481 / 6.223099 | 0 | Fail |
| trained_field | 0.01 | 184.240493 | 1.901449 / 4.587520 / 6.263939 | 0 | Fail |
| trained_phase | 0.01 | 182.754962 | 1.766325 / 4.134009 / 6.103806 | 0 | Fail |
| cg_1e-06 | 1e-06 | 109.972200 | 0.199690 / 0.412971 / 0.574495 | 0 | Pass |
| cg_0.01 | 0.01 | 82.678175 | 0.303316 / 0.358723 / 0.485925 | 0 | Pass |
| cgdt_0.005_tol_1e-06 | 1e-06 | 69.673613 | 0.794879 / 1.548112 / 2.162953 | 0 | Pass |
| cgdt_0.005_tol_0.01 | 0.01 | 42.025412 | 1.213408 / 1.418882 / 1.923893 | 0 | Pass |
| cgdt_0.01_tol_1e-06 | 1e-06 | 44.376265 | 3.144053 / 5.874126 / 8.153122 | 0 | Fail |
| dst | 0.0 | 3.881786 | 0.000000 / 0.000000 / 0.000000 | 0 | Pass |

The combined training improves the three displayed initial-scaled errors compared with both the original and new field-only heads. It does not improve every error normalization: current-relative displacement changes from 3.640762% to 3.681009%, while current-relative velocity improves from 7.712200% to 6.417503%. All nonlinear heads in this screen still miss the all-state target. Initial fitting, numerical rank and half-step checks pass. These separately trained endpoints are not online changes to one trained network.

## Reflective waves: correction coordinates help more than retraining a larger head

Increasing the latent dimension and retraining with the same phase-aware loss worsened both runtime and accuracy on the opened screen. The next arm preserves the trained nonlinear head and adds fixed linear correction directions from training data. This enlarges the nonlinear manifold while containing the original one exactly; it is a separately prepared decoder.

The enriched decoder is $h_{40}(z,y)=h_{32}(z)+B_8y$. The extra directions are frozen training principal components. All initial coordinates are fitted from the supplied fields, and the full enlarged state evolves through nonlinear latent dynamics. The initial-guess library uses appended principal-component coordinates without a residual correction; that limitation is frozen for confirmation.

Larger retrained head: all comparisons below use this screen's own paired timings.

| Method | Step / CG tolerance | GPU ms | Worst u / v / energy-state error % | GPU outliers | All-state 5% |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.0025 | 4496.899827 | 1.812375 / 4.167162 / 6.223986 | 0 | Fail |
| chol_guard | 0.01 | 182.031151 | 1.810916 / 4.167481 / 6.223099 | 0 | Fail |
| trained_phase | 0.01 | 181.880917 | 1.766325 / 4.134009 / 6.103806 | 0 | Fail |
| trained_phase40 | 0.01 | 202.700617 | 2.109531 / 4.252749 / 6.167913 | 0 | Fail |
| cg_1e-06 | 1e-06 | 110.558403 | 0.199690 / 0.412971 / 0.574495 | 0 | Pass |
| cg_0.01 | 0.01 | 82.596072 | 0.303316 / 0.358723 / 0.485925 | 0 | Pass |
| cgdt_0.005_tol_1e-06 | 1e-06 | 69.765648 | 0.794879 / 1.548112 / 2.162953 | 0 | Pass |
| cgdt_0.005_tol_0.01 | 0.01 | 42.045034 | 1.213408 / 1.418882 / 1.923893 | 0 | Pass |
| cgdt_0.01_tol_1e-06 | 1e-06 | 44.650749 | 3.144053 / 5.874126 / 8.153122 | 0 | Fail |
| dst | 0.0 | 3.876942 | 0.000000 / 0.000000 / 0.000000 | 0 | Pass |

Nested correction directions: all comparisons below use this screen's own paired timings.

| Method | Step / CG tolerance | GPU ms | Worst u / v / energy-state error % | GPU outliers | All-state 5% |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.0025 | 4508.179159 | 1.812375 / 4.167162 / 6.223986 | 0 | Fail |
| chol_guard | 0.01 | 183.175590 | 1.810916 / 4.167481 / 6.223099 | 0 | Fail |
| trained_phase | 0.01 | 182.978820 | 1.766325 / 4.134009 / 6.103806 | 0 | Fail |
| trained_phase40 | 0.01 | 204.389192 | 2.109531 / 4.252749 / 6.167913 | 0 | Fail |
| trained_nested40 | 0.01 | 199.464938 | 1.477907 / 3.260997 / 5.041071 | 0 | Fail |
| cg_1e-06 | 1e-06 | 110.326429 | 0.199690 / 0.412971 / 0.574495 | 0 | Pass |
| cg_0.01 | 0.01 | 82.659176 | 0.303316 / 0.358723 / 0.485925 | 0 | Pass |
| cgdt_0.005_tol_1e-06 | 1e-06 | 69.839773 | 0.794879 / 1.548112 / 2.162953 | 0 | Pass |
| cgdt_0.005_tol_0.01 | 0.01 | 42.017093 | 1.213408 / 1.418882 / 1.923893 | 0 | Pass |
| cgdt_0.01_tol_1e-06 | 1e-06 | 44.453260 | 3.144053 / 5.874126 / 8.153122 | 0 | Fail |
| dst | 0.0 | 3.860532 | 0.000000 / 0.000000 / 0.000000 | 0 | Pass |

The nested head reaches 5.041071% worst energy-state error and 199.464938 GPU ms on the opened screen. Its 22.601361× acceleration is relative to the same-job original ROM, not a qualified FOM speedup. It still misses the all-state target. Its weights, initializer, step and solver are frozen before multiresolution evaluation on additional development cases.

## Work still in progress

Poisson bank-capacity results are undergoing acceptance auditing. A training-only correction-direction diagnostic and a broader Burgers training-coverage comparison are being prepared. Frozen wave multiresolution confirmation is running. Completed audits will be added here, including unsuccessful arms.

## Reproduction and evidence

Heat job `3563072` contains the paired GPU measurements; scientific source and checkpoint hashes are in the linked owner panel. Independent coordinator checks cover each model's worst saved trajectory on every mesh.

- [Heat complete panel](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/accuracy10/analysis/summary.md)
- [Poisson fixed-capacity panel](../worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/staged_accuracy08/panel.json)
- [Burgers training and solver panel](../worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/accuracy08/PANEL.json)
- [Wave geometry audit](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/accel06/audit.json)
- [Wave follow-up audit](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/accel07/audit.json)
- [Wave training audit](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/accel08/audit.json)
- [Wave larger-head audit](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/accel09/audit.json)
- [Wave correction-head audit](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/accel10/audit.json)
- [Normalized values, repetition arrays and source hashes](2026-09-11-accuracy-improvements-and-wave-speed.json)

Run `reports/generate_accuracy_campaign.py` with the repository Python environment to rebuild. All source hashes and generator identity are embedded in the adjacent JSON. These are single-training-seed development studies on the recorded families; they do not establish broad PDE generalization or final paper performance.

## Plain-language glossary

- **Intervals / mesh:** subdivisions along each spatial axis; larger values request more output points.
- **Bank / head / latent:** learned spatial functions / network choosing their coefficients / compressed coordinates solved online.
- **POD / bank projection:** a span built from training-snapshot singular vectors / the closest unrestricted bank combination in the stated field norm. These are diagnostic controls, not deployed nonlinear networks.
- **FOM / ROM / NMROM:** full-grid solver / reduced solver / reduced solver constrained to a nonlinear decoder.
- **GPU / host ms:** blocked complete GPU input-to-output query time / the same heat invocation including input and output transfers.
- **Relative error:** error magnitude divided by the specified reference magnitude. Heat uses the current true field at each time; the displayed wave screen uses initial physical scales.
- **Median / worst:** middle case error / largest case error, with each case scored at its worst saved output time. Runtime uses the median of all retained repetitions.
- **CG / DST:** iterative conjugate-gradient solver / direct discrete sine-transform solver. A CG tolerance is its stopping threshold, not its measured field error.
- **Stationarity:** sufficiently small gradient of the reduced solve objective. This does not itself guarantee physical accuracy.
- **Tail emphasis:** training loss that assigns more influence to large reconstruction errors within a training batch.
- **All-state / energy-state:** checking displacement, velocity and their combined energy norm / the norm combining velocity and spatial-gradient error.
- **Guard / parity / refinement:** a numerical check with a more robust fallback / agreement with unchanged equations / agreement after reducing the integration step.
- **Cholesky / tangent velocity:** a factorization for solving a positive-definite small matrix system / the decoder Jacobian multiplied by latent velocity.
- **Outlier:** heat repetition above one-and-a-half times its case median; wave repetition above twice its panel median. Counts and every duration are retained.
- **Development / sealed final:** cases used in diagnosis and method selection / untouched cases reserved for the paper's later final evaluation.
