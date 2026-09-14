# Accuracy improvements and reflective-wave speed

This report records the completed controlled accuracy-training and wave-acceleration round. These development numbers are finalized and independently audited; final publication validation remains unopened.

Absorbing waves are excluded. Final paper cohorts remain unopened, and the existing experiment branches remain separate. Tables are generated from saved invocation records; no wall-clock ratio crosses jobs or GPUs.

## What to retain from this round

All rows below use 1024 intervals per axis and each PDE's full development cohort. Errors are worst-case values under the stated normalization; they are not comparable across different norms. The timing ratio compares each chosen ROM with its own job's fastest tested passing iterative FOM.

| PDE (cases) | Error normalization | Before → after error % | ROM GPU ms | Iterative FOM / ROM | Decision |
| --- | --- | --- | --- | --- | --- |
| Poisson (42) | Static field L2 | 7.280248 → 6.110576 | 6.043163 | 14.761360×* | Accuracy improves; target still missed |
| Heat (16) | Current field L2 | 7.595346 → 4.762515 | 11.623313 | 4.633307× | Keep targeted training |
| Burgers (6) | Initial field L2 | 3.907620 → 3.884680 | 55.924140 | 1.217479× | Keep original network; reject both retrained heads |
| Reflective waves (4) | Initial energy-state | 6.213781 → 5.145194 | 202.941565 | 2.963889×* | Keep acceleration; accuracy target still missed |

*Poisson and reflective waves miss their declared physical target. Their ratios describe runtime only, not a speedup at matched target accuracy. Direct DST is faster for the linear PDEs. Burgers before/after accuracy comes from its controlled stopping comparison; the displayed runtime ratio uses the latest same-job control pair, whose original-head fields were checked against that comparison.

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

## Poisson: more spatial features help the bank, but not yet the complete solver

The capacity arm widens the learned spatial bank without changing its initial decoded function or latent dimension. The added head outputs start at zero. Both sizes undergo bank, head and joint stages, with the smaller control matched to each larger-model phase's measured optimizer time. This isolates bank capacity from a simultaneous latent increase or difficult-case reweighting.

All cases in this comparison were already opened during development. The head-projection diagnostic uses stationary full-field fits and is a best-found value, not a proof of the global nonlinear minimum. It was run only on the middle mesh; dashes on other meshes mean it was not evaluated there. All recorded online solves are stationary.

| Intervals | Model | Latent / bank size | Bank projection error % | Best-found head error % | Online physical error % | GPU ms | GPU outliers | 5% physical target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | original_relative | 16 / 64 | 6.668704 | — | 7.281747 | 2.403331 | 1 | Fail |
| 64 | r64_head | 16 / 64 | 7.175609 | — | 7.786738 | 2.438977 | 2 | Fail |
| 64 | r64_joint | 16 / 64 | 7.170453 | — | 7.683120 | 2.359436 | 0 | Fail |
| 64 | r128_head | 16 / 128 | 4.706641 | — | 7.387691 | 2.396871 | 0 | Fail |
| 64 | r128_joint | 16 / 128 | 4.519957 | — | 7.555330 | 2.355760 | 0 | Fail |
| 256 | original_relative | 16 / 64 | 6.499383 | 7.291654 | 7.280264 | 2.368029 | 1 | Fail |
| 256 | r64_head | 16 / 64 | 6.998636 | 7.796886 | 7.785401 | 2.364624 | 0 | Fail |
| 256 | r64_joint | 16 / 64 | 6.994698 | 7.693113 | 7.681675 | 2.311711 | 1 | Fail |
| 256 | r128_head | 16 / 128 | 4.546357 | 7.397654 | 7.386228 | 2.355973 | 0 | Fail |
| 256 | r128_joint | 16 / 128 | 4.363977 | 7.565149 | 7.553790 | 2.306459 | 0 | Fail |
| 1024 | original_relative | 16 / 64 | 6.489096 | — | 7.280248 | 2.610272 | 1 | Fail |
| 1024 | r64_head | 16 / 64 | 6.987864 | — | 7.785384 | 2.623935 | 0 | Fail |
| 1024 | r64_joint | 16 / 64 | 6.984004 | — | 7.681653 | 2.584367 | 0 | Fail |
| 1024 | r128_head | 16 / 128 | 4.536676 | — | 7.386210 | 2.870484 | 0 | Fail |
| 1024 | r128_joint | 16 / 128 | 4.354563 | — | 7.553765 | 2.849253 | 0 | Fail |

At 1024 intervals on all 42 cases, the expanded joint endpoint lowers the bank projection error from 6.489096% to 4.354563%, but complete online error changes from 7.280248% to 7.553765%. GPU query time also rises from 2.610272 to 2.849253 ms. Thus this endpoint is not accepted as an online accuracy or speed improvement.

The proposed bank target is 3%, separately from the 5% online physical target; both remain missed. A subsequent training-only residual decomposition motivates fixed correction directions. It uses saved optimized training codes, not independently certified stationary training fits, and its truth-assisted corrections are reconstruction diagnostics rather than online PDE results.

## Poisson: nested linear corrections improve accuracy

The final predeclared experiment freezes the larger bank and nonlinear head, then adds nested prefixes of one correction basis constructed only from training reconstruction residuals. Evaluation truth is not used to build this basis or initialize a query. The largest prefix is the primary configuration, fixed before these queries.

The decoder is $D_q(z,y)=G(h(z)+C_qy)$. For each $z$, solve the linear least-squares problem for $y$ exactly, then optimize the projected objective in $z$. Thus additional correction coordinates increase expressiveness while leaving the nonlinear optimizer dimension unchanged. This tests both added capacity and analytic elimination; it does not isolate a width-only change.

Every neural timing in this job includes the supplied-source contraction, initial-guess search, nonlinear solve, linear recovery, full decoding, and final gradient/rank diagnostics. Both the full and projected objective gradients must satisfy their stationarity tests. Added diagnostics change the timing contract from previous jobs, so compare only paired times below.

The prefix count is a fixed-weight tuning option after preparing each prefix's projected operators, initial-guess cache and compiled solver offline. Switching among those prepared presets requires no neural retraining. It is not a guarantee that arbitrary unprepared prefix sizes can be selected at zero setup cost.

| Intervals | Method | Nonlinear / total coordinates | Worst physical error % | GPU ms | Host ms | GPU outliers | Invalid solves | Physical and numerical criteria |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | cg_1e-01 | — | 9.627059 | 4.033062 | 4.919995 | 1 | 0 | Fail |
| 64 | cg_1e-06 | — | 0.273175 | 8.385411 | 9.331011 | 0 | 0 | Pass |
| 64 | cg_3e-02 | — | 1.376769 | 4.725828 | 5.616743 | 0 | 0 | Pass |
| 64 | dst | — | 0.273175 | 0.256050 | 1.892254 | 0 | 0 | Pass |
| 64 | original_relative | 16 / 16 | 7.281747 | 4.830095 | 6.358523 | 1 | 0 | Fail |
| 64 | r128_q0 | 16 / 16 | 7.555330 | 4.766349 | 6.275453 | 0 | 0 | Fail |
| 64 | r128_q16 | 16 / 32 | 7.034354 | 4.893498 | 6.425868 | 0 | 0 | Fail |
| 64 | r128_q32 | 16 / 48 | 6.111946 | 4.910128 | 6.465681 | 1 | 0 | Fail |
| 64 | r128_q8 | 16 / 24 | 7.209430 | 4.852020 | 6.369687 | 0 | 0 | Fail |
| 256 | cg_1e-01 | — | 3.036038 | 15.021542 | 16.278673 | 0 | 0 | Pass |
| 256 | cg_1e-06 | — | 0.016506 | 34.804448 | 36.210974 | 0 | 0 | Pass |
| 256 | cg_3e-02 | — | 0.721700 | 17.501463 | 18.643897 | 0 | 0 | Pass |
| 256 | dst | — | 0.016506 | 0.209141 | 2.184999 | 6 | 0 | Pass |
| 256 | original_relative | 16 / 16 | 7.280264 | 4.622388 | 6.362127 | 1 | 0 | Fail |
| 256 | r128_q0 | 16 / 16 | 7.553790 | 4.633263 | 6.336176 | 2 | 0 | Fail |
| 256 | r128_q16 | 16 / 32 | 7.032166 | 4.813237 | 6.573889 | 0 | 0 | Fail |
| 256 | r128_q32 | 16 / 48 | 6.110598 | 4.744899 | 6.652622 | 0 | 0 | Fail |
| 256 | r128_q8 | 16 / 24 | 7.207542 | 4.764289 | 6.587218 | 0 | 0 | Fail |
| 1024 | cg_1e-01 | — | 1.182643 | 89.205305 | 96.314497 | 1 | 0 | Pass |
| 1024 | cg_1e-06 | — | 0.000785 | 251.533784 | 258.338489 | 0 | 0 | Pass |
| 1024 | cg_3e-02 | — | 0.269520 | 103.618530 | 110.952986 | 0 | 0 | Pass |
| 1024 | dst | — | 0.000785 | 0.395368 | 8.098022 | 3 | 0 | Pass |
| 1024 | original_relative | 16 / 16 | 7.280248 | 3.927122 | 11.218030 | 5 | 0 | Fail |
| 1024 | r128_q0 | 16 / 16 | 7.553765 | 4.412962 | 11.692039 | 1 | 0 | Fail |
| 1024 | r128_q16 | 16 / 32 | 7.032119 | 5.232792 | 12.490237 | 3 | 0 | Fail |
| 1024 | r128_q32 | 16 / 48 | 6.110576 | 6.043163 | 13.092479 | 0 | 0 | Fail |
| 1024 | r128_q8 | 16 / 24 | 7.207505 | 5.031713 | 12.251328 | 5 | 0 | Fail |

At 1024 intervals on all 42 development cases, the primary correction lowers worst error from the original model's 7.280248% and its larger-bank parent's 7.553765% to 6.110576%. Its GPU time is 6.043163 ms, compared with 3.927122 ms for the original and 4.412962 ms for the parent in this job.

The primary uses 53.882742% more GPU time than the original and 36.941183% more than its uncorrected parent. This is an accuracy–cost tradeoff. All these cases were already opened in the preceding development comparisons; this is not independent final validation.

On that finest mesh, target-failing cases fall from 8 to 2 out of 42. This is useful progress, while the full-cohort target remains unmet.

The added directions help the solver use its existing spatial bank, but neither the physical target nor the separate tighter bank-projection target is met across all development cases. Preserve the enriched family as an accuracy-improving candidate; do not describe it as a completed target-accuracy solution. No further correction-basis search followed this result.

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

## Burgers: broader initial coverage does not repair trajectory accuracy

The final bounded training arm expands the initial-condition draw while retaining the spatial bank, head size, replay weight, update count, batch sizes and strict solver. More training codes also enlarge the initial-guess library and optimizer state, so this compares complete training procedures rather than isolated head weights. EQ is refitted using the same old training-code indices; the added initial codes are not substituted into its fitting set.

| Intervals | Strict method | Worst trajectory error % | Worst initial error % | GPU ms | Host ms | GPU outliers | Physical and numerical criteria |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | frozen_stationary | 10.856979 | 3.956378 | 52.257794 | 54.049926 | 0 | Fail |
| 64 | trained576_stationary | 11.045568 | 4.868142 | 44.811392 | 46.187700 | 0 | Fail |
| 64 | trained4608_stationary | 10.936769 | 4.411058 | 43.433198 | 45.114398 | 0 | Fail |
| 64 | fft_loose | 10.391989 | 0.000000 | 13.026291 | 14.623495 | 0 | Fail |
| 64 | fft_tight | 10.989131 | 0.000000 | 69.699107 | 71.376884 | 0 | Fail |
| 256 | frozen_stationary | 4.554611 | 2.562872 | 51.693426 | 54.547014 | 0 | Pass |
| 256 | trained576_stationary | 5.794182 | 4.544334 | 44.711987 | 47.535326 | 0 | Fail |
| 256 | trained4608_stationary | 5.308770 | 2.287985 | 47.097540 | 49.796811 | 0 | Fail |
| 256 | fft_loose | 2.473687 | 0.000000 | 16.259475 | 19.176364 | 0 | Pass |
| 256 | fft_tight | 4.026515 | 0.000000 | 97.495110 | 100.472711 | 0 | Pass |
| 1024 | frozen_stationary | 3.884680 | 3.856220 | 55.924140 | 76.348107 | 0 | Pass |
| 1024 | trained576_stationary | 5.459554 | 4.592752 | 49.021904 | 69.911559 | 0 | Fail |
| 1024 | trained4608_stationary | 5.277323 | 2.824662 | 49.600265 | 70.179544 | 0 | Fail |
| 1024 | fft_loose | 2.389937 | 0.000000 | 68.086457 | 89.113863 | 0 | Pass |
| 1024 | fft_tight | 2.141611 | 0.000000 | 512.905665 | 533.218171 | 0 | Pass |

At 1024 intervals, the broader head improves the unsuccessful smaller-training arm from 5.459554% to 5.277323% worst trajectory error, but the original strict head remains better at 3.884680%. All online strict solves meet stationarity. Both retrained heads are rejected as replacements for the original.

On case 3, initial reconstruction improves from 3.856220% to 2.824662%, while trajectory error worsens from 3.884680% to 5.277323%. This identifies a limitation of improving initial-field reconstruction alone. Because the decoder, initial-guess library and refitted EQ weights change together, the experiment does not uniquely attribute the later error to one mechanism.

Both inherited controls reproduce their saved parent trajectories and solver counters. Training and reference bytes have explicit inherited provenance; all displayed query costs and errors were rerun in this job. No runtime ratio crosses GPU allocations. Initial code-fit failures are retained in the training audit; passing online stationarity does not certify every offline training fit. The next accuracy direction would require trajectory-aware training and a controlled quadrature-fidelity comparison; neither was added after seeing these results.

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

Wave displacement error is divided by the initial displacement norm. Velocity and energy-state errors are divided by $\sqrt{2E_0}$, where $E_0$ is initial physical energy; initial velocity can be zero, and its norm is not the denominator. The energy-state error measures displacement-gradient and velocity error, not energy-conservation drift. Current-relative displacement and velocity errors are separately retained in the JSON and exclude times flagged as numerical zeros; near-vanishing reference times are recorded separately. DST is the same-grid semidiscrete reference, so its zero discrepancy is not zero continuum error.

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

Increasing the latent dimension and retraining with the same phase-aware loss worsened both runtime and accuracy on the opened screen. The next arm preserves the phase-trained `trained_phase` head and adds fixed linear correction directions from training data. This enlarges the nonlinear manifold while containing that parent decoder exactly; it is a separately prepared decoder.

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

The nested head reaches 5.041071% worst energy-state error and 199.464938 GPU ms on the opened screen. Its 22.601361× acceleration is relative to the same-job original ROM, not a qualified FOM speedup. It still misses the all-state target. Its weights, initializer, step and solver were frozen before multiresolution evaluation on additional development cases.

## Reflective waves: frozen multiresolution confirmation

The selected nested decoder, initializer and integration step were frozen before introducing the new development cases. Confirmation uses two previously opened cases plus two fresh development cases. The original head, its accelerated implementation, named CG controls and direct DST are retimed together in this job. The phase-only head and independently retrained larger head were not confirmed across these meshes.

| Intervals | Method | GPU ms | Worst initial-scaled u / v / energy-state error % | GPU outliers | Physical and numerical criteria |
| --- | --- | --- | --- | --- | --- |
| 64 | baseline | 4485.909332 | 1.812375 / 4.167162 / 6.223986 | 0 | Fail |
| 64 | chol_guard | 178.981540 | 1.810916 / 4.167481 / 6.223099 | 0 | Fail |
| 64 | trained_nested40 | 199.438048 | 1.477907 / 3.260997 / 5.041071 | 0 | Fail |
| 64 | cgdt_0.005_tol_0.01 | 42.195606 | 1.213408 / 1.418882 / 1.923893 | 0 | Pass |
| 64 | dst | 3.760324 | 0.000000 / 0.000000 / 0.000000 | 0 | Pass |
| 256 | baseline | 4472.278085 | 1.793941 / 4.212107 / 6.211970 | 0 | Fail |
| 256 | chol_guard | 179.116437 | 1.791871 / 4.212644 / 6.210170 | 0 | Fail |
| 256 | trained_nested40 | 200.158698 | 1.542841 / 3.419016 / 5.136114 | 0 | Fail |
| 256 | cgdt_0.005_tol_1e-06 | 138.838246 | 0.806122 / 1.572060 / 2.203322 | 0 | Pass |
| 256 | dst | 4.422321 | 0.000000 / 0.000000 / 0.000000 | 0 | Pass |
| 1024 | baseline | 4477.732209 | 1.796197 / 4.204148 / 6.213781 | 0 | Fail |
| 1024 | chol_guard | 183.441437 | 1.794093 / 4.204697 / 6.211926 | 0 | Fail |
| 1024 | trained_nested40 | 202.941565 | 1.546468 / 3.421022 / 5.145194 | 0 | Fail |
| 1024 | cgdt_0.005_tol_0.01 | 601.496361 | 2.106725 / 2.889742 / 3.531910 | 0 | Pass |
| 1024 | dst | 16.732217 | 0.000000 / 0.000000 / 0.000000 | 0 | Pass |

At 1024 intervals on all 4 cases, the selected ROM takes 202.941565 GPU ms versus 4477.732209 ms for the same-job original ROM (22.064145× acceleration). The fastest tested passing CG setting takes 601.496361 ms, a 2.963889× timing ratio. The ROM still misses the all-state target at 5.145194% energy-state error, so this is not an accuracy-qualified speedup at that target. Direct DST remains faster.

For each mesh, select the fastest tested CG setting that passes on all four cases; exclude a failing setting on that mesh. A setting can qualify on another mesh. Retained repetitions and all rejected CG settings remain in the normalized JSON.

| Fine-grid cohort | Method | Initial-scaled u / v / energy-state error % | Current-relative u / v error % | Physical and numerical criteria |
| --- | --- | --- | --- | --- |
| opened | baseline | 1.796197 / 4.204148 / 6.213781 | 3.568422 / 7.813101 | Fail |
| opened | trained_nested40 | 1.546468 / 3.421022 / 5.145194 | 2.519641 / 5.557525 | Fail |
| fresh_development | baseline | 1.565365 / 3.933681 / 5.147017 | 4.384534 / 6.730975 | Fail |
| fresh_development | trained_nested40 | 1.106656 / 2.697065 / 3.853874 | 2.703948 / 4.710923 | Pass |

The new development cohort stays separate from the opened selection cases; neither cohort is the paper's sealed final test. Every returned trajectory and refinement comparison passed the recorded numerical checks. A configuration-key error stopped the first confirmation attempt before any query cases were generated; the retry changed only that operational lookup and retained the frozen scientific configuration.

## What remains for the paper

The bounded experiment round is complete. Heat's targeted training and the reflective-wave implementation acceleration are useful changes. Poisson's nested corrections improve accuracy but still need a better representation to meet the physical target. Burgers needs a training objective that preserves time evolution; both initial-field retraining arms remain negative results. Wave energy-state accuracy also remains short of its target.

A subsequent round should freeze its protocol before measuring new cases: trajectory-aware Burgers training with a controlled EQ comparison, wave displacement-gradient/velocity training, and better Poisson correction coverage. Multiple training seeds and the sealed final cohorts are still required before a publication-level generalization claim. None of that additional search or final testing was performed here.

## Reproduction and evidence

Heat job `3563072` contains the paired GPU measurements; scientific source and checkpoint hashes are in the linked owner panel. Independent coordinator checks cover each model's worst saved trajectory on every mesh.

- [Heat complete panel](../worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/accuracy10/analysis/summary.md)
- [Poisson fixed-capacity panel](../worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/staged_accuracy08/panel.json)
- [Poisson larger-bank panel](../worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/capacity_accuracy09/panel.json)
- [Poisson nested-correction panel](../worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/correction_accuracy10/panel.json)
- [Burgers training and solver panel](../worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/accuracy08/PANEL.json)
- [Burgers broader-coverage panel](../worktrees/2026-09-07-mr-burgers2d/experiments/mr-burgers2d/runs/accuracy09/PANEL.json)
- [Wave geometry audit](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/accel06/audit.json)
- [Wave follow-up audit](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/accel07/audit.json)
- [Wave training audit](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/accel08/audit.json)
- [Wave larger-head audit](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/accel09/audit.json)
- [Wave correction-head audit](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/accel10/audit.json)
- [Wave multiresolution confirmation audit](../worktrees/2026-09-07-mr-wave2d/experiments/multiresolution-wave/runs/accel12/audit.json)
- [Normalized values, repetition arrays and source hashes](2026-09-11-accuracy-improvements-and-wave-speed.json)

- [Selected implementation and artifact inventory](../reports/2026-09-11-accuracy-integration.json)

Run `reports/generate_accuracy_campaign.py` with the repository Python environment to rebuild. All source hashes and generator identity are embedded in the adjacent JSON. These are single-training-seed development studies on the recorded families; they do not establish broad PDE generalization or final paper performance.

## Plain-language glossary

- **Intervals / mesh:** subdivisions along each spatial axis; larger values request more output points.
- **Bank / head / latent:** learned spatial functions / network choosing their coefficients / compressed coordinates solved online.
- **POD / bank projection:** a span built from training-snapshot singular vectors / the closest unrestricted bank combination in the stated field norm. These are diagnostic controls, not deployed nonlinear networks.
- **FOM / ROM / NMROM:** full-grid solver / reduced solver / reduced solver constrained to a nonlinear decoder.
- **GPU / host ms:** blocked complete GPU input-to-output query time / the same timed invocation including input and output transfers.
- **Relative error:** error magnitude divided by the specified reference magnitude. Heat uses the current true field at each time; the displayed wave screen uses initial physical scales.
- **Median / worst:** middle case error / largest case error, with each case scored at its worst saved output time. Runtime uses the median of all retained repetitions.
- **CG / DST:** iterative conjugate-gradient solver / direct discrete sine-transform solver. A CG tolerance is its stopping threshold, not its measured field error.
- **Stationarity:** sufficiently small gradient of the reduced solve objective. This does not itself guarantee physical accuracy.
- **Correction prefix / analytic elimination:** the first selected directions of one fixed training basis / solving their linear coefficients exactly inside the nonlinear solve. Nonlinear coordinates are searched iteratively; total coordinates include the analytically recovered ones.
- **Tail emphasis:** training loss that assigns more influence to large reconstruction errors within a training batch.
- **EQ / replay / coverage:** fitted quadrature weights approximating weak sums / preserving old decoded outputs during training / the range and number of training initial conditions.
- **All-state / energy-state:** checking displacement, velocity and their combined energy norm / the norm combining velocity and spatial-gradient error.
- **Guard / parity / refinement:** a numerical check with a more robust fallback / agreement with unchanged equations / agreement after reducing the integration step.
- **Cholesky / tangent velocity:** a factorization for solving a positive-definite small matrix system / the decoder Jacobian multiplied by latent velocity.
- **Outlier:** heat and Poisson repetition above one-and-a-half times its case median; Burgers repetition above twice its case median; wave repetition above twice its panel median. Counts and every duration are retained.
- **Development / sealed final:** cases used in diagnosis and method selection / untouched cases reserved for the paper's later final evaluation.
