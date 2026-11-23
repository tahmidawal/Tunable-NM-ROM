# Frozen-choice cost confirmation and accuracy–cost plots

The experiment is complete with a scoped negative confirmation result. Saved-development plots are final; fresh nominal error diagnostics include explicitly rejected solves and do not establish a new speed claim. All numbers below are generated from retained run JSONs.

The fixed learned-head configurations and their prior selected comparators were evaluated on 16 new physical cases, without reselection. No fresh comparison passes all fixed accuracy, numerical and timing criteria. Each method's worst fresh error exceeds the corresponding old head worst-error limit; therefore this failure is substantive even before considering numerical qualification.

## Nominal accuracy, including rejected returned fields

These values cover every case. They are diagnostics of returned fields, not evidence that rejected solves satisfy the registered stopping condition. They are not combined with separately timed calls into an accuracy–cost point.

| Method | Cases | Median evolved error (%) | Worst evolved error (%) | Nominal invalid cases | Tightened invalid cases | Fractional reference failures |
|---|---:|---:|---:|---|---|---|
| `learned_head_m292_dt0.02` | 16 | 0.091866379 | 0.274210071 | [15] | [4, 5, 8, 10, 15] | [] |
| `learned_head` | 16 | 0.068535548 | 0.247308689 | [15] | [4, 5, 8, 10, 15] | [] |
| `pod_head_m292_dt0.02` | 16 | 0.079984022 | 0.287262780 | [] | [8, 9, 11] | [] |
| `pod_head_m292_dt0.01` | 16 | 0.050438650 | 0.259189658 | [] | [8, 11] | [] |
| `fom_m40_rk4_s10` | 16 | 0.018402720 | 0.296832321 | [] | [] | [1, 2, 6, 10, 13] |
| `fom_m64_cnab2_s50` | 16 | 0.052597679 | 0.578692293 | [] | [] | [] |

Both learned heads fail nominal initialization on the same case. The returned fields remain finite, but the initial fit misses the frozen stationarity threshold. Tightening additionally exposes fit/stopping failures, despite maximum nominal–tighter field discrepancy of 1.07540100224e-10. A small field discrepancy does not retrospectively waive the registered stopping criterion. All reference absolute checks pass, with maximum refinement discrepancy 4.14457992043e-06; the coarse comparator's smaller errors fail the relative reference bound on the cases listed above.

## Timing evidence remains separate

All 516 timed invocation records are retained, including 10 failures. The two head timing summaries contain only their nominally successful subset and therefore are not reported as complete-cohort cost evidence here. The fresh figure displays those paired observations as explicitly unqualified diagnostics. The coarse comparator also fails the unchanged drift threshold. No old absolute time is pooled with this allocation.

The saved selection plot preserves all complete finite configurations, shows off-scale inaccurate settings at its upper boundary, and provides a full-range companion. Incomplete configurations have a separate failure table. Its marginal envelopes are visual summaries for one error statistic; qualification still requires both median and worst errors. Saved confirmation ratios and timing-repetition variation are separate figures.

## Completion and remaining scope

Job 7385300 completed in 269 allocated GPU-seconds (0.074722222 GPU-hours) on NVIDIA RTX PRO 6000 Blackwell Server Edition. The independent evidence audit passes: raw repetitions, case/cohort summaries, paired ratios, source hashes and precision/backend records agree. This audit validates the retained evidence, not the scientific qualification. The result archive and every listed file were checksum-verified locally before the owned remote job directory was removed.

An all-case production claim requires an initializer robustness correction developed outside this now-exposed confirmation cohort, followed by a different untouched confirmation. Further strict comparison to the coarse classical solver also needs a more accurate reference and a clean registered timing repeat; these alone cannot erase the current head initialization and worst-error failures. No change to the frozen method, comparator search, new cohort or further job is performed in this batch. The previous development evidence remains valid within its stated scope.

Builders, all plotted values, source hashes and PNG/PDF/SVG exports are in `build_plots.py` and `plots/`. The lane remains separate and unmerged pending the user's decision. Root records the canonical lab-log closure and publishes the combined main report.

## Glossary

**Nominal:** the original fixed numerical settings. **Tightened:** the same method with stricter stopping tolerances, used as a sensitivity check. **Stationarity:** the gradient stopping condition for the least-squares solve. **Initial fit:** finding the starting reduced coordinates from the physical field. **Fractional reference bound:** requiring reference uncertainty to be small relative to the measured error. **Case:** one physical initial condition and viscosity. **Median/worst:** the middle/largest casewise evolved error. **Invocation:** one complete input-to-output execution. **Partial timing subset:** cases whose nominal solver termination passed; it is not the full cohort. **Drift:** change in repeated measured time through the measurement block. **Frozen:** fixed before fresh truth solutions were generated. **Qualification:** the combined numerical, accuracy and timing criteria. **POD:** a snapshot-derived spatial basis. **Scoped negative:** the fixed experiment failed its claim without disproving every possible version of the method.
