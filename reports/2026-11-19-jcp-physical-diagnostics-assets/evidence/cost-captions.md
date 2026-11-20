# Accuracy–cost figure captions

Final saved development measurements are replotted here; a fresh panel is included only when its result file is supplied. Every displayed value is generated from the linked JSON sources.

**Selection frontier.** Complete-query time versus cohort median and worst casewise maximum evolved field error for 107 tested configurations on 16 exposed development cases. Only the three candidate-selection repetitions are used. Lines join each family's marginal observed time/error envelope; these are not a joint median-and-worst eligibility criterion or an interpolated predictor. Filled stars mark learned heads. The primary view shows errors up to ten percent; finite settings above that limit are shown at the upper edge and retain their exact values in the full-range companion and plotted-data file. The 5 incomplete configurations are listed below instead of plotted as successful points. No candidate is omitted from the evidence.

**Confirmed case ratios.** The separately measured five-repetition confirmation on the same development cases, after comparator selection. Each point is the ratio of comparator to ROM case-median complete-query time. The dashed line at one is a timing tie. Legend statistics are medians of paired case ratios, not ratios of cohort-median times. Eligibility uses both cohort median and worst error; it does not guarantee pairwise error matching. The near POD tie and coarse classical losses remain visible.

**Repetition spread.** All confirmation timing repetitions normalized by their own method–case median. This displays measurement variation; it is neither case-generalization uncertainty nor trained-model seed uncertainty. Tukey outliers remain displayed and retained.

**Fresh confirmation: physical errors.** Each left-panel point is one case's maximum nominal field error over the requested evolved outputs. Learned/POD values come from the nominal qualification calls; classical values come from their scored timing calls. Crosses mark failure of a numerical check, including initial fitting, tighter stopping or fractional reference accuracy. Returned fields from rejected solves remain visible as diagnostics. These error diagnostics are not combined with separately timed measurements to manufacture an accuracy–cost point. All 16 nominal physical cases appear in the error panels. The shaded timing case index 15 has no nominally valid head query; each head ratio panel therefore contains only 15 available cases, not a complete-cohort timing result.

The dashed horizontal line is the frozen **cohort-median** accuracy limit: individual cases may exceed it while the cohort median still passes. It is not a per-case requirement. The dotted line is the frozen **cohort-maximum** limit: a case above it makes the cohort maximum fail. Both limits remain at their original development values.

**Fresh confirmation: observed timing ratios.** The right panels show comparator/head case-median ratios for the available nominally valid timed calls. The dashed line at one is a timing tie. Legend numbers are medians of these paired ratios. The legend's `fixed` decision requires complete-case coverage, numerical and timing qualification, and both the head and comparator to satisfy the unchanged cohort-median and cohort-maximum limits. The separate `match` decision requires complete-case coverage, numerical and timing qualification, and the comparator's fresh cohort median and maximum errors to be no greater than the fresh head's. `match` does not redefine or relax the frozen limits. Every fresh comparison fails full qualification; the plotted partial ratios support no accepted speed claim. This is frozen-choice validation, not a new selection frontier.

**Hardware scope.** Saved development measurements used NVIDIA L40S. Each fresh plot uses its own recorded allocation; no absolute time is pooled across hardware.

## Incomplete selection configurations

| Configuration | Failed invocations | Total invocations |
|---|---:|---:|
| `pod_head_m292_dt0.005` | 12 | 48 |
| `fom_m64_rk4_s5` | 3 | 48 |
| `fom_m64_rk4_s10` | 24 | 48 |
| `fom_m64_rk4_s15` | 9 | 48 |
| `fom_m64_rk4_s20` | 3 | 48 |

## Glossary

**Complete query:** all work from the declared host initial field to returned host fields, including native initialization and diagnostics conversion. **Head:** nonlinear coefficient map inside a frozen bank. **Span:** arbitrary linear combinations of the bank. **POD:** a snapshot-derived spatial basis; POD-head settings use a frozen coefficient map. **Coarse/same-grid:** a classical solver using fewer/the same grid points as the returned field. **Median/worst error:** middle/largest casewise maximum evolved error. **Selection:** the development comparison used to choose settings. **Confirmation:** repetitions or fresh cases evaluated after choices were fixed, identified separately. **Paired ratio:** comparator time divided by ROM time for the same case. **Marginal envelope:** observed best time–error choices for one plotted error statistic. **Tukey outlier:** a sample more than one and a half interquartile ranges beyond the lower or upper quartile.
