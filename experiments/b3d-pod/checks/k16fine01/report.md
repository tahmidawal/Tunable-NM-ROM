# Archived POD support at the selected no-HR endpoint

Final four-case diagnostic evidence at the independently selected no-HR endpoint. Both own-objective nonlinear checks pass, while shared field consistency and the separate additional POD gradient checks fail; no complete-cohort HR release follows.

The frozen endpoint is `POD512_none_CN_25steps_g1e-4`. Original support failures and the stopped constructor remain preserved. These are exposed validation cases, not sealed confirmation or a complete accuracy/cost frontier.

| arm | own-objective Q1 | nonstationary steps | median E0 error | worst E0 error |
|---|---|---:|---:|---:|
| k16 (sampled objective) | True | 0 / 100 | 2.036665% | 2.280985% |
| none (full residual) | True | 0 / 100 | 2.040161% | 2.280650% |

Errors are nominal relative to approved E0 fields with conditional reference uncertainty; their small inter-arm differences do not establish an accuracy ranking.

| case | k16 to matching-none field distance | common bar | common pass | native-tight distance lower bound (diagnostic) |
|---|---:|---:|---|---:|
| 0 | 0.0001392422102 | 0.00025 | True | 0.0001289083349 |
| 5 | 0.0004652805312 | 0.00025 | False | 0.0003638171948 |
| 30 | 0.0002268497943 | 0.00025 | True | 0.000130292451 |
| 43 | 0.0001899891437 | 0.00025 | True | 0.0001820543222 |

The native-tight lower bound is a post-processing diagnostic from the triangle inequality and the two saved production-to-tight distances; it is not an additional admission predicate.

| case | maximum gradient discrepancy | maximum full stationarity | gradient misses | full-stationarity misses |
|---|---:|---:|---:|---:|
| 0 | 0.00131383278 | 0.001313832869 | 4 | 4 |
| 5 | 0.001097126752 | 0.001067030793 | 2 | 3 |
| 30 | 0.0005475672654 | 0.000550209429 | 0 | 0 |
| 43 | 0.0009085713097 | 0.0009085681008 | 0 | 0 |

The extra gradient and full-stationarity bars are 0.001. The gradient bar applies at every live step. There are 6 gradient misses and 7 / 100 full-stationarity misses; the allowed pooled fraction is 1%.

Matching-none endpoint reproduction passed: True. All actual timed payloads remain exactly hash-linked or separately saved; 4 ABA panels are retained and 4 pass diagnostic drift admission. No rejected row enters an admitted cost comparison.

Job `6529465` completed with exit `0:0` in 77 allocated seconds (0.0213888889 GPU-hours). The full-residual fallback remains qualified in its original complete-cohort scope. A qualified HR comparison remains unresolved; this result does not show POD loses to the bank.

## Glossary

- arm: one frozen numerical model; k16 uses the archived weighted support, none uses every residual row.
- own-objective Q1: trajectory validity, fresh stationarity and agreement with stricter solves of that same objective.
- nonstationary step: a returned state whose fresh normalized objective gradient exceeds its frozen bar.
- E0: the approved refined reference, with conditional numerical uncertainty.
- median / worst error: middle / largest error among the diagnostic cases.
- field distance: largest output-field difference divided by the common initial-field norm.
- native-tight: the stricter native solver; the lower bound uses its saved distance from production.
- gradient discrepancy: sampled versus full-residual gradients in common full-grid objective units.
- full stationarity: the full-residual gradient norm normalized by full Jacobian and residual norms.
- miss: a step that fails the corresponding frozen numerical bar.
- ABA panel: k16, matching-none, k16 timed successively on one GPU to check drift.
- HR: hyper-reduction, using weighted selected residual rows.
- GPU-hours: scheduler-allocated GPU time including setup and diagnostics.
