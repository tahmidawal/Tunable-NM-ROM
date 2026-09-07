# Heat head refinement: additional optimization versus training coverage

These provisional development results hold the spatial bank and neural architecture fixed while refining the coefficient head and training codes. Every scheduled endpoint and failed reconstruction/rollout gate is retained; this is not final confirmation or broader multi-bump heat coverage.

Generated from `experiments/mr-heat2d/runs/pilot03/archive/outputs/results.json`. Source `0597f0dd505de537ebbda459d9db330132126d44`, GPU job `3352849`, `NVIDIA A100 80GB PCIe`. Native audit verified 112 files and 83 arrays, with maximum recomputed metric difference 3.33067e-16. Frozen spatial parameters match exactly across every checkpoint; bank rank remains 32.

Expanded-coverage worst initial errors are ['0.0395305', '0.0403031'], compared with ['0.0941133', '0.0943119'] after matched refinement on the original cohort. The spatial bank and head architecture did not change. [Accuracy and complete-query figure](figures/heat-head.svg).

## Controlled training and offline cost

Each refinement uses 8000 updates and minibatches of 192 snapshots. The two paired sampling seeds are [790714, 790715]; additional training uses seed 790713. Validation is unchanged and absent from the loss and training-code initialization library. The final cohort stays unopened.

| Model | Training trajectories | Observed refinement wall s | Final training relative MSE |
|---|---|---|---|
| frozen | 32 | — | — |
| original_seed790714 | 32 | 7.32151 | 4.63302e-05 |
| original_seed790715 | 32 | 4.04499 | 4.63239e-05 |
| expanded_seed790714 | 160 | 4.7907 | 0.000344319 |
| expanded_seed790715 | 160 | 4.68038 | 0.000343335 |

Refinement seconds are observed wall durations including first-call compilation and host work. No dedicated warm-GPU timing block preceded each training arm; these are offline costs, not a paired steady-state training-speed comparison.

The QR-compressed objective equals full-field relative reconstruction loss, including the constant error outside the bank. A GPU test checks both loss and gradient parity. More training on the original cohort is the control for additional optimization; improvement over the frozen head alone does not prove a benefit from added coverage.

## Strict reconstruction and rollout gate

Initial errors are separate from time-maximum errors after the initial state. The same nearest-training-code/mean-code multistart fitting budget is applied to every model. All starts and selected gradients/stops are preserved.

| Model | Initial error median / worst | Later error median / worst | Initial cases above 5% | Initial / all-time selected nonstationary fits | Rollout gate |
|---|---|---|---|---|---|
| frozen | 0.0503788 / 0.0973564 | 0.0308089 / 0.0623549 | 2 | 0 / 0 | False |
| original_seed790714 | 0.046531 / 0.0941133 | 0.0250067 / 0.0589801 | 2 | 0 / 0 | False |
| original_seed790715 | 0.0467689 / 0.0943119 | 0.0252941 / 0.0591517 | 2 | 0 / 0 | False |
| expanded_seed790714 | 0.0265592 / 0.0395305 | 0.0174011 / 0.0216825 | 0 | 0 / 0 | True |
| expanded_seed790715 | 0.026697 / 0.0403031 | 0.0173339 / 0.0211244 | 0 | 0 / 0 | True |

The unchanged unrestricted bank's worst reconstruction error is 0.0103101. The predeclared rollout gate requires every selected initial fit to be stationary and below 0.05 relative error. This gate does not certify autonomous dynamics. A stationary fit is not proof of a global minimum.

Each endpoint uses its own training-code lookup library. Expanded coverage therefore changes both the fitted head/codes and the available starting-code coverage. This comparison establishes an expanded-coverage pipeline improvement; it does not isolate those mechanisms or certify globally optimal fits.

## Paired complete-query results

Every cost and error comes from the same saved invocation. Full host input, online initial fitting, evolution, requested dense fields and host output are charged. The frozen original head and direct DST FOM are timed in this same job; no cross-job raw times are compared. Both solver meshes use one shared observation grid. Query times are median case-medians; ratios are medians across cases of median paired-repetition FOM/ROM ratios.

| Intervals | Method | Query ms | FOM / method | Current error median / worst | All solver / parity checks pass | Empirical targets |
|---|---|---|---|---|---|---|
| 64 | fom_dst_exact_time | 1.13795 | 1 | 0.000675205 / 0.000709211 | True / True | [0.1, 0.05, 0.01, 0.001] |
| 64 | frozen_gtol1e-09 | 27.3496 | 0.0416115 | 0.0585754 / 0.0973564 | True / True | [0.1] |
| 64 | frozen_gtol1e-05 | 13.734 | 0.082163 | 0.0585779 / 0.0973565 | True / True | [0.1] |
| 64 | expanded_seed790714_gtol1e-09 | 20.6945 | 0.0548196 | 0.0265592 / 0.0395305 | True / True | [0.1, 0.05] |
| 64 | expanded_seed790714_gtol1e-05 | 12.3825 | 0.0911443 | 0.0265592 / 0.0395311 | True / True | [0.1, 0.05] |
| 64 | expanded_seed790715_gtol1e-09 | 19.9698 | 0.0573213 | 0.026697 / 0.0403031 | True / True | [0.1, 0.05] |
| 64 | expanded_seed790715_gtol1e-05 | 12.1714 | 0.0949884 | 0.0266971 / 0.0403035 | True / True | [0.1, 0.05] |
| 128 | fom_dst_exact_time | 1.2259 | 1 | 0.000168684 / 0.000177175 | True / True | [0.1, 0.05, 0.01, 0.001] |
| 128 | frozen_gtol1e-09 | 26.804 | 0.0435881 | 0.0585451 / 0.0973564 | True / True | [0.1] |
| 128 | frozen_gtol1e-05 | 13.7157 | 0.0886615 | 0.0585477 / 0.0973565 | True / True | [0.1] |
| 128 | expanded_seed790714_gtol1e-09 | 20.2088 | 0.057371 | 0.0265592 / 0.0395305 | True / True | [0.1, 0.05] |
| 128 | expanded_seed790714_gtol1e-05 | 12.2136 | 0.0948647 | 0.0265592 / 0.0395311 | True / True | [0.1, 0.05] |
| 128 | expanded_seed790715_gtol1e-09 | 19.8398 | 0.0588248 | 0.026697 / 0.0403031 | True / True | [0.1, 0.05] |
| 128 | expanded_seed790715_gtol1e-05 | 12.0105 | 0.0965626 | 0.0266971 / 0.0403035 | True / True | [0.1, 0.05] |

Physical-reference refinement evidence is empirical, with delta 1.81003e-10. Eligibility uses $(e+\delta)/(1-\delta)$ and the reference-budget check; rigorous physical bounds remain null. These are development findings, not a mathematical continuum guarantee.

## Plain-language glossary

- **Bank / head / code:** frozen spatial functions / nonlinear coefficient map / compressed coordinate fitted to a training snapshot.
- **Original / expanded / frozen:** original training cohort / added independent training draws / unchanged initial checkpoint.
- **QR / MSE / rank:** exact orthonormal field compression / mean squared error / number of independent bank directions.
- **Median / worst / cases above:** middle cohort error / largest cohort error / count missing the stated threshold.
- **Multistart / stationary / rollout gate:** fitting from several training-derived starts / sufficiently small gradient / predeclared condition for testing evolution.
- **FOM / ROM / DST:** full-grid solver / reduced solver / fast sine transform.
- **Query ms / paired ratio / parity:** complete input-to-output milliseconds / same-case repetition cost ratio / matching compiled and modular fields and counters.
- **Current error / empirical target / rigorous bound:** error divided by current reference magnitude / target supported by refinement evidence and solver checks / proved physical error limit, unavailable here.
- **Offline / validation / final cohort:** work before online queries / development evaluation cases / untouched independent confirmation cases.
