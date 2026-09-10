# Fresh-wave device-resident comparison

These are new bounded development measurements using verified post-reset wave operators and frozen heads. All values below are generated from the retained timed outputs; no final cohort or continuum-accuracy claim is included.

| Boundary | Intervals | Method | Query median (ms) | FOM/query ratio | Worst initial-scaled displacement / velocity / energy | Worst current-relative displacement / velocity / energy | Failed / nonstationary / refinement |
|---|---:|---|---:|---:|---|---|---|
| dirichlet | 256 | frozen_mlp32_seed691200 | 4504.97 | 0.000920045 | 0.0179394 / 0.0421211 / 0.0621197 | 0.0356899 / 0.0781912 / 0.0621197 | 0 / 0 / 0 |
| dirichlet | 256 | frozen_mlp16_seed691200 | 3296.77 | 0.00125723 | 0.286216 / 0.353563 / 0.552425 | 0.421768 / 0.52847 / 0.552425 | 0 / 0 / 0 |
| dirichlet | 256 | dst | 4.14478 | 1 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| dirichlet | 512 | frozen_mlp32_seed691200 | 4504.8 | 0.00129467 | 0.017957 / 0.042063 / 0.0621334 | 0.0356842 / 0.078146 / 0.0621334 | 0 / 0 / 0 |
| dirichlet | 512 | frozen_mlp16_seed691200 | 3300.31 | 0.00176719 | 0.28581 / 0.353086 / 0.551737 | 0.421204 / 0.527932 / 0.551737 | 0 / 0 / 0 |
| dirichlet | 512 | dst | 5.83223 | 1 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| absorbing | 256 | frozen_mlp32_seed691200 | 5003.07 | 0.017838 | 0.0213979 / 0.0425544 / 0.059136 | 2.46539 / 2.95848 / 3.22436 | 0 / 0 / 0 |
| absorbing | 256 | frozen_mlp16_seed691200 | 3222.06 | 0.0276967 | 0.0387474 / 0.0570523 / 0.0765974 | 3.12313 / 3.23432 / 4.10075 | 0 / 0 / 0 |
| absorbing | 256 | rk4 | 89.2555 | 1 | 1.10135e-07 / 2.63401e-06 / 3.71916e-06 | 6.58535e-06 / 0.000403405 / 0.00037026 | 0 / 0 / 0 |
| absorbing | 512 | frozen_mlp32_seed691200 | 4999.11 | 0.0518636 | 0.0213933 / 0.0425512 / 0.0591352 | 2.46553 / 2.9572 / 3.22745 | 0 / 0 / 0 |
| absorbing | 512 | frozen_mlp16_seed691200 | 3218.86 | 0.080535 | 0.0387209 / 0.0570363 / 0.0765613 | 3.1254 / 3.23617 / 4.10612 | 0 / 0 / 0 |
| absorbing | 512 | rk4 | 259.308 | 1 | 7.73899e-09 / 2.15383e-07 / 3.04724e-07 | 4.94283e-08 / 4.27783e-06 / 4.57191e-06 | 0 / 0 / 0 |

Ratios compare complete device queries to the named same-grid FOM. Cold initialization, all projection, speed preparation, evolution and full GPU fields are charged. Host transfers are outside this timer. Inspect the adjacent errors and failure counts before interpreting a ratio as useful acceleration.

All recorded same-grid reference refinement gates passed: **True**. Integrity checks and completed execution do not establish numerical-gate or target-accuracy acceptance. The JSON explicitly qualifies each target using reference refinement, ROM refinement, completed evolution, fit stationarity and all three error components.

The summary JSON retains component timings, every repetition, per-case error curves, reference temporal refinement and separate nonlinear time refinement. A timing outlier is a repetition above twice its case median. Error-outlier counts are retained for each predeclared target. Reference refinement is empirical and does not establish a continuum bound.

## Glossary

- **Intervals / method:** spatial cells per axis / the named frozen reduced model or full solver.
- **FOM/query ratio:** the median of paired per-case FOM time divided by method time; above one means the named method is faster under this timing contract.
- **Initial-scaled / current-relative:** error divided by the fixed reference initial scale / the reference norm at the observation time.
- **Energy:** the joint displacement-gradient and physical-velocity norm.
- **Failed / nonstationary / refinement:** incomplete trajectories / initial fits missing their fixed stopping criteria / trajectories failing the smaller-step comparison.
- **Median / worst / outlier:** the middle value / largest value / a timing or error exceeding the declared threshold.
- **Device query / cold initialization:** GPU input through full GPU output / fitting the initial reduced coordinates from supplied fields.
