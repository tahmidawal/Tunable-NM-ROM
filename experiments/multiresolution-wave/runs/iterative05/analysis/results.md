# Fresh-wave iterative FOM multiresolution comparison

These are new bounded development measurements using verified post-reset wave operators and frozen heads. All values below are generated from the retained timed outputs; no final cohort or continuum-accuracy claim is included.

| Boundary | Intervals | Method | Query median (ms) | CG 1e-6/query ratio | Worst initial-scaled displacement / velocity / energy | Worst current-relative displacement / velocity / energy | Failed / nonstationary / refinement |
|---|---:|---|---:|---:|---|---|---|
| dirichlet | 64 | frozen_mlp32_seed691200 | 4509.71 | 0.0245086 | 0.0181237 / 0.0416716 / 0.0622399 | 0.0364074 / 0.0771491 / 0.0622399 | 0 / 0 / 0 |
| dirichlet | 64 | cg_1e-06 | 110.527 | 1 | 0.0019969 / 0.00412971 / 0.00574495 | 0.00352078 / 0.00738239 / 0.00574495 | 0 / 0 / 0 |
| dirichlet | 64 | cg_0.01 | 82.6188 | 1.33779 | 0.00303316 / 0.00358723 / 0.00485925 | 0.00596143 / 0.00709512 / 0.00485925 | 0 / 0 / 0 |
| dirichlet | 64 | dst | 3.71379 | 29.8639 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| dirichlet | 256 | frozen_mlp32_seed691200 | 4530.3 | 0.0299065 | 0.0179394 / 0.0421211 / 0.0621197 | 0.0356899 / 0.0781912 / 0.0621197 | 0 / 0 / 0 |
| dirichlet | 256 | cg_1e-06 | 135.485 | 1 | 0.00203903 / 0.00426775 / 0.00601802 | 0.00353387 / 0.00790952 / 0.00601802 | 0 / 0 / 0 |
| dirichlet | 256 | cg_0.01 | 98.6948 | 1.37278 | 0.0969524 / 0.123594 / 0.170932 | 0.181074 / 0.232985 / 0.170932 | 0 / 0 / 0 |
| dirichlet | 256 | dst | 4.02287 | 33.6925 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| dirichlet | 1024 | frozen_mlp32_seed691200 | 4531.54 | 0.232504 | 0.017962 / 0.0420415 / 0.0621378 | 0.0356842 / 0.078131 / 0.0621378 | 0 / 0 / 0 |
| dirichlet | 1024 | cg_1e-06 | 1053.46 | 1 | 0.00203836 / 0.00426563 / 0.00607856 | 0.00353176 / 0.0079011 / 0.00607856 | 0 / 0 / 0 |
| dirichlet | 1024 | cg_0.01 | 758.618 | 1.39688 | 0.0769918 / 0.106497 / 0.131735 | 0.171077 / 0.248137 / 0.131735 | 0 / 0 / 0 |
| dirichlet | 1024 | dst | 16.6257 | 63.3503 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| absorbing | 64 | frozen_mlp32_seed691200 | 5026.19 | 0.0341259 | 0.0214912 / 0.0426224 / 0.05919 | 2.46291 / 2.95057 / 3.12066 | 0 / 0 / 0 |
| absorbing | 64 | cg_1e-06 | 171.524 | 1 | 0.000285579 / 0.000614907 / 0.000865223 | 0.017263 / 0.350678 / 0.303935 | 0 / 0 / 0 |
| absorbing | 64 | cg_0.01 | 93.6198 | 1.83211 | 0.00762001 / 0.00779742 / 0.010753 | 2.26247 / 0.372439 / 0.297031 | 0 / 0 / 0 |
| absorbing | 64 | rk4 | 14.3893 | 11.9205 | 1.10079e-05 / 0.000136805 / 0.00017701 | 0.0067202 / 0.15099 / 0.12723 | 0 / 0 / 0 |
| absorbing | 256 | frozen_mlp32_seed691200 | 5040.49 | 0.0657745 | 0.0213979 / 0.0425544 / 0.059136 | 2.46539 / 2.95848 / 3.22436 | 0 / 0 / 0 |
| absorbing | 256 | cg_1e-06 | 331.545 | 1 | 0.000286947 / 0.000703976 / 0.000997667 | 0.00200499 / 0.00498989 / 0.00521 | 0 / 0 / 0 |
| absorbing | 256 | cg_0.01 | 161.747 | 2.04943 | 0.0376895 / 0.0524928 / 0.0734411 | 13.4456 / 10.4627 / 9.62024 | 0 / 0 / 0 |
| absorbing | 256 | rk4 | 86.3997 | 3.84105 | 1.10135e-07 / 2.63401e-06 / 3.71916e-06 | 6.58535e-06 / 0.000403405 / 0.00037026 | 0 / 0 / 0 |
| absorbing | 1024 | frozen_mlp32_seed691200 | 5027.97 | 0.719867 | 0.0213922 / 0.0425504 / 0.059135 | 2.46557 / 2.95689 / 3.22823 | 0 / 0 / 0 |
| absorbing | 1024 | cg_1e-06 | 3619.95 | 1 | 0.00028663 / 0.000714399 / 0.00101236 | 0.00230577 / 0.00490361 / 0.0050984 | 0 / 0 / 0 |
| absorbing | 1024 | cg_0.01 | 1508.49 | 2.40178 | 0.0404198 / 0.0210936 / 0.0297068 | 19.9515 / 7.87988 / 7.045 | 0 / 0 / 0 |
| absorbing | 1024 | rk4 | 2314.95 | 1.56359 | 5.10808e-10 / 1.49446e-08 / 2.11324e-08 | 3.17779e-09 / 1.04295e-07 / 1.09675e-07 | 0 / 0 / 0 |

Ratios compare complete device queries to the new same-grid implicit-midpoint CG FOM at tolerance 1e-6. This is a matched-time-step iterative control using fresh wave mathematics, not a literal historical wave algorithm. Cold initialization, all projection, speed preparation, evolution and full GPU fields are charged. Host transfers are outside this timer. Inspect the adjacent errors and failure counts before interpreting a ratio as useful acceleration.

All recorded same-grid reference refinement gates passed: **True**. Integrity checks and completed execution do not establish numerical-gate or target-accuracy acceptance. The JSON explicitly qualifies each target using reference refinement, ROM refinement, completed evolution, fit stationarity and all three error components.

The summary JSON retains component timings, every repetition, per-case error curves, reference temporal refinement and separate nonlinear time refinement. A timing outlier is a repetition above twice its case median. Error-outlier counts are retained for each predeclared target. Reference refinement is empirical and does not establish a continuum bound.

## Glossary

- **Intervals / method:** spatial cells per axis / the named frozen reduced model or full solver.
- **CG 1e-6/query ratio:** the median of paired per-case FOM time divided by method time; above one means the named method is faster under this timing contract.
- **Initial-scaled / current-relative:** error divided by the fixed reference initial scale / the reference norm at the observation time.
- **Energy:** the joint displacement-gradient and physical-velocity norm.
- **Failed / nonstationary / refinement:** incomplete trajectories / initial fits missing their fixed stopping criteria / trajectories failing the smaller-step comparison.
- **Median / worst / outlier:** the middle value / largest value / a timing or error exceeding the declared threshold.
- **Device query / cold initialization:** GPU input through full GPU output / fitting the initial reduced coordinates from supplied fields.
