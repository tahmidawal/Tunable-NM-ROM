# Burgers 2D: fixed physical initialization in the complete query

Audited development results for unchanged coordinate-separable network weights across the requested meshes. No tested configuration establishes a complete-query ROM advantage over the eligible efficient FOM envelope.

At 1024 intervals, the primary fixed-Gauss rollout lowers worst complete-grid error from 0.08695337 to 0.0390762, with complete-query costs 54.3917 and 54.9038 ms respectively. It meets the empirical development target in the selection table; the unchanged efficient FOM envelope remains cheaper.

Source `d73fb4119057d4826c783d02829dd08835fe7a24`, job `3352857`, GPU `NVIDIA A100 80GB PCIe`; backend `gpu`, f64 `True`, precision `highest`. Checkpoint SHA-256 `18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589`.

The clipped Gaussian family and seed 7090702 retain all 4 physical cases. Each configuration has 3 timing repetitions; final cases remain unopened. The inherited checkpoint was trained on 255 intervals and is evaluated at 256,512,1024 intervals.

The scalar equation is $u_t+u(u_x+u_y)=\nu\Delta u$ on the unit square, with zero Dirichlet walls. Both methods use backward Euler and sign-dependent upwinding. The FOM uses adaptive Newton/BiCGStab with exact Helmholtz preconditioning through FFT sine transforms.

The query starts with the supplied dense host field and viscosity, and ends with all requested dense host fields. Fixed Gauss sampling charges bilinear interpolation from that input. Weak evolution, sign-dependent upwinding and nonnegative advection quadrature remain unchanged. Initial fit, evolution, input/output transfers and requested dense reconstruction are included in query cost. Setup and compilation are separate.

Output times are [0, 0.05, 0.1, 0.15, 0.2, 0.25]. The common-grid metric observes 256 intervals; the complete-grid metric scores every node on the requested mesh. Both divide by the initial reference-field norm. Configuration cost is the median of per-case repetition medians. Each paired ratio is the median of per-case FOM/ROM median-time ratios from this job. GPU burn-in precedes every timed call, and the recorded configuration-order seed is 89004.

## Initial fitting and subsequent evolution

The table scores actual returned fields. Initial and later errors use the same initial-field normalization. A later error increase does not by itself isolate spatial representation, the nonlinear head, quadrature or optimization.

| Intervals | ROM setting | Query ms | Initial complete-grid worst | Later complete-grid worst | Complete-grid same-mesh worst | Current-relative complete-grid worst | IC budget / improvement stops | Evolution budget / improvement stops |
|---|---|---:|---:|---:|---:|---:|---|---|
| 256 | `rom_L256_edge_ic60_dt0.005_stall0.01_starts1` | 36.6762 | 0.026004511 | 0.045398754 | 0.026004511 | 0.058684175 | 3 / 9 | 0 / 600 |
| 256 | `rom_L256_fixed_gauss_ic180_dt0.0025_stall0.01_starts1` | 57.3590 | 0.025628708 | 0.041635828 | 0.025628708 | 0.052333088 | 0 / 12 | 0 / 1200 |
| 256 | `rom_L256_fixed_gauss_ic180_dt0.005_stall0.001_starts1` | 38.5150 | 0.025628708 | 0.045542267 | 0.025628708 | 0.058871669 | 0 / 12 | 0 / 600 |
| 256 | `rom_L256_fixed_gauss_ic180_dt0.005_stall0.01_starts1` | 36.9518 | 0.025628708 | 0.045507001 | 0.025628708 | 0.058806982 | 0 / 12 | 0 / 600 |
| 256 | `rom_L256_fixed_gauss_ic60_dt0.005_stall0.01_starts1` | 37.0179 | 0.025628708 | 0.045507001 | 0.025628708 | 0.058806982 | 3 / 9 | 0 / 600 |
| 512 | `rom_L512_edge_ic60_dt0.005_stall0.01_starts1` | 39.7608 | 0.05467109 | 0.043777122 | 0.05467109 | 0.05467109 | 3 / 9 | 0 / 600 |
| 512 | `rom_L512_fixed_gauss_ic180_dt0.0025_stall0.01_starts1` | 59.4750 | 0.032835851 | 0.034484122 | 0.032835851 | 0.043208555 | 0 / 12 | 0 / 1200 |
| 512 | `rom_L512_fixed_gauss_ic180_dt0.005_stall0.001_starts1` | 42.8585 | 0.032835851 | 0.037161855 | 0.032835851 | 0.046563751 | 0 / 12 | 0 / 600 |
| 512 | `rom_L512_fixed_gauss_ic180_dt0.005_stall0.01_starts1` | 40.0065 | 0.032835851 | 0.037133408 | 0.032835851 | 0.046528107 | 0 / 12 | 0 / 600 |
| 512 | `rom_L512_fixed_gauss_ic60_dt0.005_stall0.01_starts1` | 39.6757 | 0.032835851 | 0.037133408 | 0.032835851 | 0.046528107 | 3 / 9 | 0 / 600 |
| 1024 | `rom_L1024_edge_ic60_dt0.005_stall0.01_starts1` | 54.3917 | 0.086953369 | 0.066847426 | 0.086953369 | 0.086953369 | 3 / 9 | 0 / 600 |
| 1024 | `rom_L1024_fixed_gauss_ic180_dt0.0025_stall0.01_starts1` | 74.2636 | 0.0385622 | 0.037079305 | 0.0385622 | 0.042135143 | 0 / 12 | 0 / 1200 |
| 1024 | `rom_L1024_fixed_gauss_ic180_dt0.005_stall0.001_starts1` | 57.2498 | 0.0385622 | 0.038863444 | 0.0385622 | 0.043873796 | 0 / 12 | 0 / 600 |
| 1024 | `rom_L1024_fixed_gauss_ic180_dt0.005_stall0.01_starts1` | 54.9038 | 0.0385622 | 0.039076203 | 0.0385622 | 0.043874041 | 0 / 12 | 0 / 600 |
| 1024 | `rom_L1024_fixed_gauss_ic60_dt0.005_stall0.01_starts1` | 56.0849 | 0.0385622 | 0.039076203 | 0.0385622 | 0.043874041 | 3 / 9 | 0 / 600 |

## Reference refinement and empirical eligibility

The finest reference has 4096 intervals and timestep 0.0003125. The largest reference nonlinear relative residual is 9.252006e-12. Spatial and temporal orders are observed from three levels; the margin is the larger of their raw-difference sum and the Richardson estimate for each case. Qualification also requires this margin to be at most one tenth of the target. These estimates do not supply a rigorous reference bound.

| Scoring intervals | Worst empirical margin | Minimum / maximum spatial order | Minimum / maximum time order |
|---|---:|---|---|
| 256 | 0.0037366692 | 0.97372 / 0.99459 | 0.97016 / 0.99226 |
| 512 | 0.0037366692 | 0.97372 / 0.99459 | 0.97016 / 0.99226 |
| 1024 | 0.0037366692 | 0.97372 / 0.99459 | 0.97016 / 0.99226 |

Every eligible case must satisfy measured error plus its reference margin at the target. The FOM envelope retains all declared coarse solves and the same-mesh solve, charging output interpolation. Configurations may be selected separately for each metric and requested resolution.

| Norm | Output intervals | Target | Reference budget | Selected ROM | Selected FOM | ROM / FOM ms | Paired FOM/ROM |
|---|---:|---:|---|---|---|---|---:|
| common | 256 | 0.1 | empirical pass | `rom_L256_edge_ic60_dt0.005_stall0.01_starts1` | `fom_L128_out256_dt0.01_ntol0.01` | 36.6762 / 10.2767 | 0.29294 |
| common | 256 | 0.05 | empirical pass | `rom_L256_edge_ic60_dt0.005_stall0.01_starts1` | `fom_L256_out256_dt0.01_ntol0.01` | 36.6762 / 10.6652 | 0.30413 |
| common | 256 | 0.01 | unresolved | `unattained` | `unattained` | — / — | — |
| common | 512 | 0.1 | empirical pass | `rom_L512_fixed_gauss_ic60_dt0.005_stall0.01_starts1` | `fom_L128_out512_dt0.01_ntol0.01` | 39.6757 / 12.0976 | 0.31227 |
| common | 512 | 0.05 | empirical pass | `rom_L512_fixed_gauss_ic60_dt0.005_stall0.01_starts1` | `fom_L256_out512_dt0.01_ntol0.01` | 39.6757 / 12.2615 | 0.32117 |
| common | 512 | 0.01 | unresolved | `unattained` | `unattained` | — / — | — |
| common | 1024 | 0.1 | empirical pass | `rom_L1024_edge_ic60_dt0.005_stall0.01_starts1` | `fom_L128_out1024_dt0.01_ntol0.01` | 54.3917 / 24.6337 | 0.45953 |
| common | 1024 | 0.05 | empirical pass | `rom_L1024_fixed_gauss_ic180_dt0.005_stall0.01_starts1` | `fom_L256_out1024_dt0.01_ntol0.01` | 54.9038 / 25.5489 | 0.48201 |
| common | 1024 | 0.01 | unresolved | `unattained` | `unattained` | — / — | — |
| dense | 256 | 0.1 | empirical pass | `rom_L256_edge_ic60_dt0.005_stall0.01_starts1` | `fom_L128_out256_dt0.01_ntol0.01` | 36.6762 / 10.2767 | 0.29294 |
| dense | 256 | 0.05 | empirical pass | `rom_L256_edge_ic60_dt0.005_stall0.01_starts1` | `fom_L256_out256_dt0.01_ntol0.01` | 36.6762 / 10.6652 | 0.30413 |
| dense | 256 | 0.01 | unresolved | `unattained` | `unattained` | — / — | — |
| dense | 512 | 0.1 | empirical pass | `rom_L512_fixed_gauss_ic60_dt0.005_stall0.01_starts1` | `fom_L128_out512_dt0.01_ntol0.01` | 39.6757 / 12.0976 | 0.31227 |
| dense | 512 | 0.05 | empirical pass | `rom_L512_fixed_gauss_ic60_dt0.005_stall0.01_starts1` | `fom_L256_out512_dt0.01_ntol0.01` | 39.6757 / 12.2615 | 0.32117 |
| dense | 512 | 0.01 | unresolved | `unattained` | `unattained` | — / — | — |
| dense | 1024 | 0.1 | empirical pass | `rom_L1024_edge_ic60_dt0.005_stall0.01_starts1` | `fom_L128_out1024_dt0.01_ntol0.01` | 54.3917 / 24.6337 | 0.45953 |
| dense | 1024 | 0.05 | empirical pass | `rom_L1024_fixed_gauss_ic180_dt0.005_stall0.01_starts1` | `fom_L256_out1024_dt0.01_ntol0.01` | 54.9038 / 25.5489 | 0.48201 |
| dense | 1024 | 0.01 | unresolved | `unattained` | `unattained` | — / — | — |

![Complete-query accuracy and cost](accuracy-cost.svg)

## Where the primary Gauss error appears

The primary corrected setting uses the predeclared larger fitting budget with one start and the original loose evolution threshold. This table distinguishes an initial target miss from a miss that appears later. A configured improvement stop is not a stationary or globally optimal fit.

| Intervals | Case | Initial complete-grid error | Later complete-grid maximum | Time of maximum |
|---|---:|---:|---:|---:|
| 256 | 0 | 0.0083172873 | 0.012899595 | 0.25 |
| 256 | 1 | 0.01302074 | 0.013085103 | 0.05 |
| 256 | 2 | 0.015053412 | 0.045507001 | 0.15 |
| 256 | 3 | 0.025628708 | 0.028620634 | 0.25 |
| 512 | 0 | 0.012581488 | 0.0098268462 | 0 |
| 512 | 1 | 0.013015672 | 0.01281491 | 0 |
| 512 | 2 | 0.015055721 | 0.037133408 | 0.15 |
| 512 | 3 | 0.032835851 | 0.028292717 | 0 |
| 1024 | 0 | 0.0162891 | 0.010502314 | 0 |
| 1024 | 1 | 0.013028027 | 0.012954762 | 0 |
| 1024 | 2 | 0.01505597 | 0.035015236 | 0.15 |
| 1024 | 3 | 0.0385622 | 0.039076203 | 0.05 |

The next diagnostic uses the same saved primary fields. The near-wall band contains nodes closer to a wall than the training mesh first interior node. Its squared-error fraction locates the error; it does not independently measure a bank projection floor.

| Intervals | Case | Cold rule | Initial near-wall squared-error fraction | Target | Target diagnosis |
|---|---:|---|---:|---:|---|
| 256 | 0 | edge | 0.05412623 | 0.05 | measured error plus margin meets target |
| 256 | 0 | fixed_gauss | 0.179391 | 0.05 | measured error plus margin meets target |
| 256 | 1 | edge | 0.0105309 | 0.05 | measured error plus margin meets target |
| 256 | 1 | fixed_gauss | 0.01614473 | 0.05 | measured error plus margin meets target |
| 256 | 2 | edge | 0.008698409 | 0.05 | measured error plus margin meets target |
| 256 | 2 | fixed_gauss | 0.00889775 | 0.05 | measured error plus margin meets target |
| 256 | 3 | edge | 0.1202537 | 0.05 | measured error plus margin meets target |
| 256 | 3 | fixed_gauss | 0.1717976 | 0.05 | measured error plus margin meets target |
| 512 | 0 | edge | 0.1504951 | 0.05 | measured error plus margin meets target |
| 512 | 0 | fixed_gauss | 0.555251 | 0.05 | measured error plus margin meets target |
| 512 | 1 | edge | 0.01452451 | 0.05 | measured error plus margin meets target |
| 512 | 1 | fixed_gauss | 0.01158765 | 0.05 | measured error plus margin meets target |
| 512 | 2 | edge | 0.007215595 | 0.05 | measured error plus margin meets target |
| 512 | 2 | fixed_gauss | 0.00722158 | 0.05 | measured error plus margin meets target |
| 512 | 3 | edge | 0.1673018 | 0.05 | initial field already exceeds target |
| 512 | 3 | fixed_gauss | 0.4416153 | 0.05 | measured error plus margin meets target |
| 1024 | 0 | edge | 0.1549664 | 0.05 | initial field already exceeds target |
| 1024 | 0 | fixed_gauss | 0.7082176 | 0.05 | measured error plus margin meets target |
| 1024 | 1 | edge | 0.02539713 | 0.05 | measured error plus margin meets target |
| 1024 | 1 | fixed_gauss | 0.01117442 | 0.05 | measured error plus margin meets target |
| 1024 | 2 | edge | 0.006191251 | 0.05 | measured error plus margin meets target |
| 1024 | 2 | fixed_gauss | 0.006120309 | 0.05 | measured error plus margin meets target |
| 1024 | 3 | edge | 0.1520296 | 0.05 | initial field already exceeds target |
| 1024 | 3 | fixed_gauss | 0.5778798 | 0.05 | measured error plus margin meets target |

## Complete measured configuration set

| Configuration | Query ms | Common-grid median / worst | Complete-grid median / worst | Error outliers common / complete | Timing outliers | Failed calls |
|---|---:|---|---|---|---:|---:|
| `fom_L128_out256_dt0.01_ntol0.01` | 10.2767 | 0.02748078 / 0.05795444 | 0.02748078 / 0.05795444 | 4 / 4 | 0 | 0 |
| `fom_L256_out256_dt0.01_ntol0.01` | 10.6652 | 0.02004901 / 0.03616819 | 0.02004901 / 0.03616819 | 3 / 3 | 0 | 0 |
| `fom_L128_out256_dt0.01_ntol0.003` | 13.2736 | 0.03572886 / 0.07306105 | 0.03572886 / 0.07306105 | 4 / 4 | 0 | 0 |
| `fom_L256_out256_dt0.01_ntol0.003` | 14.3945 | 0.02880155 / 0.05054089 | 0.02880155 / 0.05054089 | 3 / 3 | 0 | 0 |
| `fom_L128_out256_dt0.005_ntol0.01` | 16.5497 | 0.02748078 / 0.05482779 | 0.02748078 / 0.05482779 | 3 / 3 | 0 | 0 |
| `fom_L256_out256_dt0.005_ntol0.01` | 17.1126 | 0.01305757 / 0.02473687 | 0.01305757 / 0.02473687 | 3 / 3 | 0 | 0 |
| `fom_L128_out256_dt0.005_ntol0.003` | 17.7089 | 0.02748078 / 0.05811261 | 0.02748078 / 0.05811261 | 4 / 4 | 0 | 0 |
| `fom_L256_out256_dt0.005_ntol0.003` | 18.1845 | 0.01230123 / 0.02943456 | 0.01230123 / 0.02943456 | 2 / 2 | 0 | 0 |
| `fom_L128_out256_dt0.0025_ntol0.003` | 30.1503 | 0.02777728 / 0.05708146 | 0.02777728 / 0.05708146 | 3 / 3 | 0 | 0 |
| `fom_L256_out256_dt0.0025_ntol0.003` | 31.6822 | 0.01172413 / 0.02721718 | 0.01172413 / 0.02721718 | 2 / 2 | 0 | 0 |
| `rom_L256_edge_ic60_dt0.005_stall0.01_starts1` | 36.6762 | 0.02075496 / 0.04539875 | 0.02075496 / 0.04539875 | 4 / 4 | 0 | 0 |
| `rom_L256_fixed_gauss_ic180_dt0.005_stall0.01_starts1` | 36.9518 | 0.02085287 / 0.045507 | 0.02085287 / 0.045507 | 4 / 4 | 0 | 0 |
| `rom_L256_fixed_gauss_ic60_dt0.005_stall0.01_starts1` | 37.0179 | 0.02085287 / 0.045507 | 0.02085287 / 0.045507 | 4 / 4 | 0 | 0 |
| `rom_L256_fixed_gauss_ic180_dt0.005_stall0.001_starts1` | 38.5150 | 0.02094475 / 0.04554227 | 0.02094475 / 0.04554227 | 4 / 4 | 0 | 0 |
| `rom_L256_fixed_gauss_ic180_dt0.0025_stall0.01_starts1` | 57.3590 | 0.01932472 / 0.04163583 | 0.01932472 / 0.04163583 | 4 / 4 | 0 | 0 |
| `fom_L128_out512_dt0.01_ntol0.01` | 12.0976 | 0.02748078 / 0.05795444 | 0.03634001 / 0.05798938 | 4 / 4 | 0 | 0 |
| `fom_L256_out512_dt0.01_ntol0.01` | 12.2615 | 0.02004901 / 0.03616819 | 0.02217648 / 0.03627495 | 3 / 4 | 0 | 0 |
| `fom_L128_out512_dt0.01_ntol0.003` | 15.1953 | 0.03572886 / 0.07306105 | 0.03686974 / 0.07311454 | 4 / 4 | 0 | 0 |
| `fom_L256_out512_dt0.01_ntol0.003` | 15.7169 | 0.02880155 / 0.05054089 | 0.02880847 / 0.05059877 | 3 / 4 | 0 | 0 |
| `fom_L512_out512_dt0.01_ntol0.01` | 16.5284 | 0.02072672 / 0.03304613 | 0.02071964 / 0.03304613 | 3 / 3 | 0 | 0 |
| `fom_L128_out512_dt0.005_ntol0.01` | 18.3160 | 0.02748078 / 0.05482779 | 0.03634001 / 0.05487062 | 3 / 3 | 0 | 0 |
| `fom_L256_out512_dt0.005_ntol0.01` | 18.9579 | 0.01305757 / 0.02473687 | 0.01921324 / 0.02520692 | 3 / 4 | 0 | 0 |
| `fom_L128_out512_dt0.005_ntol0.003` | 19.1792 | 0.02748078 / 0.05811261 | 0.03634001 / 0.05815377 | 4 / 4 | 0 | 0 |
| `fom_L256_out512_dt0.005_ntol0.003` | 20.1258 | 0.01230123 / 0.02943456 | 0.0194265 / 0.02947628 | 2 / 3 | 0 | 0 |
| `fom_L512_out512_dt0.01_ntol0.003` | 22.2445 | 0.02541756 / 0.04168595 | 0.02540853 / 0.04168595 | 3 / 3 | 0 | 0 |
| `fom_L512_out512_dt0.005_ntol0.01` | 26.6977 | 0.01215344 / 0.02063806 | 0.01214955 / 0.02063806 | 3 / 3 | 0 | 0 |
| `fom_L512_out512_dt0.005_ntol0.003` | 28.7093 | 0.00982102 / 0.01915143 | 0.009817653 / 0.01915143 | 2 / 2 | 0 | 0 |
| `fom_L128_out512_dt0.0025_ntol0.003` | 31.9253 | 0.02777728 / 0.05708146 | 0.03634001 / 0.0571262 | 3 / 3 | 0 | 0 |
| `fom_L256_out512_dt0.0025_ntol0.003` | 33.0502 | 0.01172413 / 0.02721718 | 0.0194265 / 0.02726691 | 2 / 3 | 0 | 0 |
| `fom_L512_out512_dt0.0025_ntol0.003` | 48.3475 | 0.006124359 / 0.01134245 | 0.006122112 / 0.01134245 | 1 / 1 | 0 | 0 |
| `rom_L512_fixed_gauss_ic60_dt0.005_stall0.01_starts1` | 39.6757 | 0.02066393 / 0.03713245 | 0.02292576 / 0.03713341 | 3 / 4 | 0 | 0 |
| `rom_L512_edge_ic60_dt0.005_stall0.01_starts1` | 39.7608 | 0.0311749 / 0.05274055 | 0.03139523 / 0.05467109 | 4 / 4 | 0 | 0 |
| `rom_L512_fixed_gauss_ic180_dt0.005_stall0.01_starts1` | 40.0065 | 0.02066393 / 0.03713245 | 0.02292576 / 0.03713341 | 3 / 4 | 0 | 0 |
| `rom_L512_fixed_gauss_ic180_dt0.005_stall0.001_starts1` | 42.8585 | 0.02051402 / 0.0371609 | 0.02292576 / 0.03716185 | 3 / 4 | 0 | 0 |
| `rom_L512_fixed_gauss_ic180_dt0.0025_stall0.01_starts1` | 59.4750 | 0.01958492 / 0.03448294 | 0.02292576 / 0.03448412 | 3 / 4 | 0 | 0 |
| `fom_L128_out1024_dt0.01_ntol0.01` | 24.6337 | 0.02748078 / 0.05795444 | 0.04062111 / 0.0579981 | 4 / 4 | 0 | 0 |
| `fom_L256_out1024_dt0.01_ntol0.01` | 25.5489 | 0.02004901 / 0.03616819 | 0.02623869 / 0.03630153 | 3 / 4 | 0 | 0 |
| `fom_L128_out1024_dt0.01_ntol0.003` | 27.8684 | 0.03572886 / 0.07306105 | 0.04062111 / 0.07312789 | 4 / 4 | 0 | 0 |
| `fom_L256_out1024_dt0.01_ntol0.003` | 28.4307 | 0.02880155 / 0.05054089 | 0.02880742 / 0.05061321 | 3 / 4 | 0 | 0 |
| `fom_L512_out1024_dt0.01_ntol0.01` | 29.7891 | 0.02072672 / 0.03304613 | 0.02070701 / 0.03307223 | 3 / 3 | 0 | 0 |
| `fom_L128_out1024_dt0.005_ntol0.01` | 31.0251 | 0.02748078 / 0.05482779 | 0.04062111 / 0.0548813 | 3 / 3 | 0 | 0 |
| `fom_L256_out1024_dt0.005_ntol0.01` | 31.5239 | 0.01305757 / 0.02473687 | 0.02142133 / 0.03333581 | 3 / 4 | 0 | 0 |
| `fom_L128_out1024_dt0.005_ntol0.003` | 31.9815 | 0.02748078 / 0.05811261 | 0.04062111 / 0.05816404 | 4 / 4 | 0 | 0 |
| `fom_L256_out1024_dt0.005_ntol0.003` | 32.5404 | 0.01230123 / 0.02943456 | 0.02376906 / 0.03333581 | 2 / 3 | 0 | 0 |
| `fom_L512_out1024_dt0.01_ntol0.003` | 35.5774 | 0.02541756 / 0.04168595 | 0.0254078 / 0.04170888 | 3 / 3 | 0 | 0 |
| `fom_L512_out1024_dt0.005_ntol0.01` | 39.4839 | 0.01215344 / 0.02063806 | 0.01504179 / 0.02061413 | 3 / 3 | 0 | 0 |
| `fom_L512_out1024_dt0.005_ntol0.003` | 40.5228 | 0.00982102 / 0.01915143 | 0.01373429 / 0.01917962 | 2 / 2 | 0 | 0 |
| `fom_L128_out1024_dt0.0025_ntol0.003` | 44.4563 | 0.02777728 / 0.05708146 | 0.04062111 / 0.05713736 | 3 / 3 | 0 | 0 |
| `fom_L256_out1024_dt0.0025_ntol0.003` | 45.8565 | 0.01172413 / 0.02721718 | 0.02266537 / 0.03333581 | 2 / 3 | 0 | 0 |
| `fom_L1024_out1024_dt0.01_ntol0.01` | 48.7969 | 0.02173782 / 0.02891918 | 0.0217264 / 0.02891918 | 4 / 4 | 0 | 0 |
| `fom_L512_out1024_dt0.0025_ntol0.003` | 61.4547 | 0.006124359 / 0.01134245 | 0.01050156 / 0.017819 | 1 / 2 | 0 | 0 |
| `fom_L1024_out1024_dt0.01_ntol0.003` | 65.0843 | 0.02391871 / 0.0381257 | 0.02390672 / 0.0381257 | 3 / 3 | 0 | 0 |
| `fom_L1024_out1024_dt0.005_ntol0.01` | 74.9674 | 0.01326684 / 0.02389937 | 0.01326041 / 0.02389937 | 3 / 3 | 0 | 0 |
| `fom_L1024_out1024_dt0.005_ntol0.003` | 79.1325 | 0.01050882 / 0.01656559 | 0.01050334 / 0.01656559 | 2 / 2 | 0 | 0 |
| `fom_L1024_out1024_dt0.0025_ntol0.003` | 131.6614 | 0.00625118 / 0.01048777 | 0.006248106 / 0.01048777 | 1 / 1 | 0 | 0 |
| `rom_L1024_edge_ic60_dt0.005_stall0.01_starts1` | 54.3917 | 0.04266856 / 0.08497164 | 0.04275201 / 0.08695337 | 4 / 4 | 0 | 0 |
| `rom_L1024_fixed_gauss_ic180_dt0.005_stall0.01_starts1` | 54.9038 | 0.02401685 / 0.03910892 | 0.02565217 / 0.0390762 | 4 / 4 | 0 | 0 |
| `rom_L1024_fixed_gauss_ic60_dt0.005_stall0.01_starts1` | 56.0849 | 0.02401685 / 0.03910892 | 0.02578724 / 0.0390762 | 4 / 4 | 0 | 0 |
| `rom_L1024_fixed_gauss_ic180_dt0.005_stall0.001_starts1` | 57.2498 | 0.02401675 / 0.03889607 | 0.02565207 / 0.03886344 | 4 / 4 | 0 | 0 |
| `rom_L1024_fixed_gauss_ic180_dt0.0025_stall0.01_starts1` | 74.2636 | 0.02332282 / 0.03710934 | 0.02495827 / 0.0385622 | 3 / 4 | 0 | 0 |

## Runtime components and setup

Component timings are separately synchronized diagnostic invocations with field parity against the fused query. They do not replace complete-query timing or establish kernel-launch counts.

| Intervals | Cold rule / budget | Input ms | Initial fit ms | Evolution ms | Decode ms | Output ms | Staged total ms | LM attempts |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | edge / 60 | 0.3772 | 4.4698 | 29.6202 | 0.4507 | 0.8094 | 36.9800 | 134.5 |
| 256 | fixed_gauss / 180 | 0.3783 | 4.3283 | 30.5095 | 0.3388 | 0.7960 | 36.8133 | 137.5 |
| 512 | edge / 60 | 0.6616 | 4.2567 | 29.4065 | 0.9651 | 2.3498 | 39.4365 | 133.0 |
| 512 | fixed_gauss / 180 | 0.7158 | 4.2158 | 30.2155 | 0.8549 | 2.2790 | 39.5054 | 137.0 |
| 1024 | edge / 60 | 1.8101 | 4.7388 | 31.2849 | 2.8484 | 13.6647 | 51.9814 | 140.5 |
| 1024 | fixed_gauss / 180 | 2.3538 | 4.2871 | 30.2663 | 2.8709 | 11.2916 | 54.7053 | 134.5 |

| Intervals | Sampled rank | K / R / M / m | Bank/operator setup s | Gauss setup s | Compilation/warmup s |
|---|---:|---|---:|---:|---:|
| 256 | 512 | 16 / 512 / 64 / 256 | 29.6502 | 2.9875 | 27.6207 |
| 512 | 512 | 16 / 512 / 64 / 256 | 23.7956 | 1.1035 | 28.9022 |
| 1024 | 512 | 16 / 512 / 64 / 256 | 25.1429 | 1.0828 | 34.6809 |

The dominant measured reduced phase is evolution. A bounded larger-timestep study is the next controlled cost test, keeping the fitted initializer, checkpoint, field family and FOM envelope fixed. Halving the timestep and tightening the evolution stopping threshold are already measured controls above; neither proves global or stationary optimization. Tighter accuracy also remains sensitive to the initial-fit error, especially on the finest mesh.

## Audit and scope

The independent NumPy audit checked 720 invocations across 60 configurations and 240 saved complete-grid artifacts. Maximum error-recomputation discrepancies are 4.16334e-17 on the common grid and 4.16334e-17 on complete grids. It found 0 FOM tolerance failures and 0 nonfinite invocations. 720 of 720 recorded dense-output hashes match their corresponding first repetitions.

Source, checkpoint and closed-output checksums are retained. Every repeated output is checked against an actual complete-grid artifact. Initial clipping and physical cases are preserved. Independent confirmation, rigorous reference bounds and per-resolution weight retraining remain open. The historical cold-only bank-floor diagnostics are not substituted for the returned rollout errors.

## Plain-language glossary

- **Intervals / complete grid / common grid:** cells per axis / every requested output node / fixed observation nodes shared by all resolutions.
- **FOM / ROM / frozen checkpoint:** full spatial model / nonlinear-manifold reduced model / saved neural weights kept unchanged.
- **Bank / head / IC:** learned coordinate-dependent spatial features / nonlinear map from latent coordinates to feature coefficients / initial condition.
- **Newton / BiCGStab / Helmholtz preconditioning / FFT:** nonlinear iteration / iterative linear solution / inversion of the diffusion-plus-identity part to help that solve / fast Fourier transform.
- **Gauss / edge / QR / cold:** fixed physical Gaussian quadrature / original mesh-dependent initial samples / orthogonal triangular factorization / initial reduced-state fitting.
- **K / R / M / m / sampled rank:** latent coordinates / learned spatial features / smooth weak tests / advection quadrature nodes / independently resolved bank directions on deterministic sampled rows.
- **Query ms / case median / paired FOM/ROM:** complete host-input-to-host-output latency in milliseconds / middle timing repetition for one physical case / casewise full-model time divided by reduced-model time, then a cohort median.
- **Initial / later / median / worst error:** first returned field / remaining output times / middle case error / largest case error; each is divided by the corresponding initial reference norm.
- **Current-relative error:** discrepancy divided by the current reference field norm, which exposes relative error as the field decays; it is a diagnostic separate from the initial-norm eligibility target.
- **Same-mesh error:** discrepancy from a tightly converged full solve using the same mesh and timestep.
- **Budget / improvement / failed stops:** configured trial cap / small-step or relative-improvement exit / rejected or nonfinite exit. The first two do not prove stationarity.
- **Empirical margin / Richardson / observed order / target / reference budget:** estimated reference error / extrapolation assuming the observed convergence rate continues / that rate inferred from three levels / requested accuracy / allowed fraction of target consumed by reference uncertainty.
- **Envelope / selected / unattained:** least-cost eligible candidate including coarse FOM interpolation / chosen development configuration / no tested configuration meets the stated error condition.
- **Weak evolution / sign-upwind / nonnegative quadrature:** PDE equations averaged against smooth functions / local-sign-selected differences / positive weighted sampling of advection.
- **Input / initial fit / evolution / decode / output / staged total / LM attempts:** device transfer / initial latent optimization / reduced timestepping / dense reconstruction / host transfer / sum of diagnostic phases / actual Levenberg–Marquardt trial steps.
- **Error / timing outliers:** physical cases above relative error 0.01 / calls taking more than twice that case’s timing median. All outliers remain in the results.
- **Near-wall squared-error fraction:** share of total squared field discrepancy at nodes closer to the boundary than the first interior node of the training mesh.
- **Setup / compilation / artifact / hash / parity:** reusable numerical preparation / executable preparation / saved numerical file / content fingerprint / agreement of two implementations or outputs.
- **Validation / final / f64 / highest / backend:** development cases / reserved independent cases / double precision / required matrix arithmetic precision / execution device type.
