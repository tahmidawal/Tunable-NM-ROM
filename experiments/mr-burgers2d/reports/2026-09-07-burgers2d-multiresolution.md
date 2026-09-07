# Burgers 2D frozen-network resolution transfer and complete-query cost

Development results; the frozen-network ROM does not establish a complete-query advantage over the efficient FOM envelope. All tables are generated from saved invocation records, and physical-error qualifications remain empirical rather than certified.

Source commit `9a2025c1f0624db91fb8ecbcfef0d0d433373187`, job `3350134`, GPU `NVIDIA A100 80GB PCIe`. Backend `gpu`, f64 `True`, matmul precision `highest`. Checkpoint SHA-256 `18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589`.

The frozen checkpoint was trained on 255 intervals (256 nodes per axis). The new validation seed is 7090702, with 4 physical cases; the final cohort is unopened.

The query starts with a dense initial field in host memory and ends with all requested dense fields in host memory. Input handling, cold fitting, evolution, output reconstruction/interpolation and transfers are included. Compilation and reusable setup are separate. FOM candidates use sign-upwind backward Euler with adaptive Newton/BiCGStab and FFT sine-transform Helmholtz preconditioning. ROM quadrature retains sign-dependent upwinding on decoded undershoots.

## Reference refinement

The finest reference uses 4096 intervals and timestep 0.0003125. The largest final nonlinear relative residual over all saved reference solves is 9.25201e-12.

| Case | Spatial difference | Time difference | Sum estimate |
|---|---:|---:|---:|
| 0 | 0.000598323 | 0.000378638 | 0.000976961 |
| 1 | 0.000349528 | 0.000869916 | 0.00121944 |
| 2 | 0.00226861 | 0.00133939 | 0.003608 |
| 3 | 0.00137635 | 0.000889014 | 0.00226536 |

| Case | Observed spatial order | Observed time order | Empirical Richardson estimate |
|---|---:|---:|---:|
| 0 | 0.9938 | 0.9923 | 0.000985155 |
| 1 | 0.9946 | 0.9852 | 0.00123919 |
| 2 | 0.9737 | 0.9702 | 0.00373667 |
| 3 | 0.9868 | 0.9820 | 0.0023078 |

The worst empirical additive margin is 0.00373667; it passes the reference budget for target 0.05 and does not pass the budget for target 0.01. Eligibility additionally requires the measured case error plus this margin to meet the target.

These are differences between independently converged refinement levels, not rigorous continuum-error bounds. The physical norm uses exact nested-node restriction onto the observation grid and the initial-state reference norm. Current-field normalization is reported separately.

## Mesh setup

| Intervals | Interior unknowns | K / R / M / m | Sampled bank rank | Setup s | Stored arrays MiB | Quadrature fit |
|---|---:|---|---:|---:|---:|---:|
| 256 | 65025 | 16 / 512 / 64 / 256 | 512 | 30.848 | 303.477 | 0.0051591 |
| 512 | 261121 | 16 / 512 / 64 / 256 | 512 | 25.041 | 1069.477 | 0.00515699 |
| 1024 | 1046529 | 16 / 512 / 64 / 256 | 512 | 26.242 | 4137.477 | 0.0073836 |

## Measured configurations

![Complete-query accuracy and cost](accuracy-cost.svg)

Errors are maximum-in-observation-time values, aggregated over every declared case and repetition. A configured budget exit is retained as an early stop; a small-step/relative-improvement stop is not a stationarity certificate. Failed/nonfinite solver stops are excluded from target eligibility. Outliers count cases above relative error 0.01. Failed invocations are retained. Wall times are medians over all cases/repetitions in the same job.

| Configuration | Query ms | Physical median | Physical worst | Current-relative worst | Same-grid worst | Outliers | Failed invocations | IC budget / stall | LM budget / stall |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `fom_L128_out256_dt0.01_ntol0.01` | 9.975 | 0.0274808 | 0.0579544 | 0.0819259 | — | 4 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out256_dt0.01_ntol0.01` | 10.359 | 0.020049 | 0.0361682 | 0.0423984 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out256_dt0.01_ntol0.003` | 13.311 | 0.0357289 | 0.073061 | 0.102169 | — | 4 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out256_dt0.01_ntol0.003` | 14.081 | 0.0288016 | 0.0505409 | 0.0702504 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out256_dt0.005_ntol0.01` | 16.639 | 0.0274808 | 0.0548278 | 0.077506 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out256_dt0.005_ntol0.01` | 17.190 | 0.0130576 | 0.0247369 | 0.0349687 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out256_dt0.005_ntol0.003` | 17.660 | 0.0274808 | 0.0581126 | 0.0821495 | — | 4 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out256_dt0.005_ntol0.003` | 18.302 | 0.0123012 | 0.0294346 | 0.0416095 | — | 2 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out256_dt0.0025_ntol0.003` | 30.405 | 0.0277773 | 0.0570815 | 0.0806919 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out256_dt0.0025_ntol0.003` | 31.325 | 0.0117241 | 0.0272172 | 0.0384749 | — | 2 / 4 | 0 | 0 / 0 | 0 / 0 |
| `rom_L256_dt0.005_stall0.01_starts1` | 36.418 | 0.020755 | 0.0453988 | 0.0586842 | 0.0260045 | 4 / 4 | 0 | 3 / 9 | 0 / 600 |
| `rom_L256_dt0.005_stall0.01_starts4` | 38.717 | 0.0208767 | 0.0453989 | 0.0586842 | 0.0260048 | 4 / 4 | 0 | 0 / 12 | 0 / 600 |
| `rom_L256_dt0.005_stall0.001_starts1` | 38.967 | 0.0209095 | 0.0454302 | 0.0587421 | 0.0260045 | 4 / 4 | 0 | 3 / 9 | 0 / 600 |
| `rom_L256_dt0.005_stall0.001_starts4` | 41.193 | 0.0210266 | 0.0454303 | 0.0587422 | 0.0260048 | 4 / 4 | 0 | 0 / 12 | 0 / 600 |
| `rom_L256_dt0.0025_stall0.01_starts1` | 56.427 | 0.0195388 | 0.0415383 | 0.0522023 | 0.0260045 | 4 / 4 | 0 | 3 / 9 | 0 / 1200 |
| `rom_L256_dt0.0025_stall0.001_starts1` | 58.129 | 0.0195388 | 0.0415534 | 0.0522256 | 0.0260045 | 4 / 4 | 0 | 3 / 9 | 0 / 1200 |
| `rom_L256_dt0.0025_stall0.01_starts4` | 58.999 | 0.0195389 | 0.0415385 | 0.0522023 | 0.0260048 | 4 / 4 | 0 | 0 / 12 | 0 / 1200 |
| `rom_L256_dt0.0025_stall0.001_starts4` | 60.286 | 0.0195389 | 0.0415535 | 0.0522257 | 0.0260048 | 4 / 4 | 0 | 0 / 12 | 0 / 1200 |
| `fom_L256_out512_dt0.01_ntol0.01` | 12.140 | 0.020049 | 0.0361682 | 0.0423984 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out512_dt0.01_ntol0.01` | 12.236 | 0.0274808 | 0.0579544 | 0.0819259 | — | 4 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out512_dt0.01_ntol0.003` | 15.540 | 0.0357289 | 0.073061 | 0.102169 | — | 4 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out512_dt0.01_ntol0.003` | 15.825 | 0.0288016 | 0.0505409 | 0.0702504 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L512_out512_dt0.01_ntol0.01` | 16.739 | 0.0207267 | 0.0330461 | 0.0385881 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out512_dt0.005_ntol0.01` | 18.713 | 0.0274808 | 0.0548278 | 0.077506 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out512_dt0.005_ntol0.01` | 19.202 | 0.0130576 | 0.0247369 | 0.0349687 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out512_dt0.005_ntol0.003` | 19.801 | 0.0274808 | 0.0581126 | 0.0821495 | — | 4 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out512_dt0.005_ntol0.003` | 20.006 | 0.0123012 | 0.0294346 | 0.0416095 | — | 2 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L512_out512_dt0.01_ntol0.003` | 22.841 | 0.0254176 | 0.0416859 | 0.0527866 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L512_out512_dt0.005_ntol0.01` | 27.059 | 0.0121534 | 0.0206381 | 0.0240991 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L512_out512_dt0.005_ntol0.003` | 28.524 | 0.00982102 | 0.0191514 | 0.0220206 | — | 2 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out512_dt0.0025_ntol0.003` | 32.382 | 0.0277773 | 0.0570815 | 0.0806919 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out512_dt0.0025_ntol0.003` | 32.951 | 0.0117241 | 0.0272172 | 0.0384749 | — | 2 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L512_out512_dt0.0025_ntol0.003` | 48.840 | 0.00612436 | 0.0113424 | 0.016034 | — | 1 / 4 | 0 | 0 / 0 | 0 / 0 |
| `rom_L512_dt0.005_stall0.01_starts1` | 39.980 | 0.0311749 | 0.0527405 | 0.0527405 | 0.0527405 | 4 / 4 | 0 | 3 / 9 | 0 / 600 |
| `rom_L512_dt0.005_stall0.01_starts4` | 42.713 | 0.0310307 | 0.0527411 | 0.0527411 | 0.0527411 | 4 / 4 | 0 | 0 / 12 | 0 / 600 |
| `rom_L512_dt0.005_stall0.001_starts1` | 43.232 | 0.0311896 | 0.0527405 | 0.0527405 | 0.0527405 | 4 / 4 | 0 | 3 / 9 | 0 / 600 |
| `rom_L512_dt0.005_stall0.001_starts4` | 45.100 | 0.0310454 | 0.0527411 | 0.0527411 | 0.0527411 | 4 / 4 | 0 | 0 / 12 | 0 / 600 |
| `rom_L512_dt0.0025_stall0.01_starts1` | 60.091 | 0.0298578 | 0.0527405 | 0.0527405 | 0.0527405 | 4 / 4 | 0 | 3 / 9 | 0 / 1200 |
| `rom_L512_dt0.0025_stall0.001_starts1` | 60.463 | 0.0298646 | 0.0527405 | 0.0527405 | 0.0527405 | 4 / 4 | 0 | 3 / 9 | 0 / 1200 |
| `rom_L512_dt0.0025_stall0.01_starts4` | 62.342 | 0.0297138 | 0.0527411 | 0.0527411 | 0.0527411 | 4 / 4 | 0 | 0 / 12 | 0 / 1200 |
| `rom_L512_dt0.0025_stall0.001_starts4` | 62.705 | 0.0297206 | 0.0527411 | 0.0527411 | 0.0527411 | 4 / 4 | 0 | 0 / 12 | 0 / 1200 |
| `fom_L128_out1024_dt0.01_ntol0.01` | 24.969 | 0.0274808 | 0.0579544 | 0.0819259 | — | 4 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out1024_dt0.01_ntol0.01` | 25.047 | 0.020049 | 0.0361682 | 0.0423984 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out1024_dt0.01_ntol0.003` | 28.401 | 0.0357289 | 0.073061 | 0.102169 | — | 4 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out1024_dt0.01_ntol0.003` | 28.517 | 0.0288016 | 0.0505409 | 0.0702504 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L512_out1024_dt0.01_ntol0.01` | 29.686 | 0.0207267 | 0.0330461 | 0.0385881 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out1024_dt0.005_ntol0.01` | 31.251 | 0.0274808 | 0.0548278 | 0.077506 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out1024_dt0.005_ntol0.01` | 31.698 | 0.0130576 | 0.0247369 | 0.0349687 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out1024_dt0.005_ntol0.003` | 32.535 | 0.0123012 | 0.0294346 | 0.0416095 | — | 2 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out1024_dt0.005_ntol0.003` | 32.569 | 0.0274808 | 0.0581126 | 0.0821495 | — | 4 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L512_out1024_dt0.01_ntol0.003` | 35.548 | 0.0254176 | 0.0416859 | 0.0527866 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L512_out1024_dt0.005_ntol0.01` | 39.647 | 0.0121534 | 0.0206381 | 0.0240991 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L512_out1024_dt0.005_ntol0.003` | 41.455 | 0.00982102 | 0.0191514 | 0.0220206 | — | 2 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out1024_dt0.0025_ntol0.003` | 45.089 | 0.0277773 | 0.0570815 | 0.0806919 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out1024_dt0.0025_ntol0.003` | 46.198 | 0.0117241 | 0.0272172 | 0.0384749 | — | 2 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L1024_out1024_dt0.01_ntol0.01` | 48.842 | 0.0217378 | 0.0289192 | 0.033769 | — | 4 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L512_out1024_dt0.0025_ntol0.003` | 61.829 | 0.00612436 | 0.0113424 | 0.016034 | — | 1 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L1024_out1024_dt0.01_ntol0.003` | 64.619 | 0.0239187 | 0.0381257 | 0.0451372 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L1024_out1024_dt0.005_ntol0.01` | 73.728 | 0.0132668 | 0.0238994 | 0.0279074 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L1024_out1024_dt0.005_ntol0.003` | 77.762 | 0.0105088 | 0.0165656 | 0.0180387 | — | 2 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L1024_out1024_dt0.0025_ntol0.003` | 128.985 | 0.00625118 | 0.0104878 | 0.0122466 | — | 1 / 4 | 0 | 0 / 0 | 0 / 0 |
| `rom_L1024_dt0.005_stall0.01_starts1` | 55.544 | 0.0426686 | 0.0849716 | 0.0849716 | 0.0849716 | 4 / 4 | 0 | 3 / 9 | 0 / 600 |
| `rom_L1024_dt0.005_stall0.01_starts4` | 57.045 | 0.042426 | 0.0849689 | 0.0849689 | 0.0849689 | 4 / 4 | 0 | 0 / 12 | 0 / 600 |
| `rom_L1024_dt0.005_stall0.001_starts1` | 57.356 | 0.0426685 | 0.0849716 | 0.0849716 | 0.0849716 | 4 / 4 | 0 | 3 / 9 | 0 / 600 |
| `rom_L1024_dt0.005_stall0.001_starts4` | 59.487 | 0.0424259 | 0.0849689 | 0.0849689 | 0.0849689 | 4 / 4 | 0 | 0 / 12 | 0 / 600 |
| `rom_L1024_dt0.0025_stall0.01_starts1` | 74.549 | 0.0419788 | 0.0849716 | 0.0849716 | 0.0849716 | 4 / 4 | 0 | 3 / 9 | 0 / 1200 |
| `rom_L1024_dt0.0025_stall0.001_starts1` | 76.541 | 0.0419846 | 0.0849716 | 0.0849716 | 0.0849716 | 4 / 4 | 0 | 3 / 9 | 0 / 1200 |
| `rom_L1024_dt0.0025_stall0.01_starts4` | 76.672 | 0.0417366 | 0.0849689 | 0.0849689 | 0.0849689 | 4 / 4 | 0 | 0 / 12 | 0 / 1200 |
| `rom_L1024_dt0.0025_stall0.001_starts4` | 78.056 | 0.0417423 | 0.0849689 | 0.0849689 | 0.0849689 | 4 / 4 | 0 | 0 / 12 | 0 / 1200 |

## Target-qualified validation selections

Both methods may choose a configuration within the declared pilot search, using the pooled median over all case/repetition query times shown above. This native selection statistic differs from taking the median of per-case medians; the final speed ratio itself uses per-case paired medians. Eligibility includes the measured reference estimate as an additive error margin. The margin is the larger of the raw spatial-plus-time difference and the empirical Richardson estimate. The FOM envelope includes coarser solves with aligned interpolation to the same requested dense output. Solver choices are tuned per resolution; both neural networks remain frozen. This does not test retraining benefits.

| Output intervals | Target | Qualification | Selected ROM | Selected FOM envelope | Envelope speedup |
|---|---:|---|---|---|---:|
| 256 | 0.1 | provisional refinement budget passed | rom_L256_dt0.005_stall0.01_starts1 | fom_L128_out256_dt0.01_ntol0.01 | 0.275x |
| 256 | 0.05 | provisional refinement budget passed | rom_L256_dt0.005_stall0.01_starts1 | fom_L256_out256_dt0.01_ntol0.01 | 0.287x |
| 256 | 0.01 | unresolved reference uncertainty/order; ROM target unattained; FOM target unattained | — | — | — |
| 256 | 0.001 | unresolved reference uncertainty/order; ROM target unattained; FOM target unattained | — | — | — |
| 512 | 0.1 | provisional refinement budget passed | rom_L512_dt0.005_stall0.01_starts1 | fom_L256_out512_dt0.01_ntol0.01 | 0.324x |
| 512 | 0.05 | provisional refinement budget passed; ROM target unattained | — | fom_L256_out512_dt0.01_ntol0.01 | — |
| 512 | 0.01 | unresolved reference uncertainty/order; ROM target unattained; FOM target unattained | — | — | — |
| 512 | 0.001 | unresolved reference uncertainty/order; ROM target unattained; FOM target unattained | — | — | — |
| 1024 | 0.1 | provisional refinement budget passed | rom_L1024_dt0.005_stall0.01_starts1 | fom_L128_out1024_dt0.01_ntol0.01 | 0.456x |
| 1024 | 0.05 | provisional refinement budget passed; ROM target unattained | — | fom_L256_out1024_dt0.01_ntol0.01 | — |
| 1024 | 0.01 | unresolved reference uncertainty/order; ROM target unattained; FOM target unattained | — | — | — |
| 1024 | 0.001 | unresolved reference uncertainty/order; ROM target unattained; FOM target unattained | — | — | — |

## Runtime components

Separate synchronized component invocations are diagnostic; their timings never replace the fused complete-query measurements. Each phase record comes from the same staged invocation and its field output was checked against the fused function.

| Intervals | Cold starts | Input ms | Cold fit ms | Evolve ms | Dense decode ms | Output ms | Staged total ms | LM attempts median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 0.484 | 4.273 | 31.984 | 0.519 | 0.932 | 39.624 | 146.5 |
| 256 | 4 | 0.555 | 6.762 | 32.492 | 0.540 | 0.887 | 42.076 | 146.5 |
| 512 | 1 | 1.090 | 4.692 | 32.513 | 1.126 | 2.655 | 44.215 | 145.5 |
| 512 | 4 | 0.917 | 6.583 | 33.528 | 1.062 | 2.394 | 45.752 | 152.5 |
| 1024 | 1 | 1.886 | 5.871 | 33.812 | 3.019 | 14.528 | 59.593 | 154.0 |
| 1024 | 4 | 1.961 | 6.837 | 33.851 | 2.919 | 14.318 | 60.251 | 157.0 |

LM attempts count reduced-solver trial steps, not GPU kernel launches; kernel launch counts were not profiled.

The independent query audit recomputed 828 invocation errors with maximum discrepancy 4.16334e-17. It found 0 FOM nonlinear-tolerance failures and 0 failed/nonfinite ROM evolution stops. The configured initial-fit budget exits remain visible in the table.

## Frozen-bank initial-state fitting diagnostic

A separate cold-only run, job `3350594` at source `0fb42607811425d639ba58714da2210f152cf463`, retained the same checkpoint and all 4 physical cases. It fitted initial states only. The complete-query table above still contains the original initializer, and no cost or error below is substituted into that table.

The original rule samples 48 equally weighted positions per axis between the first and last interior grid nodes. Those physical coordinates move toward the walls with mesh refinement. The fixed midpoint and weighted Gauss–Legendre rules keep their physical coordinates fixed and charge bilinear interpolation from the supplied field. These rules fit field values, never a pointwise PDE residual. The edge QR control changes only the numerical factorization; full-grid QR is a dense diagnostic within the original learned bank, with no POD replacement.

The next table uses budget 180 and 4 starts for every rule. Entries are worst-case initial-field errors on each complete requested grid, normalized by that grid's initial-field norm. The unrestricted bank floor allows arbitrary linear feature coefficients and is less restrictive than the nonlinear head. Returned local fits are not stationary or globally optimal oracles.

| Intervals | Edge Gram | Edge QR | Fixed midpoint | Fixed Gauss | Full-grid QR | Bank projection floor |
|---|---:|---:|---:|---:|---:|---:|
| 256 | 0.026004771 | 0.026004771 | 0.025466842 | 0.025628715 | 0.025446628 | 0.0018980278 |
| 512 | 0.054671614 | 0.054671614 | 0.032958472 | 0.032835851 | 0.032828466 | 0.014953484 |
| 1024 | 0.086950746 | 0.086950746 | 0.038736854 | 0.038562198 | 0.038548927 | 0.022593 |

Across every control, changing Gram to QR on the same edge samples changes the full-grid error by at most 2.46008e-11. At 1024 intervals, the largest casewise difference between fixed Gauss and full-grid QR is 1.3271e-05. The sampled objective's coordinate/weight drift therefore contributes materially; Cholesky regularization does not explain the transfer failure. The growing unrestricted bank floor also shows genuine representation loss under the full-grid norm. The continuous hard-wall bank and clipped Gaussian initial data make additional near-wall nodes a plausible contributor, but this diagnostic does not isolate the spatial location or separate all nonlinear-head and local-optimization limitations.

The query benchmark observes the fixed 256-interval grid. It can miss errors at additional fine near-wall nodes. This is visible when the same returned fit is scored under both norms:

| Intervals | Rule | Full-grid worst | Common-grid worst | Full-grid median |
|---|---|---:|---:|---:|
| 256 | `edge_gram` | 0.026004771 | 0.026004771 | 0.014063611 |
| 256 | `fixed_gauss` | 0.025628715 | 0.025628715 | 0.014037077 |
| 256 | `full_qr` | 0.025446628 | 0.025446628 | 0.014035163 |
| 512 | `edge_gram` | 0.054671614 | 0.052741118 | 0.020189301 |
| 512 | `fixed_gauss` | 0.032835851 | 0.025686942 | 0.014035696 |
| 512 | `full_qr` | 0.032828466 | 0.025604895 | 0.014035561 |
| 1024 | `edge_gram` | 0.086950746 | 0.084968854 | 0.032527162 |
| 1024 | `fixed_gauss` | 0.038562198 | 0.025886147 | 0.015672538 |
| 1024 | `full_qr` | 0.038548927 | 0.02572399 | 0.015669635 |

Budget/start controls at the largest grid, 1024 intervals, retain every case:

| Rule | Iteration budget | Starts | Full-grid worst | Budget exits | Improvement stops | Failed stops |
|---|---:|---:|---:|---:|---:|---:|
| `edge_gram` | 60 | 1 | 0.086953369 | 1 | 3 | 0 |
| `edge_gram` | 60 | 4 | 0.086950746 | 0 | 4 | 0 |
| `edge_gram` | 180 | 1 | 0.086953369 | 0 | 4 | 0 |
| `edge_gram` | 180 | 4 | 0.086950746 | 0 | 4 | 0 |
| `fixed_gauss` | 60 | 1 | 0.0385622 | 1 | 3 | 0 |
| `fixed_gauss` | 60 | 4 | 0.038562198 | 0 | 4 | 0 |
| `fixed_gauss` | 180 | 1 | 0.0385622 | 0 | 4 | 0 |
| `fixed_gauss` | 180 | 4 | 0.038562198 | 0 | 4 | 0 |
| `full_qr` | 60 | 1 | 0.038548927 | 1 | 3 | 0 |
| `full_qr` | 60 | 4 | 0.038548927 | 0 | 4 | 0 |
| `full_qr` | 180 | 1 | 0.038548927 | 0 | 4 | 0 |
| `full_qr` | 180 | 4 | 0.038548927 | 0 | 4 | 0 |

The largest normalized gradient diagnostic among the selected fits is 6.3445e-06; small-step/improvement stopping is not a stationarity certificate. Across all 240 returned fits, 15 exhausted their budget and 0 failed. Longer budgets and more starts do not remove the worst-case objective bias or bank floor. Timings in the raw cold records contain one warmed call per case and are diagnostic only; no repeated cold-fit cost frontier is claimed.

The independent audit checked all 240 declared fits, source/checkpoint hashes, unchanged cases and collection checksums. It recomputed each retained common-grid error with maximum discrepancy 2.77556e-17; the incumbent cold fields agree with the original query run within 5.19029e-14. Full-grid errors at finer resolutions and bank projection floors are in-job scalar diagnostics: the archive retains restricted fields, so those full norms are not independently reconstructed. The corrected initializer's evolution accuracy and complete-query latency remain unmeasured.

No strict physical-error certificate or rigorous reference bound is supplied by the empirical refinement estimates. No final-cohort result, unchanged-weight cross-PDE transfer, universal speed advantage, or fully optimized per-resolution training claim is established by this bounded pilot. Raw invocation arrays and fields remain the source of truth.

## Plain-language glossary

- **Intervals / interior unknowns:** grid cells along an axis / non-wall values solved for.
- **K / R / M / m:** latent coordinates / learned spatial features / smooth weak test functions / advection quadrature points.
- **Sampled bank rank:** independent feature directions on deterministic rows, with a relative singular-value cutoff of 1e-12.
- **Setup s / stored arrays MiB:** mesh-specific preparation seconds / stored numerical-array memory in binary megabytes.
- **Quadrature fit:** normalized residual of nonnegative fitting on decoder-output advection training rows; it is not a physical accuracy certificate.
- **Configuration / dt / stall / ntol:** solver setting name / timestep / ROM relative-improvement stopping threshold / FOM relative nonlinear-residual tolerance.
- **Query ms:** median complete input-to-output latency in milliseconds, including transfers and requested dense output.
- **Physical median / physical worst:** median / maximum over casewise time-maximum errors against the refined reference, normalized by initial reference norm.
- **Current-relative worst:** maximum error normalized by the current reference field norm, which can grow as the field decays.
- **Same-grid worst:** maximum ROM discrepancy from tightly converged FOM on the same grid and timestep, with initial-state normalization.
- **IC budget / stall; LM budget / stall:** initial-fit and evolution stops at their declared iteration budget / small-step or relative-improvement threshold. These counts cover all retained invocations; neither is a proven stationary optimum.
- **Outliers / failed invocations:** cases exceeding the stated error threshold / timed calls with nonfinite output, failed ROM fitting/evolution stops, or unmet FOM nonlinear tolerance.
- **Edge Gram / edge QR:** original mesh-dependent initial-field samples with Cholesky regularization / the same samples with orthogonal triangular factorization.
- **Fixed midpoint / fixed Gauss:** physical quadrature coordinates and weights kept unchanged across meshes, with charged input-field interpolation.
- **Full-grid QR / bank projection floor:** dense initial-fit objective within the learned bank / smallest error allowing unrestricted linear bank coefficients. Neither replaces the bank with POD or certifies a globally optimal nonlinear fit.
- **Full-grid / common-grid initial error:** initial-field discrepancy at all requested nodes / only at the fixed observation nodes, divided by the corresponding initial-field norm.
- **Normalized gradient:** objective-gradient norm divided by Jacobian norm times residual norm; a diagnostic of local optimization, not a global optimality guarantee.
- **Observed spatial/time order / Richardson estimate:** convergence rates inferred from three refinement levels / extrapolated remaining error assuming that rate continues; these are empirical diagnostics, not rigorous error bounds.
- **Cold starts / input / cold fit / evolve / dense decode / output / staged total / LM attempts:** number of training-code initial guesses / transfer into GPU memory / initial latent optimization / autonomous reduced evolution / dense-field reconstruction / transfer to host / all staged phases / actual Levenberg–Marquardt trial steps.
- **Spatial difference / time difference / sum estimate:** observed reference changes after spatial / temporal refinement / their sum; these diagnose uncertainty without proving a bound.
- **Qualification / target / selected ROM / selected FOM envelope / envelope speedup:** development accuracy status / requested error limit / cheapest eligible reduced configuration / cheapest eligible full configuration including coarser solves / median of per-case paired ratios (FOM median query time divided by ROM median query time).
- **Frozen / validation / final:** unchanged network weights / cases used for development selection / separate unopened confirmation cases.
- **FOM / ROM / sign-upwind / NNLS:** full model / reduced model / sign-selected spatial differences / nonnegative least squares.
- **Newton / BiCGStab / FFT sine transform / Helmholtz preconditioning:** nonlinear iteration / iterative linear solve / fast transform / exact inversion of the diffusion-plus-identity part to help that solve.
- **Commit / SHA-256 / backend / f64:** saved source revision / content fingerprint / execution device type / 64-bit floating-point arithmetic.
