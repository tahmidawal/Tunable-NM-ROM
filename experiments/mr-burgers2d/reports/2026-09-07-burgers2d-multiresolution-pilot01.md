# Burgers 2D frozen-network resolution transfer and complete-query cost

Development pilot; all tables below are generated from the saved invocation records. Physical-error qualifications remain provisional wherever the measured reference-refinement estimate misses the target budget.

Source commit `96befb8815a4c20c8d9f2862502e454ab8402032`, job `3350012`, GPU `NVIDIA A100 80GB PCIe`. Backend `gpu`, f64 `True`, matmul precision `highest`. Checkpoint SHA-256 `18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589`.

The frozen checkpoint was trained on 255 intervals (256 nodes per axis). The new validation seed is 7090702, with 4 physical cases; the final cohort is unopened.

The query starts with a dense initial field in host memory and ends with all requested dense fields in host memory. Input handling, cold fitting, evolution, output reconstruction/interpolation and transfers are included. Compilation and reusable setup are separate. FOM candidates use sign-upwind backward Euler with adaptive Newton/BiCGStab and FFT sine-transform Helmholtz preconditioning. ROM quadrature retains sign-dependent upwinding on decoded undershoots.

## Reference refinement

| Case | Spatial difference | Time difference | Sum estimate |
|---|---:|---:|---:|
| 0 | 0.00235856 | 0.000748746 | 0.0031073 |
| 1 | 0.00137857 | 0.00172215 | 0.00310071 |
| 2 | 0.00855218 | 0.00255498 | 0.0111072 |
| 3 | 0.00533584 | 0.00173097 | 0.0070668 |

These are differences between independently converged refinement levels, not rigorous continuum-error bounds. The physical norm uses exact nested-node restriction onto the observation grid and the initial-state reference norm. Current-field normalization is reported separately.

## Mesh setup

| Intervals | Interior unknowns | K / R / M / m | Sampled bank rank | Setup s | Stored arrays MiB | Quadrature fit |
|---|---:|---|---:|---:|---:|---:|
| 256 | 65025 | 16 / 512 / 64 / 256 | 512 | 31.265 | 303.477 | 0.0051591 |
| 512 | 261121 | 16 / 512 / 64 / 256 | 512 | 24.508 | 1069.477 | 0.00515699 |

## Measured configurations

Errors are maximum-in-observation-time values, aggregated over every declared case and repetition. A configured budget exit is retained as an early stop; a small-step/relative-improvement stop is not a stationarity certificate. Failed/nonfinite solver stops are excluded from target eligibility. Outliers count cases above relative error 0.01. Failed invocations are retained. Wall times are medians over all cases/repetitions in the same job.

| Configuration | Query ms | Physical median | Physical worst | Current-relative worst | Same-grid worst | Outliers | Failed invocations | IC budget / stall | LM budget / stall |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `fom_L256_out256_dt0.01_ntol0.01` | 10.099 | 0.019704 | 0.0316119 | 0.0361345 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out256_dt0.01_ntol0.01` | 10.242 | 0.0274808 | 0.0506562 | 0.0720461 | — | 4 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out256_dt0.01_ntol0.003` | 13.380 | 0.0331091 | 0.0660738 | 0.0923914 | — | 4 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out256_dt0.01_ntol0.003` | 14.117 | 0.0263751 | 0.0442473 | 0.0606111 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out256_dt0.005_ntol0.01` | 16.667 | 0.0274808 | 0.0476648 | 0.0677916 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out256_dt0.005_ntol0.01` | 17.106 | 0.0118022 | 0.0195889 | 0.0268389 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out256_dt0.005_ntol0.003` | 17.811 | 0.0274808 | 0.0507792 | 0.0722211 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out256_dt0.005_ntol0.003` | 18.213 | 0.0101784 | 0.0222605 | 0.0316601 | — | 2 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out256_dt0.0025_ntol0.003` | 30.334 | 0.0274808 | 0.0498071 | 0.0708385 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out256_dt0.0025_ntol0.003` | 31.247 | 0.00865268 | 0.0202157 | 0.0287519 | — | 2 / 4 | 0 | 0 / 0 | 0 / 0 |
| `rom_L256_dt0.005_stall0.01` | 37.209 | 0.0195388 | 0.0395798 | 0.0498492 | 0.0260045 | 4 / 4 | 0 | 3 / 9 | 0 / 600 |
| `rom_L256_dt0.005_stall0.001` | 39.309 | 0.0195388 | 0.0396088 | 0.0498857 | 0.0260045 | 4 / 4 | 0 | 3 / 9 | 0 / 600 |
| `rom_L256_dt0.0025_stall0.001` | 58.412 | 0.0195388 | 0.0361079 | 0.0454765 | 0.0260045 | 3 / 4 | 0 | 3 / 9 | 0 / 1200 |
| `rom_L256_dt0.0025_stall0.01` | 58.415 | 0.0195388 | 0.0360941 | 0.0454591 | 0.0260045 | 3 / 4 | 0 | 3 / 9 | 0 / 1200 |
| `fom_L128_out512_dt0.01_ntol0.01` | 11.621 | 0.0274808 | 0.0506562 | 0.0720461 | — | 4 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out512_dt0.01_ntol0.01` | 11.977 | 0.019704 | 0.0316119 | 0.0361345 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out512_dt0.01_ntol0.003` | 15.061 | 0.0331091 | 0.0660738 | 0.0923914 | — | 4 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out512_dt0.01_ntol0.003` | 15.787 | 0.0263751 | 0.0442473 | 0.0606111 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L512_out512_dt0.01_ntol0.01` | 16.726 | 0.0211456 | 0.0281378 | 0.0329965 | — | 4 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out512_dt0.005_ntol0.01` | 18.428 | 0.0274808 | 0.0476648 | 0.0677916 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out512_dt0.005_ntol0.01` | 18.786 | 0.0118022 | 0.0195889 | 0.0268389 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out512_dt0.005_ntol0.003` | 19.441 | 0.0274808 | 0.0507792 | 0.0722211 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out512_dt0.005_ntol0.003` | 19.831 | 0.0101784 | 0.0222605 | 0.0316601 | — | 2 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L512_out512_dt0.01_ntol0.003` | 22.086 | 0.0232018 | 0.0368047 | 0.0441311 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L512_out512_dt0.005_ntol0.01` | 26.742 | 0.012999 | 0.0229967 | 0.0269676 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L512_out512_dt0.005_ntol0.003` | 27.687 | 0.0100467 | 0.0158182 | 0.0171707 | — | 2 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L128_out512_dt0.0025_ntol0.003` | 32.510 | 0.0274808 | 0.0498071 | 0.0708385 | — | 3 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L256_out512_dt0.0025_ntol0.003` | 32.857 | 0.00865268 | 0.0202157 | 0.0287519 | — | 2 / 4 | 0 | 0 / 0 | 0 / 0 |
| `fom_L512_out512_dt0.0025_ntol0.003` | 48.521 | 0.00615772 | 0.0105628 | 0.0123867 | — | 1 / 4 | 0 | 0 / 0 | 0 / 0 |
| `rom_L512_dt0.005_stall0.01` | 40.192 | 0.0287896 | 0.0527405 | 0.0527405 | 0.0527405 | 4 / 4 | 0 | 3 / 9 | 0 / 600 |
| `rom_L512_dt0.005_stall0.001` | 42.157 | 0.0288015 | 0.0527405 | 0.0527405 | 0.0527405 | 4 / 4 | 0 | 3 / 9 | 0 / 600 |
| `rom_L512_dt0.0025_stall0.01` | 60.678 | 0.0279125 | 0.0527405 | 0.0527405 | 0.0527405 | 4 / 4 | 0 | 3 / 9 | 0 / 1200 |
| `rom_L512_dt0.0025_stall0.001` | 61.041 | 0.027918 | 0.0527405 | 0.0527405 | 0.0527405 | 4 / 4 | 0 | 3 / 9 | 0 / 1200 |

## Target-qualified validation selections

Both methods may choose a configuration within the declared pilot search. Eligibility includes the measured reference estimate as an additive error margin. The FOM envelope includes coarser solves with aligned interpolation to the same requested dense output. Solver choices are tuned per resolution; both neural networks remain frozen. This does not test retraining benefits.

| Output intervals | Target | Qualification | Selected ROM | Selected FOM envelope | Envelope speedup |
|---|---:|---|---|---|---:|
| 256 | 0.1 | unresolved reference uncertainty/order | rom_L256_dt0.005_stall0.01 | fom_L256_out256_dt0.01_ntol0.01 | — |
| 256 | 0.05 | unresolved reference uncertainty/order | rom_L256_dt0.0025_stall0.001 | fom_L256_out256_dt0.01_ntol0.01 | — |
| 256 | 0.01 | unresolved reference uncertainty/order; ROM target unattained; FOM target unattained | — | — | — |
| 256 | 0.001 | unresolved reference uncertainty/order; ROM target unattained; FOM target unattained | — | — | — |
| 512 | 0.1 | unresolved reference uncertainty/order | rom_L512_dt0.005_stall0.01 | fom_L128_out512_dt0.01_ntol0.01 | — |
| 512 | 0.05 | unresolved reference uncertainty/order; ROM target unattained | — | fom_L256_out512_dt0.01_ntol0.01 | — |
| 512 | 0.01 | unresolved reference uncertainty/order; ROM target unattained; FOM target unattained | — | — | — |
| 512 | 0.001 | unresolved reference uncertainty/order; ROM target unattained; FOM target unattained | — | — | — |

No final-cohort result, unchanged-weight cross-PDE transfer, universal speed advantage, or fully optimized per-resolution training claim is established by this bounded pilot. Raw invocation arrays and fields remain the source of truth.

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
- **Outliers / failed invocations:** cases exceeding the stated error threshold / timed calls with nonfinite output or unmet FOM nonlinear tolerance.
- **Observed spatial/time order / Richardson estimate:** convergence rates inferred from three refinement levels / extrapolated remaining error assuming that rate continues; these are empirical diagnostics, not rigorous error bounds.
- **Cold starts / input / cold fit / evolve / dense decode / output / staged total / LM attempts:** number of training-code initial guesses / transfer into GPU memory / initial latent optimization / autonomous reduced evolution / dense-field reconstruction / transfer to host / all staged phases / actual Levenberg–Marquardt trial steps.
- **Spatial difference / time difference / sum estimate:** observed reference changes after spatial / temporal refinement / their sum; these diagnose uncertainty without proving a bound.
- **Qualification / target / selected ROM / selected FOM envelope / envelope speedup:** development accuracy status / requested error limit / cheapest eligible reduced configuration / cheapest eligible full configuration including coarser solves / median of per-case paired ratios (FOM median query time divided by ROM median query time).
- **Frozen / validation / final:** unchanged network weights / cases used for development selection / separate unopened confirmation cases.
- **FOM / ROM / sign-upwind / NNLS:** full model / reduced model / sign-selected spatial differences / nonnegative least squares.
- **Newton / BiCGStab / FFT sine transform / Helmholtz preconditioning:** nonlinear iteration / iterative linear solve / fast transform / exact inversion of the diffusion-plus-identity part to help that solve.
- **Commit / SHA-256 / backend / f64:** saved source revision / content fingerprint / execution device type / 64-bit floating-point arithmetic.
