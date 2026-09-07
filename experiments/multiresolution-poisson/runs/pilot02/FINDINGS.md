# Frozen Poisson test-space and query-fusion findings

These generated results are provisional development evidence using the same frozen checkpoint and source cohort as the first pilot. They diagnose weak-test truncation and representation error, and measure a parity-checked query fusion against efficient direct solvers.

Job `3350408`, source `0175877d40c96bc4c0e28cd2c8e2e3b0f6cbc966`, GPU `NVIDIA A100 80GB PCIe`. No network training or sealed-final-cohort evaluation ran. The cohort has 6 sources and 4 timing repetitions per configuration.

## Accuracy and complete-query cost

| Intervals | Arm | Requested/retained modes | Tau | Latency ms | Median physical error | Worst physical error | Invalid | Nonstationary | Time outliers |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | dst | — | — | 1.48076 | 5.59264e-05 | 0.0001546 | 0 | 0 | 0 |
| 256 | dst_coarse128 | — | — | 1.91706 | 0.000207491 | 0.000552816 | 0 | 0 | 0 |
| 256 | rom_modular | 64/64 | 0.01 | 3.79338 | 0.00892707 | 0.0736259 | 0 | 5 | 0 |
| 256 | rom_fused | 64/64 | 0.01 | 3.80094 | 0.00892707 | 0.0736259 | 0 | 5 | 0 |
| 256 | rom_modular | 64/64 | 0.0 | 5.03205 | 0.00764969 | 0.0731865 | 0 | 0 | 0 |
| 256 | rom_fused | 64/64 | 0.0 | 4.96676 | 0.00764969 | 0.0731865 | 0 | 0 | 0 |
| 256 | rom_modular | 128/129 | 0.01 | 4.36265 | 0.00770178 | 0.0725461 | 0 | 3 | 0 |
| 256 | rom_fused | 128/129 | 0.01 | 4.16409 | 0.00770178 | 0.0725461 | 0 | 3 | 0 |
| 256 | rom_modular | 128/129 | 0.0 | 4.90132 | 0.0076023 | 0.0725461 | 0 | 0 | 0 |
| 256 | rom_fused | 128/129 | 0.0 | 4.60383 | 0.0076023 | 0.0725461 | 0 | 0 | 0 |
| 256 | rom_modular | 256/257 | 0.01 | 4.38638 | 0.00770254 | 0.0725424 | 0 | 3 | 0 |
| 256 | rom_fused | 256/257 | 0.01 | 4.20693 | 0.00770254 | 0.0725424 | 0 | 3 | 0 |
| 256 | rom_modular | 256/257 | 0.0 | 4.57034 | 0.00760189 | 0.0725424 | 0 | 0 | 0 |
| 256 | rom_fused | 256/257 | 0.0 | 4.43418 | 0.00760189 | 0.0725424 | 0 | 0 | 0 |
| 512 | dst | — | — | 1.91714 | 1.33138e-05 | 3.67666e-05 | 0 | 0 | 0 |
| 512 | dst_coarse128 | — | — | 2.27591 | 0.000207491 | 0.000552816 | 0 | 0 | 0 |
| 512 | rom_modular | 64/64 | 0.01 | 4.25784 | 0.00892788 | 0.0736296 | 0 | 5 | 0 |
| 512 | rom_fused | 64/64 | 0.01 | 4.08257 | 0.00892788 | 0.0736296 | 0 | 5 | 0 |
| 512 | rom_modular | 64/64 | 0.0 | 5.27070 | 0.00764839 | 0.0731891 | 0 | 0 | 0 |
| 512 | rom_fused | 64/64 | 0.0 | 5.15627 | 0.00764839 | 0.0731891 | 0 | 0 | 0 |
| 512 | rom_modular | 128/129 | 0.01 | 4.72902 | 0.00770196 | 0.0725459 | 0 | 3 | 0 |
| 512 | rom_fused | 128/129 | 0.01 | 4.43965 | 0.00770196 | 0.0725459 | 0 | 3 | 0 |
| 512 | rom_modular | 128/129 | 0.0 | 4.87035 | 0.00760201 | 0.0725459 | 0 | 0 | 0 |
| 512 | rom_fused | 128/129 | 0.0 | 5.04168 | 0.00760201 | 0.0725459 | 0 | 0 | 0 |
| 512 | rom_modular | 256/257 | 0.01 | 4.37800 | 0.0077028 | 0.0725423 | 0 | 3 | 0 |
| 512 | rom_fused | 256/257 | 0.01 | 4.45762 | 0.0077028 | 0.0725423 | 0 | 3 | 0 |
| 512 | rom_modular | 256/257 | 0.0 | 5.18359 | 0.00760168 | 0.0725423 | 0 | 0 | 0 |
| 512 | rom_fused | 256/257 | 0.0 | 4.86217 | 0.00760168 | 0.0725423 | 0 | 0 | 0 |

Latency is the median over sources of each source's median repetition time. All cases and timing outliers remain included. Physical error is measured on common nested observation nodes against independently refined reference; it is not a rigorous continuum norm bound.

## Cheapest qualifying development configurations

| Requested intervals | Target | ROM arm / modes / tau | FOM | ROM ms | FOM ms | Median case cost ratio |
|---:|---:|---|---|---:|---:|---:|
| 256 | 0.1 | rom_modular / 64 / 0.01 | dst | 3.79338 | 1.48076 | 0.391318 |
| 256 | 0.05 | unattained | dst | unattained | 1.48076 | unattained |
| 256 | 0.01 | unattained | dst | unattained | 1.48076 | unattained |
| 256 | 0.001 | unattained | dst | unattained | 1.48076 | unattained |
| 512 | 0.1 | rom_fused / 64 / 0.01 | dst | 4.08257 | 1.91714 | 0.4691 |
| 512 | 0.05 | unattained | dst | unattained | 1.91714 | unattained |
| 512 | 0.01 | unattained | dst | unattained | 1.91714 | unattained |
| 512 | 0.001 | unattained | dst | unattained | 1.91714 | unattained |

For each source, the cost ratio is its median FOM time divided by its median ROM time; the table reports the median of those source ratios. A ratio above unity favors the ROM. The ratio of the two displayed aggregate latencies is a different statistic and is retained separately in JSON. Every source must meet solver/parity conditions and the target after empirical reference adjustment $(e+\delta)/(1-\delta)$, with $\delta$ also within the reference budget. This is development selection, not independent confirmation.

## Representation diagnostics

| Intervals | Case | Full-bank same-grid error | Best head same-grid error | Best head physical error | Best head stationarity | Budget / start |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0 | 0.0238673 | 0.0269703 | 0.0269319 | 1.10738e-09 | 300 / 1 |
| 256 | 1 | 0.00304856 | 0.00619061 | 0.00619178 | 2.21797e-09 | 300 / 2 |
| 256 | 2 | 0.00540344 | 0.00631753 | 0.00631233 | 5.17283e-11 | 300 / 3 |
| 256 | 3 | 0.0010462 | 0.00212136 | 0.0021216 | 3.08088e-10 | 300 / 0 |
| 256 | 4 | 0.00771344 | 0.00889882 | 0.00889138 | 2.05321e-10 | 300 / 0 |
| 256 | 5 | 0.0574652 | 0.0726348 | 0.0725423 | 4.62947e-09 | 300 / 3 |
| 512 | 0 | 0.0238418 | 0.026941 | 0.0269318 | 2.28662e-09 | 300 / 3 |
| 512 | 1 | 0.00304902 | 0.00619143 | 0.00619168 | 2.65005e-10 | 300 / 1 |
| 512 | 2 | 0.00539961 | 0.00631342 | 0.00631214 | 5.18799e-11 | 300 / 3 |
| 512 | 3 | 0.00104555 | 0.00212131 | 0.00212131 | 5.51209e-11 | 300 / 3 |
| 512 | 4 | 0.00770724 | 0.008893 | 0.0088912 | 1.06348e-09 | 300 / 3 |
| 512 | 5 | 0.0573878 | 0.0725643 | 0.0725423 | 4.60929e-09 | 300 / 3 |

At 256 intervals, increasing the tight-solve test request from 64 to 256 changes worst physical error from 0.0731865097 to 0.0725423651. The enlarged-test solutions differ in physical error from the best recorded full-field head fits by at most 5.39141853e-08 across sources. The maximum change in best head-fit error after doubling the oracle budget is 0.

At 512 intervals, increasing the tight-solve test request from 64 to 256 changes worst physical error from 0.0731890805 to 0.0725423124. The enlarged-test solutions differ in physical error from the best recorded full-field head fits by at most 1.89345598e-08 across sources. The maximum change in best head-fit error after doubling the oracle budget is 0.

The full-bank projection is a higher-dimensional least-square diagnostic. The head oracle minimizes the full-field objective in exact QR coordinates from fixed deterministic starts. Both see the reference answer only for diagnosis; neither initializes or selects deployed queries. All starts, budgets, gradients, attempts and latent outputs are retained. A stationary local fit is not proof of a global optimum.

## Fusion parity and component cost

| Intervals | Modes | Tau | Median modular/fused case ratio | All field/latent/counter parity gates |
|---:|---:|---:|---:|---|
| 256 | 64 | 0.01 | 1.02215 | True |
| 256 | 64 | 0.0 | 1.00706 | True |
| 256 | 128 | 0.01 | 1.0408 | True |
| 256 | 128 | 0.0 | 1.04728 | True |
| 256 | 256 | 0.01 | 1.04078 | True |
| 256 | 256 | 0.0 | 1.03722 | True |
| 512 | 64 | 0.01 | 1.02596 | True |
| 512 | 64 | 0.0 | 1.01073 | True |
| 512 | 128 | 0.01 | 1.00453 | True |
| 512 | 128 | 0.0 | 0.992238 | True |
| 512 | 256 | 0.01 | 1.01149 | True |
| 512 | 256 | 0.0 | 1.04529 | True |

| Intervals | Arm | Modes | Tau | Input ms | Projection/init ms | Solver ms | Fused device ms | Output ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | dst | — | — | 0.40782 | 0.00000 | 0.14014 | — | 0.89780 |
| 256 | dst_coarse128 | — | — | 0.35407 | 0.00000 | 1.24675 | — | 0.28975 |
| 256 | rom_modular | 64 | 0.01 | 0.37417 | 0.57130 | 2.46327 | — | 0.46993 |
| 256 | rom_fused | 64 | 0.01 | 0.36765 | — | — | 3.08629 | 0.39632 |
| 256 | rom_modular | 64 | 0.0 | 0.39559 | 0.70509 | 3.37631 | — | 0.48560 |
| 256 | rom_fused | 64 | 0.0 | 0.37826 | — | — | 4.22304 | 0.38937 |
| 256 | rom_modular | 128 | 0.01 | 0.37489 | 0.81360 | 2.74406 | — | 0.48116 |
| 256 | rom_fused | 128 | 0.01 | 0.37945 | — | — | 3.38473 | 0.39579 |
| 256 | rom_modular | 128 | 0.0 | 0.39487 | 0.72804 | 3.23310 | — | 0.53943 |
| 256 | rom_fused | 128 | 0.0 | 0.38192 | — | — | 3.94221 | 0.35149 |
| 256 | rom_modular | 256 | 0.01 | 0.39788 | 0.59414 | 2.90699 | — | 0.50542 |
| 256 | rom_fused | 256 | 0.01 | 0.35717 | — | — | 3.47654 | 0.33806 |
| 256 | rom_modular | 256 | 0.0 | 0.36015 | 0.68232 | 3.06600 | — | 0.48536 |
| 256 | rom_fused | 256 | 0.0 | 0.37671 | — | — | 3.65220 | 0.35905 |
| 512 | dst | — | — | 0.61526 | 0.00000 | 0.17376 | — | 1.08383 |
| 512 | dst_coarse128 | — | — | 0.61056 | 0.00000 | 1.16420 | — | 0.51710 |
| 512 | rom_modular | 64 | 0.01 | 0.62087 | 0.63502 | 2.28147 | — | 0.70220 |
| 512 | rom_fused | 64 | 0.01 | 0.61047 | — | — | 2.94291 | 0.52316 |
| 512 | rom_modular | 64 | 0.0 | 0.62765 | 0.66641 | 3.19753 | — | 0.71702 |
| 512 | rom_fused | 64 | 0.0 | 0.59137 | — | — | 3.93656 | 0.57378 |
| 512 | rom_modular | 128 | 0.01 | 0.61350 | 0.70864 | 2.57320 | — | 0.72733 |
| 512 | rom_fused | 128 | 0.01 | 0.61164 | — | — | 3.21388 | 0.55354 |
| 512 | rom_modular | 128 | 0.0 | 0.62967 | 0.56357 | 3.02237 | — | 0.69550 |
| 512 | rom_fused | 128 | 0.0 | 0.59947 | — | — | 3.81995 | 0.56134 |
| 512 | rom_modular | 256 | 0.01 | 0.61288 | 0.61326 | 2.62522 | — | 0.68108 |
| 512 | rom_fused | 256 | 0.01 | 0.59689 | — | — | 3.31778 | 0.55163 |
| 512 | rom_modular | 256 | 0.0 | 0.63936 | 0.63622 | 3.10095 | — | 0.68929 |
| 512 | rom_fused | 256 | 0.0 | 0.61365 | — | — | 3.62933 | 0.57740 |

A fused invocation has input, combined device pipeline and output timestamps. Its internal projection/solve/decode times cannot be separated without reintroducing synchronization. The segmented control supplies actual same-invocation component timings; they are not substituted into fused results. Component medians need not sum to the total median. The original generic small linear-system kernel remains unchanged, so this run does not exhaust kernel optimization.

## Reference and statistic limits

Reference intervals are [512, 1024, 2048]; observations use 256 intervals. Maximum empirical final refinement difference is 7.35117705e-06. No Richardson reduction or rigorous error-bound claim is made. Larger test spaces are deployment changes on frozen networks; no per-resolution training was tested.

| First-pilot intervals | Earlier median of paired-repetition ratios | Campaign median of case-median ratios |
|---:|---:|---:|
| 256 | 0.421264144 | 0.394957513 |
| 512 | 0.49944498 | 0.499504621 |

The two first-pilot values use the same raw observations and differ only in aggregation order. Earlier values are preserved explicitly; no data or scientific conclusion was silently replaced. Raw times are never compared across the first and second jobs.

## Plain-language glossary

- **Intervals / modes / retained:** cells per grid axis / smooth functions averaging the PDE / actual complete sine shells used after the requested cutoff.
- **Arm / tau / latency:** measured implementation / requested initial-residual reduction / median of each source's median complete-query time.
- **ROM / FOM / DST:** reduced model / full discrete model / direct sine-transform solver.
- **Physical / same-grid error:** relative discrepancy from refined reference at common observation nodes / from the full solver on the same mesh.
- **Invalid / nonstationary / time outlier:** failed solver or parity gate / gradient above stationary threshold / repeated time above three configuration latencies; all are retained.
- **Full bank / head oracle / QR:** unrestricted learned spatial span / reference-only fit of nonlinear coefficients / exact orthonormal coordinates preserving field least squares.
- **Budget / start / stationarity:** maximum LM attempts / fixed initial latent choice / normalized gradient measuring local first-order optimality.
- **Modular / fused / parity:** separately dispatched stages / one compiled pipeline / field, latent and exact counter agreement.
- **Input / projection / solver / output:** host field upload / source weak-mode preparation / latent or direct equation solution / complete host field and metadata return.
- **Cost ratio / target / qualification:** FOM median cost divided by ROM median cost per source, then median over sources / requested error ceiling / all-source validity and uncertainty-adjusted accuracy.
- **Reference difference / delta / e:** last nested-refinement discrepancy / that discrepancy normalized by fine-reference norm / measured error against fine reference.
- **ms / s / checkpoint / cohort:** milliseconds / seconds / frozen trained weights / fixed development source group.
