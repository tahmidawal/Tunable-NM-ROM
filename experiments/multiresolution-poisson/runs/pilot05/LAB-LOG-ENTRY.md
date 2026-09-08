## 2026-09-07

### Poisson continuation factorial and projection/initialization speed study closed

The user retained separate worktrees and requested continued development. This session stayed in the approved Poisson tree, branch and remote namespace; no branch or merge was created. The reviewed result archive and generated findings are committed through `a970e274165085e3e4373113d8598853952172c0`. No further Poisson GPU study is launched this round.

Training factorial job `3353137` used immutable source `0ed8384c30ff150d0f25adfe1dead41a8a739c26`; speed factorial job `3354845` used `20893525a16603a99f527dbd9e10affcdc206b9d`. Each used its own attempt directory, GPU preflight, float64/highest precision, paired complete-query repetitions and burn-in. Source and checkpoint bytes were staged from committed history with direct checksums; data were regenerated from seeds. Sealed final sources stayed closed.

The training factorial crossed original coverage with expanded coverage and shared-global with full-field per-snapshot relative normalization. All 4 fixed endpoints reached 10000 updates. Original reproduction used the first 512 sources of the original 576 draw; expanded coverage added 1536 independently seeded sources. Training uses 256 nodes, while queries use [256, 512] intervals. Both bank and head were continued, along with training codes, under matched updates and batches rather than equal visits per source. Actual offline source/code-fit/compile/train/diagnostic elapsed was 47.1393422 seconds. Offline durations are not a paired warm training-speed comparison.

The owner audit checked 7680 training-study query invocations, 896 unique timed fields and 300 reference-only bank/head diagnostic panels. Maximum independent CPU query-metric difference was 5.30084409e-16. The root coordinator separately audited every timed field and independently reproduced the endpoint selection.

Training stationary-control physical errors, generated from native data:

| Intervals | Development cohort | Model | Median / worst physical error | Invalid / nonstationary |
|---:|---|---|---:|---:|
| 256 | existing_development | original_frozen | 0.00760189002 / 0.0725423651 | 0 / 0 |
| 256 | existing_development | original_global | 0.00758192806 / 0.0719184612 | 0 / 0 |
| 256 | existing_development | expanded_global | 0.00905150484 / 0.0746116862 | 0 / 0 |
| 256 | existing_development | original_relative | 0.0131921547 / 0.0608459011 | 0 / 0 |
| 256 | existing_development | expanded_relative | 0.0136182999 / 0.0576748392 | 0 / 0 |
| 256 | fresh_development | original_frozen | 0.0195452025 / 0.0766875934 | 0 / 0 |
| 256 | fresh_development | original_global | 0.0194650646 / 0.0776348521 | 0 / 0 |
| 256 | fresh_development | expanded_global | 0.0185411662 / 0.0849998522 | 0 / 0 |
| 256 | fresh_development | original_relative | 0.0208308225 / 0.068015643 | 0 / 0 |
| 256 | fresh_development | expanded_relative | 0.0184733094 / 0.0747453733 | 0 / 0 |
| 512 | existing_development | original_frozen | 0.00760168447 / 0.0725423124 | 0 / 0 |
| 512 | existing_development | original_global | 0.00758172043 / 0.0719184032 | 0 / 0 |
| 512 | existing_development | expanded_global | 0.00905132552 / 0.0746116238 | 0 / 0 |
| 512 | existing_development | original_relative | 0.0131920174 / 0.0608457768 | 0 / 0 |
| 512 | existing_development | expanded_relative | 0.0136181807 / 0.0576747557 | 0 / 0 |
| 512 | fresh_development | original_frozen | 0.0195451089 / 0.0766875155 | 0 / 0 |
| 512 | fresh_development | original_global | 0.0194649613 / 0.077634775 | 0 / 0 |
| 512 | fresh_development | expanded_global | 0.0185410352 / 0.0849997815 | 0 / 0 |
| 512 | fresh_development | original_relative | 0.0208306791 / 0.068015574 | 0 / 0 |
| 512 | fresh_development | expanded_relative | 0.0184731351 / 0.0747452342 | 0 / 0 |

The predeclared minimax rule across all sources and both meshes selects `original_relative`, SHA256 `81f945571da60bbfe9adfba5969727ade137c940b212bc6a6e25c525273d417a`, with worst empirically adjusted error 0.0680196727. This choice uses every development case, not the previously inspected narrow-source example. Relative normalization changes the tradeoff between worst-case and median errors; expanded coverage does not uniformly improve the fixed-compute endpoints. Bank/head fits remain local reference-only diagnostics and are never online initializations.

The speed factorial retains the original and selected checkpoint, crossing sine-product versus forward-DST source projection with mean-code versus nearest training-prediction initialization. It records 1800 repeated primary calls and 480 separate single-call stationary controls. The cache uses only training codes and decoder-predicted weak coefficients. Lookup, projection, guarded linear solve/fallback, nonlinear evolution, decoding and full host-field transfer are charged. Cache/operator assembly and compilation are offline.

Full host source through returned host field, stop reason/counters/latents, guard diagnostics and initialization metadata. Independent stationarity recomputation, physical-error and parity scoring occur after the timer, using that same invocation.

The residual threshold remains relative to each chosen start. A nearer start can demand a tighter absolute threshold and force a stationary exit. Initial/final residuals, absolute thresholds, stop reasons and nearest-index gaps are retained. Projection parity compares the same initialization choice; different mean/nearest local minima are evaluated directly rather than rejected by equality.

| Intervals | Checkpoint | Projection | Initialization | Primary query ms | Worst physical error | Invalid / nonstationary | Paired same-grid DST/ROM |
|---:|---|---|---|---:|---:|---:|---:|
| 256 | original_frozen | skinny_sine_products | mean_training_code | 4.3612505 | 0.0766875934 | 0 / 11 | 0.412807273 |
| 256 | original_frozen | skinny_sine_products | nearest_cached_scaled_weak_prediction | 3.79991601 | 0.0766875934 | 0 / 0 | 0.426915035 |
| 256 | original_frozen | forward_dst_and_gather | mean_training_code | 4.17404401 | 0.0766875934 | 0 / 11 | 0.410709182 |
| 256 | original_frozen | forward_dst_and_gather | nearest_cached_scaled_weak_prediction | 3.903693 | 0.0766875934 | 0 / 0 | 0.43268999 |
| 256 | original_relative | skinny_sine_products | mean_training_code | 4.36738844 | 0.068015643 | 0 / 6 | 0.398234345 |
| 256 | original_relative | skinny_sine_products | nearest_cached_scaled_weak_prediction | 3.93937051 | 0.068015643 | 0 / 0 | 0.445042553 |
| 256 | original_relative | forward_dst_and_gather | mean_training_code | 4.34452749 | 0.068015643 | 0 / 6 | 0.394674063 |
| 256 | original_relative | forward_dst_and_gather | nearest_cached_scaled_weak_prediction | 3.7348675 | 0.068015643 | 0 / 0 | 0.456780496 |
| 512 | original_frozen | skinny_sine_products | mean_training_code | 4.68914851 | 0.0766875155 | 0 / 11 | 0.480681009 |
| 512 | original_frozen | skinny_sine_products | nearest_cached_scaled_weak_prediction | 4.18058998 | 0.0766875155 | 0 / 0 | 0.545576953 |
| 512 | original_frozen | forward_dst_and_gather | mean_training_code | 4.59367101 | 0.0766875155 | 0 / 11 | 0.484046831 |
| 512 | original_frozen | forward_dst_and_gather | nearest_cached_scaled_weak_prediction | 4.32107202 | 0.0766875155 | 0 / 0 | 0.532466198 |
| 512 | original_relative | skinny_sine_products | mean_training_code | 4.60523053 | 0.068015574 | 0 / 6 | 0.480207839 |
| 512 | original_relative | skinny_sine_products | nearest_cached_scaled_weak_prediction | 4.01824253 | 0.068015574 | 0 / 0 | 0.561288028 |
| 512 | original_relative | forward_dst_and_gather | mean_training_code | 4.83629951 | 0.068015574 | 0 / 6 | 0.458062264 |
| 512 | original_relative | forward_dst_and_gather | nearest_cached_scaled_weak_prediction | 4.20458097 | 0.068015574 | 0 / 0 | 0.542128391 |

Both complete-query development envelopes use median case-median latencies and medians of per-case cost ratios. The efficient FOM panel includes same-grid DST and charged coarse-grid interpolation. Every source must meet the solver, parity and empirical reference-adjusted accuracy gates; reference differences are not rigorous continuum bounds. Separate stationary controls do not enter speed selection.

| Intervals | Target | Selected speed-study ROM | ROM / selected FOM ms | Paired FOM/ROM |
|---:|---:|---|---:|---:|
| 256 | 0.1 | original_relative / forward_dst_and_gather / nearest_cached_scaled_weak_prediction | 3.7348675 / 1.75096601 | 0.456780496 |
| 256 | 0.05 | unattained | — / 1.75096601 | — |
| 256 | 0.01 | unattained | — / 1.75096601 | — |
| 256 | 0.001 | unattained | — / 1.75096601 | — |
| 512 | 0.1 | original_relative / skinny_sine_products / nearest_cached_scaled_weak_prediction | 4.01824253 / 2.26715201 | 0.561288028 |
| 512 | 0.05 | unattained | — / 2.26715201 | — |
| 512 | 0.01 | unattained | — / 2.26715201 | — |
| 512 | 0.001 | unattained | — / 2.26715201 | — |

The speed-study CPU audit checks 668 preserved full fields, reconstructs neural outputs from saved coordinates and weights, verifies the training-only cache and selected codes, and recomputes physical errors, residuals and stationarity. Maximum query-metric discrepancy is 5.30084409e-16, decoded-field relative discrepancy 7.25450008e-15, and stationarity discrepancy 2.31287212e-13. Projection-gate failures: 0; primary fallbacks: 0; stationary fallbacks: 0. All failures remain in native records and errors. The audit distinguishes exact saved source-parameter hashes from tiny cross-architecture exponential roundoff in independent seed regeneration.

- `pilot04`: archive 1877205838 bytes in 38 checked parts, SHA256 `486713c64ffb9cbc60c51fe31f828af61b30b0efd51895e17b3374e723edb4e0`. Exact remote `/cluster/tufts/paralab/tawal01/mr_poisson2d_20260907/pilot04` deleted and absence checked.
- `pilot05`: archive 858639891 bytes in 18 checked parts, SHA256 `831001c363bb364c86107f57231e5df70ab12f33c6db16eea212273f0983cb2c`. Exact remote `/cluster/tufts/paralab/tawal01/mr_poisson2d_20260907/pilot05` deleted and absence checked.

All raw JSON, source manifests, logs, native audits, PNG/PDF figures and generated findings are preserved under `worktrees/2026-09-07-mr-poisson2d/experiments/multiresolution-poisson/runs/`. The four trained checkpoints are also directly tracked under `runs/pilot04/checkpoints/`, not only inside the checked archive.

No prior numerical result is retracted. Cross-architecture exact input-hash regeneration is not asserted; recorded within-job source hashes, exact persisted draw prefixes and numerical CPU reconstruction are the supported checks. These bounded development studies do not establish an optimal training recipe, a globally optimal head fit, rigorous physical certification or a paper-complete speed claim. Further training, solver changes and final confirmation remain open; stop for coordinator review with separate worktrees.
