# Restricted heat development pilot: frozen mesh transfer

These provisional findings establish the new heat port and measure a bounded continuous-coordinate NM-ROM pilot against direct sine-transform propagation. They await independent review and do not cover the archived multi-bump heat family, per-resolution optimization, or an independent final cohort.

Generated from `experiments/mr-heat2d/runs/pilot01/archive/outputs/results.json`. Source commit: `ad882df3ecb8614d4cafa6a25761e920f3074e3d`; job `3349961`; GPU `NVIDIA A100 80GB PCIe`. The run records GPU execution, float64 arithmetic and highest matmul precision. Checkpoint SHA-256: `c60ebb97c62329c750f86504f0585678248ed1f20f2c054db304e1091bc969c3`.

The complete query supplies a host initial interior field and returns host interior fields at every requested time. Input transfer, full-input projection and latent fitting, evolution, dense field readout and output transfer are included. The FOM propagates each requested time directly, with no imposed timestep count. The same frozen weights are reevaluated and the operators rebuilt on every mesh.

## Pinned development configuration

| Setting | Value |
|---|---|
| Training intervals | 64 |
| Evaluation intervals | [64, 128] |
| Training / validation trajectories | 32 / 4 |
| Latent / bank / weak test dimensions | 8 / 32 / 64 |
| Output times | [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] |
| CN timestep choices | [0.025, 0.0125] |
| Diffusivity | 0.02 |
| Center / width / amplitude bounds | [0.35, 0.65] / [0.1, 0.15] / [0.8, 1.2] |
| Train / validation / model seeds | 790710 / 790711 / 790712 |
| Optimizer steps / trained parameters | 12000 / 42561 |
| Timing repetitions per case and method | 5 |

The initial condition is a single smooth Gaussian times the decoder's polynomial zero-boundary factor. Data-generation descriptors never enter the head or the initial-fitting routine. Initial fitting uses exact QR compression of full-field least squares, with nearest-training-code and mean-code starts. Both starts' counters and normalized gradients are retained.

## Independent reference and state-advancement checks

| Check | Measured value |
|---|---|
| dst_vs_scipy | 2.97192e-16 |
| dst_involution | 3.42827e-16 |
| laplacian_diagonalization | 2.7844e-16 |
| discrete_eigenmode_decay | 3.54509e-16 |
| continuum_eigenmode_decay | 3.55634e-16 |
| initial_fidelity | 0 |
| weak_discrete_operator_identity | 3.21256e-15 |
| linear_weak_rollout_exact_cn | 1.79537e-14 |
| second_output_advancement | 0.226398 |
| frozen_negative_control_error | 0.630101 |
| cn_time_refinement_ratio | 4.00119 |
| hard_boundary_max | 0 |
| Maximum spectral reference refinement error | 1.81003e-10 |
| Minimum / maximum spatial refinement ratio | 4.00239 / 4.01254 |

All declared reference gates passed. The independent analytic linear-mode test uses the same LM/scan implementation as the learned rollout. The deliberately frozen carry fails that test. These are implementation and truth checks; learned accuracy is reported below.

## Accuracy across the complete validation cohort

Each error is the maximum over requested output times, then aggregated across trajectories. These first tables use each mesh's own sampled physical norm; the common-observation audit below is the transfer comparison. Current normalization divides by the current physical-reference field; initial normalization divides by its initial field. All numbers here are provisional development-cohort results, not final confirmation.

| Intervals | Method | Current error median / worst | Initial error median / worst | Initial-fit median / worst | Cases above 5% current error |
|---|---|---|---|---|---|
| 64 | fom_dst_exact_time | 0.000675205 / 0.000709211 | 0.000491381 / 0.000516928 | 0 / 0 | 0 / 4 |
| 64 | rom_cn_dt0.025 | 0.0585754 / 0.0973564 | 0.0557945 / 0.0973564 | 0.0503788 / 0.0973564 | 3 / 4 |
| 64 | rom_cn_dt0.0125 | 0.0588214 / 0.0973564 | 0.0560134 / 0.0973564 | 0.0503788 / 0.0973564 | 3 / 4 |
| 128 | fom_dst_exact_time | 0.000168684 / 0.000177175 | 0.000122797 / 0.000129182 | 0 / 0 | 0 / 4 |
| 128 | rom_cn_dt0.025 | 0.0585452 / 0.0973565 | 0.0557677 / 0.0973565 | 0.0503789 / 0.0973565 | 3 / 4 |
| 128 | rom_cn_dt0.0125 | 0.0587906 / 0.0973565 | 0.055986 / 0.0973565 | 0.0503789 / 0.0973565 | 3 / 4 |

## Complete query timing

Costs and errors come from the same captured invocations. Reported costs are the cohort median of each case's repetition median. Speed ratio is the median of paired FOM/ROM case ratios. The target column checks observed error only; it does not certify solver validity or a reference-error margin. The FOM and ROM are timed on the same physical GPU with alternating order and clock burn-in.

| Intervals | Method | Query ms | Paired FOM / method | Observed-error-only targets |
|---|---|---|---|---|
| 64 | fom_dst_exact_time | 1.22103 | 1 | [0.1, 0.05, 0.01, 0.001] |
| 64 | rom_cn_dt0.025 | 27.7753 | 0.0439615 | [0.1] |
| 64 | rom_cn_dt0.0125 | 37.1525 | 0.0319443 | [0.1] |
| 128 | fom_dst_exact_time | 1.36912 | 1 | [0.1, 0.05, 0.01, 0.001] |
| 128 | rom_cn_dt0.025 | 27.7379 | 0.0492523 | [0.1] |
| 128 | rom_cn_dt0.0125 | 37.1543 | 0.0365823 | [0.1] |

| Intervals | Method | Initial fit ms | Evolution ms | Readout ms | Input / output transfer ms |
|---|---|---|---|---|---|
| 64 | fom_dst_exact_time | 0 | 0.43299 | 0 | 0.348477 / 0.289749 |
| 64 | rom_cn_dt0.025 | 6.23627 | 20.0818 | 0.193649 | 0.203772 / 0.268745 |
| 64 | rom_cn_dt0.0125 | 6.21397 | 29.9139 | 0.157609 | 0.210787 / 0.282547 |
| 128 | fom_dst_exact_time | 0 | 0.338656 | 0 | 0.363138 / 0.495417 |
| 128 | rom_cn_dt0.025 | 6.23272 | 20.3843 | 0.203083 | 0.226856 / 0.313553 |
| 128 | rom_cn_dt0.0125 | 6.35212 | 29.8642 | 0.203292 | 0.234872 / 0.393739 |

For the direct FOM, the evolution column includes its inverse-transform field readout. Phase medians need not sum to the median total. These are measured modular query pipelines; a fused implementation could change dispatch overhead.

## Representation, dynamics and numerical-error diagnostics

| Intervals | Unrestricted-bank error median / worst | Reconstruction error median / worst | Discrete spatial error median / worst |
|---|---|---|---|
| 64 | 0.00644916 / 0.0103101 | 0.0503788 / 0.0973564 | 0.000675205 / 0.000709211 |
| 128 | 0.00644974 / 0.0103106 | 0.0503789 / 0.0973565 | 0.000168684 / 0.000177175 |

These reconstruction fits use extra truth-based starts only as diagnostics; they do not initialize autonomous rollouts. Unrestricted projection has the full bank dimension and is not a matched-dimension competitor.

| Intervals | CN dt | FOM CN-vs-semidiscrete error median / worst | ROM dt-refinement difference median / worst |
|---|---|---|---|
| 64 | 0.025 | 0.00019399 / 0.000217198 | 0.000785912 / 0.00146944 |
| 64 | 0.0125 | 4.84148e-05 / 5.42023e-05 | 0.000785912 / 0.00146944 |
| 128 | 0.025 | 0.000195118 / 0.000218478 | 0.000788764 / 0.00146576 |
| 128 | 0.0125 | 4.86954e-05 / 5.45213e-05 | 0.000788764 / 0.00146576 |

The FOM CN error isolates the timestep formula on the same spatial grid. The ROM coarse/fine difference includes nonlinear projection and solver effects and is not by itself a time-convergence order certificate.

## Solver status, decay and actual advancement

| Intervals | Method | Selected initial fits nonstationary | Rollout steps nonstationary / total | Budget-exhausted steps | Energy-increase trajectories | Minimum final/initial truth norm | Minimum first-to-final relative state change |
|---|---|---|---|---|---|---|---|
| 64 | rom_cn_dt0.025 | 0 / 4 | 0 / 80 | 0 | 0 / 4 | 0.633741 | 0.364408 |
| 64 | rom_cn_dt0.0125 | 0 / 4 | 0 / 160 | 0 | 0 / 4 | 0.633741 | 0.364401 |
| 128 | rom_cn_dt0.025 | 0 / 4 | 0 / 80 | 0 | 0 / 4 | 0.633741 | 0.364567 |
| 128 | rom_cn_dt0.0125 | 0 / 4 | 0 / 160 | 0 | 0 / 4 | 0.633741 | 0.36456 |

Stationarity uses the saved normalized gradient and the pinned tolerance. Small accepted steps and damping exhaustion remain nonstationary if their gradient misses that criterion. Raw reasons, gradients, attempts, accepted steps, all timing repetitions and every output field are preserved.

## Offline and new-mesh costs

| Work | Seconds |
|---|---|
| Reference verification | 10.0326 |
| Training data generation | 0.190981 |
| Training, including first-step compilation | 16.8017 |

| Intervals | Bank evaluation s | QR/operator assembly s | Bank MiB | Initialization projection MiB | Weak operator / Jacobian relative mismatch |
|---|---|---|---|---|---|
| 64 | 0.224513 | 2.63363 | 0.968994 | 0.968994 | 1.23881e-15 / 1.41542e-15 |
| 128 | 1.94386 | 1.27202 | 3.93774 | 3.93774 | 1.14128e-15 / 2.17725e-15 |

No amortization benefit is claimed without positive complete-query savings at attained accuracy. This first pilot does not include a classical coarser-grid output envelope, per-resolution retraining, a broad component-count heat cohort or independent training/data repeats. Those remain open.

## Common observation-grid and artifact audit

Native audit passed with 18 file checksums verified against the pulled archive and source content checked against git commit `ad882df3ecb8614d4cafa6a25761e920f3074e3d`. Saved field/error consistency maximum discrepancy: 0.

Every mesh's saved fields are restricted to the same 64-interval interior observation grid. The table uses every case and repetition. Eligibility requires selected initial fits and all rollout steps to be stationary, finite outputs, observed error plus reference uncertainty below target, and a reference uncertainty budget below one tenth of target. Eligibility is only for this development cohort; it is not independent confirmation.

| Solver intervals | Method | Common current error median / worst | All repetitions valid | Eligible development targets |
|---|---|---|---|---|
| 64 | fom_dst_exact_time | 0.000675205 / 0.000709211 | True | [0.1, 0.05, 0.01, 0.001] |
| 64 | rom_cn_dt0.025 | 0.0585754 / 0.0973564 | True | [0.1] |
| 64 | rom_cn_dt0.0125 | 0.0588214 / 0.0973564 | True | [0.1] |
| 128 | fom_dst_exact_time | 0.000168684 / 0.000177175 | True | [0.1, 0.05, 0.01, 0.001] |
| 128 | rom_cn_dt0.025 | 0.0585451 / 0.0973564 | True | [0.1] |
| 128 | rom_cn_dt0.0125 | 0.0587905 / 0.0973564 | True | [0.1] |

Unused initial starts that missed stationarity across all timed repetitions: 20. They remain recorded; each selected initial fit passed. The selected-fit criterion is explicit and does not hide the failed alternatives.

## Plain-language glossary

- **Intervals / bank / latent / weak tests:** cells along an axis / learned spatial functions / compressed state coordinates / smooth sine functions averaging the PDE.
- **FOM / NM-ROM / ROM:** full-grid solver / nonlinear-manifold reduced solver / reduced solver.
- **DST / FFT / CN:** sine transform / fast Fourier transform / Crank–Nicolson timestep. **Semi-discrete:** finite-difference space with exact-in-time propagation.
- **Physical reference / spatial error:** independently refined continuum sine solution / difference caused by the discrete spatial operator.
- **Current error / initial error / initial fit:** L2 discrepancy divided by current reference norm / initial reference norm / discrepancy in the fitted initial state.
- **Median / worst / cases above:** middle cohort value / largest cohort value / count missing the stated threshold. **Time maximum:** worst requested output time.
- **Query ms / paired ratio:** full input-to-output milliseconds / same-case FOM time divided by method time. A ratio above one means lower method latency.
- **Reconstruction / unrestricted bank / rollout:** best recorded truth fit / projection allowing every bank coefficient to vary / autonomous state evolution.
- **LM / QR / nonstationary:** damped least-squares solver / exact orthonormal–triangular compression of initial fitting / fitted gradient exceeds the declared tolerance.
- **Budget-exhausted / energy / decay:** iteration limit reached / half squared spatial L2 norm / field magnitude relative to its starting value.
- **Readout / transfer / assembly / MiB:** field reconstruction / movement between host and GPU / precomputing mesh operators / bytes divided by the binary megabyte.
- **Checkpoint / source commit / SHA-256 / provisional:** saved trained weights / pinned code revision / content-verification hash / development evidence awaiting review.
