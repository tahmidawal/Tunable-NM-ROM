# Smooth-mixture Burgers 3D architecture screen

Generated from the raw validation artifacts. Scientific conclusions remain provisional pending the coordinator's independent campaign audit; this screen does not establish rollout accuracy or reflective-wave transfer.

The two-expert smooth mixture retains the fixed spatial bank, latent dimension, training objective, cohort, and common affine initialization. The control rows below use the matching optimizer seeds and update schedule. Their runs use different GPU instances; no elapsed-time comparison is made.

| Model | Seed | Training mean | Validation mean | Median | Worst | Above worst gate | Unconverged | Tangent mean | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp128 | 200 | 2.576505% | 5.513715% | 4.672066% | 17.015734% | 5 | 0 | 21.947347% | 41472 |
| mlp128 | 201 | 2.576206% | 5.492084% | 4.654377% | 17.165202% | 4 | 0 | 22.053594% | 41472 |
| mlp192 | 200 | 2.221864% | 5.623399% | 4.599069% | 18.734846% | 4 | 0 | 21.655485% | 72320 |
| mlp192 | 201 | 2.228268% | 5.556837% | 4.731979% | 20.043463% | 3 | 1 | 21.586303% | 72320 |
| mixture | 200 | 2.160648% | 5.519008% | 4.652903% | 21.265062% | 4 | 0 | 21.173147% | 78786 |
| mixture | 201 | 2.157031% | 5.503608% | 4.619719% | 18.147213% | 3 | 0 | 21.178645% | 78786 |

The unrestricted bank mean error is 2.182214%. The retained POD comparison gives an effective mean limit of 4.305823%, using POD mean 8.611645% from the [inherited reference](runs/b3d_repair/pilot_head33/out/result.json). The worst-error limit remains 15.000000%. These are representation-screen requirements; a successful screen still requires the inherited pilot and all negative controls.

| Mixture seed | Initial mean | Later mean | p95 | Mean passes | Worst passes | Normalized-gradient passes | Budget relative change | Invariant stationarity max | Minimum rank |
|---|---:|---:|---:|---|---|---|---:|---:|---:|
| 200 | 8.554169% | 4.507287% | 12.017454% | False | False | True | 0.000000000e+00 | 7.158444085e-08 | 32 |
| 201 | 8.533771% | 4.493553% | 12.097519% | False | False | True | 0.000000000e+00 | 6.402768049e-08 | 32 |

Every completed mixture repeat fails the representation screen. This bounded outcome concerns the declared architecture and training protocol; it does not establish that all mixtures fail.

The selected fits are local multistart solutions, not certified global minima. The same errors, all starts and both solver budgets are retained in the JSON. Tangent errors measure the actual truth-state Burgers velocity outside the available decoder directions; they do not certify curvature or rollout.

| Mixture seed | Mean expert usage | Mean entropy | Saturated >0.95 | Saturated >0.99 | Relative expert disagreement RMS | Routing collapse | Identical experts |
|---|---|---:|---:|---:|---:|---|---|
| 200 | 0.497211547, 0.502788453 | 0.672229458 | 0 | 0 | 0.677916546 | False | False |
| 201 | 0.506958520, 0.493041480 | 0.682873295 | 0 | 0 | 0.729406838 | False | False |

Every validation routing weight, entropy and expert disagreement is saved. Routing-collapse and identical-expert flags use the declared descriptive thresholds in the [arm design](B3D-MIXTURE-DESIGN.md). Neither is a scientific acceptance gate, and healthy routing does not imply useful specialization. Equal updates and approximate parameter-count controls do not establish equal compute.

Source artifacts:

- [mlp128 raw result](runs/b3d_architecture/mlp128/out/result.json): job 3332190, NVIDIA A100 80GB PCIe, commit `8484a9e00b41d0bf79872e14f69ed3c17fc185c9`.
- [mlp192 raw result](runs/b3d_architecture/mlp192/out/result.json): job 3332191, NVIDIA A100 80GB PCIe, commit `8484a9e00b41d0bf79872e14f69ed3c17fc185c9`.
- [screen40 raw result](runs/b3d_architecture/screen40/out/result.json): job 3336230, NVIDIA A100-PCIE-40GB, commit `fd986d8d39483f7a1064d9f61e9593ccc5011fbc`.

[Implementation-agent provenance and independent NumPy checks](runs/b3d_architecture/screen40/audit.json) verify pulled checksums, source bytes, GPU/f64/highest configuration, seed/state membership, unchanged bank and QR, checkpoint hashes, independent coefficient reconstruction, and routing diagnostics. The coordinator's separate scientific audit remains pending. No host CPU-affinity warning occurred in this job. Backend preflight confirms GPU execution. No pilot, test cohort, larger grid, wave run, or online timing was opened for this arm.

## Glossary

- **Model / seed:** head architecture and optimizer randomness; optimizer repeats reuse the same generated data cohort.
- **Training / validation:** snapshots used for fitting, and unseen states reserved for selecting the architecture.
- **Mean / median / p95 / worst:** average, middle, 95th percentile, and largest relative full-field reconstruction errors.
- **Above worst gate:** number of validation states exceeding the retained worst-error limit.
- **Unconverged / normalized gradient:** selected latent fits failing the inherited first-order stopping requirement.
- **Tangent mean:** average fraction of the true PDE velocity unavailable through an infinitesimal latent change.
- **Parameters:** trained shared head weights, excluding separately optimized per-snapshot latent codes.
- **Initial / later:** validation at initial time, or after time stepping.
- **POD / bank / QR:** the inherited linear projection comparator, the fixed learned spatial features, and orthonormal coordinates for that bank.
- **Budget / multistart:** latent-solver attempt limit and several initial guesses; the minimum-error fit is kept even if unconverged.
- **Invariant stationarity:** normalized reconstruction residual projected onto the decoder's available local directions.
- **Rank:** numerically independent Jacobian directions under the declared SVD threshold; full rank does not guarantee small error.
- **Expert / usage:** one residual head, and its average smooth routing weight.
- **Entropy / saturation:** routing uncertainty in natural-log units, and a near-exclusive routing weight.
- **Disagreement RMS:** root mean square of expert-output differences normalized by the fixed training output scale.
- **Collapse / identical experts:** the separate, cohort-specific usage and output-similarity flags defined in the design.
- **Pilot / rollout:** inherited validation acceptance checks, and online time evolution from an initial condition.
- **f64 / highest:** double-precision arithmetic and the repository's required highest matrix-multiplication precision.
