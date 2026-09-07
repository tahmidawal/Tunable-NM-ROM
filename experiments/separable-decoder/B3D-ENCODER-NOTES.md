# Burgers shared-encoder architecture screen

Generated from the raw run JSONs. Results are provisional pending the coordinator's independent result review; this validation screen establishes neither rollout accuracy nor reflective-wave transfer.

The candidate uses exactly the control decoder architecture. It replaces independent training codes with a shared offline encoder of solution coefficients. All rows use the same frozen bank, cohort, common affine initialization, optimizer schedule and multistart fitting protocol.

| Training scheme | Seed | Validation mean | Median | Worst | Above 15% | Unconverged | Tangent mean | Training mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Free codes, matched MLP | 200 | 5.5137% | 4.6721% | 17.0157% | 5 | 0 | 21.9473% | 2.5765% |
| Free codes, matched MLP | 201 | 5.4921% | 4.6544% | 17.1652% | 4 | 0 | 22.0536% | 2.5762% |
| Shared encoder | 200 | 5.4097% | 4.5502% | 16.1582% | 2 | 0 | 21.5663% | 2.7137% |
| Shared encoder | 201 | 5.4940% | 4.6763% | 17.6365% | 4 | 0 | 21.8504% | 2.7193% |

The primary table uses latent fits from zero and the codes of the same predeclared training states. It does not use the validation snapshot encoder as an initial guess. Local fits are not established global minima.

| Seed | Direct encoder mean | Median | Worst | Above 15% | Initial-state mean | Later-state mean |
|---|---:|---:|---:|---:|---:|---:|
| 200 | 7.0759% | 5.8282% | 30.4328% | 22 | 11.4411% | 5.6208% |
| 201 | 7.3712% | 6.1066% | 24.7863% | 28 | 11.9682% | 5.8389% |

Direct encoder reconstruction is a separate snapshot diagnostic that consumes the solution coefficients. It is not an online PDE prediction or a cost result.

| Seed | Decoder parameters | Encoder parameters | Free training coordinates | Initial-state fitted mean | Later-state fitted mean | Minimum Jacobian rank | Maximum invariant stationarity | Maximum budget change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 41472 | 37152 | 0 | 8.3954% | 4.4145% | 32 | 7.280469e-08 | 0.000000e+00 |
| 201 | 41472 | 37152 | 0 | 8.6941% | 4.4274% | 32 | 7.262946e-08 | 3.169064e-12 |

The matched control has 41472 decoder parameters and 262144 independently optimized training coordinates. The encoder adds offline shared parameters without increasing decoder capacity. Equal updates are not equal compute; no cross-job timing comparison is made.

The shared encoder improves fitted validation mean in 1 of 2 repeats and tangent mean in 2 of 2 repeats. It does not consistently improve reconstruction or establish a passing representation. The direct encoder has an additional approximation gap relative to latent fitting.

Acceptance status:

- Seed 200: fails mean error, worst error. The inherited full pilot has not run; POD comparison and all operator/negative controls are still required for promotion.
- Seed 201: fails mean error, worst error. The inherited full pilot has not run; POD comparison and all operator/negative controls are still required for promotion.

Source commit `5f54f29642192d1d4dff2c80aaca6b46ae3e09ed`, job 3336240, GPU NVIDIA A100-PCIE-40GB, backend `gpu`, f64 `True`, matrix precision `highest`. The two optimizer repeats share one data seed. No final-test data were opened.

[Encoder raw result](runs/b3d_architecture/screen40/out/result.json). [Matched-control raw result](runs/b3d_architecture/mlp128/out/result.json). [Source and artifact verification](runs/b3d_architecture/screen40/verification.json).

The original A100-80GB request was canceled while pending and replaced with an available A100-40GB resource request. The scientific configuration was unchanged. [Cancellation record](runs/b3d_architecture/screen/completion.log).

## Glossary

- **Training scheme / shared encoder / free codes:** how training snapshots obtain their reduced coordinates: a shared solution-to-code network or a separate optimized vector for every snapshot.
- **Seed:** optimizer randomness; these repeats use the same data.
- **Mean / median / worst:** relative full-field reconstruction errors over the validation cohort.
- **Above 15% / unconverged:** counts exceeding the inherited worst-error or normalized-gradient requirements.
- **Tangent mean:** mean relative physical velocity missing from the decoder derivative range.
- **Training mean:** reconstruction error using assigned training codes, rather than latent-fitted validation codes.
- **Direct encoder:** solution-coefficient encoding without a subsequent latent fit; not an online PDE solve.
- **Initial / later states:** validation at the initial time or after evolution.
- **Decoder / encoder parameters:** weights in the coefficient generator or offline snapshot-to-coordinate network.
- **Free training coordinates:** per-snapshot optimized unknowns in addition to shared model weights.
- **Jacobian rank / invariant stationarity:** number of independent decoder derivative directions and the residual projected along those directions.
- **Budget change:** maximum relative reconstruction change when the latent-solver attempt budget is doubled.
- **Bank / POD comparison / pilot:** frozen spatial patterns, comparison with the inherited linear projection baseline, and the complete numerical acceptance checks.
- **f64:** double-precision arithmetic.
