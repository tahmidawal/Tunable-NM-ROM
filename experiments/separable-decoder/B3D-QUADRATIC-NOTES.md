# Burgers 3D quadratic head screen

Generated from the retained run JSON and independent polynomial audit. These validation results remain provisional pending coordinator result review; they do not establish rollout accuracy or wave transfer.

| Head | Optimizer seed | Mean | Median | Worst | Above 15% | Unconverged | Tangent mean | Trainable parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP width 128 | 200 | 5.5137% | 4.6721% | 17.0157% | 5 | 0 | 21.9473% | 41472 |
| MLP width 128 | 201 | 5.4921% | 4.6544% | 17.1652% | 4 | 0 | 22.0536% | 41472 |
| MLP width 192 | 200 | 5.6234% | 4.5991% | 18.7348% | 4 | 0 | 21.6555% | 72320 |
| MLP width 192 | 201 | 5.5568% | 4.7320% | 20.0435% | 3 | 1 | 21.5863% | 72320 |
| Quadratic | 200 | 4.6894% | 3.8924% | 20.3053% | 2 | 0 | 18.0592% | 71808 |
| Quadratic | 201 | 4.6886% | 3.8873% | 20.2729% | 2 | 0 | 18.0514% | 71808 |

The quadratic head uses one undoubled product per latent-variable pair. Its affine initialization and zero quadratic weights are deterministic. Optimizer seeds change minibatches, not the model initialization or PDE cohort. The wider MLP supplies a close parameter-count control; equal updates do not imply equal compute. No cross-job timing comparison is made.

The effective inherited mean-error ceiling is 4.305823%, obtained from the POD comparison on the same cohort (POD mean 8.611645%). A mean below the looser standalone threshold is insufficient. Both the effective mean and worst-error checks remain required; this screen does not execute the full inherited pilot.

- Quadratic seed 200: training mean 2.1106%, initial-state validation mean 7.3711%, later-state mean 3.7955%. Maximum budget change 0.000000e+00. Maximum invariant stationarity 9.643750e-08. Worst-state normalized gradient 7.596399e-09; unconverged snapshot IDs []. Effective mean gate: fail; worst-error gate: fail; oracle/POD ratio 0.544542. Independent polynomial curvature audit found 0 selected stationary fits with negative objective curvature at the declared numerical cutoff.
- Quadratic seed 201: training mean 2.1104%, initial-state validation mean 7.3683%, later-state mean 3.7954%. Maximum budget change 0.000000e+00. Maximum invariant stationarity 7.537142e-08. Worst-state normalized gradient 5.951290e-09; unconverged snapshot IDs []. Effective mean gate: fail; worst-error gate: fail; oracle/POD ratio 0.544452. Independent polynomial curvature audit found 0 selected stationary fits with negative objective curvature at the declared numerical cutoff.

Job 3336338 ran on NVIDIA A100-PCIE-40GB from source commit `24799fe4c9a7e169a43eebf8d1ad62efe3ce0a2b`. Source content, output/log checksums, unchanged source bank, validation membership, and the common training schedule were independently verified. The audit recomputes loss, gradient, singular values, tangent error and projected stationarity using the explicit polynomial without importing the model or its differentiation implementation. Host affinity warnings are recorded separately in the audit. The bank-only mean validation error is 2.1822%.

Local multistart fits do not certify global reconstruction minima. Converged worst cases and unconverged fits must be distinguished before diagnosing representation. No inherited pilot, test-data evaluation, rollout or cost experiment is included in this bounded screen.

[Raw screen](runs/b3d_architecture/screen40/out/result.json) · [Independent audit](runs/b3d_arch_quadratic/screen40_audit.json) · [Design](B3D-QUADRATIC-DESIGN.md)

## Glossary

- **Head / latent code:** coefficient-generating model / its adjustable reduced coordinates.
- **Mean / median / worst:** relative field reconstruction errors over validation states.
- **Above 15%:** count exceeding the unchanged worst-error threshold.
- **Unconverged:** selected fits exceeding the inherited normalized-gradient threshold.
- **Tangent mean:** average fraction of truth-state PDE velocity outside available decoder directions.
- **Trainable parameters:** shared model weights, excluding per-snapshot optimized codes.
- **Optimizer seed:** minibatch randomness with the same PDE data and deterministic quadratic initialization.
- **Initial / later states:** unseen states before / after PDE time stepping.
- **Budget change:** largest relative error change when doubling latent-fit attempts.
- **Invariant stationarity:** reconstruction residual projected into decoder directions and normalized by full residual.
- **Normalized gradient:** the inherited first-order local-fit convergence diagnostic.
- **Snapshot ID:** trajectory index times the number of saved times plus its time index.
- **Polynomial curvature:** second derivatives; negative objective curvature can rule out a local minimum.
- **Numerical cutoff:** declared tolerance for deciding whether a computed quantity differs from zero.
- **Bank:** fixed learned spatial features; its unrestricted coefficients give a reconstruction floor.
- **POD / oracle:** training-derived linear projection comparator / truth-assisted local latent fit.
- **Pilot / rollout:** inherited validation checks / prediction forward in time.
