# Burgers 3D protected-anchor screen

Generated from saved raw results. These are provisional validation-only architecture results pending independent result review; no pilot, rollout or wave-transfer result is claimed.

All rows retain the learned spatial bank and use the common affine initialization. The anchor arm fixes the linear directions and projects its nonlinear correction away from them. The MLP controls are the previously saved campaign controls.

| Head | Seed | Training mean | Validation mean | Median | Worst | Above worst gate | Unconverged | Tangent mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP width 128 | 200 | 2.576505% | 5.513715% | 4.672066% | 17.015734% | 5 | 0 | 21.947347% |
| MLP width 128 | 201 | 2.576206% | 5.492084% | 4.654377% | 17.165202% | 4 | 0 | 22.053594% |
| MLP width 192 | 200 | 2.221864% | 5.623399% | 4.599069% | 18.734846% | 4 | 0 | 21.655485% |
| MLP width 192 | 201 | 2.228268% | 5.556837% | 4.731979% | 20.043463% | 3 | 1 | 21.586303% |
| Protected anchor | 200 | 2.935201% | 6.072002% | 5.115527% | 17.703013% | 5 | 0 | 23.385305% |
| Protected anchor | 201 | 2.929575% | 5.948452% | 5.066594% | 17.878533% | 6 | 0 | 23.079271% |

Equal updates are not equal compute. Optimizer repeats share the same data; no cross-job timing comparison is made. The legacy normalized-gradient criterion defines the unconverged count; invariant stationarity is reported separately below.

| Anchor seed | Initial-state mean | Later-state mean | Maximum gradient | Maximum invariant stationarity | Budget change |
|---|---:|---:|---:|---:|---:|
| 200 | 9.377268% | 4.970247% | 8.74180176e-09 | 5.71822622e-08 | 0.00000000e+00 |
| 201 | 9.259685% | 4.844708% | 9.29274715e-09 | 6.72787125e-08 | 4.64972058e-13 |

| Anchor seed | Recovery error | Anchor-Jacobian error | Minimum Gram eigenvalue | Minimum singular value | Maximum condition number |
|---|---:|---:|---:|---:|---:|
| 200 | 3.55271368e-14 | 1.55431223e-15 | 1.00075559e+00 | 1.00037772e+00 | 6.48511055e+00 |
| 201 | 2.84217094e-14 | 1.77635684e-15 | 1.00055418e+00 | 1.00027705e+00 | 5.59958142e+00 |

## Necessary screen checks

These checks do not replace the full inherited pilot, POD comparison or negative controls. Passing them alone would not permit rollout promotion.

| Anchor seed | Mean within gate | Worst within gate | Gradient within gate | Stable budget |
|---|---|---|---|---|
| 200 | FAIL | FAIL | PASS | PASS |
| 201 | FAIL | FAIL | PASS | PASS |

## Provenance and limits

Source commit `d0167ee8dca5f2aa58befddc141df101e64db60f`; job `3336232` on `pax051`, NVIDIA A100-PCIE-40GB. Backend `gpu`, x64 `True`, matmul precision `highest`.

Bank validation mean 2.182214%; truth residual maximum 9.99862718e-13; backward-Euler velocity identity maximum 9.94026471e-13.

The source/checkpoint hashes, output pull, precision flags and validation membership passed the local integrity audit. Actual data were regenerated from seed; archived parameter metadata supplied only the already bounded provenance reference.

Earlier queued job `3336132` was cancelled before execution to use the available GPU class. It produced no numerical result; its submission, cancellation and checked metadata pull remain archived. Replacement job `3336232` retained the same scientific settings.

The protected head imposes a graph over the chosen fixed anchor. Recovery and rank protection do not establish adequate physical tangent coverage, good conditioning or successful dynamics. A failed screen concerns this anchor and bounded optimization protocol; local multistart fits are not certified global minima.

[Raw result](runs/b3d_architecture/screen40/out/result.json), [local integrity audit](runs/b3d_architecture/screen40/local_audit.json), [implementation and geometry tests](B3D-ANCHOR-DESIGN.md).

## Glossary

- **Head / bank:** latent-to-coefficient map, and frozen learned spatial features.
- **Seed:** optimizer randomness; each repeat uses the same data cohort.
- **Training / validation:** snapshots used to fit the model, and unseen snapshots used to assess it.
- **Mean / median / worst:** relative full-field reconstruction errors over the indicated states.
- **Above worst gate / unconverged:** counts exceeding the error or normalized-gradient criteria.
- **Tangent mean:** mean relative truth velocity outside the decoder Jacobian range.
- **Initial / later:** snapshots before or after physical time stepping.
- **Gradient / invariant stationarity:** legacy normalized objective gradient, and residual projected onto available field directions.
- **Budget change:** maximum relative reconstruction change when latent-fit attempts are doubled.
- **Recovery / anchor-Jacobian error:** numerical violations of exact latent recovery or its derivative identity.
- **Gram / singular value / condition number:** derivative metric, directional stretching, and ratio of largest to smallest stretching.
- **Gate / pilot:** a required criterion, and the inherited complete validation procedure.
- **POD:** a linear comparator fitted from training snapshots.
- **Backward Euler:** the implicit time discretization used for truth trajectories.
- **Rollout / wave transfer:** online evolution, and any later evaluation on wave equations.
