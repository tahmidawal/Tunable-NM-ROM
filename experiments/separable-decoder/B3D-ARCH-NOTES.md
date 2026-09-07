# Burgers 3D decoder architecture comparison

The numerical screen is complete. Generated from audited raw JSONs across the isolated worktrees. These are bounded validation results under the recorded training schedule; they do not establish global representation minima, rollout accuracy, or reflective-wave transfer.

Grid nodes per axis: 33; latent coordinates: 32; spatial bank size: 128. Each repeat uses 60000 updates, learning rate 0.0003, batch 4096, 8192 training states and 256 validation states. The unrestricted bank projection has mean error 2.1822% and worst error 8.0134%.

The inherited POD comparator gives an effective mean-error ceiling of 4.3058% (POD mean 8.6116%). The worst-error ceiling is 15.0000%. Passing this preliminary screen still requires the actual inherited pilot, including all negative controls, before rollout promotion. The POD value is the unchanged historical comparator for this grid, latent size and validation cohort; it is recomputed inside any promoted pilot.

## Reconstruction and acceptance

| Head | Optimizer seed | Training mean | Validation mean | Median | Worst | Above 15% | Unconverged | Preliminary failure |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MLP control 128 | 200 | 2.5765% | 5.5137% | 4.6721% | 17.0157% | 5 | 0 | mean/POD ratio, worst |
| MLP control 128 | 201 | 2.5762% | 5.4921% | 4.6544% | 17.1652% | 4 | 0 | mean/POD ratio, worst |
| MLP control 192 | 200 | 2.2219% | 5.6234% | 4.5991% | 18.7348% | 4 | 0 | mean/POD ratio, worst |
| MLP control 192 | 201 | 2.2283% | 5.5568% | 4.7320% | 20.0435% | 3 | 1 | mean/POD ratio, worst, stationarity |
| Protected anchor | 200 | 2.9352% | 6.0720% | 5.1155% | 17.7030% | 5 | 0 | mean/POD ratio, worst |
| Protected anchor | 201 | 2.9296% | 5.9485% | 5.0666% | 17.8785% | 6 | 0 | mean/POD ratio, worst |
| Quadratic | 200 | 2.1106% | 4.6894% | 3.8924% | 20.3053% | 2 | 0 | mean/POD ratio, worst |
| Quadratic | 201 | 2.1104% | 4.6886% | 3.8873% | 20.2729% | 2 | 0 | mean/POD ratio, worst |
| Shared encoder | 200 | 2.7137% | 5.4097% | 4.5502% | 16.1582% | 2 | 0 | mean/POD ratio, worst |
| Shared encoder | 201 | 2.7193% | 5.4940% | 4.6763% | 17.6365% | 4 | 0 | mean/POD ratio, worst |
| Smooth mixture | 200 | 2.1606% | 5.5190% | 4.6529% | 21.2651% | 4 | 0 | mean/POD ratio, worst |
| Smooth mixture | 201 | 2.1570% | 5.5036% | 4.6197% | 18.1472% | 3 | 0 | mean/POD ratio, worst |

All models use the fixed learned spatial bank and the same initial linear map. Two optimizer repeats share one data cohort. The historical warm-refined head used a different initialization/training history and is not an otherwise matched architecture control. Equal updates are not equal compute; no cross-job timing comparison is made.

## Tangent quality and model size

| Head | Seed | Tangent mean | Tangent median | Initial-state mean | Later-state mean | Minimum rank | Shared weights | Optimized code values |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP control 128 | 200 | 21.9473% | 19.6557% | 8.6194% | 4.4785% | 32 | 41472 | 262144 |
| MLP control 128 | 201 | 22.0536% | 19.2097% | 8.5264% | 4.4806% | 32 | 41472 | 262144 |
| MLP control 192 | 200 | 21.6555% | 18.9600% | 8.7402% | 4.5845% | 32 | 72320 | 262144 |
| MLP control 192 | 201 | 21.5863% | 18.9098% | 8.6560% | 4.5238% | 32 | 72320 | 262144 |
| Protected anchor | 200 | 23.3853% | 20.7358% | 9.3773% | 4.9702% | 32 | 37376 | 262144 |
| Protected anchor | 201 | 23.0793% | 21.0151% | 9.2597% | 4.8447% | 32 | 37376 | 262144 |
| Quadratic | 200 | 18.0592% | 15.5419% | 7.3711% | 3.7955% | 32 | 71808 | 262144 |
| Quadratic | 201 | 18.0514% | 15.5926% | 7.3683% | 3.7954% | 32 | 71808 | 262144 |
| Shared encoder | 200 | 21.5663% | 19.0791% | 8.3954% | 4.4145% | 32 | 78624 | 0 |
| Shared encoder | 201 | 21.8504% | 19.5677% | 8.6941% | 4.4274% | 32 | 78624 | 0 |
| Smooth mixture | 200 | 21.1731% | 19.2743% | 8.5542% | 4.5073% | 32 | 78786 | 262144 |
| Smooth mixture | 201 | 21.1786% | 19.3387% | 8.5338% | 4.4936% | 32 | 78786 | 262144 |

Tangent errors use the PDE velocity at the truth state, projected into the decoder Jacobian range at the fitted code. A smaller tangent error or full Jacobian rank alone does not certify time-stepping accuracy. The encoder operates offline; its shared-weight count includes both encoder and decoder, while its online decoder is the matched narrow MLP.

## Architecture-specific diagnostics

- Protected anchor, seed 200: minimum sampled Jacobian singular value 1.00038; anchor-coordinate recovery error 3.552714e-14; maximum sampled condition number 6.48511.
- Protected anchor, seed 201: minimum sampled Jacobian singular value 1.00028; anchor-coordinate recovery error 2.842171e-14; maximum sampled condition number 5.59958.
- Quadratic, seed 200: outlier snapshot IDs 27438, 28917; all are initial states: True. The initial parameters are deterministic, so these repeats vary minibatch randomness rather than data or parameter initialization.
- Quadratic, seed 201: outlier snapshot IDs 27438, 28917; all are initial states: True. The initial parameters are deterministic, so these repeats vary minibatch randomness rather than data or parameter initialization.
- Shared encoder, seed 200: direct encoder validation mean 7.0759%, median 5.8282%, worst 30.4328%. This direct encoding was not used to initialize the primary validation fits.
- Shared encoder, seed 201: direct encoder validation mean 7.3712%, median 6.1066%, worst 24.7863%. This direct encoding was not used to initialize the primary validation fits.
- Smooth mixture, seed 200: average expert weights 49.7212%, 50.2788%; mean routing entropy 0.672229 nats; routing-collapse flag False; identical-expert flag False; normalized expert disagreement RMS 0.677917. Flags describe these validation codes and are not acceptance gates.
- Smooth mixture, seed 201: average expert weights 50.6959%, 49.3041%; mean routing entropy 0.682873 nats; routing-collapse flag False; identical-expert flag False; normalized expert disagreement RMS 0.729407. Flags describe these validation codes and are not acceptance gates.

## Interpretation

Quadratic lowers mean error by 14.6293%–14.9503% and tangent mean by 17.7159%–18.1477% relative to the matched narrow MLP. These are relative improvements, not percentage-point differences. It is the strongest candidate in this bounded screen, but both the effective mean and worst-error conditions remain unsatisfied. Its large outliers are unseen initial states; converged local fitting does not certify a global minimum.

The fixed protected anchor preserves its intended geometry but worsens state and tangent accuracy. Shared encoder training gives an inconsistent reconstruction gain across repeats, while direct encoding has a further error gap. The smooth mixture improves training and tangent errors without improving validation mean or worst error; its declared collapse tests do not explain that failure. These findings apply to the tested forms, bank, data and training schedule.


## Convergence and provenance

- MLP control 128, seed 200: maximum inherited stationarity 2.917104e-08, maximum invariant stationarity 2.482900e-07, worst-state inherited stationarity 2.358739e-09, maximum budget change 2.373578e-09. Job 3332190, NVIDIA A100 80GB PCIe, source `8484a9e00b41d0bf79872e14f69ed3c17fc185c9`. [Raw result](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/runs/b3d_architecture/mlp128/out/result.json).
- MLP control 128, seed 201: maximum inherited stationarity 9.989707e-09, maximum invariant stationarity 7.951533e-08, worst-state inherited stationarity 7.728372e-09, maximum budget change 8.196901e-15. Job 3332190, NVIDIA A100 80GB PCIe, source `8484a9e00b41d0bf79872e14f69ed3c17fc185c9`. [Raw result](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/runs/b3d_architecture/mlp128/out/result.json).
- MLP control 192, seed 200: maximum inherited stationarity 9.452589e-09, maximum invariant stationarity 6.731785e-08, worst-state inherited stationarity 6.833635e-09, maximum budget change 0.000000e+00. Job 3332191, NVIDIA A100 80GB PCIe, source `8484a9e00b41d0bf79872e14f69ed3c17fc185c9`. [Raw result](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/runs/b3d_architecture/mlp192/out/result.json).
- MLP control 192, seed 201: maximum inherited stationarity 5.504878e-06, maximum invariant stationarity 3.811556e-05, worst-state inherited stationarity 6.660306e-09, maximum budget change 4.667303e-07. Job 3332191, NVIDIA A100 80GB PCIe, source `8484a9e00b41d0bf79872e14f69ed3c17fc185c9`. [Raw result](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/runs/b3d_architecture/mlp192/out/result.json).
- Protected anchor, seed 200: maximum inherited stationarity 8.741802e-09, maximum invariant stationarity 5.718226e-08, worst-state inherited stationarity 5.859317e-09, maximum budget change 0.000000e+00. Job 3336232, NVIDIA A100-PCIE-40GB, source `d0167ee8dca5f2aa58befddc141df101e64db60f`. [Raw result](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-06-b3d-anchor/experiments/separable-decoder/runs/b3d_architecture/screen40/out/result.json).
- Protected anchor, seed 201: maximum inherited stationarity 9.292747e-09, maximum invariant stationarity 6.727871e-08, worst-state inherited stationarity 7.413567e-09, maximum budget change 4.649721e-13. Job 3336232, NVIDIA A100-PCIE-40GB, source `d0167ee8dca5f2aa58befddc141df101e64db60f`. [Raw result](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-06-b3d-anchor/experiments/separable-decoder/runs/b3d_architecture/screen40/out/result.json).
- Quadratic, seed 200: maximum inherited stationarity 8.162339e-09, maximum invariant stationarity 9.643750e-08, worst-state inherited stationarity 7.596399e-09, maximum budget change 0.000000e+00. Job 3336338, NVIDIA A100-PCIE-40GB, source `24799fe4c9a7e169a43eebf8d1ad62efe3ce0a2b`. [Raw result](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-06-b3d-quadratic/experiments/separable-decoder/runs/b3d_architecture/screen40/out/result.json).
- Quadratic, seed 201: maximum inherited stationarity 7.687744e-09, maximum invariant stationarity 7.537142e-08, worst-state inherited stationarity 5.951290e-09, maximum budget change 0.000000e+00. Job 3336338, NVIDIA A100-PCIE-40GB, source `24799fe4c9a7e169a43eebf8d1ad62efe3ce0a2b`. [Raw result](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-06-b3d-quadratic/experiments/separable-decoder/runs/b3d_architecture/screen40/out/result.json).
- Shared encoder, seed 200: maximum inherited stationarity 9.368969e-09, maximum invariant stationarity 7.280469e-08, worst-state inherited stationarity 6.587931e-09, maximum budget change 0.000000e+00. Job 3336240, NVIDIA A100-PCIE-40GB, source `5f54f29642192d1d4dff2c80aaca6b46ae3e09ed`. [Raw result](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-06-b3d-encoder/experiments/separable-decoder/runs/b3d_architecture/screen40/out/result.json).
- Shared encoder, seed 201: maximum inherited stationarity 9.611860e-09, maximum invariant stationarity 7.262946e-08, worst-state inherited stationarity 5.841227e-09, maximum budget change 3.169064e-12. Job 3336240, NVIDIA A100-PCIE-40GB, source `5f54f29642192d1d4dff2c80aaca6b46ae3e09ed`. [Raw result](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-06-b3d-encoder/experiments/separable-decoder/runs/b3d_architecture/screen40/out/result.json).
- Smooth mixture, seed 200: maximum inherited stationarity 9.511717e-09, maximum invariant stationarity 7.158444e-08, worst-state inherited stationarity 9.511717e-09, maximum budget change 0.000000e+00. Job 3336230, NVIDIA A100-PCIE-40GB, source `fd986d8d39483f7a1064d9f61e9593ccc5011fbc`. [Raw result](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-06-b3d-mixture/experiments/separable-decoder/runs/b3d_architecture/screen40/out/result.json).
- Smooth mixture, seed 201: maximum inherited stationarity 8.710164e-09, maximum invariant stationarity 6.402768e-08, worst-state inherited stationarity 2.399448e-09, maximum budget change 0.000000e+00. Job 3336230, NVIDIA A100-PCIE-40GB, source `fd986d8d39483f7a1064d9f61e9593ccc5011fbc`. [Raw result](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-06-b3d-mixture/experiments/separable-decoder/runs/b3d_architecture/screen40/out/result.json).

The audit verifies committed source bytes, source checkpoint, output/log checksums, precision/backend, parameter provenance, validation membership and summaries recomputed from raw arrays. Independent GPU regeneration can produce roundoff differences in derived parameters and coefficient arrays; shared hashes need not be byte-identical. Raw random draws and membership must match, and numerical arrays must pass the recorded comparison. Host CPU-affinity warnings are recorded separately; these runs are not used for timing claims. [Audit evidence](runs/b3d_architecture/review/campaign-audit.json).

## Incomplete or invalid attempts

None among the discovered result files.

- [b3d_arch_anchor, attempt screen](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-06-b3d-anchor/experiments/separable-decoder/runs/b3d_architecture/screen/submission.json): no numerical output; preserved submission/accounting records distinguish queue cancellation from execution failure.
- [b3d_arch_encoder, attempt screen](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-06-b3d-encoder/experiments/separable-decoder/runs/b3d_architecture/screen/submission.json): no numerical output; preserved submission/accounting records distinguish queue cancellation from execution failure.

## Glossary

- **Head / bank / code:** nonlinear coefficient map, fixed learned spatial features, and reduced coordinates.
- **Optimizer seed:** fitting randomness; both repeats share the same PDE cases.
- **Training / validation:** states used in fitting, and unseen states used to assess the model.
- **Mean / median / worst:** full-field relative reconstruction errors on the indicated cohort.
- **POD:** the inherited linear projection comparator with the same number of latent coordinates.
- **Above 15%:** validation-state count exceeding the unchanged worst-error ceiling.
- **Unconverged / stationarity:** selected fits above the inherited normalized-gradient tolerance.
- **Invariant stationarity:** residual fraction projected along available decoder directions.
- **Tangent:** field changes available through the decoder Jacobian; error measures missing PDE velocity.
- **Initial / later:** states at the initial condition, and after time stepping.
- **Rank / singular value / condition number:** independent local decoder directions, directional sensitivity, and their largest-to-smallest ratio.
- **Shared weights / optimized code values:** learned network parameters, and separately optimized per-snapshot latent coordinates.
- **Encoder:** offline map from solution coefficients to a latent code.
- **Routing / entropy / nats:** mixture probabilities, their uncertainty, and its natural-log units.
- **Expert collapse / RMS:** negligible use or indistinguishable expert outputs under declared thresholds, and root-mean-square magnitude.
- **Budget change:** largest relative error change when increasing the latent-solver attempt budget.
- **Pilot / promotion / rollout:** inherited validation checks, advancement after all checks pass, and online prediction through time.
