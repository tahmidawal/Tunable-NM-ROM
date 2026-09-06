# Burgers 3D architecture screen

Generated from the linked raw run JSONs. These are provisional validation results; a representation screen does not establish rollout accuracy or wave transfer.

The four new architecture worktrees await base/name confirmation. This initial table records the common MLP controls; the four candidate heads have not run.

| Model | Width | Optimizer seed | Mean | Median | Worst | Above 15% | Unconverged | Tangent mean | Trainable parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_control | 128 | 200 | 5.5137% | 4.6721% | 17.0157% | 5 | 0 | 21.9473% | 41472 |
| mlp_control | 128 | 201 | 5.4921% | 4.6544% | 17.1652% | 4 | 0 | 22.0536% | 41472 |
| mlp_control | 192 | 200 | 5.6234% | 4.5991% | 18.7348% | 4 | 0 | 21.6555% | 72320 |
| mlp_control | 192 | 201 | 5.5568% | 4.7320% | 20.0435% | 3 | 1 | 21.5863% | 72320 |

All rows use the fixed learned bank and common linear initialization. Equal updates are not equal compute; the wider control has more parameters. Optimizer repeats share the same data. No cross-job timing comparison is made.

Independent Codex review accepted the controls as bounded negative representation results. Increasing width improved training fits and slightly improved tangent errors, but worsened validation mean and worst error in both repeats. All measured selected-fit Jacobians retained full latent rank. The wider model's unresolved fit is a different state from its stationary worst case. These observations do not establish global representation minima or rule out gains from other training choices.

The source bank, QR transform and anchor match across jobs. Independent GPU data regeneration produced roundoff-level differences in derived parameters and coefficient arrays, so shared hashes differ. Raw random draws and state membership match exactly; the bounded comparisons and source/output checks are saved in [control-provenance.json](runs/b3d_architecture/review/control-provenance.json). Logs contain host CPU-affinity warnings; both jobs used the GPU and completed. No timing claims use these runs.

- mlp_control, width 128, seed 200: initial-state mean 8.6194%, later-state mean 4.4785%, maximum invariant stationarity 2.482900e-07, maximum budget change 2.373578e-09. Job 3332190, NVIDIA A100 80GB PCIe, source commit `8484a9e00b41d0bf79872e14f69ed3c17fc185c9`. [Raw result](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/runs/b3d_architecture/mlp128/out/result.json).
- mlp_control, width 128, seed 201: initial-state mean 8.5264%, later-state mean 4.4806%, maximum invariant stationarity 7.951533e-08, maximum budget change 8.196901e-15. Job 3332190, NVIDIA A100 80GB PCIe, source commit `8484a9e00b41d0bf79872e14f69ed3c17fc185c9`. [Raw result](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/runs/b3d_architecture/mlp128/out/result.json).
- mlp_control, width 192, seed 200: initial-state mean 8.7402%, later-state mean 4.5845%, maximum invariant stationarity 6.731785e-08, maximum budget change 0.000000e+00. Job 3332191, NVIDIA A100 80GB PCIe, source commit `8484a9e00b41d0bf79872e14f69ed3c17fc185c9`. [Raw result](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/runs/b3d_architecture/mlp192/out/result.json).
- mlp_control, width 192, seed 201: initial-state mean 8.6560%, later-state mean 4.5238%, maximum invariant stationarity 3.811556e-05, maximum budget change 4.667303e-07. Job 3332191, NVIDIA A100 80GB PCIe, source commit `8484a9e00b41d0bf79872e14f69ed3c17fc185c9`. [Raw result](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/runs/b3d_architecture/mlp192/out/result.json).

Incomplete attempts:

None among the input files.

## Glossary

- **Model / width:** coefficient-generating head and units in each MLP hidden layer.
- **Optimizer seed:** fitting randomness; it does not regenerate an independent data cohort.
- **Mean / median / worst:** full-field relative reconstruction error over validation states.
- **Above 15%:** count of validation states exceeding the unchanged worst-error gate.
- **Unconverged:** selected fits whose inherited normalized-gradient measure exceeds 1e-6.
- **Tangent mean:** mean fraction of the actual PDE velocity outside the decoder tangent space.
- **Trainable parameters:** shared model weights; per-snapshot free latent codes are counted separately in JSON.
- **Initial / later states:** validation at the initial time, or after time stepping.
- **Invariant stationarity:** relative residual projected into available decoder directions.
- **Budget change:** largest relative error change between the two latent-solver attempt budgets.
- **Bank / latent code:** fixed learned spatial features, and reduced coordinates controlling their coefficients.
- **Validation / rollout:** unseen states for architecture selection, and online prediction through time.
