# Heat3D coverage and frozen final comparison handoff

The earlier pilot, tune02, extra03 and head04 diagnostics are collected and independently audited. Coverage05 is running. Final data remain unopened. Root alone owns the canonical LAB-LOG and main reports.

## Resume here

Coverage05 job `4018922` is running on pax105 A100 80 GB, submitted after the overnight jobs had completed and the queue was empty. Its immutable source commit is `91d60ace229bbd4a27001b312ced73ebac225d86`. Its unique directory is `/cluster/tufts/paralab/tawal01/paper_h3d_20260920/coverage05`. GPU/f64/highest preflight passed. Queue checks before and after submission are retained in the session record; the account cap is four allocated single GPUs and one live job per PDE lane. No other heat job may be submitted until this one completes.

The original deadline was missed during an agent usage-limit interruption. The resumed worker submitted this already-staged job at approximately 15:27 UTC, with a three-hour safety wall limit and an estimated two-hour training duration. Root is reallocating the worker slot to the fourth PDE while training proceeds. Reactivate the heat worker well before completion.

New `developmentA.json` and `developmentB.json` are generated from the declared coverage recipe by `prepare_development.py`. Stage the primary with `cluster/stage.py development06 --config developmentA.json` **after coverage collection/audit**. The stage helper now supports a development companion without requiring a final-freeze record. It runs the seedA native/fine full panel followed by seedB native accuracy/stationarity. Both candidate heads remain in this development-only panel. The final primary head has not been selected.

Run `audit_frozen_replay.py <coverage05-out/seedA> <development06-out> <audit-path>` and the corresponding seedB command after collection. Its prospectively fixed absolute field tolerance checks the actual trained checkpoints against real native-grid query inference and representation fits, and verifies byte-identical model files. This replay uses development fields only. The final can source accepted development06 output (seedB subdirectory for robustness), which also retains the primary POD assets.

Head04 passed the independent field/PCA/reference/analytic-gradient audit. Its full output is retained at archive commit `a2216043`, with actual Git-byte verification of every split chunk. The exact completed head04 cluster directory has been removed after those checks. Its original finite-budget DeepONet continuation remains weak; the longer Transolver improves. This diagnostic does not supply a new paired runtime comparison. A local audit bug shadowed the evaluation case count with a nonstationarity count, potentially skipping repeated-field equality comparisons. The repaired `audit_panel.py` now uses distinct variables and records the checked group count. Both tune02 and extra03 were independently rerun in `audit-panel-repetition-confirmed.json` and pass; no measured value changed.

The exact configuration is `coverage05.json`. The driver trains two fresh bank/head/operator seeds, plus an NM-ROM-only original-data control. SeedA was designated primary before submission; seedB is independent initialization robustness. All methods in the primary/robustness cohorts receive the same training members and six-time heat trajectory contract. The original training draw is an exactly checked prefix, not an assumed extension. No seed is selected using final results.

After coverage05 ends, collect with:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/tahmid/Dev/.venv/bin/python experiments/paper-h3d/cluster/collect.py coverage05 --remove-verified
/home/tahmid/Dev/.venv/bin/python experiments/paper-h3d/retain_fields.py coverage05 --all-output
```

The collector dispatches `audit_coverage.py`: original source/Git bytes, seeded cohorts, independent SciPy fields, NumPy bank reconstruction, every saved head latent/gradient, checkpoint selection and training metrics, operator development field errors, and independent weighted-POD/trunk/branch teacher checks. Commit the split archive and manifests, then run `retain_fields.py coverage05 --verify-git <that_commit>` and commit its verification. The complete saved output is retained; no training data were synced to the cluster.

## Required next comparison and final freeze

1. Inspect seedA/seedB development curves and saved fit states under `runs/coverage05/archive/out/{seedA,seedB}/`. The control128 record uses the same training recipe and random seeds as seedA. A changed number of training rows changes the trainable code array, so describe this as matched recipe/seeds, not identical full optimizer arrays.
2. Create a development panel from seedA with `frozen_source_attempt="coverage05"`, `frozen_source_seed="seedA"`, `frozen_input_directory="inputs/coverage05_seedA"`, `operators=[]`, and all four names in `frozen_operators`. Retain both candidate head dimensions for the development cost/error comparison, fixed weak-test count and correction ladder. Set `retain_solver_states=true`. Prefer `fit_quadrature=false` for these new heads: sampled initialization is untested, and the prior failed certificates remain archived. Do not claim certified hyper-reduction. `representation_oracles` may be true on development only.
3. Measure the primary development panel at both meshes in one allocation. `run.py` now saves `pod_N*.pkl` training artifacts. N32 is the fully matched native comparison; N64 separately labels frozen neural-weight transfer and POD rebuilt offline from the same training members. The corrected FNO physical-padding and DeepONet native-sensor/continuous-trunk flags are available. Keep original transfer variants visible. Select the final primary head and fixed q ladder using development accuracy and cost, not a desire to exaggerate the q span.
4. Prepare primary final config from the accepted development panel, changing `frozen_source_attempt` to that accepted development attempt and removing the primary `frozen_source_seed`: same full controls and N32/N64, `evaluation_cohort="final"`, `representation_oracles=false`, `fit_quadrature=false`, `retain_solver_states=true`, and `companion_config_file="code/finalB.json"`. The already reserved seed and count remain unchanged. Run `final_freeze.py --config <primary-config> --source <accepted-development-out> --output <primary-freeze.json>`; this hashes bank, selected head/corrections, all four operators and both saved POD artifacts. Its `final_freeze_path` must point to the staged `code/<primary-freeze.json>`.
5. Prepare `finalB.json` from the seedB training configuration, with the same physical problem/query settings/selected head/fixed q ladder as primary. Source the accepted development attempt / `seedB` after its real replay audit; all operators are frozen. Set `confirmation_role="independent_seed_accuracy"`, `primary_freeze_sha256` to the exact primary freeze hash, `include_linear_controls=false`, `include_half_step_control=false`, `evaluation_intervals=[32]`, `repetitions=1`; final seed/count and solver tolerance remain identical. Prepare its freeze from the accepted seedB output. The primary configuration only references the companion filename, avoiding a circular hash; the companion explicitly binds the primary freeze hash.
6. Commit both configurations and freeze records before staging the primary final attempt. `cluster/stage.py` recognizes the companion config, stages both immutable input bundles, and runs the primary followed by seedB within the same allocation. `final_freeze.verify` validates BOTH bundles before the primary draws any final parameters. Its positive protocol fixture and companion-tamper rejection are saved under `smokes/final-freeze-fixture/`; no final parameters were generated by that test. Perform real checkpoint replay on the development data before this launch.
7. Final collection automatically checks primary and seedB fields, all retained online states/analytic gradients, complete repetitions and independent reference/source audits before verified cleanup. Retain the complete output with `--all-output`, commit the chunks and verify their actual Git blobs. Report primary same-job costs, and the second seed as accuracy/stationarity robustness. Do not create cross-job speedup ratios or tune from final errors.

Selection target is 10:00–10:15 UTC; final runs, collection and manuscript handoff should finish by 12:07 UTC. The primary final panel includes classical controls, while seedB repeats only neural models and the fixed correction ladder at the native mesh. This preserves the full final cohort for initialization robustness without repeating identical classical timing sweeps. No merge or push is authorized.

## Prior audited evidence and retained artifacts

Pilot01, tune02 and extra03 raw outputs, source manifests, full field audits and panel audits remain under `runs/<attempt>/`. Their field sidecars are durable under `retained-fields/<attempt>/` with split-tar manifests, streaming restoration checks and actual committed-Git-blob audits. Head04 and coverage05 use `--all-output` because diagnostic states live outside a `fields/` subdirectory. Keep the original failed sampled certificates and direct-transfer variants. The newer code does not change any archived result.

## Glossary

- **Bank/head:** learned spatial functions and a nonlinear latent-to-coefficient map.
- **q / correction rank:** extra frozen-bank coordinates solved online together with the head coordinates.
- **Current-relative:** error divided by the reference field norm at the same time.
- **Stationarity:** small objective gradient; not proof of the global best fit.
- **POD/DST:** a linear basis fitted to training snapshots and an efficient sine-transform full-order solver.
- **Development/final:** tuning data and a cohort reserved until all configurations and model bytes are frozen.
- **Seed robustness:** variation between independent training initializations under the same recipe; distinct from variation across final inputs.
- **Teacher pretraining:** training-only supervision of the same DeepONet branch/trunk, followed by joint learning; the POD teacher is absent at inference.
