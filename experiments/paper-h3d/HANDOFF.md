# Heat3D tuning and operator handoff

Both the initial pilot and tune02 are complete, checksum-collected and independently audited. The frozen-checkpoint four-operator panel extra03 is running; no final cohort has been opened.

## Additional operator extension completed

Shared source `5933b3706c119e8ffbcfdf98db3940464af7b4fc` adds actual CNN-branch DeepONet and patchified structured-mesh 3D Transolver behind the same API. Exact file hashes are in `checks/operators-extra-source.json`. Copy **both** `operators/models3d.py` and `operators/extra_models3d.py`, retaining `operators/upstream/` attribution/license and the README. Existing FNO/U-Net paths are structurally identical after removal of only the new dispatch clauses; `training.py` is byte-identical to the previous shared source. This extension did not alter tune02 source or its artifacts.

Both new kinds require three explicitly supplied coordinate channels (default last three). Transolver also declares `coordinate_bounds`, matching the adapter's coordinate scaling. Its positional encoding uses actual grid origins/spacings, preserving interior/nodal/periodic conventions. Complete details and architecture distinctions are in `operators/README.md`; do not label this patchified baseline as Transolver++ or Transolver-3. The DeepONet trunk width is at least its architectural rank and its output is an actual branch/trunk product.

Meaningful bounded checks are retained in `checks/operators-extra-smoke.json`, `smokes/extra_{deeponet,transolver}/`, and `smokes/extra_native_{deeponet,transolver}/`. They cover independent NumPy slice-attention arithmetic, the complete 3D convolutional attention block against pinned upstream PyTorch with identical f64 parameters, periodic and Dirichlet gradients, odd-grid patch reconstruction, actual tiny training, independent selected-checkpoint errors, exact checkpoint replay, and both declared-capacity native-grid backward passes and direct finer-grid output. These are synthetic implementation smokes, never paper accuracy/training-convergence evidence. Native-capacity smoke checkpoints are regenerable and omitted; tiny replay checkpoints are retained.

`extra03.json` is the submitted same-cohort H configuration with the two new operators, the frozen earlier operators, and NM-ROM/POD/free-bank/DST controls measured again in its own allocation. Its source and active job are listed below; current resource cap remains one GPU job per PDE and four in total. Increasing data/bank capacity or further head/operator training requires a recorded development amendment, not opening the final cohort. No additional local GPU process remains active after this handoff.

Active job: `3995709`, `/cluster/tufts/paralab/tawal01/paper_h3d_20260920/extra03`, source `4ac8b16455f5b71bcdd560df3432cac43b8816d9`; config SHA256 `de2e24174b472855d30d646807e2003fa23a876009ce377c5e5f981cf60a5de9`. One GPU on `pax051`, `NVIDIA A100-PCIE-40GB`. GPU/f64/highest preflight and frozen-input/cohort gates passed. Static checkpoint parameters were uploaded during setup. Do not compare this allocation's wall clock with tune02's different GPU.

## Resume here

- Monitor extra03 `out/result.json`, `out/head_K32_curve.json`, operator `curve.json` files, `job.out` and `job.err` in the exact active directory. The new head has already completed training and the job has entered DeepONet training. Its live training-selection values remain provisional until collection.
- Once extra03 leaves the account queue, run `/home/tahmid/Dev/.venv/bin/python experiments/paper-h3d/cluster/collect.py extra03 --remove-verified` in this worktree. This checks original source/output manifests, independently audits saved fields into `runs/extra03/audit-local.json`, rechecks original bytes and removes only the exact verified remote directory.
- Then run `/home/tahmid/Dev/.venv/bin/python experiments/paper-h3d/audit_panel.py experiments/paper-h3d/runs/extra03/archive experiments/paper-h3d/runs/extra03/audit-panel.json`. It independently regenerates every reference with SciPy, verifies source against git blobs, and checks complete paired invocation coverage and summary aggregates. Keep original archives immutable.
- Read DESIGN A2/A3 and `extra03.json`: existing bank/K8/K16/FNO/U-Net are frozen; K32 is trained on the same bank/cohort and selected by development full-field fit error; only DeepONet/Transolver are newly trained among operators. Every model is remeasured with current ROM/POD/bank/DST controls in one allocation. Initial solves have their own larger cap; evolved settings are preserved.
- Frozen input provenance is `runs/extra03/inputs/tune02/ORIGIN.json`. The loader in `frozen.py` verifies input hashes and cohort identity and device-places model parameters once. Its strict quadrature reuse gate rejects any changed mesh/test/bank/weak-operator/file/config hash; a mismatch refits rather than borrowing a rule. Failed certificates remain visible, and known-uncertified sampled arms are omitted from the timed panel.
- Inspect native-grid operator training and separate direct transfer from charged native-grid interpolation. A future FNO arm should preserve padded physical domain size: the current fixed number of padded grid points changes that domain under mesh refinement. Treat this as an adapter/control diagnostic, not evidence of an intrinsic FNO limitation.
- Final cohort remains unopened. Freeze chosen configurations and perform independent training-seed checks before requesting root's final-cohort execution decision. No merge/push is authorized.

## Next bounded head diagnostic, if needed

The K32 candidate's live selected validation fit remains close to K16 despite its larger latent dimension. Its selected model is `extra03/out/head_K32.pkl`; resumable selected state is `head_K32_selected_partial.pkl`, and the complete curve/fit exits are `head_K32_curve.json`. After collection, those paths are under `runs/extra03/archive/out/`. Existing bank and K8/K16 checkpoints remain in `runs/tune02/archive/out/`, and are also copied into extra03's frozen input tree. Do not mistake the last optimizer checkpoint for the selected model.

Keep the same training trajectories and learned bank. Regenerate fields from the recorded seed; construct field-orthonormal coefficient targets as done in the extra-head branch of `run.py`. A useful next test initializes latent training codes from a training-only coefficient PCA/SVD and sets the head's linear skip to its reconstruction map, with any mean and scale recorded. This changes head initialization, not the spatial bank. Keep the initial affine/linear reconstruction as a control and include the initial checkpoint among validation candidates. Train the nonlinear residual/head and codes with the unchanged loss, retain the selection curve and all fit exits, and compare the same field norm on training and development. No validation data enter the PCA or gradient updates.

Compare held-out multistart fitting with the two declared start counts rather than silently changing the optimizer. `head_validation_starts` already controls checkpoint-selection fitting; `best_found_fields` currently uses a literal four starts and needs an explicit parameter if this is increased. Online initialization has its separate `initial_starts` setting. Keep these distinct and record them beside the result. A good head fit still needs its own frozen correction ladder and same-allocation measurements; it is not itself a trajectory result.

## Audited tune02

The complete result, generated summary and retained training/checkpoint provenance are in `runs/tune02/archive/out/`. `audit-local.json` covers all saved prediction fields; `audit-panel.json` covers independent references, source, complete repetitions and generated aggregates. The exact remote directory is removed. These are audited development results, not final-cohort confirmation.

The audits passed for 1312 saved fields, 3744 paired invocations and 78 independently recomputed summary rows.

| Intervals | Method | Median device ms | Worst evolved current-relative error (%) | Nonstationary cases |
|---:|---|---:|---:|---:|
| 32 | dst_exact | 1.319281 | 0.000000 | 0 |
| 32 | fno3d_w16_m6 | 7.241724 | 0.394686 | 0 |
| 32 | linear_bank_galerkin_exact | 0.156514 | 1.332380 | 0 |
| 32 | nmrom_K16_q0_dense | 17.226479 | 5.885451 | 2 |
| 32 | nmrom_K16_q96_dense | 23.816638 | 2.989914 | 1 |
| 32 | nmrom_K8_q0_dense | 12.967290 | 6.197678 | 0 |
| 32 | nmrom_K8_q96_dense | 15.397253 | 2.278149 | 1 |
| 32 | pod128_exact | 0.153810 | 1.244093 | 0 |
| 32 | unet3d_w8 | 3.562622 | 0.318500 | 0 |
| 64 | dst_exact | 1.328585 | 0.000000 | 0 |
| 64 | fno3d_w16_m6 | 10.980905 | 17.664133 | 0 |
| 64 | fno3d_w16_m6_native_grid_interpolated | 7.140024 | 0.989116 | 0 |
| 64 | linear_bank_galerkin_exact | 0.514389 | 1.316564 | 0 |
| 64 | nmrom_K16_q0_dense | 18.392188 | 5.842724 | 2 |
| 64 | nmrom_K16_q96_dense | 24.580931 | 2.962772 | 1 |
| 64 | nmrom_K8_q0_dense | 13.891935 | 6.147374 | 0 |
| 64 | nmrom_K8_q96_dense | 16.566503 | 2.262503 | 1 |
| 64 | pod128_exact | 0.507446 | 1.233758 | 0 |
| 64 | unet3d_w8 | 7.047398 | 92.675594 | 0 |
| 64 | unet3d_w8_native_grid_interpolated | 3.319375 | 0.946824 | 0 |

The efficient full-order solve remains more accurate than the ROM and operators. The free linear bank/POD controls are faster at their larger approximation error. All tune02 sampled cold-start certificates failed; dense trajectories and those failed sampled arms keep distinct interpretations. The original capped cold-start failures, direct-transfer failures and complete timing arrays remain in the archive.

## Audited first pilot

Job `3989545`, source `e6460d73c7d4d3292c9e9ef313ddb79c88bf59dd`; 2784 complete paired invocations and 992 independently checked saved fields. Exact remote directory deleted after checksum verification. Physical spectral-reference refinement maximum `1.435745163571614e-07` against the declared `0.0001` threshold.

| Intervals | Method | Median device ms | Worst evolved current-relative error | Nonstationary cases |
|---:|---|---:|---:|---:|
| 32 | dst_exact | 1.26438658 | 0 | 0 |
| 32 | linear_bank_exact | 0.407506479 | 0.0975118527 | 0 |
| 32 | linear_bank_galerkin_exact | 0.133202411 | 0.0962019062 | 0 |
| 32 | nmrom_K16_q0_dense | 17.151361 | 0.119299969 | 0 |
| 32 | nmrom_K16_q32_dense | 22.290098 | 0.0972628106 | 0 |
| 32 | nmrom_K8_q0_dense | 12.4540341 | 0.158873238 | 0 |
| 32 | nmrom_K8_q32_dense | 14.928741 | 0.0991073312 | 0 |
| 32 | pod64_exact | 0.136435614 | 0.0352881982 | 0 |
| 64 | dst_exact | 1.18603464 | 0 | 0 |
| 64 | linear_bank_exact | 0.802417984 | 0.0968920787 | 0 |
| 64 | linear_bank_galerkin_exact | 0.298453029 | 0.0959507108 | 0 |
| 64 | nmrom_K16_q0_dense | 17.5876725 | 0.118658843 | 0 |
| 64 | nmrom_K16_q32_dense | 22.8519139 | 0.0966814975 | 0 |
| 64 | nmrom_K8_q0_dense | 12.5164188 | 0.157870285 | 0 |
| 64 | nmrom_K8_q32_dense | 15.2603845 | 0.0985013455 | 0 |
| 64 | pod64_exact | 0.29366836 | 0.0353342226 | 0 |

The initial bank is inferior to rank-matched POD; corrections reach that bank limit. Every sampled cold-start certificate failed. These are valid negative development findings, not an accepted competitive final configuration.

## Artifact retention

Pilot source provenance, checksums, JSON metrics, trained bank/head checkpoints, optimizer checkpoints and quadrature arrays are tracked on this branch. Full compressed field sidecars remain in `runs/pilot01/archive/out/fields/` under their original checksum manifest; they are intentionally retained locally without adding the large arrays to git. Preserve or separately archive that directory before removing this worktree. The same retention requirement applies to the active run after collection.

Root owns the canonical LAB-LOG and main reports; this handoff supplies the closing facts for the root entry.

## Glossary

- **Bank / head:** learned spatial functions / nonlinear latent-to-coefficient map.
- **Correction rank:** additional coordinates solved with frozen learned weights.
- **Current-relative:** field error divided by the reference norm at the same time.
- **Stationarity:** small solver-objective gradient; distinct from accuracy.
- **POD / DST:** linear snapshot basis / direct discrete sine-transform solver.
- **Certificate:** held-out agreement of sampled and dense weak moments.
- **PCA / SVD:** training-data linear coordinate constructions; here considered only for initializing a head, not replacing the learned spatial bank.
- **Native grid / transfer:** the training mesh / direct network use on another mesh.
- **Development / final:** data used for tuning / data reserved until configurations freeze.

## Active continuation preparation

The original pilot and tune02 full-field sidecars are now durable in Git as split tar archives under `retained-fields/{pilot01,tune02}/`, committed at `30d7e58a`. Their manifests describe restoration, every member hash and every chunk hash. Streaming restoration passed for all fields. `git-retention-audit.json` independently read every committed Git blob and verified its actual bytes. Original local sidecars remain intact. Do not remove these retained archives during integration.

`head04.json` is prepared for the next allocation after extra03 is collected and audited. It first measures train/development error of DeepONet and Transolver, including DeepONet's learned trunk-span projection floor; then tests weighted-PCA code/linear-skip initialization for each declared latent dimension on the same frozen learned bank and training data. Every original random-initialized head is compared with explicit four/eight-start fitting. Selected latent states are saved, allowing `audit_head.py` to reconstruct the fields and normalized fitting gradients independently with NumPy/SciPy. After these diagnostics, DeepONet and Transolver continue from their prior selected checkpoints with freshly initialized Adam moments and the new declared cosine schedule. The initial checkpoint remains eligible for selection. This is a two-stage recipe, not independent-seed evidence.

The PCA smoke and independent analytic Jacobian/gradient checks pass. The continuation smoke passes initial-checkpoint selection and saved-weight replay. One preliminary continuation smoke could not initialize CUDA because the shared-memory machine had almost all free memory in file cache; its failure log is retained. Releasing cache for H's own completed archives resolved it. No failed GPU result was used.

Before staging head04, collect extra03 using the commands above, run its full independent panel audit, and check account queue occupancy with root. Frozen-input staging follows retained original input paths for reused earlier operator checkpoints and records each copied path. The staging source must be committed. Final data remain unopened; root owns canonical lab logging and paper reports.
