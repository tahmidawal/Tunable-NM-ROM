# Heat3D tuning and operator handoff

The original development pilot is complete and independently field-audited. The longer learned-bank/operator job is running; no final cohort has been opened.

## Additional operator extension completed

Shared source `5933b3706c119e8ffbcfdf98db3940464af7b4fc` adds actual CNN-branch DeepONet and patchified structured-mesh 3D Transolver behind the same API. Exact file hashes are in `checks/operators-extra-source.json`. Copy **both** `operators/models3d.py` and `operators/extra_models3d.py`, retaining `operators/upstream/` attribution/license and the README. Existing FNO/U-Net paths are structurally identical after removal of only the new dispatch clauses; `training.py` is byte-identical to the previous shared source. This extension does not alter active tune02 source or its artifacts.

Both new kinds require three explicitly supplied coordinate channels (default last three). Transolver also declares `coordinate_bounds`, matching the adapter's coordinate scaling. Its positional encoding uses actual grid origins/spacings, preserving interior/nodal/periodic conventions. Complete details and architecture distinctions are in `operators/README.md`; do not label this patchified baseline as Transolver++ or Transolver-3. The DeepONet trunk width is at least its architectural rank and its output is an actual branch/trunk product.

Meaningful bounded checks are retained in `checks/operators-extra-smoke.json`, `smokes/extra_{deeponet,transolver}/`, and `smokes/extra_native_{deeponet,transolver}/`. They cover independent NumPy slice-attention arithmetic, the complete 3D convolutional attention block against pinned upstream PyTorch with identical f64 parameters, periodic and Dirichlet gradients, odd-grid patch reconstruction, actual tiny training, independent selected-checkpoint errors, exact checkpoint replay, and both declared-capacity native-grid backward passes and direct finer-grid output. These are synthetic implementation smokes, never paper accuracy/training-convergence evidence. Native-capacity smoke checkpoints are regenerable and omitted; tiny replay checkpoints are retained.

`extra03.json` is a prospective same-cohort H configuration with the two new operators and the existing NM-ROM/POD/free-bank/DST controls measured again in its own allocation. It is **not submitted**. Parent/root decides its queue priority after tune02 finishes and is audited; current resource cap remains one GPU job per PDE and four in total. Increasing data/bank capacity or further head/operator training requires a recorded development amendment, not opening the final cohort. No additional local GPU process remains active after this handoff.

Active job: `3990698`, `/cluster/tufts/paralab/tawal01/paper_h3d_20260920/tune02`, source `4a93e5868acf83cbe07d84e6c451dc50ffbd9ea2`; config SHA256 `db3ba6452aa35b544ed0699a6b1a2f9aa6eb25c52acefe283af2c3b0327baf7e`. One GPU on `pax106`, `NVIDIA A100 80GB PCIe`. GPU/f64/highest preflight passed.

## Resume here

- Monitor `out/result.json`, `out/bank_curve.json`, operator `curve.json` files, `job.out` and `job.err` in the exact active directory.
- After the job leaves the account queue, run `/home/tahmid/Dev/.venv/bin/python experiments/paper-h3d/cluster/collect.py tune02 --remove-verified` in this worktree. The collector writes the independent audit outside the hashed archive and rechecks original manifests before deletion.
- Read `tune02.json` and DESIGN amendment A1: larger bank, training-only least-squares coefficient refresh, validation bank selection, longer heads, fixed-test rank ladder and operator models.
- The common FNO/U-Net implementation is independently pinned at `84d302a706bf06a460d04f419d9c06a6c424932e`; its model/training sources have not changed in the later heat-only adapter commit. P and NS owners have the API/source hashes.
- Inspect native-grid operator training first, then separate direct transfer from native-grid prediction with charged zero-wall nodal interpolation.
- Keep failed quadrature arms separate until the unchanged moment certificate passes. Larger bank training is promising, but projection error is not a trajectory result.
- Retain failed pilots and operator optimization curves; freeze configurations before touching the reserved final seed. No merging or pushing is authorized.

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
- **Native grid / transfer:** the training mesh / direct network use on another mesh.
- **Development / final:** data used for tuning / data reserved until configurations freeze.
