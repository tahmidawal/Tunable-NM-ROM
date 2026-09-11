# Poisson staged-accuracy job handoff

Status: RUNNING, incomplete and unaudited. The GPU preflight, focused training/rank/oracle controls and complete integration smoke passed. The real run has completed all staged phases and the matched joint control, generated all references, and begun multiresolution evaluation at N64. Do not interpret the first `STAGED ACCURACY COMPLETE` log marker as the real study: it belongs to `out/smoke`. Real completion requires `out/pilot/result.json` with `complete=true`, successful accounting and `EXIT_CODE=0`.

- Source commit: `db66d194efa56a9fecbee9f0c116f42d7f356e7e`.
- Attempt: `staged_accuracy08`; Slurm job: `3563323` on `pax049`.
- Remote: `/cluster/tufts/paralab/tawal01/mr_poisson2d_20260907/staged_accuracy08`.
- Log: `/cluster/tufts/paralab/tawal01/mr_poisson2d_20260907/staged_accuracy08/logs/3563323.out` and `.err`.
- Local record: `experiments/multiresolution-poisson/runs/staged_accuracy08` in the existing approved Poisson worktree.
- Scientific files: `config-staged-accuracy.json`, `STAGED-ACCURACY-DESIGN.md`, `staged_training.py`, `staged_oracles.py`, `staged_accuracy.py` in this experiment.

The current network remains k16/r64. Frozen original, ordinary joint continuation matched to staged optimizer-block seconds, staged-head endpoint and staged-joint endpoint are timed. Staged-bank parameters and free coefficients are separately preserved. All 30 existing development cases plus 12 new preregistered development cases are evaluated at [64, 256, 1024], with 3 repetitions, common stationary solve controls, three CG tolerances and DST. No final cases, no retraining on evaluation sources, no new worktree or merge.

## Monitor and collect

Use read-only `squeue`/`sacct` and log reads. Never cancel another owner's job. After this exact job leaves the queue:

```bash
/home/tahmid/Dev/.venv/bin/python experiments/multiresolution-poisson/staged_cluster.py collect staged_accuracy08
```

The collector verifies all source/result/pull hashes, builds a restorable split archive and removes only the exact completed remote attempt. It also handles terminal failed attempts, whose evidence must be retained before any corrected rerun. Never overwrite or resubmit this attempt directory.

## Remaining owner work after reactivation

Implement/adapt the independent CPU audit from `reports/audit_online_tuning.py` and the existing per-checkpoint `reports/audit04.py`. The new run has multiple archived checkpoints, model-tagged operators and 42 cases, so the old tuning audit cannot be run unchanged. Audit all sources/checkpoints against frozen git bytes; full field/reference metrics; bank/QR ranks and projections; every online latent/initial code/weak residual/gradient/stop; all reference-only head candidate fields, perpendicular residual and best stationary selector; training subtree freezes, saved endpoint weights and field-norm QR loss; normalized training-only POD span; and every optimizer timing block/time-match sum, rounding rule and update exposure. All optimizer blocks are retained. Local ARM/x86 `exp` can differ by an ulp: regenerate source values numerically while preserving exact archive hashes, as in earlier audits.

The original frozen checkpoint is repackaged with run metadata under `out/pilot/checkpoints/original_relative.pkl`; prove its weight/code identity to `in/model.pkl`, rather than requiring identical pickle bytes. Every other endpoint is in that same checkpoint directory; staged-bank free and optimal coefficients are in `bank_free_coefficients.npz`. POD basis/truth/predictions are diagnostic only, stored in `pod_diagnostic.npz` on the training mesh. `head_oracles` exists only at the configured diagnostic mesh and includes all candidate fields and real stationarity; missing stationary fits must remain explicit.

After audit passes, copy the native result to this run's `result.json`, save `audit.json`, then generate:

```bash
/home/tahmid/Dev/.venv/bin/python experiments/multiresolution-poisson/reports/summarize_staged_accuracy.py experiments/multiresolution-poisson/runs/staged_accuracy08
```

That generator is implemented but intentionally gated on audit success. It produces `panel.json` with separate existing/new/all development aggregates, every method, full-bank/head diagnostics and same-job comparators. Independently review it before publication. Commit the scoped run and split archive, verify remote absence and a clean owned tree, append numerical closure to the absolute canonical LAB-LOG, and send the machine panel/commit to root. Root owns main reports and current-status block.
