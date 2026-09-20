# Resumed NS3D capacity and final-comparison handoff

Capacity05 is an active development job. The final cohort remains unopened;
all trained and timed results are provisional until collected and audited.

The running scientific source is `4d675dd97c12ea47b90b7ca9720bc9bf0623da7f`,
job `4019914`, launched at 15:38:30 UTC on September 20, on A100 80GB/pax106.
The isolated remote path is
`/cluster/tufts/paralab/tawal01/paper_ns3d_20260920/capacity05`.
Its log contains `jax_backend=gpu`; standard error was empty at the last check.
The source is immutable. The future confirmation/final implementation is a
separate commit, `d0e8970d`, and must not modify the running stage.

Coverage04 is closed: independent membership/bank/head/timed-field audits,
all retained representation gradients, source hashes, archive roundtrip and
actual Git-blob archive hashes passed. The exact completed remote directory was
removed. Its source outputs remain restored locally under
`runs/coverage04/collected/output`, with durable split archive under
`artifacts/coverage04`. The history audit is explicitly not applicable because
its exploratory NM rollout gate failed; it is not a rollout success.

## Current job collection

After scheduler completion, collect the original `output`, `reuse`, `logs`,
`experiments`, `COMMIT.txt`, `PROVENANCE.json`, `MANIFEST.sha256`, `OUTPUTS.sha256`,
`run.sbatch` and `stage.json` directly from the exact paralab directory into
`runs/capacity05/collected`. Do not copy regenerated data into a successor job.
Use the absolute local Python and CPU thread bounds to run:

```bash
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/tahmid/Dev/.venv/bin/python experiments/ns3d/audit_coverage.py experiments/ns3d/runs/capacity05/collected --out experiments/ns3d/runs/capacity05/audit.json
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/tahmid/Dev/.venv/bin/python experiments/ns3d/audit_dense_histories.py experiments/ns3d/runs/capacity05/collected --out experiments/ns3d/runs/capacity05/history_audit.json
/home/tahmid/Dev/.venv/bin/python experiments/ns3d/audit_source.py experiments/ns3d/runs/capacity05/collected --out experiments/ns3d/runs/capacity05/source_audit.json
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/tahmid/Dev/.venv/bin/python experiments/ns3d/audit_representation_gradients.py experiments/ns3d/runs/capacity05/collected --out experiments/ns3d/runs/capacity05/representation_gradient_audit.json
```

Retain the complete collection with `retain_archive.py`, commit its split parts,
verify the actual Git blobs against the archive's hashes, and only then remove
the exact quiescent remote attempt. Preserve every failed numerical gate.
The expanded bank is warm tuning, not a complete independent training seed.

## Prepared confirmation implementation

The helpers are implemented and their tiny GPU/NumPy checks are retained in
`checks/frozen_pipeline_readiness.json`. The first all-method local compilation
smoke exceeded the required local wall cap; its incomplete output is retained.
The bounded follow-up passed complete-query, independent reference, field,
weak-gradient, decoder, correction and Galerkin parity checks. The DeepONet
teacher smoke independently verifies the vector denominator and exact learned
trunk Gram objective. A CPU fixture rejects configuration/checkpoint mutations
without calling any final parameter generator. These are implementation checks,
not larger-grid results.

After current-job audits pass, prepare the successor with:

```bash
/home/tahmid/Dev/.venv/bin/python experiments/ns3d/prepare_confirmation_reuse.py --coverage experiments/ns3d/runs/coverage04/collected --capacity experiments/ns3d/runs/capacity05/collected --output experiments/ns3d/runs/confirmation06_reuse --config experiments/ns3d/configs/confirmation06.json
```

This selects the smallest worst development representation error among trained
same-cohort heads, retaining the R1536 trained incumbent as a candidate. Affine
initialization candidates remain recorded but cannot silently replace a trained
NM-ROM in this comparison. The helper records actual selected training source,
seed and dimensions, copies checkpoints/retained replay outputs but no training
or reference data, and sets the bounded common confirmation configuration.

Commit the newly generated configuration before staging. Stage a fresh
`confirmation06` with driver `confirmation06.py`, A100 80GB, two-hour safety
limit. Move the prepared reuse directory into that stage as `reuse/`, then
regenerate its full manifest. Do not redirect a helper's stdout into a file
inside a stage while that helper hashes the stage: the changing log would
invalidate the manifest. Transfer directly to the isolated paralab path. Check
the account queue before/after submission, with one NS job and four campaign
allocations at most.

The confirmation regenerates exactly the same augmented training membership,
trains one bounded DeepONet candidate with a seeded approximate teacher plus
joint fine-tuning, and preserves the original checkpoint. Only full development
error selects between them. The scalar teacher is approximate and is not an
optimal-rank lower bound; the learned-trunk projection is checkpoint-specific.
All three velocity components are summed before the per-time vector loss.
The candidate keeps the original online architecture.

The job prepares actual frozen bank/head/POD/correction/operator assets and
replays the saved head representation. It then measures the full fixed-M neural
ladder, matched weak POD, efficient dense CNAB2 Galerkin POD/free-bank controls,
all four neural operators, and the FOM step ladder on development cases. The
cheapest passing FOM will be identified in the final freeze. Every invocation
includes initial fitting/projection and all requested fields. Dense FFT-based
weak/Galerkin evaluation remains grid-bound.

Collect and run `audit_frozen_panel.py`, `audit_dense_histories.py`,
`audit_source.py` and `audit_deeponet_pretraining.py`, saving respectively
`audit.json`, `history_audit.json`, `source_audit.json`, and `teacher_audit.json`
under `runs/confirmation06`. All four audit gates are mandatory in the freeze
helper. Retain and close the attempt by the same checksum/archive/Git procedure.

## Final evaluation

Copy the accepted confirmation config into `configs/final07.json`, changing only
its evaluation cohort to `final`, its timed count to the prospectively fixed
final count, setting `frozen_source_directory` to `reuse`, and setting
`final_freeze_path` to `experiments/ns3d/configs/final07freeze.json`. Retain the
reserved final seed and every trained/online choice. The authoritative final
count and seed are in `configs/final32_protocol.json`; the original missed and
resumed estimated deadlines are both recorded there.

Run `final_freeze.py` using the accepted confirmation's collected output as
`--source`; it requires all four accepted development audits, hashes every
actual offline asset, identifies the cheapest passing development FOM, and
refuses to overwrite an existing freeze. Commit both final config and freeze.
Stage a fresh `final07` with driver `final07.py` and that committed configuration.
Its reuse bundle contains the confirmation's `assets/`, `result.json`,
`timing_rows.json` and `timed_fields.npz` only. Rebuild the complete stage manifest.

The final driver verifies every frozen hash, performs an actual development
query replay with the fixed tolerance and identical stopping records, and only
then calls the final parameter generator. The final path never trains or refits
POD/correction/weak assets. It retains all final cases, timing arrays, physical
references, fields, latent histories, failures and freeze/replay proof. Collect,
audit and archive it before reporting final results. Do not tune after access.

Root owns the canonical lab log and main reports. Send source/job/audit hashes,
all qualified outcomes, and actual ETA to root. No merge or push is authorized.
An independent complete training seed remains pending; a warm expansion does
not establish seed robustness. Do not hide a negative NS result or change the
physical family to obtain a positive one.
