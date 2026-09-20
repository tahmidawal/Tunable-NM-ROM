# Burgers 3D overnight execution handoff

This is an operational handoff for the approved development campaign. New scientific results are provisional until collection and independent auditing; the sealed final cohort has not been opened.

## Active attempt

- Attempt `b3d003`, Slurm job `3992083`, namespace `/cluster/tufts/paralab/tawal01/paper_b3d_20260920/b3d003`.
- Scientific source `dc91edb2c81712fde9d1b05fec7028c47bd213b4`; exact frozen configuration and queue checks are in `runs/b3d003/submission.json`.
- The job started on `pax106`, passed `jax_backend=gpu`, and had empty Slurm stderr at handoff. The two-hour limit and one-GPU contract remain in effect.
- Stage order: regenerate training data; spatial-bank/head training; independent two-case fine-space/time diagnostic; FNO training; U-Net training; complete same-allocation ROM/POD/FOM panel; operator timing and interpolation controls; NumPy saved-field audit.
- Monitor `training.log`, `training/bank_projection_curve.json`, `reference-screen.log`, `out/reference_screen/result.json`, `operator-training.log`, `driver.log`, `operator-evaluate.log`, `audit.log`, and `slurm.3992083.err` under the exact attempt directory.

The new spatial optimizer uses a prospectively declared longer budget, exact updates to training coefficients and invertible output-layer whitening. Worst projection error on opened validation snapshots selects its bank checkpoint. Projection and optimizer curves distinguish representation limitation from a poor jointly fitted coefficient vector. Validation fields do not enter gradients or coefficient updates. Keep rejected checkpoints and curves. Operator outputs use only the declared observed training times and interpolate to the full requested trajectory; the truth-knot interpolation control must remain visible beside operator errors.

## Collection

Wait until this numeric job is absent from the account queue. From this worktree:

```bash
/home/tahmid/Dev/.venv/bin/python experiments/paper-b3d/cluster/collect.py b3d003
OPENBLAS_NUM_THREADS=8 /home/tahmid/Dev/.venv/bin/python experiments/paper-b3d/audit.py \
  experiments/paper-b3d/runs/b3d003/collected/out \
  --checkpoint experiments/paper-b3d/runs/b3d003/collected/training/checkpoint.pkl \
  --audit-output audit-local.json
```

The local auditor accepts a distinct output filename so it preserves the remote audit bytes and their checksums. The collector verifies the job's scientific manifest plus a complete post-exit manifest including logs. After successful checksums and auditing, remove only the literal verified remote attempt directory, record that removal, and Git-retain the trained model checkpoints and compact result/provenance files. Dense fields remain in their local archive with checksum retention records. No merging or pushing is authorized.

## Previous attempt retained

Attempt `b3d002`, job `3989876`, generated its full reference cohort and fresh checkpoint, then stalled compiling the large batched multistart fit. It was preserved before cancellation, cancelled only by explicit numeric ID through the repository cancellation helper, checksum-collected again after exit, and its exact remote directory removed. `runs/b3d002/cancellation.json` records the disposition. Its final stage is `directions`, with no reduced trajectory invocations: there is no complete panel result to report.

The trained checkpoint is Git-retained at `runs/b3d002/collected/training/checkpoint.pkl`. Complete local scientific data and optimizer states are retained under the same collected directory and covered by `COLLECTION.sha256`. `out/audit-partial-local.json` independently checks all saved validation reference defects, bank orthogonality and finite NumPy network evaluation; it explicitly does not validate a nonexistent reduced trajectory panel. The raw bank/head training curves and projection summaries remain alongside it.

The replacement bounds multistart fitting to fixed tiles, preserving original inputs and starts. `checks/fit-tile-parity.json` verifies exact solver counters and decoded coefficients under the declared floating tolerance. `checks/operator-smoke.json` verifies fresh coefficient refresh, validation checkpoint selection, both operator training/inference paths, exact initial-field pass-through and full-trajectory interpolation. These are correctness smokes, not accuracy claims.

## Remaining scientific work

Inspect the empirical spatial/time discrepancy before promoting any physical-accuracy statement. Diagnose the selected bank floor, head best-found projection, solved initial compression and evolved error separately. Freeze development-selected models and solver settings before an independent final cohort. Training-seed replication, resolved physical references for every final case, larger-mesh comparisons and empirical quadrature remain unfinished until actually measured. Current dense initial fitting and weak advection do not establish a grid-independent online path.

The coordinator owns the canonical lab-log and generated paper reports; this worker has edited only the approved Burgers worktree and namespace.

## Glossary

- **Checkpoint:** persisted trained parameters and their training metadata.
- **Bank/head:** learned spatial functions / neural map from latent state to their coefficients.
- **Projection floor:** the smallest field error attainable by unrestricted coefficients in the current spatial span.
- **FNO/U-Net:** Fourier neural operator / convolutional encoder and decoder with skip connections.
- **POD/FOM:** a basis fitted by linear snapshot compression / the full numerical PDE solve.
- **Interpolation control:** true observed-time fields interpolated between outputs, exposing error due to the chosen temporal output parameterization.
- **Development/final:** opened cases used for model selection / untouched cases used after choices are frozen.
