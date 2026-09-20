# Burgers 3D overnight execution handoff

This is an operational handoff for the approved development campaign. New scientific results are provisional until collection and independent auditing; the sealed final cohort has not been opened.

## Active attempt

- Attempt `b3d004`, Slurm job `3995688`, namespace `/cluster/tufts/paralab/tawal01/paper_b3d_20260920/b3d004`.
- Scientific source `c026d56b83905293e71268c249b57324c7d01baf`; frozen configuration and queue checks are in `runs/b3d004/submission.json`.
- Started on `pax106`; `jax_backend=gpu` passed and Slurm stderr was empty at handoff. Four campaign lanes are active, one GPU each. Do not submit another B job while this one remains queued or running.
- Reuses audited B003 bank, K32 head, FNO and U-Net by manifest, while regenerating all physical training fields from seed. Trains DeepONet and Transolver next. The explicit coordinate channels are `[1,2,3]`; viscosity occupies the last channel.
- Runs the full K32 ROM/POD/FOM/four-operator panel in `out/`, including interpolation and two-case reference diagnostics. Then trains the K64/wider-head candidate on that same frozen spatial bank and original training fields, and runs its independent complete ROM/POD/FOM panel in `out/head64/` with checkpoint/config/optimizer artifacts in `head64/`.
- Eight diagnostic fitting starts are used on validation representation states for both heads. Those truth-assisted fits are upper bounds, not globally solved representation floors. The online nearest-code initializer is unchanged.
- Monitor `training.log`, `operator-training.log`, `driver.log`, `operator-evaluate.log`, `audit.log`, `head64-training.log`, `head64-driver.log`, `head64-audit.log` and `slurm.3995688.err`. Training code has passed a bounded native-grid forward/backward smoke and a tiny head-fit/unchanged-bank check.

## Collection

Wait until the numeric job is absent from the account queue. From this worktree:

```bash
/home/tahmid/Dev/.venv/bin/python experiments/paper-b3d/cluster/collect.py b3d004
OPENBLAS_NUM_THREADS=8 /home/tahmid/Dev/.venv/bin/python experiments/paper-b3d/audit.py \
  experiments/paper-b3d/runs/b3d004/collected/out \
  --checkpoint experiments/paper-b3d/runs/b3d004/collected/training/checkpoint.pkl \
  --audit-output audit-local.json
OPENBLAS_NUM_THREADS=8 /home/tahmid/Dev/.venv/bin/python experiments/paper-b3d/audit.py \
  experiments/paper-b3d/runs/b3d004/collected/out/head64 \
  --checkpoint experiments/paper-b3d/runs/b3d004/collected/head64/checkpoint.pkl \
  --audit-output audit-local.json
```

Run `audit_stationarity.py` with the same output/checkpoint pairs after the field audits. It independently reconstructs the sine test space, analytic network Jacobian and upwind-advection Jacobian in NumPy, then checks the largest recorded evolution and initialization gradients per method. It also reports the coefficient-Jacobian smallest singular value; inspect the K64/q192 endpoint for near redundancy. This is explicitly a sample, not an all-state stationarity certificate.

The collector verifies both the job's scientific manifest and a complete post-exit manifest including logs. The trap now retains a manifest and exit marker on intermediate failure too. Local audits use distinct filenames, preserving remote audit bytes. Only after successful checksums and auditing remove the literal verified remote attempt directory. Git-retain selected model checkpoints, compact records and provenance; retain dense fields and optimizer states in the checksum-covered local archive. No merging or pushing is authorized.

## Audited B003 panel

Attempt `b3d003`, job `3992083`, is complete, checksum-collected and independently audited; the exact remote directory was removed. Compact records and trained checkpoints were committed at `df968f70`. Its full local archive is `runs/b3d003/collected/`; `summary.json` is generated from raw invocations, and `COLLECTED.json` records disposition. The root report generator can read `collected/out/result.json` and `collected/out/audit-local.json` directly. The later independent analytic stationarity sample is `collected/out/audit-stationarity-local.json` and has its own `LOCAL_AUDITS.sha256`.

The current head has a substantial held-out best-found gap above its learned bank, despite a much smaller training fit gap. This is not a terminal-output rank bottleneck: its final hidden width equals the bank rank. The K64 candidate and increased fitting-start diagnostic address capacity and local-minimum alternatives. FNO, U-Net, higher-rank POD and full-order controls are strong and must remain visible. The generated summary contains exact numbers; do not recopy provisional numbers into paper prose.

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

## Next confirmation steps for the coordinator

After B004 completes and is audited, compare the original and wider heads using the declared initial-norm evolved metric, with the matched-dimension POD rows and all four operators. Check whether eight-start best-found fits materially alter the earlier head-gap diagnosis. If the larger head remains poor, a focused head-training or larger-bank development run is justified before freezing; do not claim that one architecture change proved the cause.

The concrete final-cohort proposal is a fresh parameter-table seed `920399`, rows `0` through `15`, with no parameter or field generation until configuration freeze. This is a proposal recorded before final-data access, not an assertion that it was pre-registered before all development. The current driver only draws its shared development table, so a separate final-evaluation seed/table path and a truthful final-cohort status field must be implemented before use. Frozen training rows, POD basis, correction directions and neural checkpoints must remain unchanged during final evaluation. A final job should evaluate both retained training seeds on the exact same final cases and GPU allocation, including all operator, classical and selected correction-rank rows.

For independent initialization confirmation, retrain the selected bank/head architecture with seed `920401`, and the four selected operator architectures with seeds `920410` through `920413`, using the unchanged original seed-zero training fields and opened development rows. The K64 head, if selected, can use head seed `920421`. Retain each seed's selected validation checkpoint and all curves; do not select a winning training seed. Freeze architectures, budgets, solver settings and selection rules in a committed JSON manifest before generating the new final fields. Aim to freeze by the coordinator's campaign hour-six deadline. These confirmation jobs have not been submitted.

Before a physical-accuracy claim, extend the coarse/fine comparison to every final case and check a further spatial/time refinement of the numerical reference. Current N33-vs-N65 differences can be a substantial fraction of the target and do not certify the N65 reference. A same-grid final comparison is still interpretable if an adequate physical reference cannot be established within the remaining budget, provided that limitation is printed beside the table. Higher-grid transfer and sampled quadrature remain unfinished; neither is needed to fabricate a claim of grid independence.
