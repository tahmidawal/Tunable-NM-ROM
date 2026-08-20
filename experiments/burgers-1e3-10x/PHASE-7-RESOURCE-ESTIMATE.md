# Phase 7 G1 seed-11 training preflight

Status: prospective implementation/resource checkpoint. No scientific Phase-7 job has been
submitted. The single permitted cell is `p7_g1_s11_r1`; it binds the immutable audited P4-D,
P5-D target, and P6-D repair chains and cannot access model-validation or confirmation data.

## Fixed work and retained memory

The cell regenerates 704 train and 112 exposed-selection trajectories at all 51 times. It reads,
but never refits, the 35,904 immutable P5 H1 coefficient targets. It executes exactly 10,000 G1
plus mirrored-encoder warmup updates, a bitwise encoder-to-autolatent handoff, 30,000 joint G1 plus
autolatent updates, 20,000 direct-predictor updates, and three 10,000-update exposed q-oracle starts
with only the preregistered complete-checkpoint early stop. All full-field evaluations use the P6
Cox-weak/K3-full actual route; this cell does not run weak/EQ or N1024 timing.

The dominant host arrays are approximately 3.5 GB of regenerated train/selection fields, 0.96 GB
of normalized P5 coefficients, and 0.92 GB of immutable staged target files. Autolatents, optimizer
states, deterministic schedules, checkpoints, and batched full-grid work add comfortably less than
10 GB. The implementation never duplicates the denormalized and normalized target matrix, streams
sampled field batches from host memory, and batches full evaluation by eight snapshots. A 96 GiB
request therefore has a conservative margin over both host and accelerator-resident live arrays.

The earlier dense Phase-4 seed-11 cell regenerated the same field cohorts and completed in 8:37 on
an H200, but the G1 convolutional generator adds full 48x48 and 32x32 head work to every training
batch. That prior elapsed time is resource-planning context only, never a timing or speed gate. The
request is one H200, 8 CPUs, 96 GiB, and 16 hours. BLAS/OpenMP are pinned to one thread; CPUs support
field generation and host sampling rather than a claimed parallel training speedup.

## Excluded local execution

The first synthetic-only N16 smoke used two training cases, two selection cases, two times, one
update per stage, and no locked P4/P5/P6 data. Under the mandated GB10 `jaxrun`, GPU/f64/highest path,
it completed with `ALL-DONE` in 48 wall seconds. K3/Cox identity passed; deliberately untrained
accuracy gates were false and were recorded as an excluded negative result, never as scientific
evidence. Exact-staged rerun plus the shared independent audit is required before submission.

## Lifecycle

`cluster/make_phase7_train_cell.sh` admits only the fixed prior cells and exact clean commit, verifies
their local checksums/audits, binds the inherited Burgers runtime hashes to the P4 root manifest,
copies every immutable P5 target with its recorded name/hash, and creates one root manifest excluding
only itself. `cluster/launch_phase7_train_cell.sh` stages directly into the assigned namespace after
queue/disk checks. `cluster/pull_phase7_train.sh` requires completed Slurm/GPU/ALL-DONE health,
checksums the complete artifacts, rejects warning/OOM/disk/captured-constant evidence, runs the same
negative-aware audit, and deletes only the exact remote cell after audit success.

Pre-submit review rejected the first exact stage because its local smoke created Python bytecode
cache files after root-manifest generation. No scientific job used that stage. The lifecycle now
requires exact equality between the manifest path set and the actual stage path set locally and on
the remote before submission, repeats the exact-set check at pull with only the five named scientific
outputs/logs admitted, and sets `PYTHONDONTWRITEBYTECODE=1` in the batch. This is infrastructure-only
hardening; no scientific execution, method, data, or gate changed.
