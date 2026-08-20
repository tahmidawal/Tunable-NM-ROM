# Phase-8 P8-D resource and implementation checkpoint

Status: prospective implementation only. No scientific Phase-8 job has been staged remotely or
submitted. P8-D is exactly one cluster-only H200 diagnostic cell, `p8_d_r1`; T1/T2 are absent.

The cell regenerates the locked 35,904 training and 5,712 exposed-selection snapshots, binds the
immutable P4/P5/P6/P7 artifacts, and evaluates exact full grids for the free-H1, r3 encoder,
autolatent, predictor, and oracle controls. It then performs at most 40 matrix-free q19
trust-region attempts from each of two starts on all 5,712 selection snapshots. Full-grid fields
remain host-resident by resolution; decoder/trust batches are eight, and every JVP/VJP receives
truth, coordinates, masks, model parameters, and normalization as explicit JIT arguments.

Regenerated fields require about 3.5 GB, immutable targets/artifacts about 2 GB, and persisted
q/step traces about 0.2 GB before compression. Temporary N256 batches and compiled executables
fit well below the 96 GiB host request and one H200. The conservative request is one H200, eight
CPUs, 96 GiB, and 16 hours. This is an offline diagnostic; its elapsed time is not an online-cost
claim and no cross-job timing is compared.

The exact lifecycle verifies local dependency bundles and full hashes, creates one exact root
manifest, stages directly into the paralab namespace only after queue/disk/absence checks, and
requires GPU/f64/highest plus clean logs and completed Slurm state. Pull admits only the two named
scientific outputs and two logs, rechecks checksums, runs an independent negative-aware trace and
metric audit, and deletes only the exact remote cell after PASS. Synthetic smoke evidence is
recorded below only after a mandated local `jaxrun` execution under one minute.
