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

The excluded GB10 all-selection 512-point attempt was terminated after more than 28 minutes and
never produced admissible science. Its duration is used only as a lower-bound resource warning:
the 16-hour request is more than 34 times that observed incomplete wall interval, but the excluded
run cannot establish an upper bound or a GPU speed ratio. The driver therefore writes atomic
health-only progress after data generation and every completed control, and both health-only
progress plus a latent-work checkpoint after every trust attempt. Partial metrics may not be
inspected or acted upon. The final audit binds the completed progress/checkpoint to the full trace.

The primary r1 request remains H200 because current availability and runtime/memory risk favor the
known Phase-7 device class. An H100 with at least 80 GB is an infrastructure-only fallback, not a
scientific arm: it may be considered only with new root authorization if r1 is still pending with
zero runtime, the exact numeric job is safely cancelled through the repository cancel helper, no
output exists, and the identical commit/manifest is restaged after exact remote cleanup. There is
no automatic GPU substitution, cancellation, or resubmission.

The exact lifecycle verifies local dependency bundles and full hashes, creates one exact root
manifest, stages directly into the paralab namespace only after queue/disk/absence checks, and
requires GPU/f64/highest plus clean logs and completed Slurm state. Pull admits only the four named
scientific/progress outputs and two logs, rechecks checksums, runs an independent negative-aware trace and
metric audit, and deletes only the exact remote cell after PASS. Synthetic smoke evidence is
recorded below only after a mandated local `jaxrun` execution under one minute.

## Excluded final synthetic smoke

The exact staged code from commit `77ab4f54a4f4333e3c97b929a1ddf6c0419f7a08` and local
manifest SHA-256 `e09b57371b7e070e95ccca6278000ea27b5a57a4cb416c9c7bb3a93c2bd1a290`
completed under `jaxrun`, GPU/f64/highest, in 30.521 seconds. It used two synthetic N16 cases and
two times for train and selection, one attempt, and no locked artifact argument. All eight
snapshot/start attempts were accepted with 160 JVP and 160 VJP calls, zero CG breakdown, zero
unhealthy exhaustion, exact boundary, and passing identity/health. The independent negative-aware
audit passed and retained `p8_d_valid=false` plus no T1/T2 authorization. JSON/NPZ/progress/work-
checkpoint/audit SHA-256 values are respectively `63f47fdc8af06841e54364abbc42914c20c04b0b905de3ec19f92270d8db4af0`,
`6740165ab68439eb96665a0d94f50eca2227739629056193c149e8afcaa72199`,
`dc36ac3135581dda52666366ffcf58cb0d4dfba2c06e371b97e26ac27fc2655c`,
`b140d6ae33be147b5bc566596bb13d509cf31a2b834f7ee1af9d69cc972ff409`, and
`d52229267d55f37f389e219cc7a6b6da0f208d45a99ea684bf5cb4b8e0f1c1f6`.
These are excluded implementation evidence, not scientific metrics.
