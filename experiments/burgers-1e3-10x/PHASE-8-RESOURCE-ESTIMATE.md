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

The exact staged code from commit `0feb11f834f194e209903720b1416fb3fd15570e` and local
manifest SHA-256 `e97d6266b9cb2ad00c91daef1d9ccd4d075cc9614ea8f14c6fefe83c348c13cb`
completed under `jaxrun`, GPU/f64/highest, in 23.300 seconds. It used two synthetic N16 cases and
two times for train and selection, one attempt, and no locked artifact argument. All eight
snapshot/start attempts were accepted with 160 JVP and 160 VJP calls, zero CG breakdown, zero
unhealthy exhaustion, exact boundary, and passing identity/health. The independent negative-aware
audit passed and retained `p8_d_valid=false` plus no T1/T2 authorization. JSON/NPZ/progress/work-
checkpoint/audit SHA-256 values are respectively `533ec8d801bd4ae237f3555a54944d8affe1e5af264228def42b9450563d48cb`,
`7f1b9b4b1254439bbaa5068aae00b7affe9f4f866709571d8b3e8a2bbe0a9588`,
`f54f13030d8abc55cd8119a83e6e9a3badf774f4d84ff65bd5df0278fdaddaf4`,
`7459fa1fe86328dcc47b2bf0a274c9a6a5514687e054dab9b46fecb43a81f4ea`, and
`e11b8e2dda4740b574b4f9d3fce9ac3c9ba64c1011b4d50df0908a79d3fc1a35`.
These are excluded implementation evidence, not scientific metrics.
