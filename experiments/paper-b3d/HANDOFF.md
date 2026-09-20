# Burgers 3D confirmation and final-evaluation handoff

B005 is fully collected and independently audited development evidence. B006 is independently audited, durably retained and rejected as a weaker common operator recipe. B007 is running the frozen final workflow; reserved cases remain unopened until both real development replays pass. The original overnight deadline was missed during the session interruption; completion is still pending.

## Active attempt

- Job `4021709`, attempt `b3d007`, scientific source `5eb99849c062f72c5dd9f6422f30928a14a0301d`; config SHA `e8ca941d120e36c6c38409597fa24e43380d4fa1d518d87043eb43f4090bf299`. Remote: `/cluster/tufts/paralab/tawal01/paper_b3d_20260920/b3d007`.
- Freeze SHA `d3d1dbda9e89da8e5d05d651994cd4d8f94ce30e32aa58176bf065cf90c5bb2f`. GPU preflight passed on pax106, an A100 allocation. Requested wall limit: two hours. The actual startup log says `jax_backend=gpu`.
- Both frozen real development replays run first. No reserved input may be generated unless both `out/replay-seed{0,1}/replay-audit.json` gates pass and `out/replay-complete.json` is written.
- Then both 32-case seed panels run with all four frozen operators and the complete classical control set, followed by the prospectively frozen streamed reference checks. Monitor `workflow.log`, `out/replay-seed{0,1}-driver.log`, `out/seed{0,1}-driver.log`, `out/seed{0,1}-operators.log`, `out/physical-reference.log`, Slurm stderr, and finally `out/complete.json`.
- No second Burgers allocation may be queued or running while this job is active.

B006's two complete, independently audited conditioned DeepONet candidates are rejected by the common-recipe selection rule:

| Seed | Original 100k worst (%) | Conditioned worst (%) |
| ---: | ---: | ---: |
| 0 | 14.953016 | 17.427401 |
| 1 | 13.880043 | 16.118442 |

Both original long-schedule DeepONets remain frozen. The candidate architecture was unchanged and its training teachers used training data only. B006 job 4018959 exited successfully, both checksum manifests and the independent NumPy/source audits pass, its scientific split archive passes actual-Git restoration, and its exact remote directory is removed. Its original local collection remains intact.

## B005 accepted evidence

Both panel audits pass across 2688 paired invocations and 26 sampled analytic NumPy gradient checks. Source auditing verifies 57 staged files against Git content. Both checksum manifests pass. Raw panels are `runs/b3d005/collected/out/seed{0,1}/result.json`; `runs/b3d005/summary.json` retains every timing repetition. Each model receives the same 512 training trajectories and eight observed times, and both panels use the same 16 opened development trajectories.

| Seed | Method | Worst evolved error (%) | Median GPU ms |
| ---: | --- | ---: | ---: |
| 0 | `rom_q0` | 8.987516 | 223.392410 |
| 0 | `rom_q192` | 2.434515 | 379.655695 |
| 0 | `pod_256` | 1.135743 | 310.113549 |
| 0 | `fno3d` | 2.499541 | 6.065142 |
| 0 | `unet3d` | 2.657453 | 2.174425 |
| 0 | `deeponet3d` | 14.953016 | 2.604709 |
| 0 | `transolver3d` | 2.148732 | 3.575348 |
| 1 | `rom_q0` | 8.228348 | 222.188354 |
| 1 | `rom_q192` | 2.554136 | 404.159067 |
| 1 | `pod_256` | 1.135743 | 310.231638 |
| 1 | `fno3d` | 2.469130 | 6.064135 |
| 1 | `unet3d` | 2.354178 | 2.176164 |
| 1 | `deeponet3d` | 13.880043 | 2.613644 |
| 1 | `transolver3d` | 2.668687 | 3.556121 |

The bank has rank 256 and the head dimension is 64. The actual fixed test count is 642; requested cutoff 640 completes a degenerate sine-eigenvalue shell. Both nonlinear ladders are stationary under the recorded tolerance, but remain slower than stronger classical/operator controls. The independent second seed supports reproducibility of this same-grid correction ladder; it does not establish a manifold advantage.

The streamed opened-case spatial discrepancy is 1.982912% versus the unchanged 0.500000% budget. The time discrepancy is 0.254331% versus 0.100000%. Both gates fail and remain explicit. The independent native residual-pair audit passes. The primary final comparison must therefore remain same-grid.

Full local arrays and optimizer states remain in the checksum-covered collected directory. In addition to the compact checkpoint/replay records, every scientific field from B003, B004 and B005 is now durably retained in committed lossless split archives with complete actual-Git restore/hash audits. `SCIENTIFIC-RETENTION.md` and each run's `SCIENTIFIC-RETENTION.json` link the evidence; earlier local-only retention records remain as history. B005 remote cleanup is recorded separately in `COLLECTED.json` and must be checked there.

## Required next work

First observe both real development replay results. If either fails, retain the failed gate and verify that the reserved cohort is still unopened; do not loosen the declared replay threshold. If both pass, monitor completion of both final panels and every prospective reference case without changing any frozen solver, model, rank, tolerance or row rule.

After B007 exits, checksum-collect its exact directory. The two final checkpoint paths are `collected/code/frozen/seed{0,1}/checkpoint.pkl`; the final panel paths are `collected/out/seed{0,1}`. Run both `audit.py` field checks with distinct `audit-local.json` output files, both `audit_stationarity.py` checks, `audit_contract.py collected/out`, and `audit_source.py b3d007`. The strengthened contract audit requires all reserved cases and every reference level, checks frozen hash links and all initial fields, and independently recomputes the full refinement/discretization discrepancy arrays.

Only after successful numerical/source audits, create an accurate `COLLECTED.json` and run `summarize_attempt.py b3d007`. Preserve the full collected directory. Use `retain_attempt.py` for compact records and `scientific_archive.py create/verify` for every scientific array, committing all split parts before verifying actual Git bytes. Regenerate `SCIENTIFIC-RETENTION.md` with `generate_retention_report.py`. Use `git -c gc.auto=0` for commits to avoid an unrequested shared-repository repack. After durable retention is verified, `cluster/cleanup.py b3d007` removes only the literal verified completed directory.

The prospective final seed is 920399 with 32 cases, and primary training seed index zero was selected before final access. The actual test count is fixed at 642. Empirical spatial/time refinement thresholds remain unchanged even if they fail; the primary result stays a same-grid comparison. Neither a failed physical-reference gate nor a negative method comparison is a failed numerical audit.

The coordinator owns canonical LAB-LOG.md and main paper reports. No merge or push is authorized. Untracked `checks/conditioned/{joint,pretraining}` are preserved from the previous owner; do not remove them.

## Glossary

- **Bank/head:** learned spatial functions / nonlinear latent-to-coefficient map.
- **Correction rank:** number of additional online linear coordinates, denoted $q$.
- **POD/FOM:** linear snapshot basis / full numerical PDE solver.
- **Development/final:** opened selection cases / reserved evaluation cases drawn after freezing choices.
- **Same-grid error:** discrepancy against the converged numerical solver on the identical mesh and time step, normalized by the supplied initial-field norm.
- **Median GPU ms:** median full-query device time, with dense requested inputs/outputs and interpolation charged; comparisons are within the allocation.
- **Stationarity:** the recorded normalized local gradient stopping rule; sampled gradients are independently recomputed.
- **Empirical refinement:** change under a finer numerical solve, not a rigorous continuum-error certificate.
- **Frozen offline arrays:** exact precomputed basis and correction data reused during evaluation.
