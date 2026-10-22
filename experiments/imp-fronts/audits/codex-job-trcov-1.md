**WRONG — fix before submission:** production mode and successful completion are not enforced.

- **CORRECT:** staged manifest hashes and shell syntax pass; baseline diff shows only viscosity mapping, registered gates, and provenance edits.
- **CORRECT:** coverage uses `lo=max(0.0016,a/1000)`; other parameter draws, RNG order, normalization, architecture and optimization remain unchanged.
- **CORRECT:** recipe seed is 1; data seeds remain 0 and 1000, with test seed 1, matching jcp-robust.
- **CORRECT:** default recipe runs 300k Stage A, full Stage B extraction/head fitting, and deployment rotation.
- **WRONG:** inherited `TRAIN2D_DRYRUN`, `TRAIN2D_SMOKE`, or `STAGES` can override production. Explicitly force `TRAIN2D_DRYRUN=0 TRAIN2D_SMOKE=0 STAGES=A,B,R`.
- **WRONG:** `ALL-DONE` requires only exit zero; `run_seed.sh` can return zero with incomplete stages. Require `SEED-COMPLETE`, production `run_record.json`, and expected outputs before declaring success.
- **CORRECT:** per-chunk finite/residual gate covers both stages, including test trajectories; its failure propagates to `EXHAUSTED.json` with “arm NOT RUN.”
- **CORRECT:** `gpu`, A100-80G, absolute cluster venv, x64, highest precision, and GPU preflight exit 42 are configured.
- **CORRECT:** wall is 04:36:00; work timeout subtracts elapsed setup and reserves 120 seconds, with up to 60 seconds termination grace.
- **NEEDS-RESTATEMENT:** “no self-timer” means no fractional job cutoff; the unchanged Stage B recipe retains `TIME_CAP=1500` and records truncated fitting.
- **CORRECT:** SM-b projects 4.34416 h against 4.56667 h available, about 801 seconds margin; scaling includes both seed-0 generations.
- **NEEDS-RESTATEMENT:** the timing anchor is seed 2’s slowest incumbent-recipe retrain, not K1’s own seed-1 run.
- **CORRECT:** outputs land under `/cluster/tufts/paralab/tawal01/imp-fronts/trcov/output/train2d_s1/`: Stage A `bank.pkl`, Stage B dense-mid checkpoint, rotation NPZ/JSON, and `run_record.json`.
- **CORRECT:** temporary files and configured caches stay in that namespace; output checksums are generated.
- **NEEDS-RESTATEMENT:** disk capacity remains unverified. `df` only prints availability; the coefficient array alone is 512 MiB uncompressed, plus checkpoints/caches. Require space evidence covering large outputs and retrieval.
- **NEEDS-RESTATEMENT:** this job produces the coverage bank; subsequent paired K1 floor/ROM evaluation answers whether coverage improves steep-front performance.

Read-only local audit completed; no files, jobs, SSH, scratch, or lab-log writes.