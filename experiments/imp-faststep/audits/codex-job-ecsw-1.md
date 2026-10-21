**SUBMIT — the staged `ecsw` pilot passes this read-only job audit.** No blocking defect found. This does not certify its eventual results.

1. **CORRECT — staged provenance and manifest.** All **35 provenance entries** match the committed blobs at `aa1231cdd11d7d1705d90b745aebe8c0e6c05e1f`, including byte counts and SHA256 digests. All **40 manifest entries** verify; no staged files are missing or uncovered, excluding the manifest itself. Submission scripts match that commit. The ECSW driver, config and cluster scripts are unchanged from the prior READY audit.

2. **CORRECT — batch allocation and execution.** `gpu` partition, one H200, `200G` host memory, and **01:15:00 = 1.25 hours** agree with the reservation and config. Absolute cluster venv, highest matmul precision, x64 and GPU preflight exit 42 are present. Logs, temporary files, caches and outputs use the lane’s paralab directory; the staged `logs/` directory exists. Driver timeout is **71 minutes**, with a 30-second kill grace. Driver failures propagate; successful exit without `COMPLETE` becomes exit 5. Output checksums are generated.

3. **CORRECT — POD inputs and artifact contract.** Config pins basis digest `8ae2be237bd849482c3bdf024e3f80c757544ec77bb746a67a60fcf632341a86`, checked before loading. Ranks are **128/256/384**, with NNLS checkpoints **κ=2/4/8/16**. The driver writes `GECSW_r{128,256,384}.npz` in the format consumed by `podarm.py`, plus `ecsw.json` containing checkpoint availability, recomputed C11 residuals, witness evidence, attempt summaries and acceptance status. Unreached checkpoints remain unavailable, as registered. Failed C11 sets `accepted=false`, exits 3 and withholds `COMPLETE`.

4. **CORRECT — admission and concurrency.** Local ledger records **0 spent, 0 reserved**, with no jobs or unresolved intents. This reservation gives **1.25 ≤ 4 GPU-hours**, leaving 2.75 hours. Submission refreshes accounting under the lane lock before admission. Under the shared campaign lock, the remote script rejects queue-query failures, campaign counts ≥8 and lane counts ≥2. Existing directories and duplicate submissions are refused; ambiguous submission outcomes retain reconciliation evidence.

5. **NEEDS-RESTATEMENT — non-blocking operational limits.**
   - `submit.sh`’s header incorrectly says rejected `sbatch` removes the directory; implementation correctly preserves ambiguous outcomes.
   - The configured NNLS attempt budgets total at most **2,400 seconds**, but overshoot, finalization, state generation and compilation add time. Completion within the allocation remains a pilot measurement, consistent with A3.5.
   - Memory fit remains unmeasured. The rank-384 contribution matrix alone occupies **12.08 GB on device**, before assembly temporaries and solver buffers.
   - Acceptance here means construction/C11 acceptance; downstream rollout qualification and collected-artifact checksum pinning remain required.

Live queue/accounting, remote free space and archived basis availability were not checked under the no-cluster-command restriction. No files were changed, no jobs submitted, and no scratch space used.