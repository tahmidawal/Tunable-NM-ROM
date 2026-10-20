**Overall: SUBMIT for all eight staged jobs.** No blocking defect found at `0c581ba20`. Two nonblocking issues are noted below.

| Job | GPU | Projection (h) | Time limit (h) | Verdict |
|---|---|---:|---:|---|
| refN0 | A100-80G | 2.7326 | 4.25 | SUBMIT |
| refN0s | A100-80G | 1.4080 | 2.25 | SUBMIT |
| refB1 | A100-80G | 2.7326 | 4.25 | SUBMIT |
| refB2 | A100-80G | 2.7326 | 4.25 | SUBMIT |
| rom1024 | A100-80G | 1.9914 | 3.00 | SUBMIT |
| ps | A100-80G | 1.5167 | 2.50 | SUBMIT |
| refAm | H200 | 4.6167 | 7.00 | SUBMIT |
| refC1b | H200 | 4.6167 | 7.00 | SUBMIT |

A9’s scientific rationale is sound **as a resource change**. `analyze.py`, `rscore.py`, and `make_report.py` do not use execution timings to determine scientific results; GPU-hours appear as accounting. Configurations and numerical algorithms are unchanged. However, “GPU type cannot affect any registered result” is too strong: floating-point differences can affect borderline qualification or reference-acceptance decisions. Hardware independence of the *decision rules* is established; identical outcomes across hardware are not.

The ledger records refC1 as cancelled with zero elapsed time and no allocated node. refC1b retains its original runs and configurations. I did not independently query Slurm to verify the historical queue-saturation explanation.

All eight stages passed:
- Complete inventories: **154 source files per job**, matching HEAD blobs and provenance; all manifest hashes verified.
- Config hashes, certificate parts, GPU fields, reservations, and recomputed projections agree.
- GRES/constraints and time limits match the plan; limits satisfy the 1.5× rule.
- Job-local CUDA/JAX/temp caches, disabled bytecode writes, f64/highest precision, and GPU preflight are present.

Budget arithmetic is correct: **16.2994 committed + 34.5 new = 50.7994 GPU-hours**, leaving **9.2006 hours** under the cap. The plan’s total projection is **32.730 hours**.

Memory supports submission. ROM smoke peak was **17,278,585,600 bytes (17.279 GB)** with the same mesh, bank, and setting shapes. Production has more cases, so this is evidence of headroom rather than a measured production peak. E0’s refined `test64/ST/c036` reached **40,562,322,176 bytes (40.562 GB)** at \(L=12288\). Both fit comfortably below the A100’s configured memory allowance. Ordinary-reference refinement is unexpected, not prohibited; the refined evidence supports memory feasibility if it occurs.

Nonblocking findings:
- **Timing-basis discrepancy:** E0’s stored 22 A100 / 80 H200 ST cases give medians **326.019 / 145.061 seconds**, versus A9’s 326 / 145.5. Using the exact ratio changes **none** of the eight rounded time limits.
- **Replan hardening:** `resources.py` writes certificates before its final cap check and does not enforce zero elapsed time for renamed cancellations. These weaknesses do not invalidate this checked plan: its cap passes and refC1’s recorded elapsed time is zero.

No files changed, no jobs submitted, and no lab-log entry written.