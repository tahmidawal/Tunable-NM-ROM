**Overall: GO for submission.** No blocking defect found.

| Item | Verdict | Evidence |
|---|---|---|
| 1. Staged payload | **CORRECT** | All **114** manifest entries pass. All **21** staged source/config/checkpoint files match HEAD; staged imports resolve. All **13** training artifacts match tr1b’s `OUTPUTS.sha256`, including all **5** rotations in `LARGE-FILES.sha256`. All **77** reference files match refdv’s manifest. |
| 2. Batch execution | **CORRECT** | GPU partition; six A100s, one node, 12 CPUs, 300G, 10 hours; highest precision and x64; per-device GPU preflight; individual PID waits; timing on device 0 only after all workers succeed; output checksums and failure propagation. Shell syntax passes. |
| 3. Variant mapping | **CORRECT** | Frozen/deployed uses `rotation_R512.npz`; frozen/lane uses **base’s `rotation_frozen.npz`**. Retrained base/sob01/sig2/sig1 checkpoints and rotations match their registered recipes. |
| 4. Metrics and gates | **CORRECT** | Outputs support amended projection/gradient metrics, three-population quadrature ladders, spectra, S/ST rollouts, and paired timing. R1, C4, C5a, C6 and completeness checks are wired through evaluation, collection and reporting. Passing these remains a post-run requirement. |
| 5. Post-audit diff | **CORRECT** | `fom_comparators` is disabled in every ev1 task. Added `ref_*_evolved_129` fields correctly restrict fields and initial-condition normalization to 129 nodes; existing solver paths and metrics are unchanged. |

**NEEDS-RESTATEMENT:** provenance records pre-staging commit `88679c0c2`; HEAD `d178c9492` only adds staging records. No source discrepancy. A6 permits **60 allocated GPU-hours** for ev1; A100 runtime/peak memory remain unmeasured, and the earlier lane-total budget is stale.

Read-only audit completed; no repository or lab-log files modified.