**SUBMIT — both calibration jobs.** No blocking defect found in their staged payloads.

| Item | Verdict | Evidence |
|---|---|---|
| Original refusal | CORRECT | Failed result and log show only `Urng` failed; calibration never started. GPU, f64 and highest precision were confirmed. |
| A7 comparison | CORRECT | Shape checked; integer arrays compared exactly; floats use `rtol=1e-14`, `atol=0`. One-ulp viscosity perturbation passes. |
| Seed/draw-order protection | CORRECT | In-memory tests reject changed seeds in both dimensions, swapped 2D coordinate draws and changed integer blob counts. |
| Absolute claim “cannot pass a real generator change” | WRONG | Three golden rows cannot cover every seed/row. Changes within the float tolerance also pass; `1e-14` permits more than one ulp. This does not invalidate this small, inspected delta. |
| Code-change scope | CORRECT | Requested diff contains only `gates.py:u_rng` and `audit_ref.py` changes. Generator, solver and golden data are unchanged. |
| Both manifests and provenance | CORRECT | Each has 19 verified manifest entries and 16 staged source blobs matching commit `84b66688ed3bd80f475d1d78ad4d0b94a41f1559`, including recorded sizes and hashes. |
| `calib2d2` | CORRECT | Calibration/dimension 2; A100-80GB, 120G RAM, 12 hours, 8 CPUs. Script identical to predecessor except attempt name/paths. |
| `calib3d2` | CORRECT | Calibration/dimension 3; H200, 240G RAM, 12 hours, 8 CPUs. Same script equivalence. |
| Execution safeguards | CORRECT | GPU partition, backend refusal, f64/highest precision, manifest verification and isolated paralab paths retained. Both scripts pass shell syntax checks. |
| `audit_ref.py` zero-denominator robustness | WRONG, nonblocking | Line 284 sets `eta=None` when `eps<=0`, then line 286 passes it to `np.isfinite`, causing `TypeError`. This file is neither staged nor invoked by these calibration jobs. |

The one-ulp diagnosis is recorded in A7; the failed result alone lacks raw RNG values to independently establish it.

No files modified, including the lab log; no jobs submitted.