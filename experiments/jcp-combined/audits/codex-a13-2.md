**c2grid (2D): READY for production E0 scoring.**  
**c3grid (3D): READY once the first-order solver selection is frozen; NOT READY to run E0 analysis before then.**

| Item | Verdict | Verification |
|---|---|---|
| Mandatory loader contract | **PASS** | `score.e0_load` enforces allowed cohorts, pinned store version, normaliser agreement and initial-field identity. |
| Every E0 consumer | **PASS** | Both scoring functions, the 2D FOM path, `analyze2d`, `budget2d` and `budget3d` supply both `n0` and `u0`. The remaining argument-free call is restricted to first-order negative controls. |
| H3 uncertainty | **PASS** | Under E0, `analyze2d` adds case-wise `eta` to anchor uncertainty before evaluating all three guards. Output discloses the conditional Richardson interpretation. |
| Budget certification | **PASS** | Both budgets require `--certified` and `--frozen`, check original-reference passing gates, bind the certificate to the same `result.json`, and require the frozen file’s hash to equal the job’s recorded binding. |
| Eligibility metadata | **PASS** | Both budgets carry `not_rankable`, including the three unresolved 2D S case-kinds. |
| `label_of` wording | **PASS** | Qualifies order verification by eligibility, discloses conditional uncertainty and post-data E0 A10, and labels unmatched S diagnostics. |
| A13.4 | **PASS** | Movement bounds are narrowed to corresponding ST/S errors on identical fields/lattices; old TX → new ST and broader statistics are excluded. |
| A13.5 | **PASS** | Documents the fixes and explicitly postpones 3D E0 analysis until the first-order solver selection is frozen. |
| E0 reporting | **STILL OPEN** | `make_report.py` still reads first-order outputs and presents separate ST/TX columns. This blocks declaring A13 reporting complete, but does not block generating production E0 scores. |

The existing c2grid first-order G-score passes and matches its archived `result.json` hash. The frozen file currently contains `2d` and `3d_rules`, consistent with your stated pending 3D solver selection.

Verification comprised source inspection, five syntax checks, and in-memory loader tests: three valid inputs accepted and seven invalid inputs rejected. No production rescoring was run.

No files were created, modified or appended.