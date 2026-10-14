**HOLD: c2budA, c2budB, c2budC.** Audited HEAD `d1beec3d2d87b56fa1e63ea90cb65a977cf9e9e8`. The original single-field mutations now fail closed, but additional in-memory mutations expose remaining gaps.

| Item | Verdict | Finding |
|---|---|---|
| 2D required-row coverage and multiplicity | **CORRECT** | Rejects deleted rows/cases, duplicate rows, missing controls, missing setting evidence and failed ordinary rows. |
| B1 evidence counts | **CORRECT** | Both dimensions reject zero counts with `all_equal=true`. |
| 3D exact coverage | **WRONG** | Expected coverage still trusts result-side `case_sets` and `budget_cases`, rather than deriving both from configuration. |
| `analyzer_sha256` additions | **CORRECT** | Both analyzers emit their own source hash; committed smoke analyses match current sources. Wrong analyzer hashes are rejected. |
| Offline result binding | **WRONG** | Validates the analysis’s referenced result file, then compares only its `job_id` with the supplied smoke. Different smoke contents with the same job ID pass. |
| Offline B0/G-score validation | **WRONG** | Rejects explicit false gate flags, but accepts missing G-score controls, zero reproduction comparisons and a truthy `"not run"` B0 status. |
| Regenerated certificates at HEAD | **CORRECT** | All three return `True` from `inventory.verify_certificate()` using HEAD config/source blobs and the requested allocation. |
| “Smoke gates fail closed” | **NEEDS-RESTATEMENT** | The previous simple failures are repaired; the broader claim remains unsupported. |

Additional mutations accepted by the current gates:

| In-memory mutation | 2D | 3D |
|---|---|---|
| Remove G-score control evidence, retain reproduction `passed=true` | Accept | Accept |
| Set reproduction `compared=0`, retain `passed=true` | Accept | Accept |
| Set B0 `passed="not run"` | Accept | Accept |
| Supply different smoke contents with unchanged job ID to `offline_budget_gates()` | Accept | Accept |
| Downgrade selected-arm coverage from `all` to `first 2`, delete case 2 | — | Accept |
| Reduce recorded budget population and matching evidence from two cases to one, leaving config unchanged | — | Accept |

The last two tests exercise `budget_smoke_gates()` itself. Specifically, the selected `lat32768|fs|CN|0.025|K2` trajectory can lose its required third case while passing.

Certificate verification results:

| Job | Projected hours, including ×1.5 | Peak GB | Allocation | Decision |
|---|---:|---:|---|---|
| c2budA | 46.181540 | 40.820703 | A100-80G, 48 h, 128G | **HOLD** |
| c2budB | 46.181540 | 40.820703 | A100-80G, 48 h, 128G | **HOLD** |
| c2budC | 4.587112 | 40.820703 | A100-80G, 48 h, 128G | **HOLD** |

Release requires configuration-derived 3D coverage, binding offline evidence to the exact supplied smoke, and validating required offline evidence/counts with strict Boolean success checks. Regenerate certificates afterward.

These findings concern gate enforcement; they do not establish that the actual smoke computations failed. No files were created, modified or appended, and no jobs were submitted.