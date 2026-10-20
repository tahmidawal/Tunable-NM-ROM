**HOLD: c2budA, c2budB, c2budC.** Audited HEAD `a87c14df41293796d8552e30989d3a19b1095991`. Configuration-derived coverage is repaired, but G-score evidence validation still accepts failed controls—even through certificate verification.

| Item | Verdict | Finding |
|---|---|---|
| `c3run.py` `bud_case_plan` | **CORRECT** | Matches the configured bud loop: role mapping, selected-arm full coverage, budget subsets, finer-rule copies of convergence runs, and control baseline. Executing the actual loop’s planning statements in memory reproduced the helper’s output. |
| `runs/smk3bud` reproduction | **CORRECT** | Exact match: **37 rule/spec pairs, 77 rows**, and every recorded `case_sets` entry. |
| 3D configuration-derived coverage | **CORRECT** | Both round-3 coordinated coverage reductions now fail. Expected populations no longer come from result-side declarations. |
| Required rows, multiplicity, B1 counts | **CORRECT** | Both dimensions reject tested row deletions, duplicates, missing setting evidence, zero B1 counts and failed ordinary rows. |
| Strict B0/reproduction checks | **CORRECT** | Rejects `B0.passed="not run"` and zero reproduction comparisons. Untouched counts are 3,888 for 2D and 462 for 3D. |
| Offline evidence binding and callers | **NEEDS-RESTATEMENT** | Analysis hashes now bind to `<path>/result.json`. `main` and `verify_certificate` load that file and pass its path correctly. However, the helper still does **not** establish that supplied `sm` equals those bytes; the round-3 different-smoke/same-job mutation remains accepted. |
| G-score evidence validation | **WRONG** | 2D checks control-key presence but not their values. 3D checks only `all_break is True`. Failed or incomplete evidence can pass. |
| HEAD certificates | **CORRECT** | All three unmodified certificates return `True` against HEAD source/config blobs and their requested allocations. |
| “Smoke gates fail closed” | **NEEDS-RESTATEMENT** | Still broader than demonstrated enforcement. |

The six round-3 mutations now behave as follows:

| In-memory mutation | 2D | 3D |
|---|---|---|
| Remove G-score control evidence entirely | Reject | Reject |
| Set reproduction `compared=0` | Reject | Reject |
| Set B0 `passed="not run"` | Reject | Reject |
| Supply different smoke contents with unchanged job ID | **Accept** | **Accept** |
| Downgrade selected coverage and delete its third case | — | Reject |
| Reduce recorded budget population and matching evidence, leaving config unchanged | — | Reject |

**Remaining release blocker:** [inventory.py:360](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/inventory.py:360) accepts `case_swap=0` with `all_break=true` in both dimensions. In 3D, deleting both numerical control fields while retaining `all_break=true` also passes. With the 2D zero-control mutation applied only in memory, **all three `verify_certificate()` calls still returned `True`**. Require every dimension-specific control value to be numeric, finite and greater than the analyzers’ `1e-6` threshold.

The supplied-smoke limitation is narrower: changing a 2D row’s `e_ST` to `987.654` passes both offline and full smoke gates when the unchanged result path is supplied. Current callers load the correct file, so this is **not evidence of a mismatch in the actual certificates**. Either validate object/file equality or explicitly make that caller precondition part of the helper’s contract.

| Job | Projected hours, including ×1.5 | Peak GB | HEAD verification | Decision |
|---|---:|---:|---|---|
| c2budA | 46.181540 | 40.820703 | Pass | **HOLD** |
| c2budB | 46.181540 | 40.820703 | Pass | **HOLD** |
| c2budC | 4.587112 | 40.820703 | Pass | **HOLD** |

The `c3run.py` change adds only the pure helper; it changes no solver execution. It is absent from the 2D staging manifest, so it does not invalidate these regenerated 2D certificates. It **does** change the source hash required by older 3D certificates, which cannot authorize fresh staging at HEAD. Existing staged copies for `c3descB`, `c3descW1024`, `c3descW768` and `c3extW1024` still match their recorded provenance and contain no new helper; their execution is unaffected. Live queue status was not checked.

No files were created, modified or appended; no jobs were submitted.