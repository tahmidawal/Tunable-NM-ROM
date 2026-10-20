**NOT READY — one blocking coverage defect remains.** Audited `fa1f2e6b2`.

Both smoke outputs were freshly scored on CPU and passed K4. Reconstructed regressions passed for the earlier timing, missing-evidence, publication-cleanup, freeze and replication defects. All **35 p1 panels and 20 m1 panels** passed identity/population checks. No repository, LAB-LOG or main writes; no jobs submitted. Scratch usage: **3.6 MB**, with large inputs read in place.

| Item | Verdict | Assessment |
|---|---|---|
| H1.4/H2 ratio direction | CORRECT | Directions, thresholds and identity shortcuts hold. |
| Confirmation identity/completeness | CORRECT | Wrong A and missing invocations rejected. |
| Raw timing recomputation | CORRECT | Stale-ratio fixture returns 3.63636 and rejects drift 10. |
| Case alignment and η | CORRECT | Incomplete scores suppress comparisons and statistics. |
| A10/A12 coverage and acceptance | **WRONG** | Failed same-mesh FOM calibration can retain complete coverage. |
| H3.1 profile/G3 | CORRECT | Invalid/missing profiles and stale-plot regressions pass. |
| H3.2 baseline/qualification | CORRECT | Missing evidence remains unavailable. |
| K5 | CORRECT | Required controls participate; unresolved evidence remains nullable. |
| H3.3 batching/G4/K6/K6+/2× | CORRECT | Failed controls, invalid hashes and OOM cases block claims and coverage. |
| A1.13 δ_quad | CORRECT | Complete, valid converged-rule evidence required. |
| A2.7(d) reduced ADT | CORRECT | Validity checks and finalist gating hold. |
| H4.1 | CORRECT | Invalid panels excluded; incomplete meshes suppress verdicts. |
| H4.2 | CORRECT locally | Timing filtering and A23 statistics suppression hold; coverage defect below. |
| H4.3/A6 | CORRECT | Mesh-specific qualification and exhaustion checks remain enforced. |
| H-comp unions | CORRECT | Unmatched cells excluded; unresolved families prevent unsupported success. |
| Governing-plan K2 | CORRECT | Three-state outcomes preserved. |
| Freeze/r1 contents | CORRECT | 44 H1.4 labels included among 52 configurations; rename-failure/retry checks pass. |
| 923901 guard | CORRECT | Digest/membership checks pass; documented subsets remain permitted. |
| `resources.project_rep` | CORRECT | Independent expansion agrees, including warm-up burns; proxy estimate only. |
| Previously hand-typed report numbers | CORRECT | Cited result-dependent literals remain data-derived. |
| EXPORT completeness | **WRONG** | Exports the false complete-coverage flag. |
| Replication analysis | CORRECT | ADT schema, partial scores, missing ADT, incomplete panels/FOMs and cleanup regressions pass. |
| A23 failed-FOM statistics | CORRECT | Failed/unscored cases suppress median/worst; counts and convergence are exposed. |

**Blocking: failed same-mesh FOM calibration still permits `coverage_complete=True`.**

The honest writer catches calibration failures, records `infeasible`, and omits that FOM from timing subjects at [s3modes.py:453](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/s3modes.py:453). Analysis validates only the surviving subjects at [analyze_pm.py:428](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/analyze_pm.py:428). Its coverage predicate never checks the required same-mesh FOM diagnostics at [analyze_pm.py:538](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/analyze_pm.py:538).

Using real m1 data, I modelled a calibration allocation failure for **L=64, UB, Δt=0.01**, removing its timing subject exactly as the writer does. Results:

- FOM diagnostic: `infeasible`, with no configuration or timing.
- `every_mesh_panel_valid=True`.
- `coverage_complete=True`, including in EXPORT.
- Report status omits **“some items unavailable.”**

The diagnostic table retains the failure, but the aggregate completeness claim contradicts it. This is blocking under the stated A5 rule for honest cluster failures.

Coverage must account for every configured same-mesh FOM diagnostic and its required evidence. A missing diagnostic need not invalidate unrelated supported verdicts.

[Reproduction log](/tmp/speed3d-audit-scratch/fom-coverage.log) · [Generated report](/tmp/speed3d-audit-scratch/fom-coverage-report.md) · [Regression log](/tmp/speed3d-audit-scratch/regress.log)

**Overall: NOT READY.**