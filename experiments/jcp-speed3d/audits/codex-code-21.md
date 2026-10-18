Audited `b9da77852`. Both smoke outputs were rescored on CPU and passed K4. Reconstructed regressions passed; production reader/writer checks covered 35 p1 panels and 20 m1 panels.

| Item | Verdict | Assessment |
|---|---|---|
| H1.4/H2 ratio direction | CORRECT | Directions, thresholds and identity shortcuts hold. |
| Confirmation identity/completeness | CORRECT | Wrong subjects and incomplete populations rejected. |
| Raw timing recomputation | CORRECT | Stale-ratio fixture recomputes 3.63636 and rejects drift 10. |
| Case alignment and η | CORRECT | Incomplete scores suppress comparisons and statistics. |
| A10/A12 coverage and acceptance | CORRECT | Missing evidence affects coverage; failed integrity suppresses verdicts. |
| H3.1 profile/G3 | CORRECT | Missing or invalid profiles cannot establish G3. |
| H3.2 baseline/qualification | CORRECT | Baselines and missing-evidence handling agree with amendments. |
| K5 | CORRECT | Expected controls participate; unresolved outcomes remain nullable. |
| H3.3 batching/G4/K6/K6+/2× | CORRECT | Failed controls and OOM records block unsupported conclusions. |
| A1.13 δ_quad | CORRECT | Requires complete, valid converged-rule evidence. |
| A2.7(d) reduced ADT | CORRECT | Missing evidence blocks dependent finalist verdicts. |
| H4.1 | CORRECT | Invalid panels and incomplete meshes cannot establish flatness. |
| H4.2 | CORRECT | Same-mesh filtering and diagnostic qualifications hold. |
| H4.3/A6 | CORRECT | Mesh-specific eligibility and matching checks remain enforced. |
| H-comp unions | CORRECT | Unresolved families prevent unsupported ≥2× conclusions. |
| Governing-plan K2 | CORRECT | Three-state outcomes preserved. |
| Freeze/r1 contents | CORRECT | Includes all H1.4 labels among 52 configurations; rename-failure/retry passes. |
| 923901 guard | CORRECT | Digest/membership scope agrees with A12; subsets remain permitted. |
| `resources.project_rep` | CORRECT | Independent inventory agrees, including warm-up burns; estimate remains proxy-based. |
| Previously hand-typed report numbers | CORRECT | Cited result-dependent values remain data-derived. |
| EXPORT/publication | CORRECT | Coverage qualifications propagate; failed reads remove stale publications. |
| Replication analysis | CORRECT | Writer-shaped ADT, partial scores, missing ADT and incomplete panel/FOM regressions pass. |
| A23 failed-FOM statistics | CORRECT | Failed/unscored cases suppress median/worst and expose population counts. |
| A24 same-mesh FOM coverage | CORRECT | Calibration-failure fixture now makes coverage incomplete in analysis, report and EXPORT. |

**Blocking findings: none.** Saved-hash and audited matcher-input trust remain nonblocking under A5.

[Regression evidence](/tmp/speed3d-audit-scratch/regress.log) · [Replication evidence](/tmp/speed3d-audit-scratch/rep-regress.log)

No repository, LAB-LOG or main writes; no jobs submitted. Scratch usage was 3.6 MB; large inputs were read in place. An external `HEARTBEAT.json` change appeared during the audit and was left untouched.

**Overall: READY.**