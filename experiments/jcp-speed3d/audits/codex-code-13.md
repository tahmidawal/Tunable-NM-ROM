**NOT READY.** Audited commit `ae31606e4`.

Both smoke outputs were freshly rescored on CPU; both passed K4. Fixtures and generated reports stayed under `/tmp/speed3d-round7-audit`. This audit changed no repository files, wrote nothing to LAB-LOG/main, and submitted no jobs. Production p1 was checked for reader/writer contracts only; its scores were absent when checked.

A16 fixes both round-6 reproductions:

- Partial spectral coverage now prints **“neither confirmed nor retracted (1 of 2 measured)”**.
- Successful B=1/4 plus OOM-infeasible B=16/64 now exports `K6_fires_all=null`, `K6plus_all=null`, and an unavailable throughput bar.

[Regression evidence](/tmp/speed3d-round7-audit/new-probes.log).

Two blocking classes remain.

1. **Unavailable verdicts still become definitive false booleans in analysis and EXPORT.**

   At [analyze_pm.py:507](experiments/jcp-speed3d/analyze_pm.py:507), an unresolved family produces:

   ```json
   {"K2": "undetermined", "K2_fires": false}
   ```

   The string verdict and union calculation are correct, but the exported boolean collapses unknown into false. Failed integrity also explicitly sets this boolean false at line 527.

   Similarly, [analyze_pm.py:223](experiments/jcp-speed3d/analyze_pm.py:223) emits this for an absent mesh-free block:

   ```json
   {"qualified": false, "status": "unavailable", "reason": "not run"}
   ```

   Both contradictory representations survive into `EXPORT.json`. Use null for unavailable verdict booleans, or explicitly rename and document them as predicates meaning “established true.” The current three-state contract does not make that distinction.

2. **Coverage can be declared complete despite unavailable required evidence.**

   Two independently reproduced paths:

   - **p1:** changing only a profile panel’s raw drift to 2 makes `panel_valid=false` and `G3=null`, but leaves `coverage_complete=true`. [Line 323](experiments/jcp-speed3d/analyze_pm.py:323) checks profile presence, not validity.
   - **m1:** removing ROM score entries leaves `scores_ok=false`, no median/worst, and K2 undetermined, yet `coverage_complete=true`. [Lines 512–521](experiments/jcp-speed3d/analyze_pm.py:512) check feasibility and timing coverage without checking score coverage.

   The combined fixture exports both coverage flags as true. The generated report consequently omits its **“some items unavailable”** status qualification. Local G3/K2 decisions remain unavailable; the false claim is completeness.

These involve missing evidence and invalid measured timing, not fabricated hashes. [Reproductions](/tmp/speed3d-round7-audit/round7-probe.log), [coverage report](/tmp/speed3d-round7-audit/r7-coverage-report.md), [coverage export](/tmp/speed3d-round7-audit/r7-coverage-EXPORT.json).

Every round-1 item was reassessed:

| Item | Verdict | Round-7 evidence |
|---|---|---|
| H1.4/H2 ratio direction | **CORRECT** | Selected/comparator ratio, reciprocal H2 speedup, thresholds agree. |
| Confirmation identity/completeness | **CORRECT** | Wrong A and missing invocations rejected. All 31 real p1 confirmation panels satisfy the reader contract. |
| Raw timing recomputation | **CORRECT** | Stale-ratio fixture returns **3.63636**, rejecting drift **10**. Saved-hash trust remains **NOTED under A5**. |
| Case alignment and η | **CORRECT** | Missing scores handled; incomplete populations cannot pass b/c; η consistency checked. Coverage defect is separate. |
| A10 coverage/acceptance | **WRONG** | Invalid profiles and missing m1 scores can retain complete-coverage claims. |
| H3.1 profile/G3 | **CORRECT locally** | Invalid timing suppresses G3. All four real p1 profile contracts agree. |
| H3.2 baseline/qualification | **WRONG overall** | Baseline and incomplete-row/score handling work; an entirely absent block still exports `qualified=false`. |
| K5 | **CORRECT** | Missing/invalid expected controls remain unresolved; report prints unavailable. |
| H3.3 batching/G4/K6/K6+/2× | **CORRECT** | Failed K6+ blocks throughput acceptance; partial/all-OOM aggregates now remain unavailable. |
| A1.13 δ_quad | **CORRECT** | Complete, valid converged-rule rows and finite distances required. |
| A2.7(d) reduced ADT | **CORRECT** | Empty validity rejected; population/finite-distance checks gate finalist H2. |
| H4.1 | **CORRECT** | Invalid panels excluded; missing meshes unavailable; integrity failure suppresses verdict. |
| H4.2 | **CORRECT** | Valid same-mesh subjects only. |
| H4.3/A6 | **CORRECT** | Panel identities and mesh-specific eligibility checked; spectral publication regression repaired. Matcher trust remains **NOTED under A5**. |
| H-comp family/union cells | **CORRECT** | Unmatched cells excluded; unresolved families prevent definitive ≥2× conclusions. |
| Governing-plan K2 | **WRONG overall** | Three-state string works; exported `K2_fires` still collapses undetermined to false. |
| Freeze/r1 contents | **CORRECT under A12** | All **44 H1.4 labels**, **52 ROM configurations** included. Final-rename failure leaves no freeze; retry succeeds. Safe ordering, not a two-file transaction. |
| 923901 guard | **CORRECT for restated scope** | Full r1 accepted; wrong digest/mode and unfrozen ROM/FOM members rejected. Documented subsets remain permitted. |
| `resources.project_rep` | **CORRECT for stated proxies** | Independent expansion agrees at **4.652881254921977 h**, including warm-up burns. Arithmetic verification, not runtime validation. |
| Hand-typed report numbers | **CORRECT for cited defects** | Previously identified result-dependent literals remain data-derived. |
| EXPORT completeness | **WRONG overall** | Retains contradictory unavailable/false verdicts and false coverage claims. |

All A-subject contracts agree: `profile|` → configuration key; `confirm|a|b` → `a`; `mesh|L…|label` → `label|n…`; `match|label` → `label|meshpath|n65`.

Additional evidence: [round-1 replay](/tmp/speed3d-round7-audit/probes.log), [real p1 contracts](/tmp/speed3d-round7-audit/real-contract.log), [union/K2](/tmp/speed3d-round7-audit/k2-probe.log), [freeze/resources](/tmp/speed3d-round7-audit/freeze-probe.log), [constructor guard](/tmp/speed3d-round7-audit/guard-probe.log).