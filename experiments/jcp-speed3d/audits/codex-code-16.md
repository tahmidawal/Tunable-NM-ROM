**NOT READY.** Audited `08f42a919`. Both smoke outputs were freshly scored on CPU and passed K4. No repository files, LAB-LOG, or main were written; no jobs submitted. Production scores were not used.

The three exact round-9 reproductions are repaired. Two blocking publication/coverage defects remain.

1. **Unresolved mesh-specific matching still permits `coverage_complete=True`.**

   A19 correctly discovers a ladder member that matches at another mesh and marks it **“ladder member matches at this mesh (untimed)”**. But [analyze_pm.py:541](/tmp/speed3d-round10-audit/lane/analyze_pm.py:541) checks only **primary** union resolution when declaring coverage complete.

   Reproduction with valid primary matching, complete ROM scores and otherwise valid panels:

   - UC does not match primary ROM error **0.0100**.
   - UC error **0.0101** matches larger-mesh ROM error **0.0102**.
   - Its required larger-mesh timing is absent.
   - K2 correctly becomes **undetermined**, but `coverage_complete=True` survives into EXPORT.
   - The report omits its **“some items unavailable”** qualification.

   Include unresolved mesh-specific evidence in coverage. This is a false completeness claim, although the repaired K2 calculation itself remains conservative.

2. **A19 suppression breaks missing-profile publication and leaves stale artifacts.**

   [analyze_pm.py:367](/tmp/speed3d-round10-audit/lane/analyze_pm.py:367) changes an absent profile’s `{"missing": true}` to `{"missing": null}` when integrity fails. The report then bypasses its missing-profile branch and accesses `fused_seconds` at [make_report.py:366](/tmp/speed3d-round10-audit/lane/make_report.py:366).

   An interrupted-p1 fixture produces:

   ```text
   integrity_ok=False
   H31.C_old={"missing": null}
   KeyError('fused_seconds')
   ```

   After an earlier successful publication, both `report.md` and `EXPORT.json` remain byte-identical to their previous versions, retaining the previous accepted status. The runner saves intermediate results before profiling, so this is an honest interrupted-job path.

   Make rendering tolerate suppressed/missing records and publish unavailable status without retaining stale accepted artifacts.

Both reproductions and publication evidence are in [new-round10-probe.log](/tmp/speed3d-round10-audit/new-round10-probe.log).

Every round-1 item was reassessed:

| Item | Verdict | Round-10 assessment |
|---|---|---|
| H1.4/H2 ratio direction | **CORRECT** | Directions, thresholds and non-null identity shortcuts agree. |
| Confirmation identity/completeness | **CORRECT** | Wrong A and missing invocations rejected; 31 real p1 confirmation panels pass. |
| Raw timing recomputation | **CORRECT** | Stale-ratio fixture returns **3.63636**, rejects drift **10**. |
| Case alignment and η | **CORRECT** | Missing scores handled; incomplete populations cannot pass b/c; η checked. |
| A10 coverage/acceptance | **WRONG** | False complete-coverage claim and failed unavailable-status publication above. |
| H3.1 profile/G3 | **WRONG overall** | Calculations and invalid-panel gating pass; missing-profile rendering regresses. |
| H3.2 baseline/qualification | **CORRECT** | Baseline, incomplete evidence and component suppression checks pass. |
| K5 | **CORRECT** | Expected controls participate; missing evidence stays unresolved; nested booleans suppressed. |
| H3.3 batching/G4/K6/K6+/2× | **CORRECT** | Failed controls block throughput claims; partial/all-OOM aggregates remain unavailable. |
| A1.13 δ_quad | **CORRECT** | Valid complete evidence required; supplementary boolean now suppressed. |
| A2.7(d) reduced ADT | **CORRECT** | Population/validity checks and finalist gating remain enforced. |
| H4.1 | **CORRECT** | Invalid panels excluded; incomplete meshes and failed integrity suppress verdicts. |
| H4.2 | **CORRECT** | Valid same-mesh subjects only. |
| H4.3/A6 | **CORRECT locally** | A19 rechecks exhaustion against each mesh; missing/infeasible ROM eligibility is null. |
| H-comp family/union cells | **CORRECT locally** | Unmatched cells excluded; newly discovered untimed matches prevent definitive ≥2×. Coverage remains defective separately. |
| Governing-plan K2 | **CORRECT** | String and boolean preserve three states; round-9 fixture now gives undetermined/null. |
| Freeze/r1 contents | **CORRECT under A12** | All **44 H1.4 labels**, **52 ROM configurations** included; final-rename failure leaves no freeze; retry succeeds. |
| 923901 guard | **CORRECT under A12** | Full r1 accepted; wrong digest/mode and unfrozen members rejected. Documented subsets permitted. |
| `resources.project_rep` | **CORRECT for stated proxies** | Independent expansion agrees at **4.652881254921977 h**, including warm-up burns. |
| Hand-typed report numbers | **CORRECT for cited defects** | Previously identified result-dependent literals remain data-derived. |
| EXPORT completeness | **WRONG** | Exports false coverage; failed missing-profile publication leaves the previous EXPORT intact. |

All A-subject contracts agree with their writers. The **31 confirmation + 4 profile** real p1 panels and **20 real m1 panels** pass the reader checks.

Saved-hash trust and audited matcher-input trust remain **NOTED under A5**, not blockers. Freeze publication is safe ordering, not a two-file atomic transaction.

Additional evidence: [fixture replay](/tmp/speed3d-round10-audit/replay-status.log), [A19 regressions](/tmp/speed3d-round10-audit/new-round9-probe.log), [suppression](/tmp/speed3d-round10-audit/suppression-probe.log), [freeze/resources](/tmp/speed3d-round10-audit/freeze-probe.log), [guard checks](/tmp/speed3d-round10-audit/guard-probe.log).