**NOT READY.** Audited `e074dc49f`. Both smoke outputs were freshly scored on CPU and passed K4. No repository files, LAB-LOG, or main were written; no jobs submitted. Production scores were not used.

A20 fixes both exact round-10 reproductions. Two blocking paths remain.

1. **Invalid batch timing still counts as complete coverage.**

   [analyze_pm.py:326](/tmp/speed3d-round11-audit/lane/analyze_pm.py:326) checks batch presence, but not validity.

   In a production-shaped smoke fixture, changing a batch’s recorded hash outcome to `hashes_ok=False` gives:

   ```text
   usable=False
   bar_2x=None
   batch_all_present=True
   coverage_complete=True
   ```

   The baseline’s throughput bar was `True`. EXPORT preserves the incorrect complete-coverage claim. The report also prints the batch’s throughput without exposing its failed hashes or usability.

   This models an honestly recorded nondeterministic-output failure, **not a saved hash that lies**. The throughput bar correctly becomes unavailable; coverage and publication must reflect why.

2. **Failures before publication cleanup still leave accepted artifacts behind.**

   Two reproduced paths:

   - [make_report.py:210](/tmp/speed3d-round11-audit/lane/make_report.py:210) reads JSON inputs **before** deleting previous artifacts at line 217. An interrupted `analysis-pm.json` write causes `JSONDecodeError`; the previous report and EXPORT remain byte-identical.
   - [analyze_pm.py:568](/tmp/speed3d-round11-audit/lane/analyze_pm.py:568) encounters a missing score file and exits before replacing its output. An existing accepted `analysis-pm.json` survives, and a subsequent report invocation successfully republishes its old acceptance.

   These are honest missing/incomplete-input paths. Both scoring and analysis use direct `write_text`, so partial JSON needs no deliberate corruption. A20’s “failed run never leaves an earlier publication” guarantee is therefore still false.

   Invalidate stale outputs before fallible reads, or publish an explicit unavailable result with input provenance that prevents an older analysis from being reused as current evidence.

[Reproduction log](/tmp/speed3d-round11-audit/new_round11_probe-round11.log).

Every round-1 item was reassessed:

| Item | Verdict | Round-11 assessment |
|---|---|---|
| H1.4/H2 ratio direction | **CORRECT** | Directions, thresholds and non-null identity shortcuts agree. |
| Confirmation identity/completeness | **CORRECT** | Wrong A and missing invocations rejected; real confirmation contracts pass. |
| Raw timing recomputation | **CORRECT** | Stale-ratio fixture returns **3.63636**, rejecting drift **10**. |
| Case alignment and η | **CORRECT locally** | Incomplete score populations cannot pass b/c; η checked. Whole-file failure is finding 2. |
| A10 coverage/acceptance | **WRONG** | Invalid batches retain complete coverage; failed input processing can preserve accepted artifacts. |
| H3.1 profile/G3 | **CORRECT** | Invalid panels suppress G3; missing-profile publication now succeeds. |
| H3.2 baseline/qualification | **CORRECT** | Correct baseline; missing rows/scores yield unavailable; component suppression holds. |
| K5 | **CORRECT** | Every expected control participates; missing evidence stays unresolved. |
| H3.3 batching/G4/K6/K6+/2× | **WRONG overall** | Control gates and partial/all-OOM handling pass; invalid-batch coverage/publication remains defective. |
| A1.13 δ_quad | **CORRECT** | Complete, valid, finite evidence required; supplementary verdict suppressed when appropriate. |
| A2.7(d) reduced ADT | **CORRECT** | Population/validity checks and finalist gating hold. |
| H4.1 | **CORRECT** | Invalid panels excluded; incomplete meshes and failed integrity suppress verdicts. |
| H4.2 | **CORRECT** | Valid same-mesh subjects only. |
| H4.3/A6 | **CORRECT** | Mesh-specific qualification and exhaustion checks hold. |
| H-comp family/union cells | **CORRECT** | Unmatched cells excluded; unresolved mesh-specific evidence now makes coverage incomplete. |
| Governing-plan K2 | **CORRECT** | Three-state string and boolean preserved; unresolved families prevent unsupported ≥2× conclusions. |
| Freeze/r1 contents | **CORRECT under A12** | All **44 H1.4 labels**, **52 ROM configurations** included; final-rename failure leaves no freeze; retry succeeds. |
| 923901 guard | **CORRECT under A12** | Full r1 accepted; wrong digest/mode and unfrozen members rejected. Documented subsets permitted. |
| `resources.project_rep` | **CORRECT for stated proxies** | Independent expansion agrees at **4.652881254921977 h**, including warm-up burns. |
| Hand-typed report numbers | **CORRECT for cited defects** | Previously identified result-dependent literals remain data-derived. |
| EXPORT completeness | **WRONG** | Preserves false batch coverage; input failures can retain or republish stale acceptance. |

All A-subject contracts agree with their writers: **31 confirmation + 4 profile** real p1 panels and **20 real m1 panels** passed. Freeze publication remains safe ordering, **not a two-file atomic transaction**.

Saved-hash trust and audited matcher-input trust remain **NOTED under A5**, not blockers.

Additional evidence: [fresh fixture replay](/tmp/speed3d-round11-audit/replay-round11.log), [round-10 regressions](/tmp/speed3d-round11-audit/new_round10_probe-round11.log), [freeze/resources](/tmp/speed3d-round11-audit/freeze_probe-round11.log), [guard checks](/tmp/speed3d-round11-audit/guard_extra-round11.log).