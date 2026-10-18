**NOT READY.** Audited `8ccc88ea0`; scratch source copies were verified against that commit. Both smoke outputs were freshly scored on CPU and passed K4. This audit made no repository/LAB-LOG/main writes and submitted no jobs.

The three exact round-8 reproductions are repaired: missing scores no longer force false eligibility/measurement outcomes, null winner/B* labels no longer imply confirmed 1×, and invalid profiles no longer retain or embed stale plots.

Three blocking paths remain.

1. **A primary-mesh “no ladder match” is incorrectly transferred to other meshes.**

   [analyze_pm.py:481](/tmp/speed3d-round9-audit/lane/analyze_pm.py:481) copies that family state without reconsidering the mesh-specific ROM errors. `union_of()` then treats the family as resolved.

   Reproduction:
   - Primary ROM error: **0.0100**; UC error: **0.0101**, η **0.000001**—correctly no match.
   - Larger-mesh ROM error: **0.0102**—UC now matches.
   - Valid same-job timings give UB **3×**, UC **0.5×**.

   Nevertheless, analysis exports **union=3×**, `reaches_2x=true`, `coverage_complete=true`, and **K2 “does not fire.”** The report prints the same conclusion, while H4.2 already contains UC’s 0.5× timing.

   Recheck exhaustion against each mesh’s scores, or mark that family unresolved there. This does not require a dishonest matcher file: its primary-mesh conclusion is correct. [Reproduction and publication evidence](/tmp/speed3d-round9-audit/new-round9-probe.log).

2. **Missing or OOM-infeasible mesh ROMs still receive definitive negative eligibility.**

   [analyze_pm.py:407](/tmp/speed3d-round9-audit/lane/analyze_pm.py:407) emits `eligible=false` for both an absent ROM block and a recorded allocation failure. No solver eligibility measurement is required.

   Both fixtures survive into EXPORT; the report prints **“eligible: no.”** A18 defines eligibility through solver evidence, so unavailable evidence should produce `eligible=null`; resource infeasibility can remain separately recorded. [Evidence](/tmp/speed3d-round9-audit/new-round9-probe.log).

3. **Integrity suppression still misses component verdicts.**

   [analyze_pm.py:365](/tmp/speed3d-round9-audit/lane/analyze_pm.py:365) suppresses `b` but misses `b_median`, `b_worst`, and `converged_rule_Eb`.

   The **unmodified, freshly scored smoke** produces and exports:

   ```json
   {
     "integrity_ok": false,
     "b": null,
     "b_median": true,
     "b_worst": true,
     "qualified": null
   }
   ```

   K5’s nested comparison verdicts and `converged_rule_Eb=true` also survive. These are verdict booleans, beyond the diagnostic numbers A12 permits retaining. Suppress them consistently or explicitly separate ungated diagnostic predicates from accepted verdicts. [Reproduction](/tmp/speed3d-round9-audit/suppression-probe.log), [EXPORT](/tmp/speed3d-round9-audit/suppression-EXPORT.json).

Every round-1 item was reassessed:

| Item | Verdict | Round-9 assessment |
|---|---|---|
| H1.4/H2 ratio direction | **CORRECT** | Directions, thresholds and non-null identity shortcuts agree. |
| Confirmation identity/completeness | **CORRECT** | Wrong A and missing invocations rejected; all 31 real p1 confirmation contracts pass. |
| Raw timing recomputation | **CORRECT** | Stale-ratio fixture gives **3.63636**, rejecting drift **10**. |
| Case alignment and η | **CORRECT** | Incomplete populations cannot pass b/c; η consistency checked. |
| A10 coverage/acceptance | **WRONG** | Findings 1–3 leave unsupported outcomes despite the repaired coverage checks. |
| H3.1 profile/G3 | **CORRECT** | Validity gating works; stale plot removed and no longer embedded. |
| H3.2 baseline/qualification | **WRONG overall** | Baseline and missing-score fixes pass; component verdicts escape suppression. |
| K5 | **CORRECT locally** | Three-state aggregate and qualification blocking work; nested comparison suppression remains finding 3. |
| H3.3 batching/G4/K6/K6+/2× | **CORRECT** | Failed controls block claims; partial/all-OOM aggregates remain unavailable. |
| A1.13 δ_quad | **WRONG overall** | Measurement checks work; supplementary `converged_rule_Eb` escapes integrity suppression. |
| A2.7(d) reduced ADT | **CORRECT** | Population/validity checks and finalist gating remain enforced. |
| H4.1 | **CORRECT** | Invalid panels excluded; missing meshes and failed integrity suppress verdicts. |
| H4.2 | **CORRECT** | Valid same-mesh subjects only. |
| H4.3/A6 | **WRONG overall** | Primary matching works; mesh-specific no-match state transfer fails. |
| H-comp family/union cells | **WRONG** | Minimum excludes unmatched cells, but inherited exhaustion can falsely resolve the union. |
| Governing-plan K2 | **WRONG overall** | Three-state representation works; finding 1 supplies an incorrect resolved premise. |
| Freeze/r1 contents | **CORRECT under A12** | All **44 H1.4 labels**, **52 ROM configurations** included; final-rename failure leaves no freeze and retry succeeds. |
| 923901 guard | **CORRECT under A12** | Full r1 accepted; wrong digest/mode and unfrozen members rejected. Documented subsets remain permitted. |
| `resources.project_rep` | **CORRECT for stated proxies** | Independent expansion agrees at **4.652881254921977 h**, including warm-up burns. |
| Hand-typed report numbers | **CORRECT for cited defects** | Previously identified result-dependent literals remain data-derived. |
| EXPORT completeness | **WRONG** | Preserves the unsupported outcomes demonstrated above. |

All A-subject naming contracts agree with their writers. The real p1 panels and all **20** real m1 panels passed the contract checks; production scores were not used.

Saved-hash trust and audited matcher-input trust remain **NOTED under A5**, not blocking. Freeze publication is safe ordering, not a two-file atomic transaction.

Additional evidence: [round-1 replay](/tmp/speed3d-round9-audit/probes-fresh.log), [round-8 regressions](/tmp/speed3d-round9-audit/round8_probe-fresh.log), [stale-plot replay](/tmp/speed3d-round9-audit/stale_plot_probe-fresh.log), [freeze/resources](/tmp/speed3d-round9-audit/freeze-probe.log), [guard checks](/tmp/speed3d-round9-audit/guard-probe.log).