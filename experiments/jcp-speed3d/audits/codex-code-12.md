**NOT READY.** Audited commit `571af18a6`. Two blocking missing-evidence paths remain.

Both smoke outputs were freshly rescored with `score3.py` on CPU; both passed K4. All fixtures and generated outputs stayed under `/tmp/speed3d-round6-audit`. No repository files, LAB-LOG, or main were written; no jobs submitted.

1. **Partial spectral coverage still produces a false confirmation.**  
   [make_report.py:486](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/make_report.py:486) drops missing spectral ratios before evaluating `max(ps_sp) < 1`.

   With job integrity true, one selected ROM’s spectral ratio **0.5×**, and another selected ROM’s panel missing, the report says **“spectral FOM clearly beats the ROM at 64³” is confirmed**. Supplying the missing panel at **2×** changes that to **not confirmed**.

   A15 correctly suppresses the statement when integrity fails, but integrity does not establish spectral coverage. Require complete relevant spectral evidence for the collective confirmation, or explicitly restrict the statement to named measured configurations. [Reproduction](/tmp/speed3d-round6-audit/spectral-probe.log).

2. **Partial OOM batches yield false “all controls passed” aggregates.**  
   [analyze_pm.py:294](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/analyze_pm.py:294) aggregates only available batches.

   A fixture with successful B=1/4 and OOM-infeasible B=16/64 returns:
   - `K6_fires_all=True`
   - `K6plus_all=True`
   - `integrity_ok=True`, `coverage_complete=False`

   Both aggregates survive into **EXPORT.json**. The throughput bar correctly remains unavailable. Aggregate controls should consume every expected applicable batch: a measured failure establishes false; otherwise missing evidence prevents true. Alternatively, explicitly rename and scope these fields to available batches. [Reproduction and exported values](/tmp/speed3d-round6-audit/new-probes.log).

Both follow honest runner paths—caught panel failures and recorded OOM infeasibility—not fabricated hashes or dishonest files.

The round-1 items, reassessed against this commit:

| Item | Verdict | Evidence |
|---|---|---|
| H1.4/H2 ratio direction | **CORRECT** | Selected/comparator ratio, reciprocal H2 speedup, and thresholds agree. |
| Confirmation identity/completeness | **CORRECT** | Wrong A and missing invocations rejected; exact case×repetition population checked. |
| Raw timing recomputation | **CORRECT** | Stale-ratio fixture returns **3.63636**, rejects drift **10**. Saved-hash trust remains **NOTED under A5**. |
| Case alignment and η | **CORRECT** | Missing scores handled; incomplete populations cannot pass b/c; η consistency checked. |
| A10 coverage/acceptance | **WRONG overall** | Coverage gaps are recorded, but the two claims above remain definitive. |
| H3.1 profile/G3 | **CORRECT** | Genuine smoke subjects agree; invalid panels suppress G3; arithmetic correct. |
| H3.2 baseline/qualification | **CORRECT** | Correct baseline and ADT initialization; missing rows/scores produce unavailable status. |
| K5 | **CORRECT** | A15 includes every expected control. Missing/invalid control plus nonfiring measured control now yields unavailable. |
| H3.3 batching/G4/K6/K6+/2× | **WRONG overall** | Local gates and all-OOM handling work; partial-OOM aggregate claims fail as above. |
| A1.13 δ_quad | **CORRECT** | Complete, valid converged-rule rows and finite distances required. |
| A2.7(d) reduced ADT | **CORRECT** | Population/validity checks reject empty evidence; gate affects finalist H2. |
| H4.1 | **CORRECT** | Invalid panels excluded; missing meshes unavailable; failed integrity suppresses verdict. |
| H4.2 | **CORRECT** | Valid same-mesh subjects only. |
| H4.3/A6 | **CORRECT locally** | Panel identities and mesh-specific eligibility checked. Spectral publication defect is separate. Matcher trust remains **NOTED under A5**. |
| H-comp family/union cells | **CORRECT** | Unmatched cells excluded; unresolved families prevent definitive ≥2× union claims. |
| Governing-plan K2 | **CORRECT** | Missing/untimed/unmatched-only fixtures remain undetermined; matched **0.5×** evidence makes K2 fire. |
| Freeze/r1 contents | **CORRECT under A12** | All **44 H1.4 labels** included; **52 ROM configurations**. Final-rename failure leaves r1 without freeze; retry succeeds. Safe ordering, not a two-file transaction. |
| 923901 guard | **CORRECT for restated scope** | Full r1 accepted; wrong digest/mode and unfrozen ROM/FOM members rejected. Documented subsets remain permitted. |
| `resources.project_rep` | **CORRECT for stated proxies** | Independent expansion agrees at **4.652881254921977 h**, including warm-up burns. Arithmetic verification, not runtime validation. |
| Hand-typed report numbers | **CORRECT for cited defects** | Previously identified result-dependent literals remain data-derived. |
| EXPORT completeness | **WRONG overall** | Timing/validity context retained, but partial-batch control aggregates falsely pass. |

All A-subject contracts agree with their writers: `profile|` → `setting|rule|spec`; `confirm|a|b` → `a`; `mesh|L…|label` → `label|n…`; `match|label` → `label|meshpath|n65`.

Additional evidence: [round-1 fixture replay](/tmp/speed3d-round6-audit/probes.log), [A15 K5 replay](/tmp/speed3d-round6-audit/partial-probe.log), [union/K2](/tmp/speed3d-round6-audit/k2-probe.log), [freeze/resource arithmetic](/tmp/speed3d-round6-audit/freeze-probe.log), [constructor guard](/tmp/speed3d-round6-audit/guard-probe.log).