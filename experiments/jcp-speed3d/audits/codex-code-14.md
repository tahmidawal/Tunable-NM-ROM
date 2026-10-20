**NOT READY.** Audited commit `b416285713fb538e05093d7f4c2f855920227fa5`.

Both smoke outputs were freshly rescored on CPU; both passed K4. Fixtures and generated artifacts stayed under `/tmp/speed3d-round8-audit`. No repository, LAB-LOG, or main files were changed; no jobs submitted. Production p1 scores were absent and were not used.

Three blocking findings remain.

1. **Missing scores still become definitive false booleans.**

   - At [analyze_pm.py:414](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/analyze_pm.py:414), missing m1 scores produce `eligible=false`, even with all recorded E-a/E-b/E-c/ADT gates passing. EXPORT preserves this, and the report prints **“eligible: no.”**
   - At [analyze_pm.py:237](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/analyze_pm.py:237), missing mesh-free scores produce `measured_pass=false` alongside `qualified=null`, `b=null`, `c=null`, and unavailable status. EXPORT retains that contradictory negative.

   A17 repairs the specifically cited fields, but not the whole missing-evidence class. Use nullable outcomes consistently; keep evidence-presence predicates separate. [Fresh reproductions](/tmp/speed3d-round8-audit/round8-probe.log).

2. **No winner and no B* yield a confirmed 1× decomposition.**

   [analyze_pm.py:202](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/analyze_pm.py:202) treats `w == bs` as sufficient for identity. When both are `None`, analysis and EXPORT contain:

   ```json
   {
     "winner": null,
     "B_star": null,
     "verdict": "unavailable (no winner)",
     "S_scheme_confirmed": 1.0
   }
   ```

   This is an honest selection outcome: `analyze.select()` returns neither configuration when the comparator is unavailable. Require non-null configurations before applying the identity shortcut. [Reproduction](/tmp/speed3d-round8-audit/no-winner-probe.log).

3. **Invalid profiles can publish a stale graph.**

   [make_report.py:109](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/make_report.py:109) skips graph generation when no profile panel is valid, but [line 370](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/make_report.py:370) still embeds `plots/profile.png`.

   Using the actual plotting code: generate a valid report, then rerun with every profile’s raw drift set to 2. All four profiles become invalid, yet the previous graph remains byte-identical and embedded in the new report. Publish the image only when generated from the current accepted evidence. [Reproduction](/tmp/speed3d-round8-audit/stale-plot-probe.log).

These findings concern unavailable selections, missing scores, and invalid timing—not fabricated hashes—so they are blocking under A5.

The exact round-7 regressions are repaired: unavailable K2 and absent mesh-free qualification export null; invalid profiles and missing m1 scores make coverage false; the report includes “some items unavailable.” [Replay](/tmp/speed3d-round8-audit/round7-probe.log).

| Round-1 item | Verdict | Round-8 assessment |
|---|---|---|
| H1.4/H2 ratio direction | **CORRECT** | Directions and thresholds agree. The separate no-winner decomposition defect is above. |
| Confirmation identity/completeness | **CORRECT** | Wrong A and missing invocations rejected; all 31 real p1 confirmation contracts pass. |
| Raw timing recomputation | **CORRECT** | Stale-ratio fixture returns **3.63636**, rejects drift **10**. |
| Case alignment and η | **CORRECT** | Incomplete populations cannot pass b/c; η consistency checked. |
| A10 coverage/acceptance | **WRONG overall** | Round-7 coverage repairs pass, but unavailable outcomes still leak through publication. |
| H3.1 profile/G3 | **WRONG overall** | Calculation and validity gating work; stale graph survives publication. |
| H3.2 baseline/qualification | **WRONG overall** | Baseline and nullable qualification work; `measured_pass` still collapses unavailable to false. |
| K5 | **CORRECT** | Every expected control participates; missing evidence remains unresolved. |
| H3.3 batching/G4/K6/K6+/2× | **CORRECT** | Failed controls block throughput claims; partial/all-OOM aggregates remain unavailable. |
| A1.13 δ_quad | **CORRECT** | Complete, valid converged-rule rows and finite distances required. |
| A2.7(d) reduced ADT | **CORRECT** | Population/validity checks and finalist H2 gating remain enforced. |
| H4.1 | **CORRECT** | Invalid panels excluded; incomplete meshes unavailable; integrity failure suppresses verdicts. |
| H4.2 | **CORRECT** | Valid same-mesh subjects only. |
| H4.3/A6 | **CORRECT** | Matching requires mesh-specific evidence; spectral coverage regression passes. False eligibility publication is finding 1. |
| H-comp family/union cells | **CORRECT** | Unmatched cells excluded; unresolved families prevent definitive ≥2× conclusions. |
| Governing-plan K2 | **CORRECT** | String and boolean now preserve all three states. |
| Freeze/r1 contents | **CORRECT** | All **44 H1.4 labels**, **52 ROM configurations** included; failed final rename leaves no freeze; retry succeeds. Safe ordering under A12, not a two-file transaction. |
| 923901 guard | **CORRECT** | Full r1 accepted; wrong digest/mode and unfrozen members rejected. Subsets remain permitted under A12. |
| `resources.project_rep` | **CORRECT** | Independent proxy expansion agrees at **4.652881254921977 h**, including warm-up burns. |
| Hand-typed report numbers | **CORRECT** | Previously identified literals remain data-derived. |
| EXPORT completeness | **WRONG** | Preserves the false eligibility/measurement booleans and unsupported confirmed decomposition. |

All A-subject contracts agree: `profile|` → configuration key; `confirm|a|b` → `a`; `mesh|L…|label` → `label|n…`; `match|label` → `label|meshpath|n65`. All four real p1 profile contracts pass.

Saved-hash trust and audited matcher-input trust remain **NOTED under A5**, not blockers.

Additional evidence: [round-1 replay](/tmp/speed3d-round8-audit/probes.log), [real p1 contracts](/tmp/speed3d-round8-audit/real-contract.log), [batch/spectral regressions](/tmp/speed3d-round8-audit/new-probes.log), [freeze/resources](/tmp/speed3d-round8-audit/freeze-probe.log), [constructor guard](/tmp/speed3d-round8-audit/guard-probe.log).