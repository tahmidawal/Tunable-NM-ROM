**NOT READY.** Audited commit `0b70d78d42317e23b19e63d10f44bc0eddf1fa7f`.

Both smoke outputs were rescored successfully with `score3.py` on CPU; both passed K4. All fixtures and generated reports stayed under `/tmp/speed3d-round4`. No repository files, LAB-LOG, or main were written; no jobs submitted. The existing `HEARTBEAT.json` modification was left untouched.

The two specific round-3 reproductions are repaired. Two additional missing-evidence paths remain blocking:

1. **Unavailable K5 evidence becomes a definitive negative.** At [analyze_pm.py:231](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/analyze_pm.py:231), `any(...)` collapses unavailable per-setting K5 results into `False`.

   In a production-shaped fixture built from smoke records, the complete baseline qualifies both mesh-free paths and fires K5. Removing only the coarse-rule score entries produces:
   - Per-setting `K5.fails=None`.
   - Aggregate `K5_fires=False`.
   - Both paths labelled `blocked (K5 did not fire)`.
   - `integrity_ok=True`, `coverage_complete=True`.
   - Generated report: **“K5 control … DOES NOT fire.”**

   K5 needs three-state aggregation: true if any measured control fires; false only if every required control is resolved and none fires; otherwise unavailable.

2. **Missing mesh-free rollout rows become measured disqualification.** [analyze_pm.py:223](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/analyze_pm.py:223) folds incomplete rows into `eligible=False`; the fallback at line 243 then emits `status='not qualified'`.

   Removing only the mesh-free rows from the otherwise complete fixture leaves both paths with `rows_complete=False`, `b=True`, `c=True`, but **`qualified=False`, `status='not qualified'`**. Both integrity and coverage still report true. Missing required evidence should make qualification unavailable and coverage incomplete.

These reproduce missing-score/missing-block failures, not fabricated-hash findings under A5. [Fixture and publication evidence](/tmp/speed3d-round4/missing-probe.log).

References in the table are relative to `experiments/jcp-speed3d/`.

| Round-1 item | Verdict | Round-4 evidence |
|---|---|---|
| H1.4/H2 ratio direction | **CORRECT** | Selected/comparator ratio, reciprocal H2 speed-up and thresholds remain correct. |
| Confirmation identity and completeness | **CORRECT** | Wrong A and missing invocations are rejected; exact case×repetition coverage enforced. |
| Validity recomputed from invocations | **CORRECT** | Stale-ratio fixture returns **3.63636** and rejects drift **10**. Saved-hash limitations remain **NOTED under A5**. |
| Case alignment and η | **CORRECT** | Missing m1 score keys no longer crash; incomplete H3.2 scores cannot pass b/c; η consistency checked. |
| A10 coverage and acceptance | **WRONG** | Missing K5 scores and mesh-free rows can leave `coverage_complete=True` and yield definitive negative outcomes. |
| H3.1 profile and G3 | **CORRECT** | Genuine profile subjects match; invalid panels suppress G3; component arithmetic remains correct. |
| H3.2 baseline and qualification | **WRONG overall** | Baseline and b/c arithmetic correct; missing rollout rows become “not qualified.” |
| K5 | **WRONG** | Rank/rule checks and qualification blocking work, but unavailable evidence becomes aggregate false and is published as a negative. |
| H3.3 batching, G4, K6, K6+, 2× bar | **CORRECT** | Failed K6+ blocks the otherwise **16×** fixture. All-OOM batches yield unavailable controls and bar. |
| A1.13 δ_quad | **CORRECT** | Invalid converged-rule rows make the measurement unavailable; population and finite-distance checks enforced. |
| A2.7(d) reduced ADT | **CORRECT** | Empty validity arrays fail locally; population and finite-distance checks gate finalist H2. |
| H4.1 | **CORRECT** | Hash-invalid panels contribute no timings; missing panels yield unavailable; failed integrity suppresses pass/fail. |
| H4.2 | **CORRECT** | Only valid, same-mesh subjects enter diagnostic ratios. |
| H4.3 and A6 | **CORRECT** | Primary panel identity matches the runner; mesh-specific matching requires eligible ROM scores. Audited matcher-input trust remains **NOTED under A5**. |
| H-comp family/union cells | **CORRECT** | Both unions exclude unmatched cells and consume unresolved-family status. Primary **3× + unresolved** now yields `reaches_2x=None` and incomplete coverage. |
| Governing-plan K2 | **CORRECT** | Missing, untimed and unmatched-only fixtures yield undetermined; completing a matched **0.5×** family makes K2 fire. |
| A1.11 freeze and r1 contents | **CORRECT under A12** | All **44 H1.4 labels** included; **52 ROM configurations** total. Final-rename failure leaves r1 without a freeze; retry succeeds. Safe ordering, not a two-file transaction. |
| 923901 guard | **CORRECT for restated scope** | Actual constructor accepts full generated r1, including H1.4 entries; rejects bad digest, wrong mode and unfrozen ROM/FOM members. Subsets remain permitted as documented. |
| `resources.project_rep` | **CORRECT for stated proxies** | Independent expansion agrees exactly at **4.652881255 h**, including warm-up burns. This verifies arithmetic, not runtime. |
| No hand-typed report numbers | **CORRECT for cited defects** | Previously cited result-dependent literals remain sourced from data. |
| EXPORT completeness | **WRONG overall** | Timing and FOM validity context retained; H32 and acceptance exports inherit the missing-evidence defects above. |

The A-subject contracts agree throughout:

| Panel | Expected and written A |
|---|---|
| `profile|label` | `setting|rule|spec` |
| `confirm|a|b` | `a` |
| `mesh|L…|label` | `label|n…` |
| `match|label` | `label|meshpath|n65` |

The round-3 publication fixture now prints **K5 “unavailable”** when integrity fails, and the primary unresolved union prints **3.00, ≥2× unavailable**, explicitly labelled an upper bound with unresolved UC.

Evidence: [fixture replay](/tmp/speed3d-round4/probes.log), [batching controls](/tmp/speed3d-round4/local-probe.log), [union/K2](/tmp/speed3d-round4/k2-probe.log), [publication regression](/tmp/speed3d-round4/report-probe.log), [freeze/resource checks](/tmp/speed3d-round4/freeze-probe.log), [actual constructor guard](/tmp/speed3d-round4/guard-probe.log).