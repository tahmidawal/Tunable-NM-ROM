**NOT READY.** Audited commit `72c11c26cf325a252c7eeee13e6184880d26ba52`.

Both smoke outputs were rescored successfully with `score3.py` on CPU; both passed K4. Fixtures used checksum-matched copies under `/tmp/speed3d-round5-audit`. No repository, LAB-LOG, or main files were changed; no jobs submitted. The pre-existing `HEARTBEAT.json` modification remains untouched.

Two blocking defects remain.

1. **K5 still reports a definitive negative when an entire required control is unavailable.**

   [analyze_pm.py:215](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/analyze_pm.py:215) skips K5 creation when the mesh-free block is absent or marked invalid. The aggregate at line 233 then excludes that required control entirely.

   A production-shaped fixture with one resolved, non-firing control and another missing/invalid mesh-free block produces:

   - `K5_fires=False`, `K5_resolved_present=True`.
   - The surviving path: `blocked (K5 did not fire)`.
   - Report: **“K5 control … DOES NOT fire.”**

   This can occur honestly: `s3modes.mode_prof` skips the control after a mesh-free Gram-condition failure. Coverage being false does not prevent the false negative.

   Aggregate over **every expected control**, retaining unavailable entries as `None`. A missing control must prevent aggregate `False`, although an observed firing control can establish `True`. [Fresh fixture and publication evidence](/tmp/speed3d-round5-audit/partial_probe-fresh.log).

2. **The report creates a new definitive claim after m1 integrity suppression.**

   [make_report.py:486](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/make_report.py:486) derives “confirmed”/“not confirmed” directly from retained spectral speedups, without checking integrity.

   With `complete=False`, analysis correctly returns `integrity_ok=False`, `accepted_for_claims=False`, and suppressed hypothesis verdicts. Nevertheless, a retained diagnostic spectral ratio of **0.5** makes the generated report say:

   > “spectral FOM clearly beats the ROM at 64³” is **confirmed**

   Gate this derived claim on accepted evidence; retaining diagnostic numbers must not restore a suppressed conclusion. [Fixture and generated wording](/tmp/speed3d-round5-audit/integrity-report-probe.log).

Both are honest missing-evidence/incomplete-job paths, so they are **blocking under A5**.

The two **exact** round-4 reproductions are repaired: removing coarse-rule scores now produces unavailable K5, and removing mesh-free rows now produces unavailable H3.2 with incomplete coverage. [Fresh replay](/tmp/speed3d-round5-audit/missing_probe-fresh.log).

| Round-1 item | Round-5 verdict | Evidence |
|---|---|---|
| H1.4/H2 ratio direction | **CORRECT** | Selected/comparator ratio, reciprocal H2 speedup, and thresholds agree. |
| Confirmation identity and completeness | **CORRECT** | Wrong A and missing invocations rejected; exact case×repetition coverage enforced. |
| Validity recomputed from invocations | **CORRECT** | Stale-ratio fixture returns **3.63636**, rejects drift **10**. Saved-hash trust remains **NOTED under A5**. |
| Case alignment and η | **CORRECT** | Missing score keys handled; incomplete populations cannot pass b/c; η consistency checked. |
| A10 coverage and acceptance | **WRONG overall** | Partial-control K5 gives a false negative; report-derived spectral conclusion bypasses integrity. |
| H3.1 profile and G3 | **CORRECT locally** | Genuine smoke subjects agree; invalid panels suppress G3; component arithmetic correct. |
| H3.2 baseline and qualification | **WRONG overall** | Baseline and missing-row repair correct; unresolved K5 can still falsely block another path. |
| K5 | **WRONG** | Three-state aggregation works for retained `None` entries but drops entirely absent controls. |
| H3.3 batching, G4, K6, K6+, 2× bar | **CORRECT** | Failed K6+ blocks otherwise **16×** throughput. All-OOM records yield unavailable controls/bar. |
| A1.13 δ_quad | **CORRECT locally** | Invalid converged-rule rows make the measurement unavailable; population and finite-distance checks enforced. |
| A2.7(d) reduced ADT | **CORRECT** | Empty validity arrays rejected; population/finite-distance checks gate finalist H2. |
| H4.1 | **CORRECT** | Invalid panels contribute no timings; incomplete meshes unavailable; failed integrity suppresses verdict. |
| H4.2 | **CORRECT locally** | Valid, same-mesh subjects only. |
| H4.3 and A6 | **CORRECT locally** | Panel identities and mesh-specific ROM eligibility checked. Spectral publication defect remains separate. Matcher-input trust remains **NOTED under A5**. |
| H-comp family/union cells | **CORRECT** | Both unions exclude unmatched cells and consume unresolved-family status. |
| Governing-plan K2 | **CORRECT** | Missing/untimed/unmatched-only cases remain undetermined; adding a matched **0.5×** family makes K2 fire. |
| A1.11 freeze and r1 contents | **CORRECT under A12** | All **44 H1.4 labels** included; **52 ROM configurations**. Final-rename failure leaves r1 without freeze; retry succeeds. Safe ordering, not a two-file transaction. |
| 923901 guard | **CORRECT for restated scope** | Actual constructor accepts full generated r1; rejects wrong digest/mode and unfrozen ROM/FOM members. Subsets remain permitted as documented. |
| `resources.project_rep` | **CORRECT for stated proxies** | Independent expansion agrees at **4.652881254921977 h**, including warm-up burns. Arithmetic verification, not runtime validation. |
| No hand-typed report numbers | **CORRECT for cited defects** | Previously cited result-dependent literals remain data-derived. |
| EXPORT completeness | **WRONG overall** | Timing/validity context retained, but H32 and acceptance exports inherit the partial-control K5 defect. |

The A-subject contracts agree with the runners:

| Panel | Expected and written A |
|---|---|
| `profile\|label` | `setting\|rule\|spec` |
| `confirm\|a\|b` | `a` |
| `mesh\|L…\|label` | `label\|n…` |
| `match\|label` | `label\|meshpath\|n65` |

Additional evidence: [fresh round-1 fixture replay](/tmp/speed3d-round5-audit/probes-fresh.log), [batching](/tmp/speed3d-round5-audit/local_probe-fresh.log), [union/K2](/tmp/speed3d-round5-audit/k2-probe.log), [freeze/resource arithmetic](/tmp/speed3d-round5-audit/freeze-probe.log), [constructor guard](/tmp/speed3d-round5-audit/guard-probe.log).