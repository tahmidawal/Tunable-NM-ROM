**NOT READY.** Audited `bdc207e992e4048b506d0dac5fd6c443f1a92ca9`.

Both smoke outputs were rescored with `score3.py` on CPU; both passed K4. Fixtures and generated reports stayed under `/tmp/speed3d-round3`. No repository files, LAB-LOG, or main were written; no jobs submitted. The existing `HEARTBEAT.json` modification was left alone.

Two blocking defects remain:

1. **Primary H4.3 union still treats unresolved families as resolved.** [analyze_pm.py:456](experiments/jcp-speed3d/analyze_pm.py:456) copies the best observed family’s `reaches_2x` into the union without considering unavailable or untimed families. A production-population fixture with a matched **3×** family and an unavailable family returns `union.reaches_2x=True`. The generated report prints **“3.00 | yes | cheapest matched FOM over families.”** Completing the missing family with a matched **0.5×** timing reverses that conclusion. The largest-mesh K2 implementation handles this correctly; the primary union needs the same asymmetric treatment.

2. **Integrity suppression does not survive publication consistently.** [analyze_pm.py:324](experiments/jcp-speed3d/analyze_pm.py:324) suppresses selected field names, leaving verdicts such as G4, K6/K6+, and reduced-ADT `ok` intact. It sets `K5_fires=None`, but [make_report.py:381](experiments/jcp-speed3d/make_report.py:381) converts that into **“DOES NOT fire.”** The genuine smoke—correctly rejected as non-production—reproduces this false negative in the generated report. A12 requires unavailable verdicts when integrity fails.

These are missing-evidence/publication paths, not forged-file findings under A5.

References in the table are relative to `experiments/jcp-speed3d/`. “Locally” means the calculation and its direct evidence checks are correct; the publication defect above remains separate.

| Round-1 item | Round-3 verdict | Evidence |
|---|---|---|
| H1.4/H2 ratio direction | **CORRECT** | Selected/comparator ratio, reciprocal H2 speed-up, and thresholds remain correct. |
| Confirmation identity and completeness | **CORRECT** | Wrong A and missing invocation are rejected. Exact case×repetition checks remain enforced. |
| Validity recomputed from invocations | **CORRECT** | Stale-ratio fixture returns **3.63636** and rejects drift **10**. Retained-hash limitation remains **NOTED under A5**. |
| Case alignment and η | **CORRECT** | Missing m1 score keys no longer crash. Incomplete populations cannot pass b/c; η consistency remains checked. |
| A10 required-block coverage / acceptance | **WRONG overall** | A12’s integrity-versus-coverage distinction is implemented, and smoke cohorts are rejected. But suppression/publication is incomplete; primary unresolved-family coverage can also report `coverage_complete=True`. |
| H3.1 profile and G3 | **CORRECT locally** | All four genuine smoke profile panels satisfy subject identity checks. Invalid panels suppress G3; component arithmetic and report validity column are correct. |
| H3.2 baseline and qualification | **CORRECT locally** | Correct mesh-path baseline and mesh-free-initialised ADT. Otherwise eligible paths with incomplete scores now report unavailable. |
| K5 | **WRONG overall** | Rank/rule checks and qualification blocking are correct. The report still converts unavailable K5 into a definitive failure. |
| H3.3 batching, G4, K6, K6+, 2× bar | **CORRECT locally** | With integrity true, failed K6+ blocks the otherwise **16×** fixture. All-OOM records yield unavailable controls and bar, not vacuous passes. |
| A1.13 δ_quad | **CORRECT locally** | Invalid converged-rule rows now make the measurement unavailable; complete rows and finite distances are required. |
| A2.7(d) reduced ADT | **CORRECT locally** | Empty validity arrays fail; population and finite-distance checks remain; the gate affects finalist H2 acceptance. |
| H4.1 | **CORRECT** | Hash-invalid panels supply no timings. Missing panels yield unavailable. Failed integrity suppresses the pass verdict. |
| H4.2 | **CORRECT locally** | Only valid, same-mesh subjects enter the diagnostic ratios. Invalid-panel fixture exports empty maps. |
| H4.3 and A6 | **CORRECT locally** | Primary and mesh-panel identities match the runner; mesh-specific matching requires eligible ROM scores. The union defect is recorded separately below. |
| H-comp family/union cells | **WRONG overall** | Both unions exclude unmatched cells. Largest-mesh unresolved-family handling is repaired, but primary H4.3 still produces a false definitive ≥2× union claim. |
| Governing-plan K2 | **CORRECT** | Missing family, matched-but-untimed family, and unmatched-only fixtures yield **undetermined**. Completing the fixture with a matched **0.5×** family yields **fires**. |
| A1.11 freeze and r1 contents | **CORRECT under A12** | All **44 H1.4 labels** are included; r1 contains **52 ROM configurations**. Final-rename failure leaves r1 present and freeze absent; retry succeeds. This is safe publication ordering, not a two-file transaction. |
| 923901 guard | **CORRECT for its restated scope** | Actual constructor accepts generated r1, including H1.4 configurations. It also accepts subsets, as A12 explicitly acknowledges. Completeness remains a job-audit responsibility. |
| `resources.project_rep` | **CORRECT for the stated proxy inventory** | Independent expansion agrees exactly: **4.652881255 h**. Warm-up burns are now included. This verifies arithmetic, not actual runtime. |
| No hand-typed report numbers | **CORRECT for the cited defects** | Previously cited result-dependent literals remain sourced from data. |
| EXPORT completeness | **WRONG overall** | FOM timing, convergence, calibration, and infeasibility context are now retained. Export still inherits the primary-union and incomplete-suppression defects. |

The A-subject contract agrees across reader and writer:

| Panel | Expected and written A |
|---|---|
| `profile|label` | `setting|rule|spec` |
| `confirm|a|b` | `a` |
| `mesh|L…|label` | `label|n…` |
| `match|label` | `label|meshpath|n65` |

Evidence: [fresh-score fixture replay](/tmp/speed3d-round3/probes.log), [integrity-true batching fixtures](/tmp/speed3d-round3/local-probe.log), [K2/union fixtures](/tmp/speed3d-round3/k2-probe.log), [generated-report reproduction](/tmp/speed3d-round3/report-probe.log), and [actual guard checks](/tmp/speed3d-round3/guard-probe.log).