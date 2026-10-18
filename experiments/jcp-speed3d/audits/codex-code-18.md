**NOT READY.** Audited `ce10aecbd`. Both smoke outputs were freshly scored on CPU and passed K4. No repository, LAB-LOG, or main writes; no jobs submitted. Production scores were not used.

A21 fixes both exact round-11 reproductions. Two blocking findings remain.

1. **Incomplete scores still publish unqualified median/worst errors.**

   [make_report.py:435](/tmp/speed3d-round12-audit/lane/make_report.py:435) prints m1’s retained partial statistics without displaying `scores_ok` or the scored population.

   With otherwise valid solver evidence:

   | Evidence | Worst error | `scores_ok` | Report |
   |---|---:|---|---|
   | All 64 cases | 0.90 | true | Full-population statistic |
   | Last score missing | 0.01 | false | **eligible yes; worst 0.0100** |

   Matching correctly becomes unavailable and coverage becomes incomplete. However, the per-mesh table still presents the partial statistic as the ROM’s error. The general coverage warning does not identify these numbers as partial.

   The new replication markdown has the same omission at [analyze_r1.py:86](/tmp/speed3d-round12-audit/lane/analyze_r1.py:86): it prints median/worst but omits `scores_ok` and population counts. This downstream behavior was tested using an explicitly labelled in-memory adapter to bypass finding 2.

   Suppress these statistics when scores are incomplete, or label them explicitly as partial with observed/expected counts.

   [Unmodified m1 reproduction](/tmp/speed3d-round12-audit/extra-probe.log) · [Generated report](/tmp/speed3d-round12-audit/partial-mesh-score-report.md)

2. **`analyze_r1.py` crashes on the actual ADT row schema.**

   [analyze_r1.py:46](/tmp/speed3d-round12-audit/lane/analyze_r1.py:46) passes ADT rows to `PM.rows_elig()`. That helper unconditionally sums `r['steps_n']` at [analyze_pm.py:91](/tmp/speed3d-round12-audit/lane/analyze_pm.py:91).

   `s3ctx.rollouts()` writes `steps_n` only for non-ADT certified runs. A complete writer-shaped replication fixture therefore raises:

   ```text
   KeyError: 'steps_n'
   ```

   The real p1 ADT rows independently confirm that this field is absent. No replication analysis is published.

   ADT validity/population checking must work without certified-step counters.

   **Correction to my intermediate observation:** the ADT key is correct—`s3common.spec_key()` inserts `LSPG`. The defect is the row-field contract, not key naming.

   [Faithful schema reproduction and traceback](/tmp/speed3d-round12-audit/r1-schema-probe.log)

Every round-1 item was reassessed:

| Item | Verdict | Round-12 assessment |
|---|---|---|
| H1.4/H2 ratio direction | **CORRECT** | Directions, thresholds and non-null identity shortcuts agree. |
| Confirmation identity/completeness | **CORRECT** | Wrong A and missing invocations rejected. |
| Raw timing recomputation | **CORRECT** | Stale-ratio fixture returns **3.63636**, rejecting drift **10**. |
| Case alignment and η | **CORRECT for verdict gating** | Incomplete populations cannot pass comparisons; partial-statistic publication remains finding 1. |
| A10 coverage/acceptance | **WRONG overall** | Coverage flags now work; report still hides incomplete score coverage beside statistics. |
| H3.1 profile/G3 | **CORRECT** | Invalid panels suppress G3; missing-profile rendering and stale-plot regressions pass. |
| H3.2 baseline/qualification | **CORRECT** | Correct baseline; missing evidence remains unavailable. |
| K5 | **CORRECT** | Expected controls participate; unresolved evidence stays three-state. |
| H3.3 batching/G4/K6/K6+/2× | **CORRECT** | Invalid hashes now make coverage incomplete; report exposes usability. Partial/all-OOM regressions pass. |
| A1.13 δ_quad | **CORRECT** | Complete, valid, finite evidence required. |
| A2.7(d) reduced ADT | **CORRECT** | Population/validity checks and finalist gating remain enforced. |
| H4.1 | **CORRECT** | Invalid panels excluded; incomplete meshes and failed integrity suppress verdicts. |
| H4.2 | **CORRECT** | Valid same-mesh subjects only. |
| H4.3/A6 | **CORRECT for matching** | Mesh-specific qualification and exhaustion checks hold. |
| H-comp family/union cells | **CORRECT** | Unmatched cells excluded; unresolved families prevent unsupported ≥2× conclusions. |
| Governing-plan K2 | **CORRECT** | Three-state string and boolean preserved. |
| Freeze/r1 contents | **CORRECT under A12** | All **44 H1.4 labels**, **52 ROM configurations** included; failed final rename leaves no freeze; retry succeeds. |
| 923901 guard | **CORRECT under A12** | Full r1 accepted; wrong digest/mode and unfrozen members rejected. Documented subsets remain permitted. |
| `resources.project_rep` | **CORRECT for stated proxies** | Independent expansion agrees at **4.652881254921977 h**, including warm-up burns. |
| Hand-typed report numbers | **CORRECT for cited defects** | Previously identified result-dependent literals remain data-derived. |
| EXPORT completeness | **CORRECT for reviewed p1/m1 fields** | Coverage and score-validity flags survive; stale accepted exports no longer survive the reproduced failures. |
| New A1.11 replication analysis | **WRONG** | ADT-schema crash; downstream partial-score publication needs correction. |

All A-subject contracts agree with their writers: **31 confirmation + 4 profile** real p1 panels and **20 real m1 panels** passed.

The round-11 recheck confirms invalid batches produce `coverage_complete=False`; malformed analysis removes previous publications; missing scores remove the previous analysis, and subsequent reporting does not republish its acceptance. [Evidence](/tmp/speed3d-round12-audit/round11-recheck.log)

Saved-hash trust and audited matcher-input trust remain **NOTED under A5**, not blockers. Freeze publication remains safe ordering, not a two-file atomic transaction.