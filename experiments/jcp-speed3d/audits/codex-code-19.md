**NOT READY.** Audited `81b4251c0`. A22 fixes both exact round-12 reproductions, but one blocking publication path remains.

Both smoke outputs were freshly scored on CPU and passed K4. All work stayed under `/tmp`; no repository, LAB-LOG, or main writes and no jobs submitted. Production scores were not used.

**Blocking: failed FOM trajectories disappear from error statistics without disclosure.**

The writer at [s3modes.py:135](/tmp/speed3d-round13-audit/lane/s3modes.py:135) honestly omits score fields when a FOM trajectory is nonfinite. However:

- [analyze_pm.py:441](/tmp/speed3d-round13-audit/lane/analyze_pm.py:441) computes median/worst over surviving scored cases while reporting `n_cases=len(cs)`.
- [make_report.py:454](/tmp/speed3d-round13-audit/lane/make_report.py:454) prints those statistics and the full count, displays the calibration status, and omits the recorded convergence failure.
- EXPORT preserves the partial statistics and full count. It retains `converged=false`, but supplies no scored-population count or partial-statistic label.

The writer-shaped fixture reproduces:

| Evidence | Scored cases | Published worst | Published cases | Report status |
|---|---:|---:|---:|---|
| Complete FOM scores | 8 | 0.90 | 8 | qualified |
| One nonfinite trajectory | 7 | **0.01** | **8** | **qualified** |

This is an honest solver-failure path, not fabricated hash evidence. A22’s ROM-statistics repair does not cover these FOM statistics. Suppress incomplete statistics or explicitly label their population and failure status beside them.

[Reproduction log](/tmp/speed3d-round13-audit/new_round13_probe.log) · [Generated report](/tmp/speed3d-round13-audit/nonfinite-fom-report.md) · [EXPORT](/tmp/speed3d-round13-audit/nonfinite-fom-EXPORT.json)

Every round-1 item was reassessed:

| Item | Verdict | Round-13 assessment |
|---|---|---|
| H1.4/H2 ratio direction | **CORRECT** | Directions, thresholds and non-null identity shortcuts agree. |
| Confirmation identity/completeness | **CORRECT** | Wrong A and missing invocations rejected. |
| Raw timing recomputation | **CORRECT** | Stale-ratio fixture returns **3.63636**, rejecting drift **10**. |
| Case alignment and η | **CORRECT for ROM comparisons** | Population checks hold; partial ROM scores now produce null statistics. |
| A10 coverage/acceptance | **WRONG overall** | Earlier gating regressions pass; incomplete FOM statistics still become unqualified published claims. |
| H3.1 profile/G3 | **CORRECT** | Invalid panels suppress G3; missing-profile and stale-plot regressions pass. |
| H3.2 baseline/qualification | **CORRECT** | Correct baseline; missing evidence remains unavailable. |
| K5 | **CORRECT** | Every expected control participates; unresolved evidence stays three-state. |
| H3.3 batching/G4/K6/K6+/2× | **CORRECT** | Invalid batches block claims and complete coverage; partial/all-OOM regressions pass. |
| A1.13 δ_quad | **CORRECT** | Complete, valid, finite evidence required. |
| A2.7(d) reduced ADT | **CORRECT** | Population/validity checks and finalist gating hold. |
| H4.1 | **CORRECT** | Invalid panels excluded; incomplete meshes and failed integrity suppress verdicts. |
| H4.2 | **WRONG overall** | Timing subjects and validity filtering are correct; accompanying FOM error statistics have the blocker above. |
| H4.3/A6 | **CORRECT for matching** | Mesh-specific qualification and exhaustion checks hold. |
| H-comp family/union cells | **CORRECT** | Unmatched cells excluded; unresolved families prevent unsupported ≥2× conclusions. |
| Governing-plan K2 | **CORRECT** | Three-state string and boolean preserved. |
| Freeze/r1 contents | **CORRECT under A12** | All **44 H1.4 labels**, **52 ROM configurations** included. Failed final rename leaves no freeze; retry succeeds. |
| 923901 guard | **CORRECT under A12** | Full r1 accepted; wrong digest/mode and unfrozen members rejected. Documented subsets remain permitted. |
| `resources.project_rep` | **CORRECT for stated proxies** | Independent expansion agrees at **4.652881254921977 h**, including warm-up burns. |
| Hand-typed report numbers | **CORRECT for cited defects** | Previously identified result-dependent literals remain data-derived. |
| EXPORT completeness | **WRONG overall** | Retains the misleading FOM statistics/count described above. |
| New A1.11 replication analysis | **CORRECT in exercised scope** | Writer-shaped ADT rows work without an adapter. Partial scores suppress statistics/comparisons; missing ADT evidence suppresses eligibility; incomplete panels/FOM populations suppress their respective results. |

All A-subject contracts agree with their writers: **31 confirmation + four profile** real p1 panels and **20 real m1 panels** passed.

The round-12 ROM fixture now prints **63 scored cases with median/worst blank**. The replication fixture covers mesh/mf paths, H1.4 configurations, comparator timing and FOM comparisons. [ROM regression](/tmp/speed3d-round13-audit/extra_probe.log) · [Replication fixtures](/tmp/speed3d-round13-audit/r1-full-probe.log)

Earlier missing-input/publication regressions also pass: failed analysis removes the previous analysis, and failed report reads remove previous publications. [Regression results](/tmp/speed3d-round13-audit/regression-status.log) · [Freeze/resources](/tmp/speed3d-round13-audit/freeze_probe-round11.log) · [Guard checks](/tmp/speed3d-round13-audit/guard_extra-round11.log)

Saved-hash trust and audited matcher-input trust remain **NOTED under A5**, not blockers. Freeze publication remains safe ordering, not a two-file atomic transaction.