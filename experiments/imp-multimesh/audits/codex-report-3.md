**NEEDS-RESTATEMENT — not CLEAN.** The primary **FIRES** conclusion survives. In-memory regeneration exactly reproduced `report.md`, `EXPORT.json`, `checks/analysis.json`, `checks/probe_numbers.json`, and `checks/nomatch.json`. No files were written, jobs submitted, or large arrays loaded.

| Previous required fix | Status | Round-3 verdict |
|---|---|---|
| Markdown table escaping | **FIXED** | **CORRECT.** Escaped configuration/workload separators remain intact. |
| Numerical provenance | **NOT FIXED fully** | **NEEDS-RESTATEMENT.** Measurements reproduce from sources, but the generator still reads logs directly. Writing `probe_numbers.json` as an output does not make it the report’s input as COMMON-BRIEF §10 requires. |
| Compilation conditionality | **FIXED** | **CORRECT.** Break-even now carries its own conditional label; all registered verdicts remain robust within the stated \(B\). |
| Measured-item scope and projected fit | **FIXED** | **CORRECT** in the report’s headline, offline discussion, and figure caption. |
| A2.1 criteria and NO-MATCH explanations | **NOT FIXED fully** | **NEEDS-RESTATEMENT.** Detailed alternatives now exist for `A_dep`, but not the other reported accounting variants. |
| A_sel construction categories | **FIXED** | **CORRECT** as the explicitly disclosed descriptive scenario: construction **2,963.050609 s**, maximum ratio **0.255314**. |
| Glossary and LaTeX | **NOT FIXED fully** | **NEEDS-RESTATEMENT.** Previously named terminology was added; remaining gaps are listed below. |
| Standalone plot qualifications | **NOT FIXED fully** | **NEEDS-RESTATEMENT.** Caption repaired; PNG still omits exclusions and projected-fit disclosure. |
| EXPORT completeness | **NOT FIXED fully** | Added results are **CORRECT**, but grid metadata is **WRONG**, and A_sel source attribution remains incomplete. |
| Explicit withdrawal | **FIXED** | **CORRECT.** Comparative multi-mesh cost claim expressly withdrawn. |

The earlier PM-1024 compile correspondence, query-time medians, and timing-block counts **22/30/33** remain **FIXED / CORRECT**.

Further findings:

- **CORRECT — claim strength and outcomes.** All 15 criterion/accounting combinations reproduce. Primary: **126 MATCH**. Casewise: **84 MATCH, 42 NO-MATCH**. Rollout: **126 NO-MATCH**. The NO-MATCH cells have no admissible competitor, and the report explicitly says cost advantage is undetermined. It does not turn these sensitivities into a comparative advantage.

- **NEEDS-RESTATEMENT — incomplete NO-MATCH coverage.** [make_report.py:137](experiments/imp-multimesh/make_report.py) generates details only for `casewise|A_dep` and `rollout|A_dep`: **168 records**. The outcome table contains **504 NO-MATCH cells** across the three accounting variants. Competitor choices can be shared, but our costs and cost ratios differ. Export variant-specific details, or explicitly provide shared alternatives plus variant-specific costs and mappings. “Every NO-MATCH is listed” currently overstates coverage.

- **WRONG — break-even grid description.** [report.md:239](experiments/imp-multimesh/report.md) and `EXPORT.json` claim **241 points**. `cost.breakeven` rounds and deduplicates 241 logarithmic samples, leaving **208 distinct integer query counts**. The crossings reproduce; the metadata does not describe the evaluated grid accurately. Distinguish nominal samples from evaluated points and qualify the approximate 6% spacing.

- **NEEDS-RESTATEMENT — standalone PNG.** Axes, logarithmic scales, queries **per mesh**, H200 seconds, **R384/CNR**, and cheapest aggregate-worst-admissible CP are correct. Developmental/in-sample and compilation conditions are visible. The PNG itself still lacks “rotation and rule selection uncosted” and “bank fit projected.” These appear only in the Markdown caption.

- **NEEDS-RESTATEMENT — glossary/notation.** The closing glossary still does not explain **FIRES/NOT FIRED**, **lower envelope**, **phase-wise projection**, or the configuration tokens `R…` versus `r…|k…`. It uses unexplained concepts such as residual, snapshot, and successor rule. Mathematical expressions outside tables still include `κ = 4`, `10⁻³`, and `ε_ref` rather than LaTeX.

- **NEEDS-RESTATEMENT — export provenance.** Compilation counts, A_sel components, ratios, crossings, PM estimate, and qualifications are exported and reproduce. However, the cited `checks/analysis.json` does **not** contain A_sel. Add explicit sources for its competitor-run sum and `runs/GPU-LEDGER.json` J5 contribution.

- **CORRECT — PM wording.** PM-4096 remains an **estimated snapshot-only cost**, with sample range and unmeasured remaining setup/accuracy. No “loses regardless” claim appears.

**Required fixes:** complete NO-MATCH accounting coverage; make extracted timing JSON an actual input or explicitly resolve the §10 exception; correct grid metadata; qualify the standalone PNG; finish glossary/LaTeX cleanup; add A_sel source paths.