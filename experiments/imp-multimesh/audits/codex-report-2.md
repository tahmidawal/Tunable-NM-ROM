**NEEDS-RESTATEMENT — not CLEAN.** The primary **FIRES** conclusion survives. In-memory regeneration exactly reproduced `report.md`, `EXPORT.json`, and `checks/analysis.json`. No files changed, jobs ran, or large arrays were loaded.

| Previous required fix | Status | Round-2 verdict |
|---|---|---|
| Markdown table escaping | **FIXED** | **CORRECT.** No inconsistent table-column counts remain. |
| Numerical provenance | **NOT FIXED fully** | **NEEDS-RESTATEMENT.** Historical hard-coded measurements are replaced by source reads. However, COMMON-BRIEF §10 requires run-JSON generation; H200 rates/head timings, historical generation time, and A100 validation still originate directly from logs. The report now honestly discloses this deviation. |
| A7.1 compilation conditionality | **NOT FIXED fully** | **NEEDS-RESTATEMENT.** Headline, workload table and figure now carry the condition. The break-even section still lacks its own compile-conditional label. All recomputed verdicts are robust within \(B=40.4757755\) seconds. |
| Measured-item cost scope and projected fit | **FIXED** | **CORRECT** beside the headline and offline table: exclusions and projected bank fit are explicit. |
| A2.1 criteria and NO-MATCH explanations | **NOT FIXED fully** | **NEEDS-RESTATEMENT.** Criterion definitions are repaired; required per-workload trade-offs remain missing. |
| A_sel omitted construction categories | **FIXED** | **CORRECT** as the disclosed descriptive scenario. All construction categories now total **2,963.050609 s**; the resulting maximum ratio is **0.255** rounded. This remains a deviation from exact A5.2 subtraction, explicitly disclosed. |
| Glossary and mathematical notation | **NOT FIXED fully** | **NEEDS-RESTATEMENT.** Expanded substantially, but still incomplete; examples below. |
| Standalone plot qualifications | **NOT FIXED fully** | **CORRECT** numerically; **NEEDS-RESTATEMENT** for exclusions and projected-fit disclosure. |
| EXPORT completeness | **NOT FIXED fully** | **NEEDS-RESTATEMENT.** Many requested fields were added, but compile robustness and A_sel results remain absent. |
| Explicit withdrawal | **FIXED** | **CORRECT.** The comparative multi-mesh cost claim is expressly withdrawn. |

The three already-fixed items from `codex-results-3.md` remain **FIXED / CORRECT**: PM-1024 compile-arm correspondence, query-time medians, and timing-block counts **22/30/33**.

Further findings:

- **CORRECT — claim strength.** All 15 verdict/accounting combinations reproduce. Primary: **126 MATCH**. Casewise: **84 MATCH, 42 NO-MATCH**. Rollout: **126 NO-MATCH**. Every NO-MATCH lacks an admissible competitor; the report correctly says cost advantage is undetermined. No comparative advantage is inferred from those rows.

- **NEEDS-RESTATEMENT — NO-MATCH detail.** [report.md:39](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/report.md:39) groups workloads into ratio ranges and substitutes the *cheapest primary-admissible* competitor. A2.1 requires, for **each NO-MATCH**, the cheapest CP combination and failure reason, plus the most accurate combination and its cost. Neither configuration choices nor the most-accurate alternative are supplied.

- **NEEDS-RESTATEMENT — local cost labels.** [Break-even:224](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/report.md:224) needs the compilation condition beside its estimates. The criterion description and glossary still call `A_dep` “H200 measured offline cost,” despite its projected bank-fit component.

- **CORRECT numerically; NEEDS-RESTATEMENT for standalone use — plot.** Log axes, queries **per mesh**, H200 seconds, **R384/CNR**, and cheapest aggregate-worst-admissible CP are correct. Developmental/in-sample and compile-conditional labels are visible. Neither the PNG nor its caption names the uncosted rotation/rule selection or identifies the projected bank fit.

- **NEEDS-RESTATEMENT — glossary/LaTeX.** [Glossary:307](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/report.md:307) still leaves NNLS, LM, trust radius, non-stationarity, quadrature and conditional Richardson estimation unexplained in plain language. Expressions such as `R′ ∈ {…}`, `support size = κ r`, and `t̄(L)` remain Unicode mathematics outside LaTeX.

- **NEEDS-RESTATEMENT — export.** Ratios, counts, no-admissible counts, \(B\), crossings, qualifications and historical source paths are present and reproduce. Missing are explicit compile-robust/dependent counts or flags, the corrected A_sel components/ratio, and break-even grid-resolution metadata. A path to the detailed analysis does not itself export these key results.

- **CORRECT — PM wording and status.** PM-4096 remains an **estimated snapshot-only cost**, with sample range and unmeasured remaining setup/accuracy. No “loses regardless” claim appears. Title, status line, developmental/in-sample and conditional-reference labels are present.

**Required fixes:** complete per-NO-MATCH trade-offs; close or explicitly resolve the run-JSON provenance requirement; add local cost/compile qualifications; finish glossary/LaTeX cleanup; export compile robustness and corrected A_sel results with sources and break-even resolution metadata.