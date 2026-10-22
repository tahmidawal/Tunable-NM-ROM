**NEEDS-RESTATEMENT — not CLEAN.** The primary **FIRES** conclusion survives.

Read-only, in-memory regeneration exactly reproduced `report.md`, `EXPORT.json`, `checks/analysis.json`, `checks/probe_numbers.json`, and `checks/nomatch.json`. Extracted probe numbers agree with their original sources. No files changed, jobs submitted, or large arrays loaded.

| Previous required fix | Status | Verdict |
|---|---|---|
| Markdown table escaping | **FIXED** | **CORRECT** |
| Numerical provenance / JSON inputs | **NOT FIXED fully** | **NEEDS-RESTATEMENT** — J2 now uses extracted JSON, but A100 validation still reads logs during report generation. |
| Compilation conditionality | **FIXED** | **CORRECT** — includes break-even and robustness conditional on \(B\). |
| Measured-item scope and projected bank fit | **FIXED** | **CORRECT** — headline, cost discussion and figure disclose these. |
| A2.1 criteria and NO-MATCH coverage | **FIXED** | **CORRECT** — all **504** NO-MATCH cells across all accounting variants are exported. |
| A_sel construction categories | **FIXED** | **CORRECT** as the disclosed descriptive deviation. |
| Glossary / LaTeX | **NOT FIXED fully** | **NEEDS-RESTATEMENT** — previously named gaps repaired; remaining gaps below. |
| Standalone PNG exclusions and projection labels | **FIXED** | **CORRECT** — both are visible within the PNG. |
| Break-even grid metadata | **FIXED** | **CORRECT** — distinguishes **241 nominal samples** from **208 evaluated integer query counts**. |
| EXPORT A_sel provenance | **FIXED** | **CORRECT** — competitor runs, construction, J5 ledger and historical head sources identified. |
| Explicit withdrawal | **FIXED** | **CORRECT** — comparative multi-mesh cost claim expressly withdrawn. |

The earlier PM-1024 compilation correspondence, query-time medians and timing-block counts **22/30/33** remain **FIXED / CORRECT**.

Further audit findings:

- **CORRECT — numerical outcomes and claim strength.** All 15 criterion/accounting combinations reproduce. Primary: **126 MATCH**, CP/ours ratios **0.099119–0.130717**, median **0.104847**. Casewise: **84 MATCH / 42 NO-MATCH** per accounting variant. Rollout: **126 NO-MATCH** per variant. Every NO-MATCH lacks an accuracy-admissible competitor; the report correctly calls the cost advantage undetermined. Neither sensitivity is presented as an advantage.

- **CORRECT — A_sel and EXPORT.** Construction **2,963.050609 s** and maximum ratio **0.255314** reproduce. EXPORT carries the principal outcomes, ratios, offline costs, compilation sensitivities, crossings, PM estimate, qualifications and source references.

- **CORRECT — plot and PM wording.** The PNG uses logarithmic axes, queries **per mesh**, H200 seconds, **R384/CNR**, and the cheapest aggregate-worst-admissible CP combination. Its exclusions and projection labels are readable. PM-4096 remains an **estimated snapshot-only cost**, with sample range and remaining setup/accuracy explicitly unmeasured.

- **NEEDS-RESTATEMENT — remaining JSON-input violation.** [make_report.py:40](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/make_report.py:40) still parses the A100 training log directly to generate its validation projection and **0.134%** discrepancy. Calling `historical.records()` also reads a log. These numbers are traceable and reproduce, but the report is still not generated exclusively from run JSONs as §10 requires. Extract these historical validation inputs once, then consume that JSON. Correct the status line’s stale J2 extraction-path description too.

- **NEEDS-RESTATEMENT — glossary completeness and query boundary.** The closing glossary still leaves **bank/head dependency, bank rotation, warm query, tight/tighter solves, and fresh-process versus in-process compilation** unexplained. In particular, those compilation-table columns need plain-language definitions. The query-time discussion should also state A2.7’s boundary: input already on device, outputs ready on device; **host transfers and storage excluded**. “Init + steps + decode” does not convey that boundary.

**Required fixes:** finish the historical JSON-input conversion; complete the named glossary definitions; state the device-only query boundary and excluded transfers/storage.