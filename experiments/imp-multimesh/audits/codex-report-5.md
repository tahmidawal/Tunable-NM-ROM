**CLEAN.** The primary **FIRES** conclusion survives round 5.

Read-only regeneration in memory exactly reproduced `report.md`, `EXPORT.json`, `checks/analysis.json`, `checks/probe_numbers.json`, and `checks/nomatch.json`. No files changed, jobs submitted, or large arrays loaded.

| Round-4 required fix | Status | Verdict |
|---|---|---|
| Historical A100 JSON-input conversion | **FIXED** | **CORRECT** — report generation consumes `inputs/historical.json`; its values reproduce from the original sources. |
| Stale extraction-path description | **FIXED** | **CORRECT** — distinguishes historical extraction and J2’s extracted timers. |
| Named glossary gaps | **FIXED** | **CORRECT** — bank/head dependency, rotation, warm queries, tight/tighter solves, and fresh/in-process compilation are defined. |
| Device-only query boundary | **FIXED** | **CORRECT** — inputs and outputs on device; host transfers and storage explicitly excluded. |

Further adversarial checks:

- **CORRECT — numerical provenance.** Historical and J2 extracted numerical values match their sources. Independently re-expressing admissibility and enumerating combinations checked **1,890** criterion/accounting cells. Primary: **126 MATCH**, CP/ours ratios **0.099119–0.130717**, median **0.104847**.
- **CORRECT — claim strength.** The comparative multi-mesh cost claim is expressly withdrawn. Casewise has **42 NO-MATCH** cells per accounting variant; rollout has **126**. These lack admissible competitors and are explicitly described as **cost advantage undetermined**, not advantages.
- **CORRECT — labels and scope.** Developmental/in-sample status, conditional $\varepsilon_{\rm ref}$, compilation conditionality, robustness conditional on $B$, projected bank-fit cost, and uncosted rotation/rule selection are disclosed. Break-even estimates retain their compilation condition.
- **CORRECT — PM wording.** PM-4096 is an **estimated snapshot-only cost**, with the sample range and remaining setup/accuracy explicitly unmeasured.
- **CORRECT — formatting and glossary.** Title, status, LaTeX, table escaping, closing definitions, and retractions satisfy the requested checks. Previously repaired compilation correspondence, medians, and timing-block counts remain correct.
- **CORRECT — plot.** Visually inspected: logarithmic axes, queries **per mesh**, H200 seconds, **R384/CNR**, and cheapest aggregate-worst-admissible CP. Configuration, accounting, exclusions, projection, and conditionality labels are visible.
- **CORRECT — EXPORT.** Key outcomes, ratios, offline costs, compilation sensitivity, break-even grid, PM estimate, A_sel quantities, qualifications, and source pointers are present and reproduce. All **504** NO-MATCH details remain available through the referenced JSON.

**Required fixes: none. CLEAN.**