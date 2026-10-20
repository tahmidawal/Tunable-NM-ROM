**Overall: NOT READY.** The new numerical tables check out, but several reporting requirements remain unresolved.

Read-only verification reproduced `report.md` **byte-for-byte** and all seven plots **pixel-for-pixel in memory**. I checked every numeric entry in the new timing, budget-diagnostic and orthogonal-pair tables against the JSONs, recomputed timing statistics from archived invocations, and verified the historical budget triples. **No numerical mismatch found. No files were created, modified or appended.**

| Requested item | Verdict | Finding |
|---|---|---|
| Per-time identity; `delta_form` / `delta_time` | **RESOLVED** | Definitions now match the implemented ablations and distinguish per-time identities from aggregated values. |
| \(R_S\) restricted to 2D | **RESOLVED** | Explicitly limited to 2D BE at \(\Delta t_0\). |
| Tie definitions | **PARTIAL** | Generic first-order definition still incorrectly encompasses 3D; see below. |
| `pp`; grid versus budget arm keys | **RESOLVED** | Percentage points and absolute-step versus step-factor keys are defined. |
| Closure interpretation | **RESOLVED** | Descriptive ratio; no orthogonality or additivity claim. |
| LaTeX | **PARTIAL** | Glossary largely repaired; mathematical prose outside it still uses unrendered formulas. |
| \(3U_\star\) threshold | **PARTIAL** | Factor repaired, but reported inequality differs from implementation. |
| Budget diagnostics | **RESOLVED** | Worst form sensitivity, refinement, mixed differences, unavailable totals, time-unresolved counts and reference-dependent worst values are present and correct. |
| Orthogonal pair at \(t^\ast\) | **RESOLVED** | Values reproduce saved common-time pairs; per-case identities verify. |
| Historical \(e_P/e_{\rm dyn}\) beside E0 | **RESOLVED** | Both dimensions’ historical triples reproduce their first-order JSONs. |
| No ROM second-order wording | **PARTIAL** | Mermaid and corrections repaired; job table retains prohibited wording. |
| Timing IQR, MAD outliers, absolute times, common ratios | **NUMERICALLY RESOLVED** | All entries reproduce archived panels; common baseline needs identification. |
| Provenance statement | **NOT RESOLVED** | Measured counts remain hard-coded in generator prose. |
| Audit-statement wording | **RESOLVED** | No longer claims all reporting findings have been fixed. |

Remaining findings:

1. **The glossary still gives 3D an unavailable first-order tie policy.**  
   [report.md:546](experiments/jcp-combined/report.md#L546) describes “first-order references” generically using ST–TX distances. `analyze3d.py:217–221` explicitly says **no verified or heuristic threshold exists for the 3D first-order reference**. Scope that definition to 2D and retain 3D’s descriptive, reference-limited status.

2. **The uncertainty inequality is still inaccurate.**  
   The introduction, diagnostic headings and glossary say **at most / \(\le3U_\star\)**. Both analyzers use **strictly \(<3U_\star\)** (`budget2d.py:279`, `budget3d.py:262`), matching DESIGN’s “smaller than.” Printed counts match the implementation; repair the wording.

3. **ROM order wording survives in the job table.**  
   `report.md:473` calls `c3grid` **“3D second-order grid + timing.”** This is imported ledger text, but appears as the report’s own description. Render it using CN/BDF2-family terminology or explicitly label it historical terminology.

4. **The blanket provenance claim remains false.**  
   `make_report.py:848` hard-codes the measured **“one case”** escalation result; lines 863–864 hard-code **“two of its three transitions reverse direction.”** These are observed counts, not registered design constants. Removing the elapsed-time literal did not eliminate all hand-typed measured numbers.

5. **Some reporting details remain incomplete.**
   - The 3D common-panel ratio does not name its denominator: **`W512|lat32768|fs|BE|0.01|K1`**.
   - The historical **3D BE comparator error**, explicitly missing in round 2, remains absent from the grid comparison.
   - The H3 criterion at `report.md:125` remains Unicode/ASCII mathematical prose rather than LaTeX.

The remaining work is reporting repair; these checks found no need to rerun experiments.