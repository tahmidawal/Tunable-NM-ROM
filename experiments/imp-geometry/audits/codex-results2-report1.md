Read-only audit completed: no files changed, jobs submitted, LAB-LOG writes, or large-array copies.

**PART 1 — Results round 2**

- **CORRECT:** Revised `analysis/decide_s1.py` reproduces `runs/decision_S1.json` exactly in memory.
- **CORRECT:** H is **WITHDRAWN (measured loss)**; L remains **not decidable** with reference-ineligible targets. A10 is **not supported at S1; S2 untested**.
- **CORRECT:** Descriptive validation/test S values are retained for both geometries, including nulls. Missing cells/tests now produce “incomplete.”
- **CORRECT:** Report §4 discloses family-level-only test T-dbl flags and excludes failed test finalists; it does not claim all finalists passed.

**PART 2 — Report audit**

- **CORRECT — numerical consistency:** Independently recomputed all 480 raw timing ratios and all validation/test summaries, per-family values, winning keys and nulls.
- **CORRECT — provenance:** Decision inputs match corresponding `runs/cells/*` JSONs byte-for-byte. In-memory generation reproduces `report.md` and `EXPORT.json` exactly.
- **CORRECT — reported measurements:** S tables/ranges, reference counts, allowances, boundary diagnostics, job charges, total GPU-hours and smoke overrun match their JSON sources.
- **CORRECT — plot values:** `S_per_family.png` visually matches the validation worst-error series, logarithmic scale and threshold lines.
- **WRONG — “every number … from run JSONs”:** `make_report.py` hard-codes numerical protocol values, including `cap=62`, cohort/timing counts, diagnostic/win thresholds and smoke allowance. Measured results are sourced correctly, but the literal provenance claim fails. Read available values from records and distinguish sourced protocol constants.
- **NEEDS-RESTATEMENT — headline/status and inline qualification:** “final” and the blanket “bank is not faster” headline obscure L’s unqualified reference status. Label L’s table rows and plot panel **descriptive; reference-ineligible**, and distinguish completed measurements from qualified conclusions.
- **WRONG — WIN versus attribution:** Report §4 says a WIN requires box controls. A4.2 requires both wide levels to win; box controls establish **attribution**. Preserve the prescribed A10 readout wording, but correct its explanatory parenthesis.
- **NEEDS-RESTATEMENT — redesign/retraction:** §5 acknowledges diagnostic-guided development but omits A3.1’s explicit retraction: solver/ROM smoke results had already been inspected, so redesign was **not outcome-blind**.
- **CORRECT — timeout, smoke overrun and deferrals:** These are disclosed consistently with the records.
- **NEEDS-RESTATEMENT — reductions:** “Registered reductions were applied” is too vague; identify the reduced rank ladder, implementation-variant screening and SEM grading restrictions.
- **NEEDS-RESTATEMENT — glossary:** Define L/H, S1/S2, wide family, E0/corner, Q1/Q2, T-dbl, FD/RK4/SEM, CN-R/LSPG, Richardson estimates and max–min/finalists. Explain test statistics too; “eligible” also requires the allowance gate.

**RESULTS: CORRECT** — requested round-1 fixes verified.
**REPORT: NEEDS-FIX** — numerical-provenance claim; inline qualification/headline; WIN-versus-attribution wording; redesign retraction; reduction details; glossary.