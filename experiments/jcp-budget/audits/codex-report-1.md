Audited HEAD `244223039`. No files changed, no SSH, no submissions. Regenerated `report.md` and `runs/EXPORT.json` in memory: **both byte-identical**. Inspected all five PNGs.

1. **WRONG — numerical provenance and correction prose.**

   The generated tables trace to the selected JSONs and ledger, but “every number below is read from [analysis JSONs]” is false: numerical correction claims are literal strings in [report_inputs.json](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/report_inputs.json:31).

   Checking the twelve correction bullets in order:
   - **1–3, 5, 7–8, 11–12: CORRECT** against the cited amendments and saved analysis records.
   - **4 and 9: NEEDS-RESTATEMENT.** The approximately 27× and 22× measurements concern the **4096² level**, not the entire reference triple: DESIGN A9 gives 765–769 s versus 28 s; A14 gives 1251 s versus 57 s.
   - **6: NEEDS-RESTATEMENT.** The examples are approximately consistent with certificates and ledger, but these are approximate projection/elapsed ratios, not a generated universal “3–7×” result.
   - **10: NEEDEDS-RESTATEMENT.** “Terminated under memory pressure” asserts an unverified cause. A12.2 records termination without an OOM record and high shared-machine load.

   EXPORT reproduces correctly and now includes T=1 and front-order records; the earlier results audit’s statement that these are absent applies to its older HEAD.

2. **NEEDS-RESTATEMENT — evidence strength and scope.**

   **Correct:** conditional Richardson language; inline W3 provisional/no-verdict labels; explicit 3D Q2 failures and qualification failures; GAL-CN diagnostic designation; B8 3D deferral; disclosure of failed references and unmatched reference levels.

   **Overbroad:** the bold lane statement says “qualified 2D front,” although its predicate checks only LSPG members. The adjacent table contains `R384 GAL CN 1` with **mixed** W1 sampling. The statement must explicitly say **LSPG**, retaining the 1025²-route restriction. See [report.md:315](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/report.md:315).

   **Plot qualification mismatch:** B7 error and floor plots select every row containing the arrays, without eligibility filtering. At 1024² that includes **38 cases**, including non-rankable `val32|10` and `val32|31`; the growth table uses **36**. The curves are not labelled as descriptive results containing excluded references.

   DESIGN §5.3 also requires current-time-normalised error beside fixed-initial-time error. `e_rel_t` exists in the JSONs but is absent from the report, leaving the decay-related qualification incomplete.

3. **WRONG — report conventions and presentation.**

   The level-1 title, status paragraph and final glossary exist. The plots have readable text and their numerical series follow the generator, subject to the population mismatch above.

   Blocking presentation defects:
   - All three B8 headers contain unescaped pipes in `max |E_rom/E_ref - 1|`. The header has more cells than its separator and data rows, breaking GFM table rendering. See [report.md:284](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/report.md:284).
   - The glossary omits `complete/eligible`, ingredient-letter mappings, sampling/pair classifications, `n`, energy and overshoot normalisation, and configuration tokens such as BE, CNR, BDF2, gl, lat, fs and qn.
   - `U*` is described as though effects below it are unresolved; the implemented trajectory threshold is **3U***, with additional reference and tie criteria. “Dominant” also omits resolution/tie handling.
   - `s` and near-wall density are defined, but the latter’s displayed median pools cases and output times without saying so.
   - Mathematical prose includes ASCII expressions such as `||u_all - u_0||`; mesh headings and glossary formulas also bypass the required LaTeX convention.
   - The 3D budget plot lacks a vertical-axis definition and labels its form effect “form (+ stopping rule)” instead of identifying the **tight-solver-end** comparison.

4. **CORRECT — prohibited comparison scope.**

   No heat POD-128 comparison appears. No scientific result or performance conclusion about another lane is asserted; other-lane mentions concern inherited reference labels and archive provenance.

**NOT-READY**

1. Generate numerical correction claims from structured evidence; restrict the 27×/22× claims to the measured level, qualify the projection ratios, and remove the unsupported memory-pressure attribution.
2. Explicitly scope every lane adequacy statement—including EXPORT—to **LSPG members**, W1 and the 1025² comparison route.
3. Filter B7 plots to the stated eligible population, or clearly label excluded-reference curves as descriptive; identify horizon-unqualified members and include the registered current-time-normalised error comparison.
4. Repair the B8 header escaping so all three tables render correctly.
5. Complete and correct the glossary, including eligibility, aggregation, normalisation, configuration tokens and resolution/tie rules; convert mathematical prose to LaTeX.
6. Give the 3D plot its normalised vertical-axis definition and correct tight-end form-effect label.