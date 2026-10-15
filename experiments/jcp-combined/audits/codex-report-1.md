**Overall: NOT READY.** The generated numerical tables reproduce their source JSONs, but several claims and definitions contradict the design amendments or repeat findings already corrected in earlier audits.

Read-only verification reproduced `report.md` **byte-for-byte in memory**. All eight PNGs reproduced **pixel-for-pixel** from their corresponding JSON inputs. No files were created, modified, or appended; no GPU experiments were rerun.

| Requested item | Verdict | Finding |
|---|---|---|
| 1. Every number generated from JSON | **WRONG** literally; result tables **CORRECT** | Results are generated, but experimental metadata and numerical assertions remain hard-coded. |
| 2. Claims, verdicts and labels | **NEEDS-RESTATEMENT** | Main hypothesis results agree; several blanket statements and definitions do not. |
| 3. Retractions | **NEEDS-RESTATEMENT** | Named events are substantially recorded, but A10 qualifications and earlier scientific retractions regress. |
| 4. Report requirements | **NEEDS-RESTATEMENT** | Title, status, Mermaid and some LaTeX exist; glossary and mathematical presentation are incomplete or misleading. |
| 5. Plots | **WRONG** as publication-ready figures | Data reproduce exactly, but budget stacking violates A1.4, and labels need correction. |
| 6. Missing or overstated material | **NEEDS-RESTATEMENT** | Required uncertainty, interaction, timing and provenance disclosures are missing. |

1. **WRONG — the CN failure set is not identical across rules.**  
   The [summary and CN explanation](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/report.md:10) claim “the same at every rule” and infer rule independence. The report’s own W1024 table says **“no.”** `cnfail3d-c3descW1024.json` records seven common non-control failures, plus **case 46 at `gl24`**. The deliberately inadequate `lat256` control has distance failures on every case, and its W1024 adaptive comparator is not universally verified.

   This exact qualification was already required by `codex-a10-2.md` and confirmed by `codex-a10-3.md`. State the common non-control failure subset, the W1024 exception, and the control exclusion. The evidence supports a persistent solver problem; it does not support universal rule independence.

2. **WRONG — the reference labels contradict the data being printed.**  
   The [status line](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/report.md:3) says every accuracy number uses first-order references. The budget introduction repeats that assertion, and both budget plots label their reference-dependent components **PROVISIONAL**, although their plotted values come from **E0** JSONs.

   Use separate labels: historical first-order results are PROVISIONAL; E0 results have the conditional Richardson-uncertainty interpretation already stated correctly elsewhere. Also remove the copied **BE-matched \(R_S\)** statement from the 3D grid and budget descriptions: 3D uses only \(R_{ST}\), and its \(R_S\) step does not match the ROM. `codex-results3d-e0-1.md` explicitly identified this problem.

   Historical 3D H6 should retain **“reference-limited (descriptive)”**, not merely “PROVISIONAL, descriptive.” A13’s side-by-side continuity is also incomplete: historical 3D reference errors and budget components are not presented beside their E0 replacements.

3. **NEEDS-RESTATEMENT — “second order” is asserted beyond the established scope.**  
   [DESIGN A1.1](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/DESIGN.md:379) explicitly withdraws formal-order claims for new settings and fixed-sweep solvers, requiring **CN/CN-R/BDF2-family** terminology and measured accuracy/cost claims. The title, summary, section headings and `speedup3d.png` repeatedly restore the stronger wording.

   H4a’s failure also does **not** establish “no speed at equal accuracy” without its registered threshold. The diagnostic equal-accuracy ratios are approximately **0.848–0.962**: modest measured speed gains exist, but none meets **ratio ≤ 0.6**. Say that the **registered fast-equal-accuracy target is unavailable**.

4. **WRONG — the budget definition misdescribes the form ablation.**  
   The [budget introduction](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/report.md:192) and glossary describe every sensitivity as changing one ingredient directly from the deployed arm, including changing its “test space.” Actual `delta_form` is the **tight-solver LSPG-to-Galerkin comparison**:
   \[
   \delta_{\mathrm{form}}
   =\lVert u_{a[\sigma\to\mathrm{tight}]}
   -u_{a[\sigma\to\mathrm{tight},\,f\to\mathrm{GAL}]}\rVert/n_0.
   \]
   The number of tests and sine-test space remain unchanged. Both analyzers implement this correctly; the prose does not.

   Restore A1.1’s temporal-package explanation too: scheme, step, startup, LSPG weights and adaptive fraction can change together. The summary’s sensitivity qualification is helpful, but does not repair these definitions.

5. **WRONG — stacked budget bars and the unqualified Pythagorean identity repeat earlier rejected presentations.**  
   [A1.4](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/DESIGN.md:419) requires grouped bars/points. Both budget plots stack non-additive sensitivities and discrepancies. “Stacked for display only” acknowledges the problem without satisfying the amendment.

   The glossary states \(e_\star^2=e_P^2+e_{\mathrm{dyn}}^2\) exactly after the report defines these quantities using independent time maxima. The identity holds **at a common time**, not generally after separate maximization or cohort aggregation. This was already finding 11 of `codex-design-1.md`. Restore the per-time qualification and the saved orthogonal pair at \(t^\ast\).

6. **NEEDS-RESTATEMENT — required budget diagnostics and failure visibility are missing.**  
   The JSONs contain material that the generator omits:

   - `U_star`, unresolved/unavailable counts, within-scheme refinement, mixed differences, closure diagnostics, context sensitivity and common-time orthogonal pairs.
   - The restriction of general 2D converged-rule checks to **16/38 cases**, distinguished from full-population finalist checks.
   - Explicit B2/B3 outcomes and component worst-case values.

   This omission matters: **every 3D quadrature sensitivity is unresolved against \(3U_\star\)** on the budget population. Larger-step CN diagnostics also have severe solver-sensitive outliers hidden by the median table—for example, W1024’s `CN/0.025/K1` worst `delta_solve` is approximately **0.297**, while its displayed median is **6.10e-5**.

   M2_512’s withheld attribution is **correctly stated** as unverified convergence, not demonstrated inadequacy. Explain why its selected-arm finalist checks can nevertheless pass: those checks cover different trajectory requirements.

7. **NEEDS-RESTATEMENT — several important results are numerically correct but need tighter presentation.**  
   The JSONs support:

   - **H1:** pass / fail / does not pay off under E0.
   - **H2:** both parts pass; **H3:** cancellation not supported.
   - **H5:** **17.7× overall**, **13.6× for selected LSPG-CNR**, versus **3.89×** remeasured comparator.
   - **H4:** fail/pass throughout; no registered fast-equal-accuracy selections.
   - **H6:** the stated selected-configuration outcomes, including worsening worst error at W512→W768.
   - The six 2D fallback decisions and their cost ratios.

   However, the summary places H5 under “2D scheme” without identifying its **LSPG-CN** winner; the detailed paragraph says it “may be” LSPG-CN although it **is**. Retain the fixed mesh, development/validation cohort and calibrated-ladder scope, and explicitly identify E0 changes as **reference-dependent rematching**, not new timings.

   Distinguish H4b’s witness criterion from the separately selected accurate-equal-cost arm. The former permits ratio ≤1.1 at the base step; the latter uses ≤1 at any eligible step.

8. **NEEDS-RESTATEMENT — retraction coverage is substantial, but not complete in effect.**  
   The report accurately records **A11 escalation, A11.3’s gl112 OOM, A11.4 hash binding, A12’s relative control, A13 outcome reversals, A14 finalist failures, A15 selection retraction and GB10 corruption**. **A16** appears accurately in the failed-job ledger, although it should also appear in the dedicated corrections section.

   Nevertheless, the A10 failure-set qualification is contradicted, and A1’s withdrawals concerning formal order, stacked budgets and common-time identities have been lost. Consequently, [“Every WRONG / HOLD was fixed and re-audited before the next step”](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/report.md:332) is too sweeping. Distinguish operational release decisions from unresolved or recurring reporting findings.

9. **WRONG under the literal provenance requirement; otherwise formatting needs repair.**  
   [`make_report.py`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/make_report.py:268) hard-codes experimental quantities in prose, including cohort sizes, reference resolutions and the **141 GB** hardware assertion. Thus byte-identical generation does not establish that **every number** comes from JSON. No mismatched generated result-table value was found.

   The closing glossary needs corrections and additions: dimension-specific selection rules, historical versus E0 uncertainty, \(t^\ast\), guards a/b/c, FOM, LM, `fs`/`ad`, H4b witness, arm-key step units, timing statistics and percentage points. Mathematical expressions in prose should use LaTeX. The workflow diagram also stops at A1–A12 and omits the later rescoring/fallback/rerun stages.

   Add the historical frozen-selection citation requested by `codex-results2d-e0-1.md`, particularly `FROZEN-SELECTION-306e226a.json`. Timing reporting still omits required spread/outlier statistics and absolute/common-panel 3D costs.

The Pareto coordinates, budget values, J4 distances and speed-up coordinates are reproducible. `pareto2d_TX.png` correctly reproduces the historical first-order data, but the current `main()` no longer generates it. The blocking issues are the report’s scientific definitions, qualifications and presentation—not a discovered mismatch between its generated tables and the committed JSONs.