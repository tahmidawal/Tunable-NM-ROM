**Verdict: numerical aggregates reproduce; the final report still overstates mesh invariance and understates the failed head confirmation.** Read-only, offline audit; no files changed.

**1. Regeneration and numerical provenance**

- **PASS:** in-memory regeneration matches the report exactly: **146,767 characters**.
- **FAIL, literally, “every number comes from data”:** [make_report.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-study/reports/make_report.py:553) hard-codes “18 major issues,” “about ten times,” timing population sizes, model dimensions, thresholds and reference sizes. Configuration constants are distinguishable from measured results, but the blanket claim is false.
- **Actual hand-typed error:** the earlier audits contain **17 MAJOR findings**, not 18: eight in audit 1, nine in audit 2.

Independent recomputation from test `result.json` files, supplemented by archived per-state NPZs for rho/iv:

| Check | Comparisons | Mismatches |
|---|---:|---:|
| Rollout error maxima/medians | 1,776 | 0 |
| Dense-matched ST/S aggregates | 732 | 0 |
| B1 verdicts | 183 | 0 |
| B3 maxima/verdicts | 165 arms | 0 |
| Rho maximum/median/p95 | 2,556 | 0 |
| Worst-state descriptors/correlations/top-tail statistics | 540 | 0 |
| Timing medians | 275 | 0 |
| B4 solve times/ratios | 160 | 0 |
| Test per-case maximum spreads | 12 | 0 |

Examples reproduced:

- Accurate Gauss \(96^2\), test ST worst: **2.638 / 2.647 / 2.647%**.
- Accurate Gauss \(64^2\), \(1024^2\): continuum \(\rho_{\max}=0.0237794143\); case **test64:1**, step **1**, width **0.158978**, width correlation **0.257555**.
- Test `gref` maximum per-case ST spread: **0.009031 / 0.011475 / 4.715496 pp**, accurate/fast/head.

These are aggregate recomputations; I did not repeat the expensive independent bank/gradient evaluation or a full field-level audit.

**2. Freeze chronology**

**PASS, within the local evidence.**

- All three test `COMMIT.txt`, `PROVENANCE.json`, archived provenance and result commits pin **`9468164336ff1bc6bb914757507596ed48597849`**, the freeze-manifest commit, dated **2026-10-02 06:00:56 EDT**.
- All **48 staged-file provenance hashes** match files at that commit.
- Frozen selection and historical development-summary hashes match the manifest. The current selection also retains the frozen hash.
- A4 explicitly followed test submission; the report now discloses that chronology correctly.
- Selection reads development summaries and requires `role == 'dev'`. No test-driven selection change was found.

This supports “nothing in this lane was selected from test64.” It does **not** make historically reused test64 a newly unseen cohort or independently establish remote submission times.

**3. “Answers in brief” verdicts**

| Claim | Verdict | Evidence / required correction |
|---|---|---|
| Resolved off-mesh solutions are mesh-invariant on fixed cases | **WRONG as stated** | Approximately true for linear rungs, false for head. Test `gref` worst ST over all 64 cases is **4.916 / 8.794 / 9.377%**, B2 ratio **1.9074**, versus the **1.02** bar. Development already has a **1.221 pp** head per-case spread; test reaches **4.715 pp**. |
| Dense improves with mesh | **NEEDS-RESTATEMENT** | Supported on the common six cases. The displayed 64/64/6 head maxima instead read **6.588 / 8.843 / 1.902%**. State the matched population when asserting improvement. |
| Off-mesh is “more accurate than dense” at coarse meshes | **NEEDS-RESTATEMENT** | Supported for `gref` and appropriately resolved rules using the stated cohort-worst metrics. It is neither every rule nor a case-by-case dominance result. Under-resolved Gauss and other exceptions remain. |
| At \(4096^2\), the two “agree against S” | **NEEDS-RESTATEMENT** | Their **worst S-error magnitudes are close**; that alone does not establish rollout agreement. Matched test `gref` versus dense S errors are **0.098964 vs 0.097133%**, **0.707182 vs 0.706347%**, **1.354805 vs 1.347451%**. `gref` is slightly worse against S in all three settings. |
| B1 fails favourably for most coarse-mesh off-mesh arms | **NEEDS-RESTATEMENT** | Appropriate for resolved linear arms; not uniform across settings. Test head Gauss32 **passes B1 at 1024² and 4096²**, while failing B3. |
| Matched-\(m\) rho comparison | **CORRECT numbers; NEEDS-RESTATEMENT interpretation** | Lattice/EQ errors use the mesh target; Gauss/Fibonacci use the continuum target. These measure different discrepancies. Also, 3969/4096/4181 is approximately matched \(m\), not exactly matched. |
| Our bank needs more points than Hari’s | **CORRECT qualitatively** | The ladders support slower convergence. They compare different banks/test spaces; this is not a controlled isolation of the cause. |
| Solve cost is flat; full-query cost is not | **CORRECT, scoped** | All **40** measured test B4 entries pass, ratios **0.972–1.040**. This is the operational **median(query) − median(decode)** estimate, not a separately timed solver. Cost generally increases with \(m\) within a family; no universal scaling law follows. |
| Worst off-mesh states lean early and wide | **NEEDS-RESTATEMENT** | Gauss/Fibonacci case-maximum width correlations generally support “wider.” Argmax timing and tail timing differ: test accurate Gauss64 at 1024² peaks at **step 1**, but only **15.625%** of its top 1% states occur by step 5; their median step is **42.5**. Sobol supplies further exceptions. |
| Head spread “comes from the nonlinear head solve … not from quadrature” | **NEEDS-RESTATEMENT** | Shared sensitivity across arms supports a common solver/model explanation, but does not isolate it or exclude quadrature interactions. The report itself says the cause was not isolated. |
| Frozen recommendations confirm on test | **NOT ESTABLISHED; material omitted failure** | Accurate Gauss96 and fast Fibonacci1597 retain B1′/B3/B5 success. **Head Gauss32 fails B3 at every test mesh:** **0.585 / 1.105 / 1.258%**, against **0.5%**. Its favourable development result did not confirm. |

The test results therefore support the principal **linear-rung** findings while contradicting a blanket mesh-invariance conclusion and failing to confirm the frozen head choice.

**4. “What went wrong” verdicts, in bullet order**

1. **WRONG count / NEEDS-RESTATEMENT chronology:** three blockers is correct; **18 majors should be 17**. “Before submission” should specify **ROM submission**: A1 says reference jobs were already submitted.
2. **NEEDS-EVIDENCE:** I could not independently substantiate the specific timing-accumulator exception and calibration kill from the inspected local records. This is not evidence that they did not happen.
3. **CORRECT with scope:** B1/A3/A4/A5 chronology is disclosed correctly; favourable-direction failure applies to resolved rules.
4. **CORRECT target; NEEDS-RESTATEMENT point requirement:** G6 supports Gauss640 certification. “Needs about ten times” lacks a defined common accuracy threshold. DESIGN already reports Gauss512 agreement of **\(2.4\times10^{-6}\)** against Gauss640; 640 is not established as the minimum.
5. **CORRECT:** restricted reference scoring and limited full-field audit coverage are disclosed.
6. **WRONG blanket assurance:** direct B1 comparisons are matched, but **D4 dense B2 ratios still mix 38/38/6 cases**. The earlier audit’s population-mixing defect remains.
7. **CORRECT:** timing population and subtraction order match raw invocations.
8. **CORRECT:** GPU labels and within-job timing comparisons agree with records.
9. **CORRECT within local provenance:** historical test reuse is disclosed; no lane-specific test-driven selection found.

**5. Remaining report defects**

- **Major:** no T4/B2 test table. Add the head failures prominently rather than referring test readers to development B2.
- **Major:** the frozen head rule is still listed without its failed test-confirmation verdict.
- **Major:** D4’s dense ratios remain invalid as matched-cohort mesh comparisons.
- **Labels:** `test641` ambiguously concatenates cohort and index; use `test64:1`. Test prose points to **D5/D6** rather than the relevant test sections.
- **Units:** D4 worst-ST columns omit `%`. Distinguish percent error from percentage-point spread consistently.
- **Method/glossary:** “linear coefficients are solved directly” can suggest a closed-form time-step solve; the coefficients undergo nonlinear LM stepping. Only initialization is closed-form.
- **Notation gaps:** define \(L,N,p,\psi,\Phi,G',S\); \(S\) is overloaded between preconditioner and reference. Identify “Hari” and “paper eq. 13” with local source references.
- **Tail statistics:** give top-1% counts—**19 development states, 32 test states**—and preserve half-step medians where useful.
- **Formatting:** B1’s expression splits LaTeX around `pp`; use one expression. Mermaid source uses the requested format and role colours; rendering was not tested.