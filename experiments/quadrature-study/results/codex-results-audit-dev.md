**Verdict: the checked arithmetic reproduces; several conclusions overstate the evidence.** Read-only, offline audit completed. No files were changed.

**1. Numerical recomputation — PASS**

From archived JSONs and per-state NPZs, I checked:

| Check | Comparisons | Mismatches |
|---|---:|---:|
| Arm error maxima/medians, including cohort subsets | 5,294 | 0 |
| B1 / B3 verdicts | 183 / 170 | 0 |
| FOM reference-error aggregates | 36 | 0 |
| \(\rho\) maximum/median/p95 | 2,556 | 0 |
| Timing medians | 275 | 0 |
| B2 / B4 ratios | 61 / 40 | 0 |

Additionally, 153 error recomputations from saved restricted fields differed by at most \(4.51\times10^{-17}\). Sample report entries reproduce:

| Accurate Gauss \(96^2\) | ST worst % | S worst % | sg worst % | vs-dense % | vs-gref % |
|---|---:|---:|---:|---:|---:|
| \(256^2\) | 2.747376 | 1.078286 | 4.336480 | 4.257946 | 0.029934 |
| \(1024^2\) | 2.746514 | 1.063627 | 1.523983 | 1.155627 | 0.029348 |
| \(4096^2\) | 2.746641 | 1.061992 | 1.094078 | 0.211837 | 0.029316 |

The report also matches its current generator byte-for-byte, regenerated **in memory without writing**.

Selections independently reproduce:

| Setting | Registered | Post-hoc B1′ |
|---|---|---|
| acc | none | Gauss \(96^2\) |
| fast | none | Fibonacci 1597 |
| head | none | Gauss \(32^2\) |

**2. Registration and chronology — PASS WITH QUALIFICATIONS**

- **B1/B3/B5 and selection:** applied as written. B1 uses dense-matched cases and \(\max(0.0002,0.02E_{\rm dense})\) in fractional units. B3 uses the correct target and setting-specific threshold. Selection correctly requires B1/B3/B5, not B2/B4.
- **B2:** valid for the off-mesh arms’ common 38-case population. **Dense’s displayed ratios mix 38 cases at the smaller meshes with six at \(4096^2\)**; these are not matched-cohort mesh comparisons.
- **B4:** correctly uses the same H200 allocation and **median(query) − median(decode)**.
- **G-gates:** no observed violation. Logs show GPU/f64/highest; parity discrepancies are at most \(1.34\times10^{-13}\); truth residuals stay below \(10^{-6}\). All 76 refined references meet \(2\times10^{-11}\). G6 checks both populations; the additional rollout distances are \(1.54\times10^{-11}\)–\(1.87\times10^{-10}\). Controls fire throughout.
- **Audit limitation:** I recomputed error-field samples and per-state \(\rho\) statistics, but did not repeat the expensive independent bank/gradient evaluation underlying G8’s recorded \(\rho\) audit.

Local git chronology, October 2, EDT:

| Event | Commit | Time |
|---|---|---|
| Extra post-hoc descriptives | `73da7d329` | 05:41:19 |
| A3 / B1′ selection | `21ea03111` | 06:00:44 |
| Freeze manifest | `946816433` | 06:00:56 |
| A4 / regenerated summaries | `f01cd417f` | 06:08:04 |

All three test staging records pin `946816433`. A4 explicitly records submission of jobs 4739520–4739522 **before A4**. Therefore, “all amendments preceded test submission” is false.

The frozen selection hash matches. All three historical summary hashes match the manifest. Comparing historical/current summaries shows **only** added `worst_S`, `dense_worst_S`, and `worst_same_grid` fields; selection is unchanged.

No local test `result.json` exists. The records support pre-submission freezing of B1′ and the extra descriptives; they do **not independently establish the cluster’s exact submission/start times or absence of remote test output when A4 was written**.

**3. Requested conclusions**

| Item | Verdict | Evidence / correction |
|---|---|---|
| **D1** | **WRONG** | Accurate Gauss \(32^2\) has B2 ratio **1.0214**, failing 1.02 and contradicting “every off-mesh arm ≤1.007.” Dense remains non-invariant on matched dev6: ratios **2.1683 / 1.4909 / 1.3863** for acc/fast/head. |
| **D2** | **NEEDS-RESTATEMENT** | True for `gref` and suitable resolved rules at the smaller meshes, not all off-mesh rules. Accurate Gauss \(32^2\) is worse; at \(1024^2\), accurate Sobol and fast Gauss \(48^2\) are worse against S. At \(4096^2\), matched dev6 `gref` is **slightly worse against S**: acc **0.240313 vs 0.238564%**, fast **1.907277 vs 1.898790%**, head **2.449117 vs 2.428859%**. It is better against ST. Describe the upwind explanation as supported interpretation, not a separately isolated causal result. |
| **D3** | **WRONG as written** | G6 worst gaps are **\(6.1\times10^{-9}\)–\(4.3\times10^{-8}\)**, not \(10^{-10}\); the latter scale describes rollout agreement. \(640^2\) is a certified target, not a demonstrated minimum requirement: Gauss \(512^2\)’s point-form worst gaps against it are already approximately \(2.1\times10^{-7}\)–\(1.0\times10^{-6}\). The Hari-size observation is correct: accurate Gauss \(64^2\)/Fibonacci 4181/6765 yield roughly **0.011–0.033** continuum \(\rho_{\max}\). |
| **D4** | **CORRECT, narrowly** | B1 is the decisive obstacle to obtaining any registered recommendation: changing only B1 to B1′ produces the three stated choices. It is **not** the sole failure of every candidate; several also fail B3/B5. |
| **D5** | **WRONG as written** | All **40 measured B4 entries** pass, with solve-time ratios **0.93717–1.06004**. Full-query cost is not equally flat: `lat64` ratios are **1.2781 / 1.2478 / 1.4782** for acc/fast/head. The panel excludes several rule families. Cost generally increases with \(m\), but these data do not establish universal proportional scaling. |
| **D6** | **NEEDS-RESTATEMENT** | Accurate `lat64` B3 errors are **0.13336 / 0.26733 / 0.01341%**: fail/fail/pass. Its continuum \(\rho_{\max}\) is **0.13868 / 0.07092 / 0.05722**: above 0.116 **only at \(256^2\)**. Head EQ B3 errors are **0.64591 / 0.80790 / 0.05743%**: fail/fail/pass against 0.5%. The final passes cover only six dense cases. |
| **D7** | **NEEDS-RESTATEMENT** | The displayed Gauss/Fibonacci rules generally have positive width correlations and frequently peak at \(k=1\), especially fast/head. Exceptions matter: accurate Fibonacci 17711 peaks at **\(k=32\)–33**, accurate Gauss64 at \(256^2\) peaks at **50**, and Sobol has negative width correlations. `lat64` is not uniformly narrow/mid-time: accurate \(256^2\) and fast \(1024^2/4096^2\) peak at **\(k=1\), widths 0.182**. Argmax timing and the top-1% timing distribution must remain distinct. |
| **D8** | **CORRECT** | Both controls fail B3 and B5 in all nine mesh/setting combinations. Even the smallest control continuum \(\rho_{\max}\) is **0.3342**, above 0.116. |

**4. Draft-report defects**

In the [draft report](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-study/reports/2026-10-01-burgers2d-offmesh-quadrature.md):

- **Major — cohort mixing:** D4’s dense ratios compare 38/38/6 cases. D3 also mixes six-case dense/G6-check rows with 38-case ordinary rows. Add explicit row/metric case counts; B1’s matched scope is otherwise easy to misread.
- **Major — “equal accuracy”:** D5 compares B3 errors against **different targets and populations**: off-mesh versus `gref` over 38 cases, `lat64` versus dense over six. Passing their respective tolerances does not establish equal accuracy.
- **Missing disclosure:** the glossary calls test64 “held-out” without repeating the design’s **historically reused cohort** caveat.
- **Missing chronology:** B1′ and extra displayed descriptives are labelled appropriately, but A4’s post-submission regeneration and historical manifest hashes are not explained in the report.
- **Missing measurement caveats:** state the restricted \(257^2\) reference scoring, limited full-field audit coverage, and six-case/three-repetition timing population beside the relevant tables.
- **Misleading method diagram:** “Gauss-48 initial fit” describes the linear rungs; the head uses an LM initialization.
- **Glossary/labels:** define \(m\) versus \(M\), pp, Spearman, and top-1% population size; clarify that “LM its/query” is a **case mean**, while non-stationary exits are aggregate counts. “Sobol/Halton error about \(1/m\)” is not a universal guarantee.

Finally, [DESIGN A3](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-study/experiments/quadrature-study/DESIGN.md:315) itself overstates the explanation: “every off-mesh arm … because … smaller” is contradicted by accurate Gauss \(32^2\) and the controls. The frozen selection remains numerically correct.