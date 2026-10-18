**NOT-CONFIRMED for the complete registered results package.** The audited numerical results and principal decisions are correct. Missing registered deliverables and defects in the report prevent full acceptance.

I made no repository or cluster changes and submitted no jobs. Downloaded files were checksum-verified; scratch peaked at approximately 623 MB and was deleted.

1. **CORRECT — sampled raw E0 errors and Q1/Q2.**

   I rescored all 38 cases in each requested configuration and recomputed qualification distances and quadrature diagnostics from the archived arrays. Every E0 error matched EXPORT exactly; qualification recomputation found no disagreements.

   | Configuration | Median E0 error | Worst E0 error | Q1 | Q2 | Maximum successor distance |
   |---|---:|---:|---|---|---:|
   | H16s0 BE | 1.413847% | 5.839069% | pass | pass | 7.20374e-5 |
   | H32s1 BE | 1.203475% | 4.263265% | pass | pass | 1.56609e-4 |
   | L128 BE | 1.095338% | 4.843301% | pass | pass | 6.59242e-5 |
   | L384 CNR | 0.197875% | 1.779067% | pass | **fail** | **2.60521e-4** |
   | H4s0 BE | 6.105074% | 29.527335% | **fail** | pass | 1.68414e-4 |

   L384 CNR genuinely exceeds the 2.5e-4 Q2 bar. H4s0 BE fails anchor verification and the tight/tighter predicate despite passing production stationarity and Q2. Q1 certificate checks used the saved certificates; I did not independently rebuild every step Jacobian.

2. **CORRECT — qualification pools and “unqualified size.”**

   For both schemes, the 1024² pools are:

   | K | Qualified seeds |
   |---|---|
   | 4 | none |
   | 8 | none |
   | 16 | 0, 1 |
   | 32 | 1 |

   Thus every adjacent-K transition involves a size with fewer than two qualified seeds. “Unqualified size” and the inconclusive range verdict follow the registration.

   Rule-trail pass counts and first-passing selections agree with RULES. A2.5’s two-seed rule-selection threshold does **not** waive subsequent Q1/Q2 qualification. A9’s initial-fit tolerance amendment is disclosed as post-smoke, pre-production; I found no additional outcome-dependent relaxation.

3. **CORRECT — K=16 Q-LH-dim and Q-LH-cost.**

   Independent SciPy signed-rank/TOST calculations, 10,000-resample cost intervals, and the 32-component Holm procedure reproduce the exported decisions.

   For BE, pooled head median error is **1.440246%**. Against r=16, the head is more accurate, but costs **3.77667×**, CI **[3.60282, 3.89641]**: correctly **neither (trade-off)**.

   | Linear rung | Linear/head cost, median [95% CI] | Error winner | Holm rejects head-null |
   |---|---|---|---|
   | 4 | 0.20459 [0.19401, 0.22215] | head | yes |
   | 8 | 0.22173 [0.20606, 0.23867] | head | yes |
   | 16 | 0.26481 [0.25666, 0.27756] | head | yes |
   | 32 | 0.29457 [0.28344, 0.30899] | head | yes |
   | 64 | 0.31374 [0.30181, 0.34241] | head | yes |
   | 128 | 0.59100 [0.57731, 0.65051] | linear | no |
   | 256 | 1.05379 [1.00481, 1.12957] | linear | no |
   | 384 | 1.60845 [1.55275, 1.75932] | linear | no |

   Head-superiority p-values are 3.63798e-12 for r=4–32, 2.69832e-5 for r=64, and 1 for r≥128. Reverse-direction p-values for r≥128 are 3.63798e-12.

   Certified-affordable rungs are **4–128**; possibly-affordable rungs additionally include **256**, whose interval crosses 1.1. There is no equal-cost rung. The registered result correctly names **r=128: linear at no more cost is at least as accurate**.

   For CNR, every rung’s tests and bands also match. Only r=4 and r=128 are numerically qualified; only r=4 is Holm-rejected. The at-most-cost result again names r=128. Matched-dimension comparison is correctly **unqualified**, because L16 CNR fails Q1. L256 CNR’s equal-cost band cannot rescue its failed qualification.

4. **CORRECT — F5 is inconclusive.**

   The frozen family remains `{4,8,16,32}`. Unavailable sizes contribute p=1. K=16 does not beat every possibly-affordable rung, so there is no positive reading. The negative reading requires every family member to be qualified with a resolved negative result; K=4/8/32 prevent that.

   A4.1 correctly limits Holm to an exploratory decision rule. Neither “no head benefit across K” nor “linear dominates the entire family” follows.

5. **CORRECT numerically; NEEDS-RESTATEMENT in presentation — timing and H-cost-N.**

   Direct reconstruction from timing-job JSONs reproduced A-B-A ratios, per-case fused costs, and:

   - H16s0: **1.063593**, CI **[1.045102, 1.131014]**, pass.
   - Incumbent: **1.029107**, CI **[1.021626, 1.093765]**, pass.

   Seed-0 K=4/8/32 correctly receive no verdict. K=4 fails anchor verification; K=8 fails solver-distance/anchor and/or finer-rule convergence checks; K=32 fails solver-distance and/or quadrature checks, depending on mesh.

   Raw H16s0 timing qualification passes. Its recomputed timing/accuracy trajectory distance is **4.61e-14**, versus cached **5.76e-14**—roundoff-level variation, far below 2.5e-4.

   The descriptive table’s numbers are correct, but “all heads” is inaccurate: it includes only heads present at all three meshes, effectively seed 0 plus the incumbent. The main cost tables should show **usable**, alongside timing completeness.

6. **CORRECT — training, reproduction, and A1.7 probe.**

   All exported training means and acceptance gates match training `result.json`. The probe/full oracle ratio is **22.090726**; A1.7 correctly treats it as sensitivity evidence rather than a mandatory negative control.

   Seed-0 K=16 equality is plausible: the driver calls training with the same recipe/data/seed, and recorded progress reaches 200,000 steps. Direct weight comparison finds exact equality except one bias difference of **4.44e-16**. Oracle and ROM ratios differ from 1 only at roundoff.

   “Reproduces” is correct. Explain that this is same-seed recipe replay, not independent evidence across datasets; replace the categorical glossary claim that retrains “are not bitwise reproducible” with “are not guaranteed to be.”

7. **CORRECT from derived records — 256² and cross-mesh results.**

   Recomputed 256² decisions match: K=16 uses seeds **[0,2]**, K=32 **[1,2]**. The error-all38/cost-dev6 scope is correctly disclosed.

   All **42** cross-mesh summaries reproduce exactly from exported per-case records. H16s0 BE’s maximum change is **1.005043 percentage points**, at val32|20. Its small median change therefore does not establish uniformly mesh-independent accuracy.

   This panel was checked against derived/per-case records, not independently downloaded 256² trajectories.

8. **WRONG / NEEDS-RESTATEMENT — report completeness and presentation.**

   In-memory regeneration reproduced the current report byte-for-byte; I found no sampled numerical transcription discrepancy. However:

   - **Broken tables:** literal `|` in configuration names, case IDs, and `|change|` headings creates extra Markdown columns. The defect originates in [`make_report.py:43`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/make_report.py:43).
   - **Missing registered deliverables:** per-iteration cost, iterations per step, the bootstrapped \(t(K)=a+bK^p\) fit, seed-spread summaries, cost-versus-K plots, and seed-spread plots. These remain required by DESIGN §§2.4, 5.3, 10.
   - **Error-versus-K plot:** connects qualified and unqualified heads without distinguishing qualification; it also omits the registered pooled series.
   - **Incomplete reporting:** bracket results exist in EXPORT but are not displayed; initial-fit stationarity diagnostics and training code-convergence gains are not adequately summarized.
   - **Glossary errors:** Q1 permits ≤1% nonstationary steps, rather than requiring every step stationary. Q2 includes rho and finer-rule convergence. Q-LH-cost must distinguish certified-affordable from possibly-affordable rungs.
   - State explicitly that pooled errors/costs summarize selected seeds and need not describe one deployable model, as required by A2.3.

**Ranked required fixes**

1. Escape table delimiters and distinguish qualification/timing usability in tables and plots.
2. Complete the missing registered cost, iteration, seed-spread, and plotting deliverables; document any unidentifiable fit explicitly.
3. Add the pooled-model caveat, bracket/diagnostic summaries, and correct glossary definitions and descriptive-table scope.
4. Preserve the current principal conclusions: **F5 inconclusive**, K=16 BE **trade-off at matched dimension**, and **linear r=128 at no more cost is at least as accurate**.

**Final verdict: NOT-CONFIRMED — numerical core verified, registered results package incomplete.**