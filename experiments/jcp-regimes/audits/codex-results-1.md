**RESULTS-WRONG — one substantive rule violation affects the GAL diagnostic on C2.** The sampled primary LSPG results are confirmed.

Audited commit `f1af0ba81`. No files changed, no jobs submitted, and no archives downloaded. All calculations used CPU-only Python.

1. **Reference outcomes, eligibility, uncertainty and coverage — CORRECT within the slim-store evidence.**

   I independently recomputed order, Richardson uncertainty, acceptance predicates and precedence from all 158 reference JSON records. E0’s dimension status and designated cross-checks also permit all 38 dev6/val32 references.

   | Cohort | Eligible / sampled | Exclusions | Recorded ε_ref range, independently recomputed |
   |---|---:|---|---:|
   | dev6 | 6/6 | None | 1.298e−6–7.308e−6 |
   | val32 | 32/32 | None | 8.670e−7–9.520e−6 |
   | N0 | 14/16 | c4, c15: order-fail | 4.788e−7–8.341e−6 |
   | N0s | 8/8 | None | 4.075e−6–1.542e−5 |
   | A+ | 14/16 | c4, c8: unresolved-floor | 1.976e−6–5.746e−6 |
   | A− | 16/16 | None | 7.972e−7–6.831e−6 |
   | C1 | 16/16 | None | 5.171e−6–4.322e−5 |
   | C2 | 16/16 | None | 1.308e−5–7.462e−5 |
   | B1 | 16/16 | None | 5.937e−7–6.269e−6 |
   | B2 | 15/16 | c7: order-fail | 8.666e−7–8.846e−6 |

   Ranges include ineligible records. Every stored per-case ε_ref matches `analysis.json`.

   A+ c4/c8 first failed order, then failed the floor after refinement. A3.2 prohibits a tight refined triple, so both are correctly excluded despite the unused allowance of one. Refinement, tight-use and allowance-unresolved counts match [report.md:178](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/report.md:178).

   **Limit:** earlier reference fields and residual histories are archived. Thus direction-consistency and floor-input diagnostics were checked from recorded JSON values, not reconstructed from those unavailable arrays.

2. **ROM errors for two configurations across every regime — CORRECT.**

   Independently decoded all 158 cases for each configuration using `npcheck.decode257`, then scored against the finest local reference fields with the reference IC norm. All **316 errors** agree with `analysis.json` within relative `1e−11` / absolute `1e−14`.

   | Regime | R128 LSPG-BE median error | R384 LSPG-CNR median error |
   |---|---:|---:|
   | N0 | 0.757867% | 0.118690% |
   | N0s | 4.222132% | 1.220121% |
   | A+ | 1.372354% | 0.245870% |
   | A− | 3.886165% | 0.882836% |
   | C1 | 9.278063% | 5.679664% |
   | C2 | 18.552333% | 16.601444% |
   | B1 | 1.153005% | 0.200173% |
   | B2 | 1.404208% | 0.244904% |

   These are eligible-case medians; qualification remains separate.

3. **Q1/Q2 for those two configurations — CORRECT.**

   Independently recomputed production–tight, tight–tighter, successor and check distances across every cohort. Both configurations pass Q1/Q2 outside C1/C2, including dev6, val32, N0 and N0s.

   | Configuration / regime | Q1 production–tight maximum | Q2 successor maximum | Q2 check maximum |
   |---|---:|---:|---:|
   | R128 BE / C1 | 1.69084e−4: pass | 1.07564e−3: fail | 4.64880e−4: fail |
   | R128 BE / C2 | 3.15394e−4: fail | 3.59163e−3: fail | 5.94693e−4: fail |
   | R384 CNR / C1 | 7.60721e−4: fail | 2.27861e−3: fail | 3.16587e−4: fail |
   | R384 CNR / C2 | 2.21503e−3: fail | 1.74930e−2: fail | 2.13949e−3: fail |

   Each displayed distance has a `2.5e−4` threshold. Tight–tighter checks pass; these failures are not caused by the ρ threshold.

   Example raw successor recomputation: **R384 CNR, C1 c0 = 0.000533979450673**, already above threshold. C2 c9 gives **0.0142554608134**.

   Independently recomputed 120 ρ values, covering one case in every cohort for both configurations; maximum discrepancies were `8.17e−16` and `1.21e−14`. Certificate/recomputed-certificate agreement passes. These findings match [report.md:119](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/report.md:119).

4. **Bootstrap, paired contrast, trends, separation, floors and PS — CORRECT for the tested sample.**

   Bootstrap was independently implemented with 10,000 resamples and seed 0, using freshly recomputed errors and the registered baseline ordering.

   | Configuration / cell | Ratio [95% interval] | Accuracy label |
   |---|---|---|
   | R128 BE / A+ | 1.37401 [0.94245, 2.10125] | INCONCLUSIVE |
   | R128 BE / A− | 3.89086 [1.74824, 6.71025] | INCONCLUSIVE |
   | R384 CNR / A− | 4.60467 [1.47765, 13.21967] | INCONCLUSIVE |
   | R384 CNR / C1 | 29.62380 [16.17253, 58.68639] | DEGRADES, **unqualified** |

   Additional independent checks:

   - **Paired C1/A−:** R128 BE `2.58016 [1.93869, 3.85173]`, INCONCLUSIVE; R384 CNR `4.69873 [3.42631, 12.00811]`, DEGRADES. Both Wilcoxon p-values are `0.000030517578125`; C1 qualification caveats apply.
   - **A− distance trend, R384 CNR:** Spearman `0.45`, permutation p `0.07829217`, slope `1.72045951`; the registered “grows” predicate is false.
   - **In-range consistency:** R128 BE `0.69190 [0.45650, 1.47717]`; R384 CNR `0.59983 [0.41151, 1.57551]`. Both intervals contain one.
   - **σ_a separation:** in-range bounds `[3.932447, 19.266043]`, q90 `14.839772`. Fractions inside: A+ `2/14`, A− `15/16`, B1 `15/16`, B2 `12/15`. A+ correctly receives the confounding label. C1 has exactly `75%` above q90 and median `33.922963 > 11.868158` for A−; C2 median `78.543667` exceeds C1.
   - **Projection floors:** independently used QR projection for R128 across every cohort, rather than the analysis’s SVD implementation. All medians/maxima match. Example medians: N0 `0.379714%`, C1 `8.149188%`, C2 `15.550999%`.
   - **PS:** independently reconstructed all seven cross-checks from local arrays; all PASS under the disclosed E0 A10 rule. Six would be UNRESOLVED under the old rule; C2 c9 passes both.

5. **Frozen decision rules — WRONG for GAL/C2 validity; otherwise the examined rules are correctly applied.**

   [DESIGN.md:197](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/DESIGN.md:197) explicitly includes **budget exits** among invalid production runs. However, [p1common.py:332](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/p1common.py:332) explicitly permits budget exits, and [rscore.py:272](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/rscore.py:272) excludes reason 3 but not reason 0.

   Raw `R_R384_GAL_CN_1.npz` contains:

   - **C2 c8:** seven budget exits, zero-based steps 43–49; error `2.64002298817`.
   - **C2 c13:** three budget exits, steps 47–49; error `2.19189747580`.

   Both are marked valid. Under §7.1, **GAL/C2 must be FAILS, with two invalid runs**, rather than NOT QUALIFIED. Excluding those runs leaves **14/16 used**, median **21.4361901%**, worst **66.4917028%**; the report currently gives approximately **22.1% / 264%**.

   A1.7 imports qualification semantics but does not explicitly retract §7.1’s budget-exit rule. This conflict cannot support a claim that every rule was applied as frozen.

   Other checks: primary-label count and diagnostic scope, coverage minima, bootstrap bands, worst-error flags, absolute-error flags and A3.2 exclusions agree. No individual-case configuration-ranking claim requiring the `2ε_ref` rule was identified.

6. **Report traceability — CORRECT mechanically; NEEDS-FIX substantively.**

   Rendering `make_report.py` entirely in memory reproduced **report.md and EXPORT.json byte-for-byte**. Ledger elapsed hours sum to **14.865**, exactly as reported. All nine R-file hashes match the passing full-field verification record.

   Traceability nevertheless propagates the GAL validity defect into [report.md:51](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/report.md:51), **:65, :79, :113 and :127**: label, coverage, error summaries, pooled trend population and Q1 failure detail. The report exposes the large error and qualification failure, but omits the registered invalid-run outcome.

**Numbered fix list**

1. Enforce §7.1’s budget-exit validity rule in scoring. If the permissive inherited rule is intended instead, disclose an explicit protocol deviation; do not describe it as unchanged frozen compliance.
2. Recompute GAL/C2 validity, exclusions, both baseline comparisons, error summaries and the pooled A−/C1/C2 trend. Under §7.1, the headline is **FAILS; two invalid runs; 14/16 used**.
3. Regenerate `analysis.json`, `report.md` and `EXPORT.json`, and disclose the correction. The confirmed primary LSPG findings need no numerical revision based on this audit.