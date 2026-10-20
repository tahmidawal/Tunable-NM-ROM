**NOT-CONFIRMED — the numerical core remains verified, but the new deliverables contain one incorrect statistic and incomplete diagnostics.**

I made no repository or cluster changes and submitted no jobs. The five downloaded arrays passed checksum verification. Scratch peaked at approximately 567 MB and was deleted. The pre-existing HEARTBEAT modification remains untouched.

**Round-1 required fixes**

| Required fix | Verdict | Evidence |
|---|---|---|
| Escape Markdown table delimiters | **CORRECT** | No inconsistent column counts remain. Report regeneration reproduced the file byte-for-byte with writes disabled. |
| Show timing usability and correct descriptive-table scope | **CORRECT** | Tables now distinguish completeness from usability; the three-mesh description correctly identifies seed-0 heads plus incumbent. |
| Distinguish qualification in error plot | **CORRECT** | Filled/open markers distinguish qualified/unqualified configurations. |
| Add registered pooled error series | **WRONG** | The marker uses the median of seed medians, rather than pooling seeds within each case first. |
| Add iteration and per-iteration cost tables | **CORRECT** | Reconstructed added statistics from timing invocation records at all three meshes. |
| Add bootstrapped cost fits and cost plot | **NEEDS-RESTATEMENT** | The 256²/1024² calculations reproduce, but the 4096² fit is silently omitted; fit interpretation overstates the evidence. The cost plot does not mark unusable timings. |
| Add seed-spread summaries and plot | **CORRECT** | Both schemes’ exported spreads reproduce; the BE table marks unqualified seeds. |
| Add initial-fit diagnostics | **NEEDS-RESTATEMENT** | Exit counts are correct, but initial-fit iteration and stationarity values remain absent. |
| Add training code-convergence gains | **CORRECT** | Every added gain matches training `result.json`. |
| Add brackets, pooled-model caveat, corrected glossary and replay explanation | **CORRECT** | These changes address the identified omissions. |
| Preserve principal conclusions | **CORRECT** | Every pre-existing top-level EXPORT value is unchanged. |

**Requested results checks**

1. **CORRECT — raw E0 errors and Q1/Q2.** I rescored all 38 cases in each requested configuration and recomputed qualification distances and quadrature diagnostics. Every E0 error matched EXPORT exactly; no qualification disagreements occurred.

   | Configuration | Median E0 error | Worst E0 error | Q1 | Q2 |
   |---|---:|---:|---|---|
   | H16s0 BE | 1.413847% | 5.839069% | pass | pass |
   | H32s1 BE | 1.203475% | 4.263265% | pass | pass |
   | L128 BE | 1.095338% | 4.843301% | pass | pass |
   | L384 CNR | 0.197875% | 1.779067% | pass | fail |
   | H4s0 BE | 6.105074% | 29.527335% | fail | pass |

   L384 CNR’s successor distance is **2.605212e-4**, exceeding **2.5e-4**. H4s0 fails anchor verification and the tight/tighter predicate. Certificate checks use saved certificates; I did not independently rebuild every step Jacobian.

2. **CORRECT — qualification pools.** Unchanged for both schemes: K=4 → none; K=8 → none; K=16 → seeds 0,1; K=32 → seed 1. Consequently, K=4/8/32 have fewer than two qualified seeds. No changed qualification logic or post-hoc relaxation was introduced.

3. **CORRECT — K=16 comparisons and newly displayed brackets.** Re-execution from derived records reproduced both schemes’ paired tests, cost bootstraps, Holm decisions and brackets exactly.

   BE remains **neither (trade-off)** at matched dimension. Certified-affordable rungs are **4–128**; possibly-affordable rungs additionally include **256**. The bracket is **128 below / 384 above**. CNR has certified/possibly-affordable rungs **4 and 128**, with **no rung above**. Both correctly report **“linear at no more cost is at least as accurate,” naming r=128**. CNR’s matched-dimension comparison remains unqualified.

4. **CORRECT — F5 inconclusive.** Recomputed outcomes agree. The frozen family is unchanged; unavailable components remain non-rejections. K=16 cannot establish superiority over every possibly-affordable rung, while unqualified sizes prevent the resolved negative family-wide reading. Exploratory wording remains appropriate under A4.1.

5. **CORRECT for existing timing verdicts; NEEDS-RESTATEMENT for additions.** H-cost-N, timing gates and trajectory-consistency records are unchanged from round 1. The corrected tables expose usability. The new descriptive iteration/cost numbers reproduce.

   However, `cost_vs_K.png` gives unusable timings the same marker treatment as usable timings and lacks a standalone descriptive/unqualified caveat. Those points must not appear to be qualified deployment comparisons.

6. **CORRECT — training and reproduction.** Added code-refit gains match source results. The report now correctly explains seed-0 K=16 as same-data, same-seed recipe replay. The unchanged Q-repro and A1.7 conclusions retain round-1 acceptance; no fresh checkpoint comparison was necessary for these presentation changes.

7. **CORRECT — 256² and cross-mesh panel changes.** Their underlying EXPORT values are exactly unchanged. Table escaping repairs presentation without altering results. Round-1 verification from derived/per-case records therefore carries forward; I did not download fresh 256² trajectories.

8. **WRONG / NEEDS-RESTATEMENT — remaining new defects.**

   - **Incorrect pooled marker.** [make_report.py:197](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/make_report.py:197) plots **1.433223%** for K=16 BE. DESIGN A1.6 requires the median across cases of the per-case seed-median errors: **1.440246%**. The plot disagrees with the registered statistic and EXPORT’s pooled comparison.
   - **Incomplete initial-fit summary.** [analyze.py:473](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/analyze.py:473) uses rollout `it_sum`. Its caption correctly calls this rollout iterations, but it does not supply the requested initial-fit diagnostics. For H16s0 BE, the displayed **82** is rollout iterations; actual median initial-fit iterations are **17**, with median recorded `ic_gn` **6.17709e-7**. Surface initial-fit iterations and stationarity, with accessible per-case records.
   - **Missing 4096² fit.** [analyze.py:432](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/analyze.py:432) requires all three seeds, although this mesh’s registered panel contains seed 0. Consequently EXPORT contains `fit: null` and the report says nothing. This is missing coverage, not demonstrated non-identifiability.
   - **Overstated fit interpretation.** [make_report.py:277](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/make_report.py:277) claims cost “is not a power law” and “is driven by LM iteration counts.” The actual finding is that the chosen constrained fits fail the inherited relative-RMS acceptance threshold: **0.235622** and **0.328512**. Report that limited finding; it does not establish causation.

**Ranked required fixes**

1. Correct the pooled plot statistic to per-case seed pooling.
2. Complete initial-fit iteration/stationarity reporting and explicitly resolve the missing 4096² cost-fit deliverable.
3. Replace the causal/power-law claims with the actual fit diagnostics; mark unusable timings in the cost plot.
4. Preserve the verified conclusions: **F5 inconclusive**, **K=16 BE matched-dimension trade-off**, and **r=128 at no more cost is at least as accurate**.

**Final verdict: NOT-CONFIRMED.**