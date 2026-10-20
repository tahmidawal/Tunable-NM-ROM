**Verdict: ACCEPT-WITH-FIXES for E2 2D.** The numerical correction passes. Two report-generation/wording fixes remain; no new GPU runs are needed.

1. **CORRECT — registered dominance restored.**  
   `qstats.py` is byte-identical to qualify’s version. Both fronts use
   \[
   \tau_{XY}=\max\!\left(2\max(\bar\varepsilon_X,\bar\varepsilon_Y),\,0.02\max(e_X,e_Y)\right),
   \]
   with the registered 1.05 cost inequalities. Each configuration is tested against every original qualified candidate, separately for median and worst error.

2. **CORRECT — reference margins and candidate population.**  
   All 38 reference cases are eligible; their median uncertainty is **3.76151275454283×10⁻⁶**, matching every stored front point. The registered E2 universe contains 18 selected-rule slots per bank, with **15/14/14/13** qualified candidates for banks 0–3. Deployed configurations remain separate comparisons. Timing certification checks reveal no failing admitted candidate.

3. **CORRECT — corrected front findings.**  
   Independent scalar calculations reproduce every front state. **No R256 front row remains seed-dependent.** Selected-rule median-front membership is identical across all four banks; R384 CN-R worst-front membership still varies across seeds.

4. **CORRECT — full numerical reproduction.**  
   CPU replays from archived coefficients reproduce **all four per-bank analyses byte-for-byte**. The cross-seed roll-up also reproduces byte-for-byte: **140 seed-robust, 21 seed-dependent, 22 inconclusive**, across 183 rows.

5. **CORRECT — previously accepted evidence is unchanged.**  
   All non-front analysis values, non-front verdict rows, spread statistics and intervals equal their pre-correction versions. All **721 locally retained output files** pass their recorded checksums. Remote archive bytes were not inspected.

6. **CORRECT — selection and gates retained.**  
   All three ranks retain seed-dependent selected rules. G-E2.1 still misses deployed-rule recovery at incumbent R256 and remains non-blocking. G-E2.3 passes; no target-unverified selection qualifier applies.

7. **CORRECT — accuracy and deployed-rule conclusions retained.**  
   The six selected-rule H-acc-N cells pass; twelve rank transitions improve; six range summaries are monotone. Deployed R128 fails qualification on every retrained bank, and deployed R384 has the previously reported failures. These do not invalidate selected-rule accuracy findings.

8. **CORRECT — data-hash qualification repaired.**  
   The report distinguishes identical generation settings from byte-identical arrays and identifies seed 2’s differing hash. It makes no unsupported claim about the magnitude or cause of that difference.

9. **CORRECT — regenerated report content and retraction.**  
   The E2 report text and E2 export payload exactly match regeneration. The report carries the three-seed, validation-population, conditional-reference and log-mean-interval qualifications. It correctly withdraws the old counts and R256 front-dependence claim.

10. **NEEDS-FIX — two reporting details remain.**  
    The empirical hash grouping and deployed-rule failure seed lists are still literal prose in [make_report.py:216](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/make_report.py:216) and [make_report.py:445](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/make_report.py:445). They are accurate today, but do not satisfy **“every number generated”**; derive them from the evidence files. Also replace **“opposite definitive labels”** with **“different definitive labels”**, matching A2.7 and the implemented categorical classifier.

I made no file changes, wrote no scratch files, and ran no cluster or SSH commands. E2 3D was excluded.