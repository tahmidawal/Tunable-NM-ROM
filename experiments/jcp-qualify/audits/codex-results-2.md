**Overall: RESULTS-CONFIRMED. Job audit: SUBMIT-q3d3.** No computational must-fix found. Reporting corrections below remain required.

| Item | Verdict | Evidence |
|---|---|---|
| Raw timing, identities, drift | **CORRECT** | Recomputed every ratio and validity flag: **2016/2016 split**, **2012/2016 fused** valid; all 112 subjects usable. **684 scoring hashes** reconstruct exactly; no recorded identity failures. Costs `c` and `c_ev` match exactly. |
| Timing-job Q1/Q2 | **CORRECT** | All **112 pass** on dev6. Independent replay-distance discrepancy ≤ **1.11e-16**; independently recomputed rho discrepancy ≤ **1.81e-14**. |
| Candidates and unqualified panel | **CORRECT** | At 256²/1024²/4096²: **32/33/33 candidates**, **6/4/4 excluded**. Both numerical labels, usable timing and all 38 eligible cases enforced. |
| Fallback handling | **CORRECT** | 256² R384 CN-R 2.5Δt₀ original fails accuracy qualification despite passing timing qualification; fallback passes both and enters the candidate set. Separate costs **0.467956 / 0.638755** preserved. |
| Dominance fronts | **CORRECT** | Every membership matches independent all-pairs DESIGN §6 recomputation. Primary median/worst sizes: **7/7, 7/7, 7/8**; secondary: **7/8, 7/7, 7/7**. |
| Roles, §4.4 | **CORRECT** | All **98** labels match: **5 both roles**, **25 accurate-only**, **68 no role**. Statistic-defining cases supply the reference margins. |
| Timing controls | **CORRECT** | Split/fused fields and iterations agree. Doubled-work medians **1.99883 / 1.99360 / 2.00966**. Smoke identity injection invalidates all four split and four fused triplets of its subject. |
| 3D round 2 | **CORRECT** | **19 fail, one passes:** `W256|gl24|fs|CN|0.01|K1`. Successor distance **1.51462e-4**, finer distance **7.15063e-6**, rho **0.00449822**, reproduction **1.24929e-15**. Populations, ladders and offline agreement match. |
| 3D front/open candidates | **CORRECT** | The sole pass occupies both qualified-subset fronts; **365 open candidates** recomputed exactly. No completeness claim justified. |
| Staged q3d3 | **CORRECT** | Exact expected **18-member** round-next config. All **142 staged repository files** match commit `58de6c0637c5f926e46918c352e6ef499956287b`; manifest passes. Projection **1.04613 h**, limit **1.75 h**; spent plus reservation **5.7077 < 35 GPU-h**, no pending core reservation. |

**Ranked reporting must-fixes:**

1. **NEEDS-FIX — combined:** explicitly correct the qualification scope of R128/R256 CN-R 2Δt₀ fast arms: they fail this lane’s every-case Q1. Preserve historical measurements.
2. **NEEDS-FIX — speed3d:** both H2 winners fail Q2 here. Historical speed ratios remain valid; Q1+Q2-qualified wording does not.
3. **CORRECT — P1:** no numerical retraction identified. Retain its 16-case 4096² population and frozen-mode qualifications.

CPU-only, read-only; no files or scratch written, no cluster activity or submission.