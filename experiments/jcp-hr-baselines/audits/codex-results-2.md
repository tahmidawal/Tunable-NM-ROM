**Overall: RESULTS-CONFIRMED.** The seven domination conclusions and two withheld cell conclusions reproduce independently. The headline needs the qualifications below; no numerical disagreement reverses it.

1. **CORRECT — provenance, eligibility and X errors.** Verified verdict, tuning, Stage-B and frozen-membership hashes, plus all 53 baseline coefficient-file hashes. Independently checked the 64 reference metadata files against the pinned manifest: the eligible set is exactly `val64|0–63`. Reference uncertainties range from `8.910676602e-6` to `1.158854232e-4`. Every X median and maximum reproduces from the comparator export.

2. **CORRECT — Q1 and replay.** Reaggregating **6,890 production/anchor certificates** gives **20 Q1 passes and 33 failures**. All failures exceed the production–tight distance threshold; production has zero nonstationary steps. Largest passing distance: `2.492920958e-4`; largest tight–tighter distance: `3.365268651e-8`.

   Independently recomputed distances from saved coefficients using POD orthonormality and reference initial norms: maximum discrepancies from recorded lattice distances were `6.02e-17` and `2.03e-17`, with no gate disagreement. This uses the recorded orthonormality certificate (`1.51e-14`); I did not reload the full basis.

   All 53 coefficient replays against tuning reproduce, maximum discrepancy `1.708669714e-15`. All nine G-ours gates pass in both jobs; Stage-B maximum relative error discrepancy is `1.558983084e-14`.

3. **CORRECT — timing contract.** Independently checked **180 pairs, 3,246 attempted triples, six rejected attempts and 3,240 accepted triples**. Retry inventories pass; maximum attempted drift is `0.348241273`. Every recorded timed-output discrepancy is zero.

   Measurements share job **5932490**, H200 UUID `GPU-64b76fb8-65c0-ee22-c559-6d9549c2d207`. The log confirms GPU execution and successful completion. The staging script enables f64/highest precision; timing code burns before each synchronized call. No timing-contract artefact was found that changes a conclusion.

4. **CORRECT — X-specific fronts, tie-breaks and roles.** All fronts and picks reproduce. Here **E512** means `ecsw|512|k8|BDF2|0.016666667|common|central`; **Q1024** means Q-DEIM r1024/p4096, CN/common; **N1024** means unreduced r1024, CN/common.

   | X | Batch | Front size | B_acc: T [95% interval] | B_cost |
   |---|---:|---:|---|---|
   | M2_512 gl32 FS BE .01 | 1 | 10 | E512: .254553 [.240267, .288249] | E512; dominates |
   | W256 gl24 FS BE .01 | 1 | 11 | E512: .795892 [.741348, .927494] | E512; dominates |
   | W512 lat32768 FS BE .01 | 1 | 11 | E512: .220970 [.208737, .252410] | Q1024 .025; unresolved |
   | W768 lat32768 FS BE .01 | 1 | 11 | E512: .098762 [.094538, .111811] | Q1024 .01; dominates |
   | W1024 lat65536 FS BE .01 | 1 | 11 | E512: .029521 [.028178, .032838] | N1024 .0125; dominates |
   | W256 gl24 FS CN .01 | 2 | 11 | E512: .789849 [.739265, .918697] | E512; dominates |
   | W256 gl24 FS CN .0125 | 2 | 11 | E512: .953866 [.899066, 1.076519] | E512; unresolved |
   | W256 lat8192 QN BDF2 .025 | 2 | 11 | E512: 1.270145 [1.187660, 1.364778] | ECSW r128/k16 BDF2 .0125; trade-off |
   | W512 lat16384 QN BDF2 .025 | 2 | 11 | E512: .566849 [.510626, .613120] | E512; dominates |

   B_acc dominates in seven rows; its remaining outcomes are unresolved and trade-off, respectively. T denotes baseline/X full-query time.

5. **CORRECT — envelopes, bootstrap, TOST and cell lines.** Independent NumPy calculations, without invoking `analyze.py` or `stats.py`, reproduce conditional envelopes, **10,000-resample seed-0** intervals, equivalence margins and labels.

   Across **127 unique assessed pairs**: **21 baseline dominates, 100 neither dominates, six unresolved**. No matches or ours-dominates outcomes occur. Assessed counts are 14 per X except W256 QN, which has 15.

   The two withheld conclusions are correctly attached to **W256 FS CN .0125** and **W256 QN BDF2 .025**. Neither is evidence of equivalence or of baseline non-domination beyond the assessed set.

6. **CORRECT — restrictions and their relevance to picks.** Adequacy ratios independently reproduce:

   - r512: **0.944502**, passes.
   - r768: **1.178152**, fails.
   - r1024: **1.584391**, fails.

   The tuning inventory contains **1,056 complete and 72 construction-failed configurations**; the latter are r1024 ECSW configurations affected by the recorded **11.95 GiB allocation OOM**. Coverage remains incomplete.

   The adequacy failure directly concerns the r1024 B_cost picks, but **not E512**, which supplies all seven domination witnesses. Registered restrictions correctly withhold favourable/neutral cell conclusions without retracting observed baseline dominations.

7. **CORRECT — cumulative V; NEEDS-RESTATEMENT — construction-cost scope.**

   | Attempt | Completed V, hours | Allocation, hours |
   |---|---:|---:|
   | t3d64b | 5.876768300 | 7.5028 |
   | s3d64b | 2.736143150 | 2.8253 |
   | cancelled s3d64a | .190554729 | .2239 |
   | cancelled t3d64a | .032411909 | .8875 |
   | **Total** | **8.835878088** | **11.4395** |

   Thus **8.836–11.44 h** is correct as the recorded-phase lower bound and allocation upper bound for V.

   Reported **T = 1.068084737 h** describes the selected tuning attempt, **not cumulative construction expenditure**. Cancelled t3d64a records another **.636062460 h** of completed T phases and an unfinished `ecsw_r512` phase. Cumulative completed T is therefore **at least 1.704147197 h**. Do not present 1.068 h as total campaign construction cost.

8. **NEEDS-RESTATEMENT — headline interpretation.**

   - Split “7 of 9” into **five diagnostic batch-1 dominations and two of four qualified batch-2 dominations**, with the other two qualified cell conclusions withheld.
   - E512 uses **BDF2 at Δt = 1/60**, unlike every X here. This is permitted by A1.4’s registered ladder, but establishes a tuned configuration comparison—not an advantage at fixed scheme or time step.
   - “Dominates” concerns **query time and worst-case error only**. For qualified W256 FS CN .01, E512 has **worse median error: 1.10984% versus .569816%**, despite lower worst error: **3.23361% versus 6.36983%**.
   - FS and QN denote solver implementations for the off-mesh least-squares residual. Comparing them against POD-LSPG is permitted, but does not isolate basis, hyper-reduction or solver effects.
   - Membership was frozen; winners were selected using Stage-B outcomes. Conclusions remain **developmental, in-sample, unadjusted and conditional on ε_ref**.

No files were changed, jobs submitted, GPU work performed, or scratch files created.