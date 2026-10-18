**Overall: RESULTS-WRONG — the cumulative validation-cost accounting is incomplete.** The eight numerical domination conclusions reproduce; I found no disagreement in fronts, picks, timing statistics, or pair labels.

1. **CORRECT — provenance and eligibility.** Verified the verdict checksum, frozen-membership hash, Stage-B hash, supplied tuning hash `536163…be26`, and all 35 baseline coefficient-file hashes. Every X uses the same 38 eligible cases: `dev6|0–5` and `val32|0–31`. P1 per-case errors reproduce every reported X median and maximum. Store-derived ε_ref ranges from `8.6699383e-7` to `9.5195524e-6`; baseline score records agree with that mask and those values.

2. **CORRECT, with an audit limitation — Q1 and replay.** Independently aggregated 2,730 production/anchor certificates, including saved stepwise γ, exit reasons and damping. Production has zero nonstationary steps. Combining these certificates with the recorded lattice distances reproduces **33 passes and two failures**:
   - ECSW r32/k4: maximum distance `5.763102852e-4`.
   - ECSW r32/k8: maximum distance `5.906214221e-4`.

   Both exceed `2.5e-4`. Largest passing distance is `1.497405810e-4`; largest tight–tighter distance is `3.780408663e-7`. All baseline replay records pass, maximum discrepancy `2.535302193e-13`; all eight G-ours gates pass, maximum `1.272392532e-13`.

   **NEEDS-RESTATEMENT:** these distances were **not independently reconstructed in the lattice norm**. The coefficient files are local, but their required `Phi_lat.npz` is offloaded. I did not copy the large basis. Likewise, baseline replay discrepancies were checked against their recorded evidence, not independently regenerated from offloaded tuning coefficients.

3. **CORRECT — timing contract.** Recomputed every ratio and drift for **264 pairs**, covering six cases and three repeats: **4,754 attempts**, including two retries, yielding 4,752 accepted triples. Maximum attempted drift was `0.277089129`; rejected attempts were retried correctly. Every recorded timed-output discrepancy is zero. Repeat inventories, positive finite durations, per-case median ratios and T all pass.

   Both methods ran in job **5840961**, on the same **H200**, UUID `GPU-00f941a0-47e4-a46a-3ac2-257f0a4c43cc`. The log confirms GPU execution; the batch script enables f64/highest precision. Timing code warms both queries and burns for 0.1 seconds before each synchronized call. No cross-job timing comparison was found.

4. **CORRECT — X-specific fronts and registered picks.** Independently reconstructed all fronts and applied the registered tie-breaks. All match. Below, Q256 denotes Q-DEIM r256/p512; Q512 denotes Q-DEIM r512/p1024; E256/E384 denote ECSW r256/r384, κ=8. All picks use common-solver CN-R, Δt=0.005, central-generator snapshots.

   | X | Front size | B_acc | T, 95% CI | B_cost | T, 95% CI |
   |---|---:|---|---|---|---|
   | R128 BE | 15 | Q256 | 0.944965 [0.923368, 0.970687] | Q256 | same |
   | R128 CNR | 15 | Q256 | 0.945158 [0.920819, 0.967276] | Q256 | same |
   | R256 BE | 15 | Q256 | 0.391966 [0.382995, 0.399973] | Q512 | 0.796054 [0.785015, 0.815105] |
   | R256 CNR | 16 | E384 | 0.604127 [0.589885, 0.626745] | Q512 | 0.802936 [0.786456, 0.824685] |
   | R384 BE | 14 | E256 | 0.241272 [0.237644, 0.246093] | Q512 | 0.473255 [0.458872, 0.482861] |
   | R384 CNR | 16 | E384 | 0.354102 [0.348849, 0.362080] | Q512 | 0.473643 [0.462814, 0.478073] |
   | R512 BE | 14 | Q256 | 0.086036 [0.084493, 0.087593] | Q512 | 0.178832 [0.174861, 0.182401] |
   | R512 CNR | 16 | E384 | 0.138428 [0.135044, 0.141440] | Q512 | 0.182964 [0.179691, 0.186942] |

5. **CORRECT — envelopes, bootstrap, equivalence and labels.** Independent NumPy calculations, without invoking `analyze.py` or `stats.py`, reproduce the conditional envelopes, 10,000-resample seed-0 bootstrap intervals, equivalence margins and decision-table outcomes.

   Both picks dominate for every X. Across **122 unique assessed pairs**: **52 baseline dominates, 68 neither dominates, two unresolved**. Per-X assessed counts are `18, 18, 17, 12, 16, 12, 17, 12`, in table order. Exploratory membership and labels match; no matches or ours-dominates outcomes occur.

6. **CORRECT — restrictions and cell lines.** The 256² saturation check fails at r384: `0.000895807224 / 0.000688355291 = 1.30137334 > 1.10`. The 1024² cell therefore remains snapshot-limited with its 12-step basis. Recorded in-cell adequacy ratios pass, maximum `1.08236379`. All 172 tuning configurations are complete, including 84 ECSW configurations; recorded ECSW construction qualities are `ok`. No coverage or quality restriction is missing from those inventories. Snapshot restrictions propagate, and neutral conclusions are withheld as required.

7. **WRONG — cumulative V accounting.** The reported arithmetic correctly describes only the two selected attempts:
   - T: **0.987778594 h**
   - tuning V: **0.756969234 h**
   - successful Stage-B V: **1.265429762 h**
   - reported V: **2.022398995 h**

   However, [analyze.py’s accounting](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/analyze.py:87) omits cancelled attempt **s2d1024a**, job **5674029**. Its [raw checkpoint](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/runs/s2d1024a/archive/output/stageb.json) contains all 35 member results, 67 completed timing pairs, and **1.034590999 h of completed V phases**, plus an unfinished `aba` phase. Its measured allocation is **2.1294 h**.

   Under A3.1’s “all validation-informed work” rule, cumulative V is therefore **at least 3.056989995 h**, with additional unfinished-phase expenditure. `V_is_lower_bound=false` is wrong for cumulative expenditure. The reported allocation upper bound `3.088 h` excludes that attempt; the three-attempt allocation total is **5.2174 h**. The report generator consumes the incomplete V total for its budget comparison, so this requires correction before budget claims are accepted.

8. **NEEDS-RESTATEMENT — headline scope.** Supported wording is:

   > For each of the eight X configurations, a baseline selected by the registered role rule exhibits lower full-query time and lower worst-case error on this developmental cohort; the comparisons are in-sample, unadjusted, conditional on ε_ref, and snapshot-limited.

   Membership was frozen beforehand; **picks were selected using Stage-B results**, as A26.5 permits. “Registered pick” must not imply an independently frozen winner or held-out confirmation. The verdict already discloses in-sample selection. No timing or eligibility artefact was found that reverses its numerical headline.

No files were changed, jobs submitted, GPU work run, or scratch files created.