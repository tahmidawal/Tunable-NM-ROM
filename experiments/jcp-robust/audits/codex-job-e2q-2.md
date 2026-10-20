Audited HEAD `3ad62e1a95eff7d0319dabe42b885763835076d2`. **FIX-FIRST remains: A2.5 qualifier propagation is incomplete.** No files changed, scratch created, jobs submitted, or SSH/cluster commands run.

1. **CORRECT — diagnostic selection handling.**  
   `select3d.py` records `target_certified` per slot. Diagnostic picks retain own-family qualification downstream. Target-unverified “none within ladder” produces a non-definitive selection label and front state `missing`, preventing false agreement on absence.

2. **NEEDS-FIX — aggregate H-acc-r verdict loses the diagnostic qualifier.**  
   [e2analyze3d.py:265](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e2analyze3d.py:265) stores `range_256_1024` as a bare string. [quals_3d:102](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e2verdicts.py:102) only propagates qualifiers from dictionaries.

   Executing the actual analysis block with target-unverified diagnostic picks at every width reproduced:
   - Individual transitions: `improves`, with the A2.5 qualifier.
   - Aggregate: `monotone`, **without any qualifier**.

   The aggregate can therefore appear as an unqualified seed-robust verdict. Retaining the error verdict is permitted; dropping its diagnostic label violates A2.5. Preserve the qualifier on the aggregate and propagate it through `quals_3d`. Missing-rule/missing-evidence early returns also omit the qualifier.

3. **CORRECT — selected-rule H-acc-r comparisons added.**  
   The new loop covers all three specs, both families, and all three adjacent-width transitions using each width’s realised rule. It retains own-family qualification for diagnostic picks, filters paired cases by E0 eligibility, and reports unavailable when no rule exists.

4. **CORRECT — 3D spread-ratio calculation.**  
   The gl64 calculation implements A2.7’s lower-width cross-seed median-error spread divided by the median absolute paired change. In-memory fixtures verified ordinary division, positive/zero → `inf`, and zero/zero → `undefined`.

5. **CORRECT — 2D behavior preserved.**  
   The 2D extraction and statistical paths remain unchanged. An in-memory end-to-end fixture produced exactly identical old/new 2D output, including qualifiers and spread ratios. Seed classification follows A2.7, which supersedes A1.10’s incumbent-inclusive definition.

6. **CORRECT — previously accepted job definitions remain unchanged.**  
   Compared with the prior audited commit, configs, job specs, drivers, scoring interfaces, bank-binding implementation, and timing inputs are unchanged. Each inventory still contains **11,424 unique executions**. Independently rehashed seed-1 bank bytes match `BANK.json` and the previously audited hash. The prior host-memory evidence limitation remains; no new evaluation MaxRSS evidence was established.

7. **CORRECT — all four resource certificates match HEAD.**  
   Independently verified code-tree, job-spec, config, inventory, and timing hashes; reconstructed inventories and projections. Certificate commit `f554295f6` and HEAD have the same bound code-tree hash.

   | Jobs | Recomputed projection | Limit |
   |---|---:|---:|
   | e2q0a, e2q1a | 1.823194 h | 2.75 h |
   | e2q0b, e2q1b | 4.801016 h | 7.25 h |

**Per-job verdicts:** `e2q0a` **FIX-FIRST**; `e2q0b` **FIX-FIRST**; `e2q1a` **FIX-FIRST**; `e2q1b` **FIX-FIRST**.

Finding **2** is the remaining blocker. Refresh the certificates after correcting the downstream code.