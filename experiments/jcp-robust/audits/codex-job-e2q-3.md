Audited HEAD `08677b2c68c860cf808873f6eda956b965507e8c`. **The remaining blocker is resolved.**

1. **CORRECT — aggregate qualifier propagation (e2q-2 #2).**  
   A13.7 preserves the A2.5 qualifier through `h_acc_r_qualifiers` and `quals_3d`, including the final cross-seed output. In-memory execution passed certified, diagnostic, mixed, no-rule, missing-score, and missing-qualification fixtures.

2. **CORRECT — diagnostic selection and absence handling.**  
   Slots retain `target_certified`; diagnostic picks retain own-family qualification. Target-unverified “none within ladder” remains non-definitive for selection and front membership, preventing false agreement on absence.

3. **CORRECT — selected-rule H-acc-r coverage.**  
   All three specs × both families include all three adjacent-width transitions and the aggregate. Missing-rule and missing-evidence returns now retain their diagnostic qualifiers. This satisfies A1.9 and A2.5.

4. **CORRECT — cross-seed semantics and regression checks.**  
   A2.7 supersedes A1.10’s incumbent-inclusive classification. The 2D paths and previously accepted 3D spread-ratio calculation are AST-identical to the prior audited version. No regression found.

5. **CORRECT — previously accepted job definitions and bank binding.**  
   Configs, job specs, drivers, scoring interfaces, and timing inputs are unchanged since audit 2. Each reconstructed inventory contains **11,424 unique executions**. Rehashed seed-1 bank bytes match `BANK.json` and the previously audited hash.

6. **CORRECT — all four resource certificates match the committed tree.**  
   Verified code-tree, job-spec, config, inventory, and timing hashes. Certificate commit `07a303a0d` has the same bound code-tree hash as HEAD.

   | Jobs | Recomputed projection | Limit |
   |---|---:|---:|
   | e2q0a, e2q1a | 1.823194 h | 2.75 h |
   | e2q0b, e2q1b | 4.801016 h | 7.25 h |

   The prior limitation remains: evaluation host-memory headroom lacks a local MaxRSS measurement. Timing projections remain estimates.

**Per-job verdicts:** `e2q0a` **SUBMIT**; `e2q0b` **SUBMIT**; `e2q1a` **SUBMIT**; `e2q1b` **SUBMIT**.

No files changed, scratch created, jobs submitted, or cluster/SSH commands run.