**Verdicts: e2s1 — SUBMIT; e2s2 — SUBMIT; e2s3 — SUBMIT.** The selection jobs are sound. Two downstream safeguards need fixing before their outputs drive verdict jobs or reporting; neither requires changing these selection runs.

1. **CORRECT — Config equivalence and replay scope.**  
   Structural comparison finds only `attempt`, `bank`, and removal of `replay` differ from e2s0. All three retain the same 38 cases, ranks, candidates, successors, schemes, target/check and tighter populations. DESIGN A3.6 explicitly says incumbent replay is never applied to newly trained banks.

2. **CORRECT — Runtime bank binding.**  
   `r2acc.py:98–108` verifies both configured hashes; `p1common.Bank` passes those exact paths into `Q.Model`. Results record checkpoint and rotation hashes. `e2bank.use()` verifies those recorded identities and clears NumPy decoder caches. Independently verified registered identities:

   | Seed | Checkpoint SHA256 prefix | Rotation SHA256 prefix |
   |---|---|---|
   | 1 | `244e1107c33a` | `73bb603b48f2` |
   | 2 | `bf88ac1f0e2e` | `7a36117785e5` |
   | 3 | `af877976023c` | `fb9e79f2e304` |

   Each matches its config, BANK.json and archived source; none matches the incumbent.

3. **CORRECT — Registration and recipe completion.**  
   All three `SEED-COMPLETE` markers exist. Per seed, all **21 locally present manifest entries** hash correctly. The sole absent entry, `stageB/sep_coeff_N256_K16_R512.npz`, has the identical hash in `LARGE.sha256`. This verifies archive-manifest linkage; remote bytes were not accessed. Each seed records Stage A **300,000 steps**, mid head **200,000 steps**, `time_cap_fired: false`, production mode and GPU execution. Registered records equal archived records.

4. **CORRECT — Main downstream binding and gate scope.**  
   `e2analyze.py:220–255` checks requested-bank checkpoint/rotation identities, input attempt/mesh roles, verdict selection hashes, realised rules and selection-job ID. G-E2.3 runs only for bank 0; `e2verdicts.py` propagates its failure dimension-wide. G-E2.1 is the incumbent’s non-blocking rule-recovery comparison. Target certification remains per bank/rank under both schemes.

5. **NEEDS-FIX — Selector’s requested bank is not independently bound.**  
   `select2d.py:27–31` validates the bank declared by the supplied result, but accepts CLI bank `b` solely as the output label. Thus e2s1 output supplied with argument `2` can produce `select2d-2.json`. Final analysis catches the mismatch, but verdict-config generation can already consume that mislabelled selection. Before generating follow-on jobs, require `attempt == e2s<b>` and both hashes to match `inputs/banks/s<b>/BANK.json`.

6. **NEEDS-FIX — Preserve “target unverified” on downstream rows.**  
   `select2d.py` correctly marks diagnostic picks, and `e2verdicts.keys_2d()` excludes them from selected-rule agreement. However, `e2analyze.py:261–275` and cross-bank output rows omit that qualifier from individual downstream verdicts. A read-only synthetic check confirmed this. A2.5 requires the qualifier to travel downstream; **successor-Q2 and E0 verdicts may remain available**. Fix the annotation before reporting.

7. **CORRECT — Resources and certificate freshness.**  
   Each inventory contains **5,304 unique executions**, keyed to its seed checkpoint. Independent recomputation gives **0.543904 h**, matching certified **0.5439 h**. The required staging allowance is **0.81585 h**; **1 h clears it**. e2s0 recorded **1,629.3 s** driver elapsed and **0.455 h** ledger elapsed. This supports scheduling headroom, not a cross-job speed claim.

   All three job/config/inventory/timing hashes match. Certificate code-tree hash `995bdb51…` matches both its recorded commit and audited HEAD `11d600a1…`; audited code, configs and bank files also match committed bytes.

No files changed, no scratch created, and no jobs, cluster or SSH commands ran.