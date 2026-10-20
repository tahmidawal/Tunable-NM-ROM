**All four jobs: SUBMIT.** No submission blocker found at committed HEAD `3548238a40be`.

1. **CORRECT — Configs and populations.** In-memory generator outputs exactly match all four committed configs and job specs. Accuracy configs match bank 0 except bank/selection identity and required rule rows:
   - Selected: R128 `gauss96→gauss128`; R256 `fib10946→fib17711`; R384 `gauss160→gauss192`.
   - Separate deployed rows exist at every rank.
   - `e2v1a`: 256², 38 cases, factor 1.
   - `e2v1b`: 4096², registered 16-case subset, factor 1.
   - `e2v1c`: 1024², 38 cases, factors 2/2.5.
   
   Both BE and CN-R are present. Verdict jobs correctly omit cross-family targets; selection supplies that evidence.

2. **CORRECT — Timing workload.** `e2t1` retains dev6 × three repetitions and the bank-local R384 gauss128 CN-R reference. There are **35 timed subjects**; reference A occupies the deployed slot and is excluded from subject timing. Successor/check ladders are correct, including `fib10946→fib17711→gauss384`. Inventory: **2,358 unique entries, zero duplicates**, including 630 split triplets, 630 fused triplets and 18 doubled-work repetitions.

3. **CORRECT — Independent bank binding.** Selection job **6083351**, its recorded hashes, local files, committed files and `BANK.json` agree:
   - Checkpoint: `244e1107c33ae38a…`
   - Rotation: `73bb603b48f2883d…`
   
   All three selected ranks have certified targets. Both drivers verify the supplied bank files. Incumbent replay is correctly restricted to bank 0; its recorded G-E2.3 passes.

4. **CORRECT — Downstream consumption.** `e2analyze.py` checks checkpoint **and** rotation identity, registered attempt/mesh roles, selected/deployed rule identities, selection-job ID, and selection-file SHA256. Timing additionally checks its attempt, selection hash and bank identity. `e2verdicts.py` consumes the resulting slot/fixed labels, preserves qualifiers, and compares seeds 1–3 separately from historical incumbent agreement.

5. **CORRECT — Time reservations.** Independently recomputed projections include the certificate’s 1.25× margin; every request also passes staging’s additional 1.5× requirement.

   | Job | Certified hours | Minimum request | Requested | Bank-0 measured driver seconds |
   |---|---:|---:|---:|---:|
   | e2v1a | 0.1488 | 0.2232 | 1 h | 267.2 |
   | e2v1b | 0.2329 | 0.3494 | 1 h | 285.3 |
   | e2v1c | 0.2652 | 0.3978 | 1 h | 329.7 |
   | e2t1 | 1.0883 | 1.6325 | 2 h | 1,054.8 |

   Historical elapsed times support scheduling headroom; they are not cross-job performance comparisons. The dtime estimate is explicitly labelled as required by A10.2.

6. **CORRECT — H200 placement.** `e2v1b` preserves the successful bank-0 request: H200, 240G host memory, allocator fraction 0.95. Bank 0’s measured device peak was **96.59 GB**, below its **142.61 GB** allocator limit. Bank dimensions remain unchanged, and rule tables are released between blocks. This supports the request; bank-1 peak memory remains unmeasured.

7. **CORRECT — Certificates and committed contents.** All four job-spec, config, inventory, timing-source and code-tree hashes match. Certificate commit `4e8492db…` and current HEAD have the same certified code-tree hash. Regenerated inventories and projections match exactly.

8. **CORRECT — Banks 2 and 3 need a focused diff check, not a repeat full implementation audit**, provided the generator/drivers/consumers remain unchanged. Check each bank’s independent selection identity, hashes, resulting rules/deployed rows, inventories and fresh resource certificates. Any changed code or newly exercised exceptional selection path needs targeted re-audit.

| Job | Verdict |
|---|---|
| e2v1a | **SUBMIT** |
| e2v1b | **SUBMIT** |
| e2v1c | **SUBMIT** |
| e2t1 | **SUBMIT** |

No files changed, scratch written, jobs submitted, or SSH/cluster commands run.