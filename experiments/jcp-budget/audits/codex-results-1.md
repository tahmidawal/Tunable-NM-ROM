**NOT-READY.** The main numerical conclusions survive, but two result-label defects need correction. I audited HEAD `c24de2f044`, read all DESIGN amendments, and made no file changes, submissions or SSH connections.

1. **Production integrity — NEEDS-RESTATEMENT.**

   All **17 result-producing production jobs** have `jax_backend=gpu`; the 14 completed jobs have `ALL-DONE`. The three partial collections have `PARTIAL` markers and the documented completed populations:
   - `ref2b`: val32 7–9.
   - `ref2c`: val32 20–30.
   - `ref2t1`: dev6 0–5.

   No Traceback, OOM or captured-constants warning appeared in their archived logs. Both smokes also pass these checks.

   **Exception:** cancelled `ref3` has no local archive/log. A9 documents cancellation after 30 seconds, before any case. Therefore “every production job has a GPU log” is unsupported; “every result-producing production job” is supported.

   I rehashed **428 locally present output files: zero mismatches**. All **215 reference-manifest entries** match the job-output manifests. Offloaded payloads cannot be freshly verified without remote access; their recorded hashes remain evidence of collection, not a new remote verification.

   Ledger arithmetic is correct: **25.9104 GPU-h spent, zero reserved, cap 45**.

2. **References — CORRECT under A13/A14.2.**

   [refs-all.json](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/checks/refs-all.json) records:
   - No evaluable G-R2 failures.
   - Four non-evaluable cases: val32 10/31 at T=0.5; dev6 1/5 at T=1.
   - G-R3 passes for all 55 fine-enabled references; local fine-field comparisons are available for 13.
   - Offline judge agreement passes.
   - T=1 prefix passes for 15 cases; dev6 1/5 have no matched levels and fail eligibility.

   The B7 T=1 population is exactly **dev6 {0,2,4}, val32 {4,…,9}: nine cases**. Accepted dev6 5 remains non-rankable; val32 19 is cross-check-only.

   W1/W2 tiers are validated. **W3 is unvalidated**, because val32 0 is FD-INVALID, despite val32 19 passing.

   Git-history verification succeeded: all **55 T=0.5 case records** match commit `b6fa6beb3` exactly after removing the newly added `tight_base` field; their tiers also match. The earlier B7/B8 analyses are not invalidated by that reference-file update.

3. **B2 — WRONG tie counts; dominant and pair conclusions otherwise CORRECT.**

   The recorded numerical gates and controls pass. All 37 configuration/mesh dominance labels and pair summaries reproduce from the case records. Mesh labels reproduce: **seven “same”, three “inconclusive”**.

   However, [ana2d.py:418](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/ana2d.py:418) checks the largest candidate against only the second-largest. With different reference/reference and mixed thresholds, the third-largest can tie the maximum even when the second-largest does not.

   Two actual errors, both `R384|GAL|CN|1`:
   - **1024², val32 13:** dynamics–M difference `1.15590e-5`, below mixed tie threshold `1.24274e-5`. Correct counts: dynamics **10**, tied **4**, rather than 11/3.
   - **4096², val32 6:** dynamics–M difference `1.23949e-5`, below threshold `1.37818e-5`. Correct counts: dynamics **4**, tied **2**, rather than 5/1.

   Both remain **“no single dominant”**. The same implementation defect exists in `largest3`, although no current 3D count changes.

   Gate-reporting qualification: G-B5 is inapplicable to the structural-zero 3D IC switch. The 3D JSON does not explicitly report G-B1/G-B6. Independent checks found bitwise B2/B7-prefix reproduction for all four members and correct logical settings in **768 corner records**; those checks should be exposed explicitly.

4. **B7 — CORRECT analysis labels; WRONG T=1 report attribution.**

   Independent threshold checks reproduce Q1/Q2, stability, growth populations and W1→W2 front-order results:
   - T=0.5 Q1 fails for R128 BE factor 2.5 at 256²/1024², and R256 CNR factor 2.5 at 1024².
   - All 2D Q2 checks pass.
   - **All four 3D Q2 checks fail**: successor distances range from `3.23948e-4` to `1.88440e-3`, above `2.5e-4`. W512 also fails Q1. These failures are correctly reported.
   - Stability passes at recorded output times. Growth labels use 36, 16 or nine eligible cases as appropriate.
   - T=1 W3 growth labels correctly say **“no verdict … provisional”**.

   But [report.md:276](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/report.md:276) presents **W1→W2** output as **W2→W3**, without W3 qualification. `ana7.front_order()` only computes W1→W2; the generator supplies the wrong caption.

   Recomputing W2→W3 gives four resolved reversals among all members, zero among LSPG members, on nine cases. These are **provisional comparisons, not verdicts**.

5. **B8 — CORRECT, with the registered scope.**

   All **1,896 eligible case-window classifications** and 66 sampling summaries reproduce from the saved numbers. C-A, C-E and G-D1 pass. Fine-route recomputation covers 60/66/72 member-case records at 256²/1024²/4096²; worst relative discrepancies are below `4.7e-12`, versus the `1e-9` bar.

   Every primary 1024² LSPG member is adequate, with no resolved route-order reversal. The lane statement is supported **for LSPG, relative to the 1025² route**.

   It must not include every diagnostic GAL member: R384 GAL CN factor 1 is mixed in W1 and inadequate in W2 at 256²/1024². B8 3D remains explicitly deferred.

6. **Late-amendment claims/provisional status — NEEDS-RESTATEMENT.**

   A14.2 correctly supersedes A14.1’s ten-case population. Its rankability argument is supported by the purpose-matching rule.

   One timing claim is incorrect: A14’s **“13,502 s for one case”** is a cumulative log timestamp. The saved dev6 1 metadata gives **12,915.84 s**. Likewise, dev6 5’s “accepted after 28,005 s” is cumulative; its case duration is **12,683.06 s**.

   The substantive missing provisional qualification is the purported W2→W3 front-order result identified above. The blanket GPU-log statement also needs the `ref3` exception.

**Required fixes — blocking only:**

1. Compare the maximum against **every resolved candidate** using the registered pair-specific tie threshold; regenerate affected counts and downstream outputs.
2. Correct the T=1 front-order window attribution. Any actual W2→W3 comparison must carry the unvalidated/provisional tier and no verdict.
3. Correct the cumulative-versus-case timing claims and blanket GPU-log claim; explicitly distinguish reported, independently reconstructed and inapplicable 3D gates.

**Overall: NOT-READY.**