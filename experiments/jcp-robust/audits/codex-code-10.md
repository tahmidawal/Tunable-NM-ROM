**Overall verdict: NEEDS-FIX.** Acceptance code should not yet be released for reading smoke outputs or production.

Audited HEAD `af2ba0606`. No files changed, scratch created, smoke outputs read, or SSH/cluster commands executed. Paths below are relative to `experiments/jcp-robust/`.

1. **Round 9 #1 — RESOLVED — CORRECT — formerly BLOCKING.**

   `smoke_accept.py:264–274` now decodes all four pilot trajectories before subtraction and requires every decoded value to be finite, including the initial state. Failed finiteness makes the mandatory evidence check fail at `:277–278`.

   The small CPU overflow counterexample now produces `fields_ok=False`.

   **Fix:** None.

2. **Round 9 #5 — NOT RESOLVED completely — NEEDS-FIX — BLOCKING.**

   The new checks correctly reject mismatched job IDs, mismatched cohort seeds and missing cases **within the supplied archive’s keys**. However, two holes remain:

   - **Extra score keys pass and can overwrite another job’s evidence.** `e2analyze3d.py:90–100` checks required keys without rejecting extras. At `:117–120`, `scores.update(s_)` lets the second file overwrite first-half scores. The same incomplete validator exists at `e1analyze3d.py:48–59`.
   - **Reference and decoder provenance remain unchecked.** Neither validator checks `cohort`, `store_version` or `status_sha`. Score output at `score3.py:110–114` records no coefficient-archive or decoder-bank hash. Running the ordinary incumbent `score3.py` on a retrained-bank job can therefore produce matching job ID, cohort seed and case mappings while decoding on the wrong bank. The correct wrapper installs that bank at `e2score3.py:33–52`, but its use is not established by the score artifact.

   **CPU evidence:** Both actual validators accepted deliberately wrong reference metadata and an extra first-half score key. Applying the actual merge operation replaced that key’s two-case population with one case. Wrong-job and missing-own-case controls were rejected.

   **Fix:** Require exact score-key equality, reject conflicting keys across halves, validate reference identity, and bind scores to coefficient and decoder-bank hashes—or regenerate scores through the bank-aware scorer.

3. **NEW — Extra 1024² job’s backend is discarded — WRONG — BLOCKING.**

   `e2analyze.py:200–203` merges `--v1024` configurations and completion status, but retains only the selection job’s backend and job ID. The bank check at `:212–214` does not check backend. Consequently, the reported job list at `:217–218` omits the extra job, and `e2verdicts.py:108` cannot reject its CPU results.

   **CPU evidence:** Executing the actual `main()` with in-memory I/O and analysis stubs admitted a complete CPU extra job’s configuration, reported only the GPU selection job, retained `integrity=True`, and passed the downstream bank-validity predicate.

   **Fix:** Preserve every contributing job separately and require completion and GPU backend for each before its configurations can contribute to verdicts or fronts.

4. **Smoke-specific control wiring — CORRECT on the examined paths; overall release remains BLOCKING because of findings 2–3.**

   This is a source-level assessment, not acceptance of actual outputs. “Accept” assumes complete supporting evidence and the resource contract also pass.

   | Smoke | Passing evidence can reach ACCEPT? | Required failed/missing controls prevent ACCEPT? |
   |---|---|---|
   | **smk1b** | Yes | Yes for E1 controls, reproduction, target certification and integrity checks: `smoke_accept.py:86–128`. Its 3D scores are regenerated before analysis. |
   | **smk2** | Yes | Yes on the examined paths: `:142–193`. A6.3 legitimately permits “verdict not reached”; numerical qualification itself need not succeed. |
   | **smk3** | Yes | Yes on the examined paths: `:201–305`. Memory, rung evidence, pilot decoding, resource contract and incumbent replay are mandatory. Finding 1’s bypass is closed. |
   | **smk4** | Yes | Yes: `:321–332`, including the CPU drift test and E5 acceptance conjunction. |
   | **smk5** | Yes | Yes: `:309–317`, for the H200 E5 panel. The shared CPU drift test runs through smk4. |

   `e5analyze.py:722–764` explicitly conjoins job status, populations, doubled-work control, injected/null fixtures, reproduction and P-cross gates. Small CPU checks confirmed its positive path and rejection under ten individual veto mutations; expensive estimator functions were stubbed. Actual drift and missing/duplicate/empty-population predicates also passed their checks.

   No additional blocking certificate/collection defect was established. G-E5.2 correctly remains descriptive.

5. **Carried-forward CLI discrepancy — NEEDS-FIX — MINOR.**

   `e2verdicts.py:13` still advertises `--seeds`, while `:102` always uses the registered three seeds and silently ignores the option.

   **Fix:** Remove the advertised option or reject unsupported arguments. The fixed three-seed population is correct.

All 17 requested Python files parsed successfully. **Two blockers remain: findings 2 and 3.**