**Overall verdict: NEEDS-FIX.** Both round 10 blockers are resolved, but one additional production evidence-binding blocker remains.

Audited HEAD `d3257f563`. No files changed, scratch created, smoke outputs read, or SSH/cluster commands executed. Paths below are relative to `experiments/jcp-robust/`.

1. **Round 10 #2 — RESOLVED — CORRECT — formerly BLOCKING.**

   Both score validators now require coefficient-archive hashes, bank-aware scorer identity, pinned reference metadata, decoder-bank identities, exact score keys and exact case coverage: `e1analyze3d.py:48–73`, `e2analyze3d.py:90–115`.

   `e2score3.py:49–61` records the provenance after bank-aware scoring. `e2analyze3d.py:135–137` rejects overlapping score keys before merging halves.

   **CPU evidence:** Both actual validators accepted matching synthetic evidence and rejected extra keys, wrong reference identity, wrong decoder hash, wrong archive hash and missing cases.

   **Fix:** None for the reported defect.

2. **Round 10 #3 — RESOLVED — CORRECT — formerly BLOCKING.**

   `e2analyze.py:217–221` retains the extra 1024² job separately as `1024-factors` and requires **every contributing job** to be complete and GPU-backed before constructing the report. This closes the discarded-backend bypass.

   **Fix:** None.

3. **NEW — Verdict inputs are not bound to their mesh and selected-rule roles — WRONG — BLOCKING.**

   Two related holes remain in `e2analyze.py`:

   - `records()` preserves the actual mesh at `:32–33`, but `main()` assigns records to mesh slots solely from `--sel`, `--v256`, `--v4096` and `--v1024` at `:199–207`. The subsequent checks validate bank, selection-job ID and backend, **never the recorded mesh**.
   - `ver_labels()` trusts the labels `sel` and `dep` at `:191–197`. Nothing verifies that their actual rules equal the current selection or deployed rule. Generated verdict configs carry a selection-file hash (`make_configs_e2.py:78–79`), but the analyzer does not validate it.

   Consequently, a wrong-directory argument or stale verdict job can contribute valid numerical evidence for the wrong experimental role. `h_acc_n()` uses the assigned mesh keys and does not enforce rule equality (`e2analyze.py:50–71`).

   **CPU evidence:** The actual `h_acc_n()` returned `pass` when synthetic records from one actual mesh/job occupied all three mesh slots. It also returned `pass` after changing the 4096² “selected” rule from `fib4181` to `gauss96`. These were predicate checks; the missing caller validations were confirmed by source inspection.

   **Fix:** Before merging records, require the recorded/configured mesh and registered population to match each input role. Validate verdict-config selection hashes against the selection artifact being analyzed, and require each `sel`/`dep` row’s actual rule to match its intended rule. Reject mismatches as integrity failures.

4. **Smoke control wiring — CORRECT on the examined paths; overall release remains BLOCKING because of finding 3.**

   This assesses code, not actual smoke results. ACCEPT also requires complete supporting evidence and the resource contract.

   | Smoke | All required controls/gates pass → ACCEPT reachable? | Required control/gate fails or is missing → ACCEPT prevented? |
   |---|---|---|
   | **smk1b** | Yes | Yes: E1 controls, reproduction, target certification and integrity, `smoke_accept.py:86–128`. |
   | **smk2** | Yes | Yes on the registered paths: replay, bank identity, analysis integrity and selection-chain outcome, `:142–193`. A6.3 permits “verdict not reached”; qualification need not succeed. |
   | **smk3** | Yes, with the fetched pilot bank and required accounting/archive evidence | Yes: memory gates, rung populations/validity, hash-verified pilot decoding, finite fields, certificates and incumbent replay, `:200–303`. |
   | **smk4** | Yes | Yes: CPU drift tests and E5 acceptance conjunction, `:318–326`. |
   | **smk5** | Yes | Yes for its H200 E5 controls/gates, `:307–314`. The shared CPU drift test runs through smk4. |

   `e5analyze.py:722–764` explicitly conjoins job status, populations, doubled-work control, injected/null fixtures, reproduction and P-cross gates. G-E5.2 correctly remains descriptive.

   **CPU evidence:** The actual drift test passed. The actual E5 smoke conjunction accepted a positive synthetic case and rejected 11 individual veto mutations; estimator functions were stubbed for this wiring check.

   **Fix:** No additional smoke-specific fix established.

5. **Carried-forward CLI discrepancy — NOT RESOLVED — NEEDS-FIX — MINOR.**

   `e2verdicts.py:13` advertises `--seeds`, while `:102` always uses the registered three seeds and ignores additional arguments.

   **Fix:** Remove the advertised option or reject unsupported arguments. The fixed three-seed population itself is correct.

All 17 requested Python files parsed successfully. No additional blocking certificate, collection or ledger defect was established in the examined paths. **Finding 3 must be fixed before production acceptance is released.**