**Overall: READY for (A), with nonblocking collection fixes.** No remaining blocker found for submitting the three tuning jobs. This does not authorize Stage B.

The scoped sources and DESIGN.md match `b08690fdb` byte-for-byte; observed HEAD was `e974c704a9`. Paths below are relative to `experiments/jcp-hr-baselines/`.

1. **CORRECT — nonblocking: round-22 item 1, search accounting.**  
   `tune.py:80–100` records the active phase, records completed/failed duration, then recomputes `search_hours` before saving. An in-memory CPU experiment using the actual method confirmed immediate accounting for both successful and failed screening blocks. Interrupted expenditure remains bounded rather than falsely reported as completed.  
   **Fix:** none; retain.

2. **CORRECT — nonblocking: round-22 item 2, atomic reservations.**  
   `cluster/ledger.py:23–33` locks the complete read/check/update/replace operation. Duplicate-attempt and budget checks occur inside that lock (`45–50`); staging invokes this operation (`cluster/stage.py:154–158`). Concurrent reservations cannot overwrite one another.  
   **Fix:** none; retain.

3. **CORRECT — nonblocking: round-22 item 3, reservation reuse.**  
   `cluster/ledger.py:51–58` consumes the unbound reservation by changing its status before transfer, then binds one job ID. Subsequent submission checks reject it, including after collection. `cluster/submit.sh:28` disables requeue. An interrupted submission can strand a reservation, but cannot reuse it through this submission path.  
   **Fix:** none required for submission safety.

4. **CORRECT — nonblocking: round-22 item 4, anchor custody.**  
   Individual anchors are checkpointed at `tune.py:374–381`. Their independent array artifact and certificates are saved before production begins (`382–387`). Production failures (`428–430`) no longer delete this evidence.  
   **Fix:** none; retain.

5. **CORRECT — nonblocking: round-22 item 5, probe completeness.**  
   Probe configurations are predeclared at `tune.py:152–153`. Partial cases carry `running` status (`494–503`); `complete` follows the coefficient-file write (`504–505`). Interrupted probes cannot masquerade as complete.  
   **Fix:** none; retain.

6. **CORRECT — nonblocking: round-22 item 6, limiting-comparator inventory.**  
   `tune.py:146–151` predeclares the FOM cases before expensive phases. Execution preserves convergence failures and checkpoints each case (`461–471`).  
   **Fix:** none; retain.

7. **NEEDS-RESTATEMENT — nonblocking: round-22 item 7, reconciliation is implemented but incomplete on early failures.**  
   Normal collection now reconciles the matching job under the ledger lock (`cluster/collect.sh:25–29`; `cluster/ledger.py:59–62`). However, a preflight failure can leave no `output/` directory: collection then aborts at `collect.sh:17–19`, before archiving logs or reconciling. The remote evidence and full reservation remain intact, so this is conservative for the cap.  
   **Fix:** support log-only collection when output is absent, record that absence, and reconcile verified terminal accounting before optional deletion.

8. **WRONG — nonblocking: collection violates the absolute-Python-path rule.**  
   `cluster/collect.sh:28` invokes bare `python3`. Line 27 also silently substitutes one GPU when allocation parsing fails. All current staged jobs request one GPU (`cluster/stage.py:36–37`), so this does not establish an over-cap path for these jobs, but it weakens the claimed accounting validation.  
   **Fix:** use `/home/tahmid/Dev/.venv/bin/python`; require validated elapsed time and allocation count, retaining the reservation if accounting is unavailable.

9. **CORRECT — nonblocking: nominal budget, cohorts, mappings and tuning artifacts.**  
   The configurations contain **140 / 140 / 1,008** base configurations. Their walls are **2 / 5 / 12 hours**; including **1.512 spent + 7.5 reserve gives 28.012 GPU-h**. Evidence: configuration files at `:35`, `:35`, `:76`; `runs/GPU-LEDGER.json:30`; reservation enforcement at `cluster/stage.py:62–66` and `cluster/ledger.py:48`. Screening remains reference-eligibility-based (`tune.py:159,410–418`), comparator mappings and batch labels remain explicit (`ours.py:86–148`; `tune.py:175–179`), and reconstruction artifacts are persisted (`tune.py:270–273,309,338,363–367,405–409`). Export keys and positive replay denominators checked successfully. No additional blocking numerical, cohort, or missing-output defect was identified.  
   **Fix:** none within (A); retain the separate Stage-B audit gate.

No files changed, no scratch created, and no jobs, GPU work, or `tests/test_core.py` ran.