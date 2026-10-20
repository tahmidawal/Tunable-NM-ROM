**NOT READY** at `1876181dc`: required full-JSON boundary saves are missing.

All paths below are under `experiments/jcp-hr-baselines/`.

1. **WRONG — blocking: not every required boundary forces JSON.**  
   `stageb.py:214` uses unforced `save()` after comparator cases; `:377` does likewise after completed/failed timing pairs. Budget-skipped pairs reach `continue` without saving (`:347–349`). Successful timing comparators have no end checkpoint (`:381–388`); only their exception path forces one. This violates the requested boundary contract and, for pairs/comparators, `DESIGN.md:1880–1882`.  
   **Fix:** force saves at `:214` and `:377`, before the budget-skip `continue`, and after each timing comparator on both success and failure. Phase/group ends (`:131,319`), member ends (`:317`), replay-comparator ends (`:227`), and final completion (`:392`) already force saves.

2. **CORRECT — nonblocking: JSONL captures the required completed evidence.**  
   Member records contain identity, case, certificate, E0 score and coefficient trajectory (`stageb.py:287–291`); anchors include group, level, case, certificate and trajectory (`:270–280`); comparator cases include E0, exported E0 and replay discrepancy, after writing their arrays (`:210–214`). Each triple is appended before its callback (`:76–79`), which writes pair/case identity and every triple field before subsequent work (`:368–371`). Callback defaults bind the intended pair, case and record.  
   **Fix:** none.

3. **CORRECT — nonblocking: computation and decision rules are unchanged.**  
   The checkpoint diff preserves Q1 and replay qualification (`stageb.py:292–314`), total ordering (`:327–330`), budget predicates (`:336,347`), warmups/output checks (`:362–373`), and triple repetitions, retries and burn-in (`:62–81`). Reduced checkpoint overhead can change how much work fits within the budget, as A42 intends.  
   **Fix:** none.

4. **CORRECT for phase exceptions and normal completion — nonblocking cleanup limitation.**  
   There is one synchronous writer, opened line-buffered (`stageb.py:103`); every event ends with a newline (`:115–116`). Completed writes are flushed before later phase exceptions. Normal completion closes the handle (`:393`), with no later event writes. However, an uncaught exception outside phase handling can bypass explicit closure.  
   **Fix:** use a context manager or outer `try/finally` to guarantee closure on exceptional exits. Line buffering protects against process termination after a completed write; it does not promise storage durability against machine failure.

5. **CORRECT — nonblocking: analysis remains conservative.**  
   `analyze.py:111` reads `stageb.json`, not JSONL; normal completion forces the final JSON before closing (`stageb.py:390–394`). Missing/incomplete timing is rejected (`analyze.py:233–242`) and yields **“role incomplete”** (`:250–255`).  
   A truncated job’s JSONL should remain recovery/audit evidence unless an explicit validated reconstruction step is implemented. Not consuming it today sacrifices recoverable progress, not conclusion correctness. **Fix:** none required for analysis.

Read-only inspection only; no jobs, GPU work, file changes or scratch artifacts.