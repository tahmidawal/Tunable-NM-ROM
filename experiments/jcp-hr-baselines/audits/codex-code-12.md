Audited A16 commit `66a0d96e7` against `12a34bcc3`, focusing on the requested changes. CPU-only probes; no files changed or jobs submitted. Paths below are relative to `experiments/jcp-hr-baselines/`.

1. **MAJOR — WRONG: A16.3 does not preserve both NNLS attempts.**  
   `build.py:350–365` replaces `out` when the retry reaches more targets. The returned summary then contains the selected retry twice, losing the original attempt’s banned count, termination reason, iterations, KKT diagnostics and checkpoint results. When the retry loses, its checkpoint results and several diagnostics are also absent. A mocked retry-selection probe confirmed this information loss.  
   **Fix:** retain separate `initial` and `retry` summaries, including per-target results, and an explicit selected-attempt identifier. Preserve these through `pipeline.py:184–186`.

2. **MAJOR — WRONG: acceptance does not validate the new retry contract.**  
   `smoke_accept.py:233–235` checks only reported checkpoint sizes and finite residuals. It never reads `ecsw_info` or `ban_retry`, although `smoke.py:286` records that metadata. Missing initial-attempt evidence, an omitted required retry, or an incorrectly selected attempt therefore cannot fail this gate. Checkpoint nodes and weights are also discarded by `smoke.py:250–251`, preventing independent support verification.  
   **Fix:** persist both attempts’ checkpoint evidence; have acceptance reconstruct the retry trigger, compare target coverage, validate selection, and check actual positive-weight support. Missing evidence must fail. A retry that was correctly unnecessary should remain acceptable.

3. **MINOR — CORRECT: larger-cap checkpoint behavior matches A16.2.**  
   The defaults in `build.py:335` and `pipeline.py:175`, used by `smoke.py:251`, request 2r/4r/8r/16r. In `nnls.py:154–156`, only final-batch clipping depends on the cap. A CPU probe with caps 80 and 160 produced identical earlier checkpoints and identical paths before the shorter path’s final batch; its terminal checkpoint changed from support 80 to 81, as explicitly permitted.  
   **Fix:** no algorithm change needed; add this regression test. This invariance applies to the same banning policy, not to switching to the no-ban retry.

4. **MINOR — CORRECT: the retry is iteration-bounded and effectively disables banning.**  
   `build.py:346–351` limits each attempt to 20,000 outer iterations. With at most one drop per index per outer iteration (`nnls.py:172–174`), `max_drops=10**9` cannot ban an index within that bound. The inner refit is also bounded (`nnls.py:33–34`). This is an iteration bound, not a practical wall-time guarantee. Missing checkpoints fail the updated smoke-support predicate.  
   **Fix:** retain these bounds and report iterations separately for both attempts.

5. **MINOR — CORRECT: inherited identity claims hold.**  
   All 17 `INHERITED.json` entries match their recorded hashes; all entries naming a source commit also match that committed source byte-for-byte.  
   **Fix:** none.

**Overall: NOT READY for smoke `smk2`.** Fix preservation of both NNLS attempts and fail-closed validation of the A16 retry evidence before spending the smoke allocation. Phase II must additionally enforce extension-only use of κ=16, incomplete-coverage labels, and A16 construction/validation accounting. This targeted review does not recertify unchanged phase-I internals.