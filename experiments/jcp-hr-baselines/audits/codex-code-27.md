**READY for (B)** at `f59050a68`. No additional blocking defect found. Paths below are relative to `experiments/jcp-hr-baselines/`.

1. **CORRECT — nonblocking — comparator lifetimes, round-26 item 4.**  
   `stageb.py:189–214,322–366` scopes comparators and capturing closures inside helpers. Actual-body weak-reference probes returned construction counts **[1, 1]** for successful and failing G-ours replays, and for A–B–A success, construction failure, query failure and timing failure. A–B–A retained **zero** comparators afterward.  
   **Fix:** none; retain helper scoping and exception handling.

2. **CORRECT — nonblocking — oversized lattice backing, round-26 item 5.**  
   `ours.py:111–113` now returns a contiguous copy when downsampling. At L=1024, the actual-method probe returned **3,170,352 bytes**, no backing array and no shared memory with the **50,430,000-byte** input. This resolves the accumulated full-grid retention affecting `freeze_stageb.py:202–207`. Resource-admission probes at `cluster/stage.py:143–151` still accept H200/200G and reject A100/200G and H200/20G for the tested 60/50 GB model.  
   **Fix:** none for the blocking defect.

3. **NEEDS-RESTATEMENT — nonblocking — “independent copies” is too broad.**  
   `ours.py:41,113,145` uses `ascontiguousarray`, which can preserve existing contiguous storage. Probes show shared backing at 2D L=256 (**3,170,352 bytes**) and for the single-batch 3D output (**12,002,256 bytes**). These backing allocations are already lattice-sized, so this does not recreate item 5.  
   **Fix:** qualify A37.2 and the comments as “contiguous lattice-sized storage without oversized backing,” or explicitly use `.copy(order='C')`.

4. **CORRECT — nonblocking — qualified pair labels, round-26 item 1.**  
   `analyze.py:181–195,294` retains the shared formatter. The repeated probe produced a diagnostic-only pair label with the failed-adequacy withholding restriction.  
   **Fix:** none.

5. **CORRECT — nonblocking — truncated-job accounting, round-26 item 2.**  
   `analyze.py:81–96` preserves lower-bound reporting and measured allocation bounds. The repeated 2/3 GPU-hour fixture returned **5 GPU-hours** as the upper bound; absent allocations remain unknown.  
   **Fix:** none.

6. **CORRECT — nonblocking — KKT evidence, round-26 item 3.**  
   `freeze_stageb.py:97–101` rejects non-finite certificate values and negative tolerances. The repeated all-infinite KKT fixture was rejected.  
   **Fix:** none.

7. **CORRECT — nonblocking — projection wording, round-26 item 6.**  
   `freeze_stageb.py:176,208–210` consistently calls the projection an estimate; the stale certification comment is removed. `cluster/stage.py:144–148` checks usable driver time.  
   **Fix:** none.

8. **CORRECT — nonblocking — replay and score failures, round-26 item 7.**  
   `analyze.py:131–132,224–232,250–255` still prevents incomplete candidate sets from filling roles. Repeated replay-failure, NaN-score and missing-score probes each produced **“role incomplete (missing)”**, **zero assessed pairs**, and no favourable cell line.  
   **Fix:** none.

All probes were CPU-only and in memory, with writes intercepted. No files changed, scratch created, jobs launched, GPU work performed, or `tests/test_core.py` executed.