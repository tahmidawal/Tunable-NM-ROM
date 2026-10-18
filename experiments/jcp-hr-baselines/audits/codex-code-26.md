**NOT READY for (B)** at `14a4c8149`. Paths below are relative to `experiments/jcp-hr-baselines/`.

1. **CORRECT — nonblocking — round-25 item 3: pair labels.**  
   `analyze.py:181–195,294` applies qualification and restrictions to the pair inventory. The in-memory probe returned **“DIAGNOSTIC ONLY … B_cost: neither dominates (withheld: basis adequacy failed …)”**.  
   **Fix:** retain the shared formatter.

2. **CORRECT — nonblocking — round-25 item 4: truncated-job accounting.**  
   `analyze.py:81–96` retains the V lower-bound flag and adds measured allocation bounds. A truncated-job probe with tuning/Stage-B allocations of 2/3 GPU-hours returned an upper bound of **5 GPU-hours**; unavailable allocations produce `None`.  
   **Fix:** retain explicit attempt binding and unknown bounds.

3. **CORRECT — nonblocking — round-25 item 6: infinite KKT evidence.**  
   `freeze_stageb.py:97–101` now requires finite certificate values and nonnegative tolerance. The synthetic all-infinite KKT certificate was rejected.  
   **Fix:** retain this predicate.

4. **WRONG — blocking — round-25 item 7: comparator lifetimes still violate A36.3.**  
   Successful G-ours iterations release their comparator, but exceptions bypass `del o_, fx` (`stageb.py:189–206`). Executing the actual function body with weak-reference doubles produced live-object counts **[1, 2]** when the first replay raised.

   The normal A–B–A path also overlaps comparators: `fX` captures `fx` at `stageb.py:332`, survives `del o_, fx` at `353`, and remains alive during the next construction at `316`. The actual loop-body probe again produced **[1, 2]**. The real 3D callable captures `self` (`ours.py:139–142`).

   **Fix:** scope each comparator’s construction and execution inside a helper, with cleanup covering exceptions and all capturing closures before constructing the next comparator. Recheck the memory projection after correcting lifetimes.

5. **WRONG — blocking for resource admission — remaining memory-model omission.**  
   `stageb.py:193–197` retains comparator lattice outputs in `XS`. For 2D, `ours.py:111–113` returns a strided view backed by the full-grid host array. Yet `freeze_stageb.py:202–207` charges only lattice-sized comparator fields.

   Executing the actual lattice method at L=1024 returned a **3,170,352-byte view retaining 50,430,000 bytes**. These retained arrays accumulate across timing cases and comparators.

   The requested-resource checks themselves work: executing `cluster/stage.py:143–151` accepted a 60/50 GB device/host model on H200/200G and rejected A100/200G and H200/20G. Their input model remains incomplete.

   **Fix:** store independent lattice-sized copies, release full-grid backing arrays, and model the resulting simultaneous allocations.

6. **CORRECT — nonblocking — round-25 item 8: time-projection wording.**  
   `freeze_stageb.py:208–210` exports `hours_estimate` and explicitly separates estimates from hard wall/ledger controls. `cluster/stage.py:144–148` compares it against usable driver time.  
   **Fix:** retain this wording; remove the stale “certified factor 2” comment at `freeze_stageb.py:176` as an editorial cleanup.

7. **CORRECT — nonblocking — round-25 items 10 and 11: replay failure and invalid scores.**  
   `analyze.py:131–132,224–232,250–255` marks these candidate sets incomplete. Separate probes for replay failure, NaN eligible score and missing eligible score all returned **“role incomplete (missing)”**, zero assessed pairs, and no favourable cell line.  
   **Fix:** retain these checks.

All probes were CPU-only and in memory, with writes intercepted. No files, scratch, jobs, GPU work, or `tests/test_core.py` were used.