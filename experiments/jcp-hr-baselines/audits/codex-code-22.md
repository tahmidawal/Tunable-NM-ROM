**Overall: NOT READY for (A).** Remaining blockers affect accounting, reservations and failure evidence.

Audited sources match `7b4d202d2`; observed HEAD `b1e60125e` differs only in `HEARTBEAT.json`. Paths below are relative to `experiments/jcp-hr-baselines/`. Stage-B implementation findings are excluded.

1. **WRONG — blocking: `search_hours` still undercounts interrupted jobs (round 21 item 8).**  
   `tune.py:419–420` calculates the total **inside** the block; `Out.phase` adds that block’s duration only after it returns (`80–95`). Consequently, every checkpoint trails by one completed block; failed blocks do not refresh it. A CPU probe using the actual `phase` method confirmed a completed 10-second screening phase with `search_hours = 0`. Normal completion repairs this at `536–537`, but timeout does not.  
   **Fix:** refresh accounting after recording each phase, before saving. Persist active-phase elapsed time or explicitly identify unfinished expenditure as unknown/bounded.

2. **WRONG — blocking: reservations are not atomic (round 21 item 19).**  
   `cluster/stage.py:65–76` performs an unlocked read/check/append/write. Concurrent staging can approve against the same balance and overwrite reservations. Submission’s check (`cluster/submit.sh:14–20`) does not lock against staging; its remote lock protects only queue admission (`27–36`).  
   **Fix:** use one shared lock and atomic replacement for all ledger mutations and reservation checks. Verify unique attempt reservations under that lock.

3. **WRONG — blocking: a reservation can authorize repeated execution.**  
   `cluster/submit.sh:17` accepts any matching ledger entry, including a completed attempt; submission never marks its reservation consumed or records the job ID. After `cluster/collect.sh:25` deletes the remote directory, submitting the same local attempt passes the directory guard again (`submit.sh:34–36`) without charging another wall limit.  
   **Fix:** bind each reservation to exactly one submission/job ID and reject reuse. Require a fresh reservation for retries; disable automatic requeue or account for it explicitly.

4. **WRONG — blocking: completed anchors can still disappear on production failure (round 21 item 10).**  
   Individual anchor checkpoints now exist (`tune.py:366–374`). However, production exceptions are caught at `415–416`, after which `418` unconditionally removes the shared anchor checkpoint. If neither solver reaches the array write at `392–396`, no durable anchor output remains.  
   **Fix:** persist anchors independently and retain them regardless of production outcomes; remove temporary checkpoints only after verifying their durable replacements.

5. **WRONG — blocking: interrupted probe configurations are labelled complete.**  
   `tune.py:484` initializes `status='complete'` before running cases. Lines `492–493` checkpoint that status after each case. A timeout after the first case therefore leaves a “complete” six-case configuration with one result. Later probe configurations are not predeclared.  
   **Fix:** predeclare the probe inventory, use `running` during execution, and mark complete only after all expected cases and coefficient artifacts are saved. Per-case coefficients themselves are now preserved correctly (`489–493`).

6. **CORRECT — nonblocking: FOM inventory now exists at setup (round 21 item 14).**  
   `tune.py:141–146` declares every expected cohort/case/option before reference loading or expensive execution. Execution checks that inventory and saves each result (`443–461`).  
   **Fix:** retain this implementation.

7. **WRONG — nonblocking for initial tuning submission: collection does not reconcile reservations.**  
   A33.2 promises replacement by measured expenditure, but `cluster/collect.sh:23–25` only saves `sacct` text and optionally deletes the remote directory. It never updates `GPU-LEDGER.json`. Keeping full reservations is conservative, but misstates measured expenditure and can unnecessarily prevent subsequent work.  
   **Fix:** validate allocation GPU-hours, then atomically reconcile the matching attempt/job entry under the ledger lock before deletion.

8. **CORRECT — nonblocking: nominal workload, cohort routing and comparator mappings remain consistent.**  
   `make_phase2_configs.py:8–34` registers 140 base configurations per 2D cell and 1,008 in 3D, with walls of 2, 5 and 12 hours. Including spent time and Stage-B reserve gives **28.012 GPU-h**. Screening eligibility is reference-derived (`tune.py:126–153,397–405`); required grids precede probes and extension (`431–435,468–535`). Comparator mappings use P1 and speed3d queries (`ours.py:86–116,126–148`), with batch labels and replay checks retained (`tune.py:168–214`).  
   **Fix:** retain these paths while repairing the enforcement and persistence defects above.

No files were changed, no scratch was created, and no jobs, GPU work or `tests/test_core.py` ran.