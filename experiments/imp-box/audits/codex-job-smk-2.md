**DO-NOT-SUBMIT.** Most round-1 fixes are present, but failure-memory attribution and reliable collection/accounting remain incomplete. I also reproduced a concurrent ledger update losing one job.

Audited HEAD `53e550a3e` and the on-disk `RELEASE.json`. No files changed, jobs submitted, or lab-log writes made. Shell syntax checks passed for the scripts and both rendered job templates.

1. **Sbatch — CORRECT execution configuration; NEEDS-FIX failure-memory evidence.**

   [job.sbatch.tmpl](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/cluster/job.sbatch.tmpl:1) correctly provides:

   - `gpu` partition, one H200, runtime H200 check.
   - Absolute cluster venv interpreter.
   - `JAX_DEFAULT_MATMUL_PRECISION=highest`; `cell.py:29` enables x64.
   - GPU preflight exiting 42.
   - Logs/data beneath the unique lane job directory and an execution `flock`.
   - Stage exit-code accounting and final nonzero exit on failure. Early hardware/preflight failures still rely on Slurm accounting.

   **Round-1 memory fix is only partial.** `record_peak()` now correctly uses `null` for unavailable device peaks, and external samples survive a Python-process failure. However, the sampler’s `free -m` reports **whole-node memory**, not this job’s host usage. Its GPU query takes the first reported device without recording its UUID or establishing allocation identity. Thirty-second samples are also sampled usage, not exact peaks.

   **Required:** collect job/cgroup or process-tree host memory and identify the allocated GPU in device samples. Preserve scheduler failure/exit and available memory accounting, including batch/step records. Label samples separately from measured peaks.

2. **submit.sh — CORRECT for these specifications.**

   [submit.sh](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/cluster/submit.sh:18) stages the committed tree and reads both template and body from that commit. Approval binds job name, body hash, GPU, memory and wall limit.

   Both body hashes exactly match the supplied values and on-disk approval. The approval remains uncommitted; the script correctly blocks until that changes.

   Reservation and submission use **two separate acquisitions** of the shared campaign lock, with both caps rechecked immediately before `sbatch`. Directory reservation and the submitted sentinel protect against ordinary duplicate submissions. Queue checks occur before and after submission.

   Live permitted queries found **one account job, zero imp-box jobs**, H200 resources on `gpu`, and neither smoke directory present. Thus the pair fits the concurrency caps; submission must still perform its live rechecks.

3. **Bodies — CORRECT.**

   The bodies select W40/A2 and W40/A4 and run `ref,data,pod,bank,ladder,time,controls`. They contain no `--smoke` or size overrides. Defaults retain 96 training cases, full 300,000-step bank training and TRANS.

   A nonzero stage exit increments `FAILED`, stops subsequent stages and fails the job. Mandatory controls write their registry and exit nonzero unless all required outcomes pass.

   **Round-1 fixes verified:**

   - Unscorable references now cause explicit exit 3 before selection/control execution, preventing false `JOB DONE`.
   - The template clears all three identified debug variables; `cell.py` also rejects them in full-size execution.
   - Completion markers are written only after successful stage execution.

   One budget concern remains: the unscorable check occurs at `ladder`, so `data`, `pod` and `bank` can still consume substantial time after `ref` has already established that the smoke cannot complete. Moving that failure earlier would avoid wasted work.

4. **Smoke evidence — CORRECT successful-run coverage; NEEDS-FIX failure evidence.**

   Completed jobs would supply the intended control registry, stage times, normal-exit host/device peaks, reference gates, cohort identity, bank metadata, coverage/termination records and frozen identities. A4 selects **N_ref=768**, including spatial, temporal, padding and retry checks.

   As registered, numerical controls use generated cell data; selector/authentication controls include fixtures. “Every must-fail control” should not be interpreted as every registry entry being a physical-data experiment.

   **Runtime remains unmeasured.** There is no demonstrated basis to guarantee completion within either wall limit—or to assert inevitable overrun. Earlier completed stages and bank progress logs provide partial evidence, but bank training/ordering does not save a completed model until it finishes.

   The new ECSW deadline check improves construction budgeting, but the comment calling it a **hard budget is WRONG**: substantial matrix products and allocation occur before the first check, and individual operations between checks remain uninterruptible. Treat 1,800 seconds as a cooperative limit.

   The `resources.py` follow-up remains **production-only**: expected coverage is not reconciled against every termination record, and memory evidence is not validated. These jobs can generate certificate inputs without themselves proving that the later certificate is sound.

5. **Budget — CORRECT bounded pilot exposure; campaign affordability unestablished.**

   | Quantity | GPU-h |
   |---|---:|
   | smk-A2 maximum | 12 |
   | smk-A4 maximum | 24 |
   | Pair maximum | **36** |
   | Remaining from 87 | **51** |
   | Pair × 1.15 plus one 24-hour failure allowance | **65.4** |

   With stated spend zero, these allocations fit the cap. This is a defensible reservation bound, **not a measured runtime projection**. Subsequent releases must use collected costs and revised scope.

6. **Collection/accounting — NEEDS-FIX.**

   **Round-1 repairs verified:** `smk-*` names can collect early failures; `sacct -P` replaces width-limited display output; both pull bookkeeping files are excluded from checksums. Archive/local verification still precedes deletion.

   Remaining blockers:

   - **The one-hour procedure is incomplete.** [collect.sh:23](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/cluster/collect.sh:23) allows one hour for bulk archival **before** pulling logs, followed by a further 30-minute transfer allowance. Small failure evidence is not collected first.
   - **The waiter hides failures.** [wait_collect.sh:14](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/cluster/wait_collect.sh:14) proceeds after ledger failure, attempts collection once, then exits zero even if collection failed. It has no retry/escalation path. Its default deadline starts before queue waiting, so it can expire before a queued job completes.
   - **Concurrent ledger updates lose accounting.** [ledger.py:33](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/ledger.py:33) performs an unlocked read/modify/write. An in-memory reproduction using the actual `add()` function and two one-GPU-hour records retained only one record and reported **1 GPU-h instead of 2**. Direct writes also permit partially written JSON.
   - **Scheduler failure evidence is insufficient.** The ledger retains allocation-level state/time/TRES/node, but not exit codes or batch/step memory accounting.

   **Required:** serialize ledger updates and replace the file atomically; collect accounting, logs, markers and memory samples before bulk archival; make monitoring retry failures and report nonzero/unresolved status; ensure monitoring persists through queue wait and completion. Establish and record the active monitor for each submitted job.

**DO-NOT-SUBMIT until the memory-attribution and collection/accounting fixes above are completed and reviewed.** Committing the current approval alone does not close these blockers.