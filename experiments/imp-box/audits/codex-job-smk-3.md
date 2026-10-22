**DO-NOT-SUBMIT.** HEAD `22ee8036e` closes several round-2 findings, but allocated-GPU attribution and reliable one-hour collection/accounting remain incomplete.

Read-only audit: no files changed, jobs submitted, lab-log writes, or large arrays allocated. Both rendered jobs and shell scripts passed syntax checks; Python AST checks passed. Both body hashes match the supplied values and the **committed** release approval.

1. **Sbatch — CORRECT execution configuration; NEEDS-FIX memory attribution.**

   `cluster/job.sbatch.tmpl:3–22` specifies the GPU partition, one H200, absolute cluster venv, highest matmul precision, GPU preflight exiting 42, unique paralab paths and execution lock. `cell.py:29` enables x64. Stage failures propagate to a nonzero job exit; early failures remain visible through Slurm.

   Round-2 host-memory repair is present: lines 25–28 read the shell’s cgroup memory counter and explicitly label measurements as samples. UUIDs are now recorded.

   **Remaining blocker:** line 28 still uses an unrestricted `nvidia-smi ... | head -1`. Recording `CUDA_VISIBLE_DEVICES` separately does not establish that the sampled UUID belongs to the allocation. The runtime H200 check similarly searches all reported devices. The live inventory reports eight-GPU H200 nodes, making this distinction relevant.

   **Required:** resolve and verify the allocated GPU UUID, then query that UUID explicitly for both type and memory. Validate the cgroup counter or provide a job-attributed fallback; currently an unresolved counter silently yields `na`. Thirty-second samples must remain distinguished from exact peaks.

2. **submit.sh — CORRECT for these specifications.**

   `cluster/submit.sh:18–49` stages committed content using `git archive`, reads the body/template from that commit, and binds approval to name, body hash, GPU, memory and wall limit.

   Reservation and submission use **two separate campaign-lock acquisitions**, with both account `<8` and lane `<3` checks repeated immediately before submission. Directory reservation and the submitted sentinel prevent ordinary duplicate submission. Queue checks occur before and after.

   Live permitted queries found **one account job, zero imp-box jobs**, and neither proposed directory exists. These observations do not replace submission-time rechecks.

   **Hardening:** correct the introductory claim that one lock acquisition covers reservation through submission. Approval does not calculate or reserve GPU-hours automatically.

3. **Bodies — CORRECT.**

   Both bodies select the requested W40/A2 or W40/A4 cell and run `ref,data,pod,bank,ladder,time,controls`. Neither includes `--smoke` or size overrides. Defaults retain full cohorts, 300,000-step bank training and TRANS.

   Nonzero stage exits increment `FAILED`, stop subsequent stages and fail the job. The mandatory controls registry writes its outcomes and exits nonzero unless all required outcomes pass (`controls.py:198–223`).

   Earlier fixes remain intact: debug variables are cleared and rejected in full-size execution; unscorable references produce exit 3 rather than false successful completion; completion markers follow successful execution.

4. **Smoke evidence — CORRECT successful-run coverage; NEEDS-FIX failure evidence.**

   Completed jobs provide control outcomes, stage durations, normal-exit host/device peaks, reference gates, cohort/bank metadata, coverage/termination records and frozen identities needed as certificate inputs. A4 selects **N_ref=768**, with the registered reference checks. Numerical controls use generated data; selector/authentication controls also use fixtures, as the code audits describe.

   Failure-memory evidence still needs item 1’s repair. Python `finally` cannot record peaks after SIGKILL; external samples and scheduler accounting are essential.

   **Runtime is unmeasured.** There is no demonstrated basis to guarantee completion within 12/24 hours or declare an inevitable overrun. Completed stages and bank progress logs provide partial evidence; the completed bank is saved only after training and ordering.

   **Hardening, not additional pilot blockers:**

   - Fail immediately after unscorable references: currently `data`, `pod` and `bank` can run before the check at `cell.py:435`.
   - Treat ECSW’s 1,800-second deadline as cooperative; operations between checks can exceed it.
   - Before production release, finish expected-coverage/termination reconciliation and memory validation in `resources.py`. These remain production-certificate requirements, consistent with code audit 6.

5. **Budget — CORRECT bounded pilot exposure; full-campaign projection unestablished.**

   | Quantity | GPU-h |
   |---|---:|
   | smk-A2 maximum | 12 |
   | smk-A4 maximum | 24 |
   | Pair maximum | **36** |
   | Remaining from 87, given zero spend | **51** |
   | Pair × 1.15 plus one 24-hour failure allowance | **65.4** |

   Submitting both fits the stated cap. This is a defensible **reservation bound**, not a measured runtime projection or proof that the remaining campaign fits. Later releases must use collected costs.

6. **collect.sh / ledger.py — NEEDS-FIX.**

   Round-2 repairs verified:

   - **CORRECT:** `ledger.py:43–50` locks the entire read/modify/write and atomically replaces JSON, closing the lost-update defect.
   - **CORRECT:** accounting now requests allocation exit codes and batch/step state, exit codes and MaxRSS.
   - **CORRECT:** local-file collection precedes bulk archival; checksum verification precedes deletion.
   - **CORRECT:** waiter retries failures and returns nonzero when unresolved.
   - **CORRECT, qualified:** its running deadline starts at first observed RUNNING, rather than invocation. A separate 96-hour ceiling can still expire during queue waiting.

   Remaining blockers:

   - **Accounting failure does not prevent cleanup.** `wait_collect.sh:19–22` proceeds after all ledger retries fail; `collect.sh:32` can then delete the remote directory. Scheduler records may remain recoverable, but no durable automatic accounting-recovery path is established.
   - **One-hour collection remains unreliable.** The first transfer selects *every* file below 100 MB, not a bounded critical-evidence set. Aggregate size is unbounded. A failed transfer can consume 30 minutes, then another retry plus its ten-minute delay already exceeds an hour. Bulk archival can additionally occupy an hour before the waiter retries or reports failure.
   - **Monitoring is documented, not wired into submission.** Submission does not launch it; the heartbeat contains no collector monitors. This is expected before submission, but launch-and-record must be an explicit submission step.

   **Required:** persist accounting and critical logs/markers/memory evidence independently, retry that phase within the one-hour window, and prohibit remote deletion until accounting is durably recorded. Separate bulk archival from critical-evidence retries. Record an active monitor for each submitted job, with explicit unresolved-status handling and monitoring renewal if its ceiling is reached.

**DO-NOT-SUBMIT until allocated-GPU sampling is verified and the collection/accounting blockers are closed.** The committed approval and valid budget reservation do not close those round-2 requirements.