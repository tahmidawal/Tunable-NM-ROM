**SUBMIT — these two registered H200 pilots only.** HEAD `809e395a9` closes round 6’s required fixes. No remaining submission blocker found.

Read-only audit: no files changed, jobs submitted, lab-log writes, or arrays allocated. Shell syntax checks—including both rendered jobs—and Python AST checks passed. Paths below are relative to `experiments/imp-box/`.

1. **Sbatch — CORRECT.**

   `cluster/job.sbatch.tmpl:3–23` specifies the `gpu` partition, one H200, runtime device-type check, absolute cluster venv interpreter, highest matmul precision, and GPU preflight exiting **42**. `cell.py:29` enables x64.

   Logs and data resolve beneath `/cluster/tufts/paralab/tawal01/imp-box/<job>/`. Unique directory reservation and execution `flock` protect against overlapping execution. Stage failures are counted and produce a failed job; early preflight failures and forced termination remain visible through scheduler accounting.

   **Hardening:** resolve and verify the allocated GPU UUID once. Numeric `CUDA_VISIBLE_DEVICES` versus `nvidia-smi` indexing remains an attribution assumption. External memory measurements are correctly labelled samples, distinct from exact peaks.

2. **submit.sh — CORRECT for these specifications.**

   `cluster/submit.sh:18–49` stages committed content and reads the template/body from the recorded commit. Approval binds job name, body SHA-256, GPU, memory and wall limit. Both hashes match the supplied values.

   **The exact approvals are already committed in HEAD**, including `240G` and the respective 12-/24-hour limits.

   Both campaign-lock acquisitions check account `<8` and lane `<3`; the second rechecks immediately before `sbatch`. Directory reservation and the submitted sentinel prevent ordinary duplicates. Queue queries occur before and after submission.

   Live permitted queries found **one account job, zero imp-box jobs**, H200 resources on `gpu`, and neither smoke directory present. Submission-time checks remain necessary.

3. **Bodies — CORRECT.**

   The bodies select **W40/A2** and **W40/A4**, respectively, and run:

   `ref → data → pod → bank → ladder → time → controls`

   Neither uses `--smoke`, reduced-size arguments or debug overrides. Defaults retain full cohorts, 300,000-step bank training and TRANS. The template clears the identified debug variables; full-size Python execution also rejects them.

   A nonzero stage exit increments `FAILED`, stops subsequent stages and fails the job. Completion markers follow successful execution. The mandatory control registry persists its outcomes and exits nonzero unless all required outcomes pass (`controls.py:198–224`).

4. **Smoke informativeness — CORRECT for the registered pilot.**

   Successful completion supplies control outcomes, stage durations, host/device peaks, external memory samples, reference gates, cohort/bank metadata, coverage/termination records and frozen identities needed for subsequent certification.

   A4 uses **N_ref=768** (`geom.py:20`; `cell.py:83–85`), with spatial, temporal, padding and retry gates. Numerical controls exercise generated cell data; selector/authentication controls include registered fixtures. Claiming that *every* control is a real-data numerical experiment would be **WRONG**.

   Runtime remains unmeasured. There is no demonstrated basis to guarantee completion within these limits or assert inevitable overrun. Completed stages and progress logs retain partial evidence.

   **Hardening:** stop earlier after unscorable references; checkpoint bank training/ordering; treat ECSW’s deadline as cooperative. OOM/timeout may prevent recording the failing stage’s exact peaks; external samples and scheduler evidence survive as partial evidence.

   **Production-only NEEDS-FIX:** `resources.py:56–69` still needs reconciliation of expected coverage against termination records and validation of memory evidence before its certificate can justify production release. That does not prevent these pilots from generating the inputs.

5. **Budget — CORRECT bounded pilot exposure.**

   | Quantity | GPU-h |
   |---|---:|
   | smk-A2 maximum | 12 |
   | smk-A4 maximum | 24 |
   | Pair maximum | **36** |
   | Remaining from 87 at zero spend | **51** |
   | Pair × 1.15 plus one additional 24-hour failure allowance | **65.4** |

   This is a defensible reservation bound, **not a measured runtime forecast or proof that the remaining campaign fits**. Subsequent releases must use collected costs and revised scope.

6. **Collection/accounting — CORRECT for critical evidence within the one-hour procedure.**

   **Round-6 fixes verified:**

   - `cluster/wait_collect.sh:18–30` establishes one **45-minute deadline** shared by accounting and critical collection. Each collection attempt—including local extraction/checksums—is enclosed by the remaining timeout. Exhaustion records unresolved status, exits nonzero and preserves the remote directory.
   - With nominal five-minute polling, query time and retry sleeps, this leaves margin within one hour. Scheduler/transport outages can still prevent successful collection; the procedure reports missing evidence rather than claiming success.
   - `cluster/collect.sh:30–31` now sets remote `pipefail`, so inventory failures propagate.
   - Critical files of every size are transferred and checksum-verified; critical-only mode deletes nothing.
   - Bulk archival remains separate and may exceed one hour. Deletion requires accounting and verified preservation.
   - `ledger.py:36–50` serializes updates and replaces JSON atomically, retaining GPU-hours, exit status and step-memory records.

   The supplied successful quoting test and the stated per-job waiter launch/PID-recording procedure address the earlier operational concerns.

**SUBMIT. Required pre-submission code fixes: none.** This verdict covers only the two exact approved smoke specifications; production and test release remain separately gated.