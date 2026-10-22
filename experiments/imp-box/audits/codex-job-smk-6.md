**DO-NOT-SUBMIT.** Round 5’s preservation fix is substantially implemented, but its required one-hour critical-collection deadline remains unenforced.

Audited HEAD `30e0c6d58`, read-only. No files changed, jobs submitted, lab-log writes, or arrays allocated. Python AST and shell syntax checks passed, including both rendered templates. Paths below are relative to `experiments/imp-box/`.

1. **Sbatch — CORRECT, with attribution hardening.**

   `cluster/job.sbatch.tmpl:3–23` specifies `gpu`, one H200, the absolute cluster interpreter, highest matmul precision, allocated-device H200 check, and GPU preflight exiting 42. `cell.py:29` enables x64.

   Logs/data remain under the unique paralab job directory. Execution is locked; stage failures produce nonzero job status. Early preflight failures and process kills remain visible through scheduler accounting.

   Normal-exit host/device peaks are supplemented by cgroup/session host samples and GPU UUID-labelled samples. **Hardening:** resolve the CUDA selector to a verified allocated UUID once; numeric CUDA versus `nvidia-smi` indexing remains an assumption.

2. **submit.sh — CORRECT for these specifications.**

   Committed-tree staging, committed template/body reads, and approval binding to name, body SHA-256, GPU, memory and wall limit are present (`cluster/submit.sh:18–49`). Both hashes match the supplied values. **Matching approval is already committed in HEAD.**

   Both campaign-lock acquisitions check account `<8` and lane `<3`; submission rechecks immediately before `sbatch`. Directory reservation and the submitted sentinel prevent ordinary duplicates. Queue checks occur before/after submission.

   Live permitted queries found **one account job, zero imp-box jobs**, H200 resources on `gpu`, and neither smoke directory present. Submission-time rechecks remain necessary.

3. **Bodies — CORRECT.**

   Both select the requested W40/A2 or W40/A4 cell and execute:

   `ref → data → pod → bank → ladder → time → controls`

   Neither contains `--smoke`, reduced-size arguments or debug overrides. Defaults retain full cohorts, 300,000-step bank training and TRANS. Identified debug environment variables are cleared by the template and rejected by full-size Python execution.

   A nonzero stage exit increments `FAILED`, stops subsequent stages and fails the job. Completion markers follow successful execution. `controls.py:198–224` persists the mandatory registry and exits nonzero unless all required outcomes pass.

4. **Smoke informativeness — CORRECT for the registered pilot.**

   Successful jobs supply control outcomes, stage durations, host/device peaks, external memory samples, reference gates, cohort/bank metadata, coverage/termination records and frozen identities needed for subsequent certification.

   A4 selects **N_ref=768** (`geom.py:20`; `cell.py:83–85`), with spatial, temporal, padding and retry gates. Numerical controls exercise generated data; selector/authentication controls include registered fixtures. “Every control on real data” would overstate that distinction.

   Runtime remains unmeasured. I found no demonstrated basis to guarantee completion within 12/24 hours or assert inevitable overrun. Completed stages and progress logs provide partial evidence.

   **Hardening, not pilot blockers:** stop immediately after unscorable references, checkpoint bank training/ordering, and treat ECSW’s deadline as cooperative. Expected-coverage reconciliation and memory validation in `resources.py` remain prerequisites for trustworthy **production certification**, not for generating pilot evidence.

5. **Budget — CORRECT bounded pilot exposure.**

   | Quantity | GPU-h |
   |---|---:|
   | smk-A2 maximum | 12 |
   | smk-A4 maximum | 24 |
   | Pair maximum | **36** |
   | Remaining from 87 at zero spend | **51** |
   | Pair × 1.15 plus one 24-hour failure allowance | **65.4** |

   This is a defensible allocation bound, not a measured runtime forecast or proof that the remaining campaign fits. Subsequent releases must use collected costs and revised scope.

6. **Collection/accounting — NEEDS-FIX.**

   **Round-5 repairs verified:**

   - Byte-exact predicates partition critical files without gaps: `<20 MiB` and `≥20 MiB` (`collect.sh:32–33`).
   - Both groups are transferred and checksum-verified before the completeness marker.
   - Critical-only mode deletes nothing.
   - Job-level files of any size receive a separate verified archive before deletion (`:49–53`).
   - Ledger updates remain locked and atomic, retaining allocation cost, exit status and step-memory records.
   - The supplied quoting test and waiter launch/PID-recording procedure address those earlier concerns.

   **Remaining blocker — the deadline claim is WRONG.** `wait_collect.sh:21–24` says “within ~40 min” but has no elapsed-time deadline. Three attempts permit **75 minutes of transfer timeout allowances alone**: `3 × (300 + 1200)` seconds. Inventory, scheduler queries, accounting, sleeps and completion detection add further time. Local extraction/checksum work also lacks an enclosing deadline. Consequently, unresolved critical collection need not be reported within one hour.

   **Additional incomplete repair:** the inventory command (`collect.sh:30–31`) lacks remote `pipefail`; `find` failure can be hidden by successful `sort`. A local read-only reproduction confirmed that pipeline returns zero. Later pull pipelines have `pipefail`, reducing the risk, but inventory failure itself still does not reliably produce the promised `INCOMPLETE` status.

**Required fixes:** enforce one overall deadline for accounting and critical collection, including retries and local verification, allowing for polling delay; report unresolved status and retain the remote directory when it expires. Make remote inventory generation propagate pipeline failures. Keep bulk archival separate and preserve the existing checksum/accounting deletion gates.

**DO-NOT-SUBMIT until these remaining round-5 collection requirements are repaired.**