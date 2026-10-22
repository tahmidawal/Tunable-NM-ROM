**DO-NOT-SUBMIT.** Round 4’s accounting fix is verified, but critical-evidence collection still has an incomplete-success and potential data-loss path.

Audited HEAD `e9b4930ef`. No files changed, jobs submitted, lab-log writes, or arrays allocated. Python AST and shell syntax checks passed, including both rendered templates. Paths below are relative to `experiments/imp-box/`.

1. **Sbatch — CORRECT, with attribution hardening.**

   `cluster/job.sbatch.tmpl:3–23` specifies the GPU partition, one H200, absolute cluster interpreter, highest matmul precision, allocated-device H200 check, and GPU preflight exiting 42. `cell.py:29` enables x64.

   Logs/data stay beneath the unique paralab job directory; an execution `flock` prevents simultaneous execution there. Stage failures propagate to job failure. Early failures and SIGKILL remain accounted through Slurm.

   Host cgroup/session sampling and GPU UUID-labelled samples supplement normal-exit peaks. **Hardening:** resolve the GPU selector once to a UUID; numeric CUDA versus `nvidia-smi` identifier mapping remains an assumption.

2. **Submission — CORRECT for these specifications.**

   `cluster/submit.sh:18–49` stages committed content and binds approval to name, body hash, GPU, memory and wall limit. Both hashes match the supplied values. **The matching release approval is already committed.**

   Both campaign-lock acquisitions check account `<8` and lane `<3`; the second rechecks immediately before `sbatch`. Directory reservation and the submitted sentinel protect against ordinary duplicates. Queue queries occur before and after submission.

   Live permitted queries found **one account job, zero imp-box jobs**, H200 resources on `gpu`, and neither smoke directory present. Submission-time rechecks remain necessary.

3. **Bodies — CORRECT.**

   Both select the requested W40/A2 or W40/A4 cell and execute:

   `ref → data → pod → bank → ladder → time → controls`

   Neither uses `--smoke`, reduced-size arguments or debug overrides. Defaults retain full cohorts, 300,000-step bank training and TRANS. The template clears the three identified debug variables; full-size Python execution also rejects them.

   Nonzero stage exits increment `FAILED`, stop subsequent stages and fail the job. Completion markers follow successful execution. `controls.py:198–224` persists the mandatory registry and fails unless every required outcome passes.

4. **Smoke informativeness — CORRECT for the registered pilot.**

   Completed runs produce control outcomes, stage durations, host/JAX allocator peaks, external memory samples, reference gates, cohort/bank metadata, coverage/termination records and frozen identities.

   A4 uses **N_ref=768** (`geom.py:20`), with spatial, temporal, padding and retry gates. Numerical controls exercise generated data; selector/authentication controls also use fixtures, consistent with the audited design.

   Runtime remains unmeasured; I found no demonstrated basis to guarantee completion within 12/24 hours or assert inevitable overrun. Completed stages and progress logs provide partial evidence.

   **Hardening, not pilot blockers:** reject unscorable references before data/POD/bank; treat ECSW’s deadline as cooperative; checkpoint bank training/ordering. Production certification still needs expected-coverage reconciliation and memory validation in `resources.py`; those do not prevent generating pilot evidence.

5. **Budget — CORRECT bounded exposure; campaign projection remains unestablished.**

   | Quantity | GPU-h |
   |---|---:|
   | smk-A2 maximum | 12 |
   | smk-A4 maximum | 24 |
   | Pair maximum | **36** |
   | Remaining from 87 at zero spend | **51** |
   | Pair × 1.15 plus one 24-hour failure allowance | **65.4** |

   Both fit the budget and concurrency caps. This is a defensible reservation bound, not a measured runtime forecast. Further releases require collected costs and revised scope.

6. **Collection/accounting — NEEDS-FIX.**

   **Round-4 repairs verified:**

   - `wait_collect.sh:18–20` attempts accounting independently before collection; `collect.sh:22` also attempts it before transfers.
   - Ledger updates remain locked and atomically replaced, retaining exit codes and step-memory records.
   - Critical pulls now have a 300-second timeout.
   - Accounting and checksum gates remain before deletion.
   - The supplied remote quoting test and explicit waiter launch/PID-recording procedure close those concerns.

   **Remaining required repair:**

   - **The oversized-file predicates leave a gap.** `collect.sh:28` uses `-size -20M`; line 30 uses `-size +20M`. GNU `find` rounds upward to MiB units, so files **larger than 19 MiB through exactly 20 MiB** enter neither critical collection nor the oversized list.
   - **Oversized evidence does not prevent a false completeness claim.** Line 29 prints `critical evidence verified` before oversized detection. Detection errors are suppressed with `|| true`; a nonempty list causes no incomplete status.
   - **Large logs can be deleted without preservation.** A critical log of 100 MiB outside `out/cell` is excluded from both local pulls. The archive covers only `out/cell` (`:35–38`), yet cleanup deletes the entire job directory (`:41`). Listing the missing log does not preserve it.
   - **Critical completion lacks its own deadline/status.** The waiter retries the whole collector (`wait_collect.sh:22`). Oversized critical evidence can remain dependent on bulk transfer, and archival can delay the unresolved result beyond one hour. Bulk archival may legitimately take longer; critical-evidence completion must be tracked separately.

   **Required fix:** partition critical files without gaps; preserve every critical file before deletion; treat failed inventory or missing critical files as explicitly incomplete; and give critical collection its own retry/deadline and unresolved status within one hour. Keep bulk archival separate and retain the existing accounting/checksum deletion gates.

**DO-NOT-SUBMIT until item 6 is repaired. No additional pilot-blocking changes identified.**