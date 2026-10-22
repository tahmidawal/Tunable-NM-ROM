**DO-NOT-SUBMIT — collection/accounting still has a round-3 blocker.** Execution, full-size configuration and bounded pilot budget pass.

Audited HEAD `f5c6065ec`. No files changed, jobs submitted, lab-log writes or arrays allocated. Python AST and shell syntax checks passed, including both rendered templates. Paths below are relative to `experiments/imp-box/`.

1. **Sbatch — CORRECT, with attribution hardening.**

   `cluster/job.sbatch.tmpl:3–32` provides the GPU partition, one H200, absolute venv interpreter, highest matmul precision, GPU preflight exiting 42, unique paralab output directory and execution lock. `cell.py:29` enables x64.

   **Round-3 repair:** runtime type checking and memory sampling now explicitly select the first `CUDA_VISIBLE_DEVICES` entry; they no longer inspect an arbitrary first GPU. Samples record UUIDs. Host sampling uses cgroup memory with a process-session RSS fallback, and labels measurements as samples.

   Normal stage failures propagate to a failed job; early failures and SIGKILL remain dependent on scheduler accounting.

   **Hardening:** resolve the selector once to a UUID and reuse that UUID. The current implementation still assumes numeric CUDA and `nvidia-smi` device identifiers agree; this audit did not validate that mapping inside an allocation.

2. **submit.sh — CORRECT for these specifications.**

   `cluster/submit.sh:18–49` stages committed content and binds approval to name, body hash, GPU, memory and wall limit. Both supplied hashes match. **The matching release approval is already committed.**

   Both campaign-lock acquisitions check account `<8` and lane `<3`; the second rechecks immediately before submission. Directory reservation and the submitted sentinel prevent ordinary duplicates. Queue queries occur before and after submission.

   **Round-3 repair:** the comment now accurately describes two lock acquisitions.

   Live permitted queries found one account job, zero imp-box jobs, H200 availability on `gpu`, and neither proposed job directory. Submission must still perform its own rechecks.

3. **Bodies — CORRECT.**

   Both bodies select the requested W40/A2 or W40/A4 cell and run:

   `ref → data → pod → bank → ladder → time → controls`

   Neither uses `--smoke`, reduced-size arguments or debug environment overrides. Defaults retain full cohorts, bank training and TRANS. The template clears the three identified debug variables; full-size Python execution also rejects them.

   Nonzero stage exits increment `FAILED`, stop subsequent stages and fail the job. Completion markers follow successful execution. `controls.py:198–224` explicitly checks the mandatory registry, persists failures and exits nonzero unless all required outcomes pass.

4. **Smoke evidence — CORRECT for a completed pilot; runtime remains unmeasured.**

   Successful runs provide control outcomes, stage durations, host/JAX allocator peaks, external memory samples, reference gates, cohort/bank metadata, coverage/termination records and frozen identities.

   A4 uses **N_ref=768**, with spatial, temporal, padding and retry gates. Numerical controls exercise generated data; selector/authentication controls also use fixtures, as permitted by the audited design.

   These jobs generate certificate inputs; they do **not** establish that `resources.py` is production-ready. Expected-coverage reconciliation and memory validation remain production-release requirements.

   **Hardening, not pilot blockers:**

   - Stop immediately after unscorable references; currently data/POD/bank can run before the rejection at `cell.py:435`.
   - Treat the ECSW deadline as cooperative.
   - Preserve completed-stage evidence and progress logs if bank training/ordering times out.

   I found no demonstrated basis to guarantee completion within 12/24 hours or declare an inevitable overrun. Missing measurements are the purpose of this registered pilot, not an additional blocker.

5. **Budget — CORRECT bounded exposure; campaign projection unestablished.**

   | Quantity | GPU-h |
   |---|---:|
   | smk-A2 maximum | 12 |
   | smk-A4 maximum | 24 |
   | Pair maximum | **36** |
   | Remaining from 87, given zero spend | **51** |
   | Pair × 1.15 plus one 24-hour failure allowance | **65.4** |

   Both fit the lane budget and concurrency limits. This is a defensible allocation bound, not a measured runtime forecast. Further releases require collected costs and revised scope.

6. **collect.sh / ledger.py — NEEDS-FIX for the one-hour requirement.**

   **Round-3 fixes verified:**

   - Critical evidence precedes bulk transfer; `code/` and pull bookkeeping are excluded. The supplied successful remote quoting/checksum test closes that concern.
   - `collect.sh:28` requires a successful ledger update before cleanup.
   - Ledger updates remain locked and atomically replaced, retaining allocation exit codes and step memory records.
   - The waiter retries and reports unresolved collection with a nonzero exit.
   - Your explicit launch-and-record procedure closes the monitor-wiring requirement; no monitor is expected before submission.

   **Remaining blocker: critical transfer still gates accounting, and retry timing exceeds one hour.**

   `collect.sh:26–28` records accounting only **after** the critical pull succeeds. `wait_collect.sh:19` permits four 900-second pulls, with 180-second delays. Three timed-out attempts plus delays already consume **54 minutes before attempt four**, excluding polling and other SSH calls. Accounting can consequently remain absent past one hour despite reachable scheduler records.

   Also, `collect.sh:26` silently excludes critical files at or above 20 MiB. Such files move into the later bulk phase without an explicit “critical evidence incomplete” result.

   Bulk archival taking longer than an hour is **not itself a blocker once critical evidence and accounting are safely recorded**.

**Required fixes:** persist accounting independently of transfer success; give critical-evidence collection its own retry/deadline and unresolved status within the one-hour window; explicitly handle oversized critical files rather than silently deferring them. Retain the existing accounting/checksum gates before deletion.

**DO-NOT-SUBMIT until that collection/accounting repair is reviewed.**