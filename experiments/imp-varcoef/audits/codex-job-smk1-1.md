**Overall verdict: NOT CLEAN — GPU-hour cap enforcement has a concurrency race.** The staged job itself passes the static checks below.

| Item | Verdict | Finding |
|---|---|---|
| Partition, GPU, memory, time | CORRECT | `gpu`, one `h200`, `200G` host memory, `03:00:00`; reservation agrees. Appropriate GPU request for the 4096² bank. Actual peak memory and completion within three hours remain smoke outcomes. |
| Venv and precision | CORRECT | Absolute cluster venv; `JAX_ENABLE_X64=true`, `JAX_DEFAULT_MATMUL_PRECISION=highest`; explicit dtype/environment assertions. |
| GPU preflight | CORRECT | Prints `jax_backend`; exits **42** on non-GPU backend. `set -e` stops execution. |
| Reference store | CORRECT | `JCPREF_STORE` points to the shared paralab store; `JCPREF_META` points to staged metadata. Remote store availability was not checked. |
| Manifest and provenance | CORRECT | Verified **152 source files** against their recorded Git blobs at `3117fd59187d6412587985a72f41504027934604`; **156 manifest entries** match. No unmanifested payload files. Batch checks manifest before execution. |
| Output paths | CORRECT | Logs, outputs, temporary files and cache resolve beneath `/cluster/tufts/paralab/tawal01/imp-varcoef/smk1`. Staged `logs/` exists. |
| `smk-impl.json` | CORRECT | `small:false`; real-size implementation checks, PSI/FDMG T-loose on dev0/C100, and T-nan. |
| `smk-ref.json` | CORRECT | Both contrasts; dev0/dev3; 1024/2048/4096 levels; cross-checks and reference controls enabled. |
| `smk-ladder.json` | CORRECT | All four solver families represented; two development cases; screening, extensions and qualification enabled. |
| `smk-time.json` | CORRECT | 4096² bank, six anchor slots, two ROM targets, four FOM families, kernel/setup subjects and bank-resident pilot. All three timing controls configured. |
| Relative dependencies | CORRECT | From `$TASK_ROOT/experiments/imp-varcoef`, `../../output/...` resolves to `$TASK_ROOT/output/...`. Producers run before consumers. |
| Current T-ref fixtures | CORRECT | Six A2/A3 fixtures implemented; A4 order mutation used. Requires an accepted real reference baseline and checks that only the intended predicate fails. |
| Historical A1 controls | NEEDS-RESTATEMENT | T-ref-a/b/c and the T-loose tight-solve difference are absent. Coverage is complete **if A2/A3 supersede those A1 requirements**; avoid claiming every historical fixture runs. |
| T-dbl / T-fault / T-id | CORRECT | Separate synchronized doubled executions; injected drift; corrupted identity through a control-only alias. Firing remains to be measured. |
| Local T-gate | CORRECT | Ran staged `gate.py selftest` read-only: **12/12 fired**. |
| `smoke_accept.py` inputs | CORRECT | Exact output names, backend/completion fields, control results, timing gates, pilot evidence and log markers are supplied. Acceptance is a subsequent local step. |
| `resources.py` inputs | CORRECT | Reference solve/compile/cross-check times, ladder seconds for every family, timing elapsed and bank-build/memory records are supplied. Projection accuracy remains provisional. |
| Submission concurrency caps | CORRECT | Under the shared lock: fail-closed `squeue`, account count `<8`, lane count `<2`, fresh final directory, then `sbatch`. Queue checks occur before and after submission. |
| Submission lock description | NEEDS-RESTATEMENT | Header claims transfer and manifest verification occur inside the lock; both actually precede it in `.incoming`. Final-directory move and submission are locked. |
| GPU-hour cap / ledger | **WRONG** | Budget check occurs before locking; ledger recording occurs after unlocking. Concurrent different-job submissions can both pass against the same committed total. Atomic file replacement does not prevent lost ledger updates. |

The blocking issue is in [submit.sh](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/cluster/submit.sh:12): serialize the budget check, reservation and ledger update across concurrent submissions.

No files changed, no jobs submitted, and no lab-log writes.