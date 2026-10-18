**Overall: FIX-FIRST.** The staged payload is intact and runnable, but it omits a registered control and has gaps in acceptance and cap enforcement.

| Item | Verdict | Finding |
|---|---|---|
| 1. Batch script | **CORRECT** | `jd_smk3`, `gpu` partition, one H200, two-hour limit, correct activated venv, x64/highest precision, GPU preflight exiting 42, manifest verification, and output/cache/temp paths under the specified paralab namespace. |
| 2. Staged provenance and dependencies | **CORRECT** | All **157** provenance entries match commit `c0fe7edd436760a61bc62b12a643008b2985a012` byte-for-byte. Manifest passes. Staged driver imports succeed without importing project modules outside staging. W1024, M2, all three required rule archives, and the fixture are present; bank/rule SHA pins match. |
| 3. Smoke evidence and acceptance | **WRONG** | The explicitly requested heavy paths are present, but DESIGN §7’s 3D Smolyak must-fail control is missing. Acceptance also under-enforces adaptive evidence. |
| 4. Reservation and caps | **WRONG / NEEDS-RESTATEMENT** | The 55 GPU-hour reservation arithmetic and normal concurrency checks work. The locked campaign check fails open on `squeue` failure, and the two-hour reservation does not preserve the recorded pre-production smoke allowance. |

The concrete fixes are:

1. **Add the missing 3D `smol8` control and its acceptance predicate.** [DESIGN.md:333](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/DESIGN.md:333) requires W512 κ=4 to produce Q1 fail or Q2 fail. The staged config instead uses `smolyak13` only in `fixgate`; that supplies neither the registered fixture nor its predicate. Neither A1–A4 nor the acceptance script replaces this requirement.

2. **Strengthen adaptive smoke acceptance.** [smoke_accept.py:193](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/smoke_accept.py:193) requires only that some square and default adaptive files exist. It does not require the κ=8 inventory or both forms, and records G-cert through non-required `rep(...)` calls. Consequently, failed G-cert can coexist with `ACCEPTED`. Require the exact expected inventory and certificate-consistency checks; keep legitimate scientific qualification failures distinct from implementation failures.

3. **Make the locked account-cap check fail closed.** [submit.sh:35](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/cluster/submit.sh:35) runs `squeue | wc -l` inside `bash -c` without error propagation. If `squeue` fails, `wc` can return zero and permit submission. Explicitly check the queue query’s status inside the shared lock.

4. **Resolve the pre-production smoke allowance.** The canonical [LAB-LOG.md:19221](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md:19221) allows ≤2 GPU-hours of smokes while C3 remains outstanding. `smk2` has already consumed positive time; reserving another full two hours cannot guarantee that bound. Waiting for `smk2` ends overlap, not cumulative usage. Reconcile its actual usage and fit the remaining allowance, or record an authorized allowance change.

The requested evidence paths otherwise check out: independent M-override comparison; both default-M reproduction runs, with matching keys found in speed3d archives; W1024 Gauss/lattice tails with ADT anchors; κ=1/8 GAL and LSPG certification outputs; heavy 64-case round plus under-resolution control; fixgate; T1 and both heavy timing subjects; uncertified-panel rejection; and memory peaks. The writer’s output names match `smoke_accept.py`’s archive paths.

The current lane ledger would charge **28.5 GPU-hours**, including pending core work and both smoke reservations, below 55. Read-only queue inspection showed `jd_smk2` and one HRB job running.

No files changed, jobs submitted, GPU computations run, or scratch created.