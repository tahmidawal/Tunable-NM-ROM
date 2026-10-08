Read-only re-audit of `fff4cc20d`; no files modified, jobs submitted, or GPU work performed. Syntax checks, isolated binding/failure probes, both config-generator comparisons, and all 77 pinned local reference-file checksums passed.

| Item | Verdict | Assessment |
|---|---|---|
| 1 | **CLOSED** | Normalization, shared-node restriction, and evolved-time indexing remain correct. |
| 2 | **CLOSED** | M-prefix slicing remains mathematically correct, without establishing bitwise parity of complete rollouts. |
| 3 | **CLOSED** | Eligibility, ordered m* selection, tolerances, and separately named diagnostics remain correct. |
| 4 | **NOT-CLOSED** | Eligibility was correctly removed from control discrimination, but negating `>` incorrectly classifies NaN metrics as passing. |
| 5 | **CLOSED** | The committed configs now pin all 76 reference NPZs plus `result.json`, and the runner verifies their file hashes before use. |
| 6 | **NOT-CLOSED** | Solver/cold bindings and burn compilation are fixed, and offline paired-ratio computation is accepted, but job-wide K-time withdrawal remains absent and the smaller-m fallback changes the registered deployment rule. |
| 7 | **NOT-CLOSED** | Rank zero is guarded and `check_passed` is recorded, but a failed consistency check still leaves populated floors without an enforced unavailable/rejected state. |
| 8 | **CLOSED** | The reached-state Jacobian formula remains correct and its full singular-value spectrum is now persisted. |
| 9 | **NOT-CLOSED** | Dangling rule references are released correctly, but sampling still crosses interval boundaries and retained final-panel blocks/slice allocations lack explicit aggregate accounting. |
| 10 | **ACCEPTED-AS-DOCUMENTED** | The inherited 8 MiB parameter capture is accepted as proposed, nonfinite Jacobians now bypass SVD, and recompilation must remain included in smoke-based budget calibration. |
| 11 | **NOT-CLOSED** | The shared lock improves serialization, but its queue recheck fails open, submitted job ID/directory verification is absent, and production budget enforcement remains absent. |
| 12 | **NOT-CLOSED** | Deferring `audit_w.py` until before reporting is accepted as an acceptance gate, but cleanup’s scheduler check fails open and partial collection still overwrites `OUTPUTS.sha256`. |

The changes introduce these additional problems:

- **Submission guard fails open:** [submit.sh:17](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/cluster/submit.sh:17) masks SSH/`squeue` failure with `|| true`; an isolated failing-command probe produced `N2=0` and passed the cap check. Capture successful scheduler output first, then count names.
- **Cleanup guard fails open:** [collect.py:36](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/cluster/collect.py:36) likewise converts a failed remote `squeue` into `0`; with a completion marker already present, deletion can proceed without knowing whether the job is terminal.
- **NaN controls invalidate selection incorrectly:** [w2d.py:517](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w2d.py:517) uses `not(value > threshold)` as “passes”; require finite values and explicit `<=` comparisons.
- **Deployment fallback is unregistered:** [w2d.py:567](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w2d.py:567) selects smaller m after K-time failure, still attaches `median_ms`, and permits final timing; this needs an explicit diagnostic-only designation plus job-wide timing withdrawal before cost claims.
- **Sampler locking is incomplete:** [w2d.py:120](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/w2d.py:120) reads memory before acquiring the lock, so a pre-reset sample can enter the next interval; there are also no synchronous boundary samples.

The lambda defaults are correct: the isolated probe retained `(solver, data, cold)` as `(512,512,512)` and `(128,128,128)` after outer variables changed. Final-panel defaults also bind correctly. The `del`/`None` assignments do not invalidate retained callables or introduce an observed unbound-variable path.

The EXIT trap’s quoting is correct: `$NS` expands when the trap executes. Interrupted SSH or failed cleanup can leave a stale lock, which blocks subsequent submissions rather than bypassing the cap.

The reporting/acceptance gaps above need closure before accepting results; calibrated production budgets necessarily follow the smoke.

**OK to run the real-size cluster smoke: NO — blocking fix: make the under-lock scheduler recheck fail closed on SSH or `squeue` failure.**