**J0: SUBMIT. J5: SUBMIT.** Within this narrow audit, no blocking findings.

- **Deadline/arithmetic:** Allowance is evaluated after contributions. Initial and retry each receive the halved allowance, budgeting for both. Contributions, `ecsw_weights`, and kappa-4/8 paths retain baseline arithmetic. This is a cooperative time budget, not a hard combined deadline.
- **Timers:** Source records retain `snap_seconds`, `pod_seconds`, `codes_seconds`, `states_seconds`, and per-rank ECSW `seconds`, plus contribution/NNLS timers required for accounting.
- **Staging:** Manifest checks passed for 139 J0 / 144 J5 small files. All 136 checked provenance files per job match HEAD. Both certificate config hashes match local and staged configs. Batch syntax passes.

Recorded source commit `fc9de7e5d` is consistent with HEAD `8d356b3e0`’s restaging-only change. A9.2 treated as non-blocking.

Four large files per job were skipped. No files changed, jobs submitted, arrays loaded, or LAB-LOG/main writes performed.