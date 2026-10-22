**J0: FIX-FIRST. J5: FIX-FIRST.**

| Blocking item from codex-job-8 | Round 4 verdict |
|---|---|
| J0 resource certificate | **NOT FIXED.** Mostly historical hard-coded timings, rather than measured smoke phases scaled to current workloads. The additional 60 seconds is unexplained; contribution costs remain unitemized. |
| J5 resource certificate | **NOT FIXED.** Historical whole-job times ×1.3 replace phase/workload projections; the offline checker’s 120 seconds per mesh remains unmeasured. |
| J0 incremental artifacts and terminal states | **FIXED.** Atomic rule replacement and persisted pending `UNRESOLVED` entries retained. |
| J0 missing `seconds` crash | **FIXED.** Access is guarded. |
| J0 path versus outer budget | **NOT FIXED.** Budget is still computed before contributions and reused for both NNLS attempts. Subtracting 300 seconds assumes a contribution bound; it does not enforce a shared deadline. |
| J0 construction burn-ins | **FIXED**, retained. |
| J5 finer-rule non-convergence | **FIXED**, retained. |
| J5 independent acceptance and descriptor identity | **FIXED**, retained. Staged checker copies match their sources. |

**J5’s 4096 timeout mismatch is fixed under the revised estimate**, but the same calculation now exposes shorter-mesh mismatches:

| Mesh | Certificate’s solver estimate ×1.5 | Command timeout |
|---|---:|---:|
| 256 | 3,081 s | 1,500 s |
| 1024 | 2,593 s | 2,400 s |
| 4096 | 2,373 s | 3,600 s |

The 256 estimate exceeds its timeout even **before** the safety factor: 2,054 seconds. The 165-minute allocation does not override individual command timeouts.

Before submission:

- **J0:** Recompute remaining time after contributions and enforce one deadline across initial NNLS, retry and persistence. Provide traceable phase/workload estimates.
- **J5:** Reconcile per-mesh estimates with command timeouts and substantiate checker costs.

S1c records all controls passing. Small-file manifest checks passed for **139 J0 / 144 J5** files; certificate config hashes and batch syntax passed. Large files were skipped.

No files changed, jobs submitted, large arrays loaded, or LAB-LOG/main writes performed.