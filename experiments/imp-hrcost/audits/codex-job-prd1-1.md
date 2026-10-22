**Overall: SUBMIT.** No production-verdict-invalidating WRONG item found in this staged attempt.

| Item | Verdict | Finding |
|---|---|---|
| Committed-tree equality | **CORRECT** | All 105 staged source/input files match commit `5af988b57` (current HEAD), including provenance sizes/hashes. Expected staging set is complete. |
| Manifest | **CORRECT** | All 109 entries verify; no unmanifested payload files. |
| Production config | **CORRECT** | 1024², r=384, frozen policy, E5 CN at dt₀=0.005, all38, Q2 sub16, three fit families, timing all38 with 5 repetitions/3 burn-ins. |
| Classical ladders | **CORRECT** | Gauss 64,80,96,112,128,160,192; Fibonacci 4181,6765,10946,17711,28657. Eight candidates ≤16384, with successor/check coverage. |
| F2 search | **CORRECT** | All candidates receive Q2. Q1 proceeds in increasing point count for potentially qualifying candidates until the first qualification in each reading; Gauss128 is always evaluated. Q2 failures resolve candidates; unresolved smaller candidates prevent an identified minimum. |
| Stop-rule evidence | **CORRECT** | Driver saves bank operators, Q1 trajectories/anchors/residuals, Q2 own/successor/check outputs, fitted supports/weights, and timing evidence consumed by `hscore.py`. Missing fits remain incomplete evidence. |
| C5′ | **CORRECT** | Production independently runs original/doubled/quadrupled Jacobian benchmarks despite `controls:false`, saving repetition arrays and `jac_bench.ratio_quad`. Speed predicates require ratio_quad ≥1.5. |
| Timing | **CORRECT** | Same-job A–B–A, immediate burn-in, repetition arrays, outlier counts, and timed/scored coefficient agreement gates are implemented. |
| E5R | **CORRECT** | Staged E5 source matches pinned SHA256 `3626350c…cbec5`; registered production/tight/tighter tolerances and table gates are present. |
| Smoke evidence | **CORRECT** | Acceptance records complete analysis coverage, no offline disagreements/G-cert failures, and required controls except C5. A9 explicitly replaces that failed fixture with production C5′. |
| Previous acceptance blocker | **CORRECT** | Empty/unrelated-path analysis is now rejected; actual supplied analysis covers the smoke’s arms. Source commit is recorded separately from the stepper hash. |
| Independent-control wording | **NEEDS-RESTATEMENT** | “Every predicate recomputed from raw outputs” remains overstated. Several checks still consume in-job assertions/comparisons; acceptance now explicitly identifies these. |
| Smoke analysis freshness | **NEEDS-RESTATEMENT** | `analysis-smk1.json` contains speed predicates from before C5′ gating. It labels all stop outcomes non-production; those predicates cannot support a production speed claim. |
| Resource projection | **CORRECT** | Independently reproduced 2.06873 h including 1.5× safety. Scales Q2 ladder/population, conservatively budgets F2 Q1 search, separates fitting from qualification, and scales timing. Plausible estimate, not guaranteed completion. |
| Budget | **CORRECT** | Spent 0.4469 + projected 2.06873 = 2.51563 GPU-h. Spent + 2.25 h reservation = **2.6969 GPU-h <3**; no outstanding ledger reservations. |
| Memory | **CORRECT** | Smoke allocator peak 51.20 GB against 138.10 GB allocator limit supports the H200 allocation; identical mesh/r/fit sizes. Per-phase `nvidia-smi` fallback is present. |
| Batch requirements | **CORRECT** | GPU partition, one H200, 8 CPUs, 200G, 02:15:00; absolute venv, paralab output/cache/tmp, f64/highest, GPU preflight exit 42, manifest verification and output checksums. Reservation agrees. |
| Offline scoring | **NEEDS-RESTATEMENT** | Batch produces scoring inputs; `hscore.py` runs after collection using local E0/pin resources, which are intentionally outside staging. |
| Static checks | **CORRECT** | Staged Python syntax and batch shell syntax pass. Live queue/disk/remote-directory checks remain submission-time responsibilities. |

Read-only audit completed: no file changes, jobs, SSH, or LAB-LOG writes.