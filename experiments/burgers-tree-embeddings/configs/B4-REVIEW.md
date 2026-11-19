# B4 closeout and complete-query advancement decision

All44 development trajectories pass physical accuracy, full returned-step convergence, actual4096 output, half-time-step and doubled-integration checks for both frozen methods.

B4 used7620 allocated GPU-seconds of its14400-second ceiling. Cumulative usage is35096 seconds (9.748888889h), including failures. All66 jobs are terminal; maximum pending-or-running owned concurrency is2. Raw outputs, logs and source records have verified local and designated archive copies. No trained recipe, final-test opening, nonlinear solver restart or merge occurred.

|Run|Method|Worst physical error + allowance|Largest refinement change|Nonconverged steps|
|---|---|---:|---:|---:|
|b4base|tree_direct24|0.455525103%|0.000000000%|0|
|b4base|global_stable960|0.348966234%|0.000000000%|0|
|b4time|tree_direct24|0.464702707%|0.025347235%|0|
|b4time|global_stable960|0.351447956%|0.015783711%|0|
|b4quad|tree_direct24|0.455596895%|0.000595775%|0|
|b4quad|global_stable960|0.348966234%|0.000000754%|0|

The common physical setting remains L16384, Crank–Nicolson step0.00125, Q512 and22016 tests. This is development fixed-tree qualification, not deployable adaptation or a runtime win. The original nonlinear PDE optimizer and frozen tangent ablation remain stopped. No native NM-ROM competitor has been defeated by these controls.

The passed gate justifies B5 under B5-COMPLETE-QUERY-DIAGNOSTIC.md and B5-COMPACT-AMENDMENT.md: at most7200 additional allocated GPU-seconds, zero training recipes, two concurrent jobs and45 minutes per job. The next question is complete query cost with equivalent full-cell caches, normal-matrix reuse and one compact-patch alternative. Begin only with the20-minute catalogue equivalence job; later jobs require their retained predecessor gates. Unused B4 allowance is not carried over.

Recorded 2026-11-19T04:59:19.323381+00:00. Accounting: manifests/allocation-audit-B4.json. Archive: manifests/archive-closeout-B4.json. Old campaign remains paused; no unrelated3D work.
