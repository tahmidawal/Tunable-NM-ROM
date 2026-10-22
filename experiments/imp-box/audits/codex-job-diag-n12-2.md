**DO-NOT-SUBMIT for the stated diagnostic purpose.** Resource safeguards pass; round 1’s diagnostic blockers are only partly fixed.

| Round-1 item | Round-2 finding |
|---|---|
| Missing staged entry point | **Fixed.** Remote SHA256 matches local for both job files and `spec.py`, `geom.py`, `rom.py`, `vendor/pscore.py`. |
| Cache isolation | **Partly fixed.** Separate processes and explicit cache disabling are present. Cache directory currently absent, but replay provides no evidence of actual persistent-cache hits. |
| Production query shape | **Fixed; round-1 objection retracted.** The loop uses the same unbatched compiled query as A9.1 over all 32 validation cases. Steps 5 and 40 provide coarse/finer controls. |
| First-failure trace / independent reference | **Still blocking.** Trace and CPU comparison do not isolate the failing production computation. |

Blocking fixes in [n12.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/diag_jobs/n12.py):

- **Verify cache replay** (lines 10–15): use a job-specific fresh cache and record misses/writes followed by verified hits for the relevant queries. Two `cache` invocations alone cannot establish a cache-related cause.
- **Trace an observed failing case/configuration** (lines 26–40): the current trace arbitrarily uses `val[0]`, `n=10`, whereas the cohort probe tests `n=5,40`. Its separately compiled step also closes over precomputed coefficients, unlike the production query’s internal coefficient calculation and nested loops. Retain case-level failures and instrument the matching production path, including coefficient, initial-state and nonlinear-stage finiteness.
- **Add an independent nonlinear reference at the first failing stage** (lines 41–47): CPU execution of the same FFT implementation at the initial state cannot exclude a shared code bug or a later-stage failure. A small NumPy direct sine/cosine evaluation on the **same padded grid** would supply the missing independent check.

The A100 allocation is bounded to **0.5 GPU-h**, uses the correct environment, GPU preflight, f64 and highest precision, and writes under paralab. Arrays and stepping workload are small; compilation completion within 30 minutes is unverified. A passing A100 result must remain **inconclusive about the H200 failure**.

No files changed, jobs submitted, or LAB-LOG writes performed.