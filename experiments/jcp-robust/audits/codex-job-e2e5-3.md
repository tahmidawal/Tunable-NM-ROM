**E2 accuracy jobs: SUBMIT. e2t0: FIX-FIRST. E5 smokes: SUBMIT. E5 production: SUBMIT after the registered smoke-acceptance gates pass.**

1. **CORRECT — E2 accuracy configurations match the incumbent selection and registered populations.** In-memory regeneration exactly matches all four checked-in configurations, including selection hashes:
   - R128: selected/deployed `fib4181`, successor `fib6765`.
   - R256: selected `fib10946`, successor `fib17711`; separate deployed `gauss128`, successor `gauss160`.
   - R384: selected/deployed `gauss128`, successor `gauss160`.

   e2v0a covers 256²/all38; e2v0b covers 4096²/the P1 16-case subset on H200; e2v0c covers 1024²/all38, factors 2 and 2.5 under both schemes. Qualification, tighter checks and successor runs are present; targets are absent as required. Evidence: [make_configs_e2.py:67](experiments/jcp-robust/make_configs_e2.py#L67).

2. **CORRECT — H-acc-N, H-acc-r and G-E2.3 have the required evidence paths.** The analyzer combines e2s0’s factor-1/1024² rows with the verdict jobs, maintains selected/deployed identities, checks selection binding, and enumerates all eight incumbent replay cells—including deployed R256 `gauss128`. The three accuracy jobs’ computed minimum limits remain below their one-hour requests. Evidence: [e2analyze.py](experiments/jcp-robust/e2analyze.py).

3. **NEEDS-FIX — Prior e2t0 finding 3 is only partially resolved.** Removing A from `subjects` eliminates duplicate certification. However, `front()` now assigns A `cost=1.0` unconditionally, bypassing its equivalence evidence. An in-memory execution of the actual function with A’s equivalence marked **failed** still returned `member` for both its selected and fixed rows.

   Normalizing A to one is defensible only after validating the reference. Require valid reference equivalence and timing identity evidence before assigning that cost. Evidence: [e2analyze.py:141](experiments/jcp-robust/e2analyze.py#L141).

4. **NEEDS-FIX — Prior e2t0 finding 4: doubled work is repaired; inventory accounting remains wrong.** The doubled specification now exactly matches one subject and executes all 18 case/repetition slots. The old **66 duplicates are gone**: the regenerated inventory contains **1,602 unique keys**.

   But it counts **432 split and 432 fused triplets**, while the driver executes **414 of each**: 23 subjects × 6 cases × 3 repetitions. The extra 36 slots belong to reference A, which is no longer a timed subject. Separate reference certification from subject timing in `inv_dtime()`. Evidence: [certify.py:112](experiments/jcp-robust/certify.py#L112), [dtime.py:263](experiments/jcp-robust/dtime.py#L263).

5. **CORRECT — Prior e2t0 finding 5 is resolved.** `front()` now returns separate fixed-configuration states and checks both the registered timing attempt and selection SHA256. A wrong-hash probe returned `missing`, as required. Evidence: [e2analyze.py:95](experiments/jcp-robust/e2analyze.py#L95).

6. **CORRECT — Prior E5 finding 6 is resolved.** Staging preserves the scheduler’s `CUDA_VISIBLE_DEVICES` mapping for preflight and execution. The driver resolves JAX’s executing device and queries telemetry by its UUID. The executing-device test passed. Evidence: [stage.py:139](experiments/jcp-robust/cluster/stage.py#L139), [e5time.py:315](experiments/jcp-robust/e5time.py#L315).

7. **CORRECT — Prior E5 finding 8 is resolved; distinct-job and coverage repairs remain intact.** Acceptance checks start, between-panel and end telemetry, requires CUDA-driver/plugin/PJRT versions, and rejects `[Unknown Error]`. Targeted telemetry and full-panel smoke tests passed. Duplicate job IDs are rejected; same-UUID repetitions retain explicit labels. All **35 provenance hashes matched**. Evidence: [e5common.py:202](experiments/jcp-robust/e5common.py#L202), [e5analyze.py:156](experiments/jcp-robust/e5analyze.py#L156), [e5_tests.py:586](experiments/jcp-robust/e5_tests.py#L586).

8. **CORRECT — Prior finding 9’s E5 certificate defect is resolved.** `certify()` now sets E5 setup to zero because whole-smoke elapsed already includes it, then uses the GPU-specific per-execution rate authorized by A10.1. Write-intercepted certificate probes succeeded for both GPU types using synthetic positive rates. Certificates now explicitly label the estimates. This validates generation logic, not production runtime sufficiency before smoke measurements exist. Evidence: [certify.py:178](experiments/jcp-robust/certify.py#L178), [timings.py:120](experiments/jcp-robust/timings.py#L120).

9. **NEEDS-FIX — e2t0’s requested time fails the A10 certificate margin.** Applying A10.2 to the existing measured rollout classes yields a projection of **0.7968 hours**, requiring **1.1952 hours** under staging’s 1.5× rule. The job requests **1 hour**. Even removing the 36 nonexistent timing slots leaves a minimum of **1.1647 hours**.

   Correct the inventory and raise the request to at least **1.25 hours** under quarter-hour scheduling. This is a workload/time-limit mismatch, not a complaint about certificates awaiting regeneration. Evidence: [timings.py:133](experiments/jcp-robust/timings.py#L133), [stage.py:76](experiments/jcp-robust/cluster/stage.py#L76), [e2t0.json](experiments/jcp-robust/jobs/e2t0.json).

| Job | Verdict | Condition / blocker |
|---|---|---|
| e2v0a | **SUBMIT** | Accuracy workload passes |
| e2v0b | **SUBMIT** | P1 subset and H200 placement pass |
| e2v0c | **SUBMIT** | Front-accuracy workload passes |
| e2t0 | **FIX-FIRST** | Findings 3, 4 and 9 |
| smk4 | **SUBMIT** | E5 smoke code blockers resolved |
| smk5 | **SUBMIT** | E5 smoke code blockers resolved |
| e5a1 | **SUBMIT** | After registered smoke acceptance and certificate checks |
| e5a2 | **SUBMIT** | Same |
| e5h1 | **SUBMIT** | Same |
| e5h2 | **SUBMIT** | Same |

No repository or LAB-LOG edits, submissions, SSH or cluster commands were performed. Six targeted E5 tests passed; the full suite was interrupted. **Scratch constraint deviation:** the initial test helper wrote temporary fixtures under its hard-coded `/tmp/claude-1002/jcp-robust-e5-scratch`; those fixtures were cleaned on interruption. Subsequent tests used the authorized scratch directory.