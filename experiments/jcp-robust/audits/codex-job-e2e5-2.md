**Verdict: SUBMIT e2v0a, e2v0b and e2v0c. FIX-FIRST e2t0 and all six E5 jobs.** The E2 accuracy jobs collect the required evidence, but the front-timing path has reproducible defects. E5 items 4 and 5 are resolved; items 3, 7 and 8 remain partially unresolved.

1. **CORRECT — E2 verdict configurations match the registered incumbent selection.**  
   I regenerated all four configurations **in memory**; they equal the checked-in JSONs, including the selection-file SHA256.
   
   - R128: `sel=fib4181`, successor `fib6765`.
   - R256: `sel=fib10946`, successor `fib17711`; separate `dep=gauss128`, successor `gauss160`.
   - R384: `sel=gauss128`, successor `gauss160`.
   
   e2v0a uses 256²/all38; e2v0b uses 4096²/dev6 plus val32|0–9 on H200; e2v0c uses 1024²/all38 with factors 2 and 2.5 under both schemes. All have full qualification runs, tighter checks on dev6, and no target/check runs. This implements the requested A1.8/A4.2 populations and rule separation. Evidence: [make_configs_e2.py:67](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/make_configs_e2.py:67).

2. **CORRECT — H-acc-N, H-acc-r and G-E2.3 have the required inputs and analysis paths.**  
   The analyzer combines e2s0’s factor-1/1024² results with the verdict jobs, aliases selected rows to deployed rows where appropriate, and calls the registered P1 statistics. G-E2.3 enumerates all eight required incumbent replay cells, including **deployed R256 gauss128**, with all38 and pinned archive hashes. Verdict-job selection hashes and selection-job identity are checked. Evidence: [e2analyze.py:158](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e2analyze.py:158), [binding checks:220](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e2analyze.py:220).  
   Current measured classes cover all three accuracy jobs; their computed minimum time limits are approximately 0.168, 0.226 and 0.232 hours, below their one-hour requests.

3. **WRONG — dtime’s reference/subject collision makes a required front point unavailable.**  
   A is certified once as the persistent reference and again in its subject block. Both append equivalence records under the same label. `front()` requires exactly `[True]`, so `[True, True]` rejects every timing for R384/gauss128/CNR/factor-1. An in-memory execution of the actual `front()` returned `cost=None, state="missing"` despite valid triplets. This also removes that candidate from dominance comparisons. Evidence: [dtime.py:249](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/dtime.py:249), [subject certification:317](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/dtime.py:317), [e2analyze.py:120](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e2analyze.py:120).  
   **Fix:** explicitly handle reference reuse and retain unambiguous equivalence evidence.

4. **WRONG — e2t0’s doubled-work control never executes; its inventory also misrepresents execution.**  
   Subjects contain `slot` and `role`; `doubled=[A]` does not. dtime selects controls using whole-dictionary equality, producing **zero matching subjects**. Match computational identities instead. Evidence: [generator:100](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/make_configs_e2.py:100), [dtime.py:300](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/dtime.py:300).
   
   Independently reproduced inventory: **1,656 keys, 1,590 unique, 66 duplicates**. Those comprise 30 certification keys and 36 timing keys. The driver actually repeats the certification, while subject triplets execute once. Deduplicating everything therefore does not describe actual reuse. Doubled-work inventory also omits the repetition dimension. Evidence: [certify.py:106](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/certify.py:106).

5. **NEEDS-FIX — E2 front analysis omits the fixed-configuration comparison and timing-selection binding.**  
   `front()` discards every `dep` row, so the separately timed R256 gauss128 configurations receive no fixed-configuration front states, contrary to A2.6/A3.5. It also never verifies the timing configuration’s selection hash or registered attempt. An in-memory probe with an incorrect timing-selection hash still returned `state="member"`. Evidence: [e2analyze.py:132](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e2analyze.py:132).  
   These analysis repairs do not require changing the three accuracy-job workloads.

6. **NEEDS-FIX — E5 item 3: executing-device attribution is repaired, allocation preservation is not.**  
   The driver now resolves JAX’s executing device through CUDA UUID/PCI evidence and queries telemetry by that UUID. The no-I/O device-resolution unit test passed. However, staging still replaces inherited `CUDA_VISIBLE_DEVICES` with `0` for these single-GPU jobs, including preflight. The previous audit’s allocation-mapping requirement remains unresolved; correct attribution alone does not preserve the scheduler’s mapping. Evidence: [e5time.py:315](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e5time.py:315), [stage.py:139](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/cluster/stage.py:139).

7. **CORRECT — E5 items 4 and 5 are resolved.**  
   Duplicate directories, SLURM job IDs and staged attempts invalidate repeated inputs; same-UUID jobs receive explicit labels. A direct duplicate-job-ID probe passed. Smoke acceptance now requires at least five cases with two valid repetitions across **every required cell** of each panel, including all comparators. Evidence: [distinct_jobs:102](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e5analyze.py:102), [same-device labels:633](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e5analyze.py:633), [smoke coverage:804](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e5analyze.py:804).

8. **NEEDS-FIX — E5 item 7: telemetry still fails open in required places.**  
   Acceptance validates only start/end, ignoring the required between-panel record. `telemetry_problems()` accepts missing CUDA-driver and plugin versions; an in-memory probe returned no problems. Additionally, `[Unknown Error]` is converted to `unsupported` for application clocks/power draw, concealing collection failures. Evidence: [e5analyze.py:156](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e5analyze.py:156), [e5common.py:174](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e5common.py:174).  
   **Fix:** validate all three records and required version fields; distinguish unsupported capabilities from query errors.

9. **WRONG — E5 item 8: A10 authorizes extrapolation, but certificate generation still fails.**  
   `timings.py` now extracts the A10.1 per-execution rate for each GPU type, resolving the former unmeasured-4096²-class objection. However, `certify()` first requires `e5time|setup|None`, which extraction never emits. Feeding the actual certificate function the new extraction-output shape caused both A100 and H200 certification to exit 2 with that missing class. Whole-smoke elapsed already includes setup. Evidence: [certify.py:176](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/certify.py:176), [timings.py:127](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/timings.py:127).  
   A10.2’s dtime formulas are present, but the generated certificate lacks the required explicit estimate label, and finding 4 invalidates its execution accounting. This finding concerns generation logic, **not certificates awaiting regeneration**.

| Job | Verdict | Reason |
|---|---|---|
| e2v0a | **SUBMIT** | Correct accuracy/replay workload |
| e2v0b | **SUBMIT** | Correct P1 subset and H200 placement |
| e2v0c | **SUBMIT** | Correct front-accuracy workload |
| e2t0 | **FIX-FIRST** | Findings 3–5 and 9 |
| smk4 | **FIX-FIRST** | Findings 6 and 8 |
| smk5 | **FIX-FIRST** | Findings 6 and 8 |
| e5a1 | **FIX-FIRST** | Findings 6, 8, 9; accepted smoke required |
| e5a2 | **FIX-FIRST** | Same |
| e5h1 | **FIX-FIRST** | Same |
| e5h2 | **FIX-FIRST** | Same |

Validation used source inspection and small in-memory probes; all 35 E5 provenance hashes matched. No files changed, no scratch files created, and no GPU, SSH, cluster or submission commands ran.