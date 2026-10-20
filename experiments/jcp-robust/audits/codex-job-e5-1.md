**Verdict: FIX-FIRST for all six jobs.** The registered workloads are correct, but device attribution and smoke acceptance need fixes before these runs can reliably authorize production.

1. **CORRECT — Configs match the registered populations.** All four configs agree with `E5_CONTRACT`: frozen R128/fib4181/LSPG-BE and R384/gauss128/LSPG-CN-R, Δt₀ = 0.005; reference A is R384 CN-R at 1024². Production uses dev6 × 3 for P-cost and dev6|0–5 + val32|0–9 × 3 for P-cross; both smokes use dev6|0–4 × 2. Both subjects have doubled-work controls. Only H200 production includes 4096². Evidence: [contract definitions](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e5common.py:202).

2. **CORRECT — Comparators are pinned without re-selection.** Independently checked the frozen file and every configured selection:
   - Union = PS joint: `ps|N96|0.0125`.
   - R128 CC same-mesh: `fd|CC|L1024|0.0125|t2`.
   - R384 CC same-mesh: `fd|CC|L1024|0.005|t1`.

   All 35 provenance hashes match; all 34 git-backed sources also match their recorded source commits. The driver checks the frozen-file hash, exact selections, and union/PS equality. Evidence: [pinning code](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e5time.py:66).

3. **NEEDS-FIX — Recorded UUID is not explicitly bound to the executing GPU.** Telemetry queries all devices visible to `nvidia-smi`; `gpu_type_ok()` checks only the first row. Staging overwrites the inherited `CUDA_VISIBLE_DEVICES` with `0`. On systems where device isolation/remapping does not make these identities equivalent, the checked GPU and recorded telemetry can differ from JAX’s device. Preserve the allocation mapping and record an explicitly resolved executing-device UUID; use it for type/memory checks and telemetry. This is a portability risk established by the code, not a claim about the uninspected cluster configuration. Evidence: [GPU identification](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e5common.py:68), [batch generation](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/cluster/stage.py:136).

4. **WRONG — Same-UUID repeats are not labelled.** Scheduling permits different GPUs: A100 jobs constrain memory type, all jobs exclude `pax007`, and none pins a particular node. It does not guarantee different devices. The analysis never compares UUIDs or labels same-device repeats, despite §4.2. It also counts supplied entries without rejecting duplicate directories/job IDs; passing one result twice can satisfy the two-job count. Add distinct-job validation and same-UUID labels. Evidence: [job grouping and completeness](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e5analyze.py:574).

5. **NEEDS-FIX — Smoke `ALL` can pass without usable timing coverage across the whole panel.** Population checks require rows to exist, not two valid repetitions in every required cell. Fixtures exercise R128 at 1024² and its union comparator. Thus invalid timings elsewhere—for example, the CC comparator—can leave a per-job statistic unavailable while `ALL` remains true. Require both single-job panel intersections to retain at least five cases, with two valid repetitions for every required subject/comparator. Evidence: [smoke acceptance](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e5analyze.py:722).

6. **CORRECT — Core evidence and pooled estimators are implemented.** The driver retains timing repetitions, retries, identity checks, doubled-work evidence, scoring coefficients, and P-cross reproduction gates. Analysis requires G-E5.1 against checksum-verified P1 coefficients and applies the registered pooled estimators, separate case intersections, fixture blocking, and q4 disagreement rules. Cross-type intersections exclude 4096². Small in-memory probes passed all three A3.7 fixtures and rejected the 1.5× drift fixture. Evidence: [analysis](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e5analyze.py:294).

7. **NEEDS-FIX — Telemetry collection is comprehensive but fails open.** Start/between/end records request clocks, application clocks, power, temperature, driver/CUDA, package versions, precision flags and UUIDs. However, command failures become `"ERROR: …"` strings, and acceptance does not validate telemetry completeness. Require successful mandatory fields; explicitly distinguish unsupported application clocks from failed collection. GPU backend, x64 and highest precision are enforced by the driver. Evidence: [recording helpers](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e5common.py:60).

8. **NEEDS-FIX — H200 certification cannot be derived solely from these smokes under the current class rules.** H200 production requires **16 distinct 4096² execution classes absent from both smokes**. `certify.py` refuses unmeasured classes. Additionally, `timings.py` accepts `--smk4/--smk5` but currently never processes their E5 outputs. Supply measured coverage or an explicitly approved estimation procedure and implement extraction. This flags the generation path, **not absent production certificates**. Evidence: [class definition](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/e5common.py:193), [certificate refusal](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/certify.py:161).

9. **CORRECT — Resource placement is sensible; runtime remains conditional.** A100 jobs exclude 4096². H200 production requests 240G host memory and a 0.95 device-memory fraction, consistent with the stated 97–113 GB device peak. Smokes request 120G/30 minutes; production requests 200G or 240G/one hour. No obvious resource contradiction was found, but the one-hour limits require the forthcoming measured projections.

| Job | Verdict |
|---|---|
| smk4 | **FIX-FIRST** — findings 3, 5, 7 |
| smk5 | **FIX-FIRST** — findings 3, 5, 7 |
| e5a1 | **FIX-FIRST** — shared fixes, distinct-job/UUID handling, accepted smokes |
| e5a2 | **FIX-FIRST** — same |
| e5h1 | **FIX-FIRST** — above plus 4096² certification coverage |
| e5h2 | **FIX-FIRST** — same |

Read-only audit completed: no files changed, no scratch files created, and no SSH, cluster commands, or submissions executed.