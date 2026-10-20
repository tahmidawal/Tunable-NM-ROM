**FIX-FIRST for all four jobs.** Coverage, staging, and output layout pass. Resource certification and the A5 release prerequisites do not. This audit made no changes, submitted nothing, used no SSH, and copied no arrays.

**1. Case coverage — CORRECT**

| Job | Cohort and cases | Count |
|---|---|---:|
| ref2a | dev6 0–5; val32 0–6 | 13 |
| ref2b | val32 7–19 | 13 |
| ref2c | val32 20–31 | 12 |
| ps2 | val32 0, 1, 5, 19 | 4 |

The FD jobs cover exactly 38 distinct cohort/case pairs, without overlap or omissions. PS matches A1.5/A2.3. All configurations use \(T=0.5\); FD enables fine output and refinement.

The driver restricts cohorts to dev6/val32 in 2D and rejects sealed cohorts. Staged commands write beneath their respective job directories, never the E0 store.

**2. Batch scripts and provenance — CORRECT; memory certification NEEDS-FIX**

All four scripts request one H200 on `gpu`, 240G host memory, eight CPUs, x64 and `highest`. They include the GPU-backend preflight, manifest verification, separate output/cache/tmp directories, and distinct job names.

Each job’s **148 provenance files match HEAD `5155ccfaca793860ac8d3fb00d9430baea341c64`**. All four staged manifests verify; config hashes and reservations match.

| Job | Certified projection | Required \(1.5\times\) | Staged limit | Arithmetic |
|---|---:|---:|---:|---|
| ref2a | 1.6389 h | 2.4584 h | 2.50 h | CORRECT |
| ref2b | 1.4722 h | 2.2084 h | 2.25 h | CORRECT |
| ref2c | 1.3718 h | 2.0577 h | 2.25 h | CORRECT |
| ps2 | 1.8500 h | 2.7750 h | 3.00 h | CORRECT |

However, the certificates contain **no memory accounting**, despite A2.7 requiring output buffers, refinement levels, and retained fine arrays. The smoke recorded about 18.9 GB peak device allocation at \(L=8192\). The refinement output buffer alone is about 12.37 GiB for eleven float64 fields at \(L=12288\). This does not demonstrate an OOM, but 240G host memory does not certify GPU memory.

**3. Resource projections — NEEDS-FIX**

The FD arithmetic reproduces the smoke measurement exactly: **278.1113 seconds**, multiplied by 1.3 per case, plus 600 seconds per driver invocation. But that smoke used **neither tightening nor refinement**.

[resources.py:76](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/resources.py:76) assigns the same flat allowance regardless of the retry paths in [refrun.py:184](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/ref/refrun.py:184).

For scale, the measured \(L=8192\) level took 239.5 seconds. Scaling only spatial work and step count gives:

\[
239.5(12288/8192)^2(240/160)\approx808\text{ seconds}
\]

**for the added refinement level alone**, versus an 83-second per-case refinement allowance. Tightening can repeat entire triples; failed production/refinement solves can also retry. This scaling is an estimate, not a runtime bound, but it shows that the current certificates do not establish sufficiency for the permitted paths.

**PS: 3 hours is not defensible as covering the stability-rerun path.** The code hard-codes 0.4 hours/case; it does not derive PS timing from an H200 smoke. Using the supplied E0 measurement:

- Base ladder at \(T=0.5\): approximately \(2(0.17)=0.34\) A100-hours/case.
- A stability rerun repeats the ladder at half the step, approximately doubling its work; initial plus rerun therefore costs approximately \(3(0.34)=1.02\) hours/case.
- Four cases, 0.25 hours setup, and the required margin give:
  \[
  1.5[4(1.02)+0.25]=6.495\text{ hours}.
  \]

**Use 6.5 hours as a provisional planning limit for attempt 0**, pending H200 timing validation. This assumes no H200 speed credit and is not a certified bound. Separately reserve the A2.3 resolution-escalation ladder: the staged attempt-0 job does not execute that later escalation automatically.

**4. Collection and analysis — CORRECT layout; NEEDS-FIX collection procedure**

The writer, promoter, and analyzer agree on:

```text
2d/<cohort>/T0.5/c###.{json,npz}
2d/<cohort>/T0.5/c###_fine.npz
2d/val32/T0.5/PS/c###_a0.{json,npz}
```

[collect.py:51](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/cluster/collect.py:51) strips `output/<step>/`, checks source hashes, refuses existing destinations, verifies copied files, and writes the reference manifest. Run promotions **serially**: they share the manifest temporary filename and have no promotion lock.

`--no-fine` verifies retained outputs and requires remote fine-file verification before cleanup. But using it indiscriminately on **ref2a** violates A1.8’s requirement to pull dev6 fine fields for offline recomputation. Collect ref2a normally, or separately checksum-collect those six files.

Run `anaref.py` with all four `archive/output` directories—not the promoted `refs/` root. Its discovery expects `ref_refrun*` step directories. Missing fine files yield no local G-R3 recomputation; the stored flag is still used, so analyzer success alone cannot establish dev6 fine-field retention.

**5. Other release blocker — NEEDS-FIX**

[A5](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/DESIGN.md:763) explicitly requires **both** smoke acceptance files to pass. `checks/smoke-smk2.json` is absent; smk2 currently has staging artifacts only. The coordinator’s production release remains conditional on those gates.

| Job | Verdict |
|---|---|
| ref2a | **FIX-FIRST** |
| ref2b | **FIX-FIRST** |
| ref2c | **FIX-FIRST** |
| ps2 | **FIX-FIRST** |

Blocking fixes only:

1. Satisfy A5’s missing smk2 acceptance prerequisite, or obtain an explicit recorded amendment allowing 2D production independently.
2. Reprice FD certificates for refinement, tight solves, and retries; add required memory accounting. Update walltimes/reservations and restage against the resulting committed source.
3. Replace PS’s unsupported 0.4-hour constant with a documented workload estimate or H200 measurement. Provisionally allow **6.5 hours for attempt 0**, separately reserve resolution escalation, and recheck the 45-hour lane budget.
4. Specify serial promotion and checksum-verified dev6 fine-field collection before cleanup; analyze all four collected output trees together.