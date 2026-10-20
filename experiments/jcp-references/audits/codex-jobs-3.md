**ps2d1: DO NOT SUBMIT. ps3d1: DO NOT SUBMIT.**

Both have a reproducible stability-metadata inconsistency on non-finite runs. The 3D job also lacks a credible 12-hour budget. No files were modified, no jobs submitted, and no lab-log entry written, per your read-only instruction.

| Item | Verdict | Finding |
|---|---|---|
| Case lists, §6/A1.7 | **CORRECT** | Executed `cohorts.crosscheck_cases`: exactly 2D `[0,1,2,3,4,5,19,26]`; 3D `[0,1,2,3,5,33,43,47]`. Both configs match. |
| Spatial ladder and base step | **CORRECT** | 2D: `1024,1536,2048`, `5e-5`. 3D: `128,192,256`, `2.5e-4`. Padding is `M=3N/2`; lattice evaluation uses direct synthesis. |
| Finest half-step, A2.3 | **CORRECT** | Every case gets the finest-grid half-step run. Collectors compare that field against FD; spatial differences use the full-step triple. |
| Stability retry and selected IDs | **CORRECT**, finite path | One retry repeats all three resolutions at `dt/2`, plus finest at `dt/4`. Initial selection: `N<N>_dt0_a0`, half `N<N3>_dt1_a0`. Retry selection: `N<N>_dt1_a1`, half `N<N3>_dt2_a1`. Original evidence remains stored. |
| Non-finite stability bookkeeping | **WRONG** | Producer and independent auditor disagree when a later run has `umax=NaN`; reproduction below. |
| Per-run IC/contour checks | **CORRECT** | Every run checks IC at 256/512 panels and contour at 32/64 points using its own `N,dt,nu`. Bars match amended A4.2: IC relative ≤`1e-12`, absolute ≤`1e-13`; contour relative ≤`1e-10`. |
| In-job gates | **CORRECT** | Units and both G1 dimensions run first. PS dimension gets G3a at `N=128`; G3b/G3c at production maximum `2048`/`256`; G4 spatial `32,48,64`, temporal `N=64`, plus projection check. Failures stop production. |
| Output schema and normalization | **CORRECT** | `cNNN_a0.{json,npz}` includes selected IDs, all run fields/checks/maxima/finiteness, `t0_ps`, canonical `t0`, and `ladder_attempt`. Both consumers use FD `judge.n0`, derived from canonical `t0`, not PS initialization. |
| Escalation | **CORRECT** | These stage attempt 0 only. Unresolved cases remain pending escalation; attempt 1 is a separate job, not silently omitted or promoted. |
| Memory | **NEEDS-RESTATEMENT** | Static sizes suggest adequate capacity, but compiled peak/workspace usage is unmeasured. `240G` requests **host RAM**, not H200 device memory. |
| Twelve-hour feasibility | **NEEDS-RESTATEMENT** | The original compute estimate omits the amended workload and substantial expected stability retries; details below. |
| Sbatch/provenance/manifests | **CORRECT** | Both scripts pass shell syntax checks. Each has 19 valid manifest entries and 16 source-provenance entries, all matching commit `415f535e1860f6b817bfe03b8416ae77cd82613a` and current source bytes. Correct GPU partition, separate directories, absolute venv, GPU preflight, f64/highest precision, output checksums. |

The blocking inconsistency is in [refjob.py:445](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/refjob.py:445). Python’s `max` can discard a later NaN:

```text
run maxima: [1.0, NaN, 1.0, 1.0]
producer stability_ok: True
independent auditor stability: False
```

The auditor explicitly converts non-finite maxima to infinity. Consequently, this path skips the producer’s stability retry and stores a flag that fails independent audit. The separate `finite=False` prevents a false PASS, but legitimate failed-run evidence cannot audit consistently. Explicitly reject non-finite maxima before aggregation, then restage both jobs.

The workload in [ps_case](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/refjob.py:412) is larger than “finest plus half-step”:

| Budget quantity | ps2d1 | ps3d1 |
|---|---:|---:|
| Base steps across eight complete ladders | 200,000 | 40,000 |
| Stability threshold on recorded maximum | 1.4361 | 1.5319 |
| Predicted retry cases from initial amplitudes | 5, 19 | 1, 2, 5, 43 |
| Steps including those retries | 300,000 | 80,000 |
| Approximate finest-grid-equivalent steps¹ | 226,429 | 56,218 |
| Maximum average finest-step cost to fit 12 h² | 191 ms | 768 ms |

¹ Scaling work as padded-grid volume × logarithm of transform length. Retry predictions assume resolved initial peaks and no additional stage overshoot.  
² Before gates, compilation, coefficient checks, cohort construction, and output overhead.

As an **illustrative budget estimate**, reweighting [DESIGN §12’s outdated estimates](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/DESIGN.md:234) for three resolutions, eight half-step runs, and those retries gives approximately **9.2 h for 2D and 24.8 h for 3D**, before additional overhead. These are not measured A100/H200 runtimes; the original table is itself approximate. They support a plausible but tight 2D budget and do not support the staged 3D limit.

Memory is less concerning: one finest coefficient field is approximately **32/127 MiB**, one padded real field **72/429 MiB**, and one FFT real extension **144/860 MiB** for 2D/3D. Multiple stages, complex FFT outputs, workspaces and cached executables increase these substantially, but no obvious capacity blocker appears.

**Required before submission:** repair the non-finite stability aggregation and restage both jobs. For `ps3d1`, additionally split the workload or establish measured throughput comfortably below the 768-ms threshold; its present eight-case, 12-hour allocation is unsupported.