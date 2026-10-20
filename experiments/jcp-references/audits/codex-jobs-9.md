**SUBMIT the seven non-sealed jobs for provisional generation. DO NOT SUBMIT the five sealed/OOD jobs.** HEAD `0870698bb` still has a sealed-factor release bypass.

No files modified, jobs submitted, or lab-log entry appended.

I executed HEAD’s actual `gather()`, `verified_calibrations()`, and `release_ok()` in memory, using simulated files and intercepted links.

| Counterexample | Observed result |
|---|---|
| Gate-failed calibration used as evidence | Held |
| Final `provisional: null` | Held |
| Collector/final/calibration factor 100; non-sealed case judged at 10 | Held |
| Resumed provisional case under job-level `ok` | Held |
| Sealed job with provisional/null final status | Held |
| Sealed job with gate-failed calibration evidence | Held |
| Valid final status and verified matching evidence | Released |
| **Sealed job produced at 10; collector/final/calibration subsequently use 100** | **Incorrectly released** |

**Sealed blocker.** [collect.py:72](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/collect.py:72) substitutes the collector’s current `V.FLOOR_STAB` for the producing job’s factor. The sealed result/QC does not record that original factor in its public metadata. Consequently, changing the collector and verified calibration to 100 releases unchanged QC produced at 10. A11.4’s job-level exception for sealed jobs does not remove the factor-binding requirement.

Before submitting those five jobs, record the producing factor in the job result and have collection validate that recorded value.

**Calibration handoff remains manual and unverified.** Future calibration statuses correctly include `provisional=False` at [refjob.py:407](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/refjob.py:407).

However, live job **5209895 (`calib2d3`)** still runs commit `2568b18d3`. Its remote writer hash matches the older staged writer, which omits `provisional`. I found no exporter in the inspected workflow that converts its status. Copying it verbatim into `checks/calib_2d.json` **leaves production held**.

After successful completion and evidence verification, the handoff must extract the nested `calib_status`, explicitly add Boolean `provisional: false`, and preserve its dimension, factor and evidence hash. This is a release prerequisite; it does not prevent provisional non-sealed generation.

**Additional collector inconsistency.** A synthetic calibration with an additional missing manifested output is refused by `gather()` as `pull incomplete`, yet remains eligible in `verified_calibrations()` and releases FD outputs without linking calibration evidence. The normal calibration payload contains only the two files that the prepass verifies, so this does not block the staged non-sealed jobs. Nevertheless, the two validation paths should agree before release.

**Restaging passes.** All 12 manifests and every provenance entry match staged bytes and git content at **`19ab153e946476c0824f02800bec6018f4371ecb`**; those source bytes also match HEAD. Coverage is exact and duplicate-free: **262 cases / 524 case-kind outputs**. GPU partition, A100-80G, 120G RAM, absolute venv, f64/highest precision, GPU preflight, isolated paths, custody records and walltimes all pass.

| Job | Limit | Verdict |
|---|---:|---|
| `fd2ddev6p0` | 12 h | **SUBMIT** |
| `fd2dval32p0` | 12 h | **SUBMIT** |
| `fd2dval32p1` | 12 h | **SUBMIT** |
| `fd2dtest64p0` | 12 h | **SUBMIT** |
| `fd2dtest64p1` | 12 h | **SUBMIT** |
| `fd2dtest64p2` | 12 h | **SUBMIT** |
| `fd2dtest64p3` | 12 h | **SUBMIT** |
| `fd2dsealed128p0` | 24 h | **DO NOT SUBMIT** |
| `fd2dsealed128p1` | 24 h | **DO NOT SUBMIT** |
| `fd2dsealed128p2` | 24 h | **DO NOT SUBMIT** |
| `fd2dsealed128p3` | 24 h | **DO NOT SUBMIT** |
| `fd2dood32p0` | 24 h | **DO NOT SUBMIT** |

These submission verdicts do not certify reference release or guarantee completion through every tight-refinement path. The previous audit’s walltime limitation remains.