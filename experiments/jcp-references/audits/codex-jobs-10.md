**SUBMIT all five jobs for provisional generation.** A11.5 closes the previous sealed-factor bypass.

| Job | Verdict |
|---|---|
| `fd2dsealed128p0` | **SUBMIT** |
| `fd2dsealed128p1` | **SUBMIT** |
| `fd2dsealed128p2` | **SUBMIT** |
| `fd2dsealed128p3` | **SUBMIT** |
| `fd2dood32p0` | **SUBMIT** |

Executed the actual collector functions against an in-memory filesystem:

| Counterexample / control | Result |
|---|---|
| Produced at 10; collector, final status and calibration changed to 100 | **Held** |
| Provisional sealed job with missing/null factor | **Held** |
| Matching factor and verified final calibration | Released |
| Final calibration `provisional: null` | Held |
| Gate-failed calibration | Held |
| Legacy final-status 3D sealed job without factor | Released |

The two existing 3D sealed/OOD records have `calib_status: "ok"`, passing gates, complete outputs and valid result hashes.

All five staged manifests and provenance records pass against commit **`4df6a834af7362035a73dc1b26d401195c1e7890`**; staged source bytes match HEAD. Coverage is exact: **160 cases / 320 case-kind outputs**, without duplicates. Configs contain custody paths and committed seed hashes, with no raw secrets. Resource settings, isolation and GPU/precision preflight pass.

**Release remains conditional.** The 2D calibration is still provisional. Audit 9’s separate incomplete-calibration-manifest inconsistency remains reproducible: calibration evidence can authorize release despite that calibration job being refused as `pull incomplete`. Resolve that before release; it does not block these provisional generation jobs. The previous walltime limitation also remains.

No files modified, lab-log entry appended, or jobs submitted.