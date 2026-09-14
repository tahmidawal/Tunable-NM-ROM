## 2026-09-11 — Reflective-wave acceleration and accuracy campaign

### Reflective-wave owner: guarded geometry, phase supervision and nested enrichment

Continued only the approved `exp/2026-09-07-mr-wave2d` worktree/namespace. Absorbing waves and the sealed final-paper cohort remained excluded. All accepted runs used fresh post-reset bank/head lineage, regenerated query inputs, f64/highest GPU execution, full displacement and physical-velocity outputs, paired same-job iterative CG and direct DST controls, saved repetitions, source/output hashes and independent NumPy/SciPy field/geometry audits. Every exact remote attempt directory was removed after checksum collection.

| Attempt | Job | Scientific source | Timed invocations | Distinct timed fields | Refinement fields |
|---|---|---|---:|---:|---:|
| accel06 | 3563074 | `2826ce49f385d34a65229798893ec1f2977c714a` | 54 | 18 | 4 |
| accel07 | 3563590 | `b55fe991cff4512063c8109dcdcc72d3ed21d060` | 90 | 30 | 6 |
| accel08 | 3563875 | `2514d3954d7a8c83429d698f4788f3176a2816fd` | 60 | 20 | 8 |
| accel09 | 3564476 | `160b338a7d61a5111e46bff0b168610ae4a77282` | 60 | 20 | 8 |
| accel10 | 3565033 | `452cbabf21404a32300b93747e1254117fefa62f` | 66 | 22 | 10 |
| accel12 | 3565786 | `3eb9b6586ba019b9bc076d9da21f7a0600fb1f9d` | 324 | 108 | 36 |

The parity-preserving guarded-Cholesky geometry reduced the original ROM median by 6.929743702× at the unchanged timestep; retaining the verified larger timestep gave 24.993359802× in the same screen. Shared analytic derivatives, conservative rank bounds with exact-SVD fallback and measured normal-solve backward residuals preserve the original guard. Independent saved-state audits are empirical finite-precision checks, not exhaustive interval certificates for every internal stage.

The further timestep enlargement failed its predeclared refinement requirement and was rejected. Linear-bank evolution with stationary nonlinear output reconstruction was tested separately with a larger internal state; it reduced some errors but still missed the all-state target and was not selected. Physical output velocity used the full implicit stationarity Hessian, with saved neighboring-time kinematic checks.

Matched fixed-encoder field-only and field/energy/tangent training separated the effect of added supervision. Phase supervision modestly improved initial-scaled errors; a newly trained larger head did not improve the worst rollout error. The final additive architecture preserves the accepted nonlinear head and appends fixed linear training directions, improving both displacement and velocity errors. The fixed training-library appended scores are not residual-corrected; all initial coordinates are nevertheless fitted to supplied fields.

Final confirmation settings/checkpoint were frozen before the separately seeded fresh development cohort. The first confirmation attempt failed at an operational baseline lookup before generating any query case; its complete failure archive is retained under `runs/accel11`, and the corrected retry preserves every scientific setting and selection hash. No earlier accepted numerical result was retracted. Audit provenance was strengthened by binding every accepted audit to its raw-result hash, auditor hash and matching source/job identities.

| Intervals per axis | Original ROM GPU ms | Selected nested ROM GPU ms | Worst initial-scaled u / v / energy-state % | Fastest tested passing CG GPU ms | CG / selected ROM | Direct DST GPU ms |
|---:|---:|---:|---|---:|---:|---:|
| 64 | 4485.909332056 | 199.438048410 | 1.477906550 / 3.260997024 / 5.041070830 | 42.195605929 (`cgdt_0.005_tol_0.01`) | 0.211572497× | 3.760324442 |
| 256 | 4472.278085188 | 200.158698019 | 1.542840760 / 3.419016401 / 5.136114487 | 138.838245883 (`cgdt_0.005_tol_1e-06`) | 0.693640832× | 4.422320519 |
| 1024 | 4477.732209489 | 202.941564610 | 1.546467806 / 3.421021638 / 5.145194102 | 601.496360614 (`cgdt_0.005_tol_0.01`) | 2.963889442× | 16.732216580 |

These confirmation medians pool both opened and both fresh development cases, with the full repetition arrays retained. The selected nested manifold improves high-resolution iterative-FOM timing and physical errors, but the pooled all-state accuracy target remains missed. Direct DST remains faster. Fresh cases are reported separately in the generated panels; they were not used for further tuning. Some loose CG settings converge numerically while failing the physical target and are excluded from the passing comparator.

Displacement uses the supplied initial displacement norm. Velocity and energy-state use the square root of twice the supplied initial physical energy; initial velocity can vanish in some cases. Energy-state error measures trajectory mismatch and differs from energy drift. Current-relative errors and defined phase errors remain in every full result/panel. The selected model has forty configuration coordinates and eighty phase coordinates in the frozen sixty-four-function bank; original controls have thirty-two configuration coordinates.

Artifacts, scientific sources, checkpoints, audits, generated panels and verified bounded archive parts live under `experiments/multiresolution-wave/runs/`. The coordinator owns the generated main campaign report. No further wave search, new worktree, merge, final-cohort opening or presentation replacement occurred. Remaining paper work includes meeting the all-state target and broader independent families/seeds; this is a bounded development result.
