**SUBMIT: ps2dp0, ps2dp1, ps3dp0, ps3dp1.** A8 repairs the NaN inconsistency and makes the revised wall times credible as planning estimates. Completion within those limits remains unmeasured.

| Audit item | Verdict | Evidence |
|---|---|---|
| Previous NaN failure | **REPAIRED** | Executed each staged `stability_ok` with `[1, NaN, 1, 1]`: producer and independent-auditor logic both return **False**. NaN and ±Inf in every position also fail consistently; the finite control passes. |
| Base steps and output spacing | **CORRECT** | 2D `dt=0.00003125`; 3D `dt=0.00015625`. Base, half, and quarter steps divide `0.05` into exactly `1600/3200/6400` and `320/640/1280` steps respectively. Float multiplication also reconstructs `0.05` exactly. |
| Half-step and stability retry | **CORRECT** | Initial triple uses `dt`, finest companion uses `dt/2`. A failed stability check repeats the complete triple at `dt/2`, plus finest at `dt/4`, retaining original evidence. |
| Run IDs and selection | **CORRECT** | Initial IDs: `N<N>_dt0_a0`, companion `N<N3>_dt1_a0`. Retry: `N<N>_dt1_a1`, companion `N<N3>_dt2_a1`. Selected fields and stability metadata use the same attempt. Escalation remains separately staged. |
| Case coverage | **CORRECT** | Executed `cohorts.crosscheck_cases`. Each dimension’s two lists are disjoint and their union exactly matches the selector; lists below. |
| Stability/retry prediction | **REPAIRED** | Maximum permitted recorded amplitude becomes **2.29780 / 2.45099** for 2D/3D. No selected case predicts a retry from its initial amplitude. This prediction assumes resolved peaks and no additional stage overshoot. |
| Wall-time feasibility | **CREDIBLE, ESTIMATED** | Revised baseline estimates are **4.90 h per 2D job** and **9.91 h per 3D job**, before overhead; details below. |
| Memory qualification | **UNCHANGED LIMITATION** | No obvious static capacity blocker. Compiled peak/workspace memory remains unmeasured. `120G/240G` requests host RAM, not GPU memory. |
| Sbatch | **CORRECT** | All four pass `bash -n`; correct GPU partition, A100-80G/H200 requests, distinct directories, absolute venv, GPU preflight, f64/highest precision, and output checksums. The 24-hour requests fit the stated two-day partition limit. |
| Provenance/manifests | **CORRECT** | Each bundle has **19 valid manifest entries** and **16 provenance entries**. All staged source bytes match current files and commit `605f2f9b3d4e863f16eece9c9a0c081052f504cc`. Log directories exist. |

The workload includes three full-step trajectories and one finest half-step trajectory per case, each reaching \(T=0.25\).

| Budget quantity, per job | 2D | 3D |
|---|---:|---:|
| Full-step trajectory steps | 8,000 | 1,600 |
| Total steps, four cases, no retries | 160,000 | 32,000 |
| Additional steps per retried case | 80,000 | 16,000 |
| Approximate finest-grid-equivalent steps | 120,762 | 22,487 |
| Average finest-step ceiling before overhead | 358 ms | 3,842 ms |
| Estimated hours: zero / one / two retries | 4.90 / 7.34 / 9.79 | 9.91 / 14.87 / 19.83 |
| Requested wall time | 12 h | 24 h |

These estimates use the previous audit’s padded-volume × FFT-log weighting and the same approximate §12 cost anchors—not measured A100/H200 throughput. Gates, compilation, coefficient checks, cohort construction, and output add overhead. Three retries would already exceed either allocation on this model; **none is predicted from initial amplitudes**.

| Job | Cases | Resources | Verdict |
|---|---|---|---|
| `ps2dp0` | `[0,1,2,3]` | A100-80G, 12 h | **SUBMIT** |
| `ps2dp1` | `[4,5,19,26]` | A100-80G, 12 h | **SUBMIT** |
| `ps3dp0` | `[0,1,2,3]` | H200, 24 h | **SUBMIT** |
| `ps3dp1` | `[5,33,43,47]` | H200, 24 h | **SUBMIT** |

No files modified, no jobs submitted, and no lab-log entry written, per your read-only instruction.