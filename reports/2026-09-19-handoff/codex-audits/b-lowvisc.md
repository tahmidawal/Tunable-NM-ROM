# Independent report audit: b-lowvisc

Audited from retained records on September 20. This audit recomputes aggregates independently of the lane generator and reviews the documented protocol; it does not rerun solvers or independently validate every saved field.

Source checkout: `5760a7e256ef3c003956bef7a185a86c88b823b6`. Numeric checks: **108/108 pass**. Provenance checks: **297/297 pass**. Methodological findings remain open below.

## Verified-number sample

Values retain the units of their named metric; percentage metrics are explicitly suffixed. The complete machine-readable audit retains every checked row.

| Subject | Metric | Reported | Recomputed | Job | Source and calculation | Pass |
| --- | --- | ---: | ---: | --- | --- | :---: |
| `dense_tight` | `median_gpu_ms` | 112.288489006 | 112.288489006 | `3817807` | `worktrees/2026-09-17-b-lowvisc/experiments/b-lowvisc/runs/lvp01/archive/output/panel/result.json`: 1000 * median(invocations.gpu_seconds) | True |
| `nt1e-4_dt01` | `median_gpu_ms` | 88.7121069245 | 88.7121069245 | `3817807` | `worktrees/2026-09-17-b-lowvisc/experiments/b-lowvisc/runs/lvp01/archive/output/panel/result.json`: 1000 * median(invocations.gpu_seconds) | True |
| `q0_M1088_dense_g1em06` | `median_gpu_ms` | 557.445451966 | 557.445451966 | `3817807` | `worktrees/2026-09-17-b-lowvisc/experiments/b-lowvisc/runs/lvp01/archive/output/panel/result.json`: 1000 * median(invocations.gpu_seconds) | True |
| `q32_M1088_dense_g1em06` | `median_gpu_ms` | 772.200720967 | 772.200720967 | `3817807` | `worktrees/2026-09-17-b-lowvisc/experiments/b-lowvisc/runs/lvp01/archive/output/panel/result.json`: 1000 * median(invocations.gpu_seconds) | True |
| `fft_tight` | `worst_evolved_percent` | 0 | 0 | `3817807` | `worktrees/2026-09-17-b-lowvisc/experiments/b-lowvisc/runs/lvp01/audit-panel.json`: max_case max_evolved_time per_time_same_grid_percent | True |
| `nt1e-2_dt005` | `worst_t0_compression_percent` | 0 | 0 | `3817807` | `worktrees/2026-09-17-b-lowvisc/experiments/b-lowvisc/runs/lvp01/audit-panel.json`: max_case per_time_same_grid_percent[0] | True |
| `nt1e-3_dt01` | `worst_evolved_percent` | 3.27981945454 | 3.27981945454 | `3817807` | `worktrees/2026-09-17-b-lowvisc/experiments/b-lowvisc/runs/lvp01/audit-panel.json`: max_case max_evolved_time per_time_same_grid_percent | True |
| `pod128_M512_dense` | `worst_all_times_percent` | 23.2032370208 | 23.2032370208 | `3817807` | `worktrees/2026-09-17-b-lowvisc/experiments/b-lowvisc/runs/lvp01/audit-panel.json`: max_case max_all_times per_time_same_grid_percent | True |
| `pod256_M1024_dense` | `worst_evolved_percent` | 11.2539919782 | 11.2539919782 | `3817807` | `worktrees/2026-09-17-b-lowvisc/experiments/b-lowvisc/runs/lvp01/audit-panel.json`: max_case max_evolved_time per_time_same_grid_percent | True |
| `pod64_M256_dense` | `worst_all_times_percent` | 35.1339580387 | 35.1339580387 | `3817807` | `worktrees/2026-09-17-b-lowvisc/experiments/b-lowvisc/runs/lvp01/audit-panel.json`: max_case max_all_times per_time_same_grid_percent | True |
| `q0_M256_dense_g1em06` | `worst_evolved_percent` | 8.39330667326 | 8.39330667326 | `3817807` | `worktrees/2026-09-17-b-lowvisc/experiments/b-lowvisc/runs/lvp01/audit-panel.json`: max_case max_evolved_time per_time_same_grid_percent | True |
| `q16_M1088_dense_g1em06` | `worst_all_times_percent` | 8.75624861552 | 8.75624861552 | `3817807` | `worktrees/2026-09-17-b-lowvisc/experiments/b-lowvisc/runs/lvp01/audit-panel.json`: max_case max_all_times per_time_same_grid_percent | True |
| `q256_M1088_dense_g1em06` | `worst_t0_compression_percent` | 3.21846770401 | 3.21846770401 | `3817807` | `worktrees/2026-09-17-b-lowvisc/experiments/b-lowvisc/runs/lvp01/audit-panel.json`: max_case per_time_same_grid_percent[0] | True |
| `q32_M256_dense_g1em06` | `worst_all_times_percent` | 8.71908960667 | 8.71908960667 | `3817807` | `worktrees/2026-09-17-b-lowvisc/experiments/b-lowvisc/runs/lvp01/audit-panel.json`: max_case max_all_times per_time_same_grid_percent | True |
| `q64_M256_dense_g1em06` | `worst_t0_compression_percent` | 4.54366059228 | 4.54366059228 | `3817807` | `worktrees/2026-09-17-b-lowvisc/experiments/b-lowvisc/runs/lvp01/audit-panel.json`: max_case per_time_same_grid_percent[0] | True |

## Mismatches and provenance

No numerical mismatch was found among the recomputed aggregates.
The source/job checks performed here pass.

## Gate history, comparisons, retractions and weakest claims

- The mesh-refinement failure remains explicit. Reduced-versus-reduced comparisons on this mesh do not establish continuum-accurate low-viscosity performance.
- The report records a failed POD orthogonality audit check and bounds its effect using an oblique projector. This is a disclosed exception, not a clean all-checks-pass record.
- The main panel fails its full-order-frontier criterion; the reduced-only frontier and the correction-ladder span answer different questions. Retain these separate verdicts.
- The source-label inheritance bug was repaired in commit 5760a7e256ef3c003956bef7a185a86c88b823b6. Gate, training and panel sources are now attributed separately, and the panel labels are rechecked here against its retained result. The lane's source catalog traces archived bytes to Git objects; historical incumbent-training records lacking a recoverable run commit are explicitly marked unknown with retained artifact hashes. The combined report must not invent one shared scientific source commit.
- Further work: a resolved refinement confirmation with the same physical family and fair tuned FOM controls, then any claim of a nonlinear advantage beyond the reduced-model comparison.

Remaining methodological findings are not resolved by aggregate agreement. Full-field audits already retained by the lane are supporting evidence, not new independent field checks performed here. Any altered claim must be applied to the manuscript and its source report before submission.

## Glossary

- **Subject:** a particular model or solver configuration.
- **Metric:** the named error, time or statistic; same-grid error compares against a numerical solution on the same mesh.
- **Reported/recomputed:** value in the lane summary/value independently calculated from retained case or invocation arrays.
- **Job/source:** the cluster allocation identifier/the file containing the checked inputs.
- **Provenance:** evidence identifying the exact code and data that produced a result.
- **Gate/amendment:** an acceptance rule/a documented change to the prospective protocol.
- **FOM/POD:** a full-grid numerical solver/a linear basis derived from training snapshots.
- **Bank/head/correction rank:** learned spatial functions/their nonlinear coefficient network/the number of additional solved linear directions.
- **Frontier:** configurations for which no eligible comparator is both no slower and no less accurate, with one strict improvement.
- **Held-out/validation:** cases excluded from training/cases used to select model settings.
- **Oracle:** a best-found representation estimate, not a guarantee of the global optimum.
