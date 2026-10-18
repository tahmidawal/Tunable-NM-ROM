# Independent report audit: w-ladder

Audited from retained records on September 20. This audit recomputes aggregates independently of the lane generator and reviews the documented protocol; it does not rerun solvers or independently validate every saved field.

Source checkout: `9b84d0556888ebc052b52bd165a61dcd56b2b53d`. Numeric checks: **477/477 pass**. Provenance checks: **795/795 pass**. Methodological findings remain open below.

## Verified-number sample

Values retain the units of their named metric; percentage metrics are explicitly suffixed. The complete machine-readable audit retains every checked row.

| Subject | Metric | Reported | Recomputed | Job | Source and calculation | Pass |
| --- | --- | ---: | ---: | --- | --- | :---: |
| `cg_1e-06` | `median_complete_ms` | 1278.83663448 | 1278.83663448 | `3780450` | `worktrees/2026-09-17-w-ladder/experiments/w-ladder/artifacts/wl1024b/result.json`: 1000 * median device_plus_output_transfer_seconds | True |
| `dst` | `worst_t0_energy_state` | 0 | 0 | `3780450` | `worktrees/2026-09-17-w-ladder/experiments/w-ladder/artifacts/wl1024b/result.json`: max_case energy_state.initial_normalized[0] | True |
| `linear_bank64_cn` | `worst_energy_state` | 0.0792225216717 | 0.0792225216717 | `3780450` | `worktrees/2026-09-17-w-ladder/experiments/w-ladder/artifacts/wl1024b/result.json`: max_case max_time energy_state.initial_normalized, repetition=0 | True |
| `nested_q8` | `median_gpu_ms` | 206.632015062 | 206.632015062 | `3780450` | `worktrees/2026-09-17-w-ladder/experiments/w-ladder/artifacts/wl1024b/result.json`: 1000 * median complete_device_query seconds | True |
| `pod_k64` | `median_energy_state` | 0.00959376485456 | 0.00959376485456 | `3780450` | `worktrees/2026-09-17-w-ladder/experiments/w-ladder/artifacts/wl1024b/result.json`: median_case max_time energy_state.initial_normalized, repetition=0 | True |
| `cg_1e-06` | `worst_velocity` | 0.00440138575387 | 0.00440138575387 | `3783805` | `worktrees/2026-09-17-w-ladder/experiments/w-ladder/artifacts/wl256c/result.json`: max_case max_time velocity.initial_normalized, repetition=0 | True |
| `dst_coarse64` | `worst_evolved_energy_state` | 0.0641197761864 | 0.0641197761864 | `3783805` | `worktrees/2026-09-17-w-ladder/experiments/w-ladder/artifacts/wl256c/result.json`: max_case max_evolved_time energy_state.initial_normalized | True |
| `linear_bank64_rk4` | `worst_displacement` | 0.0138952679234 | 0.0138952679234 | `3783805` | `worktrees/2026-09-17-w-ladder/experiments/w-ladder/artifacts/wl256c/result.json`: max_case max_time displacement.initial_normalized, repetition=0 | True |
| `pod_k128` | `median_evolution_ms` | 0.8668435039 | 0.8668435039 | `3783805` | `worktrees/2026-09-17-w-ladder/experiments/w-ladder/artifacts/wl256c/result.json`: 1000 * median evolution seconds | True |
| `rk4_fom` | `median_complete_ms` | 40.0827190606 | 40.0827190606 | `3783805` | `worktrees/2026-09-17-w-ladder/experiments/w-ladder/artifacts/wl256c/result.json`: 1000 * median device_plus_output_transfer_seconds | True |
| `cgdt_0.005_tol_0.01` | `worst_t0_energy_state` | 4.04320142169e-15 | 4.04320142169e-15 | `3780447` | `worktrees/2026-09-17-w-ladder/experiments/w-ladder/artifacts/wl64b/result.json`: max_case energy_state.initial_normalized[0] | True |
| `linear_bank64` | `worst_energy_state` | 0.0493231069498 | 0.0493231069498 | `3780447` | `worktrees/2026-09-17-w-ladder/experiments/w-ladder/artifacts/wl64b/result.json`: max_case max_time energy_state.initial_normalized, repetition=0 | True |
| `nested_q32` | `median_gpu_ms` | 2500.06009161 | 2500.06009161 | `3780447` | `worktrees/2026-09-17-w-ladder/experiments/w-ladder/artifacts/wl64b/result.json`: 1000 * median complete_device_query seconds | True |
| `pod_k40` | `median_energy_state` | 0.0240994452374 | 0.0240994452374 | `3780447` | `worktrees/2026-09-17-w-ladder/experiments/w-ladder/artifacts/wl64b/result.json`: median_case max_time energy_state.initial_normalized, repetition=0 | True |
| `trained_nested40` | `worst_velocity` | 0.0512872839989 | 0.0512872839989 | `3780447` | `worktrees/2026-09-17-w-ladder/experiments/w-ladder/artifacts/wl64b/result.json`: max_case max_time velocity.initial_normalized, repetition=0 | True |

## Mismatches and provenance

No numerical mismatch was found among the recomputed aggregates.
The source/job checks performed here pass.

## Gate history, comparisons, retractions and weakest claims

- DESIGN amendment A2 introduces an integrator tie band after the local smoke showed a strict head accuracy advantage. It predates cluster jobs but is still informed by observed numerical results. Report strict and banded D1, not only the amended verdict.
- The tie band is the difference between worst-case errors from two integrators; it is an empirical diagnostic rather than a certified bound on every case's integration error.
- Closed-form linear evolution and nonlinear stepping have different algorithmic costs. The matched-integrator control helps distinguish this from a neural-manifold cost effect and should remain visible.
- The retained-value gate was changed after a failed job under amendment A4 to handle tied initializers. Preserve the failure and the field/objective parity evidence; do not describe every original gate as having passed.
- Further work: initial/velocity representation and physical trajectory accuracy. Energy conservation alone does not establish accurate wave propagation. The linear-bank result is near the best corrected head in accuracy, not uniformly strictly more accurate.

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
