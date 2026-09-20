# Independent report audit: p-linear

Audited from retained records on September 20. This audit recomputes aggregates independently of the lane generator and reviews the documented protocol; it does not rerun solvers or independently validate every saved field.

Source checkout: `a366980a46ce79741f7c1e12f576887678efcc6d`. Numeric checks: **408/408 pass**. Provenance checks: **546/546 pass**. Methodological findings remain open below.

## Verified-number sample

Values retain the units of their named metric; percentage metrics are explicitly suffixed. The complete machine-readable audit retains every checked row.

| Subject | Metric | Reported | Recomputed | Job | Source and calculation | Pass |
| --- | --- | ---: | ---: | --- | --- | :---: |
| `a_neural@K32_w128_L2` | `median_total_ms` | 12.9349560011 | 12.9349560011 | `3783883` | `worktrees/2026-09-17-p-linear/experiments/p-linear/artifacts/plhead1/result.json`: 1000 * median(invocations.total_seconds) | True |
| `cg_0.0001` | `median_same_grid` | 1.33936972682e-06 | 1.33936972682e-06 | `3783813` | `worktrees/2026-09-17-p-linear/experiments/p-linear/artifacts/plin1024b/result.json`: median_case max_rep same_grid_error | True |
| `d_linear_qr_m256@incumbent` | `median_device_ms` | 1.08694157097 | 1.08694157097 | `3783813` | `worktrees/2026-09-17-p-linear/experiments/p-linear/artifacts/plin1024b/result.json`: 1000 * median(invocation device solver plus initialization seconds) | True |
| `e_pod256_m256@trainset` | `worst_same_grid` | 0.0654976563401 | 0.0654976563401 | `3783813` | `worktrees/2026-09-17-p-linear/experiments/p-linear/artifacts/plin1024b/result.json`: max_case max_rep same_grid_error | True |
| `e_pod64_m4@trainset` | `worst_physical` | 0.0572767341331 | 0.0572767341331 | `3783813` | `worktrees/2026-09-17-p-linear/experiments/p-linear/artifacts/plin1024b/result.json`: max_case max_rep physical_error | True |
| `q128_m256@new_K32` | `median_total_ms` | 6.62033667322 | 6.62033667322 | `3783813` | `worktrees/2026-09-17-p-linear/experiments/p-linear/artifacts/plin1024b/result.json`: 1000 * median(invocations.total_seconds) | True |
| `q32_m4@incumbent` | `median_same_grid` | 0.00941916648926 | 0.00941916648926 | `3783813` | `worktrees/2026-09-17-p-linear/experiments/p-linear/artifacts/plin1024b/result.json`: median_case max_rep same_grid_error | True |
| `q64_m4@incumbent` | `median_same_grid` | 0.00786213006891 | 0.00786213006891 | `3783813` | `worktrees/2026-09-17-p-linear/experiments/p-linear/artifacts/plin1024b/result.json`: median_case max_rep same_grid_error | True |
| `cg_0.01` | `median_device_ms` | 22.2897790372 | 22.2897790372 | `3780692` | `worktrees/2026-09-17-p-linear/experiments/p-linear/artifacts/plin256/result.json`: 1000 * median(invocation device solver plus initialization seconds) | True |
| `d_linear_qr_m256@incumbent` | `worst_same_grid` | 0.0233531056149 | 0.0233531056149 | `3780692` | `worktrees/2026-09-17-p-linear/experiments/p-linear/artifacts/plin256/result.json`: max_case max_rep same_grid_error | True |
| `e_pod256_m4@trainset` | `worst_physical` | 0.00811693056413 | 0.00811693056413 | `3780692` | `worktrees/2026-09-17-p-linear/experiments/p-linear/artifacts/plin256/result.json`: max_case max_rep physical_error | True |
| `q0_m256@incumbent` | `median_total_ms` | 7.75477651041 | 7.75477651041 | `3780692` | `worktrees/2026-09-17-p-linear/experiments/p-linear/artifacts/plin256/result.json`: 1000 * median(invocations.total_seconds) | True |
| `q128_m4@incumbent` | `median_same_grid` | 0.00368452211194 | 0.00368452211194 | `3780692` | `worktrees/2026-09-17-p-linear/experiments/p-linear/artifacts/plin256/result.json`: median_case max_rep same_grid_error | True |
| `q32_m4@new_K32` | `median_device_ms` | 6.89274759497 | 6.89274759497 | `3780692` | `worktrees/2026-09-17-p-linear/experiments/p-linear/artifacts/plin256/result.json`: 1000 * median(invocation device solver plus initialization seconds) | True |
| `q64_m4@new_K32` | `worst_same_grid` | 0.0208082301112 | 0.0208082301112 | `3780692` | `worktrees/2026-09-17-p-linear/experiments/p-linear/artifacts/plin256/result.json`: max_case max_rep same_grid_error | True |

## Mismatches and provenance

No numerical mismatch was found among the recomputed aggregates.
The source/job checks performed here pass.

## Gate history, comparisons, retractions and weakest claims

- DESIGN amendment A8 changes the deciding interpretation of falsification after the coarse-mesh result was observed. It is prior to the fine-mesh job but post-observation for the campaign. Report literal and amended conclusions separately.
- The direct QR free-bank endpoint is appropriate; the redundant eliminated full-rank neural endpoint contains inert iterations and is not the efficient top-rung cost. The endpoint replacement is documented in amendment A4.
- Augmented best-found reconstruction with nonzero correction rank was retracted under amendment A10 because its projector was not orthonormal. The corrected implementation was checked locally, but the old full-cohort oracle column was not rerun. Do not reinstate it through later table generation.
- The within-job direct-solver and POD comparisons remain distinct from cross-mesh scaling. The head-capacity job does not justify comparing its timings to the fine-mesh job on another GPU.
- Further work: recompute the corrected oracle and diagnose the learned bank/head gap. The algebraic linear limit is supported without claiming a neural advantage over the linear controls.

This audit does not mark open findings resolved. Full-field audits already retained by the lane are supporting evidence, not new independent field checks performed here. Any altered claim must be applied to the manuscript and its source report before submission.

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
