# Independent report audit: no-second

Audited from retained records on September 20. This audit recomputes aggregates independently of the lane generator and reviews the documented protocol; it does not rerun solvers or independently validate every saved field.

Source checkout: `ea812685e7386e6af631a646e996c5428794f560`. Numeric checks: **78/78 pass**. Provenance checks: **78/78 pass**. Methodological findings remain open below.

## Verified-number sample

Values retain the units of their named metric; percentage metrics are explicitly suffixed. The complete machine-readable audit retains every checked row.

| Subject | Metric | Reported | Recomputed | Job | Source and calculation | Pass |
| --- | --- | ---: | ---: | --- | --- | :---: |
| `fno-large` | `mean_fixed_initial_error` | 0.0228109340376 | 0.0228109340376 | `3710846` | `worktrees/2026-09-17-no-second/experiments/neural-operator-audit/runs/fno_burgers02/field-audit.json`: mean over per-case maxima recomputed from per_time_errors | True |
| `fno-refine` | `mean_fixed_initial_error` | 0.0235012875205 | 0.0235012875205 | `3710846` | `worktrees/2026-09-17-no-second/experiments/neural-operator-audit/runs/fno_burgers02/field-audit.json`: mean over per-case maxima recomputed from per_time_errors | True |
| `fno-small` | `worst_fixed_initial_error` | 0.0748116589502 | 0.0748116589502 | `3710846` | `worktrees/2026-09-17-no-second/experiments/neural-operator-audit/runs/fno_burgers02/field-audit.json`: worst over per-case maxima recomputed from per_time_errors | True |
| `ctrl-medium-f64` | `worst_fixed_initial_error` | 0.0167210089329 | 0.0167210089329 | `3783831` | `worktrees/2026-09-17-no-second/experiments/no-second/runs/ctrl01/audit.json`: worst over per-case maxima recomputed from per_time_errors | True |
| `ctrl-medium-seed2` | `worst_fixed_initial_error` | 0.0154790685454 | 0.0154790685454 | `3783831` | `worktrees/2026-09-17-no-second/experiments/no-second/runs/ctrl01/audit.json`: worst over per-case maxima recomputed from per_time_errors | True |
| `ctrl-tsol-small-f64` | `worst_fixed_initial_error` | 0.0313289989887 | 0.0313289989887 | `3783831` | `worktrees/2026-09-17-no-second/experiments/no-second/runs/ctrl01/audit.json`: worst over per-case maxima recomputed from per_time_errors | True |
| `tsol-large` | `median_fixed_initial_error` | 0.0306903131372 | 0.0306903131372 | `3780139` | `worktrees/2026-09-17-no-second/experiments/no-second/runs/tsol01/audit.json`: median over per-case maxima recomputed from per_time_errors | True |
| `tsol-medium` | `median_fixed_initial_error` | 0.0160074915538 | 0.0160074915538 | `3780139` | `worktrees/2026-09-17-no-second/experiments/no-second/runs/tsol01/audit.json`: median over per-case maxima recomputed from per_time_errors | True |
| `tsol-refine` | `median_fixed_initial_error` | 0.0107633915233 | 0.0107633915233 | `3780139` | `worktrees/2026-09-17-no-second/experiments/no-second/runs/tsol01/audit.json`: median over per-case maxima recomputed from per_time_errors | True |
| `tsol-small` | `median_fixed_initial_error` | 0.0141559364191 | 0.0141559364191 | `3780139` | `worktrees/2026-09-17-no-second/experiments/no-second/runs/tsol01/audit.json`: median over per-case maxima recomputed from per_time_errors | True |
| `unet-large` | `mean_fixed_initial_error` | 0.0170294016001 | 0.0170294016001 | `3780138` | `worktrees/2026-09-17-no-second/experiments/no-second/runs/unet01/audit.json`: mean over per-case maxima recomputed from per_time_errors | True |
| `unet-medium` | `mean_fixed_initial_error` | 0.0110226167244 | 0.0110226167244 | `3780138` | `worktrees/2026-09-17-no-second/experiments/no-second/runs/unet01/audit.json`: mean over per-case maxima recomputed from per_time_errors | True |
| `unet-refine` | `mean_fixed_initial_error` | 0.0110429974346 | 0.0110429974346 | `3780138` | `worktrees/2026-09-17-no-second/experiments/no-second/runs/unet01/audit.json`: mean over per-case maxima recomputed from per_time_errors | True |
| `unet-small` | `mean_fixed_initial_error` | 0.0108582515685 | 0.0108582515685 | `3780138` | `worktrees/2026-09-17-no-second/experiments/no-second/runs/unet01/audit.json`: mean over per-case maxima recomputed from per_time_errors | True |
| `unet-small` | `worst_fixed_initial_error` | 0.0559365021818 | 0.0559365021818 | `3780138` | `worktrees/2026-09-17-no-second/experiments/no-second/runs/unet01/audit.json`: worst over per-case maxima recomputed from per_time_errors | True |

## Mismatches and provenance

No numerical mismatch was found among the recomputed aggregates.
The source/job checks performed here pass.

## Gate history, comparisons, retractions and weakest claims

- The declared main comparison is accuracy across identical cases, not speed across jobs. Resolution speed ratios are within each operator's own job; this does not establish a ROM/operator cost ranking.
- Validation selects mean case-maximum error; the selected arm need not minimize the worst case. The report discloses this choice and retains alternatives. Preserve it in paper comparisons.
- The initial protocol explicitly permits float32 U-Net/Transolver internals and supplies a float64 control. That historical exception must not silently become the new 3D policy.
- Most arms ended at wall budgets, with the still-improving label based on checkpoint recency. This is a heuristic, not a convergence proof. Describe achieved finite-budget errors; calling errors a lower bound on attainable error is mathematically misleading.
- Further work: validation-selected continuations and independent seeds, plus a same-allocation ROM/operator/FOM panel. The reported operator resolution knob must remain in the paper; the earlier no-knob premise was withdrawn.

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
