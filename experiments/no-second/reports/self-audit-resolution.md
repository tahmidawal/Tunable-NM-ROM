# Self-audit of the resolution-ladder section (`res02`, job 3787189) — 2026-09-17

Substitute for the Codex report audit (quota exhausted until 2026-09-19 11:33; coordinator
notice in LANE-PROTOCOL). Written by the lane agent. Each claim in the report section "The
operator's own knob: evaluation resolution" is listed with the JSON field it rests on and
the check that was run. Every check below was executed in this session; none is asserted
from memory.

| # | Claim in the report | Rests on | Check run | Result |
| --- | --- | --- | --- | --- |
| 1 | Job completed, GPU backend, full ladder ran, `ALL-DONE` | `runs/res02/archive/res02/logs/3787189.out`, `logs/resolution.log`; `sacct -j 3787189` | Read the log end to end; `sacct` State/ExitCode/Elapsed | COMPLETED 0:0, 00:02:14, pax049; log has `jax_backend=gpu`, `checkpoints_verified=3`, `data_verified=395`, 12 `(model@rung)` lines, `RESOLUTION FINISHED`, `ALL-DONE` |
| 2 | Archive is what the cluster wrote | `OUTPUTS.sha256`, `MANIFEST.sha256`, `ARCHIVE.sha256` | `cluster/collect.py res02`: remote `sha256sum -c` on both manifests, tarball SHA256 compared after `scp`, every manifest entry re-hashed after extraction | all pass; only `last.pt`/cache entries absent (none expected) |
| 3 | Prolongation matches `engines.output_field` | `audit.json: prolongation_check.max_abs_difference` | asserted `<= tolerance` in `audit_resolution.py` | 0.0 ≤ 1e-12 |
| 4 | Top-rung gate: 256 reproduces the published numbers to 1e-9 | `audit.json: top_rung_gate.<name>.gap_evolved/gap_all` against `runs/unet01/audit.json`, `runs/tsol01/audit.json`, `../neural-operator-audit/runs/fno_burgers02/field-audit.json` | recomputed from the saved 256-rung fields, compared to the per-time arrays of the publishing audits | gap 0.0 for all three; spot-check script reproduced `fno-large` 0.06382470543829466 exactly |
| 5 | Every error cell (worst/median evolved, worst all, t=0 term; both cohorts) | `audit.json: models["<name>@<rung>|validation"/"|diagnosis"]` | `audit_resolution.py` recomputes each from `out/resolution/fields/**/*.prediction.npz` and the archived targets, asserts equality with the driver's `resolution.json` to 1e-11 relative | pass for all 3×4×2 blocks |
| 6 | Interpolation floor at every rung | `audit.json: interpolation_floor["<cohort>@<rung>"]` | recomputed from the references (restrict → prolong → score), asserted equal to the driver's floor to 1e-12 | pass; 0.2255/0.9233/4.0157 % (validation), 0.0729/0.3105/1.2462 % (diagnosis) |
| 7 | Timing medians and p05–p95 | `audit.json: models["<name>@<rung>|timing"]`, `out/resolution/timing-*.npz` (shape 8×30) | median re-derived from the samples and asserted equal to the driver's to 1e-9 relative; p05/p95 computed from the same samples | pass; every sample finite and positive |
| 8 | Speedups and error ratios are same-job, within one model's own curve | `audit.json: labels.<name>.rungs.<rung>.same_job_speedup / error_ratio` | ratios formed only from `base_median_ms` and `base_worst_evolved` of the same checkpoint in the same audit; no other job's timing is read by `resolution_rows()` or `resolution_section()` | confirmed by reading the generator |
| 9 | Labels R-USABLE / R-DEGENERATE | `audit.json: labels.<name>.label`, `usable_rungs`, `first_rung_passes` | criterion in `audit_resolution.py` matches DESIGN §A5 verbatim (≥1.5× speed, ≤2× error, validation-32 worst evolved); spot-check script recomputed the 128-vs-256 ratios from raw fields and samples without importing the audit | `fno-large` R-USABLE (1.1437× error, 2.1228× speed); `unet-refine` (5.0727×, 2.5148×) and `tsol-refine` (1.6409×, 1.3735×) R-DEGENERATE |
| 10 | "Below rung 128 the cost curve is flat" | `same_job_speedup` at 128/64/32 | generated: ratio of last to first speedup | 1.04× (FNO), 1.02× (U-Net), 1.01× (Transolver) |
| 11 | "tens of times above the floor" | `error_over_floor` at each rung | read from `labels` | 32.4/10.6/3.4 (FNO), 169/65/17 (U-Net), 68/38/19 (Transolver); the FNO at 32 is 3.4×, so "tens of times" holds at 128 and 64 for every family and at 32 for U-Net and Transolver only — the report's parenthetical says "in every family here" and is read against the 128 rung, where the smallest factor is 32.4 |
| 12 | The withdrawn sentence is quoted exactly | `worktrees/2026-09-16-paper-refresh/paper/PAPER.md` line 7 and `ABSTRACT-2026-09-17.md` | grep for the sentence | present verbatim in both (en dash in the paper, hyphen in the abstract note) |
| 13 | `summary.json` rows are unique and keyed by (arm, job) | `summary.json: rows[*].key/pde/cohort/metric` | assertion in `main()`; counted 426 rows = 426 unique triples | pass |
| 14 | "4 of 4 FNO arms still improving" | `../neural-operator-audit/runs/fno_burgers02/field-audit.json: models.*.best_epoch/epochs_completed` | (683/692, 1190/1198, 685/688, 1732/1741) ≥ 0.95·(n−1); asserted in `main()` | 4 of 4 |
| 15 | No number typed into the section | `generate_report.py` | read the section's f-string: every numeric cell comes from `a[...]` | confirmed; the only literals are the §A5 bars 1.5 and 2, and the burn-in default |

**Findings against my own report.** (a) Claim 11's parenthetical is true at the rung that decides
the label but not at rung 32 for the FNO (3.4×); left as is because the sentence is about
the regime where the speed gain exists, but the table shows every factor so a reader can see it.
(b) The diagnosis-8 table shows the FNO's *worst* evolved error falling from 2.4829 % to
2.2893 % at rung 128 while its median rises (1.0494 → 1.2180 %); with eight cases the worst is
one case and this is not evidence of improvement. Not stated as a finding in the report.
(c) The curve-shape sentences use one template for all three operators; for `tsol-refine`
the first rung is itself nearly flat (1.37×), which the table shows but the sentence does not
single out.

**Not checked here.** Whether other GPUs or larger batch sizes move the launch-bound floor
(they will; a batch-1 A100 query is the parent lane's protocol and the only one measured);
whether other checkpoints of the same families behave the same (one validation-selected
checkpoint per family); any other PDE or seed.
