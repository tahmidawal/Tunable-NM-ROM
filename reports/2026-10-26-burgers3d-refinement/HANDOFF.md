# Burgers 3D refinement handoff

Status: checkpoint, work ongoing. Completed allocations total 1.41361 GPU-hours against the
predeclared campaign cap 12; live jobs are listed explicitly and are not counted as zero.

## Work locations

- Coordinator/report: `worktrees/2026-10-26-b3d-refine`, branch `exp/2026-10-26-b3d-refine`.
- Solver/quadrature: `worktrees/2026-10-26-b3d-solver`, branch `exp/2026-10-26-b3d-solver`.
- POD comparator: `worktrees/2026-10-26-b3d-pod`, branch `exp/2026-10-26-b3d-pod`.
- Canonical log: `/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md`.
- New work must retain the consolidated fork recorded in CLAUDE.md, not the frozen main baseline.

## Exact job checkpoint

| Lane | Attempt | Job | State | Completed GPU-hours |
| --- | --- | --- | --- | --- |
| solver | spilot02 | 6525183 | COMPLETED | 0.0177778 |
| solver | sdiag01 | 6525518 | COMPLETED | 0.0536111 |
| solver | qpilot01 | 6525527 | FAILED | 0.00361111 |
| solver | qpilot02 | 6525763 | FAILED | 0.00388889 |
| solver | qpilot03 | 6526456 | COMPLETED | 0.0155556 |
| solver | qcheck801 | 6527167 | COMPLETED | 0.0297222 |
| solver | qwitness01 | 6527478 | COMPLETED | 0.0188889 |
| solver | tdiag02 | 6527289 | COMPLETED | 0.198611 |
| solver | podtune02 | 6527566 | COMPLETED | 0.0672222 |
| pod | pilot | 6525154 | FAILED_FINAL_PRINT_AFTER_COMPLETE_CHECKPOINT | 0.0255556 |
| pod | fit32 | 6525584 | COMPLETED_CONSTRUCTION_REJECTED | 0.92 |
| pod | none64 | 6525585 | COMPLETED | 0.0591667 |

Inspect each ledger's exact directory, Slurm state and existing outputs before any restart.
Do not duplicate a submitted job. Use direct staging to paralab and checksummed collection.
The POD final-print failure recovered complete numerical output; it did not warrant a rerun.
The quadrature cohort-identity preflight stop is a provenance failure until resolved, not a
measured quadrature failure. Keep every failed attempt in the ledger.

## Completed and accepted evidence

The frozen historical inventory and exact timing-output re-scoring passed independent review
for their declared scope. Selected-case solver and POD diagnostics have separate independent
results gates. Their approved scope does not promote them into a whole-validation frontier.
See the generated report and the snapshotted lane exports for numerical details and current
whole-validation qualification. Every scientific number is generated from saved JSON inputs.

## Failed or retracted interpretations

- A configuration-map count does not cover later round-only qualified candidates.
- A reused-QN fresh restart does not restore the original cached nonlinear solver state.
- Pooled stationarity alone does not guarantee agreement with a tight full trajectory.
- Trust clipping alone is not established as the cause of the replayed convergence failure.
- Certificate repair alone is not an accuracy improvement.
- Old POD observed domination does not supply the newly requested HR convergence gate.
- Neither the new frozen-support failures nor an unsampled POD comparison establish a
  general advantage over tuned qualified POD.
- Historical mesh-free deployment curves are not fixed-mode full-field output curves.

## Deferred or still open unless closed in the final exports

Missing quadrature successor evidence; new-policy own-state quadrature and complete nonlinear
qualification; time-step separation; complete matched solver/POD/FOM accuracy–cost comparisons;
fixed-mode mesh and output phase costs; larger-support training fit and its independent
admission. The report must label every unresolved part instead of filling it with an estimate.
Fresh 3D extrapolation-regime cohorts, bank retraining and sealed replication are not implied.

## Preservation and next-session rules

All original Research Update slide bytes and their order are preserved; generated refinement
slides are appended and original PDF hashes/backups are recorded. Read the latest canonical
saved user edits again before modifying presentation sources. Keep code-only mirrors current;
never push archive-bearing experimental branches directly. Do not merge any lane without the
user's decision. No unique data may be deleted; remote job directories are removed only after
verified complete collection. No unsupported background watcher is relied on.


The actual fitting matrix (13.6902 GB) is retained at `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-26-b3d-pod/experiments/b3d-pod/runs/fit32/collected/output/contributions.npy` with SHA256 `0bd92ac754fabccb89652d47f80f068dba2c1a7dd069cd2872cdee4f1816b67e`. It is outside Git; the durable retention manifest is [LOCAL-RETENTION.json](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-26-b3d-pod/experiments/b3d-pod/runs/fit32/collected/LOCAL-RETENTION.json). Preserve or move this data before any worktree archival. Candidate nodes and the target are listed by that manifest.


## Glossary

- Lane: an isolated worktree and cluster namespace for one scientific question.
- GPU-hour: allocated GPU count multiplied by elapsed hours, including failed attempts.
- QN: nonlinear solve that reuses its small matrix; a fresh restart resets that cache.
- Q1/Q2: nonlinear-solver and quadrature qualification on the declared case population.
- POD/HR/FOM: a standard reduced basis / weighted sampling of its equations / a full solver.
- Frontier: admissible settings not beaten in both cost and the chosen error metric.
- Diagnostic: explains selected cases; it is not a population performance estimate.
