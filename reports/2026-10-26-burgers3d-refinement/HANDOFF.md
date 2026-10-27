# Burgers 3D refinement handoff

Status: bounded campaign closed. Completed allocations total 3.90333 GPU-hours against the
predeclared campaign cap 12; live jobs are listed explicitly and are not counted as zero.

## Work locations

- Coordinator/report: `worktrees/2026-10-26-b3d-refine`, branch `exp/2026-10-26-b3d-refine`.
- Solver/quadrature: `worktrees/2026-10-26-b3d-solver`, branch `exp/2026-10-26-b3d-solver`.
- POD comparator: `worktrees/2026-10-26-b3d-pod`, branch `exp/2026-10-26-b3d-pod`.
- Canonical log: `/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md`.
- [Consolidated report](2026-10-27-burgers3d-refinement.md) and [updated Research Update PDF](../2026-10-23-research-update/Research-Update.pdf).
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
| solver | podtune03 | 6528236 | FAILED | 0.00916667 |
| solver | podtune04 | 6528531 | FAILED | 0.00861111 |
| solver | podtune05 | 6529066 | COMPLETED | 0.0366667 |
| solver | front6402 | 6533177 | COMPLETED | 1.7575 |
| solver | mesh01 | 6541155 | FAILED | 0.154722 |
| solver | mesh02 | 6542062 | COMPLETED | 0.501667 |
| pod | pilot | 6525154 | FAILED_FINAL_PRINT_AFTER_COMPLETE_CHECKPOINT | 0.0255556 |
| pod | fit32 | 6525584 | COMPLETED_CONSTRUCTION_REJECTED | 0.92 |
| pod | none64 | 6525585 | COMPLETED | 0.0591667 |
| pod | k16fine01 | 6529465 | COMPLETED_FAILED_HR_ADMISSION | 0.0213889 |

Foreground queue check at `2026-10-27T05:30:46.487534+00:00`: `[]`.
Live job ids absent from the source lane ledgers: `[]`. Any such lag is explicit;
active allocation costs are still unaccounted rather than zero.

Inspect each ledger's exact directory, Slurm state and existing outputs before any restart.
Do not duplicate a submitted job. Use direct staging to paralab and checksummed collection.
The POD final-print failure recovered complete numerical output; it did not warrant a rerun.
The quadrature cohort-identity preflight stop is a provenance failure until resolved, not a
measured quadrature failure. Keep every failed attempt in the ledger.

## Completed and accepted evidence

- Historical inventory, source hashes, exact saved timing-output re-scoring and generated figures.
- Missing original finer-rule witness joined to inherited complete-population evidence; legacy gl56 qualification closes within its original scope.
- Audited time-step diagnostic: median improves while worst worsens; coarse/fine endpoint choice is post-diagnostic and the valid intermediate is retained.
- Complete-validation unsampled POD fallback with independent reference/array audit; its separate allocation supplies no cross-job speed ratio.
- Frozen same-allocation complete-validation frontier: 16/16 subjects pass their registered method-specific admission. All declared timing-coverage masks remain explicit; final POD costs come from this same allocation, separately from the old fallback job.
- A faster spectral member jointly matches conditional median/worst accuracy for 7/7 tested bank settings. Refreshed-QN time refinement improves the median while worsening the worst case and cost; no universal method advantage follows.
- Frozen complete bounded mesh study: fixed reduced settings, separate fixed-output/growing-output tasks and supplied-coefficient operator controls. Qualification remains selected-case/common-reference only; no exponent or general POD mesh advantage is inferred.

Selected-case diagnostics retain their separate scope and independent gates. See the
generated report and frozen exports for exact cases, current admission and provenance.
Scientific numbers are generated from saved JSON inputs; publication review is a separate gate.

## Failed experiments and controls

- Historical quadrature/nonlinear failures and square-system controls remain in the inventory.
- Mesh attempt mesh01, job 6541155: frozen120GB checkpoint guard; collection pending. Its allocation remains in the resource total; an incomplete checkpoint is not an admitted mesh dataset.
- Sole larger-support constructor: incomplete (KKT violated), terminal support 13673/16384. No admitted support or rollout; no fit retry.
- The existing k16 support at the selected improved policy passes both own-objective Q1 checks, but fails common field agreement and stricter whole-objective gates. No complete-validation HR release; scalar reference-error changes do not demonstrate a physical-accuracy loss.

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
- The unstaged worst-only time-endpoint rejection rule was abandoned, not called predeclared.
- The inherited bank uses analytic advection with discrete diffusion, not FOM-exact upwind.
  Solver/quadrature certification does not prove identical discrete equations or spatial consistency.
- Changing the step or diffusion eigenvalues also changes inherited residual-row weights;
  those interventions do not isolate pure temporal truncation or physical diffusion bias.

## Deferred, conditional and still-open work

- Qualified hyper-reduced POD remains unestablished within the tested supports/policies. No further fit or policy ladder is released by the completed campaign.
- Sealed/out-of-distribution replication, bank retraining, further support fitting and branch merges are outside the released campaign.

The report must label every unresolved part rather than fill it with an estimate.

## Preservation and next-session rules

All original Research Update slide bytes and their order are preserved; generated refinement
slides are appended and original PDF hashes/backups are recorded. Read the latest canonical
saved user edits again before modifying presentation sources. Keep code-only mirrors current;
never push archive-bearing experimental branches directly. Do not merge any lane without the
user's decision. No unique data may be deleted; remote job directories are removed only after
verified complete collection. No unsupported background watcher is relied on.


The actual fitting matrix (13.6902 GB) is retained at `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-26-b3d-pod/experiments/b3d-pod/runs/fit32/collected/output/contributions.npy` with SHA256 `0bd92ac754fabccb89652d47f80f068dba2c1a7dd069cd2872cdee4f1816b67e`. It is outside Git; the [portable retention manifest](data/pod-fit32-retention.json) has SHA256 `19b8f62d573b223139c3dd2be4d3197c788eda93bf87b71e18fee1b3000d2276` and its original is `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-26-b3d-pod/experiments/b3d-pod/runs/fit32/collected/LOCAL-RETENTION.json`. Preserve or move this data before any worktree archival. Candidate nodes and the target are listed by that manifest.


Solver attempt `front6402` / job 6533177 retains 384 bulk arrays (4.3722 GB) outside Git. The [portable retention manifest](data/solver-retention-front6402.json) has SHA256 `0560db049551617d55d6cbb133715d55b8f4cae585fed560c7d0502106f7d475`; its original is `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-26-b3d-solver/experiments/b3d-solver/results/front6402/RETENTION.json`. The verified collection archive is `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-26-b3d-solver/experiments/b3d-solver/runs/front6402/collection.tar` with SHA256 `6b88621e9e363b309f0772710182ade7fcf57594a48f27678b589ec6bc65bd31`. Before archiving/removing this worktree, copy all excluded arrays or the verified collection archive to permanent storage and verify these hashes. Ignored files are not preserved by managed worktree archival.


Solver attempt `mesh01` / job 6541155 retains 348 bulk arrays (1.64114 GB) outside Git. The [portable retention manifest](data/solver-retention-mesh01.json) has SHA256 `ae2b91e877ff46a506cf4e87ea23e0c565c0991acf7e97f3c551c35cbf8aab04`; its original is `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-26-b3d-solver/experiments/b3d-solver/results/mesh01/RETENTION.json`. The verified collection archive is `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-26-b3d-solver/experiments/b3d-solver/runs/mesh01/collection.tar` with SHA256 `74a9c667e97b438485c4e9ec882f55743b86882ca1948f997a4ca47762362940`. Before archiving/removing this worktree, copy all excluded arrays or the verified collection archive to permanent storage and verify these hashes. Ignored files are not preserved by managed worktree archival.


Solver attempt `mesh02` / job 6542062 retains 648 bulk arrays (2.21026 GB) outside Git. The [portable retention manifest](data/solver-retention-mesh02.json) has SHA256 `205a9ffba85f4272a5311ea0d60b1db5f6dbbaa4ee1960038f244b36c8cae05b`; its original is `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-26-b3d-solver/experiments/b3d-solver/results/mesh02/RETENTION.json`. The verified collection archive is `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-26-b3d-solver/experiments/b3d-solver/runs/mesh02/collection.tar` with SHA256 `297a06edfb9dbffb58e43e74ea55015c763f329004162e48eef76456a21c2bc6`. Before archiving/removing this worktree, copy all excluded arrays or the verified collection archive to permanent storage and verify these hashes. Ignored files are not preserved by managed worktree archival.


## Glossary

- Lane: an isolated worktree and cluster namespace for one scientific question.
- GPU-hour: allocated GPU count multiplied by elapsed hours, including failed attempts.
- QN: nonlinear solve that reuses its small matrix; a fresh restart resets that cache.
- Q1/Q2: nonlinear-solver and quadrature qualification on the declared case population.
- POD/HR/FOM: a standard reduced basis / weighted sampling of its equations / a full solver.
- Frontier: admissible settings not beaten in both cost and the chosen error metric.
- Diagnostic: explains selected cases; it is not a population performance estimate.
