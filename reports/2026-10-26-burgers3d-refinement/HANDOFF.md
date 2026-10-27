# Burgers 3D refinement handoff

Status: checkpoint, work ongoing. Completed allocations total 1.64417 GPU-hours against the
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
| solver | podtune03 | 6528236 | FAILED | 0.00916667 |
| solver | podtune04 | 6528531 | FAILED | 0.00861111 |
| solver | podtune05 | 6529066 | COMPLETED | 0.0366667 |
| solver | front6402 | 6533177 | RUNNING | active/unaccounted |
| solver | mesh01 | 6541155 | FAILED | 0.154722 |
| pod | pilot | 6525154 | FAILED_FINAL_PRINT_AFTER_COMPLETE_CHECKPOINT | 0.0255556 |
| pod | fit32 | 6525584 | COMPLETED_CONSTRUCTION_REJECTED | 0.92 |
| pod | none64 | 6525585 | COMPLETED | 0.0591667 |
| pod | k16fine01 | 6529465 | COMPLETED_FAILED_HR_ADMISSION | 0.0213889 |

Foreground queue check at `2026-10-27T04:12:14.137999+00:00`: `[{"job_id": 6533177, "workdir": "/cluster/tufts/paralab/tawal01/b3d-solver-20261026/front6402", "state": "RUNNING", "name": "b3ds_front6402"}]`.
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

## Deferred, conditional and still-open work

- Same-allocation complete-validation endpoint comparison, own new-policy qualification and paired accuracy-matched full-solver/POD costs.
- Bounded fixed-mode mesh/output costs, explicit operator interventions and same-reference projection floor.
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


## Glossary

- Lane: an isolated worktree and cluster namespace for one scientific question.
- GPU-hour: allocated GPU count multiplied by elapsed hours, including failed attempts.
- QN: nonlinear solve that reuses its small matrix; a fresh restart resets that cache.
- Q1/Q2: nonlinear-solver and quadrature qualification on the declared case population.
- POD/HR/FOM: a standard reduced basis / weighted sampling of its equations / a full solver.
- Frontier: admissible settings not beaten in both cost and the chosen error metric.
- Diagnostic: explains selected cases; it is not a population performance estimate.
