# hires-poisson — HANDOFF (kept current)

**Lane:** worktree `worktrees/2026-09-20-hires-poisson`, branch `exp/2026-09-20-hires-poisson`,
namespace `/cluster/tufts/paralab/tawal01/hires_p_20260920/`. Contract:
`reports/2026-09-20-speed-accuracy-campaign-protocol.md` (main). Design + amendments A1–A4:
`DESIGN.md`. Speed loop: `SPEED-LOG.md`. Generated numbers: `reports/summary.json`,
`reports/tables.generated.md` (run `reports/generate_hires.py` after every collection).

## STATE: LANE CLOSED (2026-09-21) — budget 8/8 spent, every job collected, namespace empty, nothing running

## Jobs

| # | attempt | mesh | job id | result |
|---|---|---|---|---|
| 1 | hp2048 | square 2048² | 4049279 | audited; 0.965 % / 28.7× vs CG 1e-2 → bar MET |
| 2 | hp3d128 | cube 64³ + 128³ | 4051032 | audited; 128³ 0.160 % / 2.70× → bar MISSED (speed) |
| 3 | hp4096 | square 4096² | 4051236 | audited; 0.965 % / 41.1× (146× device) → bar MET |
| 4 | hpl1024 | L-shape 1024², 12 sources | 4053801 | audited; bar verdict WITHDRAWN (A8, subset); timings stand |
| 5 | hp3d256 | cube 128³ + 256³ | 4056288 | audited; 256³ 0.160 % / 4.18× (23× device) → bar MISSED (speed) |
| 6 | hpl32 | L-shape 1024² + 2048², 32 sources | 4057694 | 1024²: audited, 2.196 % / 9.76× → MISSED (accuracy). 2048²: audit rejected the CG 2n reference (A10) |
| 7 | hpl32fix | repair of that reference + unchanged audit | 4071217 | audited; 2048²: 2.195 % / 19.3× → MISSED (accuracy) |
| 8 | hp4096b | square 4096² re-measure (f32-I/O, q256m8, q384m4, CG 0.3/0.2/0.1) | 4071227 | audited; Slurm FAILED 1:0 = wrapper bug after the audit (A11), collected by hand with verified checksums; 37.6× named, 27.4× vs matched CG 0.2 |

## How to run

```bash
PY=/home/tahmid/Dev/.venv/bin/python
$PY experiments/hires-poisson/cluster/stage.py <attempt> --config <cfg> [--set 3d --driver hp3d_solve.py --audit hp3d_audit_np.py --subsample 32] [--memfrac 0.95]
experiments/hires-poisson/cluster/submit.sh <attempt>      # squeue before/after, one dir per job, refuses if account has 6 running
experiments/hires-poisson/cluster/collect.sh <attempt>     # requires ALL-DONE; checksum pull; remote delete
$PY experiments/hires-poisson/reports/generate_hires.py
```
The job runs the driver, then the independent NumPy audit ON THE CLUSTER (fields are tens of
GB), which writes `audit.json` + strided subsamples and deletes the full fields only if every
gate passed. `runs/*/archive/output/sub/` is pulled but git-ignored.

## Next steps (for whoever continues; no GPU budget is left in this lane)

1. Nothing to collect. To regenerate everything: `reports/generate_hires.py` (summary.json, tables, report);
   `SPEED-LOG.md` is generated from the same audits (script inline in the git history of that file's commit).
2. The coordinator owns the "Where things stand" block of the lab log; this lane only appended dated entries
   (INTERIM 1–6 and the closing entry). Fold the closing entry's verdict table in there.
3. Ask the user whether to merge or archive this worktree (repo rule); not merged, not pushed.
4. Unrun ideas are listed at the bottom of `SPEED-LOG.md`.
