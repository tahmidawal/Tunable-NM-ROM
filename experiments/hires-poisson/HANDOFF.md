# hires-poisson — HANDOFF (kept current)

**Lane:** worktree `worktrees/2026-09-20-hires-poisson`, branch `exp/2026-09-20-hires-poisson`,
namespace `/cluster/tufts/paralab/tawal01/hires_p_20260920/`. Contract:
`reports/2026-09-20-speed-accuracy-campaign-protocol.md` (main). Design + amendments A1–A4:
`DESIGN.md`. Speed loop: `SPEED-LOG.md`. Generated numbers: `reports/summary.json`,
`reports/tables.generated.md` (run `reports/generate_hires.py` after every collection).

## Jobs (budget 8 total, 2 running)

| # | attempt | mesh | job id | GPU | state |
|---|---|---|---|---|---|
| 1 | hp2048 | square 2048² | 4049279 | H200 | DONE, audited on cluster, collected, remote deleted. Bar MET (0.965 %, 28.7× vs CG 1e-2) |
| 2 | hp3d128 | cube 64³ + 128³ | 4051032 | H200 | DONE, audited, collected, remote deleted. 128³: 0.160 %, 2.70× vs CG 1e-2 → bar MISSED on speed (I/O-bound; 6.8× device) |
| 3 | hp4096 | square 4096² | 4051236 | H200 | DONE, audited, collected, remote deleted. Bar MET (0.965 %, 41.1× total / 146× device) |
| 4 | hpl1024 | L-shape 1024², 12 sources | 4053801 | H200 | DONE, audited, collected, remote deleted. 0.895 % / 9.88× — **bar verdict WITHDRAWN (A8): 12-source subset excludes the hard sources**; timings stand |
| 5 | hp3d256 | cube 128³ + 256³ (DST-assembled operator, A7) | 4056288 | H200 | RUNNING |
| 6 | hpl32 | L-shape 1024² AND 2048², all 32 development sources (two driver runs → output/, output2/) | 4057694 | H200 | RUNNING/PENDING; ~3–4 h (2n=4096 reference by tight GPU CG) |
| 7 | hp4096b | square 4096² re-measure: f32-I/O twins, q256m8, q384m4, CG 0.3/0.2/0.1/0.01 (A6, A9) | — | H200 | STAGED at runs/hp4096b; a background chain submits it when the lane has < 2 jobs and the account < 6. If the session died: `cluster/submit.sh hp4096b` |
| 8 | (spare) | — | — | — | one job left in the budget |

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

## Next steps

1. Collect each job when its log says ALL-DONE: `cluster/collect.sh <attempt>` (pulls output/ and output2/, verifies
   checksums, deletes the remote dir), then `reports/generate_hires.py`, then a lab-log append under flock
   (`reports/make_lab_entry.py <attempt>` prints the generated paragraph; hpl32 has two audits — output2 is 2048²).
2. After all jobs: final lab-log entry + rewrite nothing on main except the append; commit; do not push.
3. Ideas not run (spare job): sine-mode enrichment of the bank to break the 0.742 % floor (labelled accuracy arm);
   a 1024² square anchor on the same H200 for a three-point same-GPU mesh series; pinned host buffers.
