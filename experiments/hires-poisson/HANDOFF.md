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
| 4 | hpl1024 | L-shape 1024² | 4053801 | H200 | RUNNING |
| 5 | hp3d256 | cube 128³ + 256³ (DST-assembled operator, A7) | — | H200 | STAGED at runs/hp3d256, NOT submitted (account had 6 running). Submit: `cluster/submit.sh hp3d256` |
| 6 | hpl2048 | L-shape 2048² (2n reference by tight GPU CG, A5) | — | H200 | not staged yet: `stage.py hpl2048 --config config-lshape-2048.json --set lshape --driver hpl_solve.py --audit hpl_audit_np.py --hours 8` |
| 7 | hp4096b | square 4096² re-measure: f32-I/O twins + q256m8/q384m4 (A6) | — | H200 | not staged yet: `stage.py hp4096b --config config-4096b.json --memfrac 0.95 --hours 6` |

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

1. Collect `hp3d128` and `hp4096` when ALL-DONE; regenerate summary; lab-log append (flock).
2. L-shaped 1024² / 2048²: port `worktrees/2026-09-17-lshape/experiments/lshape/lsh_solve.py`
   (source commit d80fed7a) into the lane with coarse-grid arms, 5 reps, UUID guard, primary models
   only, no POD at 2048²; M=257, q ∈ {0,32,64,128}; fine reference at 2n by SuperLU. Own amendment first.
3. Speed loop re-measure ideas not yet run: pinned/f32 I/O for all subjects alike; larger M at q=256;
   sine-mode enrichment of the bank as a labelled accuracy arm (floor 0.742 % is what binds).
