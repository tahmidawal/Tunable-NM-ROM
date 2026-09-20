# HANDOFF — hires-heat lane (kept current; read this first after any interruption)

Branch `exp/2026-09-20-hires-heat`, lane dir `experiments/hires-heat/`, cluster namespace `/cluster/tufts/paralab/tawal01/hires_h_20260920/<job>`. Design: `DESIGN.md`. Codex design audit: `CODEX-DESIGN-AUDIT.txt` (15 findings; dispositions in `CODEX-DESIGN-AUDIT-DISPOSITION.md`). Speed loop: `SPEED-LOG.md`.

## Mechanics
- Submit (clean committed tree only): `cluster/submit.sh <job> <config> <gpu> <HH:MM:SS> <mem> [trainname]`.
- Collect + checksum + NumPy/SciPy audit + generated summary: `cluster/collect.sh <job>`; add `--remove` to delete the remote dir after the audit passes.
- Generated per-job outputs: `runs/<job>/summary.json`, `runs/<job>/SUMMARY.md`, `runs/<job>/audit.json`.

## Job ledger (budget: 8 total, 2 running)
(see bottom of file; appended at each submit/collect)

| job | slurm id | config | GPU | submitted (UTC) | state |
|---|---|---|---|---|---|
| h2d-ladder01 | 4051290 | configs/h2d-ladder01.json (audited R=32 checkpoint, q ladder, 256→4096) | H200 pax011 | 2026-09-20 ~23:20 | running |
| h2d-wide02 | 4051298 | configs/h2d-wide02.json (train R=128 bank + K=8/16 heads, panel 256/1024) | A100 pax105 | 2026-09-20 ~23:20 | running |
| h3d-profile03 | — | configs/h3d-profile03.json (frozen paper-h3d K32 checkpoint, 64³/128³, stepping + speed arms) | H200 | waiting for a slot | prepared |

Source commit of jobs 1-2: 1cc6920f. Local findings so far (smoke only, not results): the audited 2D bank floor is ~1-1.7 % at t=0 so its ladder cannot reach 1 %; the 3D paper-h3d 0.754 % headline is evolved-times only (t=0 moments fit is 1.9 %, field fit ~1.1 % = bank floor); at K+q=R the corrected ROM coincides with the linear-bank solve.
| h2d-wide02 | 4051298 | CANCELLED by owner after 25 min: bank trained in 58 s (validation projection floor 0.114 %), then XLA stalled >20 min compiling the head step because `train.py` closed the jit over the training arrays (the CLAUDE.md captured-constant landmine, my port's fault). No results used. Fixed: arrays are jit arguments. Counts as job 2 of 8. | A100 | | cancelled |
