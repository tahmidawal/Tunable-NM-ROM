# HANDOFF — hires-burgers (kept current; read this first after an interruption)

**Lane:** `experiments/hires-burgers/` in `worktrees/2026-09-20-hires-burgers`
(`exp/2026-09-20-hires-burgers`, fork `exp/2026-09-17-b-panel` @ `25434a27`). Cluster namespace
`/cluster/tufts/paralab/tawal01/hires_b_20260920/`. Contract:
`reports/2026-09-20-speed-accuracy-campaign-protocol.md` (on main). Design: `DESIGN.md`.

## State (update at every milestone)

- 2026-09-20: design written; code written (`hops.py` Φ-free operators, `hfast.py` structured
  accurate-rung kernel + Φ-free dense truth, `hires.py` driver, `audit_hires.py`,
  `cluster/{stage,collect}.py`). b-speed kernels and b-eqtop rules were ALREADY in the fork
  point byte-identical (DESIGN §3) — nothing copied.
- Jobs used: 0 / 8. Running: none.

## How to run a job

```bash
cd worktrees/2026-09-20-hires-burgers        # commit first: stage.py refuses uncommitted files
PY=/home/tahmid/Dev/.venv/bin/python
$PY experiments/hires-burgers/cluster/stage.py <attempt> config-2048.json --gpu h200 --mem 240G
ssh tufts-login 'squeue -u $USER'                                    # before
rsync -a experiments/hires-burgers/runs/<attempt>/ tufts-login:/cluster/tufts/paralab/tawal01/hires_b_20260920/<attempt>/
ssh tufts-login 'cd /cluster/tufts/paralab/tawal01/hires_b_20260920/<attempt> && sbatch run.sbatch'
ssh tufts-login 'squeue -u $USER'                                    # after: exactly one job for this dir
# when done:
$PY experiments/hires-burgers/cluster/collect.py <attempt>
$PY experiments/hires-burgers/audit_hires.py experiments/hires-burgers/runs/<attempt>/archive --out experiments/hires-burgers/checks/<attempt>-summary.json
ssh tufts-login 'rm -rf /cluster/tufts/paralab/tawal01/hires_b_20260920/<attempt>'   # only this attempt dir
```

## Next step

See the bottom of this file's job table.

| attempt | job id | mesh | GPU | state | summary |
|---|---|---|---|---|---|
