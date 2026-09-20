# hires-poisson — HANDOFF (kept current)

**Lane:** worktree `worktrees/2026-09-20-hires-poisson`, branch `exp/2026-09-20-hires-poisson`,
namespace `/cluster/tufts/paralab/tawal01/hires_p_20260920/`. Contract:
`reports/2026-09-20-speed-accuracy-campaign-protocol.md` (main). Design: `DESIGN.md`.

## State (update on every change)

- 2026-09-20: design + code committed (`85e323d3`); local N=64 smoke passed (parity, audit,
  audit negative control). Codex design audit running. No GPU job submitted yet.

## Jobs (budget 8 total, 2 running)

| # | attempt | mesh | job id | GPU | state |
|---|---|---|---|---|---|

## How to run

```bash
PY=/home/tahmid/Dev/.venv/bin/python
$PY experiments/hires-poisson/cluster/stage.py <attempt> --config config-<mesh>.json   # needs a clean commit
experiments/hires-poisson/cluster/submit.sh <attempt>      # squeue before/after, one dir per job
experiments/hires-poisson/cluster/collect.sh <attempt>     # checksum pull + remote delete
```
The job runs `hp_solve.py`, then the independent `hp_audit_np.py` ON THE CLUSTER (fields are
tens of GB), which writes `audit.json` + strided subsamples and deletes the full fields.

## Next step

Apply Codex findings → stage/submit `hp2048` → from its memory/timing decide `hp4096` config.
