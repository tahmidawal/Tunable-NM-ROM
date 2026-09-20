# HANDOFF — hires-heat lane (kept current; read this first after any interruption)

Branch `exp/2026-09-20-hires-heat`, lane dir `experiments/hires-heat/`, cluster namespace `/cluster/tufts/paralab/tawal01/hires_h_20260920/<job>`. Design: `DESIGN.md`. Codex design audit: `CODEX-DESIGN-AUDIT.txt` (15 findings; dispositions in `CODEX-DESIGN-AUDIT-DISPOSITION.md`). Speed loop: `SPEED-LOG.md`.

## Mechanics
- Submit (clean committed tree only): `cluster/submit.sh <job> <config> <gpu> <HH:MM:SS> <mem> [trainname]`.
- Collect + checksum + NumPy/SciPy audit + generated summary: `cluster/collect.sh <job>`; add `--remove` to delete the remote dir after the audit passes.
- Generated per-job outputs: `runs/<job>/summary.json`, `runs/<job>/SUMMARY.md`, `runs/<job>/audit.json`.

## Job ledger (budget: 8 total, 2 running)
(see bottom of file; appended at each submit/collect)
