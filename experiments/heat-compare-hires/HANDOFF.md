# HANDOFF — heat-compare-hires (read first after an interruption)

Branch `exp/2026-09-23-heat-compare-hires` (LOCAL ONLY — never push: the base branch carries
multi-GB archive blobs and `git push` drove pack-objects to 73 GB RSS; coordinator instruction).
Lane dir `experiments/heat-compare-hires/`. Design + amendments: `DESIGN.md`. Namespace
`/cluster/tufts/paralab/tawal01/hcmp_20260923/<job>/`. At most 2 lane jobs at once
(`cluster/submit.sh` refuses a third). Jobs named `opstune_*` / `bcmp_*` belong to other lanes.

## Mechanics
- Submit (clean committed tree): `cluster/submit.sh <job> <gpu> <HH:MM:SS> <mem> '<cmd>'`
  (cmd runs in `code/heat-compare-hires`, `$PY` and `$OUT` provided).
- Panel command: `"$PY" panel.py --config configs/pn<N>.json --out "$OUT"`.
- Collect: `cluster/collect.sh <job>` (checksums, `jax_backend=gpu`, audit, summary);
  `--remove` deletes the remote dir only after the audit passed. Training dirs `tr1024a`,
  `tr2048` must stay until their panel has run (the panels read the checkpoints there).
- Report: `reports/make_report.py` → `reports/heat-compare-hires.md`, `reports/summary.json`.

## Ledger
| job | slurm | what | GPU | state |
|---|---|---|---|---|
| tr1024 | 4196056 | operator training 1024² | H200 | CANCELLED by me while PENDING; never ran |
| tr1024a | 4196355 | operator training 1024² (4 × 3000 s) | A100-PCIE-40GB pax051 | COMPLETED 3:20:53, collected; remote kept for pn1024 |
| tr2048 | 4196062 | operator training 2048² (4 × 3000 s) | H200 pax008 | COMPLETED 3:21:12, collected; remote kept for pn2048 |
| pn1024 | 4203972 | panel 1024² | H200 | submitted |
| pn2048 | 4203979 | panel 2048² | H200 | submitted |
| pn4096 | — | panel 4096² (no operators) | H200 | staged |
