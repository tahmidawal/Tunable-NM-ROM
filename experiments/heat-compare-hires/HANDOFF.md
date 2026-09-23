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
| pn1024 | 4203972 | panel 1024² | H200 | CANCELLED by me while PENDING (no H200); never ran |
| pn1024a | 4205399 | panel 1024² | A100-80G pax049 | COMPLETED 0:21, collected, remote kept; v1 order gate FAILED (noise) -> provisional/diagnostic, see DESIGN A2 |
| pn1024b | 4206383 | panel 1024² (A2) | A100-80G pax049 | COMPLETED, collected, audit passed, status final |
| pn2048 | 4203979 | panel 2048² | H200 | CANCELLED by me while PENDING (pre-A2 code); never ran |
| pn2048b | 4206387 | panel 2048² (A2) | H200 pax008 | FAILED: GPU OOM at qm32 set-up (A3); no number used |
| pn4096 | 4207404 | panel 4096² | H200 | CANCELLED by me while PENDING (pre-A3 code) |
| pn2048c | 4207537 | panel 2048² (A3) | H200 pax010 | FAILED: torch OOM at FNO behind the JAX pool (A4); partial results diagnostic only |
| pn4096b | 4207540 | panel 4096² (A3) | H200 pax011 | FAILED: 70 GiB QM32 bank vs fragmented pool (A5); partial diagnostic only |
| pn4096c | — | panel 4096² (A5, QM first) | H200 | submitted |
| pn2048d | 4211204 | panel 2048² (A4) | H200 pax010 | COMPLETED, audit passed, status final; remote removed (and tr2048) |
| pn4096 | — | panel 4096² (no operators) | H200 | staged |
