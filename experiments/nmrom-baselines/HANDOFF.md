# HANDOFF — nmrom-baselines (kept current; read DESIGN.md first)

**State 2026-09-20 (session 1):** code written, local smokes pass for the gate (`gate_kim2d.py`) incl. sub-network parity 6e-14.
`family.py` local smoke in progress. No GPU job submitted yet. Codex design audit running -> `checks/codex-design-audit-01.txt`.

| job | attempt dir | id | status |
|---|---|---|---|
| J1 gate Kim 2D Burgers | `nmrombase_20260920/gate01` | — | not submitted |

Jobs used: 0 / 8.

**Next:** apply audit fixes -> commit -> `python cluster/stage.py gate01 --gpu a100 --hours 8 -- gate_kim2d.py --out output`
-> rsync `cluster/stage/gate01/` to `tufts-login:/cluster/tufts/paralab/tawal01/nmrombase_20260920/gate01/` -> squeue check -> `sbatch run.sbatch` -> squeue check.
Then J2 = 128² tuning sweep (`configs/sweep128.json`, cohort `tune`), J3–J5 finals.
