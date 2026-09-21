# HANDOFF — heat3d-bank lane (RESUMED 2026-09-21 ~15:40 EDT)

Branch `exp/2026-09-21-heat3d-bank` (fork of hires-heat `4fb12a6d`). Lane dir `experiments/heat3d-bank/`.
Cluster namespace `/cluster/tufts/paralab/tawal01/h3dbank_20260921/`. Budget: ≤ 8 jobs total, ≤ 1 running (≤ 2 if account < 5 running).
Design + pre-registered selection rule: `DESIGN.md`. Codex audit: `CODEX-DESIGN-AUDIT.txt` (+ disposition below once read).

## State
- Divergence root cause verified locally (`diagnostics/refit_smoke.py`, JSONs beside it): Adam moments zeroed at refit with the shared
  step count kept -> ×3 full-batch loss spike 50 steps after refit (0.0448 -> 0.1357) vs 0.0531 with count reset / 0.0412 keeping moments.
- Fix = code-free variable-projection trainer `train_vp.py` (local smokes pass; `configs/smoke_vp.json`, `configs/smoke_panel.json`).
- Pipeline copied from hires-heat @4fb12a6d: `core.py`, `train.py` (head only), `run.py` (+ errors-only/timed-prefix, saved-field prefix),
  `audit.py` (skips unsaved cases, counts them), `summarize.py` (timed prefix), `cluster/{submit,collect}.sh` (train -> N panels).
- Configs from `make_configs.py`; selection from `select.py` (validation summaries only) -> `configs/final01.json`, `selection.json`.

## Jobs
| job | id | status | what |
|---|---|---|---|
| valR256 | - | not submitted | train R=256 K16/K32 + validation panels |
| valR320 | - | not submitted | train R=320 K16/K32 + validation panels |
| final01 | - | not submitted | frozen selection, cohorts 920399 + 921099, 32^3/64^3/128^3 |

## Next step
Codex disposition -> commit -> submit valR256 + valR320 (H200, 240G) -> collect (`cluster/collect.sh <job> --remove`) ->
copy `runs/<job>/pull/out/trained_vp_R*` to `inputs/vp_R*` and commit -> `select.py runs/val*/val_vp_*` -> commit -> submit final01.

## Earlier local findings (pre-GPU, orientation only; see diagnostics/podfloor*.py)
POD floor (training-only POD, worst relative L2, t=0 worst): 2048 training draws, R=256 -> 0.14 % (920399) / 0.15 % (921777 val256);
R=320 -> 0.08 % / 0.07 %. R=128 cannot reach 1 % (1.71 %). Disclosure in DESIGN.md: 920399 was touched by this POD diagnostic only.
