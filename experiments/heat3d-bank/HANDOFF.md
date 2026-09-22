# HANDOFF — heat3d-bank lane (RESUMED 2026-09-21 ~15:40 EDT)

Branch `exp/2026-09-21-heat3d-bank` (fork of hires-heat `4fb12a6d`). Lane dir `experiments/heat3d-bank/`.
Cluster namespace `/cluster/tufts/paralab/tawal01/h3dbank_20260921/`. Budget: ≤ 8 jobs total, ≤ 1 running (≤ 2 if account < 5 running).
Design + pre-registered selection rule: `DESIGN.md`. Codex audit: `CODEX-DESIGN-AUDIT.txt`, disposition `CODEX-DESIGN-AUDIT-DISPOSITION.md`.

## State
- Divergence mechanism supported locally (Codex: "supported", not "verified") (`diagnostics/refit_smoke.py`, JSONs beside it): Adam moments zeroed at refit with the shared
  step count kept -> ×3 full-batch loss spike 50 steps after refit (0.0448 -> 0.1357) vs 0.0531 with count reset / 0.0412 keeping moments.
- Fix = code-free variable-projection trainer `train_vp.py` (local smokes pass; `configs/smoke_vp.json`, `configs/smoke_panel.json`).
- Pipeline copied from hires-heat @4fb12a6d: `core.py`, `train.py` (head only), `run.py` (+ errors-only/timed-prefix, saved-field prefix),
  `audit.py` (skips unsaved cases, counts them), `summarize.py` (timed prefix), `cluster/{submit,collect}.sh` (train -> N panels).
- Configs from `make_configs.py`; selection from `select.py` (validation summaries only) -> `configs/final01.json`, `selection.json`.

## Jobs
| job | id | status | what |
|---|---|---|---|
| valR256 | 4141159 | CANCELLED (head-validate compile stall, see runs/valR256-cancelled/CANCELLED.md). Bank had reached 0.122 % at 64^3/128^3 | train R=256 K16/K32 + validation panels |
| valR320 | 4142297 | CANCELLED while pending (same fix) |
| valR256b | 4143174 | DONE, collected, remote removed; R256 floor 0.120 % |
| valR320b | 4143180 | DONE, collected, remote removed; R320 floor 0.060 % | train R=320 K16/K32 + validation panels |
| final01 | - | not submitted | frozen selection, cohorts 920399 + 921099, 32^3/64^3/128^3 |

## Next step
Codex disposition -> commit -> submit valR256 + valR320 (H200, 240G) -> collect (`cluster/collect.sh <job> --remove`) ->
copy `runs/<job>/pull/out/trained_vp_R*` to `inputs/vp_R*` and commit -> `select.py runs/val*/val_vp_*` -> commit -> submit final01.

## Earlier local findings (pre-GPU, orientation only; see diagnostics/podfloor*.py)
POD floor (training-only POD, worst relative L2, t=0 worst): 2048 training draws, R=256 -> 0.14 % (920399) / 0.15 % (921777 val256);
R=320 -> 0.08 % / 0.07 %. R=128 cannot reach 1 % (1.71 %). Disclosure in DESIGN.md: 920399 was touched by this POD diagnostic only.

## 2026-09-21 ~20:30 — validation done, pre-registered selection applied (R320/K32/q288 via rule 5), addendum 1 speed jobs next
Next: submit speedR256 (4146429) + speedR320 (4146432) RUNNING (configs speed_*.json, no training: inputs/vp_R* committed), then `select_speed.py`, then final01
(two panels: A = configs/final01.json, B = addendum). Jobs used: 4 of 8.

## 2026-09-21 ~23:00 — addendum 1 done (R256/K16/q160 panel B), addendum 2 speed2 job 4149861 RUNNING (job 7 of 8)
speedR256 4146429 / speedR320 4146432: collected, all checks passed, remotes removed. After speed2: collect, `select_speed2.py
runs/speed2/speed2_vp_R256_K16`, commit, then submit final01 (LAST job, 8 of 8):
`cluster/submit.sh final01 h200 14:00:00 240G - - final01.json final01b.json`. Nothing else may be submitted.

## 2026-09-22 — speed2 (4149861) collected, all checks passed, remote removed; addendum-2 selection applied
Panel B arms (`configs/final01b.json`, `selection_addendum2.json`): R256/K16, q in {0,160}: `field_cn_tol1e-4_chol_s1` (13.2 ms val) and
`direct_tol1e-4_chol_mom_s1` (6.3 ms val); all variants 0.543 % at q160; CN dt0.05 variants had non-stationary exits -> ineligible.
Next: final01 = LAST job (8 of 8), panels A (final01.json) + B (final01b.json).

## final01 = job 4153878 RUNNING (H200 pax009, 16 h wall) — LAST job, 8 of 8. Sealed 921099 + repeated benchmark 920399 opened here.
On completion: `cluster/collect.sh final01` (then `--remove` if all checks pass), `make_report.py`-style generated table, lab log.

## 2026-09-22 ~04:00 — LANE CLOSED: final01 (4153878) collected, all checks passed, remote removed; namespace empty; 8 of 8 jobs used
Generated table: `runs/final01/FINAL-TABLE.md` / `final_table.json` (from `runs/final01/{final01,final01b}/summary.json`).
Sealed 921099 (64 draws), worst all-times same-grid error: panel A (pre-registered, R320/K32/q288) 0.1137 % at 32^3/64^3/128^3;
panel B (addendum, R256/K16/q160) 0.4733 %. Accuracy bar (<=1 %) PASS at every mesh. Speed bar (>=5x named CN-CG) NOT met:
best 4.86x (panel B direct, 128^3); vs the paper FOM rule 1.92x (B direct) / 0.94x (B cn) / 2.22x (A direct) / 0.28x (A cn) at 128^3.
Labelled controls beat every ROM arm: linear solve in the learned bank 0.150 % in 1.3-2.2 ms; DST 2.1 ms; coarse 64^3 CN-CG 0.46 % 4.9 ms.
Nothing running. Worktree not merged (ask the user).

## 2026-09-22 — REOPENED for 256³ (user-authorised, ≤2 extra jobs): final256 = job 4171513 (H200, 6 h), job 9
Frozen panel A, sealed 921099 (64 cases), config `configs/final256.json`, DESIGN addendum 3. On completion:
`cluster/collect.sh final256`, check COLLECT-STATUS, remove remote, extend `make_final_table.py` output for the 256³ rows.

## 2026-09-22 — 256³ NOT obtained: both authorised extra jobs failed on memory/kernel issues (no numbers, nothing retracted from final01)
- 4171513: 256³ decode as one f64 [6,16.6M] GEMM -> XLA autotuning failure. Fixed by blocked decode.
- 4175066: blocked decode sliced the 42 GB device bank -> slice copies duplicated it -> OOM 39.62 GiB in jit_query.
  Fixed by STORING the bank as row blocks (no slicing). Parity vs committed 64³ numbers 7.5e-15 (1 block and 7 forced blocks).
- Ready to run: `cluster/submit.sh final256c h200 06:00:00 240G - - final256.json` (frozen panel A, sealed 921099, 64 cases).
  Needs a new job authorisation (the 2 extra jobs are spent). Memory after the fix at 256³: 8 blocks x 5.3 GB = 42 GB, no duplicate.

## 2026-09-22 — 256³ DONE: final256c (job 4175680) collected, all checks passed, remote removed, namespace empty, nothing running
Frozen panel A at 256³, sealed 921099, 64 cases x 5 reps, one H200. `runs/final256c/final256/summary.json`, table `runs/final256c/FINAL-TABLE.md`.
Accurate q288: direct 0.1138 % all-times / 0.0258 % evolved in 26.42 ms = 12.49x named CN-CG (330.02 ms) and 6.74x the paper-rule FOM
(dt 0.025 rtol 1e-4, 0.0823 %, 178.07 ms) -> BOTH bars met at 256³; CN 0.1137 % in 77.71 ms = 4.25x named, 2.29x rule (2 non-stationary exits).
Fast q0: 2.0042 %, direct 26.99 ms (12.23x named). Linear-bank control 0.0696-0.0698 % in 11.6-21.7 ms (baseline, not the NM-ROM).
Query profile (q288 direct): encode 11.0 + init 1.9 + evolve 2.6 + decode 10.7 ms -> now bank-read (memory) bound, not solver bound.
