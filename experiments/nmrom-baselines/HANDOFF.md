# HANDOFF — nmrom-baselines (kept current; read DESIGN.md first)

**State 2026-09-21 ~03:10 EDT** (session 2, Opus agent, took over from the credit-limited session at 5da47517).
Read DESIGN.md §6–§7 first (audit dispositions and post-gate03 amendments).

| job | dir | id | status |
|---|---|---|---|
| gate01 | — | 4051709 | cancelled by me (no epoch in 22 min; scatter-add). logs `runs/gate01/logs` |
| gate02 | — | 4055132 | cancelled by me <1 min (submitted with 6 account jobs running). no output |
| gate03 attempt 1 (swish / per-feature / f32) | removed | 4059370 | DONE, audited: **FAIL** 1.67/1.45/1.73 %, median 1.67 % > 1.5 %; LS-LSPG 31.6 %; HR fails. `runs/gate03/` |
| gate04: HR-SNS on attempt-1 weights + attempt 2 (global scale) | removed | 4072224 | DONE, audited: attempt 2 **FAIL** 709/11.2/10.9 %; HR-SNS fails. `runs/gate04/` |
| gate05: attempt 3 (sigmoid / per-feature / f32) — the LAST allowed attempt | removed | 4077574 | DONE, audited: **PASS** 1.45/1.83/1.44 %, median 1.45 %; LS 31.6 %; HR diverges (HR gate FAIL). `runs/gate05/` |
| fam128a: 128² **swish** sweep (11 variants) -> selection -> finals K=8/16/32 + ours q0/q256 + FOMs + timing, `--provisional` | `fam128a` | 4073272 | RUNNING (started 23:08, ~7 h). Kim rows INADMISSIBLE (swish failed the gate); POD/ours/FOM rows valid |
| fam256: sigmoid mini-sweep K16 -> select on tune -> finals K8/16/32 (+HR expl.) + data-matched K16 fit576 + published-M1 precheck + POD + ours + FOMs + timing, `--gate gate05` | `fam256` | 4095408 | RUNNING (submitted 03:05, est. 7–8 h) |

Jobs used: 7 / 8. One left: **fam512** (H200, 240G) — config from fam256's SELECTED line (`python configs/make_mesh_configs.py 512 <copy of live fam256 summary.json>`), submit when fam128a frees the lane slot. No job budget for Lee & Carlberg (B): cut, as DESIGN §5 foresaw.

## Session 2 changes (committed)

- `8fafab83` untracked 1.8 GB of gate04 weights/fields that `ac769088` had committed (nested paths escaped the ignore
  pattern). Files stay on disk; history NOT rewritten because gate05 records `ac769088` as its source commit.
- `7d18d670` `family.py`: (1) variant key `fit_traj` (> 112) adds further TRAIN-split draws, indices 128.. (the shared
  protocol's `future_train_prefixes`; tuning 112–127 and validation untouched) — the **data-matched arm**: the frozen
  bank was trained on 576 trajectories (head on 4608), the Kim AE on 112, a confound a reviewer would name;
  (2) device-memory precheck: an arm whose dense-encoder weights+grad+Adam exceed the device is recorded in `dropped`
  (`exceeds_device_memory_precheck`), never attempted. 16² local smoke passed. kimae.py / lspg.py unchanged.
- `95a8863a` `configs/make_mesh_configs.py`: writes fam256.json / fam512.json from the 128² selection (rules in its docstring).

## Next steps

1. fam128a done -> collect, `audit_family.py runs/fam128a`, commit small files, delete remote dir.
2. When fam256's log shows `SELECTED`: copy its live `output/summary.json` to the scratchpad, `python configs/make_mesh_configs.py 512 <copy>`,
   commit fam512.json, `python cluster/stage.py --gpu h200 --hours 14 --mem 240G fam512 -- family.py --gate experiments/nmrom-baselines/runs/gate05/output/summary.json --config experiments/nmrom-baselines/configs/fam512.json --out output`
   (OPTIONS BEFORE the attempt name — argparse REMAINDER swallows anything after it), then `bash cluster/submit.sh fam512`.
3. fam256 / fam512 done -> collect + audit, `python reports/gen_report.py`, lab-log entry, final message.

**Collect a job:** `rsync -a tufts-login:/cluster/tufts/paralab/tawal01/nmrombase_20260920/<attempt>/{output,logs,OUTPUTS.sha256,run.sbatch,COMMIT.txt} runs/<attempt>/`,
`cd runs/<attempt> && sha256sum -c OUTPUTS.sha256`, run `audit_gate.py` / `audit_family.py`, then delete the remote attempt dir (that dir only).
Arrays (`*.npz *.pkl *.npy`) are gitignored at any depth; commit only summary.json, audit.json, logs, sha files.
