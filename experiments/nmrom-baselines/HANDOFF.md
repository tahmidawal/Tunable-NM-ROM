# HANDOFF — nmrom-baselines (kept current; read DESIGN.md first)

**State 2026-09-21 ~05:00 EDT** (session 2, Opus agent, took over from the credit-limited session at 5da47517).
Read DESIGN.md §6–§7 first (audit dispositions and post-gate03 amendments).

| job | dir | id | status |
|---|---|---|---|
| gate01 | — | 4051709 | cancelled by me (no epoch in 22 min; scatter-add). logs `runs/gate01/logs` |
| gate02 | — | 4055132 | cancelled by me <1 min (submitted with 6 account jobs running). no output |
| gate03 attempt 1 (swish / per-feature / f32) | removed | 4059370 | DONE, audited: **FAIL** 1.67/1.45/1.73 %, median 1.67 % > 1.5 %; LS-LSPG 31.6 %; HR fails. `runs/gate03/` |
| gate04: HR-SNS on attempt-1 weights + attempt 2 (global scale) | removed | 4072224 | DONE, audited: attempt 2 **FAIL** 709/11.2/10.9 %; HR-SNS fails. `runs/gate04/` |
| gate05: attempt 3 (sigmoid / per-feature / f32) — the LAST allowed attempt | removed | 4077574 | DONE, audited: **PASS** 1.45/1.83/1.44 %, median 1.45 %; LS 31.6 %; HR diverges (HR gate FAIL). `runs/gate05/` |
| fam128a: 128² **swish** sweep (11 variants) -> selection -> finals K=8/16/32 + ours q0/q256 + FOMs + timing, `--provisional` | removed | 4073272 | DONE, collected, audited (`runs/fam128a/`). Kim rows INADMISSIBLE (swish); POD/ours/FOM valid; ours q0 3.41 % / q256 0.51 % |
| fam256: sigmoid mini-sweep K16 -> select on tune -> finals K8/16/32 (+HR expl.) + data-matched K16 fit576 + published-M1 precheck + POD + ours + FOMs + timing, `--gate gate05` | `fam256` | 4095408 | RUNNING (submitted 03:05, est. 7–8 h). SELECTED `sig_K16_zero_feature` (tune 70.6 %, validation 145 %); published M1 dropped by precheck (136 > 77 GB) |
| fam512: fam256-selected sigmoid variant K8/16/32 (+HR expl.) + published-M1 precheck + POD + ours + FOMs + timing, H200 | `fam512` | 4107272 | RUNNING (submitted 04:55, est. 8 h) |

Jobs used: **8 / 8 (budget spent)**. Lee & Carlberg (B): cut.

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

1. fam256 / fam512 done -> collect (below) + `audit_family.py`, `python reports/gen_report.py`, commit small files, delete that remote dir.
2. Lab-log closing entry (numbers from the generated report), final message. No job budget remains.

**Collect a job:** `rsync -a tufts-login:/cluster/tufts/paralab/tawal01/nmrombase_20260920/<attempt>/{output,logs,OUTPUTS.sha256,run.sbatch,COMMIT.txt} runs/<attempt>/`,
`cd runs/<attempt> && sha256sum -c OUTPUTS.sha256`, run `audit_gate.py` / `audit_family.py`, then delete the remote attempt dir (that dir only).
Arrays (`*.npz *.pkl *.npy`) are gitignored at any depth; commit only summary.json, audit.json, logs, sha files.
