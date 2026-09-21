# HANDOFF — nmrom-baselines (kept current; read DESIGN.md first)

**State 2026-09-20 ~23:45 EDT.** Read DESIGN.md §6–§7 first (audit dispositions and post-gate03 amendments).

| job | dir | id | status |
|---|---|---|---|
| gate01 | — | 4051709 | cancelled by me (no epoch in 22 min; scatter-add). logs `runs/gate01/logs` |
| gate02 | — | 4055132 | cancelled by me <1 min (submitted with 6 account jobs running). no output |
| gate03 attempt 1 (swish / per-feature / f32) | remote dir **still present** (AE weights needed by gate04) | 4059370 | DONE, collected, audited: **FAIL** 1.67/1.45/1.73 %, median 1.67 % > 1.5 %; LS-LSPG 31.6 %; HR fails. `runs/gate03/` |
| gate04 = HR-SNS diagnostic on attempt-1 weights (`output/hr_attempt1_sns`) then attempt 2 (global scale, `output/attempt2`) | `gate04` | 4072224 | RUNNING (~3 h) |
| fam128a: 128² sweep -> selection on tuning subset -> finals K=8/16/32 + ours q0/q256 + FOMs + timing, `--provisional` | `fam128a` | 4073272 | RUNNING (~7 h, 12 h limit) |

Jobs used: 5 / 8. Remaining plan: [gate05 = attempt 3 sigmoid, only if attempt 2 fails], fam256, fam512 (H200? 512²: n=261k; encoder capped).

**When gate04 finishes:** collect (see below; two summaries: `output/hr_attempt1_sns/summary.json`, `output/attempt2/summary.json`; run `audit_gate.py` needs `runs/gate04/output/attempt2` layout -> call `audit_gate.py runs/gate04/attempt2view` after `ln -s`, or pass a dir that has `output/` + `logs/`), then delete BOTH remote dirs gate03 and gate04.
If attempt 2 passes -> family runs become admissible if kimae.py/lspg.py hashes match (they were not changed since gate03's commit? CHECK: `summary.json: source_sha256`).
If it fails -> stage gate05: `python cluster/stage.py --gpu a100 --hours 8 --mem 64G gate05 -- gate_kim2d.py --out output --attempt 3 --act sigmoid`.

**When fam128a finishes:** collect, `audit_family.py runs/fam128a`, read `selection.selected_variant`; write configs/fam256.json and fam512.json = that variant with `M1` capped (4096), `variants: []`-style direct finals (put the selected variant as the only sweep variant per K, `finals` omitted, `timed: true`), `rom_qs [0,256]`, timing block. 512²: use `--gpu h200 --mem 240G`.

**Collect a job:** `rsync -a tufts-login:/cluster/tufts/paralab/tawal01/nmrombase_20260920/<attempt>/{output,logs,OUTPUTS.sha256,run.sbatch,COMMIT.txt} runs/<attempt>/`,
`cd runs/<attempt> && sha256sum -c OUTPUTS.sha256`, run `audit_gate.py` / `audit_family.py`, then delete the remote attempt dir (that dir only).

**Next:** if `output/summary.json: gate.passed` -> stage J2: `python cluster/stage.py --gpu a100 --hours 8 --mem 120G sweep128 -- family.py --config experiments/nmrom-baselines/configs/sweep128.json --gate experiments/nmrom-baselines/runs/gate01/output/summary.json --out output`
(the gate summary must be committed first so staging carries it; add it to FILES in cluster/stage.py). If the gate fails: attempt 2 = `--scale global --attempt 2`, attempt 3 = `--act sigmoid --attempt 3` (DESIGN §6 finding 3). After three failures: report, use nothing.
