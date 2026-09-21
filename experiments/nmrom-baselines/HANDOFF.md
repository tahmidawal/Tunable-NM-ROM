# HANDOFF — nmrom-baselines (kept current; read DESIGN.md first)

**State 2026-09-20 ~23:45 EDT.** Read DESIGN.md §6–§7 first (audit dispositions and post-gate03 amendments).

| job | dir | id | status |
|---|---|---|---|
| gate01 | — | 4051709 | cancelled by me (no epoch in 22 min; scatter-add). logs `runs/gate01/logs` |
| gate02 | — | 4055132 | cancelled by me <1 min (submitted with 6 account jobs running). no output |
| gate03 attempt 1 (swish / per-feature / f32) | remote dir **still present** (AE weights needed by gate04) | 4059370 | DONE, collected, audited: **FAIL** 1.67/1.45/1.73 %, median 1.67 % > 1.5 %; LS-LSPG 31.6 %; HR fails. `runs/gate03/` |
| gate04: HR-SNS on attempt-1 weights + attempt 2 (global scale) | removed | 4072224 | DONE, collected, audited (`runs/gate04/`): attempt 2 **FAIL** 709/11.2/10.9 %; HR-SNS fails. remote gate03+gate04 deleted |
| gate05: attempt 3 (sigmoid / per-feature / f32) — the LAST allowed attempt | `gate05` | 4077574 | RUNNING (~2.7 h) |
| fam128a: 128² sweep -> selection -> finals K=8/16/32 + ours q0/q256 + FOMs + timing, `--provisional` | `fam128a` | 4073272 | RUNNING (~7 h) |

Jobs used: 6 / 8. Two left: fam256, fam512 (if fam128a needs a rerun, 512² is dropped).
First family number (provisional, inadmissible): Kim base K=16 at 128² = 84 % worst evolved (autoencode 41 %), POD-16 41 %.

**If gate05 fails:** the Kim baseline is NOT validated; report family tables only under "unvalidated implementation — not admissible". Nearest miss = attempt 1, 1.67 % vs the 1.5 % bar (published < 1 %). Do not move the bar.

**When fam128a finishes:** collect, `audit_family.py runs/fam128a`, read `selection.selected_variant`; configs for 256²/512² = that variant, `M1` capped at 4096, one sweep variant per K with `timed: true`, no `finals`, `rom_qs [0,256]`, timing block, HR list on the variants. 512²: `--gpu h200 --mem 240G`.

**Collect a job:** `rsync -a tufts-login:/cluster/tufts/paralab/tawal01/nmrombase_20260920/<attempt>/{output,logs,OUTPUTS.sha256,run.sbatch,COMMIT.txt} runs/<attempt>/`,
`cd runs/<attempt> && sha256sum -c OUTPUTS.sha256`, run `audit_gate.py` / `audit_family.py`, then delete the remote attempt dir (that dir only).

**Next:** if `output/summary.json: gate.passed` -> stage J2: `python cluster/stage.py --gpu a100 --hours 8 --mem 120G sweep128 -- family.py --config experiments/nmrom-baselines/configs/sweep128.json --gate experiments/nmrom-baselines/runs/gate01/output/summary.json --out output`
(the gate summary must be committed first so staging carries it; add it to FILES in cluster/stage.py). If the gate fails: attempt 2 = `--scale global --attempt 2`, attempt 3 = `--act sigmoid --attempt 3` (DESIGN §6 finding 3). After three failures: report, use nothing.
