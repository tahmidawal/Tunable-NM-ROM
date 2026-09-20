# HANDOFF — nmrom-baselines (kept current; read DESIGN.md first)

**State 2026-09-20 ~19:50 local:** gate job submitted. Codex design audit done (22 findings, dispositions in DESIGN.md §6;
codex sandbox could not read files here, so files were inlined — sandbox NOT loosened). `family.py` smoke being re-run locally.

| job | attempt dir | id | status |
|---|---|---|---|
| J1 gate Kim 2D Burgers, attempt 1 (swish / per-feature / f32) | `nmrombase_20260920/gate01` | 4051709 | RUNNING (A100, 10 h limit; ~5 h expected) |

Jobs used: 1 / 8.

**Collect a job:** `rsync -a tufts-login:/cluster/tufts/paralab/tawal01/nmrombase_20260920/<attempt>/{output,logs,OUTPUTS.sha256,run.sbatch,COMMIT.txt} runs/<attempt>/`,
`cd runs/<attempt> && sha256sum -c OUTPUTS.sha256`, run `audit_gate.py` / `audit_family.py`, then delete the remote attempt dir (that dir only).

**Next:** if `output/summary.json: gate.passed` -> stage J2: `python cluster/stage.py --gpu a100 --hours 8 --mem 120G sweep128 -- family.py --config experiments/nmrom-baselines/configs/sweep128.json --gate experiments/nmrom-baselines/runs/gate01/output/summary.json --out output`
(the gate summary must be committed first so staging carries it; add it to FILES in cluster/stage.py). If the gate fails: attempt 2 = `--scale global --attempt 2`, attempt 3 = `--act sigmoid --attempt 3` (DESIGN §6 finding 3). After three failures: report, use nothing.
