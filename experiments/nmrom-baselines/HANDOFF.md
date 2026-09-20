# HANDOFF — nmrom-baselines (kept current; read DESIGN.md first)

**State 2026-09-20 ~21:00 local:** `cluster/submit.sh gate03` is waiting in the background for account capacity (< 6 jobs) and then submits.
If this session died, re-run `cluster/submit.sh gate03` (it refuses a duplicate). Codex design audit done (DESIGN.md §6).

| job | attempt dir | id | status |
|---|---|---|---|
| gate01 | `gate01` | 4051709 | CANCELLED by me after 22 min: no training epoch finished (XLA scatter-add pathological). Logs in `runs/gate01/logs`. Only the FOM + LS-LSPG control ran: LS-LSPG 31.6 % (published LS ≈ 34–38 % with HR) |
| gate02 | `gate02` | 4055132 | CANCELLED by me < 1 min after submit: I submitted while the account already had 6 running (protocol breach, corrected). No output |
| gate03 (= gate01 recipe, slice-based layer) | `gate03` | pending submit | guarded submit waiting |

Jobs used: 2 / 8 (both cancelled ones are counted); gate03 will be the 3rd.

**Collect a job:** `rsync -a tufts-login:/cluster/tufts/paralab/tawal01/nmrombase_20260920/<attempt>/{output,logs,OUTPUTS.sha256,run.sbatch,COMMIT.txt} runs/<attempt>/`,
`cd runs/<attempt> && sha256sum -c OUTPUTS.sha256`, run `audit_gate.py` / `audit_family.py`, then delete the remote attempt dir (that dir only).

**Next:** if `output/summary.json: gate.passed` -> stage J2: `python cluster/stage.py --gpu a100 --hours 8 --mem 120G sweep128 -- family.py --config experiments/nmrom-baselines/configs/sweep128.json --gate experiments/nmrom-baselines/runs/gate01/output/summary.json --out output`
(the gate summary must be committed first so staging carries it; add it to FILES in cluster/stage.py). If the gate fails: attempt 2 = `--scale global --attempt 2`, attempt 3 = `--act sigmoid --attempt 3` (DESIGN §6 finding 3). After three failures: report, use nothing.
