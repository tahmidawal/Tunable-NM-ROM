# HANDOFF — burgers-heldout (kept current; read this first after an interruption)

**Lane:** `experiments/burgers-heldout/` in `worktrees/2026-09-21-burgers-heldout` (branch
`exp/2026-09-21-burgers-heldout`, fork of hires-burgers @ 0ab60014). Namespace
`/cluster/tufts/paralab/tawal01/bheld_20260921/`. Contract: main's
`reports/2026-09-20-speed-accuracy-campaign-protocol.md`. Design: `DESIGN.md` (pre-registered, Codex-audited,
dispositions §9). Budget ≤ 2 running / ≤ 8 total GPU jobs; only scancel own ids.

## State (update at every milestone)

- Resumed 2026-09-21 ~15:40 EDT after the pause. Plan = DESIGN §5: bh1 build → bh2 select (1024², dev6+sel32)
  → bh3 (4096², dev6+hold64) ∥ bh4 (4096², fresh64).
- **bh1 = job 4139115 DONE**, collected (checksums), remote deleted. cpod512 selected (state-weighted POD); sel32 floor
  0.183 % vs incumbent 0.513 % (256²). Head oracle 4.34e-3 (incumbent 5.20e-3). Directions rank 512.
  Records `checks/bh1/`; model+directions in git-ignored `ckpt/bh1/`, hashes in `CKPT-MANIFEST.json`. Lab-log milestone 1 appended.
- **bh2 = job 4142941** (A100-80G, staged at 925a59a4, `config-1024-select.json`): 1024², dev6+sel32, selection.
  When done: `cluster/collect.py bh2` → `audit_bh.py runs/bh2/archive --out checks/bh2-summary.json` → delete remote →
  `make_eval_configs.py checks/bh2-summary.json` → DESIGN amendment naming the headline arm → commit → stage bh3/bh4
  (`--gpu h200 --mem 320G --hours 20`), submit both (≤2 running).
- Jobs used: 2 / 8.

## How to run / collect

```bash
cd worktrees/2026-09-21-burgers-heldout            # commit first: stage.py refuses uncommitted files
PY=/home/tahmid/Dev/.venv/bin/python
$PY experiments/burgers-heldout/cluster/stage.py <attempt> build|hires [config] --gpu ... --mem ...
ssh tufts-login 'squeue -u $USER'                  # before AND after
rsync -a experiments/burgers-heldout/runs/<attempt>/ tufts-login:/cluster/tufts/paralab/tawal01/bheld_20260921/<attempt>/
ssh tufts-login 'cd .../<attempt> && sha256sum -c MANIFEST.sha256 --quiet && sbatch --parsable run.sbatch'
```

## Next step

When bh1 finishes: collect (checksums), check gates (parity, Gram identity, orthogonality, truth ≤1e-8, selection
floor < 0.6 % and below incumbent's), commit compress.json/hfit json/directions json + CKPT-MANIFEST.json (model
and directions git-ignored under ckpt/), delete remote dir, then stage bh2 (`config-1024-select.json`).
