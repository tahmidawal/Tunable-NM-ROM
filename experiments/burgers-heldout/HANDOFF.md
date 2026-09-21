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
- **bh2 = job 4142941 DONE** (collected, audited `checks/bh2-summary.json`, remote deleted): no certified arm ≤ 1 % on
  dev6+sel32 (q256/M1088 1.73 %, q384/M1600 0.93 % uncertified ρ 0.128); stop rule → no H200 yet. DESIGN A2.
- **bh2b = 4144023 DONE** (collected, audited `checks/bh2b-summary.json`, remote deleted): lat128 arms uncertified
  (ρ outliers) but accurate (q512/M2112 0.41 %, q448 0.65 %) and slow (300–635 ms at 1024²). Pre-registered pick
  still q256/M1088 lat64 1.73 % → bar not met under the rules. DESIGN A3: labelled exploratory 4096² jobs.
- **bh2c = 4144025 DONE** (incumbent attribution, `checks/bh2c-summary.json`): incumbent sel32 q256/M1088 1.49 %,
  q512 1.35 %, floor 0.594 % (new: 1.73 / 0.41 / 0.259 %). Incumbent lat64 q256 uncertified at 1024² (ρ 0.119).
- **bh3 = 4146491** (H200, dev6+hold64, new model, exploratory per A3) RUNNING; **bh4 = 4147539** (fresh64) RUNNING.
- **bh5 STAGED (runs/bh5, commit 7912b033), NOT submitted**: coordinator task A4 (incumbent re-timed with lat64
  j=1 at 4096², dev6+hold64, eqcert certificate). Submit as soon as bh3 or bh4 leaves the queue (≤2 running):
  rsync runs/bh5 → sbatch. Collect with cluster/collect.py bh5; audit with eqcert/audit_eqcert.py AND audit_bh.py.
- Jobs used: 6 submitted / 8 (bh5 = 8th). No budget left for retries after bh5.

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
