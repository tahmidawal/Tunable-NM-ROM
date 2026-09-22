# ops-timing-panel — HANDOFF

**State (2026-09-22):** `opt101` = job **4179247** RUNNING on pax049 (a100-80G, 6 h wall).
Staged from source commit `e664fe45`. Expected ~45 min.

- Worktree `worktrees/2026-09-22-ops-timing-panel`, branch `exp/2026-09-22-ops-timing-panel`,
  forked from `exp/2026-09-17-no-second` @ `ea812685`.
- Namespace `/cluster/tufts/paralab/tawal01/opstime_20260922/`, one directory per job.
- Jobs used: 1 of 3. Running: 1 of 1. squeue checked before (empty) and after (exactly one).

## What exists

- `DESIGN.md` — pre-registered question, arms, timing scope, FOM rule, pass/fail, stop rules.
- `COPIED-FROM.json` — every copied file with source worktree, path, commit, SHA256.
- `operators.json` — the 9 operator checkpoints with the SHA256 their producing job recorded.
- `config-256-ops.json` — b-panel `config-256.json` trimmed to 21 subjects (endpoints only).
- `cluster/stage.py` — b-panel's stager, lane paths + operator phase over all 9 checkpoints.
- `audit_panel.py` — b-panel's audit, `--fno-name` now a list.
- `smoke_operators.py` — local check: all 9 checkpoints load, hash-verified, right contract.

## Done

- Local smoke `smoke_operators.py`: **PASS**, all 9 checkpoints, SHA256 verified, shape
  (6, 257, 257), $t_0$ bitwise, boundary masked, f64.
- `declare_subjects` on the trimmed config: 21 subjects, exactly the intended set.

- Codex design audit (`checks/codex-design-audit.md`): six findings, all accepted, disposition
  table in `DESIGN.md` §11.

## Next

1. Watch 4179247. Preflight must print `jax_backend=gpu`; the training-index assert exits 43.
2. On completion: collect with checksums; run
   `audit_panel.py output/result.json --fields output --out checks/opt101-audit.json
   --fno-name fno-large unet-small unet-medium unet-large unet-refine tsol-small tsol-medium
   tsol-large tsol-refine`; **refuse the job if the audit's `failed` list is non-empty**;
   delete the remote directory; generate the report; append the lab log.
