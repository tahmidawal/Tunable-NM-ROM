# ops-timing-panel — HANDOFF

**State (2026-09-22):** design written, harness copied and smoked, nothing submitted yet.

- Worktree `worktrees/2026-09-22-ops-timing-panel`, branch `exp/2026-09-22-ops-timing-panel`,
  forked from `exp/2026-09-17-no-second` @ `ea812685`.
- Namespace `/cluster/tufts/paralab/tawal01/opstime_20260922/`, one directory per job.
- Jobs used: 0 of 3. Running: 0 of 1.

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

## Next

1. Commit, stage `opt101`, `squeue` before/after, submit to a100-80G.
2. On completion: collect with checksums, run `audit_panel.py` over all 9 operator names,
   delete the remote directory, generate the report, append the lab log.
