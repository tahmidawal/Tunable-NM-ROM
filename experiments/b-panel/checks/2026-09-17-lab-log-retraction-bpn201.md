## 2026-09-17
### b-panel — bpn201 (1024²) RETRACTED: a config-parsing crash after 20 min, no number produced; fixed, smoked, resubmitted as bpn202 (job 3787247)

Worktree `worktrees/2026-09-17-b-panel`, branch `exp/2026-09-17-b-panel` at `494c3f485954`, namespace
`/cluster/tufts/paralab/tawal01/b_panel_20260917/`. Session resumed after the coordinator restart.

**What ran.** `bpn201` (job 3783817, NVIDIA H200 on pax008, staged from `e330cca4`)
exited 1:0 after 00:20:23. Disk checked first (92% used, logs non-empty): not disk-full.
Peak host RSS 72 GB of 240: not memory. Cause, verbatim: `TypeError: string indices must be integers, not 'str'` at
`panel.py` — `priority_override` is a list of subject names in `config-1024.json`; the driver indexed it as dicts.
Neither smoke config carried the key. The job had completed the preflight (`jax_backend=gpu`), six 4096-interval references,
3328 snapshots and all six rule transfers (742 s) and died on the next line. No subject was built; **no timing or error number exists**.

**Retracted.** `bpn201` in full. Archived as `experiments/b-panel/artifacts/bpn201-retracted/` (FAILURE.json, verbatim logs,
scheduler record, partial result.json, six transferred rule files, remote hashes); remote directory deleted. It counts against the cap.

**Fix (DESIGN A6, commits `330dc05e`, `494c3f48`).** `priority_override` read as names, every name asserted to be a declared
subject, declaration moved to the top of `main()` so a bad config fails at once; the declared list is otherwise byte-identical.
`config-smoke128.json` now carries an override; the smoke asserts the build order honours it and that config-256/512/1024's names
all declare. Smoke: baseline 1.42e-14, all gates, 330 s
(`checks/smoke-panel-a6.json`). No arm, rule, seed, population, tolerance or metric changed.

**Seen before the crash, bearing on the A5.2 prediction (retracted-attempt values, enter no table).** A5.2 predicted the
q=128 and q=256 transferred rules (14 and 8 fit states) would come back uncertified. Observed:

| q | fit states | rho_max | rho_p95 | basis |
|---|---|---|---|---|
| 0 | 64 | 0.0180 | 0.0161 | primary |
| 16 | 64 | 0.0594 | 0.0562 | primary |
| 32 | 42 | 0.0313 | 0.0180 | primary |
| 64 | 25 | 0.1316 | 0.1254 | none |
| 128 | 14 | 1.0295 | 0.5218 | none |
| 256 | 8 | 0.4144 | 0.3980 | none |

The prediction held for its two rungs and q=64 (25 states) also missed the 0.116 bar. The fit-state count is not raised in
`bpn202` (science unchanged between attempt and resubmission; the larger refit is unsmoked) and is the first thing `bpn301` fixes.

**Submitted.** `bpn202` = job 3787247, H200, `--mem 240G`, same `config-1024.json` (attempt label only changed), one `bpn_` job
queued before and after, PENDING behind the per-user GPU cap. Jobs used: 3 of 8 (bpn101, bpn201 retracted, bpn202).

**Open.** `bpn202` landing → collect, audit, report. `bpn301` (256² full panel with both rule sets, raised fit states) held for the
coordinator's signal after b-eqtop's job 3783811 lands. `bpn401` (512²) after that. Codex unavailable until 2026-09-19 11:33; self-audit stands in.
