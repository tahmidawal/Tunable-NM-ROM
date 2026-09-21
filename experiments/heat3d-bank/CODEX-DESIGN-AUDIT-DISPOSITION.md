# Codex design audit — disposition (2026-09-21)

Audit: `CODEX-DESIGN-AUDIT.txt` (codex exec, read-only, gpt-5.6-sol; attempt 1 hung on stdin, `CODEX-DESIGN-AUDIT-attempt1-stdin.txt`).
All fixes re-smoked locally (train_vp, run.py panel with two cohorts, audit exact discrepancy 1.1e-15, summarize).

| # | severity | finding | disposition |
|---|---|---|---|
| 1 | blocker | select.py opens sealed cohorts even if no bank passes the gate | FIXED: no gated bank -> `selection.json` STOP, no final config; fallback restricted to gated banks |
| 2 | blocker | unquoted JOB in rm -rf | FIXED: job-name regex in submit/collect, quoted paths, prefix `case` guard before `rm -rf` |
| 3 | major | varpro via normal equations + ridge is not an exact projector | FIXED: thin Householder QR projector `Y - Q(QᵀY)` in training and validation; condition assert tightened to 1e8 |
| 4 | major | summarize dropped the per-cohort `stats` slice (my edit commented it out) | FIXED (real bug; would mix failure counts across cohorts); refinement max also sliced per cohort |
| 5 | major | audit covers only a prefix of validation cases | FIXED for selection: every validation case saves the selection-driving `_field_cn` arms (audited); all arms saved for the first 32. Final job saves every arm of every case (sub-grid exact). FOM full-grid errors are exactly audited only where full fields are saved (case 0 at ≤64³, random-node samples on cases 0–1): labelled as sampled |
| 6 | major | collect can accept/delete a partial job | FIXED: refuses while the job id is in squeue; requires `run_exit=0` (checksummed `out/EXIT_STATUS.txt`), every expected panel present, `complete=true`, audit passed |
| 7 | major | reporting scripts not staged/hashed | FIXED: audit.py/summarize.py staged + checksummed in the job; collect runs the staged copies; `summary.json` records `reporting_sha256` |
| 8 | major | remote dir creation not atomic | FIXED: `mkdir` without `-p` on the job dir (fails if it exists) |
| 9 | major | 920399 not statistically sealed | ACCEPTED: sealed verdict cohort = 921099 (64 draws, never evaluated by anyone; its first 16 were reserved by hires-heat but never opened); 920399 reported as the repeated paper benchmark cohort (comparability with the 1.93 % row) |
| 10 | major | "root cause verified" too strong; resetcnt also restarts the lr schedule | ACCEPTED wording: "supported mechanism". The `keepmom` arm (no schedule confound) shows no spike, isolating the zeroed-moment/stale-count reset; the at-scale interaction with bank ill-conditioning is not separately tested (the old trainer is retired, not repaired) |
| 11 | minor | timing spread | FIXED: p10/p90 per row in summary.json |
