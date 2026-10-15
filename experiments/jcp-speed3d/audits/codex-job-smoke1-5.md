**HOLD** at HEAD `0543aae5b196`. Isolation and stderr selection are resolved; one blocking false-PASS remains.

[smoke_accept.py:371](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/smoke_accept.py:371) searches the **entire log**, selecting the last line containing case-sensitive `Error` or `RESOURCE_EXHAUSTED`. This does not establish the terminating cause. CPU execution of the actual branch confirmed that an earlier recoverable allocation error followed by 45 progress lines and Slurm’s terminal **TIME LIMIT** message produces **PASS**, even with `--sacct-state TIMEOUT`.

Require terminal resource evidence within the last 40 lines, and reject timeout/cancellation or unrelated terminal failures; retain explicit `OUT_OF_MEMORY` acceptance.

Other checks passed:

- Both manifests verified: **44 / 38 entries**; **40 / 34 provenance files** match HEAD; commit markers match.
- Both staged shell scripts pass syntax checks; GPU preflight, f64/highest precision and separate paralab paths are configured.
- Pilot: separate H200 job, **240G**, 24 hours; result targets `output/smkmesh384/result.json`.
- Main smoke excludes the pilot.
- Live `squeue` returned empty. The pair occupies **2 of 3** permitted slots; recheck before submission.

No repository/git changes, GPU execution, or submissions.