**S1: SUBMIT. J2: SUBMIT** — for this narrow audit.

Paths below are relative to `experiments/imp-multimesh/`.

| Fix | Status | Evidence |
|---|---|---|
| **(1) Preserve uncertain reservations** | **FIXED** | `cluster/submit.sh:59–68` checks both lookup statuses and requires exactly one candidate before binding. Empty lookup still exits at `:63` under `set -e`, retaining `RESERVED` rather than recording `AMBIGUOUS`; budget remains protected. |
| **(2) Early-exit evidence and checksums** | **FIXED — both jobs** | Templates: `cluster/stage.py:115–122`, `cluster/stage_probe.py:108–111`. Staged scripts: `runs/s1/run.sbatch:24–32`, `runs/j2/run.sbatch:17–21`. Both emit valid failure JSON and generate `OUTPUTS.sha256`, then exit nonzero. S1 also copies failure evidence to `ACCEPT.json` with `all_passed: false`. |
| **(3) Restaging and hash refresh** | **FIXED** | All **146 S1** and **63 J2** manifest entries verify. `checks/resources-s1.json:7` matches the staged smoke config (`runs/s1/MANIFEST.sha256:18`). `checks/resources-j2.json:7` matches the staged batch script (`runs/j2/MANIFEST.sha256:63`). S1’s batch script also verifies at `runs/s1/MANIFEST.sha256:146`. |

Executed the generated JSON-formatting expressions without file writes: both manifest and GPU-preflight cases parse correctly. Shell syntax checks pass. Early finalization preserves nonzero status even if checksum generation fails.

No new GPU-hour-waste or cross-lane namespace defect found in these fixes.

Read-only: no files changed, jobs submitted, large arrays loaded, or writes to main or `LAB-LOG.md`.