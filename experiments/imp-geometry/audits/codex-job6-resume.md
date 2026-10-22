Fix verdicts (paths relative to `experiments/imp-geometry/`):

1. **CORRECT** — `run_cell.py:380` filters the expansion frontier to registered, base-tolerance PCG rows. Checkpoint replay confirms 20 new SEM variants; no excluded grades or variant feedback.
2. **WRONG** — `run_cell.py:491` still truncates to 12 using stored cross-job/form screens; `run_cell.py:498` chooses the three variant-bearing rows using that same ordering. Fresh sorting at `run_cell.py:510` cannot recover discarded candidates or unexplored variants. Re-screen eligible candidates consistently before truncation and K=3 selection.
3. **CORRECT** — `DESIGN.md:870–871` explicitly states that coordinator authorization beyond A8’s 12 GPU-h cap is required. This verifies disclosure only, not authorization.

No additional blocking issues found for `cluster/bodies/resume_L_wide_S1.sh`. Shell syntax passes.

Read-only: no changes, submissions, log writes, or large-array copies.

**JOB(resume): NOT-READY (blocker 2: cross-job/form screens still control candidate and variant selection).**