Paths relative to `experiments/imp-geometry/`.

**Blocker 2: WRONG — not fixed.** `run_cell.py:503–510` implements `DESIGN.md:872–875`, but still excludes candidates using cross-job/form timings.
- Counterexample (illustrative): stored screens A=1, B=3.1; fresh screens A=1, B=0.5. Only A is re-screened; measured drift=1, so B—the actual fastest—is dropped.
- Drift measured only inside the window cannot bound excluded candidates; even widening to 6× leaves the same defect.
- `run_cell.py:513` correctly chooses variant parents using fresh timings, but only among surviving candidates.

**Re-screening cost:** finite call count, no established runtime bound. For N eligible rows, each selection makes at most N+3 screens (`run_cell.py:502–521`), each with four query executions (`finalize.py:57–65`). Selection repeats across six target/statistic combinations without a shared screen cache (`run_cell.py:571–574`).
**NEW blocker:** the expanded screening workload lacks a revised cost certificate: `runs/cost/projection_A9.json:19` retains an assumed 2.5 h timing stage, without bounding this workload or compilation cost.

No other new blocking issues found in `cluster/bodies/resume_L_wide_S1.sh`; shell syntax passes. No files changed, jobs submitted, log writes, or large-array copies.

**JOB(resume): NOT-READY (blocker 2: potentially fastest candidates excluded; new blocker: expanded re-screening cost uncertified).**