Paths relative to `experiments/imp-geometry/`.

- **CORRECT — blocker 2 fixed.** `run_cell.py:485–500` re-screens every eligible registered candidate, without stored-time filtering. `SCREEN_CACHE` (`:54`, `:493–500`) shares fresh screens across targets/statistics; variant parents use fresh rankings (`:503–514`). Matches `DESIGN.md:876–877`.
- **CORRECT — revised screening cost projection.** Independently recomputed `sum(wall − cohort_seconds)` over 2,809 unique eligible registered checkpoint rows: **2.369114 h**, confirming the ≈2.4 h proxy (`DESIGN.md:878`). The accounting includes single-query compilation/warm-up and screening (`ladder.py:86–109`).
- `resources.py:80–89` and `runs/cost/projection_A9.json:19,28` now allocate **4.0 h** for selection/timing and **5.1 h** for the resume. These remain projected allowances, not measured completion times.
- **CORRECT — requested wall time:** 5.1 × 1.5 = **7.65 h**, rounded to 7.7 h, within the **8 h** request (`cluster/RELEASE-PLAN.md:24–25`).
- **NEW blocking issues: none found.** Restore verifies checksums; stage failures stop progression (`cluster/bodies/resume_L_wide_S1.sh:4–7`). Shell syntax passes.

No files changed, jobs submitted, log writes, or large-array copies.

**JOB(resume): READY (pending authorization)**