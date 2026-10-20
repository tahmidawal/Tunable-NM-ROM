**Overall: NEEDS-FIX — one new blocker.**

The round 6 fix passes CPU boundary tests: both 3D accuracy Q1 and timing qualification reject incorrect pooled counts for FS and QN and accept exactly `cases × spec_steps(spec)`.

1. **2D timing qualification accepts missing per-step certificates.** [analyze2d.py:226](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze2d.py:226) uses the reported `s['steps']` and row counts without checking certificate-array dimensions. `gcert_rows` also accepts empty arrays through vacuous `np.all`.

   In-memory CPU fixtures with all six cases and complete timing inventories returned **`qualified=True`, `usable=True`** with **0, 1, 24 or 26 certificate steps**, while reporting 25. This accepts missing evidence under A1.1 and can admit uncertified costs into A3 and Shapley results.

   Require certificate inventories to match the registered timestep-derived count, and recompute non-certified totals from those arrays.

| File | Verdict |
|---|---|
| `dscore.py` | READY |
| `analyze2d.py` | **NEEDS-FIX** |
| `analyze3d.py` | READY |
| `score3.py` | READY — mapping diff checked |
| `make_configs_late.py` | READY |
| `make_round.py` | READY |
| `waitjobs.sh` | READY |
| `d3run.py` | READY — scoped adaptive storage change |

No production results read, files changed, jobs submitted, or scratch files created.