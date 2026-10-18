**Overall: NEEDS-FIX — one new blocker.**

The round 8 rho fix passes: incomplete shapes, all-NaN evidence, infinities and values above `0.116` are rejected; a complete `(6,6)` array containing a trivial NaN remains accepted.

1. **3D Q1 accepts incomplete anchor trajectories through broadcasting.**  
   [analyze3d.py:155](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze3d.py:155) computes distances before validating trajectory shapes. The population check validates IDs and summary-array lengths, but omits `W`, `W_tight` and `W_tighter` dimensions.

   In-memory CPU fixtures returned **every Q1 predicate true** with:
   - `W_tighter.shape == (8,1,r)`: five output states missing per check case.
   - `W_tighter.shape == (1,6,r)`: seven check-case trajectories missing.

   NumPy broadcasts the surviving evidence across missing states/cases. The same omission affects ladder Q1 at line 146. Require exact `(n,6,r)` production/tight shapes and `(n_check,6,r)` tighter shapes before distance calculations; incomplete evidence must remain **missing** under A1.5.

| File | Verdict |
|---|---|
| `dscore.py` | READY |
| `analyze2d.py` | READY |
| `analyze3d.py` | **NEEDS-FIX** |
| `score3.py` | READY — mapping diff checked |
| `make_configs_late.py` | READY |
| `make_round.py` | READY |
| `waitjobs.sh` | READY |
| `d3run.py` | READY — scoped adaptive storage change |

No production results read, files changed, jobs submitted, or scratch files created.