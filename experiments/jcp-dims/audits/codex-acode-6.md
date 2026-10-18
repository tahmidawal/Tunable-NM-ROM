**Overall: NEEDS-FIX — one new blocker.**

Both round-5 findings are resolved:
- Adaptive accuracy/timing certificates reject step counts 0, 1, 24 and 26, and accept the registered 25 at `dt=0.01`.
- Incomplete round populations return `missing (incomplete population)` even when an available run is invalid.

Ranked new finding:

1. **Fixed-sweep/QN certificate inventories remain unchecked.** At [analyze3d.py:148](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze3d.py:148), accuracy Q1 requires only a positive reported step count. At [analyze3d.py:545](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze3d.py:545), timing qualification does not even require that.

   CPU fixtures with otherwise complete case/anchor inventories reproduced:
   - Accuracy: **1 step accepted; 1,600 required** (`64 × 25`).
   - Timing: **0 steps accepted; 200 required** (`8 × 25`).

   Missing certification evidence can therefore pass Q1 and timing qualification, affecting the qualified front and H2q. Validate each case’s certificate inventory against the registered spec’s step count, and verify pooled counts against those records.

| File | Verdict |
|---|---|
| `dscore.py` | READY |
| `analyze2d.py` | READY |
| `analyze3d.py` | **NEEDS-FIX** |
| `score3.py` | READY — setting-mapping diff checked |
| `make_configs_late.py` | READY |
| `make_round.py` | READY |
| `waitjobs.sh` | READY |
| `d3run.py` | READY — scoped `Wstore`/`Cstore` change |

No other new blocker established. Read-only audit: no files changed, production result archives opened, jobs submitted, or scratch files created.