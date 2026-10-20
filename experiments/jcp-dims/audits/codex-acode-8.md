**Overall: NEEDS-FIX — one new blocker.**

1. **2D timing Q2 accepts incomplete ρ evidence.** [analyze2d.py:255](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze2d.py:255) checks `nanmax(rho)` without validating its inventory. A1.1 requires six output states for every dev6 case: shape `(6, 6)`.

   CPU fixtures with complete certificates and timing inventories returned **`qualified=True`, `usable=True`** with ρ shapes **`(1,)`, `(1,6)`, `(5,6)`, and `(6,5)`**. Missing quadrature evidence can therefore admit costs into downstream results.

   Require the complete `(6,6)` inventory before evaluating the maximum, preserving the registered trivial-ρ convention.

The **round 7 certificate fix passes**: non-square LSPG, square LSPG and GAL reject 0, 1, 24 and 26 steps and accept the registered 25; recomputed non-certified totals override reported zero counts; malformed GAL companion arrays and empty G-cert rows are rejected.

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

No other new blocking defects found. No production results read, files changed, jobs submitted, or scratch files created.