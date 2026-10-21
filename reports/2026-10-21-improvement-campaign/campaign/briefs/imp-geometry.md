# Lane imp-geometry — E2 (pilot Part B′): no fast transform, against strong solvers

Worktree `worktrees/2026-10-21-imp-geometry`, branch `exp/2026-10-21-imp-geometry`, lane dir
`experiments/imp-geometry/`, namespace `/cluster/tufts/paralab/tawal01/imp-geometry/`.
**GPU-hour cap: 62. Lane concurrent-job cap: 3.** Read `COMMON-BRIEF.md` first (binding).

## Phasing
- **Now (wave 1, design mode): DESIGN, Codex design audit, all code, code audit, competitor builds, and local
  CPU/small smoke. No cluster production jobs** until the coordinator sends "release production" (after the
  wave-1 lanes hand back: E5 decides the best reduced stepper you use, and the wave is re-priced). Cluster
  smoke/memory-pilot jobs ≤ 1 GPU-h total are allowed before release. When ready, set heartbeat step
  `"ready for production, waiting for release"` and send the coordinator a short ready note.
- Leave the reduced stepper a parameter: default = the incumbent CN-R LSPG; switch to E5's winner if it passes.

## Experiment (pilot report §7 Part B′; DESIGN in `worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/`)
Burgers 2D on an **L-shape** and a **box with a circular hole**; narrow family and wide family W = 0.4 (same
bump widths as A′), two steepness levels; 32 test cases and a held-out test-once cohort per cell
(2 geometries × 2 families × 2 levels = 8 cells).
- Competitors (decisive part), each tuned over order, mesh, step and solver tolerance: **high-order**
  (spectral elements / $p$-FEM with static condensation, or multi-domain Chebyshev on the L-shape), finite
  differences IMEX with **multigrid**, and **POD + ECSW**.
- Reduced arms: the bank with an R-function/distance envelope (retrained per cell), POD–Galerkin, POD-ECSW.
- Box control per cell (as the pilot's AB) against the spectral FOM.
- Speed-up S at matched median and worst error (1 %, 3 %, 10 %), same GPU, one job.

## Win / withdraw rule (verbatim; pre-register it)
Win: $S \ge 2$ against the best competitor on the wide family. Otherwise withdraw the geometry claim. Also
withdraw any geometry attribution if the box control gives the ROM the same S against the spectral FOM.
If a high-order competitor cannot be built in budget: say so, use the strongest available, label S "against
the strongest tested competitor". The E2 verdict decides whether E4 (3D cube with small spherical holes) runs.

## Build on
The spectral pilot (Part B code, FD-IMEX/SBDF2, references), jcp-fom2d, jcp-hr-baselines (POD-ECSW),
jcp-references (reference recipe), bank training code (jcp-robust `e2bank.py`, jcp-combined).
