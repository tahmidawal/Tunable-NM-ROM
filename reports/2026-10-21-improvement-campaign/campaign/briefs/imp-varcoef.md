# Lane imp-varcoef — E3: variable-coefficient Burgers, FOM-only benchmark first

Worktree `worktrees/2026-10-21-imp-varcoef`, branch `exp/2026-10-21-imp-varcoef`, lane dir
`experiments/imp-varcoef/`, namespace `/cluster/tufts/paralab/tawal01/imp-varcoef/`.
**GPU-hour cap: 15 (FOM-only phase ≤ 8). Lane concurrent-job cap: 2.** Wave 1. Read `COMMON-BRIEF.md` first.

## Problem it answers
Problem 1 (a coarse sine spectral solver wins on the box because of fast transforms and an exactly integrated
constant-coefficient Laplacian). A high-contrast diffusion coefficient $\nu(x)$ removes the exact spectral
integration. Source: `reports/2026-10-20-spectral-investigation/where-we-win/report.md` (ranked settings,
competitors, kill criteria) and why-spectral §cost model.

## Phase 1 — FOM-only benchmark (do this first)
Viscous Burgers 2D on the box with $\nabla\cdot(\nu(x)\nabla u)$, high-contrast $\nu(x)$ (define the
coefficient family and contrast levels in DESIGN, e.g. contrast 10 and 100). Competitors, each tuned and
accuracy-matched at 1 %, 3 %, 10 % worst error on a new dev cohort, with a second-order reference built with the
E0 recipe:
- spectral IMEX with **warm-started CG** (constant-coefficient preconditioner);
- exponential integrators (e.g. ETD with a constant-coefficient split, or Krylov-exponential);
- finite differences with multigrid.
Our box cost = the deployed bank ROM per-query cost on the box at the same targets (from jcp-p1-scaling /
jcp-fom2d, same GPU type; re-measure in the same job if possible).

## Gate (verbatim; pre-register it)
Train a ROM only if the fastest FOM is ≥ 2× slower than our box cost. If not, stop and report (no ROM trained).
If the gate passes, Phase 2: retrain the bank on the variable-coefficient family (single change), score vs
the fastest FOM with the scoring contract; write Phase 2's DESIGN amendment and get it audited before any
training. Phase 2 must fit in the remaining cap.

## Build on
jcp-fom2d (`psjax2d.py`, `ladder.py`, `match2.py`, `targets.py`), jcp-references (`refcore.py` recipe),
jcp-p1-scaling (box cost), spectral-pilot (`worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/`).
