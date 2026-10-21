# Lane imp-faststep — E5 (ideation W1): a faster reduced time step

Worktree `worktrees/2026-10-21-imp-faststep`, branch `exp/2026-10-21-imp-faststep`, lane dir
`experiments/imp-faststep/`, cluster namespace `/cluster/tufts/paralab/tawal01/imp-faststep/`.
**GPU-hour cap: 4. Lane concurrent-job cap: 2.** Wave 1. Read `COMMON-BRIEF.md` first (binding).

## Problem it answers
Problem 2 (tuned POD + ECSW dominates us, 2D 8/8, 3D 7/9) and part of Problem 1: our per-step cost is about
two dense $M\times r$ Jacobian assemblies over POD-ECSW (r = 384: 1.74 ms vs 0.613 ms/step, 1024², H200), and
LSPG's Newton loop plus CN-R time error dominate. Source: ideation report §2.2, F1, F5, F6 and §5 row W1.

## Arms (all in one job per mesh, same GPU, paired timing)
1. current LSPG CN-R (the incumbent deployed 2D configuration, from jcp-combined / jcp-p1-scaling);
2. precontracted Galerkin CN (offline precontraction of the linear and quadratic parts onto the test space);
3. the same with ETDRK4 at Δt ∈ {0.0125, 0.025};
4. one-pass linearised Galerkin CN (F5: single Newton pass / Jacobian reuse);
5. **POD + ECSW with the same ETDRK4 step** (shared integrator control; reuse jcp-hr-baselines code/tuning);
6. an M = 1152 LSPG CN-R timing control (F6 bounded control).
Grid: r ∈ {128, 384}; meshes 1024² and 4096²; cohorts dev6 + val32 (selection on dev only).

## Stop rule (verbatim; pre-register it)
Stop if any run is unstable or invalid, its worst error is above the current one (arm 1), or the r = 384
evolution is < 3× faster. Report the outcome for each new arm separately; the E5 line is "fixed" only if at
least one Galerkin/ETDRK4 arm passes all three conditions.

## Hand-off obligations
- **E6 (lane imp-hrcost) consumes your precontracted Galerkin arm** (arm 2/3). As soon as arm 2's code passes
  its Codex code audit, commit it and write `experiments/imp-faststep/HANDOFF-E6.md` (module path, function
  signatures, frozen config hash, how to time it). Do this before production, so E6 can build on it.
- Export per-step and per-query cost decompositions (evolution, initial projection, decode) in EXPORT.json;
  E1/E2 will reuse "the best stepper" you identify.

## Build on
jcp-combined (`c2common.py`, `c2grid.py`, `timing.py`), jcp-p1-scaling (`p1core.py`, `p1time.py`, component
timing), jcp-hr-baselines (`rom.py`, `hrcore.py`, `tune.py`, POD-ECSW), jcp-fom2d (`psjax2d.py` ETDRK4),
jcp-budget (Galerkin endpoint), jcp-references (E0). Ideation analysis scripts in
`reports/2026-10-20-improvement-ideation/analysis/`.

## Defer order if the cap binds
4096² r = 128 → M = 1152 control at 4096² → Δt = 0.025 at 4096².
