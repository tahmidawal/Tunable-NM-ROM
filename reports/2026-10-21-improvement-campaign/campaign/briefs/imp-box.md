# Lane imp-box — E1 (pilot Part A′): the box, family width × steepness

Worktree `worktrees/2026-10-22-imp-box`, branch `exp/2026-10-22-imp-box`, lane dir `experiments/imp-box/`,
namespace `/cluster/tufts/paralab/tawal01/imp-box/`. **GPU-hour cap: 87. Lane concurrent-job cap: 3.**
Wave 3. Read `COMMON-BRIEF.md` first (binding; note the worktree is dated 2026-10-22).

## Problem it answers
Problem 1: a coarse sine spectral solver is faster at equal accuracy on the box (2D 4096² S = 0.05–0.71×).
The pilot (`worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/`, report §7 Part A′, post-hoc
control AB) found the reduced model wins only with a narrow family; this lane tests family width × steepness.

## Experiment
Burgers 2D box, family width W ∈ {0.08, 0.16, 0.28, 0.4} (bump centres range; bump width U[0.035, 0.05]) ×
steepness levels A1–A4 ($s = a/\nu$, paired ν, factor 3 per level; A4 needs $N_{\rm ref}$ ≈ 768 on GPU) = 16
cells; 32 test cases per cell plus a dev/val split by seed; a test-once cohort per cell opened once at the end.
Arms (each tuned, accuracy-matched, same GPU, one job per comparison):
1. the tuned sine spectral FOM (ETDRK4, fixed and adaptive step) — from jcp-fom2d `psjax2d.py`;
2. **our real bank, retrained per cell** (linear span, off-mesh rule) — on the 8 cells W ∈ {0.08, 0.4}
   (pilot's planning), with the incumbent CN-R LSPG stepper **and** the Galerkin ETDRK4 Δt = 0.0125 stepper from
   imp-faststep (`worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/`, its 14–24× faster evolution;
   it failed E5 only by a small worst-error margin, irrelevant at 1–10 % targets — still Q1/Q2-qualify it here);
3. the pilot's POD–Galerkin stand-in (tensor / grid ETDRK4);
4. one front-tracking or nonlinear (transport-aware) model, on the 4 cells W = 0.4;
5. POD + ECSW (jcp-hr-baselines / imp-faststep ETDRK4 variant).
Speed-up S at matched median and worst error (1 %, 3 %, 10 %), H200, same job. Report the width at which S
crosses 1 and 2 per target and level, and per-arm query-cost decomposition.

## Stop rule (verbatim; pre-register it)
Stop the line if every reduced model has $S < 0.5$ at every target on the wide family (W ≥ 0.28 cells).
Also stop the bank arm if its projection floor at its own equal-cost size $r_\times$ is above τ in every cell.
Run the wide-family cells (W = 0.4, then 0.28) **first** so the stop rule can fire before the narrow cells spend.

## Defer order if the cap binds
transport/nonlinear arm → A4 level → W = 0.16 cells → the second data size (384 trajectories) → bank on W = 0.08.

## Build on
spectral pilot (Part A code, references, POD-Galerkin, AB control), jcp-fom2d (spectral ladder, `match2.py`),
jcp-references (reference recipe), bank training code (jcp-robust `e2bank.py`, jcp-combined), imp-faststep
(Galerkin ETDRK4 precontraction, `fscore.py`), imp-geometry (bank retrain-per-cell pipeline and the
measured bank-training cost: `worktrees/2026-10-21-imp-geometry/experiments/imp-geometry/`, read-only),
jcp-hr-baselines and imp-multimesh (POD-ECSW).
