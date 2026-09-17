# Table plan — ICLR 2027 submission

One table per experiment, with the reviewer requirement it answers and its status on disk. Written
2026-09-17 during the nine-lane campaign; status column updated as lanes close. Reviewer requirement
codes refer to the NeurIPS 2026 reviews (`REVIEWER-RESPONSE-MAP.md`): R1 one checkpoint per curve,
R2 competitive full-order solvers, R3 tuned baselines on shared data, R4 SMA cold start, R5
advection case with a linear POD baseline under the same knobs, R6 seeds with variance, R7 whether
the family is really free from one model including quadrature refits, R8 timing protocol, R9 precise
PDE specification, R10 harder domains, R11 the intrusive/non-intrusive asymmetry, R12 no hand-typed
numbers, R13 a correct time-step derivation, R14 when the nonlinear manifold is needed.

**Status vocabulary.** `done` = audited run JSONs on disk and a generator reads them. `running` = job
in flight. `failed` = job failed, diagnosis or resubmission pending. `not run` = nothing on disk.

| # | Table | Answers | Lane | Status |
|---|---|---|---|---|
| T1 | Problem specification per PDE: equation, domain, boundary treatment, source or initial-condition family, mesh, time step, reference solver and tolerance, cohort sizes, seeds | R9, R13 | all | done (configs + generators) |
| T2 | Provenance and protocol per result: checkpoint SHA, K, R, commit, job id, GPU, precision, burn-in, repetitions, sync policy | R1, R8, R12 | all | done |
| T3 | The tunability curve, Burgers 256²: rungs q, dense and certified-EQ arms, both error metrics, t=0 compression, median GPU ms, same-job full-order controls | R1, R7 | b-panel, b-eqtop | done (256²), 1024² running |
| T4 | Rank versus test count, crossed q × M, dense | the confound objection | b-qxm | **done** — fixed-M ladder passes the bar alone (2.44× error, 5.16× cost); rank is 85 % of the variance |
| T5 | Final same-job panel at 256² and 1024²: ROM rungs, POD-LSPG ranks, FNO, Newton at several tolerances, exact direct solve | R2, R3, R5, R11, R14 | b-panel | 256² done (0 of 27 reduced arms non-dominated); 1024² pending |
| T6 | Head ablation at matched latent dimension, both PDEs | R5, R14 | inherited (head-ablation) | done |
| T7 | Three-layer decomposition: bank floor, best-found, solved | R14 | inherited + all lanes | done |
| T8 | Solver knobs at a fixed checkpoint: cap, tolerance, quadrature count, cold start | fxe8 F7/F8, 5mgh M11 | inherited | done |
| T9 | Quadrature rule certification: NNLS fit versus held-out error on reachable states, per rung | R7, 5mgh M8/M9 | b-eqtop | running — top rungs certify once the fit-state count is raised; draw-replication in flight |
| T10 | Mesh ladder, frozen checkpoint, 64 → 1024 | R2, sub-cubic scaling | inherited (mesh-ladder) | done |
| T11 | Where the family collapses: Poisson ladder with POD and the direct solve; heat linear bank; wave | R14, GwrW G5 | p-linear, w-ladder | Poisson 256² done, 1024² rerunning; wave 64²/1024² done, 256² rerunning |
| T12 | Seeds: per-rung mean and spread over three training seeds, monotone-seed count | R6 | b-seeds | running (three trainings) |
| T13 | Sealed cohort: development versus sealed worst error per rung, difficulty-normalised | R6, pre-registration | b-seeds | pending the seed jobs |
| T14 | Neural operators on shared data: FNO, U-Net, Transolver capacities, matched cohort against the ROM and the full-order solver | R3, 5mgh M1 | no-second | **done** — U-Net and Transolver beat the ROM; accuracy claim withdrawn |
| T15 | Speed at parity: fused residual and Jacobian, folded head, block solve | R8 | inherited (b-speed) | done |
| T16 | Training study (appendix): data ladder, objective, K, smoothness penalty, with the reproduction caveat | 5mgh originality | inherited (b-head-train) | done (negative) |
| T17 | SMA-NM-ROM under the same cold-start protocol | R4 | none | **not run** — state as a limitation |
| T18 | L-shaped Poisson: signed-distance boundary factor, sparse-direct and preconditioned-CG controls, where the direct solver stops applying | R10 | lshape | running |

## What the tables must not claim

- No accuracy superiority over neural operators. T14 overturned it.
- No speedup over an efficient full-order solver in 2D. T5 shows zero of 27 reduced arms non-dominated
  at 256².
- No cost ratio across jobs or GPUs. Same-allocation only, per the timing protocol.
- No quadrature rule certified by its NNLS fitting residual; only held-out error on reachable states.
- No performance ratio in the abstract until T12 and T13 exist.

## Decisions still with the user

1. Burgers headline metric: worst over evolved times with the t=0 compression reported separately,
   against worst over all times. The fixed-M ladder is monotone on both, so this now moves the
   headline number rather than the ladder's shape.
2. The code-only GitHub mirror push, prepared and dry-run, not run.
3. Merge or archive the campaign worktrees.
