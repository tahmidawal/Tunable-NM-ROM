# Paper positioning and table structure

Editorial recommendation based on the archived submission, current manuscript, and collected campaign evidence. This document contains no new numerical results; pending final evaluations cannot support final paper claims yet.

## Central claim

A trained nonlinear-manifold reduced model can expose a deployment-time accuracy–cost tradeoff by activating precomputed correction directions. Its practical value depends on the equation, resolution, and comparator. Frame measured acceleration against explicitly named solvers; do not promise universal superiority over full-order solvers or neural operators.

Preserve the original three contributions: the tunable method, the architecture enabling it, and the numerical implementation. Integrate limitations into the experimental findings rather than making them an additional contribution. The manuscript must distinguish reduced-model error against a converged discrete reference from physical error against a refined reference.

## Preserve the original table roles

| Original role | Revised content | Essential columns |
|---|---|---|
| Method comparison | Matched NM-ROM, neural operators, POD and named FOMs; separate dimension panels | Problem, mesh, method, relative L2 error (%), runtime (ms), speedup versus named FOM |
| Best configurations | Development-selected fast and accurate settings from the same frozen model, then evaluated on held-out cases | Problem, mesh, setting, correction rank, error, runtime, speedup |
| Which knob to turn | Measured effects of actual deployment controls | Control, meaning, observed accuracy effect, observed cost effect |

For a wide comparison table, preserve paired error/speedup columns per method and put runtimes in an immediately adjacent timing table. Do not compress away the FOM identity, tolerance, or its error. Fast settings must satisfy the declared numerical and accuracy requirements; if no setting passes, say so. Do not select a different winner using each final test case's truth.

## Meaningful controls

- Correction rank $q$: primary representation-accuracy control. Freeze the bank, nonlinear head, correction ordering, and residual test space when isolating its effect. A schedule changing the test count alongside $q$ is a separate experiment.
- Solver tolerance: controls numerical convergence. Tighter tolerance need not improve field accuracy once representation error dominates.
- Iteration limit: computational budget and safeguard. A truncated or stalled solve is not silently promoted as a converged fast setting.
- Empirical-quadrature size: selection among precomputed, validated rules where empirical quadrature is actually used. It is not an available control in every PDE or dimension.

Keep latent dimension, bank size, training schedule, and residual test count in the setup and ablation material. They remain necessary for reproducibility but are not interchangeable with deployment-time controls from one frozen model.

## Speedup convention

Use $S=T_{\mathrm{FOM}}/T_{\mathrm{method}}$. All operands come from paired synchronized measurements in the same allocation, with identical input/output scope and retained repetitions. Separate GPU query time from complete host-to-host time.

In the main comparison use a common, development-selected FOM per problem, mesh, and declared accuracy target; show its error. If reporting a separate cost-to-accuracy comparison, explicitly select the fastest tested passing FOM meeting that target and state that the comparator can differ by target. Never silently change the denominator between methods in a shared table.

Use CG for applicable symmetric positive-definite systems; name the actual nonlinear or time-integration solver for other problems. Keep efficient direct-transform controls visible where applicable. Use efficient production solver timings, with audit traces replayed separately when instrumentation would materially affect cost.

## Evidence and labels

Burgers supplies the most developed evidence for correction-rank tunability, with same-grid and physical-reference scope distinguished. Linear PDEs test when the nonlinear restriction helps relative to a reduced linear solve. Navier–Stokes tests the present method's limitations and must not be portrayed as an established speed or target-accuracy success. New three-dimensional findings enter final tables only after their frozen evaluations and audits complete.

Label the linear-bank baseline “Linear ROM using the learned bank.” Its basis is part of the learned construction, but its direct reduced solve bypasses the nonlinear manifold. Do not attribute its speedups to nonlinear NM-ROM. Keep POD and strong neural operators in the comparison even when they win.

## Source documents

- [Archived submission](../worktrees/2026-09-16-paper-refresh/private/old-submission/main.tex)
- [Current manuscript](../worktrees/2026-09-16-paper-refresh/paper/main.tex)
- [Collected CG comparisons](2026-09-20-iterative-cg-comparisons.md)
- [Collected direct-FOM inventory](2026-09-20-2d-3d-error-speedup.md)

## Glossary

- NM-ROM: nonlinear-manifold reduced-order model; solves the PDE using a learned nonlinear representation.
- Correction rank: number of precomputed correction directions activated during a solve.
- Bank: fixed collection of spatial functions used to reconstruct the solution.
- Head: learned nonlinear map from latent variables to bank coefficients.
- POD: proper orthogonal decomposition, a conventional way to construct a reduced basis from snapshots.
- FOM: full-order numerical model on the stated mesh.
- CG: conjugate gradient, an iterative solver for symmetric positive-definite linear systems.
- Empirical quadrature: precomputed weighted sampling used to approximate specified residual terms.
- Residual test count: number of functions onto which the PDE residual is projected.
- Relative L2 error: norm of prediction minus reference divided by the stated reference norm; its time aggregation must be defined per problem.
- Runtime: median measured time to produce the requested solution outputs, using the stated timing scope.
- Speedup: named full-order runtime divided by method runtime; values below unity indicate a slowdown.
- Development selection: choosing settings on cases available during method development.
- Held-out evaluation: testing frozen settings on reserved cases not used to choose them.
- Non-dominated: no tested alternative is both at least as accurate and at least as fast, with a strict improvement in one.
- Physical reference: refined numerical solution used to assess error beyond the discretization on the working mesh.
