# Current heat NMROM versus the historical iterative FOM algorithm

Prospective development comparison requested by the user. No new speedup is established by this design.

Continue the existing approved heat worktree from `9891754`. The user selected the older iterative CG FOM as the main comparison. Keep the current frozen decoder, supplied initial conditions, weak objective, original initialization, nonlinear tolerance and output times. Compare the original NMROM and its tested Cholesky optimization to full-grid CG on the same allocation. The primary pair is Cholesky NMROM and CG with the same Crank–Nicolson time grid; a separate backward-Euler arm retains the historical time-step count. This ports the historical algorithm to the current physical problem, rather than claiming a replay of the old checkpoint, geometry or data family.

## Full-order algorithms

Use the positive finite-difference Dirichlet Laplacian $L$ and solve

$$[I+\theta\Delta t\nu L]u_{j+1}=[I-(1-\theta)\Delta t\nu L]u_j.$$

The main CG comparison uses the current ROM's Crank–Nicolson values of $\theta$ and $\Delta t$. The historical stepping control uses backward Euler and the archived number of steps. All concrete tolerances, meshes, seeds and budgets are frozen in `config-iterative-cg.json`. Looser CG controls test the effect of requiring less linear-solve accuracy; they are retained whether or not they meet the declared physical target. There is no claim that this small tolerance screen finds the globally fastest classical method.

Implement the standard unpreconditioned CG recurrence with a previous-state starting guess, the same relative stopping rule as JAX CG, and a hard iteration cap. Compile both inner iterations and the complete rollout. Charge an explicit true-residual verification at every step and retain attempted iterations, recurrent and true relative residuals, and convergence flags. Store only requested full fields and small step diagnostics; do not store all internal full-grid states inside the timed query.

The direct transform FOM and free-coefficient linear bank remain separately labeled diagnostics. They do not replace the user-selected primary iterative comparator. Reference generation still uses the independently checked spectral heat solution, and all physical errors use the same refined reference. The archived direct-FOM results are preserved.

## Measurement and acceptance

Use the existing development cases at the declared mesh resolutions, with full input processing and all requested full outputs. Preserve timing repetitions and full fields from each actual invocation, burn in before timed blocks, alternate method order, report medians/outliers, and record GPU type, source and checkpoint hashes, seeds and configuration. Compilation and offline assembly are recorded separately. Every job has its own directory and uses the mandatory GPU/f64/highest-precision preflight.

Before submission, compare the new CG implementation with dense finite-difference solves and the installed JAX CG on a nonsingle-mode fixture, then perform a sub-minute frozen-decoder GPU smoke. After collection, independently check all saved field errors, original nonlinear trajectory parity and stopping diagnostics, and compare CG outputs with independent SciPy discrete sine-transform propagation of the exact corresponding time-step formula. Physical target failure is a reported outcome, not an audit failure to conceal. Checksum the archive before deleting only the completed attempt directory.

Report the same-grid CG speed ratio with its tolerance and time-step formula beside it. Also report the looser-tolerance and historical stepping controls. A win over this iterative baseline does not establish a win over every FOM algorithm or at globally optimized classical cost. No training, new worktree, merge or final-cohort opening is part of this round.

## Glossary

- **NMROM / FOM:** nonlinear-manifold reduced model / full-grid numerical solver.
- **CG / Cholesky:** iterative conjugate-gradient full-grid solve / factorization used in the small nonlinear update system.
- **Crank–Nicolson / backward Euler:** centered implicit time discretization / fully implicit first-order time discretization.
- **Dirichlet Laplacian / diffusivity:** spatial second-derivative operator with zero boundary values / heat diffusion coefficient.
- **Tolerance / recurrent residual / true residual:** relative stopping threshold / recursively updated equation discrepancy / discrepancy recomputed using the actual final iterate.
- **Primary pair / control:** declared main comparison / additional algorithm or setting that tests its interpretation.
- **Invocation / median / outlier:** one solve producing both the timed and graded output / middle timing value / repetition above the stated within-case threshold, retained.
- **Spectral reference / discrete transform propagation:** heat reference using spatial modes / independent application of the same discrete time-step formula in those modes.
- **Development / final cohort:** cases used to choose and study the method / separate confirmation cases, unopened here.
