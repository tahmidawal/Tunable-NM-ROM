# Matched head/span trajectory mechanism

Prospective development experiment. No new numerical results exist yet. The frozen learned NS3D bank and nonlinear head are reused without training. The machine-readable design and gates are in `configs/pilot.json`.

## Falsifiable question

Does restricting the same spatial bank to the frozen nonlinear head reduce physical trajectory drift after matching the initial reconstruction, weak rows, row weights, free frame rule, integration grid, operator and nonlinear solve accuracy? A vanished advantage is an informative outcome.

For the shared-initial-state comparison, both arms start from exactly $a_0=h(z_0)$ and the same frame. The original physical initial field supplies the same reference trajectory. A separately labeled native-span projection diagnoses the initialization contribution. Neither the head fit nor the trajectory solve is claimed globally optimal.

The minimized residual uses the existing diffusion-preconditioned weak midpoint objective. The expanded test residual is evaluated independently on additional Fourier rows. At common previous head states we additionally seed a full-span solve from the reached head solution, making any physical deterioration despite objective improvement observable without different previous states.

## Implementation and execution order

1. Hash-copy the source modules and frozen checkpoints, preserving the source commit and SHA256 manifest. Build a driver with explicit array arguments so no large constants are captured.
2. Regenerate the exposed development prefix from its original seed, sample the frozen bank, and assemble the exact quadratic weak tensor with dense-grid independent checks.
3. Fit the head initial condition from multiple frozen training-code candidates. Retain all candidate values and qualification data.
4. Solve matched head, shared-initial span and native-initial span trajectories using adaptive LM with the same termination rule and retain every iteration including rejected trials. Run tight/tighter pairs, test-space enrichment and time-grid refinement on the initial two cases.
5. Save reconstructible trajectories, physical error histories and frame/shape diagnostics. Verify analytic Jacobian actions independently by finite differences. Include frame directions in the physical trial tangent. Fit reached span states to the head using multiple starts and label the result a best-found fixed-frame distance.
6. Apply the predeclared gates before any cohort expansion. Diagnose failures rather than deleting cases or selecting optimistic curves. If passed, expand the same cohort prefix to the full exposed development panel under the parent's compute reservation.

## Numerical clarification

The reference reproduces the inherited CNAB2 spatial and initial-field contract, checks output-grid parity, and compares reference time steps on each pilot case. Both the absolute and ROM-relative reference gates in the JSON must pass. This checks the same-grid FOM, not continuum accuracy. All trajectory and reference differences are normalized by the original physical initial-field norm. The physical-tangent Gram includes translations and reports every eigenvalue, the rank cutoff and retained rank, excluding unresolved directions rather than dividing by them. Time-step refinement changes the inherited diffusion row preconditioner and is therefore method-setting robustness, not a pure temporal-order proof. Solver cap and stagnation are failures even when trajectories happen to agree. Zero residual has zero gradient; otherwise relative stationarity is required without a roundoff waiver.

## Reporting contract

Measured fields and diagnostics are generated from saved arrays. Report median and worst-case time histories and all case outcomes, not a favorable case alone. This lane makes no timing advantage or uniform stability claim. Boundary conditions are periodic by construction; report spectral divergence and energy diagnostics. Source hashes, seed, mesh, model dimensions, scalar rows, total unknown count, hardware, job id and commit accompany each run.

## Glossary

A head maps a few latent coordinates to spatial-bank coefficients. A span permits arbitrary coefficients in that bank. A free frame adds three solved translations. Weak rows are scalar tests of the PDE residual. LM is an adaptive least-squares solver. Stationarity measures the remaining local objective gradient. Tight/tighter means two accuracy settings for that solver. A common initial state is an identical reconstructed physical field in both arms. A best-found fit is the smallest value found by the recorded starts, not a guaranteed global optimum.
