# Prospective centered-difference diagnostic amendment

The original ladder01 remains failed and retained. Independent source and saved-output review confirms Newton convergence before the manufactured h1 Jacobian diagnostic failure. Error times step is approximately constant, consistent with cancellation, not a proved causal measurement.

The actual fixed-mesh, fixed-quadrature residual has linear viscous/pressure/continuity terms and quadratic convection; forcing is state-independent. Centered differences therefore have no truncation term in exact arithmetic. Adopt steps `[1e-2,1e-3,1e-4]` uniformly for all future cases, preserving direction, normalization, row scales, signal and plateau thresholds, nonlinear solver and physical gates. Persist reference/perturbed coefficients, free mask, scales, direction, assembled action and plus/minus residuals before checking acceptance.

This authorizes source preparation only. An exact reviewed candidate and one bounded allocation release must precede execution. No original result is retrospectively relabeled; no reference or training admission follows from a diagnostic pass.
