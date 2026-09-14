# Reflective-wave latent-capacity screen

Predeclared follow-up to the bounded phase-supervision experiment. Test one 40-coordinate nonlinear head in the same frozen 64-function bank; this is a capacity change, not a continuation of the 32-coordinate endpoint. Training-only affine projection diagnostics support the test but do not establish full physical or online accuracy.

Keep all original 64 training trajectories, seed, fixed affine encoder convention, 10000-step endpoint, optimizer and normalized field-plus-energy-plus-tangent objective. Extend the original standardized training coefficient-PCA encoder to 40 coordinates and differentiate that encoder consistently for velocity supervision. The bank and its weak operators remain unchanged. The frozen accepted 32-coordinate phase-trained head is a paired control, alongside the original deployed head and its faster geometry implementation.

There are 64 weak bank equations and 40 latent unknowns: 24 extra equations, with $M/k=1.6$. This retains more overdetermination and lower online cost than a 48-coordinate test; no 48-coordinate arm is included. Keep the same rank thresholds, conservative Cholesky bound, original QR/SVD fallback, scaled acceleration backward-residual monitoring and full curvature. Record exact sampled singular ratios and weak normal-force residuals so that approaching a square system cannot masquerade as improved accuracy.

Use only the same two opened reflective development inputs at 64 intervals, three repetitions, the retained timestep and its halved-step accuracy control. Pair named CG timestep/tolerance controls and direct DST. All full fields, actual initial fits, u/v/energy-state errors, phase and energy drift remain measured. The 5% all-state target and numerical criteria are unchanged. If the new head fails these criteria, preserve the failure; do not relax thresholds or promote a faster incomplete solve.

Freeze a global method/checkpoint selection after this screen, before the planned three-resolution confirmation and fresh development seed. Do not select epochs or per-case methods using confirmation truth. Final-paper cases stay sealed.

## Glossary

- **Latent capacity:** number of nonlinear configuration coordinates in the decoder.
- **Bank rank:** number of frozen learned spatial functions and independent weak bank equations.
- **Overdetermination:** having more weak equations than latent unknowns.
- **Fixed encoder:** unchanged linear formula for both training codes and their derivatives.
- **Weak normal force:** component of the reduced physical acceleration force outside the decoder tangent span.
- **Numerical criteria:** initial stationarity, finite/rank-valid trajectory and halved-step consistency.
- **CG / DST:** iterative and direct full-order wave controls.
