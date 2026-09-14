# Additive linear enrichment of the accepted nonlinear wave head

Final bounded architecture screen before multiresolution/fresh-development confirmation. Preserve the accepted 32-coordinate phase-trained head exactly and add eight fixed linear training-PCA directions in the same 64-function bank. This is distinct from relearning a 40-coordinate head.

Let $B_8$ contain columns 33 through 40 of the original standardized training coefficient-PCA map. Define

$$h_{40}(z,y)=h_{32}(z)+B_8y.$$

The implementation appends eight zero rows to the first hidden-layer weight matrix and eight fixed columns to the linear skip. All original nonlinear weights, biases, scales and 32-coordinate responses remain unchanged. The additional coordinates use the original dimensionless standardized PCA-score units. The original manifold is included exactly at $y=0$; this inclusion is not a guarantee of better projected dynamics.

Verify value, original-coordinate Jacobian, physical tangent and curvature inclusion, the eight new linear derivatives, and full Jacobian rank on fixed training-code probes. Fit all 40 initial coordinates from the supplied displacement/velocity. The affine initializer and fixed-start library are derived only from the already accepted training artifacts; no development reference enters initialization. Retain all 64 weak equations, exact curvature, rank guards/fallbacks and physical velocity from latent evolution.

`nested40-config.json` fixes the two opened cases, 64 intervals, three timed repetitions, the retained timestep and every nonlinear method's halved-step control. Pair the original implementation, accelerated original head, accepted phase-trained 32-coordinate head, retrained 40-coordinate head, new nested head, CG timestep/tolerance controls and direct DST. No new bank/head training occurs. The unchanged accuracy, stationarity, conditioning, refinement, complete-output and timing rules apply.

After this screen, freeze the best eligible true nonlinear manifold and complete the three-resolution comparison on the opened cohort plus preregistered fresh development inputs even if the all-state accuracy target remains missed. No further wave architecture or training search is planned in this round. Keep the larger linear-bank/hybrid controls and rejected retrained/timestep arms in their existing evidence archives.

## Glossary

- **Additive enrichment:** additional freely solved linear coefficients around a preserved nonlinear decoder.
- **Nested manifold:** a larger set of representable states that contains the previous set exactly.
- **PCA directions:** fixed spatial-coefficient directions learned from training snapshots.
- **Tangent / curvature:** first / directional second derivative of the coefficient decoder.
- **True nonlinear evolution:** time integration in the nonlinear manifold's coordinates and velocities.
- **CG / DST:** iterative conjugate-gradient midpoint / direct discrete-sine-transform full-order controls.
