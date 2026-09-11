# Reflective-wave geometry and timestep acceleration

Predeclared development experiment continuing the validated fresh-wave MLP32/rank-64 bank. Numbers are not yet measured. The frozen `fresh-wave-head` mathematical sources and checkpoint remain unchanged. Absorbing waves and the sealed final cohort are excluded.

The control solves the weak manifold equation with the existing curvature-inclusive RK4 evolution, cold-fit starts and output contract. With bank-orthonormal coefficients $a=h(z)$ and physical velocity $b=J(z)w$, the acceleration satisfies

$$\ddot z=\arg\min_x\|Jx+H_h(z)[w,w]+cD b+c^2K a\|_2.$$

## Geometry arms

The analytic SiLU implementation shares the value, tangent, Jacobian and directional second derivative in one layer traversal. Independent automatic differentiation and saved-trajectory checks test all four quantities. The first arm retains QR acceleration and computes singular values from its triangular factor. The second uses QR acceleration with a sufficient rank bound. The third uses Cholesky normal equations only in a comfortably conditioned region and otherwise restores QR with the original exact singular-value test. No ridge or reduced checking frequency is introduced.

For $J=QR$, the sufficient bound is

$$\frac{1}{\|R\|_F\|R^{-1}\|_F}\leq\frac{\sigma_{\min}(J)}{\sigma_{\max}(J)}.$$

Only a bound above $10^{-6}$ can bypass the original singular-value ratio threshold of $10^{-8}$. Otherwise the original singular-value calculation is performed. For $J^TJ=LL^T$, the corresponding bound is

$$\frac{1}{\|J\|_F\|L^{-1}\|_F}\leq\frac{\sigma_{\min}(J)}{\sigma_{\max}(J)}.$$

The Cholesky arm requires a bound above $10^{-3}$ and finite factors; all other cases return to original QR/SVD. These are exact-arithmetic inequalities with conservative finite-precision margins, not interval-arithmetic certificates. Every RK stage records its rank diagnostic, fallback count and scaled normal-equation backward residual. Lower bounds are labeled as such, not reported as exact singular ratios.

The guard smoke includes full-rank and rank-deficient diagonal examples around the original threshold. Saved nonlinear states receive independent exact singular-value and QR-acceleration checks. Geometry candidates must preserve the complete displacement, physical velocity and energy-state trajectory to below $10^{-8}$ on the original physical scales and retain the original finite/rank decisions. The scaled normal-equation residual must remain below $10^{-10}$.

## Screen and confirmation

`acceleration-screen-config.json` freezes two already-opened reflective development cases on 64 intervals, three repetitions, the original timestep, and separate doubled/quadrupled-timestep Cholesky arms. The original CG tolerances and direct DST comparison are timed within this same job. Standalone synchronized geometry, factorization and complete-RHS timings diagnose costs but are not additive shares of a compiled RK stage. Complete-query timing remains the headline measurement.

Timestep changes are separate numerical settings, not parity optimizations. Their all-state physical errors may worsen by at most 0.2 percentage point relative to the original method, and their halved-step discrepancy must stay below 1% on the reference initial scales. Phase and physical energy behavior are recorded alongside the displacement/velocity/energy-state errors. All initial fits must meet the existing stationarity/rank criteria. Select the fastest median configuration satisfying these numerical gates, without claiming an accuracy-qualified speedup if the original physical target remains missed.

Freeze a selector file before confirmation. Confirmation uses 64, 256 and 1024 intervals, the two opened inputs plus a separately seeded small fresh development cohort if the screen supports continuation. The final paper cohort stays sealed. Baseline, selected method, explicit halved-step accuracy controls, CG and DST remain paired within each confirmation job. Any changed modal-propagation/projection method requires a separate design with implicit differentiation of stationary nonlinear projection for physical velocity; it is not equivalent to latent RK4.

## Measurement and evidence

Inputs are full supplied GPU displacement/velocity fields and scalar wave speed. Charge full-source projection, every initialization fit, every RK stage and both full fields at all 49 output times. All large arrays are explicit compiled arguments. Mesh-only bank/operator construction is offline. Burn-in precedes every timed invocation. Preserve timing arrays, all outliers, actual output hashes, complete fields, code/config/checkpoint provenance, and stationarity/rank/energy/phase diagnostics. Pull all artifacts with checksums before deleting the exact cluster attempt directory.

## Glossary

- **Bank:** learned spatial functions; their coefficients form the small weak system.
- **MLP32:** frozen nonlinear coefficient network with 32 latent coordinates.
- **Physical velocity:** time derivative of the decoded displacement, $J(z)w$.
- **Curvature:** the decoder's directional second derivative needed in acceleration.
- **QR / SVD / Cholesky:** matrix factorizations for least squares, singular-value rank measurement, and positive-definite linear solves.
- **Rank bound:** conservative lower bound on the smallest-to-largest singular-value ratio.
- **Backward residual:** normalized discrepancy in the small acceleration equation after solving it.
- **Parity:** numerical agreement of the changed implementation with the frozen control.
- **CG / DST:** iterative conjugate-gradient midpoint FOM and direct discrete-sine-transform semidiscrete FOM.
- **Initial normalization:** field error divided by a fixed norm of the supplied initial state.
- **Development cohort:** inputs available for selecting model/method settings; not a sealed paper test.
