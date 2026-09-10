# Modified CP and empirical quadrature for fresh two-dimensional waves

This implements the user-approved single-seed architecture pilot. Numerical results
are pending; implementation checks do not establish accuracy or speed advantages.

## Frozen comparison

Train CP, lightly modulated CP and FiLM-INR separately for reflective Dirichlet and
absorbing boundaries. The common decoder source is content-hash-identical to the
approved Burgers companion experiment. All arms decode displacement and velocity
jointly from **32 total latent coordinates**, with CP rank 64 and 128 smooth tests
per component. This has fewer phase-space coordinates than the earlier displacement
manifold configuration using independent 32-dimensional position and tangent-velocity
coordinates; that earlier configuration is not a matched architecture control.

Train once on 256 intervals, then freeze parameters for 256 and 512. The reflective
decoder clamps coordinates to the first and last training interior positions inside
a one-cell boundary strip and multiplies the entire output by a linear envelope.
This preserves all training-grid values and prevents unconstrained masked table
endpoints from contaminating finer-grid predictions. Absorbing endpoints are trained
and unchanged. Modulation occurs before SiLU within coordinate branches, with zero
output initialization that exactly recovers the shared pretrained CP.

The approved compact Gaussian-core generator is unchanged. New training, validation
and evaluation seeds, all hyperparameters and solver sweeps are in `wave/config.json`.
Training uses physical displacement and velocity with a fixed, training-only
normalization: the RMS over initial training displacement mass norms and initial
energy-state norms. The latter remains positive for zero initial velocity. These
same two persisted constants normalize online fitting and weak residuals. No query
norm, truth state, physical initial-condition descriptors or time enters a decoder.

## PDE and quadrature

Use the independently verified post-reset edge-stiffness and trapezoid-mass wave
operators, with first-order Crank–Nicolson for both ROM and CG FOM. The full solve is

$$
\left(M+\frac{\Delta t}{2}C+\frac{\Delta t^2}{4}K\right)u_{n+1}
=\left(M+\frac{\Delta t}{2}C-\frac{\Delta t^2}{4}K\right)u_n+\Delta t Mv_n,
$$

followed by $v_{n+1}=2(u_{n+1}-u_n)/\Delta t-v_n$. This matrix is symmetric positive
definite in Euclidean coordinates, including absorbing boundary mass weights.
Report true CG residuals. Retain the exact reflective spectral propagator and fresh
RK4 absorber as efficient additional FOM controls.

Weak tests are discrete sine modes for Dirichlet and cosine modes including the
constant for absorbing boundaries. Transfer stiffness to these smooth tests using
their exact generalized eigenvalues; never minimize the strong residual. Fit NNLS
on decoder outputs with a constant measure row, full-grid targets and RMS row scaling.
Volume support has four or eight times the test count. Absorbing damping has four
separate positive face rules, retaining both corner half-weights. Supports are refitted
per frozen checkpoint, mesh and requested budget. Perturbed latent states and full
weak residual checks diagnose quadrature mismatch.

## Timing and acceptance

The primary query begins with full initial fields on the GPU and ends with all
requested displacement/velocity outputs on the GPU. Charge sampled initialization,
latent evolution and reconstruction; exclude training, quadrature construction,
compilation and host transfer. Every architecture and FOM comparator is measured on
the same job/device. Preserve two discarded warm-ups, seven repetitions, full-grid
physical errors and hashes from each actual timed output, and median/outlier counts.
Validation alone selects configurations; new evaluation cases are generated only
after selection is persisted. Numerical completion and stationarity are separate;
budget or stalled solves cannot qualify as converged target passes.

Measure displacement mass error relative to initial displacement mass norm, velocity
mass error relative to initial energy-state norm, and joint energy-state error on
that same initial energy-state norm. Both one-percent and five-percent targets require
every component for every evaluation trajectory. Primary reference is the verified
semidiscrete PDE. Absorbing temporal refinement is rechecked per new reference case.
The inherited conditional spatial estimate does not universally certify one-percent
continuum accuracy; continuum claims remain provisional until supported by refinement.

## Glossary

CP: sum of products of one-dimensional spatial factors. FiLM: latent-dependent scale
and shift before a neural activation. INR: coordinate-input neural representation.
EQ: offline positive empirical quadrature. NNLS: nonnegative least squares used to
fit quadrature weights. $M$: full-order mass matrix; the phrase test count denotes
the separate number of weak test modes. $C$: absorbing boundary damping matrix.
$K$: speed-scaled stiffness matrix. CG: conjugate gradients on the symmetric full
time-step matrix. FOM: full-order model. ROM: reduced-order model. A checkpoint is
frozen learned model parameters; mesh transfer evaluates those same parameters on a
new grid. Stationarity means the residual is orthogonal to every latent derivative
within the declared normalized-gradient tolerance. A complete query includes all
online work required to return the requested fields.
