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
operators, with Crank–Nicolson on the first-order displacement/velocity system
for both ROM and CG FOM. The full solve is

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

For original CP only, contract these same approximate EQ weights with the frozen
spatial factors offline. Online weak moments then multiply the resulting small maps
by the nonlinear coefficient head; include the masked bias contribution. This is an
algebraically equivalent optimization of the EQ residual, not a switch to exact
full-grid operators. Sampled cold initialization remains charged. ModCP and FiLM
retain state-dependent sampled evaluation. Sparse-rule value and latent-Jacobian
parity tests cover both boundaries and the finer-grid boundary strip.

## Timing and acceptance

The primary query begins with full initial fields on the GPU and ends with all
requested displacement/velocity outputs on the GPU. Charge sampled initialization,
latent evolution and reconstruction; exclude training, quadrature construction,
compilation and host transfer. Every architecture and FOM comparator is measured on
the same job/device. Preserve two discarded warm-ups, seven repetitions, full-grid
physical errors and hashes from each actual timed output, and median/outlier counts.
Primary ROM timing uses one compiled complete query and one completion barrier;
separately warmed initialization/evolution/reconstruction calls are diagnostic only.
The coordinator-approved validation refinement evaluates every configuration on all
validation cases once, with seven additional timing repetitions on predetermined
case zero used only as a selection-cost proxy. These proxy rows remain separate.
Validation and evaluation run in separate allocations. All Burgers and both wave
boundary/mesh selections are persisted and globally sealed before drawing any
shared evaluation cohort. Evaluation imports hashes of the complete global seal,
its own validation handoff, and both mesh selection proofs. Validation timings
remain in the imported proof; final timing rows all come from the new allocation.
Every selected configuration receives the full seven-repetition evaluation protocol.
Final prediction fields preserve all original values in lossless stored NPZ files.
One shared full reference file per case and mesh is identified by its SHA256 in
every associated timing row. Validation observation-grid fields retain their
bounded format. Full local archives remain outside Git with tracked manifests;
the coordinator verifies a durable main/artifacts copy before any worktree cleanup.
Validation alone selects configurations; new evaluation cases are generated only
after selection is persisted. Numerical completion and stationarity are separate;
budget and small-step exits can qualify by physical accuracy while remaining explicitly
nonstationary. Damping-ceiling, nonfinite and zero-tangent breakdowns cannot qualify.

Measure displacement mass error relative to initial displacement mass norm, velocity
mass error relative to initial energy-state norm, and joint energy-state error on
that same initial energy-state norm. Both one-percent and five-percent targets require
every component for every evaluation trajectory. Primary reference is the verified
semidiscrete PDE. Absorbing temporal refinement is rechecked per new reference case.
The inherited conditional spatial estimate does not universally certify one-percent
continuum accuracy; continuum claims remain provisional until supported by refinement.

New wave EQ fits use economy QR on each selected support before positive least
squares. The greedy residual remains in the original integrand coordinates, and
final metadata records its original-space objective and support KKT conditions.
Actual decoder-integrand and near-dependent-support checks are stored under
`checks/nnls-probe02/`. This is an offline implementation choice; its local
timings are implementation probes. Previously completed direct fits are reused
unchanged with original source and rule hashes.

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
