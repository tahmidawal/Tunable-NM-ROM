# Current NM-ROM for three-dimensional Poisson

Prospective development protocol for the approved overnight paper campaign. No numerical result exists at registration. This lane starts from the corrected Poisson linear-control branch and imports independently exercised three-dimensional learned-bank primitives from the contemporaneous heat lane, with exact source hashes retained.

## Equation, information and cohorts

Solve $-\Delta u=f$ on $[0,1]^3$ with homogeneous Dirichlet boundary conditions. There are $N$ intervals and $(N-1)^3$ interior unknowns. The full-order discrete reference is the exact separable inverse of the seven-point negative Laplacian using an orthonormal type-I discrete sine transform (DST). The forcing is a positive, boundary-masked Gaussian with a continuously varying three-dimensional center, width and amplitude. Every model receives only the supplied nodal forcing field; no method receives the generator descriptors as an online shortcut.

The initial configuration fixes independently seeded training and validation draws. Final seed 920499 is reserved and will not be instantiated until validation selection freezes. The principal mesh is 32 intervals; later 64-interval transfer is a separate bounded continuation. Accuracy is relative Euclidean field error, equivalent to relative discrete volume-weighted $L^2$ on this uniform mesh. No reconstruction or solver outcome changes the cohort or accuracy thresholds.

## Reference checks and physical interpretation

Before training, check the DST against SciPy, the discrete operator against its spectral eigenvalues and an independently assembled Kronecker sparse matrix, and a manufactured sine solution against independent sparse direct inversion. Compare a continuous sine forcing under mesh refinement to establish second-order finite-difference convergence. For validation inputs, retain finer continuum-spectral solutions at 64 and 128 intervals and their restricted differences. The physical-reference budget is $10^{-4}$ relative error; a failed empirical budget makes physical-error statements provisional, while exact same-grid errors remain usable. The development target is 1% worst same-grid error; the separate physical target is 2% subject to reference qualification. These targets do not imply that the model will pass.

## Learned model and correction variable

Train a coordinate Fourier-feature multilayer network as a spatial bank, jointly with free training coefficients. Its last hidden width is at least its output rank, so no width bottleneck silently caps the bank rank. Whiten the bank by a checked full-rank thin QR. Train nonlinear coefficient heads with latent dimensions 8 and 16, jointly fitting training latent codes. The bank is learned from fields, not substituted with POD. Record full training curves, partial checkpoints, representation errors and whether training has demonstrably plateaued; a finite step budget is not convergence.

Form nested correction directions from the training reconstruction residuals in the bank's orthonormal field coordinates. For each frozen head evaluate $q\in\{0,8,16,32\}$ with the principal smooth test count fixed at $M=256$. The unknown field is $u=G[h(z)+C_q y]$. Test the residual against sine modes divided by their discrete Laplacian eigenvalues, so the weak least-squares problem is $\min_{z,y}\|B[h(z)+C_qy]-b(f)\|_2$. Eliminate $y$ through an exact thin QR of $BC_q$ and optimize only $z$. Cold initialization chooses training latent codes by distance in the supplied forcing's weak solution moments, followed by multistart Levenberg--Marquardt. Record actual full augmented gradient norms, iteration counts and reasons; no stalled solve is called stationary. Reorthogonalize correction images at every query mesh.

The full-rank endpoint is the direct free-bank QR solve, without redundant nonlinear variables. Also report the unrestricted bank's field projection and multistart best-found augmented reconstructions as untimed diagnostics, distinctly from actual PDE solves. A candidate tunability claim requires a nested validation curve with no greater than 5% relative error regressions between adjacent retained rungs, at least a factor of two worst-error span, and a measured cost span. Failures remain explicit, and validation selections require an untouched final cohort for confirmation.

## Quadrature and controls

Fit nonnegative quadrature weights on decoder-output snapshots times the smooth test modes, never residual snapshots. Use at most $m=4M$ candidate grid nodes; retain all nonzero weights and measure held-out forcing-moment error against the dense weak projection. This is sampled integration of the weak residual, not strong-form random collocation. A maximum relative moment discrepancy above 1% leaves the quadrature path uncertified. The dense path remains an explicitly grid-dependent diagnostic. Refit whenever $N$ or $M$ changes.

Mandatory same-data controls are POD at latent-matched and correction-augmented ranks, the free learned-bank weak QR endpoint, a learned-bank Galerkin solve, and the exact DST full-order solver. Construct POD from the identical training solutions. Include a coarse DST control only with its physical error and interpolation cost retained. FNO3D and U-Net3D are the next operator tranche, with no generator descriptors and the same forcing/solution pairs, f64 arithmetic and validation selection. DeepONet and Transolver remain explicit operator backlog until implemented and evaluated.

## Measurement, provenance and first tranche

All arrays and model internals are float64 with highest matrix-multiply precision. Real runs require GPU preflight, their own paralab attempt directory, recorded GPU/job/source/configuration/seed hashes, and regenerated data. The first single-GPU job requests no more than two hours. Save training checkpoints before evaluation so a later failure cannot erase learning progress.

Time complete cold queries in one allocation: upload supplied forcing, weak projection or full solve, latent fitting where applicable, dense readout and output transfer. Preserve synchronized device time and total host time separately. Warm compiled methods and burn the GPU before each timed invocation, randomize method order, retain repetition arrays and fields from those same invocations, and report medians, worst errors, nonfinite outputs, stationarity failures and timing outliers. Offline training, setup and quadrature costs are separate. No cross-job wall-clock ratio or unearned neural advantage is permitted.

The owner independently recomputes saved field errors and verifies representative outputs, then checksum-collects the exact attempt and removes it only after verification. The root coordinator owns the canonical lab log. No final-cohort opening, worktree merge or public-paper claim happens automatically from a successful pilot.

## Glossary

- NM-ROM: a reduced numerical solver whose field coefficients depend on a smaller neural latent state.
- Bank/head: learned spatial functions/neural map from latent variables to bank coefficients.
- $N$, $R$, $K$, $q$, $M$, $m$: intervals per axis, bank rank, latent dimension, correction rank, weak test count, and quadrature candidate count.
- DST: discrete sine transform, which diagonalizes this rectangular Dirichlet Laplacian.
- POD: the best linear training-snapshot subspace under a squared field-error objective.
- QR: an orthonormal matrix factorization used for stable projection and direct linear solves.
- Galerkin/weak: residuals tested against reduced basis functions/declared smooth functions.
- NNLS: nonnegative least squares, used to fit integration weights.
- Same-grid/physical error: error against the exact stated discrete solver/a qualified finer continuum reference.
- Stationarity: a measured sufficiently small objective gradient, separate from field accuracy.
- Validation/final: development selection cases/independent cases kept unopened until selection freezes.
- FNO/U-Net/DeepONet/Transolver: Fourier, convolutional, branch--trunk and transformer learned solution-map baseline families.
