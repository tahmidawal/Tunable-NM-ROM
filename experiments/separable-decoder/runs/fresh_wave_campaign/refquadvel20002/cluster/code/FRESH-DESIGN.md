# Fresh two-dimensional wave head experiment

This document specifies a new benchmark. Numerical verification and learned results are pending; no old wave source, artifact, threshold, or conclusion supplies evidence.

## Equation and new spatial discretization

On $\Omega=[0,1]^2$, solve $u_t=v$ and $v_t=c^2\Delta u$. The requested fixed-wall reflection is homogeneous Dirichlet ($u=v=0$ on each wall), which reverses displacement sign. The absorbing case uses first-order Sommerfeld $v+c\partial_nu=0$ on all four faces. This absorber is exact for a normally outgoing plane wave and approximate for oblique waves. Its continuum reflection must be distinguished from numerical discretization error.

A grid with `n` intervals has spacing $h=1/n$. Each one-dimensional mass is $H=h\operatorname{diag}(1/2,1,\ldots,1,1/2)$, edge stiffness is $S=D^TD/h$ for nearest-neighbor incidence $D$, and endpoint selector is $B=\operatorname{diag}(1,0,\ldots,0,1)$. Use the tensor edge/weak discretization

$$M=H_x\otimes H_y,\quad K=c^2(S_x\otimes H_y+H_x\otimes S_y),\quad C=c(B_x\otimes H_y+H_x\otimes B_y),$$

$$M\dot v+Cv+Ku=0.$$

Absorber corners sum both face contributions; $C_{ii}/M_{ii}=4c/h$ at a corner. Dirichlet DOFs are eliminated, leaving zero damping. This is a trapezoidal weak/edge discretization, not a claim of exact quadrature for bilinear finite elements. Mixed/periodic axes are correctness controls only.

$$E=\tfrac12v^TMv+\tfrac12u^TKu,\qquad \dot E=-v^TCv.$$

Classical RK4 integrates the first-order system. Step sizes begin at $\Delta t\le0.2h/c_{\max}$, rounded to divide the observation interval. RK4 does not exactly conserve energy. An augmented state integrates $q'=v^TCv$ using the same RK stages to measure $E+q-E_0$. The absorbing constant-displacement nullspace is retained: $\mathbf1^T(Mv+Cu)$ is invariant, and zero energy does not imply zero displacement.

## Verification before training

`test_fresh_fom.py` independently assembles dense incidence/Kronecker matrices and uses a matrix exponential; it verifies both state components, exact semidiscrete energy derivative, absorber constant states/invariant, smooth initial compatibility, and detection of stiffness-sign, omitted-damping and dropped-initial-velocity mutations. Boundary/corner-concentrated states are included in the scientific verification.

`fresh_verify.py` records continuum standing modes with nonzero initial velocity; periodic traveling modes; oblique manufactured waves with separate signed face loads (corners sum); analytic normal-incidence compact-pulse wall reflection/absorber exit; a fresh larger-domain periodic FFT reference for two-dimensional absorber behavior; and space/time refinements. Actual minimum-width localized family members receive their own mesh refinement through the full training horizon. Analytic standing-wave checks use intervals 16, 32, 64, 128. Actual-family checks use 32, 64, 128. Time studies hold space fixed and halve step sizes; physical-space studies make temporal error negligible.

New truth acceptance: independent operator errors at roundoff; fixed-space RK4 order at least 3.5 before roundoff; standing/manufactured space order at least 1.7 in the joint mass norm; every deliberate mutation must be detected; no nonfinite or unstable boundary control; exact invariants agree at roundoff. Approximate absorber versus large-domain mismatch is reported as absorber-model error and does not silently become a numerical gate. Family refinement determines a mesh uncertainty budget before neural training. The primary candidate is 32 intervals (33 nodes), promoted to 64 if needed; no new learned result is inspected to choose it.

## Fresh data declaration

Independent NumPy PCG64 seeds: training 690601, validation 690602, final 690603. Counts are 64, 16, 16 trajectories per boundary configuration. The two BCs share parameter rows for a controlled physics comparison. The final cohort remains unopened during development. No model takes these parameter descriptors or time as an input.

Initial displacement is a product of smooth compact bumps $u_0=A\,b((x-x_c)/s_x)b((y-y_c)/s_y)$, with $b(r)=\exp(1-1/(1-r^2))$ for $|r|<1$ and zero otherwise. Half-widths are uniform in $[0.27,0.34]$, centers are uniform in $[s+0.025,1-s-0.025]$, amplitude in $[0.7,1.3]$, and speed in $[0.85,1.15]$. This keeps every initial boundary derivative exactly zero. Velocity is $v_0=-c(d_x\partial_xu_0+d_y\partial_yu_0)$ with $d_x,d_y$ uniform in $[-0.5,0.5]$. Every fourth training/validation row has zero initial velocity; independent verification additionally uses nonzero-mean compact velocity. All parameters, membership and initial compatibility defects are persisted.

Observe displacement and physical velocity every 0.05 through time 2.4, long enough for several fixed-wall interactions and for absorbing exit. Data are regenerated from seed inside each scientific GPU job. All calculations are f64 with highest matmul precision.

## Learned models and outcomes to freeze after truth verification

A fresh coordinate network produces a spatial bank of rank 64, independently trained for each BC from that BC's training trajectories. No POD spatial bank is used for the claimed neural method. Freeze the bank and use weighted QR to change coordinates without changing its span. Latent dimension is 16. Smooth weak test dimension is 64, comfortably larger than the latent dimension. All wave operators are linear, so their projection on the learned bank is precomputed exactly; online residuals require no spatial sampling or quadrature fit.

Compare an affine-skip SiLU MLP and affine-plus-unique-quadratic-products head, each with displacement-only and displacement-plus-velocity-tangent objectives. The velocity term is the least-squares distance of $v$ from $GJ_h(z)\dot z$. Each arm has the identical frozen bank, training observations, affine initialization, latent dimension, normalization and update budget. Optimizer repeats use seeds 691200 and 691201. Weight reuse across PDEs is excluded. A fresh POD model is a labeled baseline only.

The concrete bank/head optimizer settings, weak dynamics/time integrator and neural acceptance budget are finalized in a tracked config after reference verification and before training. Acceptance must include reconstruction, tangent, actual rollout displacement/velocity/error-state energy, energy balance, phase, outliers and solver completion. Accuracy is never selected by the best time-step outcome; predetermined step refinements remain separate results. Incomplete or nonfinite solves are failures. No speed claim is in this first experiment.

## Glossary

- **DOF:** a stored grid value; fixed-wall boundary values are removed from the unknown vector.
- **Weak/edge discretization:** spatial equations derived from weighted integrals and differences across neighboring nodes.
- **Mass, stiffness, damping:** matrices weighting velocity, spatial restoring force, and absorption.
- **RK4:** a four-stage explicit time update with fourth-order accuracy at fixed spatial grid.
- **Sommerfeld:** a local approximate radiation boundary condition.
- **Refinement/order:** decreasing a step size and measuring the rate of error reduction.
- **Manufactured solution:** an analytic field whose known boundary/interior forcing makes it an exact solution.
- **Bank/head:** learned spatial features and the latent-dependent coefficients multiplying them.
- **Latent dimension/rank:** the number of internal manifold coordinates / independent bank features.
- **POD:** an independent linear subspace baseline fitted from fresh training data.
- **QR:** a coordinate change that makes the bank orthonormal under the mass-weighted inner product.
- **Tangent consistency:** whether a physical velocity lies in the decoder's local derivative directions.
- **Final cohort:** unused trajectories reserved for later evaluation after model choices are frozen.

## Pre-training revision after failed reference verification

The first committed reference trial failed the unchanged compact-plane spatial-order checks. Independent sine-transform propagation of the saved reflected fields isolates a large spatial-dispersion error while agreeing closely with the semidiscrete time evolution. No bank or head was trained and the failed original family remains recorded.

Before training, revise the localized family to a Gaussian core multiplied by the same smooth compact taper. The two Gaussian standard deviations are uniform in $[0.12,0.16]$; support half-widths are independently uniform in $[0.36,0.42]$. Centers remain in $[s+0.025,1-s-0.025]$, so the new center coverage is explicitly narrower than the original family. Amplitude, speed, velocity direction, counts, seeds and horizon remain unchanged. The field and all derivatives remain exactly zero at the walls initially. This revision suppresses high-frequency edge content; its suitability is itself subject to new verification.

`fresh_verify_refined.py` retains and extends the original plane controls to intervals 128, 256, 512 and the original narrow reflected-family control to intervals 128, 256. It compares RK4 with independently assembled sine-transform modal propagation using both semidiscrete and continuum frequencies. The new Gaussian-core family receives intervals 32, 64, 128, 256, same-stage energy balance/invariant/finiteness checks, and independent enlarged-domain FFT resolution and box-size checks. A semidiscrete time error below $2\times10^{-4}$ is required for these sharper controls; this is distinct from the fixed-space fourth-order RK4 test. The physical-reference target is at most 2% energy-state uncertainty over the full horizon; a second-order coarse/fine difference below 1.5% is a necessary estimate for the coarse-grid target, subject to observed refinement. If intervals 128 do not meet it, further mesh verification is required before learning.

The coefficient heads and the weak manifold dynamics are being implemented independently while these checks run. The fresh online ODE minimizes the weak acceleration residual in the orthonormal learned-bank test coordinates:

$$\ddot z=\arg\min_a\|J_h(z)a+H_h(z)[\dot z,\dot z]+D_rJ_h(z)\dot z+K_rh(z)\|_2^2.$$

Thus physical velocity is always $GJ_h(z)\dot z$. QR solves the overdetermined weak system with a stage-wise singular-value rank guard and no hidden ridge. Classical RK4 integrates $(z,\dot z)$ and same-stage boundary power. It has no exact discrete energy-conservation claim. The curved-map analytic acceleration and energy identity, independent DOP853 propagation, and a nonorthogonal linear-head matrix-exponential reference are separate component tests.

## Frozen bounded learning and evaluation protocol

Reference mesh selection is complete before training: use intervals 256. The initial compact-only trial and the subsequent intervals-128 Gaussian trial retain their failed gates. `campaign-config.json` records the accepted new verification job, source/result hashes, measured evidence and independent-review scope. Exact semidiscrete versus refined continuum sine propagation and measured RK4 error support the reflective uncertainty budget; the absorbing estimate is conditional on observed contraction in the declared empirical sample, with boundary-model error assessed separately.

The bank has 64 outputs and two SiLU hidden layers of width 128, with a fixed coordinate-only Fourier lift. Train it by differentiating the mass-QR projection error of normalized training displacement and velocity, for 6000 Adam updates with minibatches of 96 and cosine-decayed learning rate from 0.001 to one tenth of that value. Reject nonfinite or rank-deficient raw banks before QR; the minimum raw singular-value ratio is $10^{-8}$. Check preservation of the learned span, mass orthogonality, exact transformed stiffness, positive semidefinite frozen operators, and decoded-field/operator-table parity.

The frozen bank feeds a latent dimension of 16. A PCA of training *coefficient vectors* provides the common affine head initialization and dimensionless standardized latent scores; the spatial bank remains the trained coordinate network. The MLP has two SiLU layers of width 128 with a zero-initialized nonlinear output and affine skip. The quadratic uses one undoubled product per latent pair. Compare displacement-only and unit-weight velocity-tangent objectives for each architecture, with a common $10^{-6}$ mean-square code penalty. Each arm receives 10000 Adam updates, minibatches of 64, head learning rate 0.001 and code learning rate 0.003, both cosine-decayed to one tenth. Use the final fixed-budget checkpoint, without validation checkpoint selection. Optimizer seeds are 691200 and 691201.

Every unseen snapshot receives eight initial latent guesses: its common affine fit, zero, and six evenly spaced rows of that arm's trained training-code table. Run independent 400- and 800-iteration LM budgets from the exact same eight guesses; save all initial guesses, fitted codes, objectives, gradients, projected-residual stationarity, ranks, finiteness, damping and stopping reasons. The longer budget's best finite result is the common reporting rule. The normalized raw-gradient tolerance is $10^{-7}$ and the projected-residual stationarity tolerance is $10^{-6}$, with a separate near-zero residual tolerance. Report nonstationary and rank-deficient cases as failures; local multistart stationarity is not a global optimum certificate.

For initial-state energy $E_0=E(u_{\mathrm{FOM}}(0),v_{\mathrm{FOM}}(0))$, the physical error scales are

$$\epsilon_u(t)=\frac{\|u_{\mathrm{ROM}}(t)-u_{\mathrm{FOM}}(t)\|_M}{\|u_{\mathrm{FOM}}(0)\|_M},\qquad
\epsilon_v(t)=\frac{\|v_{\mathrm{ROM}}(t)-v_{\mathrm{FOM}}(t)\|_M}{\sqrt{2E_0}},$$

$$\epsilon_E(t)=\sqrt{\frac{E(u_{\mathrm{ROM}}(t)-u_{\mathrm{FOM}}(t),v_{\mathrm{ROM}}(t)-v_{\mathrm{FOM}}(t))}{E_0}}.$$

The provisional engineering target is that all 16 validation trajectories complete and their time-maximum values of all three errors are at most 0.10. Persist means, medians, worst cases and outlier counts; any finite-only summary is labeled conditional and never makes a failed trajectory disappear. The predeclared RK4 steps are 0.005, 0.0025 and 0.00125. The middle step is primary, with no selection by best accuracy; the finest two must additionally agree within 0.01 on all three trajectory scales. Every generated FOM trajectory records same-stage flux/energy balance, absorbing invariant, finiteness and initial compact-support margin. Every RK stage checks latent rank and finite power; failed trajectories freeze and retain their failure status. Nonfinite decoded energy also fails the trajectory.

Physical velocity is $GJ_h(z)\dot z$. Reflective modal phases use the semidiscrete frequencies; absorbing sine projections are labeled diagnostics, not eigenmodes. Vanishing predicted modes have undefined phase and explicit flags; phase drift is reported over contiguous valid segments without alignment. The wall-strip energy peak is labeled as a peak-time proxy, not an exact wavefront arrival. Absorbing mean displacement is assessed separately from energy.

Fresh randomized POD of normalized training displacement and velocity is an explicitly approximate linear baseline with fixed oversampling and power iterations. Both latent-sized and bank-sized POD baselines, and the unrestricted learned-bank linear dynamics, evolve with independent matrix exponentials. Initial ROM fits in this accuracy experiment use full-field initial-state projections; a grid-independent cold start remains required before any online-cost claim. The final-test generator is never called during this campaign.

## Additional glossary

- **PCA coefficient initialization:** a linear statistical initialization inside the already learned spatial span, shared by all heads.
- **Variable projection:** solve the linear coefficients exactly while optimizing the spatial network.
- **Projected stationarity:** the remaining residual component along the decoder's available derivative directions.
- **Rank ratio:** smallest divided by largest singular value, used to detect collapsed derivative directions or bank features.
- **Conditional error estimate:** an estimate assuming the measured refinement trend continues, not a proven upper bound.
- **Time-maximum error:** the largest error over the stored trajectory times, normalized using its initial physical state.
- **Phase / vanished mode:** an oscillation angle / an amplitude too small for that angle to be meaningful.
- **Matrix exponential:** an independent exact time propagator, up to numerical roundoff, for a linear finite-dimensional system.
