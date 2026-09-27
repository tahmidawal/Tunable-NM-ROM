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
