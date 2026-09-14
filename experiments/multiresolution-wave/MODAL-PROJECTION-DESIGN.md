# Linear bank evolution with nonlinear output reconstruction

Predeclared reflective-wave accuracy follow-up. This method changes the internal evolution from 32 nonlinear configuration coordinates to 64 linear bank coefficients. It must not be presented as a parity acceleration of the original latent dynamics. The retained 32-coordinate nonlinear decoder reconstructs every returned displacement; a free linear-bank control is timed beside it.

## Evolution and output projection

Project the supplied initial displacement and velocity once into the frozen orthonormal bank, obtaining $a_0,b_0$. With the exact mesh-specific weak stiffness $K=V\Lambda V^T$, propagate these coefficients analytically using sine and cosine at each requested time. The eigensystem is mesh-only offline preparation; query wave speed, projections and propagation are charged. No trajectory reference or true field after the supplied initial state enters the online method.

These internal initial moments are the raw bank projections of the supplied fields. They differ from the baseline's fitted manifold displacement and tangent-projected initial velocity. The hybrid subsequently reconstructs its returned initial fields with the same nonlinear projection used at other output times. The actual returned fields at time zero are included in every physical error and initialization diagnostic; internal moment matching must not be described as matched fitted initial states.

For each propagated coefficient target $a(t)$, minimize

$$F(z,t)=\frac12[h(z)-a(t)]^TW[h(z)-a(t)].$$

The two frozen output metrics are $W=I$ and $W=K/(\operatorname{tr}(K)/64)$. The implementation applies the triangular square root of $W$ to the head and targets and uses the original trust-region fit. This is a coefficient-coordinate transformation, not retraining. All eight original types of start are charged at every output: affine, zero and the six fixed training codes. Select by the weighted objective; never by truth error. Every selected fit must satisfy the original gradient, stationarity and rank criteria.

At a stationary fit, $J^TW(h-a)=0$. Differentiate this condition to obtain the physical latent velocity:

$$\left[J^TWJ+\sum_i[W(h-a)]_i\nabla^2h_i\right]w=J^TW\dot a,$$

and return $u=Gh(z)$, $v=GJ(z)w$. The full projection Hessian is used without a hidden ridge. Require finite positive Hessians with smallest-to-largest eigenvalue ratio above $10^{-10}$ and scaled linear-solve backward residual below $10^{-10}$. A pseudoinverse tangent projection alone would generally not differentiate the fitted displacement when the normal residual is nonzero.

## Bounded screen

`projection-screen-config.json` fixes the two opened reflective cases at 64 intervals and three repetitions. Compare the original latent RK4, Cholesky latent RK4 at the selected timestep and a separately labeled larger timestep, free linear bank, both nonlinear reconstruction metrics, named iterative FOM timestep/tolerance combinations, and direct DST. The larger latent timestep must satisfy the original temporal/physical degradation checks; it is not a new geometry-parity result.

For each nonlinear reconstruction, independently refit neighboring targets at $t\pm\delta$ from the selected code on that branch, for both declared small offsets, and compare centered differences of the returned coefficients with the reported physical velocity. These diagnostic fits occur after the primary timer and cannot improve the timed outputs. Require both neighboring fits stationary and the discrepancy below 0.1% on the initial velocity scale. A branch switch or local-minimum failure remains a failed diagnostic; stationarity is not a global optimum certificate.

Report displacement, physical velocity, energy-state and phase errors; all-state 5% eligibility; initial-fit distortion; physical energy drift; every fit's objective/gradient/iteration count; positive-Hessian ratios; kinematic differences; all timing repetitions and outliers. Both changed-method dimensionality and free-bank accuracy/cost remain visible. If no nonlinear reconstruction meets the physical target, retain it as an accuracy diagnostic and proceed only with an explicit head-training experiment, not a hidden relaxation of the target.

After screening, freeze the retained choices and confirm at 64, 256 and 1024 intervals on both the opened cohort and a separately seeded fresh development cohort. Keep the final paper cohort sealed. Iterative FOMs can also increase timestep at declared tolerances; speed factors use the fastest tested all-state-passing combination, beside the original comparator and direct DST.

## Glossary

- **Linear bank:** 64 learned spatial functions with freely evolving coefficients.
- **Nonlinear output reconstruction:** stationary fitting of the 32-coordinate frozen decoder to each propagated target.
- **Weighted projection:** fitting a target using the specified spatial metric.
- **Hessian:** matrix of second derivatives of the fitting objective.
- **Physical velocity:** time derivative of the reconstructed displacement on its selected stationary branch.
- **Kinematic check:** numerical test that displacement derivatives match reported velocity.
- **Phase / energy-state:** oscillation timing / combined displacement-gradient and velocity error.
- **CG / DST:** iterative conjugate-gradient midpoint and direct discrete-sine-transform full-order controls.
- **Development cohort:** cases used for screening and further research; distinct from the sealed final paper cohort.
