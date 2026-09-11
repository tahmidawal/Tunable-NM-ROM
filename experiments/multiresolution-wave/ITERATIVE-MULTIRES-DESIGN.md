# Fresh-wave multiresolution comparison with iterative FOM

Predeclared development comparison of the frozen post-reset separable nonlinear model with a compiled iterative full-order wave solver. No pre-reset wave evidence, retraining, final cohort, or matched-accuracy speedup is assumed.

The configuration JSON fixes both development cases, both boundary conditions, meshes, repetition count, output times, time steps and CG tolerances. The primary model is the previously retained first-seed MLP32 head in its learned spatial bank. Its initial fitting and RK4 evolution are unchanged. The original direct reflective / explicit absorbing full solvers remain diagnostics.

For the verified fresh mass, stiffness and boundary damping, the full equation is

$$M\ddot u+cD\dot u+c^2Ku=0.$$

The new implicit midpoint method uses $u_{j+1}=u_j+\Delta t(v_j+v_{j+1})/2$. With $x=M^{1/2}v$, $q=M^{1/2}u$, $L=M^{-1/2}KM^{-1/2}$ and $C=M^{-1/2}DM^{-1/2}$, its solve is

$$[I+\Delta t cC/2+\Delta t^2c^2L/4]x_{j+1}=[I-\Delta t cC/2-\Delta t^2c^2L/4]x_j-\Delta t c^2Lq_j.$$

The matrix is symmetric positive definite even with trapezoid endpoint masses and absorbing damping. Counted, unpreconditioned CG starts from the previous velocity; the entire solve and rollout are compiled. Every step retains iterations, recursive relative residual and a charged independently recalculated true relative residual. This is an older-style iterative baseline using fresh mathematics, not an old wave replay.

Both methods use the same physical time grid and outputs; ROM uses RK4 and FOM uses implicit midpoint, so temporal error need not match. Independent dense small-grid stepping checks the implementation and matrix symmetry, and comparison with the dense exponential checks second-order refinement. Reflective full fields receive an independent discrete modal check. Absorbing scoring uses a fine RK4 reference with both members of its temporal refinement pair archived. ROM half-step trajectories remain accuracy-only controls.

The timer starts with supplied full displacement, velocity and wave speed on the GPU, charges projection, every cold fit, dynamics and both full fields at all requested outputs, and ends only after completion. Output transfer is retained separately; that is not a host-to-host query. Mesh-only assembly and compilation are offline. Every timed output is hashed and scored; byte-identical repetitions share a field artifact. Burn-in precedes each timed invocation; order reverses across repetitions. All timings and outliers remain recorded.

Accuracy includes displacement, velocity and energy-state errors, each with both fixed initial normalization and current-reference normalization. Vanishing current reference norms remain flagged; they are not hidden by substituting another metric. Numerical completion, rank, initial stationarity, CG convergence, time refinement, and physical accuracy are distinct gates. The cohort supports development observations against same-grid semidiscrete truth, not continuum or broad generalization claims.

## Glossary

- **CG:** conjugate gradient, the iterative method for the symmetric positive definite full system.
- **FOM / ROM:** full-order / reduced-order model.
- **MLP32 / bank:** frozen nonlinear coefficient map with its recorded latent dimension / frozen learned spatial functions.
- **Intervals:** number of spatial cells along each coordinate axis.
- **Implicit midpoint:** second-order time integration that solves a coupled linear wave update.
- **RK4:** explicit fourth-order Runge–Kutta time integration.
- **Current / initial normalization:** division by the reference norm at that time / a fixed physical initial scale.
- **Energy-state:** combined displacement-gradient and physical-velocity norm.
- **Refinement:** changing the time step and comparing resulting fields; an empirical check, not a mathematical error bound.
- **Device query:** supplied GPU fields through all requested GPU outputs.
- **Development cases:** already exposed inputs; not independent final evaluation.
