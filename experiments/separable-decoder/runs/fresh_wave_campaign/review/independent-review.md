# Fresh wave experiment: independent design and source review

Review of the new implementation only. Numerical acceptance remains conditional on the
fresh scientific verification and result audit; no historical wave evidence is used.

## Reference discretization

Root read the initial `fresh_fom.py`, `test_fresh_fom.py` and `fresh_verify.py` from the
approved wave worktree. A separate numerical reviewer independently derived the tensor
mass, edge stiffness and boundary damping and checked endpoint and corner weights.
The spatial scheme is a trapezoidal weak/edge discretization; its name was corrected
before scientific execution because ordinary exact Q1 stiffness uses a consistent
transverse mass instead.

The first-order wave system needs both displacement and initial velocity. This is also
set out in J. Schöberl's [finite-element wave derivation](https://jschoeberl.github.io/iFEM/timedependent/intro/waveequation.html).
For the fresh scheme, the semidiscrete mass, stiffness and damping obey the independently
derived identity

$$\frac{d}{dt}\left(\frac12v^TMv+\frac12u^TKu\right)=-v^TCv.$$

RK4 approximates this balance; it does not preserve the energy exactly. Outgoing work
must be integrated at the same RK stages and temporal error must be measured. The
absorber's constant-displacement nullspace is retained, and the invariant

$$I=\mathbf1^T(Mv+Cu)$$

provides another independent check. Mean displacement error remains necessary even when
energy becomes small. The weak radiation-boundary construction is consistent with the
independent [ETH finite-element absorbing-wave derivation](https://people.math.ethz.ch/~grsam/NUMPDEFL/Parts/NPDEFL_Problems_solutions_9-4.pdf);
the new code is not copied from that reference or from old repository wave modules.

The review requires exact/independent standing and traveling solutions, oblique forced
boundary signs and both corner contributions, a signed reflected pulse, initial-velocity
mutations, small-grid matrix-exponential comparison, and actual-family mesh refinement.
Absorber-versus-open-domain discrepancy contains physical boundary-model reflection and
cannot be interpreted as only discretization error. The enlarged Fourier reference must
have its own mesh/domain checks before it is used as certified open-domain evidence.

## Curvature-aware weak reduced dynamics

An independent ROM reviewer and root derived the fresh continuous system. In mass-QR
coordinates for the newly learned spatial bank, let $w=\dot z$, $J=J_h(z)$ and $H$ be
the head's second derivative. Solve the overdetermined weak-acceleration problem

$$a=\arg\min_a\|Ja+H[w,w]+D_rJw+K_rh(z)\|_2^2,$$

then integrate $\dot z=w$, $\dot w=a$. The weak bank dimension must comfortably exceed
the latent dimension. Use a rank-revealing solve without a hidden ridge. At full rank,
the normal equations imply the correct continuous reduced-energy balance. Every RK
stage must reevaluate the head, Jacobian and curvature, holding $w$ constant inside the
spatial-direction Hessian-vector calculation. Finite differences of decoded displacement
are not the primary velocity; physical velocity is $GJ_h(z)w$.

The review rejected two shortcuts before implementation: unconstrained fitting of next
position and velocity can break their kinematic relationship; an endpoint momentum
least-squares residual does not automatically inherit second-order midpoint accuracy.
The chosen continuous equation avoids both ambiguities. Explicit RK4 still needs an
independent nonlinear stability/refinement assessment; the FOM CFL is not sufficient.

Independent tests should include a linear head against direct matrix evolution and a
curved polynomial head against its analytic acceleration and a separate ODE integrator.
Dropping curvature must be a detectable mutation. Rank loss and nonfinite evolution are
failed trajectories. Report displacement, tangent-lifted velocity, error-state energy,
phase, mean displacement and balance separately, with all attempted trajectories kept.

## Review provenance and status

Implementation owner: `fresh_wave_impl`, sole writer of the approved wave worktree.
Numerical reviewer: `fresh_wave_review`, read-only independent derivation and source review.
ROM reviewer: `quadratic_head`, read-only fresh derivation; no old wave source was used.
Root owns this review record, cluster staging and independent result audit in the repair
tree. Any scientific values and acceptance decisions will be generated from the new raw
run outputs, not transcribed into this design review.

## Glossary

- **Weak/edge scheme:** equations weighted over space, with spatial derivatives represented by edge differences.
- **Q1:** bilinear finite-element basis functions on rectangles.
- **Mass / stiffness / damping:** weights on acceleration, restoring force and absorption.
- **RK4 / CFL:** the four-stage time integrator and a mesh-based time-step restriction.
- **Semidiscrete:** space has been discretized while time remains continuous.
- **Invariant / nullspace:** a quantity that remains constant and a field direction with zero restoring force.
- **Mass-QR / learned bank:** an orthonormal coordinate change and the trained spatial functions whose span it preserves.
- **Jacobian / curvature:** first and second derivatives of the coefficient head.
- **Rank-revealing / ridge:** a solve exposing lost independent directions and an added regularization term.
- **Tangent lift / error-state energy:** the velocity induced by latent motion and the physical energy norm of the prediction error.
