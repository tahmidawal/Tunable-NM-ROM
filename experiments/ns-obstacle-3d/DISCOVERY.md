# Three-dimensional pipe–sphere discovery and proposed reference pilot

This is a design inventory, not a numerical-results report. No flow has been computed in this lane, no training is authorized, and all proposed acceptance thresholds must enter the coordinator's reviewed design before the corresponding run.

## Recovery and what can be reused

The worktree is `exp/2026-10-27-ns-obstacle-3d`, forked from consolidated `93f399faad67f94ea2b56fe25f13003143e64694`; its mirror base is `5a64899fdcfbeb6cfe5b3371af1a8d8a3d2b568f`. Namespace: `/cluster/tufts/paralab/tawal01/ns-obstacle-3d-20261027/`. The canonical root lab log remains the only session log.

Inspected inherited code at that exact fork:

| Component | Evidence and status | Reuse decision |
|---|---|---|
| `experiments/ns3d/ns3d_fom.py` | Periodic spectral Galerkin velocity solver with strict cutoff, Leray projection, CNAB2 and RK4. Velocity only; pressure eliminated. Its validation is for a periodic domain. | Reuse ideas for separate time-order controls, arrays/provenance and paired timing; do not use its periodic operator as the pipe FOM. |
| `experiments/ns3d-coordnet/coordnet.py` | Curl of a periodic Fourier-feature MLP vector potential. Analytical continuum divergence cancellation, autodiff derivatives, variable projection, QR/SVD ordering. Sampling also applies discrete spectral projection and mesh-dependent orthonormalization. | Reuse mathematical differentiation and ordering patterns with new geometry-aware basis/lift and weighted inner products. Revalidate all boundary traces and derivatives. Do not use the frozen periodic bank weights. |
| `experiments/ns3d-shift/shift_rom.py` | Solves translations in a co-moving frame because periodic operators commute with translations. Fourier tests and precomputed quadratic tensors. | Fixed pipe and obstacle break this invariance; shifts, tests and diffusion eigenvalue shortcut are ineligible. Tensor construction ideas may transfer after weak-form rebuilding. |
| `experiments/ns3d-coordnet/rom_mesh.py` | Same-allocation paired controls, disjointness checks, floor diagnostics, source hashes, stored repetitions, discrete derivatives. | Reuse workflow only. Its normalization, periodic norms and initial-field error denominator need replacement. |
| Earlier NS code evidence | Canonical 2026-10-01 entry retains the original representation/accuracy gate failures and the later user-directed acceptance amendment; lower-resolution bank equivalence did not pass as registered. | No historical speedup, boundary validity, or mesh-independence assertion transfers to this case. |

The latest Burgers conclusions also apply as warnings about inference: a fitted sampling failure does not establish bank superiority, projection floors do not identify a unique physical-error cause, and changing geometry alone does not create a competitive method.

## Primary-source interpretation

[Yin and Ong (2020)](https://doi.org/10.1063/5.0017349) directly studies a stationary sphere in a circular pipe. It distinguishes Reynolds numbers based on inlet mean speed, the undisturbed mean speed across the sphere's projected disk, and mean gap speed. Its wake patterns depend jointly on confinement and Reynolds number. The publisher abstract and an indexed excerpt of the accepted manuscript were accessible; direct full-manuscript retrieval failed. This is relevant physical evidence, but it does not certify the proposed low-Reynolds pilot or supply a threshold for it.

[Barber and Emerson (2001)](https://epubs.stfc.ac.uk/manifestation/255/dltr-2001001.pdf) provides a primary numerical low-Reynolds confined-sphere check in its continuum no-slip limit. Its Reynolds number uses the sphere radius and pipe mean speed. Its normalized drag is $C_D Re_a/12$, equivalent to our $C_D Re_D/24$. The relevant source row and domain are machine-recorded in `SPEC.proposed.json`. The fully developed inlet and vanishing downstream velocity-gradient condition provide a useful consistency diagnostic; its published precision and finite domain are not a rigorous uncertainty certificate for our case.

[Sheard and Ryan (2007)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/pressuredriven-flow-past-spheres-moving-in-a-circular-tube/8A2E452C6720959CC2C904D168605A09) uses a moving periodic array of spheres and a translating reference frame. It is not the fixed isolated-sphere problem. Its transition values and driving parameters must not be imported into this campaign.

The proposed production geometry below is therefore a **new project-defined case informed by primary literature**, with external checks rather than a claimed reproduction of an established benchmark.

## Physics and proposed fixed geometry

Use dimensionless coordinates scaled by the sphere diameter $D$ and velocities by the prescribed final inlet cross-sectional mean $U$. Time is scaled by $D/U$. Density is constant and set to unity; $p$ means kinematic pressure (physical pressure divided by density), scaled by $U^2$. Thus

$$
\partial_t\mathbf u+(\mathbf u\cdot\nabla)\mathbf u=-\nabla p+\nu\Delta\mathbf u,
\qquad \nabla\cdot\mathbf u=0,
\qquad \nu=Re_D^{-1}.
$$

The proposal is in `SPEC.proposed.json`: sphere diameter unity at the origin; pipe radius $2.5$; axial interval $[-10,20]$. The diameter blockage is $\beta=D/D_p=0.2$ and area blockage is $\beta^2=0.04$. These are different quantities and both must be labelled. The initial steady target is $Re_D=10$ with $U=1$, hence $\nu=0.1$. This is an economical starting hypothesis, not an asserted transition threshold or verified steady solution.

At the inlet prescribe

$$
\mathbf u_{\rm in}(y,z,t)=2U s(t)\left(1-\frac{y^2+z^2}{R_p^2}\right)\mathbf e_x.
$$

The peak is twice the mean. Pipe and sphere surfaces have all three velocity components zero. No symmetry plane constraints are imposed in the full three-dimensional calculation. An axisymmetric independent control may reduce verification cost, but that symmetry is a physical restriction and must be offered to strong FOM/POD comparators too.

The primary outlet is the **Laplacian natural condition**

$$
\nu\partial_n\mathbf u-p\mathbf n=0.
$$

This is the condition associated with $\nu\int\nabla\mathbf u:\nabla\mathbf v-\int p\nabla\cdot\mathbf v$. It is not the symmetric-stress traction-free condition. Its zero ambient pseudotraction fixes pressure level; do not also impose a mean-zero pressure constraint or pin an arbitrary pressure degree of freedom. Pressure regularization must be absent in the reference or explicitly tightened and shown negligible.

For the steady solve use a Stokes initial guess followed by bounded damped Newton continuation. Steady comparisons time direct steady FOM and ROM solves to the same output tolerance; they do not charge the FOM a long transient. The separate transient control starts at zero velocity with $s(t)=\sin^2(\pi t/(2T_r))$ until $T_r=2$, then $s=1$. Proposed short horizon is $T=10$, with outputs every $0.1$ and time-step ladder in the JSON. This verifies temporal treatment only and does not claim the flow has reached steady state at the endpoint.

Compute physical stress and force separately:

$$
\boldsymbol\sigma=-pI+\nu(\nabla\mathbf u+\nabla\mathbf u^T),
\qquad
\mathbf F=\int_{\Gamma_s}\boldsymbol\sigma\mathbf n_s\,dS,
\qquad
C_i=\frac{F_i}{\tfrac12 U^2(\pi D^2/4)}.
$$

Here $\mathbf n_s$ points from the solid sphere into fluid. If a mesh supplies the fluid-outward normal, reverse it. Save pressure and viscous force components. The transverse coefficients should approach zero in the symmetric steady case; use an absolute tolerance, not relative error against zero. Pressure drop uses area averages on two fixed planes common to all length variants, with the empty-pipe contribution reported separately. Subtracting an arbitrary mean for plotting does not change the imposed outlet gauge or license altering pressure-gradient/force calculations.

## Solver choice and dependency feasibility

Use the same NGSolve/Netgen implementation family as the 2D lane: body-fitted curved tetrahedra, a Taylor–Hood velocity/pressure pair, primary velocity degree three and pressure degree two, then independent mesh and polynomial-order refinement. The [official NGSolve example](https://docu.ngsolve.org/nightly/i-tutorials/unit-3.2-navierstokes/navierstokes.html) establishes the mixed weak form and available discretization API. Its tutorial settings are not a campaign specification; in particular do not copy a pressure-penalty term without analysis.

Local and cluster read-only package discovery found none of NGSolve, Netgen, Gmsh, meshio, FEniCS, Firedrake, scikit-fem, PETSc or MPI Python bindings in the prescribed venvs. The coordinator/2D discovery identified matching pinned cluster wheels. A future per-lane dependency target installed with `--no-deps` and an explicit `PYTHONPATH` can preserve the shared JAX installation. This lane has installed nothing.

The finite-element solves are CPU computations within the authorized GPU allocation; count the entire allocation in the campaign budget. The mandatory JAX GPU preflight proves GPU availability, not GPU acceleration of the finite-element solver. Report CPU model, thread counts, linear backend and memory separately. Compare FOM and ROM on the same allocated node, including CPU–GPU transfers and reconstruction in each matched output contract.

## Reference and feasibility gates proposed before results

The exact numerical proposal is stored in the JSON for review. For every refinement retain velocity, pressure, force, divergence, flux, boundary mismatch, solver history, iterations, memory and timings rather than just a success flag.

1. **Geometry:** exact pipe/sphere surfaces represented through curved elements; check signed Jacobians, boundary labels, surface normals, fluid volume, sphere area and inlet area against analytic values using refined integration. No points inside the obstacle. Isolate geometric-order error from field-order error.
2. **Empty pipe:** remove the obstacle as a separate control. Exact $\mathbf u=2U(1-r^2/R_p^2)\mathbf e_x$ and $p=8\nu U(x_{out}-x)/R_p^2$ satisfy the chosen outlet. Check velocity, pressure gradient, flux and integrated wall shear. A symmetric traction-free outlet would not reproduce this exact control.
3. **Manufactured flow:** before real training data, exercise all three components and pressure with a smooth divergence-free manufactured field and corresponding body force. Use exact nonzero boundary data on a separate manufactured-control configuration and verify expected spatial/temporal order; do not claim it validates the real no-slip obstacle by itself.
4. **Sphere convergence:** hold integration and nonlinear tolerance tight while refining the mesh; then refine field order on fixed geometry; refine curved geometry independently. Treat nonmonotone errors conservatively. Reference uncertainty is the observed finest-level envelope plus independently assessed temporal, integration, tolerance and domain effects; do not call it a theorem or assume Richardson asymptotics.
5. **Solver tolerance:** tighten both linear and nonlinear tolerances by a fixed factor and require output changes below one tenth of the respective reference gate. Record accepted nonlinear residuals and increments. A converged coefficient residual alone cannot certify field or drag accuracy.
6. **Length/outlet:** extend upstream alone and downstream alone using the JSON variants and the same local sphere mesh density. Compare fields on a common interior fluid region, all forces and fixed-plane pressure drops. Then assess a second physically defined outlet model on the longer domain. If the wake reaches the outlet or backflow persists, the original case is ineligible for ranking; do not silently stabilize or shorten the wake.
7. **Temporal:** the short ramped transient gets three time steps on a spatially qualified mesh. An observed temporal order claim needs its own acceptance interval and a regime where spatial/solver uncertainty is smaller. Endpoint-only agreement is insufficient; compare all scheduled fields/forces.
8. **Conservation:** report signed inlet/outlet flux and weak and strong divergence separately. Check physical momentum balance and the driven energy balance, including inlet/outlet momentum and power fluxes. In exact incompressible form,

$$
\frac{d}{dt}\frac12\int_\Omega|\mathbf u|^2
+2\nu\int_\Omega\epsilon(\mathbf u):\epsilon(\mathbf u)
+\frac12\int_{\partial\Omega}|\mathbf u|^2\mathbf u\cdot\mathbf n
=\int_{\partial\Omega}\mathbf u\cdot\boldsymbol\sigma\mathbf n.
$$

The natural outlet generally has nonzero **physical** traction, which belongs on the right. A decay-only periodic energy rule would be wrong here. Quantify discrete divergence defects when assessing this identity.

No result is rankable until all relevant field, force, conservation, sensitivity and reviewer gates pass. The literature low-Re drag row is an external check, not a substitute for these tests.

## ROM implications and bounded progression

A new lifting must satisfy inlet, pipe and sphere traces and incompressibility together. Multiplying velocity by a scalar wall envelope generally breaks divergence freedom. Taking a curl preserves divergence, but does not impose no slip unless the vector potential and appropriate derivatives satisfy surface constraints. Pressure is still needed for forces even with a divergence-free trial space. The first geometry-aware construction should therefore be chosen jointly with the ROM lane and tested against a stable mixed/supremizer POD control before any larger network search.

Use actual fluid-element integration and a separate sphere-surface rule. The first credible rule is mapped positive element quadrature; any compressed rule must pass operator, Jacobian, divergence/pressure and reached-state field/force refinement checks. Geometry, PDE solution and force errors remain separate. Dense sampling tensors scale as the number of points times trial width times component/derivative count; profile these explicitly before training and evaluate derivatives in chunks.

The coordinator's initial resource envelope is in the JSON. This lane's first approved smoke has a strict twenty-minute allocation, at most sixteen CPU threads and sixty-four GiB memory. It must report mesh generation, degrees of freedom, factorization/solve and derivative-evaluation memory before an expansion proposal. Stop rather than inflate the geometry or train a large bank if the reference cost/memory projection exceeds the remaining envelope. A later small representation study and full ROM/POD/FOM query are conditional on reference acceptance. A later unsteady wake campaign requires distinct source-backed selection and verified simulation evidence; smooth steady flow cannot support turbulent generalization.

## Glossary

**FOM:** full-order numerical solver. **ROM:** reduced-order solver. **POD:** snapshot-derived orthogonal basis. **Bank:** learned coordinate functions whose coefficients are solved online. **Prefix:** first columns of an ordered bank. **Lifting:** field that supplies nonzero boundary data before a homogeneous correction is added. **Taylor–Hood:** mixed velocity/pressure finite elements with compatible polynomial degrees. **Pseudotraction:** the natural outlet quantity for a Laplacian weak formulation; distinct from physical stress traction. **Pressure gauge:** convention fixing otherwise arbitrary pressure level. **Blockage:** sphere-to-pipe diameter ratio, or its square for area blockage. **Reynolds number:** inertia-to-viscosity ratio under the explicitly named speed and length. **Reference gate:** declared condition for admitting computed truth data. **Representation floor:** best field fit attainable inside a frozen trial space, with no claim of an attainable PDE trajectory. **Quadrature:** weighted numerical integration. **Reached state:** state actually visited by a numerical solve. **Supremizer:** velocity enrichment used to maintain a stable reduced velocity–pressure coupling. **Allocation hours:** reserved compute time including setup and failed attempts, even if only CPUs perform a phase.
