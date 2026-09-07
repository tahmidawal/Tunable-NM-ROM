# Burgers 2D frozen-network multiresolution pilot

This is an approved development pilot for the current coordinate-separable NM-ROM versus efficient FOM solvers. Scientific results are pending. The checkpoint and family are inherited from the consolidated branch; the validation cohort is new and the final cohort remains unopened.

The scalar PDE is

$$u_t+u(u_x+u_y)=\nu\Delta u,$$

on the unit square with zero Dirichlet walls. Data retain the incumbent single-Gaussian distribution, including its boundary clipping. Physical descriptors generate initial fields only and never enter the reduced model. `L` denotes intervals per axis, with $(L+1)^2$ total nodes and $(L-1)^2$ interior unknowns. The selected dense-mid checkpoint was trained with the legacy node count; the exact conversion and source hash are emitted by the driver.

The first study freezes both neural tracks and latent/bank dimensions, rebuilds the exact weak linear terms at each mesh, and refits nonnegative quadrature from decoder-output advection snapshots. The online weak nonlinear term retains the FOM's sign-dependent upwind stencil at its quadrature nodes, including on negative decoded states. The positive-field tensor is excluded. Low-frequency sine tests have a test count four times the latent dimension and the quadrature budget is four times the test count.

The FOM uses backward Euler with tolerance-adaptive Newton and matrix-free BiCGStab. Its exact discrete Helmholtz preconditioner applies orthonormal FFT sine transforms. This avoids the dense sine-matrix multiplication used in the inherited baseline. Component checks compare the transform to explicit sine matrices, invert the discrete Helmholtz operator, check mixed-sign Newton against a dense Jacobian solve, check the weak diffusion identity, and validate the small-state Gauss-Jordan solve against LU. FOM tolerance failures remain visible per timed invocation.

The input/output contract starts with a host-resident dense initial field on the requested grid and a viscosity scalar, and ends with six host-resident dense fields at fixed physical output times. Every timed call includes input transfer, input handling, cold latent fitting where needed, autonomous evolution, dense decoding/interpolation, and output transfer. FOM candidates may solve on a coarser nested mesh and interpolate onto the requested output grid. Cold initialization uses a bounded regular set of input values, followed by a local fit in latent coordinates. No learned Gaussian-descriptor encoder or POD replacement is introduced.

The common observation grid is the smaller requested mesh. The refined reference is restricted exactly onto its nested nodes. Spatial and temporal refinement differences are saved separately; their sum is a conservative diagnostic estimate, not a rigorous continuum error bound. A target is unresolved when that estimate exceeds one tenth of the requested error target. The primary error is the largest observation-time field discrepancy normalized by the initial reference norm. Current-field relative errors are reported separately. Same-grid tight-FOM discrepancies are also retained for each reduced timestep.

The bounded per-resolution search varies timestep and nonlinear-solver stalling tolerance while preserving the checkpoint. This is solver configuration selection on validation, and does not establish the benefit of retraining at each resolution. The module accepts any compatible native separable checkpoint; new weight training is deferred until these measured costs/errors locate its value.

Compilation, bank/operator construction, quadrature fitting, memory, singular-value rank diagnostics, and all timing repetitions are saved. Every timed invocation saves its actual observation-grid fields, full-output hash, error, iterations and stop conditions. Timings alternate forward/reverse configuration order with GPU burn-in before each timed sweep. There are no cross-job timing comparisons.

`cluster/stage.py` verifies every staged byte against a commit, `submit.sh` directly copies into a unique approved paralab attempt, and `collect.sh` checksums outputs and closed logs before deleting only that exact attempt directory. Job names begin with `ctol_` so the repository numeric cancellation guard applies.

The report generator reads the saved JSON and writes all numerical tables. Completed results will remain development evidence until reference checks, convergence, provenance, and independent review pass.

## Plain-language glossary

- **FOM / NM-ROM:** full discrete PDE solve / nonlinear-manifold reduced solve.
- **Bank / head / latent coordinate:** learned spatial features / neural map producing their coefficients / compressed state solved online.
- **Frozen transfer:** unchanged network weights evaluated and evolved on a different grid.
- **Intervals / nodes / interior unknowns:** cells along an axis / grid points including walls / values solved away from the walls.
- **Weak residual / sine tests / quadrature:** PDE equations averaged against smooth functions / those functions here / a weighted subset of grid points for advection evaluation.
- **NNLS:** nonnegative least squares, used to fit those weights.
- **Sign-upwind:** backward or forward spatial differences selected by the local field's sign.
- **Backward Euler / Newton / BiCGStab / preconditioner:** implicit timestepping / nonlinear solve / iterative linear solve / transform improving that linear solve.
- **DST-I / FFT:** a sine transform / fast Fourier transform used to compute it.
- **Validation cohort / final cohort:** development cases used to select settings / independent cases reserved for later confirmation.
- **Physical-reference error / same-grid discrepancy:** difference from a refined reference / difference from a converged full solve on the same mesh and timestep.
- **Refinement estimate:** the difference after making a mesh or timestep finer; it is evidence about reference uncertainty, not a proof.
- **Complete query / offline setup / compilation:** measured input-to-output solve / reusable preparation / constructing executable GPU code.
- **Median / worst / outlier:** middle measured value / largest case error / case above a stated threshold.
