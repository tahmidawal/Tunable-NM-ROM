# Three-component reference controls: implementation checkpoint

Candidate `g1-smoke04` contains revised, unexecuted finite-element code for independent diff review. Independent review found an invalid boundary-gradient evaluation in earlier candidates; their release is withdrawn. The earlier staged `g1-smoke01` remains immutable and has not run; candidate 02 was never staged. See `checks/BOUNDARY-TRACE-AUDIT-CORRECTION.md`. The only completed checks are Python/shell parsing and an independent symbolic manufactured-field derivation; no numerical reference, dependency import, geometry or solver gate has passed.

## Authoritative specification

`CAMPAIGN.predata-A3.json` copies the coordinator's central specification; the exact SHA256 is recorded and enforced by `configs/g1-smoke04.json`. The earlier A2 copy remains for provenance. Central geometry $x\in[-5,15]$ supersedes the longer discovery proposal before any data. The original `SPEC.proposed.json` remains unchanged as historical design provenance. Pressure means kinematic pressure with an outlet-fixed level; there is no pressure penalty or additional gauge.

The smoke uses inexpensive $P_2/P_1$ Taylor–Hood fields and cubic geometric mapping only to check execution and establish resource feasibility. It is not the planned $P_3/P_2$ reference or its $P_4/P_3$ cross-check. Mesh seed is recorded; the relationship of NumPy's seed to Netgen's deterministic defaults remains to be verified at runtime, so no stochastic mesher reproducibility assertion is made.

## Code path

`reference3d.py` creates an actual circular pipe minus sphere by constructive solid geometry. It checks boundary names and saves geometric integral controls. It then solves an empty-pipe Poiseuille control and a manufactured flow on the actual sphere-excluded fluid region. Physical sphere flow is implemented as a separately selectable case, but is absent from this first smoke configuration.

The manufactured polynomial is

$$u=(Y^2+Z,\;Z^2+X,\;X^2+Y),\qquad p=X+Y^2-Z,$$

with $X=x/L$, $Y=y/R$, $Z=z/R$. It has three nonzero components, exact zero divergence and nonzero convection in all components. The physical source is derived from the full steady momentum equation. All essential boundaries receive the exact nonzero manufactured velocity, and the outlet receives its exact nonzero natural traction. `verify_manufactured.py` uses independent symbolic differentiation to verify the manually coded gradient, Laplacian, pressure gradient, forcing and stress divergence. The resulting `MANUFACTURED.symbolic.json` is algebraic evidence, not flow validation.

The nonlinear form uses $\nu\nabla u:\nabla v+(\nabla u)u\cdot v-p\nabla\cdot v-q\nabla\cdot u$, plus prescribed forcing and outlet load. Fixed continuation starts from the Stokes problem and ends with full convection. Damped Newton solves the true linearization, checks the direct linear solve's free residual, and rejects iteration-limit or failed line-search exits. At acceptance an independent `LinearForm` recomputes the residual from the returned velocity and pressure. Serialized row units are $U^2D^2$ for velocity and $UD^2$ for continuity; normalization is the maximum of the initial Stokes-problem residual at the exact-boundary/zero-interior initial vector and the configured floor. That denominator stays fixed throughout continuation and Newton; it is not an output-error estimate.

Saved diagnostics distinguish symmetric physical stress from gradient stress, physical surface force from weak reaction force, body-force and boundary flux terms, and finite-divergence corrections to gradient-form momentum/energy identities. Boundary integration is repeated at two orders. Pressure errors use $U^2\sqrt V$ scaling without pressure shifts. The manufactured-force check compares the FE solution's surface force, the exact polynomial integrated on the FE sphere, and the analytic integral on the exact sphere, thereby separating discretization from geometric integration.

## Memory and release conditions

The first solve is limited to at most fifteen thousand total mixed degrees of freedom. Before matrix factorization it applies the conservative screen

$$B_{screen}=128n_{dof}^2+4\,\mathrm{GiB},\qquad B_{screen}\le48\,\mathrm{GiB}.$$

This is a screening allowance, not a mathematical upper bound for library memory. The batch reservation remains sixty-four GiB; observed peak resident memory, degrees of freedom and factor/solve time are saved. A later larger level needs two cheap-level observed memory measurements and a new independent release. No scalable iterative solver or large factorization is silently substituted when a guard fires.

The proposed batch script is exactly one twenty-minute GPU allocation with sixteen reserved CPU cores, four numerical threads and a mandatory JAX GPU preflight. NGSolve solves run on CPUs and are labelled accordingly. Installation uses designated cluster Python and a new per-job `deps/` prefix, preserving native libraries under its `lib/` directory. Four matching binary wheels are pinned by filename, version, size and SHA256; `--no-deps`, `--require-hashes`, `--no-index`, `--ignore-installed` and the private prefix prevent shared dependency replacement. Raw RECORD paths, sizes and hashes, OCCT wheel-native entries, and package-relative OpenBLAS files are checked before importing. JAX/JAXlib/NumPy versions are checked before and after. Download, installation and import logs remain in the isolated allocation. Neither the installer nor the batch has been executed.

The coordinator requires the 2D smoke to pass and an independent memory/configuration audit before releasing the 3D solve. The central scientific design also requires independent Claude review. This implementation checkpoint does not override any release gate.

## Explicit remaining infrastructure and scientific work

- Runtime NGSolve/Netgen import and API compatibility, actual direct backend and curved FE behavior remain untested.
- The geometry report deliberately says that a full signed-curved-Jacobian certificate is missing. The backend/geometry preflights check one mapped volume point, which does not certify all curved elements. Volume/area and normal-integral checks alone cannot supply it.
- Boundary RMS and essential-coefficient mismatch are saved. A pointwise maximum trace certificate is not implemented and is not replaced by RMS.
- A full $h/p$/geometry/solver-tolerance/domain/outlet series, reference uncertainty and strong FOM ladder are future released work. The first smoke cannot admit training data.
- Mesh serialization preserves the emitted mesh/solution and coefficients; independent reload/reconstruction parity must pass before using these for data transfer.
- Runtime boundary pressure-probe failures are retained as unresolved, with no nearest-node substitution.
- The transient integrator is deliberately absent pending its separate implementation/review gate.
- No bank, POD or fitted quadrature training has been performed.

## Candidate g1-smoke02 practical review fixes

The independent Claude implementation review found the manufactured derivatives and force identity correct. Its practical suggestions adopted here are detailed in `checks/CLAUDE-IMPLEMENTATION-RESPONSE.md`: independent per-boundary normal samples, a tiny sparse-backend and mapped-export preflight, reaction-versus-exact force metrics, invalid-point rejection, explicitly raw physical balance labels and bounded termination handling. There has been no installation or numerical FE execution. Strict linear residual admission remains unchanged; speculative dependency and tolerance changes were rejected.

## Candidate g1-smoke03 boundary derivative correction

All computed surface stresses use `BoundaryFromVolumeCF(Grad(u_h))` before multiplication by a boundary normal. The tiny preflight checks the volume-side gradient, divergence, separate-face gradient/symmetric tractions and stress powers on a labelled unit cube. Volume operators and analytic prescribed tractions are unchanged. Source hashes and the correction to earlier readiness assessments are retained in `checks/BOUNDARY-TRACE-AUDIT-CORRECTION.md` and its source inventory. No numerical run has occurred.

## Candidate g1-smoke04 dependency layout and crash diagnostics

The private prefix repair follows independently reviewed 2D commit `d345b0b05`. Its original wheel pins, scientific settings and release gates are unchanged. `checks/DEPENDENCY-PREFIX-REPAIR.md` records the adaptation and metadata-only checks. The 2D lane has reported successful prefix/import checks followed by an unresolved native API-preflight crash; no flow case ran there. Candidate 04 therefore adds Python fault reporting and flushed markers before/after native preflight calls. All mapped transformations remain alive through point evaluation and array copying. The 3D lane has not installed packages or run numerical cases.

## Glossary

**Taylor–Hood:** compatible mixed velocity/pressure finite elements. **Mixed degrees of freedom:** all velocity and pressure coefficients combined. **Natural traction:** outlet load implied by the chosen weak form. **Continuation:** gradually increasing the convection factor while retaining each solved state. **Newton:** nonlinear iteration using the derivative of the residual. **Free residual:** equations remaining after essential boundary values are imposed. **Reaction force:** force obtained by testing the assembled residual with prescribed boundary motion. **Peak RSS:** maximum resident host memory observed by the process. **Screen:** conservative resource rejection rule, not a numerical-accuracy certificate. **Geometry order:** polynomial degree of curved element coordinates. **Reference admission:** the independent checks required before a result can serve as truth data.
