# Parameter-family reference preparation

This is a nonexecuting plan. Training remains closed until reference admission is supported by saved evidence and independent review. Re20 qualification alone does not admit the other parameters.

Choose the actual qualified same-domain mesh after spatial, solution-order, geometry-order, solver and quadrature controls. Copy its binary mesh byte-for-byte, preserving geometry order and the recorded FE layout. Use the existing immutable-mesh loader with exact mesh, identity and fresh-process reload hashes. Do not regenerate a mesh independently for each parameter.

The native common-lift control in `immutable_lift_control.py` loads the actual saved Stokes coefficients, confirms the immutable reuse/layout path, tests velocity reuse and viscosity-proportional pressure at Reynolds endpoints and the central case, and saves the raw scaled-residual evidence. It exports a candidate common-lift artifact with `physical_admission=false`. Only a separately reviewed G2 admission may unlock the production `CommonStokesLift` loader; an interface control is not a substitute.

Generate the declared nine training, two development and four validation references in disjoint case directories on that same mesh, retaining each nonlinear history, residual, trace, mass, divergence, pressure/force and balance diagnostics. Preserve exact parameter values and source/configuration hashes. Same geometry/layout evidence can be shared; parameter-dependent physical and refinement uncertainty cannot be silently inherited from Re20. Apply the external DFG benchmark anchors only to their declared Reynolds case. Keep validation data out of bank fitting and quadrature fitting; inspect development only under the frozen selection policy.

Every exported family case must reproduce the identical full physical quadrature points and weights. Retain velocity, pressure and physical velocity gradients from the actual invocation. The ROM preparation exports dimensionless velocity and pressure and gradients scaled by D/U, with gradient units explicitly `d(u/U)/d(x/D)`. Retain the common Stokes lift and its gradient separately so correction snapshots and reconstructions use the same boundary-compatible field. No nearest-node remapping, pressure shifting or coordinate nudging is permitted.

Capacity is reviewed for the measured selected mesh, not inferred from the old coarse adapter cap. Any streamed or blocked implementation must reproduce the complete full-rule objective and exact source-bound ordering. Report all offline generation/preparation/training costs. Timed FOM/POD/bank queries must share output contracts, hardware allocation and actual-returned-state accuracy evidence, with cached common initialization and reviewed convergence tolerances.

## Glossary

Re: Reynolds number, the dimensionless parameter controlling viscosity here. Immutable mesh: the exact saved binary mesh reused without changing its coefficients. FE layout: the ordering of finite-element coefficients. Stokes lift: a boundary-compatible linear flow used for initialization and reconstruction. Correction snapshot: the nonlinear solution after subtracting that lift. Full rule: every positive mapped quadrature point and its weight. G2: independently reviewed physical-reference qualification. Validation: held-out cases used only after model choices are frozen.
