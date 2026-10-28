# Prospective G2 producer controls and shared lift

This source-only milestone is unexecuted and grants no numerical release or physical admission. The released G1 attempt and its staged source remain immutable. Root-level `INPUT-MANIFEST.json` and `SOURCE-SHA256SUMS` retain that historical G1 binding; `SOURCE-MANIFEST.DRAFT.json` beside this document inventories the new development source and is deliberately not a staging manifest. Regenerating a submit-ready inventory requires independent review and a new coordinator release.

## Controls and export contract

`reference_export_controls=true` is permitted only for `G2_steady_control_candidate` or future `G2_steady_training_export`. Existing G1 configurations do not enable these controls. Future G2 drafts bind the exact central A6 campaign/design at `de62124b1`. The released G1 binding is unchanged. The layout algorithm and export contract retain reviewed ROM commit `97847a479`; additional A6 controls are prospective.

The full mapped volume rule preserves every point and positive weight. For the exact circular solid it records signed radius error, the number and total weight of points strictly inside, their weight fraction and minimum signed distance. No point is removed or reweighted. A G2 candidate fails if an inside point occurs. The exporter adds ordered mapped Jacobians and the exact ordered mixed-space layout digest used by the reviewed adapter. Its pure test compares the layout function's syntax tree with the reviewed source.

`reload_checks.json` is produced by actual independent loading of `mesh.vol`, rebuilding the FE spaces, loading both solution and Stokes coefficients, and re-evaluating every volume/boundary sample. It checks curve order, coefficient order, physical points, positive weights, Jacobians, values, gradients, pressure, normals and boundary labels. For each numeric array the saved metric is

$$e=\frac{\max|a-b|}{\max(\max|b|,s)}.$$

The denominator scale $s$ is explicitly recorded through the numerator/denominator pair: position $D$, velocity $U$, gradient $U/D$, pressure $U^2$, and weights/Jacobians/normals one. The fixed threshold is $10^{-10}$. Reload calls never re-curve a mesh. Success writes a mesh identity bound to both mesh bytes and actual reload evidence; later immutable reuse copies those exact bytes, checks geometry/construction/library/layout identities and repeats the reload checks.

The producer's `EXPORT-MANIFEST.json` binds each output, source file, case identity, campaign and exact library versions, with `physical_admission=false`. It cannot create the outer passed G2/training manifest. No training or held-out snapshot is read here.

## Essential traces and gradient orientation

The existing sampled analytic-data trace error remains unchanged. A separate certificate compares the final FE trace with the FE trace actually prescribed before the solve. On a reference edge this difference is a vector polynomial of degree at most $p$. Values at $p+1$ Chebyshev–Lobatto nodes determine its components; independent samples check the fit at tolerance $10^{-12}U$. Endpoint and stationary values give a pointwise maximum estimate. If $c_{ij}$ are the component Chebyshev coefficients, the bound used for enforcement is

$$\max_{s\in[0,1]}\|e(s)\|_2\leq\sqrt{\sum_i\left(\sum_j|c_{ij}|\right)^2}+10^{-12}U.$$

The fixed allowance accounts for the verified numerical fit; this is floating-point evidence, not an interval-arithmetic proof. Every element's coefficients, candidates, fit error, estimate and bound are saved. The central trace gate applies to the prescribed-FE bound; actual analytic-data approximation is retained separately, particularly for the non-polynomial manufactured trace on the curved obstacle. Physical inlet/no-slip data remain their analytic targets.

Reloaded normals additionally pass analytic side orientation, inward obstacle radial orientation and normal closure separately on the obstacle and the union of outer faces. The closure error is normalized by each closed component’s boundary measure and must be at most $10^{-10}$. Closure is not tested on an individual open face and cannot replace the sign test.

The new affine witness $u=(y,0)$ checks the non-symmetric gradient $\begin{bmatrix}0&1\\0&0\end{bmatrix}$ in volume and through the wrapped boundary derivative at the existing $10^{-10}$ tolerance. The fixture uses degree $\max(p,g)$ so physical affine coordinates remain representable during the independent $p=3,g=4$ check; production spaces are unchanged. Transformation objects remain alive during mapped-point access, and exported arrays own their memory.

The downstream centerline diagnostic samples actual FE velocity at the obstacle center height with declared target and actual spacing. It saves each point's validity, never nudges invalid points, and does not infer wake length or alter exact pole pressure probes.

## One common Stokes lift and future query initialization

The draft matrix marks only its canonical fine DFG case for `export_common_stokes_lift=true`. Its one validated Stokes state supplies `common_stokes_lift.npz`: velocity coefficients and pressure coefficients divided by the declared base viscosity. The companion JSON records mesh/layout hashes, scales, base viscosity, unchanged natural-outlet pressure convention, linear-control evidence, coefficient hash and diagnostic Stokes phase time. That phase includes linear assembly/factorization/solve/check; preceding mesh, space and form setup is recorded separately, and no tuned offline timing is claimed. This candidate is not admitted merely by export.

`shared_lift.CommonStokesLift.from_admission` requires a future coordinator-produced `ns-obstacle-common-stokes-admission-v1` record with `gate=G2-steady`, a passed decision, exact common-lift artifact hashes and source-bound review/reference-admission digests. No such passed record is created by this milestone. After offline verification/loading, `state_for_nu(nu)` copies the common velocity and scales only its pressure block; the cached arrays stay immutable. Its mesh, layout, scales and outlet convention must match the future query problem. All training states must refer to this single lift identity; independently solved lift equality is not the definition.

The robust reference generator retains independent Stokes solves as control work. The future tuned paired FOM query runner must use the cached shared lift, charge one-time construction/loading offline, and include the necessary per-query coefficient copy/scaling and nonlinear work online. That runner, tolerances and paired timing experiments are not implemented or measured here. Per-point reference export/reload/diagnostic loops are timed separately and cannot be counted as competitor query work. Actual CPU concurrency remains unmeasured; no performance claim is made.


## Prospective normal-orientation repair propagation

The isolated G1 normal repair at `26e9b8272` is under review. Its helper is now
propagated into this separate G2 development source, with no G2 release. Both
original generation and actual mesh reload derive and independently verify
fluid-outward signs from adjacent volume topology. The raw curved normals,
per-label factors and complete certificates are retained. Reload checks compare
both raw and oriented normals and require the same inferred factors.

The helper's ordinary boundary coefficient function is used consistently by
loads, fluxes, balances, forces and export; volume-facet controls retain native
normals. Source provenance includes the helper. This propagation does not
advance reference admission, update the currently provisional mesh-size matrix,
or authorize any G2 execution. Setup verification is offline/reference work;
a future matched-query benchmark must reuse a validated common-mesh certificate
fairly instead of charging repeated helper assemblies only to the FOM.

## Resource projection and remaining gates

`plan.py` generates the draft matrix and optionally projects from explicit collected G1 mesh counts, reading array headers rather than field values. Exact mesh sizes are unresolved; no G2 config is runnable. The existing adapter point, coefficient, element and archive caps remain fixed. A projected cap violation requires a reviewed resource-only amendment or separately qualified deterministic rule, never a silently weaker reference rule.

The matrix includes three mesh levels, independent geometry/p-order controls, solver and quadrature tightening, upstream/downstream extensions and the separately named symmetric outlet problem. Common-domain field comparison and uncertainty aggregation still need a reviewed comparison implementation. Every eventual training/development/validation reference must pass the common gates; Re20 does not confer admission on other viscosities. The shared-lift adapter connection, actual curved-file round trip, native gradient/trace APIs, exact resource counts and the eventual tuned FOM query runner all remain unverified until their appropriate reviews and authorized runs.

## Glossary

G1: initial bounded smoke controls. G2: reference accuracy and domain-sensitivity admission. FE: finite element. DOF: a solver coefficient. Mapped rule: quadrature points and weights transformed to the curved physical elements. Layout digest: a hash of ordered coefficients, free masks and local element numbering. Prescribed trace: the FE boundary data held fixed by the solver. Shared lift: one fixed incompressible Stokes velocity field and its pressure per unit viscosity. Offline cost: reusable preparation performed once. Query cost: work required for each new viscosity. Physical admission: an independent decision that all reference gates passed, never implied by successful serialization.
