# Obstacle ROM/POD infrastructure contract

This implementation is ready for independent code review. It has synthetic algebra checks only; it has not passed an obstacle reference, training, pressure-stability, quadrature or physical-accuracy gate. The coordinator's current `DESIGN.md` and dated amendments supersede discovery suggestions.

## Files and entry points

| Module | Public interface | Scope |
|---|---|---|
| `fe_adapter.py`, `fe_pod_a6.py`, `fe_controls_a6.py`, `fe_algebra.py` | `TrainingFEAdapter.open/build/export`, `coefficient_pod_a6` | Prospective A6 gated 2D FE loader, coefficient-metric QR/SVD POD, supremizers and fixed test eigenspaces; no FE execution or integration admission. See `FE-ADAPTER-CONTRACT.md`. |
| `geometry.py` | `Geometry.envelope`, `fluid_mask` | Exact circle/sphere geometry; physical coordinate derivatives. Inlet factor uses the actual inlet coordinate. |
| `bank.py` | `initialize`, `FrozenBank.evaluate/save/load` | Frozen tanh model, NumPy values and analytic first derivatives, portable non-pickle NPZ. |
| `train_bank.py` | `train`, `array_hash` | One admitted nine-state training family only, GPU/f64 QR projection loss, fixed architecture/seed/step/time cap. No dataset is bundled or opened automatically. |
| `spaces.py` | `ordered_transform`, `weighted_pod`, `pressure_basis`, `projection_floor` | Weighted training ordering/POD, constant-first pressure space, retained field reconstruction coefficients. |
| `spaces.py` | `canonicalize_eigenmodes`, `select_test_columns`, `inf_sup` | Deterministic eigenblock orientation and test enumeration, explicit standard velocity-H1/pressure-L2 inf-sup check. |
| `weak.py` | `WeakRule.tensor/direct`, `WeakOperator.residual/jacobian` | Shared bank/POD gradient-form mixed weak residual, exact quadratic tensor, independently assembled direct-rule control. |
| `solver.py` | `solve_steady`, `jacobian_check` | CPU weak least squares; square mixed Galerkin is separately tagged and requires a root residual, not merely stationarity. |
| `quadrature.py` | `output_moments`, `fit_positive`, `convection_candidate` | Positive output/gradient moment rule for convection only; linear/pressure/divergence operators remain exact full-rule controls. Failed fits cannot deploy. |
| `interpolation.py` | `CoefficientPCHIP` | Mandatory zero-PDE-solve interpolation baseline using training-only POD coefficients; no extrapolation. |
| `metrics.py` | `field_errors`, `representation_admission` | Unshifted pressure-scaled RMS, full-field velocity relative error; development floor stop before online PDE solve. |
| `weak.py` | `force_moments` | Separate linear Cauchy-stress boundary moments with solid-to-fluid normals. |
| `timing.py` | `paired_cpu` | Alternating same-process CPU repetitions, retained actual scored outputs/checksums and work counts. |

## Reference exporter bridge

Reference agent retains `.vol` mesh, `.sol` state, exact configuration and FE evaluators. Requested arrays at a mapped full integration rule are physical points `(n,d)`, strictly positive physical weights `(n,)`, velocity lift `(n,d)`, lift gradient `(n,d,d)`, pressure lift per physical viscosity `(n,)`, test values `(n,d,Mv)`, test gradients `(n,d,d,Mv)`, and continuity tests `(n,Mq)`. Gradient axes are always component first, derivative coordinate second. The same evaluator supplies pressure/supremizer/POD traces and derivatives at arbitrary requested points and on both exact and FE boundaries.

`WeakRule` receives **dimensionless** fields: coordinates $x/D$, weights divided by $D^d$, velocity lift divided by $U$, lift gradients multiplied by $D/U$, and pressure lift per dimensionless viscosity multiplied by $D/U$. Neural velocity values are already in $u/U$ units; their physical-coordinate gradients need multiplication by $D$. FE test values are treated as dimensionless; their physical gradients need multiplication by $D$. Pressure trial functions are in $p/U^2$ units. `physical_to_dimensionless` handles the lift/weight portion. No hidden unit conversion occurs in assembly. The reference exporter and adapter must be independently reviewed together before a numerical release.

The physical pressure field is $p_L(\nu)+\Psi c$. Pressure POD construction uses training corrections $p_{train}-p_L(\nu)$, with a constant mode first and weighted-centered pressure POD modes next. Removing training means to construct the remaining modes does not shift the physical pressure gauge: the constant pressure coefficient remains an unknown. Pressure count is the minimum of the velocity prefix and one plus the centered snapshot numerical rank. Both methods use exactly this rule.

## Fixed test enumeration

Momentum candidates are all maximum-prefix pressure supremizers first, then scalar velocity-order H1 generalized eigenmodes with homogeneous essential boundary and natural outlet, each explicitly copied into the first and second physical component. Actual component DOF maps and scalar/vector matrix blocks are checked. Continuity candidates are the constant, all pressure modes, then scalar Laplace modes. Eigenpairs are sorted by eigenvalue. Within eigenvalue clusters, a canonical basis uses successive largest squared residual FE-DOF row norms, numerical ties to the smallest DOF index, with positive pivot orientation. Rotation invariance is an exact-arithmetic property; the declared numerical tie rule does not promise cross-platform uniqueness. The frozen mesh DOF numbering and eigenproblem matrices must be hashed. A look-ahead gap is not proof of complete scalar eigenvalue counting. No validation-state information enters enumeration.

`select_test_columns` retains the first independent columns using the fixed rank rule until the required momentum/continuity counts are reached; insufficiency is an error. It normalizes using the full-rule dimensionless H1 or L2 metric. The resulting transform, raw candidate indices, rejected indices, eigenblock pivots and every Gram are saved. Physical pressure level remains observable through tests that are free at the outlet; the constant continuity test alone is not a substitute. Actual nonzero outlet-flux ability and inf-sup are runtime gates, not guarantees of this enumeration.

The square mixed Galerkin control is mandatory. `WeakRule.tensor(formulation="mixed_Galerkin")` verifies its momentum and continuity tests equal the actual velocity and pressure trial functions. The main least-squares operator requires more independent rows than total unknowns. Both use the same direct and tensor convection formulas; there is no prior cost advantage claimed for either basis at equal dimensions.

## Fitting and output admission

The positive fitting matrix consists, point by point, of constant and coordinate moments, reconstructed training velocity components, reconstructed training velocity gradients, and their tested products $v_i\cdot[(u\cdot\nabla)u]$. There is no net PDE residual, pressure/viscosity cancellation or validation snapshot in that matrix. A fixed physical row magnitude normalizes moments. NNLS reports its active support, weights, row scaling, fit residual and restricted-support KKT checks. This is fitted positive quadrature, not an unproved general energy-conserving ECSW implementation. Boundary force quadrature is separate. The nonlinear operator and Jacobian, and returned physical fields/forces, still need all central admission checks on reached states.

The current fitted-rule implementation precontracts convection into the same online tensor shape as the full-rule control. Fewer fitting/integration points alone therefore do not establish an online coefficient-solve speedup. Charge the changed assembly/setup cost separately. A future direct sampled-point evaluator would be a distinct implementation requiring review and its own matched timing evidence.

Initial neural fit is rank eight, two hidden tanh layers of width sixty-four, fixed recorded seed, constant Adam step size, and fixed step/time ceilings. QR eliminates training coefficients without a hidden Gram ridge. The final model uses training QR/SVD ordering. G2-steady training admission contains exact case order, training-array hash, independent review record and reference manifest hash. The training API rejects missing/mismatched admission before importing JAX. Its time checks occur at synchronized step boundaries; the cluster launcher must also impose a hard process deadline. Compilation and every failed attempt are charged.

Before online solves, compute actual development projection fields against full reference velocity and unshifted pressure and apply the frozen one-fifth-target stop. Preserve gradients and forces of those achieved fits separately; small velocity projection error is not a pressure or force certificate. PCHIP coefficients require the same output reconstruction and physical checks as a solved model. Its zero-solve nature is a legitimate competitor for this smooth one-parameter family.

## Verification and limits

`smoke.py` exercises exact boundary zeros including the negative 3D inlet coordinate, neural finite-difference derivatives, model save/load identity, direct/tensor weak residual agreement, centered derivative checks, weighted POD/order reconstruction, deterministic degenerate-eigenmode orientation, test-Gram normalization, inf-sup rejection, positive fit acceptance/failure, square Galerkin root logic, interpolation and analytically integrated pressure force. It is CPU-only and uses synthetic arrays, not physical benchmark data.

`CPU-SMOKE.json` is generated by that program. `SMOKE-FAILURES.md` retains the first serialization failure. No measured training or Navier–Stokes result is claimed. JAX fitting code is syntax-checked but has not executed before its gate. The A6 adapter rejects non-float64 archives before array payload loading, builds mass-metric coefficient QR/SVD POD, and checks actual Gram, spectrum and registered-prefix residual consistency before pressure coupling. The old quotient-mode construction and its failing synthetic evidence remain archived. Its inf-sup pressure Gram comes from the actual FE evaluations. It still requires narrow independent review of these corrections, actual pinned-library FE integration and genuine G2 exports. Full runtime pressure/conditioning admission and actual mixed POD assembly remain dependent integration work.

The timing recorder deep-copies each invocation's complete output/accounting record before hashing or retaining it and marks stored output arrays read-only. A regression reuses one output array and nested mutable accounting dictionary across every method/repetition, then mutates the owner again; earlier saved values, hashes, work counts and component times must remain unchanged. Evidence-copy/hash overhead is reported separately from query execution. Any output copy required by the actual user-facing query contract belongs inside that method's callable and its timed cost.

## Glossary

POD: weighted snapshot basis. FE: finite elements. G2-steady: admitted stationary reference/data pipeline. H1/L2: gradient-plus-value and value-only function-space norms. Supremizer: velocity enrichment exposing pressure through divergence. Weak residual: integrated PDE mismatch. Galerkin: trial-space test equations solved as a root. QR/SVD: matrix factorizations used for stable ordering and rank checks. NNLS: nonnegative least squares. KKT: constrained optimization stationarity/dual checks. PCHIP: shape-preserving piecewise cubic interpolation. Prefix: the first columns of one frozen ordered bank. Full rule: the independently converged geometric integration control. Jacobian: derivative of the reduced equations with respect to solved coefficients.
