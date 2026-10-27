# Gated FE adapter and exporter integration

This is implementation infrastructure for independent review. No NGSolve import, mesh, flow case, FE eigenproblem or FE supremizer has been executed by this lane. `FE-ADAPTER-SYNTHETIC.json` is only a small array/matrix regression; it cannot release reference or training gates. The current G1 FOM files are deliberately insufficient for this adapter.

## Ownership and current sources

The ROM lane owns `fe_adapter.py`, `fe_algebra.py` and their checks. The adapter never imports `fom.py`, runs `run_case`, writes into the reference worktree, or updates the canonical lab log. The reference lane owns the FOM and G2 exports. The coordinator owns reference admission and the source-bound adapter review record.

The inspected producer is `experiments/ns-obstacle-2d/fom.py` in the `2026-10-27-ns-obstacle-2d` worktree: it saves `config.json`, `mesh.vol`, `full_rule.npz`, `boundary_rule.json`, `solution.sol` and `result.json`. Its full rule already contains physical points, positive weights, returned velocity/gradient/pressure and both mixed coefficient vectors. Stokes coefficients are copied before Newton; the returned solution coefficients are restored after its directional-Jacobian diagnostic. The producer's gradient convention is component, derivative coordinate.

The reference author explicitly has **not** certified that `.vol` retains the curved geometry. Both sides require a future reload check. The adapter calls `Mesh(mesh.vol)` and never calls `Curve` after loading. A mismatch stops the bridge; no re-meshing, interpolation, re-curving, point nudging or mesh substitution repairs it silently. Only the fixed nine-state 2D training family is implemented; the conditional 3D family needs its own reviewed extension.

## Exact future G2 producer additions

| Artifact | Required content |
|---|---|
| `full_rule.npz` | Existing arrays plus `jacobians[n,2,2]` from the same ordered mapped volume rule. Retain solution and Stokes coefficient vectors. All numeric arrays must already be stored as finite float64. The adapter checks every NPY member header and declared allocation size before loading array payloads, rejecting other dtypes; it never upgrades float32 provenance to float64. |
| `fe_layout.json` | `{"layout": record, "sha256": digest}` using the exact `layout_record(mesh,V,Q,ng)` algorithm below. |
| `reload_checks.json` | `passed: true`, `policy: "saved_curved_verified_no_recurve"`, saved/reloaded geometry order evidence and numerical comparisons of mapped points, weights, Jacobians, volume/boundary values and gradients. This must come from a real reload, not a source inspection. |
| `config.json` | Existing exact configuration, `release: "G2_steady_training_export"`, `precision: "float64"`, central campaign hash. A G1 release is rejected. |
| outer training manifest | Exact schema below, with genuine independent G2 admission/review hashes and per-file SHA256 values. |

The versioned layout record has `schema: "ns-obstacle-fe-layout-v1"`, mesh dimension, velocity/pressure DOF counts, ordered free-DOF masks as integer zero/one lists, and an ordered list from `mesh.Elements(VOL)`. Each element stores `number=int(el.nr)`, `type=str(el.type)`, and ordered `V.GetDofNrs(el)`/`Q.GetDofNrs(el)` lists. Digest the JSON serialization with `sort_keys=True`, `separators=(',', ':')`, UTF-8 and SHA256. Save both the record and digest. The helper is import-safe without NGSolve; producer reuse requires staging the reviewed adapter sources, or an independently checked identical implementation of this specification.

Use one byte-identical immutable admitted mesh and the same FE/geometry/rule orders for all training states. The reference author agreed this is feasible for the viscosity-only family. If references need other meshes, construct the declared common training export with a separately reviewed transfer/error gate; this adapter does not invent one.

The outer manifest is `ns-obstacle-fe-training-v1`. Required fields are `gate: "G2-steady"`, `passed: true`, `split: "training"`, `mesh_reload: "saved_curved_verified_no_recurve"`, `adapter_source_sha256`, `adapter_review_sha256`, `reference_admission_sha256`, `campaign_sha256`, and `cases`. `adapter_digest()` binds the exact adapter/algebra/space/operator/training-hash sources and frozen model contract; the campaign digest must also match that contract. `cases` must have the frozen ordered Reynolds numbers from the central campaign; each case contains its `Re`, `split: "training"`, relative `directory`, and a `files` mapping of exactly the eight named producer artifacts to their SHA256 hashes. No dummy admitted example is distributed. Read-only manifest checking rejects a closed gate, a validation population, mismatched source, missing files or mixed mesh hashes before importing NGSolve.

## Frozen construction

For physical diameter $D$, dimensionless integration weight is $w/D^2$. Velocity and pressure coefficients are divided by $U$ and $U^2$. The shared Stokes velocity lift is recovered from the first admitted training state and compared against all other Stokes vectors. The pressure lift coefficient per dimensionless viscosity is $p_LD/(U\nu)$. The adapter checks that this is shared, and forms training corrections only after subtracting the lift. It preserves the physical natural-outlet pressure convention.

All velocity correction essential DOFs must be within the fixed roundoff threshold before being set exactly to zero; the removed maximum is recorded. Weighted POD uses the existing full-rule algorithm and carries its snapshot combinations back to FE coefficients, so every retained mode can be evaluated at arbitrary physical points. Pressure uses the same existing constant-first weighted-mean-removed training-correction construction and rank threshold. POD singular-value clusters use the existing coefficient-row pivot convention and save the rotation, keeping coefficient and sampled field representations consistent. POD prefixes above the actual weighted numerical rank are marked ineligible; smaller registered prefixes remain available. There is no manufactured zero-filled mode. No data outside training enter any basis or eigenproblem.

After carrying the retained SVD modes back to FE coefficients, independently evaluate those coefficients on the full mapped rule. For every velocity and pressure mode require

$$
\frac{\|\phi_j^{FE}-\phi_j^{SVD}\|_Q}{\|\phi_j^{SVD}\|_Q}\le10^{-10},\qquad
\|G^{FE}-I\|_2\le10^{-10}.
$$

These are fixed numerical reconstruction/orthonormality checks, separate from the physical field targets. Cancellation in $CV/\sigma$ near the existing singular cutoff causes a hard failure with the discrepancy/Gram diagnostics retained in the exception. Do not re-orthogonalize, drop extra columns, change the rank threshold or quietly substitute the ideal sample modes after failure. The pressure Gram used by the full-velocity reference inf-sup calculation is formed from these **actual reconstructed FE evaluations**, matching the pressure coefficients used in the divergence coupling and supremizers. Both velocity and pressure checks, their actual Grams and fixed thresholds are exported in metadata.


Let $M$ denote dimensionless FE mass and $K$ the dimensionless gradient Gram. The velocity supremizer solves

$$
(M+K)s_j=b_j,\qquad (b_j)_i=\int_{\widehat\Omega}\psi_j\widehat\nabla\cdot v_i\,d\widehat x.
$$

Velocity essential DOFs are removed; outlet DOFs remain free. Matrix assembly supplies explicit per-element-type integration rules matching the exported full rule. Matrix symmetry, solve residuals and agreement between assembled and sampled test Grams are checked. The full-velocity/same-pressure reference inf-sup value is recovered from $B(M+K)^{-1}B^T$ with pressure-L2 whitening for each registered pressure prefix. It is a reference value, not a passed reduced inf-sup gate.

Velocity eigenmodes solve $(M+K)e=\lambda Me$ on the same free velocity space. Scalar modes use the pressure H1 space with natural Neumann boundary conditions; the same shifted problem has Laplacian eigenvalues $\lambda-1$ and avoids factoring the singular Neumann stiffness. Fixed candidate counts are 96 velocity and 32 scalar modes, each with one extra eigenpair. A cut through an eigenvalue cluster stops construction. Sparse shift-invert uses SciPy `eigsh`, deterministic seed 27102026, tolerance $10^{-12}$, maximum 2000 iterations and SuperLU COLAMD. There is no fallback or enlargement after failure. Original and oriented eigenpair residuals, matrix hashes, look-ahead gap and canonical block pivots are saved. This is the documented generalized-eigenproblem and shift-invert contract of [SciPy eigsh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html); use on the pinned FE matrices still needs runtime review.

The fixed enumeration is all maximum-prefix supremizers then velocity eigenmodes, and constant then all pressure modes then scalar eigenmodes. Existing full-rule dependency rejection and normalization must produce exactly 64 H1 momentum tests and 16 L2 continuity tests. The duplicated constant in the scalar enumeration is deliberately rejected as dependent. The actual retained indices, transforms, Grams and candidate hashes are persisted. No label is taken as proof of pressure stability, outlet observability or good Jacobian conditioning.

## Bounded use and export schema

The prospective caps are encoded in `CAPS`: mixed unknowns, elements, volume/boundary rule points, sparse nonzeros, individual arrays/archives, estimated factor storage and process peak RSS. They are hard stops. The factor estimate repeats the FOM's conservative fill model; it is not a memory guarantee. A reviewed cluster launcher must additionally impose a hard memory/time limit and charge all failed setup work. This source provides no launcher and authorizes no job.

Future invocation is `TrainingFEAdapter.open(manifest).build().export(new_directory)`. There is no CLI that searches for snapshots automatically. NGSolve is imported only after admission and source/file checks. Scalar coefficient-function evaluation accepts only a finite real scalar or an array/tuple with exactly one element. Larger, empty, complex or nonfinite results are errors; this explicitly handles the pinned-library singleton-tuple behavior. The `.sol` vector must match the archived mixed coefficients, and evaluating those coefficients must reproduce all archived volume and boundary fields/gradients, mapped points/weights/Jacobians and normal labels. Raw-point evaluation rejects a point outside the saved FE geometry; exact/approximate curved-boundary consistency is a separate gate.

The export contains:

| File | Contents and units |
|---|---|
| `coefficients.npz` | FE coefficients for dimensionless lift velocity, pressure lift per dimensionless viscosity, POD velocity, pressure, supremizers and both fixed test banks. |
| `volume.npz` | Physical points/Jacobians, dimensionless weights, each coefficient field's values and dimensionless first derivatives. Velocity arrays have axes point/component/column, gradients point/component/coordinate/column; scalar arrays point/column. Lift fields retain a singleton column. |
| `boundary.npz` | Same field arrays on the separate positive mapped boundary rule, labels and outward-fluid normals. Only obstacle normals are negated for `force_moments`; other labels retain fluid orientation for flux/balance diagnostics. Surface weights divide by $D$. |
| `training.npz` | Declared training-only dimensionless velocity/pressure corrections and velocity gradients, physical points, dimensionless weights, ordered Reynolds numbers. Its neural-training array digest uses the existing training API's exact hash convention. |
| `manifest.json` | Input/source hashes, layout, rank/ordering/test/eigen/supremizer evidence, caps, versions, setup/export timing, peak RSS, units and hashes of all output archives. `physical_admission` remains false. |

For an eligible registered velocity prefix $r$, take the first $r$ POD columns (or later neural columns) and the first $p_r$ supremizers, with $p_r$ the minimum of $r$ and the saved pressure count. The pressure unknowns are the first $p_r$ pressure columns. Weak LS uses the saved fixed 64/16 tests; mixed POD Galerkin tests equal its actual enriched velocity and pressure trial columns. Squeeze singleton lift columns when constructing `WeakRule`. Boundary gradients use `BoundaryFromVolumeCF` while normals remain on the boundary transformation, following the repaired FOM convention. The [NGSolve NumPy bridge](https://docu.ngsolve.org/nightly/how_to/howto_numpy.html) documents coefficient-array views and sparse COO export; saved evidence is always copied before it can alias mutable FE storage.

All setup, FE recovery, eigen/supremizer construction, sample export and tensor assembly costs are offline costs to retain. Same-CPU online timing still requires the frozen output contract. The current fitted quadrature contracts into the same tensor shape: fewer points alone establish no online solve speedup.

## Remaining release blockers

Independent adapter/source review and a real pinned-library serialization/evaluation/matrix test remain mandatory before future FE use. G2 has not passed. The producer has not yet added the declared G2 layout/Jacobian/reload exports. Actual full-rule convergence, geometry consistency, full reference validation, reduced inf-sup/Jacobian gates, training execution and full end-to-end bank/POD assembly remain unpassed. The adapter builds smooth FE H1 functions; it does not assert analytic smoothness across element interfaces or exact pointwise incompressibility. The pressure-POD and test eigensolver APIs must not be described as integrated merely because synthetic algebra passes.

## Glossary

FE: finite elements. DOF: one coefficient in an FE space. G1: preliminary controls. G2: admitted reference/data gate. POD: weighted training snapshot basis. H1: value-plus-gradient function norm. L2: value-only function norm. Supremizer: a velocity enrichment exposing a pressure mode through divergence. Inf-sup: a scaled pressure/velocity coupling diagnostic. Eigenmode: a generalized matrix eigenvector interpreted as an FE function. Prefix: the first columns of one fixed ordered space. Round-trip: saving, reloading and comparing the same geometric/field data. Shift-invert: solve-based eigensolver transformation. COLAMD: a fixed sparse-factor ordering algorithm. RSS: measured process memory. Gram matrix: weighted inner products between basis functions. SVD: singular-value matrix factorization. Spectral norm: largest singular value of a matrix. Cancellation: loss of numerical precision when subtracting nearly equal quantities. SHA256: content digest. COO: sparse coordinate-list representation. Physical admission: all required scientific checks on actual flow results, distinct from source or array-algebra checks.
