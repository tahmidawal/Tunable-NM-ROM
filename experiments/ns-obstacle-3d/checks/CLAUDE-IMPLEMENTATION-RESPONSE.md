# Response to independent Claude implementation review — 2026-10-27

Review SHA256: `cbbc08dce5d72baf034faade4d261cfdfebfa7a5cf636d0082ad6224f6ba73ca`.

Source review read in full: coordinator `experiments/ns-obstacle/reviews/CLAUDE-IMPLEMENTATION-REVIEW.txt`. This response prepares a new candidate `g1-smoke02` for independent Codex diff review. It does not mark execution or scientific gates passed. The unexecuted `g1-smoke01` local/remote package and checksum record remain unchanged.

| Review item | Response in candidate 02 |
|---|---|
| B1 normal orientation | On each curved boundary element, independent mapped quadrature samples compare the mesh normal against the geometric sphere radial, pipe radial or axial normal. Sphere dot must be at most −0.99; pipe-wall dot at least 0.99; inlet/outlet axial components must equal −1/+1 to the explicit tolerance. Invalid data, missing labels/samples or a failed check stop before case factorization. These are point checks, not a continuum geometry proof. |
| B2 physical balances | Rename physical outputs `physical_momentum_raw_uncorrected_continuum` and `physical_energy_raw_uncorrected_continuum`. The separate gradient-form identities retain the required finite-divergence and body-force terms; no PDE change was made. |
| B3 reaction checks | Save reaction-minus-analytic-exact-sphere force and reaction-minus-exact-polynomial-on-FE-surface metrics for MMS, alongside reaction-minus-gradient/symmetric-surface force for each integration order. The existing analytic expression and symbolic derivation remain unchanged. |
| A4 probes | Require `MeshPoint.nr >= 0` before field evaluation; exceptions and invalid points remain unresolved. No probe displacement or nearest-node substitute is introduced. |
| A8 termination/time | SIGTERM raises a caught termination exception so summary and output checksums can be saved when Python regains control. Native calls can delay handling. The outer timeout uses a kill-after interval; remaining solve time is computed after installation and capped, leaving finalization/Slurm margin. The installer also receives a hard external timeout. |
| A2/A3 backend/export preflight | A tiny scalar FE mass-plus-diffusion solve with the exact selected UMFPACK backend and a small DOF cap runs before actual cases. Mapped volume/boundary transforms, Jacobian matrix conversion, normals, coefficient values and vector gradients are checked on that tiny mesh. Each actual case also exercises one mapped volume and boundary point immediately after curving. |
| B5 residual normalization | Output states explicitly that floor activation makes the criterion absolute in fixed row-scaled units. The denominator and actual residual remain saved. |
| B6 memory | The geometry-only first case constructs the planned spaces, prints/saves their actual DOF counts and conservative screen, and stops before a case factorization if the unchanged memory/DOF guard fails. |
| B7 precision/path | JAX preflight explicitly asserts x64 and the GPU backend; lane-local dependency path is prepended while retaining an existing PYTHONPATH. |

Not adopted: Claude's proposed relaxation of the direct linear-residual threshold to $10^{-6}$. The registered $10^{-10}$ threshold stays unchanged; any actual failure must be retained and analyzed before an amendment. Not adopted: speculative MKL/intel-openmp dependency additions. Retrieved metadata pins NGSolve's matching OpenBLAS package; the tiny backend preflight will supply evidence about availability. No unrelated packages or library-path changes were introduced.

The mapped API choices were checked against the pinned NGSolve source: explicit `ElementId(Ngs_Element)` and `Mesh.GetTrafo(ElementId)` in `comp/python_comp_mesh.cpp`; mapped `jacobi`, `point`, `measure` and transformation calls in `fem/python_fem.cpp`. Runtime compatibility is still an unpassed smoke condition. API source URLs:

- https://raw.githubusercontent.com/NGSolve/ngsolve/v6.2.2608/comp/python_comp_mesh.cpp
- https://raw.githubusercontent.com/NGSolve/ngsolve/v6.2.2608/fem/python_fem.cpp

The actual PDE, force normalization, inlet, outlet, geometry, strict solver criteria and conservative memory thresholds are unchanged. Assembly uses additional integration order, while diagnostics use absolute integration order; that distinction remains explicit. Future reference gates still require full signed-curved-Jacobian checks, stronger trace checks, convergence/refinement, tolerance and domain/outlet sensitivity, restart parity and independent numerical review. No job or installer has run.
