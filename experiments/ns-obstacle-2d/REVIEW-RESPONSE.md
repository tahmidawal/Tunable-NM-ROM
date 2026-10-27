# First smoke implementation review response

2026-10-27. The independent Codex numerical reviewer returned CONDITIONAL READY for G1 smoke on source commit `4e38c8265`, with no mathematical blocker reported through the coordinator. This is not G2 reference admission or training release. The central campaign/design binding is refreshed to predata-A4; the copied compatibility filename `campaign-A2.json` now contains version A4, identified by its internal version and SHA256. No equation, mesh setting, solver stopping policy, threshold, or numerical source changed in this rebinding.

The coordinator requires the independent Claude implementation consultation before submission. The remote directory may contain reviewed source bytes only until explicit release. Neither installer nor numerical solver has run.

The reviewer identifies later G2 work: save actual exact-obstacle distance and excluded-point/weight diagnostics for the volume rule; save boundary velocity gradients or independently reload and evaluate the FE state for force audit; and measure real CPU concurrency before comparative timings. Existing positive-weight/signed-Jacobian and sampled boundary checks do not close those requirements. The current exporter preserves mesh, FE coefficients, and full-rule field gradients; their independent reload still requires verification. These gaps remain explicit and are not G1 blockers.

All constants and manifest identities must match the staged release; a later source fix requires a new recorded attempt rather than silently replacing live files.

## Claude implementation consultation response — before second staged attempt

The independent Claude consultation re-derived the core mixed weak form, force sign, reaction force, gradient and physical balances, and manufactured fields without finding a mathematical blocker. Its raw review is preserved by the coordinator. The response adopts concrete execution checks while retaining original scientific tolerances.

- Added `preflight.py`: a tiny finite-element UMFPACK solve plus mapped volume/boundary point, Jacobian, weight and normal checks before actual flow cases. The existing `GetTrafo(Ngs_Element)`, `np.asarray(mip.jacobi)` and coefficient normal evaluation are retained because the pinned official bindings document their conversion/properties; this smoke now tests them cheaply. No speculative API replacement is made.
- Added normal checks for all four straight sides and inward-fluid normal at the obstacle before factorization. Manufactured exact closed-surface gradient and symmetric forces must agree to roundoff. Their symbolic witness remains independent of NGSolve derivatives.
- Exact pressure probes require `MeshPoint.nr >= 0` and finite interpolation. Invalid probes are saved as unresolved, with explicit G2 pressure ineligibility. No nearest-node substitution or nudged primary measurement is used.
- Boundary names are sorted. `TaskManager` wraps case assembly/solves, with actual effective CPU concurrency explicitly unmeasured until timing qualification.
- A batch exit trap always attempts the robust summary for ordinary failures, saving stage exit codes and missing/failed case states. A fundamental dependency/API failure stops later expensive cases. A scheduler hard kill can still prevent finalization; partial stage files remain evidence.
- The summary adds nongating raw DFG literature gaps, reaction/surface differences at both integration orders, and exact pressure-probe status. These do not create reference admission.
- The strict linear relative residual criterion remains `1e-10`. Claude's proposed relaxation was speculative and is not adopted. No iterative-refinement change is introduced absent an actual failure.
- No MKL/intel-openmp dependency is installed: current pinned PyPI metadata declares OpenBLAS. The import/tiny inverse preflight establishes runtime availability rather than assuming another build's dependencies.
- `SOURCE_COMMIT.txt` will be generated from the exact committed source at staging, transferred explicitly and independently recorded in the attempt manifest. The prior staged `g1-smoke01` is preserved; the candidate is for a distinct `g1-smoke02` after re-review.

Checks performed without NGSolve execution: Python syntax, shell syntax, symbolic witness equality, and the missing-case summary path. No packages installed and no numerical allocation submitted.

The job name now starts with `ctol_` so the existing explicit-ID cancellation helper can validate it if cancellation becomes necessary. No alternate cancellation policy is introduced. The first smoke's exact submission command specifies `--gres=gpu:h200:1`, subject to the coordinator checking current availability; this is recorded in the manifest. The hard twenty-minute limit, CPU FE interpretation, and total allocation accounting are unchanged.

## Boundary differential-operator audit miss — readiness retracted before execution

An additional independent physics review identified a concrete missed API issue in the approved candidate before any job was submitted. Plain `Integrate(..., definedon=mesh.Boundaries(...))` selects boundary transformations in pinned NGSolve 6.2.2608. VectorH1 volume `Grad(uh)` and `div(uh)` cannot be assumed to evaluate there as the desired volume derivatives; the old diagnostics could fail or evaluate an inappropriate trace. Earlier readiness for those diagnostics is retracted. No numerical result existed, so no measured result is retracted.

Primary implementation evidence inspected:

- [Pinned Integrate region handling](https://github.com/NGSolve/ngsolve/blob/v6.2.2608/comp/python_comp.cpp#L3714), especially selection of `vb` from a boundary region.
- [Pinned GridFunction derivative dispatch](https://github.com/NGSolve/ngsolve/blob/v6.2.2608/comp/gridfunction.cpp#L1731), selecting only a differential operator supporting the transformation's region type.
- [Pinned BoundaryFromVolumeCF binding and documentation](https://github.com/NGSolve/ngsolve/blob/v6.2.2608/comp/python_comp_mesh.cpp), explicitly documenting adjacent-volume evaluation for boundary mechanical stresses.
- [Pinned wrapper implementation](https://github.com/NGSolve/ngsolve/blob/v6.2.2608/comp/meshaccess.cpp), mapping boundary points into the associated volume element.
- [NGSolve developer clarification](https://forum.ngsolve.org/t/exterior-trace-on-an-internal-material-interface-using-dx-skeleton-true/3879), applying this wrapper to GridFunctions/coefficient functions, not symbolic trial/test functions.

The repair applies `BoundaryFromVolumeCF` to the **GridFunction gradient and divergence only**, before building stress or multiplying by boundary normals. It fixes every surface-force, wall-shear, boundary-work and finite-divergence boundary term; volume weak forms and all thresholds are unchanged. Native H1 pressure/velocity boundary traces are retained. Boundary gradient arrays are now saved with the boundary rule, enabling later independent surface-force reconstruction.

The preflight now sets a tiny VectorH1 field exactly to $u=(x,y)$, checks its traced identity gradient and constant divergence on boundary quadrature, and integrates known gradient/symmetric-stress tractions independently on each side. This exercises the actual derivative-wrapper/normal/integration path before flow solves. It is not a substitute for the physical/MMS checks. These checks are implemented and syntax-checked, not yet executed with NGSolve.

Both staged attempts remain untouched. `g1-smoke02` is explicitly held; a new `g1-smoke03` may be staged only after focused diff review. No dependency installation, numerical job, equation change, or tolerance relaxation occurred.

## Executed G1 smoke failed at import — bounded dependency-layout repair

The first actual allocation is recorded in `ledger/6566869.json`; `artifacts/g1-smoke03/FINAL-SACCT.txt` is authoritative accounting. The JAX GPU/f64 check and exact wheel hashes passed, but `import ngsolve` failed inside Netgen's OCCT loader with `KeyError: 'tkernel'`. No finite-element preflight or flow case ran. The failed package tree, wheel inputs, full logs, source/config files and stage summary are retained byte-for-byte in checksum-verified archive chunks that are tracked and mirrored to GitHub. `RETENTION.json` gives recovery and exact hashes. Only the checksum-verified remote duplicates were removed.

`DEPENDENCY-LAYOUT-DIAGNOSIS.json` derives the missing-library evidence from the retained wheel and installed raw RECORD. The OCCT wheel places its libraries in the wheel data scheme, whereas the failed flattened target contains none of them. OpenBLAS uses package-relative files and is present; the pinned NGSolve config says `USE_MKL=OFF`. This is a packaging-layout failure, not a scientific or GPU result, and does not justify changing packages or tolerances.

The coordinator approved preparing repair counter one for a distinct `g1-smoke04`: use the same four pinned wheels with pip `--prefix` into the fresh job-owned `deps`, preserving the standard library/data layout. [Pip documents prefix installation](https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-prefix) as retaining `lib`, `bin` and other top-level folders. `--no-deps`, `--no-index` and `--ignore-installed` keep dependency resolution and shared-environment writes out of this operation. Isolated pip operation, no pip cache and the existing job-local temporary directory prevent user-home output. The designated interpreter is unchanged.

The installer checks every raw RECORD file for presence, prefix containment, size and declared SHA256, checks wheel-declared OCCT data paths and the TKernel/OpenBLAS libraries, and saves a layout inventory before importing. The batch process then adds only job-local prefix paths, verifies that Netgen/NGSolve actually import from that prefix, and repeats the GPU, f64 and highest-precision check after import. The repair is bound to its exact new attempt; a further failure stops and preserves evidence. Equations, meshes, controls, scientific thresholds and job limits are unchanged. This is prepared for exact diff review only; no repaired installation or resubmission has occurred.

## Native preflight failure: measured evidence and separate lifetime repair

The executed prefix-repair attempt is preserved in `artifacts/g1-smoke04/`, including exact allocation accounting, all installed RECORD/native hashes, successful import/GPU/f64/highest checks, and the stage-level segmentation-fault report. No flow case ran. A recursive job-directory search found no core file. A read-only attempt to inspect node core routing was blocked by SSH host-key verification, which was not bypassed. No native backtrace or per-call output exists for this attempt; its precise failing call and root cause remain unresolved.

The pinned source establishes a concrete ownership hazard independently of the crash. `Mesh.GetTrafo` transfers transformation ownership to Python; the transformation's point-call binding returns a mapped point without a keep-alive link to that transformation. `BaseMappedIntegrationPoint` stores a raw transformation pointer and does not own it by default. Thus `mesh.GetTrafo(el)(ip)` can leave the mapped point with a dangling pointer once the temporary transformation is released. The integration-rule indexing lambda instead returns its point by value, so the rule temporary was not identified as the same defect. The source files, hashes and exact URLs are preserved in `source-evidence/`.

The bounded candidate retains named transformation owners throughout every mapped-point access in the early geometry probe. Existing FOM export loops already retained named owners. Arrays that persist in export lists are now explicit owning copies; other mapped values become Python lists/scalars while their owners are live. Named finite-difference residual vectors likewise remain alive while their NumPy views are read. This changes ownership only, not the weak forms or numerical values requested from the solver.

The preflight adds flushed before/after markers around mesh generation, mapping, normal and gradient evaluation, UMFPACK factorization/solve and stress integration. The batch process enables Python fault handling from interpreter startup and unbuffered output. Detailed markers run only in the tiny preflight, while the same mapping helper remains quiet in full control cases. The separate `PREFLIGHT-REPAIR.json` records the native repair limit and distinguishes it from the successful dependency-layout repair. The prefix installer is unchanged except binding its fresh attempt to `g1-smoke05`.

Equations, controls, meshes, scientific thresholds and allocation caps are unchanged. A later pass would validate the corrected execution path, not prove where the earlier crash occurred. This candidate has only syntax/ownership checks; it has not been staged or run.
