# 2D bank/POD driver

`pilot2d.py` implements the source-only accuracy pilot. The native interface control is a separate accepted interface test; no qualified training population or trained bank currently exists.

The preparation entry requires an exact-source release and a genuinely admitted adapter export for the nine fixed training Reynolds numbers. Training uses the unchanged registered recipe. The pilot evaluates bank and POD prefixes 2, 4 and 8 with their shared lift, pressure modes, supremizers and fixed weak test space. POD also retains its square mixed Galerkin and coefficient PCHIP controls.

The maximum-bank development projection gate precedes all bank PDE solves. Training-output positive quadrature fits use the registered support sequence; development reached-state operator, Jacobian, field and force checks select the first passing support. Both full and fitted solves must converge. The selected support is written before the validation loader is invoked. No validation refit is implemented. Failed construction, solver and quadrature arms remain explicit; unsampled POD remains available.

Each query returns its actual coefficient state, required outputs and work counts. `paired_queries` accepts separately reviewed FOM and ROM callbacks and retains full snapshots of each timed invocation for later accuracy scoring. There is no built-in substitute FOM and no performance claim. The FOM callback and production runtime package still require review once qualified reference data exists.

Small synthetic tests in `pilot2d_smoke.py` exercise both model constructions, all output contracts, direct/tensor weak-form agreement, Jacobians, direct symmetric traction, POD Galerkin/PCHIP, and coefficient-state retention with reused mutable buffers. These tests are pure NumPy and confer no physical admission. Run with the absolute local interpreter and one BLAS thread; they complete below one second.

`save_evidence` writes arrays and JSON references with hashes; failures and nonfinite diagnostics are explicit. All driver reports retain `physical_admission=false`: acceptance belongs to independent saved-result review, not the executable.

## Production preparation entry

`prepare2d.py --training-manifest ... --release ... --out ...` is a source-only entry for the admitted FE adapter. The release schema is `ns-obstacle-native-preparation-release-v1`, stage `G3-2d-steady-preparation`; it binds `preparation_digest()`, the exact training manifest, independent adapter review and reference admission hashes. It never creates an admission record. Before native import or output creation, the existing adapter checks the full training population, identical binary mesh and common-lift admission. It exports to `prepared/`, writes exact pressure-pole basis evaluations to `probes.npz`, and retains native evidence and a completion record with hashes. Partial failures remain in their fresh directory with their stage recorded. A completed execution still needs independent review and the separate pilot/training release. No production preparation has run yet.

`fom_outputs2d.py` is a source-only correctness evaluator factory for the reference lane's `SteadyFOMQuery` factory API. All evaluators bind its actual mesh instance and receive actual viscosity. It preserves exact mapped rules, pressure poles, dimensionless fields and physical symmetric stress. Its explicit per-point loops provide an independent correctness oracle only: they are not yet a tuned FOM timing baseline. Cached/batched native field evaluation and precontracted force functionals must be validated against this oracle before matched-output performance claims.
