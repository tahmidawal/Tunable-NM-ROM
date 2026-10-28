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

The accuracy orchestration retains only one full field model at a time. First it records all development projection floors and fingerprints, then reconstructs each eligible model for development and again for validation. Rebuilt operators, training projections and force coefficients must match their recorded fingerprints exactly. No model or quadrature choice is fitted to validation. Additional offline rebuild time and total pilot elapsed time are recorded. The synthetic lifecycle regression uses explicitly unadmitted arrays with only the population gate mocked inside the test; it verifies one live full model and that validation is opened after frozen choices are written. No synthetic admission file is created.

Development projection and solve records now include dimensionless velocity-gradient errors as required by the design: raw weighted L2 error, raw reference-gradient norm, relative denominator and scaled error. Scaling divides by the square root of dimensionless volume; the relative denominator floors the reference norm at 1e-12 times that scale. These are diagnostic fields with no added acceptance threshold. Reference population archives must contain `velocity_gradient` along with velocity and pressure and declare units `d(u/U)/d(x/D)`. Validation summaries also retain gradient and boundary-trace medians/worst cases.

The declared point-block layout now has a production consumer. It streams fixed-test arrays and assembles complete weak moments before whitening, while retaining only bounded small-prefix trial fields. Dense/block synthetic controls preserve fields, weak operators, direct residuals, Galerkin and force outputs. Fitting uses hash-retained derived RAM moment blocks with explicit capacity checks and full selected-column/target/timing evidence; no point subsampling or dense oversized-gradient fallback occurs. The newly bound source still requires independent review, qualified inputs and measured capacity release.

Native-batch03 independently passed all four scalar/vector volume/boundary value and
physical-gradient replays, full-rule test-space selection and retained sparse eigen
checks. Preparation now calls that accepted batched primitive for coefficient reload
checks, POD fields, training export, candidate TSQR and point-block export. The original
pointwise `evaluate` remains the independent oracle and handles explicit raw probes.
Full-output assembly checks original point coverage and the existing allocation cap;
no point/DOF/resource ceilings or physical admission requirements are raised here.

A serialization smoke initially still patched the former pointwise helper after this
integration and failed on its intentionally non-native stub; its patch now targets the
batched helper. The retained native array serialization then passes unchanged, including
dimensionless gradient conversion and all original points. This fixture tests storage,
while native-batch03 supplies the actual native evaluation evidence.
