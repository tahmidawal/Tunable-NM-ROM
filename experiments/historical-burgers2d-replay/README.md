# Replay of the original Burgers tensor comparison

This cell restores the archived Burgers comparison at the user's request. New measurements are pending; the historical source and checkpoint payloads are frozen copies, while timing-output capture is instrumented explicitly below.

The four archived stages contain the same numerical source bytes. `HISTORICAL-SOURCES.json` records their original source commit, file hashes, checkpoint hashes and original locations. Each mesh reuses its own original $K=16$, $R=64$, $M=64$ checkpoint. At $N=512$, the original run trained its checkpoint in the job; this replay loads that exact resulting checkpoint. The encoder uses the same independent seed, training state selection, regenerated initial fields, fitted nodes and update count. Data are regenerated on the cluster.

The primary timer starts with the full initial field already on the GPU and ends with all 51 full decoded fields on the GPU. It includes initialization, 50 latent evolution steps of size $0.005$, and decoding. Both host transfers are outside this historical timer. The separate reduced evolution timer measures only the middle part. The FOM is the original tolerance-terminated Newton method with dense sine-matrix Helmholtz preconditioning, on the same mesh and timestep. Its original Newton/linear tolerance ladder selects the cheapest configuration with cohort mean current-state relative error no larger than each ROM arm's. This is the original comparison, not a claim against every possible FOM algorithm.

The arms are the original full-grid upwind residual, sampled residual, tensor residual, and archived learned-node control where available. The tensor's fixed-backward stencil requires its original positivity and full-upwind discrepancy audits; negative decoder undershoots are retained and reported. The selected old mesh label counts nodes per axis, including boundaries.

Instrumentation changes are restricted to `code/sep_b2d_tensor.py` and `code/sep_common.py`:

- Keep the final actual timed output of each paired ROM/FOM call and calculate its per-time errors after timing. The timer boundary and numerical function are unchanged.
- Retain per-time errors for the original FOM ladder, in addition to its original means.
- Burn the GPU before every FOM and paired timing block. Preserve every repetition and alternate invocation order as before.
- Save the fitted initialization network. Save sampled paired fields for every case, and full tensor/FOM/truth fields for predeclared case zero. Compression happens after the timed block.
- Treat large captured-constant warnings as failures rather than suppressing them. Record explicit replay commit and checkpoint hashes; do not inspect ancestor git repositories on the cluster.

`prepare.py <historical_attempt>` creates a private staged payload and hash manifest. `run_ladder.py` runs all four meshes sequentially inside one GPU allocation, so within-ladder times share one physical GPU. Real runs require the cluster absolute Python environment, GPU preflight, float64 and highest matrix-multiplication precision. Inputs are in `in/`; results are collected under `runs/<attempt>/out/n<N>/`. No old result or checkpoint is edited.

The replay keeps mesh-specific learned weights; it does not measure frozen-weight transfer. Report the full cohort mean, median and worst per-time error, timing medians, retained repetition arrays and outlier counts. Historical and newly replayed absolute times belong to different allocations and must not be subtracted as an isolated algorithm effect.

## Plain-language glossary

- **FOM / ROM:** the full grid solver / the solver using a learned reduced family.
- **K / R / M:** latent coordinates / learned spatial features / projected weak equations.
- **Checkpoint / encoder:** saved decoder weights / a learned initial guess from sampled field values.
- **Full / sampled / tensor:** evaluating every grid point / using fitted weighted nodes / using precomputed quadratic coefficients in the residual.
- **Weak residual:** the PDE mismatch projected onto smooth test functions.
- **Newton / Helmholtz / sine preconditioner:** nonlinear equation correction / the linear diffusion-plus-identity operator / the original dense transform used to approximately invert it.
- **Paired / repetition / burn / outlier:** alternated ROM and FOM calls / one retained measurement / premeasurement GPU activity / a retained unusually long measurement.
- **Current-state relative error:** field error divided by the reference field norm at that same time.
- **Cohort / case / mesh:** the eight original seeded test trajectories / one trajectory / the spatial grid.
- **Captured constant:** an array embedded in a compiled function instead of passed as a runtime input.
