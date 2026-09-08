# Frozen heat head transfer and efficient coarse output controls

This bounded development experiment implements the coordinator's approved transfer proposal. Both expanded-coverage checkpoints, spatial features, head parameters and training-code libraries stay frozen; it contains no new training and leaves the final cohort unopened.

The machine-readable declaration is `config-transfer.json`. It retains the restricted polynomial-boundary Gaussian family, original development draws and a separately seeded fresh development cohort. It rebuilds the full-field QR projection and weak operators for each requested mesh. The same Crank–Nicolson step, multistart fit budgets and previously checked relaxed gradient tolerance apply at every mesh. Complete compiled and modular query fields, latent states, counters and stopping reasons must agree before timing.

For every FOM, the initial output is an exact copy of the supplied requested-grid input. Coarse restriction happens on the host inside the timer; only that restricted field is transferred to the GPU. Direct-time sine-transform propagation supplies later outputs, physically aligned bilinear interpolation reconstructs them at every requested node, and the complete contiguous host output is constructed inside the timer. A ROM instead returns its actual fitted initial field, charging the full input transfer, exact QR compression, two-start nonlinear fit, evolution and all dense outputs. Neither path is given descriptors of the localized initial condition. Full arrays contain interior nodes; known homogeneous boundary values are identical and implicit.

The full-query timer records input restriction/transfer, device solve/readout and complete host output separately. Diagnostics, data generation, compilation, mesh assembly and archive compression are outside it. Every paired block burns in the GPU; method order reverses on alternate repetitions. All requested fields and errors come from that actual invocation. Identical same-grid/coarse solver aliases are timed only once. Different solver meshes are never deduplicated.

The complete nested continuum-spectral reference pair is retained. Its observed refinement difference is empirical evidence, not a rigorous certificate. Full requested-grid and shared observation-grid norms remain separate. For observed current-relative error $e$ and empirical relative reference discrepancy $\delta$, the diagnostic adjusted quantity is $(e+\delta)/(1-\delta)$. A strict physical upper bound remains unspecified. Same-grid discrete truth and its difference from the physical reference are recorded separately. The current reference norm, initial reference norm, absolute error and decay flags are retained at every output time.

The declared envelope has 1,152 measured invocations. Anticipated GPU peak is below 10 GiB, host peak below 16 GiB, archive 7–10 GiB, and runtime 15–35 minutes with a two-hour cap. These are planning estimates, not measurements. First-repeat complete output fields are saved; every repetition hashes its actual full field and saves an additional field if bytes differ. The spectral pair and all scientific output bytes will be retained in checked archive chunks smaller than the repository blob limit. Small metadata and reports are tracked directly; large extracted arrays may be omitted only as duplicate blobs. The tested restore helper validates every chunk, the joined archive, and member manifests and refuses to overwrite differing files.

Results must preserve all failed/nonstationary cases. Configuration eligibility uses every repetition and each cohort's empirical adjusted physical errors, with full-grid and common-grid decisions separately. Summaries use the median of case timing medians; paired ratios use the median of per-case ratios of those medians. No crossover against a same-grid FOM alone is a speed claim if a cheaper interpolated FOM qualifies.

## Plain-language glossary

- **Frozen head / bank / code library:** unchanged trained mapping from latent coordinates / unchanged learned spatial functions / stored training coordinates used to start input fitting.
- **QR projection / weak operator:** exact compression of full-field least squares / heat residual projected onto smooth test functions.
- **Intervals / requested grid / solver grid:** cells along each spatial axis / nodes at which all outputs are delivered / internal FOM mesh.
- **Passthrough / restriction / interpolation:** returning the supplied initial field / selecting physically coincident coarse nodes / reconstructing values between solver nodes.
- **DST / Crank–Nicolson:** sine transform for direct heat propagation / the ROM's fixed time-stepping formula.
- **Current-relative / initial-relative / absolute error:** discrepancy divided by the current truth norm / divided by its initial norm / physical area-weighted discrepancy without normalization.
- **Empirical adjustment / strict bound:** allowance based on observed reference refinement / proven upper error limit, which is not available here.
- **Case median / paired ratio / envelope:** middle repetition time for one physical input / FOM time divided by ROM time for that input / cheapest qualifying solver configuration.
- **Development / final cohort:** inputs used to assess this study / separately reserved confirmation inputs that remain unused.
