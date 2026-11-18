# Burgers continuation: physical accuracy and complete query cost

Measured development evidence from the authorized continuation beyond A0. Completed numerical results are retained; active cells and unfinished qualifications are identified below. The required accuracy and speed objective has not been declared achieved.

## Contract and scope

Each physics query receives the same resident initial-field array and viscosity and returns one final field on exactly4096×4096 nodes. Errors are evaluated at all five prescribed evolved times against qualified independent references, normalized by the initial reference norm. The target is1% per trajectory including reference uncertainty. Projection diagnostics use known future fields only for assessment; PDE queries do not. Final unseen tests remain closed.

C1 replaces the biased coarse upwind operator with a separately qualified continuous sine weak Galerkin formulation and ETDRK4 time integration. FamilyA uses the registered TRAIN128/DEV32 prefixes; familyB initially uses its own TRAIN32/DEV8. This expands training coverage independently from the operator change. A0 checkpoints remain preserved and their selector branch remains closed.

## Matched complete-output timing

All times below include input restriction, initialization, input-dependent selection, PDE advancement, final dense decoding and synchronization. Timing uses two randomized blocks of five repetitions per case. Reusable full-policy cache setup and cold calls are reported separately; offline training is not hidden inside a warm-time speed claim. Different panels are separate allocations, so compare paired methods within a panel.

### Initial family-A panel

|Method|Worst physical error|With reference uncertainty|Median warm time (ms)|All accuracy controls pass|
|---|---|---|---|---|
|local_hybrid|0.5271%|0.5460%|20.627|True|
|spectral16_dt005|3.3864%|3.4053%|13.077|False|
|spectral32_dt01|0.1158%|0.1348%|8.197|True|
|spectral32_dt005|0.1156%|0.1346%|14.963|True|
|spectral64_dt005|0.0508%|0.0698%|14.858|True|
|spectral64_dt00125|0.0510%|0.0699%|53.581|True|

local_hybrid / spectral32_dt01 median time ratios by block: 2.529, 2.524. Paired case-bootstrap95% intervals: [2.487, 2.549], [2.504, 2.546]. Useful-speed gate: **False**; substantial-speed gate: **False**.

Reusable cache preparation: 7.606s. A warm method that is already slower has no positive break-even query count after adding nonnegative setup/training cost.

![Independent complete-output cost panels](2026-11-18-burgers-continuation-assets/accuracy-cost.png)

The dashed line is the1% target. Crosses indicate a failed or incomplete numerical qualification, not eligible matched-accuracy speed competitors.

## Capacity, integration and conditional-frame studies

|Compressed coordinates|Quadrature order|Time step|Worst error + uncertainty|Quadrature change|Half-step change|
|---|---|---|---|---|---|
|384|128|0.0125|0.5559%|0.0019%|0.0398%|

Only passing compression settings are shown here; every failed setting remains in the evidence snapshot. Compression changes the trial space. Smaller quadrature for the unchanged bank and the first positive empirical quadrature rules failed; DECISION-C1-INTEGRATION.md closes those recipes. Increasing training alone is not treated as an operator repair.

|Frame recipe|Global + local coordinates|Worst projection|Evolved finer-grid score change|
|---|---|---|---|
|diffusive_transferred|64 + 64|7.2153%|0.0088%|
|initial_aligned|64 + 128|6.8914%|0.0044%|
|initial_transferred|64 + 64|5.4090%|0.0058%|

The first conditional frame, geometry-only repair and newly aligned neural teacher all fail representation before rollout. The aligned teacher uses only training data. Its decreasing training objective does not establish development generalization. These checkpoints are preserved and this recipe is closed; no PDE or speed claim is inferred.

## Neural and non-neural controls

These local ablations hold global rank, selected patches and nominal coefficient count fixed. They are prepared-input PDE diagnostics, not complete-query timing winners. The local-POD teacher uses bilinear interpolation and is not claimed to be an optimal smooth basis.

|Local feature family|Worst projection|Worst PDE error|
|---|---|---|
|local_pod|0.4891%|0.7122%|
|neural|0.6744%|0.9331%|
|poly|0.6998%|0.8794%|

|Comparator|Training provenance|Worst PDE error|Scope|
|---|---|---|---|
|Historical direct bank rank128|frozen historical training|6.5718%|all development cases|
|Historical direct bank rank384|frozen historical training|2.1953%|all development cases|
|Historical direct bank rank512|frozen historical training|1.6411%|all development cases|
|POD rank128|matched training|18.6052%|all development cases; span-limited|
|POD rank256|matched training|8.5678%|all development cases; span-limited|
|POD rank384|matched training|4.8824%|all development cases; span-limited|
|POD rank512|matched training|2.6164%|all development cases; span-limited|

The historical latent head has worst finite-start optimized projection 5.6723%. It was first reproduced with its inherited algorithm in A0; the C1 continuous-operator variant is explicitly a different online algorithm, not a reproduction or matched-data retraining.

|Nonlinear manifold|POD truncation|Encoder error|Optimized projection|Max gradient residual|
|---|---|---|---|---|
|POD autoencoder|0.4638%|16.8440%|16.8396%|5.92e-10|
|Whitened wider repair|0.4638%|16.7492%|16.7167%|1.26e-07|

Finite-start latent optimization is not a proof of globally optimal manifold distance. Encoder reconstruction, optimized projection and actual PDE solves are distinct measurements.

|Autoencoder PDE diagnostic case|Worst evolved error|
|---|---|
|0|2.9652%|
|9|18.7507%|

The separately registered matched-data global coordinate-neural bank comparator is not yet complete. It is distinct from the completed nonlinear POD-autoencoder and historical frozen-bank comparisons.

## Allocation, limitations and reproducibility

Completed allocation: **2.41167 GPU-hours** across 38 terminal jobs. The registered finite cumulative ceiling is 8.35583 GPU-hours; it is an accounting limit for this batch, not a user-imposed total cap. Failed numerical recipes are included. Active/pending ledger entries: c1compresscost (7346653), c1bcomplete (7346814).

All owned jobs use isolated paths, A10080GB preflight, float64 and highest matrix precision. At most two owned GPU jobs run concurrently. The branch is `exp/2026-11-17-burgers-quadtree`, backed up through `codeonly/exp/2026-11-17-burgers-quadtree`; no implementation merge occurred. Raw outputs, trained weights, checksums, source commits and failures are retained. Large derived arrays remain explicitly remote-only where indicated by retention manifests.

No final-test, wider-seed, output-size scaling, five-dense-output or host-transfer superiority claim follows from these development panels. Those confirmation panels remain gated by a feasible accuracy/cost candidate. The bounded quadtree replaces refined parents and selects a fixed tree from initial data; it does not demonstrate online temporal refinement or state transfer. Failure of these finite recipes does not exclude other neural architectures, adaptive solvers or PDE regimes.

Reproduce report tables with the implementation lane’s `reports/build_report.py --publish`; numerical drivers, prospective amendments and `cluster/ledger.json` identify the inputs and jobs. The evidence snapshot beside this report and its SHA manifest make the reported values auditable.

## Glossary

- **Projection:** best fitted coefficients for a known reference field; a representation diagnostic, not a forecast.
- **PDE solve / rollout:** advance the prescribed initial field using the governing differential equation.
- **ROM / NM-ROM:** reduced-order model / nonlinear-manifold reduced-order model.
- **POD:** a linear basis fitted to training snapshots.
- **Encoder:** network that maps a field to latent coordinates; optimized projection additionally adjusts those coordinates.
- **Quadrature / integration order:** numerical sampling used to evaluate the weak PDE operator.
- **Half-step / finer-grid change:** change after refining time or spatial evaluation; separate from reference error.
- **Reference uncertainty:** conditional error estimate from independent convergent space/time ladders.
- **Worst error:** maximum across the prescribed evolved times and cases, normalized by each initial reference norm.
- **Warm query:** repeated solve after reusable compilation/cache preparation. Cold and setup costs remain separate.
- **Time ratio:** ROM time divided by eligible classical time; below one is faster.
- **Case bootstrap:** resampling whole trajectories to estimate uncertainty in the median paired ratio.
- **Coordinates / rank:** number of active reduced coefficients.
- **FamilyA / familyB:** the registered Gaussian and more localized boundary-compatible Burgers populations.
- **Development / final test:** data used for architecture selection / untouched data reserved for frozen confirmation.
- **Quadtree:** a square partition whose cells can be divided into four children.
- **GPU-hour:** one allocated GPU for one hour, including setup and failed recipes.
