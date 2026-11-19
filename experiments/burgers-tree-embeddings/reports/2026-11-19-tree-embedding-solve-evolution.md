# Burgers solve evolution and full-resolution spatial errors

Saved, qualified B4 PDE trajectories decoded on actual 4096 × 4096 grids. These are time-evolved solves, not independently fitted reference snapshots. The reference is a numerically qualified spectral solution, not an analytic exact solution. Error maps compare each model with that reference; a reference error against exact truth is unavailable.

[Download all full-resolution field and signed-error PNGs](2026-11-19-tree-embedding-solve-evolution-assets/full-resolution-images.zip). Each standalone PNG has one pixel per grid node. Overview figures are reduced for display. The underlying float64 arrays remain in the checksum-verified run output.

Our model here is the admitted direct local-coefficient tree. The nonlinear-child PDE route failed solver convergence and is not substituted into these plots. The global control is the POD–Fourier model, not a reproduced native neural baseline. Field colors share one scale within each case; signed-error colors share another scale across both methods and all times. Negative errors are blue, positive errors red. Rows at time zero show initialization.

## Representative pulse: dev000

![Reference, direct tree, global control and signed errors over time](2026-11-19-tree-embedding-solve-evolution-assets/dev000-evolution.png)

![Full-grid errors over time](2026-11-19-tree-embedding-solve-evolution-assets/dev000-errors.png)

|Time|Tree relative L2|Global relative L2|Tree maximum absolute error|Global maximum absolute error|
|---:|---:|---:|---:|---:|
|0.00|0.186869%|0.013018%|0.00387940|0.00012344|
|0.05|0.259661%|0.088977%|0.00727554|0.00262723|
|0.10|0.150363%|0.093330%|0.00409074|0.00190376|
|0.15|0.112945%|0.095686%|0.00234334|0.00175214|
|0.20|0.107873%|0.099604%|0.00201457|0.00161068|
|0.25|0.104106%|0.100987%|0.00166792|0.00145381|

The reference convergence allowance is 0.00006689 percentage points. The largest final-field discrepancy from the originally retained B4 output is 1.536e-16 in the current-reference norm.

## Worst direct-tree development case: dev013

![Reference, direct tree, global control and signed errors over time](2026-11-19-tree-embedding-solve-evolution-assets/dev013-evolution.png)

![Full-grid errors over time](2026-11-19-tree-embedding-solve-evolution-assets/dev013-errors.png)

|Time|Tree relative L2|Global relative L2|Tree maximum absolute error|Global maximum absolute error|
|---:|---:|---:|---:|---:|
|0.00|0.249163%|0.004247%|0.00345476|0.00004880|
|0.05|0.455470%|0.149395%|0.01518796|0.00524978|
|0.10|0.343652%|0.164750%|0.00952036|0.00503707|
|0.15|0.252705%|0.167813%|0.00897171|0.00438652|
|0.20|0.209477%|0.173601%|0.00569446|0.00387736|
|0.25|0.197278%|0.178864%|0.00407289|0.00337595|

The reference convergence allowance is 0.00005490 percentage points. The largest final-field discrepancy from the originally retained B4 output is 1.964e-16 in the current-reference norm.


[Browse all individual full-resolution images](2026-11-19-tree-embedding-solve-evolution-assets/index.html).

## Glossary

- **Relative L2 error:** the norm of the model-minus-reference field divided by the reference norm at the same time; computed using all grid entries.
- **Signed spatial error:** model value minus reference value at each location, in field units.
- **Maximum absolute error:** largest pointwise discrepancy anywhere on the complete grid.
- **Reference allowance:** empirical numerical convergence uncertainty, not an exact-error certificate.
- **Direct tree:** retained global parent plus independently adjustable local correction coefficients.
- **Global control:** one spatial basis formed from training POD directions and prescribed Fourier completion.

