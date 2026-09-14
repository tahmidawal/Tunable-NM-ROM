# Poisson neural-operator pilot

First-stage experiment source: common discrete dataset, empirical fine-reference
calibration and same-case diagnosis of the inherited selected ROM. These are pilot
comparisons with unmatched historical ROM training, not matched-data paper evidence.

`protocol.json` fixes the first-stage settings. `dataset.py` produces independent
calibration/train/validation seeds and rejects any final cohort. Every model case
contains exactly `input`, `target`, `parameters`, `times`; Gaussian descriptors
remain offline metadata. Training targets are explicitly discrete. Fine-reference
sidecars are evaluation-only; the empirical refinement margin is not a rigorous
continuum bound and is checked separately on every validation case.

`diagnose.py` compares full bank projection, augmented bank projection, multistart
best-found nonlinear-plus-linear fitting and the actual stationary solve on each
case/reference. The selected corrections are coefficient directions within the
learned bank, so the augmented bank has the same span; rank-revealing SVD checks
this instead of creating spurious QR directions. Oracle fits are not deployable
methods or certified global optima. Same-case comparisons are not additive error
attributions. Direct DST and a tolerance ladder of zero-start CG are timed in the
same allocation, with burn-in, complete fields and all repetition records retained.
No FNO speed ratio is available until shared paired evaluation is complete.

`cluster.py LABEL` stages exact committed bytes directly to the approved paralab
namespace and submits one one-hour A10080GB job. Calibration must pass before bulk
data. Sequential dataset generation and diagnosis follow; the job exits when done.
The lane's total authorized first-block budget remains eight GPU-hours. The
staging code never consults an unrelated cluster ancestor repository.

`checks/` contains bounded local smoke artifacts, which are never production
accuracy or timing evidence. The diagnosis smoke exercises both references,
projection rank, fits and all query paths. Source hashes there identify the smoke
version; subsequent checksum-manifest changes do not rewrite earlier artifacts.

## Glossary

- Bank: learned spatial functions with free linear coefficients.
- Head: the frozen network mapping a latent vector to bank coefficients.
- Correction: extra free coefficients analytically eliminated during solving.
- Stationarity: the normalized weak-objective gradient satisfies its threshold.
- DST: direct discrete sine-transform Poisson solver.
- CG: conjugate-gradient iterative Poisson solver.
- Oracle: diagnostic using a reference answer unavailable to a deployed method.
- Empirical refinement: comparison of two increasingly fine numerical solutions.
- Mesh: intervals per axis; nodal fields have one additional point per axis.

## Conditional matched-data capacity screen

The fresh independent validation pilot exposed a spatial-bank floor and a larger
head/representation gap, while converged online solves nearly reach the best-found
fit. The authorized follow-up uses `train_matched.py`, `training-protocol.json`
and `matched_cluster.py`. Both arms use bank/head hidden widths of 256, the same
Fourier features, latent dimension 16 and 32 training-fitted correction directions;
only the feature count varies between 128 and 256. The former is a common-width
matched-data control, not the historical checkpoint's exact architecture.

Fresh random weights and PCA scores derived solely from the common training fields
initialize both arms. Fixed joint warmup, free-bank, head and final joint phases
use equal update counts, source/point batches and random streams. Actual optimizer
time, every timing block, sample exposures, loss histories and every phase checkpoint
are retained; equal update count is not equal floating-point operation cost.
Validation is opened only by the subsequent diagnosis process. Correction directions
are fitted from the normalized training-field residuals, with no validation access.

The dataset is copied directly between cluster job directories with unchanged
indices and a recorded calibration-path relocation. Future freshly generated
indices contain a portable calibration copy; existing indices are never rewritten.

- **PCA:** principal component analysis, a linear training-field summary used to
  initialize the trainable latent codes.
- **Phase:** a recorded training stage with a fixed objective and parameter set.
- **Exposure:** an input case or spatial point sampled by a training update;
  repeated draws count as repeated exposures.
