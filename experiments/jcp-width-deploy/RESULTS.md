# Burgers width confirmation and sparse deployment closure

Final numerical results for the registered frozen-choice same-family confirmation and bounded developmental sparse bridge. The dense confirmation passes; neither sparse rule is admitted for deployment.

## Frozen confirmation

| Configuration | Qualified cases | Median primary error (%) | Worst primary error (%) |
|---|---:|---:|---:|
| Baseline R128/M2048 | 8/8 | 0.632154781 | 2.972355170 |
| Selected R512/M8192 | 8/8 | 0.066820426 | 0.203092149 |

All fresh cases improve: True. Median paired selected/baseline error ratio: 0.078259534. Ratio of worst errors: 0.068327012. Registered confirmation gate: True.

The primary metric is the maximum over every stored state including initialization, normalized by the original initial-field norm. The reference is the same-grid, same-step discrete FOM. Width and test count both change, so the joint gain is not attributed to width alone. This is an accuracy confirmation, with no cross-job cost claim.

## Sparse bridge

| Rule | Nonzero samples | Actual stencil sites | Held-out moment error | Residual error / initial norm | Relative Jacobian-action error | Initial-fit field difference | Admitted |
|---|---:|---:|---:|---:|---:|---:|---|
| checkerboard | 32513 | 65025 | 7.53940747e-05 | 7.05067828e-07 | 0.000337638111 | 0.000146582105 | False |
| three_quarters | 48896 | 65025 | 0.00140209782 | 9.40886966e-05 | 0.00510581073 | 0.000219402993 | False |

Both rules fail the initial-fit and site-economy gates. The larger rule also fails residual and Jacobian-action gates. Its bounded NNLS fit reaches the iteration cap; its optimizer does not report convergence even though the separately registered projected-gradient threshold passes. The checkerboard fit has a successful optimizer exit, but that does not qualify its initialization or physical evaluation cost. Exact nonzero-weight closure equals proposed-support closure; no pruning threshold is used.

The gate closes the registered continuation: no sparse rollout, time-step interaction or sparse speed claim is admitted. This is a negative result for these frozen supports and fit recipe, not an impossibility theorem for every quadrature. The successful dense wider model remains grid-bound.

## Derivative interpretation

The saved derivative panels retain large discrete Laplacian discrepancies despite small field error. These are not caused by zero reference denominators: the saved denominator audit records every reference norm and absolute error numerator. The derivative targets are the implemented discrete operators, not analytic network derivatives or continuum certificates.

## Evidence and plots

- [Fresh numerical audit](checks/fresh/analysis.json)
- [Sparse independent audit](checks/bridge/analysis.json)
- [Saved derivative and width plot data](figures/saved/plotted-data.json)
- [Fresh confirmation plot](figures/closure/fresh-width-confirmation.png)
- [Sparse bridge plot](figures/closure/sparse-bridge-qualification.png)

## Glossary

**R:** active bank width. **M:** actual scalar weak-test count. **Qualified:** passed registered reference, solve, array and environment checks. **Primary error:** largest original-initial-norm field error over the registered trajectory. **FOM:** full-order discrete model. **Stencil closure:** every distinct physical site needed for the exact possible upwind neighbors. **Jacobian action:** response of the residual to a physically normalized perturbation. **NNLS:** least-squares integration-weight fit constrained to nonnegative weights. **Cold start:** finding coefficients from the supplied initial physical field. **Development:** previously exposed cases used for the sparse audit. **Fresh confirmation:** new physical cases generated only after the method settings were frozen.
