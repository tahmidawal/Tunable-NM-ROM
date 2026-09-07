# Poisson follow-up interpretation and next bounded options

Pilot02 is complete; generated findings and provisional status are in
`runs/pilot02/FINDINGS.md`. No additional job is submitted by this note.

Increasing smooth test-space size brings the deployed weak solve extremely close
to the best recorded stationary full-field head fit, while leaving the difficult
source above the smaller accuracy targets. The unrestricted frozen bank itself
also has a substantial approximation floor on that source. This separates a small
weak-truncation contribution from bank/head representation limitations on the
fixed development cohort. It does not prove the head oracle found a global minimum.

Query fusion preserves every tested field, latent and counter exactly, but its
paired gains are modest and do not produce a DST advantage. The next isolated
speed experiment is the coordinator's proposed tiny linear-system kernel ablation:
retain the original `ctol_tol` control and compare a validated in-kernel small
solver on the same normal equations and damping schedule. Test conditioning,
solution/gradient residuals, actual solved fields and counter changes explicitly.
Do not presume bitwise identity when changing numerical linear algebra. Preserve
all same-invocation physical metrics and full host input/output costs, including
unfavorable cases; a kernel improvement is not an accuracy improvement.

If smaller accuracy targets remain the priority, the evidence calls for improving
or changing the bank/head, rather than adding further weak modes to this fixed
checkpoint. Any larger inherited checkpoint or new training must be a separately
labeled selection/training study with its actual retained rank and provenance,
and must retain the compact frozen model as a control. Existing final cohorts stay
closed during this diagnosis and development selection.

## Plain-language glossary

- **Weak truncation:** error from keeping only a finite set of smooth PDE tests.
- **Bank / head / oracle:** learned spatial span / nonlinear coefficient map /
  diagnostic fit that can see the reference solution.
- **Stationary / global:** locally zero objective gradient / best possible fit
  over all allowed coefficient coordinates.
- **Fusion / kernel:** combining query stages in one compiled pipeline / small
  accelerator operation implementing a particular numerical task.
- **Normal equations / damping:** small linear equations formed from a residual
  Jacobian / regularization controlling the proposed nonlinear solve step.
- **Conditioning / counter / provenance:** sensitivity to numerical perturbation /
  solver work statistic / recorded source and checkpoint origin.
