# Numerical-mechanism reviewer notes

Condensed faithfully from the completed advisory review by `/root/reviewer_dynamics`, November 19, 2026. This agent formed priorities from the results and manuscript before comparing with the existing outline. No files or experiments were changed by the reviewer. These notes preserve recommendations and source tracing, not new measurements.

## Main recommendation

Strengthen the explanation of existing results before expanding experiments. Promote phase-versus-shape and derivative/physical-fidelity diagnostics to main-paper evidence. Saved scalar arrays already support those analyses; their trends have not been newly evaluated by this review.

## Recommended analyses

**Approximation versus evolution:** essential. Plot projection/best-found fit and evolved error versus time/dimension with matched norms. Same bank, physical initial field, frame freedom, tests, operator and solver qualify a causal head/span comparison. Keep native initialization and older reference-centroid fits separate. A nonlinear fitted distance is not certified global projection.

**Residual visibility:** essential. Use the same-state residual/error ratio scatter, physical-tangent singular spectra and actual correction outside the head-plus-frame tangent. Match row scaling and rank thresholds. A sensitivity improvement after subspace restriction is partly algebraic; actual field-error/correction evidence matters. Saved `reset_*.json` records contain the needed quantities. Example: `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-18-jcp-head-mechanism/experiments/jcp-head-mechanism/runs/p03b/output/case_000/reset_m584_dt0.01.json`.

**Phase versus shape:** high priority. The physical centroid is circular and weighted by squared velocity magnitude, not the learned frame parameter. `phase_displacement` is the wrapped predicted/reference centroid difference. `shape_aligned_relative_error` compares a Fourier-shifted predicted field to truth. This is reference-assisted diagnostic alignment, neither an online correction nor global registration. Plot phase, ordinary error and aligned error versus time. Examine circular-moment concentration before interpreting diffuse/multimodal cases; the reviewer did not find that concentration metric stored. Existing plot: `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-18-jcp-head-mechanism/experiments/jcp-head-mechanism/runs/p03b/output/figures/drift_diagnostics.png`.

**Physical/derivative fidelity:** high priority. NS records have energy, reference energy and normalized divergence. Burgers width records have gradient, upwind-advection, Laplacian and energy histories. Derivative errors use current reference-quantity denominators, whereas field error uses initial field norm. These are discrete metrics. Compare viscous energy decay to reference rather than enforcing constant energy. Data: `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-18-jcp-weak-visibility/experiments/jcp-weak-visibility/checks/width-results/analysis.json`.

**Width/test interaction:** essential. Show measured cells only, with error, projection and evolution-distance panels plus the retained transient regression. Separate common-test width changes from joint enrichment. Do not subtract error norms to manufacture evolution distance.

**Refinement:** essential supporting evidence. Reuse qualified temporal, tolerance, test and quadrature studies. Same-operator dense/sparse checks differ from changing the target operator. Keep nonmonotonicity and failures rather than fitting unsupported convergence orders.

**Long-horizon/perturbation growth:** conditional on stronger stability claims. A new paired perturbation study would use equal physical initial perturbations, compare ROM and FOM growth, and test several directions/magnitudes. Historical horizon studies exist but have reference/rule limitations. Finite-time amplification is not automatically a Lyapunov exponent or stability theorem.

## Verified primary literature

Lee–Carlberg, JCP (2020), [author text](https://arxiv.org/html/1812.08373v3) and [publisher](https://www.sciencedirect.com/science/article/pii/S0021999119306783): projection/evolution separation and conditional manifold analysis. Barnett–Farhat, JCP (2022), [author text](https://arxiv.org/html/2204.02462) and [publisher](https://www.sciencedirect.com/science/article/pii/S0021999122004107): field and physical-output validation. SNF-ROM, JCP (2025), [author text](https://arxiv.org/html/2405.14890v2) and [publisher](https://www.sciencedirect.com/science/article/pii/S0021999125002402): derivatives and time/sampling interaction. Carlberg–Choi–Sargsyan, JCP (2018), [author manuscript/DOI record](https://arxiv.org/abs/1711.11550): conservative reduced models, whose guarantees do not transfer to the current method. Publication metadata was checked; detailed sections refer to accessible author versions.

## Glossary

Physical tangent means a locally reachable field change normalized in a spatial norm. Visibility means sensitivity of tested equations to such a change. Phase means periodic physical displacement; shape error follows the specified reference-assisted alignment. A best-found fit is an optimizer result without global-optimum certification. Conditional work is required only for the stronger claim it tests.
