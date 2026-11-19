# Representation and hyper-reduction reviewer notes

Condensed faithfully from `/root/reviewer_hyperreduction`, November 19, 2026. The reviewer formed independent priorities before comparing with the manuscript outline and performed read-only literature/data inspection. No new measurements or experiments were produced.

## Main recommendation

Add the saved derivative/operator-error panel. The substantive missing experiment, if sparse deployment is central, is a coherent qualification of the selected final configuration. Do not convert the completed dense Burgers span result into a nonlinear-head or hyper-reduced speed claim.

## Recommended analyses

**Width–test reliability map:** essential. Active width versus actual scalar test count; error, reference-projection distance, independent-shell residual and qualification panels. Use one common independent observer. Leave untested cells blank. Common-test width changes, within-width test changes and combined changes answer different questions. Saved source: `/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-width-final.json`.

**Field versus derivatives/operators:** essential supporting panel. Plot field, gradient, upwind-advection and Laplacian error against time, retaining medians and case ranges. Field error is initial-field normalized; each derivative/operator error uses its corresponding reference norm at that time. Inspect near-zero denominators. These are discrete-operator checks, not analytic-network or continuum derivative certificates. Definitions: `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-18-jcp-weak-visibility/experiments/jcp-weak-visibility/core.py:130`; exports: `analyze_width_results.py:181` in the same directory. Arrays were verified in the saved width analysis.

**Reached-state quadrature residual and Jacobian action:** essential if sparse deployment is claimed. Plot sample count or sample/test ratio against discrepancies using declared fixed scales and absolute errors; avoid normalization solely by near-zero converged residuals. Apply both maps to the same physically normalized tangent directions. Cover dense- and sparse-reached states and initialization, with actual unique evaluated stencil sites. Historical checks are partial for a newly selected coupled configuration.

**Operator versus nodes:** essential for off-mesh attribution. Same saved fields, common tests and norm; compare upwind, central differences, analytic derivatives at mesh nodes and sparse analytic integration. Separate quadrature approximation of a fixed operator from replacing the operator. Existing A1/A2 findings have preliminary physical-reference/solver caveats. Source: `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/report_numbers.json`.

**Time-step × quadrature interaction:** essential for a quadrature-centered article, otherwise conditional. Use a small error and dense–sparse discrepancy matrix; freeze bank/tests/tolerances/cases. Existing ladders and sealed decisions have mixed outcomes. An isolated good cell can reflect cancellation. Source: `/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-15-jcp-sealed-closeout/experiments/jcp-sealed-closeout/results/four-rule-decisions.json`.

**One-bank reuse:** essential for basic tunability. Width versus projection and trajectory errors, then cost versus error; identify points sharing a checkpoint and preparation repeated per width. Ordering-superiority claims need unordered controls; no-loss-versus-retraining claims need retrained controls. Current prefix accuracy is done; complete cost of the latest dense widths is not established.

**Mesh/output and storage/runtime crossover:** essential for efficiency claims. Output size against input/evolution/recovery, plus error; stored bytes and runtime against width in separate panels. Disclose rebuilds and avoid cross-GPU absolute-time curves. Storage success with slower runtime remains useful evidence. Existing scaling and tensor records are under `reports/2026-11-18-jcp-results-audit-assets/evidence/{scaling,tensor}.json`.

## Verified primary literature

[SNF-ROM author text](https://arxiv.org/html/2405.14890v2), [JCP publication](https://www.sciencedirect.com/science/article/pii/S0021999125002402): derivative Appendix B and coupled time/sample tables. [Kim et al. author text](https://arxiv.org/html/2009.11990), [JCP publication](https://www.sciencedirect.com/science/article/pii/S0021999121007361): sampled decoder/Jacobian and evaluation cost. [Romor author text](https://arxiv.org/html/2308.03396), [JCP publication](https://www.sciencedirect.com/science/article/pii/S0021999125000129): reconstruction/evolution/sampling and overhead. [LiCROM author text](https://arxiv.org/html/2310.15907), [SIGGRAPH Asia conference DOI](https://doi.org/10.1145/3610548.3618158): continuous linear neural bases, cubature and preparation. LiCROM is a conference paper; detailed numbering generally refers to author versions. The proposed maps/fidelity panels are reviewer adaptations rather than claims about identical published plots.

## Glossary

Residual is the tested equation error; a Jacobian action measures its local change under a specified perturbation. A reached state is visited during evolution, unlike a training snapshot. A stencil includes neighboring sites needed by a discrete derivative. Dense/sparse refer to integration or evaluation of the same specified target. A changed target operator is a separate mathematical change. A reuse frontier lists measured choices obtainable from one frozen model.
