# Choose plots that explain reduced dynamics

The most useful additions are **phase-versus-shape diagnostics for Navier–Stokes and field-versus-derivative diagnostics for Burgers**. Their computed values are already saved. This November 19, 2026 review recommends strengthening the presentation of existing evidence before commissioning more experiments; it introduces no new measured result. The principal remaining experiments concern untouched confirmation of a frozen choice and, if claimed, sparse deployment of the selected wider model.

Three advisory review agents examined numerical mechanisms, representation/hyper-reduction, and physical validation/performance. These are independent perspectives, not actual JCP referee reports or an official journal checklist. The numerical reviewer emphasized the meaning of drift and local observability; the hyper-reduction reviewer emphasized operator identity and coupled numerical errors; the validation reviewer emphasized physical fidelity, case variation and fair cost. Their recommendations complement the [manuscript outline](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-manuscript-outline.md), especially where it made physical diagnostics optional.

## Prioritize questions, then group them into figures

The catalogue contains **analysis types, not a requirement for twelve new experiments or twelve separate figures**. “Essential” means essential to the stated paper claim. “Replot” means reviewers verified computed values in saved records; it does not mean that the proposed interpretation has already been established. Historical studies count only where their model, operator, reference and scoring definitions apply.

| Analysis | Priority and placement | Current evidence status |
|---|---|---|
| Approximation versus evolution | Essential; main | Existing figures and qualified matched dynamics |
| Phase versus shape drift | High priority; main | Existing diagnostic figure and computed histories |
| Field, derivative and physical fidelity | High priority; main | Computed histories ready for assembly |
| Residual visibility and actual corrections | Essential for mechanism; main | Scatter, tangent spectra and correction diagnostics saved |
| Width–test interaction | Essential for tunability; main | Sparse measured configuration set; replot |
| Numerical refinement and interactions | Essential qualification; compact main, full supplement | Substantial historical controls; changed configurations need checks |
| Operator change versus node change | Essential for off-mesh attribution; main/supplement | Historical factorial evidence with reference caveats |
| Reached-state quadrature fidelity | Essential if claiming sparse deployment | Partial historical checks; final coupled configuration not established |
| Complete-query accuracy–cost frontier | Essential for efficiency; main | Selection and confirmation records available separately |
| Output scaling and storage crossover | Essential if claiming grid-independent work; main/supplement | Existing scoped component and storage records |
| Coverage, case variation and failures | Essential for generalization; main plus supplement | Existing case records; new frozen-choice confirmation remains |
| Initialization, training and horizon sensitivity | Supporting or claim-dependent | Some saved comparisons and historical studies; stronger claims need targeted work |

### Approximation and evolution answer different questions

Plot physical error against time and online dimension in adjacent panels: span projection floor, best-found head reconstruction, and evolved trajectories. Label latent dimension, active prefix width and full bank width separately. Use a common mass norm and denominator within each comparison. The causal control needs the same bank, physical initial field, frame freedom, operator, tests, time grid and solve accuracy; native initialization is a separate practical comparison.

A head with worse reconstruction but better evolution supports constrained dynamics. A trajectory near its projection floor instead points to insufficient representation. Best-found nonlinear fits are not global minima, and older reference-centroid fits must not be pooled with new common-initial-state runs. Existing [matched evidence](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments.md) supports assembly now. The approximation/evolution distinction is central in [Lee–Carlberg’s author manuscript](https://arxiv.org/html/1812.08373v3).

### Phase and shape reveal what “drift control” means

Use time on the horizontal axis for ordinary field error, wrapped physical-centroid displacement, and centroid-aligned field error. Report displacement in declared domain units, preferably fractions of periodic-domain length; retain the original initial-field denominator for both field errors. Show paired head/span curves and case ranges.

The saved centroid is circular and weighted by squared velocity magnitude: **it is not the learned frame parameter**. Alignment Fourier-shifts the predicted field using its centroid difference from truth. It is reference-assisted diagnosis, not an online correction or globally optimal registration. If alignment removes the gap, the benefit is predominantly phase control; if a gap remains, shape evolution also differs. Neither outcome has been inferred here from the saved scalar names alone.

The [existing drift figure](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-18-jcp-head-mechanism/experiments/jcp-head-mechanism/runs/p03b/output/figures/drift_diagnostics.png) already contains phase/shape panels. Its additional manifold-distance panel retains the best-found/nonstationary-fit caveat. Check circular-moment concentration before interpreting centroids for diffuse or multimodal fields; that reliability quantity was not among the stored metrics inspected. This particular plot is the reviewers’ method-specific recommendation, not a claimed reproduction of a published figure.

### Derivative and physical fidelity expose errors hidden by field norms

For Burgers, place time against field, gradient, upwind-advection and Laplacian errors in aligned panels. Compare common-test widths first, then test enrichment at fixed width. The denominators matter:

$$
e_u(t)=\frac{\|u-u_\star\|}{\|u_\star(0)\|},\qquad
e_D(t)=\frac{\|D_h(u)-D_h(u_\star)\|}{\|D_h(u_\star)\|},
$$

where $D_h$ denotes the stated discrete gradient, advection or Laplacian. Despite a broad source docstring, derivative/operator denominators use the corresponding reference quantity **at each time**. Inspect near-zero denominators and retain absolute errors where needed. These are discrete-operator diagnostics, not continuum-derivative certificates. The [computed arrays](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-18-jcp-weak-visibility/experiments/jcp-weak-visibility/checks/width-results/analysis.json) and [metric implementation](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-18-jcp-weak-visibility/experiments/jcp-weak-visibility/core.py:130) support replots.

For Navier–Stokes, add reference and ROM energy normalized by initial reference energy, signed energy discrepancy, and the saved divergence ratio $\|\nabla\!\cdot u\|/\|\nabla u\|$. The mechanism case is unforced and viscous: compare reference decay, not constant energy. Translation can preserve energy, so energy alone does not explain drift. A shared-scale field/error snapshot for a declared representative and difficult case is useful supporting context. [SNF-ROM Appendix B](https://arxiv.org/html/2405.14890v2) motivates derivative checks; [Barnett–Farhat’s flow studies](https://arxiv.org/html/2204.02462) motivate physical outputs alongside global error.

### Residual visibility must connect to the actual error

Retain the scatter with span/head weak-residual norm horizontally and span/head physical error vertically. Pair it with tested-Jacobian singular values on mass-orthonormal physical tangents and the fraction of the actual correction outside the head-plus-frame tangent. Use identical states, tests, row scaling and rank cutoffs; include the independent residual observer.

Raw latent-coordinate conditioning is not intrinsic physical sensitivity. Restricting a tangent space can improve its minimum sensitivity for algebraic reasons, so spectra alone do not establish the mechanism. Connect them to actual corrections and field errors. If that connection disappears, narrow the explanation. [Retained probe records](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-18-jcp-head-mechanism/experiments/jcp-head-mechanism/runs/p03b/output/case_000/reset_m584_dt0.01.json) already contain these quantities; their use as mechanism diagnostics is our extension of residual-based analysis, not a uniform stability proof.

### A sparse width–test map is more honest than a smooth phase diagram

Place active width $R'$ horizontally and actual scalar test count $M$ vertically. Use panels for maximum field error, initial projection, distance from the reference’s mass projection, independent-shell residual, and qualification. **Leave untested cells blank.** Accompany the map with the retained fixed-test transient regression. Separate common-test width changes, within-width enrichment and joint changes.

Better projection with worse evolution identifies a dynamics/test limitation; joint improvement cannot be credited solely to width. Compute evolution distance from field differences, not by subtracting scalar norms. The [frozen width results](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-width-final.json) support this replot. [Romor et al.](https://arxiv.org/html/2308.03396) motivate separating reconstruction, evolution and sampling; the two-dimensional map is a reviewer-proposed synthesis.

## Numerical credibility requires distinguishing errors that can cancel

### Refine the ingredients without changing the question

Plot physical error and successive-solution differences against time step, solver tolerance, test count and quadrature size, with reference uncertainty indicated. Reuse the applicable [historical verification studies](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-results-audit-and-priorities.md); they are not all missing. Keep failures and nonmonotonic behavior visible rather than fitting an unsupported convergence slope.

If quadrature is central, add a small time-step–sample-count matrix of error and dense–sparse trajectory discrepancy. Freeze representation, tests, solver tolerance and cases. A plateau surviving temporal refinement supports the rule; an isolated low-error cell suggests cancellation. Preserve both existing space-and-time and space-only reference decisions. [SNF-ROM’s author-version tables](https://arxiv.org/html/2405.14890v2) explicitly motivate this interaction check. A full grid for every benchmark is unnecessary.

### Separate operator attribution from integration accuracy

On identical fields, compare weak-advection discrepancies for upwind, central-difference, analytic-at-mesh-node and sparse-analytic evaluations against the declared dense target. Use mesh spacing on the horizontal axis, consistent norms vertically, and a companion operator/rule table. If analytic derivatives at existing nodes explain the improvement, point placement is not required to explain that particular result.

The [historical mechanism records](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/report_numbers.json) can be replotted within their preliminary reference and solver limitations. Sparse versus dense evaluation of one operator is a different comparison from discrete versus continuous/hybrid operators. The published derivative/sampling discussion in [SNF-ROM](https://doi.org/10.1016/j.jcp.2025.113957) supports asking this question; the factorial design is a project-specific attribution study.

### Sparse deployment needs residual and Jacobian-action fidelity

Plot quadrature sites $m$, or $m/M$, against scaled residual discrepancy, Jacobian-action discrepancy, rollout difference and solve qualification. Use declared scales and absolute errors, not only division by a nearly zero converged residual. Apply both Jacobians to the same physically normalized tangent directions. Check states reached by dense and sparse trajectories and the initial-fit path. Count actual distinct evaluated sites including stencil neighbors.

A small residual discrepancy with large action discrepancy can explain changed solver behavior; a field-only pass does not qualify every ingredient. Historical checks exist, but the [deployment boundary](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments.md) remains for a newly selected wider configuration. The [frozen rule decisions](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-15-jcp-sealed-closeout/experiments/jcp-sealed-closeout/results/four-rule-decisions.json) must retain their distinct reference criteria. [Kim et al.](https://arxiv.org/html/2009.11990) motivate sampled decoder/Jacobian accounting; the proposed fidelity plot is an additional diagnostic.

## Performance and validity plots must preserve the measurement contract

### Show the accuracy–cost frontier and confirmed comparisons separately

Use complete-query milliseconds horizontally and cohort median or worst casewise maximum evolved error vertically. Keep initial error separate. Plot head, span, POD and classical families, using **candidate-selection repetitions only** for the selection frontier. Present selected confirmation measurements separately, with paired casewise ratios $T_{\rm baseline}/T_{\rm ROM}$ and a reference line at one.

All ratios need a common hardware allocation and output contract. Eligibility is the stated cohort median-and-worst rule, not per-case equal-error matching. A mixed casewise result qualifies a favorable aggregate; the current POD near-tie and coarse-classical losses stay visible. [Selection records](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p04-selection.json) and [confirmation results](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p04.json) support assembly. Case range, timing dispersion and uncertainty in a trained model are different quantities. [Romor et al.](https://arxiv.org/html/2308.03396) provide a relevant overhead-aware comparison precedent.

### Separate output work and storage from reduced evolution

Plot requested degrees of freedom $N$ against input preparation, reduced evolution and output recovery, with error alongside. Plot stored tensor/quadrature bytes and runtime against active width in separate panels. Use directly measured components; never manufacture one by subtracting unrelated jobs or join absolute times across different GPUs. Disclose rebuilt normalization, operators and rules.

Flat reduced evolution with growing output cost supports a limited internal-computation claim. Storage improvement with slower execution remains an informative crossover. Existing [scaling](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-results-audit-assets/evidence/scaling.json) and [tensor-counterexample records](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-results-audit-assets/evidence/tensor.json) are available. [LiCROM](https://arxiv.org/html/2310.15907) separates continuous basis preparation from sampled simulation; continuity alone is not a new efficiency result.

### Show where the cases lie and which comparisons fail

Plot parameter locations labeled training, development and untouched confirmation; accompany them with casewise maximum error against meaningful physical parameters. Distinguish interpolation, extrapolation and mesh transfer. Add paired error scatter or an empirical cumulative distribution, plus a small case/configuration qualification matrix.

Separate setup failures, invalid references, solver nonconvergence and finite inaccurate predictions. Failed development repairs are not estimated method-failure probabilities; repeated invocations are not independent cases. Existing records support distributions, but the latest mechanism/width cohorts are exposed development data. New confirmation must follow a frozen choice, not simply fill an empty plot. [Kim](https://arxiv.org/html/2009.11990), [SNF-ROM](https://arxiv.org/html/2405.14890v2) and [Romor](https://arxiv.org/html/2308.03396) motivate parameter-resolved validation.

### Add initialization and learning studies only for the corresponding claim

Plot initial reconstruction error against subsequent maximum error for common-state and native-initialization arms, using saved [initial-fit and trajectory records](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/head-p03b.json). Complete-query totals include fitting, but do not establish a separately replicated fitting-cost breakdown. A practical initialization gap can narrow deployment conclusions without undoing the matched evolution result.

Training-data curves require error versus training trajectories with independently trained seed traces; initial-condition cases and timing repetitions cannot substitute. Historical seed and longer-horizon studies exist, with qualifications. New learning curves, physical-perturbation growth studies or longer integrations are justified by data-efficiency or stability claims, not as automatic decorations. Energy agreement does not prove conservation; [conservative ROMs](https://arxiv.org/html/1711.11550v3) enforce additional structure. A quadratic or other nonlinear baseline becomes important for claims of uniquely beneficial neural nonlinearity, as illustrated by [quadratic manifolds](https://doi.org/10.1016/j.jcp.2022.111348) and [PROM-ANN](https://doi.org/10.1016/j.jcp.2023.112420).

## Assemble the main paper before extending the experiments

Use a compact sequence of result figures: approximation/evolution; residual mechanism with phase/shape; width/test interaction; derivative fidelity; numerical qualification; operator/quadrature qualification; complete-query cost; and coverage/case variation. Put output/storage scaling in the main paper when efficiency across meshes is central, otherwise in the supplement. The architecture diagram is separate. These are grouping recommendations, not a prescribed figure count; a crowded figure should be split, and repeated curves can move to the appendix.

Compared with the previous outline, promote focused phase/shape and derivative diagnostics instead of adding a broad application collage. Give operator attribution and rule sufficiency distinct panels. Keep the selection frontier separate from confirmation, and put coverage/failures near the main accuracy claims. Retain full solver histories, additional physical quantities, initialization sensitivity and applicable seed/horizon studies in supporting material.

**First, assemble saved evidence:** phase/shape, derivative fidelity, measured width/test cells, case distributions and the selection/confirmation cost views. Check centroid concentration, near-zero denominators, label definitions and case-selection rules before interpretation. **Second, freeze the intended claim and configuration.** **Third, run only the missing confirmation or sparse-deployment checks that claim requires.** No new broad PDE suite, universal theorem, rectangular parameter sweep, large training campaign or exhaustive baseline catalogue is automatically required. The separate adaptive-tree project remains outside this review.

The reviews used primary publication records and accessible author manuscripts. Figure/section numbering cited in the notes refers to those versions and can differ from the final journal articles. The proposed combined plots are reviewer designs inspired by the literature, not claims that each cited paper used the same plot. Focused source notes are preserved in [numerical review](</home/tahmid/Dev/Tunable-NM-ROM-Claude/research_notes/JCP reviewer analyses and plots/dynamics-review.md>), [hyper-reduction review](</home/tahmid/Dev/Tunable-NM-ROM-Claude/research_notes/JCP reviewer analyses and plots/hyperreduction-review.md>) and [validation review](</home/tahmid/Dev/Tunable-NM-ROM-Claude/research_notes/JCP reviewer analyses and plots/validation-review.md>).

## Glossary

**Bank, span, head:** spatial functions, their arbitrary linear combinations, and a nonlinear map restricting their coefficients. **Prefix width:** number of leading ordered bank functions used. **Projection floor:** best field error available in a specified linear space. **Trajectory:** solution through time. **Weak tests:** weighted measurements of equation error. **Residual visibility:** how strongly those measurements detect physical changes. **Tangent/Jacobian:** locally reachable changes/local sensitivity. **Mass norm:** spatially weighted field norm.

**Phase/shape:** position of a structure/its form after specified alignment. **Circular centroid:** a centroid suitable for a periodic domain; concentration measures whether that location is well determined. **Quadrature/hyper-reduction:** numerical integration/reducing equation-evaluation work. **Jacobian action:** applying a derivative map to a specified perturbation. **Stencil:** neighboring samples used by a discrete operator. **POD:** a snapshot-derived linear basis. **Frontier:** best observed accuracy–cost choices within a tested candidate set. **Complete query:** all work from declared input to requested output.

**Qualification:** passing declared numerical or measurement checks. **Development:** cases available during method design and selection. **Timing confirmation:** new repetitions after comparator choices are frozen, using the existing cases. **Fresh-cohort confirmation:** untouched physical cases evaluated after the method is frozen; the existing cost confirmation is not this kind of held-out test. **Replot:** assemble already computed values; no new PDE solve is implied. **Historical/conditional:** evidence tied to an earlier configuration/work required only for a chosen stronger claim. **Median/worst:** middle/largest casewise statistic under its stated time aggregation. **Case/seed/repetition:** a physical input/a trained-model randomization/a repeated measurement. **Cohort:** the group of physical cases being compared. **Empirical cumulative distribution:** the fraction of observed cases below each error threshold. **Paired ratio:** a comparison computed within the same case before summarizing across cases. **Main/supplement:** central argument/supporting detail; table priorities express editorial recommendations, not journal rules.
