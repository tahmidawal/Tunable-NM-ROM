# JCP manuscript outline and evidence map

This is a drafting plan prepared on November 19, 2026, using the numerical evidence finalized on November 18 and qualified historical studies. The new campaign results are completed **development evidence**, not fresh-test confirmation; proposed figures, theoretical arguments and remaining experiments below are explicitly distinguished from completed work.

**Recommended title:** *Tunable nonlinear-manifold reduced models: approximation, weak-residual dynamics, and computational cost.* A shorter alternative is *Reliable tuning of continuous nonlinear-manifold reduced models*.

The paper should argue that **representation capacity, restrictions on reduced dynamics, residual visibility and integration accuracy must be controlled separately**. A compact head can evolve more accurately than its containing linear span; a wider span needs sufficiently informative tests; inexpensive reduced evolution does not by itself imply inexpensive complete field queries. This gives the journal article a connected numerical argument, with both improvements and counterexamples.

Treat this as a **consolidated journal extension** of the existing framework. The architecture is already in the conference manuscript. The new contribution is the controlled explanation, qualification and comparison of its behavior—not the introduction of continuous neural bases, nonlinear coefficient maps or cubature as individual ideas.

## Claims to organize the manuscript around

1. **Constrained dynamics:** under matched controls on the original Navier–Stokes family, the nonlinear head reduces trajectory error relative to its own full span; same-state probes explain how minimizing the tested residual can favor physically inaccurate directions.
2. **Qualified tunability:** frozen ordered prefixes provide different representation capacities without retraining neural weights, but reliable evolution also depends on the test space. Best-approximation nesting does not imply monotone rollout accuracy.
3. **Qualified integration:** continuous evaluation enables alternative integration rules, whose adequacy depends on the target operator, reference, states and numerical configuration. Storage savings and runtime savings are separate findings.
4. **Complete cost:** comparisons count the declared input-to-output task and use eligible classical/POD alternatives. The completed evidence defines useful regimes and limits; it does not establish universal superiority.

The quantitative anchors below are generated from retained JSONs. They are drafting aids, not a suggestion to combine unlike error definitions in one paper table.

| Evidence | Generated numerical anchor | Required interpretation |
|---|---|---|
| Matched head/span, M=584, $\Delta t=0.01$ | Median maximum evolved error: 0.0967% / 0.6697%; median paired span/head ratio 7.06; head better in 16/16 cases | Original-family learned bank; common initial field; initial-field-normalized norm; completed development evidence |
| Frozen Burgers span and tests | Worst initialization-inclusive error: 4.2979% at R′=128, M=2048, to 0.3918% at R′=512, M=8192; 6/6 cases qualify | Dense discrete-upwind dynamics at fixed time step; width and test count both change; not head or sparse-cost evidence |
| Cheapest admitted head versus eligible alternatives | Complete-query medians: head 26.51 ms; POD 26.62 ms; coarse classical 23.95 ms | Same allocation; cohort median-and-worst accuracy eligibility; near POD tie and no coarse-classical advantage |

## 1. Introduction: explain the numerical problem and the journal addition

**Argument.** Compressing a field is only the first problem. The reduced model must evolve accurately, observe relevant equation errors, and deliver the requested output economically. Introduce these as the questions the paper answers, then state the four contributions above in bounded language.

**Related-work progression.** Start with nonlinear-manifold projection and its approximation/evolution distinction, then continuous neural fields, nonlinear maps inside a linear basis, and hyper-reduction. Bring the closest precedents together: Lee–Carlberg for manifold projection, LiCROM for continuous banks and cached integration values, PROM-ANN for nonlinear coefficient corrections, and SNF-ROM for neural-field evolution. Use Kim et al. and Romor et al. to motivate decoder/Jacobian work, sampling sensitivity and complete cost. Cite quadratic manifolds as a non-neural alternative. The paper need not reproduce every comparator to make a scoped claim, but should not claim that a neural head is uniquely necessary. [Core references](#core-literature-and-its-role)

End the introduction with the prior-work/extension distinction. Avoid an abstract organized around a headline speedup. Write the abstract last: problem, inherited framework, new controlled analyses, principal measured outcomes, and their scope.

## 2. Method: specify what is frozen, solved and changed

**2.1 Continuous bank and two online representations.** Define the boundary lift, bank, optional moving frame and physical norm. In a common frame,

$$
u_h(x;z)=b(x)+G_R(x)h_\theta(z),\qquad
u_{R'}(x;a)=b(x)+G_{R'}(x)a.
$$

Distinguish full bank width $R$, active span width $R'$, head dimension $k$, number of tests $M$, quadrature sites $m$, requested output degrees of freedom $N$, and any frame unknowns. **Retraining-free runtime tuning currently concerns the linear prefix; changing $k$ is generally a separate training choice.** Prefix-specific operator or quadrature setup still costs work.

**2.2 Fixed tested residual and online solve.** State the actual objective,

$$
\min_q\frac12\|S\widehat r_{M,m}^{\,n}(u(q))\|_2^2,
$$

where $q$ contains the active coordinates and any solved frame, $S$ is the recorded row scaling, and $\widehat r_{M,m}^{\,n}$ approximates the chosen weak residual. Specify time discretization, initialization, termination and returned-state checks. Explain its relation to projection methods without importing full-residual LSPG guarantees unchanged.

**2.3 Operator and integration variants.** Give separate labels to dense discrete-upwind evaluation, sampled evaluation of that same discrete operator, and continuous/hybrid weak operators. A quadrature lattice is an integration device; a change from a stencil to analytic derivatives also changes the target operator. State which variant each experiment uses.

**2.4 Offline and online work.** Present training/ordering, operator construction, quadrature preparation, initial projection, evolution and output recovery in **Figure 1**, a new architecture diagram. **Table 1** should list benchmark, bank type, solver/operator, training/development/test population, reference, error norm, horizon and input/output contract. Reuse the [actual methodology](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-13-manuscript-rebuttal/paper_revision/main.tex:311), explicitly attributed as the inherited method.

## 3. Analysis and verification: establish what the experiments can mean

**3.1 Structural facts.** For a fixed common frame,

$$
\mathcal M_k\subseteq b+\operatorname{range}(G_R).
$$

Allowing both models the same frame family preserves inclusion after taking the same union over frames; that union need not be a linear span. Therefore the head cannot beat the full span's globally optimal reconstruction in the same norm. Its advantage can concern evolution, compression or conditioning. Nested prefixes likewise give nonincreasing optimal projection error, not guaranteed improvement in time-dependent error or cost. Include concise proofs; label nonlinear fitted distances as best-found upper bounds.

**3.2 Local residual visibility.** Introduce the tested PDE Jacobian on a mass-orthonormal tangent basis. Small singular values identify physically normalized perturbations weakly detected by the objective. Explain the assumptions needed for a local error estimate, including a positive lower sensitivity bound and controlled nonlinear remainder. Existing sampled diagnostics do **not** constitute a uniform stability theorem.

**3.3 Numerical qualification.** State reference refinement, tight/tighter solver comparisons, action/derivative checks, and dense/sparse consistency. Distinguish these from accuracy rankings. Use the completed historical time-order and error-intervention studies; keep nonadditive interactions instead of inventing additive error percentages. Heat and Poisson can provide concise correctness context, with detailed ladders in Appendix A. [Historical analysis ledger](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-results-audit-and-priorities.md)

**Status:** structural statements can be written now; numerical checks exist for specified configurations. A stronger stability theorem remains optional unless the title or conclusions promise one.

## 4. Navier–Stokes: distinguish representation from trajectory drift

**4.1 Approximation versus evolution.** Begin with full-span projection, best-found head fit and evolved trajectories on the same bank. Use the [existing fit/trajectory panel](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-results-audit-assets/measured-ns3d.png) as historical context, with its reference-centroid frame and finite-budget fit caveats. Then show the newly [matched trajectories](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/head-p03b-trajectories.png): common reconstructed initial state, equal bank, test space, time grid, frame freedom and converged solves. Assemble these as **Figure 2**, clearly labeling the different cohorts/configurations rather than treating historical and new panels as paired measurements.

**Argument and takeaway:** the head's smaller trajectory error is compatible with its more restricted approximation space. The matched comparison supports a constrained-dynamics explanation on this family and horizon. **Status:** completed development evidence; the numerical anchor above identifies the representative registered setting.

**4.2 Why the residual can reward drift.** Use the [same-previous-state residual/error scatter](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/head-p03b-residual-versus-error.png), physically normalized tangent diagnostics, and [richer-test control](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/head-p04-richer-tests.png) in **Figure 3**. Explain that the span can lower the same objective while increasing physical error, and that the tested richer spaces do not erase the observed gap. Shading is a case range, not a confidence interval; probes within a trajectory are correlated.

**Caption argument:** “A smaller tested residual need not imply a more accurate field; restricting accessible directions changes the resulting trajectory.” Put solver failures/repairs and optional nonstationary manifold-distance fits in Appendix B, while retaining their limitations in the main text. Harder-flow POD-bank results cannot substitute for this learned-bank causal control. [Completed mechanism evidence](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments.md)

## 5. Burgers: show how tests and width control different errors

**5.1 Increase tests with the representation frozen.** **Figure 4** uses [test enrichment](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/weak-full6-test-enrichment.png): physical error, independent residual and the difference from the reference's mass projection. The unchanged worst error is tied to initialization/projection, even while other trajectory errors improve. The argument is that adding equations can reduce an evolution error without repairing a deficient representation. **Status:** completed qualified development study, using dense evaluation of the original discrete-upwind target. Retain pointwise regressions and earlier failed controls in Appendix C.

**5.2 Increase frozen-prefix capacity and recheck dynamics.** **Figure 5** combines [projection floors](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/weak-width-w00-projection-floors.png), [qualified width dynamics](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/weak-width-final-width-dynamics.png), and the [fixed-test transient counterexample](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/weak-width-final-counterexamples-width-counterexample.png). Keep common-test width changes separate from joint width/test enrichment. The strongest measured setting improves the initialization-inclusive maximum error, while width alone can worsen intermediate times.

**Caption argument:** “Nested prefixes improve available approximation, but reliable trajectories require sufficiently informative tests.” **Status:** completed dense linear-span evidence. It is neither a nonlinear-head result nor a sparse deployment speed result. Ordering superiority over unordered prefixes, or equivalence to separately retrained widths, needs its own matched comparison only if claimed. Historical prefix and head-dimension studies already exist; their trained-head sweeps are not retraining-free tunability. [Prior studies and outstanding distinctions](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-results-audit-and-priorities.md)

## 6. Quadrature and mesh transfer: qualify the integration claim

**6.1 Separate operator change from node placement.** Use the saved derivative/node-placement study to show which change explains the historical improvement. Its preliminary physical references limit strong accuracy attribution. Do not silently connect it to the dense discrete-upwind width result. **Figure 6** should combine this controlled comparison with rule-size ladders; panel labels must name the operator, quadrature and reference. [Mechanism records](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/report_numbers.json)

**6.2 Select and qualify an integration rule.** Assemble **Table 2** from the [frozen rule decisions](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-15-jcp-sealed-closeout/experiments/jcp-sealed-closeout/results/four-rule-decisions.json), preserving primary space-and-time versus secondary space-only outcomes. Pair it with [per-case rule ladders](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-05-quadrature-per-case/data.json). A pass establishes the stated sufficiency criterion relative to the selected reference; it is not universal quadrature convergence. **Status:** completed, with mixed decisions. Reached-state residual/Jacobian checks remain necessary for a newly selected coupled deployment not already covered. [Final interpretation](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-15-experiment-closeout.md)

**6.3 Separate reduced evolution, storage and field delivery.** **Figure 7** plots error and the recorded cost components against requested output size, with stored nonlinear-operator bytes separately from peak memory. State which weights remain fixed and which normalization, tests, operators or rules are rebuilt. Include the small-rank tensor counterexample and continuous-POD controls. **Caption argument:** “A representation can transfer across meshes while input/output work grows, and quadrature can save storage without saving query time.” Existing [scaling records](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/checks/analysis.json) and [storage/tensor evidence](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-15-jcp-submission-readiness.md) support assembly now; changes to the final method require only targeted requalification.

## 7. Complete-query comparisons: show the real performance boundary

Use **Figure 8**, based on the [final complete-query plot](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/cost-p04-complete-query.png), with **Table 3** listing every qualified head setting, span control, eligible comparator, accuracy, median runtime, paired ratios and failures. Include same-mesh and any-mesh classical comparisons separately. Competitors qualify under the cohort median-and-worst error rule; this is not per-case equal-error matching.

Count physical input preparation, frame finding, initial fitting, evolution and required output. Report timing repetitions/outliers and selection versus timing-confirmation blocks. Accuracy comes from the returned timed solution. The cost study uses requested evolved output times; the head mechanism uses its denser trajectory history, and the Burgers width metric includes initialization. Do not compare their headline errors as if the metrics were identical.

**Argument:** fair cost accounting reveals where the method is competitive and where alternatives remain better. The cheapest measured head is effectively tied with eligible POD and slower than the selected coarse classical solver. Same-grid advantages remain restricted findings. **Status:** completed development comparison within a finite candidate grid; no general speed superiority. Historical prepared-input timings, single first-call activation records and lifetime-cost estimates belong in Appendix E, with their distinct contracts intact. Activation records exclude fresh process/runtime startup; amortization cannot create savings when online cost is already greater. [Complete measurement record](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments.md)

## 8. Discussion and conclusions: state usefulness and its limits

Return to the causal sequence: representation sets what is reachable; the head changes permitted evolution; tests determine which errors are visible; integration approximates those tests; complete cost includes reaching the answer and delivering it. Offer a practical tuning sequence grounded in these distinctions, with numerical checks before performance claims.

Use an optional **Figure 9** only if physical generality is central: assemble selected harder-flow, energy/force/divergence and mesh-transfer evidence, clearly separating learned banks, mesh-specific POD banks and the cylinder's Galerkin variant. Otherwise put it in Appendix D. Retain the cylinder's unfavorable neural/POD result and quadrature ambiguity. The [historical evidence ledger](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-results-audit-and-priorities.md) already identifies the completed temporal, long-horizon, seed, CROM/CoLoRA and operator studies; use their admitted scope rather than reopening them indiscriminately.

Conclude with the demonstrated mechanism and tuning lessons. State that broader families, stronger stability claims and efficient sparse deployment of changed widths require further evidence. The separate adaptive-tree study is outside this manuscript's fixed method and is not silently added to its results.

## Figure, table and appendix assembly map

| Item | Main purpose | Existing material / work needed |
|---|---|---|
| F1 | Bank, head, prefix, tests and integration | New explanatory diagram; no new measurements |
| F2–F3 | Approximation, matched drift and residual visibility | Existing measured panels; assemble with distinct controls and captions |
| F4–F5 | Test enrichment and frozen-width tuning | Existing measured panels; retain transient counterexample |
| F6 / T2 | Operator attribution and quadrature sufficiency | Assemble historical ladders and frozen decisions; preserve reference differences |
| F7 | Mesh transfer, output and storage | Assemble recorded components; do not join cross-job absolute times |
| F8 / T3 | Complete-query accuracy and cost | Final same-allocation panel; include all qualified target settings |
| F9, optional | Physical generality and failure boundaries | Existing scoped application evidence, usually Appendix D |
| T1 | Benchmark and method definitions | Compile configurations and populations before comparing results |
| Appendices A–E | Verification; head diagnostics; tunability; generality; cost/reproduction | Reuse qualified historical records and retain failures |

## What is new relative to the previous manuscript

| Evidence category | Journal treatment |
|---|---|
| Bank/head, ordering, fixed tests, preassembly and quadrature | Attributed framework; precise assumptions and variant definitions |
| Broad PDE suite, harder Navier–Stokes and cylinder | Supporting evidence with the original bank/solver distinctions |
| Off-mesh scaling and output costs | Prior evidence; restyling the figure is not a new contribution |
| New since that source: matched head, visibility/width and complete-query controls | Central additional numerical evidence and interpretation |

The authors must settle the relationship between manuscripts before submission; this outline does not assert current venue eligibility or conference disposition. Compare against the [actual conference source](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-13-manuscript-rebuttal/paper_revision/main.tex) and [overlap assessment](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-15-jcp-submission-readiness.md). If a distinct companion article is needed, use *Classical quadrature for continuous reduced models of nonlinear advection*: concentrate on Section 6 and its cost consequences, attribute the representation, and leave the broad head/application narrative with the original paper.

## Remaining work, in drafting priority

1. **Assemble and freeze the claim.** Choose the final configurations, operator labels, reference populations and input/output contracts; build the figures from their saved records. No new broad sweep is needed for this step.
2. **Confirm the frozen choice.** Use untouched cases only after selection is fixed. Existing historical tests remain historical evidence; the latest mechanism and width studies remain development evidence until separately confirmed.
3. **Qualify changed deployment components.** If the wider Burgers configuration is claimed as efficiently hyper-reduced, establish its sparse residual/action fidelity, reference/time sensitivity and complete-query cost. Dense accuracy does not supply these results.
4. **Add only claim-dependent controls.** A harder-flow causal head claim needs matched controls there; ordering superiority needs unordered/retrained comparisons; a uniform stability claim needs a proof; a broader nonlinear-method ranking needs a matched quadratic competitor. Otherwise narrow the claim and use existing evidence.

## Core literature and its role

The references below are methodological precedents, not an official JCP checklist. Publication metadata was checked against primary records; detailed analysis was often inspected in author versions, whose numbering can differ from the journal article. Published experiments were not reproduced for this outline.

| Paper | Placement and reason |
|---|---|
| Lee and Carlberg, JCP (2020), [nonlinear-manifold model reduction](https://doi.org/10.1016/j.jcp.2019.108973) | Sections 1, 3–4: separate best approximation from evolved error and state theory assumptions |
| Kim et al., JCP (2022), [shallow masked-autoencoder NM-ROM](https://doi.org/10.1016/j.jcp.2021.110841) | Sections 2, 6–7: sampled decoder/Jacobian work and hyper-reduced error |
| Barnett and Farhat, JCP (2022), [quadratic approximation manifolds](https://doi.org/10.1016/j.jcp.2022.111348) | Sections 1, 8: credible non-neural nonlinear alternative |
| Barnett, Farhat and Maday, JCP (2023), [PROM-ANN](https://doi.org/10.1016/j.jcp.2023.112420) | Sections 1–2: nonlinear coefficient correction inside an outer basis; inspected [author text](https://arxiv.org/html/2212.08939) is an AIAA precursor |
| Romor, Stabile and Rozza, JCP (2025), [explicable hyper-reduced models](https://doi.org/10.1016/j.jcp.2025.113729) | Sections 4, 6–8: reconstruction, evolution, sampling, generalization and unfavorable cost regimes |
| Puri et al., JCP (2025), [SNF-ROM](https://doi.org/10.1016/j.jcp.2025.113957) | Sections 1–3, 6: continuous fields, derivative regularity and coupled time/sampling effects |
| Chang et al., SIGGRAPH Asia conference (2023), [LiCROM](https://doi.org/10.1145/3610548.3618158) | Sections 1–2, 6: continuous neural banks, cached cubature and resolution-independent representation |
| Hernández, Caicedo and Ferrer, CMAME (2017), [empirical cubature](https://doi.org/10.1016/j.cma.2016.10.022) | Sections 1, 6: fitted integration points/positive weights; metadata and excerpts, not full-text reinspection |

## Provenance and glossary

The [builder](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/build_2026_11_18_jcp_manuscript_outline.py) generates the empirical anchors from frozen campaign data; [source hashes and derived values](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-manuscript-outline-assets/evidence-manifest.json) identify their exact input. The outline introduces no new numerical result. Assembly recommendations are editorial judgments based on the cited literature and local evidence.

**Bank / span / head:** spatial functions; their arbitrary linear combinations; a nonlinear map producing restricted coefficients. **Prefix / width:** the first ordered functions / their count. **Projection floor:** the smallest field error available in a stated linear space. **Trajectory:** the evolving solution. **Weak tests / residual visibility:** weighted equation measurements / how well they detect physical changes. **Tangent / Jacobian:** locally reachable field changes / local sensitivity of a map. **Mass norm:** a spatially weighted measure of field error. **Stationarity:** a small objective gradient, not proof of global optimality.

**Quadrature / cubature / hyper-reduction:** numerical integration / multidimensional integration / reduced equation-evaluation work. **Stencil:** neighboring grid values used by a discrete derivative. **POD:** a linear basis obtained from example fields. **LSPG:** least-squares Petrov–Galerkin, a residual-minimizing projection formulation. **Complete query:** all work from the declared physical input to the requested output. **Median / worst:** middle / largest casewise error after the specified time aggregation. **Development / untouched confirmation:** cases available during design / cases reserved until choices are fixed. **Qualification:** passing stated numerical and measurement checks. **Reference:** the comparison solution and its documented uncertainty. **Same-allocation timing:** compared methods timed on the same allocated hardware. **F / T:** proposed figure / table; **existing material** identifies available evidence, while **work needed** distinguishes assembly from new validation.
