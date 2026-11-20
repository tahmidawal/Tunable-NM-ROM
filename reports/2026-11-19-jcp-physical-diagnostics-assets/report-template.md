# Physical diagnostics and frozen confirmation for the JCP paper

The requested phase/shape, derivative-error, width–test and accuracy–cost plots are complete, and the targeted follow-up experiments have finished. The Burgers fresh-case accuracy confirmation passes; fresh Navier–Stokes and cost comparisons retain failed qualification checks, and neither tested sparse rule is admitted for deployment. These are final experiment records with the claim limits stated beside the results.

Three experiment agents used separate branches and cluster directories. Models, numerical choices, comparison rules and new-case seeds were recorded before generating new reference solutions. A further agent reviewed the scientific interpretation. The coordinator independently checked arithmetic, retained failures and consistency between the two Navier–Stokes implementations. The run total is **{{GPU_JOBS}} jobs, {{GPU_SECONDS}} allocated GPU-seconds ({{GPU_HOURS}} GPU-hours)**. No new model training or historical reserved test cohort was used. [Numerical audit]({{SOURCE_coordinator-audit}}), [scheduler accounting]({{SOURCE_scheduler}}).

## What belongs in the paper

| Figure family | Argument and placement | Result status |
|---|---|---|
| Phase and centroid-aligned shape | Main head-mechanism section: the error gap remains after the declared physical alignment | Saved development results qualified; fresh confirmation has incomplete initial-fit qualification |
| Field versus derivative/operator errors | Main numerical-analysis section: small field error does not certify accurate spatial derivatives | Saved discrete-reference diagnostics verified, including denominators |
| Width–test map and retained transient regressions | Main tunability section: representation and test enrichment interact; trajectory improvement is not monotone | Development map verified; frozen combined choice confirms on new Burgers cases |
| Accuracy–cost selection and separate confirmation | Main performance section: complete-query costs and eligible comparisons determine the claim | Development evidence remains qualified; fresh fixed-comparator experiment has no qualified comparison |
| Sparse-rule admission | Quadrature section or supplement: equation fidelity, initial fitting and actual evaluation sites all matter | Both tested recipes rejected; sparse rollout and time-step interaction were not admitted |

These plots implement the [review recommendations](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-19-jcp-reviewer-analysis-and-plots.md) and refine the [manuscript outline](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-manuscript-outline.md). Literature positioning remains in those documents. The figures here contain measured values only.

## Phase and shape: the head advantage survives centroid alignment

![Development phase, shape and ordinary field error]({{FIGURE_phase-development}})

For the qualified development comparison, the median paired span/head maximum-error ratio is **{{DEV_FIELD_RATIO}}** before alignment and **{{DEV_SHAPE_RATIO}}** after alignment. Physical position error improves on **{{DEV_PHASE_WINS}}/{{DEV_COUNT}} cases**. The strong field-error advantage therefore persists after this particular alignment; it cannot be described solely as centroid drift control. This is not an optimal-registration experiment or proof of a unique dynamical mechanism.

Both field errors use the original physical initial-field norm. The position measure is the norm of the wrapped difference between energy-weighted physical circular centroids, already expressed in periodic-domain fractions. It is not the learned frame variable. Alignment Fourier-shifts the prediction using the reference centroid and is a diagnostic only. Replaying the saved states reproduces the stored metrics; the concentration check supports using these centroids. [Replay evidence]({{SOURCE_phase-replay}}).

{{TABLE_development-head}}

The table takes each case's maximum over evolved times, then the cohort median/worst. Curves instead show the median at each time with the full case range; these aggregations need not coincide. The common-initial-state arms isolate evolution. Native span initialization is a separate practical arm.

![Fresh phase and shape, incomplete initial-fit qualification]({{FIGURE_phase-fresh}})

**Fresh result remains formally unqualified:** the observed paired ratios are **{{FRESH_FIELD_RATIO}}** before alignment and **{{FRESH_SHAPE_RATIO}}** after alignment, but **{{IC_FAILURE_COUNT}} initial fits miss the frozen strict stationarity rule**. All cases remain visible. Trajectory, reference, common-state and Jacobian checks pass. The same-previous-state probes retain the lower-residual/higher-physical-error pattern in all **{{PROBE_COUNT}}** observations; correlated time steps are not independent samples.

The failed initial fits selected a marginally lower objective whose strict solve stagnated. Stationary alternatives existed outside the predeclared objective tie band. Tight/tighter selected initial fields were identical, but that agreement does not override the registered stationarity criterion. The initializer or threshold was not changed after seeing these cases. [Exact diagnosis]({{SOURCE_phase-initial-fit}}), [qualification audit]({{SOURCE_phase-audit}}), [frozen protocol]({{SOURCE_phase-protocol}}).

Fresh centroid concentration has minimum **{{MIN_CONCENTRATION}}**, with no registered ambiguity flag. Position-error improvement is mixed (**{{FRESH_PHASE_WINS}}/{{FRESH_COUNT}} cases**), while the field gap persists after centroid alignment. Individual-case curves and the unchanged fitting caveat are available in the [full-case figure]({{FIGURE_phase-all-cases}}).

## Derivatives: a small field error can conceal a large Laplacian discrepancy

![Saved Burgers field and derivative-error histories]({{FIGURE_derivative-time}})

The saved selected configuration has worst maximum field error **{{DEV_FIELD_WORST}}%**, yet worst maximum relative discrete-Laplacian error **{{DEV_LAPLACIAN_WORST}}%**. This discrepancy must remain in the paper. Reference Laplacian norms do not approach zero: their minimum Euclidean norm is **{{MIN_LAPLACIAN_NORM}}** in the checked records. The displayed large relative errors are not division by a vanishing reference. [Denominator and metric audit]({{SOURCE_derivative-summary}}).

The normalization is deliberately different:

$$
e_u(t)=\frac{\|u(t)-u_\star(t)\|}{\|u_\star(0)\|},\qquad
e_D(t)=\frac{\|D_h(u(t))-D_h(u_\star(t))\|}{\|D_h(u_\star(t))\|}.
$$

Here $D_h$ denotes the implemented gradient, sign-upwind advection or Laplacian; advection is nonlinear. These are discrete-operator diagnostics. They neither certify continuum derivatives nor prove a particular spatial-frequency explanation without a further spectral analysis. The frozen bank and exact discrete weak target are unchanged. The broad normalization sentence in the old metric-function docstring is not the definition used in this report.

{{TABLE_development-width}}

Each entry is the median/worst of the casewise maximum over all saved times, including initialization. The [case-maximum plot]({{FIGURE_derivative-maxima}}) displays all measured configurations and individual cases. Absolute derivative-error numerators and reference norms remain in the [plotted data]({{SOURCE_width-plotted-data}}).

## Width and tests: keep the measured cells and the counterexample

![All measured width and test configurations]({{FIGURE_width-tests}})

Untested cells remain blank. The main map combines compatible field, initial projection and evolution-distance records; the [independent-residual map]({{FIGURE_width-observer}}) uses only configurations with the same observer rows. A larger prefix lowers best linear projection error, but it does not guarantee a better trajectory at every time. The [fixed-test transient-regression figure]({{FIGURE_width-counterexample}}) retains the cases where increasing width temporarily worsens evolution.

![Fresh Burgers confirmation]({{FIGURE_width-fresh}})

**Fresh Burgers confirmation passes on all {{WIDTH_CASES}} new cases.** The frozen combined width/test choice reduces worst initialization-inclusive field error from **{{WIDTH_BASELINE_WORST}}% to {{WIDTH_SELECTED_WORST}}%**, and the median from **{{WIDTH_BASELINE_MEDIAN}}% to {{WIDTH_SELECTED_MEDIAN}}%**. Every case improves and every arm passes the declared numerical checks. This is a joint change of width and tests with native initialization, not an isolated width effect.

{{TABLE_fresh-width}}

The reference is the same-grid, same-step discrete full model. The result establishes neither continuum convergence nor sparse online speed. The [case-coverage figure]({{FIGURE_width-coverage}}), [raw-array audit]({{SOURCE_fresh-width}}) and [registered protocol]({{SOURCE_width-protocol}}) preserve parameters, reference checks and the distinction from the exposed development cases.

## Accuracy and cost: selection, repeated timings and new cases stay separate

![Development accuracy–cost selection frontier]({{FIGURE_cost-selection}})

Only candidate-selection timing repetitions form this frontier. Lines join observed marginal envelopes for the displayed error statistic; they are not an interpolated predictor or the joint median-and-worst eligibility test. Finite settings above the display limit are explicitly marked and appear in the [full-range view]({{FIGURE_cost-selection-full}}). Incomplete candidates remain in the underlying records.

The [separate development confirmation]({{FIGURE_cost-development-confirmation}}) retains the near POD tie and losses to eligible coarse classical computation. Its paired ratios compare the same case on the same allocation. The [repetition-spread plot]({{FIGURE_cost-repetitions}}) retains outliers. Timing repeats, physical cases and independently trained models represent different types of variation.

![Fresh fixed-comparator validation with failures retained]({{FIGURE_cost-fresh}})

**No fresh comparison qualifies.** Both learned heads have a nominal initial-fit failure on the marked case; stricter solves expose further initialization/stopping failures. Every frozen subject exceeds the applicable previous worst-error target. The coarse classical subject also fails fractional reference checks and a timing-drift check. None of these results is silently converted into a speed claim. [Complete cost audit]({{SOURCE_cost-audit}}), [fixed protocol]({{SOURCE_cost-protocol}}).

The fresh plot shows returned-field diagnostics even when numerical qualification fails. Its dashed target is a **cohort-median criterion**, not a per-case requirement; the dotted target constrains the cohort maximum. Crosses mark numerical-check failures separately from accuracy-target exceedances. Timing ratios use only available successful paired invocations and are explicitly unqualified. A missing nominal solve is not assigned a favorable time or omitted without explanation.

{{TABLE_fresh-cost}}

The table's error columns use all nominal returned-field diagnostics, including rejected numerical solves. No full-cohort head time is reported when a case fails. Complete timings for other subjects are measured values, not qualified matched-accuracy comparisons. The records contain **{{TIMED_ROWS}} timed invocations**, including **{{FAILED_TIMED_ROWS}} failures** from repeated measurements of the same physical input. These are not independent method failures. The fresh timings are never pooled with development absolute timings or compared across allocations. Cost accuracy uses requested evolved outputs; the denser mechanism time grid and initialization-inclusive Burgers metric are different contracts.

## Sparse evaluation: qualification completed, deployment not admitted

![Sparse-rule admission checks]({{FIGURE_sparse-qualification}})

The bounded development-only study fitted nonnegative weights to decoder-output snapshots on fixed on-grid supports. It preserved the original sign-upwind operator and exact cached linear terms. Evaluation used held-out development states, common physically normalized Jacobian directions and a sampled initial fit. The new confirmation cases were not used to choose weights.

{{TABLE_sparse-bridge}}

Residual and initial-fit discrepancies use original initial-field mass norms; Jacobian-action discrepancies use the corresponding dense action norm, with absolute values also retained. These are probes along the registered directions, not a full Jacobian operator-norm bound.

The checkerboard rule passes the measured residual/action checks but fails the initial-fit requirement. The other rule fails residual/action checks and reaches its optimization iteration limit; it is an approximate fit with that termination retained. Every fitted weight remains positive. Both rules therefore require every interior grid site when including their fixed closure over both possible upwind directions. Neither passes the site-economy gate. This is a result for these recipes and evaluation strategy, not an impossibility theorem for all sparse or adaptive implementations.

The predeclared admission rule consequently stops before sparse PDE rollout, time-step–sample-count interaction and complete-query sparse timing. Those experiments were **not run** and are not presented as validated. [Fidelity records and explicit decisions]({{SOURCE_sparse-bridge}}).

## What further experiments would still be needed?

The requested saved-data figures and finite follow-up protocols are complete. The scoped paper can now present qualified development mechanism evidence, confirmed dense Burgers accuracy, derivative limitations, fair development cost comparisons and the failed deployment/robustness checks. It cannot claim that all fresh validations pass or that the wider Burgers result already has efficient sparse deployment.

| Stronger claim desired | Necessary next work | Priority |
|---|---|---|
| Reliable initialization across new Navier–Stokes inputs | Develop a numerically robust initial-fit selection/stopping procedure on exposed cases, preserve strict numerical checks, freeze the resulting algorithm, then evaluate a different untouched cohort | First if retaining an all-case robustness claim |
| A new qualified complete-query performance claim | Qualify all solvers and sufficiently refined references; choose comparators/accuracy targets on development data; freeze them and repeat same-allocation measurements and fresh-case evaluation with retained repetitions | Follows numerical qualification; the current unqualified ratios cannot substitute |
| Efficient sparse deployment of the wider Burgers model | Qualify sampled initial fitting and a rule/kernel that actually reduces evaluated work; then reached-state residual/action checks, sparse trajectories, time–quadrature interaction and complete-query timing | Required only for that deployment claim |
| Accurate continuum derivatives, stronger long-time behavior or data efficiency | Corresponding spatial/temporal, horizon or training-data studies with appropriate references | Conditional; not established by the present field-error plots |

No additional broad PDE catalogue or training sweep is required merely to populate figure slots. The authors' conference/journal overlap decision remains a separate manuscript issue. The new results do not waive existing submission or attribution obligations.

## Reproduction and figure files

Every main figure has PNG, SVG and PDF versions in the adjacent asset directory. Useful vector exports: [phase/shape]({{FIGURE_phase-development-vector}}), [derivatives]({{FIGURE_derivative-time-vector}}), [width–test map]({{FIGURE_width-tests-vector}}), [accuracy–cost]({{FIGURE_cost-selection-vector}}), [fresh Burgers]({{FIGURE_width-fresh-vector}}). The [source registry](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-19-jcp-physical-diagnostics-assets/sources.json) records exact original paths; immutable copies and hashes accompany this report. The report builder generates its numerical tables and prose values, and each lane retains its plot builder, plotted data, configurations and raw evidence. The three experiment branches are backed up separately and remain unmerged pending the user's decision. The cost lane's large regenerable deployment arrays and transport archive are retained locally and in Git, with hashes and regeneration instructions, but excluded from its code-only remote mirror. [Retention details]({{SOURCE_cost-retention}}), [verified branch closeout]({{SOURCE_lane-closeout}}), [independent review]({{SOURCE_independent-review}}).

## Glossary

**Bank, span, head:** frozen spatial functions, their unrestricted linear combinations, and a nonlinear map restricting their coefficients. **Width $R'$ / tests $M$ / sample sites:** active bank functions / weak residual measurements / integration locations. **Weak residual:** equation error averaged against specified test functions. **Projection floor:** best field approximation available in a specified linear space. **Evolution distance:** field distance from the reference's actual linear projection, not subtraction of error norms. **Native/common initialization:** each method's own fit / an identical initial reduced field.

**Phase / centroid-aligned shape:** physical position / the field after the specified reference-assisted translation. **Concentration:** how distinctly a periodic centroid is defined. **Gradient / advection / Laplacian:** the stated discrete spatial derivative / transport operator / second-derivative operator. **Initial/current norm:** reference magnitude at initial time / the same time as the evaluated derivative. **Median/worst:** middle/largest casewise statistic; the time aggregation is stated separately. **Case range:** observed minimum to maximum, not a confidence interval.

**Qualification:** satisfying the predeclared numerical or measurement checks. **Stationarity:** a sufficiently small normalized objective gradient, not a global-optimality guarantee. **Tie band:** the fixed allowance for considering objective values numerically equivalent. **Fresh/development:** previously unused physical cases / cases available during choices and diagnosis. **Nominal/tighter:** production numerical settings / stricter checks. **All-case/failure count:** physical input coverage / failures under the stated classification; repeated timings are not new physical cases.

**POD:** a basis derived from snapshots. **Complete query:** work from declared input to returned outputs, including initialization and transfers. **Paired ratio:** comparison within the same physical case. **Selection/confirmation:** choosing settings on development data / evaluating frozen choices or repeating their measurements, with the kind identified. **Marginal envelope:** the best observed time/error trade-off for one plotted error statistic. **Outlier:** an unusual timing retained in the data. **Drift:** change in timing during the measurement block.

**Quadrature/NNLS:** weighted numerical integration / nonnegative least-squares fitting. **Jacobian action:** local sensitivity applied to a specified perturbation. **Mass norm:** field magnitude with spatial integration weights. **Stencil closure:** all grid sites required by the sampled operator's neighboring evaluations. **Admission/site economy:** permission to proceed under a predeclared gate / reduced evaluated-site requirement. **Sparse rollout:** time evolution using a qualified reduced integration rule. **Discrete/continuum reference:** a specified grid-and-step full model / the limiting PDE solution. **GPU-seconds:** elapsed allocated time multiplied by the allocated GPU count. **Integrity pass:** evidence is internally consistent; it does not mean the scientific hypothesis or qualification passed.
