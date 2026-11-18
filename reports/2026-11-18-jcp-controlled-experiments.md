# Controlled experiments for the JCP paper

This report records the completed development experiments authorized on November 18, 2026, with independently reviewed numerical results, measured plots and retained failures. Every numerical result below is generated from saved records; these exposed development studies do not replace fresh-cohort confirmation.

The central questions are whether the nonlinear head improves trajectories under matched numerical controls, whether richer weak tests detect errors missed by the minimized residual, and whether the selected method saves complete-query cost at matched accuracy. The earlier optimistic sketches remain hypotheses, not acceptance criteria or promised outcomes.

<!-- GENERATED EXPERIMENT SUMMARY BEGIN -->
These measurements support a constrained-dynamics explanation for the head and distinguish test-space and representation errors in Burgers.

- **Head mechanism (qualified):** at the last registered setting, M=584 and $\Delta t=0.01$, median maximum evolved error is 0.0967% for the head versus 0.6697% for the span. The median paired span/head error ratio is 7.06; 16/16 cases favor the head.
- **Weak-test enrichment:** increasing M from 256 to 2048 reduces median maximum saved-time error from 2.0064% to 1.3228%. Worst error remains 4.2979%, set by the initial projection. This is the original frozen-width baseline.
- **Frozen-width PDE dynamics:** worst maximum saved-time error is 4.2979% at R=128, M=2048, and 0.3918% at the largest registered setting, R=512, M=8192. The latter qualifies 6/6 cases. Common-test width comparisons are retained below.

**Latest complete-query comparison: cost-p04, 16 development cases.** Each comparator satisfies the stated median-and-worst accuracy rule. The ratios divide comparator median time by target median time; above one favors the target. All times in this table come from the same allocation.

| Target | Worst evolved error (%) | Time (ms) | Any-mesh classical / target | POD / target |
|---|---:|---:|---:|---:|
| Head M292, Δt=0.02 | 0.1956 | 26.506 | 0.904 | 1.004 |
| Head M584, Δt=0.02 | 0.2005 | 27.343 | 0.876 | 0.973 |
| Head M584, Δt=0.01 | 0.1856 | 33.557 | 0.714 | 0.958 |
| Span R64 | 1.2643 | 37.113 | 0.437 | 0.713 |

Every tested head setting is slower than its selected eligible coarse-grid classical comparator on every case. A broad complete-query speedup is therefore not established by this panel. Same-grid comparisons remain separately reported below.
The lowest median-cost head's POD comparison has ratio 1.004200; casewise medians favor the head in 9/16 cases, with ratios ranging from 0.969336 to 1.039652. This small aggregate difference does not establish a consistent head advantage over POD.

The figures below contain measured values. All cohorts here are exposed development data; untouched confirmation and broader JCP claims remain separate work.
<!-- GENERATED EXPERIMENT SUMMARY END -->

## Separate ownership and initial resources

The following are **initial resource reservations**, not measured costs. Every lane starts from the canonical consolidated fork and has its own worktree, experiment code, cluster namespace, ownership record and job ledger. The coordinator remains on the main checkout for this report and the canonical lab log.

| Lane | Branch and worktree suffix | Initial allocated GPU-hour reservation | Responsibility |
|---|---|---:|---|
| Head mechanism | `2026-11-18-jcp-head-mechanism` | 3 | Matched head/span dynamics and errors over time |
| Weak-test visibility | `2026-11-18-jcp-weak-visibility` | 3 | Physical tangent conditioning, independent residuals and test enrichment |
| Paired cost and review | `2026-11-18-jcp-paired-cost` | 2 | Independent protocol/result review, then selected complete-query comparisons |

Branches use `exp/` and directories live under `worktrees/`. The initial total is eight allocated GPU-hours, including compilation and failed attempts. Each job uses one GPU and a fresh directory. At most two campaign GPU jobs run concurrently; the cost lane waits for a scientifically qualified configuration. Individual initial jobs have bounded wall times. The coordinator can reallocate unused reservations between lanes before additional submissions.

## Matched head/span experiment

Start with the existing original-family Navier–Stokes learned bank and its frozen head on exposed development cases, using the saved development configuration identified in the [results audit](2026-11-18-jcp-results-audit-and-priorities.md). Copy source and checkpoint assets into the owned lane, record their original locations and content hashes, and regenerate physical data from the recorded seed on the cluster.

For the causal comparison, hold the bank, PDE operator, physical initial field, frame rule, smooth test functions, residual row scaling and time grid fixed. Include a genuinely common representable initial-state control: native head and span projections need not return the same initial field. Frame variables are part of the online state when solved, so their effects must not be silently attributed to the coefficient head.

Replace fixed iteration counts with explicit convergence diagnostics and tight/tighter returned-trajectory checks. Begin with exact or sufficiently resolved nonlinear evaluation to isolate the representation. Record the actual number of scalar residual rows; an inherited Fourier-mode parameter is not necessarily that number.

Persist physical error histories, reconstructible states, frame/phase information, minimized and independent expanded-test residuals, solver termination reasons, iteration histories, Jacobian checks and relevant physical quantities. Fit the head to selected reached span states with multiple initial guesses to estimate departure from the learned manifold; label such fits best-found, not globally optimal distances. Distinguish errors leaving the manifold from errors moving along it.

An advantage that survives matched controls supports a constrained-dynamics explanation. A vanished advantage narrows the earlier finding to the selected numerical configuration. Numerical failure first triggers diagnosis, not a larger sweep or weaker acceptance condition. A harder-flow extension follows a working and qualified implementation control.

## Weak-test visibility experiment

Use a selected existing Burgers configuration with documented test-space issues. Preserve its weak formulation and exact discrete upwind advection target. Vary admissible smooth test spaces with frozen representation, time grid and qualified nonlinear evaluation. Keep the historical failed square-system verdict; a new protocol may study qualified overdetermined spaces without changing that verdict.

Normalize decoder tangents in the physical mass norm before evaluating the tested PDE Jacobian. Report raw-coordinate conditioning separately. Measure singular values, independently expanded weak residuals, physical errors, solver convergence, and dense/sparse residual and Jacobian-action agreement. Include frame variables if present. Oversampling and positive quadrature weights are not proofs of stability.

Refit any fitted quadrature on decoder-output snapshots when its relevant bank/test/mesh configuration changes, and qualify the result independently. A rule-of-thumb point budget is only a starting choice. No strong-form random or importance collocation is introduced.

If enrichment reduces physical error and independently measured residual leakage, it supports the test-visibility hypothesis. If not, examine representation, operator consistency and optimization before adding more points. Improvements to scaling or test selection are separate development amendments with the original baseline retained.

## Complete-query comparisons

The cost lane prepares the measurement pipeline and reviews the other protocols immediately. It submits its numerical comparison after a suitable configuration is selected. Compare head/span, the strongest eligible POD model and relevant classical solvers using the same physical input information, requested output, precision and reference/accuracy requirements.

Mechanism isolation and best-performance comparison answer different questions. The first matches numerical settings; the second permits each method to select qualified settings under a declared development procedure and matched accuracy.

Count input preparation, frame finding, initial projection, time evolution, required recovery and output. Separate compilation and offline costs. Burn in the GPU, interleave timing order, retain repetition arrays and perform an independent confirmation block. Accuracy must come from the returned timed solution. Speed ratios are within one allocation; failed-drift panels remain excluded. A quantity-of-interest contract is distinct from complete field output and must be labelled.

If the ROM has no admitted complete-query advantage, retain that finding and report any demonstrated accuracy, storage or reuse benefit. Amortization cannot repair a positive extra offline investment when the ROM has no per-query saving.

## Development, review and confirmation

Each worker writes a versioned, falsifiable lane protocol before submission. The coordinator and independent review worker inspect concrete controls and implementation hazards. Small preliminary panels precede expansion. All failed attempts, source hashes, job IDs, hardware, precision, data seeds and configuration changes remain in the ledger.

Scientific progress may justify a development amendment; it does not justify changing the physical reference, discarding inconvenient cases, relaxing a qualification after seeing its result, or relabelling exploratory data as untouched confirmation. New model architectures and training are not initial steps. Fresh confirmation is opened only after the configuration and selection policy are frozen. Existing opened tests remain historical diagnostic evidence.

The existing tree-embedding and paused quadtree campaigns retain their separate ownership and state. This authorization starts new JCP analysis lanes. Each lane commits and backs up through its code-only mirror; experiment branches are not merged automatically. A merge decision will be requested when the lanes close.

<!-- GENERATED EXPERIMENT RESULTS BEGIN -->
## Measured development results

These are new measured development runs. A numerical failure remains visible even when a physical-error comparison is favorable. The earlier optimistic schematics are separate hypothetical artifacts.

### Matched head/span dynamics

| Retained run / job | Cases | Numerical qualification | Failed gates |
|---|---:|---|---|
| [head-p01](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/head-p01.json) / 7354236 | 2 | Excluded: failed gates | all_stationary, ic_fit_passed, reference_passed |
| [head-p02](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/head-p02.json) / 7354339 | 2 | Pass | None |
| [head-p03](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/head-p03.json) / 7354386 | 16 | Excluded: failed gates | ic_fit_passed |
| [head-p03b](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/head-p03b.json) / 7354705 | 16 | Pass | None |

**Numerically qualified development comparison — head-p03b.** Errors below are maxima over each evolved trajectory, normalized by its original physical initial field. Both primary arms have identical reconstructed initial states. Median and worst are across the same cases.

| Scalar weak rows M | Time step | Head median / worst error (%) | Span median / worst error (%) | Cases with lower head error |
|---:|---:|---:|---:|---:|
| 292 | 0.02 | 0.113993 / 0.195600 | 0.794107 / 1.294029 | 16 / 16 |
| 584 | 0.02 | 0.124681 / 0.200474 | 0.649119 / 1.340350 | 16 / 16 |
| 584 | 0.01 | 0.096675 / 0.185633 | 0.669708 / 1.270691 | 16 / 16 |

![Measured matched head/span trajectories](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/head-p03b-trajectories.png)

The admitted comparison isolates the head constraint under the registered numerical settings.
Its scope excludes a universal stability guarantee, fresh-test generalization, a global manifold distance, and a complete-query speedup.

In the same-previous-state control, **640 of 640 stationary probes** reduce the minimized weak residual while increasing physical error; 0 additional probes do not meet stationarity. Each span solve starts from the head's next coefficients and uses the identical previous field and objective. These are correlated step diagnostics across 16 cases, not independent statistical samples.

![Same-previous-state residual and physical-error comparison](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/head-p03b-residual-versus-error.png)

At the same reached head states, the median physical-norm fraction of the least visible span direction outside the head-plus-frame tangent is **0.999718**. The corresponding median fraction of the actual span departure, evaluated with the local linearized field map, is **0.995103**. Retained physical (head, span) ranks are [(11, 67)]. These are norm fractions of sampled local directions, not fractions of cases or directions, energy fractions, global manifold distances, or uniform stability constants. The run's qualification status above applies to this diagnostic too.

### Richer-test check for the head mechanism

This follow-up retains the fixed pilot cases, model, common initial reconstruction, time step and convergence gates. It tests whether additional residual information closes the span/head gap. The original development-panel result remains separate.

| Run / job | Cases | Numerical qualification | Rows |
|---|---:|---|---|
| [head-p04](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/head-p04.json) / 7354832 | 2 | Pass | 1168, 2336 |

![Measured richer-test head/span comparison](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/head-p04-richer-tests.png)

Optional manifold-distance diagnostics retain their convergence limitations:

| Run | Best-found fits | Nonstationary fits | Interpretation |
|---|---:|---:|---|
| [head-full16-distance-diagnostics](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/head-full16-distance-diagnostics.json) | 1376 | 48 | Upper bounds only |
| [head-richer-distance-diagnostics](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/head-richer-distance-diagnostics.json) | 168 | 4 | Upper bounds only |

These optional fits are separate from the converged initial fits and equation solves used to qualify the trajectory comparisons. Even stationary local fits do not certify global distance to the nonlinear manifold; nonstationary fits retain that additional limitation.

### Weak-test visibility

The Burgers diagnostic model evolves arbitrary coefficients in the linear span of a frozen learned spatial basis; there are no frame variables. Every panel uses the same-grid, same-time-step discrete Burgers reference. No continuum-error claim is made. Maximum field error includes the initial saved time. Spectral visibility and physical accuracy are separate diagnostics.

| Retained run | Cases | Numerically qualified case/row settings |
|---|---:|---:|
| [weak-p02](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-p02.json) | 2 | 4 / 8 |
| [weak-p03](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-p03.json) | 2 | 8 / 8 |
| [weak-full6](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-full6.json) | 6 | 24 / 24 |

Latest registered panel:

| Weak rows M | Median / worst maximum field error (%) | Numerically qualified cases |
|---:|---:|---:|
| 256 | 2.006441 / 4.297893 | 6 / 6 |
| 512 | 1.530516 / 4.297893 | 6 / 6 |
| 1024 | 1.368581 / 4.297893 | 6 / 6 |
| 2048 | 1.322786 / 4.297893 | 6 / 6 |

![Measured weak-test enrichment](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/weak-full6-test-enrichment.png)

The largest total error remains limited by the initial projection. At M=2048, 4 of 6 cases attain their maximum at initialization. The worst actual distance from the reference's mass projection decreases from 2.998368% to 1.067425%. This distance is computed from the difference of fields, not by subtracting error norms. [Casewise source table](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-full6-table.json).

**Retained counterexample:** 6 of 18 adjacent case/test-space transitions increase error at some times, even though their maximum-over-time errors do not worsen. The largest pointwise increase is 0.021036 percentage points. This is a separately labeled post-run diagnostic; it precludes pointwise error-dominance claims. [All comparisons](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-pointwise-counterexamples.json).


Adding unchanged rows cannot decrease the smallest singular value at an identical state. That algebraic property alone does not demonstrate improved physical prediction.

### Frozen-width follow-up: projection screen

This prospectively bounded follow-up targets the initial projection floor diagnosed above. It takes longer prefixes of the same frozen ordered bank and uses all original development cases. Each curve fits the reference at each saved time; it measures representation capacity and does not predict a trajectory. The physical operator, reference, parameters and error denominator are unchanged.

| Basis width R | Median / worst all-time projection error (%) | Worst initial projection error (%) | Admission to equation solves |
|---:|---:|---:|---|
| 128 | 1.293561 / 4.297893 | 4.297893 | Baseline |
| 256 | 0.452384 / 0.815003 | 0.815003 | Pass |
| 512 | 0.143086 / 0.391845 | 0.391845 | Pass |

[Projection audit and admission](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-width-w00.json) bind the results to the saved model, basis ordering, references and numerical settings. All registered rank, projection and original-baseline reproduction checks pass. A prefix enters the PDE study only if both worst initial and worst all-time projection errors are at most 80% of baseline. No training was performed.

![Measured frozen-bank projection floors](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/weak-width-w00-projection-floors.png)

### Frozen-width follow-up: actual equation solves

These trajectories initialize once from each prefix's native projection and subsequently solve the reduced PDE without reference-state resets. Width changes both the representation and its initial approximation. The common-test comparison holds residual rows fixed; each within-width enrichment adds tests. The fresh baseline reproduces the original accepted trajectory. All widths use the same disjoint observer, and the dense implementation carries no online speed claim.

| Width R | Weak rows M | Median / worst maximum field error (%) | Qualified cases | Prospective improvement versus baseline |
|---:|---:|---:|---:|---|
| 128 | 2048 | 1.322786 / 4.297893 | 6 / 6 | Baseline |
| 256 | 2048 | 0.461987 / 1.019474 | 6 / 6 | Pass |
| 256 | 4096 | 0.458617 / 0.834834 | 6 / 6 | Pass |
| 512 | 2048 | 0.221961 / 0.654043 | 6 / 6 | Pass |
| 512 | 8192 | 0.148417 / 0.391845 | 6 / 6 | Pass |

[Saved-array analysis](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-width-final.json) records original-baseline reproduction to a maximum error-curve difference of 4.857e-17. The prospective improvement rule requires every numerical gate, at least a 20% reduction of the worst maximum-over-time error, and no case's maximum error worsening by more than 2%. Initialization is included. The registered matrix is complete.

![Measured frozen-width PDE trajectories](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/weak-width-final-width-dynamics.png)

The [width-study pointwise audit](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-width-final-counterexamples.json) retains 14 of 60 tested transitions with increases at some saved times; the largest increase is 0.228597 percentage points. These post-run diagnostics are separate from the prospective maximum-over-time criterion.

| Change being compared | Case/transitions checked | Transitions with a pointwise increase | Largest increase (percentage points) |
|---|---:|---:|---:|
| Width at common tests | 18 | 6 | 0.226726 |
| Width and tests together | 30 | 6 | 0.228597 |
| Tests at fixed width | 12 | 2 | 0.000730 |

The largest increase occurs for case 5 at time 0.015, changing $(R,M)=(256,4096)$ to $(R,M)=(512,2048)$. The comparison type matters: some transitions change both width and test count.

![Measured fixed-test width counterexample](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/weak-width-final-counterexamples-width-counterexample.png)

### Complete-query cost

These are instrumented queries from the original host-resident physical input to six complete host-resident fields, including native initial fitting, transfers, evolution and recovery. Primary accuracy is the maximum error over the five evolved requested fields; initial error is retained separately. This differs from both the denser mechanism-history grid and the Burgers metric that includes initialization. Candidate selection and independent timing repetitions use exposed development cases within each allocation. Eligibility requires both the median and worst casewise evolved error no greater than the target; it does not require every individual case to match. A finite candidate search does not certify the fastest possible comparator.

**cost-p01**

| Run / target | Cases | Comparator | Target time (ms) | Comparator time (ms) | Comparator / target | Qualification |
|---|---:|---|---:|---:|---:|---|
| [cost-p01](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p01.json) / learned_head | 2 | pod | 31.5855 | — | — | no eligible selection comparator |
| [cost-p01](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p01.json) / learned_head | 2 | fom_m64_cnab2_s60 | 31.5855 | 68.2701 | 2.161439 | Pass: same-allocation accuracy and timing |
| [cost-p01](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p01.json) / learned_head | 2 | fom_m48_rk4_s15 | 31.5855 | 34.8359 | 1.102907 | Pass: same-allocation accuracy and timing |
| [cost-p01](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p01.json) / learned_span64 | 2 | pod_span32 | 36.6945 | 26.0182 | 0.709051 | Pass: same-allocation accuracy and timing |
| [cost-p01](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p01.json) / learned_span64 | 2 | fom_m64_cnab2_s40 | 36.6945 | 49.8211 | 1.357729 | Pass: same-allocation accuracy and timing |
| [cost-p01](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p01.json) / learned_span64 | 2 | fom_m28_rk4_s5 | 36.6945 | 14.8756 | 0.405390 | Pass: same-allocation accuracy and timing |

cost-p01 used **NVIDIA L40S**, job **7354479**. Ratios above one favor the target; no times are compared across jobs. A missing eligible POD comparator is an unresolved match in this candidate set, not a POD speed defeat.

The main table divides cohort median times. The following diagnostics pair case medians; outliers use the per-case Tukey rule on all retained confirmation repetitions, without deleting them.

| Target / comparator | Median paired time ratio | Cases with faster target | Timing outliers: target / comparator |
|---|---:|---:|---:|
| learned_head / fom_m64_cnab2_s60 | 2.161538 | 2 / 2 | 0 / 0 |
| learned_head / fom_m48_rk4_s15 | 1.102958 | 2 / 2 | 0 / 0 |
| learned_span64 / pod_span32 | 0.709127 | 0 / 2 | 1 / 2 |
| learned_span64 / fom_m64_cnab2_s40 | 1.358406 | 2 / 2 | 1 / 2 |
| learned_span64 / fom_m28_rk4_s5 | 0.405582 | 0 / 2 | 1 / 1 |

Accuracy of every retained confirmation subject, scored at the requested output times:

| Subject | Median / worst evolved maximum error (%) | Worst initial error (%) | Complete numerical records |
|---|---:|---:|---|
| learned_head | 0.065171 / 0.076069 | 0.039679 | Yes |
| learned_span64 | 0.609294 / 0.817315 | 0.039578 | Yes |
| pod_span32 | 0.516467 / 0.636411 | 0.130740 | Yes |
| fom_m28_rk4_s5 | 0.496102 / 0.604733 | 0.000129 | Yes |
| fom_m48_rk4_s15 | 0.017310 / 0.031768 | 0.000000 | Yes |
| fom_m64_cnab2_s40 | 0.107182 / 0.135496 | 0.000000 | Yes |
| fom_m64_cnab2_s60 | 0.047318 / 0.059689 | 0.000000 | Yes |

These cost-panel accuracies are not a new improvement over the mechanism panel: their saved-time sets differ. Initial error is reported separately; it is not part of the evolved-error eligibility statistic.

**cost-p03**

| Run / target | Cases | Comparator | Target time (ms) | Comparator time (ms) | Comparator / target | Qualification |
|---|---:|---|---:|---:|---:|---|
| [cost-p03](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p03.json) / learned_head | 16 | pod_head_m292_dt0.01 | 33.4974 | 32.1862 | 0.960858 | Pass: same-allocation accuracy and timing |
| [cost-p03](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p03.json) / learned_head | 16 | fom_m64_cnab2_s50 | 33.4974 | 59.4616 | 1.775113 | Pass: same-allocation accuracy and timing |
| [cost-p03](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p03.json) / learned_head | 16 | fom_m40_rk4_s10 | 33.4974 | 23.9027 | 0.713570 | Pass: same-allocation accuracy and timing |
| [cost-p03](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p03.json) / learned_span64 | 16 | pod_span32 | 37.1220 | 26.4462 | 0.712414 | Pass: same-allocation accuracy and timing |
| [cost-p03](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p03.json) / learned_span64 | 16 | fom_m64_cnab2_s40 | 37.1220 | 50.2342 | 1.353217 | Pass: same-allocation accuracy and timing |
| [cost-p03](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p03.json) / learned_span64 | 16 | fom_m32_cnab2_s20 | 37.1220 | 16.2370 | 0.437397 | Pass: same-allocation accuracy and timing |

cost-p03 used **NVIDIA L40S**, job **7355021**. Ratios above one favor the target; no times are compared across jobs. A missing eligible POD comparator is an unresolved match in this candidate set, not a POD speed defeat.

The main table divides cohort median times. The following diagnostics pair case medians; outliers use the per-case Tukey rule on all retained confirmation repetitions, without deleting them.

| Target / comparator | Median paired time ratio | Cases with faster target | Timing outliers: target / comparator |
|---|---:|---:|---:|
| learned_head / pod_head_m292_dt0.01 | 0.960995 | 0 / 16 | 11 / 6 |
| learned_head / fom_m64_cnab2_s50 | 1.776485 | 16 / 16 | 11 / 8 |
| learned_head / fom_m40_rk4_s10 | 0.713438 | 0 / 16 | 11 / 9 |
| learned_span64 / pod_span32 | 0.716496 | 0 / 16 | 7 / 6 |
| learned_span64 / fom_m64_cnab2_s40 | 1.353591 | 16 / 16 | 7 / 5 |
| learned_span64 / fom_m32_cnab2_s20 | 0.437530 | 0 / 16 | 7 / 4 |

Accuracy of every retained confirmation subject, scored at the requested output times:

| Subject | Median / worst evolved maximum error (%) | Worst initial error (%) | Complete numerical records |
|---|---:|---:|---|
| learned_head | 0.085866 / 0.185633 | 0.062162 | Yes |
| learned_span64 | 0.673384 / 1.264262 | 0.061962 | Yes |
| pod_head_m292_dt0.01 | 0.073134 / 0.134687 | 0.065178 | Yes |
| pod_span32 | 0.522900 / 0.957675 | 0.345731 | Yes |
| fom_m32_cnab2_s20 | 0.418537 / 0.996812 | 0.000013 | Yes |
| fom_m40_rk4_s10 | 0.035395 / 0.163923 | 0.000000 | Yes |
| fom_m64_cnab2_s40 | 0.091706 / 0.384403 | 0.000000 | Yes |
| fom_m64_cnab2_s50 | 0.058487 / 0.099053 | 0.000000 | Yes |

These cost-panel accuracies are not a new improvement over the mechanism panel: their saved-time sets differ. Initial error is reported separately; it is not part of the evolved-error eligibility statistic.

**cost-p04**

| Run / target | Cases | Comparator | Target time (ms) | Comparator time (ms) | Comparator / target | Qualification |
|---|---:|---|---:|---:|---:|---|
| [cost-p04](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p04.json) / learned_head_m292_dt0.02 | 16 | pod_head_m292_dt0.02 | 26.5064 | 26.6177 | 1.004200 | Pass: same-allocation accuracy and timing |
| [cost-p04](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p04.json) / learned_head_m292_dt0.02 | 16 | fom_m64_cnab2_s50 | 26.5064 | 59.4806 | 2.244011 | Pass: same-allocation accuracy and timing |
| [cost-p04](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p04.json) / learned_head_m292_dt0.02 | 16 | fom_m40_rk4_s10 | 26.5064 | 23.9535 | 0.903687 | Pass: same-allocation accuracy and timing |
| [cost-p04](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p04.json) / learned_head_m584_dt0.02 | 16 | pod_head_m292_dt0.02 | 27.3435 | 26.6177 | 0.973458 | Pass: same-allocation accuracy and timing |
| [cost-p04](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p04.json) / learned_head_m584_dt0.02 | 16 | fom_m64_cnab2_s50 | 27.3435 | 59.4806 | 2.175313 | Pass: same-allocation accuracy and timing |
| [cost-p04](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p04.json) / learned_head_m584_dt0.02 | 16 | fom_m40_rk4_s10 | 27.3435 | 23.9535 | 0.876022 | Pass: same-allocation accuracy and timing |
| [cost-p04](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p04.json) / learned_head | 16 | pod_head_m292_dt0.01 | 33.5568 | 32.1311 | 0.957512 | Pass: same-allocation accuracy and timing |
| [cost-p04](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p04.json) / learned_head | 16 | fom_m64_cnab2_s50 | 33.5568 | 59.4806 | 1.772532 | Pass: same-allocation accuracy and timing |
| [cost-p04](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p04.json) / learned_head | 16 | fom_m40_rk4_s10 | 33.5568 | 23.9535 | 0.713818 | Pass: same-allocation accuracy and timing |
| [cost-p04](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p04.json) / learned_span64 | 16 | pod_span32 | 37.1126 | 26.4742 | 0.713347 | Pass: same-allocation accuracy and timing |
| [cost-p04](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p04.json) / learned_span64 | 16 | fom_m64_cnab2_s40 | 37.1126 | 50.2315 | 1.353489 | Pass: same-allocation accuracy and timing |
| [cost-p04](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p04.json) / learned_span64 | 16 | fom_m32_cnab2_s20 | 37.1126 | 16.2207 | 0.437066 | Pass: same-allocation accuracy and timing |

cost-p04 used **NVIDIA L40S**, job **7355440**. Ratios above one favor the target; no times are compared across jobs. A missing eligible POD comparator is an unresolved match in this candidate set, not a POD speed defeat.

The main table divides cohort median times. The following diagnostics pair case medians; outliers use the per-case Tukey rule on all retained confirmation repetitions, without deleting them.

| Target / comparator | Median paired time ratio | Cases with faster target | Timing outliers: target / comparator |
|---|---:|---:|---:|
| learned_head_m292_dt0.02 / pod_head_m292_dt0.02 | 1.003479 | 9 / 16 | 13 / 8 |
| learned_head_m292_dt0.02 / fom_m64_cnab2_s50 | 2.243473 | 16 / 16 | 13 / 6 |
| learned_head_m292_dt0.02 / fom_m40_rk4_s10 | 0.901766 | 0 / 16 | 13 / 5 |
| learned_head_m584_dt0.02 / pod_head_m292_dt0.02 | 0.972102 | 1 / 16 | 8 / 8 |
| learned_head_m584_dt0.02 / fom_m64_cnab2_s50 | 2.176083 | 16 / 16 | 8 / 6 |
| learned_head_m584_dt0.02 / fom_m40_rk4_s10 | 0.874718 | 0 / 16 | 8 / 5 |
| learned_head / pod_head_m292_dt0.01 | 0.958545 | 1 / 16 | 11 / 13 |
| learned_head / fom_m64_cnab2_s50 | 1.772533 | 16 / 16 | 11 / 6 |
| learned_head / fom_m40_rk4_s10 | 0.712779 | 0 / 16 | 11 / 5 |
| learned_span64 / pod_span32 | 0.717648 | 0 / 16 | 8 / 8 |
| learned_span64 / fom_m64_cnab2_s40 | 1.354035 | 16 / 16 | 8 / 5 |
| learned_span64 / fom_m32_cnab2_s20 | 0.437074 | 0 / 16 | 8 / 10 |

Accuracy of every retained confirmation subject, scored at the requested output times:

| Subject | Median / worst evolved maximum error (%) | Worst initial error (%) | Complete numerical records |
|---|---:|---:|---|
| learned_head_m292_dt0.02 | 0.113308 / 0.195600 | 0.062162 | Yes |
| learned_head_m584_dt0.02 | 0.124681 / 0.200474 | 0.062162 | Yes |
| learned_head | 0.085866 / 0.185633 | 0.062162 | Yes |
| learned_span64 | 0.673384 / 1.264262 | 0.061962 | Yes |
| pod_head_m292_dt0.02 | 0.091243 / 0.151112 | 0.065178 | Yes |
| pod_head_m292_dt0.01 | 0.073134 / 0.134687 | 0.065178 | Yes |
| pod_span32 | 0.522900 / 0.957675 | 0.345731 | Yes |
| fom_m32_cnab2_s20 | 0.418537 / 0.996812 | 0.000013 | Yes |
| fom_m40_rk4_s10 | 0.035395 / 0.163923 | 0.000000 | Yes |
| fom_m64_cnab2_s40 | 0.091706 / 0.384403 | 0.000000 | Yes |
| fom_m64_cnab2_s50 | 0.058487 / 0.099053 | 0.000000 | Yes |

These cost-panel accuracies are not a new improvement over the mechanism panel: their saved-time sets differ. Initial error is reported separately; it is not part of the evolved-error eligibility statistic.

![Measured complete-query accuracy and cost](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/cost-p04-complete-query.png)

The pilot and expanded panels differ in cohort, candidate grid and initial-fit implementation. Their historical results are retained, but cross-panel timing changes do not isolate any one of those factors. Every reported speed comparison uses its own allocation.

Cold measurements, where retained in the experiment lane, cover asset activation, compilation and the first query within an already initialized runtime. They retain the operating-system page cache and exclude runtime/context startup. They cannot be relabeled process-cold deployment costs.

Latest panel's [single first-call records](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p04-cold.json), separate from the repeated warm timings:

| Subject | Case | Activation + compilation + first query (s) | Returned result |
|---|---:|---:|---|
| learned_head_m292_dt0.02 | 0 | 8.7949 | ok |
| learned_head_m584_dt0.02 | 0 | 9.0093 | ok |
| learned_head | 0 | 7.4434 | ok |
| learned_span64 | 0 | 3.8749 | ok |
| pod_head_m292_dt0.02 | 0 | 8.9902 | ok |
| pod_head_m292_dt0.01 | 0 | 9.2298 | ok |
| pod_span32 | 0 | 4.6756 | ok |
| fom_m32_cnab2_s20 | 0 | 2.4500 | ok |
| fom_m40_rk4_s10 | 0 | 2.4400 | ok |
| fom_m64_cnab2_s40 | 0 | 2.2268 | ok |
| fom_m64_cnab2_s50 | 0 | 2.1694 | ok |

These single-call records do not estimate timing variability, total training cost or lifetime break-even.

[cost-p03-audit raw audit](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p03-audit.json): 51 effective failed invocations among 6019 retained records, all outside accepted comparisons. The audit separately identifies 12 legacy rows with undefined error metrics. Such rows and their candidates never enter an eligible, selected or confirmed set; the raw records remain unchanged. Subsequent scoring explicitly fails nonfinite metrics.

[cost-p04-audit raw audit](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p04-audit.json): 57 effective failed invocations among 6370 retained records, all outside accepted comparisons. The audit separately identifies 0 legacy rows with undefined error metrics. Such rows and their candidates never enter an eligible, selected or confirmed set; the raw records remain unchanged. Subsequent scoring explicitly fails nonfinite metrics.

Retained cost attempts that stopped before accepted timing:

| Run / job | Outcome | Allocated GPU-seconds |
|---|---|---:|
| [cost-p02-failure](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p02-failure.json) / 7354813 | Native learned-head initialization on case14 did not reach declared stationarity; no timing result accepted. | 223 |

A scalar compiled replay subsequently tests the same targets and comparator grid with unchanged convergence thresholds. Its different GPU allocation prevents attributing any successful repair exclusively to the earlier batched arithmetic.

### Reproducibility

The [builder](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/build_2026_11_18_jcp_controlled_experiments.py) reads the [registered sources](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/sources.json), freezes the JSON evidence, and generates every numerical table and plot above. [Source hashes](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/source-manifest.json) and [derived values](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/measured-values.json) are saved separately. Full histories, configurations, numerical arrays, hardware logs and checksum manifests remain in the linked experiment lanes. No synthetic values enter these figures.

The [environment review](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/head-p02-environment.json) documents retained host CPU-affinity messages in job 7354339. Explicit GPU and numerical witnesses support accuracy-only admission; diagnostic elapsed times are not used for a speed claim. The message matches the [upstream XLA report](https://github.com/openxla/xla/issues/45484).

Independent numerical and measurement reviews: [head-full16-independent-review](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/head-full16-independent-review.json), [head-richer-independent-review](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/head-richer-independent-review.json), [weak-width-independent-review](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-width-independent-review.json), [cost-final-independent-review](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-final-independent-review.json).

Lane retention and closeout records: [weak-closeout](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-closeout.json), [head-closeout](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/head-closeout.json), [cost-closeout-final](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-closeout-final.json).

### Completed-job accounting

All attempts count, including failed and excluded runs. Allocated time includes setup, compilation, diagnostics and scoring; it is separate from online query timing.

| Lane | Attempts | Allocated GPU-seconds | Allocated GPU-hours | Record |
|---|---:|---:|---:|---|
| Head mechanism | 5 | 1179 | 0.327500 | [Ledger](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/head-job-ledger.json) |
| Weak tests and frozen width | 9 | 1418 | 0.393889 | [Ledger](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-job-ledger.json) |
| Complete-query cost | 4 | 2693 | 0.748056 | [7354813](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p02-failure.json) · [7354479](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p01-accounting.json) · [7355021](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p03-accounting.json) · [7355440](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p04-accounting.json) |
| **Total** | **18** | **5290** | **1.469444** | |

Unique numerical evidence, model checkpoints, source and manifests are backed up through the lane mirrors. Regenerable large sampled operators in the head/cost lanes also have verified local copies and regeneration recipes; their code-only mirrors do not contain every dense deployment array. This distinction is preserved in each lane's retention record.

<!-- GENERATED EXPERIMENT RESULTS END -->

## How to use these results in the JCP paper

| Proposed claim | Evidence supplied by this campaign | Boundary and next step |
|---|---|---|
| The nonlinear head reduces trajectory drift relative to its own full span. | Matched original-family dynamics, common initial states, converged solves, same-previous-state probes and richer-test controls. | Keep the claim within the tested family and horizon. A causal claim for the historical harder-flow example still needs its own matched control. |
| A small weak residual can coexist with substantial physical error. | Residual-versus-error probes, physically normalized tangent diagnostics and a disjoint residual observer. | Sampled local diagnostics support an explanation; a uniform stability theorem requires separately justified assumptions. |
| Test enrichment and frozen-width tuning can improve the learned-basis solver. | Separate test-count, projection-capacity and native-initialized PDE studies, with pointwise exceptions retained. | Use the measured maximum-over-time criterion. Neither monotone error at every time nor universal convergence follows. |
| The selected method offers a useful accuracy–cost tradeoff. | Complete physical-input/full-field-output measurements, strong finite comparator grids, separate selection and timing-confirmation repetitions. | Report all qualified target settings and negative comparisons. A finite search does not establish global superiority over POD or classical solvers. |
| Wider diagnostic models provide efficient hyper-reduced deployment. | The width study establishes accuracy using exact dense evaluation. | A deployment claim needs the chosen quadrature's reached-state residual/Jacobian checks and complete-query cost. Dense diagnostic accuracy alone does not establish it. |

The next scientific decision is to freeze the paper's method configuration and accuracy/output contract using the completed development results. Fresh-cohort confirmation can then assess that frozen choice. Reuse the already qualified temporal, mesh, quadrature, seed and physical-fidelity analyses identified in the [pre-campaign audit](2026-11-18-jcp-results-audit-and-priorities.md), checking that they apply to the selected configuration. Extend only the missing control or changed component. Full training/lifetime amortization and process-cold deployment remain separate from the warm complete-query measurements here.

## Glossary

**Head, bank and span:** the bank contains spatial functions; the head maps a small latent state to their coefficients; the span allows arbitrary coefficients. **Weak tests** measure smooth weighted averages of the PDE residual. **Residual visibility** describes how well those measurements detect changes in the physical state. **Tangent** means a field change reachable by a small coordinate change; mass normalization accounts for physical spatial volume. **Frame** is the coordinate shift used to follow moving structures.

**Quadrature** approximates integrals with weighted sample points. **Jacobian** measures local sensitivity. **POD** is a linear basis derived from training snapshots. **Matched accuracy** requires comparable physical error under the same reference and eligibility rules. **Complete query** includes all work needed to turn the agreed input into the requested answer. **Allocated GPU-hours** count the reserved GPU time actually charged to a job, including setup. **Development amendment** is a recorded change tested on exposed development cases. **Timing confirmation** uses new repetitions after comparator choices are frozen, on the same development cases. **Fresh-cohort confirmation** evaluates a frozen choice on previously untouched cases; these are different forms of confirmation.

**Case** means one initial field and its PDE parameters. **M** is the actual number of scalar weak residual equations. **Time step** is the interval between successive numerical updates. **Trajectory error** compares a predicted field with its reference at each saved time; the stated normalization specifies the field norm used as denominator. **Median / worst** denote the middle / largest casewise error, after first maximizing over the stated times. **Numerical qualification** requires the registered reference, operator, derivative and solver checks; an attractive error alone does not pass it. **Stationarity** is a small objective gradient and does not certify a global minimum. **Independent shell residual** measures equations outside every solved test space, using the same extra rows for every compared method. **Source hash** identifies the exact saved file bytes.

**FOM** means the full discretized PDE solver. **Same-mesh / any-mesh classical** distinguish comparators restricted to the requested mesh from those allowed a different internal mesh while returning the same requested fields. **Initial projection** is a model's approximation of the supplied physical initial field. **Mass projection** is the closest field in a linear span under the spatially weighted field norm. **Singular value** measures the sensitivity of residual measurements to a particular perturbation; its smallest value identifies a least visible direction. **Physical rank** counts independent field changes after normalization. **Outside-head norm fraction** measures the part of a specified perturbation that cannot be expressed by the local head-plus-frame tangent. It is a fraction of a norm, not of cases or directions.

**Median paired time ratio** first divides comparator time by target time for each case, then takes the median; it can differ from dividing two cohort medians. **Tukey timing outlier** means a repetition outside the lower or upper quartile by more than one and a half times the interquartile range. Outlier counts are descriptive; every repetition remains in the reported medians and raw records. **Frozen prefix** means the first ordered functions from an unchanged learned bank. **Projection floor** is the closest error available in that linear span for the specified reference field.
