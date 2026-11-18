# Controlled experiments for the JCP paper

This is the development protocol for the experiments authorized on November 18, 2026. New measured development runs and their qualification status are generated below from retained records; failed attempts remain visible.

The central questions are whether the nonlinear head improves trajectories under matched numerical controls, whether richer weak tests detect errors missed by the minimized residual, and whether the selected method saves complete-query cost at matched accuracy. The earlier optimistic sketches remain hypotheses, not acceptance criteria or promised outcomes.

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
| [head-p01](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/head-p01.json) / 7354236 | 2 | Provisional: failed | all_stationary, ic_fit_passed, reference_passed |
| [head-p02](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/head-p02.json) / 7354339 | 2 | Pass | None |
| [head-p03](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/head-p03.json) / 7354386 | 16 | Provisional: failed | ic_fit_passed |
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

### Weak-test visibility

Every panel uses the same-grid, same-time-step discrete Burgers reference. No continuum-error claim is made. Maximum field error includes the initial saved time. Spectral visibility and physical accuracy are separate diagnostics.

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

### Complete-query cost

These are instrumented queries from the original host-resident physical input to complete host-resident fields, including native initial fitting, transfers, evolution and recovery. Accuracy is scored on the six requested output fields, not the denser mechanism-history grid. Candidate selection and independent timing repetitions use exposed development cases within each allocation. Eligibility requires both median and worst error no greater than the target. A finite candidate search does not certify the fastest possible comparator.

| Run / target | Cases | Comparator | Target time (ms) | Comparator time (ms) | Comparator / target | Qualification |
|---|---:|---|---:|---:|---:|---|
| [cost-p01](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p01.json) / learned_head | 2 | pod | 31.5855 | — | — | no eligible selection comparator |
| [cost-p01](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p01.json) / learned_head | 2 | fom_m64_cnab2_s60 | 31.5855 | 68.2701 | 2.161439 | Pass: same-allocation accuracy and timing |
| [cost-p01](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p01.json) / learned_head | 2 | fom_m48_rk4_s15 | 31.5855 | 34.8359 | 1.102907 | Pass: same-allocation accuracy and timing |
| [cost-p01](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p01.json) / learned_span64 | 2 | pod_span32 | 36.6945 | 26.0182 | 0.709051 | Pass: same-allocation accuracy and timing |
| [cost-p01](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p01.json) / learned_span64 | 2 | fom_m64_cnab2_s40 | 36.6945 | 49.8211 | 1.357729 | Pass: same-allocation accuracy and timing |
| [cost-p01](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/cost-p01.json) / learned_span64 | 2 | fom_m28_rk4_s5 | 36.6945 | 14.8756 | 0.405390 | Pass: same-allocation accuracy and timing |

cost-p01 used **NVIDIA L40S**, job **7354479**. Ratios above one favor the target; no times are compared across jobs. A missing eligible POD comparator is an unresolved match in this candidate set, not a POD speed defeat.

![Measured complete-query accuracy and cost](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/cost-p01-complete-query.png)

Cold measurements, where retained in the experiment lane, cover asset activation, compilation and the first query within an already initialized runtime. They retain the operating-system page cache and exclude runtime/context startup. They cannot be relabeled process-cold deployment costs.

### Reproducibility

The [builder](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/build_2026_11_18_jcp_controlled_experiments.py) reads the [registered sources](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/sources.json), freezes the JSON evidence, and generates every numerical table and plot above. [Source hashes](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/source-manifest.json) and [derived values](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/measured-values.json) are saved separately. Full histories, configurations, numerical arrays, hardware logs and checksum manifests remain in the linked experiment lanes. No synthetic values enter these figures.

The [environment review](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/head-p02-environment.json) documents retained host CPU-affinity messages in job 7354339. Explicit GPU and numerical witnesses support accuracy-only admission; diagnostic elapsed times are not used for a speed claim. The message matches the [upstream XLA report](https://github.com/openxla/xla/issues/45484).

<!-- GENERATED EXPERIMENT RESULTS END -->

## Glossary

**Head, bank and span:** the bank contains spatial functions; the head maps a small latent state to their coefficients; the span allows arbitrary coefficients. **Weak tests** measure smooth weighted averages of the PDE residual. **Residual visibility** describes how well those measurements detect changes in the physical state. **Tangent** means a field change reachable by a small coordinate change; mass normalization accounts for physical spatial volume. **Frame** is the coordinate shift used to follow moving structures.

**Quadrature** approximates integrals with weighted sample points. **Jacobian** measures local sensitivity. **POD** is a linear basis derived from training snapshots. **Matched accuracy** requires comparable physical error under the same reference and eligibility rules. **Complete query** includes all work needed to turn the agreed input into the requested answer. **Allocated GPU-hours** count the reserved GPU time actually charged to a job, including setup. **Development amendment** is a recorded change tested on exposed development cases. **Confirmation** evaluates a frozen choice on previously untouched cases.

**Case** means one initial field and its PDE parameters. **M** is the actual number of scalar weak residual equations. **Time step** is the interval between successive numerical updates. **Trajectory error** compares a predicted field with its reference at each saved time; the stated normalization specifies the field norm used as denominator. **Median / worst** denote the middle / largest casewise error, after first maximizing over the stated times. **Numerical qualification** requires the registered reference, operator, derivative and solver checks; an attractive error alone does not pass it. **Stationarity** is a small objective gradient and does not certify a global minimum. **Independent shell residual** measures equations outside every solved test space, using the same extra rows for every compared method. **Source hash** identifies the exact saved file bytes.
