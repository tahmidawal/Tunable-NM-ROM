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

**Numerically qualified development comparison — head-p02.** Errors below are maxima over each evolved trajectory, normalized by its original physical initial field. Both primary arms have identical reconstructed initial states. Median and worst are across the same cases.

| Scalar weak rows M | Time step | Head median / worst error (%) | Span median / worst error (%) | Cases with lower head error |
|---:|---:|---:|---:|---:|
| 292 | 0.02 | 0.100137 / 0.116196 | 0.904549 / 1.008631 | 2 / 2 |
| 584 | 0.02 | 0.112733 / 0.135521 | 0.766687 / 1.063164 | 2 / 2 |
| 584 | 0.01 | 0.088656 / 0.093940 | 0.611201 / 0.820088 | 2 / 2 |

![Measured matched head/span trajectories](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/head-p02-trajectories.png)

The admitted comparison isolates the head constraint under the registered numerical settings.
Its scope excludes a universal stability guarantee, fresh-test generalization, a global manifold distance, and a complete-query speedup.

### Weak-test visibility

Every panel uses the same-grid, same-time-step discrete Burgers reference. No continuum-error claim is made. Maximum field error includes the initial saved time. Spectral visibility and physical accuracy are separate diagnostics.

| Retained run | Case | M | Maximum field error (%) | Numerical qualification | Independent shell residual |
|---|---:|---:|---:|---|---:|
| [weak-p02](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-p02.json) | 0 | 256 | 1.904368 | Pass | 0.00243535 |
| [weak-p02](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-p02.json) | 0 | 512 | 1.462042 | Pass | 0.00196867 |
| [weak-p02](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-p02.json) | 0 | 1024 | 1.190747 | Pass | 0.00164771 |
| [weak-p02](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-p02.json) | 0 | 2048 | 1.014660 | Pass | 0.00139583 |
| [weak-p02](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-p02.json) | 1 | 256 | 0.215723 | Provisional: failed | 0.000162037 |
| [weak-p02](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-p02.json) | 1 | 512 | 0.215723 | Provisional: failed | 0.000116262 |
| [weak-p02](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-p02.json) | 1 | 1024 | 0.215723 | Provisional: failed | 9.31746e-05 |
| [weak-p02](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/weak-p02.json) | 1 | 2048 | 0.215723 | Provisional: failed | 8.1317e-05 |

Adding unchanged rows cannot decrease the smallest singular value at an identical state. That algebraic property alone does not demonstrate improved physical prediction.

### Reproducibility

The [builder](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/build_2026_11_18_jcp_controlled_experiments.py) reads the [registered sources](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/sources.json), freezes the JSON evidence, and generates every numerical table and plot above. [Source hashes](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/source-manifest.json) and [derived values](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/measured-values.json) are saved separately. Full histories, configurations, numerical arrays, hardware logs and checksum manifests remain in the linked experiment lanes. No synthetic values enter these figures.

The [environment review](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-11-18-jcp-controlled-experiments-assets/evidence/head-p02-environment.json) documents retained host CPU-affinity messages in job 7354339. Explicit GPU and numerical witnesses support accuracy-only admission; diagnostic elapsed times are not used for a speed claim. The message matches the [upstream XLA report](https://github.com/openxla/xla/issues/45484).

<!-- GENERATED EXPERIMENT RESULTS END -->

## Glossary

**Head, bank and span:** the bank contains spatial functions; the head maps a small latent state to their coefficients; the span allows arbitrary coefficients. **Weak tests** measure smooth weighted averages of the PDE residual. **Residual visibility** describes how well those measurements detect changes in the physical state. **Tangent** means a field change reachable by a small coordinate change; mass normalization accounts for physical spatial volume. **Frame** is the coordinate shift used to follow moving structures.

**Quadrature** approximates integrals with weighted sample points. **Jacobian** measures local sensitivity. **POD** is a linear basis derived from training snapshots. **Matched accuracy** requires comparable physical error under the same reference and eligibility rules. **Complete query** includes all work needed to turn the agreed input into the requested answer. **Allocated GPU-hours** count the reserved GPU time actually charged to a job, including setup. **Development amendment** is a recorded change tested on exposed development cases. **Confirmation** evaluates a frozen choice on previously untouched cases.

**Case** means one initial field and its PDE parameters. **M** is the actual number of scalar weak residual equations. **Time step** is the interval between successive numerical updates. **Trajectory error** compares a predicted field with its reference at each saved time; the stated normalization specifies the field norm used as denominator. **Median / worst** denote the middle / largest casewise error, after first maximizing over the stated times. **Numerical qualification** requires the registered reference, operator, derivative and solver checks; an attractive error alone does not pass it. **Stationarity** is a small objective gradient and does not certify a global minimum. **Independent shell residual** measures equations outside every solved test space, using the same extra rows for every compared method. **Source hash** identifies the exact saved file bytes.
