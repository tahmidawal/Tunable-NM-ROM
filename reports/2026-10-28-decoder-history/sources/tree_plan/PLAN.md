# Quadtree NM-ROM development plan

Updated October 28, 2026 after three independent Claude reviews and coordinator reconciliation. This is a proposed architecture and staged validation plan; it contains no measured performance results. The next stage is a bounded representation diagnostic, not a full neural training campaign.

## Objective

Represent most of a solution with a small global bank and add local capacity only where it improves the accepted solution efficiently. Establish separately whether localization improves representation, whether the physics solve realizes that accuracy, whether adaptation is affordable, and whether learned features improve on classical local bases.

Success means lower complete cost at matched physical accuracy and identical requested output against the strongest applicable baseline. Smaller active dimension alone is not success.

## Architecture

For a scalar field, use

$$
u(x)=u_{\mathrm{bc}}(x)+B_0(x)c_0+
\sum_{P\in\mathcal A}w_P(x)
\sum_{j=1}^{k_P}c_{P,j}\phi_{\theta,\tau_P,j}(T_P^{-1}x).
$$

The boundary lift supplies prescribed values. The global bank captures broad behavior and communicates changes across the domain. Active patches supply additive detail. Each patch has its own active rank and coefficients. The map converts physical positions into patch coordinates; the optional bank type distinguishes explicitly tested scale or boundary classes.

Start with features linear in the online coefficients. Neural weights, when introduced, are trained offline and frozen during solves. An encoder or learned selector is not required.

Use fixed smooth overlapping windows. Retain parent features when adding children, and initialize new coefficients at zero to preserve the pre-solve field. This does not guarantee lower physical error after solving. Never renormalize existing windows silently. Additive corrections do not require their windows to sum to one.

Apply exact homogeneous boundary constraints in the initial problem, including local corrections. Start with interior sources and eligible interior correction patches; extend boundary coverage explicitly later. Compare every method on the same physical domain.

Keep three objects separate: the mesh used to obtain reference solutions, the active correction tree, and the quadrature used to integrate equations. Refining quadrature does not enlarge the representable solution space.

## What can adapt

| Action | Intended use | Required check |
|---|---|---|
| Increase global rank | Missing behavior is broad or shared across many patches | Improvement versus local alternatives |
| Increase local rank | Current patch scale can represent the missing detail efficiently | Added features supply independent useful directions |
| Activate child corrections | Smaller supports represent concentrated detail more efficiently | Test individual and bounded sibling-group actions |
| Improve tests or integration | Current representation is adequate but equations are poorly resolved | Stable trial/test pairing and independent integration convergence |
| Stop or use the classical solver | Accuracy is accepted, budget is exhausted, or no useful action remains | Report acceptance or failure reason and all incurred cost |

Keep configurations fixed within each coefficient solve. Coarsening is deferred until the steady mechanism works; before transients, it must include field/history transfer and verification.

## Adaptation and acceptance

First define the target error. For scalar diffusion, use energy error as the primary representation/solver diagnostic and report volume-weighted field error as well. Choose physical acceptance tolerances before launching experiments. Do not change the target norm to favor a configuration.

Separate three roles for functionals:

- Solve tests determine coefficients.
- Selection probes rank candidate actions.
- Audit probes evaluate the frozen policy independently. If their outcomes drive policy changes, they become development evidence and require a fresh final assessment.

Include global and local scales in selection and audit coverage. Different function names alone do not make probes informative: examine overlap with the solve-test span and sensitivity to fine trial directions. Boundary checks are required whenever boundary conditions are not imposed exactly.

Use a defined test-space metric, with Gram scaling or a justified approximation. Raw sums of overlapping residual entries can change merely because a region has more tests. A finite probe score is not a certified physical error bound. A restricted dual norm is a lower bound only under its precise mathematical assumptions.

For each adaptation round:

1. Solve the current coupled coarse/local system to a controlled tolerance.
2. Check integration, source resolution, boundary conditions, and trial/test conditioning.
3. Use multilevel selection probes to shortlist global-rank, local-rank and child-group actions.
4. Pair each trial enrichment with sufficiently resolving solve tests. Compare candidate outcomes using a common selection metric and integration accuracy.
5. Evaluate candidate benefit and complete additional cost, including rejected trials and checking overhead.
6. Accept a stable action with meaningful absolute improvement; a high improvement/cost ratio alone is insufficient.
7. Re-solve and recheck integration and physics at the accepted state. Stop, enrich further, or fall back according to registered budgets and stagnation rules.

For linear least squares with fixed operator, tests, weighting and quadrature, incremental QR can evaluate candidate updates without repeated full factorizations. Changed tests or quadrature require consistent rebuilding or a common union test space. Union-space overhead must be counted. This optimization is not assumed valid unchanged for nonlinear problems.

Keep the original compactly supported physical columns when screening redundancy. QR projections used to detect dependence must not silently replace them with globally supported basis functions. Monitor scaled rank and conditioning of the assembled system. More tests than coefficients is necessary here but does not by itself establish stability.

Candidate actions must include cases where multiple siblings work together. Fix the grouping rule and candidate budget in advance rather than searching every configuration without accounting for its cost.

## Training the local bank

The neural bank is introduced only after local POD and polynomial controls establish a functioning architecture.

### Data and coarse representation

Split whole problem instances into training, development and untouched test sets before extracting overlapping patches. Separate ordinary held-out testing from deliberate extrapolation in source width, position or tree depth.

Train and freeze the coarse bank with recorded provenance. Sharing training instances between the coarse and local banks is permitted; it is not automatically test leakage. Use separate-training-data or cross-fitting controls if needed to quantify optimistic residual training.

Compute best coarse reconstructions in a stated physical norm and form the missing solution detail. This is field error, not a pointwise PDE residual. On development data, compare best-approximation error with actual weak-solve error so test-space, solver and operator failures are not mistaken for missing capacity.

### Ordered features

Evaluate several local prefixes and refit coefficients for every prefix. Do not obtain a small-prefix fit by truncating coefficients fitted at maximum rank.

For a sampled active configuration C and solution instance s, the definitive reconstruction objective is

$$
\mathcal L(\theta)=
\mathbb E_{s,C}\left[
\min_{c_0,\{c_P\}}
\left\|u_s-u_{\mathrm{bc},s}
-B_0c_0-\sum_{P\in C}w_P
B_{\theta,P}^{(k_P)}c_P\right\|_{E_s}^2
\right].
$$

For diffusion, the squared energy norm integrates conductivity times squared physical gradients. Use volume-weighted field error as an additional reported metric; any mixed training loss must state its weights and physical units. Specify the configuration distribution and sampling weights before training.

Independent patch fits may initialize the bank, but evaluate and refine it through the assembled objective. Account for overlaps and changing coarse coefficients. Compute physical derivatives of both windows and mapped features. In two dimensions, for fixed similarity-mapped shapes, the field L2 integral scales with patch area while gradient energy is scale invariant; reference-coordinate losses cannot be substituted without the correct transformations.

Compare a shared bank with a child-detail variant only if redundancy warrants it. Child training must state which parent ranks/configurations generated its targets. A bank trained for one parent prefix may not transfer to others.

Keep ordering within common supports. Check useful new span, coefficient stability, and assembled conditioning. State any regularization and its bias. Local POD optimality applies only to the corresponding fixed common inner product and linear-space problem; heterogeneous metrics or coupled configurations require care.

Avoid per-instance network conditioning initially so feature tables can be reused. Add material or scale conditioning only if simpler shared or discrete bank types fail, and count the recurring evaluation cost.

## Example sequence

### A. Constant-conductivity Poisson: mechanism control

Use a square with exact homogeneous Dirichlet data, a broad smooth source and an additional movable localized smooth source. Specify source normalization: use a fixed-integral localized source for the first width study, with nonzero widths bounded away from the unresolved limit. Keep the initial source sufficiently interior; boundary behavior is a later registered extension.

Vary position continuously, including patch interiors, edges and dyadic corners. Vary width across resolved scales. Verify reference and source-integration convergence independently.

This example tests rank versus splitting, overlap and alignment, conditioning, and selector behavior. It does not establish general scale transfer or speed advantage. Include a simple source-informed activation rule and a translation-aware representation control.

### B. Smooth heterogeneous diffusion: discriminating extension

Use a broad source and a smooth localized conductivity variation, with conductivity uniformly positive and contrast bounded in advance. Vary its position, width and strength. Add source/material interactions only after the simpler family is understood.

Measure whether the coarse bank captures the nonlocal response and locals capture useful detail. Compare against material-informed activation as well as residual selection. Difficulty may still be partly predictable from inputs; no selector advantage is assumed.

### C. Moving-source transient diffusion: conditional later work

Proceed only after a useful steady regime exists. Add activation/deactivation, history transfer, temporal convergence and complete time-stepping cost. Keep spatial and temporal errors separate. Burgers and incompressible flow remain outside the initial scope.

## Staged experiments and decisions

| Stage | Work | Evidence required to continue | Interpretation if it fails |
|---|---|---|---|
| Preparation | Freeze norms, parameter ranges, source normalization, outputs, budgets and numerical gates; verify references | Data resolve the target and finest candidate support | Repair reference/discretization setup |
| Representation | Compare global POD, same-window local POD/polynomials, uniform and truth-assisted adaptive configurations | Useful held-out accuracy/capacity tradeoff with plausible complete-cost headroom | Narrow or stop the tested family; a greedy failure does not disprove all adaptive configurations |
| Physics fidelity | Solve fixed selected configurations before building an adaptive loop | Physical solve error remains close to best-approximation error; stable tests and quadrature | Repair solver/test/operator issues |
| Deployable selection | Compare rank/split/global actions against simple selectors | Physical gains survive audit checks and all selection costs | Improve or simplify the selector; do not blame representation automatically |
| Neural features | Train ordered local coordinate features in the same machinery | Improvement over local POD/polynomials in accuracy, useful evaluation cost or a specified transfer objective | Stop neural extension or retain classical local bases |
| Complete cost | Benchmark strongest configurations at matched accepted error and requested output | A reproducible cost advantage in a stated regime | No speed claim; report the scoped outcome |

Truth-assisted configuration selection is a diagnostic using the reference solution, not a deployable method. Label greedy searches as feasible searches, not certified optimal oracles.

Numeric gates remain to be registered before compute: error tolerances, reference margin, ranks/depths, conditioning limits, candidate count, absolute improvement floor, stagnation rule, runtime/compute caps and repetition counts. This document is an architecture plan, not an executable batch protocol. Choose gates from the requested physical accuracy and resource budget, not observed test outcomes.

## Fair comparisons and measurements

Use the same coarse bank, windows, active configurations and integration accuracy when isolating the effect of local feature type. Compare adaptive choices against uniform enrichment, global enrichment, residual-only selection and input-informed selection. Add translation-aware controls for the moving-source family.

For complete cost, use the strongest applicable tuned classical solvers: transform methods on the constant-coefficient box, factorization reuse for fixed operators, and adaptive FEM/multigrid for heterogeneous problems. Additional specialized updates belong only where their assumptions and implementation cost fit the workload.

Report active coefficients and total stored dictionary size separately. Count feature evaluation, source/input handling, assembly, conditioning checks, candidate trials, accepted solves, adaptation, audit checks, fallback and requested output. Report offline training/setup and amortization separately. Account for cached/precomputed operator blocks fairly on both sides. Overlapping and nested patches do not imply independent local algebra blocks.

Record per-instance physical errors, best-approximation versus solve gaps, probe underestimation, conditioning, selected actions, failures and complete cost. Report medians, tails and outlier counts; preserve timing repetitions. Timings must be paired on the same GPU with warmup, f64 and highest matmul precision. Follow the parent project's cluster isolation, GPU preflight, provenance and checksum-retention rules for any later numerical jobs.

## Next concrete task

Prepare the bounded representation diagnostic for example A using local POD and polynomial corrections. Specify the numerical protocol and reference construction before running it. The immediate deliverable is a comparison of adding global rank, local rank and grouped child corrections on the same saved solutions, with held-out alignment/width cases and explicit resource accounting. No neural training, learned selector or transient campaign is part of that first task.

## Review provenance

The three raw Claude reviews and the reconciled assessment are in [reviews/2026-10-28/](reviews/2026-10-28/). The [coordinator assessment](reviews/2026-10-28/coordinator-assessment.txt) records reviewer overstatements and mathematical corrections; raw recommendations are not all adopted.

Earlier architecture and literature background: [adaptive-tree report](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-26-adaptive-trees-for-coordinate-roms.md). This plan does not change accepted numerical findings in the parent project.

## Plain-language glossary

- **Bank / feature:** reusable functions whose weighted sum represents a field.
- **Coefficient / local rank k:** a solved weight; rank is the number of active weights in a patch.
- **Prefix:** the first selected number of features in an ordered bank.
- **Patch / parent / child:** a spatial region, its coarser ancestor, or one of its subdivisions.
- **Window / support:** a function that localizes a correction; support is where it can be nonzero.
- **Boundary lift:** a field supplying prescribed boundary values.
- **Weak test:** a smooth function used to check an integrated PDE balance.
- **Quadrature:** points and weights used to approximate an integral.
- **Energy error:** gradient error weighted by conductivity for the diffusion problem.
- **Volume-weighted field error:** field mismatch integrated with physical area or volume.
- **POD:** a linear basis fitted to compress training snapshots under a specified norm.
- **Projection / best approximation:** the best fit within a chosen function space, using the known reference.
- **Conditioning / Gram matrix / QR:** sensitivity of a solve; a matrix of function inner products; a stable matrix factorization used for solving and checking independent directions.
- **FEM / multigrid:** finite-element discretization; a solver using several spatial scales.
- **Audit / held-out / extrapolation:** independent checking; instances excluded from fitting/tuning; testing beyond trained parameter or scale ranges.
- **Fallback / amortization:** use of a conventional solver when the reduced method fails; spreading one-time setup cost over repeated queries.
