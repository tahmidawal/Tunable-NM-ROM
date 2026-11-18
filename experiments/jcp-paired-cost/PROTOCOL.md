# Paired complete-query cost and independent mechanism review

This is a prospective development protocol, not a measured result. It prepares a cost comparison for the selected original-family Navier–Stokes model and independently reviews the neighboring head and weak-test experiments.

## Scope and decision order

The initial physical problem is the clean mesh-64 original-family NS3D problem, using the existing frozen rank-64 coordinate bank and latent-dimension-8 head. No architecture, training, final cohort, or historical result is changed. The parent chooses the scientific configuration after the matched mechanism/visibility pilots. This lane may not submit its cost job before that choice. The first job uses one GPU for at most thirty minutes; all jobs together must fit this lane's two allocated GPU-hours, including failed jobs. Subsequent choices require a recorded prospective amendment and remaining budget.

The existing development seed is eligible for development only. Its configuration, training seed and all excluded seeds are pinned before staging. Fresh final confirmation is not authorized. Repetition confirmation below means new timing repetitions on the development cases, not a fresh scientific test cohort.

The head/span mechanism experiment starts both trajectories from an identical reconstructed head field and frame. This cost experiment instead gives every method the identical original physical initial field and viscosity, then includes its native initialization. Native head fitting and full-span projection have different representation errors; that is part of complete-query performance, not evidence identifying the head mechanism.

## Common scientific contract

Every method receives the same host-resident f64 initial velocity field on the requested mesh and the same viscosity. Every method returns the same six complete f64 velocity fields at the same requested times and mesh. Parameter-only initial-condition shortcuts are prohibited. A scalar parameter may control the PDE only when available to every method. No method may reuse the last repetition's evolved state, fitted code, frame, coefficients, or input transform.

Primary query timing starts with that physical input and includes host/device transfer, frame finding, centering, projection or code fitting, input restriction/FFT, time stepping, output resampling/reconstruction, synchronization, and the host output. A secondary device-resident query may be recorded, but is labeled separately and cannot substitute for the primary measurement. Pretrained parameters and assembled weak operators are persistent deployment assets. Their construction is measured separately as offline setup; asset transfer and compilation are separately measured cold deployment costs. No unmeasured cold/offline cost is presented as measured, and no additive estimate is labeled a directly timed query.

Use each method's best eligible mathematical configuration from the development candidate set, not equal iteration counts masquerading as matched cost. Head and span share the selected bank and exact weak operator. The POD competitor includes the frozen POD head and full-span bank and any previously eligible lower-rank candidates. Classical candidates include both CNAB2 and RK4 spectral solvers on the scoring mesh and allowed coarser meshes, with restriction and full output interpolation included. Keep any-mesh and same-mesh comparisons separate. These classical methods are already spectral; inventing a second similarly named baseline adds no evidence.

Physical accuracy is scored against regenerated, reproducible truth. For case $i$ and output time $t_j$,

$$e_{ij}=\frac{\|u^{\mathrm{method}}_i(t_j)-u^{\mathrm{truth}}_i(t_j)\|_2}{\|u^{\mathrm{truth}}_i(0)\|_2},\qquad E_i=\max_{j>0}e_{ij}.$$

For a fixed target arm, a comparator is eligible only when its casewise evolved-error median and maximum both do not exceed that target's corresponding values. The two comparisons use the same cases, denominator and returned output. Both arms must have finite outputs for every case and no recorded solver failure. Initial error, final error, all per-time errors and failure counts are retained. The mathematical eligibility rule is frozen before observations; target values are computed from the designated selection block, never adjusted to make a winner. Reference uncertainty or a separate relaxed-accuracy sensitivity must be explicit and cannot replace this primary rule.

## Timing and selection

Before each timed block, burn in the GPU. Build/compile candidate functions explicitly before warm timing and persist those durations separately. Check any optimized implementation against its pinned equivalent on all pilot cases before using its cost, including output difference and error difference. Full GPU arrays are function arguments, avoiding captured-large-constant recompilation.

Use randomized, interleaved order across methods and cases, with a fixed recorded order seed. Every timed invocation returns the exact output used to compute that invocation's accuracy. Synchronize before stopping a GPU timer; convert to host before stopping the primary full-query timer. Store every repetition, case, order position, predecessor, output checksum, accuracy vector, failure and elapsed time. Summaries use case medians, overall median and outlier counts, not a favorable minimum.

There are distinct selection and repetition-confirmation blocks in the same allocation. Select the fastest eligible POD, same-mesh classical and any-mesh classical candidates using selection-block timings and accuracies only; freeze their identities to a file before the confirmation block. Confirm only those selected identities and the target arms, with new randomized repetitions. Check median/maximum accuracy eligibility again using confirmation outputs. A change in eligibility is reported as a failed match, not repaired by substituting a different winner. Report start/end sentinel drift and first/last repetition-block drift; the frozen drift threshold is ten percent. Failure of a timing gate prevents a speed claim but does not erase the accuracy observation.

Cold model activation is a separately labeled directly timed first call after asset activation and compiled-cache reset, not “warm minus compile” arithmetic. The pilot keeps the JAX/CUDA runtime and device context alive and does not flush the operating-system page cache; runtime/context startup is excluded. Consequently it is not a new-process cold deployment measurement. The full source fixture records this scope. Query timings include conversion of returned solver diagnostics and are labeled instrumented complete queries. Warm query timing always repeats physical cold initialization even when compiled code is reused.

## Independent review checks

For P1, verify exact coefficient/frame equivalence at the common reconstructed-head initial condition, keep its discrepancy from original truth visible, and confirm no silent divergence projection changes only one arm. Count all free-frame variables. Compare the same residual and time grid with demonstrated tight/tighter stationarity, retaining nonconverged cases. A lower head residual cannot be interpreted as superiority to a global full-span optimum of the same residual.

For P2, normalize tangent perturbations in physical field norm and report retained rank/cutoff. Its selected Burgers span has no frame variables. For any free-frame NS diagnostic such as P1, include the three frame tangent directions and their dependencies. Record actual scalar test rows separately from any Fourier-mode label. The restricted weak Jacobian must act on the same physical perturbation whose norm is used. Verify exact weak action or a quantified quadrature approximation and retain finite-difference/action checks. Do not mistake an arbitrary code-coordinate condition number for physical observability or claim stability from a singular value alone.

For both, record frozen asset and source hashes, regenerated-data seeds, common truth, discretization, convergence, failures, GPU type, job ID and budget. No test-cohort tuning. Review concrete discrepancies; additional speculative requirements do not block a well-scoped pilot.

## Execution and preservation

Only the owned namespace in `ownership.json` is used. Each job has its own directory and `gpu` partition allocation, mandatory backend preflight, f64 and highest matmul precision. Check the queue before/after submit, stage directly to paralab, regenerate data, and never use login `/tmp`. Pull all logs, sources, configs, provenance and outputs with checksums, verify them, then remove only the checksum-verified duplicate in the owned job directory. CPU fallback, OOM, captured constants, truncated logs or full disks invalidate GPU acceptance. No bare `scancel`.

Every commit is backed up through the lane's code-only mirror script. Raw results remain reviewable even if a hypothesis fails. Historical classical wins and invalidated timings stay explicit. The parent owns the canonical report and lab-log coordination; no neighboring worktree is edited and no branch is merged without the user's merge decision.

## Glossary

- **Arm:** one fixed method and numerical configuration.
- **Bank:** the frozen set of spatial decoder functions; **rank** is its coefficient count.
- **Head:** the frozen nonlinear map from a small latent vector to bank coefficients.
- **Span:** all coefficient combinations in a fixed bank.
- **Frame:** the translating spatial coordinate system solved alongside the model coefficients.
- **Weak operator:** the PDE residual integrated against fixed smooth test functions.
- **Scalar test row:** one real scalar residual equation, distinct from a loosely named Fourier mode.
- **POD:** a basis found by proper orthogonal decomposition of training snapshots.
- **CNAB2 / RK4:** second-order Crank–Nicolson/Adams–Bashforth and fourth-order Runge–Kutta time integrators.
- **Native initialization:** a method's own fit or projection of the supplied physical input.
- **Shared initial condition:** an identical reconstructed field and frame supplied to both mechanism arms.
- **Selection block:** repetitions used to choose a candidate; **confirmation block:** new repetitions that assess the frozen choice.
- **Eligible:** meets both the median and worst physical-error comparison and has no failed case.
- **Same-mesh / any-mesh:** classical competitors restricted to the scoring mesh / allowed to solve coarsely and return the scoring mesh.
- **Warm query:** compiled persistent model, with input preparation and initialization recomputed for every query.
- **Cold deployment:** first query including asset activation and compilation, measured separately.
- **Offline setup:** work such as training and operator construction performed before queries exist.
- **Sentinel:** a fixed method measured near the start/end to detect timing drift.
- **Tangent:** an infinitesimal physical-field change permitted by a model and its frame.
- **Stationarity:** a small residual gradient indicating local optimization convergence, not proof of a global optimum.
