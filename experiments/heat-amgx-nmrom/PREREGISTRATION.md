# Heat–AmgX pure NM-ROM preregistration

Status: **locked before implementation or numerical data generation** on 2026-08-19. This
document fixes the benchmark, data partitions, gates, search budget, confirmation protocol,
and stop rules. Later changes must be appended as deviations; they may not silently replace
this plan.

## Question and fixed success criterion

Can a genuinely pure nonlinear coordinate-manifold ROM solve a nonseparable, heterogeneous,
temperature-dependent 2-D Heat problem accurately and at least ten times faster end to end than
a tuned, like-for-like Newton–FGMRES–AmgX full-order model on the same GPU?

The experiment passes only if one locked method simultaneously achieves on the untouched
confirmation cohort:

- trajectory mean relative cell-volume-weighted L2 error at most `1e-3`;
- worst-trajectory relative L2 error at most `3e-3`;
- zero divergent, non-finite, truncated, or tolerance-censored trajectories;
- median **first-use end-to-end** speedup at least `10x` against the selected tuned FOM; and
- trajectory/case-clustered 95% bootstrap speedup lower bound at least `8x`.

The following supporting gates are conjunctive: reference/discretization error at most `1e-4`,
decoder reconstruction mean at most `3e-4`, full weak-ROM mean at most `7e-4`, hyper-reduced
mean at most `1e-3`, and hyper-reduced/full degradation at most `1.05`. No gate will be weakened
after seeing a result.

## Benchmark

### Domain and finite-volume truth

The fixed domain is the L shape

`Omega = [0,1]^2 \\ ([0.58,1] x [0,0.42])`.

The FOM is a conservative, cell-centred finite-volume discretisation on the active cells of a
uniform Cartesian background grid. Face conductivities use harmonic averaging. The re-entrant
notch is a true boundary, not filled with penalised cells. The state equation is

`c(x,T;mu) dT/dt - div(k(x,T;mu) grad T) = q(x,t;mu)`.

Time integration is backward Euler over `t in [0,0.4]`. The development default is 80 equal
steps; spatial and temporal refinement are both included in the truth-convergence gate. The
deployment resolution is selected before learning as the coarsest grid whose trajectories agree
with a `2N`, `dt/2` reference to the stated `1e-4` gate on all 12 convergence cases. If none of
`N={128,192,256,384}` passes, the benchmark is infeasible under this budget and the result is a
numerical-truth negative, not an NM-ROM result.

Boundary conditions are mixed:

- homogeneous Dirichlet on the outer left edge and the extant outer bottom edge;
- insulated Neumann on both faces of the re-entrant notch; and
- parameterised Robin cooling on the outer top and outer right edges.

The decoder enforces the Dirichlet condition exactly through an analytic boundary factor. Robin
and Neumann terms remain in the FOM-exact weak operator.

### Nonseparable coefficients and forcing

Conductivity is positive by construction and contains a background, two rotated anisotropic
Gaussian inclusions, and one curved channel. Its log field is multiplied by a bounded
temperature factor. Heat capacity has an independently translated inclusion and a quadratic,
bounded temperature factor. A Gaussian source follows a quadratic Bezier path whose endpoints
and bend are parameters. The Robin coefficient is also parameterised.

The parameter vector contains the two inclusion contrasts and centres, channel orientation and
contrast, conductivity temperature coefficient, capacity contrast and temperature coefficient,
source path endpoints/bend, source width/amplitude, and Robin coefficient. Ranges will be encoded
once in the implementation and unit-tested for positivity and nonseparability. Parameters are
sampled by scrambled Sobol draws generated from fixed seeds, never by selecting favourable cases.

The initial temperature is zero and satisfies the Dirichlet boundary condition. Coefficient and
source ranges must keep all reference trajectories finite without post-draw rejection. If a draw
is invalid because of an implementation-independent physical bound, the whole Sobol block and
the bound are recorded before redrawing; individual trajectories are never replaced.

## Numerical truth and tuned FOM

The nonlinear residual and exact sparse Jacobian are shared by truth, deployment FOM, full weak
ROM, and hyper-reduced ROM. Newton uses residual-scaled stopping plus an update test, a monotone
line search, and no fixed iteration count. Reported solution error and cost come from the same
solver invocation.

The deployment baseline is NVIDIA AmgX FGMRES with AmgX AMG as a preconditioner. The sparsity
pattern is fixed; coefficients are replaced each Newton step, and hierarchy reuse/rebuild policy
is tuned only on the development cohort. AmgX is not described as a nonlinear solver. At least
these configurations are compared in the same GPU job:

1. classical AMG, conservative strength threshold, block-Jacobi or multicolour GS smoother;
2. aggregation AMG with the same FGMRES outer tolerance ladder;
3. one reuse-heavy variant and one rebuild-heavy variant; and
4. a credible available control: cuDSS if present, otherwise a matrix-free JAX FGMRES/PCG with
   Jacobi or geometric coarse-grid preconditioning. The absence of PETSc, PCAMGX, hypre, or
   cuDSS is recorded rather than papered over.

The selected baseline minimises median time subject to matching the tight-reference trajectory
within `1e-4`, zero nonlinear failures, and a returned relative nonlinear residual at most the
locked deployment tolerance. Linear forcing terms are calibrated rather than inherited from an
over-solved truth generator.

Timing reports both:

- **first use**, including matrix construction, transfers, AmgX resources/config/setup, all
  Newton/FGMRES work, and requested full-field output; and
- **amortised reuse**, where only setup demonstrably reusable across a stream is divided by the
  fixed 32-trajectory confirmation batch.

The headline `10x` and `8x` gates use first-use time. Amortised results are secondary. Every timed
block receives a GPU burn-in, every repetition is persisted, and method order is balanced AB/BA
within each trajectory. Wall clock is compared only within one allocation on one physical GPU.

## Data partitions

All splits are disjoint Sobol blocks and are generated from seed on the cluster:

| purpose | count | seed/block | permitted use |
|---|---:|---|---|
| truth convergence | 12 | `2026081901` | choose deployment `N,dt` only |
| training | 256 trajectories | `2026081902` | optimisation and EQ fit subset |
| calibration | 32 trajectories | `2026081903` | early stopping, losses, solver and FOM tuning |
| development test | 32 trajectories | `2026081904` | staged gates and architecture selection |
| untouched confirmation | 32 trajectories | `2026082799` | exactly one final locked evaluation |

Each trajectory stores all time slices for accuracy, but training may subsample space/time. No
confirmation parameters, fields, latents, timings, or aggregate metrics may be generated before
the method, stopping rules, FOM configuration, and output protocol are locked in a committed
manifest.

## Pure nonlinear learned method

The decoder is `T(x;z) = lift(x) + d_D(x) * D_theta(features(x), z)`, with a compact coordinate
trunk and grouped FiLM modulation. It has no POD basis, linear reduced correction, grid-tied
factor, stored FOM field, or full-grid online feature. Candidate latent sizes are `k={32,48,64}`.
Fourier frequencies are capped below the training-grid Nyquist limit and are checked off grid.

The online initial latent and each warm prediction use only known parameters, time, and a bounded
history of latent states. The candidate is then corrected by a trust-region weak LSPG solve. At
most two Gauss–Newton Jacobians per time step are allowed in the deployable bracket; a zero-step
predictor-only arm and a one-step arm are mandatory controls. Trust radii are calibrated from the
training latent cloud and cannot expand beyond its recorded envelope.

Training combines relative reconstruction, volume-weighted H1/Sobolev, weak-operator, temporal
consistency, and decoder-Jacobian conditioning terms. Ablations decide whether each nonzero term
earns its cost; labels do not cross data partitions. Reconstruction is evaluated at native and
off-grid coordinates.

Smooth weak test functions are low graph-diffusion modes on the active L-shaped FV mesh. Their
boundary terms and the nonlinear FOM operator are retained exactly. Candidate counts satisfy
`M >= 2k` and `M > k` comfortably. The full weak objective sums all active cells. Empirical
quadrature uses `m` near `4M` and nonnegative weights fitted only from local weak integrands of
**decoder-output** training/calibration snapshots, never FOM residual snapshots. The cold start
uses the same hyper-reduced path.

## Finite staged search

The hard cap is 38 submitted GPU cells and 96 allocated GPU-hours, plus at most four exact
replacement cells for documented infrastructure failures. Local work is smoke testing only,
under one minute per `jaxrun` process and at most three concurrent processes.

### Stage 0 — truth and baseline, at most 6 cells

- one isolated AmgX build/backend probe and one FOM unit/smoke cell;
- at most two spatial/temporal convergence cells; and
- at most two same-GPU FOM configuration/tolerance cells.

Stop if no deployment grid passes truth error or no AmgX route produces independently verified
residuals. An unavailable AmgX build after two distinct supported CUDA/GCC combinations is an
infrastructure negative; do not replace AmgX with an unnamed solver.

### Stage 1 — representation, at most 12 cells

Screen at most six `(k,width,depth,group-size,bandwidth)` designs using short training and the
calibration set. Fully train at most three survivors and at most one loss/curriculum refinement
per survivor. A candidate advances only if reconstruction mean is at most `3e-4`, off-grid error
is at most 1.25 times native error, and no trajectory exceeds `1e-3` reconstruction error.

Stop the decoder search if all six screens exceed `6e-4`, if three fully trained candidates miss
`3e-4`, or if the best oracle-latent manifold floor exceeds `7e-4`. Do not hide a failed pure
manifold behind POD plus a nonlinear corrector.

### Stage 2 — predictor and full weak ROM, at most 8 cells

For each of at most two decoder survivors, compare zero/one/two-step history-aware prediction and
at most two trust-region/stopping settings. Search `M` only in `{2k,3k,4k}`. Advance only if the
full weak-ROM mean is at most `7e-4`, worst is at most `2.5e-3`, and censoring is zero.

Stop if the oracle previous-latent warm start misses the full-ROM gate, or if every allowed
two-Jacobian method misses it. More nonlinear iterations are outside the deployable budget.

### Stage 3 — empirical quadrature and kernels, at most 6 cells

For at most two full-ROM survivors, fit decoder-output NNLS quadrature at `m={4M,5M}`; a `3M`
diagnostic may run but cannot replace the required `4M` arm. Refit whenever `N` or `M` changes.
Advance only if mean error is at most `1e-3`, degradation is at most `1.05`, worst error is at
most `3e-3`, and censoring is zero. Kernel work may change batching/caching/fusion but not the
mathematical objective.

Stop if both `4M` and `5M` fail, or if a same-GPU oracle timing bound (zero latent iterations,
measured prediction plus required decode) cannot possibly reach `8x` versus the tuned FOM.

### Stage 4 — lock and confirmation, at most 6 cells

Use up to four development timing/robustness cells to select one method and lock a machine-readable
manifest. Then run one untouched confirmation cell and, only if its audit fails for a documented
infrastructure reason, one exact replacement with identical code/config/data seed.

The confirmation cell uses 32 trajectories, nine timed repetitions per method per trajectory,
balanced order, burn-in before each timed block, and a trajectory-clustered bootstrap with at
least 20,000 resamples. No result-dependent case removal is allowed. Any non-finite repetition is
retained and makes the robustness gate fail.

## Audit, provenance, and reporting

Every result JSON records the benchmark/config manifest, data seeds and draw indices, mesh and
time step, latent `k`, `M`, `m`, stopping tolerances, FOM and AmgX configuration, git commit,
staged-tree SHA-256, source-file hashes, JAX/x64/highest status, GPU name, node, SLURM job id,
complete timing arrays, returned residuals, iteration counts, failures, and setup/reuse costs.

Every batch script uses the `gpu` partition and the mandatory JAX backend preflight; a result is
inadmissible unless its log contains `jax_backend=gpu`. Each cell has its own directory under
`/cluster/tufts/paralab/tawal01/heat_amgx_nmrom/`. Code is copied directly there, results are
pulled with checksums, and the exact remote cell directory is deleted after verification.

The final report is generated from run JSONs by a script in `reports/`; no measured number is
typed into its prose or tables. It labels every result final, provisional, excluded, or retracted.
Completion means either all conjunctive gates pass, or this finite search is honestly exhausted
and the report gives the active accuracy and speed floors plus the best measured Pareto point.
