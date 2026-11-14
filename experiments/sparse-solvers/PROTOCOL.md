# Sparse-solver solve-only timings: protocol

Fixed on Fri 13 Nov 2026, 11:52 EST, before any timing result of this lane exists.
Later changes are appended under "Amendments" with the reason.

## Question

How long does one linear solve of the discretised operator take on a GPU with a sparse direct solver
(cuDSS) and with algebraic multigrid (AmgX), as a function of the number of unknowns, and how does that
compare with the query time of the reduced model reported in the paper?  Only the solve is built: the
sparse matrix with the correct sparsity pattern and a right-hand side of all ones.  No expected outcome is
assumed; every size where a solver is faster than the reduced model is reported as such.

## Matrices (`matrices.py`)

float64 CSR, int32 indices, finite differences on a uniform grid with n interior points per axis,
h = 1/(n+1), homogeneous Dirichlet boundary, K = minus the discrete Laplacian.

| name | matrix | type | sizes |
|---|---|---|---|
| poisson | K (5-point in 2D, 7-point in 3D) | SPD | 2D n = 256, 512, 1024, 2048, 4096; 3D n = 64, 96, 128, 192, 256 |
| heat | I + (dt kappa / 2) K, dt = 0.025, kappa = 0.02 (the Crank-Nicolson step matrix of the paper) | SPD | same |
| burgers | I + dt (diag(g) + diag(u) D_upwind(u) + nu K), dt = 0.005, nu = 0.01: backward-Euler Jacobian of u_t + u sum_i u_{x_i} = nu Lap u with sign-upwind differences, linearised about u = Gaussian bump (width 0.1, centre (0.4, 0.6[, 0.5])) + 0.5 prod_i sin(2 pi x_i); g = upwind sum_i du/dx_i | nonsymmetric | 2D as above; 3D n = 64, 96, 128 |

nu = 0.01 is the lower end of the paper's range (0.01 to 0.1), the most convection-dominated case.
A size that does not fit in device memory is recorded as a result (stage, device, memory), not dropped.

## Solvers

1. **cuDSS** (`nvidia-cudss-cu12` wheel, called through its C interface).  Matrix type SPD for poisson and
   heat, GENERAL for burgers; full matrix view; default reordering; host threading layer
   `libcudss_mtlayer_gomp` with all CPUs of the job (16).  If SPD with the full view is rejected, the upper
   triangle view is used and recorded.  Recorded separately per repeat: analysis (reordering + symbolic),
   numerical factorisation, solve, refactorisation with the same pattern (the same values are re-supplied
   through `cudssMatrixSetValues`; the phase does the full numerical work), solve after refactorisation;
   peak device memory (5 ms polling of `cudaMemGetInfo`, plus the library's own estimate); relative
   residual ||b - A x|| / ||b|| computed outside the library.
2. **AmgX** built from source (commit recorded in every JSON), mode dDDI, one GPU.  Shipped configuration
   files, named in the results:
   - SPD: `PCG_CLASSICAL_V_JACOBI`, `PCG_AGGREGATION_JACOBI`, `FGMRES_CLASSICAL_AGGRESSIVE_PMIS`
   - nonsymmetric: `FGMRES_AGGREGATION_JACOBI`, `PBICGSTAB_CLASSICAL_JACOBI`,
     `FGMRES_CLASSICAL_AGGRESSIVE_PMIS`, `PBICGSTAB_AGGREGATION_W_JACOBI`
   Only these fields of a shipped file are changed: outer `tolerance` (1e-6 and 1e-10), outer `max_iters`
   (1000), printing and internal timers off, `monitor_residual` on.  Convergence criterion as shipped
   (relative to the initial residual, L2), zero initial guess.  Setup and solve are timed separately;
   iteration count, solver status and the residual computed outside the library are recorded.  All
   configurations are kept in the JSON.  The headline AmgX number per matrix and tolerance is the
   configuration with the smallest median setup + solve among those whose externally computed relative
   residual is at most the tolerance; if none reaches it, that is reported.
3. **Jacobi-preconditioned CG in CuPy** (free extra, SPD matrices only), tolerances 1e-6 and 1e-10,
   residual checked every 10 iterations, at most 400,000 iterations.

Fallback order if AmgX cannot be built or run within about two hours, labelled as what it is:
(i) a GPU AMG from wheels (PyAMG smoothed-aggregation hierarchy moved to CuPy, V-cycle preconditioned CG
or GMRES); (ii) CPU PyAMG (smoothed aggregation + CG) with the thread count stated.

## Timing rule

- One GPU type for all headline numbers: NVIDIA H200 (141 GB), chosen for memory headroom at the largest
  sizes; it is at least as fast as the A100 80 GB used for the paper's Burgers 2D timings, which favours
  the comparators.  If a slot allows, the list is repeated on an A100 80 GB and reported separately.
  The exact model string from `nvidia-smi` is stored in every JSON.
- float64 throughout.  The device is synchronised before every clock read (`time.perf_counter`).
- Each timed unit (cuDSS: one analysis/factor/solve/refactor/solve cycle on a new data object; AmgX: one
  setup + solve on a new solver object; CG: one solve) is run once as warm-up, stored separately as
  `first_call`, and then repeated 5 times; 3 times if the warm-up unit took more than 60 s; once if more
  than 600 s (the number of repeats is stored).  Median and min-max of the repeats are reported.
- Matrix assembly on the CPU and upload to the device are not part of any reported solver time (upload
  time is stored).
- One process per (matrix, solver); a process that dies (for example device out of memory) leaves the
  stages already finished in its JSON and is recorded as crashed with the error text.  Per-process limit
  90 minutes; a timeout is recorded as a result.

## Derived per-trajectory estimate (arithmetic, not a measured end-to-end solver)

Number of linear solves of the paper's full-order trajectories: heat 20 Crank-Nicolson steps
(dt = 0.025 to t = 0.5); Burgers 50 backward-Euler steps (dt = 0.005 to t = 0.25), counted with one
linear solve per step, which is a lower bound on a Newton solver.

- heat, cuDSS: analysis + factor + 20 solves, and also 20 solves alone (factor amortised).
- heat, AmgX / CG: setup + 20 solves (the solve time for a ones right-hand side from a zero guess).
- Burgers, cuDSS: analysis + 50 (refactor + solve), and also analysis + factor once + 50 solves
  (a frozen-Jacobian variant, stated as such).
- Burgers, AmgX: 50 (setup + solve), and setup once + 50 solves.

These are placed next to the reduced model's full query time from the paper and labelled as estimates.

## Reduced-model numbers

Read from the revised paper source (table and line recorded in RESULTS.md); not re-measured here.

## Outputs

`results/<gpu tag>/<problem><dim>d_n<N>_<solver>.json`; `RESULTS.md` is generated from the JSON by
`summarize.py`.  Growth exponents are least-squares slopes of log(time) against log(unknowns).

## Amendments

1. Fri 13 Nov 2026, 11:53 EST (after the smoke job 7174901, before any production run).  The smoke run
   showed that `AMGX_solver_solve_with_0_initial_guess` does not zero the solution vector, so repeated
   AmgX solves started from the previous solution while the library assumed zero (external residual
   1, 2, ... 5 on repeats).  The solution vector is now set to zero before every solve, outside the timed
   region.  The smoke AmgX numbers are invalid and are not used; smoke files are kept in `results/smoke1`.
   No change to sizes, solvers, configurations, tolerances or the timing rule.

### Amendment 2 (Fri 13 Nov 2026, 12:41 EST): cuDSS hybrid memory mode for the size that did not fit

Reason: the in-device cuDSS factorisation of the 256^3 Poisson and heat matrices failed with ALLOC_FAILED on the H200
(library estimate 380.6 GiB of device memory against 140 GiB).  cuDSS has a documented hybrid memory mode
(`CUDSS_CONFIG_HYBRID_MEMORY_MODE = 1`) that keeps part of the factors in host memory; the library's own estimate of the
minimum device memory in that mode is 91.8 GiB, which fits.  To give the direct solver a fair chance at that size it is run
once more as solver `cudss_hybrid` (same phases, same timing rule, device memory limit left at the library default, host
memory request raised to 600 GiB), on 256^3 Poisson and heat and on 192^3 Poisson as a cross-check against the in-device run.
The in-device failure stays in the record.  Written before the hybrid run; the outcome is not known.

### Amendment 3 (Sat 14 Nov 2026): Task A, a measured implicit Burgers stepper on cuDSS; Task B, re-timing the reduced model on the same GPU type

Written before any Task A or Task B job is submitted; no outcome of either is known.  Reason: the per-trajectory figures
above are arithmetic estimates from single solves, and the reduced-model times were copied from the paper, part of them
measured on a different GPU type.  Both are replaced by measurements.

**Task A.  Stepper (`stepper.py`).**  Backward Euler for the paper's Burgers full-order discretisation (2D:
`mr-burgers2d/engines.py` residual, L intervals, (L-1)^2 unknowns; 3D: `paper-b3d` sign-upwind residual, n = L+1 nodes per
axis), float64, CuPy arrays, cuDSS through ctypes as in `bench.py` (general LU, default reordering and pivoting, 16 host
threads).  The CSR pattern is fixed; residual and Jacobian values are computed on the device by two CUDA kernels.  The
Jacobian kernel is checked against a finite-difference directional derivative of the residual kernel, and the tight
stepper against the paper's reference solution, before any timing is used.
Newton rule, as the paper's full solver: start a step from the previous state, stop when ||r|| <= ntol ||u_prev||, test
before each iteration.  Iteration cap 20 for full Newton (the paper's cap) and 100 for the variants that reuse a
factorisation; a step that hits the cap or goes non-finite is recorded as not converged and the trajectory continues.
Variants, each reported separately:
- `full`: Jacobian assembled and numerically refactorised at every Newton iteration (symbolic analysis reused);
- `k1`: modified Newton, refactorised once per time step at the previous state, then reused within the step;
- `k5`, `k10`: refactorised once every 5 / 10 time steps;
- `frozen`: factorised once per trajectory, at the initial state;
- `lin1`: linearly implicit, exactly one Newton iteration per step with a fresh factorisation and no residual test.
Sizes: 2D L = 256, 512, 1024, 2048, 4096; 3D L = 64, 96, 128.  An out-of-memory or library failure is a result.

*Cases and reference.*  2D: the first K cases of the paper's test cohort `params_draw(20260916, 64)` (checksum verified),
K = 64 at 256 and 512, 16 at 1024, 8 at 2048, 6 at 4096.  3D: the first K cases of the paper's held-out cohort
`table(923901, 32)`, K = 16 at 64, 8 at 96, 6 at 128.  The reduced model's worst-case error in the paper is over all 64
(2D) or 32 (3D) cases; where K is smaller the cuDSS worst case is over fewer cases, which can only favour cuDSS, and is
stated.  Reference: the paper's own reference solver, unchanged vendored code, run in the same job on the same mesh
(2D `iterative_paths.make_fom(L, 0.005, 'fft')`, ntol 1e-6, ltol 1e-8; 3D `common.make_fom(n, 0.005, 1e-10, 1e-11)`).
Error of a trajectory, as in the paper: max over the five evolved output times of ||u - u_ref|| / ||u_ref(t=0)|| on the
full target mesh.  At 3D L = 96 the paper has no reduced model; cuDSS is still measured there.

*Timing.*  One trajectory = from the initial field on the target mesh, on the device, to the six output fields
(t = 0, 0.05, ..., 0.25) on the target mesh, on the device, device synchronised at both clock reads.  It includes
Jacobian assembly, every numerical refactorisation (also the one of `frozen`), solves, residuals, and for coarse meshes
the restriction of the input and the interpolation of the output.  Excluded and reported separately as one-off setup:
pattern construction, cuDSS symbolic analysis, the first numerical factorisation (needed before a refactorisation),
kernel compilation.  Warm-up: one two-step trajectory per setting, untimed.  Repeats: 3 timed trajectories per case if
the first takes at most 5 s, otherwise 1; median and min-max are over all (case, repeat) timings.  Newton iterations,
refactorisations, solves and unconverged steps are counted per trajectory; an extra instrumented trajectory (case 0)
gives the split refactorise / solve / rest for the same-mesh settings.

*Same-mesh comparison.*  dt = 0.005 (50 steps), the six variants, ntol 1e-6 and 3e-3 (the paper's lean setting), next to
the reduced model's accurate and fast settings.

*Accuracy-matched comparison.*  cuDSS may choose variant in {k1, frozen, lin1}, ntol in {1e-3, 1e-2}, dt in
{0.005, 0.01, 0.025, 0.05}, and a mesh coarsened by 1, 2 or 4 per axis (input subsampled, output interpolated to the
target mesh by aligned multilinear interpolation, the paper's coarse-mesh rule), plus all same-mesh settings.  For each
reduced-model setting (accurate, fast) the matched cuDSS setting is the one with the smallest median trajectory time among
settings whose worst-case error over the cuDSS cases is no larger than the reduced model's worst-case error; settings with
any unconverged step are eligible only if flagged.  The selection uses the same cases as the report (there is no separate
cohort for cuDSS, which favours cuDSS).  Settings run coarsest and cheapest first under a per-size wall budget; a setting
not run for budget is listed as not run, never as failed.

*AmgX stepper* (setup per step / reused): only if time allows; if not run it is reported as not run.

**Task B.  Reduced model on the same GPU type.**  The paper's own drivers, configurations and frozen checkpoints are
re-staged unchanged (checksums against the original run manifests) into this lane's cluster namespace and re-run on an
H200 node of the `gpu` partition, the node type of the cuDSS runs (and on an A100 80 GB if a slot allows): Burgers 2D
(256 to 4096), Burgers 3D (64, 128), Poisson 2D, Poisson 3D, heat 2D, heat 3D, in that priority.  Only job name, paths,
GPU constraint, memory and wall-time requests change; nothing in the model, its settings or the timing code.  A re-run
timing is used only if the re-run reproduces the paper's worst-case and median errors for that row (same cases; agreement
to the printed digits, small floating-point differences between GPU types reported); otherwise the row is reported as
not reproduced.  Rows not re-run for lack of time or slots keep the paper value and are marked as not re-measured.
`RESULTS-ours-same-gpu.md` is generated from the result JSON by script.  At most 3 GPU jobs of this lane queued or
running at any time.

### Amendment 4 (Sat 14 Nov 2026, 16:35 EST): settings of the AmgX stepper (Task A item 4), fixed before it has run

Reason: amendment 3 left the AmgX stepper open ("only if time allows").  The single-solve measurements show AmgX far
cheaper than a cuDSS refactorisation in 3D, so an honest best shot for the sparse-solver side needs it.  No AmgX stepper
number exists at the time of writing; the cuDSS stepper results seen so far (256^2, part of 512^2) did not inform
anything below except the decision to keep the list short.

`amgx_step.py` reuses the cuDSS stepper's kernels, Newton loop (the same function), timing contract, warm-up, repeat rule,
checks against the paper's reference and error metric.  Only the linear solve differs: AmgX (mode dDDI, device pointers),
zero initial guess, relative linear tolerance `ltol` (RELATIVE_INI, L2), max 200 linear iterations; the Newton residual
test is ours, as for cuDSS, so a linear solve that stops early only costs Newton iterations.  "Setup" = replace matrix
coefficients + AmgX solver setup.  Variants: full (setup every Newton iteration), k1 (once per time step), frozen (once
per trajectory), lin1 (one setup and solve per step, no residual test).
Configurations: 2D PBICGSTAB_CLASSICAL_JACOBI, 3D FGMRES_CLASSICAL_AGGRESSIVE_PMIS (the shipped configurations that were
fastest on the Burgers Jacobians in the single-solve tables), and in both a non-AMG BiCGStab with one block-Jacobi sweep
written by this lane (the backward-Euler matrix is close to diagonally dominant at small sizes, where AMG setup is waste).
Same-mesh settings (dt 0.005): lin1 (ltol 1e-3); {k1, frozen, full} x (ntol 1e-6 with ltol 1e-2; ntol 3e-3 with ltol 1e-1).
Matched grid, same mesh only: dt in {0.01, 0.005} x {frozen, k1} x ntol in {1e-2, 1e-3} with ltol 1e-1; lin1 at dt 0.01
with ltol 1e-2.  Cases: the first min(16, cuDSS cases) cases of the same cohort.  A per-size wall budget applies; settings
not reached are "not run (budget)"; a setting whose AmgX call errors is "failed" with the message.  Selection rule for
the accuracy-matched table: as amendment 3, with AmgX and cuDSS settings reported in separate columns.  Mesh coarsening
and dt > 0.01 are not run for AmgX (the cuDSS grid shows whether they can meet the error at all).
