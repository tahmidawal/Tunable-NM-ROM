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
