No files were modified. Note: `pro_common.py` and `pro_colloc.py` changed externally during the review; I re-read them, and findings below refer to the current 17:36:54 versions.

## MUST

- **`spec_a1_Mall` is not exactly the reported full-field data misfit when `HARD_BC=0`.** It acts only on `coords_int` ([pro_common.py:219](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_common.py:219)), while the oracle and reported error include all grid nodes, including decoder boundary values ([pro_objective.py:97](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_objective.py:97)). The smoke output has nonzero boundary blocks, so this is not merely formal.  
  **Fix:** call it `interior_data_misfit`, or enforce hard BC/append the boundary misfit to make it equal the full data objective.

- **Biased importance weights are wrong for sampling without replacement.** [pro_colloc.py:150](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_colloc.py:150) samples PPS without replacement but [pro_colloc.py:151](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_colloc.py:151) uses the with-replacement weight \(1/(mq_i)\). At \(m=n_i^2\), all nodes are selected but the weights are still nonuniform, so it demonstrably does not converge to the full objective.  
  **Fix:** sample with replacement, or use actual inclusion probabilities and Horvitz–Thompson weights.

- **Off-grid soft-BC solves do not pose the Poisson Dirichlet problem.** The objective contains only \(-\Delta u-f\) ([pro_common.py:315](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_common.py:315)); boundary error is only recorded afterward. Harmonic components are invisible to this objective, unlike the ghost-zero FD operator, so off-grid numbers can optimize a different, underdetermined problem.  
  **Fix:** require `HARD_BC=1` for off-grid runs or append a properly scaled boundary residual.

- **The `cgK` Gauss–Newton Jacobian is generally inconsistent with its forward objective.** [pro_common.py:239](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_common.py:239) deliberately truncates JAX CG, then [pro_common.py:243](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_common.py:243) differentiates it. JAX CG uses implicit differentiation, which is valid for a converged linear solve, not the nonlinear finite-\(K\) Krylov map. Thus `J` need not be the derivative of the value used for acceptance.  
  **Fix:** implement and differentiate an explicitly unrolled \(K\)-step solver, or use a fixed linear polynomial preconditioner.

- **Hard-BC checkpoint/runtime mismatch is silent.** Training records `hard_bc` ([pro_train.py:43](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_train.py:43)), but checkpoint validation ignores it ([pro_common.py:71](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_common.py:71)); consumers independently default `HARD_BC=0`. Forgetting the flag changes every prediction and objective.  
  **Fix:** derive the setting from `cfg.get("hard_bc", 0)` and reject any conflicting environment value.

## SHOULD

- **Ritz and `spec_a0.5_Mall` have identical GN steps, but not necessarily identical finite-run stopping.** Ritz accepts raw energy containing the negative solution-dependent constant ([pro_common.py:250](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_common.py:250)); relative stopping divides by `abs(E)` ([pro_common.py:179](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_common.py:179)). This can stop earlier than the energy-norm residual. The smoke rows agree because they exhaust the eight-attempt budget.  
  **Fix:** stop Ritz on step size only, or evaluate the shifted energy gap/energy-norm residual for stopping.

- **“M modes” is not consistently M modes.** The discrete mask keeps every mode tied at the cutoff and can therefore exceed `M` ([pro_common.py:130](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_common.py:130)); the off-grid table instead slices exactly `M`, potentially splitting a degenerate \(i,j\)/(j,i) pair and breaking symmetry ([pro_common.py:287](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_common.py:287)).  
  **Fix:** use the same complete-eigenshell policy in both paths and report the actual retained count.

- **Off-grid mode enumeration is hard-coded to the N=64 experiment.** [pro_common.py:284](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_common.py:284) always uses modes 1–63; consequently off-grid `Mall` is neither grid-size-independent nor literally all continuum modes.  
  **Fix:** derive the cutoff from `grid.N` and label it explicitly as a truncated continuum basis.

- **NNLS diagnostics do not describe the weights actually used.** `rnorm` is computed before padding ([pro_colloc.py:125](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_colloc.py:125)); padding assigns arbitrary positive median weights and may rescale them ([pro_colloc.py:135](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_colloc.py:135)). Also, the 200-step inner cap and outer safety cap can return without a KKT/termination check ([pro_common.py:346](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_common.py:346)).  
  **Fix:** re-solve on the final support, verify KKT convergence, and recompute/report the residual of the final `wq`.

- **Collocation comparisons are not paired.** One shared RNG drives EQ construction and fresh uniform/biased/off-grid nodes inside the objective and initialization loops ([pro_colloc.py:89](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_colloc.py:89), [pro_colloc.py:158](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_colloc.py:158)). Changing objective/init order changes the sampled sets, confounding method differences with sampling noise.  
  **Fix:** precompute keyed subsets per `(scheme,m,test_case)` and reuse them for every objective and initialization.

- **Off-grid \(m\to\infty\) does not converge to the full FD objective.** It converges to a continuum strong-form integral with continuous eigenvalues, whereas the full arm uses the discrete ghost-zero operator. That is a legitimate alternate method, but not a collocation approximation of the same objective.  
  **Fix:** compare off-grid runs against a dense off-grid reference and label them separately from FD collocation convergence.

- **The smoke output cannot validate reproduction of 6.3e-2.** It uses only two cases and eight attempts ([obj_smoke.json:29](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/smoke/obj_smoke.json:29)); its nearest-FD mean is 0.256 ([obj_smoke.json:123](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/smoke/obj_smoke.json:123)).  
  **Fix:** add a 16-case, 60-attempt golden control comparing against the phase-C per-sample results.

## NIT

- **Nearest initialization assumes exact generative source parameters are available.** It uses `z_true_all`, not an explicit distance between supplied source arrays ([pro_objective.py:65](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_objective.py:65)). This is not held-out-field leakage, but it is optimistic if deployment supplies only opaque sampled \(f\).  
  **Fix:** compute nearest neighbors using the source lattice representation, or explicitly document parameter metadata as online input.

- **`NS` can be mislabeled.** `stages_all[:NS]` silently returns fewer stages if `NS` is too large, while the report still records the requested value ([pro_objective.py:54](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_objective.py:54)).  
  **Fix:** assert `1 <= NS <= len(stages_all)` and similarly bound `N_TEST`.

- **Raw `obj_med` values are not comparable across objectives.** Different \(\alpha\), Ritz energy, grid sums, and continuum integrals have different units/constants; this does not change each minimizer but can mislead tables.  
  **Fix:** label units and avoid cross-objective comparisons of raw objective magnitudes.

## Checks that passed

- The discrete sine basis, normalization, eigenvalues, `ij` ordering, and \(dx=1/(N-1)\) are correct for `ms_parametric.neg_lap_interior` ([pro_common.py:97](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-16-poisson2d-rom-objective/experiments/poisson2d-rom-objective/pro_common.py:97), [ms_parametric.py:78](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-14-multistage-precision/experiments/multistage-precision/ms_parametric.py:78)).

- Full-grid `fd` is operator-equivalent to phase C, and residual-objective `lm_generic` matches the imported LM schedule, acceptance rule, and attempt accounting.

- Ritz has exactly \(g=J^Tr\) and \(H=J^TAJ\), matching full `spec_a0.5_Mall` apart from the stopping caveat. The smoke results confirm identical trajectories.

- `jax.hessian` is taken with respect to the two coordinate components, so its trace is the decoder Laplacian; the sign and source scaling are correct.

- Uniform weights, discrete `S[px]`/`S[py]` indexing, continuous \(2\sin(i\pi x)\sin(j\pi y)\), `wq` broadcasting, stencil `keep`, and `ij` reshaping are correct.

- No held-out field enters the ROM path. Hard-BC multiplication is consistent in training, decoding, oracle evaluation, and objectives when the runtime flag matches the checkpoint.

In short: the full-grid FD/spectral/Ritz core is sound and field leakage was avoided. Before trusting publication numbers, fix the biased sampling, off-grid boundary formulation, truncated-CG Jacobian, and hard-BC provenance; treat current NNLS and unpaired collocation comparisons as provisional.