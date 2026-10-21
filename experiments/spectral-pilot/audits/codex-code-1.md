The numerical kernels largely check out, but qualification, timing selection, and several controls do not satisfy A1–A4. I reproduced two concrete failures. This audit was read-only; no files were written and no long jobs were launched.

I also inspected the untracked `diag_a.py` and `time_b.py` found during the review, so diagnostics and Part B timing are **present**, rather than missing.

1. **[common.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/common.py:88) — BLOCKER — WRONG**

   The shared 1024-point IC rule fails G7 at high resolutions. Comparing against `pscore.ic_coefs_2d` on test row 0 reproduced:

   | N | Relative coefficient error |
   |---:|---:|
   | 192 | `1.79e-14` |
   | 384 | `3.56e-14` |
   | 512 | `8.89e-11` |
   | 768 | `8.24e-6` |

   The required threshold is `1e-12`. This affects the A4 reference and high-resolution initialisation; sharing the inaccurate rule between methods does not satisfy G7.

   **CORRECT:** ETDRK4 stage arithmetic, output indexing, coefficient-space relative continuous \(L^2\) error, and lattice synthesis. The Parseval factor cancels between numerator and denominator. Paired Part A descriptors and training/test disjointness are implemented correctly.

   **MINOR:** f64 is enabled, but `highest` matmul precision is only recorded, not enforced.

2. **[romlib.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/romlib.py:29) — MAJOR — NEEDS-FIX**

   The Gram-eigenvalue rank cutoff, `w[0] * 1e-24`, is below meaningful double-precision Gram-matrix accuracy. A tiny rank-two input returned **six basis vectors**, with spurious relative singular values around `1e-8`. QR restores orthogonality but cannot recover accurate small POD modes from Gram roundoff. Use a numerically defensible rank criterion or a stable alternative.

   **CORRECT:** QR preserves the computed prefix spans; each prefix is separately diagonalised; tensor rotation contracts the correct axes; packing includes the off-diagonal factor two; ROM-Q uses the correct `-(2/Mq)^2` normalisation and derivative synthesis. Small independent checks gave relative errors `8.53e-16` for rotated tensor versus exact evaluation and `7.09e-15` for ROM-Q versus exact evaluation.

   **MAJOR:** large operator arrays are captured in JIT closures—particularly `U/Gd` in `tensor_full`, packed tensors, and exact-evaluation matrices. They are not explicit dynamic arguments. This needs compiler/memory validation at production sizes; the small checks do not establish that large-constant compilation is safe.

3. **[gen_a.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/gen_a.py:60) — MAJOR — NEEDS-FIX**

   `accepted` covers only space and time, excluding the subsequently required padding gate. `tau_min_eligible` retains the superseded `20 × max(...)` rule. These are misleading qualification fields under A3/A4.

   SW2 performs the prescribed perturbation, but failure only sets a Boolean; it does not prevent downstream qualification. The training qualification result is recorded but not propagated into ROM labels.

   **CORRECT:** the four training trajectories are compared at the right output times: `X[:4, ::4]`.

   **MAJOR:** generation accumulates all chunks in memory and concatenates them, rather than streaming snapshots to disk. There are no required pre-stage time/memory projections.

4. **[pad_gate.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/pad_gate.py:22) — MINOR — NEEDS-FIX**

   **CORRECT:** selects the two largest-\(s\) test cases, preserves the reference step count, compares `2Nref` padding against the stored default reference, and applies `2e-5`.

   Missing stage wall time and preflight projection. More seriously, its result is not enforced by the downstream consumers discussed next.

5. **[fom_a.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/fom_a.py:41) — BLOCKER — WRONG**

   A missing `pad.json` silently contributes **zero** to the allowance. A4 explicitly forbids this. A finite but failed space/time/padding gate also does not make the level unresolved: the code can still classify configurations as meeting targets, and timing subsequently trusts this allowance.

   **MAJOR:** the “two consecutive sizes meet every target” stopping rule is not registered in A1–A4. It can exclude a faster larger-grid configuration using fewer steps. Calling it registered in the docstring is incorrect.

   **MAJOR:** no spectral-floor D2 assertion. SW1a/SW1b execute, but their failure does not stop qualification.

   **CORRECT:** coefficient padding, per-case errors, invalid-run handling, and the trajectory-distance step stopping calculation.

6. **[rom_a.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/rom_a.py:125) — MAJOR — NEEDS-FIX**

   **CORRECT:** prefix-specific rotation, initial projection, reference projection, tensor construction, ROM-Q basis rotation, and continuous-norm floors. I found no operative late-binding bug: these closures are consumed synchronously before the loop variables change.

   Control deficiencies:

   - G4 records failures without preventing qualification.
   - SW4 can falsely pass a zero nonlinear state: `0 >= 1e-6 * 0` is true. A3 requires finite, nonzero \(N(z)\), otherwise “unavailable.”
   - D2 is asserted only for ROM-T, not ROM-Q or sine-ROM.
   - G9 is correctly diagnostic, but `r = min(64, rmax)` can index a nonexistent `rom_t[r]` when a custom/effective rank below 64 is not on the ladder.
   - The 48-trajectory and random-span calculations produce **floors only**, not the corresponding evolution results needed for D1 and the random-span ROM diagnostic.

   The reference/training qualification state is not carried into the ROM results.

7. **[score.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/score.py:11) — MAJOR — NEEDS-FIX**

   **CORRECT:** finite three-way comparisons, including both equality fixtures; all SW1b fixtures passed. SW1a injects perturbations with the correct norm and denominator.

   Missing-allowance handling is not implemented: `U=None` raises an exception. Moreover, invalid errors are classified as misses before allowance availability is considered. A4 requires unavailable gates to make the level/part unresolved. This must be represented explicitly rather than left to callers that currently substitute incomplete allowances.

8. **[time_a.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/time_a.py:151) — BLOCKER — NEEDS-FIX**

   The reported winner is not established under the registered timing protocol:

   - Screening excludes candidates whose **nonlinear-step estimate** exceeds three times the minimum. That estimate omits IC, ETDRK4 setup and output synthesis; it cannot safely exclude whole-query winners. This extra pruning is not authorised by A1–A4.
   - Screening interleaves **candidate batches**, not individual F/R calls: each candidate receives all repetitions across all timing cases before the next candidate.
   - Screening has no sentinel. Fresh finalist timing cannot repair a biased candidate selection.
   - Final timing records sentinel failure but neither implements the required retry nor makes its emitted speed ratios unusable.
   - `check_env()` checks affinity and XLA settings, but not the required BLAS/OpenMP/MKL thread settings.
   - Nonlinear timing repetitions are discarded after taking medians.
   - The secondary native-output contract is absent.

   **CORRECT:** retained ROM candidates have their initialiser selected and are re-scored with \(e+U\) **before finalist selection**. Their selected initialiser is what gets timed. Both whole-query paths include IC, viscosity-dependent ETDRK4 coefficients, stepping and output synthesis. Finalist calls are properly interleaved and synchronised.

   The harness nevertheless inherits the invalid reference-qualification path from `fom_a.py` and never checks failed G4/G7/SW controls.

9. **[fd_l.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/fd_l.py:147) — BLOCKER — WRONG**

   `make_rk4(32, ..., out_lattice=False)` unconditionally calls `lattice_idx(32)` with the default `lat=64`, triggering `assert L % lat == 0`. This reproduces immediately and prevents the registered RK4 ladder from getting past its first size.

   **CORRECT:** L-shape boundary mask, five-point Laplacian, central advection, SBDF2 with IMEX-Euler start, and RK4 stage formulas. Independent small-grid checks matched sparse and stencil operators to roundoff. SBDF2 outputs matched an independent loop to `4.44e-16`; no output-time off-by-one was found.

   **MINOR:** Part B seeds are `3_000_001/4_000_001`; the design specifies `3_000_000/4_000_000`. This is an undocumented cohort change, not demonstrated train/test contamination.

10. **[gen_b.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/gen_b.py:75) — MAJOR — NEEDS-FIX**

    **CORRECT:** four-grid comparisons, lattice mappings, observed-order formulas, largest-\(s\) fine-grid time-check subset, and numerical-reference fallback allowance.

    Missing robustness and qualification:

    - Ratios/logarithms are evaluated before checking finite, positive differences; zero differences can raise division errors.
    - Missing/nonfinite time gates are not explicitly converted into unavailable qualification.
    - The allowance is labelled `"physical (Richardson allowance)"` without preserving the required conditional/extrapolated-coverage qualification in downstream outputs.
    - B basis qualification is not explicitly labelled “unestablished.”
    - SW2 is absent for B.

    `multiprocessing.Pool` also forks after JAX backend initialisation. The existing `runs/gen_B.log` contains the JAX/fork deadlock warning. Use a safe process-start arrangement.

11. **[fom_b.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/fom_b.py:70) — BLOCKER — WRONG**

    The RK4 family crashes through the `L=32` defect above.

    **MAJOR:** F-IMEX starts at `n=10`, omitting the registered `n=5` candidate. Both families also use the unregistered two-successful-sizes stopping rule. RK4 additionally prunes low step counts using the diffusion estimate instead of evaluating the amended ladder; this needs explicit amended search scope.

    B’s SW1a/SW1b checks are not run. The normalisation and case/reference lattice alignment otherwise look correct.

12. **[rom_b.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/rom_b.py:136) — MAJOR — NEEDS-FIX**

    **CORRECT:** Euclidean POD convention, reduced FD Laplacian, central-advection tensor, full-grid initial projection, lattice decoder and sampled-basis least-squares floors. D2 uses the documented tolerance.

    G4 compares nonlinear RHS evaluations at reached states, **not two independently evolved trajectories** as specified. It therefore does not establish the registered rollout agreement.

    SW4 has the same zero-state false-positive as Part A, and its rollout-change diagnostic is absent. G4/SW4 failures do not block qualification. D1 again computes only floors; D3 envelope representability and the random-span evolution diagnostic are absent. Tensor assembly also captures its large matrices in JIT closures.

13. **Additional files: `diag_a.py` and `time_b.py` — MAJOR — NEEDS-FIX**

    `diag_a.py` correctly implements G7 comparison, gradient-magnitude front sharpness and remaining energy. These are **not missing**. However, G7 failure is only recorded and never enforced by generation or timing.

    `time_b.py` includes both IMEX factorisations, nodal IC, ROM sub-lattice projection with the correct `(512/Linit)^2` factor, ETDRK4 setup and decoding. Initialiser re-scoring occurs before finalist selection.

    Its screening still runs the FOM families and ROM sequentially, uses unregistered one-repetition pruning, and has no screening sentinel. Its warm-up helper does not synchronise JAX outputs. Final timing lacks the required retry, spread/outlier reporting and B H1 diagnostic. Final output also drops the screening record’s reference-qualification label. The initialiser discrepancy uses the full-grid IC norm rather than explicitly establishing equivalence to the prescribed scoring normalisation.

Across the implementation, the required pre-stage benchmarks, written peak-memory projections, enforced unit/total budgets, native-output timings, B corner-window diagnostics and matched finer-lattice diagnostics are not implemented. Storing raw error rows also does not by itself supply all required per-configuration three-way statuses and “not searched” accounting.

`vendor/pscore.py` was used only to verify the wrapper contract; it was not audited for changes.

OVERALL: NEEDS-FIX