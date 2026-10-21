**A2 is not yet ready, even for this feasibility pilot.** It fixes several round-2 findings, but leaves contradictory decision rules and an invalid fallback for B’s uncertainty allowance. These require small design repairs, not publication-scale experiments.

I applied A2 over A1 over the original. No files were changed and no jobs were run.

1. **Round-2 item 1: hypotheses and interpretation — CORRECT.**

   A2.1 fixes all three requested changes: a paired viscosity sweep with gradient/decay diagnostics, a common 48-trajectory budget through A4, and a nonlinear-evaluation crossover distinct from whole-query speed.

   **Minor clarification:** measured costs can have multiple crossings. Specify which bracketing pair defines \(r_\times\), or report multiple crossings as unresolved. Retain “smallest observed feasible size” throughout the readouts.

2. **Round-2 item 2: proxy logic — CORRECT.**

   A1.1’s withdrawal remains effective. Neither a POD loss nor a POD win transfers automatically to the coordinate-network bank.

   **Required treatment:** retain “this tested POD–Galerkin implementation” in conclusions; keep representation, rollout accuracy, and cost separate.

3. **Round-2 item 3: step ladder — NEEDS-RESTATEMENT.**

   [A2.2](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/DESIGN.md:461) improves the stopping diagnostic, but does not define one consistent decision procedure.

   - The estimate \(\delta_n/15\) assumes fourth-order asymptotic convergence. One small trajectory difference does not establish that regime. B’s SBDF2 comparator is second order; its corresponding asymptotic factor is \(1/3\).
   - A2.2 classifies a **size** using its last run, whereas feasibility means **any observed step count** meets the target. An earlier qualifying run must remain feasible even if the final run is unresolved.
   - A2.2’s band around \(\tau\) and A2.5’s band incorporating \(U\) can assign different statuses. Neither section explicitly resolves their precedence.
   - No distance exists at the first rung, or when the preceding run is invalid.

   **Fix:** score every valid observed configuration using A2.5; a size is observed feasible if any configuration passes. Treat the distance rule solely as a search-stopping heuristic and mark unsearched refinements as such. Explicitly define missing-distance handling and label order-based remaining-error estimates conditional. Do not add temporal extrapolation uncertainty to an already observed configuration’s error merely because further steps were not explored.

4. **Round-2 item 3: ROM-Q ladder and pruning — NEEDS-RESTATEMENT.**

   A2.3 supplies the missing grid ladder and appropriately defers B ROM-Q. Independent scoring of each actual ROM-Q remains correct.

   However, \(6(M_q-1)^2r\le2r^3\) is a **workload heuristic**, not a runtime dominance test. More flops can execute faster because tensor and grid kernels have different memory traffic and implementation efficiency. The tensor estimate also differs from the earlier approximately \(r^3\) packed-evaluation count.

   **Fix:** explicitly label excluded configurations “not explored by heuristic pruning,” justify the operation-count convention, and describe winners as cheapest among evaluated configurations. No exhaustive ROM-Q campaign is needed for this pilot.

5. **Round-2 item 3: query accounting and timing — NEEDS-RESTATEMENT.**

   Separate A/B contracts, both IMEX factorizations, stored sentinel values, dispersion, outlier counts, and honest fresh-finalist reporting are repaired.

   Three ambiguities remain in [A2.3](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/DESIGN.md:474):

   - B’s initializer and decoder use \(\Phi\), immediately after requiring every operator to use the rotated \(\Psi_r\). Taken literally, these are inconsistent coordinates.
   - B’s abbreviated contract omits its ROM’s parameter-dependent ETDRK4 coefficient construction.
   - The initialization differences are written as absolute norms but compared with a relative target \(\tau/20\). Their normalization is unspecified.

   **Fix:** write B’s sampled projection and decoder using \(\Psi_r\), explicitly state the basis normalization that makes the \((512/L_{\rm init})^2\) weight correct, charge ETDRK4 coefficients, and normalize initialization discrepancies by the same initial-state norm as scoring. Apply the full \(e+U\le\tau\) acceptance test when re-scoring the selected initializer. Price that actual initializer before finalist selection.

6. **Round-2 item 4: SW1 — WRONG under the new matcher.**

   Perturbing the reference itself fixes the cancellation problem. But the asserted acceptance of \(\epsilon=\tau/2\) now requires
   \[
   U\le\tau/2.
   \]
   A2 does not guarantee this for every B target. Likewise, \(2\tau\) is a definite *miss* only when \(U<\tau\); otherwise it can be unresolved. “Not accepted” and “misses” must be distinguished.

   **Fix:** test the error calculation with \(U=0\), then independently test the three-way matcher with explicit \((e,U,\tau)\) fixtures covering pass, miss, unresolved, and both equality boundaries. Do not attach unconditional expected labels to arbitrary production allowances.

7. **Round-2 item 4: H1 and SW4 — CORRECT, with a fixture condition.**

   H1 is now properly diagnostic. SW4’s instantaneous algebraic check removes the unjustified rollout-response assertion.

   Retaining A1’s largest-component selection, for nonzero \(N(z)\),
   \[
   \max_k|N_k(z)|\ge \frac{\|N(z)\|_2}{\sqrt r},
   \]
   so the relative excitation requirement is comfortably satisfied at the registered ranks.

   **Fix:** explicitly require a finite, nonzero nonlinear vector and retain largest-component selection. If no reached state supplies one, report an unavailable fixture or use a declared synthetic fixture; do not call the physical absence of excitation a solver bug. Keep rollout sensitivity diagnostic.

8. **Round-2 item 5: B’s four-grid order calculation — CORRECT; failed-gate fallback — WRONG.**

   Four grids now provide three differences and two computable orders. The common scoring lattices fix the previous mathematical defect. Orders of maximum differences are acceptable as **conditional aggregate diagnostics**, although the maximizing case/time may change.

   The fallback is not executable safely: A2 retains
   \[
   U_B=\hat e_{\rm ref}+\delta_t
   \]
   after allowing the order gate to fail. If \(p_2=0\), the Richardson denominator vanishes; if \(p_2<0\), the proposed estimate is negative. Even a finite positive estimate loses its stated qualification when the gate fails.

   **Fix:** use Richardson-based \(U_B\) only when its gate passes and the differences are finite and nondegenerate. On failure, report raw errors against the \(L=1024\) numerical field separately, with physical-target qualification unresolved. Specify whether both scoring-lattice gates must pass or the second is diagnostic. A disclaimer alone does not repair an invalid allowance.

9. **Round-2 item 5: temporal and padding qualification — NEEDS-RESTATEMENT / WRONG.**

   - **B temporal coverage — NEEDS-RESTATEMENT.** Four actual-reference refinements plus twelve coarse-grid checks improve coverage. They do not qualify fine-grid temporal error for all twelve cases. “Conclusions apply to that check’s coverage” is too vague beside a twelve-case worst-error verdict.

     **Fix:** explicitly call the twelve-case allowance an extrapolation from partial fine-grid checks, or restrict qualified conclusions to the checked subset. Full twelve-case refinement is optional for this pilot if the limitation is carried into the verdict.

   - **A padding coverage — WRONG.** A2.4 says the spatial gate covers reference padding error because modes and padding change together. That comparison cannot isolate padding error; shared bias or cancellation can survive it. The specific round-2 repair was not implemented.

     **Fix:** compare two padding sizes at fixed reference resolution on representative hard cases, or explicitly leave padding uncertainty unqualified and restrict claims accordingly. G9 can remain diagnostic.

   - **Training gates — NEEDS-RESTATEMENT.** A now has a failure consequence. B supplies a difference but no threshold, so its statement “a failure labels … unqualified” has no defined trigger.

     **Fix:** either define a B threshold or call the B comparison diagnostic and its basis qualification unestablished.

10. **Round-2 item 5: three-way uncertainty scoring — CORRECT conditionally.**

    For a finite, nonnegative allowance, the pass/miss/unresolved inequalities are coherent. Adding checked contributions is preferable to taking their maximum.

    **Fix:** explicitly call \(U\) a conditional allowance, not an established error bound; define its aggregation over cases/times and its unavailable/failed-gate state. Reconcile it with item 3 and repair B’s fallback before using it for selection. Merely stating that errors are against numerical references does not establish physical accuracy.

11. **Round-2 item 6: metrics and floors — CORRECT.**

    Continuous coefficient-space scoring for A and sampled-basis least-squares floors for B remain valid. Per-prefix rotation preserves the selected span.

    **Required clarification:** use identical case/time maxima and normalization for floors and rollout errors, allow round-off tolerance in D2, and retain B’s restricted-output interpretation. No denser continuum qualification is required for this pilot.

12. **Round-2 item 7: competitors — CORRECT.**

    The narrowed comparison against these FD implementations remains adequate for feasibility.

    **Required treatment:** no inference that losing a global sine transform makes ROMs generally advantageous. Stronger competitors can remain deferred.

13. **Round-2 item 8: feasibility caps — NEEDS-RESTATEMENT.**

    Stage benchmarks, checkpointable units, a total eight-hour cap, explicit timeout consequences, and SBDF2 training resolve most of round 2. Finishing with too few levels is acceptable.

    However, recording projected memory does not specify what happens when it exceeds the limit. Only A4 has an explicit memory admission condition; B factorization, tensor assembly, and other stages need one too. The fixed three-times-snapshot workspace allowance is not itself evidence about actual peak allocations.

    **Fix:** require every stage/unit to fit the memory ceiling, including temporaries and factorization storage; otherwise split or skip it. Specify that the eight-hour budget includes benchmarks, retries, and already completed pilot work, and that timeouts apply to individual units as well as the total budget.

14. **Round-2 item 9: specific numerical claims — CORRECT repairs, with the qualifications above.**

    - Discrete tensor equality versus continuous Galerkin exactness: repaired.
    - Packed storage and off-diagonal weighting: unchanged and valid.
    - Measured nonlinear-cost crossover: repaired.
    - Both SBDF2-startup factorizations: repaired.
    - Separate diagonalization of every POD prefix: explicitly repaired.
    - Front-width/CFL rules withdrawn as eligibility criteria: repaired.
    - Negligible rather than exact wall compatibility: repaired.
    - R-function quotient singularity and limited D3 interpretation: repaired.

    **Required fix:** make B’s initializer/decoder notation obey the newly explicit rotation requirement; retain padding qualification as a separate outstanding issue.

The remaining blockers are decision-rule consistency, the failed-order allowance, deterministic matcher fixtures, and unresolved qualification/accounting details. None requires expanding the pilot into a publication study.

OVERALL: NOT-READY