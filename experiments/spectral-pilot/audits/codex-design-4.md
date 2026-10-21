**A3 is fit for the stated direction-and-feasibility pilot.** No BLOCKING issue remains. Not every round-3 clarification is fully incorporated: B’s fallback wording, initializer selection order, and numerical tolerance for floor checks still need restatement. These do not invalidate the narrowly scoped pilot reading.

I applied A3 over A2 over A1 over the original, read all three prior audits and the canonical log’s current context, and made no changes or job submissions. This is design approval, not verification of implementation or results.

The numbering below follows round 3.

1. **Hypotheses and interpretation — CORRECT.**

   A3.1 defines the first upward nonlinear-cost crossing and makes multiple crossings unresolved. A2’s paired-viscosity interpretation and common training-budget comparisons remain effective.

   **NON-BLOCKING edge case:** an exact measured equality is not explicitly handled by “exceeds.”

   **Concrete fix:** count an exact equality as a measured crossing; retain “unresolved” when no crossing is bracketed. These are nonlinear-evaluation comparisons, not whole-query crossover claims.

2. **Proxy logic — CORRECT.**

   A1.1’s withdrawal remains effective. This POD–Galerkin implementation is neither an accuracy bound nor a cost bound for the coordinate-network bank.

   **Concrete treatment:** retain “this tested implementation” beside positive and negative conclusions. No further design repair is required.

3. **Step ladder and configuration scoring — CORRECT.**

   A3.1 fixes all four findings: each observed configuration receives the same uncertainty-aware score; any passing configuration establishes observed size feasibility; trajectory distance only stops searching; missing distances cause continuation. The unsupported fourth-order extrapolation is removed from verdicts.

   **NON-BLOCKING limitation:** stopping can miss later feasible or cheaper configurations.

   **Concrete treatment:** report “smallest observed feasible size” and “cheapest among evaluated configurations,” with unsearched refinements identified. A3 already provides this interpretation.

4. **ROM-Q ladder and pruning — CORRECT.**

   A3 explicitly demotes the flop filter to a workload heuristic. Its packed count is consistent: \(r^2(r+1)/2\) multiply–adds gives approximately \(r^3\) flops when a multiply–add counts as two.

   **NON-BLOCKING limitation:** the retained \(2r^3\) cutoff is a permissive search threshold, not the corrected tensor flop estimate or a runtime bound.

   **Concrete fix:** describe the factor two as a heuristic margin. Do not conclude that excluded ROM-Q configurations are slower.

5. **Query accounting and timing — NEEDS-RESTATEMENT, NON-BLOCKING.**

   [A3.2](experiments/spectral-pilot/DESIGN.md# a32-part-b-accounting-corrects-a23) repairs B’s rotated coordinates, states Euclidean basis normalization, includes ETDRK4 coefficient construction, and requires initializer rescoring with the full allowance. The subsampling weight is appropriate for approximating the full-node sum under that normalization.

   The remaining sequencing omission is small but real: pricing the initializer in **finalist timing** does not explicitly require pricing it **before finalist selection**, as round 3 requested.

   **Concrete fix:** select each candidate’s initializer, rescore that actual candidate, time it, then select finalists. If selection used a different initializer, report the outcome as a screened candidate’s fresh time rather than the cheapest evaluated query.

6. **SW1 matcher checks — CORRECT.**

   SW1a now tests error calculation independently of reference uncertainty. SW1b covers all three labels and both boundary conventions. The fixtures match the stated inequalities.

   **Concrete treatment:** test matcher boundaries without introducing a broad numerical tolerance that changes the acceptance rule. No design repair is required.

7. **H1 and SW4 — CORRECT.**

   The finite, nonzero nonlinear-vector requirement and largest-component selection resolve the fixture issue. An unavailable physical fixture no longer becomes a purported solver bug. Rollout sensitivity remains diagnostic.

   **Concrete treatment:** record unavailable fixtures explicitly. No further repair is required.

8. **B order gate and fallback — NEEDS-RESTATEMENT, NON-BLOCKING.**

   A3 fixes the substantive defect: Richardson-based \(U_B\) is used only after the order gate passes with finite, decreasing differences. It also explicitly makes the second lattice diagnostic. Invalid or negative Richardson allowances are no longer permitted.

   However, “scored **two-way** … \((U=\delta_t)\)” does not define what happens inside the uncertainty band. A3.1 otherwise requires three-way scoring.

   **Concrete fix:** write:

   > On order-gate failure, report raw error against the numerical reference and apply the three-way rule with \(U=\delta_t\) for that numerical-reference comparison only. Physical-target qualification is unavailable.

   This is non-blocking because A3 already withdraws physical-accuracy qualification from **every** fallback verdict. A speed comparison under that explicitly numerical contract can still inform implementation feasibility.

9. **Temporal coverage, padding, and training qualification — CORRECT.**

   All three requested repairs are present:

   - B’s twelve-case temporal allowance is explicitly an extrapolation from partial fine-grid checks.
   - A now compares padding sizes at fixed reference resolution on two selected cases, adds the discrepancy to its allowance, and defines a failure consequence.
   - B’s training comparison is diagnostic and its basis qualification remains unestablished.

   **NON-BLOCKING limitations:** largest nominal steepness need not identify the worst padding case; partial checks do not bound unchecked errors.

   **Concrete treatment:** carry those coverage limitations beside the resulting verdicts. Do not describe the checks as exhaustive qualification.

10. **Conditional uncertainty allowance — CORRECT, subject to item 8.**

    A3 defines aggregation and explicitly calls \(U\) a conditional allowance rather than an error bound. A3.1 resolves the earlier conflict with step-search uncertainty.

    **Concrete treatment:** missing or nonfinite checks must produce an unavailable allowance, never an implicit zero. Use the item-8 restatement for B’s fallback.

11. **Metrics and projection floors — NEEDS-RESTATEMENT, NON-BLOCKING.**

    The mathematics remains correct: A’s coefficient norm measures the represented continuous fields, and B’s sampled-basis least-squares floor applies to its restricted scoring norm. Rotation preserves each selected span.

    A3 does **not** explicitly incorporate round 3’s requested round-off tolerance for D2.

    **Concrete fix:** compare rollout errors and floors using identical time/case aggregation and denominators; assert the inequality with a documented numerical tolerance. A literal zero-tolerance assertion can falsely stop a valid run, but it does not undermine the floor itself.

12. **Competitor scope — CORRECT.**

    Comparison against these FD implementations is adequate for this pilot. Missing multigrid, stabilized explicit, and high-order competitors remain limitations of any broader claim.

    **Concrete treatment:** a B win means promising against the tested implementations, not that removing a global sine transform generally favors ROMs.

13. **Memory and budget — CORRECT.**

    A3.5 extends memory admission to every unit, explicitly includes temporaries and factorization storage, and requires splitting or skipping oversized units. Benchmarks, retries, and previous work consume the total budget; unit and total timeouts both apply.

    **NON-BLOCKING limitation:** projections can underestimate actual consumption.

    **Concrete treatment:** retain incomplete units as incomplete and report resource failures. They are feasibility observations, not evidence against crossover.

14. **Specific numerical claims — CORRECT.**

    The earlier repairs remain effective: discrete versus continuous tensor exactness, packed storage, nonlinear-cost crossover, both SBDF2 factorizations, per-prefix diagonalization, withdrawal of CFL/front-width eligibility claims, approximate wall compatibility, and the envelope quotient caveat.

    A3 supplies the previously missing B coordinate consistency and separate reference-padding check.

    **Concrete treatment:** preserve these distinctions in the report; no additional mathematical repair is required.

One further **NON-BLOCKING provenance clarification** is needed: A3 honestly records that reference/training generation and a debugging ladder preceded this final amendment. Consequently, the pilot should be described as an **amended exploratory design**, not wholly preregistered before execution. Retain the chronology and distinguish debugging observations from subsequent reported readings.

These limitations permit a conditional, implementation-specific feasibility reading. They do not support physical-accuracy certification, exhaustive optimization, or transfer to the learned bank.

OVERALL: READY