**OVERALL: NOT-READY. The requested A1 closure audit cannot be completed against the supplied E2 file.**

The available [E2 design](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-geometry/experiments/imp-geometry/DESIGN.md) ends at line 311, with no amendment. A1 exists in **E5’s** [design](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/DESIGN.md:250), and the named [previous audit](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/audits/codex-design-1.md) also concerns E5. Those answers cannot silently amend E2.

Below are verdicts on the **available E2 text**, checked against both binding briefs and the pilot background. No files were changed, jobs submitted, or arrays copied.

1. **Win/withdraw rule — WRONG.**

   The quote is substantively faithful but not verbatim: it adds “and” before “label S.” More materially, “best competitor” excludes R-PODG even though that arm could beat the bank. Reporting its superiority secondarily does not rescue a headline win against the best competitor.

   Worst-error matching and identifying R-BANK as “the ROM” are sensible. The existential rule—either geometry, either of three targets independently at each level—is a preregisterable scope choice, but supports only that restricted claim. Test-once currently confirms target attainment, without explicitly confirming numerical qualification or the speed threshold on that cohort.

   **Fix:** copy the exact brief; include R-PODG in the competing frontier; freeze the complete selection and confirmation predicate, including qualification, timing population, missing controls and censoring. Distinguish a family-specific speed win from surviving geometry attribution. The factor-two definition of “same S” is an explicit conservative convention, not wording supplied by the brief.

2. **Geometry and families — WRONG.**

   The clearance arithmetic is essentially correct:

   - Outer wall: \(0.30\), giving \(e^{-18}a\approx1.52\times10^{-8}a\).
   - Hole: \(\sqrt{0.3^2+0.3^2}-0.2\approx0.224264\), giving approximately \(4.28\times10^{-5}a\).
   - Re-entrant corner: \(\sqrt{0.3^2+0.3^2}\approx0.424264\).

   The claim that the feature lies on **every** bump’s advection path is false. Diagonal transport preserves \(x-y\); a centre with \(|c_x-c_y|>0.2\sqrt2\) has a centreline that misses the disk. Likewise, reaching either feature by \(T=0.25\) is not established by \(aT\): that uses the initial peak as a sustained velocity despite diffusion and nonlinear evolution.

   Widths and S1/S2 agree with the amended pilot choices. The tiny boundary traces make a substantial incompatibility handicap unlikely, but “set to zero on the boundary” is not an exactly compatible continuous Gaussian.

   **Fix:** withdraw the universal interaction claim; preregister a quantitative dev-only interaction diagnostic against the paired box trajectory. Check sensitivity to an exactly compatible IC before attributing a high-order disadvantage to geometry. Do not redesign the family after inspecting winning cells.

3. **Cohorts — NEEDS-RESTATEMENT.**

   Geometry-paired descriptors, disjoint seeded splits, validation selection and frozen test-once confirmation are sound. Calling val32 the brief’s “32 test cases” is acceptable only with the selection role clearly disclosed.

   Level-dependent seeds mean S1/S2 are **not paired viscosity changes**. That is permissible but differs from the pilot’s paired steepness experiment. Separate seeds alone also do not prove descriptor disjointness; the proposed assertion is necessary.

   Specify that test-once references, gates and failures cannot influence candidate replacement, mesh tuning or reference-policy changes. Confirmation qualification must cover that split.

4. **References — WRONG.**

   A second-order recipe does not establish second-order convergence on these domains. The corner singularity and changing Shortley–Weller cut distances require demonstrated convergence. Jointly halving space and time can also conceal cancellation.

   The fallback contradicts per-case eligibility: a threshold of 24 eligible cases does not make the remaining eight rankable. “Two-way” scoring with a nonzero \(U\) is internally inconsistent. An unqualified fine-grid field cannot establish physical 1–10% accuracy merely by attaching a conditional label.

   The SEM cross-check is diagnostic in §3 but can disqualify SEM in §9. Disagreement does not identify which solver is wrong.

   **Fix:** define per-case reference handling, positive-time order checks, separate temporal refinement and escalation for failed spatial convergence. Failed references make physical-target qualification unavailable; numerical-reference comparisons remain descriptive. Treat independent-solver disagreement as unresolved reference evidence, not automatic competitor failure.

5. **Competitor fairness — WRONG.**

   The unsupported assertion that SEM is mass-dominated with roughly ten Jacobi-PCG iterations cannot justify omitting the requested strong solver treatment. Fine elements, grading and high order can undermine that assumption. A 50-iteration warning does not establish competitive implementation.

   Several essential knobs remain unpinned: macro-mesh connectivity, grading construction, SBDF3 startup, spectral \(N\) ladder, multigrid damping/coarsest solve/boundary transfers, iteration caps and ECSW training/support failures.

   There is also a concrete output inconsistency: FD resolutions 96 and 192 per unit are not multiples of 64, so the promised scoring-lattice subsampling cannot serve the entire ladder.

   **Fix:** implement and benchmark a credible strong SEM solve/preconditioner or explicitly invoke the brief’s limited-competitor fallback; remove the unsupported iteration claim. Specify all ladders and boundary machinery, provide interpolation for nonnested FD meshes, and include R-PODG in the best-competitor comparison. A tuned condensed or more effectively preconditioned high-order solver is the clearest missing candidate; its actual superiority is unmeasured.

6. **Reduced arms — WRONG.**

   Stage-A-only training and training-code ordering are disclosed experimental choices, not inherently unfair. The envelopes vanish on the intended boundaries; the corner representability caveat is appropriate.

   However:

   - At \(r=384\), the test prefix has \(M=512\), not \(4r=1536\). Its oversampling ratio is only \(4/3\); calling this an \(M=4r\) transplant hides a substantial change.
   - Shortley–Weller projection does not automatically produce a symmetric reduced diffusion matrix in the stated grid inner product. Orthogonal diagonalization cannot simply be assumed.
   - The initializer \(\Psi^\top Wu_0\) requires orthonormality in that **same** metric. Changing quadrature may require a mass solve.
   - A precontracted tensor is exact relative to its construction rule, not necessarily the continuum integral.
   - The ordering rule lacks a pinned construction quadrature and rank/conditioning policy.

   **Fix:** write each arm’s mass, diffusion, nonlinear and tested-residual equations explicitly; establish the appropriate metric and diffusion decomposition; specify adequate test-space dimensions, initialization and rank handling. Allow competitors their own tuned steppers rather than forcing all onto the bank’s externally selected winner.

7. **Metrics, search and timing — WRONG.**

   The three-way inequality is reasonable with a valid conditional \(U\). Its handling of ineligible or missing references is not.

   “Fine step” in the spatial screen is undefined. Discarding sizes before searching time can remove competitive configurations; neither error nor cost is guaranteed monotone in these knobs. The one-repetition 1.5× timing screen is a heuristic, not a certified frontier.

   Batch one, same-job/device timing and charging query-dependent setup are good. Timing only val[0:3] leaves the performance claim about three cases, despite accuracy covering 32. Raw repetition persistence and the precise A–B–A aggregation are unspecified.

   **Fix:** pin screen parameters and heuristic labels; score every observed valid configuration; define missing-\(U\) behavior; define case-level timing aggregation, retain raw arrays and bind scored/timed trajectory identity. Confirm the performance claim on a preregistered representative population.

8. **Q1/Q2 — WRONG.**

   These are direct binding-contract failures:

   - ETDRK4 step-halving measures truncation sensitivity; it is not an independent in-job recurrence certificate.
   - A square Newton root needs an appropriate residual certificate; normalized stationarity alone is not a satisfactory root test.
   - Q2 rollout agreement covers 16 cases instead of every case.
   - The largest rule, \(q=12\), has no specified \(q=14\) successor.
   - Tight/tighter validity and confirmation-population qualification are incomplete.
   - ECSW receives no explicit hyper-reduction agreement certificate.
   - Iteration-cap exits are described as “invalid” yet allowed up to 1%; solver failure must not be conflated with permitted nonstationary steps.

   **Fix:** supply every-step independent certificates for each stepper; valid tight/tighter anchors; all-case Q2 rollouts with fixed comparison operators and explicit zero-denominator handling; a terminal successor; ECSW certification or an unqualified label; and a consistent failed-solve policy.

9. **Must-fail controls — WRONG.**

   Most mutations are plausible, not guaranteed witnesses. One LM iteration may suffice; \(q=2\) may integrate a particular state adequately; a zeroed tensor slice may be negligible; random ECSW weights need not violate the chosen threshold. The perturbed reference ladder can accidentally exhibit acceptable order.

   Manufactured SEM/FD convergence rates depend on the unspecified manufactured solution and refinement range. A smooth manufactured solution also does not test reference behavior under the actual corner singularity.

   **Fix:** freeze real dev fixtures and demonstrate each mutation’s detector before production, with any control amendment recorded before result selection. Add controls for wrong ETD stages/cache parameters, invalid anchors, omitted Q2 cases, incorrect output times, and scored/timed configuration mismatch. Specify the ECSW training-fit normalization.

10. **Budget and defer order — WRONG as a certified executable plan.**

    The rounded total fits 62 GPU-hours, but feasibility is unsupported. The reference schedule alone entails about **18.1 million SBDF2 steps**, with S2’s finest grid spanning roughly \(4096^2\) points before masking. Eight GPU-hours implies approximately **1.6 ms per reference step averaged across all levels**, before ancillary costs.

    Training-data generation gets only about **4.7 seconds per trajectory** on average. The proposed bank cost also drops sharply from the pilot’s planning estimate without evidence. Reference multigrid, training, compilation and hundreds of SEM spatial combinations are the likely budget drivers.

    **Fix:** require measured certificates covering full-size reference solves, training, ladder sizes, all qualification replays, retries and memory. Reserve queued/running commitments. Do not defer potentially decisive SEM branches while retaining an unrestricted best-competitor claim.

11. **Other interpretability gaps — WRONG.**

    The fixed \(1/64\) scoring lattice can miss localized errors near the hole or corner. A ranking on that lattice is valid for the declared sampled-output task, but does not automatically establish domain \(L^2\) accuracy.

    Geometry attribution also conflates physical interaction with the feature and the expense of using a non-box solver. Both are legitimate questions, but they are different claims.

    **Fix:** preregister finer-lattice and boundary/corner-region diagnostics, retain the sampled-output scope, and separately report physical geometry effects and solver-cost effects. Resolve the E2/E5 document mismatch before calling this a second-round closure audit.

**OVERALL: NOT-READY**

Required fixes:

1. Supply the amended **E2** design and its corresponding prior audit.
2. Repair the best-competitor and confirmation predicates.
3. Correct the interaction claim and preregister geometry/IC diagnostics.
4. Replace the reference fallback with explicit per-case qualification and escalation.
5. Specify and validate strong competitor implementations and complete ladders.
6. Resolve reduced-space dimensions, metrics, diffusion algebra and initialization.
7. Close all-case Q1/Q2, anchor-validity and explicit-step certificate gaps.
8. Demonstrate deterministic control witnesses and preserve timing evidence.
9. Produce a full-size cost certificate and scope-preserving defer policy.