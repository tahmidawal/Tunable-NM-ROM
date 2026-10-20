**Overall: the core discretisations are sound, but the pre-registration needs revision before execution.** The main problems are error attribution, solver acceptance, gate interpretation, and the claims permitted by selection. No files were modified.

1. **CORRECT — LMM coefficient table and startup orders (§2).**  
   BE, CN, BDF2 and TH06 have the stated coefficients. A fixed number of full BE startup steps supplies \(O(\Delta t^2)\) starting errors, sufficient for global order two for CN-R/BDF2 on a smooth, stable, fixed-dimensional ODE. Calling CN-R “Rannacher-type” while explicitly distinguishing full from half steps is appropriate. This does **not** establish second-order convergence of their LSPG versions or uniform accuracy as stiffness increases.

2. **CORRECT — Weighted LSPG residual, Jacobian and BE reduction (§2).**  
   With
   \[
   D=(\alpha_0I+\beta_0\Delta t\,\nu\Lambda)^{-1},
   \]
   the stated residual and Jacobian are correct. In fact,
   \[
   D(\alpha_0A+\beta_0\Delta t\,F')
   =A+\beta_0\Delta t\,D N',
   \]
   which at BE reproduces the Jacobian in [qcore.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/vendor/quad2d/qcore.py:279). Algebraic equivalence does not imply bitwise equivalence after rearrangement.

3. **NEEDS-RESTATEMENT — LSPG’s first-order limit is a generic prediction, not a universal order theorem (§2).**  
   The mechanism is sound: a timestep-dependent test space generally introduces an \(O(\Delta t^2)\) step perturbation and an \(O(\Delta t)\) trajectory error relative to GAL. However:
   - The displayed perturbation drops \(\alpha_0\) factors; for BDF2, \(D\to(2/3)I\), not \(I\).
   - A *small but nonzero* leading coefficient generally delays first-order behaviour; it does not change the asymptotic order.
   - Cancellation can eliminate that coefficient even with a nonzero out-of-range residual.
   - Fixed stopping tolerances do not guarantee convergence as \(\Delta t\to0\).

   **Fix:** set \(\widehat D=\alpha_0D\). At the GAL step, where \(A^{\mathsf T}\delta=0\), the stationarity perturbation is
   \[
   A^{\mathsf T}(\widehat D^2-I)\delta+
   \frac{\beta_0\Delta t}{\alpha_0}F'^{\mathsf T}\widehat D^2\delta.
   \]
   State the order-one prediction conditional on smoothness, rank, stability, a consistent solution branch, negligible solve error, and a nonzero leading perturbation. The general continuous-limit relationship is consistent with [Carlberg–Barone–Antil](https://arxiv.org/abs/1504.03749); the rate for this weighting needs the lane’s own derivation.

4. **CORRECT — GAL is the LMM applied to the stated reduced ODE (§2).**  
   For full-column-rank, fixed \(A=QR\),
   \[
   Q^{\mathsf T}\mathcal R_n=0
   \]
   is exactly the LMM for \(R\dot c=-Q^{\mathsf T}F(c)\), equivalent to \(A^{\mathsf T}A\dot c=-A^{\mathsf T}F(c)\). It is Galerkin in the **tested residual space**, not necessarily physical-space Galerkin. Record the rank and conditioning of \(A\), since the equivalence and numerical tolerance interpretation require them.

5. **WRONG — The motivating ST–S gap is identified with BE time error (§1).**  
   The quoted numerical summaries match the supplied JSON. Their interpretation does not follow. Differences between error norms are not norms of temporal error; taking separate maxima over time further prevents that identification. Matching BE and \(\Delta t\) does not make FOM and LSPG temporal errors identical.

   The factor about 18 is a ratio of median reference discrepancies, not an established temporal/non-temporal error ratio. “Every arm” also exceeds the evidence script’s six selected setting/arm combinations.

   **Fix:** call these *reference-sensitive error gaps*. Report \(\|S-ST\|\), ROM refinement differences, and, where useful, vector-error alignment. Restrict the claim to the inspected arms.

6. **NEEDS-RESTATEMENT — ST and TX are useful provisional references; S is not BE-exclusive (§4).**  
   \[
   TX=(16ST-S)/15
   \]
   correctly cancels the leading BE time-error term at fixed mesh, assuming an asymptotic expansion over that unusually wide refinement ratio. It leaves spatial error and unverified higher-order temporal error.

   “Against S only the BE ROM is meaningful” is false. Any method’s distance from S is meaningful as agreement with that specific discrete solution. Even BE at a different timestep lacks the claimed matched-time interpretation.

   **Fix:** distinguish *agreement with S* from *estimated continuum accuracy*. Retain ST/TX provisional labels beside selections and headline claims, report ST–TX sensitivity, and mark ranking changes as reference-dependent. Do not infer a certified accuracy floor from these two references.

7. **CORRECT — The timestep ladder and divisibility adjustment (§3).**  
   Every listed timestep divides 0.05. Replacing \(4\Delta t_0\) by \(5\Delta t_0\) avoids interpolation and unequal observation times. Restrict factor-two order estimates to the dyadic subset, as proposed.

8. **NEEDS-RESTATEMENT — Self-convergence definition and anchor resolution (§5).**  
   The self-difference metric is sound on common output times, but the indexing should explicitly define
   \[
   p(h)=\log_2\!\frac{\|u_h-u_{h/2}\|_{\max}}{\|u_{h/2}-u_{h/4}\|_{\max}}.
   \]
   Otherwise, the ascending ladder makes \(k+1\) ambiguous.

   The BDF2 anchor’s difference-divided-by-three estimate is valid only in the second-order asymptotic regime. Its own reported anchor distance is zero, although its temporal error is not.

   **Fix:** call \(e_{\rm time}\) an *anchor discrepancy* until resolution is demonstrated. Add a predeclared finer anchor check and mark discrepancies comparable to anchor uncertainty unresolved. The LSPG-BE Richardson fallback also requires demonstrated first-order asymptotics and solver convergence. Reconcile that fallback with G2’s instruction to stop when GAL fails.

9. **NEEDS-RESTATEMENT — G1’s tolerance is plausible, but the gate is too weak (§6).**  
   Changing an outer `scan` to a `while_loop` does not inherently invalidate \(10^{-6}\) reproduction. However, compiler arithmetic changes can alter predictor choices or stopping branches, so success cannot be promised without a comparison.

   Matching two scalar error summaries does not prove trajectory reproduction: different trajectories can have identical errors.

   **Fix:** compare vendor and generalized BE outputs directly in the same job, including initial coefficients, output fields and solver diagnostics. Keep historical scalar parity as a separate check. Distinguish arithmetic differences from algorithmic changes rather than loosening the threshold automatically.

10. **NEEDS-RESTATEMENT — CN and TH06 are not guaranteed “must-fail” controls (§6).**  
    CN can agree with BE within \(10^{-6}\) on nearly stationary trajectories. TH06 is asymptotically first order, but its first-order coefficient is small enough that second-order terms or cancellation can produce apparent order near two on a finite ladder. Correct code can therefore fail the proposed control requirement.

    **Fix:** retain these as empirical controls, with an explicit “not yet asymptotic/inconclusive” outcome. Add manufactured ODE cases with known nonzero leading errors and deliberate coefficient/history faults to test that the machinery detects wrong implementations.

11. **WRONG — Tight `gtol` alone does not establish the proposed solver noise floor (§§3, 5–6).**  
    In [hfast.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/vendor/quad2d/vendor/hfast.py:51), `gtol` bounds
    \[
    \frac{\|J^{\mathsf T}r\|}{\|J\|_F\|r\|},
    \]
    not trajectory error. The deployed query also has a separate residual threshold, \(10^{-9}\times\text{scale}\). Changing `gtol` to \(10^{-9}\) alone leaves that early exit available.

    Furthermore, the same LM can exit through stationarity or a tiny step before meeting GAL’s requested root residual.

    **Fix:** specify every stopping criterion for each form. Independently enforce GAL’s root-residual requirement. Establish trajectory-level solve uncertainty through a tighter-tolerance replay, then require both differences entering an order estimate to exceed that uncertainty by a predeclared margin. Do not label \(10^{-10}\) “solver noise” without measurement.

12. **NEEDS-RESTATEMENT — G2 can reject correct physics and accept a misleading median (§6).**  
    Order bands are useful diagnostics, but a single finest triple does not establish an asymptotic regime. Stiff transients, initial boundary incompatibility, cancellation and solve error can move correct implementations outside the bands. A cohort median can conceal many failures; exclusions can leave an unrepresentative subset.

    **Fix:** separate manufactured implementation tests from real-data asymptotic checks. Predefine a refinement extension, minimum valid-case coverage, and reporting of per-case orders and outlier counts. A real-data failure should block an order claim, without automatically being diagnosed as a coding bug.

13. **WRONG — G3 admits solver failures that its wording misses (§6).**  
    Reason 2 is a tiny-step exit and need not mean stationarity or root convergence. Reason 4 need not satisfy GAL’s residual requirement. Screening only reasons 0 and 3 misses both. Allowing up to 1% failed steps can also admit an early failure that contaminates the entire trajectory; the aggregation denominator is unspecified.

    **Fix:** independently recompute acceptance conditions at every accepted step. Treat tiny-step exits as unverified unless those conditions pass. Specify per-case accounting and exclude trajectories with unresolved failures from selection, while retaining them in failure summaries.

14. **WRONG — The ringing threshold can flag purely monotone decay (§§5, 8).**  
    For a non-oscillatory scalar mode \(x_n=r^n\), \(0<r<1\), the proposed ratio is
    \[
    \frac{1-r}{1+r}.
    \]
    It exceeds 0.5 whenever \(r<1/3\). Thus fast smooth decay can trigger H3. The index also has an undefined zero-over-zero case and ignores amplitude.

    **Fix:** require an amplitude floor and evidence of alternating signed modal behaviour or negatively aligned increments. Define zero-denominator handling. Keep the existing quantity as normalized second-difference magnitude, not a standalone ringing detector. State exactly how a ringing classification affects selection.

15. **NEEDS-RESTATEMENT — A–B–A is sensible, but G4 is insufficient (§§6–7).**  
    Same-device pairing, warm-up, randomized order and pre-call burn are good choices. Matching warm-up hashes proves repeatability, not correspondence with the accuracy run. A stale timestep or wrong arm can repeat perfectly.

    **Fix:** hash timed outputs against the stored accuracy outputs for the same complete configuration. Synchronize before/after timing, persist all A/B/A samples, identify the physical GPU, and predeclare a baseline-drift diagnostic. Report outliers and uncertainty, not just the winning median from three repetitions.

16. **NEEDS-RESTATEMENT — Tracing removes recompilation differences, not all performance confounding (§2).**  
    Dynamic step counts require fixed-shape output/diagnostic buffers or another explicit strategy; the vendor’s variable-length scan outputs cannot simply be retained unchanged. Dynamic coefficients also do not guarantee that multiplication by zero removes unneeded history evaluations.

    **Fix:** specify the loop, buffer masks, runtime branches and argument shapes/dtypes. Verify no compilation occurs during timing. Define whether BE avoids unnecessary \(F(c_n)\) work. Benchmark the generic BE implementation against the vendor implementation so shared generic overhead cannot silently favour larger timesteps.

17. **NEEDS-RESTATEMENT — “Solve time” is inconsistent with “query minus nothing” (§5).**  
    The deployed query includes initialization and decoding. Timing decoding separately does not turn total query time into solve time.

    **Fix:** define the primary cost explicitly as end-to-end query latency, including initialization and six decoded fields. If solve-only time is also needed, instrument it separately. Match FOM and ROM output obligations.

18. **NEEDS-RESTATEMENT — H1/H2 and selection are operationally incomplete (§8).**  
    H1/H2 are legitimate development-set existence tests, but “best” needs an explicit objective, and “second-order arm” mislabels LSPG arms if the predicted order-one behaviour occurs. Selection references G3 without explicitly requiring the other gates. It also does not clearly restrict candidates to arms with measured timings.

    **Fix:** define H1/H2 as existence of one arm satisfying all listed inequalities; call them CN/BDF2-family arms until order is established. Freeze the exact candidate universe, required gates, tie-breaking, and no-eligible-arm outcome. Include the comparator in the eligible set only intentionally. State that speed conclusions apply to the six timed development cases, not automatically the entire accuracy cohort.

19. **NEEDS-RESTATEMENT — Within-lane selection avoids test use, but replication is historical (§§8, 11).**  
    Selection restricted to dev6 ∪ val32 is appropriate. However, test64 has already been examined in earlier work, and the quadrature choices inherit that research history. It is not a new sealed confirmation cohort.

    **Fix:** call it historical-cohort replication and disclose inherited exposure. Freeze solver settings, failure rules, references and reporting before evaluation. Include the primary \(L=1024\) setting in replication; testing only at \(L=4096\) changes mesh and cohort simultaneously.

20. **CORRECT — FOM LMM residual and diffusion preconditioner (§9).**  
    The residual and DST inverse
    \[
    (\alpha_0+\beta_0\Delta t\,\nu\lambda)^{-1}
    \]
    have the correct signs and scaling for the diffusion part of the Newton Jacobian. Omitting advection from that preconditioner is an approximation, not a mathematical error. BE recovers [engines.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/mr-burgers2d/engines.py:84).

21. **NEEDS-RESTATEMENT — Section 9 does not yet establish a fair FOM frontier.**  
    A single inherited tolerance pair may over-solve some arms and under-solve others. The existing FOM discards BiCGStab status and can return after reaching the Newton limit. A two-case order check at \(L=256\) does not establish solve accuracy at \(L=4096\). The smaller FOM timestep grid also limits what “best FOM” means.

    **Fix:** record final nonlinear and linear residuals and explicitly reject unconverged solves. Add development-only tolerance calibration and tighter replay checks. Define the equal-error target and candidate grid, reporting “best among tested configurations.” Restrict “both second order” to arms actually shown to be second order; it cannot automatically include LSPG-CN/BDF2.

22. **NEEDS-RESTATEMENT — Missing attribution and audit outputs could prevent a clear answer (§14).**  
    New time integrators visit new states. Quadrature certification inherited from BE trajectories does not automatically cover them. Production-versus-tight discrepancies may explain apparent temporal gains. Restricted \(257^2\) outputs can also miss oscillations visible on the full mesh.

    **Fix:** pre-register:
    - production-versus-tight trajectory discrepancies;
    - higher-quadrature replay on selected and worst cases;
    - denser-output audits on a small fixed subset;
    - per-case fields or sufficient coefficients for independent metric reconstruction;
    - mutation tests for wrong history, coefficients, output times and reference indexing.

    These distinguish time-discretisation effects from solve error, quadrature error and observation-grid effects.

23. **WRONG — The local smoke allocation contradicts the operating rules (§12).**  
    The proposed 0.3-hour local GPU job exceeds the repository’s under-one-minute local-smoke limit.

    **Fix:** split out a genuinely sub-minute smoke and move the complete validation grid to the cluster.

**Overall verdict: NEEDS-RESTATEMENT — revise before running.** The residuals and GAL construction provide a credible experiment. As written, however, the protocol can mistake solver error for temporal order, monotone decay for ringing, and reference agreement for removal of BE time error.