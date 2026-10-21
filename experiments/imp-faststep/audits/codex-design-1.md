**Overall verdict: WRONG as a qualification-ready pre-registration.** The proposed reduced equations and integrators are sound. The blocking problems are the Q1/Q2 qualification gaps, the unsupported stability interpretation, and competitor selection that does not satisfy the accuracy-matching contract. These can be fixed without changing the central experiment.

This was a read-only source audit. No files were changed, jobs submitted, arrays copied, or numerical experiments run. Control outcomes below are therefore assessments of their design, not observed passes.

References below use line numbers in [DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/DESIGN.md), checked against the [common brief](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-21-improvement-campaign/campaign/briefs/COMMON-BRIEF.md), [lane brief](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-21-improvement-campaign/campaign/briefs/imp-faststep.md), ideation sections requested, and the named code.

1. **Diffusion-coordinate algebra — CORRECT, with one notation correction.**  
   Lines 42–49 agree with `p1core.make_evolve`’s GAL residual and `qcore.tested_value('point')`. With full-column-rank \(A=QR_m\),
   \[
   \dot z=-V^\top Q^\top N(R_m^{-1}Vz)-\nu Dz,
   \]
   giving exactly the stated \(B_z,D_z,P_z\). Also
   \[
   \|Ac\|^2=\|QR_mc\|^2=\|R_mc\|^2=\|V^\top R_mc\|^2.
   \]
   The residual norm and tolerance scale transfer under the orthogonal transformation.

   **Fix:** line 40’s rectangular equation \(A\dot c=-F(c)\) is generally inconsistent; it is not itself the GAL ODE. Write \(Q^\top[A\dot c+F(c)]=0\). Record rank/conditioning of \(A\), QR orthogonality, eigenpair residuals, and positive diffusion eigenvalues. Use triangular solves to construct the maps.

2. **ETDRK4 coefficient use — CORRECT.**  
   Lines 75–78 use the correct split: `etdrk4_coefs(-nu*D, dt, 32)`, with only negative advection in the nonlinear function. `pscore` includes the timestep factors in \(Q,f_1,f_2,f_3\); `psjax2d` uses
   \[
   E z+f_1N_z+2f_2(N_a+N_b)+f_3N_c.
   \]
   Copying that recurrence preserves its convention. No additional timestep multiplication belongs outside those coefficients.

   **Fix:** explicitly freeze this recurrence and include coefficient construction inside timed evolution, since \(\nu\) is a query input. Certify intermediate stages as well as endpoint states.

3. **One-pass linearised CN order — CORRECT.**  
   For the autonomous square ODE \(\dot z=-g(z)\),
   \[
   \delta=-h(I+\tfrac h2J_g)^{-1}g
          =-hg+\tfrac{h^2}{2}J_gg+O(h^3),
   \]
   matching the exact solution through second order. This holds with the exact Jacobian at the current state, including the first step. The design correctly avoids the retracted rectangular-LSPG version.

   **Fix:** distinguish second-order consistency from stability. Also verify convergence of the proposed ETDRK4 reference; simply naming \(h/8\) “converged” does not establish it.

4. **Q1 analogues — WRONG as currently sufficient for qualification.**  
   Lines 118–141 have mixed quality:

   - **LSPG:** the stated fresh-Jacobian contract is correct. Require valid tight/tighter anchors explicitly, as the existing qualification code does.
   - **Galerkin Newton:** a fresh root residual is an honest replacement for normalized stationarity. Normalized \(\gamma\) need not decrease toward a square-system root. However, “at a root it is 0/0” is inaccurate for the implemented denominator with \(10^{-300}\): it returns zero. Separate the production tolerance from the \(10^{-14}\) certificate allowance, which dominates the nominal tighter tolerance.
   - **ETDRK4:** comparing two RHS evaluators at endpoints does **not** certify the time-step recurrence. A wrong stage combination can pass that check on all validation cases. A twin on dev6 does not close the every-case gap.
   - **One-pass:** fresh-Jacobian agreement and a linear residual are appropriate, but certify the RHS independently and check that the actual saved increment equals \(z_{n+1}-z_n\). Otherwise a correct linear solve can coexist with an incorrect update.
   - **POD ETDRK4:** “validity, stability, twin” lacks explicit recurrence residuals and numerical thresholds.

   **Fix:** independently reconstruct every explicit stage and final update from each saved step’s input on **every case**; certify the one-pass equation using the actual saved increment and original-table RHS/Jacobian. Map each COMMON-BRIEF requirement explicitly to its non-Newton analogue. Dev6 twins can supplement those certificates.

   I would **not** automatically impose step-halving as a Q1 algebraic-error bar: temporal truncation error is distinct from solving the chosen discrete scheme correctly. But implementation equivalence alone must not be presented as a time-accuracy certificate.

5. **Q2 — WRONG population; otherwise substantially correct.**  
   Lines 142–147 restrict rollout agreement to sub16. COMMON-BRIEF §3 requires the rollout-distance bar on **every case**. Inheriting the earlier `p1acc.py` subset does not override the new brief.

   The unprojected tested-advection difference, evaluated on the arm’s own reached states, is appropriate; retaining \(A,Q,R_m,V\) when changing only quadrature is also correct.

   **Fix:** run successor rollouts on all 38 cases for every claimed configuration. Specify that both forces are evaluated at the **same main-rule state**; require valid successor runs; retain the existing zero-denominator/non-finite handling. Six output-state checks are a declared sampling choice, not an all-step guarantee.

   POD does have an HR ladder—\(\kappa=4,8,16\), plus full integration—even though it lacks this classical-rule ladder. Define a separate HR-agreement certificate or explicitly withhold a numerically-qualified label.

6. **Stability monitor — WRONG justification; NEEDS-RESTATEMENT as an empirical gate.**  
   Lines 155–158 infer projected-norm decay from continuum energy decay. That inference fails. For this reduced ODE,
   \[
   \frac12\frac{d}{dt}\|z\|^2
   =-(Ac)^\top N(c)-\nu z^\top Dz.
   \]
   The advection term has no guaranteed sign. The tested projection is not the complete trial field; projection effects persist even with exact quadrature. Central-difference POD advection likewise does not automatically preserve the continuum energy identity.

   Consequently, a converged reduced trajectory can breach 1.05 without a time-step instability. Conversely, a bounded but inaccurate or oscillatory trajectory can remain below it.

   **Fix:** retain 1.05 only as an explicitly conservative amplitude-growth rejection threshold, not a theorem or a margin attributable solely to quadrature. Compare with a converged same-ODE trajectory to distinguish reduced-model growth from time-integrator instability. Check stage finiteness and add a stated oscillation/growth diagnostic.

7. **Stop rule — NEEDS-RESTATEMENT; core reading is faithful.**  
   Lines 162–164 reproduce the brief’s words, modulo wrapping. Separate outcomes for arms 2/3a/3b/4, the strict worst-error comparison, and the additional numerical-qualification requirement for “fixed” are correct.

   The paired median-of-case-medians evolution statistic is a reasonable preregistered operational choice; the brief does not prescribe another aggregation. It measures median performance, not a threefold gain on every case.

   **Fixes:**

   - Keep the all-case strict comparison as a conservative literal stop predicate, but do not use ineligible cases to establish an E0 accuracy ranking.
   - “\(2\varepsilon_{\rm ref}\) on the cases attaining the maxima” is ambiguous when the maxima occur on different cases. Report eligible same-case resolved comparisons separately, or define worst-error uncertainty intervals explicitly.
   - Specify handling of missing/non-finite errors, empty eligible sets, and incomplete configurations; none may pass vacuously.
   - “Every mesh run” permits a blanket “fixed” after deferring cells. Report **fixed on the completed scope**, with the original full-grid conclusion incomplete.
   - Preserve F5’s separate kill outcome. If arm 4 fails its registered order/accuracy/gain criterion, it must not silently become the recommended F5 implementation merely because the broader E5 predicate passes.

8. **POD retraining, selection and fairness — WRONG as a complete COMMON-BRIEF §5 comparison.**  
   Retraining weights for \(\Phi^\top N_h(\Phi c)\) is **correct**: `build.ecsw_contrib` currently targets \(J^\top r\). Reusing its old weights would not train the requested Galerkin force. Exact diffusion followed by orthogonal diagonalization is also correct.

   Problems requiring fixes:

   - The new contribution routine must adapt the importance distribution to the new force contributions. “Exactly as `ecsw_contrib`” must mean its sampling policy, not its LSPG arithmetic.
   - Preserve the existing positive block-norm floor and dropped-block handling; literal \(1/\|b(s)\|\) is unsafe.
   - A support checkpoint can be unavailable. `ecsw_weights` can return `None`; “otherwise choose \(\kappa=16\)” is not always executable.
   - Explain why \(\kappa=2\) is trained but excluded from the online ladder.
   - Matching a candidate to 1.1 times **POD-full’s own** worst error does not accuracy-match it to our arm. Choosing the most accurate timestep also needlessly excludes faster accuracy-matched competitors.
   - Same-rank comparisons are useful mechanism controls, but COMMON-BRIEF explicitly allows competitors to tune rank. Use the available ladder across ranks when forming the matched frontier.
   - The historical r=128 strongest comparator is not established by replaying r=128 ECSW-k8. Ideation §2.2 identifies a cross-rank Q-DEIM comparator for its reported r=128 comparison.
   - Carry forward the reused 1024² basis’s **snapshot-limited** restriction and unresolved tuning-budget restriction from the baseline report.

   **Fix:** retain the dev6-only HR pick as a diagnostic, but separately select the fastest valid, accuracy-matched competitor from the declared available ladder on dev6, then report its frozen validation outcome. Label finite-ladder speedups as upper bounds on speedup against a fully tuned competitor.

   **4096² exclusion:** defensible as a disclosed budget limitation under COMMON-BRIEF §5, but it removes the shared-integrator POD comparison there. Do not claim H-base is resolved at 4096². Building a new basis from 576 trajectories is one construction recipe, not a demonstrated lower bound on every possible POD comparator.

9. **Must-fail controls — NEEDS-RESTATEMENT overall.**

   | Control | Verdict | Assessment and concrete fix |
   |---|---|---|
   | C1/C1m | CORRECT | Good equivalence test and genuine wrong-congruence mutation. Freeze random seed, state scale, and absolute/relative norm convention; include reached states. |
   | C2/C2m | CORRECT | Compare against **CN**, not CNR, with identical initialization and valid tight anchors. Budget zero is a plausible real failure; demonstrate it rather than assume it. |
   | C3 | CORRECT | Tests exact linear propagation, but exercises none of the nonlinear ETD coefficients. |
   | C3m | NEEDS-RESTATEMENT | Verify reference convergence and an asymptotic timestep range. At \(L=0\), \(f_1=f_2\), so the mutation is ineffective in that limit. Freeze a nondegenerate real case and show detection. |
   | C4m | NEEDS-RESTATEMENT | Negative viscosity can produce growth, but crossing 1.05 within this horizon is not guaranteed; overflow may trigger invalidity first. Specify the expected detector and real fixture. |
   | C5 | NEEDS-RESTATEMENT | Euler is a sound order mutant. A fixed coarse range can nevertheless give misleading slopes. Verify the reference and retain per-case slopes/failures, not only the median. |
   | C6m | NEEDS-RESTATEMENT | A wrong Jacobian can still converge to a stationary solution. It need not yield >1% nonstationary steps. Add direct analytic-versus-fresh-autodiff Jacobian comparison; reserve stationarity failure for an empirically demonstrated fixture. |
   | C7m | NEEDS-RESTATEMENT | Gauss32 is plausibly inadequate, but failure of this new arm on the smoke cases is unproven. Freeze a real witness and specify own-state versus mutant-rollout evaluation. |
   | C8 | CORRECT | Different timesteps should change trajectory identity. Include schedule/configuration, physical output times, and iteration/exit data where applicable. |
   | C9 | CORRECT | Self-comparison should fail the 3× predicate. The ±10% identity band is a measurement-quality check requiring defined handling of noisy results. |
   | C10 | NEEDS-RESTATEMENT | Rank does not guarantee worst-error ordering. Select and pin an independently evidenced real pair, or perturb a real scored trajectory sufficiently to exercise the predicate. |
   | C11 | WRONG as stated | NNLS relres is the norm ratio of the **stacked, block-scaled** training system, not every individual force error. Recompute that exact aggregate. Uniform random weights are not guaranteed to be 10× worse; define their normalization and demonstrate a witness. |
   | C12 | NEEDS-RESTATEMENT | Existing guards are useful but insufficient for new tables. `Setting.check` fingerprints the LSPG data, not \(B_z,D_z,P_z,V,D\). Bind transformed tables, POD artifacts, rule, rank and mesh into the checked manifest. |

   **Missing important controls:** a wrong ETD stage/update caught on a non-dev case; wrong \(\nu\)/coefficient cache; a correct linear solve followed by a wrong state update; invalid tight/successor anchors; omission of a Q2 case; and output-time/schedule mismatch. These directly exercise the identified qualification gaps.

10. **GPU-hour estimate and defer order — NEEDS-RESTATEMENT.**  
    The table totals **3.5 GPU-h including reserve**, so its arithmetic fits the 4-hour cap. The defer order is copied correctly.

    Feasibility is not established. The reused baseline report records 2,784 seconds of 1024² ECSW NNLS construction across its historical workload; this does not prove the new two-rank 0.6-hour estimate impossible, but it makes a measured pilot essential. The new r=384 contribution matrix alone contains \(120\times384\times32768\) doubles, approximately **12.1 GB**, before factorization workspace and copies.

    **Fix:** budget NNLS retries/timeouts, compilation, full-node POD diagnostics, all qualification replays, stage/Jacobian certificates, burn-ins, and revised all-case Q2. Reserve projected hours for **queued/running jobs**, not only completed expenditure. Enforce remaining allocation limits.

    A 1024² smoke cannot establish the 4096² peak-memory requirement. Add an actual 4096² construction/memory pilot accounting for bank construction temporaries, decode buffers and retained compiled arrays.

11. **Other answerability gaps — NEEDS-RESTATEMENT.**

    - **Dev-only selection:** lines 213–214 select among arms passing gates that use val32. That makes eligibility for the “best stepper” validation-informed. Select the candidate and its eligibility using dev6, then confirm on val32; otherwise label the selection accurately.
    - **M=1152 implementation:** `p1common.Setting` hardcodes \(M=4r\). Arm 6 needs an explicit supported override and rebuilt, checked metadata; it cannot be obtained by unchanged use of that class.
    - **Order references:** qualify the \(h/8\) ETDRK4 reference against a finer run before treating order failures as method failures.
    - **Timing scope:** retain raw A–B–A repetitions, burn-in/drift handling and identical scored/timed outputs. Include coordinate maps and per-query ETD coefficients in evolution. Specify the same full-field query contract for POD and bank fused timings.
    - **Claims:** line 246 is false as a blanket description: the E5 speed ratio is against the incumbent, not necessarily the strongest tested competitor. Label incumbent speedup, matched-competitor speedup, and shared-integrator controls separately.

**Final disposition:** preserve the mathematical method; amend and re-audit qualification, stability semantics, competitor selection, and control definitions before production. As written, a passing result could establish fast execution of the proposed formulas without establishing the full numerical qualification or fair-comparison claim required for E5 “fixed.”