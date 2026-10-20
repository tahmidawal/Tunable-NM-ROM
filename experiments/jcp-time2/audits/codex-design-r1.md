**Overall verdict: revise A1 before execution.** The core equations remain sound, but manufactured-test independence, solver-uncertainty coverage, ringing classification, and selection still have holes. No files were modified. “RESOLVED” includes previously correct items that remain correct.

1. **RESOLVED — LMM coefficients/startup.** Still correct under the original smoothness/stability qualifications; these do not establish second-order LSPG convergence.

2. **RESOLVED — Weighted residual/Jacobian.** Algebra remains correct.

3. **RESOLVED — LSPG order prediction.** A1.2 fixes the missing \(\alpha_0\) factors and appropriately conditions the generic first-order prediction.

4. **RESOLVED — GAL equivalence.** Correct; recording rank and conditioning addresses the numerical qualification.

5. **RESOLVED — Motivating error gap.** A1.1 retracts the invalid temporal-error interpretation and restricts the evidence to inspected arms.

6. **RESOLVED — References.** S agreement, provisional ST/TX accuracy, and reference-dependent rankings are now distinguished appropriately.

7. **RESOLVED — Timestep ladder.** Divisibility and dyadic-order restrictions remain correct.

8. **NEEDS-RESTATEMENT — Anchor uncertainty.** A1.3 compares normalized anchor discrepancies with an uncertainty written as an unnormalized norm. A successive difference is also only an uncertainty *indicator*, particularly outside an established asymptotic regime. **Fix:** use the same output-time maximum and normalization; include solve uncertainty; define anchor eligibility per case and distinguish G2a failure from G2b failure.

9. **NEEDS-RESTATEMENT — G1a tolerance.** Direct trajectory comparison is a substantial improvement. However, \(10^{-8}\) agreement is not guaranteed by algebraic equivalence when production stationarity is \(10^{-3}\): reassociation can change predictor/stopping branches. This is not inherently infeasible, but remains unvalidated. Coefficient tolerances are unspecified, and historical parity was weakened to \(10^{-5}\) without justification. **Fix:** preserve vendor BE arithmetic where practicable, specify coefficient norms/tolerances, and diagnose failures before changing thresholds.

10. **STILL-WRONG — Manufactured test and mutations.** A1.4 describes a random nonlinear benchmark, not an explicitly manufactured known solution. No seed, initial state, quadratic coefficients, or guaranteed asymptotic regime is specified; same-code fine BDF2 is not an independent oracle.
    
    **Fix:** freeze a nondegenerate problem with an exact solution or independent verified reference, and independently evaluate the intended equations and output times. Specifically:
    - **m1:** sign-flipped BDF2 is inconsistent, but an equilibrium can conceal it; require nonstationary data.
    - **m2:** lagged CN generally becomes first order, but the chosen finite triple need not expose it; independently check intended CN residuals.
    - **m3:** define exactly which history value stays stale; independently check the required history at every step.
    - **m4:** output shifting leaves step residuals unchanged; test timestamps and field-to-time correspondence, including the final output.
    - **m5:** **CN has \(\beta_0=\beta_1\), so this mutation does nothing.** Use BE/BDF2 with a demonstrably nonzero least-squares residual and nonuniform weights. Require the unmutated implementation to match NumPy and the mutation to exceed a specified discrepancy threshold. Exact-root problems can hide weighting errors too.

11. **STILL-WRONG — Solve uncertainty.** In [make_fused_lm](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/vendor/quad2d/vendor/hfast.py:65), zero tolerances do not literally disable tests; comparisons remain `<=`. More consequentially, the fixed tiny-step threshold and strict residual-decrease requirement remain active. GAL \(10^{-15}\) verification can therefore be unattainable before stagnation; tighter requested tolerances do not ensure tighter solutions. Dev6 maxima also do not bound val32 solve error. **Fix:** explicitly control stopping tests, record achieved convergence, treat failed tighter replays as unresolved, and measure uncertainty on every case used for order claims. Compare each difference against uncertainty from **both** participating trajectories.

12. **NEEDS-RESTATEMENT — Real-data order gate.** Coverage/outlier rules improve substantially, but “finest valid triple” can select different regimes across cases, and the second-order band for all CN/BDF2-family arms does not express A1.2’s first-order LSPG prediction. **Fix:** predeclare form-specific claim tests, identify every selected triple, require consistency across adjacent resolved triples where available, and apply item 11’s uncertainty correction.

13. **RESOLVED — Failed-step exclusion.** Independent returned-state verification and exclusion after any failed step close the original acceptance hole. This protects validity even when item 11’s tight solves fail.

14. **STILL-WRONG — Ringing index.** The amplitude gate is disconnected from the alternating increments. For one normalized mode with increments
    \[
    (1,0,\varepsilon,-\varepsilon,\varepsilon),
    \]
    A1.6 gives \(\mathcal I=1\) and \(a=1\), however tiny \(\varepsilon\) is. It therefore labels negligible alternating noise as ringing. **Fix:** impose amplitude thresholds on the same modes/increment pairs contributing negative products and require repeated alternation; retain “alternation diagnostic” unless numerical ringing is independently established.

15. **NEEDS-RESTATEMENT — Timing integrity.** Accuracy-output hashes, UUIDs and persisted samples resolve much of this. Explicit synchronization and uncertainty/outlier reporting remain missing. **Fix:** block completion of warm-up/burn before starting the clock and block all query outputs before stopping it; predeclare aggregation across cases/repetitions and report spread/outlier counts.

16. **RESOLVED — Dynamic-loop implementation.** Fixed buffers, runtime history branching, cache checks and vendor-overhead timing address the previous concern. Implementation must initialize buffer slot zero and preserve the previous increments needed by A1.6.

17. **RESOLVED — Cost definition.** End-to-end initialization, stepping and six-field decoding is now explicit.

18. **NEEDS-RESTATEMENT — Selection.** Including the verified comparator makes “fast-equal-accuracy has an eligible arm” incapable of failing: its accuracy inequalities hold identically. With self-ratio defined as one, the same applies to accurate-equal-cost. This does **not** make H1/H2 tautological; their improvement thresholds can fail. Remaining ties across forms/schemes are unresolved. **Fix:** distinguish “baseline retained/no improvement” from “no verified candidate”; define baseline self-ratio, a final deterministic tie-break, required integrity gates, and what happens when the comparator fails verification.

19. **RESOLVED — Historical replication.** A1.9 correctly discloses exposure and includes primary-mesh replication with frozen settings.

20. **RESOLVED — FOM equations.** Residual and diffusion preconditioner remain correct.

21. **NEEDS-RESTATEMENT — FOM calibration.** Recording residuals helps, but calibration aggregation and scope across meshes/schemes/timesteps remain unspecified. Reaching the iteration limit is not failure if the last permitted iteration converged. Linear failures are recorded without an explicit handling rule. **Fix:** define per-configuration calibration across all dev6 cases, verify the tight reference, reject unconverged/nonfinite final states, and specify linear-failure handling; do not reject solely for successful convergence on the last iteration.

22. **NEEDS-RESTATEMENT — Attribution coverage.** Higher-quadrature checks omit selectable \(\Delta t_0/2\) and \(10\Delta t_0\) arms. Full-mesh anchor differences cannot replace independent reference-index checks, which remain absent. **Fix:** replay every selected arm at higher quadrature regardless of timestep; add reference/case/time-index mutations and independently reconstruct metrics. Label quadrature differences as sensitivity, not certified quadrature error.

23. **RESOLVED — Local smoke.** A1.12 withdraws the oversized local GPU allocation and moves the substantive smoke to the cluster.