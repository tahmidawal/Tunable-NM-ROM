1. **NOTE — The freezing equation and displayed residual have the correct sign.** With \(y=x-c(t)\),
   \[
   u_t=v_t-\dot c\cdot\nabla v,
   \]
   so \(v_t=\dot c\cdot\nabla v+\mathcal P[N(v)]+\nu\Delta v\). Moving the right-hand side into the residual gives **minus** \(\sum_d\delta_dD_d\bar a\), exactly as displayed. Setting \(\delta=0\) recovers the linear-bank residual in [ns3d_rom.make_run](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-22-ns3d-shift-decoder/experiments/ns3d/ns3d_rom.py:95). The periodic Leray projector, spectral cutoff, derivatives and Laplacian commute with translation. Strict dealiasing preserves translation equivariance of the retained quadratic nonlinearity. Moreover,
   \[
   \langle\phi_m,\Delta v\rangle=-\lambda_m\langle\phi_m,v\rangle,
   \]
   so the diffusion term remains valid in the moving frame.

2. **SHOULD-FIX — The increment formulation is second order, but “algebraically identical” overstates discrete equivalence.** For a smooth frame,
   \[
   c^{n+1}-c^n=\Delta t\,\dot c(t_{n+1/2})+O(\Delta t^3).
   \]
   Multiplying that increment by \(\nabla G\bar a\) gives a consistent second-order midpoint approximation. **There is no inherent order loss from solving for \(\delta\).** This assumes a consistent gauge, sufficiently accurate nonlinear solves and an unwrapped frame. However, discretizing the freezing equation is not exactly equivalent to transforming an already-discretized laboratory-frame update. The existing weak ROM uses implicit midpoint for its quadratic nonlinearity; the FOM uses CNAB2. Require a timestep-refinement check, not trajectory parity with CNAB2.

3. **SHOULD-FIX — The proposed phase condition is mathematically legitimate, but a weighted row does not enforce it exactly.** It is
   \[
   \langle\partial_d v^n,v^{n+1}-v^n\rangle=0,
   \]
   a discrete version of the usual freezing orthogonality condition \(\langle\partial_dv,v_t\rangle=0\). For periodic spectral derivatives, \(S_d^\mathsf T=-S_d\), hence
   \[
   (a^n)^\mathsf TS_d\Delta a=\bar a^\mathsf TS_d\Delta a.
   \]
   Thus using \(a^n\) here does **not** introduce a first-order lag.

   But finite \(w_g\) makes this a penalty competing with the PDE residual. Small weights permit gauge drift; large weights damage conditioning and can make the normalized-gradient stopping test misleading. Prefer an **exact linear constraint** \(B_n\Delta a=0\), with rows \(B_{n,d}=(a^n)^\mathsf TS_d\), imposed through a rank-revealing nullspace parametrization. Report the constraint rank and violation. A fixed-template phase is another option. If the intended frame must remain centroid-centered, use an explicit centroid phase instead; the existing coefficient centroid matrices permit this without a full-grid centroid calculation.

4. **BLOCKER — The design confuses frame convention with physical identifiability.** A finite centered bank is generally **not translation invariant**. Exact coefficient/shift redundancy exists only insofar as translation tangents can be represented inside its span. B1 can therefore acquire apparent identifiability from truncation error, while B2’s phase rows can be weak or redundant.

   Nonzero shift columns are insufficient: they may lie almost entirely in the coefficient-column span. Examine the scaled shift sensitivity remaining after projecting out coefficient sensitivity, as well as the full Jacobian spectrum. Separate physical-residual conditioning from conditioning supplied by gauge rows.

   **Delete the requirement that the solved frame follow the true centroid.** Orthogonality freezing does not define a centroid frame. For a reconstructed field,
   \[
   \operatorname{centroid}(u_{\rm ROM})
   =c+\operatorname{centroid}(Ga)\pmod 1.
   \]
   Compare that quantity with the truth centroid as a diagnostic. A different \(c(t)\), or an LM failure, does not establish non-identifiability.

5. **SHOULD-FIX — Existing LM damping is not a sufficient robustness argument.** [make_lm](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-22-ns3d-shift-decoder/experiments/ns2d/ns2d_rom.py:220) actually solves
   \[
   \bigl[J^\mathsf TJ+\lambda\,\operatorname{diag}(\operatorname{diag}(J^\mathsf TJ)+10^{-30})\bigr]s
   =-J^\mathsf Tr.
   \]
   Positive damping helps correlated columns, but normal equations square conditioning; the tiny diagonal floor does not meaningfully normalize a nearly invisible shift direction. `jacfwd` differentiates the residual—it does not fix this. The default trust radius is infinite.

   Scale coefficients, shifts and phase rows explicitly. Record per-step iterations, rejection/stopping reasons, physical residual, phase violation and frame increments. Budget exhaustion, tiny steps and rejected solves must not silently count as convergence. Use a rank-revealing QR/SVD reference on troublesome development states. An analytic Jacobian is particularly straightforward for this quadratic/bilinear residual and should be checked against AD.

6. **BLOCKER — B0 is not the failure control described.** Every solved arm initializes with \(c^0=\operatorname{centroid}(u_0)\). Therefore B0 holds an **initially centered frame fixed**; it does not use the centered bank at the laboratory origin. It has already removed the large initial translation responsible for much of the fixed-span problem.

   Also, diag01’s **53.860%** is the **plain, unaligned POD** evolved projection floor—not the oracle-centered POD floor. That number cannot establish that B0 must exceed 5%. B0 might pass at some rank or timestep without any harness error. Keep it as an ablation and report its outcome. Use independently validated sign/shift/reconstruction mutations to test whether the measurement harness detects an actual fault.

7. **BLOCKER — D is not an accuracy ceiling, and beating it is not grounds for retraction.** A truth-centroid-driven frame fixes one particular coordinate convention. Its truncated rollout still has projection, temporal and optimization errors. An online frame can produce lower field error by choosing a better frame for the finite bank. Even A1 is a projection floor **conditional on the centroid shift**, not the globally best approximation over all shifts and coefficients.

   D is safe to report **only as a clearly labeled oracle-assisted diagnostic**, excluded from online accuracy/speed claims. Isolate its truth-path inputs from B0/B1/B2/C. Specify interpolation: six saved centroid samples are not a resolved frame path for every reduced step. Unwrap the circular coordinates before interpolation and take signed increments. `diag_floor.torus_delta` returns unsigned distances and is unsuitable for frame integration.

8. **NOTE — Computing the initial centroid from the given \(u_0\) is legitimate.** It uses query input, not future truth. Likewise, centering training snapshots with their training-time centroids is legitimate offline preparation. Include the initial centroid, shift and projection in query timing. Check that the circular first moments defining the centroid are not nearly zero. Keep units explicit: the frame is in unit-domain coordinates, while `fourier_shift` accepts grid-sample offsets, requiring multiplication by \(N\).

9. **SHOULD-FIX — The proposed checks do not yet validate the moving-frame dynamics.** Checking linearity in \(\delta\) against a JVP of the same residual can pass with the wrong sign or units. Add independent checks of:

   - The complete tensor residual against a full-grid moving-frame residual.
   - Exact reduction to the existing weak residual when \(\delta=0\).
   - A manufactured moving-frame solution with prescribed smooth translation, testing sign and temporal order.
   - Fractional-translation equivariance of complete queries, including a torus-boundary crossing.
   - Periodic skew-adjointness of \(S_d\), finite increments and gauge violations.

   Use complete Fourier sine/cosine pairs so translating the test functions preserves their span and residual norm. Save \(a\), \(c\) and solver diagnostics alongside fields; otherwise an independent auditor cannot determine whether the reported fields came from the claimed solved trajectory.

10. **SHOULD-FIX — Choose \(M\) from observability as well as equation count.** For \(r=64\), I would start with **\(M=292\)**: complete polarization/sine/cosine groups through the full shell \(|k|^2\le10\), roughly four times the 67 unknowns. \(M=256\) is a reasonable cheaper pilot with complete pairs, but cuts a shell. Neither number guarantees that the smooth tests adequately observe this centered bank.

    Inspect singular values of the weighted \(A\) and augmented Jacobian on representative states, then confirm accuracy with a larger test space. A small residual in poorly observed modes can fake solver success. Do not inherit \(M=2048\) automatically: that risks making a small-rank speed experiment unnecessarily expensive.

11. **SHOULD-FIX — Removing transport is valuable; “costs nothing per step” is not credible.** The explicit extra contraction costs only \(O(3Mr)\), small beside \(O(Mr^2)\) quadratic work. Increasing the solve dimension from 64 to 67 is also modest. But those facts say nothing about changes in iteration count, conditioning or rejected trials.

    The whole-trajectory JIT removes repeated Python dispatch. It does **not** remove tensor contractions, Jacobian evaluation, normal-matrix formation or dense factorizations inside the LM loop. The likely dominant cost is **repeated LM work**: tensor/Jacobian work at larger \(M\), and small-system factorization plus device-operation overhead at smaller \(M\). An accepted iteration evaluates a trial residual and then reevaluates the accepted residual/Jacobian. Generic `jacfwd` need not realize the cheapest algebraic contraction automatically.

    Output is also material: the rank-64 real bank occupies about **50 MB**, and each dense reconstruction reads it. Five evolved output shifts, initial centering and initial reconstruction remain grid-sized operations. Put evolved reconstruction outside the inner step scan. The existing tracker computes a laboratory-frame field inside every step before selecting the block’s final field; do not assume compilation removes all that work.

12. **SHOULD-FIX — The speed bar is difficult, but the evidence does not prove it unreachable for every ROM.** At \(\Delta t=0.01\), a 3 ms trajectory leaves roughly **150 μs per step before initialization and output costs**. I would not expect an untuned generic LM implementation with several iterations per step to beat that reliably. A compact tensor implementation, well-scaled gauge and very few iterations—or an accurate larger timestep—could still do so.

    My assessment is: **speed is unproven and high risk, not mathematically impossible at \(N=32\)**. The old 597.826 ms weak-ROM measurement used full-grid nonlinear evaluation with \(M=2048\); it is not a prediction for this stored-tensor implementation. Conversely, eliminating the tracker’s phase sum does not establish a speedup because this proposal introduces a different nonlinear solve. Require an early complete-query timing gate.

13. **SHOULD-FIX — Tighten the timing contract before reusing the harness.** “Same job” is weaker than “same solver invocation.” The old diagnostic harness sometimes times one call and obtains accuracy from another; do not copy that pattern. Compute errors from retained timed outputs and persist timing arrays by case and repetition.

    Burn in before each timed block, interleave paired methods, include all query initialization/output work, and separate compilation from steady-state execution. Profile the compiled trajectory; separately dispatched microbenchmarks do not add up faithfully to its runtime. Distinguish a ratio of medians from a median of paired ratios. If the ROM becomes more accurate than every listed coarse FOM comparator, include a finer eligible FOM rather than leaving comparator selection undefined.

14. **BLOCKER — The multi-structure stop rule draws the wrong conclusion.** If the single-shift A1 projection already meets 5%, then a successful single-frame representation exists at the sampled times. Failure of its rollout points toward reduced dynamics, gauge choice, temporal discretization, test-space observability or optimization. It does **not** localize failure to needing several independently moving structures.

    Multiple frames also introduce relative-shift-dependent nonlinear cross-interactions; the same constant tensor is no longer sufficient. Remove this automatic escalation from the six-job experiment. It is a separate design question.

15. **SHOULD-FIX — Clarify the architecture claim and stage the budget.** Arm F measures a learned bank’s representation, while the proposed solved unknown is a free coefficient vector \(a\). That does not demonstrate a nonlinear-head rollout \(a=h_\theta(z)\). State which claim is being tested. If using the raw learned bank, \(G^\mathsf Tu\) is not its orthogonal-projection coefficient vector; whiten it or use its QR factors as the existing machinery does.

    Freeze a small development ladder for rank, timestep, \(M\), solver tolerances and gauge treatment before submission. First establish centered-POD dynamics and complete-query cost. Make network training conditional on that result, and reserve a job for independent confirmation before the sealed draw. Do not spend the budget rediscovering already measured high-rank floors or pursuing multiple frames after an unexplained solver failure.

**Disposition:** Do not submit the design unchanged. The freezing equation, residual sign, diffusion treatment and midpoint use of the frame increment are sound. The blocking defects are the interpretation of identifiability, the mandatory B0 failure, the oracle-rollout ordering requirement and the multi-structure escalation. Correct those, enforce or explicitly validate the gauge, and make complete-query timing an early gate. The mechanism deserves a bounded development test, but “free translation” and a speed win over a 3 ms FOM are currently hypotheses. No files were modified and no GPU work or cluster jobs were run.