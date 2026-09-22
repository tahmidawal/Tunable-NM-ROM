1. **NOTE — The freezing sign and proposed residual are correct.** With \(y=x-c(t)\),
   \[
   u_t=v_t-\dot c\cdot\nabla v,
   \]
   so \(v_t=\dot c\cdot\nabla v+\mathcal P[N(v)]+\nu\Delta v\). Moving everything to the residual gives the **negative** transport term \(-\sum_d\delta_dD_d\bar a\), exactly as written. Setting \(\delta=0\) reproduces the residual in `ns3d_rom.make_run`. This is implicit midpoint for the nonlinear term, with Crank–Nicolson diffusion; it is **not** the FOM’s CNAB2 discretization.

   Using \(\delta=c^{n+1}-c^n=\Delta t\,\dot c_{\mathrm{mid}}\) does **not** inherently lose an order. For smooth trajectories, \(\delta\nabla\bar v\) approximates the integrated transport term with local error \(O(\Delta t^3)\). Solver errors, an inconsistent phase update, or an inaccurate prescribed centroid path could spoil that accuracy.

2. **NOTE — Leray commutation and the diffusion expression are valid under the existing spectral assumptions.** Translation, spectral differentiation, the cutoff, and the Leray projector commute as Fourier multipliers. The retained dealiased nonlinear operator is translation equivariant. Solenoidal tests remove the need to apply Leray explicitly inside the tensor contraction. Since \(\Delta\phi_m=-\lambda_m\phi_m\),
   \[
   \Phi^{\mathsf T}\Delta G=-\operatorname{diag}(\lambda)A.
   \]
   Thus \(-\nu\lambda A\bar a\) is correct even though the bank itself is not a Laplacian eigenbasis. Preserve the projected, band-limited bank and use **spectral derivatives of that bank** for \(D_d\) and \(S_d\).

3. **SHOULD-FIX — “Algebraically identical” overstates the discrete equivalence.** The continuous substitution is exact. Applying midpoint after changing coordinates is not generally identical to applying a laboratory-frame midpoint residual directly to \(G(x-c)a\).

   Also, fixed tests in moving coordinates correspond to translated tests in laboratory coordinates. Their least-squares norms agree under translations when the test space contains complete sine/cosine pairs with matching weights. `test_modes` can stop midway through a pair. Require complete pairs—multiples of four give complete wave/polarization groups—and make the coordinate convention explicit. The diffusion weighting preserves the pairing because each pair has the same \(\lambda\).

4. **BLOCKER — The design conflates a gauge with an identifiable physical centroid.** Translation is an exact representation redundancy only when the relevant translation tangents lie inside the bank span. For a finite centered POD bank this need not hold: \(\partial_dGa\) can have substantial components outside \(\operatorname{span}(G)\). The degeneracy may therefore be approximate, directional, or absent.

   More seriously, an orthogonality-gauge frame need not equal the energy-centroid frame. For the reconstructed field,
   \[
   \operatorname{centroid}(u)=c+\operatorname{centroid}(Ga)\pmod 1.
   \]
   Stop rule 2 would reject a valid solution merely because it uses a different frame. Compare reconstructed physical fields and their centroids; only compare \(c\) directly with truth centroids if the gauge explicitly centers \(Ga\). Failure to converge also does not, by itself, prove non-identifiability.

5. **SHOULD-FIX — The proposed phase row is mathematically sensible, but enforce and diagnose it properly.** It is precisely
   \[
   \langle\partial_dv^n,v^{n+1}-v^n\rangle=0,
   \]
   a discrete version of the standard orthogonality phase condition. There is a useful stronger fact here: with periodic spectral differentiation, \(S_d^{\mathsf T}=-S_d\), so
   \[
   (a^n)^{\mathsf T}S_d\Delta a
   =\bar a^{\mathsf T}S_d\Delta a.
   \]
   Consequently, using \(a^n\) does not introduce the obvious first-order lag one might suspect.

   Appending weighted rows is nevertheless a **penalty**, not an exact constraint. Small \(w_g\) permits phase drift; large \(w_g\) worsens conditioning and can distort stopping diagnostics. Prefer equality-constrained least squares or null-space elimination of the three linear constraints, provided their rank is three.

   For this particular centroid-trained bank, another reasonable gauge is \(a^{\mathsf T}Q_{\sin,d}a=0\), with positive cosine moments selecting the origin branch. `coeff_shift.centroid_matrices` already supplies the necessary quadratic forms. This aligns the online frame with training and avoids per-step field reconstruction. Monitor moment magnitudes: a circular centroid becomes ill-defined when those moments vanish.

6. **BLOCKER — The identifiability test can report a manufactured pass.** Nonzero shift columns do not establish independence from coefficient updates. Gauge rows can improve an augmented singular-value ratio simply through their weight.

   Report separately the appropriately scaled physical blocks \(J_a,J_\delta\), and the singular values of
   \[
   (I-J_aJ_a^\dagger)J_\delta.
   \]
   This measures shift information that coefficient changes cannot absorb. Also report the constrained-system conditioning and translation-tangent rank. Define numerical thresholds and coefficient/shift scaling before running. A good spectrum after adding strong gauge rows establishes a well-conditioned **chosen frame**, not unique frame recovery from the PDE alone.

7. **BLOCKER — Both mandatory “must fail/must win” controls are invalid.** B0 uses \(c^0=\operatorname{centroid}(u_0)\), then holds that frame fixed. It is therefore an **initially centered fixed-frame ROM**, not the centered bank used unshifted in the laboratory frame. It could legitimately meet 5%, especially as rank increases. Its success would weaken the need for online translation; it would not prove a broken harness.

   D is not a guaranteed accuracy ceiling either. Prescribing the true centroid path does not guarantee the best finite-bank dynamics or numerical trajectory. A solved frame can outperform it without leakage. Likewise, A1 is a projection floor **conditional on the prescribed centroid shifts**, not the global optimum over all shifts or all learned banks.

   Replace these retraction rules with diagnostics. A valid projection inequality is: at the **same predicted shift**, orthogonal projection of truth into that shifted bank cannot be worse than the ROM field in that bank.

8. **NOTE — Using the given \(u_0\)’s centroid is legitimate.** The initial field is an authorized query input. Computing its centroid and projecting its centered version uses no future truth. Include those operations in query timing.

   Keep future truth and oracle paths outside the B0/B1/B2/C runner interfaces. Check that changing or removing evaluator-side future truth cannot change their predictions. Oracle D needs centroid increments at every ROM step; the six saved output frames do not provide these directly. Specify how D obtains its diagnostic path rather than silently interpolating sparse centroids.

9. **SHOULD-FIX — Initialization and shift units need explicit contracts.** \(a^0=G^{\mathsf T}v_0\) is correct only for an orthonormal bank. The learned bank is not generally orthonormal; `ns3d_model.whiten` and the QR-based initialization in `make_run` address precisely this. Whiten consistently or solve the appropriate projection problem.

   The reusable shift helpers take **grid samples**, while \(c,\delta\) are unit-torus coordinates: initialization needs `-c0*n`, reconstruction needs `+c*n`. Use signed, unwrapped increments across periodic boundaries; `torus_delta` returns an unsigned distance and cannot supply transport increments. Accumulate \(c\) separately from the per-step unknown \(\delta\).

10. **SHOULD-FIX — The current checks can validate the same wrong formula twice.** Testing delta-linearity against a JVP of the same residual will not catch a shared sign error. Add independent checks for the complete residual and Jacobian, including
    \[
    J_\delta[:,d]=-\frac{D_d\bar a}{1+\Delta t\,\nu\lambda/2},
    \]
    and the coefficient-Jacobian contribution from the bilinear transport term.

    Check fractional-shift equivariance, spectral derivative signs, \(S_d\)’s skew symmetry, and a manufactured translating field. Verify timestep refinement with tight solves. Record physical residuals, phase violations, accepted/rejected steps, iterations, termination reasons, and finite-field checks. The existing LM permits budget, tiny-step, and rejection exits; returning a trajectory does not establish convergence.

11. **SHOULD-FIX — “Costs nothing” is false; a small incremental arithmetic cost is plausible.** The transport evaluation costs \(O(3Mr)\), modest beside \(O(Mr^2)\) advection. But it also changes the coefficient Jacobian, adds three unknowns, and can substantially increase LM iterations.

    The actual `make_lm` uses `jacfwd`, forms \(J^{\mathsf T}J\), performs a dense solve, evaluates trial residuals, and refreshes the Jacobian after acceptance. Repeated contractions, normal-equation formation, small dense solves, and GPU kernel launches are likely to dominate evolution. An analytic Jacobian and elimination of the linear-in-\(\delta\) subproblem are worth considering before expensive solver tuning.

    Full queries still require initial centering/projection and six laboratory-frame outputs. Stored POD output requires bank expansion plus Fourier shifts. In fact, `coeff_shift.make_coeff_run` expresses laboratory-field reconstruction inside every step and retains only the last frame of each block; inspect the compiled execution before attributing its entire cost to phase reprojection.

12. **SHOULD-FIX — Be pessimistic about beating the FOM, but do not claim impossibility.** At \(\Delta t=0.01\), twenty steps must fit inside roughly 3 ms **including initialization and outputs**: less than 150 microseconds per step before those other costs. Several generic LM iterations per step make that a difficult target.

    My expectation is that a straightforward extension of this LM machinery will lose at \(N=32\). I would not spend six jobs assuming speed will emerge. But the evidence does not show that **any** ROM is unable to win: small \(M\), few iterations, larger accurate timesteps, and specialized solves could change the outcome. The historical 597.8 ms weak-query result used full-grid nonlinear evaluation through `dense_n=n`; it is not a timing bound for the proposed tensor implementation. Use one early paired profile to decide whether speed remains credible.

13. **BLOCKER — The multi-structure trigger diagnoses the opposite of what the evidence says.** If A1 already represents the snapshots below 5% using one shift, a solved trajectory above 5% does **not** localize failure to the single-frame representation. It points toward dynamics, gauge mismatch, weak-space truncation, discretization, initialization, or optimization. Adding independently shifted structures then introduces relative-shift-dependent interaction terms and additional ambiguities. It is a separate design, not a justified fallback under the stated trigger.

14. **SHOULD-FIX — Bound the first job and repair the measurement contract.** Specify the development ladders for rank, \(M\), timestep, gauge, and LM budget now; require \(M\) comfortably above \(r+3\). Compare selected solutions against a richer test space so neglected modes cannot silently invalidate residual-based claims. Reserve training arm F until the POD mechanism has earned further investigation.

    Timing needs GPU burn-in before each block, retained per-case repetition arrays, and errors from the actual timed invocations. `diag_floor.time_calls` discards outputs, and its FOM accuracy example comes from a separate case-0 call; that harness cannot be reused unchanged for the stated acceptance contract. Time complete queries, keep diagnostic SVDs outside them, and use the same output contract for FOM and ROM.

    Finally, correct the opening evidence statement: 53.860% is the **uncentered training POD** result; 0.128% uses a separately fitted centered POD. They are not the same bank with and without an evaluation shift.

**Disposition:** Do not submit the design unchanged. The freezing equation is sound and deserves a bounded development pilot, but the current gauge interpretation, oracle ordering, mandatory-control failures, and multi-structure trigger can reject valid results or misdiagnose failures. Fix those gates, specify the numerical contract, and make the first paired profile decisive. Accuracy is plausible; speed against this very cheap \(32^3\) FOM is doubtful with the existing generic LM implementation, though not ruled out. This audit made no file changes and ran no GPU work.