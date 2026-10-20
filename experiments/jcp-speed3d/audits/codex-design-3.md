**READY for implementation and its registered code/smoke gates.** A2 resolves the six Round-2 blockers. I found remaining notation and diagnostic-coverage issues, but no new blocker that permits an unsupported accuracy or speed claim through the amended gates.

This review was strictly read-only: no files—including the lab log—were changed, no git state changed, and no jobs or GPU work ran.

**Round 2: six blocking items**

| Item | Verdict | Assessment |
|---|---|---|
| 1. QN initialization, masks, failure precedence and refresh semantics | **RESOLVED** | A2.1 initializes histories and bookkeeping, masks non-finite candidates, gives failure precedence over convergence, freezes dead members, separates applied refreshes from executed contractions, and recomputes stopping information after refresh. Its “once” rule supersedes A1’s double increment. |
| 2. Executable H1.1 replay | **RESOLVED** | A2.2 defines case selection, step selection, entering damping, saved predictor index, QN-1 initialization, mutually exclusive labels and unsuccessful outcomes. Crucially, an unsuccessful baseline reproduction forbids interpreting interventions. Adaptive-startup coverage remains limited, as discussed below. |
| 3. Large-mesh distance without a mesh Gram | **RESOLVED** | A2.3 explicitly replaces the mesh metric with the scoring-lattice metric, defines normalization and requires per-mesh ADT comparisons. No full-mesh Gram is needed. The displayed matrix multiplication needs a transpose clarification. |
| 4. Streaming mode indexing | **RESOLVED** | A2.4 correctly converts one-based modes and nodes to zero-based array indices. |
| 5. Valid tightest FOM qualification reference | **RESOLVED** | A2.5 requires the tightest run itself to pass finite/stage-convergence checks and makes failure unavailable without fallback. Residual and output-distance normalization are explicit. |
| 6. K9 perturbation origin and normalization | **RESOLVED** | A2.6 anchors both perturbations to the actual ADT comparator, fixes the seed/configuration, normalizes in the registered metric, and verifies constructed distances before classification. |

**Round 1: twelve required amendments**

| Requirement | Verdict | Reason |
|---|---|---|
| 1. Correct contraction/startup accounting and qualify dominance | **RESOLVED** | A1.1 separates arithmetic, contractions, iterations and measured time; dominance is a hypothesis requiring profiling. |
| 2. Fully specify QN; remove convergence guarantee | **RESOLVED** | A1.2 plus A2.1 define the relevant branches. Agreement with ADT is empirical, subject to independent certification and output-distance gates. |
| 3. Predictor startup/history and exact G1 fallback | **RESOLVED** | Current-stage coefficients, history initialization, activation masks and static original-code fallback are specified. A2.7 withdraws the incorrect “full history” wording. |
| 4. Eligibility attribution, certification and references | **RESOLVED** | The pooled stationarity requirement is disclosed as new; comparator eligibility is recomputed; finest references and reference eligibility are required; ADT sensitivity checks extend to finalists and large meshes. |
| 5. Direct same-job timing | **RESOLVED** | Screening ratios cannot establish the final speed-up. Direct comparator/winner confirmation is required; Route 4’s smaller panel is explicitly identified. |
| 6. H1.1, unavailable outcomes and selection ties | **RESOLVED** | Replay interpretation is guarded; rescue/unavailable outcomes and screening tie rules are defined. |
| 7. Define and rename decomposition | **RESOLVED** | Factors are descriptive rather than causal. A2.7 correctly disclaims multiplication of separately aggregated medians into the total. |
| 8. Mesh-free qualification and batching controls | **RESOLVED** | Same-initialization ADT, per-case error protection, valid full-rank K5, wrong-viscosity association and positive permutation control are specified. |
| 9. FOM ladder, matching, qualification, PS and large meshes | **RESOLVED** | Missing ladder points, diagnostic versus matched scope, pure-JAX PS port, tolerance qualification, streaming tables and feasibility gates are addressed. |
| 10. Discriminating controls and fixed cases | **RESOLVED** | K9 now tests the actual distance threshold correctly; K10 rejects invalid/inaccurate cheap entries; fixed cases and claim-blocking control failures prevent case hunting. |
| 11. Freeze before replication | **RESOLVED** | A1.11, explicitly reaffirmed by A2.7, places the complete immutable selection manifest before replication. |
| 12. Timing boundaries, inventory, resources and finite-ladder limits | **RESOLVED** | Prepared-input latency is explicit; preparation/setup costs remain visible; inventories and pilot projections are required; finite-ladder speed-up limitations are disclosed. |

**Adversarial checks of A1 and A2**

1. **QN refresh and batch semantics — CORRECT, with the amendments applied together.**

   The refresh schedule measures elapsed steps from the most recent refresh, including emergency refreshes. That is coherent; it is not an absolute modulo schedule.

   A2 distinguishes:
   
   - refreshes applied to a member;
   - contractions executed for the whole batch;
   - iterations spent attempting a step.

   Counting an emergency refresh at the final iteration, despite performing no subsequent solve, correctly records work actually performed. Recomputing `gam` and `done` after that refresh prevents stale stopping records.

   Failed members cannot become eligible merely because their last finite state is retained: reason 3 persists and E-a rejects the run. Non-finite trial residuals can be rejected during backtracking without requiring the accepted state to become invalid.

   The stationary-point statement should be understood as applying to the ideal update with a nonsingular preconditioner—not to every unchanged output of a finite-budget, rejection-based algorithm. The independent gates protect against that distinction becoming an accuracy claim.

2. **IMEX algebra, BE/CN/BDF2 and first steps — CORRECT.**

   Let \(D=\nu\lambda\). Since
   \[
   S(a_0+\Delta t\,b_0D)=1,
   \]
   the predictor’s coefficient matrix is exactly \(A\). Its least-squares right-hand sides are
   \[
   \begin{aligned}
   \mathrm{BE}:&\quad S(Ac_n-\Delta t\,\widetilde{\mathrm{adv}}),\\
   \mathrm{CN}:&\quad S\left(Ac_n-\frac{\Delta t}{2}
        [F_n+\widetilde{\mathrm{adv}}]\right),\\
   \mathrm{BDF2}:&\quad S\left(2Ac_n-\frac12Ac_{n-1}
        -\Delta t\,\widetilde{\mathrm{adv}}\right).
   \end{aligned}
   \]

   CN includes the previous diffusion contribution through \(F_n\). BDF2’s first step uses BE coefficients and `adv_n`; CN’s first predictor retains CN coefficients with `adv_n`. These are predictor definitions, not independent claims of second-order integration.

   Small CPU algebra checks gave maximum discrepancies of \(8.9\times10^{-16}\). The original disabled-candidate path remains essential for G1’s bitwise requirement.

3. **H1.1 replay coverage — NEEDS-RESTATEMENT, non-blocking because interpretation is guarded.**

   [The inherited solver](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/fs3.py:146) uses adaptive LM during startup. At \(2.5\Delta t_0\), startup lasts two steps for M2_512 and three for the wide settings.

   A2.2 can select one of those steps, but its baseline is always “K = 2.” A fixed two-sweep replay generally does **not** reproduce an adaptive-startup step. Such a case will therefore become “replay not reproducible,” even with perfectly reconstructed history.

   This does not create false attribution because A2 explicitly forbids interpretation after reproduction failure. It does limit H1.1’s coverage. Distinguish unsupported adaptive-startup replay from unexpected reproduction failure.

   Also, a distance-only case has no certified non-stationary step by construction. If its baseline replay is already stationary, intervention success does not demonstrate a rescue. Report that baseline status alongside the intervention labels.

4. **H3.2 qualification and controls — CORRECT.**

   \[
   12^3=1728\ge1024.
   \]

   Thus gl12 no longer has the unavoidable row-count rank deficiency of lat256. This permits full column rank; it does not establish numerical rank or conditioning. A2.7 correctly requires those checks before K5 counts.

   H3.2 establishes the registered aggregate and per-case accuracy allowances, plus solver agreement with identically initialized ADT. It does **not** establish identical trajectories to mesh initialization.

   K6’s incorrect viscosity association and K6+’s whole-query permutation now test different, appropriate properties.

5. **FOM ladder and tolerance qualification — CORRECT as a prospective protocol.**

   The tightest calibration run must itself converge. A looser level must both converge and agree with it; production stage failures invalidate the configuration.

   The recorded residual is relative to the entering stage state, matching [fomcore.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/fomcore.py:139). The lattice-output comparison uses the initial-field norm.

   Qualification on four cases is a calibration check, not evidence that tolerance sensitivity is bounded on every production case. The design now makes a defensible calibrated claim rather than silently treating failed solves as matching candidates.

   The all-case matched ladder and eight-case large-mesh diagnostics remain distinct. The finite-ladder upper-bound direction is correct.

6. **Streaming tables: normalization, node indexing and gathering — CORRECT.**

   [The inherited builder](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/vendor/quad3d/vendor/burgers3d-retry/tables.py:64) computes \(A\) using `common.phiT`, which applies three orthonormal DST-I transforms. Therefore
   \[
   A_{m,i}=
   \sum_{a,b,c=1}^{n-2}
   \left(\frac{2}{n-1}\right)^{3/2}
   \sin\frac{\pi k_{x,m}a}{n-1}
   \sin\frac{\pi k_{y,m}b}{n-1}
   \sin\frac{\pi k_{z,m}c}{n-1}
   \widehat G_i\!\left(\frac{a,b,c}{n-1}\right).
   \]

   A1’s separable accumulation with A2’s gather computes precisely this expression:

   ```text
   T[kx - 1, ky - 1, kz - 1, :]
   ```

   There is **no extra cell-volume factor**. Nodes begin at one, and the sine denominator is \(n-1\).

   A small CPU comparison using the extracted `phiT` implementation, with NumPy executing its array operations, gave relative discrepancy \(1.45\times10^{-15}\).

   The algorithm avoids an \(N\times R'\) bank allocation. It still needs slab/intermediate storage and full-grid mode-ordering work; the registered feasibility pilot remains necessary.

7. **A2.3 lattice-distance formula — NEEDS-RESTATEMENT, notation only.**

   A2 calls \(G_{\rm lat}\) the bank **rows** at lattice points, which conventionally gives shape \(63^3\times R'\). With coefficient rows \(W_k\), decoding is then
   \[
   (W_a-W_b)_kG_{\rm lat}^{\mathsf T},
   \]
   not \((W_a-W_b)_kG_{\rm lat}\).

   Alternatively, define \(G_{\rm lat}\) explicitly as the transposed bank. The intended metric is otherwise unambiguous and equals the inherited Gram metric at \(n=65\), using the same initial-field normalization.

8. **K9 and freeze/replication order — CORRECT.**

   K9’s comparator-centered construction removes the earlier \(0.9\tau+0.5\tau\) counterexample. Verifying the constructed distances before classification also separates perturbation-generation errors from classifier errors.

   The final freeze follows mesh-free qualification, confirmation panels and FOM matching. Replication cannot select replacements or alter qualifications without becoming explicitly post hoc.

All **14 files listed in `INHERITED.json` matched their recorded hashes**.

**Required blocking changes:** None.

**Non-blocking suggestions:**

- Clarify adaptive-startup replay coverage and report “baseline already stationary” for distance-only diagnoses.
- Correct the \(G_{\rm lat}\) orientation in A2.3.
- Use an explicit mixed absolute-relative tolerance for replayed \(\gamma\) near zero; the current rejection guard is safe but may reject useful replays.
- Consolidate A1/A2’s QN overrides into one implementation checklist before coding, especially the refresh counters and bad/dead masks.

**Final verdict: READY.**