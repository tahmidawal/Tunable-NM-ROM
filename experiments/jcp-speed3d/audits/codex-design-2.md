**NOT READY.** A1 resolves much of Round 1, including the IMEX algebra, direct timing comparisons, reference handling, and freeze order. Remaining blockers concern executable solver semantics, replay definitions, distance controls, and large-mesh certification.

This review was read-only. I changed no files or git state and submitted no jobs. Small numerical checks ran on CPU only.

The table below follows the **12 required changes at the end of Round 1**, whose numbering differs from A1’s numbered responses.

| Round-1 requirement | Status | Assessment |
|---|---|---|
| 1. Correct contraction/startup accounting and qualify dominance | **RESOLVED** | A1.1 distinguishes contractions, iterations, arithmetic and measured time. Jacobian dominance is now an explicitly unverified hypothesis. |
| 2. Fully specify QN; remove convergence guarantee | **PARTLY** | The convergence claim is appropriately weakened, but refresh, non-finite handling and counters still have defects described below. |
| 3. Specify predictor startup/history and G1 fallback | **PARTLY** | The predictor algebra, activation indices and static original-code fallback are correct. QN’s initial history values and non-finite candidate masking remain unspecified. |
| 4. Correct eligibility attribution, certification and reference handling | **RESOLVED** | A1.4 correctly identifies the new pooled-stationarity requirement, recomputes comparator eligibility, adds ADT verification and pins eligible finest-field references. |
| 5. Require direct same-job timing | **RESOLVED** | A1.5 fixes the common-baseline inference problem. Route 4 also uses direct pairing, although its smaller panel needs an explicit exception to A1.5’s universal wording. |
| 6. Repair H1.1 and define unavailable outcomes/selection ties | **PARTLY** | Rescue/unavailable outcomes and screening rules are specified. H1.1 still mixes pooled and per-case eligibility, leaves some replay targets undefined, and permits overlapping attribution labels. |
| 7. Define and rename decomposition | **RESOLVED** | The factors are now descriptive, with bank/rule/step changes acknowledged. See the aggregation clarification below. |
| 8. Repair mesh-free qualification and batching controls | **RESOLVED** | Same-initialization ADT, per-case accuracy protection, wrong-viscosity semantics and permutation control repair the substantive issues. gl12 passes the necessary row-count check; actual numerical rank remains to be checked. |
| 9. Resolve FOM ladder, matching, qualification, PS and large meshes | **PARTLY** | Most specifications are repaired. The streaming transform is mathematically correct, but array indexing and the large-mesh solver-distance calculation remain incomplete. The tightest FOM qualification run must itself be verified. |
| 10. Add discriminating controls and fixed cases | **PARTLY** | Fixed cases and K10 improve the design. K9’s perturbation origin is ambiguous and can invalidate its expected classification. |
| 11. Freeze all decisions before replication | **RESOLVED** | A1.11 correctly places replication after all validation selections and calls the historical cohort replication. |
| 12. Register timing boundaries, inventory, resources and finite-ladder limits | **RESOLVED** | These are now prospective requirements. The resource estimate remains a planning estimate until the required certificates exist. |

**Adversarial findings on A1**

1. **QN refresh bookkeeping — WRONG / NEEDS-RESTATEMENT.**

   In [A1.2](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/DESIGN.md:266), the emergency branch says “refresh … as above” and then increments `n_full` again. The referenced refresh already increments it. A literal implementation double-counts emergency contractions.

   Refresh also changes `Jn`, but the emergency branch does not recompute `gam` or `done`. Consequently the next iteration and potentially the final recorded ratio use stale stopping information. For example, a refresh that increases `Jn` enough to satisfy the stopping threshold still triggers another iteration.

   Define a single refresh operation with explicit outputs and exactly one counter increment. Recompute the stopping ratio immediately afterward. Specify whether a refresh at the iteration limit is performed and counted even though no subsequent solve occurs.

   The schedule `k-last_ref >= k_ref`, with emergencies resetting `last_ref`, is otherwise coherent: periodic refresh is measured from the most recent refresh, rather than an absolute modulo schedule.

2. **QN initialization, masks and failure precedence — NEEDS-RESTATEMENT.**

   Missing pieces materially affect the promised batch/single equivalence:

   - Initialize all coefficient/advection histories and refresh bookkeeping. Masking P1–P3 does not protect the unconditional `a2*A*c_{n-1}` at the first step: zero times an undefined/NaN history is still invalid.
   - Mask **non-finite active candidates**, not only inactive candidates. Specify rejection when every candidate is invalid.
   - Define `finite` over the state, gradient, residual, norm and factor. Give non-finite failure precedence over `done`; the current final assignment selects reason 4 first, despite the subsequent statement that a non-finite factor means reason 3.
   - In batch execution, freeze completed/failed members’ states, damping and counters while other members continue. Refresh flags must be intersected with the appropriate active-member mask.
   - Distinguish refreshes applied to a member from contractions actually computed. A batched refresh computes unflagged members’ contractions too; counting only flagged updates understates executed work.

   The per-member specification is a useful starting point, but does not yet fully determine these branches.

3. **IMEX predictor algebra and startup — CORRECT.**

   Writing \(D=\nu\lambda\), the predictor residual is exactly
   \[
   A c+S\bigl(\mathrm{hist}+\Delta t\,b_0\widetilde{\mathrm{adv}}\bigr),
   \]
   because \(S(a_0+\Delta t\,b_0D)=1\). Thus the thin QR of \(A\) is the appropriate solve, assuming full column rank.

   Explicitly, the right-hand sides of the resulting least-squares problems are:

   \[
   \begin{aligned}
   \mathrm{BE}:&\quad S(Ac_n-\Delta t\,\widetilde{\mathrm{adv}}),\\
   \mathrm{CN}:&\quad S\!\left(Ac_n-\tfrac{\Delta t}{2}
       [F_n+\widetilde{\mathrm{adv}}]\right),\\
   \mathrm{BDF2}:&\quad S\!\left(2Ac_n-\tfrac12Ac_{n-1}
       -\Delta t\,\widetilde{\mathrm{adv}}\right).
   \end{aligned}
   \]

   CN retains the full previous \(F_n\), including diffusion. BDF2’s first step correctly uses BE coefficients and `adv_n`. CN’s first step correctly retains CN coefficients while falling back to `adv_n`; that is a predictor, not a claim that the explicit guess itself is second order.

   CPU algebra checks gave maximum discrepancies below \(9\times10^{-16}\).

   One wording correction: FSP does **not always enter its fixed phase with cubic history available**. For M2_512 at \(2.5\Delta t_0\), `a_eff=2`, so its first fixed step has `k=2`. The registered `k>=3` activation handles this correctly; retain that mask and remove “with full history.”

4. **H1.1 replay selection and attribution — NEEDS-RESTATEMENT.**

   [A1.6](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/DESIGN.md:324) still says cases “failing E-b,” although E-b is a pooled configuration-level condition. Define which cases enter diagnosis when the pooled condition fails.

   Additional undefined cases remain:

   - A validity-only failure can have neither a non-stationary step nor an output-distance exceedance.
   - An “output interval” is not a unique step to replay. Choose a precise step or replay the entire specified interval.
   - “Carried damping” must mean damping **entering** the selected step, not its recorded final damping.
   - Recomputed cached Jacobians can change a near-tied predictor choice through rounding. Save the actual starting candidate and require a baseline replay to reproduce the original K=2 result before interpreting interventions.
   - If both trust enlargement and damping reset succeed while ordinary sweeps require \(K>4\), both named attribution conditions hold. Give “mixed” explicit precedence.
   - Define the outcome when no \(K\le12\) succeeds and which QN variant/cache initialization is used.

   These interventions can support statements about the selected replayed step. They do not establish that an entire trajectory’s accuracy failure had that cause.

5. **H3.2 qualification and gl12 — CORRECT, conditional on the registered checks.**

   \(12^3=1728\ge1024\). The previous unavoidable row-rank defect is fixed. This inequality permits full column rank; it does not prove it.

   Same-projection ADT and the added per-case error allowance correctly make this an **accuracy-noninferiority qualification**, not trajectory equivalence. The viscosity-association and whole-query-permutation controls now have the right semantics.

   For K5 to be interpretable, gl12 must pass rank, finite-factor and conditioning checks before an accuracy failure counts. Failure solely through an invalid projection does not satisfy the stated purpose. Its anticipated accuracy failure remains an empirical control outcome, not a mathematical certainty.

6. **Streaming \(A\): normalization — CORRECT; array indexing — NEEDS-RESTATEMENT.**

   The inherited [`common.phiT`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/vendor/quad3d/vendor/burgers3d-span/common.py:117) applies three orthonormal DST-I transforms. Therefore
   \[
   A_{m,i}=
   \sum_{a,b,c=1}^{n-2}
   \left(\frac{2}{n-1}\right)^{3/2}
   \sin\frac{\pi k_{x,m}a}{n-1}
   \sin\frac{\pi k_{y,m}b}{n-1}
   \sin\frac{\pi k_{z,m}c}{n-1}
   \widehat G_i\!\left(\frac{a,b,c}{n-1}\right).
   \]

   A1’s separable contraction computes this expression. **No extra cell-volume factor belongs here.** The node index is one-based, and the denominator is \(n-1\), not \(n-2\).

   A small CPU test against the actual extracted `phiT` implementation gave relative discrepancy \(1.01\times10^{-15}\).

   However, `common.modes` returns **one-based** mode triples. For the declared array shape, executable gathering must use:
   ```text
   T[kx-1, ky-1, kz-1, :]
   ```
   Similarly, the first sine-table column represents mode 1. A1’s literal `T[kx,ky,kz]` would shift modes and potentially access beyond the array.

7. **Large-mesh solver-distance certification — NEEDS-RESTATEMENT.**

   A1.9 explicitly omits a mesh Gram matrix, but §5 still requires the inherited **Gram-metric field distance** to ADT.

   Mesh-free initialization removes the need for the *projection* Gram on the full mesh; it does not automatically provide the metric required for solver-distance certification. The rule Gram \(B^\mathsf TW B\) is not exactly the nodal mesh Gram.

   Specify one executable route: streamed full-mesh differences, a streamed mesh Gram, or an explicitly amended scoring-lattice distance. State the corresponding normalization and use it consistently for ADT agreement, E-c and controls. Large-mesh eligibility is otherwise undefined.

8. **FOM ladder and tolerance qualification — mostly CORRECT; qualification reference NEEDS-RESTATEMENT.**

   The revised ladder includes the missing CN-R runs; cross-resolution matching and eight-case large-mesh diagnostics are separated; the PS port is now appropriately explicit. The finite-ladder speed-up bound has the correct direction.

   The tightest tolerance run used as the qualification reference must itself pass all finite/stage-convergence checks. Otherwise a candidate can be compared against a failed reference solve. Specify “no qualified level” as an unavailable configuration, with no fallback to an unconverged level.

   Also identify the residual as the recorded **relative** residual, consistent with `fomcore`, and make the norm and \(n_0\) in the lattice-difference test explicit.

9. **K9 distance control — NEEDS-RESTATEMENT; one natural interpretation is WRONG.**

   Perturbing an eligible output by \(0.5\tau\), where \(\tau=2.5\times10^{-4}\), does not ensure it remains within \(\tau\) of ADT. If its original ADT distance is \(0.9\tau\), an aligned perturbation produces \(1.4\tau\).

   Construct the two synthetic outputs around the **distance comparator itself**, or explicitly test the perturbed output against its unperturbed origin. Normalize directions in the actual field metric and verify the constructed distances before invoking the classifier. Pin the random seed and case/configuration.

10. **Freeze order and reduced-rule claims — CORRECT.**

    A1.11 supersedes the old diagram with the right ordering. Replication cannot influence mesh-free qualification, FOM matches or winner selection. A1.13 also correctly distinguishes empirical surrogates from quadrature equivalence and requires finalist sensitivity measurements.

**Required blocking changes**

1. Complete QN initialization, candidate/active masks, failure precedence and refresh semantics; fix refresh counting and recompute stopping information after refresh.
2. Make H1.1’s case selection, replay step/state, baseline replay check, QN initialization and mutually exclusive outcomes executable.
3. Specify the large-mesh solver-distance metric and how it is computed without the omitted mesh Gram.
4. Correct the streaming algorithm’s executable one-based-to-zero-based mode indexing.
5. Require a valid tightest FOM qualification reference and specify the no-qualified-level outcome.
6. Anchor K9’s synthetic distances to its actual comparator and normalize them in the registered metric.

Non-blocking suggestions:

- Explicitly scope A1.5’s eight-case/three-repeat rule versus Route 4’s four-case/two-repeat panel.
- Define the decomposition’s aggregation convention: medians of paired ratios generally do not multiply into the median total ratio.
- Require gl12 to pass conditioning as well as rank before counting K5 as an accuracy control.
- Extend tight/tighter ADT sensitivity checks to finalist reduced-rule and large-mesh objectives; the current deployed-rule check does not validate those different objectives.
- Label prepared-input latency clearly and present preparation costs beside it.
- Correct FSP’s “full history” wording and state that A1.11 supersedes the original execution diagram.

**Final verdict: NOT READY.**