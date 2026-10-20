No files changed, jobs submitted, or arrays copied. This is a design audit, not a code audit. A2.3/A4.3 ratification is not re-flagged.

Evidence shorthand: **D** = [baseline DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/DESIGN.md); **Common/Brief** = the supplied campaign briefs; **R/C/S** = the supplied references/combined/speed3d experiment directories.

1. **A6 — NEEDS-RESTATEMENT: the POD repairs are sound; the underdetermination explanation overstates Q1.**

   **Evidence:** D:883–930; Common:49–56.

   A6.1–2 resolve audit 6’s two blockers: negative-tail handling, saturation, separate dimension-capped sketches, and selection of one nested basis are explicit. The timing estimator, damping rejection and metadata pin are also appropriate.

   However, “fewer residual rows than unknowns … cannot be Q1-qualified” does not follow from COMMON-BRIEF. Q1 imposes stationarity, validity and trajectory-distance tests—not uniqueness. Nonunique solutions can satisfy those tests. Excluding underdetermined controls is a defensible **additional lane restriction**, but is not a mathematical consequence of Q1.

   **Fix:** Describe actual-support `< r` rejection as a registered control-admissibility restriction. Preserve the POD certificate and the revised overdetermined ladder.

2. **A7 — CORRECT: the amendments strengthen verification without changing the competitive objective.**

   **Evidence:** D:936–962; R/refcore.py:64–87; R/refstore.py:53–76.

   Solving passive NNLS systems on original columns avoids another conditioning penalty. The stencil/Jacobian fixture, classifier exercise and every-call timing checks address concrete failure modes. Distinguishing the adapter smoke from exported-error replay prevents a premature G-ours claim.

   A11 subsequently governs the distinction between exhausted backtracking and non-finite steps.

   **Fix:** None; retain that precedence and the phase-II G-ours obligation.

3. **A8 — NEEDS-RESTATEMENT: the solver changes are defensible, but their mathematical justification contains false generalizations.**

   **Evidence:** D:968–987; C/c2common.py:32–35; C/t2core.py:175–180.

   Changing native production tolerance to \(10^{-6}\), retaining common LM at \(10^{-3}\), and using native QR anchors for both production solvers are transparent, pre-verdict choices. Neither identical production tolerances nor identical anchor algorithms are required by Q1.

   Two explanations need correction:

   - Gauss–Newton is **not generally quadratically convergent** for nonzero-residual LSPG. Consequently, an extra iteration is not guaranteed cheap. [The local convergence bound includes a residual-dependent linear term.](https://www.cs.cornell.edu/courses/cs4220/2026sp/lec/2026-04-10.html)
   - A square nonlinear residual does not necessarily have a root, and a square problem is not inherently unable to pass Q1. At an exactly represented root, the registered ratio is zero because of its denominator safeguard. Near-root floating-point residuals can instead produce misleadingly large ratios.

   **Fix:** Retain the registered settings, but describe the square-control failure as an observed dry-run outcome. Replace the convergence claim with measured iteration/cost evidence. Freeze the settings across production configurations; do not selectively relax them after verdict data.

4. **A9 — NEEDS-RESTATEMENT: contract-compatible anchor exception, overclaimed round-off diagnosis.**

   **Evidence:** D:512–514,578–582,993–1003,1013–1015; Common:49–56.

   **A9 does not change COMMON-BRIEF Q1.** Production still uses fresh-Jacobian certification, the \(10^{-3}\) threshold, pooled 1% limit, every-case \(2.5\times10^{-4}\) distance and prescribed-subset \(2.5\times10^{-5}\) anchor agreement. A11 correctly prevents non-finite steps from receiving the exception.

   But exhausting eight halvings does not establish “no residual decrease possible at working precision.” It establishes failure of that bounded line search. Ill-conditioning or an inadequate step range can also cause it.

   The effective exceptional limits are \(10^{-5}\) for tight anchors and \(10^{-6}\) for tighter anchors. Two runs can stall at the same trajectory and pass agreement without independently demonstrating the claimed round-off mechanism.

   **Fix:** Call this a **small-gradient backtracking-stall allowance**, report nominal versus exceptional acceptance separately, and label round-off as the proposed explanation. If asserting proven round-off, require an additional diagnostic capable of distinguishing it from line-search exhaustion. The common Q1 bars need no change.

5. **A10.1 — CORRECT: changing the doubled-work fixture is a legitimate pre-verdict control amendment.**

   **Evidence:** D:737–742,915–921,1005–1007.

   Using the longer `none` query improves timing resolution while preserving two distinct synchronized calls, all six timing cases, the estimator and the acceptance band. The failed CPU outcome is disclosed rather than erased.

   **Fix:** None. Treat real-GPU firing as still required; this control alone does not establish timing precision for millisecond ECSW queries.

6. **A11 — NEEDS-RESTATEMENT: exit semantics and persistence are correct, but retained evidence needs a revised storage contract.**

   **Evidence:** D:483–485,799–802,1013–1027; D:306–307.

   Separating non-finite steps into reason 5 closes the substantive A9 ambiguity. Persisting trajectories and independently rescoring evidence is appropriate. The maximal NNLS pilot’s support requirement is explicit.

   However, “every grid configuration” now retains fields and trajectories, while the older local-storage statement remains “≤2 GB per cell.”

   For the full 3D Stage A alone, assuming all configurations construct:

   \[
   6048+3024+1008=10080
   \]

   production/shared-tight/tighter runs produce approximately **121 GB** of uncompressed six-output lattice fields. The analogous base 2D cell produces approximately **4.44 GB**, before Stage B, controls, bases or other evidence.

   **Fix:** State whether A11’s field-retention requirement is smoke-only or production-wide. For production, register streamed verification and durable storage/deletion rules, and replace the obsolete local bound with a manifest-derived peak. Do not resolve the conflict by silently dropping required evidence.

7. **A12 — CORRECT: the final amended acceptance requirements are internally consistent.**

   **Evidence:** D:1033–1069; Common:98–100.

   A12.7 correctly limits NaN-aware comparison to invalid-grid accounting; timed outputs must be finite. Residual-only autodifferentiation removes the identified unnecessary Jacobian tangent. The small-mesh POD change acknowledges the method-of-snapshots resolution limit while retaining the independent production certificate.

   The narrowed Q-DEIM pivot claim appropriately distinguishes a pivot rule from LAPACK implementation equivalence. QR-based node selection is the standard Q-DEIM construction. [Primary Q-DEIM paper.](https://arxiv.org/abs/1505.00370)

   **Fix:** None at design level. The \(10^{-8}\) small-test cutoff is near the Gram-method resolution boundary, so passing remains empirical rather than guaranteed.

8. **CORRECT — The governing baseline equations are mathematically consistent.**

   **Evidence:** D:52,66,103–108,338–352,529–534; R/refcore.py:64–87.

   The residual’s diffusion sign, time coefficients and product-rule Jacobian are correct. ECSW fits the appropriate contribution:

   \[
   \nabla_c\frac12\sum_i\xi_i\mathcal R_i(c)^2
   =\sum_i\xi_iJ_i(c)^\top\mathcal R_i(c).
   \]

   The governing DEIM compression is **untruncated Householder QR**, superseding the original truncated SVD. Thus \(\|Bz\|=\|R_Bz\|\) algebraically, without inverting \(R_B\) or discarding linear terms.

   This is a credible weighted strong-residual POD-LSPG baseline. Petrov–Galerkin ECSW explicitly accommodates finite-difference discretizations. [Primary ECSW paper.](https://arxiv.org/abs/2008.02891)

   **Fix:** Continue describing the final compression as QR, and do not claim an energy-conservation property for this Burgers construction.

9. **CORRECT — Fairness and selection are defensible as a developmental comparison, not independent validation.**

   **Evidence:** D:385–394,675–685,788–791,854–857; Brief:27–31; C/report.md:3,42–44.

   The governing design does **not** select final picks only on dev6/cases 0–5. Those cases screen; Stage B selects and judges on the full development cohort. X-specific fronts prevent pruning solely because an absolute-time front favors another configuration.

   Training-only construction, validation-informed work, offline costs and unequal historical selection exposure remain visible. The full scheme ladder and complete common-solver ablation protect against straightforward baseline handicaps.

   **Fix:** Preserve “developmental, in-sample, unequal selection histories” and actual budget ratios. Neither small-cohort screening nor subsequent full-cohort selection establishes generalization.

10. **CORRECT — H-base is sufficiently frozen for descriptive use, including degenerate cases.**

    **Evidence:** D:228–243,441–464,617–619,644–655,709–713,782–786,827–829,915–921; Brief:37–46.

    Dominance requires resolved time and worst-error improvement. Offline cost and median error accompany it, as the lane brief specifies. Matching requires the registered equivalence tests and worst-envelope guard; overlap alone is insufficient. Crossing resolved objectives yield “neither dominates.”

    Deterministic ties, duplicate picks, missing eligibility, failed picks and incomplete timing coverage have explicit outcomes. Identical arms intentionally remain unresolved in the reference-limited regime.

    Multiplicity and selection invalidate confirmatory interpretations, but the governing labels expressly remain developmental and unadjusted. Worst-error envelopes are conditional reference-error envelopes, not population confidence intervals.

    **Fix:** None. Preserve these qualifications beside every conclusion.

11. **CORRECT — Required controls exist; actual firing remains unestablished by this design audit.**

    **Evidence:** D:424–436,600–610,773–780,824–825,859–877,904–906,947–957.

    Independent residual/Jacobian checks cover each baseline. G-full supplies full-rank/all-points FOM recovery. N-under requires qualified accuracy degradation; solver failure alone cannot satisfy it. An unfulfilled family control blocks that family’s comparisons.

    A8’s larger supports make an informative ECSW control more plausible, but do not guarantee degradation or qualification.

    **Fix:** Execute the frozen real-size smoke and retain unsuccessful levels. Do not claim that the controls fire from CPU dry-run explanations or code-audit acceptance alone.

12. **CORRECT — Checked dependency facts agree; production feasibility remains conditional.**

    **Evidence:** D:354–361,367–370,715–730,751–754; training JSON:5,19,33–36; `burgers2d_film.py`:160–185; `sep_burgers_r3.py`:321–332; C/inputs/model_W1024/training.json:3–27; C/t2core.py:36–37,165–180,259–263; S/DESIGN.md:60–63; S/resources.py:1–12.

    Checked facts agree: training seeds/draws, the 2D early-state selection versus the lane’s 12-step choice, 3D snapshot schedule, IC and mesh conventions, dimensional timesteps, startup conventions and query initialization.

    The old 13 GPU-hour estimate is withdrawn. Maximal matrices, full-count Gram work, the potentially enlarged Stage B, QR/NNLS workspace, compilation variants and retained evidence must enter the resource certificate. A 1,024-support pilot does not itself establish production cost at support 16,384.

    **Fix:** Require the complete smoke-derived certificate before production; report an incomplete cell when the 30 GPU-hour cap cannot accommodate it.

**Overall verdict: NEEDS-RESTATEMENT.** The governing residuals and common Q1 contract survive this audit. Top blockers are the overgeneralized A6/A8 mathematical explanations, A9’s unsupported identification of backtracking exhaustion with round-off, and A11’s unresolved production evidence-storage scope. Real-size control firing and resource feasibility remain subsequent empirical gates.