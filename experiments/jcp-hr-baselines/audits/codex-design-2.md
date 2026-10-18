**Overall verdict: WRONG — A1 repairs several important equations and provenance claims, but does not yet satisfy the binding qualification, fairness, controls, and decision contracts.**

Items 1–20 below correspond to the first audit’s findings. Additional problems appear afterward. This was read-only; no files were changed or jobs submitted.

Evidence shorthand: `D` = [baseline DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/DESIGN.md); `R`, `C`, and `S` = the supplied references, combined, and speed3d experiment directories; `P` = the supplied full-experiment-plan directory.

1. **CORRECT — Strong-residual LSPG and the ECSW gradient target remain mathematically consistent.**

   **Evidence:** D:66, 103–108, 338–343; R/refcore.py:64–87, 125–143.

   For fixed nonnegative weights, the weighted least-squares gradient is exactly \(\sum_i\xi_iJ_i^\top\mathcal R_i\). The all-node target is appropriate. A1 correctly withdraws any implied Burgers energy-conservation guarantee. Petrov–Galerkin ECSW with finite-difference discretizations is supported by the [primary ECSW paper](https://arxiv.org/abs/2008.02891).

   **Fix:** Retain the equations and explicit “weighted strong-residual LSPG” label. Mathematical validity alone does not establish that the proposed training and solver budgets produce a strong baseline.

2. **CORRECT — A1 fixes the missing temporal and diffusion factors in the Jacobian.**

   **Evidence:** D:340–343 versus the superseded D:101; R/refcore.py:64–87.

   The new expression
   \[
   J=\left[a_0I+\Delta t\,b_0\{\operatorname{diag}(D_hu)+\operatorname{diag}(u)D_h-\nu\Delta_h\}\right]\Phi
   \]
   is the correct derivative of the stated residual.

   **Fix:** Use this expression in online evaluation, ECSW training, and independent checks, including startup coefficients and BDF2 history.

3. **NEEDS-RESTATEMENT — QR fixes the truncation problem, but “different column scales … are harmless” overclaims numerical accuracy.**

   **Evidence:** D:345–352, 426.

   With an untruncated orthogonal factorization, \(\|Bz\|=\|R_Bz\|\) is correct in exact arithmetic, and rank deficiency does not require inversion of \(R_B\). Backward stability does **not** guarantee small relative residual or gradient error near cancellation. The proposed relative comparison at a converged step can divide by a nearly zero norm or gradient.

   **Fix:** Retain QR, describe equality as algebraic/up to roundoff, and specify mixed absolute-relative checks scaled by the assembled terms. Freeze the cancellation fixture and denominators. Do not require relative accuracy against a vanishing gradient.

4. **NEEDS-RESTATEMENT — Snapshot provenance is corrected, but failed saturation still leaves the principal baseline knowingly restricted.**

   **Evidence:** D:354–365; 2D training JSON:5544–5561; `experiments/separable-decoder/sep_burgers_r3.py`:321–332.

   A1 correctly distinguishes the bank’s 16,384 selected states from the lane’s 6,912 snapshots. However, if the training-only saturation check fails, the principal \(1024^2\) comparison proceeds unchanged, merely labelled “snapshot-limited.” Also, passing at \(256^2\) does not certify saturation at \(1024^2\).

   **Fix:** A failed saturation check must block claims that ours defeats an adequately trained POD baseline. Register a streaming/enriched fallback or mark that comparison incomplete. Add a bounded training-only adequacy check on the actual \(1024^2\) basis.

5. **WRONG — Disclosure does not amend the binding trajectory-generator requirement.**

   **Evidence:** D:355–359 versus P/campaign/briefs/jcp-hr-baselines.md:27–31; `engines.py`:41–51; vendored `common.py`:3–5.

   Central-FD snapshots are scientifically defensible and may strengthen the baseline, but the brief explicitly requires reuse of the bank’s generator or stored data. A1 acknowledges the departure without establishing a governing amendment.

   **Fix:** Obtain a coordinator-approved brief amendment allowing operator-matched central snapshots, or use the prescribed generator. Report these as different training fields, not identical training data.

6. **NEEDS-RESTATEMENT — The expanded 3D ladder matches speed3d, but the defer policy can immediately break that fairness.**

   **Evidence:** D:367–370, 505–509; S/DESIGN.md:60–63; C/t2core.py:36–37.

   BE/CN/BDF2 and the four step factors now match the registered speed3d ladder. Their startup descriptions agree with `t2core`. But A1.14 permits dropping the \(2.5\Delta t_0\) baseline options while D:218 still admits all qualified speed3d comparators.

   **Fix:** When an option is deferred, restrict the corresponding fairness claim/comparator set or label it incomplete. Preserve the distinction between `t2core` CNR’s two full BE steps and `refcore` CNR’s four half steps.

7. **NEEDS-RESTATEMENT — Accounting categories improve fairness, but equal tuning budgets remain unestablished.**

   **Evidence:** D:372–383; P/PLAN-v3-amendments.md:89–93; C/report.md:21–29, 181–195.

   “May spend up to” ours’ budget is an equal ceiling, not necessarily matched expenditure. The extension stops when no boundary configuration exists, even with unused budget. Conversely, the original grid may already exceed that ceiling. Rule selection is assigned to C-hr, yet `c3desc*`—which performs rule selection—is also named in C-search. Whole-job `sacct` totals cannot separate these categories automatically.

   **Fix:** Freeze a non-overlapping task-level accounting ledger, distinguish equal allocations from equal expenditure, and state what happens when the initial grid exceeds the search allocation. Label unmatched comparisons descriptive at the **pair level**, without retaining an unqualified H-base falsification claim.

8. **NEEDS-RESTATEMENT — Full-cohort selection is defensible developmentally, but Stage B can systematically discard the fast baseline.**

   **Evidence:** D:385–394, 452–454; C/DESIGN.md:864–871, 887–946; C/report.md:3.

   A1 correctly withdraws the purported shared out-of-selection view. Both methods now use the verdict cases for selection, with unequal histories explicitly disclosed.

   However, “within a factor 1.5 of the front in worst error **or** median time” has no precise frontier-distance definition. The 24-configuration cap, ordered first by error, can discard the cheap end of the Pareto front—even when the front itself has more than 24 members.

   **Fix:** Define the screening relation algebraically. Preserve every screening-front candidate or declare coverage incomplete; allocate any extra slots using a frozen rule that preserves both time and accuracy extremes. Keep all conclusions developmental.

9. **WRONG — A1’s “Q1 restated exactly” changes the binding stationarity predicate.**

   **Evidence:** D:396–405 versus COMMON-BRIEF §3, P/PLAN-v4-amendments.md:16–21, and S/DESIGN.md:95–98.

   The physical-IC normalization and explicit anchor validity repair the original problems. But A1 adds:
   > stationary iff \(\gamma\le10^{-3}\), **or** residual tolerance is met.

   The binding contract counts a step non-stationary whenever \(\gamma>10^{-3}\) **or is non-finite**, without that exception. An approximate residual tolerance is not an “exact root.” A1 can therefore qualify trajectories the common contract rejects.

   **Fix:** Keep solver termination separate from certification. Apply the common \(\gamma\) predicate exactly, or obtain an explicit campaign-wide amendment. Ensure the saved-trajectory certification runs in the same job and independently verifies tight/tighter anchors.

10. **WRONG — ECSW’s zero-target case remains undefined, and the new pooling removes startup coverage.**

    **Evidence:** D:104–112, 407–418.

    If the median target norm is zero, the normalization floor is zero; a zero block is not dropped by the strict condition \(0<0\), so division by zero remains possible.

    A1 also excludes CN-R and BDF2 startup steps from their assigned trajectories. Pooling BE states from other trajectories/options does not provide the previously promised startup coverage.

    Finally, “96 states” is false after deduplication: for BDF2 at \(\Delta t=0.025\), \(N=10\), so \(k_0=2=\lceil0.15N\rceil\). The two assigned trajectories each lose one state.

    **Fix:** Specify an absolute positive scaling floor and an all-zero-target outcome. Include actual startup states and BDF2’s two predecessor states explicitly. Generate the state count from the deduplicated schedule and report per-option coverage and training residuals.

11. **CORRECT — The independent residual/Jacobian checks now distinguish the actual baseline formulations.**

    **Evidence:** D:424–426, 345–352; R/refcore.py:64–87.

    Raw stencil checks, weighted ECSW rows, full reconstructed DEIM objective/gradient checks, and directional checks of each online residual address the original category error. The perturbation no longer vanishes at \(c=0\).

    **Fix:** Retain this decomposition, with the cancellation-safe tolerances in finding 3. Keep the independent assembly independent of the tested row/compression implementation.

12. **WRONG — Dropping the advection Jacobian still does not guarantee N-grad fires.**

    **Evidence:** D:434; C/t2core.py:74–104.

    An approximate-Jacobian iteration can converge to the correct root or sufficiently stationary solution. Choosing low viscosity does not demonstrate that this mutation exceeds either Q1 threshold within the allowed budget.

    **Fix:** Establish and freeze a discriminating fixture, initial state, and finite iteration budget before production. Separately test the certifier on a saved trajectory with independently verified non-stationarity. A harmless mutation outcome must not be called a certifier bug.

13. **WRONG — N-under now measures degradation properly, but weakens the required blocking consequence.**

    **Evidence:** D:433 versus lane brief:32–34 and superseded D:261.

    Requiring an actual error increase instead of accepting solver failure is an improvement. Nevertheless, if every ladder level fails to degrade, A1 blocks only “sampling matters.” The binding brief requires an under-sampling degradation control, and the original table blocked HR accuracy claims.

    **Fix:** Establish an independently verified degradation fixture for each HR family. If no such fixture is found, mark the required control unfulfilled and block the affected baseline-comparison acceptance, unless the coordinator explicitly amends the requirement.

14. **NEEDS-RESTATEMENT — Several repaired controls still contain incorrect or unsupported acceptance tests.**

    **Evidence:** D:428–438; C/c2common.py:32–36.

    - **G-full:** complete bases are now specified correctly, but LSPG “tight” means \((g_{\rm tol},\mathrm{tolf})=(10^{-7},0)\). That does not imply \(10^{-8}\) field recovery. Near a full-rank root, normalized stationarity can remain large as both residual and gradient approach roundoff.
    - **G-nnls:** for \(g=G^\top(G\xi-b)\), positive-weight coordinates require \(g_i\approx0\), not merely \(g_i\le10^{-9}\). The stated one-sided test accepts arbitrarily negative gradients there.
    - **G-pod:** “full computed rank” is ambiguous if computation stops at the registered rank; a truncated POD need not reproduce a training snapshot to \(10^{-8}\).
    - **N-swap:** case/time swaps still have no demonstrated \(10\varepsilon_{\rm ref}\) separation.
    - **N-dom:** case (b) can be classified “dominates” under the new precedence if its small perturbations resolve consistently in both directions.

    **Fix:** Specify root-recovery tolerances separately, correct KKT tests, define numerical snapshot rank, establish discriminating swap fixtures, and make synthetic labels follow the exact frozen decision table.

15. **CORRECT — Conditional error envelopes and basic dominance directions are repaired.**

    **Evidence:** D:228–234, 441–443; R/refstore.py:132–173; P/PLAN-v3-amendments.md:89–93.

    Clipping lower endpoints and explicitly distinguishing reference-error envelopes from population confidence intervals are appropriate. Endpoint propagation through maxima is valid conditional on the supplied bounds. Eligibility remains per case-kind.

    **Fix:** Retain this treatment and require the same eligible case set, reference version, and metric for both arms. Do not silently drop failed cases from either maximum.

16. **NEEDS-RESTATEMENT — The added worst-error guard ignores uncertainty in X’s worst error.**

    **Evidence:** D:445–450.

    A1 compares B’s entire worst envelope with a margin around the **point estimate** \(W_X\). It does not propagate X’s envelope. Moreover, \(\varepsilon_{i^*}\) from X’s point-estimate worst case can miss a different case that determines X’s upper envelope.

    The mean standardized TOST and timing bootstrap are now better specified, but this extra guard does not establish uncertainty-aware worst-error equivalence.

    **Fix:** Freeze a margin and require the worst-error **difference envelope**
    \[
    [L_B-U_X,\;U_B-L_X]
    \]
    to lie inside that margin, or register a justified paired worst-error equivalence procedure. Define worst-case ties and zero margins explicitly.

17. **NEEDS-RESTATEMENT — Decision semantics improve, but the table is not exhaustive and selection remains consequential.**

    **Evidence:** D:190–196, 452–464.

    Empty fronts, missing X errors, precedence, and exploratory multiplicity labels are improvements. Remaining gaps include:

    - A failed selected pick after replay/certification has no explicit disposition.
    - Zero \(\delta_i\), non-finite statistics, insufficient valid timing data, and tied \(i^*\) are undefined.
    - The “total order” omits \(\tau\) or a unique configuration ID; distinct ECSW fits can have equal support sizes.
    - Registered picks retain stronger-looking “H-base falsified” wording despite in-sample selection and unadjusted multiple testing.
    - A non-pick can demonstrably dominate X but only receive an exploratory label, while the cell says “not falsified by the registered picks.”

    **Fix:** Provide an executable truth table with fail-closed outcomes and a unique final tie-breaker. Label **all** selected-pair statistical conclusions exploratory/unadjusted unless multiplicity and selection are addressed. Clearly separate observed domination from formal hypothesis testing.

18. **WRONG — The new resource inventory still excludes its own permitted extensions and fallbacks.**

    **Evidence:** D:379–381, 416–417, 466–485.

    The listed base snapshot, basis, and Jacobian sizes are broadly consistent. But:

    - The 16,384-support ECSW fallback doubles the listed \(G^\top G_S\) storage and requires larger support factorizations.
    - Disk estimates still use support \(\le8,192\).
    - Rank extensions and \(p/r=8\) are excluded from the \(p\le4r_{\max}\) inventory.
    - At 2D rank 512 and \(p=4096\), B alone is approximately **42.9 GB**, before QR copies/workspace and resident basis blocks.
    - Full-rank POD checks and the thin-SVD fallback lack phase-specific workspace estimates.
    - Aggregate retained HR artifacts are not bounded by enumerating all configurations and actual sharing.

    **Fix:** Generate live-memory and disk inventories for every allowed branch, including extensions and retries. Explicitly budget host/device copies, factorization workspaces, retained artifacts, and simultaneous jobs. Certify maximum-size phases, not only reduced-training smoke shapes.

19. **NEEDS-RESTATEMENT — The workload table is useful, but does not support the 30 GPU-hour feasibility claim.**

    **Evidence:** D:487–509; S/resources.py:4–12, 66–93.

    The expanded base grid correctly totals 952 configurations across the three cells. Yet the inventory omits or leaves unbounded saturation work, extensions, NNLS retries, full-size offline pilots, control costs, comparator preparation, and some anchor certification.

    The Q-DEIM count is inconsistent: independent QR pivoting of each \(U_p^\top\) requires ten distinct \(p\) values in the 2D grid, not nine. Reusing prefixes of a larger-\(p\) pivot sequence is a different construction.

    Traced schemes can avoid baseline recompilation, but inherited `fs3` explicitly makes scheme, step, and step count static (C/fs3.py:20); comparator compilation must still be budgeted.

    **Fix:** Generate the inventory from the executable configuration manifest, including all allowed fallbacks and measured compilation/setup. Withdraw the old approximately 13 GPU-hour estimate explicitly. If the complete required work cannot fit the cap, report incomplete scope rather than a completed C3 comparison.

20. **NEEDS-RESTATEMENT — Baseline-strength claims are improved, but solver-based pruning remains asymmetric.**

    **Evidence:** D:505–521; S/DESIGN.md:57–58; C/c3common.py:95–100.

    Backtracked GN is stronger than undamped GN; the FOM is correctly renamed a limiting comparator; spectral deferral is now justified by scope; and `none` native anchors are preserved. The dimension-specific trust policy is also substantially better aligned.

    However, common-solver runs may be dropped based on the native front. A configuration that performs poorly with 30-step GN may perform well with LM; pruning it using GN defeats the purpose of the common-solver ablation.

    **Fix:** Freeze a complete common-solver comparison subset independently of native outcomes, or finish the ablation before claiming solver-robust baseline inferiority. Describe incomplete ablations explicitly.

21. **WRONG — BDF2 FOM support and independent scheme validation are missing from the dependency contract.**

    **Evidence:** D:253, 367–370, 409–412, 428, 493; R/refcore.py:202–217; S/fomcore.py:204–218; C/fom2.py:29–50.

    A1 requires central-FOM BDF2 training, limiting comparators, and full-rank recovery. Neither supplied central FOM implements BDF2. The generic LMM FOM does implement history/startup handling, but uses the upwind operator. G-fom still checks only BE/CN, so a newly written BDF2 history bug can escape the independent FOM gate.

    **Fix:** Explicitly register the central-LMM implementation and independent BDF2 history/startup tests. Test the exact CNR startup convention as well; `refcore` CNR is not `t2core` CNR.

22. **NEEDS-RESTATEMENT — Timing selection and timing verdicts target different populations.**

    **Evidence:** D:187–194, 206–211, 389–391, 449–450.

    Picks now use full-cohort Stage-B times, while verdict timing remains restricted to dev6/cases 0–5. Thus “equal cost” selection and “faster” judgment can describe different case populations. Repeated drift failure is also only flagged; the design does not say whether that pair becomes unresolved or the contaminated timing remains usable.

    **Fix:** Use the same declared timing population for role selection and verdicts, or explicitly distinguish the estimands. Freeze a repeated-drift failure policy. Ensure N-time executes and synchronizes two distinct calls so compiler elimination cannot invalidate the control.

**Top blockers:** the Q1 predicate mismatch; unapproved generator-contract departure; unfulfilled negative controls; Stage-B pruning and incomplete decision semantics; ECSW zero-target/startup handling; missing independent BDF2 validation; and resource estimates that exclude permitted execution paths.

A1 makes the baseline mathematics much more credible. It does **not** yet justify calling the experiment a fair, fully qualified, equal-budget test of H-base.