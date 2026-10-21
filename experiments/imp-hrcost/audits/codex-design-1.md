**Overall verdict: WRONG — revise and re-audit before implementation.** The four-arm mathematics is sound. The blocking problems are qualification changes presented as inherited contracts, incorrect control predicates, and insufficiently specified fitting and resource limits.

This was a read-only source audit. I changed no files, submitted no jobs, and did not test controls on real data.

1. **Four-arm formulas — CORRECT.**

   [DESIGN.md:28](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/imp-hrcost/DESIGN.md:28) correctly transcribes the prior sine-tested Galerkin equation:
   $$
   R_m\dot c=-Q^{\mathsf T}\Psi^{\mathsf T}a(c)
             -\nu Q^{\mathsf T}\Lambda Ac.
   $$
   Therefore
   $$
   P=\Psi QR_m^{-\mathsf T},\qquad
   \mathcal L=-\nu R_m^{-1}Q^{\mathsf T}\Lambda A.
   $$
   There is no transpose or sign error.

   The factor \(L\) in \(\Psi\) is necessary. `qcore.py` defines mesh-orthonormal tests as \((2/L)\sin\sin\), continuum tests as \(2\sin\sin\), and consequently \(A=\Phi^{\mathsf T}G\approx L\int\psi G\). Removing \(L\) from the nonlinear term would introduce a scaling error.

   Bank testing gives
   $$
   \mathcal M\dot c=-G_q^{\mathsf T}Wa(c)-\nu\mathcal Kc.
   $$
   Since \(\mathcal M\) is symmetric, \(P=WG_q\mathcal M^{-1}\) and \(\mathcal L=-\nu\mathcal M^{-1}\mathcal K\) are correct. Fitted weights simply replace \(W\).

   **Qualification:** F2 changes both the projection and the diffusion discretisation: mesh-discrete sine diffusion becomes continuum bank stiffness. The design acknowledges this; results must retain that qualification. Implement inverse actions with solves, and verify symmetry/positive definiteness rather than relying only on a Frobenius convergence test.

2. **Every arm costs \(6mr\) — NEEDS-RESTATEMENT.**

   This is correct as the **leading dense arithmetic count for one advection-value evaluation**, with \(G_s\) and \(P\) precomputed:
   $$
   G_qc,\quad G_sc,\quad P^{\mathsf T}(u\odot s).
   $$
   It omits lower-order multiplication/reduction work and is not a full RHS, Jacobian, Newton-step, or query cost.

   The inference at lines 52–53 that *any saving must come from smaller \(m\)* is too strong. Changing projection can change conditioning, iteration counts, convergence, and implementation efficiency even at fixed \(m\). GAL-CN still computes Jacobians and solves nonlinear systems.

   **Fix:** say “the leading advection-value arithmetic differs only through \(m\).” Report Jacobian evaluations, nonlinear iterations, failed steps, evolution time, and query time separately.

3. **Stop rules — NEEDS-RESTATEMENT.**

   The two quoted rules preserve the brief’s wording and thresholds. Requiring Q1 in addition to Q2 strengthens them appropriately. End-to-end timing matches the ideation kill criteria.

   Three operational qualifications remain:

   - “Equal worst error” becomes a conditional Richardson tie. This is compatible with the common brief’s tie convention, but must be described as **no distinguishable worsening under the registered conditional tolerance**, not literal equality.
   - F3’s median \(\ge1.5\) with confidence lower bound merely \(>1\) supports a point estimate of 1.5× and evidence of some speed-up. It does **not** establish a statistically supported gain of at least 1.5×.
   - If construction or budget prevents testing a registered rule, failure to find a passing rule is not evidence that no such rule exists. Distinguish **stop: tested failure** from **stop: incomplete evidence/budget**.

   The F2 selection rule also changes the brief’s “smallest Q2-passing rule” into “smallest Q1+Q2-passing rule.” That is conservative for acceptance, but record both minima so solver failure is not misreported as a quadrature-size requirement.

4. **Q1 and Q2 — WRONG as claimed inherited contracts.**

   **Q1:** The stated certificate is not an algebraically equivalent transcription of `jcp-qualify` §4.1b. If \(\rho_n\) denotes the coordinate CN residual in E6, the inherited certificate is
   $$
   \eta_{\rm old}
   =\frac{\|R_m\rho_n\|}{\|Ac_n\|}
   =\frac{\|R_m\rho_n\|}{\|R_mc_n\|}.
   $$
   E6 instead uses \(\|\rho_n\|/\|c_n\|\). They have the same zeros but different acceptance regions, potentially differing by factors controlled by \(\operatorname{cond}(R_m)\).

   A root-residual certificate is scientifically appropriate; the prior lane explicitly registered its departure from literal stationarity. E6 must likewise explicitly register and audit its norm change. It cannot inherit the numerical threshold merely by calling this a transcription.

   Additional gaps:

   - State explicitly that certification runs **in the allocated job**, from every saved internal state, with fresh AD Jacobians.
   - Specify adaptive tight/tighter solver settings and require both anchors to be valid.
   - Restore agreement between the independent certificate and the solver’s recorded convergence quantity—the prior G-cert gate.
   - Define zero-normalisation and non-finite handling.

   **Q2:** Replacing the tested \(M\)-vector by the precontracted \(r\)-vector weakens the inherited sine-tested check in a precise way. For an error \(e\in\ker(Q^{\mathsf T})\),
   $$
   R_m^{-1}Q^{\mathsf T}e=0
   $$
   even when the tested-advection error is arbitrarily large. Thus the new check can pass when the original ρ gate fails. The inverse factor also changes the norm weighting.

   The projected quantity is useful for assessing the dynamics, but **the old 0.116 threshold does not automatically transfer**.

   **Fix:** retain the original tested-vector ρ gate for R/F3 and report the projected indicator additionally. For bank testing, explicitly define and audit the corresponding bank-tested-vector norm; report its mass-inverted counterpart separately.

   **Parent-rule successor:** Gauss \(128^2\) is a defensible reference for a fitted compression rule. It need not be another fitted support size. This does **not** pass by construction on unseen validation trajectories: NNLS fits training snapshots, whereas Q2 evaluates the fitted arm’s own validation states.

   However, it certifies compression relative to the parent, not independent continuum accuracy. Preserve parent-versus-finer convergence, require valid \(q^{++}\) runs, independently recompute ρ, and define zero-denominator/all-trivial cases as the prior code does.

5. **Fitting and leakage — NEEDS-RESTATEMENT; no prohibited leakage found.**

   The proposed snapshots are decoded states from training-draw trajectories, and the fit uses projected advection contributions rather than residual snapshots. That is consistent with the ideation prescription.

   Selecting \(m^*\) or \(\kappa^*\) using dev6/val32 is explicitly permitted. Excluding those cohorts from **weight fitting** does not prohibit validation-based configuration selection.

   Necessary fixes:

   - Freeze and hash the actual 576-row draw table, selected 16 rows, snapshots, and parent configuration. `train_draw_2d(seed=0,count=16)` is not equivalent to taking the first 16 rows of the count-576 draw.
   - Require valid training rollouts and define what happens if a parent cannot produce them.
   - Define handling for \(\|b_j\|=0\), tiny norms, and non-finite blocks.
   - Name the procedure **support-targeted greedy NNLS**. First crossings are not generally converged global NNLS optima.
   - Specify unreached-support handling, iteration/time limits, banning/retry policy, rank-deficiency handling, QR-drift acceptance, and training-only retry selection. A15/A16 and the current `nnls.py` contain these distinctions; copying the solver does not specify their use.
   - Freeze fitting choices before validation results are inspected.

6. **Controls — WRONG in part; real-data firing remains unestablished.**

   [The control table](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-hrcost/experiments/imp-hrcost/DESIGN.md:182) has these problems:

   | Control | Verdict and required correction |
   |---|---|
   | C1, coarse quadrature | **NEEDS-RESTATEMENT.** Both rules may be inaccurate, unstable, or agree accidentally. Both bars failing is not guaranteed. Require valid fixtures and demonstrate each gate’s sensitivity separately. |
   | C2, shuffled weights | **NEEDS-RESTATEMENT.** Permutation need not cause failure—for example, nearly equal weights change little. It tests sensitivity to weight association, not independence from fitting. Freeze the permutation seed and validate the fixture. |
   | C3, corrupted state | **WRONG.** An interior saved \(c_{n+1}\) enters residuals for two adjacent CN steps. “That step and only that step” is generally false. Perturb the final state or require the affected adjacent residuals to respond. Specify the perturbation direction. |
   | C4, NaN table | **CORRECT in intent.** Require rejection by the actual validity path. |
   | C5, duplicated rule | **WRONG as a guaranteed query-time predicate.** Doubling quadrature work does not guarantee a 30% increase in total query time when decode/init/other work dominates. Use a dedicated advection benchmark for the doubled-work timing control; keep query timing descriptive. |
   | C6, A–A | **CORRECT as a sanity check**, but cannot by itself exclude workload-dependent ordering or thermal bias. |
   | C7, leakage | **CORRECT in intent**, provided it checks actual descriptors/provenance, not just cohort labels. |
   | C8, scoring identity | **CORRECT in intent**, provided independent requested identity survives the mutation. |
   | P1/P2 | **Useful but incomplete.** Two paths using the same erroneous \(P\) can agree. Add direct comparison against the original uncontracted sine-tested RHS and Jacobian. |
   | P3 | **Useful convergence check**, not proof of exact integration. |

   Missing high-value controls are an independent bank mass-form RHS comparison, an independent Jacobian check, and rejection of invalid tight/check-rule anchors. “A control that cannot fire is a bug” should distinguish an implementation bug from a defective control fixture.

7. **Timing, fairness, and predicates — NEEDS-RESTATEMENT.**

   Same-job pairing, synchronisation, f64, common inputs, full decode, and case-level bootstrap are good choices.

   Remaining loopholes:

   - Specify burn-in **after fitting and before each timed block**. Three queries at an unspecified point do not address the project’s documented post-NNLS clock-ramp bias.
   - Persist all repetition arrays, order, device identity, and outlier counts.
   - Tie scored outputs and validity to the timed configuration/invocations.
   - Clarify ratio aggregation: a median of paired ratios and a ratio constructed from separate medians are different statistics.
   - Searching several κ values and accepting any passing interval makes ordinary per-arm 95% intervals selection-sensitive. Use simultaneous inference or label them exploratory, in-sample and unadjusted.
   - Require a common eligible case set; failed cases cannot disappear from worst-error calculations.
   - The maximum of the two worst cases’ \(\varepsilon_{\rm ref}\) values is a declared tolerance convention, not a confidence bound for the maximum. Report the defining cases and per-case worsen/tie/improve counts.

   COMMON-BRIEF §5 requires tuned competitors. Fixing E5’s arm is defensible for a controlled ablation, but does not establish superiority to an accuracy-matched, tuned classical competitor. Either obtain an explicit scope exception or include the required comparator tuning and finite-ladder caveat.

8. **E5 dependency and GPU cap — NEEDS-RESTATEMENT.**

   Copying the handoff with commit/hash provenance, comparing against E5 in the same job, and retaining R even if E5 fails its own stop rule are correct.

   At audit time, `HANDOFF-E6.md` and E5’s `DESIGN.md` were absent from the specified directory. Thus the assumed interface, tolerances, and primary stepper remain provisional.

   **Fix:** include the brief’s ≤30-minute polling interval, 48-hour deadline, waiting status, and a mandatory handoff reconciliation/amendment before dependent implementation or runs. Also correct §1.1’s “positive control P1” reference: the E5 equivalence control is P2.

   The 3 GPU-h cap is preserved formally, but “NNLS fits minutes” and the production estimate lack evidence:

   - \(C\) has \(67{,}584\times16{,}384\) doubles: about **8.86 GB**.
   - The current updating-QR implementation rounds capacity for a 6144-point target to 8192; its \(Q\) buffer alone is about **4.43 GB**, before other workspace.
   - Eight non-reference configurations × 38 cases × five A–B–A repetitions require **4560 timed query calls**, before qualification and fitting.
   - A κ=8 smoke does not directly certify κ=16 fitting, refactorisation, and deletion costs.

   Use measured phase costs, explicit fit deadlines, complete reservation of remaining mandatory work, and an incomplete-evidence outcome when the cap binds. Deferring Fibonacci changes the searched family; do not silently retain an exhaustive \(m^*\) claim.

9. **Other ambiguities — NEEDS-RESTATEMENT.**

   - Define \(m^*\) as the minimum **among the registered candidate rules**. Failure there does not prove that no other rule below 16384 works.
   - Specify the mass/stiffness convergence fallback beyond \(448^2\), including a hard ceiling and failure outcome.
   - Freeze bank rotation, mesh projection, quadrature, and cold-start identities; hashing only the mode list is insufficient.
   - Define behaviour when R itself is numerically unqualified. Timing against it can remain diagnostic, but cannot establish a qualified replacement claim.
   - The optional Gaussian sketch of the NNLS fitting system is not the ideation’s sketched-LSPG-residual control. Rename it as an additional fitting experiment or implement the intended control.
   - References to the “defer order §9” should point to §8.

**Final assessment:** retain the four-arm formulas and training/validation separation. Correct the Q1 transcription claim, preserve the inherited sine-tested Q2 gate, repair C3/C5, and freeze construction/resource failure handling. Those changes are necessary before this pre-registration can support its proposed conclusions.