**Overall verdict: NEEDS-RESTATEMENT.** The support-based ECSW knob is a defensible repair, and using development-case smoke evidence before production is legitimate here. It preserves the three-point tuning grid. However, the extension rule still refers to the removed knob, the claimed continuation is not implemented by the current construction API, and the revised candidate-banning rule is not yet demonstrated to resolve the observed failure.

Evidence shorthand:

- **D** = [DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/DESIGN.md)
- **A** = [smk1 ACCEPTANCE.json](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/runs/smk1/archive/output/ACCEPTANCE.json)
- **S** = [smk1 smoke.json](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/runs/smk1/archive/output/smoke.json)

Later amendments govern superseded text.

1. **CORRECT — A14.1 separates a shape pilot from an accuracy certificate.**  
   **Evidence:** D:1108–1114, 883–897; S:796; A:165–185, 583–607.  
   An uncertified maximal advection basis can exercise allocation and factorisation shapes without becoming an admissible production basis. Production retains the A6.1 certificate. The reported 2D state numerical rank does not establish the advection rank, but A14 does not need that inference to justify this separation.  
   **Fix:** None; label the pilot explicitly and retain independent certification of every basis used for scored rollouts.

2. **CORRECT — A14.2, as corrected by A15.6, preserves POD admission rules.**  
   **Evidence:** D:1116–1120, 1161–1165, 883–897; A:1–36.  
   Parking candidate bases on the host and chunking sign normalization are reasonable memory changes. Flooring the reconstruction divisor is acceptable because finiteness alone does not admit a prefix. A15.6 correctly withdraws the implication that every prefix beyond the diagnostic numerical rank must be rejected: saturated prefixes may pass A6.1.  
   **Fix:** Describe the memory changes as implemented remedies awaiting successful real-size validation, rather than demonstrated solutions to the OOM.

3. **CORRECT — A15.1’s motivation is supported, but only for the observed configurations.**  
   **Evidence:** D:1132–1138, 1155–1158; A:187–197, 239–260, 329–350.  
   At \(r=128\), the recorded tolerance crossings have 32 and 64 positive weights. These sampled residuals have fewer rows than unknowns. The 4r fixture passes Q1 with median error 0.03697 against 0.03115 for `none`; 2r fails Q1. This supports replacing a poorly calibrated tolerance grid with a support grid.

   It does **not** establish that tolerance-controlled ECSW is intrinsically ill-posed, that every registered tolerance fails, or that \(s>r\) guarantees full column rank or Q1. A13 already correctly separates underdetermination from the literal Q1 predicate.  
   **Fix:** Narrow the heading to “the observed tolerance settings produced underdetermined sampled LSPG problems.”

4. **NEEDS-RESTATEMENT — Support-controlled ECSW is legitimate, but it is a declared variant of the cited construction.**  
   **Evidence:** D:103–115, 338–343, 1136–1138, 1159–1162; `build.py`:265–349.  
   The positive-weight approximation to the assembled LSPG gradient remains mathematically appropriate. Changing where the greedy path is sampled does not turn it into arbitrary collocation.

   However, the cited Petrov–Galerkin ECSW paper explicitly uses a relative training-residual tolerance to stop NNLS; it does not prescribe \(\kappa\in\{2,4,8\}\). The latter is a reasonable experimental support-budget parameterization, not a published universal ECSW prescription. [Grimberg et al., §4.1, equations 18–20](https://arxiv.org/html/2008.02891).

   **Fix:** Name it “ECSW with support-targeted greedy NNLS.” Report actual support, achieved training residual, construction failures and solver qualification. Do not claim that the new grid guarantees a strong or unbiased baseline.

5. **NEEDS-RESTATEMENT — A2.18 and ECSW coverage rules were not completely migrated to \(\kappa\).**  
   **Evidence:** D:657–659, 747–749, 1136–1138, 1159–1162.  
   A2.18 still triggers extension at the **smallest τ**; A3.11 still defines boundary coverage failures using τ and the old cap retry. A15 names 16r but does not explicitly replace those predicates. This leaves discretion over whether a boundary ECSW configuration receives more sampling budget.

   The base grid cardinality is unchanged: three ECSW settings replace three settings, giving **140 configurations per 2D mesh and 1,008 in 3D**, before extensions. Grid cardinality alone does not establish equal tuning cost.  
   **Fix:** Explicitly replace the ECSW boundary by \(\kappa=8\), the single extension by \(\kappa=16\), and coverage accounting by attempted/constructed \((r,\kappa)\). State the outcome when no front exists or the target cannot be reached; do not silently treat failed boundary construction as adequate coverage.

6. **NEEDS-RESTATEMENT — A15.6’s continuation requirement is not supported by the current API.**  
   **Evidence:** D:1159–1162; `pipeline.py`:175–187; `build.py`:334–349; `nnls.py`:100–110, 151–155, 179–185.  
   First-crossing checkpoints are well-defined. But `pipeline.ecsw()` reconstructs contributions, starts a fresh path, deletes the matrix, and replaces the stored records. `greedy_nnls()` exposes no resumable state. Moreover, the terminal cap affects batch size, so rerunning with a larger cap is not automatically an identical continuation.

   **Fix:** Either retain resumable path state and immutable earlier checkpoints, or explicitly register construction through 16r in advance while permitting validation of that checkpoint only when the extension rule triggers. Include the resulting construction/storage cost in the resource certificate.

7. **CORRECT, with mandatory accounting — Development-informed knob selection is permissible here.**  
   **Evidence:** D:385–394, 675–685, 788–791, 859–873; A:329–350.  
   These are development cases, not sealed confirmation cases. The design already calls the comparison developmental, in-sample and subject to unequal selection histories. Correcting a baseline that fails numerically is legitimate; freezing a known defective construction would handicap it.

   Nevertheless, this amendment used development errors and Q1 outcomes to choose the knob and fixture. “Pre-verdict” does not make that work training-only.  
   **Fix:** Explicitly include the validation-informed smoke/dry-run portions in V accounting under A3.1, retain the discarded settings’ evidence, and freeze the amended grid before production. Equal-budget language remains conditional on the registered accounting and ratification rules.

8. **NEEDS-RESTATEMENT — A15.3/A15.7 diagnose banning honestly but do not establish that three drops fixes it.**  
   **Evidence:** D:1142–1144, 1167–1170; S:20414–20439; `nnls.py`:142–174; `checks/test_core.json`, G-nnls entry.  
   The observed failure is substantial: **20,989 banned candidates**, relative residual **0.077514**, and a positive excluded-candidate gradient far above the KKT tolerance. Allowing three drops is less aggressive, but a permanent ban still changes the admissible search. The duplicated-column regression reports **zero bans**, so it does not exercise this failure mechanism.

   Global NNLS KKT satisfaction is **not** required at an intentional sparse checkpoint. The concern is excluding useful candidates and then interpreting a weakened construction as evidence against ECSW.  
   **Fix:** Register a consequence for material banning: a bounded robust retry without permanent exclusion, or an explicit construction-quality limitation blocking “adequately tuned ECSW” and exhaustive non-domination claims. Exercise the guard on a representative failing case. Report checkpoint fit quality as well as terminal diagnostics.

9. **CORRECT — A15.2 strengthens the negative-control experiment without weakening its success criterion.**  
   **Evidence:** D:1139–1141, 899–906, 1075–1077; A:329–380, 735–800.  
   Adding intermediate overdetermined support caps can distinguish accuracy degradation from solver failure. Requiring both Q1 and twice-`none` error remains appropriately strict.

   The historical 3D ECSW control fired at **4r**, which is no longer in the revised ladder. That historical success cannot certify the new ladder or the changed NNLS algorithm. Neither 2D family fired in smk1.  
   **Fix:** Re-establish the revised controls in the next authorized smoke; preserve per-family blocking when a control remains unfulfilled. Call the listed sizes **caps**, since actual support may be smaller.

10. **CORRECT — A15.4–A15.7 accurately distinguish evidence, remedies and incomplete phase I.**  
    **Evidence:** D:1145–1170; A:398–420, 609–672, 840–859; `cluster/collect.sh`:13–25.  
    A15.6 corrects the failing expression to `G.T @ b`, corrects the orthonormality and non-stationarity summaries, and acknowledges incomplete G-fd coverage. A15.5 correctly strengthens ROM-fixture eligibility; the recorded time-shift score was exactly unchanged. Qualification still does not guarantee a discriminating fixture.

    Collection now requires a recognized terminal allocation state and verifies transferred files before optional deletion. A failed or ambiguous state query refuses collection.  
    **Fix:** None to these governing rules. Keep `ALL=false` prominent; successful executed subchecks are not complete family-level acceptance.

11. **CORRECT — Governing baseline mathematics and Q1 remain consistent.**  
    **Evidence:** D:338–352, 529–534, 578–591, 980–982, 1087–1091; references `refcore.py`:64–87, 112–132; combined `t2core.py`:35–38.  
    Residual signs, the product-rule Jacobian, and \(g_i=J_i^\top\mathcal R_i\) are correct. The governing Q-DEIM compression is **untruncated Householder QR**, superseding the original truncated SVD. It preserves the assembled DEIM residual norm algebraically and keeps linear terms exact.

    The governing production Q1 rule has no residual-tolerance exemption, uses fresh in-job Jacobians, pooled stationarity and every-case anchor distance. The disclosed anchor-stall allowance does not alter those production bars.  
    **Fix:** None. Describe the actual QR-based method, not the superseded SVD-cutoff formulation.

12. **CORRECT — Cohort, stepping and H-base rules remain defensible within their stated limits.**  
    **Evidence:** D:367–394, 617–619, 644–655, 782–791, 854–857, 915–921; references `cohorts.py`:35–37; `refstore.py`:132–173; speed3d `DESIGN.md`:60–63.  
    Final baseline selection is on the **full development cohort after subset screening**, not solely dev6/cases 0–5. The actual 2D/3D step sizes and startup conventions agree with the dependencies. Training draws and snapshot provenance agree with `sample_params`:160–177, the bank training JSONs, and `sep_burgers_r3.py`:321–332.

    H-base handles ties, missing eligibility, failed/absent picks, crossing fronts and reference-limited equivalence. Worst-error envelopes are conditional reference envelopes, not population confidence intervals. Multiplicity and in-sample selection prevent confirmatory interpretation; the amended descriptive wording correctly acknowledges that.  
    **Fix:** Preserve those qualifications beside every conclusion. Operator-matched snapshots remain the explicitly disclosed generator deviation, subject to the existing release condition.

13. **CORRECT — Required controls are specified; feasibility and acceptance remain unproved.**  
    **Evidence:** D:424–436, 600–610, 715–730, 799–802, 1093–1097, 1151–1153; A:840–859; `runs/GPU-LEDGER.json`:4–19.  
    Independent residual/Jacobian checks, full-rank all-points recovery and per-family under-sampling controls are present. The small-mesh gate record reports full-rank recovery; smk1 does not establish complete real-size acceptance.

    The withdrawn 13 GPU-hour estimate cannot be revived. The smoke omitted important work and encountered OOMs. The new support targets, NNLS retries/continuation, host assembly, fallback basis coexistence, actual compilation shapes and production evidence retention must enter the complete certificate. The recorded **0.878 GPU-hours** establishes expenditure, not production feasibility.  
    **Fix:** Require the successful authorized smoke, complete resource projection under 30 GPU-hours, and promised phase-II storage amendment before production acceptance.

**Overall: NEEDS-RESTATEMENT.** Top blockers are the incomplete \(\tau\rightarrow\kappa\) extension/coverage migration, the unsupported continuation claim, and the unresolved effect of permanent NNLS banning. The knob change itself is acceptable. Audit 8’s readiness does not transfer automatically to these post-smoke changes, and smk1 remains a failed acceptance run.