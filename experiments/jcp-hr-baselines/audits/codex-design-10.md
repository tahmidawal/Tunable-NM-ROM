**Overall verdict: NEEDS-RESTATEMENT.** A16 resolves audit 9’s naming, extension migration, advance-construction and accounting issues. Its banning/retry rule remains incomplete. I found no new mathematical error in the governing A14–A16 specification. This is a design verdict, not code or production acceptance.

Evidence below uses **D** = [experiments/jcp-hr-baselines/DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines/DESIGN.md). Later amendments govern superseded text.

1. **CORRECT — A16.1 resolves audit 9 item 4: the ECSW variant is accurately named.**

   **Evidence:** D:1176–1180; D:103–108, 338–343. The sampled weighted residual produces the gradient contributions \(g_i=J_i^\top\mathcal R_i\); changing the stopping/checkpoint parameter to support size does not invalidate that construction. The published method uses a relative training-residual stopping tolerance, so explicitly naming the support-targeted variant is appropriate. [Grimberg et al., §4.1, equations 18–20](https://arxiv.org/html/2008.02891).

   **Fix:** None. Preserve the actual-support, achieved-residual, construction-failure and Q1 disclosures. This establishes a legitimate construction, not that the chosen grid necessarily yields a strong baseline.

2. **CORRECT — A16.2 resolves audit 9 items 5 and 6 at design level.**

   **Evidence:** D:1182–1188; D:657–659, 747–749, 1159–1162. The boundary is now explicitly \(\kappa=8\), the extension is \(\kappa=16\), and failed boundary construction receives the coverage-incomplete label. Constructing through \(16r\) before screening removes the need for a resumable construction API. Validation of that checkpoint remains conditional on the extension rule.

   An empty screening front produces no extension under A2.18 and falls under the existing incomplete-baseline outcome, D:458–460.

   **Fix:** None to the design choice. Freeze checkpoint identities before screening and include the entire advance construction in the resource certificate. Implementation compliance is outside this audit.

3. **NEEDS-RESTATEMENT — A16.3 does not completely resolve audit 9 item 8: substantial banning can escape its consequence.**

   **Evidence:** D:1190–1193; D:1142–1144, 1167–1170.

   The retry requires **both** an unreached target and more than 1% banned candidates. A path can ban many useful candidates, subsequently reach every support target using poorer substitutes, and receive neither a retry nor a construction-quality limitation. Reaching a prescribed number of positive weights does not establish that exclusion was harmless. Reporting the achieved residual helps readers, but supplies no decision consequence.

   A16.1’s disclaimer against guaranteed baseline strength does not explicitly prohibit an “adequately tuned ECSW” interpretation in this case.

   **Fix:** Whenever the registered material-banning threshold is crossed, either perform the bounded ban-free retry irrespective of target attainment, or attach an explicit construction-quality limitation blocking adequately-tuned/exhaustive non-domination claims. This need not require global NNLS optimality at sparse checkpoints.

4. **NEEDS-RESTATEMENT — The retry’s selection and termination rules are not fully frozen.**

   **Evidence:** D:1190–1193; D:134–138.

   “The path reaching more targets is used” does not choose a path when both reach the same number. For example, both paths may reach \(2r,4r,8r\) but miss \(16r\), while producing different weights at all three usable checkpoints. “Iteration-bounded” also gives no numerical bound or reference to an immutable bound.

   These choices affect the baseline construction and its resource projection.

   **Fix:** Freeze the retry iteration limit and a training-only total order for selecting one whole path, including ties. For example: number of reached targets, then a declared aggregate of checkpoint training residuals, then a fixed path preference. Finish selection before validation and preserve all earlier checkpoint identities thereafter.

5. **CORRECT — A16.4 resolves audit 9 item 7’s accounting requirement.**

   **Evidence:** D:1195–1197; D:675–685, 788–791.

   Smoke and dry-run evidence used to choose the knob or fixtures now counts as validation-informed work. Discarded settings remain recorded. This correctly avoids treating “before production” as synonymous with “training-only.”

   **Fix:** None to A16.4. Actual budget compliance remains conditional: report the measured V ratio, and do not call the factor-1.25 band “equal budget” without the registered ratification.

6. **CORRECT — Governing mathematics and Q1 survive A14–A16.**

   **Evidence:** D:66, 103–108, 122–126, 338–352, 529–534, 578–582, 980–982, 1087–1091; references `refcore.py:64–87,112–132`; combined `t2core.py:35–38`.

   The central operator, residual signs, product-rule Jacobian and ECSW gradient target are consistent. The governing Q-DEIM compression is **untruncated Householder QR**, replacing the original cutoff-SVD formulation. It preserves the assembled DEIM residual norm algebraically and leaves linear terms exact.

   Q1 retains fresh in-job Jacobians, pooled non-stationarity, every-case anchor distance, tight/tighter agreement and validity. The residual-tolerance exemption was withdrawn. The disclosed anchor-stall allowance does not change production Q1 thresholds.

   **Fix:** None. Describe the QR formulation when presenting the final method; do not revive the superseded SVD truncation.

7. **CORRECT — Cohorts and fairness are defensible only within the existing developmental qualifications.**

   **Evidence:** D:354–394, 544–558, 675–685, 788–791, 854–857; references `cohorts.py:35–37`, `refstore.py:132–173`; combined `c3common.py:35`, `t2core.py:35–38`; `burgers2d_film.py:160–177`; `sep_burgers_r3.py:321–332`; bank training JSONs.

   Final selection uses the full development cohort after subset screening. It is not merely selection on dev6/cases 0–5 followed by an independent evaluation. The in-sample, unequal-selection-history wording is therefore necessary.

   Training draws, the 2D early-state selection distinction, 3D training seed/count, IC definitions, dimension-specific step sizes and startup conventions agree with the inspected dependencies. Operator-matched central snapshots remain a disclosed deviation from literal generator reuse, subject to the registered release decision.

   **Fix:** Preserve these qualifications beside conclusions. A16 does not itself establish equal expenditure or ratify the generator deviation.

8. **CORRECT — A16 introduces no new H-base decision-rule defect.**

   **Evidence:** D:228–243, 441–463, 617–619, 644–655, 782–786, 915–921; lane brief:37–46.

   Dominance requires resolved improvements in time and worst error; matching requires the registered equivalence checks. Crossing objectives, reference-limited identical arms, failed picks, missing eligibility and timing failures have explicit outcomes. Worst-error envelopes are conditional reference envelopes, not population confidence intervals.

   Multiplicity and reuse of selection data prevent confirmatory interpretation; the governing descriptive, in-sample, unadjusted wording acknowledges that limitation.

   **Fix:** None for this amendment. Keep coverage counts and limitations beside every “no observed domination” statement.

9. **CORRECT — A14–A16 retain the required controls, but do not establish that they now fire.**

   **Evidence:** D:424–428, 600–606, 875–877, 899–906, 1139–1153, 1155–1158; `runs/smk1/archive/output/ACCEPTANCE.json:266,329,353,667,840–859`.

   Independent residual/Jacobian checks, full-rank all-points recovery and per-family under-sampling controls remain specified. The recorded smoke acceptance is false. Historical 3D ECSW degradation at \(4r\) does not validate the revised ladder, which excludes that level. A Q1-qualified N-swap fixture is sensible but not guaranteed to discriminate.

   **Fix:** Demonstrate the amended controls in the authorized smoke. Preserve family-level blocking and incomplete-phase-I status for unfulfilled gates.

10. **CORRECT — Memory remedies are plausible; production feasibility remains unproved.**

    **Evidence:** D:715–730, 1093–1097, 1110–1120, 1145–1153, 1185–1188.

    Separating maximal-shape pilots from POD certification is valid. Host staging, releasing competing bases and transpose-avoiding products are reasonable remedies, but A16 adds construction through \(16r\) and a possible second path. Neither successful allocation nor total runtime follows from the earlier failed smoke.

    **Fix:** Before production release, certify all allowed branches—including retry limits, checkpoint storage, maximal `none` Jacobians, compilation and production evidence retention—against measured memory, cluster disk and **spent plus projected cost ≤30 GPU-hours**. The withdrawn 13-hour estimate remains withdrawn.

**Overall: NEEDS-RESTATEMENT.** The design blockers are A16.3’s incomplete consequence for material banning and its unspecified retry tie-break/iteration bound. Successful amended controls, a complete resource certificate and the promised production-storage specification remain separate production-release gates.