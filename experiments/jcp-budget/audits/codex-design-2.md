<!-- recovered from the codex exec stdout log: the -o file was not written; final message verbatim -->
**Overall verdict: NEEDS-RESTATEMENT.** A1 resolves substantial parts of audit 1, but not every required fix. The remaining blockers include incomplete interaction coverage, controls that do not guarantee their stated verdicts, an omitted reference case, and ambiguous eligibility and missing-data rules.

Read-only audit completed: no files changed, no jobs submitted, no scratch arrays copied.

References below use **D** for [this DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget/DESIGN.md), and **Q**, **C**, **E** for the supplied `jcp-qualify`, `jcp-combined`, and `jcp-references` experiment directories.

1. **Lane coverage — NEEDS-RESTATEMENT**

   **Resolved:** B7 horizons, fresh references, long-horizon Q1/Q2, spectral checks, B8 diagnostics, and the prescribed defer order are present. A1 explicitly protects the ideal anchor and its uncertainty replays from deferral. The listed front members and four 3D configurations match the supplied JSONs; both complete artifact hashes match.

   **Still unresolved:** A1.1’s conclusion that “whichever two components are largest, their interaction is measured” is false.

   - The 3D form panel measures **F×Q and F×T at tight solver**, but not **F×S**. If form and solver are largest, their interaction remains unavailable. Moreover, these form effects have a different background configuration from deployed-end Q/T/S effects. D:416–423, 436–439.
   - The new test-space ingredient **M** is outside the factorial and absent from the candidate-component decision rules. No interaction involving M is scheduled, even if its sensitivity is largest. D:425–427 versus 206–217.
   - The cross term \(\chi_{u_a}\) measures interaction between **total numerical displacement** and ideal dynamics. It does not identify interaction between ideal dynamics and a particular largest ingredient such as Q or T. D:419–423.

   The floor’s orthogonality statement is correct for the fixed span and norm. It does not repair these other coverage gaps. Define the eligible component universe explicitly and provide the missing pair-specific measurements or narrow the claim with coordinator agreement.

2. **B2 mathematics — CORRECT, with decision-rule qualifications below**

   The Möbius definitions and vector telescoping identity are correct. Norms are correctly kept non-additive. D:179–182.

   A1.2 correctly repairs endpoint association:

   - LSPG-deployed full factorial ends at the common GAL ideal.
   - GAL-deployed trajectories reach that ideal through \(\{Q,T,S,I\}\), while their full factorial ends at LSPG.
   - The 3D Q/T/S factorial ends at adaptive LSPG, followed by a separate GAL displacement.

   D:432–439.

   The per-time Pythagorean identity is correct when all terms use the same sampled span and normalizer. In 2D, zero walls make projection on the \(255^2\) interior equivalent to including the walls of the \(257^2\) lattice. The 3D campaign lattice contains \(63^3\) interior points. C/budget2d.py:219–231 implements the corresponding calculation.

   The \(U_\star\) replay formula matches the predecessor, and A1 correctly calls it an **indicator, not a bound**, with maximum-over-time aggregation. D:187–188, 440–441; C/budget2d.py:194–216.

   These identities must remain per-time statements: separately maximized squared terms do not generally satisfy the same identity.

3. **Ingredient switches — NEEDS-RESTATEMENT**

   **Resolved:**

   - Form is explicitly a **form-plus-stopping-policy package**, with a separate tight-tolerance form comparison. D:443–447.
   - Time is explicitly a temporal-discretization package, including 3D adaptive startup. D:448–449.
   - Blocked full-grid LS avoids a dense pseudoinverse and is feasible in principle at \(4096^2\). The existing bank exposes row/column blocks. Q/p1common.py:137–153.
   - Preserving the initialization-derived stopping scale correctly isolates the IC coefficient replacement. This matters because Q/p1core.py:346–350 constructs that scale and line 192 uses it in the LSPG stopping tolerance.
   - The 3D initialization switch is structurally zero on the mesh path. Q/s3common.py:135–143 calls `TB.project`; Q/vendor/quad3d/vendor/burgers3d-retry/tables.py:120–124 solves the mesh normal equations. Projection **error** is not zero.

   **Remaining:**

   - The new M ablation needs explicit implementation semantics and qualification checks. Increasing M while retaining the deployed quadrature and solver may expose quadrature or solver error; it measures enlargement sensitivity under that package, not isolated test-space error.
   - G-B6 must compare **logical ingredients**, allowing derived changes. A literal assertion that only tuple fields named by \(S\) change would reject F’s intended tolerance change and T’s startup/weight changes. D:541–543.
   - Recording \(\operatorname{cond}(H)\) is useful, but A1 gives no conditioning acceptance/fallback rule for normal-equation LS. A small normal-equation residual alone does not certify accurate coefficients for an ill-conditioned Gram matrix. D:450–456.

   The missing 3D F×S interaction remains the coverage problem in item 1; it cannot be supplied merely by relabelling the tight-end panel.

4. **Decision rules — NEEDEDS-RESTATEMENT**

   **Resolved:** explicit ties, unavailable contrasts, ideal-dynamics uncertainty screening, inconclusive mesh invariance, uncertainty-aware sampling, and sampled-time stability wording. D:460–479.

   **Remaining problems:**

   - **Incomplete candidate sets:** excluding an unavailable component from “largest” can award dominance to another component even though the missing component could be larger. Report “largest among available,” or make dominance inconclusive unless the required candidate set is complete.
   - **Empty/small sampling populations:** A1 excludes reference-limited cases but gives no minimum population or zero-case outcome. An adequacy claim must not arise from an empty denominator or a tiny surviving subset. D:472–478.
   - **Growth at \(T=1\):** the completion explicitly screens \(W_1,W_2\), but not the \(W_3/W_2\) ratio. Apply the rule to each adjacent-window pair. D:250–252, 470–471.
   - **Front order:** freeze a common eligible case population across members and windows/routes, the Kendall variant with ties, and an inconclusive outcome when no pair is resolved. Otherwise “no resolved reversal” can pass vacuously. D:256–258, 475–477, 551–552.
   - **Other denominators remain undefined:** near-wall concentration for zero error, deployed/floor IC ratio for zero floor, and mesh component ratios with zero denominators. D:220–221, 269–272, 291–293.

   The basic pair thresholds and sampled-energy threshold are falsifiable. Their problem is incomplete population and availability handling, not an inherently impossible threshold.

5. **Reference generation — NEEDS-RESTATEMENT**

   **Correct:**

   - \(1025^2\) strides are 2/4/8 for \(L=2048/4096/8192\).
   - \(127^3\) strides are 1/2/4 for \(L=128/256/512\), using interior indices \(js-1\).
   - Refinement strides are 12 and 6.
   - Fine-route independent judging, purpose-matched replay, and a disclosed restriction on \(T=1\) refinement are sensible corrections.

   E/refcore.py:225–236 supports the indexing convention.

   **Concrete missing run:** A1 selects the smallest-viscosity case from E0’s 3D cross-check list. That is **val64 case 43**, with \(\nu=0.01000596416798915\), outside v16. Its metadata records this at E/store/3d/val64/ST/c043.json:4,36. The reference inventory still generates only cases 0–15. D:134, 499–500. Add case 43’s long-horizon FD ladder and associated cost; a PS run alone cannot cross-check it.

   **Other unresolved points:**

   - “Full E0 procedure” does not inventory E0’s spatial escalation ladder. E/refjob.py:48–49,459–460 supports a second ladder reaching 3072 in 2D and 384 in 3D. Either include its trigger and cost or explicitly register its omission.
   - Specify which route/window failures trigger tight retries and refinement now that multiple independent judges exist. Merely saying “same E0 procedure” leaves competing judge outcomes unresolved.
   - G-R2 becoming “not evaluable” has no explicit eligibility consequence. D:489–494.
   - G-R2 must distinguish matching individual levels from matching the **reference actually used for scoring**. A later horizon can select a different finest level after refinement. B8 should compare routes from the same FD realization, or explicitly account for that additional reference difference.
   - The new GPU bars are defensible stopping criteria, but not established cross-GPU guarantees. Also exempt \(t_0\) from the \(\varepsilon(t_k)\)-based bar: E0 stores uncertainty for evolved outputs only. E/verdict.py:95–102,129–132.

   Generalizing the horizon requires updating clipping indices, stage counts, canonical IC routes, metric IDs, and all FD/PS judge loops—not just output-buffer length. E/refcore.py:155–165,205–245 and E/verdict.py:95–132 show the hard-coded assumptions.

6. **Must-fail controls — WRONG as a complete specification**

   | Control | Verdict | Finding |
   |---|---|---|
   | **C-A** | **WRONG** | The triangle inequality guarantees \(s\ge1.2\), but that does **not** guarantee rejection by A1.4’s uncertainty-aware rule. |
   | **C-E** | **NEEDS-RESTATEMENT** | The scaling identity is algebraically correct for the unnormalized balance difference. The registered diagnostic is normalized by \(E_{\rm ref}(t_0)\); the stated prediction omits that division. |
   | **C-F** | **NEEDS-RESTATEMENT** | Orthogonalizing against both span and residual repairs the missing cross term. The amplitude still does not guarantee the decision code labels floor largest. |
   | **C-I** | **NEEDS-RESTATEMENT** | The expected norm is correct, but it need not differ from the unperturbed LS-distance norm. Compare perturbed versus unperturbed decoded IC directly for a guaranteed nonzero plumbing signal. |
   | **C-S** | **CORRECT for the named 2D fixture, conditional on smoke** | Named fixture, valid tight comparator, and intentional-invalid-run exemption are appropriate. |
   | **C-T** | **CORRECT for the named 2D fixture, conditional on smoke** | Prior BE temporal sensitivities support the predicate; propagation to the new implementation still requires demonstration. |
   | **C-Q** | **NEEDS-RESTATEMENT** | The cited established fixture compares Gauss32 against **Gauss64**, whereas the new \(I_Q\) compares against Gauss320. The predecessor result is supporting evidence, not proof of this changed predicate. |
   | **C-R1** | **CORRECT conditional on prerequisite checks** | Explicit order-check rejection repairs dependence on the final outcome string. Freeze the actual case and dimension-specific fixture. |
   | **C-R2 / C-score** | **CORRECT for identity rejection** | Identity-tag rejection is appropriate. Keep numerical time-shift difference descriptive unless a discriminating numerical fixture is frozen. |

   Details of the three most consequential failures:

   **C-A:** rejection requires
   \[
   e^{\rm fine}-e^{257}>0.1e^{257}+2(\varepsilon^{\rm fine}+\varepsilon^{257}).
   \]
   A 20% increase is insufficient when uncertainty exceeds the remaining margin. For example, \(e^{257}=10^{-3}\) and both uncertainties \(10^{-4}\) pass the coarse resolved-error screen, yet a 20% increase remains “consistent.” Choose amplitude against the **actual decision threshold**, with positive slack. D:472–474,520–524.

   **C-F:** ten times the largest raw component does not necessarily exceed the tie/resolution thresholds involving \(2\varepsilon_{\rm ref}\) and \(3U_\star\). Thus the identity can pass while “largest” returns tied or unresolved. Include these thresholds in the construction. Define behavior for zero residual or vanishing orthogonalized seed. D:464–467,509–515.

   **C-E:** divide the closed-form expression by \(E_{\rm ref}(t_0)\), or explicitly test the raw balance before normalization. Replace an undefined relative comparison at zero with a declared absolute/relative convention. D:278–280,516–519.

   Finally, audit 1 requested explicit dimension-specific fixtures. A1 still supplies only 2D C-S/C-T fixtures while promising every 3D control in `smk2`. Freeze the 3D inventory or explicitly mark controls inapplicable. D:350,528–536.

7. **Scoring contract — NEEDS-RESTATEMENT**

   **Resolved:** development/validation populations only, ST primary, restricted S use, carried post-hoc labels, GAL diagnostic status, long-horizon Q2 successor convergence, and explicit Q2 population scope. D:65–98,537–550.

   **Still unresolved:** A1’s cross-check status handling is not fully equivalent to E0 eligibility.

   E/refstore.py:132–141 requires:

   - accepted FD reference;
   - validated dimension;
   - PASS for a designated cross-check case.

   A1 says an **unvalidated** dimension gets provisional labels, but does not clearly prohibit ranking/verdicts or exclude a designated unresolved cross-check case. D:501–503. Explicitly separate descriptive scores, provisional comparisons, and eligible verdicts.

   Also define \(W_3\) status: no spectral cross-check is scheduled there. Passing at \(T=0.5\) does not validate the later window.

   The common-population and tie issues in item 4 also affect front-order claims.

8. **Compute, memory and storage — NEEDS-RESTATEMENT**

   **Resolved:** lane cap 3, campaign cap 8, outstanding-job reservations, explicit C3 release dependency, combined-path \(4096^2\) memory pilot, and separate retention of reference data. D:557–571.

   **Smoke allowance remains unverified.** A1 asserts a coordinator decision of “each ≤2 GPU-h,” overturning audit 1’s aggregate interpretation. The supplied binding briefs do not establish that interpretation. The canonical lab-log entry says “≤2 GPU-h smokes,” without resolving it. Cite the actual coordinator clarification; absent that evidence, the claimed reconciliation is not auditable.

   The original table totals 29 GPU-hours, but A1 adds M ablations, form-panel runs, and substantially broader check-rule replays without updating the inventory. The missing case-43 FD ladder adds further cost. Treat 29/36 hours as superseded estimates until repriced.

   Nominal f64 storage before temporaries:

   | Object | GiB |
   |---|---:|
   | \(4095^2\times512\) bank | 63.97 |
   | \(4095^2\times384\) retained prefix | 47.98 |
   | Gauss320 tables, \(r=384,M=1536\) | 1.76 |
   | Gauss640 tables, same setting | 7.03 |
   | Gauss640 tables, \(r=384,M=2048\) | 8.59 |
   | Gauss56³ tables, \(r=512,M\approx2048\) | 4.02 |
   | Gauss64³ tables, same setting | 6.00 |

   The proposed pilot is appropriate; these scalar counts are not peak-memory certificates. Include 3D construction peaks, reference refinement buffers, caches, and retained fine arrays in certificates.

   Cluster retention is compatible with the common brief’s large-array rule. Local free space was approximately 130 GiB; this audit created no scratch.

9. **Other failure paths — NEEDS-RESTATEMENT**

   **Important remaining validity hole:** A1 adopts Q’s `run_valid` for B2 corners. That function intentionally permits budget exits and relies on separate certification to judge them. Q/p1common.py:294–309 explicitly documents this. B2 does not clearly impose that additional requirement on every tight/ideal/check corner.

   Consequently, a finite but unconverged tight corner could be treated as a valid numerical anchor and generate a large ingredient effect. Require verified convergence for numerical anchors and check runs, while preserving appropriate deployed-solver diagnostics and control exemptions.

   **Resolved audit-1 findings:** A1 corrects dimensional energy weights and boundary stencils, withdraws the temporal-quadrature cancellation claim, limits sampling adequacy to the chosen finer route, expands G-B4 across corner types, and completes artifact hashes. D:553–555,574–583.

   Two qualifications remain:

   - G-B6 verifies configuration bookkeeping, not independently that the intended numerical operator was executed. Preserve inherited operator/residual checks alongside it.
   - G-B4 permits attribution only on its checked population, while the headline budget still promises dominant-component classifications over 38 cases. State exactly which full-population labels remain descriptive and which verdicts are withheld outside the checked subset.

Audit-1 required-fix closure:

| Audit-1 fix | Round-2 status |
|---|---|
| 1. Repair manufactured controls | Partial; C-A/C-E/C-F remain defective |
| 2. Freeze empirical fixtures and smoke populations | Partial; comparator changes and 3D fixtures remain |
| 3. Separate endpoints and protect anchors | Resolved |
| 4. Define switches, stopping scale and blocked LS | Substantially resolved; M and conditioning need completion |
| 5. Complete decisions and form interactions | Partial |
| 6. Restore Q2 and quadrature-corner checks | Resolved in scope; validity/population consequences need completion |
| 7. Complete reference and eligibility procedure | Partial; missing case 43 and status/retry rules |
| 8. Correct physical diagnostics and limitations | Resolved, except C-E normalization |
| 9. Reconcile compute and retention | Partial; smoke authorization and revised inventory remain |

**Overall verdict: NEEDS-RESTATEMENT. Required fixes:**

1. Complete the largest-component interaction contract, including M, 3D form–solver coverage, and ingredient-specific dynamics interactions.
2. Make C-A’s amplitude exceed the uncertainty-aware rejection threshold; repair C-F’s dominance guarantee and C-E’s normalization.
3. Freeze the complete per-dimension control inventory and distinguish inherited fixtures from changed comparators.
4. Add val64 case 43’s long-horizon FD references and update the spectral escalation inventory.
5. Specify route/window retry selection, reference-version pairing, G-R2 non-evaluability, and cross-check eligibility—including \(W_3\).
6. Complete missing-candidate, empty-population, adjacent-window, common-population, tie, and zero-denominator rules.
7. Require verified convergence for tight/ideal/check anchors; align full-population claims with the checked quadrature population.
8. Complete M-switch semantics, LS conditioning handling, and logical-ingredient assertions.
9. Cite the coordinator’s smoke allowance and reprice the amended run inventory with complete memory certificates.
