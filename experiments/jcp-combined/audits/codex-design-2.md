**Overall verdict: NOT READY.** A1 repairs much of the original design, but selection sequencing, quadrature qualification, H4 eligibility, and several mandatory controls still need amendments. No files were modified, including the lab log.

I reviewed [DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/DESIGN.md), the [27-item audit](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/audits/codex-design-1.md), and the named solver and runner code. A1 takes precedence throughout.

**Disposition of all 27 original items**

“RESOLVED” for an originally CORRECT item means its valid statement survives the amendment.

| # | Verdict | Assessment and concrete remaining fix |
|---|---|---|
| 1 | **RESOLVED** | The LMM residual and simplified Jacobian remain correct for the quadratic off-mesh operator. |
| 2 | **RESOLVED** | A1.1 correctly says “no additional Jacobian contraction,” specifies accepted-state caches, and charges their arithmetic to timing. |
| 3 | **RESOLVED** | A1.2 records predictor selection, rejected sweeps, damping and stationarity; accuracy is measured rather than inferred from extrapolation. |
| 4 | **RESOLVED** | The clipped, time-based adaptive-start rule handles the five-step trajectory and preserves the deployed count at \(h_0\). Interpret “W1024” here as all prefixes of that bank. |
| 5 | **PARTIALLY** | A1.2 preserves BE arithmetic and supplies a numerical fallback, but A1.10 still calls G-FS1 a strict gate. **Fix:** explicitly distinguish exact reproduction from approved numerical equivalence, and specify whether fallback acceptance also requires matching eligibility and verified exits. |
| 6 | **RESOLVED** | A1.1 withdraws formal-order claims for new settings and makes the hypotheses empirical accuracy/cost claims. |
| 7 | **PARTIALLY** | Oracle, refinement sequence and fixed/adaptive agreement are separated correctly. Mutation coverage is not scheme-specific; some prescribed mutations are algebraically inert. **Fix:** assign each mutation a named discriminating scheme and case, with an independent expected residual. See N13. |
| 8 | **RESOLVED** | Chaining solver then form changes at tight tolerance removes the undefined fixed-sweep GAL combinations; initialization is held fixed. This resolves definability, not numerical qualification of every tight solve. |
| 9 | **RESOLVED** | Package sensitivity is correctly renamed and accompanied by within-scheme tight refinement. |
| 10 | **RESOLVED** | The orthogonal identity remains valid pointwise, provided one common normalized inner product and the actual decoder span are used. |
| 11 | **RESOLVED** | A1.4 separates per-time identities, independently maximized summaries, and the orthogonal pair at a common maximizing time. Grouped sensitivity bars replace additive stacks. |
| 12 | **RESOLVED** | The discrepancy is explicitly composite; initial-fit error is separate and causal claims are withdrawn. |
| 13 | **PARTIALLY** | Signed mixed differences are a substantial improvement. The old factor-of-two “interacting” label remains unamended, and near-zero ratios remain unstable. **Fix:** call endpoint ratios context-sensitivity diagnostics, apply a numerical resolution floor, and qualify all mixed-difference corners. |
| 14 | **PARTIALLY** | Temporal and solve refinement now cover the stated anchor populations. Quadrature qualification remains dev6-only in 2D and selection-arm-only in 3D. **Fix:** qualify each actual ideal/budget trajectory on its claimed population, or restrict the convergence claim to checked cases and arms. |
| 15 | **PARTIALLY** | Separate quadrature/solver axes and all-64 escalation fix the original causal overreach. The operator/rollout comparison uses incompatible relative metrics, and a failed adaptive probe can conflict with a passing fixed-solver pair. **Fix:** separate selection status from diagnostic availability and remove the quantitative amplification inference. |
| 16 | **PARTIALLY** | Family cost is deterministic and finalist checks are added. They cover only 32 of 64 3D cases, do not qualify the reference under each finalist, and do not certify fallback arms. **Fix:** verify finalist/reference pairs on all selection cases; define and verify an explicit fallback inventory. |
| 17 | **PARTIALLY** | “Heuristic temporal tie threshold” is accurate. “The worst case” is ambiguous when arms have different maximizers; median tolerances are a policy, not propagated uncertainty. **Fix:** define symmetric comparison rules and separate heuristic ties from E0 uncertainty propagation. |
| 18 | **RESOLVED** | The new H3 criterion is non-tautological, uses a common time, and has a scheduled deployed-rule anchor. Explicit normalization and a margin-resolution rule are still needed; see N7. |
| 19 | **PARTIALLY** | Scheme-by-setting selection and H2’s two transitions are now explicit. Missing medians in the global sum, tied steps, and absent global-scheme arms remain undefined. **Fix:** specify a complete deterministic ranking and unavailable outcomes. |
| 20 | **PARTIALLY** | Anchors cover all 64 cases, selected arms receive budgets, and H6 separates deployment from diagnostics. H4 still has ambiguous resolved populations, insufficiently specified comparator qualification, and conflicting selection cost caps. **Fix:** define common comparison masks and one selection rule per H4 outcome. |
| 21 | **PARTIALLY** | Tolerances, empirical-control handling and damping flags are improved. B0’s coordinate-shift control need not break the identity; strict G-FS1 conflicts with its fallback. **Fix:** replace B0 controls with algebraically guaranteed inconsistent constructions and reconcile G-FS1 acceptance. |
| 22 | **RESOLVED** | The registry now pins coordinates, boundaries, cases, hashes, times and physical normalization; initialization is preserved and decoding chunked. |
| 23 | **PARTIALLY** | FOM fields and rescoring scope are now specified. Float32 storage cannot generally satisfy the stated score-reproduction tolerance, and “frozen selections” does not identify all frozen H5 decisions. **Fix:** preserve lossless fields or bound storage error, and freeze arm identities separately from reference-dependent matching. |
| 24 | **PARTIALLY** | Same-job 2D accuracy/timing, invocation arrays and common 3D costs are improvements. The common panel requires selected arms before selection; common-baseline ratios also need an explicit mapping to setting-relative selection costs. **Fix:** time all candidates before selection or predefine a same-job selection phase; specify direct pairings or the exact cost statistic. |
| 25 | **RESOLVED** | Comparator inventory, calibrated-ladder scope, unmatched outcomes and FOM-CN qualification are explicit. Remaining tie and rescoring issues are covered above. |
| 26 | **PARTIALLY** | A measured inventory gate appropriately replaces confidence in the placeholder estimate. It must include conditional successors, all qualification runs, fallback timing and memory/cache management. **Fix:** enumerate those branches before smoke-based admission; do not extrapolate only deployed-query timing. |
| 27 | **RESOLVED** | A1.12 correctly limits the lane to provisional development evidence and preserves global sealing. |

**Fresh adversarial findings**

1. **NEEDS-RESTATEMENT — The job sequence requests selected arms before selection exists.**

   A1.0 freezes selection after `c3grid`; A1.11 puts a common panel containing “selected arms” inside `c3grid`. The literal sequence cannot identify those subjects.

   **Fix:** put every potentially selectable arm in the common panel, then select offline. Alternatively, define a deterministic preliminary selection phase inside the same allocation, followed by timing and a final freeze. State which measurements drive selection.

2. **NEEDS-RESTATEMENT — Post-selection budgets are legitimate, but their fallback cannot silently replace the selected method.**

   Freezing the selected arm before inspecting its budget avoids retrospective reselection. However, A1.5 then reports a larger-rule fallback without defining its ladder for all settings, its own qualification, or its cost. A fallback changes the deployed algorithm and can invalidate equal-cost/equal-accuracy status.

   **Fix:** freeze explicit fallback arm IDs, including terminal “unavailable” outcomes. Report the original arm as failing its qualification. Treat a fallback as a separate diagnostic unless it receives full accuracy, solver, quadrature and same-job timing qualification. Preserve the original selection verdict.

3. **CORRECT — The tight-level form chain removes the undefined cross-product.**

   The two measured changes
   \[
   u_a\longrightarrow u_{a,\mathrm{tight}}
   \longrightarrow u_{a,\mathrm{tight},\mathrm{GAL}}
   \]
   are well-defined. No fixed-sweep GAL algorithm is required.

   **Implementation requirement:** use one stored initial coefficient vector and verify both solves independently. `t2core.py` correctly distinguishes LSPG stationarity from GAL root-residual acceptance; a finite return alone is insufficient. Describe the second distance as *form sensitivity at tight tolerances*, not an exactly isolated form error if solve sensitivity is material.

4. **CORRECT — The mixed-difference formulas are valid package-interaction contrasts.**

   Each expression has the correct four-corner signs. Time × solver intentionally includes the changing adaptive-start fraction as part of the temporal package.

   **Implementation requirement:** include all four corners explicitly in the inventory, with identical initial coefficients, common output times, and unchanged nonvaried settings. Do not substitute the GAL ideal for an LSPG corner merely because both use BDF2 at \(h_\star\). Failed corners make the contrast unavailable.

5. **WRONG — A selection-solver-converged rule does not qualify the 3D ideal or every finalist.**

   A1.7 checks fixed BE at \(h_0\) and fixed CN at \(2.5h_0\). A1.4 subsequently uses that rule for fine-step GAL-BDF2 ideals, tighter replays and other budget corners. Agreement under the selection solvers does not establish agreement for these different trajectories.

   The first admissible pair can also be \((56^3,48^3)\), even if the already scheduled \(56^3/64^3\) comparison fails. That supports adjacent-rule agreement, not the stated “fine enough that a finer one changes the rollout by less than the bar.”

   **Fix:** require the finer check for the accepted reference, and qualify every \(r_\infty\)-dependent budget trajectory under its actual form/scheme/solver. Include the `M2_512` bank explicitly; a check of W512 does not transfer to another bank. Limit 2D convergence claims to dev6 unless qualification expands beyond it.

6. **NEEDS-RESTATEMENT — J4 needs separate operational and diagnostic failure states.**

   A1.7 accepts a converged pair using both fixed selection solvers, then says nonfinite or ineligible runs imply “unresolved.” It is unclear whether an adaptive diagnostic failure vetoes otherwise eligible fixed trajectories.

   Also,
   \[
   \frac{\|N_p(c)-N_q(c)\|}{\|N_q(c)\|}
   \]
   is a relative tested-operator difference, whereas rollout distance is a physical-field difference normalized by \(\|u_0\|\). Comparing their numerical magnitudes does not identify amplification.

   **Fix:** define `selection_pair_status` and `adaptive_diagnostic_status` separately. Report operator and trajectory differences separately, with denominator-zero handling. To claim quantitative amplification, use a common field-equivalent perturbation metric or an actual controlled propagation experiment.

7. **CORRECT — H3’s replacement algebra is genuinely non-tautological.**

   With all vectors normalized by the same \(n_0\),
   \[
   e_{\rm BE}^2=e_\infty^2+\|\delta_{\rm BE}\|^2
   +2\langle\delta_{\rm BE},\varepsilon_\infty\rangle.
   \]
   Consequently, the new condition is equivalent to
   \[
   -2\langle\delta_{\rm BE},\varepsilon_\infty\rangle
   \ge 1.5\|\delta_{\rm BE}\|^2.
   \]
   This imposes a real requirement.

   **Required clarification:** A1.9 defines raw field differences but squares normalized errors; write the normalization explicitly. Define deterministic ties in \(t^\ast\), zero-vector angle handling, and whether ST and TX each choose their own \(t^\ast\). Withhold confirmation when anchor uncertainty can reverse the inequalities. This establishes cancellation at the chosen time, not necessarily an explanation of the CN trajectory’s maximum error.

8. **NEEDS-RESTATEMENT — H4’s relative solver bar is sensible, but eligibility and selection remain incomplete.**

   The condition
   \[
   d_{\rm solve}\le\min(2.5\times10^{-4},0.25d_{\rm anchor})
   \]
   prevents solver discrepancy from dominating a small claimed error. It does not itself make the adaptive comparator trustworthy; that run must verify and have negligible tighter-solve sensitivity.

   A1.6 defines resolution per arm, while A1.9 requires 80% resolution only “for the comparator.” It does not say whether candidate-unresolved cases are excluded, counted, or treated as censored. Candidate-specific exclusions can change medians and worst cases.

   Original H4b also allows cost ratio \(\le1.1\), while its accurate-equal-cost selection uses \(\le1\) and is not explicitly restricted to \(h_0\).

   **Fix:** define a fixed comparison population or conservative bounds for unresolved candidates; require at least 52 of 64 comparator cases resolved. Verify the BE comparator’s solver sensitivity too. Distinguish the H4b witness from the equal-cost selected arm, or unify their step and cost restrictions.

9. **NEEDS-RESTATEMENT — Aggregate ties are not completely specified.**

   Two arms may attain their worst errors on different cases. Using the threshold of “the worst case” is ambiguous and can make comparison asymmetric. Pairwise tolerance ties are also nontransitive, so “wins most settings” needs an explicit algorithm.

   Median reference thresholds are acceptable as a declared heuristic, but are not generally uncertainty bounds for medians, particularly after E0 replaces the registry values.

   **Fix:** define symmetric heuristic thresholds, an explicit nondominated/tied set, and deterministic step tie-breaking. For verified casewise uncertainty \(\eta_i\), propagate intervals:
   \[
   e_i^{\rm true}\in[\max(0,e_i-\eta_i),e_i+\eta_i],
   \]
   then take medians or maxima of the endpoints. Specify missing-arm treatment in global sums and unavailable H1/H2 outcomes.

10. **WRONG — One mandatory B0 negative control can preserve B0 exactly.**

    Replacing the reference by a shifted vector does not break
    \[
    \|v-y\|^2=\|(I-P)y\|^2+\|v-Py\|^2
    \]
    when \(P\) remains orthogonal and \(v\in\operatorname{range}(P)\). The identity holds for every \(y\), including a wrongly shifted reference. A perturbed basis also has no guaranteed \(10^{-6}\) effect.

    **Fix:** test coordinate alignment independently against known node values. For B0, construct a deterministic fixture and corrupt only one term, use a known nonorthogonal projector, or add a controlled out-of-span component to \(v\). Verify the expected defect analytically.

11. **NEEDS-RESTATEMENT — FOM rescoring needs a precision contract and a decision contract.**

    Saving full fields fixes the spatial-availability problem, but float32 serialization can change scores far beyond \(10^{-13}+10^{-10}|y|\). The existing FOM runner already distinguishes float32 archives from float64 timing evidence; that distinction cannot disappear under the new all-table rescoring promise.

    “Frozen selections” also leaves unclear whether an updated reference may change FOM matches or the H5 maximizing ROM arm.

    **Fix:** save lossless float64 fields for strict reproduction, or record a bound on serialization error and use a separate archive-reproduction gate. Pin mesh coordinates and shape explicitly. Freeze deployed ROM selections, and separately label any reference-dependent rematching or maximization over the frozen candidate inventory.

12. **NEEDS-RESTATEMENT — Common-baseline timing does not automatically provide every required paired statistic.**

    In 2D the panel baseline is `acc` BE, while selection requires ratios against each setting’s own BE. In general, dividing two medians of ratios is not the median of the desired paired ratios. H5 likewise needs a declared FOM/ROM speedup aggregation.

    In 3D the proposed common baseline W512 may be unavailable under A1.7. “Drift reported” in A1.11 also needs reconciliation with the inherited 10% gate.

    **Fix:** directly bracket the required setting-relative and FOM/ROM comparisons, or define one explicit case/repetition statistic from saved timings. Pre-register a baseline fallback, common-panel cases, randomization seed and drift-failure handling. Preserve identical timed work and synchronized completion.

13. **WRONG — `wrong_weight` is not a discriminating CN mutation.**

    A1.3 replaces \(\beta_0\) with \(\beta_1\) in \(S\). For CN, both equal \(1/2\), so neither the trajectory nor an independent correct residual check must change. For CN-R it changes only startup; BDF2 is the straightforward discriminating choice.

    **Fix:** assign `wrong_weight` to BDF2 or BE; assign `stale_fn` to CN and history/startup mutations to BDF2. State that each mutation must pass its designated negative-control fixture, rather than requiring every mutation to discriminate every scheme.

The amendment is substantially stronger, but these are executable-contract and inference problems, not editorial polish. **Overall verdict: NOT READY.**