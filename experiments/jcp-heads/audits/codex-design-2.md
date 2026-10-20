**Overall: NEEDS-RESTATEMENT.** A1 resolves several concrete round-1 defects, but the primary comparator implication is false, F5 still has ambiguous outcomes, and the reservation inventory omits required work.

Read-only throughout: no files changed, no jobs submitted, no scratch arrays copied. I ran one small, in-memory CPU statistical counterexample. Draft implementation was treated only as context.

References: **D** = [heads DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/DESIGN.md); **P** = [heads p1stats.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/p1stats.py); **Q** = the specified sister lane’s `experiments/jcp-qualify/`.

**(a) Disposition of the ten round-1 items**

| # | Item | Verdict on A1 | Justification |
|---|---|---|---|
| 1 | Lane-brief coverage | **NEEDS-RESTATEMENT** | A1.8 supplies the missing CN-R timings and distinguishes accuracy-job versus timing-job qualification populations (D:443–446). However, A1.6 replaces equal-cost comparison with an at-most-cost comparison; it does not guarantee a comparison against an equal-cost-qualified rung, or explicitly report that none exists (D:403–410). Reservation coverage also remains incomplete. |
| 2 | Incumbent recipe and matched K sweep | **CORRECT** | A1 explicitly preserves whitened initialization/optimization, identifies the changing quantities, records training time, and limits interpretation to a recipe under the M=4K policy (D:331–338). This resolves the causal-overstatement objection. |
| 3 | Frozen coefficient-file deviation | **CORRECT** | Source, complete hash, extraction provenance, physical-time correction, `hfit_pick` alignment and conditional-on-one-dataset scope are supplied (D:340–346). This resolves the design issue; it is not an independent verification of the artifact’s bytes. |
| 4 | Head Q1 adaptation | **CORRECT** | Independently certified step stationarity, no budget exits for anchors, explicit unverified-anchor handling, production IC veto and degeneracy diagnostics address the objections (D:348–363). Numerical qualification is expressly separated from physical accuracy. Actual attainability remains a smoke question. |
| 5 | Head Q2 and rule selection | **CORRECT** | Every selectable rung now has both successors; selection requires valid solves and finer-rule agreement. Both methods use the same Gauss-ladder policy (D:365–376), consistent with the sister classifier’s pass/unverified distinction (Q/qscore.py:199–212). Remaining asymmetries are deployment-policy limitations, discussed below. |
| 6 | M=4K and M=4r | **CORRECT** | A fixed-M sensitivity with fresh selection and qualification is registered; headline claims remain policy-scoped. The unsupported mode-set prediction is withdrawn (D:378–383). |
| 7 | Statistical procedures and F5 | **WRONG** | Normalized TOST, superiority precedence and same-seed pooling are repaired (D:386–402). But selecting the lowest-median linear rung does **not** justify “head more accurate than every affordable rung,” and Holm across K does not repair that implication (D:403–424). |
| 8 | Controls | **NEEDS-RESTATEMENT** | Incumbent emitted codes replace `Z_ck`; under-training becomes a sensitivity probe; the cost synthetic and H-cost-N control are corrected (D:426–441). However, one registered qualification control is unreachable through the declared population, and several classification boundary outcomes remain unspecified. |
| 9 | Budget, jobs and memory | **NEEDS-RESTATEMENT** | Pending-core reservation and full retained-table memory coverage are substantial fixes (D:448–454). But `time256`’s retained head/matched-dimension work is neither reserved as mandatory nor listed as deferrable. The expanded workload also needs a new resource certificate. |
| 10 | Prior findings and scope | **CORRECT** | A1 explicitly limits positive findings to validation, retains repair and sealed confirmation as open, corrects the CN-R order wording, and leaves K2 untouched (D:456–459). The remaining F5 decision-logic defects belong to item 7. |

**(b) Attempts to break A1**

1. **Lowest median does not imply hardest paired comparator — WRONG; blocking.**

   A1 selects \(r^*\) by lowest median error, then translates a paired signed-rank win against it into superiority over **every** affordable rung (D:403–408). Those are different orderings.

   Concrete counterexample, evaluated with the copied `p1stats.wilcoxon`:

   - Head errors: \(H_i=i\), \(i=1,\ldots,38\).
   - Linear A: \(A_i=H_i+1\), except cases 20–25, where \(A_i=H_i-10\).
   - Linear B: \(B_i=H_i\).
   - All configurations qualified, reference uncertainties negligible, both linear rungs affordable.

   Median errors are A = **15.0**, H = **19.5**, B = **19.5**. A is therefore selected. The head-versus-A superiority test gives **p = 0.00676929**, with negative median paired difference and complete resolved coverage. Against B, the head is exactly equivalent.

   Thus the registered output says “head more accurate than every affordable rung” even though an affordable rung reproduces its error exactly. This p-value can also pass the first Holm threshold for four K values.

   **Required repair:** either report only “head superior to the validation-selected lowest-median rung,” with appropriately limited inference, or test superiority against every affordable rung. For the latter, a per-K intersection-union test using the maximum component p-value is a natural route, with all required coverage/direction predicates satisfied.

2. **“Validation selection can only favour linear” — WRONG as an inferential guarantee; blocking with item 1.**

   Selecting the smallest observed linear median favours linear on that **descriptive median statistic**. It does not necessarily select the strongest paired signed-rank comparator, nor does it establish that the selected-comparator p-value retains its nominal interpretation (D:405–406).

   Holm controls multiplicity across supplied K hypotheses; it cannot validate an incorrectly constructed per-K hypothesis. Separate descriptive selection from the confirmatory test.

3. **F5 family and decision precedence — NEEDS-RESTATEMENT; blocking.**

   Three gaps remain at D:418–424:

   - “Across the tested K” does not specify treatment of unqualified K, missing costs, empty affordable sets or uncomputable p-values. Freeze the family before results; specify non-rejections for unavailable tests.
   - A positive result requires only that Q-LH-dim is **not** “linear dominates.” An **unqualified** or **unresolved** matched-dimension result therefore satisfies that condition.
   - The negative rule quantifies over “every tested K,” while the next sentence assigns deferred K to “inconclusive.” If K=32 is deferred and the remaining K all match, both readings apply unless precedence is specified.

   A positive result at one fully evaluated K can reasonably coexist with incomplete work elsewhere, but that must be stated. Likewise, decide whether matched-dimension evidence is a required affirmative gate or merely a reported diagnostic.

4. **Same seeds do not make seed-median error and cost one deployable model — NEEDEDS-RESTATEMENT in interpretation; minor if labelled.**

   A1 fixes the earlier seed-0 mismatch (D:386–389). However, the seed supplying the median error can differ from the seed supplying the median cost, and both can change by case. The resulting point is a summary of qualified retrains, not necessarily an attainable individual head.

   This is acceptable as a registered summary estimand. Explicitly label it **conditional on numerically qualified retrains**, report qualification counts, and retain the per-seed joint error–cost comparisons already promised. Do not describe it as the cost and accuracy of one selected deployed model.

5. **dev6 cost versus all38 error — NEEDS-RESTATEMENT; blocking for an unqualified accuracy–cost claim.**

   Costs use dev6; superiority uses eligible dev6+val32 cases (D:388, 396; D:168–172, 205–213). A1.8 labels qualification populations but does not address this mismatch.

   A linear rung can be affordable on dev6 and expensive on val32; the reverse can happen to a head because IC and LM work are case-dependent. Consequently, the result establishes a joint statement about **two different populations**, not matched cost across the accuracy population.

   Either measure cost on the accuracy population or scope every dominance/F5 statement explicitly to “error on eligible all38 cases, cost on dev6.” This does not require discarding the existing protocol, but its headline interpretation must be precise.

6. **Equal-cost coverage and undefined brackets — NEEDS-RESTATEMENT; blocking for the requested comparison.**

   At-most-cost is a useful additional question, but not interchangeable with equal cost (D:403–410). Moreover:

   - No outcome is defined when no qualified \(r_{\rm hi}\) exists.
   - Ties in lowest median error or cheapest eligible rung lack a deterministic tie-break.
   - “Shared cases” should be one fixed comparison population across candidate rungs.
   - An affordable rung can be arbitrarily cheaper than the head.

   Report equal-cost-qualified rungs separately, with an explicit **“no equal-cost rung”** outcome. Keep at-most-cost and upper-bracket analyses as separately named results.

7. **Quadrature policy fairness — CORRECT within its stated scope; minor limitations.**

   The common Gauss ladder removes the earlier asymmetric frozen-rule comparison (D:365–376). Nevertheless:

   - All three head seeds must pass rule selection, whereas each linear rung has one pass requirement.
   - One bad head seed can exhaust the ladder even though two other seeds would support pooled analysis.
   - The selected rule is the smallest passing node count, not necessarily the fastest total query: a larger rule could change solver iterations.
   - Rules are selected under BE and transferred to CN-R, with requalification (D:159).

   These can disadvantage heads or CN-R, but they are explicit policy choices. Replace “cheapest rule” with “smallest passing rule”; do not claim optimization of total online cost or best-achievable performance.

8. **Unreachable qualification control — WRONG; classification repair needed before acceptance tests.**

   A1 defines \(\mathcal S_K\) using configuration-level qualification, which already requires validity and anchor-distance success on every case (D:128–139, 386–389, 415–417). Therefore, when \(|\mathcal S_K|\ge2\), every case automatically has at least two seeds satisfying those per-case predicates.

   The promised end-to-end “>25% unqualified cases” branch cannot arise through that construction (D:436). It can be tested as a unit fixture for `p1stats.h_acc_r_transition` (P:129–147), but not represented as an attainable integrated heads classification. Distinguish those two tests.

9. **Empty populations, zero Jacobians and synthetic specifications — NEEDS-RESTATEMENT; mostly minor.**

   - Error classification says empty population → “no population”; dominance otherwise falls through to “unresolved,” while the control expects “no population” without naming its layer (D:394–400, 435). Define propagation and precedence.
   - The degeneracy ratio \(\sigma_{\min}/\sigma_{\max}\) is undefined for an all-zero Jacobian (D:358–361). Explicitly flag zero and nonfinite spectra.
   - “One-sided exact Wilcoxon” is inaccurate when absolute differences tie: the inherited function uses a normal approximation (D:390; P:30–65). This includes many constant-percentage synthetics.
   - Fix synthetic sample sizes, reference uncertainties and valid timing masks so the expected labels follow deterministically (D:432–439).

   The duplicated-latent fixture itself is sound: identical input and skip rows create identical derivative columns.

10. **Real-data negative controls — CORRECT as gates, outcomes still unverified.**

    A1 appropriately requires valid successor/check solves for the under-resolved-rule control and blocks Q2 if that evidence is unavailable (D:440–441; Q/qscore.py:204–211). The original requirement that both rollout and rho bars fail remains stronger than the ordinary Q2 failure predicate (D:257), but it is not mathematically impossible.

    Neither the supplied 256² reproduction evidence nor the design proves that the required 1024² reproduction, zero-update and under-resolved fixtures will fire (D:326–329 versus D:252–257). These are smoke obligations, not grounds to manufacture a design failure.

11. **Budget inventory remains incomplete — NEEDS-RESTATEMENT; blocking.**

    A1.9 reserves only `train`, `rule1024`, `acc1024`, and `time1024` (D:448–451). Yet the defer list removes only the **linear-ladder subjects** of `time256`, not its head-cost panel (D:283–285, 450–451). Required remaining `time256` work can therefore lose its budget to optional work.

    Reserve every retained obligation, including required `time256` subjects. Exclude the currently submitted job from pending totals to avoid double counting; the sister implementation explicitly does this (Q/ledger.py:168–170). Failed mandatory attempts must restore their pending obligation.

    Also reprice the added fixed-M panel, expanded linear rule search, and all-seed CN-R timings (D:368–380, 443–444). The old estimate at D:280 no longer certifies this workload. The expanded 4096² memory-pilot specification is otherwise adequate.

**Remaining blockers, ranked**

1. **Repair the false “every affordable rung” implication and its F5 statistical test.**
2. **Make F5’s family, qualification gates, deferred-K handling and outcome precedence executable.**
3. **Restore explicit equal-cost coverage and define missing comparator/bracket outcomes.**
4. **Scope accuracy–cost claims to their different case populations, or measure matching populations.**
5. **Reserve all retained mandatory work, especially `time256`, and reprice the expanded design.**

Minor repairs: distinguish unit-only from attainable controls; propagate empty-population outcomes; handle zero Jacobians; specify tie-breaks and synthetic fixtures; describe seed medians and smallest-rule selection accurately.

**Final verdict: NEEDS-RESTATEMENT before production.**