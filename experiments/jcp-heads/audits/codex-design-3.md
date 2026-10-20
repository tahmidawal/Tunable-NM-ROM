**Overall: NEEDS-RESTATEMENT.** A2 resolves the original comparator counterexample and most round-2 defects. Two issues remain blocking: the claimed family-wise error control after data-dependent comparator selection, and the handling of unavailable costs in the fixed-population comparison.

Read-only throughout: no files changed, no jobs submitted, no scratch files or large arrays copied. I ran small, in-memory CPU checks using the copied statistics module. Draft implementation is not being certified.

References below: **D** = [heads DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/DESIGN.md); **P** = [p1stats.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/p1stats.py). Line references refer to those files.

**(a) Disposition of every round-2 finding**

The numbering follows round 2’s eleven adversarial findings, covering its five ranked blockers and minor items.

| Round-2 item | Verdict on A2 | Justification |
|---|---|---|
| 1. Lowest-median comparator does not establish superiority to every affordable rung | **CORRECT** for that defect | A2 tests every included rung and uses the maximum component p-value; the lowest-median rung is descriptive only. D:473–478. The old counterexample now has component p-values approximately 0.00676929 and 1, hence aggregate p = 1. |
| 2. Validation selection “can only favour linear” | **CORRECT** for the original assertion | The offending inferential use of the lowest-median comparator is removed. D:478. A separate selection issue remains for the *affordable set*, discussed below. |
| 3. F5 family, missing tests, affirmative dimension gate, deferred sizes and precedence | **CORRECT**, with a minor lifecycle clarification | The family is frozen, unavailable error tests receive p = 1, unresolved/unqualified dimension comparisons block positives, and lane outcomes have explicit precedence. D:483–496. Later budget deferrals need clarification against that freeze. |
| 4. Seed medians need not describe one deployable model | **CORRECT** | A2 explicitly defines a summary conditional on qualified retrains, requires qualification counts and repeats comparisons per individual seed. D:498–501. |
| 5. dev6 costs versus all38 errors | **CORRECT** for the population mismatch | At 1024², timings now cover all38 and comparisons use the fixed E0-eligible population. The remaining 256² mismatch is explicitly scoped. D:465–467, 503–508. Invalid timing coverage remains unspecified. |
| 6. Equal-cost coverage, absent brackets and comparison population | **CORRECT**, with minor ordering details | Equal-cost, at-most-cost and bracket results are separated; empty sets and absent brackets are labelled; the comparison population is fixed. D:465–481. “Costliest” still needs its scalar ordering statistic specified. |
| 7. Quadrature policy fairness | **CORRECT** as a declared policy | Two passing seeds suffice; the selected rule is called “smallest passing,” and BE selection/CN-R transfer is explicit. D:510–513. This is not optimization over all head deployments. |
| 8. Unreachable qualification control | **CORRECT** | The >25% unqualified-case branch is now explicitly a unit test, not an integrated heads outcome. D:516–517; P:129–147. |
| 9. Empty populations, zero Jacobians, Wilcoxon wording and synthetic fixtures | **CORRECT** | Propagation and precedence, degenerate spectra, approximation wording and fixture parameters are supplied. D:518–527. Small CPU checks confirm the stated superiority/equivalence behavior. |
| 10. Real-data negative controls | **CORRECT** as gates; execution unverified | A1 still requires valid successor/check solves and blocks Q2 when the control cannot be demonstrated. D:440–441. Nothing in A2 makes the reproduction or under-resolution controls mathematically impossible; their success remains a smoke obligation. D:252–257. |
| 11. Mandatory reservation inventory and repricing | **CORRECT** | Required `time256` subjects are reserved; the submitting attempt is excluded from pending totals; failed mandatory jobs return to pending; estimates are updated. D:529–535. Projections sum to 22.5 GPU-h; multiplying all by 1.5 gives 33.75 GPU-h. This is an estimate, not the required smoke certificate. |

The previously accepted recipe, frozen-data provenance, Q1 anchor, fixed-\(M\) sensitivity and validation-only scope repairs remain intact: D:331–383, 456–459.

**(b) Attempts to break A2 and its interaction with A1**

**1. Adaptive affordability does not inherit the fixed-set intersection–union guarantee — WRONG as an unconditional FWER claim; blocking.**

For a **fixed** set of comparator hypotheses, taking their maximum p-value is the appropriate intersection–union construction. A2 fixes the earlier logical error on precisely that basis.

But its set is not fixed:

- Membership depends on estimated costs and numerical qualification.
- Costs and errors are measured on the same cases.
- Those quantities can be related through case difficulty, initial-fit behavior and solver work.
- A2 does not establish conditional validity of the error p-values after these selections. D:465–487.

Holm across \(K\) requires valid input p-values. It does not itself supply that missing selection guarantee.

An illustrative statistical counterexample: suppose eight component null p-values are individually uniform, but correlated cost measurements cause only the smallest-p component to be classified affordable. The maximum over the selected singleton is then the minimum over eight tests. Its probability of falling below the four-\(K\) first Holm threshold, 0.0125, is

\[
1-(1-0.0125)^8 \approx 0.0957.
\]

With the other three \(K\) tests unavailable and assigned p = 1, the nominal 5% family procedure can therefore reject with probability exceeding 5%. This is an illustration of the missing guarantee, not an allegation that the implementation reads error p-values during cost selection.

**Required repair:** distinguish the valid *observed-set logical comparison* from a statistically controlled confirmatory claim. Either justify conditional validity, use independent selection/inference data with the relevant qualification and seed-selection rules addressed, or use simultaneous inference over a predeclared component family that remains valid when a subset is subsequently selected. Simply applying Holm across the four selected-set aggregates is insufficient.

This also limits interpretation: the registered result concerns **qualified rungs classified affordable**, not every truly affordable rung or every possible linear deployment.

**2. Missing or invalid timing observations have no complete F5 propagation rule — NEEDS-RESTATEMENT; blocking.**

A2.4 supplies one repeat per val32 case and promises a fixed comparison population. The inherited timing protocol permits invalid measurements after drift retries or identity failure. D:238–246, 503–508.

Numerical qualification does not guarantee a valid timing measurement. Consequently, a qualified rung can have:

- a computable error comparison;
- an eligible case with no valid fused timing;
- no computable cost interval or affordability classification.

A2.2 assigns p = 1 for unavailable **component p-values**, but does not say what happens when an otherwise qualified comparator cannot enter the affordable set because its **cost** is unavailable. D:485–487.

The consequential ambiguity is whether to:

- remove the case, violating the fixed-population contract;
- remove the rung, potentially making “every affordable rung” easier to satisfy;
- alter the head’s seed pool, breaking the shared error/cost estimand;
- or block the affected conclusion.

**Required repair:** register an explicit timing-completeness predicate and propagation rule. For example: use valid repeats only; require coverage of every eligible comparison case for the declared seed population; preserve missing-cost rungs as “affordability unknown”; and make dependent F5 conclusions unavailable unless a registered conservative treatment establishes them. Register any rerun policy and its reservation obligation.

The particular repair is a design choice. Leaving the decision to analysis is not.

**3. Can the positive and negative at-most-cost labels co-occur? — CORRECT under the amended precedence.**

For a nonempty, fully computable affordable set:

- A positive requires every component to satisfy head-superiority predicates and the aggregate threshold.
- The alternative label requires some component’s error label to be “linear more accurate” or “matches.”
- A1 gives superiority precedence over equivalence. D:390–395, 473–478.

Thus a small but significant head improvement can have an equivalence **flag**, yet its error **label** remains “head more accurate.” That does not simultaneously trigger the alternative result.

My checks also confirm:

- identical errors: equivalence, no superiority;
- 5% lower head errors: superiority plus equivalence;
- 10% lower head errors: superiority, without equivalence under the specified minimum-error margin.

These are consistent with A1.7 and A2.6, rather than contradictory controls.

**4. F5’s Holm and negative readings — CORRECT structurally, conditional on repairing item 1.**

The non-rejection defaults and affirmative Q-LH-dim gate are now executable. Positive results take precedence over incomplete work elsewhere. D:483–496.

A \(K\) that passes unadjusted comparisons but fails Holm does **not** become a negative result automatically; absent an alternative resolved outcome, it leads to inconclusive. That is appropriate.

Likewise, “no benefit shown” is not a multiplicity-controlled proof that every head is equivalent or inferior. A2’s restrained wording is acceptable; it must not later become “absence of benefit established.”

**5. Two-of-three rule selection and seed summaries — CORRECT within scope; minor limitations.**

The revised policy removes the “one bad seed exhausts the ladder” defect. It can nevertheless favor or disadvantage heads:

- It permits a qualified subset of retrains rather than requiring all retrains.
- The smallest rule passing Q2 for two seeds might leave fewer than two seeds after full Q1 qualification, even if a larger rule would support a qualified pair.
- A third seed that would work at a larger rule is excluded at the selected rule.
- The resulting pooled error–cost point remains potentially unattainable by any individual model.

These are consequences of the registered policy, not contradictions. D:348–363, 386–389, 498–513. The qualification counts and per-seed joint comparisons are necessary, and A2 requires them.

A1’s fixed-\(M\) panel also remains necessary: the headline compares recipes under \(M=4K\), not latent dimension in isolation. D:378–383.

**6. Budget deferral after the F5 freeze — NEEDS-RESTATEMENT; minor.**

A2.2 freezes the family at `acc1024` submission and requires deferral decisions before accuracy results are read. A2.7 still allows removing \(K=32\) subjects from every job not yet submitted. D:483–485, 529–533.

The conservative reading is workable: after the family freezes, later inability to finish \(K=32\) leaves it in the family with a non-rejection/unavailable outcome; it cannot be removed to improve multiplicity or the negative conclusion.

State that explicitly. The existing “core incomplete” outcome supplies a legitimate stop when the retained work cannot fit. D:448–452. I do not regard the reservation inventory itself as still blocking.

**7. Bracket ordering and residual undefined labels — NEEDS-RESTATEMENT; minor.**

A2 specifies smaller-\(r\) tie-breaking but not the scalar being tied: median A-normalized cost, median per-case linear/head ratio, or another cost statistic. These orderings need not agree. Define it for “costliest affordable” and “cheapest costlier.” D:479–481.

Likewise, explicitly display the fourth cost class, **unresolved**, alongside affordable/equal/costlier. Its exclusion from the certified-affordable set is intelligible, but “no affordable rung” should read “no rung certified affordable,” especially when unresolved or missing-cost rungs exist.

These details do not invalidate the restored equal-cost comparison.

**Remaining blockers, ranked**

1. **Repair or appropriately limit the claimed confirmatory/FWER interpretation after affordability and qualification select the tested comparisons.**
2. **Define invalid/missing timing coverage and its propagation through affordability, seed pooling and F5 without silently shrinking the fixed population or comparator set.**

Minor: explicitly retain post-freeze incomplete sizes in the F5 family; define the bracket’s scalar cost ordering; distinguish certified affordability from unresolved affordability.

**Final verdict: NEEDS-RESTATEMENT before production.**