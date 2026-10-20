**Overall: NEEDS-RESTATEMENT.** A3 fixes the comparator-set selection problem and missing-cost propagation, but its unconditional family-wise error guarantee remains unsupported: the component tests themselves change when qualification and timing completeness select the head’s seed pool.

Read-only throughout. No files changed, jobs submitted, scratch files created, or large arrays copied. I ran one small, in-memory CPU statistical illustration. Draft implementation is not certified.

References: **D** = [heads DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/DESIGN.md); **P** = [p1stats.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/p1stats.py). Citations below use their file lines.

**(a) Disposition of round 3**

The numbering follows round 3’s own seven findings in part (b).

| Round-3 item | Verdict on A3 | Justification |
|---|---|---|
| 1. Adaptive affordability and unconditional FWER claim — blocker | **WRONG as a complete repair** | Simultaneous testing correctly protects subsequent selection **among valid component tests**. However, qualification, rule selection and now timing completeness also determine the errors entering each component test. Holm over \(K\times r\) does not establish validity of those adaptively constructed p-values. D:541–554, 564–566. |
| 2. Missing/invalid timing propagation — blocker | **CORRECT for the missing-cost defect** | Valid-repeat coverage is explicit; cases cannot disappear; missing-cost linear rungs remain possibly affordable; their component p-values become 1. Fewer than two complete head seeds blocks the comparison. No additional timing job is registered. D:545–546, 556–569. The changed seed estimand creates the separate inferential issue above. |
| 3. Positive and negative at-most-cost labels co-occurring | **CORRECT** | Certified-affordable rungs are a subset of possibly-affordable rungs. A positive requires superiority against every member of the latter; a negative requires a certified-affordable member labelled linear-superior or matching. Superiority-first error labels prevent overlap. D:390–395, 549–551, 562–569. |
| 4. F5 Holm and negative readings | **CORRECT structurally; inferential guarantee unresolved** | Component-family Holm replaces per-\(K\) Holm. The affirmative dimension gate and lane precedence remain. Failure to reject still does not establish a negative result. D:487–496, 553–554. |
| 5. Two-of-three rule selection and seed summaries — limitations | **CORRECT as a restricted deployment-policy comparison** | Two seeds suffice; smallest-passing-rule selection and individual-seed sensitivities remain explicit. The pooled point still need not describe one deployable head. A3 adds timing-based seed exclusion, whose consequences require the clarification below. D:498–513, 564–566. |
| 6. Post-freeze budget deferral — minor | **CORRECT** | An incomplete size remains in the frozen family with every component p-value 1; it cannot disappear to improve multiplicity or the negative reading. D:571–573. |
| 7. Bracket ordering and unresolved affordability — minor | **CORRECT** | The ordering statistic is now the median per-case linear/head cost ratio, with smaller-\(r\) tie-breaking. Unresolved and unknown cost classes are explicit, and “certified affordable” replaces the stronger wording. D:560–563, 573–575. |

The earlier accepted repairs to training provenance, Q1 anchors, fixed-\(M\) sensitivity and validation-only scope remain intact. D:331–383, 456–459.

**(b) Adversarial findings**

**1. The fixed component family does not fix adaptive construction of each component — WRONG; blocking.**

A3’s simultaneous-inference argument is correct **provided each component p-value is valid for its registered null**. The design has not established that premise.

There are two different operations:

- **Selecting which fixed tests to report:** simultaneous inference can protect this.
- **Selecting which observations/models define a test:** simultaneous inference over the resulting \(K\times r\) labels does not automatically protect this.

A3 still does the second:

- Rules are selected using these validation trajectories. D:365–373, 510–513.
- Numerical qualification selects the seed pool using case-dependent solver and quadrature behavior. D:348–363, 386–389.
- Timing completeness then selects a further pool, \(\mathcal S'_K\), and **recomputes pooled errors** from it. D:564–566.

Consequently, a component named “head \(K\) versus linear \(r\)” can mean the median of seeds \(\{0,1,2\}\), \(\{0,1\}\), \(\{0,2\}\), or \(\{1,2\}\), at a validation-selected rule. Those are different error statistics. Qualification and timing failure need not be independent of case difficulty or error.

Assigning p = 1 when a **fixed comparison** is unavailable is conservative. Replacing that comparison with another seed subset and computing a fresh p-value is a different operation.

For illustration, I generated 50,000 independent synthetic panels, each containing 38 cases and three null Gaussian seed-difference vectors. Every fixed two-seed average has a symmetric null distribution. Using P’s exact signed-rank null at the first 32-test Holm threshold, \(0.05/32\):

| Comparison construction | Rejection fraction |
|---|---:|
| Fixed seed pair | 0.00148 |
| Pair selected for lowest observed mean difference | 0.00414 |

This is **not** the registered qualification algorithm and does not estimate this experiment’s FWER. It demonstrates why a selected seed-pool p-value cannot simply inherit the validity of a fixed-pool p-value. A3 supplies no conditional-validity argument excluding analogous dependence through its actual selection criteria.

**Required repair:** either limit the result to exploratory validation evidence without the unconditional FWER assertion, or register valid inference addressing rule and seed-pool selection. Independent selection/evaluation data or simultaneous inference over explicitly fixed rule/subset comparisons are possible approaches; their assumptions must be stated. Enlarging the comparator-rung family alone is insufficient.

The signed-rank inference also needs its population/null assumptions stated: its exact branch is not an assumption-free test of arbitrary “more accurate” claims, and tied differences use a normal approximation. P:30–65; D:522.

**2. Missing-cost protection now works — CORRECT.**

For a numerically qualified linear rung with incomplete timing:

1. Its cost is unknown.
2. It remains possibly affordable.
3. Its component p-value is 1.
4. It blocks the positive conjunction.

Thus missing timing cannot silently remove an inconvenient comparator. D:545–551, 559–567.

For heads, removing an incomplete seed can improve or worsen the pooled error and cost. That is permitted by the new estimand, but it is not an inferentially neutral action. Report both numerical-qualification and timing-completeness counts, including excluded seed identities. The statistical consequence belongs to blocker 1.

**3. Scope of \(\mathcal S'_K\) is insufficiently explicit — NEEDS-RESTATEMENT; minor.**

A3 says “pooled error and cost use \(\mathcal S'_K\),” defined by complete **1024² timing**, without naming the affected analyses. D:564–566.

This should explicitly cover the 1024² joint comparisons and their Q-LH-dim gate. It should separately specify what happens to:

- Q-K-err, previously based on numerically qualified seeds;
- the fixed-\(M=128\) panel, which deliberately has **no timing**;
- optional 256² comparisons, whose cost population remains dev6.

A literal global application would make the accuracy-only fixed-\(M\) panel depend on unavailable timings. A cost-comparison-only reading is workable, but should be registered. D:378–389, 503–508.

**4. Comparator selection and policy fairness — CORRECT within the declared scope.**

The lowest-median affordable \(r^*\) is descriptive only; it no longer substitutes for comparisons against every relevant rung. D:473–478, 549–551.

The rule policy can favor or disfavor heads:

- Two passing retrains can support a pooled result while the third fails.
- A small rule passing Q2 for two seeds can later leave fewer than two Q1-qualified seeds.
- A larger rule might support more seeds or better solver behavior.
- Smallest quadrature order need not minimize total query cost.

These are disclosed policy limitations, not reasons to reject the experiment. The appropriate claim remains about this recipe and selection policy, not the best achievable nonlinear or linear deployment. D:348–363, 374–383, 498–513.

**5. Labels and F5 interpretation — CORRECT, with minor presentation clarification.**

I found no positive/negative co-occurrence under the amended superiority precedence.

A head may beat every possibly-affordable rung while **no rung is certified affordable**: all relevant cost intervals might be unresolved. Those statements can coexist. “No rung certified affordable” should therefore be displayed as a cost-coverage status, not an overriding terminal accuracy verdict. D:549–551, 560–569.

An empty possibly-affordable set cannot yield a positive, as explicitly required. Failure of Holm or the dimension gate remains inconclusive unless the separately registered negative conditions hold. The negative wording remains “no benefit shown,” not proof of equivalence or inferiority. D:488–496, 550–554.

**6. Population mismatch and seed medians — CORRECT as scoped.**

At 1024², error and cost use the same E0-eligible case population, with the same retained seed pool for joint comparisons. Individual-seed comparisons remain necessary because componentwise medians need not represent any deployed model. D:465–467, 498–507, 564–566.

At 256², error on eligible all38 and cost on dev6 remains a limitation, explicitly labelled and outside primary F5. It must not be described as a joint all38 accuracy–cost result. D:508, 544.

**7. Controls — CORRECT as registered obligations; execution unverified.**

No newly impossible control follows from A3.

- The unreachable “>25% unqualified cases” branch is correctly a statistics-unit test, not an integrated heads outcome. D:516–517.
- The under-trained head is a sensitivity probe, not a guaranteed failure. D:430–431.
- Superiority/equivalence fixtures have explicit parameters and precedence. D:432–437, 523–527.
- The under-resolved Q2 control requires valid finer solves; failure to demonstrate it blocks Q2. D:440–441.

Reproduction, under-resolution and real timing controls still require smoke evidence. Their success cannot be certified from the design alone.

**8. Budget reservation — CORRECT at design level.**

Mandatory `time256` subjects are included; pending reservations exclude the submitting attempt; failed mandatory work returns to pending; post-freeze incomplete sizes remain in the inferential family. D:448–454, 529–535, 571–573.

A3 registers in-job drift retries but no extra timing rerun job, so it adds no unreserved recovery campaign. D:557–558. Smoke-derived projections must include those retries and the expanded timing coverage. The copied sister-lane ledger is a pattern, not evidence that the heads inventory has already been implemented correctly.

**Remaining issues, ranked**

1. **Blocking:** establish validity of component p-values after rule and seed-pool selection, or withdraw the unconditional confirmatory/FWER claim.
2. **Minor:** explicitly scope \(\mathcal S'_K\) across joint comparisons, Q-K-err, fixed-\(M\) sensitivity and other meshes.
3. **Minor:** distinguish the cost-coverage status “no rung certified affordable” from the at-most-cost accuracy verdict.

**Final verdict: NEEDS-RESTATEMENT before production.**