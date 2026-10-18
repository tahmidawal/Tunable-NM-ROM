**AM-7 resolves the verdict overlap. The controls blocker remains PARTIALLY resolved because its doubled-work construction does not guarantee two executions.** No files were changed, no jobs submitted, and no lab-log entry written.

References below use **A** for [P1-RESTATEMENT-AMENDMENTS.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/campaign/P1-RESTATEMENT-AMENDMENTS.md) and **B** for [COMMON-BRIEF.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/campaign/briefs/COMMON-BRIEF.md).

| Round-2 blocker | Round-3 disposition | Evidence |
|---|---|---|
| Mandatory controls can reject correct implementations | **PARTIALLY** | A:137–159 removes several unsupported guarantees, but A:140–143 introduces an unsupported anti-elision guarantee. |
| H-acc-r overlap and missing whole-range aggregation | **RESOLVED** | A:165–171 establishes qualification/coverage precedence and exhaustive, mutually exclusive transition and range outcomes. |

The previously partial decision-rule findings (§3 and V4-5) are therefore **RESOLVED**. The controls findings (§4 and V3-7), and the corresponding implementation obstacle in §6, remain **PARTIALLY** resolved for the reason below.

**Disposition of every previously PARTIALLY / NOT RESOLVED control**

| Control | Disposition | Evidence and remaining obligation |
|---|---|---|
| Production timing fault | **RESOLVED** | A:137–139 makes it a reported sensitivity probe with no required outcome. The independently specified synthetic fixture remains the mandatory test. |
| Doubled work | **PARTIALLY** | A:140–143 fixes the adaptive-workload mismatch by restarting from the same state. It does **not** establish that the compiled program performs two evolutions. See the blocker below. |
| Component sum versus fused | **RESOLVED at coordinator level** | A:144–145 explicitly requires matched scope. The lane must account for the actual component invocation counts, including predictor evaluations, initial evaluations and rejected iterations where applicable. A single blanket iteration multiplier is sufficient only if those workloads match. |
| Under-resolved rule | **PARTIALLY; safely finish in lane** | A:146–150 requires independent evidence instead of assuming that \(m<M\) proves failure. However, disagreement with a converged rule does not necessarily imply disagreement with the particular successor used by Q2. The lane must establish failure of the actual Q2 predicate. |
| Wrong-gradient / replacement Q1 mutation | **PARTIALLY; safely finish in lane** | A:151–154 correctly demotes the sign-flipped Jacobian. Independent NumPy confirmation of \(\gamma>10^{-3}\) establishes a non-stationary **step**; it does not alone establish failure of the **pooled >1%** criterion in B:49–55. The fixture must also establish that aggregate predicate. |
| Zero-update solver | **PARTIALLY; safely finish in lane** | A:155–156 provides a substantial-motion requirement. The motion must be measured against the verified tight trajectory using the same initialization, normalization and scoring contract as Q1. Motion of an arbitrary frozen trajectory alone does not prove distance from the tight anchor. |
| CN-R temporal order | **PARTIALLY; safely finish in lane** | A:157–159 adds a contamination check and an explicit uninterpretable outcome. Small solver/quadrature contributions at the finest tested step do not themselves prove asymptotic temporal behavior or negligible anchor error. Those checks can be specified in DESIGN.md; failure blocks the order label. |
| Sensitivity probes | **RESOLVED** | A:160–162 correctly acknowledges the frozen seeded random shift. The implementation uses `default_rng(seed).uniform(size=2)` in `hari_quadrature.py:116`. |

The reproduction gate, previously “resolved in intent,” is now **RESOLVED**: A:176–177 explicitly separates numerical configuration equality from old/new provenance.

**Remaining blocker: the doubled-work anti-elision claim is false as a guarantee.**

A:141–142 proposes adding `0·sum(W₁)` to the second input “so it cannot be elided.” A zero-valued arithmetic dependency is not an execution barrier. Whether it survives optimization depends on compiler semantics and optimization settings. Consuming both outputs also does not prove that shared computation was executed twice.

An optimized implementation could return the first evolution’s output twice, including identical iteration counts. Thus the new iteration-count check cannot certify the intended workload. The mandatory timing ratio could then reject a correct timing harness because the supposed doubled workload was never established.

The amendment should require **two separately dispatched, synchronized executions of the compiled evolution from the same initial state**, or another construction whose two executions are demonstrated in the compiled program/profile. Remove the assertion that a zero multiplication guarantees execution. The lane can choose the exact implementation and freeze matched timing boundaries.

**Adversarial checks of AM-7 itself**

- **H-acc-r outcomes are mutually exclusive.** Qualification failure stops classification; insufficient resolved coverage stops classification; the remaining cases receive exactly one transition outcome. At range level, “every transition improves” and “any transition does not improve” cannot coexist. Mixed improves/inconclusive transitions yield inconclusive; any established non-improvement yields not monotone.
- **The numerical examples are not unconditional guarantees.** A rule can disagree with a converged comparator while agreeing with its immediate successor. One non-stationary step among 200 is only 0.5%, so it passes the pooled stationarity bar. These are lane obligations under A:146–147’s overarching requirement for independently established expected results—not reasons to demand another coordinator expansion.
- **No additional false factual claim found in the checked source assertions.** The seeded-shift correction and historical W512 `gl56` citation agree with their sources. The anti-elision assertion is the new factual defect.

Safe lane work includes the precise discriminating fixtures and aggregate predicates, component accounting, bootstrap and numerical edge cases, empty-population handling, configuration-level Q2 labels with their 16-case validation scope, temporal-anchor checks, historical `acc` versus `R384` mapping, explicit 3D Gauss comparator, and the revised cost certificate. These do not require another coordinator restatement.

NEEDS-RESTATEMENT