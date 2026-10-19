**Overall verdict: READY for 2D implementation.** A3 resolves audit 3’s design blockers. The draft E6 implementation does **not yet fully match** A3.2; those discrepancies belong in the forthcoming code audit.

Read-only review completed: I read the whole DESIGN, all three previous audits and both binding briefs. No files changed, jobs submitted or scratch files created.

References below: **D** = [DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/DESIGN.md), **F** = [refjob.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/refjob.py).

| Audit-3 fix | Verdict | Evidence |
|---|---|---|
| 1. Require valid production and tight probe members; define retry replacement | **CORRECT** | D:527–530 requires both members to pass `steps_ok`, finite deviation, and uses the last production/retry attempt. F:204–212 implements those rules. A2’s limited “observed coarse-level agreement” interpretation remains intact at D:467–472. |
| 2. Deterministic cohort allowance, explicit terminal branches and selection disclosure | **CORRECT as a design** | D:532–542 fixes the allowance per cohort, ascending ordering, resumed consumption, all three skipped-escalation branches, and adjacent reporting of resource-limited cases. It explicitly acknowledges unknown-direction selection bias. E6 discrepancies are listed below. |
| 3. Freeze reduced populations, complete reservation policy and stop branch | **CORRECT as a design** | D:544–553 makes smoke diagnostic-only and regenerates production populations. Reduced C2 means c0–c7 consistently for references, ROM, Q1/Q2 and labels. PS is dropped **before** C2 reduction, so no reduced-C2 PS population remains ambiguous. FOM is deferred first; 3D is already outside scope. Mandatory work that cannot fit triggers amendment. |

The reservation certificate remains a **pre-submission deliverable**, not an accomplished validation. It must include the smoke’s reproduction, upwind, calibration and ROM controls in spent/open reservations, as well as production and conditional probes. A3’s “every workload” requirement and A2’s spent-plus-open-reservations gate provide the contract; the historical estimates alone do not certify affordability.

Audit 3’s minor points are all **CORRECT**:

| Point | Evidence |
|---|---|
| Distinguish 80 H200 cases from the 102-case record; describe 850 seconds as planning allowance | D:555–557 |
| Wilcoxon ties concern absolute nonzero log differences | D:557–558 |
| Apply ratio coverage after zero/non-finite exclusions | D:558 |
| Explicit seven-case PS population, excluding N0s | D:558–559 |

**E6 match check: WRONG as a claim of complete implementation.**

- **Correct:** F:172–181 prevents refinement after a tight triple and prevents a tight refined triple, recording the respective skip flags. F:356–357 deducts verified-resumed tight cases.
- **Mismatch:** D:536 requires `outcome="unresolved-floor"` when allowance blocks escalation. F:169–170 sets only `tight_skipped`; F:186 preserves `jd['outcome']`. Thus the promised terminal outcome is not emitted.
- **Incomplete enforcement:** F:328–330 checks sorted indices but accepts arbitrary case subsets and an arbitrary configured allowance. It does not itself enforce the frozen A+ allowance, other-cohort allowances, or full production population. Correct generated configurations and validation could enforce these; E6 alone does not.
- The resume deduction supports the registered single-job, ascending-population workflow. It is not a general guarantee against repartitioning or changed restart configurations.

These are implementation obligations under an already explicit design, rather than reasons to reopen the experimental specification.

**New minor wording issue introduced by A3:** D:559–562 substitutes descriptor amplitude \(a\) for the previously defined nodal maximum. That is an implementable normalization convention, but the claimed identity is inaccurate. For single bumps, \(a\) is the continuous peak; for B1/B2 it is the peak on the normalization grid. Inclusion of those coarse nodes in finer grids does **not** make their maxima equal. Call the revised diagnostic explicitly
\[
\sigma_a=\max_{k=1,\ldots,5}g_k/a.
\]
The tiny single-bump approximation bound does not establish the corresponding bound for two-bump fields. This is nonblocking because A3 explicitly specifies the denominator to compute.

**Remaining required design fixes:** none.

Readiness covers coding the **2D** design only. It does not certify the draft code, resource certificate, smoke outcomes or deferred 3D phase.