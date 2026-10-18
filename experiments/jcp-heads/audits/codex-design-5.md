**Overall: READY at the design level, for exploratory validation evidence only.** A4 resolves round 4’s blocker and both requested minor clarifications. I found no remaining blocking issue. This does not certify the draft implementation or replace the required code and smoke audits.

Read-only throughout: no files changed, jobs submitted, scratch created, or arrays copied.

Citations below use **D** = [heads DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/DESIGN.md), **P** = [p1stats.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/p1stats.py), and **R4** = [round-4 audit](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/audits/codex-design-4.md).

**(a) Disposition of round 4**

| Round-4 item | Verdict on A4 | Justification |
|---|---|---|
| **Blocker: adaptive rule/seed selection invalidates the asserted unconditional FWER guarantee** | **CORRECT** | A4 explicitly withdraws confirmatory/FWER claims, acknowledges that selection and evaluation reuse validation trajectories, and retains Holm only as a descriptive decision rule. This implements the exploratory-only repair requested in R4:53. The signed-rank assumptions and excluded-seed reporting are also stated. D:581–592. |
| **Minor: scope of the timing-complete pool \(\mathcal S'_K\)** | **CORRECT** | A4 assigns it to 1024² joint comparisons, including Q-LH-dim. Accuracy-only analyses retain numerical-qualification pools; fixed-\(M\), optional 256², and H-cost-N receive separate definitions. D:594–601. The inherited definition remains mesh- and scheme-specific. D:386–389. |
| **Minor: “no rung certified affordable” versus the accuracy verdict** | **CORRECT** | A4 makes this a separate cost-coverage status. The accuracy result still uses the possibly-affordable set, so superiority and lack of certified affordability may coexist without contradiction. D:603–606. |

The other round-4 findings remain accepted, with the limitations below.

**(b) Adversarial checks**

1. **Amendment precedence and surviving confirmatory language — CORRECT.**

   A3’s strong-FWER argument remains visible at D:547–554, but A4 directly withdraws that claim and says nothing in the lane is confirmatory. The earlier stronger F5 wording is likewise superseded through A1.6, A2.2, A3.1 and A4.1. These are historical statements, not competing live rules. D:385, 418–424, 483–496, 581–592.

   A positive result cannot reinstate the paper’s nonlinear-manifold claim by itself: the separate sealed accuracy and mesh-invariance obligations remain open. D:456–459.

2. **Holm and the F5 decision procedure — CORRECT as an exploratory rule.**

   The operative adjustment is once across the fixed \(|\mathcal F|\times8\) BE component family, not an additional per-\(K\) Holm correction. Unavailable components receive \(p=1\). A positive requires every possibly-affordable component to pass both Holm and the superiority predicates, plus the resolved Q-LH-dim gate. D:541–554.

   F5’s lane precedence remains positive first, then the separately established negative outcome for every retained size, otherwise inconclusive. Non-rejection alone cannot produce the negative reading. D:487–496, 568–573.

   A4’s assumptions describe the signed-rank procedure; they do **not** prove that selection-conditioned differences satisfy those assumptions. A4 no longer claims otherwise. The implementation’s exact-versus-approximate branches agree with the stated description. D:589–591; P:30–65.

3. **Positive/negative labels co-occurring — CORRECT; no contradictory pair found.**

   Certified-affordable rungs are a subset of possibly-affordable rungs. Therefore a positive at-most-cost result requires head superiority against any rung that could establish the negative result. Superiority-first error labels prevent that same rung from simultaneously being labelled matching or linear-superior. D:390–395, 549–551, 560–569.

   Legitimate co-occurrences remain:

   - “Head more accurate” with an equivalence flag: superiority has precedence.
   - Positive at-most-cost accuracy with “no rung certified affordable”: all relevant cost intervals can be unresolved.

   Neither is a contradiction. D:393–394, 603–606.

4. **Missing timing and empty comparator sets — CORRECT, with one minor wording issue.**

   A qualified rung with incomplete timing remains possibly affordable and receives component \(p=1\); it blocks the positive conjunction. Fewer than two complete head seeds makes joint comparisons unavailable. Missing cases cannot be dropped to improve the result. D:545–546, 559–567.

   **Minor — NEEDS-RESTATEMENT:** A4.3’s “every qualified rung is certified costlier” also holds vacuously when there are **no qualified linear rungs**. The result is safely non-positive, but “no possibly-affordable rung” then obscures whether the ladder was genuinely too expensive or entirely unqualified. Add a displayed reason distinguishing those cases. This does not alter F5 and is not blocking. D:550–553, 603–606.

5. **Rule policy can favor or disfavor heads — CORRECT within the declared scope.**

   The comparison is between registered deployment recipes, not globally optimized linear and nonlinear methods. Both arms use the common rule ladder, while heads use the explicitly registered two-of-three seed criterion. D:365–376, 510–513.

   Remaining asymmetries are real but disclosed:

   - Excluding a failed third seed can improve the pooled head result.
   - Selecting the first rule passing Q2 can later leave fewer than two Q1-qualified seeds.
   - A larger rule might improve qualification or reduce iterations; smallest rule does not imply fastest query.
   - Varying \(K\) also changes optimization, initialization, trust radius and test-space size.

   These limit interpretation rather than invalidate this experiment. The fixed-\(M\) panel and per-seed reporting provide relevant sensitivities. D:331–338, 378–381, 498–513.

6. **Validation-selected comparator \(r^*\) — CORRECT.**

   The lowest-median affordable rung is now descriptive only. It cannot stand in for the universal “every possibly-affordable rung” comparison. A3’s component conjunction and A4’s exploratory scope remove the earlier unsupported inference from selecting \(r^*\). D:473–478, 541–554, 581–592.

   The descriptive \(r^*\) remains an optimistically selected linear comparator; it should not be presented as an independently validated best rung.

7. **Seed medians and error/cost populations — CORRECT as scoped.**

   At 1024², joint comparisons use the same eligible case population and retained seed pool for error and cost. Timing is available on all38, with dev6 repeated three times and val32 once; per-case summaries prevent dev6 from receiving threefold population weight. D:503–507, 564–567, 595–597.

   The pooled error–cost point still need not represent any deployable model: different seeds can supply medians across cases or objectives. A2 expressly acknowledges this and requires individual-head joint comparisons. A4 adds excluded-seed identities and reasons. D:498–501, 591–592.

   At 256², all38 error and dev6 cost cannot support a joint all38 claim. A4 states that limitation explicitly, and the primary F5 family remains BE at 1024². D:544, 599–601.

8. **Q-LH-dim’s relationship to Holm — CORRECT operationally; minor wording ambiguity.**

   The inherited Q-LH-dim procedure uses the paired error/cost labels. Holm governs the at-most-cost conjunction and F5, not a newly adjusted dominance classifier. D:390–402, 487–489, 553–554.

   **Minor — NEEDS-RESTATEMENT:** A4.1 includes Q-LH-dim among labels Holm “feeds.” Strictly, Q-LH-dim is a separate gate feeding F5. State that its paired tests remain unadjusted and exploratory. The existing operative definitions are sufficiently clear that this is not blocking. D:585–587.

9. **Controls that cannot fire — CORRECT as design obligations; execution unverified.**

   I found no new logically impossible control introduced by A4.

   - The under-trained head is a sensitivity probe, not a guaranteed failure. D:430–431.
   - The unreachable integrated “>25% unqualified cases” branch is explicitly a statistics-unit fixture. D:516–517; P:129–153.
   - Dominance fixtures have fixed populations, errors, margins and costs. D:432–437, 523–527.
   - The under-resolved Q2 control requires valid finer solves; inability to demonstrate it blocks Q2. D:440–441.
   - The driver-identity repair uses incumbent-emitted `Z_tr`, rather than the extraction file’s `Z_ck`. D:427–429; [sep_hfit_run.py:366](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/separable-decoder/sep_hfit_run.py:366).

   Reproduction tolerances and real numerical/timing controls still require smoke evidence. Their success is not established by this review.

10. **Budget reservation and post-freeze deferral — CORRECT at design level.**

    The amended inventory reserves mandatory `time256` subjects, excludes the submitting attempt from pending totals, restores failed mandatory work to pending, and stops with dependent verdicts unavailable if the core cannot finish. D:448–454, 529–535.

    After the family freezes, an incomplete size remains in it with component \(p=1\), preventing selective removal from easing Holm or manufacturing the negative reading. D:571–573.

    In-job timing retries must be included in smoke-derived projections; no additional timing recovery campaign is registered. D:557–558. The sister ledger demonstrates the reservation pattern, but does not certify the heads implementation: [ledger.py:168](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/ledger.py:168).

**Remaining issues, ranked**

1. **Blocking: none found.**
2. **Minor:** distinguish an empty qualified linear ladder from a nonempty ladder whose members are all certified costlier.
3. **Minor:** clarify that Q-LH-dim uses its existing unadjusted exploratory procedure and gates F5; Holm does not redefine its labels.

**Final verdict: READY for the next code-audit stage under A4’s exploratory-only scope.**