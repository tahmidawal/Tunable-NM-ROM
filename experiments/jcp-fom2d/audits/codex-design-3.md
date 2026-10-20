**NOT READY.** A2 resolves most round-2 findings, but the extension sequence remains ambiguous, and the decision rules do not fully respect the one-sided nature of the screening bound. One previously identified false PS diagnostic claim also remains.

Read-only audit: no files changed, no cluster access, no jobs or computations run, no arrays copied, and no scratch created.

Evidence notation: **D** refers to [DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/DESIGN.md); line numbers below refer to that file. **R2** refers to [round-2 audit](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/audits/codex-design-2.md).

**Disposition of the five round-2 blocking fixes**

| Round-2 fix | A2 verdict | Evidence |
|---|---|---|
| 1. Complete C2 selection and inference | **CORRECT**, with non-blocking clarifications below | D:348–359 defines family/union groups, common-valid selection cases, retries, ties, and frozen winners. D:360–363 explicitly limits inference to dev6. D:368–370 removes the anchor scheduling dependency. |
| 2. Repair K2 coverage and dominance inference | **CORRECT for coverage; NEEDS-RESTATEMENT for conclusions** | D:372–376 withdraws dominance and requires direct evaluation against a complete qualification manifest. However, inherited D:160–161 still declares “does not fire” when some selected-comparator cell passes; screening can miss a faster validated comparator. |
| 3. Align minimum-cost claims and population-specific execution | **CORRECT for execution and comparator wording; NEEDS-RESTATEMENT for bound interpretation** | D:364–367 disclaims identifying the minimum. D:378–383 independently matches H/X populations and enforces the size requirement. The bound direction is correct, but its implications for positive conclusions need repair. |
| 4. Resolve high-resolution PS scope honestly | **CORRECT** | D:385–388 withdraws the unsupported projections and explicitly defers the cells. This is a partial same-resolution deliverable, not evidence that those PS settings are slower. |
| 5. Finish extension/recovery rules and reconcile contracts | **NEEDS-RESTATEMENT** | Recovery is repaired at D:409–411; FD/PS contracts at D:401–405. Extension still depends on an undefined pre-C2 “selected” comparator, D:390–398. Code implementation is outside this audit. |

**Disposition of every round-2 NEEDS-RESTATEMENT item**

This table covers both R2’s blocking-fix table and its detailed disposition table; repeated findings are consolidated.

| Round-2 item | A2 verdict | Evidence and qualification |
|---|---|---|
| C2 union selection, selection validity, frozen IDs | **CORRECT** | D:348–359. Union winners are selected before inference and never reselected from stage 2. |
| Same-case selection/bootstrap interpretation | **CORRECT** | D:360–363 expressly denies selection-adjusted population inference. |
| Unsupported “cheapest” claim / top-three screening | **CORRECT wording; NEEDS-RESTATEMENT downstream** | D:364–367 replaces the minimum claim with a procedure-selected comparator. Positive “best FOM” conclusions do not follow. |
| K2 complete inventory and dominance shortcut | **CORRECT** | D:372–376 requires the complete qualified inventory and direct timing; missing inventory makes K2 inconclusive. |
| Same-mesh PS and restricted union | **CORRECT as partial scope** | D:385–388; D:260–261 retains a meaningful equal-state-resolution convention. |
| General interpolation factor | **CORRECT** | D:413 explicitly gives \(N/L\) for the stated two-dimensional orthonormal DST-I convention. |
| Prepared-input timing label | **CORRECT** | D:363 supplies the label; D:70–72 retains the preparation contract. |
| Bounded extension details | **NEEDS-RESTATEMENT** | D:390–398 supplies global scope, fallback, deduplication and added resolutions, but leaves the triggering selection stage ambiguous. |
| Production recovery qualification | **CORRECT** | D:409–411 requires the recovery level itself to pass calibration, with exactly that level accepted offline. |
| PS lattice stability proxy | **WRONG, unchanged** | D:248–250 still calls the lattice maximum a lower bound on the padded-grid running maximum. D:404–405 adds the real running maximum but does not replace that diagnostic claim. |
| PS anchor scheduling | **CORRECT**, with non-blocking clarification | D:368–370 uses existing ladder fields, eliminating selection/run circularity. These are adjacent-step differences, not necessarily half-step or finer-step anchors. |
| Cheapest-admissible-member boundary flag | **CORRECT** | D:396–397 restores a separate screening-cost boundary label. |
| “Most accurate” comparator terminology | **CORRECT** | D:398 uses “lowest-worst-error unmatched comparator.” |
| H/X population-specific selections | **CORRECT** | D:378–383 gives distinct keys, matching and groups; identical populations may be deduplicated. |
| Empty/undersized intersections and K-false ties | **CORRECT** | D:382–383 requires at least three eligible cases and deterministically chooses the first raw-error maximizer. |
| Complete resource accounting | **CORRECT at design level** | D:414–415 adds preparation, recovery, extensions and both selections to D:334–338. Anchors now reuse fields. Actual feasibility remains smoke-gated. |

The previously correct K-syn/K-false constructions remain in D:306–313. A2 does not need to repeat them. Likewise, A2’s recovery amendment supplements the inherited one-retry, all-case rerun rule at D:256–258.

**Adversarial findings**

1. **The upper-bound direction is right; the positive decision implication is not.**

For fixed positive costs on the same workload, if the selected FOM belongs to the admissible matching set,

\[
S_*=\frac{\min_{F\in\mathcal M}c_F}{c_{\rm ROM}}
\le
\frac{c_{\rm selected}}{c_{\rm ROM}}
=S_{\rm selected}.
\]

Thus missing the cheapest member makes the reported ROM advantage too favorable, exactly as D:364–367 says. The direction agrees with [PLAN-v4-amendments.md:78](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/PLAN-v4-amendments.md:78).

But suppose ROM cost is 1, the selected matching FOM costs 3, and an overlooked validated matching FOM costs 1. The selected comparison can confidently pass 2× while the best-FOM comparison fails.

Consequently:

- A selected-comparator result below 2 can provide a conservative witness against a 2× best-FOM claim, subject to the stated uncertainty scope.
- A selected-comparator result above 2 establishes only superiority over that comparator.
- D:161’s “does not fire iff some member passes” cannot establish that K2 does not fire against the **best validated FOM**.

The governing quantifier remains “every linear setting,” against the best validated setting: [governing plan:135](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-06-jcp-offmesh-paper-plan.md:135), [V3:85–88](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/PLAN-v3-amendments.md:85).

Also distinguish the optimization inequality from measurement uncertainty: a noisy stage-2 point estimate is not itself a guaranteed numerical upper bound. D:364–367 should name the bound as an optimization/scope limitation, separate from the bootstrap interval.

2. **The deterministic extension rule still lacks a deterministic selection stage.**

D:392 triggers on a “selected joint comparator.” A2 now defines selection through full-field stage 0 and paired stage 1 at each ROM mesh, D:348–359. Yet D:398 builds C2 configurations only **after** extension.

Two plausible implementations therefore differ:

- use the original lattice-screened preliminary match;
- run preliminary full-field screening/selection before deciding extensions.

These can select different resolutions and trigger different extensions. The design must choose one. It should also specify that \(L_{\min}\)/\(N_{\min}\) is the family-wide minimum admissible resolution on the **base** ladder.

The remaining mechanics are sound: one global round, P1 fallback included, no duplicate added resolutions, all ladder steps tested, and normal calibration/admissibility applied.

3. **A2 leaves a known false PS maximum comparison in force.**

D:248–250 compares maxima over different, generally non-nested spatial grids. A scoring-lattice sample can exceed every padded-grid sample; sampling at fewer times does not repair the missing spatial inclusion.

A2 now records the actual padded-grid running maximum, D:404–405. Use that quantity for the E0-style diagnostic, or remove the lower-bound/optimism assertion and call the lattice quantity an unrelated proxy. E0’s check uses the recorded `umax`: [refjob.py:448](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/refjob.py:448).

This does not undermine PS admissibility, which A2 correctly separates from the heuristic.

**Non-blocking clarifications**

- **Group construction:** Reading A2 together with A1’s retained stage-1 shortlist rule gives three candidates per family and up to nine pooled union candidates. State that explicitly; D:349 currently calls all matches the family’s “candidates.” An empty group should carry “no match,” not “invalid timing.” Same-mesh-unmatched groups should retain the lowest-worst-error choice rather than silently cost-selecting among unmatched settings.
- **Conditional inference:** D:360–363 is an honest scope restriction. The bootstrap remains a workload-resampling summary, not a timing-repeat confidence procedure or evidence about unseen cases. Keep that qualification beside H-comp/K2 results.
- **PS anchors:** D:370 may use a *larger* step and ratios other than two. Report both actual steps and whether both trajectories are valid. If the adjacent member is invalid, report the diagnostic unavailable; do not present it as a convergence certificate. No reselection should follow.
- **H/X populations:** The \(n\ge3\) matching requirement and the five-of-six timing requirement are different and compatible. A crossover identity qualified at only one mesh supplies a point, not a bracket.
- **Deferred PS:** D:385–388 is acceptable as declared partial evidence. The lane brief still requests all-family same-mesh crossover at [brief:28](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/campaign/briefs/jcp-fom2d.md:28); those cells must remain outstanding in the hand-back.
- **FD two-dispatch timing:** D:401–403 is a defensible implemented-program contract if both dispatches and completion of the output map are included in the timer. Byte-identical E0 code explains the implementation choice; it does not prove fusion would have identical performance. Avoid claims of optimal FD implementation.
- **Recovery:** D:409–411 correctly closes the calibration loophole. Production convergence and complete evidence on all cases remain required by the inherited rules.

**Ranked remaining blocking fixes**

1. **Align H-comp/K2 conclusions with the screening bound.** Explicitly distinguish passing against the selected comparator from passing against the best validated FOM. A selected-comparator pass alone must not certify “K2 does not fire” for the latter.
2. **Define the pre-extension selection procedure and ordering.** Specify exactly which cost/selection stage triggers extension, then rerun final screening and freeze selections after the extended ladder is complete.
3. **Remove or correct the retained PS lower-bound assertion.** Prefer the actual padded-grid running maximum now required by A2.

These are design-level fixes. No conclusion here certifies or rejects the implementation under the separate code audit.