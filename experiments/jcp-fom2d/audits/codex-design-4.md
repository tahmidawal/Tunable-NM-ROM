**NOT READY.** A3 resolves the extension-ordering and PS-diagnostic blockers. It corrects the direction of the selected-comparator argument, but the “conservative witness” conclusion still needs narrower wording. A4’s account-cap mitigation does not enforce the registered cap.

References below use **D** for [DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/DESIGN.md). This is a design audit, not certification of A4’s implementation.

**Round-3 blocking items**

| Item | Disposition | Evidence |
|---|---|---|
| Selected-comparator versus best-FOM conclusions | **NEEDS-RESTATEMENT**, substantially repaired | **D:421–434** correctly distinguishes the optimization bound from the bootstrap and withdraws “K2 does not fire.” But **D:426–427** still describes the descriptive-bootstrap result as a witness that the ROM lacks a best-FOM speed advantage. |
| Extension trigger and ordering | **CORRECT** | **D:436–441** explicitly chooses the base-ladder C1 lattice-cost matcher, defines the family-wide minimum admissible resolution, completes the extension, reruns matching, then performs C2 selection. No C2-dependent circularity remains. |
| PS stability lower-bound claim | **CORRECT** | **D:443–445** explicitly withdraws the lattice-proxy sentence and uses the padded-grid running maximum. The formula agrees with [E0 `refjob.py:448`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/refjob.py:448). It remains a diagnostic, not a stability certificate or admissibility condition. |

**Round-3 non-blocking clarifications**

| Clarification | Disposition | Evidence |
|---|---|---|
| Family/union shortlist sizes, empty groups, unmatched selection | **CORRECT — non-blocking** | **D:447–450** gives three per family, nine pooled, “no match” for empty groups, and one lowest-worst-error unmatched candidate. |
| Conditional inference | **CORRECT as a scope statement — non-blocking** | **D:360–363, 423–424** limits the bootstrap to workload variation and disclaims population inference. The remaining problem is the conclusion attached to it, discussed below. |
| PS anchors | **CORRECT — non-blocking** | **D:450–451** requires actual steps and validity, makes invalid adjacency unavailable, and forbids certification or reselection. |
| H/X populations | **CORRECT — non-blocking** | **D:378–383** retains population-specific matching and the three-case minimum; **D:452** limits a single-mesh identity to a point. The five-of-six timing requirement remains separate. |
| Deferred PS | **CORRECT — non-blocking** | **D:452** keeps these cells outstanding in the hand-back; **D:385–388** retains the partial-scope limitation. |
| FD two-dispatch timing | **CORRECT — non-blocking** | **D:452–453** includes both dispatches and output completion and disclaims implementation optimality. |
| Recovery | **CORRECT — non-blocking** | **D:409–411**, together with **D:256–258**, requires a calibration-passing tighter level and a complete all-case rerun. A3/A4 do not revoke these requirements. |

**Remaining blocking findings**

1. **The optimization inequality is conservative; the descriptive bootstrap is not thereby a confidence guarantee.**

   For fixed costs, a common workload and an admissible selected comparator,

   \[
   S_{\mathrm{best}}\le S_{\mathrm{selected}}.
   \]

   Therefore, a valid upper confidence bound on the selected comparator’s underlying speed ratio would also bound the best-FOM ratio. Comparator screening does not invalidate that one-sided implication.

   However, **D:423–424** expressly says the interval summarizes case-to-case variation, while **D:426–427** calls its failure outcome a conservative witness about the ROM’s speed. Those are different claims. Repeating case IDs does not quantify timing-repeat uncertainty, and using fresh repetitions on the same selected workload does not supply selection-adjusted population coverage.

   For example, timing measurements can put all six observed selected-comparator ratios below two even when their underlying repeat-aggregated ratios exceed two. Resampling those six observations cannot reveal that error. An overlooked cheaper FOM might rescue the conclusion, but its existence cannot be assumed.

   **Required repair:** retain the registered rule but label its outcome explicitly as **“fail under the registered dev6 descriptive-bootstrap rule.”** Likewise, say **“K2 fires under that workload-specific rule,”** without presenting it as a confidence-certified best-FOM conclusion. Alternatively, provide an inferential procedure supporting the stronger claim.

   **“Not shown to fire” is correct** for a selected-comparator pass: that pass cannot exclude a cheaper matching FOM. It is an inconclusive best-FOM outcome, not evidence that K2 is false. A3’s complete-manifest requirement and incomplete-universe override are sound.

2. **A4’s post-submission cancellation cannot enforce “at most eight.”**

   **D:477–480** acknowledges the missing shared lock but substitutes detection and cancellation after `sbatch`.

   Counterexample:

   - Two lanes observe seven jobs.
   - Both submit.
   - The account has nine jobs before either post-check completes.

   Cancellation can shorten the violation; it cannot prevent it. Jobs can also start before cancellation, and a failed post-check or cancellation leaves the excess unresolved. The current implementation follows exactly this sequence: [`cluster/submit.sh:35`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/cluster/submit.sh:35), with the check/cancellation at lines 43–46.

   **Required repair:** require a shared submission/reservation mechanism honored by all lanes, scheduler enforcement, or coordinator-issued slots before submission. Until then, this remains a production-submission blocker under **D:4–5**. Calling it a mitigation is honest; treating it as resolution of the cap requirement is not.

**Other adversarial checks**

- **Extension: CORRECT, with a non-blocking reproducibility detail.** A3 deliberately uses lattice screening, so extension need not follow the eventual full-field winner. That is a bounded search policy, not a guarantee of finding the optimum. Record the triggering target, population, comparator and base-ladder minimum. Explicitly adopt the C1 tie-break already implemented at [`match2.py:267`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/match2.py:267): screening cost, then lexical key. Equal-cost edge/interior candidates could otherwise imply different extension decisions.

- **Bitwise identity: CORRECT at design level; non-blocking clarification.** **D:469–470** makes bitwise identity the default and requires a dated, named-family amendment for the tolerance fallback. This is stricter than silently accepting `≤10⁻¹²`. State expressly whether that amendment governs both C1 screening and C2 identity checks, and bind the authorized policy to frozen configurations/results. A tolerance fallback establishes numerical agreement, not bitwise identity. Failed smoke evidence can motivate an amendment, but does not itself become accepted production evidence.

- **A4’s other dispositions are sensible design requirements, not verified repairs.** Fail-closed acceptance, immutable resume identities, persisted convergence evidence and complete production populations address the code audit’s failure modes (**D:462–476**). They still require the separate implementation audit. Similarly, adopting a 0.1-second burn-in (**D:483–484**) does not itself demonstrate that clock-ramp bias is absent.

The design remains **NOT READY** pending the inference wording and account-cap enforcement fixes. No files were changed, no cluster commands or jobs were run, and no arrays or scratch files were created.