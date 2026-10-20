**Overall: NEEDS-RESTATEMENT.** A4 resolves the TH06 schedule and most budget issues, but omits reservation of the enlarged **first** timing panel.

References: `D` = [DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/DESIGN.md); `Brief` = [lane brief](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/campaign/briefs/jcp-qualify.md).

| Prior blocker | A4 disposition | Judgment and evidence |
|---|---|---|
| 1. Fallback reservation lifecycle | **PARTIALLY** | **NEEDS-RESTATEMENT.** Atomic replacement, inherited defer priority, and the non-deferrable budget-failure outcome are explicit (`D:558–564`). However, the replacement covers fallback accuracy and **repeated** timing panels, omitting the increased reservation for an unsubmitted first panel receiving fallback subjects (`D:408–410`). |
| 2. TH06 ladder lacks \(L_s\) | **RESOLVED** | **CORRECT.** A4 explicitly omits the contextual larger step, registers factors \(\{2,1,\tfrac12,\tfrac14\}\), and preserves the fitted steps, anchors and checks (`D:566–568`). |

Additional adversarial checks:

- **CORRECT — sequencing is mesh-local.** A2 supersedes A1.2’s global barrier. Timing still waits for its own mesh’s optional accuracy and fallback work; A3 explicitly acknowledges this delay (`D:484–487,530–531`). Other meshes’ completion is not a logical prerequisite.
- **CORRECT — core and defer order preserved.** A3 matches the brief’s four-level defer order; A4 makes fallbacks inherit their parent’s priority (`D:521–525,560–564`; `Brief:43–44`). “Core incomplete (budget)” blocks dependent verdicts rather than silently dropping required work.
- **CORRECT — TH06 cannot pass vacuously.** At least five interpretable cases must each meet the order-1 interval; the full control runs in smoke and repeats in production (`D:428–434,546–548`).
- **CORRECT — order outcomes remain reachable.** Both integer-order intervals are compatible with the interpretability checks. Nonpositive/nonfinite fitted errors and undefined fits are rejected (`D:247–257,550–552`).
- **CORRECT — no restored 3D completeness claim.** Only individually qualified candidates enter the tested-subset front; absent check rules remain unverified (`D:474–482`).

Ranked blockers:

1. **NEEDS-RESTATEMENT — reserve enlarged initial timing panels atomically.** When Q2 triggers fallbacks before timing submission—the normal sequence—the first panel gains subjects. A4 reserves their accuracy jobs but explicitly adds timing only for a **repeat**. The existing panel reservation can therefore remain undersized while intervening optional submissions consume the difference. A later submission check protects the hard cap, but cannot recover that spent budget (`D:526–529,558–564`). Require the same transaction to update the first panel’s projected time limit to include all retained fallback subjects, or reserve a complete repeat when necessary, without double-counting.

Read-only audit: no files changed, scratch created, arrays copied, or cluster jobs submitted.