<!-- codex -o output file was not written (process exited without it); this is the final answer block copied from the codex stdout log, lines 11903-end -->
| B5 | NEEDS-RESTATEMENT | State machine fixed; selection-timeout conflict remains |
| B8 | NEEDS-RESTATEMENT | Explicit exhaustion outcomes added; ceilings need operational definitions |
| B9 | NEEDS-RESTATEMENT | Cost phases named; conservative bounds still unsupported |

**A1 — CORRECT.** Frozen normalization, nonzero $\alpha$, the lower bound on $c$, and unclipped-floor assertions close the reported degeneracies. A prerequisite failure produces INCONCLUSIVE rather than a false detector pass. The inherited unit normalization of $z$ must remain.

**A2 curvature — NEEDS-RESTATEMENT.** Zero/nonfinite $L$, a mixed absolute/relative refinement tolerance, and the explicitly descriptive fixed-spacing check repair audit 3’s substantive concerns. However, “all curvature numbers … normalised by $L$” conflicts with the independent-path tolerance containing $10^{-6}L$: are $a,b$ raw derivatives or normalized derivatives? These interpretations behave differently under amplitude rescaling.
**Fix:** define $a,b$ explicitly. For raw derivatives retain $10^{-6}L$; for normalized derivatives use $10^{-6}$. State that the maximum is an absolute infinity norm.

**A2 decision edge cases — CORRECT.** Zero/nonfinite total energy is rejected, availability is measured against the original field, retained tail coefficients avoid subtraction cancellation, and unavailable controls force INCONCLUSIVE. The inherited minimum available-case count remains applicable.

**B0 — NEEDS-RESTATEMENT.** Guard-first precedence and independently evaluated OR routes now handle both audit-3 counterexamples correctly. But “guard evaluable (≥6 eligible FI test cases)” does not explicitly require finite arm/control floors; route (a) similarly requires K1 floors without specifying finiteness. The tail route permits six valid K1 runs among eight eligible cases without saying whether both medians use the same six cases.
**Fix:** require finite paired data for floor comparisons and the guard; define missing/nonfinite evidence as UNAVAILABLE. Freeze one common case set for both tail medians, while retaining full eligible-population arm qualification.

**B2 Q1 fixture — CORRECT.** The bounded search recomputes the actual residual and fresh Jacobian and requires measured $\gamma>10^{-3}$. The previous nonlinear counterexample cannot pass falsely. Zero coefficients can make all perturbations zero, but the prescribed certification block handles that safely. Fresh tighter-anchor IDs remain fixed.

**B5 — NEEDS-RESTATEMENT.** Separate escalation/successor maps, own-state $\rho$, exceptional evidence handling, full evaluation of round 10, and BUDGET-CAP fix the listed defects. One cross-section conflict remains: Amendment 2 starts an uncertified selection at fib4181, whereas Amendment 3 B8 makes ROM routes UNAVAILABLE after selection exhaustion. A completed, qualified fallback trajectory therefore has two possible endpoint treatments.
**Fix:** explicitly choose whether selection exhaustion permits fallback policy evaluation and endpoint qualification; apply that choice consistently in B0, B5, B8 and synthetic transition tests.

**B8 — NEEDS-RESTATEMENT.** The total is 14.6 GPU-h and S3’s stated saving is corrected. But a job’s wall-time limit is not its GPU-hour limit when it requests two GPUs: a 2.8-hour reference job could consume 5.6 GPU-h. The combined 0.9-hour smoke allowance also lacks a split between SM-a and SM-b. Hard termination alone does not persist exhaustion outcomes.
**Fix:** specify wall limits as allocated GPU-hours divided by GPU count, split the smoke allowance, and reserve time for checkpoint/finalization or define collector-side timeout records. Charge failed attempts and retries against the same total.

**B9 — NEEDS-RESTATEMENT.** Naming largest-rule solves, finer factors, compilation and refinement improves coverage. However, node-count × step-count scaling does not bound adaptive tight/tighter solver iterations, and two natural development references need not measure the refinement ratio cited by the certificate.
**Fix:** enumerate solve multiplicities; use measured expensive-phase rates or explicit conservative bounds covering iteration caps, tighter anchors and refinement. If a bound is unavailable, mark the item uncertified and defer it. Retain synthetic tests for all repaired exceptional transitions.
