Read-only audit completed; no files changed, jobs submitted, or arrays copied. A4 takes precedence.

5. **WRONG — Spatial tail estimate is inconsistent with the registered differences.** Here \(d_{128}\) and \(d_{192}\) are both distances **to the base**, not adjacent-level differences. Even an aligned geometric tail with \(d_{192}/d_{128}=0.1\) gives remaining base error \(0.125d_{192}\), exceeding A4’s \(0.11d_{192}\). The “resolved” alternatives also check only \(d_{128}\) or \(e_1\), allowing a larger \(d_{192}\) or \(e_2\) to escape.
   **Fix:** use adjacent spatial differences and an explicitly derived conditional extrapolation; require both spatial differences, or both temporal differences, below the floor when invoking the resolved alternative. The conditional-indicator label and explicit limitation concerning smaller \(\epsilon_{\rm ref}\) are appropriate.

10. **NEEDS-RESTATEMENT — Acceptance after extension is unspecified.** The \(10^{-3}\) threshold fixes the missing criterion, and unresolved competitors cannot invalidate an otherwise established negative comparison. But A4 only mandates extensions after failure; it does not require the temporal check to pass afterward. A family could retain “qualified” configurations under A1.12 despite failing this separate check.
   **Fix:** repeat the check using the two finest extended steps; persistent failure makes the family unresolved and blocks a positive target verdict. Negative conclusions remain conditional on resolving item 12.

12. **WRONG — Renaming the result does not establish a conservative screen.** A4 still lacks the binding-brief exception requested in round 4. Moreover, the historical **1.5× shortlist** need not contain the current fastest ROM. An omitted configuration can become faster in-job, reducing the denominator and increasing \(G\); therefore a shortlist-based FAIL-SCREEN is not conservative for the full registered population.
   **Fix:** perform fresh Q1/Q2 before a gate verdict, or obtain an explicit exception defining the permitted historical-proxy screen. For the claimed minimum, cover the entire registered population or justify exclusions with valid current cost bounds. Otherwise report an unqualified diagnostic and leave the gate unresolved.

19. **CORRECT — Order-control fixture repaired.** The replacement gives \(d_{23}=d_{12}/2\), hence \(p=1\) and direction discrepancy zero. Starting from an accepted fixture preserves the floor check because \(d_{23}\) increases; the other named checks remain unchanged. Production is explicitly blocked if the control fails to fire.

23. **CORRECT — Zero-denominator handling resolved.** Both operators zero gives zero discrepancy; a zero denominator with a nonzero numerator gives infinity.

**NEW blocking problem introduced by A4**

- **WRONG — ROM selection now uses stage-2 inference measurements.** A4.12 takes the minimum ROM cost in stage 2, undoing the selection/inference separation retained in A1.15. The existing bootstrap does not specify how this new selection enters its estimand.
  **Fix:** select and freeze the ROM denominator in stage 1, then independently measure it in stage 2. Describe it as the frozen selected configuration, not an established population minimum; any conservative population-minimum screen needs a separate justified procedure.

**Overall verdict: WRONG — not CLEAN; items 5, 12, and the new stage-2 selection problem remain blocking.**