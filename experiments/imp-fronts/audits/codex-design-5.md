Amendment 4 closes three audit-4 findings, but introduces a missing-evidence loophole in B0. **One WRONG; two NEEDS-RESTATEMENT.**

| Item | Verdict | Finding |
|---|---|---|
| A1 fixtures | CORRECT | Prerequisites and failure handling remain intact. |
| A2 curvature | CORRECT | Raw derivatives and absolute infinity norm resolve the units conflict. |
| A2 decision | CORRECT | Availability checks and descriptive curvature remain separate from the decision. |
| B0 | WRONG | Dropping missing floor evidence can falsely certify the worst-case guard. |
| B2 Q1 fixture | CORRECT | Actual nonlinear residual/Jacobian checks and certification blocking remain intact. |
| B5 | CORRECT | Uncertified starts now have one consistent endpoint treatment. |
| B8 | NEEDS-RESTATEMENT | Initial ceilings are operational; retry admission still relies on projected consumption. |
| B9 | NEEDS-RESTATEMENT | Listed solve multiplicities conflict with inherited Q1 certification on every run. |

**B0 — WRONG (lines 654–659).** Counterexample: eight eligible FI cases, six finite paired floors passing the guard, and two missing arm floors. Amendment 4 permits guard PASS and therefore CONTINUE, although either missing case could violate the registered worst-floor bar. Pairing fixes incomparable populations but does not certify the full eligible-population maximum.  
**Fix:** require finite paired floors on every eligible FI case before guard PASS; missing evidence makes the guard UNAVAILABLE, while an observed violation may still trigger STOP. Likewise, do not certify arm (a)’s eligible-population median after dropping missing arm/control floors; require complete evidence or explicitly register a narrower endpoint.

**B8 — NEEDS-RESTATEMENT (lines 666–671).** GPU-count conversion, smoke splitting, finalization reserve, and collector outcomes fix audit 4. However, `spent + certified remaining ≤ 15` uses a prediction that B9 expressly allows to underestimate actual consumption. A retry can consume the margin while outstanding jobs retain larger enforceable ceilings.  
**Fix:** admit submissions/retries only when spent plus reserved remaining hard ceilings—including running jobs and the proposed attempt—is ≤15 GPU-h; shorten ceilings or defer work otherwise.

**B9 — NEEDS-RESTATEMENT (lines 673–680 versus 426).** `{prod, succ, half, tight}` budgets one tight solve per round, but the inherited text requires Q1 certification “on every run.” Certifying successor and half-step runs also requires their tight comparisons and applicable tighter-anchor checks.  
**Fix:** either enumerate those additional solves, or explicitly restrict full Q1 certification to production candidates and certify a reused half-step run when promoted. Enumerate tighter-anchor multiplicities per visited state. Treat scaled timings as conservative projections, with enforceable budget protection supplied by B8.

Read-only audit completed; no files changed, scratch created, or jobs submitted.