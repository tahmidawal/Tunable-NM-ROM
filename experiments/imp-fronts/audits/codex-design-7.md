Amendment 8 closes audit #6’s loophole. Amendments 9–10 have correct headline arithmetic, but the completion certificate and smoke allocation need clarification.

| Item | Verdict | Finding / fix |
|---|---|---|
| A8: B0 early STOP | CORRECT | An observed arm maximum exceeding twice the **complete eligible FI K1 maximum** proves violation despite missing arm floors. Retain FI ≥ 6; incomplete K1 evidence cannot authorize this exception. |
| A9: reference rates | NEEDS-RESTATEMENT | Replacing maxima with means is acceptable for an **expected-cost projection**, not a completion bound. Explain the H200 mean revision from ≈252 to 288.5 s and pin the records/population used. |
| A9: tight/tighter rates | NEEDS-RESTATEMENT | Node/step scaling and a 1.5 multiplier are empirical projections, not guaranteed bounds. Explicitly price tighter solves and setup/compilation; retain the larger history/smoke rate on matching hardware. |
| A9: R384-only selection | CORRECT | Only R384 feeds these endpoints/policies. The existing `select2d_r384.py` wrapper restricts ranks while preserving selection checks; name that wrapper rather than implying the unmodified entry point accepts one group. |
| A9: pooled stationarity | CORRECT | The additional job-wide pooled ≤1% requirement preserves the common-brief requirement alongside existing per-run checks. |
| A10: arithmetic | CORRECT | 28 × 288.5 × 1.5 / 3600 = **3.3658 h**; 24 cases = **2.885 h**; 0.229 × 1.5 = **0.3435 h**. S1 = **14.8**, S3 = **9.55 GPU-h**. |
| A10: work cutoff | NEEDS-RESTATEMENT | Amendment 4 stops new work at 90%: references have **2.61 GPU-h** before that cutoff, selection **0.315** per arm. Their projections exceed these amounts. Separate work/finalisation costs and certify against the actual cutoff, or explicitly amend the reserve rule. |
| A10: policy projection | NEEDS-RESTATEMENT | State the complete formula: measured per-case cost × **40 cases** ×1.5, plus uncovered overhead and terminal tighter anchors. Identify the smoke bank/start rules and justify transferring its path distribution to both retrains; exercising transitions alone does not validate that transfer. |
| A10: “worst-path upper bounds” | NEEDS-RESTATEMENT | These use empirical solve-rate scaling, so call them **worst-path cost projections**. Hard ceilings/admission enforce expenditure; they do not guarantee completion. |
| Smoke allocation—known open item | WRONG | 0.45 + 0.45 exhausts the 0.9 allocation, leaving no explicit ceiling for the separate two-dev-reference job. Register a three-job split totaling ≤0.9, or rebalance the schedules; S1 has only **0.2 GPU-h** spare. |
| E8+E9 cap and deliverables | NEEDS-RESTATEMENT | Admission still protects the shared 15-hour cap. S1 does not reserve the earlier **0.3-hour E9 timing** requirement; explicitly label fallback cost deferred unless actual savings fund it. Flagged fractions alone leave that brief deliverable incomplete. |

Read-only throughout: no changes, scratch, jobs, or lab-log writes.