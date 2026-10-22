1. **CORRECT — Decision table and selection (A4.2).**  
   Unresolved matching is separated from censoring; speed wins require matched competitors; unqualified test references provide no evidence rather than a loss. Either cohort can withdraw attribution, unsupported geometry claims are withdrawn, and deterministic max–min selection resolves the circular definition.

2. **WRONG — Qualification certificates (A4.3).**  
   Independent recomputation checks only the **final update**, not each stage as round 3 required. Intermediate stage errors can cancel in the update. The non-finite invalidation rule and zero-safe CN normalizations are corrected.  
   **Fix:** compare every production stage and final update against the independent transcription, with explicit zero-safe normalization and frozen tolerances. Define Q2’s population separately for validation (40 cases × 6 states) and confirmation (32 × 6), and explicitly skip zero-norm states even when the population maximum is zero; the skip-fraction gate then fails an all-zero population.

3. **CORRECT — Fault-control predicates (A4.4); verification remains outstanding.**  
   Both controls now require an independently established violation **and** successful detection. This fixes the conditional-test defect. However, the available v4a geometry controls do not establish these final cell-control results. Before release, supply the completed v4a evidence showing at least one violating fixture and successful detection for every required fault control. A promised ready-note entry is not verification.

4. **NEEDS-RESTATEMENT — Production cost/memory certificate (A4.5).**  
   The projection exists and honestly labels estimates, but remains a production-release blocker. Smoke6 measures bank training steps and ECSW fitting; it does not resolve the estimated selection/Q1/Q2/timing stage, SEM screening costs, or S1 scaling. The full projected panel is **88.52 GPU-h with headroom**, exceeding the **62 GPU-h** cap before spent time.  
   Require an audited, explicitly selected deferred panel, measured costs or justified conservative bounds for every remaining estimate, and total spent plus projected cost within the cap. Include quantitative host/device peak-memory bounds: chunking alone is insufficient evidence, particularly because `pod.py` also performs a full-basis device QR.

**OVERALL: NOT-READY.** Blocking fixes: complete stage-level certificates and Q2 edge-case definitions; verify final v4a fault controls; certify the selected production panel’s time and memory against the remaining budget.