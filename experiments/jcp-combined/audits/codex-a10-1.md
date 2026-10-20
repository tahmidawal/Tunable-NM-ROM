**Overall: WRONG as an end-to-end implementation.** The core post-hoc selector is correct for complete `c3run.py` outputs, but downstream freezing loses the new converged rule and the post-hoc label. Some diagnostic claims also need qualification.

1. **Selection logic — CORRECT.**  
   `posthoc_converged` combines fixed BE’s **individual selection status** with adaptive CN’s **individual diagnostic status**, preserving unresolved precedence. It chooses `gl56` from the `gl56|gl64` pair, otherwise `gl64` from `gl64|gl80`; `gl48|gl56` cannot establish convergence. All 81 status combinations passed an independent check.

   Ladder qualification preserves BE’s A6.2 eligibility, uses adaptive CN validity, compares each arm against the same arm at the new converged rule, and retains the tolerance, minimum qualifying rung, control veto, cost tie-breaking and fallbacks. Missing escalation evidence cannot establish convergence. See [select3d.py:46](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/select3d.py:46).

2. **Adaptive eligibility — CORRECT.**  
   Here, adaptive `valid` really does mean **every step passed the implemented verification**, plus finite saved coefficients. `run_valid` requires `nfail == 0`; `t2core` increments `nfail` for each failed returned-state verification. Thus reconstructing eligibility from all rows’ `valid` flags matches `elig_of(...)[all_valid]`. The selector also requires exactly the configured case multiset, catching missing or duplicate rows.

   There is **no 1% failure allowance** for adaptive arms. However, “verified” means the stopping criteria passed—not a mathematical bound on solver error. See [c3common.py:218](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c3common.py:218).

3. **Cost — NEEDS-RESTATEMENT.**  
   The implementation correctly sums median **`tB` times of the original fixed-sweep BE/CN arms**. A10 explicitly discloses this, and the JSON distinguishes `selection_arms` from `timed_arms`. It is defensible as a retained fixed-sweep timing criterion; it does **not** identify the cheapest adaptive selection pair.

   Two header statements are stale: `select3d.py` describes median paired **ratios**, and describes timing baseline A as the converged rule. For the failed original selections, `c3run` actually timed against its diagnostic baseline, typically `gl80`. Neither changes the implemented raw-`tB` criterion, but both should be corrected.

4. **Bugs, keys and ordering — WRONG.**

   - **Downstream convergence/label loss:** selection retains original `converged` and writes the replacement as `posthoc_converged`. [freeze.py:88](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/freeze.py:88) copies only original `converged`, omitting `posthoc` and `label`. [make_budget_configs.py:39](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/make_budget_configs.py:39) then skips settings whose original convergence is `None`. A10 is not integrated safely downstream.
   - **Diagnostic validity gap:** `cnfail3d.py` does not require or report adaptive validity before describing its comparator as verified. Its failing set measures distance failure, not full A6.2 ineligibility.
   - **Coverage gap:** `cnfail3d.py` skips absent adaptive arrays and can report identical failing sets for incomplete populations—or vacuously for zero/one rule. Require complete, common case coverage.
   - **Ordering:** current producer ordering is handled correctly. Selection asserts both arrays match configured case order; diagnostics assert adaptive order matches fixed order and normalize by ID. But IDs are optional, so missing metadata silently restores positional assumptions rather than enforcing A8.2.
   - The initial adaptive iteration-key bug (`it` instead of `step_it`) **was corrected by another writer during this audit**; it is not a remaining finding.

5. **A10 honesty and pre-registration — NEEDS-RESTATEMENT.**  
   A10 clearly declares the change post hoc, preserves the unavailable pre-registered outcome, and explicitly substitutes adaptive CN. That is honest; the selector does not silently relax BE eligibility or the quadrature threshold.

   “Asks the same question” is too strong. It now establishes quadrature agreement for **fixed BE and adaptive CN**, not the failing fixed-CN trajectory or every grid candidate. Likewise, per-step stationarity verification does not certify small solver error. Retain the later arm-specific solver and quadrature qualification gates.

   The promised label in **every downstream file** is currently false because freezing drops it. Also, I could not independently verify A10.1’s production-case claims: only smoke archives were available locally.

No experiments were run or code changed by me. The audit is recorded in the canonical lab log.