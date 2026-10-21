Read-only audit completed; no files changed, jobs submitted, or arrays copied. A3 takes precedence over earlier amendments.

5. **WRONG — Cross-check convergence remains insufficient.** A3 adds solver refinement and labels the departure from E0, but `d128 > d192` does not establish a convergent spatial tail. Nearly equal differences can pass while the unresolved error greatly exceeds their sum. **Fix:** use three-level observed-order/extrapolation evidence with an explicit acceptance threshold. Call \(U_{xc}\) a conditional indicator, not a certified bound; the absolute comparison floor does not validate a smaller \(\epsilon_{\rm ref}\).

10. **WRONG — Temporal acceptance and the omission exception remain incomplete.** The 20% improvement trigger addresses some inaccurate rows, but the fixed-case temporal check still has no acceptance criterion. Declaring every tested row invalid does not establish that a required family cannot compete at smaller steps. **Fix:** prescribe a temporal/stability acceptance rule; exhausted unresolved families should block a positive gate unless a budget-limited omission is explicitly justified under COMMON-BRIEF §5.

12. **WRONG — Monotonicity does not satisfy COMMON-BRIEF §3.** For a **fixed candidate population with fixed costs**, removing candidates cannot decrease its minimum cost, so the proposed inequality is mathematically valid. However:
   - §3 explicitly treats inherited qualification as prior evidence; A3 supplies no authorized exception.
   - The timed denominator is a historically selected winner, not an established minimum under the fresh timing measurements.
   - Removing every qualifying candidate leaves an undefined target, not automatically a qualified FAIL.

   **Fix:** perform fresh Q1/Q2 before issuing a qualified gate verdict, or obtain an explicit binding-brief exception permitting a conservative historical-proxy stopping screen. Pending that, a provisional PASS cannot authorize training, and a negative screen must not be presented as a §3-qualified FAIL.

14. **CORRECT — Timing decomposition resolved.** Separate query, prepared-setup kernel, and setup subjects avoid unreliable subtraction. No fix needed.

19. **WRONG — The reference-control requirements contradict the retained fixture.** A2’s `u3 = 2u2 − u1` gives equal successive differences: zero/undefined inferred order and a failed direction check. It cannot satisfy A3’s requirement that **every other** predicate remain true. **Fix:** replace it with an aligned, order-one fixture, for example \(u_2-u_3=(u_1-u_2)/2\), preserving \(t_0\); verify the remaining predicates. T-loose’s measured-margin requirement is acceptable provided failure to fire blocks production.

20. **CORRECT — Aggregate budget resolved.** Spent work, mandatory remaining work, extensions and retry reserve are included. No fix needed.

23. **CORRECT — The missing Q2 components are registered.** A3 specifies own-state operator discrepancy, both numerical bars, and all pooled cases. Define zero-denominator handling explicitly; implementation verification belongs to the code audit.

25. **CORRECT — Timing pairing resolved.** Frozen slots unambiguously map eligible FOM cases to ROM/anchor cases and bootstrap records.

27. **CORRECT — Test independence resolved.** Phase 1 owns the existing test cohort; Phase 2 gets a separately seeded, unopened cohort before training or tuning.

**NEW blocking design problems:** None beyond the unresolved items above.

**Overall verdict: WRONG — not CLEAN; items 5, 10, 12 and 19 remain blocking.**