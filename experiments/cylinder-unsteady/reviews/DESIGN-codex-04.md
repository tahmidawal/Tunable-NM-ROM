
codex
Items 1 and 2 are resolved for the intended R1b workflow, but the cycle edits introduce a new blocker.

- **1 — Resolved.** `REF/r1_gates.py:48–104` requires the roundoff control, derives metric-specific floors and checks cross-record initial-state/window identity. Startup and stencil requirements remain enforced. The thread-count control demonstrates sensitivity to reduction order; it does not independently certify algebraic accuracy.

- **2 — Resolved.** `REF/compare.py:24–35,61–67,95–121` enforces completion, snapshot SHA checks, a single fixed-duration study, 141 saves and the exact eligible coarse force grid. The previous CLI-window loophole is removed.

- **NEW BLOCKER — optional pressure NaN breaks periodicity and persistence.** `REF/cycles.py:65–66` now legitimately emits NaN for an unavailable 1996 pressure evaluation. However, `cycles.py:85` requires **every field** to be finite before testing periodicity, so an optional diagnostic can reject otherwise periodic flow. Worse, `REF/transient.py:541–543` stores those records, while `transient.py:46` serializes with `allow_nan=False`, causing a completed study to fail during persistence. An in-memory synthetic check reproduced both failures.  
  **Required:** test finiteness only for quantities defining periodicity; serialize unavailable optional diagnostics as JSON `null`, preserving availability counts. Also handle zero valid observations in `cycles.py:76–77`, where `min()`/`max()` currently raise.

- **3 — Resolved as an explicitly heuristic design**, `DESIGN-G5.md:464–471`. Statistic-specific estimates and qualified reporting language are now specified. They support “resolved under the linear-propagation estimate,” not an independently validated reference-error bound. Keep that qualification wherever accuracy conclusions appear.

- **7 — Resolved at design level**, `DESIGN-G5.md:472–476`. Comparing fixed raw functions removes coordinate ambiguity; coverage includes every ranked arm and a denominator floor. Implementation still needs verification before ranking.

- **10 — Resolved at design level**, `DESIGN-G5.md:477–478`: exclude unavailable 1996 pressure samples and report counts. Its implementation remains blocked by the NaN handling above. The explicit preservation of A4’s original four-second failures at lines 479–480 is appropriate.

No files were changed; the reproduction ran entirely in memory.

**DESIGN: RELEASE** — for the amended protocol, retaining its heuristic uncertainty qualification and all dependent-stage gates.

**CODE: NOT READY** — fix optional-diagnostic NaN handling in periodicity, serialization and empty summaries before R1b.
tokens used
186,300
Items 1 and 2 are resolved for the intended R1b workflow, but the cycle edits introduce a new blocker.

- **1 — Resolved.** `REF/r1_gates.py:48–104` requires the roundoff control, derives metric-specific floors and checks cross-record initial-state/window identity. Startup and stencil requirements remain enforced. The thread-count control demonstrates sensitivity to reduction order; it does not independently certify algebraic accuracy.

- **2 — Resolved.** `REF/compare.py:24–35,61–67,95–121` enforces completion, snapshot SHA checks, a single fixed-duration study, 141 saves and the exact eligible coarse force grid. The previous CLI-window loophole is removed.

- **NEW BLOCKER — optional pressure NaN breaks periodicity and persistence.** `REF/cycles.py:65–66` now legitimately emits NaN for an unavailable 1996 pressure evaluation. However, `cycles.py:85` requires **every field** to be finite before testing periodicity, so an optional diagnostic can reject otherwise periodic flow. Worse, `REF/transient.py:541–543` stores those records, while `transient.py:46` serializes with `allow_nan=False`, causing a completed study to fail during persistence. An in-memory synthetic check reproduced both failures.  
  **Required:** test finiteness only for quantities defining periodicity; serialize unavailable optional diagnostics as JSON `null`, preserving availability counts. Also handle zero valid observations in `cycles.py:76–77`, where `min()`/`max()` currently raise.

- **3 — Resolved as an explicitly heuristic design**, `DESIGN-G5.md:464–471`. Statistic-specific estimates and qualified reporting language are now specified. They support “resolved under the linear-propagation estimate,” not an independently validated reference-error bound. Keep that qualification wherever accuracy conclusions appear.

- **7 — Resolved at design level**, `DESIGN-G5.md:472–476`. Comparing fixed raw functions removes coordinate ambiguity; coverage includes every ranked arm and a denominator floor. Implementation still needs verification before ranking.

- **10 — Resolved at design level**, `DESIGN-G5.md:477–478`: exclude unavailable 1996 pressure samples and report counts. Its implementation remains blocked by the NaN handling above. The explicit preservation of A4’s original four-second failures at lines 479–480 is appropriate.

No files were changed; the reproduction ran entirely in memory.

**DESIGN: RELEASE** — for the amended protocol, retaining its heuristic uncertainty qualification and all dependent-stage gates.

**CODE: NOT READY** — fix optional-diagnostic NaN handling in periodicity, serialization and empty summaries before R1b.
