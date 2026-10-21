Read-only audit completed with small in-memory CPU fixtures. No files changed, jobs submitted, GPU execution, or large arrays copied. Paths below are relative to `experiments/imp-varcoef/`.

| Item | Verdict | Evidence and fix |
|---|---|---|
| **12 — diagnostic validation** | **WRONG** | `vtime.py:258,293–298` correctly orders queries before kernels and compares their lattice fields. However, `gate.py:101–110` never checks kernel scoring `repro`. A CPU fixture with `repro=False` still returned a numerical kernel cost. **Fix:** require exact scoring coverage with successful reproduction for kernels, alongside field agreement, validity, identity, and timing checks. Missing decompositions now correctly report “unavailable.” |
| **24 — reporting** | **WRONG** | `gate.py:236–249` now reports the explicit paired PSI C100/C1 ratio even when PSI is not the union winner. But only its C1 denominator receives diagnostic validation; the C100 numerator needs only complete timing coverage. **Fix:** require successful C100 scoring/reproduction and absence from `veto` before emitting the ratio; otherwise report unavailable with the numerator’s failure reason. |
| **Round-3 new diagnostic-reporting bug** | **WRONG — partially fixed** | `gate.py:101–109` correctly rejects invalid C1 scoring, kernel field mismatches, identity vetoes, and incomplete timing. Failed kernel reproduction remains accepted, as described under item 12. **Fix:** add the missing kernel reproduction predicate. |
| **13 — selection-copy hardening** | **CORRECT** | `make_configs.py:177–180` checks the opening record before copying the selection. A refused sequential replay no longer overwrites it. Deliberate replay/concurrent-submitter residuals remain **NOTED**, nonblocking under A6.4. |
| **CCv docstring correction** | **CORRECT** | `vfd.py:308–310` now agrees with denominator preparation and consumption at `vfd.py:321–327`. No further fix needed. |

**NEW bug introduced by this diff:** `gate.py:244–249` can publish a paired ratio from an invalid or vetoed C100 execution. The CPU fixture returned **3.0** despite C100 scoring having `valid=False`, `repro=False`, and the subject being vetoed. This represents an ordinary failed-execution path, not a forgery-only finding: timing identity and coverage do not establish solver validity. Apply the numerator checks described under item 24.

No new real-size crash was identified by source inspection; production-size execution was not performed.

**Overall verdict: NOT CLEAN — WRONG findings remain in diagnostic reproduction and paired-ratio acceptance.**