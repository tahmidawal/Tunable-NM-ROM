**ACCEPT at HEAD `54d557192` for applying A10 (+A10.3) to the cross-check, with its stated conditional interpretation.** The three targeted repairs pass.

| Item | Re-audit result |
|---|---|
| Auditor’s old-rule overflow discrepancy | **Repaired.** Actual 3D/0 PS arrays plus finite FD entries of `1e200` now produce old-rule `UNRESOLVED` in both implementations; `match=True`. |
| Invalid `n0` | **Repaired.** `inf`, `-inf`, `NaN`, `-1`, and `0` produce `FD-INVALID` consistently in `crosscheck`, `old_rule_verdict`, and the independent auditor. |
| Report disclosure | **Repaired.** `make_report.py:272–280` explicitly discloses the change after PS self-convergence inspection, narrows the chronology to **new second-order** FD references, states the floor assumption, and limits PASS to conditional scoring-lattice agreement. |

Rerunning all eight saved PS cases against synthetic matching FD fields reproduced the previous numerical results: current-rule PASS for all eight; old-rule UNRESOLVED for 2D/0–2 and PASS otherwise. Every auditor comparison had `match=True`.

The floor-threshold, undefined-order, mixed floor/non-floor, stale-finite/NaN-array, NaN-epsilon, gate, stability, and FD-outcome reproductions also retain the expected behavior. **No new error found in this narrow re-audit.**

Previous prose defects remain: DESIGN’s original “there” claim still incorrectly locates the initial discrepancy on the walls—recomputed wall differences are zero—and the report’s introductory “self-difference” wording remains imprecise. These are unchanged documentation issues, not remaining blockers to applying the explicitly qualified A10.3 rule.

These synthetic tests establish implementation consistency, **not actual PS–FD agreement**. No files were modified; the report generator was inspected without executing its writes.