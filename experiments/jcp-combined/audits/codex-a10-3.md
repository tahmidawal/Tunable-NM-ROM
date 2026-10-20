**Overall: CORRECT for the requested round-3 fixes.** Round-2 items 4–7 are resolved.

4. **`cnfail3d.py` and both diagnostic JSONs — CORRECT.** Expected coverage now comes from `esc_rules ∪ ladder`, adding `gl80` when `gl64|gl80` exists. Missing fixed or adaptive arrays abort instead of silently shrinking coverage. Both archives and outputs contain all **13 expected rules**, with matching case populations, diagnostic fields, failure sets and archived solver-distance maxima.

5. **A10.1 — CORRECT.** Both jobs have fixed CN K=2 ineligible at all 13 rules, all three escalation selection axes `unresolved`, and original convergence `None`. The common non-control failures are **5, 7, 11, 30, 40, 43, 62**. W1024 additionally fails case **46 at `gl24`**, with distance **0.0002588719983**, just above 0.00025. Smoke cases 0 and 1 miss these non-control failures.

   `lat256` exceeds the distance threshold on all 64 cases. W1024’s adaptive control comparator is invalid for case 62; its aggregate diagnostic verdict is correctly **`null`**, not an assertion of verified agreement or disagreement. W768’s verdict is **`true`**.

6. **Downstream label propagation / A10.2–A10.3 — CORRECT.** Both config generators carry `rule_selection_label` and `rule_selection_posthoc`. `score.py`, `analyze3d.py` and `budget3d.py` now copy those fields from the run config into their serialized outputs. Post-hoc and default expressions checked successfully. The scientific scope remains appropriately limited to fixed BE and adaptive CN, with later arm-specific gates retained.

7. **A10.2a — CORRECT.** In each archive, all **256 corresponding duplicate `gl48` row pairs** differ only in `seconds_first`. Case ordering matches across blocks, and the join selects the last block. The revised text correctly declines to claim bitwise equality of two trajectory copies when only one survives.

Checks used archived arrays, rows, diagnostics and source expressions; full bank decoding was not rerun. **No files were created, modified or appended.**