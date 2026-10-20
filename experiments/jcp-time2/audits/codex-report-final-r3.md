**Overall: NOT CLEAN** at HEAD `6477e9c37`.

The committed report and all nine plots are unchanged from the previous audit.

| Finding | Verdict | Remaining issue |
|---|---|---|
| 6 | STILL-WRONG | Wide summary and plots still omit quadrature-sensitivity context, including unavailable fine-step sensitivity. |
| 8 | STILL-WRONG | Calibration statuses and Newton–BiCGStab remain unexplained; eligibility wording still incorrectly requires timing for accuracy plots. |
| 10 | STILL-WRONG | Wide cost plot still omits “unselected”; both wide plots lack sensitivity context. |

**New blocking error:** `experiments/jcp-time2/make_report.py:650` has a confirmed `SyntaxError`: unescaped apostrophes in the new glossary string prevent report regeneration.

No files modified.