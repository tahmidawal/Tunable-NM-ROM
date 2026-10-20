**CLOSED.** `report.md:107` now reports zero rejected LM steps for all 12 deployed 2D settings. Verified directly against `result.json`’s `rows[].rejected_total`: 38 cases per deployed arm, 456 rows total, every value zero.

The generator correctly sums by setting and deployed arm. No new blocking issue found in the change. No files modified.

REPORT-OK: YES