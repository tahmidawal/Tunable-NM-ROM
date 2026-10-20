**Overall: NOT CLEAN** at HEAD `1d39f5768`.

| Item | Verdict | Evidence |
|---|---|---|
| Finding 6 | RESOLVED | Wide summary states sensitivity coverage and its absence at finer steps; both plots reference the report. |
| Finding 8 | RESOLVED | Calibration statuses and Newton–BiCGStab are explained; accuracy plots explicitly allow untimed configurations. |
| Finding 10 | STILL-WRONG | Required qualifications were added, but both wide PNG titles now overflow and are clipped. The cost plot truncates the sensitivity reference. |
| R3 SyntaxError | RESOLVED | HEAD `make_report.py` compiles successfully in memory. |

No additional blocking errors found. CPU only; no files modified.