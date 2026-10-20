**NO-GO.** The artifact-saving fix introduces a definite crash: [bankeval.py:522](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-smooth-bank/experiments/jcp-smooth-bank/bankeval.py:522) overwrites the output directory with `out = []`; line 534 then evaluates `out / filename`, raising `TypeError`. Task 4 fails and paired timing is skipped.

| Previous item | Verdict |
|---|---|
| Payload hashes | **PASS:** 117 entries; all 21 source/config files equal HEAD; 12 tr2 artifacts, 4 base artifacts and 4 large rotations verified. |
| References | **PASS:** 77 files and 76 `f257` hashes verified; acceptance checks pass. |
| Variant mapping | **PASS:** five banks/rotations correctly mapped; base included in paired timing. |
| Batch execution | **PASS:** syntax, GPU preflights, precision and failure propagation unchanged. |
| FOM grid/time alignment | **PASS:** unchanged and correct. |
| A1.7/A2.6 scoring | **PASS:** unchanged and correct. |
| FOM acceptance/report withholding | **PASS in code:** finite fields/residuals and ≤1e-8 required; comparison withheld unless all 76 rows accepted. |
| FOM audit artifacts | **FAIL:** path/list collision prevents saving. |
| Resources/provenance | **PASS:** A7 explicitly allows five A100s, any memory size, ten hours/50 GPU-hours. Recorded commit precedes HEAD only by staging records; payload matches HEAD. |

Keep the output path and results list in separate variables, then restage. No files modified.