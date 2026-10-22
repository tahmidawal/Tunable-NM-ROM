**Three blocking issues:**

1. **Resume changes the SEM search.** `run_cell.py` rebuilds `pareto()` from all recorded rows, including previously generated tolerance variants and excluded grades. Against the actual checkpoint, restarting proposes **15 additional eig variants**, including one at grading **0.7**. An uninterrupted run would not revisit this frontier. `registered()` filters final selection but does not prevent this search mutation. Preserve the original expansion frontier and apply registration filtering before adaptive decisions.

2. **Selection compares timing screens across jobs and implementation forms.** `select()` ranks old `screen` values alongside new measurements, then screens alternate implementations only for the first three candidates. Existing bank rows use point form; new ranks use tensor form. Consequently, job/device timing differences and the implementation switch can change which variants are explored and which two finalists survive. Final paired timing cannot recover discarded candidates. Re-screen eligible candidates on the resume GPU under a consistent, registered implementation policy before applying K=3.

3. **The resume exceeds the unrevised release-step cap.** A8 authorizes **12 GPU-h including meas6**. Those jobs have already consumed **11.6183 GPU-h**, leaving **0.3817 h**. The resume projects **3.6 h**, or **5.4 h with headroom**, and requests **8 h**. A9 proposes resumption but does not explicitly replace that cap. The 8-hour request exceeds the projected allowance comfortably; the blocker is authorization under the existing step cap. Amend the released budget explicitly.

Read-only audit completed; no files changed or jobs submitted.

**JOB(resume): NOT-READY (SEM replay changes the search; mixed-job/form selection screens; unrevised release-step cap).**