**REPORT READY**

- The revised “pilot” definition in [report.md:495](experiments/jcp-fom2d/report.md#L495) correctly distinguishes resource pilots—first and warm executions with timing, compilation and memory records—from C2 memory pilots: untimed FOM solves with the 4096² ROM bank resident, checking convergence, finite output and memory. This agrees with `ladder.py:330–366`, `c2time.py:391–402`, and `configs/smkc2.json`.
- The generator diff changes only that glossary entry. The report’s only other changes update the audit history to include Round 3 and supersede Round 2. No numerical results or substantive claims changed; no new error found.

No files changed, no scratch created, and no cluster commands run.