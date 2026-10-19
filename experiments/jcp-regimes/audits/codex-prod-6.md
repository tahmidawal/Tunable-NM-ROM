**NEEDS-FIX** at `693aeb8c5`.

- **Parse fix verified:** the expression closes correctly at [make_report.py:98](experiments/jcp-regimes/make_report.py:98). All top-level lane Python files compile.
- **Normal end-to-end test passed:** nine production configurations generated `report.md`, four plots, and `EXPORT.json`. Labelling precedence, exclusions, coverage, pairing, and trend aggregation passed.
- **Remaining runtime blocker:** when no cases have usable errors, analysis legitimately produces null medians ([analyze.py:99](experiments/jcp-regimes/analyze.py:99)). The report unconditionally selects logarithmic scaling at [make_report.py:47](experiments/jcp-regimes/make_report.py:47), then crashes at line 49 with `ValueError: Data has no positive values, and therefore cannot be log-scaled.` Since `main()` calls plotting first, neither report nor export is written.

Guard plots against having no finite positive values and render an explicit “no usable data” panel.

Reproducers: `/tmp/jcp-regimes-audit-scratch/check.py` and `edge.py`. Scratch usage: 808 KB. No repository changes, jobs, or lab-log writes.