ACCEPT-WITH-CORRECTIONS

Paths below are relative to `experiments/jcp-dims/`.

1. **Low — LMF setting disappears from the report.** `make_report.py:147` uses double-escaped `\\d` inside raw regex strings, so no configuration matches. Consequently, `report.md:611` ends its heading with “test set at” and no setting. Use `\d`; the data identify $(r,M)=(384,1536)$.

2. **Low — H2q PNG still lacks its provisional qualification.** `make_report.py:392` and `plots/h2q_3d.png` omit E0-provisional winner selection. The adjacent report caption now supplies it (`report.md:897`), but the standalone image does not. Include that qualification in the image.

3. **Low — Literal-number and mathematical-formatting cleanup remains incomplete.** `make_report.py:541` hard-codes screening/confirmation case and repetition counts. Derive these from configuration files. Mathematical expressions also remain outside LaTeX, including `M=4r` in `report.md:45` and `n×n`/`n³` in `report.md:955`.

**Verified:** All ten PNGs viewed. Secondary export contracts are present; the front includes all 101 qualified screened configurations, grouped by bank width, with the unqualified comparator separately marked. Heat maps cover all eight rungs, unusable timings are labelled, and the H2q annotation no longer overlaps. Requested glossary definitions, table qualifications, and comparison scope are present. All 21 generated export sections and 57 report-section fragments match their sources in memory.

`H2Q_PASS` equals `1/.6` exactly. Re-executing h2q’s confirmation branch on `analysis-3d.json` reproduces all six results exactly: primary **1.149138 / 2.952802 / 9.874135**, secondary **1.145831 / 2.929196 / 9.670805**, each **fail / pass / pass**. Winner selection was not independently recomputed: that summary JSON lacks its per-case inputs.

Read-only throughout; no files changed, submissions made, arrays copied, or scratch created.