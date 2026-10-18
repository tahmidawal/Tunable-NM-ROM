**RESULTS-AND-REPORT-CONFIRMED** — HEAD `d9081edac`.

| Round-3 item | Disposition | Evidence |
|---|---|---|
| Historical-value provenance | **RESOLVED** | `report.md:3` explicitly identifies transcribed `HIST_CLAIMS` constants. Checked the cited historical tables: values agree. |
| EXPORT population definitions | **RESOLVED** | `runs/EXPORT.json:706` defines sub16 as dev6[0:6] + val32[0:10], identifies S exclusions, and supplies per-configuration exclusions. All 44 exported population records agree with analysis. |
| Tie definition | **RESOLVED** | `report.md:257` requires qualification for ties and separates unqualified cases, matching `p1stats.py`. |
| Change-statistic populations | **RESOLVED** | `report.md:258` correctly distinguishes median over resolved cases from maximum over all eligible cases. |
| P256/P4096 | **RESOLVED** | `report.md:259` defines mesh pairs and populations. |
| W512/W1024 and lattice/Gauss names | **RESOLVED** | `report.md:260` defines bank widths and quadrature point counts. |
| Fused query / A | **RESOLVED** | `report.md:261` defines the complete-query numerator and flanking evolution-time denominator, matching `p1core.py` and `p1time.py`. |
| Previously resolved round-3 items | **RESOLVED** | No regressions in conditional labels, scientific qualifications, LaTeX, plot caption, memory inventory, or prohibited comparisons. |

Read `report.md` end to end: no additional wrong claim, unsupported numerical value, or material glossary omission found. `checks/analysis.json` and plot files are unchanged from round 3; numerical conclusions remain unchanged.

No files changed, jobs submitted, or scratch/archive copies created.