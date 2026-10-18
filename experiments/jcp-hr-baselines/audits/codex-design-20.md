**Overall: NEEDS-RESTATEMENT.** A30.2 resolves round-19 item 9. A30.1 restores the required coverage, but its stated drop priority conflicts with its execution order and budget guards. No numerical WRONG finding.

Audited HEAD: `cbf5a82dc53785b78546f705dffb5de4bf6b3283`.

References: `D` = `experiments/jcp-hr-baselines/DESIGN.md`; `T` = `experiments/jcp-hr-baselines/tune.py`. `Q` and `L` are respectively `experiments/jcp-qualify/runs/EXPORT.json` and `GPU-LEDGER.json` in the supplied qualification worktree.

1. **NEEDS-RESTATEMENT — A30.1 resolves the obsolete coverage rules, but not the defer-order ambiguity.**

   The required coverage is now consistent: both solvers run `none` at every rank and the complete HR grids; deferred options exclude corresponding comparators; missing Stage-B timings prevent filling affected roles (`D:1592–1602`, `D:556–558`, `D:633–635`, `D:1502–1506`). The implementation matches the revised execution order: HR grids → all-rank `none` → limiting comparator → extension → probes (`T:401–405`, `T:432–471`, `T:475–504`).

   However, **“first dropped first”** places extensions before probes (`D:1592–1596`), while execution prioritizes extensions (`D:1601`). Both use the same elapsed-time threshold (`T:398–399`, `T:452`, `T:487–488`), without reserving probe time. Thus an extension can start below the threshold, consume the remaining allowance, and cause probes to be deferred while the extension completes. That is not the declared drop priority.

   **Concrete fix:** Replace the global “first dropped first” claim with the actual sequential start-threshold policy, explicitly stating that extensions may consume the allowance before probes start. Alternatively, preserve probe capacity before admitting extensions and amend the execution policy accordingly. Retain the never-dropped coverage and incomplete-role rules.

2. **CORRECT — A30.2 resolves round-19 item 9 and preserves A21.3’s obligations.**

   Shared initial execution replaces a separate timing pass; joint Stage-B membership and subsequent pick freezing follow A26.5. Cost charging remains explicit rather than supposedly automatic (`D:1604–1612`, `D:1326–1331`, `D:1502–1506`). Wall limits provide the hard spending bound, not a guarantee of workload completion (`D:1494–1500`, `D:1554–1558`).

   The qualification export’s computed SHA-256 exactly matches:
   `6cb394d149be8852260b75a585db8102791ef52e9963ab6934f67d536c374431`.

   All four ledger entries agree:

   | Job | GPU-hours | Evidence |
   |---|---:|---|
   | smk3 / 5366486 | 0.0622 | `L:29–38` |
   | q3d / 5396416 | 0.1456 | `L:89–98` |
   | q3d2 / 5422698 | 0.1036 | `L:173–182` |
   | q3d3 / 5440837 | 0.1761 | `L:185–194` |
   | **Sum** | **0.4875** | `D:1609–1610` |

   Each records one GPU and one attempt. The four admitted configurations are exactly the export’s four passes (`Q:1895`, `Q:1905`, `Q:1909`, `Q:1911`).

   **Concrete fix:** None required.

3. **CORRECT — no additional substantive inconsistency found among A19–A30.**

   Later amendments consistently govern the full-field query contract (`D:1314–1324`, `D:1583–1586`), diagnostic-only batch 1 (`D:1337–1342`, `D:1612`), retrospective memory acceptance (`D:1537–1540`), and Stage-B membership/pick timing (`D:1502–1506`). Batch 2 retains the qualification export’s subset restriction (`D:1576–1577`, `Q:1864`).

   **Concrete fix:** None beyond item 1.

4. **NEEDS-RESTATEMENT — A19.4(a) is not yet discharged.**

   A26.7 permits successor-audit acceptance, but item 1 remains a substantive ambiguity about which evidence budget pressure removes (`D:1512–1513`). A28.7 correctly keeps design acceptance separate from code/gate, resource-artifact and amended-smoke acceptance (`D:1564–1566`).

   **Concrete fix:** Resolve item 1 and record acceptance against the amended revision. This audit does not authorize production.

No files changed, jobs submitted, GPU work or tests run, or scratch created.