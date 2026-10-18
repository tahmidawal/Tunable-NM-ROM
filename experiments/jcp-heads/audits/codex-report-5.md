| Round-4 fix | Verdict | Finding |
|---|---|---|
| 1. Complete, clickable historical references | **NEEDS-RESTATEMENT** | Abbreviated paths were expanded and historical entries identified accurately. References remain backticked/plain text, not clickable links. |
| 2. Complete glossary | **CORRECT** | Relative RMS and its normalisation, rung, and all jobs-table columns are now defined. The RMS definition matches the fitting code. |

1. **Numerical provenance — CORRECT.** With filesystem writes blocked and outputs captured in memory, regeneration reproduced `report.md` and all five PNGs **byte-for-byte**. Independently checked **32 numerical cells** against EXPORT: six training values for each of `mid_k4_s0`, `mid_k8_s1`, `mid_k16_s2`, and `mid_k32_s1`; median/worst errors for `H16s0|BE`, `H32s1|BE`, `L16|BE`, and `L128|BE`. All match displayed precision. Registered DESIGN constants and the disclosed ledger are accepted.

2. **Headline claims — CORRECT.** F5 remains **inconclusive**, explicitly distinguished from negative evidence. Matched dimension remains a trade-off; the at-most-cost conclusion names the qualified linear rung. “Linear dominates” appears in the specific bracket comparison, not as a universal conclusion. Qualified seed pools, exploratory scope, and E0-conditional caveats are explicit.

3. **Plots — CORRECT.** Inspected all five images and their generation logic. Qualification and timing usability are distinguished where applicable. Pooled error uses per-case seed medians before aggregation across cases. Training oracles are truth-assisted diagnostics; cost-versus-K is descriptive.

4. **Glossary — CORRECT.** The round-4 omissions are resolved. No additional material undefined table column or term was found.

5. **Earlier results — CORRECT substantively; NEEDS-RESTATEMENT for navigation.** The canonical log supports the historical matched-dimension claim and the quadrature head rule’s post-hoc, unconfirmed status. The report appropriately distinguishes this experiment from a numerical replication and states the inherited E0 qualification and combined-selection retraction plainly.

6. **Formatting — CORRECT by structural inspection.** All **27 tables** have consistent column counts. Title, status line, Mermaid diagram, inline qualifications, and closing glossary are present. No LaTeX or Mermaid syntax defect was identified; dedicated renderers were not run.

7. **Cold-reader clarity — NEEDS-RESTATEMENT.** The scientific interpretation is appropriately restricted. Historical source navigation remains the sole unresolved finding. No prohibited comparison or new unsupported scientific claim was found.

**NOT-READY. Ranked fix list:**

1. Convert the historical references in the [opening status paragraph](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/report.md:3) and [Section 6d](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/report.md:640) into complete, usable Markdown links to the relevant reports/log entries. Make this change in `make_report.py`.

No files changed, jobs submitted, GPU used, or cluster accessed. No scratch files were created.