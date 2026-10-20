| Round-2 fix | Verdict | Round-3 finding |
|---|---|---|
| 1. Complete glossary; correct timing completeness and explain cost bands | **NEEDS-RESTATEMENT** | Timing completeness and cost bands are corrected. The glossary still falls short of the required completeness. |
| 2. Fix HTML placeholders and mathematical formatting | **NEEDS-RESTATEMENT** | Configuration templates now render correctly; qualification thresholds use LaTeX. Section 5 still prints `1024^2` as ordinary prose. |
| 3. Complete standalone figure context and identification | **CORRECT** | Added caveats, model-colour legend, and seed offsets address the reported issues. |
| 4. Explain inherited qualifications/retractions with sources | **NEEDS-RESTATEMENT** | The substantive explanation is accurate. Source links remain absent, and the added historical count is hard-coded. |

The re-audit findings are:

1. **Numerical provenance — NEEDS-RESTATEMENT; checked results CORRECT.**

   With filesystem writes blocked and outputs captured in memory, regeneration reproduced **`report.md` and all five PNGs byte-for-byte**.

   Independently checked **32 numerical cells** against EXPORT:

   - K, reconstruction error, oracle error, oracle/floor, code-refit gain and seconds for `mid_k4_s0`, `mid_k8_s1`, `mid_k16_s2`, `mid_k32_s1`.
   - Median and worst rollout errors for `H16s0|BE`, `H32s1|BE`, `L16|BE`, `L128|BE`.

   All passed. Registered DESIGN constants and the disclosed ledger are accepted under your exceptions.

   One new provenance defect remains: **“1 of 8 checks”** is literal text in [make_report.py:81](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/make_report.py:81). The canonical log supports it, but it is an empirical historical result—not a registered DESIGN constant or ledger value. Generate it from a source record, or omit the count while retaining the qualification.

2. **Headline claims — CORRECT.**

   F5 remains **inconclusive**, explicitly distinguished from a negative finding. The matched-dimension result is a trade-off. The at-most-cost claim identifies the qualified linear rung rather than asserting universal linear dominance. The numerical headline explicitly identifies its qualified-seed pool, exploratory scope and E0 qualification.

3. **Plots — CORRECT.**

   Inspected all five images and checked their generator logic. Qualification and timing usability are distinguished where applicable; the error–cost plot correctly shows H8s1 BE as numerically unqualified but timing-usable. The training plot identifies its truth-assisted diagnostic role. Seed offsets expose qualification at K=32. The cost plot states dev6, H200 and descriptive scope.

4. **Glossary — NEEDS-RESTATEMENT.**

   The important round-2 corrections are present, including the correct **at least one valid fused repeat per case** criterion.

   However, the [closing glossary](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/report.md:666) still leaves table concepts insufficiently defined: `arm` and its `identity`/`probe` variants, the timing scope of training `seconds`, and the aggregation behind `fused median (ms)`. Terms used elsewhere without plain-language definitions include **conditional Richardson estimate, POD, SiLU/MLP, whitening, initial-fit basin**, and **LM branch**. These matter under the explicit “every column and term” requirement.

5. **Relation to earlier results — CORRECT substantively; NEEDS-RESTATEMENT for source access.**

   The canonical log supports the matched-dimension historical account, the distinction from a numerical replication, and the quadrature-study head rule’s post-hoc/not-confirmed status. It also supports the newly stated reference-rule change and withdrawn frozen selection.

   The requested source links were not added: the report names `LAB-LOG.md` and amendment identifiers without linking readers to the relevant records.

6. **Formatting — NEEDS-RESTATEMENT.**

   All **27 tables** parse with consistent column counts. No inline HTML placeholders remain. LaTeX and Mermaid showed no syntax defect by inspection; their rendering engines were not run.

   One residual formatting defect is [report.md:256](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/report.md:256): the generated contract prints `1024^2` outside math delimiters. Format the mathematical portion as LaTeX.

7. **Cold-reader clarity — NEEDS-RESTATEMENT.**

   The scientific conclusion is appropriately restrained. The remaining obstacles are terminology, historical-source navigation, and the small provenance/formatting defects above. No additional unsupported scientific conclusion was found.

**NOT-READY. Ranked fixes:**

1. Complete the glossary’s remaining column meanings and technical terms.
2. Remove or source-generate the hard-coded historical count; add links to the historical qualification/retraction records.
3. Format Section 5’s mathematical contract text in LaTeX.

No files changed, jobs submitted, GPU used, or cluster accessed. No scratch files were created. The existing `HEARTBEAT.json` modification was untouched.