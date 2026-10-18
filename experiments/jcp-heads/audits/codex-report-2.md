Round-1 ranked fixes:

| Fix | Verdict | Round-2 finding |
|---|---|---|
| 1. Explicit model names and precise metric/statistical meanings | **NEEDS-RESTATEMENT** | Head/linear labels and the error denominator are fixed. The equal-cost decision band and several statistical terms remain unexplained. |
| 2. Complete glossary | **WRONG** | Expanded substantially, but still incomplete; “timing complete” also overstates its actual criterion. |
| 3. Numerical provenance and ledger disclosure | **CORRECT under your stated exceptions** | Checked results trace to EXPORT; the ledger is disclosed. Registered DESIGN constants are acceptable without replacing every literal. |
| 4. Inline pooled-model and provisional caveats | **NEEDS-RESTATEMENT** | Headline fixed. Standalone figure caveats and explanations of inherited qualifications remain incomplete. |
| 5. Plot overlap and timing usability | **CORRECT** | Error–cost markers now distinguish both properties; selected annotations are readable. Horizontal offsets expose qualification in the error-versus-K plot. |
| 6. Historical scope, GPU provenance, unavailable values | **CORRECT** | Historical comparison is explicitly non-replicative; rule selection is policy-specific; GPU models and `n/a` are supplied. |

The re-audit finds no numerical discrepancy, but several reporting requirements remain unmet.

1. **Numbers — CORRECT under the allowed source exceptions.**

   With filesystem writes blocked, in-memory regeneration reproduced `report.md` and **all five PNGs byte-for-byte**.

   Independently checked **32 numerical cells** against EXPORT:

   - Six fields each for `mid_k4_s0`, `mid_k8_s1`, `mid_k16_s2`, and `mid_k32_s1`: K, reconstruction error, oracle error, oracle/floor, code-refit gain, and seconds.
   - Median/worst rollout errors for `H16s0|BE`, `H32s1|BE`, `L16|BE`, and `L128|BE`.

   All passed. The headline agrees with EXPORT: pooled head **1.440246%**, linear **10.171310% / 1.095338%**, cost ratios **0.264808 / 0.591000**. “Every number comes from EXPORT alone” remains literally false, but the disclosed ledger and registered constants satisfy your exception.

2. **Headline claims — CORRECT.**

   F5 is **inconclusive**, explicitly not a negative finding. The matched-dimension result remains a trade-off. The at-most-cost statement names rung 128 without claiming universal linear dominance. The pooled headline now states both E0-conditional status and that it does not describe one deployable model.

   “Linear dominates” in the bracket tables is a registered, rung-specific descriptive label; it is not an unsupported headline expansion.

3. **Plots — NEEDS-RESTATEMENT.**

   Data and marker logic are correct. In particular, **H8s1 BE is an open circle**: numerically unqualified, timing usable.

   Remaining issues:

   - `seed_spread.png` lacks its own validation/exploratory and E0-conditional caveats.
   - `training_oracle_vs_K.png` should identify itself as a truth-assisted pipeline-test representation diagnostic. An E0 caveat would be inappropriate for that figure.
   - `cost_vs_K.png` says “descriptive” but omits the dev6 population and H200 context.
   - The error–cost figure has no legend decoding head/linear/incumbent colours or unannotated points. Its selected labels solve the overlap problem but leave identification incomplete.
   - Seed-spread markers still nearly coincide at K=32; horizontal offsets would make qualification easier to read.

4. **Glossary — WRONG against the completeness requirement.**

   The [glossary](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/report.md:666) still does not explain, among other items:

   - `stationary`, `residual`, and `other (invalid)` exit categories;
   - transition `lo`/`hi`, improve/worsen/tie counts;
   - rule-selection group/trail and “passing arms / needed”;
   - successor distance, tight/tighter anchors;
   - dev6/val32 and the statistical meaning of the intervals;
   - the fitted exponent \(p\), distinct from the p-value;
   - the equal-cost band and unresolved cost labels.

   **Incorrect definition:** “timing complete” says every registered repeat has valid, hash-matched measurements. The implementation requires registered repeat coverage, but **at least one valid fused repeat per case**, not validity of every repeat. See [analyze.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/analyze.py:243).

5. **Earlier results — CORRECT historically; NEEDS-RESTATEMENT for inherited qualifications.**

   The canonical log supports both the September head-ablation account and the quadrature-study rule’s post-hoc, not-confirmed-on-test64 status. The report correctly distinguishes comparison arms/settings and avoids presenting this as a numerical replication.

   However, [the opening status paragraph](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/report.md:3) merely lists “E0 A10; combined A10, A11 and the retracted A15.” Those identifiers do not plainly tell a cold reader what changed. Explain the post-data reference qualification, post-hoc rule selections, and withdrawn 3D frozen selection, with source links and their relevance here.

6. **Formatting — NEEDS-RESTATEMENT.**

   All **27 tables** have consistent columns and parse as tables with MarkdownIt. No malformed LaTeX or Mermaid syntax was found by inspection; I did not run their rendering engines.

   Two concrete defects remain:

   - [Configuration-name templates](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/report.md:681) use unquoted `H<K>s<seed>` and `L<r>`. MarkdownIt parses the angle-bracket placeholders as **HTML tags**. Put the templates in code spans.
   - [Qualification thresholds](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/report.md:179) still use raw `2.5e-4` and `2.5e-5` outside tables instead of LaTeX.

7. **Cold-reader clarity — NEEDS-RESTATEMENT.**

   The report never states that “equal cost” means the entire cost-ratio interval lies within \([1/1.1,1.1]\). That definition matters to understanding certified affordability and “no equal-cost rung.” The definitions above, figure context, and unexplained amendment identifiers remain the principal obstacles.

**NOT-READY. Ranked remaining fixes:**

1. Complete the glossary and correct “timing complete”; explain the equal-cost band and label meanings.
2. Fix configuration-template HTML parsing and remaining non-LaTeX mathematical prose.
3. Complete standalone figure context and identification, especially seed spread and the error–cost plane.
4. Replace inherited amendment identifiers alone with plain-language qualifications/retractions and source links.

No files changed, jobs submitted, cluster accessed, or scratch files created.