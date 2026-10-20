| Round-3 fix | Verdict | Round-4 finding |
|---|---|---|
| 1. Complete remaining glossary definitions | **NEEDS-RESTATEMENT** | All specifically identified omissions were added correctly. The exhaustive glossary requirement still has gaps listed below. |
| 2. Remove/source the historical count; link historical records | **NEEDS-RESTATEMENT** | The hard-coded historical count was removed. Actual source links remain absent; abbreviated paths containing `...` are not usable references. |
| 3. Format Section 5’s contract as LaTeX | **CORRECT** | The replacement uses LaTeX and agrees with EXPORT’s contract. |

1. **Numerical provenance — CORRECT.** With filesystem writes blocked and outputs captured in memory, regeneration reproduced **`report.md` and all five PNGs byte-for-byte**.

   Independently checked **32 numerical cells** against EXPORT:

   - Six training values each for `mid_k4_s0`, `mid_k8_s1`, `mid_k16_s2`, and `mid_k32_s1`.
   - Median and worst rollout errors for `H16s0|BE`, `H32s1|BE`, `L16|BE`, and `L128|BE`.

   All matched at the displayed precision. The headline’s pooled error and cost ratios also agree with EXPORT. Registered DESIGN constants and the disclosed ledger are accepted under your exception.

2. **Headline claims — CORRECT.** F5 is **inconclusive**, explicitly distinguished from negative evidence. The matched-dimension comparison remains a trade-off. The at-most-cost statement identifies the qualified linear rung without claiming universal linear dominance. The pooled-head summary identifies its qualified seeds and does not describe itself as one deployable model. Exploratory and E0-conditional qualifications remain explicit.

3. **Plots — CORRECT.** Inspected all five PNGs and their generation logic. Qualification is distinguished from timing usability where applicable; the error–cost plot correctly distinguishes numerically unqualified but timing-usable H8s1 BE. Training-oracle results are labelled truth-assisted diagnostics. The pooled error uses the registered aggregation, and cost-versus-K is explicitly descriptive.

4. **Glossary — NEEDS-RESTATEMENT.** The additions correctly explain training arms, timing scope, fused-time aggregation, Richardson estimation, MLP/SiLU, whitening, POD, and initial-fit/LM terminology.

   The closing glossary still does not define:
   - **Relative RMS**, used to reject or describe the cost fits, including its normalisation.
   - **Rung**, repeatedly used in comparison tables: a tested linear-span dimension or quadrature level.
   - The jobs table’s **attempt, job, state, GPU model, GPU-hours** columns.

   These are small documentation gaps, but the requested standard explicitly covers every column and term.

5. **Relation to earlier results — CORRECT substantively; NEEDEDS-RESTATEMENT for source navigation.** The canonical log supports the historical matched-dimension claim, the distinction from a numerical replication, and the quadrature-study head rule’s post-hoc/not-confirmed status. The E0 rule change and combined-selection retraction are stated plainly.

   However, [the opening source references]( /home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/report.md:3) and [Section 6d](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-heads/experiments/jcp-heads/report.md:640) still lack clickable, complete source locations. Naming entries improves discoverability but does not complete the round-3 fix.

6. **Formatting — CORRECT by structural inspection.** All **27 tables** have consistent column counts. The level-1 title, status line, Mermaid diagram, inline qualifications, and closing glossary are present. No broken placeholders or LaTeX/Mermaid syntax defects were found. Dedicated LaTeX and Mermaid renderers were not run.

7. **Cold-reader clarity — NEEDS-RESTATEMENT.** The scientific conclusion is appropriately restrained. Remaining obstacles are the glossary gaps and inaccessible historical references. No additional unsupported scientific conclusion or prohibited historical comparison was found.

**NOT-READY. Ranked fixes:**

1. Replace abbreviated historical paths and plain-text references with complete links to the relevant log entries or reports, including the head-ablation and quadrature-study records.
2. Complete the glossary with relative RMS and its normalisation, rung, and the jobs-table column meanings.

No files changed, jobs submitted, GPU used, or cluster accessed. No scratch files were created.