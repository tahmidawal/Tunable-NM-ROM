**Overall: NEEDS-RESTATEMENT; not CLEAN.** The primary numerical conclusion survives: the stop rule **FIRES**. The report has provenance, labelling and rendering defects.

I regenerated the report, export and analysis **in memory**, suppressing writes: all matched the existing files. No files changed, jobs ran, or large arrays were loaded.

1. **CORRECT — numerical verdicts and A4.6 interpretation.**  
   Recomputed all 15 verdict/accounting combinations. Primary: **126 MATCH, zero NO-MATCH or UNRESOLVED**. Casewise: **84 MATCH, 42 NO-MATCH**; rollout: **126 NO-MATCH**. Every NO-MATCH has no admissible competitor. The report correctly says cost advantage is undetermined; it does not turn those rows into an advantage.

2. **WRONG — Markdown tables corrupt column alignment.**  
   Unescaped `|` inside configurations and workload names splits cells throughout the accuracy, workload, break-even and compilation tables. For example, `R128|BE` shifts status and numerical columns in [report.md:56](experiments/imp-multimesh/report.md#L56). Escape these separators in every generated table cell, including headers.

3. **WRONG — “every number … from the run JSONs” is false.**  
   [analysis.py:33](experiments/imp-multimesh/analysis.py#L33) parses H200 logs, while historical A100 values are hard-coded at line 51. [make_report.py:263](experiments/imp-multimesh/make_report.py#L263) hard-codes four selection timings; its A100 validation also reads an external log and hard-codes the recorded total. This does **not establish that those values are numerically wrong**, but they lack the promised JSON provenance. Store source-linked numerical records and generate from them.

4. **NEEDS-RESTATEMENT — A7.1 compilation conditionality.**  
   All recomputed cells are robust under the two adverse assignments with **$B=40.4758$ s**. However, the headline and cost tables do not explicitly state “conditional on compilation projections”; “conditional on $B$” does not explain that robustness assumes $B$ bounds true compile costs. Add that condition beside the outcome, workload costs, break-even results and plot.

5. **NEEDS-RESTATEMENT — cost scope and projected training cost.**  
   The title says simply “full cost,” contrary to A2.2. Uncosted rotation and rule selection appear later, rather than beside the verdict as A3.7 requires. The offline heading says “measured” although the bank-fit total is projected. Use **“full cost of the measured items”**, list exclusions beside the conclusion, and identify the offline total as containing a projected bank fit.

6. **NEEDS-RESTATEMENT — A2.1 explanation is incomplete.**  
   Explicitly define the primary criterion as **aggregate worst error**, distinct from casewise ranking. Define `worst_eps`, `casewise`, `worst_median` and `rollout`, including their allowance formulas. The two summary bullets do not provide the required per-NO-MATCH trade-offs: cheapest competitor and failure reason, plus most accurate competitor and cost. A generated companion table can carry these.

7. **WRONG — A_sel construction description.**  
   [make_report.py:262](experiments/imp-multimesh/make_report.py#L262) sums only `ecsw`, `states`, `resample` and `map`, yet the report calls this **“all competitor construction artefacts”**. It omits `codes`, `trust`, `snap` and `pod`; it therefore has not established the stated upper bound on non-deployed construction. Correct the accounting or describe the selected-subset scenario precisely. The disclosed failure to subtract deployed items does not explain these omissions.

8. **NEEDS-RESTATEMENT — glossary and mathematical notation.**  
   The closing glossary exists but leaves major terms/columns unexplained: POD, FOM, LSPG, BE/CNR, dev6/val32, developmental/in-sample, workload, $O$, $P$, sentinel ratios, A2/A1, identity, AOT, A_sel, construction-key meanings and qualification statuses. Several mathematical expressions remain Unicode prose rather than LaTeX. Expand the glossary in plain language and format prose mathematics consistently.

9. **CORRECT numerically; NEEDS-RESTATEMENT as a standalone plot.**  
   The plot uses **R384/CNR**, log axes, queries **per mesh**, and the cheapest Q1-qualified competitor admissible under **aggregate worst error / A_dep**. The configuration is visible. Missing: the criterion/accounting labels, developmental/in-sample status, compilation condition and measured-item exclusions. Add a caption or subtitle; the report currently does not embed or explain the PNG.

10. **NEEDS-RESTATEMENT — EXPORT.json is incomplete for compilation.**  
    Outcomes and offline totals match regeneration, and an analysis source path exists. Missing key results include ratio range/median, verdict counts, no-admissible counts, compile robustness/$B$, break-even results and evidence-status qualifications. The single analysis path does not source every exported value, notably A100 validation and the PM estimate. Add per-result source paths and conditionality metadata.

11. **CORRECT — PM wording and basic status.**  
    PM-4096 is explicitly an **estimated snapshot-only cost**, with sample range and remaining setup/accuracy unmeasured; no “loses regardless” inference appears. The level-1 title, developmental/in-sample status and conditional Richardson-reference label are present.

12. **NEEDS-RESTATEMENT — explicit withdrawal.**  
    Corrections are documented, but the report only quotes the instruction to drop the comparative claim. State the consequence directly: **“We withdraw the comparative multi-mesh cost claim under the registered primary criterion.”**

**Required fixes:** repair table escaping; complete numerical provenance and EXPORT; correct A_sel wording/accounting; add the required cost/compile qualifications, criterion definitions, NO-MATCH detail, glossary, plot caption and explicit withdrawal.