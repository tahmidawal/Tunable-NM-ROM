**Overall: NEEDS-FIX.** The primary numerical verdicts survive. Three round-1 blockers are resolved; timing-scope disclosure is only partially resolved. The report audit also found a misleading memory plot, incomplete export, hard-coded numerical claims, and glossary deficiencies.

Audited HEAD `b4bdccb2f`. No files changed, no jobs submitted, no scratch files or archive copies created. CPU-only checks reproduced all 44 configuration error summaries from saved fields against checksum-verified E0 references, all H-acc-r transition statistics, and all eight H-cost-N bootstrap results. In-memory regeneration reproduced report.md byte-for-byte and all eight plots pixel-for-pixel.

**Round-1 disposition**

| Blocker | Disposition | Evidence |
|---|---|---|
| Mixed-population accuracy plots | **RESOLVED** | Both accuracy plots now use the same 16 cases at every rank/mesh. Recomputed 4096² R384 medians: BE **1.210462%**, CNR **0.247929%**. |
| S summaries included ineligible cases | **RESOLVED** | All BE summaries filter S eligibility and match recomputation. At 1024² R384 BE, eligible-only median is **0.051261%**. Exclusions are disclosed. |
| P1e rule names and “failing” label | **RESOLVED** | gauss128/160/192 replace `None`. The report explicitly states that registered-failing gauss64 **passed** both bars: distance **4.863348e-5**, ρ **0.0553124** against gauss96. |
| Timing scope and identity-audit limitation | **PARTIALLY** | report.md now correctly describes evolution timing with internal-state recording and the unavailable per-execution hashes. EXPORT.json still omits both qualifications. |

**Report audit**

1. **Memory plot — WRONG.**  
   [make_report.py:139](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/make_report.py:139) sums **every** table role, while its title says “rule + successor + check.” R256 includes failing-rule controls; R384 includes additional historical/verification rules, especially gauss640.

   | Rank | Plotted GB | Main + successor + check GB |
   |---:|---:|---:|
   | 128 | 0.183561 | 0.183561 |
   | 256 | 1.346423 | 1.168146 |
   | 384 | 9.466149 | 1.487975 |
   | 512 | 5.234552 | 5.234552 |

   The apparent memory decrease from R384 to R512 therefore compares different inventories. Filter to `main`, `qp`, `qpp`, or label the graph explicitly as unequal whole-experiment inventories. Its title is also clipped.

2. **“Every number … from analysis.json” — WRONG.**  
   Reproducible generation does not establish data provenance. [make_report.py:293](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/make_report.py:293) hard-codes **1.85 h / 1.14 h**. Those agree with the wide-bank report, but are not loaded from analysis.json or raw accounting. The first-smoke “five” failures and recording ratio “≈1” are also literal prose; historical published values come from hard-coded `analyze.HIST_CLAIMS`.

   Move empirical values and their source provenance into the analysis pipeline. Registered constants and configuration identifiers can remain constants, but the blanket provenance statement must accurately describe them.

3. **EXPORT.json — WRONG as a complete hand-off.**  
   The exported primary verdicts agree with analysis.json, and all eight plots are listed. However, [EXPORT.json](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/runs/EXPORT.json) lacks:
   
   - Recording-executable scope and the per-execution identity-audit limitation.
   - H-cost-r fitted exponents, intervals and secondary failure classifications.
   - H-acc-N equivalence/trend statistics and H-acc-r transition counts.
   - Corrected common-population accuracy summaries and their population labels.
   - P1e judgments, including the unexpectedly passing 2D control, and P1g key numbers.
   
   Its numerical groups contain only configuration median/worst errors and H-cost-N statistics. Thus it preserves primary outcome labels, but does not satisfy the requested complete verdict/key-number export.

4. **Title/status and conditional accuracy labels — NEEDS-RESTATEMENT.**  
   The level-1 title and opening scope/status description are present. The opening correctly discloses conditional Richardson uncertainty and the post-data A10 amendment. However, the accuracy tables and figures do not carry that qualification inline. Label these as **conditional E0 accuracy estimates**, distinguishing completed computations from unconditional accuracy certification. “Final for the tranche” should not obscure this distinction.

5. **Glossary — WRONG/incomplete.**  
   The glossary exists but does not define every column or technical term. Missing explanations include BE/CNR, LSPG/GAL/FS, historical arm names, Q3, `cond`, `rank`, the matrices/operators A/J/N, `chol`, S eligibility, resolved/tied cases, range monotonicity, and statistical p-values versus fitted exponent p.

   Two existing definitions also need correction:
   
   - “Q2 … 16 validation cases” actually uses **six development plus ten validation cases**.
   - Unqualified **$M=4r$** does not apply to the fixed-$(M,m)$ kernel panels.

   Define error normalization explicitly: the denominator is the initial reference norm.

6. **LaTeX maths — NEEDS-RESTATEMENT.**  
   Some equations use LaTeX correctly, but the glossary’s margin formula remains plain Unicode/ASCII: `δ = max(2ε_ref, 0.1 e(1024²))`, as does `|Δe| ≤ δ`. Convert these to inline LaTeX. Unicode matrix notation inside table headers is permitted by the repository rules.

7. **Scientific claims — mixed disposition.**

   | Claim | Assessment |
   |---|---|
   | H-cost-r concerns component kernels at fixed $(M,m)$ | **CORRECT** in plot titles, glossary and export contract. No total-solver scaling conclusion is established. |
   | “Operation-model” explains the physical cause of measured scaling | **NEEDS-RESTATEMENT** if inferred: it is the registered classification rule, not proof that byte traffic explains the measurements. |
   | H-acc-N pass means practical equivalence | **CORRECT**; the report explicitly retains R128-CNR’s detectable trend, **p = 0.00628662**. |
   | “Monotone” means majority improvement among resolved cases | **NEEDS-RESTATEMENT**; the transition glossary helps, but the range label itself is undefined. State that every transition passes that majority rule, not that every case improves. |
   | P1g measures total effects under each rule | **CORRECT**; historical reproduction is distinguished from a fixed-test-space interpretation. |
   | The registered-failing 2D rule actually passed | **CORRECT**, explicitly acknowledged. |
   | R384→512 does not improve | **CORRECT**; BE counts **4/16/18/0**, CNR **6/22/10/0** improve/worsen/tie/unqualified. |

8. **Remaining plots — CORRECT numerically.**  
   The seven non-memory plots reproduce their analysis inputs and have consistent population/scope labels. The CNR median curve’s slight decrease at R512 does not contradict the paired resolved-case verdict; a caption explaining that distinction would prevent misreading.

9. **Forbidden historical comparisons — CORRECT.**  
   The report contains neither the heat POD-128 comparison nor a pre-September Burgers speed-up.

10. **Missing Mermaid diagram — CORRECT to omit.**  
    This is a results audit/report, adequately served by tables and plots. The repository requires Mermaid **when presenting a diagram**; it does not require adding an unnecessary architecture diagram.

Acceptance requires correcting the memory plot, completing the export, removing unsupported numerical-provenance claims, and repairing the conditional labels, glossary and mathematical notation. The primary registered results need no numerical retraction from this audit.