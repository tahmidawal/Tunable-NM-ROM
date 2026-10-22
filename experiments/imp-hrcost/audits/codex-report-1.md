**Overall: NOT CLEAN.** Both prior restatements are fixed; remaining issues concern numerical wording and presentation.

- **CORRECT — Title/status:** level-1 title and explicit status distinguish final `prd1` outcomes from conditional E0 accuracy estimates.
- **CORRECT — Generated results/export:** in-memory generation reproduces `report.md` and `EXPORT.json` byte-for-byte. Arm tables, fitted-rule results, timing ratios and stop outcomes match their source JSON fields.
- **WRONG — Stiffness claim (`report.md:71`):** “Gauss 384² is off by about 1%” misinterprets `dK_prev=0.010482`: this measures **384² versus 256²**, not 384² error against converged stiffness. Say “384² and 256² differ by about 1.05%.”
- **NEEDS-RESTATEMENT — Numerical provenance:** “every number … from … analysis, result and ledger” is too broad. Several constants are hard-coded; smoke controls come from another JSON, and the stiffness discussion uses the construction diagnostic.
- **CORRECT — Stop outcomes, verbatim:** F2: `fired (tested failure): m* = gauss96, no_worse=False, faster=False`; F3: `fired (tested failure)`. Binding and non-binding readings agree. These are supported by the registered predicates.
- **CORRECT — Previous fixes:** latency is now explicitly an unestablished hypothesis; Q1 correctly permits at most 1% pooled non-converged steps.
- **CORRECT — Scope:** gains are labelled internal comparisons against R, with exploratory, in-sample, selection-unadjusted intervals. No competitor advantage is claimed.
- **NEEDS-RESTATEMENT — Plot presentation:** plotted values agree with sources, but `error-vs-speedup.png` omits conditional/exploratory qualifications and has overlapping arm labels. Add qualifications beside the figure and separate labels.
- **NEEDS-RESTATEMENT — Glossary:** missing plain-language definitions include ECSW, NNLS, CI/bootstrap, dev6/val32, training-fit residual and condition number. Explain that repeated “NNLS seconds” is total family-fit time, not separate time per κ. Q2’s definition also omits check-rule convergence.
- **NEEDS-RESTATEMENT — LaTeX:** the governing equation and Mermaid diagram comply, but prose mathematical expressions such as `κ ≤ 16`, `m*`, and `2 ε_ref` remain outside LaTeX.
- **NEEDS-RESTATEMENT — Retractions:** A5/A6/A9 are stated correctly, but DESIGN A2.1’s withdrawal of the projected-ρ justification is not explicitly recorded among the report’s retractions.

Read-only throughout; no files, lab log, main, or jobs changed.