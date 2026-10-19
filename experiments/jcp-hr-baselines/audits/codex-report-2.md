1. **NEEDS-RESTATEMENT — numerical provenance.** Read-only, in-memory regeneration reproduced `report.md` and `EXPORT.json` exactly; verdict checksums and tuning hashes passed. Recomputed supporting statistics match. However, “every number below is read from run JSONs” is false: the generator hard-codes experimental counts in the opening, including “72 … configurations.” Derive these from the inventory.

2. **CORRECT — Markdown tables.** Configuration delimiters are escaped. All table rows have consistent column counts.

3. **CORRECT — restrictions and selection scope.** Developmental, in-sample selection is disclosed; restricted neutral conclusions remain withheld. Diagnostic batch 1 is distinguished from the qualified subset.

4. **NEEDS-RESTATEMENT — standalone summaries.** The three “By batch” paragraphs include restrictions but omit budget qualifications, contrary to A47’s claim that these lines carry them. Add “budget band unresolved” or “unequal-budget, descriptive” directly.

5. **WRONG — budget qualification in EXPORT.** Budget arithmetic and report role labels are correct. Exported nested pair/role `label` strings still lack the budget qualification; it exists only as surrounding `budget_qualification` metadata. This does not satisfy “every pair label” or A47’s stated implementation. Qualify each exported label.

6. **CORRECT — construction costs and ledger.** Lower-bound markers follow the flags. Ours’ recorded costs reproduce: 2.967958 h and 1.85 h, with missing components disclosed. Ledger recomputation gives 21.1783 h = 1.512 h smokes + 17.520560 h recorded phases + 2.145740 h remainder.

7. **CORRECT — cancellations and coverage.** Reasons, cancelled-attempt costs, 72 failed constructions, 1,056 completed 3D configurations, and deferred meshes are separately disclosed.

8. **NEEDS-RESTATEMENT — most frequent pick.** Adequacy-witness counts reproduce: 8/8, 8/8, and 7/7; unchecked rank 512 in 2D is correctly identified. The 3D ECSW pick occurs in 11 dominating role pairs. But the 1024² “Most frequent dominating pick” is tied: Q-DEIM rank 256/p512 and rank 512/p1024 each occur six times. Report both co-leaders.

9. **CORRECT — median reversals.** Reversal tables correctly distinguish worst-error improvement from median deterioration, including the substantial qualified 3D reversal.

10. **CORRECT — generator probe.** Projection values, probe errors, ratification, and the 256² 51-step versus 12-step confound are reported consistently. Unavailable 2D rank-512 probe projections are marked `n/a`.

11. **CORRECT — comparator scope.** Executed query paths, initialization/output contracts, excluded setup, admitted FRONT-2D members, and excluded categories are documented.

12. **NEEDS-RESTATEMENT — supporting evidence and missing deliverables.** Q1 counts, limiting-FOM errors, HR/none error-level discrepancies, native/common summaries, projection floors, and offline timers reproduce. Screening plots distinguish Q1 failures and batches; unavailable HR/none field distances are explicitly disclosed. However, “Break-even query counts are not computed … with equal budgets” introduces a requirement absent from DESIGN §11, which requires accuracy qualification and positive per-query saving. Explain the actual missing-cost/restriction obstacle or provide descriptive calculations. Compilation cost is also required by §11 but neither separately reported nor explicitly identified as unavailable.

13. **NEEDS-RESTATEMENT — glossary and project rules.** Domination, time schemes, FS/QN, and native/common definitions agree with the cited code. The ECSW definition “first NNLS path state with at least $4r$ nodes” omits A23’s permitted early KKT-optimum exception. Add it and distinguish target from actual support. Mathematical prose still includes non-LaTeX formulas such as “Δt = 1/60” and “γ > 1e-3.” No prohibited historical comparison or ASCII diagram was found.

Overall: NOT-READY