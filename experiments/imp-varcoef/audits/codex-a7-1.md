Read-only audit completed. No files changed or jobs submitted. The four smoke-result hashes match `smoke-accept.json`; the ledger records only the smoke job.

1. **NEEDS-RESTATEMENT — A7 is not entirely cost-only or within §11’s defer order.**  
   [DESIGN.md:477](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/DESIGN.md:477): A7.3 uses accuracy/order/Péclet evidence; A7.4 cites performance “at comparable error.” Qualification capping and reference coarsening are additional amendments, not registered deferrals.  
   **Fix:** describe these as post-smoke changes informed by development accuracy and cost, made before production. The repository evidence supports “no production run recorded,” not “only cost information examined.”

2. **CORRECT, conditional — the reference shift is numerically plausible.**  
   [DESIGN.md:487](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/DESIGN.md:487): the four smoke references give:
   
   | Quantity | Observed |
   |---|---:|
   | ε_ref | 7.117×10⁻⁶–1.433×10⁻⁵ |
   | p across output times | 1.9902–2.0003 |
   | Maximum Péclet at 1024 | 0.69325 |
   | 4096 share of level-solve time | approximately 87–88% |
   
   Assuming second-order behavior continues, coarsening predicts ε_ref ≈2.85×10⁻⁵–5.73×10⁻⁵ and maximum Péclet ≈1.3865. Those support **trying** 512/1024/2048 while retaining every acceptance predicate and fallback. They do not establish acceptance at 512 or for unseen cases.  
   **Fix:** call these extrapolations; replace “two orders” with “approximately 70–140 times below the eligibility bound,” and limit demonstrated fallback acceptance to the smoke cases.

3. **CORRECT — the three explicit deferrals follow §11; the qualification cap preserves the gate’s conservative direction.**  
   [make_configs.py:141](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/make_configs.py:141) removes CCv L512; C10 becomes dev-only and timing defaults to C100. [vladder.py:401](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/vladder.py:401) exits before validation, screening and selection for C10. The eight-attempt limit retains qualification checks; unresolved required families block a positive target verdict.  
   **Fix:** explicitly label the cap a new search truncation. “No qualified” means none found within the attempted subset, not none exists.

4. **CORRECT on code inspection — shared programs retain configuration-specific numerical inputs.**  
   [vspec.py:68](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/vspec.py:68), [vfd.py:157](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/vfd.py:157), and [vfd.py:313](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/vfd.py:313): each `Bound` owns its parameter array; query, setup and kernel pass it explicitly. Cache keys retain structural distinctions, including quadrature, output shape, preconditioner and CCv Newton limit. I found no configuration-aliasing bug.
   
   Tiny CPU comparisons against `HEAD~1` gave zero field difference for two PSI tolerances and ETD-max, including query versus setup/kernel composition. I stopped the broader check at about one minute; this is not full refactor or real-size certification.

5. **NEEDS-RESTATEMENT — persistent-cache memory is unvalidated at production breadth.**  
   [vladder.py:325](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/vladder.py:325): full-field programs now remain reachable, alongside lattice and qualification variants. The smoke exercised the previous implementation and a small configuration subset.  
   **Fix:** quantify resident executable/constants memory across the production structural set, or use bounded cache lifetimes. No actual OOM is established.

6. **WRONG — the fallback generator does not implement the promised test-reference fallback.**  
   [make_configs.py:121](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/make_configs.py:121): `fallback(job_results)` ignores its argument and always reads the development/validation eligibility file. It therefore cannot select failed test references from the supplied results. Moreover, emitted configs omit the frozen-selection/opening-record bindings required by `vref.py` for any test case; a test fallback generated this way would fail verification.  
   **Fix:** consume the specified reference results, distinguish pooled and test fallback selection, and preserve test-opening bindings. The ordinary pooled fallback’s case selection is otherwise consistent with the eligibility schema.

7. **WRONG — reference certificates do not implement A7.5’s stated unobserved-level bound.**  
   [resources.py:109](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/resources.py:109): `top = rates[max(rates)]` selects the largest **resolution’s** rate, not the largest observed rate. For unobserved L512, the required maximum is **1.528×** the selected rate at C100 and **1.071×** at C10.  
   **Fix:** use `max(rates.values())` for unobserved levels. Also reconcile the docstring’s minimum retry fraction **0.5** with the implementation’s **0.1**. Regenerate certificates after correction.

8. **WRONG — the ladder model is a projection, not a conservative upper bound.**  
   [resources.py:133](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/resources.py:133):
   
   - A 50% extension allowance does not bound the registered extensions. A four-step row with work proportional to 10+20+40+80 can add 160+320: **320% additional work**, before cheap-edge extensions.
   - Half the aggregate configuration cost does not bound the cost of up to eight attempts per family/target. Attempts can concentrate on expensive configurations.
   - Screening burns are omitted: production requests six burns of 0.1 seconds per screened configuration.
   
   **Fix:** enumerate possible extension work, bound qualification using the most expensive allowed attempt sets, and add burns explicitly. Publish the complete spent-plus-remaining plan, including fallback and test reserves. The smoke replay projection exceeds observed elapsed time, but that does not establish a production upper bound.

**Overall verdict: NOT CLEAN — WRONG findings in test fallback generation and resource certification.**