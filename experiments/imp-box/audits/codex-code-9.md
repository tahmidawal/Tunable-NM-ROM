**Overall: NEEDS-FIX.** Audited `d149f94e0..bd87f90c3`. The remaining blocker is reporting/export semantics; the A9 execution changes pass this review.

1. **A8.1 — CORRECT, with qualification.**  
   `M_dev / sqrt(R)` implements the relative Frobenius mass check; mass and stiffness checks use `1e-8` over all retained modes. This is compatible with A2.3, but successive-rule convergence does **not imply** exact-operator accuracy. This remains a disclosed post-smoke tolerance relaxation. Available artifacts contain no final selection results, consistent with the stated chronology. “Benign round-off” is an interpretation, not established by the norm alone.

2. **Unbatched cohort and screen timing — CORRECT.**  
   The loop matches the official unbatched evaluator and preserves ordering, scoring and pooled validity predicates. Screening reuses the query after a completed warm call. Small CPU AD checks found identical statistics and batched/unbatched output differences of at most `1.32e-16`; screening added no compiled specialization. Equivalent semantics, not guaranteed bitwise equality across batching strategies.

3. **Persistent cache — CORRECT for the staged execution path.**  
   No stale-executable collision mechanism found. ROM operators, tensors, eigenvectors, spectral tables and shifted-decoder coefficients can become embedded literals. Installed JAX hashes the serialized computation, including those literals, plus compiler/backend configuration. Constants supplied as executable arguments instead receive current values. Neither mechanism substitutes another configuration’s data.

   Cache loading occurs before screen and paired timing regions. The path check now covers smoke runs; normal staged execution uses `code/.jaxcache`, which collection and both archive paths exclude. `IMPBOX_LOCAL_DEBUG` explicitly bypasses containment and must remain a local-debug exception. Actual cache-hit performance on cluster JAX remains unverified.

4. **Hoisted AD tolerance — CORRECT.**  
   Small CPU comparisons against the audited implementation at `1e-3` and `1e-5` produced exactly equal outputs and statistics. Cross-tolerance executable sharing is correctly retracted.

5. **Relative stop — CORRECT resolution: withdrawn.**  
   Neither relative nor target-aware stopping remains in the executable ladder. The stopping logic matches the audited baseline. The original “only dominated rows” claim was wrong: later rungs can cross a target or qualify where an earlier rung fails, and iterative solvers need not have monotone runtime.

6. **Report generator — NEEDS-FIX.**  
   Markdown now labels failed controls, partial FOM searches and timing exclusions; conditional-reference/provisional labels and a glossary exist. Winner identity, unavailable device memory and cancelled-smoke accounting are repaired.

   **The export still loses essential qualifications.** At [make_report.py:192](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/make_report.py:192), entries export the ROM status and controls acceptance but omit FOM search statuses and conditional-reference/provisional context.

   An in-memory fixture with a `finalist-partial` FOM and a `missing-construction` alternative printed those qualifications beside `S = 2` in Markdown, yet exported `status: finalist`, `controls_accepted: true`, and `S: 2` without either qualification.

   **Required:** preserve both FOM statuses and explicit reference/evidence qualifications in `EXPORT.json`. Also correct the opening “every number … read from JSON” claim: registered widths, targets and thresholds remain literals. The glossary now distinguishes these from measurements, but the opening claim still contradicts that distinction.

7. **Other integration — CORRECT within reviewed scope.**  
   `resources.py` uses `smk2-*` and no longer accepts withdrawn termination names. The ledger’s preserved-metadata merge is now inside the lock. Empty checksum sets are handled correctly. Changed Python files parse successfully.

Validation was read-only source inspection and small CPU checks, including intercepted report writes. **No files changed, jobs submitted, or lab-log writes performed.**