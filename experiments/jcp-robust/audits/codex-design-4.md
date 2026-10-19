Read-only audit completed. No files changed, scratch created, jobs submitted, or cluster commands run.

References: **D** = [experiments/jcp-robust/DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/DESIGN.md). **Q, M, P, F, S, W** denote the qualify, dims, p1-scaling, fom2d, smooth-bank and wide-bank worktrees specified in the request; paths are relative to their experiment directories. **COMMON** denotes the binding common brief.

Items 1–15 address the requested prior-audit items, including item 18, which audit 3 already marked resolved.

1. **Prior 18 — CORRECT; RESOLVED.**  
   `D:685–687` explicitly uses `procedure_outcome` for H-cost-N(2). This matches `P:p1stats.py:181–188,205–207`: median over valid case/repetition entries, with a separate two-mesh outcome.  
   **Fix:** None.

2. **Prior 19 — NEEDS-RESTATEMENT; NOT RESOLVED — BLOCKING.**  
   The ordinary complete-data pooling is now defined, but the new missing-job fallback contradicts the intersection requirement. `D:680–683` requires valid repetitions in **every job of both types**; `D:690–691` permits a type verdict from the remaining job when another is missing or blocked. With an entirely missing job, the former rule leaves no common cases, while the latter promises a verdict.  
   **Fix:** Define retained jobs before constructing case intersections, separately for cross-type quantities and per-type q4. Alternatively, remove the single-job fallback and return unavailable when either required job is missing.

3. **Prior 20 — CORRECT; RESOLVED.**  
   `D:668–678` repairs the statistical controls; `D:481–483` separately tests physical doubled work within each GPU. The q2 injection cannot be cancelled by the original mesh ordering. Fixture failures also block dependent q4 conclusions.  
   **Fix:** None to the design; exercise these paths during code audit.

4. **Prior 21 — NEEDS-RESTATEMENT; NOT FULLY RESOLVED — IMPLEMENTATION-NOTE.**  
   `D:693–708` supplies a substantially improved workload inventory and an explicitly provisional budget. It still does not fully enumerate the qualification executions discussed in item 16 below. The hard cap, reservations and smoke certificates prevent this from being a design blocker by itself.  
   **Fix:** Generate the execution inventory and reservation certificate from the actual job configurations before submission.

5. **Prior 29 — CORRECT; RESOLVED.**  
   `D:668–678` requires nonempty, estimator-level fixtures. `D:600–601` supplies five cases and two valid repetitions for the A100 smoke, meeting the analysis minimum.  
   **Fix:** None; ensure “job 1” can refer to the smoke’s fixture input rather than requiring production first.

6. **Prior 32 — CORRECT; RESOLVED.**  
   Copying one positive array into both baseline mesh columns makes baseline \(q_2=1\). Scaling only the injected 1024² column makes injected \(q_2=1.5\) under every joint case resample (`D:673–674`). The earlier bootstrap counterexample no longer applies.  
   **Fix:** Assert equality within floating-point tolerance, rather than literal exact equality.

7. **Prior 33 — CORRECT; RESOLVED for complete jobs.**  
   `D:680–683` separates dev6 P-cost cases from the sixteen P-cross cases and excludes H200-only 4096² cells. The accidental global intersection is removed.  
   **Fix:** Retain this separation when repairing the missing-job rule in item 2.

8. **Prior 34 — NEEDS-RESTATEMENT; NOT FULLY RESOLVED — BLOCKING.**  
   `D:685–690` now defines pooled per-type interval decisions and gives conflicting definitive job labels precedence. Those constructors are sound. However, the single-job branch remains incompatible with `D:680–683`, as described in item 2. This is the same blocker, not an additional independent defect.  
   **Fix:** Specify eligible-job sets and case-set construction for every fallback branch.

9. **Prior 35 — CORRECT; RESOLVED — IMPLEMENTATION-NOTE.**  
   `D:657–666` enumerates incumbent replay configurations, populations, blocking scope and smoke coverage. A representative replay gate need not cover every downstream configuration.  
   **Fix:** Resolve 3D archive keys through the saved `wkeys` and case mapping, as constructed by `Q:q3run.py:44–67`; do not assume the 2D universal `W` key. Freeze concrete filenames and hashes at staging.

10. **Prior 36 — CORRECT; RESOLVED in its principal requirement — IMPLEMENTATION-NOTE.**  
    `D:628–636` makes production failures count against H-E1c, ranks failed aggregate errors worst, suppresses arithmetic on failed values, and distinguishes missing evidence.  
    **Fix:** Store validity separately for production, tight, tighter, successor and target runs. A single mutually exclusive enum cannot retain simultaneous failures. Derive the displayed status from those flags, with explicit precedence.

11. **Prior 37 — CORRECT; RESOLVED.**  
    `D:638–644` keys target certification by bank, rank/width, form, scheme, step and mesh; specifies the certification subset; and withdraws target-dependent quantities when certification fails. Own-family qualification remains separately available.  
    **Fix:** Label convergence as checked on the registered subset, rather than implying an every-case convergence proof. Handle individual target-run failures as noted in item 17.

12. **Prior 38 — CORRECT; RESOLVED.**  
    `D:651–655` adds unqualified, missing and blocked states and restricts dominance to qualified candidates. Three absent slots can establish agreement about **absence**, not successful frontier usefulness; the separate qualification comparison preserves that distinction.  
    **Fix:** Report the agreed categorical state beside every “seed-robust” count.

13. **Prior 39 — NEEDS-RESTATEMENT; NOT FULLY RESOLVED — IMPLEMENTATION-NOTE.**  
    The revised estimate is reviewable as a projection, but not yet a certified complete workload. Dims’s complementary ladder jobs consumed **1.5264 + 2.8592 GPU-h** (`M:runs/GPU-LEDGER.json:77–106`). A3 now allows roughly that scale per bank, instead of treating either job as the complete workload. The component sum of **65.1 GPU-h** is correct (`D:706–708`).  
    **Fix:** Complete the inventory corrections in item 16 and derive costs from representative smoke measurements. Do not present the remaining 9.9 hours as certified contingency.

14. **Prior 40 — CORRECT; RESOLVED.**  
    `D:646–649` replaces median reference uncertainty with a uniform maximum bound. Both the median and maximum are Lipschitz in the maximum norm, so the stated propagation argument works, conditional on the reference-error estimates.  
    **Fix:** Preserve the conditional-Richardson qualification; this is not a rigorous bound on unknown reference error.

15. **Prior 41 — CORRECT; RESOLVED.**  
    `D:623–626` correctly distinguishes 38 scored cases from qualification using `sub16`. It also correctly treats new successor failures outside that subset as findings rather than reproduction failures.  
    **Fix:** None.

The following are additional checks and consequences of A3.

16. **NEEDS-RESTATEMENT — The execution table is not yet a complete cost certificate — IMPLEMENTATION-NOTE.**  
    `D:700–703` leaves several accounting details unresolved:

    - E2 verdict rows count production/tight/successor executions but do not explicitly count their tighter anchors.
    - Reusing gl64 **production** targets does not supply gl64 tight/tighter solves for a qualified H-acc-r panel.
    - The 2D selection row counts three executions per candidate, then adds terminal Gauss successor executions separately; successor reuse and potential double-counting are unclear.
    - `D:701` budgets 38 cases at 4096², while the registered E2 panel at `D:189` uses sixteen.
    - The compile-signature column does not distinguish solver levels, although adaptive tight solves can have different compilation costs.

    **Fix:** Enumerate unique execution keys—including bank, mesh, rule, specification, solver level and case set—and explicitly identify reuse. Include replay gates and tighter anchors. These are code/job-audit requirements; provisional arithmetic alone does not block the scientific design.

17. **NEEDS-RESTATEMENT — Individual target failures need an explicit propagation rule — IMPLEMENTATION-NOTE.**  
    A target can pass the dev6/cases-0–7 convergence check yet fail on another evaluation case. `D:631` calls this “target failed,” whereas `D:641–644` defines fallback behavior for an uncertified **configuration**. The implementation must not count a failed target as a failed randomized draw or use its error in H-E1c’s margin.  
    **Fix:** Make target-dependent quantities unavailable on such cases, state how this propagates to configuration-level summaries, and apply the declared target-free fallback consistently. Preserve independently valid production-error evidence.

18. **NEEDS-RESTATEMENT — The emergency incumbent reduction changes the comparison universe — IMPLEMENTATION-NOTE.**  
    `D:711–713` allows the 3D incumbent evaluation to shrink to four fixed configurations. That no longer supplies the identical restricted front required by `D:439–447` for comparison with fully evaluated new seeds. Merely labelling the reduced run does not make membership comparable.  
    **Fix:** If this deferral fires, report historical agreement only for the common fixed-configuration quantities; mark incumbent agreement for selected-rule fronts unavailable. New-seed agreement can still be reported if all required new-seed evaluations exist. The brief’s protected items and earlier defer order remain intact (`D:327–331`).

19. **CORRECT — E1 population, Sobol terminology and shift construction remain sound.**  
    Reading `Q:runs/FRONT-2D.json` confirms seven randomized configurations at each mesh: R128 BE factors 1/2/2.5, CN-R factor 1, and GAL-CN factors 1/2/5. `Q:report.md:456` confirms the two randomized members of the pinned 3D subset.  
    `D:359–362` delegates to the actual generators, matching `vendor/quad2d/vendor/hari_quadrature.py:107–119`, `vendor/quad3d/rules.py:99–104`, and `rules3d_names.py:47–63`. The Sobol description agrees with [SciPy’s documented LMS plus digital shift](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html).  
    **Fix:** None; retain diagnostic-only Sobol and GAL labels.

20. **CORRECT — The E2 recipes and like-for-like selection fallback remain defensible.**  
    `S:smoothtrain.py:40–42,170–177,403–432` preserves the Stage-A recipe and emits a compatible checkpoint. `separable-decoder/sep_coeff_extract.py:101–134` supports its use with the intentional dense-extraction mismatch. `run_dn256b.sbatch:44–62` matches `D:411–420`; the historical `make_rotation.py:57–75` uses emitted head coefficients.  
    `W:train3d/train2w.py:193–224,379–444` supports validation-selected W1024 training and LS-based ordering. Fixed physical data are justified for the explicitly conditional estimand. Dims’s `report.md:47,55,71` makes deployed-rule recovery plausible, but does not guarantee the new joint selection outcome.  
    **Fix:** None to the recipe design. Preserve non-blocking recovery anchors, identical procedures across banks, and historical-software caveats.

21. **CORRECT — E5 memory, frozen comparators and controls are appropriately scoped.**  
    `P:report.md:204–207` supports the existing implementation’s A100-80GB restriction. `F:FROZEN-SELECTION-1024.json:108–119,14784–14795` confirms the same-mesh CC comparator keys used at `D:579–581`.  
    Physical doubled work now tests within-device timing (`D:481–483`); estimator fixtures test cross-hardware statistics (`D:668–678`). Neither is incorrectly claimed to substitute for the other. Case bootstrap conclusions remain conditional on observed devices (`D:478`).  
    **Fix:** Repair only the missing-job aggregation contradiction identified above.

22. **CORRECT — No additional non-degenerate-outcome or common-contract blocker found.**  
    H-E1a’s zero/one/two-or-more failures produce all three outcomes at sixteen draws. H-E1c’s percentage bands and A2.7’s three-seed agreement rule also admit all registered outcomes. All-tied typicality is explicitly handled (`D:531–540`), and reproduction gates reject empty/missing evidence (`D:384–392`).  
    E0 ST scoring, case-kind eligibility, frozen modes, full Q1, development cohorts and compute restrictions remain registered at `D:12–16,83–84,285–291,364–372`.  
    **Fix:** During code audit, explicitly enforce finite, nonempty bad-vector admission with positive target norms, following `M:dstats.py:173–176`; never rely on an empty `all(...)` reduction.

**Overall verdict: NEEDS-RESTATEMENT.**