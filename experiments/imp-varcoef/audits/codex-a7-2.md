Read-only audit completed; no files changed or jobs submitted. Checks used code inspection and CPU-only bookkeeping probes, not GPU solves.

1. **CORRECT — A7’s status is now stated accurately.**  
   [DESIGN.md:507](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/DESIGN.md:507) explicitly identifies post-smoke accuracy-and-cost information and distinguishes new amendments from registered deferrals. **Fix:** none.

2. **CORRECT, conditional — reference coarsening is appropriately qualified.**  
   [DESIGN.md:509](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/DESIGN.md:509) labels the predicted errors and Péclet numbers as extrapolations and limits demonstrated fallback acceptance to smoke cases. Production acceptance remains case-specific. **Fix:** none.

3. **CORRECT — deferrals and qualification truncation preserve the conservative gate.**  
   [DESIGN.md:512](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/DESIGN.md:512) now explains “no qualified” correctly. [vladder.py:416](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/vladder.py:416) retains the C10 dev-only exit; qualification still stops after eight attempts per group. **Fix:** none.

4. **CORRECT on inspection — shared programs retain configuration-specific inputs.**  
   The relevant cache/binding code remains unchanged: [vspec.py:156](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/vspec.py:156), [vfd.py:159](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/vfd.py:159), and [vfd.py:319](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/vfd.py:319). No new aliasing defect found. **Fix:** none; this does not constitute production numerical certification.

5. **NEEDS-RESTATEMENT — memory instrumentation is not memory validation.**  
   [vladder.py:195](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/vladder.py:195) records allocator statistics and cache size, but supplies neither a production-breadth measurement nor bounded cache lifetimes. Each save overwrites the memory snapshot; the cache remains persistent. **Fix:** retain the “production memory unvalidated” qualification until measured, or bound cache lifetimes. No OOM is established.

6. **CORRECT on inspection — test-reference fallback is implemented.**  
   [make_configs.py:121](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/make_configs.py:121) consumes the supplied test result, selects failed/ineligible cases, and copies the original test configuration’s opening bindings while replacing cases and levels. **Fix:** none for this defect; test execution must include the fallback reference directory last, as supported by [vtest.py:37](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/vtest.py:37).

7. **WRONG — rate calculation fixed, published certificates still stale.**  
   [resources.py:110](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/resources.py:110) correctly uses the maximum observed rate, and the retry-floor documentation now matches the implementation. However, recomputation gives:

   | Certificate | Published GPU-h | Current calculation |
   |---|---:|---:|
   | [ref100](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/checks/resources-ref100.json:13) | 0.41601967203001766 | 0.4186862734936872 |
   | [ref10](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/checks/resources-ref10.json:13) | 0.13763484235638043 | 0.1377292775234289 |

   Both config hashes still match, so the staging hash check will not detect the stale model output. **Fix:** regenerate both certificates. The budget plan already uses the corrected values.

8. **WRONG — L2 qualification accounting still fails to bound permitted attempts; budget completeness remains unproven.**  
   The extension superset and explicit screening burns are improvements. CPU probes of the actual extension routine produced 237, 399, and 162 extensions under different validity/temporal outcomes, all contained in the proposed superset. Recomputed L1 cost is **2.4068815540908157 GPU-h**, consistent with the rounded 2.41 reserve.

   But [resources.py:195](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/resources.py:195) **unions** each target’s eight most expensive configurations before summing. This does not bound the union of actual screening-ranked attempts.

   Counterexample: eight strict-target matches cost 10 units each; eight additional loose-target matches cost 9 each but screen faster. If all fail qualification, the strict group attempts the first eight and loose groups attempt the other eight: **152 units**. The certificate’s expensive-eight sets all select the strict eight, charging **80 units**. Even 1.3× safety does not cover that gap.

   **Fix:** sum per-group bounds without deduplication, or bound distinct attempts by the most expensive `max_tries × number_of_targets` configurations per family. Include required compilation work separately.

The **resume path is conditionally consistent, but “results are identical” is too strong**:

- With unchanged configuration and deterministic reruns, the ordinary lattice records are reused; missing records are computed by `ladder()`. Extensions are recomputed from the same base keys and dev statistics. Qualification requests fields explicitly, causing missing fields and spectral `_coefs` to be regenerated before use. I found no inherent missing-`_coefs` failure.
- Screening and qualification predicates are unchanged. Screening timings are measured afresh, however, so identical rankings, attempted subsets, and selected candidates are **not guaranteed** across allocations. Restate [DESIGN.md:516](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/DESIGN.md:516) as numerical-workflow equivalence conditional on unchanged inputs, rather than identical results.
- **Evidence continuity is incomplete:** [vladder.py:98](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/vladder.py:98) initializes empty stage arrays, while resume imports JSON records only. Stage arrays were removed from those records and saved separately at [vladder.py:159](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/vladder.py:159). Consequently, L2’s `stages.npz` omits reused runs that were not recomputed. **Fix:** import the L1 stage archive or explicitly bind and retain it as part of the final combined evidence.
- **NOTED:** the resume hash authenticates the selected JSON bytes, not compatibility of every solver/configuration input. Deliberately substituted checkpoint contents are a forgery-only concern, not a blocking finding here.

The **budget arithmetic is correct; completeness is not established**. [budget-plan.json:65](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-varcoef/experiments/imp-varcoef/checks/budget-plan.json:65) correctly sums to **7.002015551017116 GPU-h**, leaving **0.997984448982884 GPU-h** below eight. All categories required by `resources.py` appear. However, the fallback reserve covers only seven cases without a registered seven-case limit, and no explicit test-fallback allocation or shared-reserve allocation rule is given. L2, timing, and test execution remain provisional reserves rather than demonstrated bounds. **Fix:** price pooled and test fallback explicitly, state what happens when reserves are exceeded, and regenerate the plan after correcting L2.

**Overall verdict: NOT CLEAN — WRONG findings remain in items 7 and 8.**