**READY for the real-size smoke `smk1`. Production is not yet ready.**

Reviewed commit `011f38036`, the audit-7 changes, DESIGN through A5, and the binding contract. Audit-7’s decisive smoke blockers are repaired. I ran small NumPy/AST probes; no solver code changed and no GPU jobs were submitted.

1. **CORRECT — precontraction, steppers and coordinate maps.**

   The QR/eigenbasis transformation, CN residual/Jacobian and tolerance proportional to $\|z_n\|$, ETDRK4 recurrence, and one-pass linearised CN match the amended equations. Coefficients and coordinate maps remain inside evolution.

   **NEEDS-RESTATEMENT:** `phi_np` uses 40 Taylor terms below magnitude 0.5, rather than the specified 12 below 0.2. My independent coefficient comparison found a maximum absolute difference of approximately `7.1e-14` against the contour evaluator. This documentation discrepancy does not block smoke.

2. **CORRECT — the identified certificate/output mutation holes are closed.**

   Original-table diffusion gates, independent coefficient evaluation, per-step certificates and coordinate/output checks remain present.

   [fsrun.py:272](experiments/imp-faststep/fsrun.py#L272) now checks LSPG outputs against scheduled trajectory rows, including anchors and successors. [podarm.py:273](experiments/imp-faststep/podarm.py#L273) implements the corresponding native check, including anchors; the offline checker also verifies native schedules.

   My extracted production schedule-check probe accepted the original outputs and rejected a swap of output rows 1 and 2. I found no new demonstrated mutation that passes the complete main-run checks. This is not a proof against arbitrary implementation mutations.

3. **CORRECT — main Q1/Q2 execution; WRONG — pooled qualification in selection.**

   Production configurations request all 38 cases; successor rollouts and own-state force ratios cover those cases. Fresh anchors and their saved evidence are present.

   However, [fsrun.py:542](experiments/imp-faststep/fsrun.py#L542) applies stationarity allowance separately to each case inside `dev_qualified`, whereas the contract pools steps.

   **Reproduced:** one nonstationary step among six 50-step cases passes the registered `1/300 ≤ 1%` rule, but `dev_qualified` returns false. This can incorrectly eliminate a best-stepper candidate.

   **Before production:** aggregate stationarity counts over the selection population, while retaining every-case validity and anchor requirements.

4. **CORRECT — A4 classifier and main stability monitoring; NEEDS-RESTATEMENT — refinement validity completeness.**

   All six prescribed A4 boundary fixtures returned `unresolved`; positive growth/absence and invalid-refinement probes also behaved correctly. Main runs monitor accepted-step growth and ETD stage finiteness. Classification is no longer truncated.

   Audit-7’s two fixes are present: LSPG refinements use the production stationarity threshold; CN refinements pass their certificate into `g_valid`.

   Remaining diagnostic hardening: LSPG refinement validity still uses a reduced check instead of the complete production validity checks; POD refinement validity at [podarm.py:226](experiments/imp-faststep/podarm.py#L226) omits explicit `stage_max` finiteness. Apply complete validity checks before assigning a resolved classification. This does not change the original run’s conservative instability rejection.

5. **CORRECT — E0 scoring and cohort restriction.**

   Bank, POD-Galerkin and native outputs receive independent offline decoding/scoring. The implicit output-schedule checks close the previously demonstrated shared-schedule hole.

   The lane selector admits only dev6/val32. Historical tests cannot be selected through it; `refstore` additionally rejects sealed/OOD cohorts. Store, metric and reference identity checks remain present.

6. **CORRECT — POD mechanics and successor construction; NEEDS-RESTATEMENT — offline native qualification.**

   Galerkin diffusion, ECSW/full-node advection providers, independent twins, distinct-rule successors, duplicate aliases and Q2-POD successor certification match the amendments.

   `successor_plan` is now the production builder used by C19/C21. My probes confirmed missing-κ skipping and duplicate-rule merging.

   Native production/anchor schedules are checked, and reason/damping arrays are now saved. However, [checks.py:320](experiments/imp-faststep/checks.py#L320) still summarizes native validity, stationarity and anchors principally from JSON rows instead of reconstructing their predicates from those arrays.

   **Before production:** recompute native validity/anchor predicates and distances from persisted evidence; complete the corresponding bank anchor verification.

7. **CORRECT — timing mechanics; selection defect is item 3.**

   E/F/D subjects, warm-up identity checks, retry-wide identity failure propagation, burn-in, drift retries, ratio directions and direct matched/F5 triplets remain implemented. Rank-specific guards are bound correctly.

   Evolve ratios are incumbent/subject; direct fused ratios are competitor/subject, with inversion for the F5 gain. Headline timing requires six dev6 cases and three repetitions. Acceptance requires timing identity and selection agreement.

8. **WRONG for complete production acceptance; sufficiently repaired to run smoke.**

   C19/C21 now exercise production successor construction, closing the previous disconnected-helper defect. C19 still checks successor changes without replaying the registered qualification labels after checkpoint removal.

   Also, [checks.py:372](experiments/imp-faststep/checks.py#L372) unconditionally requires in-job controls at every rank. Consequently, the intentionally control-free `f4096` cannot be accepted.

   **Before production:** finish C19’s label replay and add an explicit, provenance-checked link to accepted `smk1` control evidence for `f4096`. Keep per-job identity, scoring and numerical checks mandatory. Actual control demonstrations—including C9—remain outcomes to verify from `smk1`.

9. **NEEDS-RESTATEMENT — H200 readiness; WRONG — production resource certificate.**

   I found no obvious captured full-bank/POD constants in the principal JIT paths. Large tables are arguments. A single f64 rank-384 bank occupies approximately **3.0 GiB at 1024²** and **48.0 GiB at 4096²**, before construction temporaries, retained subjects and other arrays. Actual peaks and runtime require the registered smokes.

   `resources.py` now enumerates target bank arms/ranks, but still:

   - Adds bank construction to `elapsed.setup`, which already includes it.
   - Counts controls inside rank overhead and again separately.
   - Records GPU type without enforcing compatibility. My AST probe accepted a CPU smoke for an H200-labelled target.
   - Calls missing-rank substitution conservative without establishing a bound; conditional refinement costs can be absent from the smoke.

   **Before production:** correct accounting and validate the extrapolation using `smk1`/`smk4096`. Disabling `f4096` controls resolves the earlier unpriced-control discrepancy.

10. **NEEDS-RESTATEMENT — reporting is improved but unfinished.**

   Smoke F5/direct outputs now carry `descriptive_smoke`, and the diagnostic POD pick exists. Nevertheless:

   - F5 still emits `pass`/`gain_fired` predicates on smoke subsets; explicitly suppress scientific verdicts rather than relying solely on the descriptive flag.
   - The diagnostic pick does not yet produce the registered same-rank/shared-integrator comparison.
   - Offline evidence verification remains incomplete as described above.

**Required before production:** pooled selection, complete offline qualification/control verification, linked `smk1` control evidence for `f4096`, corrected resource certification, and completed diagnostic/reporting outputs.

These remaining issues do not prevent `smk1` from collecting the real-size control, memory and timing evidence. The canonical lab log records this audit outcome.