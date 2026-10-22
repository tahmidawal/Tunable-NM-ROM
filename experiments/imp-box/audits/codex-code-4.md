**Overall verdict: NEEDS-FIX.** Most round-3 fixes work, but commit `054fd7d9b` introduces a deterministic solver-build failure, and the claim-label fix remains incomplete.

Read-only audit: no files changed, jobs submitted, or LAB-LOG writes. Tiny CPU checks used the prescribed interpreter with `JAX_PLATFORMS=cpu` and bytecode writes disabled. Paths below are relative to `experiments/imp-box/`.

1. **Problem, cohorts, IC and lattice — CORRECT**

   Descriptor mapping, Gaussian IC, sine convention, x-major ordering and full boundary-inclusive scoring lattice agree: `geom.py:35–68,98–109`, `spec.py:38–80`, `podsine.py:63–88,163–192`, `bank.py:40–46`.

   CPU checks:
   - Split disjointness passes with 384 training rows.
   - Asymmetric-mode synthesis agrees between FOM and POD evaluation to **6.13e-16**.
   - All files marked unmodified in `PROVENANCE.json` match their hashes.

   All descriptor splits are hashed at `cell.py:193–199`; test solutions are generated only in the test stage. **Required fix: none.**

2. **Spectral FOM, controller and certificate — CORRECT**

   Fixed ETDRK4 and its independent transcription agree: `spec.py:93–117,130–173`. Adaptive coefficients through J+1, integer ticks, aligned coarsening, charged rejected trials, rejection-at-J termination and unfinished-output invalidation follow the governing amendments: `spec.py:178–260`.

   Tiny CPU results:
   - Fixed coefficient discrepancy **1.70e-15**, stage/update defect **4.62e-17**.
   - Adaptive discrepancy **1.20e-13**, defect **5.63e-16**.
   - Forced rejection with J=3, j₀=2: **two rejects, one cap hit, invalid output**.

   The reduced NaN hook uses the global step index. **Required fix: none.**

3. **References and gates — CORRECT**

   The maximum of coefficient/lattice differences, two-condition round-off branch, retry, conditional ε/U, padding populations and training gates are consistent: `refgate.py:20–60`; `cell.py:112–170,193–264,277–284`. Reference-gate fixtures pass.

   MF-REF-real and MF-PAD-chk exercise the production eligibility predicate with clean-pass/fault-reject requirements. Their first-pass accepted-case population is explicitly permitted by A7.3. Unscorable selection/test populations cannot produce matched results.

   **Required fix: none.**

4. **POD, ECSW and TRANS mathematics — CORRECT**

   Parseval scaling, reorthonormalization, exact derivatives/stiffness, all-retained-mode rule checks and operator gating are implemented: `podsine.py:35–157`; `cell.py:327–358`. ECSW uses the registered training rows/snapshots and candidate sampling: `ladders.py:131–145`; `podsine.py:217–262`.

   TRANS centres the descriptor and decodes with **x−δ, δ=c−0.5**: `podsine.py:180–192`.

   G-OPS passes: point/tensor discrepancies below **1.70e-14**, mass discrepancy **3.90e-15**, stiffness discrepancy **1.11e-15**. **Required mathematical fix: none.**

5. **Ladders, coverage and statuses — CORRECT for the round-3 repairs**

   Construction failures now affect the two-finalist early return, and meeting rows with failed screens become incomplete evidence: `timing.py:158–166,196–207`; `ladder.py:108–115`.

   CPU selector fixtures returned:
   - Two qualified rows plus a construction failure → **finalist-partial**.
   - Accurate row with no screen → **missing-construction**.
   - Two qualified rows without failures → **finalist**.

   Qualified-finalist extensions, tolerance-extension closure, pruning and termination records remain connected: `timing.py:229–277`; `ladder.py:132–170`.

   **Required fix:** restore the solver constants identified next so the mandatory B-LSPG ladder can actually be constructed.

6. **Qualification, selection, timing and freeze — WRONG**

   **New blocking regression:** the cache rewrite deleted both `TOLS` and `GAL_TOLS`, while [rom.py:298](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/rom.py:298) still references them.

   A one-mode CPU build reproduces:

   ```text
   lspg_cnr: NameError: name 'TOLS' is not defined
   gal_cnr:  NameError: name 'GAL_TOLS' is not defined
   gal_etdrk4: build OK
   ```

   This breaks every B-LSPG query, including its certified anchors. The reported earlier end-to-end smoke cannot validate this commit’s implicit path.

   **Fix:** restore the previous tolerance dictionaries exactly and exercise production, tight and tighter LSPG query builds/replays.

   The sampled tensor-cache collision is fixed by retaining the space object alongside its identity key: `rom.py:160–166`. Current callers construct replacement spaces for weight faults. Making cached arrays immutable would be useful **hardening**, not a demonstrated current-path blocker.

   Paired T-id, compile checks, T-dbl invalidation and frozen query-source/artifact identity remain connected: `timing.py:215–225,322–355,375–412`.

7. **Diagnostics — NEEDS-FIX**

   Round-3’s ECSW timing-label collision is repaired: the full configuration now identifies component repetition arrays, including κ (`diag.py:74,108`). Evaluation/query crossover calculations, selectable-form filtering, rank-specific projection and finer-reference reconstruction remain consistent.

   **The claim-label finding is only partly fixed.** `diag.py:261–266` records eligible counts without original population totals. Diagnostic entries do not carry applicable truncated-search or deferred-coverage labels. Thus a conditional worst error still lacks the required “k of n” qualification within its diagnostic record.

   **Fix:** propagate a shared claim context containing eligible/total counts, applicable training qualification, search status and coverage labels into each diagnostic result. Use original cohort sizes, not inferred defaults.

8. **Must-fail controls — NEEDS-FIX because of the new regression**

   The registry is explicit, missing mandatory controls fail, and failed controls exit nonzero: `controls.py:198–223`. The exclusion fixture now chooses a currently eligible row and requires the population to decrease by one: `controls.py:183–191`. This closes the round-3 hardening finding.

   However, every controls run reaches the broken LSPG builder at `controls.py:59–60`, including non-bank cells. **Consequently neither registered H200 smoke can complete its mandatory controls.**

   **Fix:** restore the tolerance tables and rerun the clean/fault LSPG certificate control. A6.6 explicitly justifies reporting, rather than requiring, clean ECSW qualification; I do not count that exception as a defect.

9. **Test stage — NEEDS-FIX for evidence labels**

   Authentication, retained-cell freeze checks, artifact/source/descriptor verification, accepted-control requirement, unscorable early return and frozen-pair-only evaluation remain correct: `teststage.py:24–74,133–159,167–209`. Atomic opening reservation remains in `cluster/bodies/test.sh.in:3–6`.

   **Round-3’s test-label finding is unfixed.** Test entries and pairs omit training-data qualification and selection/coverage qualifications (`teststage.py:167–209`). Moreover, `timing.py:407–411` drops boundary labels and timing-exclusion details when creating the freeze, so the test stage cannot faithfully reproduce all qualifications on `S_val`.

   **Fix:** retain those qualifications in the frozen selection/pair records and copy them into test entries, alongside explicit eligible/total counts. Keep the existing no-reselection rule.

10. **Stop rules and final.json mapping — CORRECT**

   Timing-excluded finalists now receive the registered one-sided treatment: **S<0.5 becomes missing; valid S≥0.5 can establish NOT FIRED** (`stoprule.py:68–73`). Incomplete FOM searches remain missing; censoring/non-vacuity and deferral handling remain intact.

   All stop fixtures passed, including both new timing-exclusion cases. Population and training labels are now present in `cell_labels`: `stoprule.py:141–155`.

   **Required fix: none to the predicates.** Preserve these labels when rendering individual stop entries.

11. **Cluster scripts and release — CORRECT for smoke mechanics; production certificate still needs work**

   H200 enforcement, absolute venv, highest precision, GPU preflight, execution lock, campaign queue-cap rechecks and exact body/resource approval are present: `cluster/job.sbatch.tmpl:9–20`; `cluster/submit.sh:18–49`.

   The round-3 collection defect is fixed: `SHA256SUMS` is excluded consistently from source and archive listings at `cluster/collect.sh:23`. Terminal-state checks and verification-before-deletion remain present.

   **Production-only gap:** `resources.py:53–57` checks that coverage files *exist*, not that required coverage completed. Source compatibility covers query sources, not ladder policy or environment overrides such as `IGEO_NCAP` (`ladder.py:24`). It also checks N_ref against the recorded level without establishing that `smk-A4` actually recorded A4.

   **Fix before production certification:** validate smoke identity, expected coverage/termination records and effective ladder settings. These are **not prerequisites for measuring the registered memory pilot**.

12. **Real-size memory/runtime — CORRECT as a pilot plan; peaks unverified**

   A5.8/A7.8 make the H200 smoke the memory pilot. Lack of measured peaks is **not independently a smoke blocker**.

   Remaining risks include eager coefficient/POD arrays, simultaneous rule spaces, whole-rule bank ordering, retained successful outputs and whole-file test hashing: `cell.py:297–311,332–344`; `builders.py:20–29`; `bank.py:155–157`; `timing.py:182–194`; `teststage.py:73`.

   **Additional hardening:** `del sps` does not release spaces retained by builder closures such as `br`, `be` and `bt` (`ladders.py:121–164`). Delete those closures when their arm finishes, or scope each arm separately. ECSW construction can still exceed its nominal deadline before NNLS begins (`podsine.py:220–250`). Peak records are written only after successful stage completion (`cell.py:451–460`).

   Preserve failure evidence and keep resource failures classified as missing. Do not interpret these calculated allocations as measured H200 bounds.

**Required fixes before READY-TO-SMOKE:**

- Restore `TOLS`/`GAL_TOLS` and verify the mandatory LSPG path and control.
- Complete claim-context propagation through diagnostics, freeze and test results: eligible/total populations, training qualification, truncated/partial searches and timing exclusions.

**Overall verdict: NEEDS-FIX.**