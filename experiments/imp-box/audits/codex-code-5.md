**Overall verdict: NEEDS-FIX.** The restored tolerance tables work. Round-4’s claim-context repair is incomplete, and BANK-STOP can still consume an incomplete FOM search as conclusive evidence.

Audited HEAD `4b4bb3db9`, read-only. No files changed, jobs submitted, LAB-LOG writes, or large arrays copied. Checks used the prescribed interpreter, CPU backend, f64, highest precision, and disabled bytecode writes. Paths below are relative to `experiments/imp-box/`.

1. **Problem, cohorts, IC and lattice — CORRECT**

   Descriptor mapping, Gaussian IC, sine normalization and x-major ordering agree: `geom.py:35–68,98–109`, `spec.py:38–80`, `podsine.py:66–88,164–192`, `bank.py:38–43`.

   CPU checks:
   - Disjointness passes with 384 training rows.
   - Asymmetric-mode FOM/POD synthesis differs by **3.67e-16**.
   - Shifted decoding matches the analytic asymmetric mode exactly.
   - All provenance entries marked unmodified match their recorded hashes.

   Every descriptor split is hashed at `cell.py:193–199`. Test solutions are generated only in `teststage.py:152`.

   **Required fix: none.**

2. **Spectral FOM, adaptive controller and certificates — CORRECT**

   ETDRK4 stages and independent transcription agree: `spec.py:93–117`. Integer ticks, aligned coarsening, coefficients through J+1, charged rejected trials, termination at J and unfinished-output invalidation follow the governing amendments: `spec.py:178–260`.

   Tiny CPU checks:
   - Fixed: coefficient discrepancy **1.77e-14**, stage/update defect **3.97e-17**.
   - Adaptive: **1.14e-12**, **1.02e-15**.
   - Forced rejection with J=3, j₀=2: **two rejects, one cap hit, invalid output**.

   The reduced NaN hook uses the global step index: `rom.py:267–284`.

   **Required fix: none.**

3. **References and gates — CORRECT**

   Coefficient/lattice maximum, two-condition round-off branch, conditional ε/U, retry resolution, padding populations and training gates are consistent: `refgate.py:20–60`; `cell.py:112–170,193–264,277–284`.

   MF-REF-real and MF-PAD-chk invoke the production eligibility predicate and require clean acceptance, fault rejection and sufficient measured margin. A7.3 explicitly permits their first-pass accepted-case population. Reference-gate fixtures passed.

   **Required fix: none.**

4. **POD, ECSW and TRANS — CORRECT**

   Parseval normalization, reorthonormalization, exact derivatives/stiffness, all-retained-mode rule checks and operator gating are implemented: `podsine.py:35–157`; `cell.py:327–358`.

   ECSW uses the registered training rows, snapshots and candidate sampling: `ladders.py:131–145`; `podsine.py:218–262`. TRANS centres the descriptor and decodes at x−δ with δ=c−0.5: `podsine.py:180–192`.

   G-OPS passed: point/tensor discrepancies below **1.70e-14**, mass discrepancy **3.90e-15**, stiffness discrepancy **1.11e-15**.

   **Required fix: none to these mathematical paths.**

5. **Ladders, extensions, pruning and coverage — CORRECT**

   The registered flop screen and hopeless-rung prune are implemented: `ladders.py:62–72`; `ladder.py:132–170`.

   Construction failures and failed meeting-row screens propagate into incomplete statuses, including the two-finalist early return: `timing.py:158–204`. Qualified-finalist extensions and AD tolerance closure remain connected: `timing.py:228–277`.

   **Required fix: none to the ladder predicates.** Their incomplete-search status must also reach BANK-STOP; see item 10.

6. **Qualification, validation timing and freeze — CORRECT**

   **Round-4’s deterministic regression is fixed.** `TOLS` and `GAL_TOLS` are restored at `rom.py:31–32`.

   Tiny CPU production, tight and tighter LSPG builds/replays all produced finite outputs without nonstationary steps, damping saturation or unconverged steps. The clean/fault certificate probe changed the nonstationary fraction from **0 to 1**.

   Certified anchors, ETDRK4 replay, actual tensor Q2, ECSW parent qualification and centred TRANS states remain connected: `finalize.py:75–130`; `qual.py:28–115`; `timing.py:89–122`.

   Validation timing checks every timed output, compile-cache growth and T-dbl before admitting a pair: `timing.py:216–225,339–350`. Freeze now preserves selection statuses, boundary labels, construction/screen failures, termination reasons and timing exclusions: `timing.py:400–419`.

   **Required fix: none to validation timing.** Test timing has a separate gap below.

7. **Diagnostics — NEEDS-FIX**

   **Round-4 claim-context finding is only partly fixed.** `diag.py:261–268` now records original population totals and training qualification. However, neither diagnostic writer serializes applicable selection statuses, FOM boundary labels or coverage qualifications: [diag.py:271](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/diag.py:271), `diag.py:303–324`.

   In particular, crossover entries contain a comparator key, timing, floor and verdict, but no indication that the comparator came from a partial or truncated search: `diag.py:230–253`. The necessary information already exists in `sel`.

   **Fix:** attach the relevant FOM selection statuses and boundary labels to each crossover entry, and carry applicable selection/coverage context into component and finer-lattice records. Preserve explicit deferrals when producing checkpoint claims.

   Evaluation/query crossover arithmetic, selectable-form filtering, rank-specific projection, component timing keys and finer-reference reconstruction otherwise remain consistent.

8. **Must-fail controls — CORRECT, with hardening suggestions**

   The restored LSPG builder removes round-4’s controls-stage blocker. The explicit registry requires missing controls to fail and exits nonzero on failure: `controls.py:198–223`.

   Reference faults, tensor faults, certificates, production-weight faults, shifted decoding, selector exclusion and frozen-artifact mutation exercise their intended paths. The eligibility fixture now removes a currently eligible row: `controls.py:183–191`.

   A6.6 explicitly permits reporting a failed clean ECSW Q2 result without requiring that positive control.

   **Required fix: none.** Hardening: add an authenticated, valid-manifest positive fixture for MF-LEAK; its current negative fixtures primarily exercise missing/wrong-path rejection (`teststage.py:77–90`).

9. **Test stage — NEEDS-FIX**

   Authentication, campaign freeze completeness, artifact/source/descriptor verification, atomic opening, unscorable handling, Q1/Q2 and no reselection remain correct: `teststage.py:24–74,133–191`; `cluster/bodies/test.sh.in:3–6`.

   Round-4’s population/training context and frozen selection-label preservation are substantially repaired: `teststage.py:166–202`.

   **Timing validity is weaker than validation timing.** Test pairs check T-id and compile count, but never run or require T-dbl: [teststage.py:207](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/teststage.py:207)–214. Validation explicitly requires both T-dbl outcomes at `timing.py:347–350`. The earlier validation result does not verify the new test descriptors in the test allocation.

   **Fix before test release:** apply one shared timing-validity predicate to both stages, including fresh test-case T-dbl checks and persisted repetitions. Publish `S_test` only when it passes. Also make `matched_on_test` distinguish numerical matching from accepted timing; it currently remains true when timing fails.

10. **Stop rules and final.json mapping — WRONG**

   Existing speed-stop predicates and all supplied fixtures pass, including incomplete FOM and timing-exclusion cases.

   **New blocking defect: BANK-STOP bypasses the incomplete-FOM check.**

   Speed entries reject incomplete FOM searches at `stoprule.py:55–60`. Bank entries instead accept any diagnostic verdict when controls passed and `final.json` is scorable: [stoprule.py:135](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/stoprule.py:135)–140. Meanwhile, `diag.py:230–245` computes crossover verdicts from partial finalist sets.

   An in-memory fixture through `evaluate()` reproduced:
   - Fixed FOM: `finalist`.
   - Adaptive FOM: `finalist-partial`.
   - Speed entry: **missing**.
   - Bank diagnostic: `below`.
   - BANK-STOP: **NOT FIRED**, even with the other cells missing.

   An unfinished competitor search can discover a faster FOM, lower the affordable bank rank and increase its floor. Thus the current `below` verdict cannot conclusively prevent the stop.

   **Fix:** validate the relevant FOM search completeness before consuming each bank verdict; map incomplete evidence to `missing`. Add this fixture through `evaluate()`, not only through the standalone predicate.

11. **Cluster scripts — CORRECT for smoke mechanics; production certification still NEEDS-FIX**

   H200 request/check, absolute venv, highest precision, GPU preflight, directory lock, locked campaign/lane cap rechecks and exact approved job specification are present: `cluster/job.sbatch.tmpl:9–20`; `cluster/submit.sh:18–49`.

   Collection checks terminal state and verifies archive/local checksums before deletion: `cluster/collect.sh:14–30`. The earlier checksum-file mismatch remains fixed. Shell syntax checks passed.

   **Round-4 production-only gap remains:** `resources.py:53–57` checks coverage-file existence rather than completed coverage, does not bind effective ladder settings such as `IGEO_NCAP`, and does not establish that the job named `smk-A4` actually represents W40/A4.

   **Fix before production certification:** validate smoke identity, expected coverage/termination records and effective settings. This does **not** require completing the memory pilot before authorizing it. The empty `RELEASE.json` currently prevents submission.

12. **Real-size memory/runtime — CORRECT as a registered pilot; hardening remains**

   A5.8/A7.8 make the real-size H200 smoke the memory pilot. Unmeasured peaks are **not independently a smoke blocker**.

   Round-4 improvements are present:
   - Builder closures are released between ladder arms: `ladders.py:149,163,179`.
   - Peak recording runs in `finally`: `cell.py:437–440,462–473`.

   Remaining risks are eager coefficient/POD arrays, simultaneous rule spaces, whole-rule bank ordering, retained successful outputs and whole-file test hashing. Relevant locations: `cell.py:297–311,332–344`; `builders.py:20–29`; `bank.py:155–157`; `timing.py:182–197`; `teststage.py:73`.

   **Hardening:** stream artifact verification, bound retained spaces/outputs, and enforce the ECSW deadline during matrix construction (`podsine.py:222–250`). Python `finally` cannot preserve peaks after SIGKILL/host OOM; retain scheduler failure evidence and classify such jobs as missing.

**Required fixes:**

- Finish diagnostic search/coverage qualification propagation.
- Prevent incomplete FOM searches from producing conclusive BANK-STOP entries.
- Apply the complete timing-validity contract before publishing test speed-ups.
- Before production certification, validate actual smoke identity, coverage and effective settings.

**Overall verdict: NEEDS-FIX.**