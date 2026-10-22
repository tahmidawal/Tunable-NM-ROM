**READY-TO-SMOKE for the registered H200 pilot only.** Two round-5 follow-ups remain: production coverage certification is still unsound, and test timing failures retain a misleading matching label. Neither requires delaying the memory pilot.

Audited `21dd3de4e`, read-only. No files changed, jobs submitted, LAB-LOG writes, or arrays copied. Checks used the prescribed Python, CPU backend, f64, highest precision and disabled bytecode writes. Paths below are relative to `experiments/imp-box/`.

1. **Problem, cohorts, IC and lattice — CORRECT**

   Descriptor mapping, Gaussian IC, sine normalization and x-major ordering agree: `geom.py:35–68,98–109`, `spec.py:38–80`, `podsine.py:63–88`, `bank.py:38–43`.

   CPU checks passed split disjointness with 384 training rows. Asymmetric-mode FOM/POD synthesis differed by **3.67e-16**. All provenance entries marked unmodified matched their hashes.

   Test descriptors are generated and hashed before selection, as explicitly registered; test solutions are generated only at `teststage.py:152`.

   **Required fix: none.**

2. **Spectral FOM, adaptive controller and certificates — CORRECT**

   ETDRK4 stages and independent transcription agree (`spec.py:93–117`). Integer ticks, aligned coarsening, coefficients through J+1, rejected-trial work, rejection-at-J termination and unfinished-output invalidation follow the latest amendments (`spec.py:178–260`).

   Tiny CPU results:

   - Fixed: coefficient discrepancy **1.70e-15**, stage/update defect **4.62e-17**.
   - Adaptive: **1.20e-13**, **5.63e-16**, respectively.
   - Forced rejection with J=3, j₀=2: **two rejects, one cap hit, invalid output**.

   Reduced-stage injection uses the global step index (`rom.py:267–284`).

   **Required fix: none.**

3. **References and gates — CORRECT**

   The coefficient/lattice maximum, two-condition round-off branch, conditional ε/U, retry, padding populations and training gates are connected correctly: `refgate.py:20–60`; `cell.py:114–172,195–264,279–286`.

   Reference-gate fixtures passed. MF-REF-real and MF-PAD-chk invoke the production eligibility predicate and require clean acceptance, fault rejection and measured margin. Their accepted first-pass population is permitted by A7.3.

   **Required fix: none.**

4. **POD, ECSW and TRANS — CORRECT**

   Parseval scaling, reorthonormalization, exact derivative evaluation, stiffness, all-retained-mode rule checks and operator eligibility are implemented: `podsine.py:35–157`; `cell.py:329–358`.

   ECSW uses the registered rows, snapshots and candidate sampling (`ladders.py:131–145`; `podsine.py:218–262`). TRANS centres the descriptor and decodes with **x−δ, δ=c−0.5** (`podsine.py:180–192`).

   G-OPS passed: point/tensor discrepancies below **1.70e-14**, mass discrepancy **3.90e-15**, stiffness discrepancy **1.11e-15**.

   **Required mathematical fix: none.** ECSW deadline enforcement remains a hardening item under item 12.

5. **Ladders, extensions, pruning and coverage — CORRECT**

   Flop screening and hopeless-rung pruning match the amendments: `ladders.py:62–72`; `ladder.py:132–170`.

   Construction failures and failed meeting-row screens produce incomplete statuses, including the two-finalist early return (`timing.py:158–204`). Qualified-finalist spatial extensions and adaptive tolerance closure remain connected (`timing.py:228–277`).

   **Required fix: none to these selection predicates.** The resource certificate mishandles their statuses; see item 11.

6. **Qualification, validation timing and freeze — CORRECT**

   The restored tolerance dictionaries remain present (`rom.py:31–32`). LSPG anchors are certified; FOM/reduced ETDRK4 certificates, tensor Q2, ECSW parent Q2 and centred TRANS states remain connected: `finalize.py:75–130`; `qual.py:28–115`; `timing.py:89–122`.

   Validation timing requires every timed output to pass T-id, no compile-cache growth, and both T-dbl checks (`timing.py:216–225,339–350`). `finalize.py:241–265` excludes incomplete timing rows before maxmin selection.

   Freeze retains selection statuses, boundary labels, construction/screen failures, termination reasons and timing exclusions (`timing.py:400–419`).

   **Required fix: none to the validation path.**

7. **Diagnostics — CORRECT for the round-5 repair**

   Round-5’s missing search context is now supplied:

   - Per-configuration status/boundary context: `diag.py:48–52,281–284,314–321`.
   - Both FOM families’ statuses and boundary labels at crossover entries: `diag.py:246–249`.
   - Original population totals and training qualifications: `diag.py:271–278`.

   Evaluation/query crossover arithmetic, selectable-form filtering, rank-specific projection and finer-reference reconstruction remain consistent (`diag.py:137–185,190–266`).

   **Required fix: none for current cell diagnostics.** Report generation must preserve this context and the checkpoint’s explicit deferrals when rendering individual claims.

8. **Must-fail controls — CORRECT, with hardening suggestions**

   The mandatory registry rejects missing controls and exits nonzero on failure (`controls.py:198–223`). Numerical faults exercise production weights, tensors, certificates, reference gates and decoding. Selector and eligibility fixtures exercise their corresponding paths.

   A6.6 explicitly permits a failed clean ECSW Q2 result; that exception is not a defect.

   **Required fix: none.** Hardening: add a valid, authenticated positive fixture for MF-LEAK. Its current negative fixtures principally demonstrate missing/wrong-path rejection (`teststage.py:77–90`). Also directly assert the refinement transition in MF-AD-rej, rather than relying on rejection count and eventual validity (`controls.py:72–83`).

9. **Test stage — NEEDS-FIX before test release**

   Authentication, retained-cell freeze completeness, artifact/source/descriptor verification, atomic opening, unscorable handling, Q1/Q2 and no reselection remain connected: `teststage.py:24–74,133–191`; `cluster/bodies/test.sh.in:3–6`.

   **Round-5’s numerical timing fix works:** fresh T-dbl checks are now required, repetitions are persisted, and failed timing yields `S_test=None` (`teststage.py:211–217`).

   **The matching-label fix remains incomplete.** [teststage.py:201](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/teststage.py:201) sets `matched_on_test=good` before timing. Lines 212–217 never revise it when timing fails. Thus a record can simultaneously say:

   ```text
   matched_on_test = true
   timing_valid = false
   S_test = null
   ```

   **Fix:** preserve `numerically_matched=good`, and define accepted `matched_on_test=good and timing_valid`. Give timing failures an explicit reason. This is a reporting defect; invalid test speed-ups are now suppressed correctly.

10. **Stop rules and final.json mapping — CORRECT**

   **Round-5’s blocking BANK-STOP defect is fixed.** `stoprule.py:140–143` now rejects incomplete evidence from either FOM family before consuming a diagnostic verdict.

   An in-memory fixture through `evaluate()` used otherwise-above bank entries plus one `below` entry backed by `finalist-partial` adaptive FOM evidence. The entry became **missing**, and BANK-STOP became **UNRESOLVED**.

   Speed-stop incomplete-search, timing-exclusion and non-vacuity handling remain intact (`stoprule.py:48–112`).

   **Required fix: none.**

11. **Cluster scripts — CORRECT for smoke mechanics; production certificate NEEDS-FIX**

   H200 enforcement, absolute venv, precision, GPU preflight, execution lock, campaign cap rechecks and exact approved job specification are present: `cluster/job.sbatch.tmpl:9–20`; `cluster/submit.sh:18–49`. Collection checks terminal state and verifies archive/local checksums before deletion (`cluster/collect.sh:14–30`). Shell syntax checks passed.

   Smoke identity is now checked correctly (`resources.py:58`). **Coverage certification remains unsound:**

   - [resources.py:62](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/resources.py:62) rejects only `search-incomplete` and `missing-construction`, accepting `finalist-partial` despite construction failures.
   - Coverage/termination files need only exist (`resources.py:56`); expected entries and termination records are not validated.
   - Missing environment keys count as default settings (`resources.py:60`).
   - Source compatibility covers query sources, excluding ladder-policy code (`timing.py:378–379`).

   **Reproduction:** an in-memory `bounds()` fixture with `finalist-partial`, 20 construction failures, empty termination records and `env={}` still produced bounds for **A1–A4**.

   **Fix before production certification:** validate expected coverage and termination records; reject or explicitly cost unfinished construction/search work; require recorded effective settings; bind the ladder policy used for the certificate. This is not a prerequisite for running the registered pilot.

12. **Real-size memory/runtime — CORRECT as a pilot plan; unmeasured risks remain**

   A5.8/A7.8 explicitly make the real-size H200 smoke the memory pilot. Lack of measured peaks is **not a separate smoke blocker**.

   Remaining pressure points include eager coefficient/POD arrays, simultaneous rule spaces, whole-rule bank ordering, retained successful outputs and whole-file artifact hashing: `cell.py:297–344`; `builders.py:20–29`; `bank.py:155–157`; `timing.py:182–197`; `teststage.py:73`.

   Builder closures are released between ladder arms, and peak recording runs in `finally` (`ladders.py:149,163,179`; `cell.py:439–442`).

   **Hardening:** stream test artifact hashing; bound live spaces/output caches; enforce the ECSW deadline during matrix construction (`podsine.py:222–250`). Preserve scheduler evidence after SIGKILL/OOM, which Python `finally` cannot capture. Such failures remain missing evidence, not scientific outcomes.

**Required follow-up fixes:** complete production coverage/settings certification before production release; distinguish numerical matching from accepted timing before test release.

**Overall verdict: READY-TO-SMOKE — registered H200 pilot only.**