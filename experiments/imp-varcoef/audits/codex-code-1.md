<!-- recovered from the codex exec log: the -o output file was not written (run 1); final agent message below -->
1. **`vref.py:232` — WRONG. Reference jobs crash after their first cross-check.**  
   `crosscheck()` returns `delta1`, `delta2`, etc., but the logger accesses nonexistent `xc['diffs']`. Both production reference configs exercise this path.  
   **Fix:** log the returned fields and persist the completed case before subsequent diagnostics.

2. **`gate.py:103`, `gate.py:113`, `gate.py:175` — WRONG. Failed timing validation does not prevent a gate verdict.**  
   Timing gates and controls are copied into the report but never enforced. `complete`, backend, anchor veto, qualification provenance and prescribed coverage are not validated. A synthetic input with `complete=False`, every timing gate false and `T-id` false still returned **PASS-PROVISIONAL**. Furthermore, G-RR failure does not trigger the inherited-ROM fallback if triplets remain valid.  
   **Fix:** enforce completion, valid anchor, required controls, qualified FOM identity and exact six-slot/three-repetition coverage. Apply ROM scoring/identity failures to the per-ROM fallback decision.

3. **`gate.py:156` — WRONG. Incomplete stage-2 coverage can crash instead of becoming inconclusive.**  
   When `fs` contains some—but not all—slots, computing `[fs[c] for c in cases]` raises `KeyError` before the `not f_ok` branch.  
   **Fix:** check coverage before indexing or calculating cost; emit the A2.26 inconclusive outcome.

4. **`elig.py:41` — WRONG. Missing cross-checks count as success.**  
   `any(...)` over an empty cross-check dictionary is false. A synthetic complete reference record with 32 accepted cases and **zero cross-checks** produced `decidable=True`. Partial cross-check coverage has the same problem.  
   **Fix:** require successful records for exactly the prescribed dev cases 0–3 at each contrast, bound to the final reference versions. Validate the exact dev/val case set rather than just its length.

5. **`vladder.py:87`, `vladder.py:110`, `vtime.py:136`, `gate.py:103` — WRONG. Frozen artifacts are recorded but not authenticated across stages.**  
   The ladder does not verify the eligibility `.FROZEN` digest or loaded reference arrays against `ref_sha256`. Timing accepts independently supplied slots and candidate groups without checking them against the frozen mask and qualified ladder. Gate inputs can likewise come from different runs.  
   **Fix:** validate the complete hash chain: final references → eligibility/slots → qualified candidates → frozen selection → timing → gate, including the frozen ROM-box digest.

6. **`vspec.py:162`, `vspec.py:212`, `vref.py:110` — WRONG. Spectral validity can miss a final-step blow-up; cross-check validity is weaker still.**  
   PSI tracks the state entering each step, and ETD tracks stage inputs; neither checks the final accepted state after the last update. `validity()` only checks output finiteness, so a finite final output exceeding 10 can pass. Cross-check validity additionally omits the running-maximum and completion predicates.  
   **Fix:** include every accepted updated state, including the terminal state, in the maximum; apply the shared full validity predicate to cross-check runs.

7. **`vspec.py:158`, `vfd.py:199` — WRONG. Required per-stage evidence is discarded.**  
   PSI/FDMG retain iteration sums/maxima and maximum residual, not each stage’s iteration count and independently recomputed residual as required by A1.12.  
   **Fix:** return and persist per-stage arrays with their schedule indices. Keep aggregates as summaries.

8. **`vladder.py:247` — WRONG. Cheap-edge extensions select the wrong provisional winner.**  
   Selection uses unburned lattice-run wall times, excluding dev0, instead of registered full-field screening latency. This can trigger different extensions or omit necessary ones. CCv’s loosest tolerance edge is also ignored because it uses `level`, not `tol`.  
   **Fix:** screen eligible-dev provisional candidates under the registered full-field contract before selecting extension triggers; implement the CCv tolerance-edge extension explicitly.

9. **`vladder.py:319`, `vfd.py:35`, `vfd.py:303` — WRONG. CCv qualification does not use the registered E0 tight anchor.**  
   Qualification uses level `0` = `(ntol, ltol)=(1e-10,1e-11)` and `T=(1e-12,1e-13)`, both with 20 Newton iterations. The registered E0 tight setting is `(1e-13,1e-12,40)`.  
   **Fix:** provide an explicit E0-tight qualification solver and specify the tighter comparison consistently with A1.12.

10. **`gate.py:131` — WRONG. The obsolete all-unstable-family exemption survives.**  
    A family with no valid runs receives “unstable on the tested ladder” and is excluded from `unresolved_families`. This permits a positive gate despite no qualified configuration from that required family. A4.10 replaces that exemption: unresolved required families block a positive verdict.  
    **Fix:** classify every required family lacking a qualified configuration as unresolved for passing, retaining “all tested runs invalid” as descriptive evidence.

11. **`vkeys.py:75`, `vkeys.py:88` — WRONG. CCv lacks the A3.14 decomposition.**  
    `CCvProgram.kernel` is `None`; `setup_only` returns only face coefficients. A CCv `KERNEL|...` subject therefore crashes, and the setup program omits its preconditioner setup. Its query also remains two dispatches rather than the registered single jitted program.  
    **Fix:** implement genuine `setup`, `core`, fused `query`, kernel-only and setup-only programs for CCv, including all per-query denominators.

12. **`vtime.py:269`, `vtime.py:396` — WRONG. Diagnostic timing subjects are neither mandatory nor adequately validated.**  
    Kernel/setup subjects only run if manually supplied in `cfg['foms']`; a union winner can therefore lack the required measurements. Kernel reproduction failures are excluded from G-FR and do not veto the diagnostic.  
    **Fix:** construct decomposition subjects for every frozen union winner and require kernel/query trajectory identity before accepting their timings.

13. **`vref.py:199`, `vtest.py:31` — WRONG. Test-once protection is only a truthy-string check.**  
    Any nonempty `frozen_selection_sha256` unlocks the test cohort. No selection file is hashed, requested keys are not checked against the decisive frozen winners, and repeated execution is unrestricted. `vref` even constructs the unlocked cohort before checking the token.  
    **Fix:** verify the actual frozen-selection artifact before cohort generation, derive the permitted keys from it, and maintain an atomic test-opening/run record allowing only controlled recovery of the same evaluation.

14. **`checks_impl.py:173`, `checks_impl.py:193` — WRONG. Required smoke controls are incomplete and failures do not stop execution.**  
    T-nan is advertised but absent. T-loose omits the A3.19 residual-to-threshold margin. MG probes exercise only one hierarchy depth, rather than every deployed level count. Failed checks still exit successfully; reference controls similarly do not prevent a completed job.  
    **Fix:** implement the missing controls/evidence, probe deployed hierarchy depths, and make failed mandatory controls block production explicitly.

15. **`make_configs.py:114` — WRONG. The documented production configuration commands do not exist.**  
    Only `base` is dispatched; `ladder` and `time` raise `KeyError`. Production timing configuration, authenticated slots/candidates and mandatory winner diagnostics are consequently not generated by this code.  
    **Fix:** implement those generators and their provenance/schema checks before production.

16. **`cluster/stage.py:47`, `cluster/submit.sh:18`, `ledger.py:53` — WRONG. GPU-hour caps and the complete-work budget are unenforced.**  
    Staging checks an individual certificate and wall limit, but never checks `spent + outstanding reservations + all remaining certified work ≤ 8`. Submission does not consult the ledger or budget plan. Pending jobs contribute zero actual GPU-hours, and reservation limits are not included in the reported total.  
    **Fix:** under submission locking, reconcile accounting and reservations, verify the full A3.20 plan—including fallback, extensions, qualification, retry and test work—and enforce phase/lane caps.

17. **`cluster/submit.sh:12`, `cluster/submit.sh:18`, `ledger.py:25` — WRONG. Concurrency and submission bookkeeping can race.**  
    The lane-cap check and remote-directory existence check occur outside the campaign lock. Concurrent submissions can exceed the lane cap or share a directory. Inside the lock, a failed `squeue` can become a zero count through `wc -l`. Ledger updates lack locking and atomic replacement; a post-submit error can leave a live job unrecorded.  
    **Fix:** perform fail-closed queue queries, both caps, atomic directory reservation and submission registration under the shared lock; use locked, atomic, idempotent ledger updates.

18. **`vcoef.py:22`, `vcoef.py:66` — CORRECT.**  
    Row-wise RNG order, coefficient weighting, normalization parameters and paired contrasts match the design.  
    **Fix:** none to the formulas. Record normalization preparation time, currently omitted.

19. **`vspec.py:50`, `vspec.py:88`, `vspec.py:148`, `vspec.py:198` — CORRECT.**  
    Galerkin sign and factor four, SPD implicit system, PCG recurrence, SBDF2/CNAB2 coefficients and first steps, and ETDRK4 remainder splitting are consistent. Tiny CPU checks gave constant-coefficient operator relative error **4.05e-16** and one CG iteration with residual **7.43e-17**.  
    **Fix:** none to these formulas; address validity/evidence issues above.

20. **`vfd.py:47`, `vfd.py:59`, `vfd.py:111`, `vfd.py:195` — CORRECT.**  
    Conservative face fluxes, diagonal, full weighting, bilinear prolongation, symmetric V-cycle construction and FD SBDF2 signs are consistent. Tiny CPU probes gave MG symmetry error **9.37e-18**, positive quadratic form, and `P=4Rᵀ` discrepancy **8.88e-16**.  
    **Fix:** none to these formulas; real-size hierarchy checks remain necessary.

21. **`vfd.py:230`, `vref.py:113`, `vref.py:216` — CORRECT.**  
    Variable-coefficient CN-R has the correct residual and four BE half steps. Péclet uses the coarsest operator’s minimum face coefficient. Unchanged `judge`, floor proxy and stage checks are used. A5.5 adjacent-difference extrapolation and both resolved alternatives are correctly implemented.  
    **Fix:** none to these formulas; repair the cross-check crash and validity omissions.

22. **`vladder.py:189`, `vladder.py:262`, `vladder.py:338` — CORRECT, subject to the preceding defects.**  
    Pruning uses eligible-dev error while testing validity on all dev cases; no eligible dev cases disables accuracy pruning. Temporal rechecking after extension and pooled-case spectral operator/rollout qualification implement the relevant amended predicates.  
    **Fix:** preserve these predicates when repairing extension selection.

23. **`vtime.py:305`, `vtime.py:386`, `gate.py:81`, `rom_box.py:38` — CORRECT, subject to validation enforcement.**  
    The C2 diff preserves fresh selection/inference blocks, burn–A–burn–B–burn–A timing, retries and identity comparisons. Selection is written before stage 2. Slot bootstrap and median-of-slot-medians implement the registered estimand. ROM-box selection searches both qualified lists and uses historical anchor-normalized costs; its current freeze digest matches.  
    **Fix:** authenticate these artifacts and enforce their validation results as described above.

24. **`vtime.py:459`, `gate.py:169`, `vladder.py:418` — NEEDS-RESTATEMENT. Reporting remains incomplete.**  
    The registered historical-C2 ±20% consistency flag, T-kappa diagnostic, paired PSI C=100/C=1 cost reading, and explicit ratio of median absolute seconds are not implemented in these paths.  
    **Fix:** implement them before claiming the registered deliverables complete, or explicitly amend and label their omission.

**Overall verdict: NOT CLEAN — WRONG findings must be fixed before accepting the real-size smoke or production results.**
