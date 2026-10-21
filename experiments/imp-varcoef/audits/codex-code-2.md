<!-- recovered from the codex exec log: the -o output file was not written; final agent message below -->
Read-only audit completed. No files changed, jobs submitted, GPU execution, or arrays copied. Checks used CPU only and in-memory fixtures.

**The lane is not ready for production.** An adversarial gate fixture still returned **PASS-PROVISIONAL** with one slot, repetition 99, a vetoed anchor, empty scoring records, and an unqualified FOM winner.

Paths below are relative to `experiments/imp-varcoef/`.

1. **CORRECT — reference cross-check logger.** `vref.py:235–239` uses the returned cross-check fields and saves the result. The original `xc['diffs']` crash is fixed.  
   **Fix:** none for that defect; saving the reference case before calling the cross-check would additionally protect completed work.

2. **WRONG — gate validation remains insufficient.** `gate.py:118–119,156–175,197–199` trusts the supplied case count and repetition count; counts records without checking repetition identities; accepts empty `scoring` through `all([])`; and never rejects a vetoed anchor. The in-memory fixture described above passed. Per-ROM reproduction failures now correctly trigger inherited fallback, but that does not repair overall validation.  
   **Fix:** require exactly `dev6|0…5`, repetitions `{0,1,2}` once per subject/slot, exact scoring coverage, a non-vetoed validated anchor, and authenticated qualified winners. Validate timing records rather than trusting their `valid` flags alone.

3. **CORRECT — incomplete stage-2 coverage.** `gate.py:175–183` checks coverage before indexing `fs` and emits an inconclusive result.  
   **Fix:** none for the original `KeyError`; strengthen coverage validation under item 2.

4. **WRONG — cross-checks are not bound to the final reference.** `elig.py:35–36,41–43` fixes missing prescribed cross-checks and validates the pooled case set. However, later fallback cases overwrite `cases` independently of `xc`. A fallback without a new cross-check can inherit a successful cross-check against the old reference.  
   **Fix:** store the checked reference hash in each cross-check and require equality with the final case’s `ref_sha256`; invalidate stale cross-checks when replacing a reference.

5. **WRONG — the hash chain remains incomplete.** `vtime.py:126–142` verifies membership in a global candidate dictionary, not qualification for the particular family/target/contrast group. `vtime.py:192–199` still loads reference arrays without checking their frozen hashes. `gate.py:108–117` accepts `bindings={'files': {}}`, does not require specific bound files, and does not authenticate the supplied ladder or frozen selection against the timing result.  
   **Fix:** require a fixed binding schema; bind candidates to the completed qualified ladder and eligibility; validate exact group membership and ROM specifications; hash loaded reference arrays; verify the actual frozen selection and its correspondence to `T['selection']`.

6. **WRONG — terminal-state guard is only partially repaired.** `vkeys.py:102–120` now checks returned-field maxima, and `vref.py:108` uses shared validity. But `vspec.py:162,212–213` still omits the terminal accepted state from the solver’s running maximum. Checking its lattice synthesis is not identical to evaluating the maximum on the grid used by the nonlinear operator.  
   **Fix:** update `umax` from the terminal accepted state on the same native evaluation grid used for other states; retain output finiteness and output-maximum checks.

7. **CORRECT — PSI/FDMG stage evidence.** `vspec.py:160,169`, `vfd.py:204,212`, and `vladder.py:150–153,182–183` now retain and persist stage iteration/residual arrays. Array position supplies the stage index.  
   **Fix:** none for the original PSI/FDMG evidence omission.

8. **NEEDS-RESTATEMENT — cheap-edge extension.** `vladder.py:252–270` does implement the stated superset: every dev-valid, target-matching edge configuration is extended without latency selection. **That is a faithful conservative replacement of the provisional-winner trigger:** it includes every extension that trigger could require. It changes the registered workload, however, and `DESIGN.md:334` still specifies provisional winners. Also, CCv level `4` is `(1e-3,1e-4)` in `vfd.py:37`, whereas A1.10 names the `1e-2` tolerance extension.  
   **Fix:** amend the design and budget to register the superset explicitly, and either use the registered CCv loose endpoint or explicitly register the new endpoint.

9. **CORRECT — CCv qualification anchor.** `vfd.py:37–38,312–314` and `vladder.py:350–351` provide E0 tight `(1e-13,1e-12,40)` and a tighter linear-solver comparison `(1e-13,1e-13,40)`.  
   **Fix:** none.

10. **CORRECT — unresolved families.** `gate.py:143–146` no longer exempts all-invalid families from unresolved status. Their instability remains descriptive evidence.  
    **Fix:** none.

11. **WRONG — CCv decomposition remains incomplete.** `vfd.py:320–325` supplies callable setup/kernel/query programs, fixing the missing kernel and two-dispatch query. A tiny CPU check gave identical query/kernel fields and valid output. However, setup returns only `KX`, `KY`, and `nu`; preconditioner denominators are still constructed inside stepping at `vfd.py:249`.  
    **Fix:** prepare the startup and ordinary-step denominators in `setup`, return them in `S`, and consume them in `core`. Setup-only must include all registered per-query setup.

12. **WRONG — diagnostic timing validation.** `make_configs.py:145–146` generates decomposition subjects, but `vtime.py:290–292` still excludes kernels from G-FR. `vtime.py:298–302` compares each kernel execution only with that kernel’s own scoring identity, not with its query trajectory. A consistently wrong kernel can therefore retain valid diagnostic timings.  
    **Fix:** require kernel/query trajectory identity per case, veto failing diagnostics, and enforce complete decomposition measurements for every union winner.

13. **WRONG — test-once enforcement remains replayable.** `testgate.py:29–39` authenticates supplied artifacts but performs no execution claim. `make_configs.py:183–187` uses a non-atomic existence-check/write sequence. `vtest.py:32` can be invoked repeatedly with the same record; requested keys need only be a subset of winners, and `contrast` is not bound to C=100.  
    **Fix:** atomically claim a persistent evaluation identity; bind contrast, exact winners, cohort, references, code and configuration; permit only explicit recovery of that same evaluation.

14. **CORRECT — original missing smoke controls, with a new acceptance gap below.** `checks_impl.py:88–98,153–173,190–194` adds deployed MG depths, T-loose margins and T-nan. `smoke_accept.py:35–47` plus `cluster/stage.py:48–49` provides a production-blocking acceptance step. Individual drivers still exit successfully on failed predicates, but the new production acceptance check addresses the original requirement to block production explicitly.  
    **Fix:** retain this gate and repair its real-size/pilot checks described below.

15. **WRONG — production generators exist but generate an unusable timing configuration.** `make_configs.py:148–159` adds PSI `@C1` subjects while generating slots only for contrasts 100 and 10. `vtime.py:161,184–186` accesses the absent `slots['1']` during input preparation.  
    **Fix:** map C=1 diagnostics explicitly to the corresponding C=100 physical cases, and validate every generated subject’s slot mapping before constructing the bank.

16. **WRONG — budget enforcement is still not transactional.** `cluster/stage.py:50–54` checks a staging-time ledger snapshot and trusts a precomputed plan scalar. Staged jobs are not reserved there; `cluster/submit.sh:18–28` does not recheck the budget under submission locking. Several jobs can be staged against the same remaining allowance and subsequently exceed it.  
    **Fix:** reconcile accounting, reserve each job, validate the complete remaining-work plan, and enforce phase/lane caps under the submission transaction. Use the rounded Slurm wall limit when reserving cost.

17. **WRONG — submission bookkeeping still races.** Queue checks and directory promotion are now inside the campaign lock, and failed `squeue` is rejected. But ledger registration remains outside that lock at `cluster/submit.sh:28`. `ledger.py:52–68` has no read-modify-write lock, and `ledger.py:31` uses one shared temporary filename. Concurrent writers can lose entries or interfere with replacement; interruption after submission can leave an unrecorded job.  
    **Fix:** use locked, idempotent ledger transactions with unique temporary files and durable submission/recovery records. **The nested shell quoting itself is correct for staged job names**: mocked checks returned counts 0/0 and 1/1, rejected two lane jobs with exit 74, and rejected eight account jobs with exit 75.

18. **CORRECT — coefficient family.** `vcoef.py:26–40,66–78` preserves the registered sampling and normalization formulas. `vref.py:213–215,230` adds a normalization-cost measurement.  
    **Fix:** none to the formulas; label that measurement as a repeated preparation measurement rather than the original cohort-construction interval.

19. **CORRECT — spectral numerical formulas.** `vspec.py:50,88–112,148–157,198–213` retains the previously reviewed operator signs, PCG recurrence and time-stepping formulas.  
    **Fix:** none to those formulas; repair item 6.

20. **CORRECT — FD/MG numerical formulas.** `vfd.py:47–146,195–205` retains the reviewed conservative flux, restriction/prolongation, V-cycle and SBDF2 formulas.  
    **Fix:** none to those formulas.

21. **CORRECT — reference formulation and extrapolation.** `vfd.py:234–284` preserves variable-coefficient CN-R and startup stepping; `vref.py:113–145,218–225` preserves A5 extrapolation and the discrete Péclet convention.  
    **Fix:** none to these formulas; reference-version binding remains item 4.

22. **CORRECT — pruning and amended qualification predicates.** `vladder.py:201–205,275–286,347–387` preserves no-eligible-dev pruning behavior, temporal rechecking, and pooled spectral qualification.  
    **Fix:** none to these predicates.

23. **CORRECT — estimand and selection/inference structure, conditional on validation fixes.** `vtime.py:327–352,361–426`, `gate.py:72–88,185–192`, and `rom_box.py:38` retain fresh timing blocks, burn–A–burn–B–burn–A, frozen selection before inference, and the registered slot-based statistic/bootstrap.  
    **Fix:** authenticate and validate their inputs as specified in items 2 and 5; the mathematical estimand itself needs no change.

24. **NEEDS-RESTATEMENT — reporting remains incomplete.** `vladder.py:289–306` adds T-kappa, and `gate.py:190–192` adds the absolute-seconds ratio. The C=1 reading is currently blocked by item 15 and is not summarized as the registered paired cost comparison. `gate.py:165–166` also reports perfect historical consistency after substituting the historical fallback itself.  
    **Fix:** implement the paired PSI comparison, and report historical consistency only from successful fresh measurements; otherwise use “not assessed.”

**NEW defects or regressions in the changes:**

- **C=1 timing subjects crash production input preparation.** `make_configs.py:148–159` → `vtime.py:161`. Missing C=1 slots produce `KeyError`; fix as item 15.
- **New bindings validation accepts an empty file map and wrong-target candidates.** `gate.py:108–117`; `vtime.py:134–137`. Require mandatory bindings and exact qualified group membership, not global membership.
- **New scoring validation accepts empty evidence.** `gate.py:156–157,173–174`. `repro_checked` can claim coverage while `scoring=[]` passes. Derive coverage and reproduction from exact per-case records.
- **New consistency flag can falsely certify fallback as a fresh match.** `gate.py:161–166`. Inherited fallback makes `vs_inherited=1` and `consistent_20pct=True` automatically. Emit “not assessed” instead.
- **New smoke acceptance does not establish real-size smoke or pilot success.** `smoke_accept.py:33–47` never requires `imp['small'] is False` or `tim['pilots_ok']`; staging subsequently trusts `accepted` without revalidating the recorded source hashes. Require the registered real-size configuration, successful required pilots, and authenticated acceptance sources.

**Overall verdict: NOT CLEAN — WRONG findings remain.**
