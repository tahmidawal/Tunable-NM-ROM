**(A) Tuning jobs and staging: NOT READY.**

**(B) Stage-B chain: NOT READY.**

Audited execution sources match `4c84a0bde`; observed HEAD `687e465ff` differs only in `HEARTBEAT.json`. References below are relative to `experiments/jcp-hr-baselines/`. Items 1–19 follow round 20’s numbering.

1. **CORRECT — reference eligibility is fixed independently of solver outputs.**  
   `tune.py:146–147,384–392` derives the screening mask from the store and makes invalid eligible-case errors non-finite. `stageb.py:157–158` independently derives the verdict mask; `analyze.py:81` no longer intersects candidate-dependent eligibility.  
   **Fix:** retain these changes.

2. **NEEDS-RESTATEMENT — timing enforcement is substantially repaired, but the validator is not complete.**  
   `stageb.py:318–325` binds timed outputs to scored outputs. `analyze.py:123–130` blocks incomplete timing and output failures; `36–62` recomputes drift, positivity and repetition inventories. However, it accepts an unsuccessful repetition ending before its retries are exhausted. A CPU probe with one such repetition returned no problems. The missing check already exists at `smoke_accept.py:173–174`.  
   **Fix:** require exhausted retries whenever the final attempt is invalid. This particular malformed sequence is not emitted by the current writer; under A12.4, this checker-only finding is nonblocking by itself.

3. **CORRECT — reconstruction replay now blocks admission.**  
   `stageb.py:42,257–277` fixes `REPLAY_TOL`, records replay failures and excludes them from timing. `analyze.py:120–122` excludes replay failures from selection.  
   **Fix:** retain this gate.

4. **NEEDS-RESTATEMENT — freeze verification is mandatory, but “recomputed from evidence” overstates its independence.**  
   `freeze_stageb.py:81–94` makes verification unconditional and blocking; `111–125` hashes replay coefficient files and construction dependencies. `analyze.py:79` checks the frozen tuning hash. However, `freeze_stageb.py:51–57` trusts certificate summaries (`valid`, `nonstat`, `anchor_ok`), and `99–100` trusts comparator acceptance summaries. It does not independently reconstruct those predicates from the saved per-step evidence.  
   **Fix:** recompute certificates from the recorded gamma/reason/damping lists and trajectories, and recompute comparator admission from case evidence. Otherwise describe this accurately as summary aggregation plus distance verification. Summary-forgery scenarios alone are nonblocking under A12.4.

5. **WRONG — missing frozen members still disappear from selection and conclusions.**  
   `analyze.py:80,119` iterates only `B['members']`; it never reconciles that set against `FZ['members']`. Inventory counts at `98–99` do not constrain selection. The timing-incomplete guard at `138–141` therefore misses members whose accuracy runs never completed. Missing comparator records similarly disappear because `106` iterates only recorded comparators.  
   An in-memory probe with a frozen member marked `not_run` produced **“no observed domination among the 0 pairs assessed.”**  
   **Fix:** reconcile frozen members, comparators and expected cases first. Preserve missing/failed-work reasons and mark affected cells incomplete; missing evidence must not support favorable conclusions.

6. **WRONG — batch metadata and qualifications do not survive every required artifact or verdict line.**  
   Tuning and freeze propagation is present (`tune.py:162–166`; `freeze_stageb.py:104–109`). Stage-B comparator records omit it (`stageb.py:171–192`). Analysis recovers metadata from freeze (`analyze.py:107–109`), but prefixes only the successful `cell_line` branch (`178–179`); early-return labels and individual role/exploratory labels remain unprefixed. Printed qualified-batch conclusions also omit the qualified-subset limitation (`185`).  
   **Fix:** persist batch and qualification provenance in Stage B, then apply one qualification formatter to every emitted conclusion, including incomplete and not-assessed outcomes.

7. **WRONG — saturation and ECSW consequences remain incomplete.**  
   `analyze.py:84–92` makes `--sat256` optional and accepts an unpinned file without validating its cell identity. Omitting it makes the 1024² snapshot limitation false. Missing/failed adequacy or saturation phases also escape withholding because `103` checks only explicit `False`. ECSW coverage at `86–88` considers construction failures but misses unattempted boundary work. Required offline costs are absent from role records (`156–157`).  
   **Fix:** require a hashed, correctly identified 256² saturation result for 1024² analysis; treat unavailable checks as unassessed. Derive ECSW coverage from the expected inventory and attach required limitations and costs to affected verdicts.

8. **WRONG — search accounting and budget wording remain unsafe.**  
   Adding completed `B|` phase durations is progress (`analyze.py:89–96`). But tuning writes `search_hours` only at normal completion (`tune.py:521–522`), and analysis converts its absence to zero. The CPU probe confirmed the resulting **“baseline search within ours’ allocation”** claim. Selection-relevant A–B–A timing is also omitted, although it determines the picks (`stageb.py:330`; `analyze.py:145–146`). The V-budget ratio and required factor-band/descriptive wording are not implemented.  
   **Fix:** reconstruct accounting from checkpointed classified phases, represent unfinished expenditure as unknown or bounded, include selection-relevant timing, and implement the A3.1/A4.3 wording.

9. **CORRECT — the previous population-wide device-data retention is removed.**  
   `stageb.py:203–228` shares basis prefixes and replaces the HR-data cache; timing reconstructs data by group (`297–307`) instead of retaining every option’s device data.  
   **Fix:** retain this architecture. Separately certify resource admission for actual frozen membership; this repair is not itself a memory-fit certificate.

10. **WRONG — evidence preservation is still incomplete.**  
    Baseline screening cases now checkpoint (`tune.py:373–376`), and Stage-B production cases do too (`stageb.py:251–254`). Remaining losses include:
    - tuning anchors accumulated without checkpoints (`tune.py:359–361`);
    - probe cases and coefficients saved only after the configuration (`475–483`);
    - Stage-B comparator cases saved only after the comparator (`182–197`);
    - Stage-B anchors accumulated across the cohort (`242–245`);
    - timing attempts retained only in local memory until an entire case finishes (`61–78,321–322`).

    **Fix:** predeclare these work units and atomically checkpoint each completed anchor, comparator/probe case and timing attempt, including its reconstruction evidence.

11. **CORRECT — tuning execution and defer order now match A30/A31.**  
    `tune.py:415–419` runs every registered HR and `none` configuration before optional work. Probe runs precede the extension (`456–520`); the extension has the registered start guard (`501`).  
    **Fix:** retain this order.

12. **NEEDS-RESTATEMENT — the earlier optimum checks remain, but the requested offline validation extension is absent.**  
    `smoke_accept.py:112–142` retains terminal/optimum validation. Its quality validation still checks individual attempt labels (`329–334`), while the freeze/analyze chain propagates the stored combined rank quality (`freeze_stageb.py:93`; `analyze.py:137`). It does not recompute that label from selected-path drift and banning evidence.  
    **Fix:** add an offline ECSW artifact validator covering terminal evidence and the combined selected-path quality label before freezing membership. Do not describe that follow-up as completed.

13. **CORRECT — the probe lattice basis is now persisted.**  
    `tune.py:291–298` saves `Phi_probe_lat`; uncertified probe ranks remain excluded (`465–469`). Saturation replacement and seeded adequacy routing remain implemented (`234–281`).  
    **Fix:** retain these repairs; address incomplete probe checkpointing under item 10.

14. **WRONG — FOM inventory is predeclared too late to describe whole-job truncation.**  
    Convergence checks and per-case persistence remain correct (`tune.py:441–449`). But FOM inventory is created only after all registered screening work, inside `fom_runs` (`423–433`). A timeout during screening leaves no expected FOM entries.  
    **Fix:** declare the complete FOM inventory during initial job setup, before expensive execution.

15. **CORRECT — Stage-A timing agreement remains a front-admission gate.**  
    `tune.py:385–393` records agreement, and `freeze_stageb.py:81–89` requires it. Comparator timing agreement remains enforced at `tune.py:202–208`.  
    **Fix:** retain these gates.

16. **CORRECT — registered grids and selection primitives remain consistent.**  
    The production configurations match the rank/option/comparator definitions at `make_phase2_configs.py:8–34`: 140 base configurations per 2D cell and 1,008 in 3D. `stats.py:118–186` retains uncapped fronts, near-front additions, X-specific membership and role selection.  
    **Fix:** retain these primitives; repair their coverage and accounting inputs.

17. **CORRECT — comparator adapters retain the intended query mappings.**  
    `ours.py:86–116` uses the P1 implementation; `126–148` uses speed3d’s mesh query, including quasi-Newton arms. Batch-2 input identity is explicit, for example `inputs/speed3d_batch1_e0.json:1021–1024`.  
    **Fix:** retain these mappings and runtime G-ours checks; repair qualification propagation under item 6.

18. **WRONG — Stage-B staging still lacks configuration-aware preflight.**  
    M2 and configuration-specific files are staged (`cluster/stage.py:30,59`). Asset preflight remains guarded by `a.driver == 'tune'` (`134–139`). Stage B validates its external tuning directory only after GPU allocation (`stageb.py:115–125`). No staging check validates the frozen workload against a regenerated resource admission.  
    **Fix:** preflight Stage-B assets, external tuning dependencies, hashes and membership-derived resource admission before submission.

19. **WRONG — the ledger check does not enforce a cumulative spending bound.**  
    `cluster/stage.py:62–70` checks the existing ledger and wall limit, but staging creates no reservation. `cluster/submit.sh:17–26` serializes concurrency checks without reserving or rechecking GPU-hours. Multiple staged attempts can therefore pass against the same balance and subsequently exceed the cap; retries have the same hole.  
    **Fix:** atomically reserve each attempt’s maximum GPU-hours, revalidate at submission, and reconcile reservations with measured expenditure. Account explicitly for retries/requeues and outstanding Stage-B commitments.

20. **WRONG — assessed-pair counts double/triple-count identical comparisons.**  
    `analyze.py:158–168` includes registered picks again as exploratory comparisons, then adds both role entries. A CPU probe with one unique pair filling both roles reported **three assessed pairs**. This directly violates A4.10 (`DESIGN.md:827–829`). Excluded and missing pairs are also absent from the reported not-assessed count.  
    **Fix:** count unique `(X, baseline)` pairs receiving substantive labels; retain role membership separately and reconcile all expected pairs by outcome/reason.

21. **WRONG — Stage-B timing order differs from the registered total order.**  
    `stageb.py:290–292` prepends `(kind,r,hp)` and substitutes zero time/support values before applying `total_order_key`. Thus budget truncation selects a different timed subset from A30.1’s required A1.12 order.  
    **Fix:** construct the registered total-order records with actual available statistics and support sizes, then execute that order while keeping the bounded HR cache.

No prohibited cohort access was found in these paths (`score.py:25,60–74`). The committed gate-suite pass does not cover the orchestration defects above.

No files were changed, no scratch was created, no jobs or GPU work ran, and `tests/test_core.py` was not executed.