**NOT READY for (B)** at `43087b9f2`.

Paths below are relative to `experiments/jcp-hr-baselines/`. CPU probes used synthetic, in-memory evidence. No files were changed, no scratch was created, and no jobs, GPU work or `tests/test_core.py` ran.

1. **CORRECT — nonblocking — round-21 item 2: retry exhaustion.**  
   `analyze.py:49–60` now rejects an invalid final attempt before all retries are exhausted. An in-memory probe confirmed rejection.  
   **Fix:** retain this check.

2. **CORRECT — nonblocking — item 4: freeze verification is accurately restated by A33.3.**  
   `freeze_stageb.py:46–57` aggregates certificate summaries and recomputes coefficient-derived distances; `117–130` makes verification mandatory. This matches the narrowed contract in `DESIGN.md:1689–1691`, rather than independent per-step recertification.  
   **Fix:** retain the implementation and use that narrower description consistently in the module documentation.

3. **WRONG — blocking — item 5: the all-missing-member failure path crashes before reporting incompleteness.**  
   `analyze.py:84–85` creates an empty uncertainty array when no member completed. At `153–154`, it computes the comparator envelope before reaching the missing-member guard at `175`. With eligible comparator cases and every frozen member `not_run`, a CPU probe raised a broadcasting `ValueError`. No incomplete verdict is written.  
   **Fix:** handle missing/all-absent members before computing envelopes; obtain reference uncertainties independently of successful baseline runs.

4. **WRONG — blocking — additional coverage defect: incomplete tuning work disappears before frozen-inventory reconciliation.**  
   `freeze_stageb.py:114–116` omits unfinished tuning configurations from membership. Analysis reconciles only that reduced membership (`analyze.py:87–90`); its tuning-coverage check covers ECSW only (`107–110`). Unrun Q-DEIM or `none` configurations therefore do not make the cell incomplete.  
   A CPU probe containing an unrun registered Q-DEIM configuration produced **“no observed domination among the 0 pairs assessed.”** This violates A28.5’s incomplete-cell rule.  
   **Fix:** propagate and reconcile the complete registered tuning inventory, with explicit distinctions between evaluated rejection, construction failure, permitted deferral and unfinished work. Withhold favourable cell conclusions when required work is unfinished.

5. **CORRECT — nonblocking — item 6: batch and qualification propagation.**  
   `stageb.py:174–175` persists batch and qualification. `analyze.py:134–148,190–208,223` applies the shared formatter to missing, rejected, role, exploratory and cell conclusions, including diagnostic-only batch 1 and qualified-subset batch 2.  
   **Fix:** retain these changes.

6. **WRONG — blocking — item 7: saturation/adequacy handling improved, but withholding does not reach individual conclusions.**  
   The required hashed 256² saturation source and unavailable-check handling are implemented at `analyze.py:94–106,130`. ECSW inventory limitations are also recorded. However, role and exploratory labels at `195–208` remain unrestricted when `withhold=True`; only the aggregate cell line receives the withholding suffix (`219–222`). Thus an inadequate basis can still produce an unqualified **“ours dominates”** role label. ECSW coverage restrictions likewise appear only on the aggregate line.  
   **Fix:** attach withholding and applicable construction/coverage restrictions to every affected pair conclusion, while preserving observed baseline domination as allowed by A2.2.

7. **WRONG — blocking — item 8: `analyze.py` does not provide complete accounting for the future report.**  
   `analyze.py:114–115` excludes both jobs’ `ours_pass` phases, although they qualify comparators; A30.2 explicitly charges batch-2 replays and warm timings to baseline V. A CPU fixture with five hours of qualifying/selection phases reported **3.0 hours, `V_complete=True`**.  
   Truncated expenditure is represented only by a boolean (`116–124`), without a bound or active-phase accounting. Training-only T costs are absent from the verdict, including role records (`196–208`), despite A3.1 requiring them.  
   **Fix:** export a complete, nonoverlapping T/V phase classification; include comparator qualification work; provide explicit bounds or unknown expenditure for interrupted phases and a mapping to the corresponding attempts. Supply construction costs alongside selected configurations. Deferring dimension-level ratios and band wording to `make_report.py` is acceptable; incomplete inputs are not.

8. **WRONG — blocking — item 10: timeout can still lose completed evidence.**  
   In `stageb.py:249–254`, a tight anchor completes but is not checkpointed until the following tighter anchor finishes. Termination during that tighter run loses the completed tight anchor. Comparator case metadata is checkpointed before its reconstruction array is written (`192–195`), leaving another interruption window. Timing attempts now checkpoint correctly (`75–78`).  
   **Fix:** checkpoint each completed anchor immediately. Write comparator reconstruction evidence atomically before checkpointing metadata that references it. Record active phases before execution.

9. **NEEDS-RESTATEMENT — nonblocking under A12.4 — item 12: ECSW validation remains partial.**  
   Selected-path quality recomputation is implemented at `freeze_stageb.py:79–94`. Terminal validation, however, checks only counts, positive finite weights and node uniqueness (`89–92`): it omits terminal node bounds/type, terminal residual reproduction, optimum/KKT acceptance and optimum-checkpoint aliasing required by A26.2. The stronger existing helpers are `smoke_accept.py:112–142`. Furthermore, the ECSW path artifact inspected by this validator is absent from the frozen dependency set (`149–155,178`).  
   **Fix:** reuse the complete terminal/optimum validator, bind the selected online sample to its saved checkpoint, and hash the inspected ECSW path artifacts. The current code should not be described as completing the full offline validator.

10. **WRONG — blocking — item 18: resource admission is only a wall-time estimate.**  
    Remote dependency hashing and projection-versus-wall checks are present (`cluster/stage.py:131–145`). But `freeze_stageb.py:157–171` produces no memory certificate, and staging checks no memory bound or GPU compatibility. Stage B retains every admitted comparator object (`stageb.py:159,182–199`), every rank prefix (`208–213`) and every qualified member’s timing-case fields (`240,284–286`). Clearing the HR cache alone does not certify their combined footprint.  
    The time model also omits explicit comparator setup/replay and certification/checkpoint costs; its factor of two is labelled “certified” without corresponding evidence in this path.  
    **Fix:** generate membership-specific host/device peak estimates covering all retained objects and workspaces, check them against the requested resources, and either bound retention or release/reload objects. Include omitted workload components and compare usable driver time, not merely allocation wall time.

11. **NEEDS-RESTATEMENT — nonblocking — item 20: duplicate counting is fixed, but count reporting is incomplete.**  
    `analyze.py:199–213` avoids counting a shared role twice and excludes registered picks from exploratory duplication. However, incomplete/not-assessed branches (`138–183`) omit pair inventories and counts. Successful-path counting also counts role IDs without explicitly requiring a substantive label (`209`), contrary to A4.10.  
    **Fix:** construct one expected-pair inventory for every comparator and count substantive labels, exclusions and missing evidence consistently across all branches.

12. **CORRECT — nonblocking — item 21: timing order.**  
    `stageb.py:299–302` now uses the registered total-order key with actual errors, screening time and support size; the execution loop follows that order (`307`).  
    **Fix:** retain this change.

13. **CORRECT — nonblocking — Stage-B spending reservations.**  
    `cluster/stage.py:62–66,158–162` reserves the configured wall limit. `cluster/ledger.py:23–33,45–58` serializes reservation and submission-state changes, refuses duplicate attempts and enforces the cumulative cap. The submission path disables requeue (`cluster/submit.sh:32`). This repairs the earlier shared-balance reservation race for the prescribed workflow.  
    **Fix:** retain these controls; the resource-certification deficiency in item 10 is separate from the cumulative spending bound.

14. **CORRECT — nonblocking — `ours_V.json` arithmetic and cohort scope.**  
    The item sums equal the stored totals: **4.767744444444444**, **16.8108** and **17.2983 GPU-h** (`inputs/ours_V.json:3–37`); batch 2 adds **0.4875 GPU-h**. This verifies arithmetic, not independent completeness of historical accounting.  
    Stage B uses only `dev6`/`val32` or `val64` (`stageb.py:137–146`), with the reference loader enforcing allowed cohorts (`score.py:25,60–74`). No prohibited-cohort access was found.  
    **Fix:** retain the cohort restrictions and preserve source accounting evidence when the report consumes these totals.