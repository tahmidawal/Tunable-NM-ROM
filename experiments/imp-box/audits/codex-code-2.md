**Overall verdict: NEEDS-FIX.** Commit `cd75a624a` fixes many round-1 findings, but construction failures can still become censoring, the freeze does not protect every query-defining input, and the real-size resource certificate remains insufficient.

Read-only audit completed: no files changed, jobs submitted, or lab-log writes. CPU checks used the specified interpreter, `JAX_PLATFORMS=cpu`, and tiny arrays. Paths below are relative to `experiments/imp-box/`.

1. **Problem, cohorts, IC and lattice — CORRECT**

   Round-1 fixes are implemented:
   - Full scoring lattice, including TRANS wall values: `geom.py:98–103`, `spec.py:77–80`, `podsine.py:164–176`.
   - Disjointness checks the requested training population: `geom.py:60–68`, `cell.py:184`.
   - Every descriptor split is hashed; precomputing test descriptors is explicitly distinguished from generating test solutions: `cell.py:185–189`.
   - Test references are generated only in `stage_test`: `teststage.py:129–130`.

   Sine normalization, x-major synthesis ordering, Gaussian descriptors and bank coordinates agree. An asymmetric-mode CPU probe found maximum synthesis disagreement **3.67e-16**. Disjointness passed with 384 training rows.

   **Required fix:** none here; enforcing the stored hashes during test remains outstanding under item 9.

2. **Spectral FOM and adaptive controller — CORRECT**

   Fixed ETDRK4 and the independent transcription agree: `spec.py:93–117`. Adaptive ticks, aligned coarsening, coefficients through `J+1`, accepted/rejected work, immediate rejection-at-cap termination and unfinished-output invalidation implement the governing amendments: `spec.py:178–260`.

   The reduced NaN hook now uses the global step index, fixing round 1: `rom.py:263–280`.

   Tiny CPU checks:
   - Fixed coefficient discrepancy **2.72e-15**, stage/update defect **8.90e-17**.
   - Adaptive coefficient discrepancy **1.18e-12**, defect **6.17e-15**.
   - Forced rejection with `J=3`, `j0=2`: **two rejects, one cap hit, invalid output**.

   **Required fix:** none to the stepping/controller logic.

3. **References and gates — NEEDS-FIX**

   The coefficient/lattice maximum, two-condition round-off branch, retry, allowance and training checks are correct: `refgate.py:20–60`, `cell.py:102–161,253–260`. Selection now stops for unscorable cells: `cell.py:406–410`. G7 covers reference, retry and FOM sizes: `cell.py:164–180`.

   Round-1’s inconsistent retry coefficients are now explicitly retained as **first-pass control data only** (`cell.py:161`); scientific coefficient diagnostics regenerate the per-case reference resolution (`diag.py:158–163`). This is a justified resolution.

   **Remaining defect:** controls still require the *original*, failed reference to pass. After a legitimate successful retry, MF-REF-real uses `raws['val']` from the first pass and requires `clean_ok`; MF-PAD similarly uses first-pass arrays (`cell.py:205–230`). A retry that rescues val[0] can therefore produce qualified references but necessarily fail the mandatory controls registry.

   **Fix:** run clean/fault controls against the accepted reference resolution for the designated case, preserving its resolution and gate provenance. Handle an entirely ineligible test population as described in item 9.

4. **POD, ECSW and TRANS — CORRECT**

   Parseval scaling, reorthonormalization, projection, derivatives and exact stiffness are consistent: `podsine.py:35–125`. ECSW uses the registered state population and sampling seed: `ladders.py:178–184`, `podsine.py:218–259`. TRANS centres the dynamics and decodes with `delta=c−0.5`, using `x−delta`: `podsine.py:180–192`.

   Round-1 fixes:
   - Rule selection and M/K checks use **all retained modes**: `cell.py:310–323`.
   - Failed POD operators prevent selection: `timing.py:143–145`.
   - G-OPS now gates mass and blocks the stage on failure: `gops.py:70–78`, `cell.py:333–334`.

   CPU G-OPS passed; point/tensor discrepancies were below **1.70e-14**. The files declared unmodified in `PROVENANCE.json` match their recorded hashes.

   **Required fix:** none to these mathematical implementations; allocation hazards remain under item 12.

5. **Ladders, coverage and extensions — NEEDS-FIX**

   Flop screening and hopeless-rung pruning match A5: `ladders.py:102–110`, `ladder.py:149–154`. Termination reasons are now persisted (`ladder.py:164–168`), resolving the lost-prune-record finding. Qualified-finalist N extensions were added (`timing.py:208–231`).

   Two defects remain:

   - **Tolerance extension misses another boundary finalist.** `timing.py:234–240` extends the first qualifying `tol=1e-7` finalist and immediately breaks. If that extension already exists and does not change selection, the second boundary finalist is never extended. A mocked two-finalist fixture at N=384 and N=512 produced only `specad|N384|tol1e-08`.
   - **Partial construction still becomes scientific censoring.** `Arm.rows()` discards failures (`timing.py:74–75`); `select()` distinguishes missing construction only when *no* valid rows exist (`timing.py:146–150`). A failed large-rank ECSW fit plus valid inaccurate small-rank rows becomes `arm-censored`. Missing supports are visible in `ecsw_fits.json` but never consulted by selection (`ladders.py:184–195`). The same problem applies to caught build failures in `ladder.py:98–103`.

   **Fix:** extend every distinct binding finalist configuration until closure; propagate expected-versus-completed construction coverage into selection. Resource/build failures must remain missing even when another configuration ran successfully. Attach step-cap/termination labels to affected selection entries, not only the separate termination log.

6. **Selection, qualification, timing and freeze — NEEDS-FIX**

   Round-1 fixes verified:
   - Attempt-cap singleton is `finalist-partial`: `timing.py:160–162`.
   - Tensor Q2 computes actual tensor contractions: `qual.py:83–89`.
   - Weight faults affect production rollouts and vectors: `qual.py:59–79,101–110`.
   - T-dbl failure invalidates validation timing pairs: `timing.py:294–297`.
   - Certified LSPG anchors, ETDRK4 replay checks, three-way matching, paired output checks and max–min selection remain correctly wired.

   **New cache hazard:** `rom.py:159–164` keys tensors by `id(sp)` and name but retains no reference to the space. Python may reuse that identity after destruction, returning another space’s tensor. A CPU lifetime probe confirmed that the space is freed while its cache entry remains; allocator reuse was not observed in that short probe. Q1 can replay the same wrong cached tensor, so it is not sufficient protection.

   **Freeze remains incomplete:** `timing.py:326–338` hashes Phi/ECSW/bank artifacts but omits `pod.json`, whose `k_q` controls reconstructed test queries (`ladders.py:44–48`). Its stored commit is not checked against the executing solver source.

   **Fix:** use an object-safe bounded cache, with immutable space data or explicit invalidation. Freeze and verify every query-defining input, including quadrature metadata and solver-source identity. Preserve T-dbl repetition arrays; `finalize.py:223–239` still discards them.

7. **Diagnostics — WRONG**

   Several round-1 repairs are present: selectable-form filtering, rank-specific projection, dynamic TRANS shift timing, boundary defects, query crossover output and FOM coefficient diagnostics (`diag.py:88–91,192–206,211–252,267–307`).

   Remaining problems:

   - **Continuous error can discard candidate high modes.** `diag.py:306` calls `refgate.coef_diff`, which truncates the candidate to the reference’s size (`refgate.py:11–15,23`). This matters when a selected N=512 FOM is compared with an A3 N=384 reference. A tiny probe with a unit high-mode discrepancy returned **0.0**, whereas the correct normalized error was **1.0**.
   - **Query crossover does not minimize the registered statistic.** It chooses one configuration using the single-case ladder screen, then rescreens only that configuration on val[0:4] (`diag.py:215–222`). Another row can be cheaper under the required four-case statistic.
   - **Component timing arrays remain discarded.** Component calls omit `label`, while `tmed` saves repetitions only when a label is supplied (`diag.py:30–45,74–120`). `timing_raw` therefore does not contain these measurements.
   - Eligibility and training-qualification labels are still absent beside diagnostic/stop claims; stored metadata elsewhere does not meet A3.1’s “at the claim” requirement.

   **Fix:** zero-pad *both* coefficient arrays to their common maximum size; rescreen all eligible query candidates before minimizing; label and persist every timing block; propagate population counts and training qualification into each affected readout.

8. **Must-fail controls — NEEDS-FIX**

   The explicit registry closes most of round 1: G4, reference checks, G-OPS, TRANS identity/sign, stop fixtures and timing controls are now included (`controls.py:189–209`). TRANS uses an exactly representable field. ECSW parent Q2 is genuinely computed. A6 explicitly justifies not requiring clean ECSW Q2 to pass.

   However:
   - **Mandatory controls can disappear.** MF-ECSW is registered only if a fit exists (`controls.py:95–115,204–205`). With no fits, the registry can pass without testing ECSW. MF-ENV similarly disappears when the bank artifact is absent.
   - **Partial-eligibility fixture is not a fault fixture.** `controls.py:182–184` merely compares lengths for the current mask; an all-eligible smoke passes without exercising exclusion.
   - Failed controls write `all_fire=false` but return successfully (`controls.py:209–212`), and `cell.py:434` marks the stage done. Stop-rule ingestion rejects these results, but test preflight does not.

   **Fix:** declare required controls from the intended cell arms before executing them; missing controls must fail or explicitly block acceptance. Inject an exclusion in the eligibility fixture. Make control failure block test opening and release.

9. **Test stage — NEEDS-FIX**

   Fixed: committed blob checking, current-manifest index membership, frozen-pair-only timing, test Q1/Q2, raw test timings and suppression of `S_test` on invalid timing (`teststage.py:24–61,149–183`).

   Remaining gaps:
   - **Campaign completeness is not verified.** The index check validates only this cell (`teststage.py:35–38`). It does not establish that every retained cell has an authenticated frozen manifest before the first test opens.
   - **Frozen inputs are not fully enforced.** No comparison of current descriptor hashes against `fr['cohorts']`; no solver-code identity check; unhashed `pod.json` changes can alter queries.
   - **Zero eligible cases crash instead of reporting unscorable.** `stage_test` continues after `rec['scorable']=False`; empty `c.Dv` reaches `run_unbatched` and `np.stack([])` (`teststage.py:130–155`, `finalize.py:48–54`).
   - **Test-once protection is not atomic.** The archive existence check in `cluster/bodies/test.sh.in:3` permits two differently named jobs to restore the same untested cell before either is collected. Failed attempts that opened test references but never wrote `test.json` are also unrecorded as openings.

   **Fix:** validate the complete retained-cell index and all immutable inputs; require accepted controls; return a persisted unscorable result before evaluating an empty population; atomically reserve each cell’s test opening in shared storage before reference generation.

10. **Stop rules and status mapping — CORRECT, conditional on upstream statuses**

   Round-1 mapping defects are repaired:
   - Either incomplete FOM family makes the entry missing: `stoprule.py:55–60`.
   - Partial reduced searches follow A6’s one-sided rule: `stoprule.py:71–72`.
   - Missing arms are excluded from “infeasible everywhere”: `stoprule.py:92–93`.
   - Deferral matching is shared; scorable/control acceptance is checked: `stoprule.py:28–42,122–140`.

   Existing boundary/non-vacuity fixtures all passed on CPU. `operator-check-failed` is treated equivalently to unqualified in the predicates.

   **Required fix:** repair upstream incomplete-construction statuses from item 5 and carry eligible-population labels into stop entries. Otherwise a correct predicate can still consume incorrect evidence.

11. **Cluster scripts — NEEDS-FIX**

   Round-1 improvements are present: validated job/collection names, H200 request and runtime check, absolute venv, precision, GPU preflight, execution lock, locked queue-cap checks, terminal-state collection and checksum verification.

   Outstanding:
   - **Release approval binds only a job name.** `submit.sh:19–21` does not bind approved body, cell settings, memory/time request or resource certificate. An approved name can execute a different committed body.
   - **The budget certificate is still not a demonstrated bound.** `resources.py:37–51` applies A2 smoke elapsed time to A1/A2 and A4 to A3/A4, without verifying full-size settings, full bank training, ladder coverage or source compatibility. A 384-trajectory job receives the same bound as a 96-trajectory job (`resources.py:59–62`).
   - Mandatory-stage enforcement is only in `make_body.sh:6`; `submit.sh` accepts other committed bodies. Failed controls also exit successfully, as above.
   - The archive test check needs the atomic opening reservation from item 9.

   **Fix:** approve an exact job specification and certificate hash; validate measured configuration and coverage; explicitly model second-data-size costs and failed-job allowances per wave. Enforce acceptance in the executing driver.

   The current empty `RELEASE.json` safely blocks submissions, but does not repair these mechanisms.

12. **Real-size memory/runtime and result hazards — NEEDS-FIX**

   Round-1 memory findings are largely unresolved:
   - Whole-file hashing remains in `cell.py:56–57` and `teststage.py:64–66`.
   - Training/POD processing remains eager: `cell.py:273–287,308–320`, `podsine.py:38–47`.
   - All rule spaces are built simultaneously: `ladders.py:44–54`.
   - ECSW retains row arrays, concatenates another matrix, then passes it to device NNLS: `podsine.py:224–248`. Its budget starts after matrix construction.
   - Bank ordering evaluates the full rule at once: `bank.py:155–157`.
   - Failed output caches are released, but successful candidate outputs can accumulate across targets: `timing.py:165–180`.
   - MF-FROZEN now copies only the smallest artifact, improving round 1, but no fixed small-size bound exists.

   Calculated payloads, **not measured peaks**, remain approximately **9.49 GB** for one A4 training coefficient array, **2.41 GB** for 512 POD modes, and **13.29 GB** for the r=384 ECSW matrix before duplicates/workspace. Full-mode M/K checking adds large temporary rule arrays. Host peak memory is not recorded; only device peak is attempted (`cell.py:427–433`).

   **Fix:** stream hashes, bound coefficient/rule processing and caches, construct spaces lazily, preallocate or chunk ECSW construction, and include construction in its budget. Establish host/device memory budgets before the real-size smoke, then measure actual peaks and complete-stage costs. Allocation failures must propagate as missing/resource failures.

**Required fixes before READY-TO-SMOKE:** complete boundary extensions and construction-status propagation; replace the unsafe tensor-cache key; repair coefficient-space diagnostics and timing evidence; authenticate all frozen inputs and the complete campaign freeze; enforce mandatory controls and atomic test-once opening; handle empty test eligibility; and provide bounded real-size allocations plus an exact, auditable release specification.