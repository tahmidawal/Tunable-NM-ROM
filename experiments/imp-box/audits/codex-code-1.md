**Overall verdict: NEEDS-FIX.** The tiny smoke does not establish readiness for either full-size H200 smoke. Several failures can currently produce selectable or reportable results.

This audit was read-only: no files changed, no jobs submitted, no lab-log writes, no large arrays copied. CPU probes used the specified interpreter with `JAX_PLATFORMS=cpu`. The checkout changed externally during review: `stoprule.py`, `make_body.sh`, and new test/resource scripts appeared; findings below include their last observed versions. HEAD was `5a1907c3828dcfec8744faa7c544ea93991601fe`.

Paths below are relative to `experiments/imp-box/`.

1. **Problem, cohorts, IC and lattice — WRONG**

   The descriptor mapping, Gaussian IC, sine normalization and x-major ordering agree. A tiny asymmetric-mode probe found exactly zero difference between `spec.synth_fn` and `podsine.eval_modes`. Default split disjointness passes.

   **The scoring lattice violates §5.** `geom.py:98–102`, `spec.py:77–79`, and `podsine.py:164–176` use only the **255² interior nodes**, instead of 257² including walls. Omitting zero boundaries is harmless for boundary-exact arms, but **TRANS generally has nonzero wall traces**. Its scored error therefore excludes part of its boundary-condition violation.

   Further gaps:
   - `geom.check_disjoint()` checks the default 96 training rows, even when `n_train=384`: `geom.py:60–68`, `cell.py:163`.
   - Test descriptors are constructed before test (`cell.py:64–68`), though test trajectories/references are correctly deferred. Test descriptor hashes are excluded from the manifest (`cell.py:164–166`).

   **Fix:** score all arms on the registered full lattice, including actual TRANS wall values; check disjointness for the actual training population; freeze every split’s descriptor hash and row IDs. Precomputing descriptors is distinguishable from opening test solutions, but document that distinction.

2. **Spectral FOM and adaptive controller — CORRECT, with one fault-hook fix**

   The implemented ETDRK4 stages and independent transcription agree with the registered formulas: `spec.py:92–116`. The adaptive controller correctly implements:
   - integer ticks and aligned coarsening;
   - coefficients through `J+1`, inside the descriptor-only query;
   - charged accepted/rejected trials;
   - immediate termination on rejection at `j=J`;
   - exhaustion and unfinished-output invalidation.

   Evidence: `spec.py:177–259`. These follow A2/A4, superseding A1’s accept-at-cap wording.

   Tiny CPU results:
   - Fixed certificate: coefficient discrepancy `1.70e-15`, update/stage defect `4.62e-17`.
   - Adaptive certificate: `1.20e-13`, defect `5.63e-16`.
   - Forced rejection with `J=3, j0=2`: two rejects, one cap hit, nonfinite unfinished outputs.

   **Fault-hook correction:** reduced `nan_stage` fires at the first step of **each output block**, because its index resets (`rom.py:257–279`), rather than only global step 1. Pass the global step index, as the fixed FOM does. No normal-path ETDRK4 error was found.

3. **References and gates — WRONG**

   The larger-of-coefficient/lattice metrics, round-off branch requiring both differences, retry resolution, padding population and conditional allowance are substantially correct: `refgate.py:20–60`, `cell.py:102–176`. Training gates cover both spatial and temporal differences.

   Blocking problems:
   - **Unscorable splits remain usable.** `cell.py:158` records `scorable=False` after more than two exclusions, but `ladders.py:120–126` and `timing.py:114–119` simply select the remaining cases. `teststage.py:93–125` likewise permits matched results on an unscorable test split.
   - G7 checks only the original reference table on `val[0]` (`cell.py:177–184`), not every table used by reference retries and FOM configurations.
   - Retry replaces lattice references but returns the original coefficient array (`cell.py:149–159`), leaving inconsistent representations.
   - MF-REF-real/MF-PAD-chk compute a perturbation distance and compare a constant; they do **not invoke the gate being tested** (`cell.py:185–204`).

   **Fix:** enforce split scorable status before selection or test claims; explicitly handle empty eligible populations; certify each distinct IC table; retain consistent retry coefficients and provenance; pass perturbed references through the production gate and require clean-pass/fault-reject outcomes.

4. **POD, ECSW and TRANS — NEEDS-FIX**

   Parseval scaling, reorthonormalization, projection, exact derivatives and stiffness formulas are correct by inspection (`podsine.py:35–125`). The tiny POD Gram error was `5.45e-16`. TRANS correctly centres the descriptor and decodes with `delta=c−0.5` and `x−delta` (`podsine.py:180–192`). ECSW’s seed, training-state selection and sampling scheme follow the amendments.

   However:
   - A2.3 requires the Gram criterion over **all retained POD modes**. `cell.py:278–285` truncates rule selection and M/K checks to 384, although up to 512 modes are retained.
   - M/K and G-OPS failures are recorded, never enforced (`cell.py:288–298`).
   - G-OPS does not gate its mass matrix, despite A3.2 explicitly requiring M (`gops.py:68–73`).

   **Fix:** validate the full retained representation; make required M/K/G-OPS failures block release, and include M in G-OPS. The tiny G-OPS probe passed its existing checks, but that does not repair missing acceptance predicates.

5. **Ladders, extensions and pruning — NEEDS-FIX**

   The A5 flop inequalities and hopeless-rung predicate are implemented correctly (`ladders.py:102–110`, `ladder.py:146–151`).

   Coverage is not reliable:
   - FOM extensions inspect the **screen-fastest meeting row before qualification**, not the eventual qualified finalist (`ladders.py:139–162`). A boundary finalist selected after faster candidates fail Q1 can therefore miss its required extension.
   - No complete persisted termination reason distinguishes convergence, cap, consecutive invalid rungs and pruning.
   - `row['pruned']=True` occurs **after** the row was written, so the JSONL loses this flag (`ladder.py:121–127,150`).
   - Missing ECSW fits and build failures can leave no candidate rows, which selection then calls `arm-censored`; they are not necessarily evidence of completed infeasibility.

   **Fix:** perform endpoint checks on actual qualified finalists and repeat selection after extensions; persist termination and skipped/failed coverage explicitly; distinguish incomplete construction from completed accuracy censoring.

6. **Selection, qualification, timing and freeze — WRONG**

   Three-way matching, certified LSPG anchors, ETDRK4 replay checks, centred TRANS Q2 states and ECSW parent qualification are largely implemented correctly.

   Blocking gaps:
   - **Attempt-cap misclassification:** `timing.py:147–149` reports `finalist` when one candidate passed but the 18-attempt cap prevented exhaustion. A CPU fixture with 20 candidates reproduced `finalist, attempts=18`. A3.6 permits a singleton at **exhaustion**, not an undisclosed cap.
   - **Tensor Q2 checks the wrong implementation.** `qual.py:42–44,73–74` computes point-form vectors for both sides even when the selected production path is tensor form. A tensor-construction defect shared by production and its Q1 replay can escape. G4 currently tests only one small POD configuration (`controls.py:38–51`).
   - `T_dbl` results are recorded but do not invalidate timing or prevent freezing (`timing.py:235–258`).
   - Frozen manifests omit code/tree identity, cohort hashes, training qualification and key configuration metadata (`timing.py:261–272`).

   Paired timing does correctly retain raw repetitions, check every timed field output, warm queries and reject cache-growth detections during validation timing.

   **Fix:** retain `search-incomplete` at the cap; evaluate the actual tensor vector in tensor Q2; enforce mandatory timing controls; freeze the complete provenance and qualification contract.

7. **Diagnostics — WRONG**

   The A5.4 comparator uses the correct common validation rescreen statistic, and floor classifications use the correct allowance inequalities (`diag.py:199–212`).

   Problems:
   - `r_times` minimizes over **all** point forms (`diag.py:179–189`), including configurations removed by the A5 flop screen. This can change `r_x` and BANK-STOP using an unselectable form.
   - It projects the IC into the full bank and slices it (`diag.py:174–177`), instead of applying each rank’s actual projection.
   - The promised `r_x_query` diagnostic is absent.
   - FOM continuous coefficient-space errors are absent from `fom_diag` (`diag.py:236–247`).
   - `boundary_defect` is defined but never called.
   - TRANS decode timing closes over one fixed shift (`diag.py:82–84`), allowing shifted-table construction to be constant-folded out of that component measurement.
   - Kernel timing repetitions are discarded by `tmed` (`diag.py:26–34`).

   **Fix:** share the selectable-form predicate with ladders; use rank-specific IC projection; implement missing diagnostics; pass shifts dynamically in decode timing; persist timing arrays and eligibility/training labels.

8. **Must-fail controls — WRONG**

   **`all_fire` can pass while mandatory checks fail.** It includes only dictionaries with a top-level `fires` key (`controls.py:174–176`). It excludes G4, TRANS identity/sign, T-dbl, score fixtures and `MF_STOP['passed']`, plus reference/operator checks stored elsewhere. Clean positive controls are not required by this aggregate.

   Other weaknesses:
   - MF-ECSW supplies `parent_q2=dict(passed=True)` and does not require clean ECSW qualification (`controls.py:103–106`).
   - Q2 weight faults change only the diagnostic `nvec` multiplier; the faulty production rollout is unchanged (`qual.py:67–75,83–90`).
   - MF-TRANS relaxes the registered identity comparison threshold and uses a truncated Gaussian against its analytic field (`controls.py:114–122`), mixing truncation error with decoder correctness.
   - Missing forced test-Q2-failure and partial-eligibility propagation fixtures.
   - `leak_controls` copies **every frozen artifact** merely to change one byte (`teststage.py:59–70`).

   **Fix:** use an explicit mandatory-control registry with required positive and negative outcomes; invoke production acceptance predicates; mutate actual production weights; use an exactly representable shifted field; add the missing fixtures; copy only a small artifact for MF-FROZEN. Failed controls must block acceptance.

9. **Test stage — WRONG**

   Frozen configurations are evaluated without reselection, and test Q1/Q2 are recomputed. The validation-selected pair is correctly used.

   But:
   - **“Committed” is not verified.** `preflight` checks pathname and byte equality only (`teststage.py:24–38`). Two matching uncommitted manifests satisfy it.
   - No campaign-wide proof ensures all retained cells were frozen before the first test split opened.
   - Unscorable test splits can still yield matched claims.
   - `S_test` is published even if T-id fails or compilation occurs; those values are merely attached (`teststage.py:123–134`).
   - Raw test timing arrays and detailed test qualification evidence are not persisted in `test.json`.

   **Fix:** verify manifests against authenticated committed/tree content and a campaign freeze index; enforce scorable status; apply the validation timing validity predicate before publishing `S_test`; save raw timing and qualification records.

10. **Stop rules and status mapping — WRONG**

   The core speed/bank predicates and A4 boundary fixtures pass.

   **Incomplete FOM searches are lost:** if fixed FOM has a finalist but adaptive FOM is `search-incomplete`, `entry_status` still returns `match` (`stoprule.py:48–62`). A CPU fixture reproduced `('match', 0.3)`. The incomplete competitor search must remain missing.

   Also:
   - `arms_infeasible_everywhere` includes arms whose entries are missing, not necessarily infeasible (`stoprule.py:81–85`).
   - Wildcard bank deferrals accepted by `required()` are not consistently applied to BANK-STOP (`stoprule.py:34,119`).
   - Scorable/reference/control qualification is not checked when ingesting final results.

   **Fix:** propagate incomplete status from either required FOM search; distinguish missing from infeasible; unify deferral handling; consume only accepted, scorable results. The externally edited collection-path correction at lines 108–110 addresses the earlier directory-layout mismatch.

11. **Cluster scripts — NEEDS-FIX**

   Good: GPU partition, absolute venv, highest precision, GPU-backend preflight, per-directory execution lock, committed code archive, locked cap recheck and queue checks are present. Collection verifies archive and local checksums before deletion.

   Missing safeguards:
   - Any GPU string is accepted; H200 is neither required nor verified (`submit.sh:7`, `job.sbatch.tmpl:4,18–19`).
   - No submission-time enforcement of the 87 GPU-hour certificate.
   - Arbitrary stage lists can omit mandatory controls (`make_body.sh:6–7`, `cell.py:350`).
   - Collection does not establish that the job is terminal before copying and deleting its directory (`collect.sh:13–26`).
   - Name/path inputs are interpolated into remote commands and deletion paths without validation.
   - The newly added test body removes `.done_test` unconditionally, permitting repeated test opening (`cluster/bodies/test.sh.in:3`).

   **Fix:** enforce H200 and the budget certificate; validate names and required stages; require terminal job state before collection; make test rerun authorization explicit and recorded.

12. **Real-size memory/runtime and other result hazards — NEEDS-FIX**

   These are calculated allocation sizes, not measured H200 peaks:
   - One A4 training coefficient array: **9.49 GB** (`96×21×767²×8`).
   - A 512-mode A4 POD coefficient array: **2.41 GB**.
   - The r=384 ECSW matrix: **13.29 GB** (`132×384×32768×8`), with row-list and concatenated copies coexisting, followed by a device copy.

   Relevant hazards:
   - Eager training/POD arrays and whole-file hashing: `cell.py:56–57,234–256,273–287`.
   - Simultaneous rule-space construction: `ladders.py:44–54`.
   - ECSW materialization and copying: `podsine.py:224–248`; its 1800-second budget starts **after** matrix construction.
   - Unchunked bank feature evaluation during ordering: `bank.py:155–157`.
   - Repeated tensor construction for each newly built query: `rom.py:157–158`.
   - Unbounded cached official outputs: `timing.py:152–156`; each full validation field output is about 100 MB.
   - MF-FROZEN duplicates multi-gigabyte artifacts.
   - The new `resources.py:33–60` treats smoke elapsed times as bounds without verifying full-size settings, successful gates, H200 identity or code compatibility. Different levels’ adaptive work, rule sizes and retries are not proven monotone.

   **Fix:** stream hashing and coefficient processing, bound caches, construct rule spaces lazily, cache reusable tensors, reduce ECSW duplication and budget its construction. Require measured full-size peak memory/runtime and a validated resource certificate before release. An OOM or incomplete fit must become **missing/resource-failed**, never scientific censoring.

**Required fixes before READY-TO-SMOKE:** restore the full scoring lattice; enforce reference/operator/control gates; repair search coverage and status propagation; check actual tensor paths; authenticate the freeze; reject invalid test timings; complete stop-rule diagnostics; enforce cluster/test-once safeguards; and bound real-size allocations with a credible resource certificate.