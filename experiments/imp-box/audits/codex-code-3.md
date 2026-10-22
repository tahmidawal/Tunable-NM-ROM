**Overall verdict: NEEDS-FIX.** Most round-2 findings are fixed or explicitly superseded by A7, but selection completeness, tensor caching, timing completeness, and diagnostic evidence still have defects.

Audited HEAD `eaefda162`. No files changed, jobs submitted, or lab-log writes. Checks used tiny CPU arrays with the specified interpreter. Two untracked smoke bodies appeared externally during review; they are not part of the audited commit. Paths below are relative to `experiments/imp-box/`.

1. **Problem, cohorts, IC and lattice — CORRECT**

   Descriptor mapping, Gaussian IC, sine normalization, x-major synthesis, bank coordinates, and the full scoring lattice agree: `geom.py:35–68,98–109`, `spec.py:38–80`, `podsine.py:63–88,163–192`, `bank.py:40–46`.

   Disjointness passed with 384 training rows. All descriptor splits are hashed (`cell.py:193–199`); test solutions remain unopened until `teststage.py:152`.

   **Required fix:** none.

2. **Spectral FOM, adaptive controller and certificates — CORRECT**

   Fixed ETDRK4 and independent transcription agree (`spec.py:93–117`). Adaptive integer ticks, aligned coarsening, coefficient levels through J+1, charged rejected trials, immediate rejection-at-cap termination, and unfinished-output invalidation implement the governing amendments (`spec.py:178–260`).

   Tiny CPU checks returned:
   - Fixed: coefficient discrepancy **1.70e-15**, update/stage defect **4.62e-17**.
   - Adaptive: **1.20e-13**, **5.63e-16**.
   - Forced rejection, J=3 and j₀=2: **two rejects, one cap hit**, invalid output.

   The reduced NaN hook uses the global step index.

   **Required fix:** none.

3. **References and gates — CORRECT**

   Coefficient/lattice maximum metrics, the two-condition round-off branch, retry, conditional ε/U, padding population and training gates are implemented in `refgate.py:20–60` and `cell.py:112–170,277–284`.

   Round-2’s retry-control defect is resolved **under A7.3**: controls choose a case accepted at its first-pass resolution, and explicitly fail when none is available (`cell.py:224–254`). This is a justified change of control population, not evidence that every retried reference was tested by the control.

   Unscorable selection splits block selection; unscorable test splits now return without evaluating configurations.

   **Required fix:** none.

4. **POD, ECSW and TRANS mathematics — CORRECT**

   Parseval scaling, reorthonormalization, exact derivatives/stiffness, all-mode rule checks and projection are consistent (`podsine.py:35–157`, `cell.py:327–350`). ECSW uses the specified training rows, snapshots and candidate seed (`ladders.py:131–145`, `podsine.py:217–262`).

   TRANS centres the descriptor and decodes at x−δ with δ=c−0.5 (`podsine.py:180–192`).

   G-OPS passed on CPU: point/tensor discrepancies were below **1.70e-14**. All files marked unmodified in `PROVENANCE.json` matched their hashes.

   **Required fix:** none to these mathematical implementations.

5. **Ladders, construction coverage and extensions — NEEDS-FIX**

   The tolerance-extension closure is repaired (`timing.py:249–262`). Flop screening, pruning and persisted termination reasons implement A5/A7.

   **Blocking defects remain:**
   - **Two finalists bypass construction-failure propagation.** `timing.py:195–197` returns `finalist` before reaching the `nfail` handling at lines 198–199. A CPU fixture with one construction failure and two qualified rows returned **`finalist`, construction_failures=1**. A7.1 requires `finalist-partial`.
   - **Screening failures become censoring.** `ladder.py:108–111` swallows timing exceptions and stores `screen=None`; `timing.py:151–164` removes those rows and can return `arm-censored`. A valid, accurate row with a failed screen reproduced that outcome.

   **Fix:** apply incomplete-construction status at every selector return. Record screen failures explicitly and propagate them as incomplete/missing evidence, rather than scientific censoring.

6. **Qualification, selection, timing and freeze — NEEDS-FIX**

   Certified LSPG anchors, ETDRK4 replay certificates, successor/parent Q2, TRANS states, paired-output checks and compile checks remain correctly connected. T-dbl repetition arrays are now retained (`timing.py:278–289`).

   **Blocking cache defect:** [`rom.py:32`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/rom.py:32) fingerprints only strided samples and array sums. Different tensors can share the key used at lines 171–175. A four-point CPU fixture produced **identical fingerprints but tensors 30 and 25**. The replacement removes the object-ID lifetime hazard but is not a reliable content key.

   **Fix:** hash all relevant array bytes, preferably once for immutable space data, or use retained object identity with enforced immutability/invalidation.

   **Blocking timing-completeness defect:** `finalize.py:245–258` discards finalists with invalid timing and returns S from the remainder. Selection remains complete, allowing an incomplete maximum to support a stop decision; see item 10.

   Freeze now includes `pod.json`, other specified metadata, descriptor hashes and the A7 query-source set (`timing.py:373–408`). That round-2 finding is fixed.

7. **Diagnostics — NEEDS-FIX**

   Round-2 repairs verified:
   - Common-size coefficient padding preserves candidate high modes (`refgate.py:24–25`). The previous counterexample now returns **1.0**.
   - Query crossover re-screens the three cheapest rows, as newly specified by A7.6 (`diag.py:219–226`).
   - Component calls now save timing arrays.
   - Evaluation crossover, floor thresholds and finer-reference reconstruction follow A3.3/A5.4.

   **Remaining evidence defects:**
   - **ECSW timing arrays overwrite each other.** `diag.py:74` omits κ from the component label; `diag.py:39` assigns into that label. Finalists differing only in κ retain separate reported components but only the last raw repetition arrays.
   - **Claim-level qualification labels remain absent.** Diagnostic outputs (`diag.py:239–256,261–314`) and stop entries (`stoprule.py:142–144`) do not carry the required eligible-population counts and training-data qualification. Test entries also omit these labels.

   **Fix:** use the complete configuration key for timing evidence. Attach eligible/total counts, training qualification and applicable truncation/deferral labels to each affected readout.

8. **Must-fail controls — CORRECT, with hardening suggestions**

   Round-2’s main gaps are closed: ECSW is always required, ENV is required for bank cells, failed controls exit nonzero, and test preflight requires accepted controls (`controls.py:196–221`, `teststage.py:142–144`). A6 explicitly permits a failing clean ECSW Q2 result.

   Controls exercise production reference gates, tensor/point paths, certificate faults, rule-weight faults, shifted decoding, selection and frozen-artifact checking.

   **Hardening:** the exclusion fixture always sets row 0 false (`controls.py:182–189`). If that row is already excluded, it injects no new exclusion. Choose a currently eligible row and assert the count decreases by one.

   **Required blocking fix:** none beyond the selection/timing fixtures needed for items 5–6.

9. **Test stage — CORRECT for the repaired contract**

   Round-2 fixes are present:
   - Every listed retained cell must have an authenticated committed freeze (`teststage.py:32–45`).
   - Frozen artifacts, descriptors and query-source hashes are verified (`teststage.py:50–74,133–144`).
   - Empty/unscorable populations return a persisted unscorable result (`teststage.py:156–159`).
   - The standard test body atomically reserves opening before generating references (`cluster/bodies/test.sh.in:3–6`).
   - Only frozen finalists are evaluated; only the frozen validation pair can receive test speed-up, after feasibility, Q1/Q2 and timing checks (`teststage.py:167–209`).

   **Required fix:** carry the claim labels from item 7. Streaming artifact verification remains a memory-hardening suggestion: `teststage.py:73` still reads whole files.

10. **Stop rules and final.json mapping — NEEDS-FIX**

   Existing boundary, non-vacuity, missing-FOM and partial-search fixtures passed. The registered predicates themselves remain sound.

   **New blocking mapping defect:** [`stoprule.py:68`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/stoprule.py:68) ignores `final['S'][key]['excluded']`.

   Reproduction through production functions:
   - Two reduced finalists; one valid timing ratio **0.3**, the other timing invalid.
   - `maxmin` returns **S=0.3**, with the second finalist listed as excluded.
   - `entry_status` returns **match**.
   - `speed_stop` can return **FIRES**.

   The missing finalist could have S≥0.5.

   **Fix:** treat timing-excluded finalists as incomplete evidence. At minimum, apply the same one-sided rule as partial reduced searches: S<0.5 is missing; a valid S≥0.5 may establish NOT FIRED. Also repair item 5’s upstream statuses and add regression fixtures.

11. **Cluster scripts and resource release — NEEDS-FIX**

   H200 request/check, absolute venv, highest precision, GPU preflight, per-directory execution lock, campaign queue checks, terminal-state collection and checksum verification are present. Release now binds body SHA, GPU, memory and time (`cluster/submit.sh:19–28`).

   **Blocking collection defect:** [`cluster/collect.sh:23`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/cluster/collect.sh:23) excludes `SHA256SUMS` when generating the archive listing, but includes it in the source listing. Test restoration copies that file into `out/cell`; therefore test collection compares unequal file sets and fails.

   **Fix:** exclude `SHA256SUMS` consistently on both sides.

   **Before production release:** `resources.py:48–51` checks cohort counts and training steps but not N_ref, actual ladder coverage or source compatibility. A reduced-reference run can satisfy its “full_size” predicate. Validate recorded reference resolution, completed coverage and compatible source identity before using smoke costs to certify production. Release entries should also identify their supporting certificate/source.

   The empty committed `RELEASE.json` currently blocks submissions. These production-certificate gaps do **not** require a completed memory pilot before authorizing that pilot.

12. **Real-size memory/runtime hazards — CORRECT as a pilot plan; peaks unverified**

   A5.8/A7.8 explicitly make the real-size H200 smoke the memory pilot. Lack of measured peaks is therefore **not independently a smoke blocker**.

   Improvements include streamed cell hashes, preallocated ECSW matrices, construction time deducted from the fit budget, and recorded host/device peaks (`cell.py:56–61,452–460`, `podsine.py:220–250`).

   Remaining pilot risks are eager training/POD arrays, simultaneous rule spaces, whole-rule bank ordering, successful-output caches and whole-file test hashing. The stated payload sizes are not demonstrated peak bounds. ECSW construction can also exceed its budget before the NNLS deadline is consulted.

   **Hardening:** bound these allocations and deadline overshoot where practical; preserve failure/peak evidence. Allocation failures must remain missing/resource outcomes—item 5 currently prevents that guarantee.

**Required fixes before READY-TO-SMOKE:** repair construction/screen-failure status propagation; replace the sampled tensor-cache key; propagate incomplete timing into stop-rule mapping; prevent ECSW timing-evidence overwrites and attach required claim labels; fix collection’s checksum-file mismatch.

**Overall verdict: NEEDS-FIX.**