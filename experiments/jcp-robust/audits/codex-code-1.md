Read-only audit completed. No files changed, scratch created, jobs submitted, or cluster/SSH commands run. Checks used CPU-only Python with bytecode writes disabled. E5 was excluded.

Paths below are relative to `experiments/jcp-robust/`.

1. **WRONG — BLOCKING: controls and integrity failures do not govern verdicts.**  
   `e1analyze.py:290–299` computes verdicts independently of controls and reproduction results. `e1analyze3d.py:124–170` does the same. Distance mismatches are merely recorded, and the 2D `gcert_ok` computed at `r2acc.py:251–264` is never enforced by analysis. GAL rows also receive H-E1 verdicts despite being diagnostic-only.  
   **Fix:** Implement A1.4/A2.4 precedence explicitly, including dimension-wide C-E1.1 propagation, required positive gates, certificate agreement, and distance disagreements. Suppress inferential verdicts for GAL diagnostics.

2. **WRONG — BLOCKING: incomplete populations can receive definitive passing labels.**  
   `e1stats.py:23–33` accepts any nonempty draw list; `NDRAW=16` is unused. Both analyses derive populations from supplied records rather than validating registered expectations. `draw_qualified()` uses `all()` on validity lists without checking their lengths (`e1stats.py:115–128`).  
   **CPU counterexamples:** three passing draws return **robust**; empty validity lists combined with nonempty passing distance lists return `(True, True, 'ok')`. The generated three-draw smoke can therefore receive an unqualified robustness label.  
   **Fix:** Validate exact draw IDs, unique case IDs, expected case counts, certificate shapes, and anchor subsets. Label smoke statistics explicitly; missing evidence must yield unavailable.

3. **WRONG — BLOCKING: failure-state propagation is incomplete.**  
   `e1analyze.py:126–132,157–173` and `e1analyze3d.py:99–128` retain finite \(D_s\) for invalid production runs. They also use target-dependent ranks and indicator tables whenever subset certification passes, even if another target case failed.  
   **CPU counterexample:** an invalid draw zero with finite coefficients ranked “favourable” for \(D_s\), with `rollout_ok=True`; marking its target invalid did not withdraw that indicator result. Zero eligible cases instead crash at `np.max([])`.  
   **Fix:** Implement A3.2/A4.2 case states before aggregation: failed production receives the registered failed rank; failed targets withdraw dependent quantities; zero eligibility returns unavailable. Promote the explicitly labelled own-family H-E1a fallback when target certification is unavailable.

4. **WRONG — BLOCKING: non-finite serialized evidence can crash failure analysis.**  
   The trusted JSON cleaner converts non-finite numbers to `None`. Analysis nevertheless calls `np.isfinite(job)` (`e1analyze.py:31–34`) and `max()` over raw JSON indicator lists (`e1analyze3d.py:97–98`). These operations precede or bypass appropriate failure handling.  
   **CPU check:** a `None` tight-distance entry raises `TypeError` in `draw_qualified()`.  
   **Fix:** Decode missing/non-finite evidence into explicit states first. Preserve recorded numerical failures as failures; distinguish them from absent evidence without performing arithmetic on either.

5. **WRONG — BLOCKING: 3D Q1 distances are not independently recomputed.**  
   `e1analyze3d.py:87–98` recomputes target and successor distances but directly trusts `dist_adt` and `adt_tight_tighter`. Its docstring promises otherwise. Stationarity also comes from saved summary counts (`:92`) rather than recomputing the pooled predicate from the saved fresh certificates.  
   **Fix:** Resolve tight/tighter coefficient keys and case mappings from `W_e1.npz`, independently decode both distances, compare against job values, and rebuild stationarity counts from `gam_cert` rows. Disagreement must block acceptance.

6. **WRONG — BLOCKING: the default 2D Sobol diagnostic crashes analysis.**  
   `make_configs.py:39–41` creates a target with `check=None`. Nevertheless, `r2acc.py:184,207–210` saves nonempty `check_case_ids` with an empty `W_check`. `e1analyze.py:58–60` then indexes that empty array.  
   **Fix:** Store empty check IDs when no check runs. Either reuse a matching certified target explicitly or represent target certification as unavailable for that diagnostic.

7. **WRONG — BLOCKING: 3D C-E1.2 does not exercise identity rejection.**  
   `e1run3d.py:141–149` checks that draw zero matches deployed points and draw one differs by a shift. It never feeds draw-one table identity under a draw-zero claim and verifies rejection. `e1analyze3d.py:137–138` treats these checks only as G-E1.1.  
   **Fix:** Add an actual table-identity validator and invoke the deliberate mismatch through that validator in CPU and in-job controls. Require C-E1.2 to fire.

8. **WRONG — BLOCKING: retrained 3D banks leak into incumbent offline decoding.**  
   `e2run3d.py:17–26` registers new banks for execution, while generated settings retain names such as `W256` (`make_configs_e2_3d.py:39–40`). The copied scorer maps those names to incumbent `W1024` in `score3.py:32,87–95`; its cache is keyed only by setting. No lane adapter changes that mapping or clears the cache.  
   **Fix:** Configure offline decoding from the run’s bank registry and setting metadata, verify checkpoint hashes, and key caches by bank identity plus width. Apply this to every scoring and distance-analysis entry point.

9. **WRONG — BLOCKING: generated 3D front timing lacks required in-job evidence.**  
   `make_configs_e2_3d.py:80–83` generates a fresh job containing only a `panel` block. The trusted harness requires earlier certified rollouts in the same context: `d3run.py:480–483` rejects missing `Wstore`/`wall` entries. Selection-job evidence is neither loaded nor sufficient for this requirement.  
   **CPU check:** the fresh-context evidence linker returns `identity=False, valid=False`.  
   **Fix:** Prepend certified production runs with internal-state recording for the anchor and every timed subject; supply the required qualification evidence before timing.

10. **WRONG — BLOCKING: E2 selection conflates unavailable evidence with “none within ladder.”**  
    At `select2d.py:41–45`, `None` qualification values are simply false in candidate selection. If target certification passes on dev6 but a target fails elsewhere, every candidate can become unavailable and the selector reports the definitive outcome “none within ladder.” Recorded distance disagreements (`:39`) do not affect selection.  
    **Fix:** Use three-state predicates. Report “none” only when complete evidence definitively rejects every candidate; otherwise report unavailable or the registered target-unverified diagnostic outcome. Enforce analysis disagreements before selecting.

11. **NEEDS-FIX — BLOCKING: reproduction verification is not complete, especially for E2.**  
    The 2D archive routing follows A2.1, but P1 hashes are only recorded, not checked against an owning-lane pin (`e1analyze.py:186,218–222`). The 3D archive hash is likewise recorded only after use (`e1analyze3d.py:141–155`). Both compare the cases present in the new output rather than enforcing the registered population.  
    Moreover, the existing reproduction entry point is hard-coded to R128/draw-zero E1 files; E2 candidate-labelled outputs do not implement the required G-E2.3 replay gate.  
    **Fix:** Pin archive files and hashes, validate expected shapes/populations and finite values, resolve 3D archive keys from saved metadata, and implement the enumerated incumbent E2 replay checks with dimension-wide blocking.

12. **NEEDS-FIX — BLOCKING: staging does not deliver A4.2’s execution inventory and resource certificate.**  
    `cluster/stage.py:60–66` checks only a production projection and configuration hashes. It neither generates nor stages the unique-execution inventory; `checks/` is excluded at `:33–38`, so the certificate itself is absent unless separately added. The 2D dry-run output (`r2acc.py:117–133`) is aggregate counts, omits control replays, and does not identify reuse.  
    **Fix:** Generate and stage unique execution keys covering production, tight/tighter, successors, targets/checks and controls. Bind the resource certificate to that inventory, code/configuration identities, GPU count and rounded time limit.

13. **WRONG — BLOCKING: training outputs cannot satisfy collection cleanup.**  
    Training scripts emit `SEED-COMPLETE` (`train2d/run_seed.sh:184–187`, `train3d/run_seed.sh:105–108`), whereas `cluster/collect.py:29–30` requires `COMPLETE` in every output step directory. A successful training job therefore cannot complete the prescribed cleanup flow.  
    **Fix:** Define completion contracts per staged step in `JOBSPEC.json`, or emit a common verified completion marker from training wrappers. Check expected steps rather than discovered directories alone.

14. **WRONG — MINOR: cleanup fails when no output exceeds the archive threshold.**  
    `cluster/collect.py:34–36` always runs `sha256sum -c LARGE.sha256`. An empty large-file manifest exits 1; this was reproduced without writing files.  
    **Fix:** If the verified large-file list is empty, skip that checksum invocation. Continue verifying all local outputs.

15. **CORRECT — MINOR: the principal per-draw execution structure is present.**  
    `r2acc.py:243–290` performs fresh certification, tight solves for every case, tighter subset solves, successor rollouts, and own-state successor/target indicators. `e1run3d.py:54–85,110–128` implements the corresponding 3D structure. The generated E1 configurations use draw-matched successors.  
    **Fix:** None to that structure; repair the acceptance and failure handling above.

16. **CORRECT — MINOR: draw generators and deployed draw identity pass the inspected checks.**  
    `rules2d.py:50–55` delegates to the vendored Fibonacci generator; `rules3d_names.py:58–63` recovers the vector, asserts exact seed-zero regeneration, then calls the vendored generator with the requested seed.  
    CPU checks confirmed all sixteen 2D draws for both sizes match generator calls, and seed-zero identity holds for 3D sizes 8192, 16384 and 32768.  
    **Fix:** None.

17. **CORRECT — MINOR: fixture establishment uses independent NumPy evaluation and matching 3D \(M\).**  
    `establish_e1.py:56–75,84–113` uses independent NumPy bank/derivative evaluation and finite, positive-norm admission. The fixed panels assert six states. The saved establishment contains six admitted states per dimension. CPU evaluation confirmed both establishment and job shell completion give **\(M=1027\)**. `e1stats.py:137–147` checks relative agreement within 1%.  
    **Fix:** Wire the resulting control into blocking precedence. Also make `admit()` itself reject empty inputs instead of relying on caller shape assertions.

18. **CORRECT — MINOR: the 2D bank substitution is complete in the inspected paths.**  
    `p1common.py:153–168` passes checkpoint and rotation explicitly to `Q.Model`; operators, cold start and off-mesh tables derive from that model. `e2bank.py:20–28` verifies both hashes and clears the independent NumPy parameter, rotation and decode caches.  
    **Fix:** None.

19. **CORRECT — MINOR: committed training recipes preserve the registered changes.**  
    CPU hashing found no mismatches in the 17 train2d and 10 train3d provenance entries. All three production 3D configurations differ from the base only in the two registered seeds. The 2D wrapper preserves extraction seed zero, changes training/head seeds, uses the emitted head checkpoint for rotation, and records time-cap deviations (`train2d/run_seed.sh:111–150`). The 3D wrapper retains validation-selected checkpoint selection and records the selected step.  
    **Fix:** None to the inspected recipe contents; fix their completion-marker integration.

20. **CORRECT — MINOR: the main submission safeguards are present.**  
    `cluster/stage.py:70–89,127–134` stages committed blobs, verifies the manifest and cluster-input hashes, and preflights each GPU. `cluster/submit.sh:13–46` serializes lane submission, checks lane/account counts, uses the campaign flock and checks the queue afterward. Reservation is GPUs × rounded time limit; `ledger.py:48–68,103–118` implements the 75 GPU-hour check and GPU-count charging. Collection verifies copied large outputs before deletion.  
    **Fix:** Preserve these safeguards while adding the inventory/certificate and completion fixes above.

**Overall verdict: NEEDS-FIX.**