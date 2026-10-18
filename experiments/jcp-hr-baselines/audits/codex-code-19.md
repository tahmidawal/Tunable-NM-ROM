**Overall verdict: WRONG — HEAD `1193c68af` is not ready for phase-II production.** The principal blockers are incomplete staging, missing saturation fallback, loss of completed evidence on exceptions, and an unsupported resource certificate.

Paths below are relative to `experiments/jcp-hr-baselines/`. I did not run the gate suite, GPU work, or jobs, and made no file changes.

1. **CORRECT — terminal NNLS evidence and the stale construction docstring were repaired.**  
   `nnls.py:405–412` returns the terminal support, weights and residual vector. `build.py:344–345,353–378` documents optimum substitution, requires the solver’s KKT certificate, preserves terminal evidence and assigns an orthogonality-quality label. `tune.py:189–200` persists these arrays.

   **Fix:** retain these changes; complete the offline validation below.

2. **WRONG — offline optimum acceptance still permits missing or unbound terminal evidence.**  
   `smoke_accept.py:113–121` now checks `kkt.ok` and terminal-weight feasibility **when weights exist**, but missing weights bypass those checks. `legacy_no_terminal` at `308–309` merely reports absence; it does not restrict the exception to the pinned legacy smk2 run. Terminal nodes are never validated, and an optimum checkpoint is not compared with the terminal solution (`274–282`).

   Read-only, small-array probes confirmed that `opt_ok` accepts both absent terminal evidence and a terminal node of `-42`. The latter is a malformed-evidence robustness finding, not evidence that the solver produced such a node.

   **Fix:** require complete terminal arrays for post-A24 runs; allow legacy absence only for explicitly pinned provenance. Validate terminal-node identity/count and bind optimum checkpoints to the positive terminal weights and corresponding nodes.

3. **WRONG — drift detection is fixed, but quality propagation remains incomplete.**  
   `nnls.py:379–380,411` now treats NaN drift as failure. `pipeline.py:184–195` combines orthogonality and banning limitations into a rank-level label. However, `arm_data` returns ECSW data without that label (`203–208`), and Stage-A configuration records contain only `ainfo` (`tune.py:245`). The offline checker allows missing quality labels (`smoke_accept.py:307`) and does not verify the combined selected-rank label.

   **Fix:** attach construction quality to every ECSW configuration, require it for modern evidence, and make the Stage-B/report consumer enforce A16.5/A24.3 wording restrictions. The underlying rank-level evidence is available.

4. **NEEDS-RESTATEMENT — A24 corrects the append arithmetic but introduces another traffic error.**  
   `DESIGN.md:1393–1399` correctly gives approximately \(2.24\times10^{14}\) append flops and \(9.83\times10^{14}\) refactor flops. But reading a 27.4 GB matrix 19,000 times is approximately **\(5.20\times10^{14}\) bytes**, not \(4.2\times10^{15}\): an eightfold error. `resources.py:39` uses the smaller, correct gradient-traffic expression.

   **Fix:** generate the prose arithmetic from the executable model. Include residual formation, deletions, triangular solves, refactor temporaries and measured throughput before calling the resulting wall time a certificate.

5. **CORRECT — the base grid, configurations and principal Q1 routing match the amended design.**  
   `make_phase2_configs.py:8–17` registers the correct ranks, two 2D options and twelve 3D options. `tune.py:39,207–237` covers seven HR choices and both solvers: **140 configurations per 2D cell; 1,008 in 3D**. Tight native anchors cover all six screening cases; tighter anchors cover cases 0–1. `pipeline.py:278–298` checks cohort completeness and the registered distances.

   I compared all five committed `tune_*.json` files with the generator’s computed dictionaries without executing its writing loop: all match.

   **Fix:** retain this grid and routing; add explicit expected/completed/failed inventory accounting.

6. **WRONG — saturation and adequacy do not implement their registered consequences.**  
   `tune.py:130–133` uses trajectories `0..31`, whereas A2.2 specifies the first 32 of a seeded permutation (`DESIGN.md:536–542`). More seriously, saturation only reports a Boolean (`tune.py:144–158`): failure never replaces the 256² primary basis with the all-step POD, as A1.3 requires (`DESIGN.md:362–365`). HR construction continues using the original basis.

   **Fix:** use and persist the registered trajectory selection. On saturation failure, certify and install the all-step basis, recompute trust/operators and downstream construction, rerun adequacy on the actual basis, and propagate the required limitation to the 1024² cell.

7. **WRONG — the generator probe bypasses basis constructibility.**  
   Its snapshot generator and basic rank/option coverage are correct (`tune.py:160–166,307–319`). But `cp.build_basis` computes certification, then `probe_runs` directly constructs data and runs **every rank**, bypassing `cp.constructible_r`. Failed probe ranks can therefore acquire ordinary-looking E0 results.

   **Fix:** preserve and enforce the probe’s constructible-rank inventory; record failures explicitly. Persist per-case probe fields or verified scoring evidence and report central/probe projection errors alongside E0 errors.

8. **WRONG — the staged 3D job is missing the M2 bank.**  
   `cluster/stage.py:20–35` includes W1024 but omits `vendor/quad3d/inputs/model_M2/bank.pkl`. `c3common.py:38–39,95–98` requires that file. The first configured 3D comparator is M2 (`make_phase2_configs.py:12`), so `ours_pass` fails during construction and loses **all five** comparators.

   The import-isolation check (`cluster/stage.py:119–123`) cannot catch this: importing the module does not load the bank.

   **Fix:** stage and hash the M2 bank; validate every configuration-dependent asset before submission. Extend isolation checks to lazy imports/assets exercised by the selected driver.

9. **CORRECT — the comparator identities and requested provenance checks match.**  
   `ours.py:73,86–116` reproduces P1’s bank edges, rank-specific rules, frozen/nominal policy, LSPG production schedule, IC and lattice restriction. These agree with P1’s `acc256.json` and `acc1024.json` at `5b782026b`.

   `ours.py:48–68,71–72` uses the correct 3D bank, rank, adaptive-start parameter, trust fraction and mesh-path FS query. Actual \(10^{-10}\) replay remains an execution gate, not something established by this static audit.

   All four P1 entries at `INHERITED.json:132–160` match both their declared hashes and source blobs. All **30** extracted 3D errors, job identities and three source hashes in `inputs/speed3d_batch1_e0.json:1–70` match `git show` of the specified speed3d commit.

   **Fix:** retain these mappings and provenance; obtain G-ours after repairing staging.

10. **WRONG — successful work can disappear when a later operation fails.**  
    `Out.phase` checkpoints completed phases (`tune.py:64–79`), but `cell_block` holds both solver summaries locally until both finish (`224–258`). A common-solver failure discards an already completed native summary. Comparator, probe and FOM results similarly remain local until their entire passes finish (`262–304,308–319,322–336`). Basis/ref failures return without enumerating the remaining configurations.

    **Fix:** checkpoint each completed configuration/case, isolate comparator construction failures individually, and maintain a predeclared inventory with explicit `not_run`, `construction_failed`, `run_failed` and `complete` states. Persist partial phase results before continuing.

11. **WRONG — timing/scoring agreement is recorded but not enforced.**  
    The baseline has a warm call and burn, but `tune.py:236` only records a maximum discrepancy; neither Q1 nor the Stage-A record rejects a mismatch or nonfinite timed output. Ours scores the timed output, which is good, but does not compare it with the preceding invocation (`285–296`). This falls short of A6.6/A12.7.

    **Fix:** enforce finite outputs and the \(10^{-12}\) agreement check before admitting screening times. Preserve failed timing records without allowing them into absolute-time or X-specific fronts.

12. **WRONG — the full-rank limiting comparator can report an unconverged FOM solution.**  
    Cohort and option coverage are correct: 38 cases in 2D and 64 in 3D (`tune.py:326–334`). However, unlike snapshot generation (`build.py:59–63`), this code never checks `o['res']`. It stores only the scalar error, discarding convergence and eligibility evidence.

    **Fix:** validate every FOM step’s finite residual against tolerance, persist convergence/status and full E0 metadata, and checkpoint each case. Failed solves must remain failed comparator entries.

13. **CORRECT — no prohibited cohort access was found in the tuning execution path.**  
    Training construction uses training draws; Stage A and G-ours use the six screening cases (`tune.py:106–118,224–237,282–296`). The separate FOM pass uses the authorized verdict cohort. `score.py:25,60–74` restricts reference loading. No sealed/OOD/test fields are requested.

    **Fix:** keep these restrictions in Stage B. Final picks legitimately use the full developmental cohort under A1.6; describing those picks as selection-subset-only would be incorrect.

14. **NEEDS-RESTATEMENT — H200/200 GB fit is plausible, but not demonstrated.**  
    smk2 measured **107.18 GB** device peak at the 2D maximal Q-DEIM pilot (`runs/smk2/archive/output/smoke.json:149–152`). Its snapshots were reduced; production uses 576/1,536 trajectories. Production snapshot storage is approximately **57.9 GB / 36.9 GB**; retaining central snapshots while constructing the probe doubles that portion (`tune.py:122,160–171`). These arrays alone fit 200 GB, but they are not a complete peak inventory.

    The maximal 3D NNLS nominal buffers total approximately 59 GB before workspace and other resident arrays (`DESIGN.md:1348–1352`). The new path’s peak is checked only after execution (`tune.py:177–187`); it cannot prevent an OOM. smk2’s 3D device peak is masked by the preceding 2D peak.

    **Fix:** regenerate live host/device inventories for full snapshot counts, fallback, saturation, all retained comparators and maximal refactoring. Measure the missing peak in an isolated admission workload. Neither a guaranteed fit nor a guaranteed OOM follows from the existing evidence.

15. **WRONG — the current runtime certificate does not establish either budget.**  
    `resources.py:53–55` reuses reduced-smoke basis costs for full production populations. `61` assigns identical HR cost at every rank despite growing residual/Jacobian and dense-factorization dimensions. `77–79` scales the entire 256² job by one quarter, including costs that do not scale with mesh size. Stage B is simply half the tuning estimate (`83`).

    Its current estimates are:

    | Tuning job | Model estimate | With its 1.5 contingency |
    |---|---:|---:|
    | 2D 256² | 0.53 h | 0.94 h |
    | 2D 1024² | 2.10 h | 3.15 h |
    | 3D 65³ | 6.54 h | 9.81 h |

    These use smk2 grid times of 70.51 s and 54.90 s (`smoke.json:179–182,31248–31251`). As a sensitivity calculation, changing **only** HR rank scaling to quadratic raises contingency estimates to **5.34 h / 22.26 h** for 1024²/65³; cubic scaling gives **10.47 h / 58.04 h**. These are conditional projections, not measured predictions, but demonstrate that “within cap” is not robust.

    **24 h/job:** plausible under favorable scaling, unproven; the 3D sensitivity exceeds it.  
    **30 GPU-h lane:** unproven. The quadratic sensitivity alone gives approximately **30.05 GPU-h including 1.512 spent and the existing 256² estimate**, before Stage B or additional construction corrections.

    **Fix:** fit rank/support-dependent costs from measured workloads, include full snapshot and retry costs, and derive Stage-B cost from its frozen inventory. Do not label the present 22.35 GPU-h campaign projection “certified.”

16. **NEEDS-RESTATEMENT — Stage B needs an explicit, validated handoff contract.**  
    `tune.py:126–127,188–220,239–252` saves useful bases, HR data, fields, trajectories and certificates; rank-level ECSW quality is also available. However, completion is not validated, the advertised artifact hash list is supplied only by the batch wrapper, and extension screening is not implemented.

    **Fix:** before freezing Stage B, require:
    - Complete base-grid/failure inventory and any triggered A2.18/A16.2 extensions.
    - Verified Q1, eligible-case E0 statistics, and per-case timings for absolute and X-specific fronts.
    - G-ours, basis adequacy/saturation, ECSW quality/coverage and comparator qualification labels.
    - Checksummed reconstruction artifacts and a frozen membership file.
    - Full-cohort Q1, repeated G-ours and registered A–B–A evidence in Stage B.
    - Explicit **diagnostic-only** treatment of batch-1 3D comparators under A22.0.

**Overall: NOT READY.** Repair the execution and evidence defects, regenerate resource estimates, and await the source-pinned gate-suite result before production admission.