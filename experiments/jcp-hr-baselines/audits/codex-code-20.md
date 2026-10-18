**Overall verdict: WRONG — `07f46488f` is NOT READY for production.** Several round-19 repairs are correct, but the new reporting chain can admit invalid timing evidence, ignore reconstruction failures, select from incomplete results, and lose the batch-1/batch-2 distinction.

All file:line references below refer to `07f46488f`, relative to `experiments/jcp-hr-baselines/`. HEAD advanced during the audit; later amendments and changes are outside this verdict. I made no filesystem changes, used no scratch, submitted no jobs, ran no GPU work, and did not run `tests/test_core.py`.

1. **WRONG — reference eligibility is not consistently respected during selection.**

   `tune.py:372–379` computes screening worst/median errors over **all** screening cases. `freeze_stageb.py:67–76` and the extension trigger at `tune.py:451–453` use those statistics without filtering reference-ineligible case-kinds. This contradicts §6 (`DESIGN.md:187`) and §2.

   Final analysis has a different problem: `analyze.py:64–67` intersects eligibility across **every** member, including unqualified members. Because `score.py:92–97` sets `eligible=False` for an invalid numerical output, a failed baseline can remove an otherwise reference-eligible case from every comparison. Empty eligibility reaches `ST.envelope` without a guard (`analyze.py:81–82`).

   **Fix:** derive one immutable case-kind eligibility mask from pinned reference metadata. Use it for screening, extension, final fronts and labels. Reject invalid candidates without shrinking that mask. Return “not assessed” for empty eligibility or missing exported errors.

2. **WRONG — Stage-B timing validity is recorded but not enforced downstream.**

   `stageb.py:287–289` records `outputs_ok=False` if any timed output disagrees. `analyze.py:85–90` ignores that flag; `rho_of` merely discards individual bad triples (`36–45`). A pair with a failed output check can therefore remain selectable.

   The analyzer also trusts `tr['valid']` without recomputing drift, positive finite times, or repetition/retry completeness. A small CPU probe confirmed that it accepts a triple marked valid despite `tA1=1`, `tA2=10`.

   Moreover, `stageb.py:282–285` compares timed outputs with newly executed warm outputs, rather than binding them to the stored certified/scored Stage-B outputs.

   **Fix:** enforce the complete timing contract, including every output check and repetition inventory. Recompute drift from raw times and compare timing outputs against the saved certified outputs. Reuse the stronger checks in `smoke_accept.py:145–178`.

3. **WRONG — reconstruction replay is informational, not an acceptance gate.**

   The reconstruction formulas themselves match tuning: ECSW uses saved nodes/weights; Q-DEIM reconstructs `Pst` and loads `T1/T2/T3E`; `none` uses the saved basis and operators (`tune.py:341–347`, `stageb.py:199–209`).

   But `stageb.py:229–246` records `replay_vs_tune_W`, marks the member complete, and admits it to timing solely on Q1. Neither Stage B nor `analyze.py:84–96` rejects a replay mismatch.

   **Fix:** establish and enforce a finite replay tolerance before admitting a member to timing or selection. Preserve mismatches as explicit replay failures.

4. **WRONG — the frozen handoff does not fully bind or validate its evidence.**

   `freeze_stageb.py:65–76` trusts stored Q1/timing flags. Its optional verification checks only production-to-tight distances (`34–54`); a returned `ok=False` does not prevent writing the freeze (`97–105`).

   The frozen artifact hashes omit the screening coefficient files subsequently used for replay (`90–102`; `stageb.py:229`). Analysis hashes its supplied tuning JSON but does not require that hash to equal the frozen tuning hash (`analyze.py:51–76`).

   **Fix:** make verification mandatory and blocking; recompute certificates, timing acceptance and both anchor-distance checks. Hash every replay dependency and enforce the frozen tuning hash again in analysis.

5. **WRONG — truncation can produce ordinary picks and an inadequately qualified non-domination statement.**

   Freeze does not carry complete expected/finished/failed/deferred coverage into its membership contract (`freeze_stageb.py:98–102`). Analysis silently skips absent or unusable timings and selects from the survivors (`analyze.py:84–101`).

   Its reported comparison count is `len(cand)`, although some candidates are neither registered picks nor included in the envelope-filtered exploratory comparisons (`113–124`). It does not export the assessed/not-assessed counts required by A3.5.

   **Fix:** validate coverage against the frozen inventory, identify incomplete roles/cells explicitly, preserve each exclusion reason, and count actual assessed pairs. Missing evidence must not support a favorable conclusion for ours.

6. **WRONG — A29 batch identity disappears from production outputs.**

   The input correctly marks batch 1 and batch 2 (`inputs/speed3d_batch1_e0.json:207,1024,1227,1430,1633`). Neither tuning comparator records, frozen X records, nor Stage-B comparator records propagate those labels (`tune.py:160–202`, `freeze_stageb.py:80–88`, `stageb.py:168–188`).

   `analyze.py:69` then marks **every 3D comparison** diagnostic-only. Qualified batch-2 comparisons receive the same treatment as diagnostic batch 1.

   **Fix:** propagate per-X batch, qualification status and qualification provenance through every artifact and verdict line. Batch 1 must remain diagnostic; batch 2 must carry A29’s qualified-subset limitation.

7. **WRONG — saturation, construction-quality and coverage consequences are incomplete.**

   `analyze.py:125–126` withholds non-domination wording only when adequacy fails. It ignores failed saturation, although A2.2 says either failed check triggers withholding. Nothing imports the 256² saturation outcome into the 1024² cell.

   ECSW quality reaches configuration records (`tune.py:341,376`) and final role records (`analyze.py:111`), but coverage failures and quality limitations do not produce the required cell-level restrictions. Required median-baseline and offline-cost information is also absent beside role labels.

   **Fix:** explicitly propagate cross-cell saturation status, ECSW boundary failures and selected-path quality. Generate the required restrictions and accompanying statistics for every affected line.

8. **WRONG — reported C-search expenditure excludes Stage B.**

   The extension’s A28.3 trigger correctly sums classified `A|` timers (`tune.py:448–450`). Final tuning expenditure correctly sums `A|` and `EXT|` (`481–482`).

   However, `analyze.py:68–74` compares only that tuning total against allocation. Stage-B selection-relevant work is never added, contrary to A25.2/A28.3. Thus “baseline search within ours’ allocation” can be false.

   **Fix:** classify and persist Stage-B selection costs, combine them with tuning expenditure, and implement the registered budget wording, including unequal-budget descriptive outcomes.

9. **WRONG — Stage B retains potentially excessive device data across the entire selected population.**

   `stageb.py:211–219` reconstructs data separately for each `(kind,r,hp,scheme,dt)` group. `245–246` retains every qualifying group’s data until all accuracy runs finish and timing begins.

   This duplicates basis prefixes and HR data across options. In the permitted uncapped-front extreme, the six HR choices across twelve options retain approximately **248.9 GB of non-full-rank 3D basis prefixes alone**, excluding stencils, transforms, comparators and workspaces. This is a conditional workload bound, not a claim that the eventual frozen population will attain it. Small CPU inspection confirmed repeated JAX prefix slicing creates distinct buffers.

   **Fix:** share basis prefixes per rank and online HR data per `(kind,r,hp)`, and process/timestamp members in bounded groups. Certify memory against actual frozen membership; the sequential-phase memory assumption does not describe this implementation.

10. **WRONG — per-case evidence preservation remains incomplete.**

    Tuning now saves each completed solver configuration, and comparator/FOM cases are checkpointed. But baseline screening accumulates all cases before saving (`tune.py:358–390`); a later case failure loses earlier successful cases.

    Stage B similarly builds whole-cohort dictionaries before checkpointing (`stageb.py:221–249`), and timing cases are saved only after the pair completes (`278–292`). FOM/probe/comparator inventories are not fully predeclared, so truncation can leave absent entries rather than explicit unfinished states.

    **Fix:** predeclare all expected work and checkpoint each completed case, anchor and timing attempt atomically. Preserve partial results on exceptions and termination.

11. **NEEDS-RESTATEMENT — execution order and defer priority are not a coherent contract through A29.**

    At this revision, the explicit A26.3 execution order is implemented, with the 85% start guard (`tune.py:398–445,461`). However, A28.5 points defer policy back to §12, while A1.14/A2.15 had already replaced that list (`DESIGN.md:505–509,633–635`). The code can defer large-rank `none` runs despite those amendments’ never-dropped requirements.

    Running first-dropped work first also consumes budget ahead of work intended to have greater protection.

    **Fix:** consolidate one binding defer list, distinguish execution order from drop priority, and run optional work in an order consistent with that priority. Validate guards at the intended work-unit boundaries.

12. **CORRECT — the specific round-19 optimum-evidence defects are repaired.**

    `smoke_accept.py:112–131` rejects absent modern terminal evidence and validates positive weights, integer unique in-mesh nodes, support count and residual norm. `135–142` binds optimum checkpoint nodes/weights to the terminal solution. Legacy absence is restricted to the explicitly registered attempt name (`108–109,195–197`).

    Small CPU probes confirmed rejection of missing terminal evidence and node `-42`.

    **Fix:** retain these repairs. Extend mandatory terminal-evidence validation to every modern attempt, including ordinary non-optimum paths, and recompute the combined selected-rank quality label; the current loop only checks individual attempt labels (`329–334`).

13. **CORRECT — saturation replacement, seeded adequacy and probe constructibility are implemented.**

    Saturation failure installs the certified all-step basis, trust radii and constructible ranks before downstream construction (`tune.py:228–254`). Adequacy uses the seeded permutation on the actual installed basis (`256–274`). Probe runs reject uncertified ranks (`492–493`).

    **Fix:** retain the numerical routing and repair the downstream consequences in item 7. Also persist the probe lattice basis: currently only probe coefficients are saved (`285–291,505`), preventing independent reconstruction of its reported E0 results.

14. **CORRECT — the limiting FOM now checks convergence and preserves E0 metadata.**

    `tune.py:422–430` requires every residual to be finite and within tolerance, records failed solves, retains the complete E0 record, and checkpoints each case.

    **Fix:** retain this change; predeclare the complete FOM inventory so never-reached cases remain distinguishable from nonexistent cases.

15. **CORRECT — Stage-A timing agreement is enforced in front admission.**

    Baseline runs record finite agreement checks (`tune.py:360–380`); ours does likewise (`181–202`). Freeze requires `timing_ok` for both sides (`freeze_stageb.py:67,82`).

    **Fix:** retain this behavior and apply the same fail-closed principle to Stage B as described in item 2.

16. **CORRECT — the registered grids and selection/statistical primitives match the amended design.**

    All five committed tuning configurations match the generator. The production base grids contain **140 configurations per 2D cell and 1,008 in 3D**, with the registered options (`make_phase2_configs.py:8–34`, `tune.py:135–140`).

    `stats.py:118–186` implements uncapped fronts, the 24 alternating near-front additions, X-specific front union and the registered role tie-breakers. `stats.py:29–113` implements the case-weighted timing estimand and amended pair decision table. Stage-B anchors are shared by discrete problem and Q1 covers the full cohort (`stageb.py:211–227`).

    **Fix:** retain these primitives; repair their eligibility, completeness and timing inputs before relying on their outputs.

17. **CORRECT — comparator source identity and imported 3D scores verify.**

    Direct `git show` comparisons against speed3d `691465d88` in the specified worktree confirmed byte identity and declared hashes for `s3common.py`, `qn3.py`, and `fs3p.py` (`INHERITED.json:171–190`). The P1 code and exported-input pins also match their source blobs.

    All **nine comparator entries × 64 cases** match their source E0 errors, reference uncertainties and eligibility flags; all three source hashes and job identities match. `OursP1` and `OursS3` route through the intended query implementations (`ours.py:86–116,126–148`).

    **Fix:** retain these mappings and the runtime G-ours gates. Carry the verified batch metadata into outputs.

18. **CORRECT — the missing M2 staging defect is fixed.**

    `cluster/stage.py:30` includes the M2 bank. Tune and Stage-B drivers, speed3d dependencies and configuration-specific `stage_files` are staged (`20–35,47,59`). Tune asset preflight is invoked in the isolated staged tree (`125–130`).

    **Fix:** retain this repair. Extend configuration-aware preflight to Stage B, including its external tuning directory and frozen dependencies.

19. **CORRECT — A26’s arithmetic is a spending bound, not a completion certificate.**

    `resources.py:84–91` computes **28.012 GPU-h** from spent time, the three tuning wall limits and Stage-B reserve, and labels performance projections conditional. This resolves the misleading earlier runtime-certificate claim.

    The bound remains conditional on enforcing those limits: `cluster/stage.py:51,78–86` independently accepts `--minutes`; it does not enforce the configured wall bound or remaining reserve.

    **Fix:** enforce configured wall limits and ledger reservations during staging/submission, including retries. Regenerate Stage-B admission from frozen membership. Do not infer workload completion or memory fit from the spending bound.

20. **CORRECT — no prohibited cohort access was found in these execution paths.**

    Training construction uses training draws; screening uses the registered selection cases; full-cohort evaluation and final selection use the authorized developmental cohorts (`tune.py:126–128,215,413`; `stageb.py:133–143`). `score.py:25,60–74` restricts reference access. Full-cohort picks are authorized by A1.6, rather than leakage.

    **Fix:** preserve those restrictions and explicitly retain the developmental, in-sample, unequal-selection-history qualification.

**Overall: NOT READY at `07f46488f`.** The successful dry runs support nominal execution and numerical replay. They do not establish fail-closed reporting, complete evidence custody, A29 qualification labeling, or production memory feasibility. Those defects require correction independently of the pending source-pinned gate-suite result.