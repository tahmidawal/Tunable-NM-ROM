**NOT READY FOR BUDGET SMOKES.**

Read-only audit completed. No files were modified, no project modules were imported, and no GPU jobs were run. Isolated AST-extracted functions and small numerical probes reproduced the failures noted below.

Paths below are relative to `experiments/jcp-combined/`.

1. **BLOCKER — `smk2bud` cannot stage its driver.**  
   `cluster/stage.py:23,69` selects `FILES2` for `driver="c2bud"`, but that list omits `c2bud.py`. The isolated staged import fails; the batch command also lacks its executable.

   **Fix:** include the budget driver in the staged manifest and verify the isolated budget import.

2. **BLOCKER — freezing a selected \(5/3\) timestep produces an invalid schedule.**  
   [freeze.py:36](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/freeze.py:36) reconstructs the step factor from a formatted timestep and rounds it to six decimals.

   **Reproduced:** `0.0083333333 → f=1.666667 → dt=0.008333335`. The horizon mismatch is approximately `5e-8`, exceeding `t2core.sched`’s `1e-12` assertion. It also changes the saved arm key. `smk2bud` uses only integer factors, so it misses this failure.

   **Fix:** preserve the original configuration factor—already available as the selected arm’s `f`—rather than reconstructing it from its display key. Test the \(5/3\) selection-to-budget path.

3. **BLOCKER — the 3D scorer cannot read the budget’s mixed case populations.**  
   [c3run.py:390](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c3run.py:390) saves selected-arm arrays on all cases and diagnostic arrays on the budget subset. `score.py:219` nevertheless indexes every array over the full configured population.

   **Reproduced with the smoke dimensions:** `smk3bud` has three total cases and two budget cases; scoring a diagnostic array raises `IndexError: index 2 is out of bounds`.

   **Fix:** save explicit case IDs alongside every trajectory array and join by ID when scoring, comparing, or normalizing. The current `"first N"` metadata is insufficient for a robust archive contract.

4. **MAJOR — 2D converged-rule acceptance ignores verification and can accept NaNs.**  
   [c2bud.py:143](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c2bud.py:143) reduces distances with Python `max` and never checks the paired rows’ `verified` flags.

   **Reproduced:** `max(0., nan)` returns `0.`, so a non-finite comparison can leave a passing convergence distance. Finite but unverified trajectories can also pass. If both convergence pairs fail, the code retains Gauss640 as `conv`; A1.4 instead requires reporting against Gauss320 with “converged rule unverified.”

   **Fix:** require complete, finite, verified comparisons for every inventoried trajectory type. Implement the prescribed Gauss320 reporting fallback after successor failure.

5. **MAJOR — budget qualification does not consume trajectory validity or enforce its controls.**  
   [budget2d.py:96](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/budget2d.py:96) computes components, mixed differences, `U_star`, context flags, and finalist qualification without checking the participating rows’ verification.

   Thus four failed but identical ideal/replay trajectories give `U_star=0`; failed mixed-difference corners remain reportable; and close failed finalist trajectories can qualify. This violates A1.4, A2.3 and A3.6. Drivers merely record B1 failure and still write `COMPLETE`; the analyzer copies B2/B3 discrimination without withholding dependent conclusions.

   **Fix:** define validity dependencies per quantity. Unverified corners or ideal qualification runs must make the associated result unavailable. Enforce B1 as a stopping gate and propagate non-discriminating B2/B3 controls.

6. **MAJOR — 3D budget mode runs finer rules but never evaluates their qualification.**  
   `c3run.py:387–434` schedules the requisite finer-rule trajectories, including all-case selected arms, but computes neither:
   
   - convergence checks for every `conv` trajectory type;
   - all-case finalist `delta_quad` and finer-rule qualification.

   It then declares completion. No accompanying 3D budget analyzer implements these checks.

   B2 also looks exclusively for a deployed `vendor|BE|0.01|K1` archive key. `analyze3d` freezes the baseline as `fs|BE|0.01|K1`, so a budget built from that selection gets `dctl=None`. The smoke uses `vendor`, masking this integration defect.

   **Fix:** implement and enforce the budget checks, consume explicit case IDs, and make the baseline identity consistent through analysis, freeze, driver and controls.

7. **MAJOR — fallback execution and budget/grid reproduction are missing.**  
   `freeze.py` records fallback lists, but neither budget driver consumes them. Neither provides the first fallback’s accuracy, solver, quadrature and same-job timing qualification required by A2.4. Neither compares budget trajectories with their grid counterparts; the explicit 3D field-reproduction gate of `1e-8` in A1.11 is absent.

   The 3D frozen budget also contains strings, while `bud_specs` requires dictionaries containing `spec` and `selected`. **Direct handoff reproduces a `TypeError`.** No converter or frozen-selection binding is present in these drivers.

   **Fix:** add a validated freeze-to-budget configuration step preserving selected flags, exact arm identities and ordered fallbacks; bind it to the frozen file hash. Implement first-fallback qualification and grid reproduction.

8. **MAJOR — budget reproduction coverage is incomplete; several required diagnostics are absent.**  
   `budget2d.py:132–145` checks only the arm’s `K` dictionary, skips absent rows, and excludes ideal/qualification/ideal-end trajectories and controls. Any positive comparison count with zero failures passes. Changing references also compares new-reference errors with old in-job errors.

   The two implemented B0 analytic fixtures reproduce correctly, but A3.5’s real-data mismatched-normalizer fixture is missing. Section 5/A1.4’s closure ratios, triangle bounds and explicit orthogonal pair at \(t^\ast\) are absent. Context sensitivity is evaluated only on cohort medians, potentially hiding case-specific interactions; unresolved flags cover only four sensitivities.

   **Fix:** derive expected reproduction coverage from the full inventory and authenticate it against the original reference. Add the missing fixture and diagnostics, retaining per-case qualification and resolution status.

9. **MAJOR — H4 adaptive-replay qualification uses the wrong threshold and ignores replay validity.**  
   [analyze3d.py:98](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/analyze3d.py:98) correctly implements the fixed/adaptive distance bar  
   \(\min(2.5\times10^{-4},0.25d_{\rm anchor})\),  
   but tests tighter-replay sensitivity against the constant `2.5e-5`. It checks the tight run’s validity, not the tighter replay’s.

   For anchor discrepancy `1e-6`, the candidate’s solver bar is `2.5e-7`; its replay tolerance should be `2.5e-8`, **1,000 times smaller** than the implemented threshold.

   **Fix:** require both adaptive runs verified and compare replay sensitivity per case against one tenth of that case’s solver bar. Apply A3.4’s unavailable-diagnostic behavior to the comparator probe.

10. **MAJOR — H4 witnesses can be omitted from the frozen budget.**  
    `analyze3d.py:130–146` computes H4a/H4b witness lists, but budgets only the generic fast-equal-accuracy and accurate-equal-cost selections.

    The fast selection drops H4a’s step and `0.6` cost restrictions. The accurate selection deliberately differs from H4b: A2.8 allows any step at ratio `≤1`, while an H4b witness is at `dt0` with ratio `≤1.1`. Consequently, an H4 claim can lack its selected witness’s budget qualification.

    **Fix:** freeze explicit H4a and H4b witness IDs and include them in the budget, alongside the separately reported general selections and diagnostics.

11. **MAJOR — 3D analysis ignores supplied scores and H6 ignores uncertainty ties.**  
    `analyze3d.py:55` authenticates the score file, but `:85` reads errors from the original result rows. Rescoring therefore leaves the reported errors and H6 unchanged while changing the output’s `refs` label.

    H6 at `:154–160` uses raw `<` comparisons and a raw median reduction; it implements neither A1.8/A2.6’s tie policy nor uncertainty-based unavailability.

    **Fix:** consume the authenticated score values, retain frozen selections under rescoring, and implement the registered uncertainty comparison. Do not emit “extends” from unresolved raw differences.

12. **MAJOR — rule-family deployment uses a different cost formula and incomplete evidence can proceed.**  
    [select3d.py:114](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/select3d.py:114) sums median **paired ratios**; A1.7 specifies the sum of the two arms’ median **times**. Different baseline timings across panels can change the winning family.

    Missing `lat256` evidence becomes `None`, which passes the `if ctrl_q` guard. Missing qualifying-family timing likewise permits selecting the other family without establishing the registered comparison. The consumer does not verify the two expected subjects and complete case/repetition coverage.

    **Fix:** use the prescribed `tB` statistic; require an evaluated control and complete expected panels. Treat missing evidence as unavailable.

13. **MAJOR — G-score coverage is still derived from what happens to be archived.**  
    [score.py:110](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/score.py:110) builds expected counts while traversing available arrays, rather than the required run inventory. Removing a trajectory from the archive removes its expected comparisons. A missing FOM archive is silently skipped.

    Both ROM scoring paths accept paired non-finite scores without incrementing failures. Required case-swap controls are also waived when no second case exists.

    **Fix:** compare inventory, result rows and archive keys bidirectionally; require mandatory controls and explicit finite-comparison coverage. Keep invalid-rollout reporting separate from a passing score-agreement gate.

14. **MAJOR — registry authentication remains partial.**  
    `score.py:49–60` now authenticates reference files/manifests, but `times`, `nodes` and `normaliser` are required only to exist; decoding and normalization remain hardcoded. Case descriptors are not checked against the reference cohort contract. The 2D rotation is not pinned, and the run’s claimed checkpoint hash is compared with the registry without verifying the actual decoder checkpoint there.

    **Fix:** validate the supported coordinate/time/normalization contract, authenticate case descriptors and actual bank/rotation assets, and reject unsupported registry changes explicitly.

15. **MAJOR — the rescoring contract still does not work end to end.**  
    `freeze.py` writes the 2D selection under `fz['2d']`; `analyze2d.py:144` instead reads `frz['per_scheme_setting']` and `frz['global']`. Passing the documented frozen file raises `KeyError`.

    New-reference scores are either `unchecked` and rejected by analysis, or checked against old-reference in-job errors. E0 uncertainty is also implemented as `eps=eta/2` followed by scalar threshold comparisons, rather than the required propagated intervals.

    **Reproduced:** errors `0.003` and `0.0015` with `eta=0.001` have overlapping uncertainty intervals, but the implementation declares the larger error beaten.

    **Fix:** consume the actual frozen schema, separate original-reference reproduction certification from authenticated rescoring, and implement endpoint propagation for case, median and worst comparisons.

16. **MINOR — ordinary Pareto fronts retain exact duplicates.**  
    `analyze2d.py:52–55` deduplicates entire `(cost,error,id)` tuples, so identical numerical points with different IDs both survive. A3.1 requires retaining only the lowest arm ID.

    **Fix:** collapse by `(cost,error)` before computing the front.

The requested re-check of `codex-code-2.md` items 13–20 is:

| Item | Status | Assessment |
|---|---|---|
| **13 — common-panel validity** | **RESOLVED** | H2/H5 now become unavailable when the common panel is invalid. |
| **14 — G-score enforcement/coverage** | **NOT RESOLVED** | Explicit status, per-time comparisons and failure exit are fixed; coverage and non-finite/control loopholes remain—finding 13. |
| **15 — registry contract** | **NOT RESOLVED** | Reference checksums improved; identity and metadata enforcement remain incomplete—finding 14. |
| **16 — frozen rescoring/FOM/uncertainty** | **NOT RESOLVED** | FOM rescoring exists, but frozen-file consumption, new-reference gating and interval comparisons remain broken—finding 15. |
| **17 — H1 rules** | **RESOLVED** | TX-only H1(a), reference-limited precedence, and explicit median-tie handling are present. |
| **18 — H2 populations/fronts** | **NOT RESOLVED** | Required named transitions and empty-population protection are fixed; duplicate-front rule remains wrong—finding 16. |
| **19 — H3 anchor qualification** | **RESOLVED** | BE/CN and all anchor replays are checked; uncertainty is evaluated at \(t^\ast\), with the A3.3 guards. |
| **20 — global scheme ties** | **RESOLVED** | Complete schemes alone enter sum comparison; incomplete ties use fixed order; no support returns unavailable. |

The implemented 2D component formulas and four-corner mixed differences are direct measurements, **not hidden remainders**. The principal problems are qualification, missing diagnostics and integration. The 3D fixed comparator population and conservative `3U` substitution are implemented, as are A6.2/A7.1 ladder eligibility and the finer-neighbor escalation path.

**NOT READY FOR BUDGET SMOKES.**