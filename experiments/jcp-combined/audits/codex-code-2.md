**NOT READY.** Several original defects are fixed, but resource certification, scoring gates, and hypothesis evaluation still admit invalid conclusions.

No files were modified, including the lab log. I reran the complete manufactured test suite with writes intercepted in memory: it passes and reproduces the certificate. I also ran small CPU adversarial probes; no GPU workloads were run.

References below are relative to `experiments/jcp-combined/`. Items 1–12 correspond to the previous audit.

1. **BLOCKER — RESOLVED: adaptive 3D LSPG crash.**  
   `c3common.py:204–206` now prepares both forms, including `hmask`; `c3common.py:200` passes the appropriate prepared dictionary.  
   **Fix applied:** shared preparation before queries. No remaining fix for the original crash.

2. **MAJOR — RESOLVED: non-stationary trajectories accepting escalation.**  
   `c3common.py:239–244` computes pooled numerators/denominators and eligibility. `c3run.py:248–253` requires eligibility on both rules before declaring a pair passed. The earlier all-reason-zero counterexample cannot pass.  
   **Fix applied:** pooled eligibility now gates selection. One contract discrepancy remains: adaptive `run_valid()` requires zero failed steps (`c3common.py:220–221`), making its pooled 1% allowance ineffective. **Clarify A4.5 or implement its stated adaptive allowance.**

3. **MAJOR — PARTIALLY: manufactured solver equivalence.**  
   The new test **does independently execute a BE trajectory** (`test_fs.py:162–219`); it no longer merely checks an algebraic identity. All three recorded trajectory comparisons reproduced, including iteration/reason equality.

   It is not a fully faithful vendor transcription:
   - Initial non-finite residuals lack the vendor’s immediate reason-3 exit (`test_fs.py:137`; vendor `common.py:414–415`). My probe produced **vendor: 0 iterations; NumPy: 21 iterations**.
   - The tiny-step test uses the updated state (`test_fs.py:150–154`); the vendor uses the pre-update state (`common.py:437`).
   - NumPy uses a general solve instead of Cholesky and does not reproduce its failure behavior. The fixed phase omits finite-step handling; finite trust clipping is untested.
   - Mutations still use distance to a SciPy minimizer (`test_fs.py:126–127,279–292`), not A2.2’s direct independent step-residual expectation.

   **Fix:** faithfully implement exit/failure semantics; add finite-trust, rejection, tiny-step and non-finite fixtures; compare residuals/Jacobians directly against independent expectations. Include vendor dependencies in the certificate hashes (`test_fs.py:303–304`).

4. **MAJOR — PARTIALLY: G-FS1/G1a enforcement.**  
   G1a now stops on failure (`c2grid.py:145–150`). G-FS1 has per-case coefficient/field comparisons and an enforced failure exit (`c3run.py:323–346`).

   However, approved-equivalent compares `valid` and **non-stationary counts**, not the required verified-exit counts (`c3run.py:334–336`). Two invalid runs with different reason-3 counts can satisfy those comparisons. The CN control is merely recorded (`c3run.py:339–343`); a non-discriminating control does not withhold anything as A1.10 requires. It also uses the first configured case rather than explicitly requiring `923801/0`.

   **Fix:** compare explicit verified-exit counts and the prescribed eligibility; pin the control case and propagate “control unresolved” to dependent verdicts.

5. **MAJOR — PARTIALLY: timing validity.**  
   Both drivers now require nonmissing matching hashes for all three timed invocations and record panel validity (`c2grid.py:298–306`; `c3run.py:206–213`).

   **A4.4 silently weakens drift detection:** it pools all subjects into one median. A consistently bad subject can be hidden by the other subjects. The inherited 3D check instead evaluates subject drift and gates its worst value (`vendor/quad3d/qpanel.py:495–513`). No justification or injected-fault evidence accompanies this change.

   **Fix:** retain subject-level drift checks, report spread/outliers, and demonstrate detection of an injected fault exceeding 10%. Enforce validity downstream—see item 13.

6. **MAJOR — RESOLVED: missing adaptive timing and repeated GAL preparation.**  
   Preparation is outside queries (`c3common.py:204–206`), and adaptive subjects enter the panel (`c3run.py:352–354`). A4.3 explicitly narrows the adaptive timing inventory with a cost justification while retaining deployment candidates.

   **Fix applied:** adequate for the amended contract. Minor documentation mismatch: `timed_adaptive()` also includes GAL-CN at `dt0` (`c3run.py:63–64`), whereas A4.3 names only its `2.5dt0` arm. Align them.

7. **MINOR — PARTIALLY: FOM storage.**  
   Lossless shared-node fields remain saved (`c2grid.py:225–245`). A4.2 explicitly removes the full-mesh float32 requirement, so absence of that archive is now an amendment rather than an undocumented implementation deviation.

   Its justification overstates the plan: E0 permits shared-subset lattices **or audited interpolation** ([paper plan:83](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/reports/2026-10-06-jcp-offmesh-paper-plan.md:83)). The reduced archive supports the chosen subset route; it does sacrifice other rescoring options. Explicit coordinates/order/times are still absent from the archive.

   **Fix:** record coordinate/time metadata and state the narrowed capability accurately, including the rerun requirement.

8. **MAJOR — RESOLVED: escalation classification.**  
   Separate pass/fail/unresolved axes are implemented (`c3run.py:244–265`); every escalation rule receives all four arms (`c3run.py:239–240,274`). Fixed/adaptive sensitivities are saved (`c3run.py:278–282`). Operator states now come from the prescribed `gl64` adaptive BE trajectory, with zero-denominator and dropped-state counts (`c3run.py:283–299`).  
   **Fix applied:** the original classification/population defects are repaired.

9. **MAJOR — RESOLVED: discarded step evidence.**  
   Fixed-sweep `gn`, `rn`, damping and other arrays are retained; adaptive verification arrays are retained (`c3common.py:247–260`). Rows also store step counts and `a_eff` (`c3run.py:159–163`).  
   **Fix applied:** the original evidence loss is repaired.

10. **BLOCKER — NOT RESOLVED: resource gate can certify substantial underestimates.**  
    A certificate mechanism exists, but it does not implement A2.12:
    - 2D dry-run omits FOM order/calibration/evaluation, vendor checks and all timing (`c2grid.py:69–72`).
    - The estimator adds only per-setting timing; **both common panels are omitted** (`inventory.py:79,88–109`).
    - Warmups, setup, rule reconstruction, operator checks, extra decodes and serialization are absent.
    - Rates merge different rules under `(setting,spec)` (`inventory.py:49`). Rule point count disappears again from production run lists (`inventory.py:78`).
    - Memory is copied from a smaller smoke without projecting production residency. Missing peak `-1` passes; non-finite peaks explicitly pass (`inventory.py:113`).
    - Smoke completion, accepted gates and matching GPU are not required.

    A synthetic two-rule probe—first/steady times `[10,1]` and `[100,10]`—estimated **zero compilation seconds**, because rule identity was discarded.

    **Fix:** enumerate actual invocations, including conditional paths and warmups; key measurements by rule/shape/static signature; project peak live memory; fail closed on missing measurements and incomplete or incompatible smokes.

11. **MAJOR — PARTIALLY: large-rule memory residency.**  
    Escalation now drops several rules (`c3run.py:239–242,274–275`), removing the original retain-every-escalation-rule behavior. But it still keeps `gl64` while constructing `gl80`; the ladder then accumulates all rule preparations (`c3run.py:309–315`). Grid mode retains every setting’s preparation until the common panel (`c3run.py:350,358–364`).

    These are bounded by configuration, but **not bounded by a validated memory budget**. The smoke’s reduced ladder/settings do not establish the production peak.

    **Fix:** enforce a live-rule residency limit, include executable/workspace and construction peaks, and validate the largest production live set on the intended GPU. No OOM is claimed from inspection alone.

12. **MINOR — RESOLVED: common-panel seed.**  
    The common panel uses explicit `common_seed` (`c3run.py:363–364`), and every panel records its effective seed (`c3run.py:212`). Production defaults specify `20261013` (`make_configs.py:38`).  
    **Fix applied:** no remaining seed defect.

The additional findings are:

13. **MAJOR — WRONG: H2/H5 ignore common-panel invalidity.**  
    Per-setting validity gates selection (`analyze2d.py:155–160`), but H2 reads common costs unconditionally (`analyze2d.py:239–269`), and H5 checks only whether the panel exists (`analyze2d.py:323–360`).

    **Reproduced:** an in-memory fixture with `common.valid=false`, `hashes_ok=false`, and drift `4.0` returned **H2a=true, H2b=true, H5(ST)=true, H5(TX)=true**.

    **Fix:** make every dependent timing verdict unavailable when the common panel fails; reject non-finite/incomplete paired samples.

14. **MAJOR — WRONG: G-score passes without executing its gate or passing controls.**  
    `score.py:65,123` initializes `passed=True` even without `--check`. Missing rows are skipped; 3D additionally skips non-finite scores (`score.py:139–143`). Controls do not affect `gate.passed`, and scoring exits successfully after failures (`score.py:157–162`). Analysis accepts this with a permissive default (`analyze2d.py:74`).

    **Reproduced:** analysis accepted `compared=0` and `controls.all_break=false`.

    Furthermore, agreement compares only maximum errors, allowing errors at non-maximal times to escape detection. The coordinate control does not use the pinned dev6/0, `t=0.05` field comparison (`score.py:90–106`), and 3D has no controls.

    **Fix:** distinguish unchecked/pass/fail; require complete expected coverage, finite comparisons, all mandatory controls, and per-time agreement. Implement the specified coordinate-field control and fail nonzero.

15. **MAJOR — WRONG: the reference registry does not enforce the rescoring contract.**  
    Coordinates and normalizers are hardcoded (`score.py:38–46,74,119–121`), rather than supplied by the registry. The 2D manifest is never checked; the 3D registry SHA is never checked (`score.py:50–55,115`). Bank/rotation/case/time identity is not bound to the saved run. `refs.json:9` contains a textual pointer instead of a bank hash.

    The coordinate evaluation is an independent decode path, but its inputs are not authenticated.

    **Fix:** implement a validated registry schema containing hashes, descriptors, ordered coordinates, times, boundary convention and normalization; verify it against the run before decoding. Reject unsupported registry kinds explicitly.

16. **MAJOR — WRONG: rescoring mixes new ROM errors with old FOM errors and reselects arms.**  
    ROM errors come from the supplied scores, but H5 FOM errors come directly from the original run (`analyze2d.py:313–320`). `score.py` never scores `fom_f257.npz`. Changing references therefore produces mixed-reference matching.

    Analysis also reruns scheme/step selection (`analyze2d.py:151–196`), violating A1.8’s frozen-selection rule. Verified E0 uncertainty intervals are unsupported; absent ST/S references silently give zero uncertainty (`analyze2d.py:94–95`).

    **Fix:** rescore FOM archives; load the frozen selection unchanged; recompute only authorized reference-dependent matching/H5 maximization; implement interval comparisons and label rematching.

17. **MAJOR — WRONG: H1 uses different acceptance rules.**  
    H1(a) requires the 25% fall against **TX**; the code additionally requires ST and does not apply the fewer-than-half-resolved “reference-limited” rule to the first two transitions (`analyze2d.py:227–235`; `DESIGN.md:211–216,499`). H1(b)’s median reduction also lacks an explicit tie-policy improvement check.

    **Fix:** encode the registered conditions separately for each transition; apply reference-limited precedence and the specified tie policy consistently. Report ST sensitivity without silently adding it to H1(a)’s acceptance criterion.

18. **MAJOR — WRONG: H2(b) can substitute a missing dial rung.**  
    Analysis appends whichever selected settings exist, then checks `dial[:3]` (`analyze2d.py:257–267`). If R256 is unavailable, it can assess **R128→R384→R512** instead of A1.9’s mandatory **R128→R256→R384**.

    H2(a) can also pass vacuously on an empty BE population (`analyze2d.py:250–268`); ordinary numerical Pareto fronts required by A3.1 are not generated.

    **Fix:** require the exact named transitions, report unavailable if any required arm is missing, reject empty coverage populations, and produce the registered fronts separately from tie-aware coverage.

19. **MAJOR — WRONG: H3 can confirm cancellation using unverified anchors.**  
    Anchor uncertainty is computed without checking any anchor’s verification (`analyze2d.py:107–117`). H3 likewise ignores BE/CN verification and emits “confirmed” solely from numerical inequalities (`analyze2d.py:276–307`).

    Identical failed anchor/replay trajectories produce `U_anc=0`; the guards then provide no protection. The code also substitutes a sum of trajectory maxima for A3.3’s uncertainty at `t*`, potentially changing negative verdicts.

    **Fix:** qualify BE, CN and all anchor/replay trajectories before H3; otherwise report unavailable. Compute the registered per-time uncertainty at `t*`, and retain the “indicator, not certified bound” label.

20. **MINOR — WRONG: global-scheme tie handling uses forbidden partial sums.**  
    The normal A3.2 unbeaten-set selection is implemented correctly. But global ties still compare partial sums among schemes missing settings (`analyze2d.py:194–195`), although A2.6 makes such schemes ineligible for the four-setting sum comparison. It also names a winner when every setting has an empty winner set.

    **Fix:** compare sums only for complete schemes; resolve incomplete ties by the fixed order, and return unavailable when no scheme receives support.

21. **BLOCKER — WRONG: staging does not bind the resource certificate to the submission.**  
    `cluster/stage.py:57–59` checks only `fits` and requested hours. A certificate produced for an H200 can authorize the default A100 submission; neither GPU identity/capacity nor configuration/source/smoke hashes are checked. Being committed at HEAD does not establish certificate freshness.

    **Fix:** bind the certificate to source/configuration hashes, measured smoke artifacts, GPU type/capacity, production shapes and allocation limits; verify these before staging. Replace the `smk*` name-based exemption with an explicit validated smoke classification.

**A4 assessment:** A4.1’s tighter-level comparator change is justified and independently cross-checked; A4.3’s reduced adaptive timing scope is explicitly justified. A4.2 is defensible only as a declared restriction to shared-subset rescoring. **A4.4 introduces an unjustified weakening of drift detection.** A4.6 promises a resource guarantee that the implementation does not provide. A4.7’s coordinate-decode approach is sound, but its scoring gate remains incomplete.

The fixed-sweep residual, Jacobian, history advancement and startup mathematics remain consistent on inspection. Passing the manufactured certificate does not repair the acceptance and analysis defects above.

**Overall verdict: NOT READY.**