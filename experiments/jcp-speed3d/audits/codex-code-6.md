**NOT READY.** I found honest-path false failures and unchecked ADT failures. No files or git state were changed; no GPU or jobs were used. The only executable check extracted pure functions with AST to verify counts.

Paths below are relative to `experiments/jcp-speed3d/`.

1. **HONEST-PATH BLOCKING — false FAIL: an infeasible L=384 pilot fails acceptance.**

   A1.9 explicitly permits dropping a mesh whose pilot does not fit (`DESIGN.md:389–391`). The producer catches table-building and FOM-calibration exceptions and records `infeasible` (`s3modes.py:419–423`, `444–448`).

   The checker nevertheless requires complete ROM results, converged FOM cases, and timing panels at every configured mesh (`smoke_accept.py:298–327`). A correctly recorded OOM therefore fails. An OOM outside those catch blocks can also terminate the entire mesh step.

   **Fix:** give the pilot an explicit feasible/infeasible outcome. Accept a documented resource failure as a dropped mesh, skipping its dependent requirements while checking the remaining meshes. Keep unexpected exceptions as failures; isolate the pilot if necessary to survive process-level OOM.

2. **HONEST-PATH BLOCKING — false FAIL: standalone G4 requires “exact,” excluding the design’s accepted “within tolerance” outcome.**

   A1.8 permits batching distances up to `2.5e-5`, distinguishing “exact” at `1e-8` (`DESIGN.md:351–354`). The profiling checker follows that rule (`smoke_accept.py:265–267`).

   However, standalone acceptance requires `G4 <= 1e-8` (`smoke_accept.py:145`), and `test_s3.py:172–173` independently exits unsuccessfully above that threshold. Honest numerical differences between `1e-8` and `2.5e-5` therefore fail the smoke despite satisfying the amended design.

   **Fix:** change both predicates to accept `<=2.5e-5`, retaining the exact/within-tolerance classification and count.

3. **HONEST-PATH BLOCKING — false PASS: finite distances can hide invalid ADT reference solves.**

   ADT can return finite output while failing verification: `run_valid` explicitly checks `nfail == 0` (`s3common.py:223–230`). Thus finite distance is insufficient.

   Several checks omit available validity evidence:

   - Standalone tests produce `ADT_CN_0.025_valid` and `ADT_BE_0.01_valid` (`test_s3.py:108–112`), but acceptance never checks them.
   - Grid output records tight ADT validity (`s3run.py:105`); acceptance checks row presence and tighter validity, but not tight validity (`smoke_accept.py:190–198`, `212–215`).
   - Reduced-rule profiling stores only tight/tighter distance (`s3modes.py:373–378`), and acceptance checks only that distance (`smoke_accept.py:284–286`).

   Two nearby finite trajectories can therefore pass despite an unverified reference.

   **Fix:** require all relevant tight and tighter ADT validity flags, with exact case populations. Persist both flags for reduced-rule checks.

4. **HONEST-PATH BLOCKING — false PASS: mesh ADT agreement is only checked for finiteness.**

   The producer computes `adt_tight_tighter` (`s3modes.py:434–436`). Acceptance merely requires it to be finite (`smoke_accept.py:307–309`); even a large discrepancy passes this gate.

   A1.4 supplies the `2.5e-5` agreement bar (`DESIGN.md:306–311`), and A2.3 extends tight/tighter checking to route-4 meshes (`DESIGN.md:461–462`).

   **Fix:** enforce `adt_tight_tighter <= 2.5e-5` for feasible meshes, alongside the existing validity checks.

5. **HONEST-PATH BLOCKING — possible false FAIL: K2 is required to fail separately on every standalone case.**

   Both checker and test require `min(nonstationary) > 0` (`smoke_accept.py:142`, `test_s3.py:168–169`). The design requires the configuration to be **ineligible**, using pooled stationarity and the other eligibility conditions (`DESIGN.md:193–194`, `306–311`).

   An honest predictor-only run could satisfy stationarity on one case while failing sufficiently on the others. The control would correctly demonstrate ineligibility, yet this stronger per-case requirement rejects it.

   **Fix:** evaluate K2 with the registered configuration-level eligibility predicate. Retain per-case counts as diagnostics.

**NOTED — the investigated schema and count concerns are consistent.** No fixes needed for these:

| Area | Producer/checker agreement |
|---|---|
| JSON case keys | Dictionary keys become strings; FOM checks convert them back with `int`. Invocation `case` values remain integers. |
| Repetitions | `s3ctx.py:205–214` emits zero-based `rep`; checker uses `range(reps)`. Warm timing correctly uses per-case counts instead. |
| Grid | M2: 333 rollout rows including tighter ADT; 82 panel subjects and 164 A–B–A records. W1024: 225 rows; 55 subjects and 110 records. Expected populations match. |
| FOM | Four FD configurations and one PS configuration; three accuracy cases and two warm timings each. Outer FD keys omit tolerance level by design; chosen-level keys are stored separately (`s3modes.py:170–177`). |
| Profiles | FS profiles have nine subjects, QN profiles eight, matching `s3modes.py:297–311`. With two cases: 18 and 16 records respectively. Confirmation has two records. |
| Batching | Three labels × sizes 1 and 3; three timings per size. JSON size keys are read as strings correctly. |
| Mesh | Same-mesh lookup keys omit level; panel subjects use the actual chosen-level key. Fixed matched keys retain their specified level. Deduplication matches the producer. |
| `smoke1.json` | It describes the bundle; the four `smk*.json` files determine its actual populations. |

**NOTED — remaining threshold and execution checks:**

- **QN zero nonstationary steps:** not stricter for this smoke population. Three CN trajectories at `dt=.025` contain 30 steps; the design’s 1% allowance is 0.3 steps, so the permitted integer count is zero.
- **T1 `[1.8,2.2]`:** the design’s explicit band (`DESIGN.md:201`), not an added restriction. Timing noise can invalidate evidence; that is not itself a checker defect.
- **Grid/reduced ADT `2.5e-5`:** the registered threshold, not stricter. The issues are omitted validity and the missing mesh bound.
- **Environment:** the staged batch script exports `SOURCE_COMMIT` (`cluster/stage.py:107`); SLURM supplies exported `SLURM_JOB_ID`. `smoke.sh:10–15` preserves that environment for every child, including acceptance.
- **Wrong-job output and crashes:** identity checks reject different job/commit values; step exit statuses and `complete` are checked. Section exceptions become failed gates while later sections continue (`smoke_accept.py:55–64`, `105–120`, `329–331`). The shell’s final exit zero is intentional under A4.1; acceptance must be read from `ACCEPTANCE.json`.

**NOT READY FOR CLUSTER SMOKE.**