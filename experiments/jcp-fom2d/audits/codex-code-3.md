**NOT READY to submit `runs/smk1` as the acceptance/resource smoke.** A fresh launch has no deterministic failure identified, but ordinary resume destroys compilation evidence needed for production certificates. Extension handling also remains incorrect.

Audited HEAD **`dc3e4a1a1f10937df9bd9e4243cbbe2cbb18a33e`** and its staged smoke. Concurrent uncommitted C2 edits are excluded. All references below are relative to `experiments/jcp-fom2d/`.

No repository or lab-log writes, cluster commands, or submissions. CPU checks used the prescribed interpreter and environment; scratch usage was **3.1 MB**.

**Round-2 findings 1–7**

| # | Status | Evidence |
|---|---|---|
| 1. Smoke-target population crash | **FIXED** | `match2.py:357–359` intersects target case IDs with the smoke population and labels the variants. All **48** supplied P1 variants passed the exercised population/control paths. |
| 2. Compilation evidence | **PARTIAL** | Actual compilation is now recorded: `fomrun2.py:109–151`, `psjax2d.py:72–81`. However, `ladder.py:341,357` replaces persisted evidence with the current process’s log on resume. Reproduced certificate failure below. |
| 3. Divergent configuration blocks campaign | **FIXED** | `match2.py:78–80` treats mutually nonfinite scores as agreement; `match2.py:145–146` still rejects that configuration. Synthetic check returned disagreement **0**, admissibility **False**. |
| 4. Resource certification | **PARTIAL** | Accepted smoke required at `resources.py:73`; preparation/burn costs included at `resources.py:48–61`; extension pilots added at `configs/smk1.json:175–183`; H200 checked at `cluster/stage.py:47`. Clean-run coverage is complete, but resumed smoke evidence breaks the certificate path. |
| 5. Target identity / FRONT ingestion | **PARTIAL** | Bank/form/level enter identity at `targets.py:40–43`; collisions are rejected at `targets.py:55`. **FRONT ingestion remains unimplemented**, `targets.py:105`. This blocks FRONT matching/C2, not base-ladder execution. |
| 6. Extension bookkeeping | **PARTIAL** | Base-only decisions, triggering targets, requirements and completion are now represented at `match2.py:367–388`. Already-tested resolutions are not excluded, completion uses ambiguous string prefixes, and C2 configuration generation does not enforce completion. |
| 7. Accounting / account cap | **FIXED under A5’s stated mitigation** | Live accounting rows take precedence at `ledger.py:131–140`; failed queue queries abort at `cluster/submit.sh:18,25,45`; the two-slot margin is enforced at line 28. The disclosed cross-lane race remains **NOTED**, not an additional blocker. |

**Ranked fixes**

1. **Preserve compilation evidence across resume — resource-smoke blocker.**

   [ladder.py:357](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/ladder.py:357) unconditionally assigns `rep['compile_log'] = foms.compile_log`. Completed pilots/configurations are skipped on resume, so their executables need not compile again.

   A tiny actual CPU ladder run demonstrated:

   - First execution recorded `ps|N4|0.005|lattice` compilation and supplied a usable resource rate.
   - Identical resume retained `accepted=True` but saved `compile_log={}`.
   - `resources.ps_rate` then raised **`KeyError('ps|N4|0.005|lattice')`** at `resources.py:42`.

   Partial resumes similarly lose earlier executable entries. Merge persisted and new evidence, retain attempt provenance, and require complete compile-key coverage before accepting a resource smoke. This is an ordinary execution failure, not forgery.

2. **Repair extension selection and completion — required before final matching/C2.**

   At `match2.py:376–381`, the proposed smaller resolution is checked only against the lower bound, not against resolutions already tested.

   Executing the actual extension block with a tested-but-inadmissible CC32 and an admissible selected CC64 requested **CC32 again**. A2.5 explicitly excludes an already-tested resolution. Loading that repeat can subsequently hit the duplicate-group assertion at `match2.py:60`.

   At `match2.py:384`, prefix matching also confuses distinct steps:

   ```text
   "fd|CC|L16|0.0125".startswith("fd|CC|L16|0.01") == True
   ```

   Thus a missing `dt=0.01` extension group can be marked complete by `dt=0.0125`. Compare parsed configuration/group identities exactly.

   Finally, `make_c2configs.py:50` checks only `blocked`, while extension completion is separate (`match2.py:365,388`). Require `extension_complete` before generating C2 jobs.

3. **Finish FRONT ingestion before claiming production FRONT matching.**

   `targets.py:105` still exits immediately. The identity improvements do not complete that workflow. P1 fallback remains a separate, explicitly labelled route under A1.7.

**Smoke coverage and launch checks**

| Question | Assessment |
|---|---|
| Staged provenance | All **107** provenance entries match HEAD blobs and recorded hashes. Manifest verification passed; staged commit equals audited HEAD. |
| Imports / batch syntax | Staged ladder, matcher, gates, targets and resource imports succeeded. Batch shell syntax passed. |
| Execution setup | H200, GPU partition, absolute interpreter, f64/highest precision, GPU preflight, paralab paths and `--no-requeue` are present. Remote availability was not tested. |
| DESIGN §6 gates | All seven gates are requested. `ladder.py:345–348` enforces their results, including G-PS1’s separate half-step control. |
| A1.5 controls | K-loose and fixed dev6 `0→1` K-swap are wired at `ladder.py:312–328`. K-syn/K-false remain offline; the supplied smoke-target variants now work. |
| G-CT | All five registered family/resolution relationships are present at `ladder.py:162–166`. |
| A1.6/A5.4 pilots | **68 FD pilots**, covering every tolerance at every base/extension resolution; **13 PS pilots**, including N4 and informational N1024. |
| Certificate lookup coverage | Synthetic populated measurements successfully projected all four supplied production configs: **48 CC groups, 42 UB groups, 7 UB4096 groups, 88 PS configurations**. |
| Compile logging | Correct on the exercised fresh-run path; broken across resume as above. |

Additional small CPU checks:

- G-FD1 passed.
- G-FD2 at **L32** gave zero field difference and identical Newton counts.
- AOT G-PS1 passed both prescribed N32 fixtures: relative differences **7.12e-16** and **3.54e-16**; running maxima agreed; both half-step controls fired.

These establish code-path checks, not GPU gate outcomes, burn-in adequacy, memory sufficiency or the two-hour smoke runtime.

**C2: obvious remaining defects at audited HEAD only**

- `c2time.py:289–294`: stage-1 selection does not remove candidates vetoed during stage 1.
- `c2time.py:190–197`: ROM validity is recorded but not enforced by G-RR; baseline A validity is also not a panel gate.
- `c2time.py:127–137`: prepared FOM inputs accumulate; executable/input release still does not implement A1.6.
- `fomrun2.py:153–163`: the callable-only C2 path bypasses FD/PS compile-log collection.
- `make_c2configs.py:50`: extension completion is not enforced.
- `make_c2configs.py:90` versus `c2analyze.py:144`: a population-too-small target can lack `union_joint_group`, causing a K2-analysis `KeyError`.

The previously defective H-comp/K2 wording, T-fault separation check and 2× crossover output are repaired in committed `c2analyze.py`. The generated target identity also supplies a distinct alias for A’s configuration, so A’s own target need not lack stage-2 measurements.

Forgery-only acceptance/evidence substitutions are **NOTED, nonblocking**.

**Submission verdict:** **NOT READY** for the resource-smoke workflow until fix 1 is committed and restaged. After that, an accepted smoke and usable certificates can support the four base production ladder jobs; I found no additional deterministic launch/arithmetic blocker in those configurations. Fixes 2–3 and the separate C2 audit remain necessary before the downstream matching/C2 campaign is ready.