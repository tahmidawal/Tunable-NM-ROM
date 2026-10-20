**NOT READY FOR SMOKE.** All three staged jobs lack a required module. I also reproduced a false G-FS1 pass and a submission-cap failure.

Paths below refer to `experiments/jcp-combined/`.

| Previous item | Status | Evidence |
|---|---|---|
| **2 — escalation eligibility** | **RESOLVED** | `c3common.py:221–245` enforces adaptive verification and pooled fixed-sweep eligibility; `c3run.py:248–253` checks both rules. A5.1 resolves the earlier allowance discrepancy. |
| **3 — manufactured equivalence** | **PARTIALLY** | `test_fs.py:148–181,214–229` repairs most exit semantics; `:331–346` adds independent stationarity checks and finite-trust comparisons; `:350–353` hashes vendor dependencies. A non-finite-Jacobian counterexample remains—finding 3. |
| **4 — G-FS1/G1a** | **PARTIALLY** | `c2grid.py:146–151` stops on G1a failure. `c3run.py:338–354` compares exit counts and pooled eligibility, pins case 0, and records control discrimination. The exact branch can still accept NaNs—finding 2. H4 consumption remains a before-results obligation. |
| **5 — timing validity** | **RESOLVED** | `timing.py:15–27` gates drift per subject; `:30–39` detects the injected fault. Both drivers require nonmissing accuracy hashes for every timed A/B output (`c2grid.py:301–306`, `c3run.py:209–213`). |
| **6 — adaptive timing/preparation** | **RESOLVED** | `c3common.py:204–206` prepares forms outside queries; `c3run.py:63–64,359–362` includes adaptive subjects. A5.2 reconciles the inventory with A4.3. |
| **7 — FOM archive** | **RESOLVED** | `c2grid.py:236–248` saves float64 restricted fields, coordinates and times. A5.4 corrects the shared-subset versus interpolation claim. |
| **11 — residency** | **RESOLVED for amended smoke scope** | `c3run.py:239–242,274–275,293–305` releases escalation preparations; `:309–322` processes ladder rules individually. Grid retains deployed rules (`:358–372`), explicitly permitted by A5.6. Production peak certification remains deferred. |

1. **BLOCKER — every staged smoke crashes importing `timing`.**  
   [stage.py:22](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/cluster/stage.py:22) defines `COMMON`, `FILES2` and `FILES3` without `timing.py`. Both drivers import it (`c2grid.py:34`, `c3run.py:39`). The staged tree therefore raises `ModuleNotFoundError` before setup.

   **Fix:** add `timing.py` to the staged common files and verify imports against an isolated staged tree. I found no other missing module on the drivers’ runtime import paths.

2. **MAJOR — G-FS1’s “exact” branch silently accepts a later non-finite case.**  
   [c3run.py:329](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/c3run.py:329) builds per-case distances, then uses Python `max(per)` at line 334. With finite initial states, matching case 0, and NaNs in case 1’s evolved states, the reproduced distances are `[0.0, nan]`; `max(per) <= 1e-12` returns **True**. Matching iteration/reason arrays then yield “exact,” bypassing the stopping gate. This violates A2.1’s every-case criterion.

   **Fix:** require finite trajectories and finite per-case distances before either acceptance branch, then use an explicit all-case comparison. Require a finite CN-control distance as well.

3. **MINOR — the NumPy certificate oracle still differs on non-finite Jacobians.**  
   [test_fs.py:142](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/test_fs.py:142) uses SciPy Cholesky with finite checking, but lines 163 and 216 catch only `LinAlgError`. A finite residual with a NaN Jacobian produces **`ValueError`**; the vendor instead rejects steps and exits **reason 3 after 21 iterations**. I reproduced both outcomes.

   **Fix:** explicitly reproduce non-finite matrix/step handling in both NumPy phases and add this fixture before regenerating the certificate. This does not invalidate the passing finite manufactured trajectories.

4. **MAJOR — the locked submission-cap check fails open.**  
   [submit.sh:18](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/cluster/submit.sh:18) wraps the entire `squeue | grep` pipeline in `|| true`. If that queue query fails, `grep -c` supplies `0`, permitting submission. A shell probe reproduced this. The earlier queue query cannot protect against a subsequent failure inside the lock.

   **Fix:** capture `squeue` output separately and abort on failure; count names only after a successful query.

The manufactured suite passed with its output writes intercepted. All **seven certificate source hashes match both checkout and HEAD**. All three smoke configs match `make_configs.py`. The 2D mode-prefix checks pass at \(L=1024\); the 3D output index is exactly the \(63^3\) interior ordering, and every configured fixed-sweep schedule constructs successfully. No full-size GPU execution or memory-fit claim is made.

The 3D reference checksum, acceptance flag, mesh, timestep, count and smoke-field shapes pass. The 2D field checksums pass, but its cohort hash fails regeneration **locally**; target-cluster compatibility remains unverified because SSH closed the connection. This is not established as a cluster blocker.

A5 supersedes the earlier contracts as indicated above. **Production staging fails closed:** `stage.py:60–63` invokes `inventory.verify_certificate`, which unconditionally raises at `inventory.py:127`. I have not counted A5.9’s deferred resource or analysis work as smoke blockers. Both shell scripts pass syntax checks; no additional concrete defect was established in `collect.sh`.

Audit-process disclosure: one probe inadvertently imported the vendored config generator, rewriting identical config contents. Git shows no content changes, but this violated the read-only instruction. The lab log was not modified.

**NOT READY FOR SMOKE.**