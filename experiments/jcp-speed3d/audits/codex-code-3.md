**NOT READY FOR CLUSTER SMOKE.** Reviewed `550d07214`. The numerical fixes substantially address round 2, but the smoke runner and acceptance collector remain blocking.

Review was read-only: no files, git state, jobs, or GPU work changed. CPU probes ran sequentially with bytecode and persistent compilation caching disabled.

**Round-2 blocking fixes**

| Fix | Disposition | Evidence |
|---|---|---|
| 1. Executable smoke coverage and explicit acceptance | **NOT RESOLVED** | Tests are now invoked, but acceptance crashes on ordinary FD results, missing populations can pass, and staging does not supply the complete smoke bundle. |
| 2. FOM completeness and Newton-limit semantics | **RESOLVED** | `converged()` checks expected stages, completed time units, residual/count lengths, finite outputs and Newton counts in `[0,20]`. A4.2 explicitly defines limit exits. |
| 3. Real-stage profiling and preparation/compile measurements | **RESOLVED under A4.3’s amended scope** | Kernels use trajectory histories; full QN-step timing includes production refresh/backtracking. ROM compile estimates and input-preparation measurements are recorded. Static shape/history tracing found no runtime defect in registered configurations. |
| 4. Cross-resolution matched timing | **RESOLVED in code/configuration** | The L=64 restriction is removed. Each mesh gets matched timing panels; `smkmesh` includes both FD and PS matches. Execution remains unverified. |

The fabricated UB records produced:

| Probe | Accepted? |
|---|---|
| Complete record | Yes |
| Stage count short | **No** |
| Newton-count array short | **No** |
| Newton count 21 | **No** |
| Completed time units short | **No** |
| Newton count 20, residual meets tolerance | Yes |
| Newton count 20, residual exceeds tolerance | **No** |

**Smoke failure handling and coverage**

`smoke.sh` continues after an individual command fails and records its exit status, so one solver-process crash ordinarily does not prevent subsequent commands. However, the acceptance collector ignores those exit statuses. Its own exception aborts collection of subsequent gates, and the shell still exits zero.

Coverage is not equivalent to “every A1.10 gate passes”:

- G0–G2, G6v/G7/K8, G5/K7 including W256, T1, hashes, batching G4/K6 and grid K1–K3 have execution/check paths.
- G6 and G8 are separate local evidence, not checked or bound into `ACCEPTANCE.json`.
- K4 and K9 are absent from this runner.
- K10 and G3 are explicitly deferred. K5’s current smoke check establishes conditioning/execution, not its deferred scientific qualification.
- Required settings, cases, panels, profiles, matches and repetitions are not checked against a complete expected inventory.

**G6 evidence:** `checks/test_fom-local-cpu.json` is valid *recorded CPU evidence for the narrow G6 test*: it reports bitwise equality for UB, UC and CC. `test_fom.py` actually uses `np.array_equal` against the separately loaded E0 solver. I also confirmed that the solver source from `NST =` onward is identical to E0. I did not rerun the full L=64 comparison. The artifact lacks source hashes and should be bound to the reviewed sources when incorporated into acceptance.

**Profile trace**

In [s3modes.py:251](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/s3modes.py:251), FS/FSP and QN trajectories have shape `(steps+1, 1, R′)`. Thus `cn`, `cm` and `cp` are `(1,R′)`; contraction receives `(R′,)`, produces `(M,R′)`, and Cholesky/solve dimensions agree.

For the four registered timesteps, `kmid-3` is **9, 7, 4, 2**: no negative-history indexing. ADT configurations are skipped before accessing `W_all`. FS configurations also supply `W_all`; their separate adaptive-start kernel receives scalar viscosity/tolerance correctly. The QN kernels attached to an FS profile are QN diagnostic kernels evaluated on that FS trajectory, not measurements of FS iteration cost. This assessment is by source tracing, not a profile execution.

**A4.5 deferrals**

The analysis deferrals are reasonable **conditionally**. Grid jobs retain rows, trajectories, certification, distances and timing invocations; f1 retains calibrated convergence summaries, scores and warm repetitions with hash checks. Those support later selection, matching and confirmation without reconstructing missing trajectory evidence.

The conditions remain: enforce completeness/provenance before selection, implement matcher/K10 before choosing matches, qualify H3 before freezing, validate all frozen paths/associations before replication, and require complete smoke-derived resource certificates before production staging. Current smoke acceptance cannot establish those prerequisites.

**Blocking fixes**

1. **Make acceptance collection exception-safe.** At [smoke_accept.py:62](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/smoke_accept.py:62), calibration passes `status=` into `gate()`, which already constructs `dict(status=..., **ev)`. An ordinary calibrated FD entry reproducibly raises:
   ```text
   TypeError: dict() got multiple values for keyword argument 'status'
   ```
   Rename the evidence field. Malformed/missing artifacts must produce explicit failed gates while allowing other artifacts to be assessed.

2. **Reject missing coverage and failed execution.** I reproduced `PASS` using empty settings, timing panels, FOM populations, profile/batch populations and mesh populations, with empty standalone failure lists and a W256 G5 summary. Require the configured inventory and case/repetition membership; inspect command exit statuses; validate standalone check contents. Bind results to the current attempt so stale successful JSON cannot mask a failed rerun. Account explicitly for local G6/G8 and absent/deferred controls.

3. **Stage an executable complete smoke bundle.** [cluster/stage.py:63](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/cluster/stage.py:63) stages only the selected `cfgname`; `smoke.sh` needs all four smoke configs. The default command runs only one mode. Also, `TASK_ROOT` is assigned but not exported, so invoking `bash smoke.sh` through `--cmd` does not supply its required environment variable. Provide and verify one concrete staging invocation that supplies all dependencies and runs acceptance.

**Non-blocking for initial smoke**

1. Complete the A4.5 decision gates before their respective selection, production and replication steps.
2. Treat first-call-minus-later-call values as compile **estimates**: cache reuse and first-use preparation can contaminate them.
3. Preserve source provenance for local G6/G8 evidence; do not treat historical local profile/mesh artifacts as validation of this revision.
4. Fix K4’s positional case selection before scoring decisions; A1.10 specifies cases 0 and 1.

**NOT READY FOR CLUSTER SMOKE.**