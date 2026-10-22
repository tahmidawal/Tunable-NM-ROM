**Overall: WRONG — production-blocking integration and enforcement gaps remain.**
**Tests:** 133 passed; training `--static-only`: 0 failures. Test paths were redirected in memory to permitted scratch; pytest caching and bytecode writes were disabled. No numerical training probes ran.

Paths below are relative to `experiments/imp-fronts/`.

| Item | Verdict | B-1 fix status / evidence / required fix |
|---|---|---|
| Production configs | **CORRECT** | Fixed: `rom/schedule.py`, `mkconfigs_pol.py`, and `fpolicy.py` share and validate 40 cases, S1/S3, 0.5 h policies, and amended ceilings. |
| Incomplete selection | **CORRECT** | Fixed: `select2d_r384.py` persists an explicit null selection for missing/incomplete results; config generation preserves fib4181 fallback. |
| Killed-job recovery | **CORRECT** | Fixed: `fpolicy.py` persists initial states before setup, writes JSON atomically, and `fcollect.py` reconstructs from RUN/config when progress is absent. |
| Smoke shell fragment | **CORRECT** | Fixed: offending substitution is removed; rendered templates pass Bash syntax tests. |
| C-B1/C-B6 admission gate | **CORRECT** | Fixed: `rom/evidence.py` binds control records to file hashes, rechecks outcomes, and `resources.admit()` rejects missing/failed evidence. |
| Cost certificate | **WRONG** | Partly fixed: one-case/ROUND-CAP dry runs and empty training phases are rejected; cutoff arithmetic is updated. But `phase_times.py` emits `datagen_seconds`/`projection`, whereas `resources.train_item()` requires `phases`/`increment`; filenames also differ. Real smoke output cannot certify training. Implement and test the actual producer→consumer interface. |
| Dry-run evidence completeness | **WRONG** | `evidence.check_dry()` checks config IDs and policy-list length, but not exact unique policy-result IDs or exercised transitions. Require result identities, solve/history consistency, and the registered path coverage. |
| Admission | **WRONG** | Partly fixed: schedule membership, ceilings, smoke reservations and evidence are checked. With passing evidence assumed, probes admit `{'selected':'S3'}` without item certificates, and count a completed job lacking `charged_gpu_h` as zero. Validate certificate contents and fail closed on missing/invalid ledger charges. |
| Test-once | **WRONG** | Not fixed operationally: `testonce.require()` accepts the same registration for two different empty output directories; registration is never consumed. Probe reproduced this. Atomically claim the key once, bind its output/job, and permit collector-only recovery. |
| Registration → floors | **CORRECT** | Fixed: `register_arm.py` writes canonical filenames; `bankguard.py` and `phaseb/floors_b.py` resolve/hash them and support incumbent defaults. |
| B0 route integrity | **WRONG** | Route substitution is fixed, but validation remains incomplete: probes accept eligible `FI|999` and certified `fib1` with one node. Validate IDs against the scheduled cohort table and rules against the registered selection ladder; require exact integer node counts. |
| E9 routing cost | **CORRECT** | B-1 restatement fixed: input routing and post-solve fallback are separated; failures pay ROM cost. Missing paired-job timing yields “unmeasured.” |
| Training baseline / staging | **CORRECT** | Static checks verify 21 baseline files against jcp-robust committed blobs; staged base contains the expected 19 files. Arm changes exactly match declarations. |
| Arm fid single change | **CORRECT** | Solver intervention matches Amendments 1/11: ETDRK4 N128, dt=.001, capture every five steps, exact synthesis, unchanged nodal initial state and parameter draws. Both generator copies and PS dependencies are overlaid. |
| Arm cov single change | **CORRECT** | Generator diff changes only the viscosity transform, preserving RNG position and other draws. Both Stage A/B copies receive the identical replacement. |
| ARM / PROVENANCE / R0 | **CORRECT** | Hashes and declared edits pass static checks; deliberate mismatches are rejected. Shared smoothtrain changes only R0 dispatch; replacement gates fail closed. Diagnostic-cohort changes are disclosed by Amendment 11. |
| C-B3 / C-B4 implementation | **CORRECT** | C-B3 computes aggregate dev6 project error and PS self-convergence; C-B4 checks both complete parameter tables, incumbent failure, and unchanged non-viscosity draws. Production residual checks remain. Numerical passage was **not** rerun here. |
| Training smoke recipe | **WRONG** | `train_body.sh:73` uses N64/16 trajectories/20 steps via `TRAIN2D_DRYRUN`, replacing the registered full-size smoke slice. Restore the amended slice or explicitly amend the specification before use. |
| Smoke failure propagation | **WRONG** | `smoke_datagen.py` exits successfully when `passed=False`; failed/skipped chaining does not block `ALL-DONE`; timing errors are swallowed. `phase_times.py:115` uses completeness, not gate success, for fundability. Propagate failures and require successful generation, gates, chaining and timing records. |
| Training timing projection | **WRONG** | No GPU burn-in precedes timed generation. `phase_times.py` subtracts incumbent **generation+projection** from arm **generation-only**; its claim that this enlarges the increment has the sign backwards. Measure comparable work on the same GPU type and reject mismatched timing provenance. |
| Training Slurm envelope | **CORRECT** | All four templates use `gpu`, A100-80G, absolute cluster venv, f64/highest, exit-42 preflight and the required namespace. Walls: smA 12 min, smB 24 min, trfid/trcov 4 h 36 min. |
| Training cutoff | **WRONG** | Despite `SELF_TIMER_FRAC=1`, `train_body.sh:30` imposes a timeout 120 seconds before the hard wall, contradicting Amendment 11 and certification against 100%. Remove it or register and certify against the actual cutoff. |
| Training static tests | **CORRECT** | Requested static checks pass and establish staging/provenance integrity. They do not cover smoke failure propagation, timing-schema integration, or numerical gate passage. |

No repository/main/LAB-LOG writes, SSH, jobs, or sealed/OOD access. Scratch occupied approximately **29 MB** and was deleted.