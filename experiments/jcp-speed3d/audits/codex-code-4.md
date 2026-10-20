Reviewed `401cd4e42` against `550d07214`. Read-only throughout: no file/git changes, submissions or GPU execution. CPU probes used in-memory artifacts.

| Round-3 blocker | Verdict | Evidence |
|---|---|---|
| 1. Exception-safe acceptance | **RESOLVED** | The conflicting `status` field is renamed; each section catches exceptions. Injected malformed artifacts produced `FAIL` with all six section failures recorded, without aborting collection. |
| 2. Coverage, exit status, attempt binding and local evidence | **NOT RESOLVED** | Exit statuses and local G6/G8 evidence are now checked, and deferrals are explicit. Coverage and attempt binding remain incomplete; detailed probes below. |
| 3. Complete executable staging bundle | **RESOLVED** | The supplied invocation includes all four configs and local evidence. All 40 selected repository files match HEAD; the external first-order reference exists and matches its pinned SHA-256. |

The remaining blocker is acceptance integrity:

- **Standalone checks are tested for presence, not validity.** In-memory execution of the actual `tests()` section passed all five gates with every required cluster check set to `False`, provided `mandatory_failures=[]`. See [smoke_accept.py:60](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/smoke_accept.py:60).
- **Case and repetition coverage still permits missing evidence.** The FOM section passed all 16 gates with FD case IDs `91,92,93`, empty PS cases and empty warm-timing arrays. The mesh section passed all 28 gates with empty ROM validity/distance arrays, empty same-mesh FOM cases, failed matched entries and empty timing panels carrying `valid=true`. These were section-level probes, not a claimed whole-bundle PASS. See [FOM checks:105](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/smoke_accept.py:105) and [mesh checks:163](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/smoke_accept.py:163).
- **Attempt binding is partial.** Result job-ID checking disappears outside Slurm; standalone test JSONs and `STEPS.txt` have no checked attempt identity. Source commit is not checked. Thus the collector does not establish that all evidence belongs to the same execution. See [binding:53](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/smoke_accept.py:53).

I evaluated the sbatch template expression in memory, without calling staging. It requests one H200, `200G`, and `12:00:00`; exports `TASK_ROOT`, both E0 variables, f64/highest precision and `REF1_PATH`; checks the manifest; runs the GPU preflight with exit 42 on CPU; then changes into `experiments/jcp-speed3d` and executes `bash smoke.sh`. Required source, vendor, bank, rule, config and evidence files are included. Remote E0-store availability remains an execution prerequisite.

**Resources are plausible for an initial smoke, but 12-hour completion is unverified.** Grid/profile work uses bounded quadrature tables; FOM includes L=128; mesh reaches L=384 through slab streaming. At L=384, one scalar f64 field is about 0.419 GiB, six outputs 2.512 GiB, and a W1024 slab 1.119 GiB, before solver/workspace allocations. No obvious mandatory allocation defeats the requested resources. However, the L=384 path performs three calibration solves per family, accuracy solves and repeated timing calls; runtime cannot be inferred from a “one-case pilot” description.

Blocking fixes only: validate standalone check contents; require exact configured case/subject/repetition membership and nonempty finite evidence; validate matched results beyond key presence; bind every execution artifact to a recorded attempt/job/source identity, including offline acceptance.

**NOT READY FOR CLUSTER SMOKE.**