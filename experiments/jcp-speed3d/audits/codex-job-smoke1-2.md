**HOLD — round 1’s resource-failure item is only partially resolved.** Reviewed HEAD `bc5e165f2633f80ff25d7f66d49230c567eaea78`.

1. **Remaining blocker, under A5:** the new handlers cover Python exceptions from ROM, ADT tight/tighter and timing calls. But L=384 still executes inside the single `smkmesh` process. An honest host-memory OOM kill cannot be caught by `except Exception`; it leaves `complete=false` and a nonzero step status, so acceptance still reports FAIL instead of documented infeasibility. The requested process isolation remains absent. See [smoke.sh](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/smoke.sh:15).

   **Required:** isolate the large-mesh pilot with a resource limit that preserves its supervisor; persist resource termination as infeasible and skip its dependent gates. Unexpected failures must still fail.

   **Edit-introduced attribution problem:** [the timing handler](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/s3modes.py:487) marks the ROM infeasible even when a FOM timing subject exhausted memory. Acceptance then skips that ROM’s entire panel. This does not establish ROM infeasibility; record which subject failed.

2. **Staged-job checks pass.**
   - H200 ×1, `gpu`, 200G, **24 hours**, eight CPUs.
   - GPU preflight exits 42 on CPU; x64, highest precision, cluster venv and E0 paths configured; outputs/cache/tmp use paralab.
   - All **44 manifest entries** verify; all **40 provenance files** match committed HEAD; `COMMIT.txt` matches.
   - Both shell scripts pass syntax checks. Smoke runs both ROM tests, the FOM test, grid/FOM/profile/mesh modes, then acceptance. Step statuses are recorded; `ALL-DONE` means collection completed, not acceptance passed.

3. **24 hours is plausible, not demonstrated.** With both families qualifying and both ROM panels running, L=384 now schedules **16 full FOM solves**, down from 26: six calibration, two accuracy, eight warm/timed calls. At averages of 5/15/30 minutes, those consume **1.33/4/8 hours**, leaving substantial room within 24 hours for other work. ROM tables, ADT, compilation and earlier smoke steps remain additional, unmeasured costs.

No repository or git changes, GPU execution, or job submission.