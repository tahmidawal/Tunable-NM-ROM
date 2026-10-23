**The collection blocker remains, even with separate `t` names.**

Both restart bodies copy the entire original archive into `out/cell`, including `_job` ([W40 body, line 4](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/cluster/bodies/W40_A3t.sh:4)). Read-only SSH confirmed both source archives contain old Slurm files and stage logs.

[collect.sh:46](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-22-imp-box/experiments/imp-box/cluster/collect.sh:46) copies that inherited `_job` into the fresh `t` archive. Line 51 then overlays restart evidence there; old `slurm-6203321.out` / `slurm-6203563.out` and stage logs remain, so the inventory comparison still fails.

Exclude `_job` from both restores, then update the released body hashes.

Other checks passed:

- Report and stop rule prefer `t` results when `final.json` exists.
- Both body hashes match `RELEASE.json`: H200, 240G, `06:00:00`.
- Bodies clear completion/selection markers and run only time and controls; GPU preflight, f64, highest precision are configured.
- Certificate matches ledger: **41.3547 + 12 + 6 + 17.76 = 77.1147 ≤ 87**. W08_A2 production remains deferred.

No files changed, log appended, or jobs submitted.

**DO-NOT-SUBMIT**