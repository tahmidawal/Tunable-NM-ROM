# Poisson staged-accuracy acceptance

Status: COMPLETE AND AUDITED. Job 3563323 completed on NVIDIA A100 80GB PCIe / pax049. Scientific source `db66d194efa56a9fecbee9f0c116f42d7f356e7e`. Full-field, operator, initialization, solver/exit, training/rank/QR, phase-time-match and checkpoint identity checks passed.

Machine panel: `panel.json`; native recorded invocations: `result.json`; owner audit: `audit.json`; exact remote cleanup: `cleanup.json`; archive manifest: `ARCHIVE.json`; stream restoration proof: `restoration-audit.json`. Every checkpoint is also retained in `checkpoints/`. Full raw outputs remain in `cluster/out/pilot`, restorable from the tracked archive parts.

Fixed-rank staging did not improve worst-case accuracy across all development sources. No model passes the complete physical target. All online solves satisfy the declared stationarity/linear/rank gates, and every bounded head-fit case has a stationary candidate. Original/new development groups remain separate. See generated `summary.md` and full `panel.json` for exact values and same-job FOM controls.

The CPU `pod-capacity-diagnostic.json` supports testing a larger learned spatial bank but does not prove rank64 minimax impossibility or guarantee a neural result. The coordinator approved isolated r128/k16 training versus phase-time-matched r64 under `CAPACITY-ACCURACY-DESIGN.md`; that follow-up remains separate from this accepted archive. Final cases remain unopened. No merge.
