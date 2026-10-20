**HOLD.** Reviewed HEAD `3ab0e14035734ba1b3c1cc21ffcd7994b5951987` under A5. One blocking item remains.

| Audit-6 item | Verdict |
|---|---|
| 1. Infeasible L=384 pilot causes false FAIL | **NOT RESOLVED** |
| 2. G4 incorrectly requires exact agreement | **RESOLVED** — both predicates accept `≤2.5e-5`. |
| 3. Invalid ADT reference solves can pass | **RESOLVED** — standalone, grid and reduced-rule validity checks added. |
| 4. Mesh ADT agreement lacks a bound | **RESOLVED** — enforces `≤2.5e-5`. |
| 5. K2 must fail separately on every case | **RESOLVED** — uses pooled configuration-level ineligibility. |

The remaining blocker is incomplete resource-failure handling. In [s3modes.py:434](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/s3modes.py:434), tight/tighter ADT calls remain outside the new exception handlers; timing calls are also unprotected. An honest resource exhaustion there aborts `smkmesh`, leaving `complete=false` and a nonzero step status. Acceptance then fails instead of reporting the mesh as infeasible. Process-level OOM is likewise not isolated. The new checker correctly accepts resource failures **when recorded**, but the producer does not reliably record them.

**Required fix:** isolate the large-mesh pilot and persist a feasible/infeasible outcome covering its entire execution. Skip dependent requirements for documented resource failures; retain failures for unexpected errors.

The staged-job checks otherwise pass:

- One H200, `gpu` partition, `200G`, 12 hours; GPU preflight exits 42 on CPU.
- x64 and highest precision enabled; E0 store configured; logs, outputs and caches point into the requested paralab namespace.
- All **44 manifest entries** verify; all **40 provenance-listed files** match committed HEAD and `COMMIT.txt`.
- CPU import checks from `/tmp` passed for every smoke entry point and E0 loader. All loaded repository modules resolved inside the staged tree.
- Collection writes `OUTPUTS.sha256` and prints `ALL-DONE`; acceptance remains separately determined by `ACCEPTANCE.json`.
- Configurations cover the required smoke gates, controls and L=384 pilot work, with A4’s explicit offline/deferred checks preserved. Per-run, certification, compile-estimate, panel and table-build timing evidence is implemented.

**Runtime assessment:** 12-hour completion is plausible but unverified. L=384 alone schedules **26 full FOM solves** when both families qualify. At an average 5/15/30 minutes each, that consumes **2.17/6.5/13 hours**, before ROM tables, ADT, grid, profiling and compilation. No measured L=384 runtime supports a narrower estimate; the design’s original one-hour smoke estimate is not substantiated for this bundle.

No repository files or git state changed; no GPU execution or jobs submitted.