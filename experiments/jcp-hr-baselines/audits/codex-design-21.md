**Overall: CLEAN.** Audited HEAD: `47bc19088ea4713121ff32646c32f9b5b83a2ebf`.

References below: `D` = `experiments/jcp-hr-baselines/DESIGN.md`; `T` = `experiments/jcp-hr-baselines/tune.py`.

1. **CORRECT — A31 resolves round-20 item 1; the tuning order and guards match.**

   A31 explicitly replaces extension-before-probe execution with probe-before-extension (`D:1618–1623`). The implementation runs HR grids, all-rank `none` with both solvers, the limiting comparator, probe runs, then extension (`T:401–405`, `T:432–501`).

   Probe configurations check elapsed time against `0.85 × budget` before starting (`T:398–399`, `T:449–450`). Extension admission applies that same guard plus the classified Stage-A search-allocation threshold (`T:470–482`). Once admitted, the extension may continue without another threshold check, as A31 explicitly permits.

   Budget-skipped probe configurations are `deferred`; triggered extension configurations are predeclared `deferred` when admission fails or `not_run`, stage `EXT`, when admitted (`T:449–450`, `T:486–492`). Completed configurations update individually; interrupted extension work therefore retains `not_run` (`T:385–390`). Its required incomplete reporting is explicit in `D:1622–1623`.

   **Concrete fix:** None required.

2. **CORRECT — no remaining substantive inconsistency found among A19–A31.**

   Later amendments consistently resolve the consequential changes:
   
   - Full-field 3D queries supersede mesh-free comparison wording (`D:1314–1324`, `D:1583–1586`).
   - Batch 1 remains diagnostic; batch 2 admits only the four qualified configurations, without claiming complete coverage (`D:1337–1342`, `D:1572–1577`, `D:1612`).
   - Memory acceptance is retrospective; resource bounds limit spending without promising completion (`D:1494–1500`, `D:1537–1540`).
   - Stage-B membership precedes execution; picks follow complete candidate timing, with incomplete roles left unfilled (`D:1502–1506`, `D:1594–1596`).
   - Batch-2 execution and accounting obligations are expressly retained (`D:1604–1612`).

   **Concrete fix:** None required.

3. **CORRECT — this CLEAN successor audit satisfies A19.4(a).**

   A26.7 expressly permits a CLEAN successor to discharge the round-14 design condition (`D:1264–1266`, `D:1512–1513`). With round-20’s substantive ambiguity resolved, this audit supplies that acceptance for the revision identified above.

   This satisfies **design acceptance only**. A28.7 separately retains revision-specific code/gate, resource-artifact and amended-smoke acceptance requirements (`D:1564–1566`).

   **Concrete fix:** None required.

No files changed, jobs submitted, GPU work or tests run, or scratch created.