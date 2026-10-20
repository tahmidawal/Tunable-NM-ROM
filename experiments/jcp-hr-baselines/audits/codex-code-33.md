**READY** for re-submitting the 3D Stage-B job within this focused audit. No blocking defect found.

Actual HEAD is `53b1b6355`; its only change from `0dc4f4406` is `HEARTBEAT.json`. Paths below are relative to `experiments/jcp-hr-baselines/`.

1. **CORRECT — nonblocking: A44.1 step recovery.** CPU-only extraction of `canon_dt`/`parse` verified every frozen member against its tuning configuration: **53/53 c3d64, 46/46 c2d256, 35/35 c2d1024**, with exact float equality. Recovered steps are 0.005 in 2D and 0.05/k for k = 5, 4, 3, 2 in 3D (`stageb.py:85–95`; `configs/tune_3d64.json:12`; both 2D tuning configs:11). Invalid, nonfinite and nonpositive probe values fail closed. The assertion requires ordinary Python execution without `-O`.

   The diff against audit 32’s `c820d3c4e` contains **only** the new canonicalizer and replacement of `float(dt)` in `parse`. Other printed-step uses are harmless for these memberships: `analyze.py:268–271` supplies numeric dt to tie-breaking (`stats.py:157–161`), whose ordering is preserved; `freeze_stageb.py:192–195` round-trips it into the same eight-digit lookup key. No additional erroneous numerical schedule use found.

2. **CORRECT — nonblocking: A44.2 cumulative accounting.** Both extra-attempt flags parse directory/attempt pairs correctly (`analyze.py:124–132`). Completed validation phases increase the lower bound; unfinished attempts propagate the lower-bound flag; missing allocations make the upper bound unknown (`analyze.py:97–112`). The report’s dimension-level budget consumes all three cumulative fields (`make_report.py:114–122`).

   The regenerated verdict matches results audit 1, item 7:
   **2.0223989954037176 + 1.034590999125503 = 3.0569899945292205 h**, lower-bound flag **true**, upper bound **5.2174 h** (`checks/VERDICT-c2d1024.json:27–45`). Extra-attempt completeness remains caller-supplied; the analyzer does not discover omitted attempts automatically.

3. **CORRECT — nonblocking for resubmission: s3d64a must count.** Its recorded validation phases total **0.19055472859569514 h** (`runs/s3d64a/archive/output/stageb.json:25–164`); an unfinished phase remains at `:211450`. Its measured allocation is **0.2239 h** (`runs/GPU-LEDGER.json:90–97`). Include it through `--extra-stageb=<archive/output>:s3d64a` in the eventual 3D analysis: add those completed phases to cumulative V, add its allocation to the upper bound, and retain **`V_cumulative_is_lower_bound=true`**. Omitting it would violate A44.2 (`DESIGN.md:1933–1935`) and invalidate cumulative budget claims, but does not require delaying the rerun.