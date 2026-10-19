SUBMIT — audit passes; **wait for a lane slot before submitting.**

Paths below are relative to `experiments/jcp-dims/`.

- **Selection reproduced:** per-case E0 endpoint comparisons and the 3% cost tie-break select `M2_512|gl32|qn|BDF2|0.016666667|k4` for all three comparators. Matching candidates: **11 / 18 / 8**; each cheapest-cost tie set contains only this winner. Screening cost **0.8407922771**, median/worst E0 **0.0104466921 / 0.0424121122**. `dstats.py:226`, `checks/analysis-3d.json:17985`.
- **Qualification verified:** `cert_of_job` passes all **102/102** t3s subjects and agrees with every screening Q1 flag. Winner’s round-5 Q2 passes; screening costs reproduce from invocation arrays. Panel valid; T1 median **1.991005818**. `analyze3d.py:561`, `checks/analysis-3d.json:16967`.
- **Secondary pairs correct:** qualified BE comparators are M2_512/gl32, W768/lat32768, W1024/gl40; all select the same winner. `checks/analysis-3d.json:18212`.
- **Panels correct:** six panels, A=comparator, B=winner, **24 A–B–A invocations each**. Five distinct subjects receive certification and tighter anchors on cases 0–7 before timing. Generator settings/blocks reproduce in memory. Full mesh-path query and per-call burn verified. T1 is registered in t3s, not t3c. `configs/t3c.json:46`, `configs/t3c.json:168`, `d3run.py:359`, `d3run.py:394`, `DESIGN.md:342`.
- **Resources reproduce:** **1.535462668 h → 2.5 h**, including config SHA-256 match. Budget: **29.1701 + 8 + 0.25 + 3.75 + 2.5 = 43.6701 ≤ 55 GPU-h**; pending allowance **2.0 ≥ projection**. `checks/resources-t3c.json:6`, `runs/GPU-LEDGER.json:293`.
- **Staging passes:** manifest verified; all **177 provenance entries** match staged files and HEAD blobs at `09f2d4b9f114967272a763c77746814998aabc03`. H200, gpu partition, exit-42 GPU preflight, f64/highest, and paralab paths verified. `runs/t3c/PROVENANCE.json:1`, `runs/t3c/run.sbatch:3`.
- **Operational hold:** live SSH queue shows t2 **6056624 running**, f2l **6057843 pending**, b3m **6065773 pending**: lane **3/3**. Recheck for a free slot before submission. `runs/GPU-LEDGER.json:259`.

No files changed, scratch created, or jobs submitted.