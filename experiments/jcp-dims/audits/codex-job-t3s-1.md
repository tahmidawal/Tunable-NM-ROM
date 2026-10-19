FIX-FIRST

All paths below are relative to `experiments/jcp-dims/`.

1. **18 qualified configurations are omitted.** `analyze3d.py:546` restricts `qualified_3d` to five deployment setting names, excluding configurations already marked `qualified: true`: eight adaptive κ configurations, seven fixed-sweep κ configurations, and three `W256M2052` configurations. Examples: `checks/analysis-3d.json:9824` and `:11197`. The staging exactly matches the supplied 83-entry list, but that list violates DESIGN §4’s “every numerically qualified” scope. Fix the inventory, regenerate certification blocks, and reprice—including adaptive subjects.

2. **Doubled-work control is executed but not enforced.** `d3run.py:363` runs C_old twice, but `dstats.py:272–304` checks only hashes, inventory, certification links, positive ratios and drift. It never requires the T1 ratio to lie in `[1.8, 2.2]`; `analyze3d.py:610` trusts this panel-valid flag. A failed control can therefore yield accepted screening costs.

3. **Confirmation reservation is underpriced.** `runs/GPU-LEDGER.json:252` reserves 1 hour, while `make_configs_late.py:113–124` allows three comparator pairs plus three Cᵠ pairs. Using `resources.py:188,214`, six panels plus startup already project **1.50798 hours before certification**. Three pairs alone project 0.83732 hours before certification. Increase the pending allowance using a defensible full-work estimate.

4. **Live concurrency remains unverified.** Both read-only `ssh tufts-login squeue -u tawal01` attempts failed: connection reset, then banner timeout. Neither lane `<3` nor account `<8` is established.

Checks that passed for the **current, incomplete inventory**:

- All four qualify passes, three comparators and five Cᵠ candidates are covered. All 38 certification blocks precede screening and run tight/tighter on every panel case. Panel identity links, full-payload hashing, A–B–A ordering and drift checks are present (`make_configs_late.py:85`; `d3run.py:233,371`).
- Reproduced **6.5897413693 h → 10 h** without writes. Every block is priced; current subjects contain no adaptive `|prod` entries. Table sizes fit the smoke envelope, whose recorded peak is **52.646 GB** (`resources.py:183`; `checks/resources-t3s.json:6`).
- All **175 provenance entries** match HEAD `cd83db6e9`; all **179 manifest hashes** pass. H200, gpu partition, exit-42 preflight, f64/highest and paralab paths are correct (`runs/t3s/run.sbatch:3`).
- Recorded reservation arithmetic passes: **21.6656 + 17 + 10 + 1 + 2 = 51.6656 ≤ 55**. The t2 allowance remains provisional pending its generated certificate (`runs/GPU-LEDGER.json:248`; `DESIGN.md:787`).

No files changed, submissions made, or scratch arrays copied.