FIX-FIRST

Paths below are relative to `experiments/jcp-dims/`.

1. **t2 reservation is stale.** `runs/GPU-LEDGER.json:253` reserves **4.8 h**, but generating the config in memory from the current 2D analysis produces **284 subjects and 5.04893 h**, versus A12.3’s 268 subjects. Update the reservation and estimate (`DESIGN.md:814`). The cap still fits: **28.6212 + 12 + 2 + 5.04893 = 47.67013 ≤ 55**.

2. **A13’s current alias is valid, but its future invalidation guarantee is incomplete.** Independently reproduced the old `c6e315cf…` hash; the dstats diff contains only the T1 hunk. However, `analyze2d.py:44` binds `q2_label`’s source without its `BAR` and `RHO_BAR` globals (`dstats.py:89`). Changing a threshold leaves `_code_version()` unchanged. Bind those values or narrow A13.1’s guarantee.

Checks passed:

- **Inventory and certification:** reconstructed **101 qualified configurations**; staged inventory equals their union with comparators exactly. There are **101 ordinary subjects plus T1**, with C_old as A. All certification blocks and settings match regeneration; eight adaptive specs correctly include `|prod` (`analyze3d.py:548`, `make_configs_late.py:85`).
- **Pricing:** reproduced **7.968066 h → 12 h**. k3b adaptive units measured **1.746–3.672 s**, below the **11.178 s** smoke invocation price. Six confirmation panels plus conservatively priced certification project **1.59517 h**, supporting t3c’s **2 h** (`resources.py:183`).
- **T1:** enforced; all **14 panel tests passed**, executed without the test runner’s file-writing step (`dstats.py:302`, `test_dims.py:230`).
- **Staging:** all **175 provenance entries match HEAD `f7e1977a1`**; all **179 manifest hashes pass**. H200, gpu partition, exit-42 preflight, f64/highest, paralab paths and 12-hour limit are correct (`runs/t3s/run.sbatch:3`).
- **Concurrency:** SSH retry succeeded; `squeue -u tawal01` returned no jobs: lane **0 < 3**, account **0 < 8**.

No files changed by this audit, submissions made, or scratch arrays copied.