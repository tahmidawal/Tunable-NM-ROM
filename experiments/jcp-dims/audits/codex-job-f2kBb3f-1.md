**f2kB: SUBMIT**  
**b3f: SUBMIT**

No blocking findings at HEAD `94ba57b8b917731d3de456fd658b6c498fa5eda0`. **Only one can be submitted now:** at `2026-10-18T21:58:11Z`, read-only `squeue` showed **2 lane jobs / 5 account jobs**. Recheck before each submission; the second must wait for a lane slot.

All paths below are relative to `experiments/jcp-dims/`.

- **Inventory passes.** `f2kA ∪ f2kB` equals `f2k.json` exactly, with no overlap: 96 + 64 settings. `f2kB` contains all 152 registered scheme–rung configurations for κ ∈ {3,8}, each on all38. `b3f` contains four shifted-lattice ladders, five Sobol ladders, the six-rung Smolyak ladder, bad-vector rollout, and twelve-rule fixgate. Gauss and lattice draw 0 correctly come from `m3a`. (`configs/f2kB.json:1`, `configs/b3f.json:1`)

- **Qualification and discovery pass.** Tight/tighter populations and same-draw successor links match registration: all38/dev6 in 2D, all64/cases 0–7 in 3D. Neither audited job contains square cells; the shared square certificate retains A2.1/A6.1 behavior. Both analyzers discover these jobs and their registered ladders. (`dacc.py:166`, `d3run.py:232`, `dcert.py:13`, `analyze2d.py:27`, `analyze3d.py:108`, `analyze3d.py:383`)

- **Certificates reproduce without writes**, including configuration hashes:
  
  | Job | Projection | Required/staged limit |
  |---|---:|---:|
  | f2kB | 1.7404015357 h | 2.75 h |
  | b3f | 2.4419265386 h | 3.75 h |

  Every `b3f` block is priced. Its fixgate receives **300 s**, versus **6.163 s** measured for the five-rule smoke block; production has twelve rules at the same setting and maximum point count. No unpriced fixgate or square-adaptive block is hidden here. (`resources.py:183`, `resources.py:216`, `checks/resources-f2kB.json:1`, `checks/resources-b3f.json:1`)

- **Memory and fixture pass.** Requested table sizes lie within the H200 smoke envelope: recorded peaks **29.661 GB** in 2D and **52.646 GB** in 3D, below 120 GB. Retained smoke metadata matches its archived checksums. The six-state 3D fixture matches both its runtime hash and the A2.2 source hash. (`d3run.py:49`, `d3run.py:326`, `checks/archive-listings/smk2.sha256-listing:1`, `checks/archive-listings/smk3.sha256-listing:1`)

- **Staging passes.** For each job, all **173 provenance entries** match HEAD blobs and staged bytes; every MANIFEST hash verifies. Scripts specify H200, `gpu`, separate paralab directories, GPU preflight exit 42, and f64/highest. (`runs/f2kB/PROVENANCE.json:1`, `runs/b3f/PROVENANCE.json:1`, both `run.sbatch:3`)

- **Budget passes.** Spent + live limits + job limit + remaining pending projections:
  - f2kB: **15.9648 + 20 + 2.75 + 7.4719 = 46.1867 h**
  - b3f: **15.9648 + 20 + 3.75 + 6.7704 = 46.4852 h**
  
  Reserving both would total **47.4948 h ≤ 55 h**. Mandatory pending inventory validation passes. (`runs/GPU-LEDGER.json:185`, `ledger.py:35`)

No files changed, jobs submitted, or scratch arrays copied.