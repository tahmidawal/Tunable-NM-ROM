**Overall: SUBMIT fb256/fb4096; FIX-FIRST time1024.** Read-only audit; no scripts executed, files changed, GPU use, or cluster jobs.

Paths below are relative to `experiments/jcp-qualify/`.

| Item | Verdict | Evidence |
|---|---|---|
| Prior must-fix 1: timing controls gate fronts | **RESOLVED** | Fronts now block on failed controls; doubled-work median uses drift-valid measurements. `analyze.py:384`, `:414`. |
| Prior must-fix 2: missing doubled-work fixtures | **RESOLVED** | A fixture present in both panels; refreshed certificates match config hashes and reservations. `configs/time256.json:424`; `configs/time4096.json:413`; `runs/GPU-LEDGER.json:115`. |
| Prior must-fix 3: fallback completion **and labels** before timing staging | **PARTIALLY** | fb1024 completed, but analysis contains 108 original configurations and **no fallback entry**. `runs/GPU-LEDGER.json:101`; `checks/analysis.json:2`; `analyze.py:189`. |
| fb256/fb4096 fallback selection | **PASS** | Recorded parents match Q2 failures. Registered successor ladders, all38 and sub16 populations match. `runs/GPU-LEDGER.json:131`; `configs/fb256.json:13`; `configs/fb4096.json:13`; `DESIGN.md:106`. |
| Resource certificates | **PASS** | Independently reproduced projections → limits: fb256 **0.362937 → 0.75 h**; fb4096 **0.335901 → 0.75 h**; time1024 **2.134049 → 3.25 h**. Exact config hashes match. `checks/resources-{fb256,fb4096,time1024}.json:1`. |
| Staged provenance | **PASS** | Each tree: **140 source files** equal their blobs at `d905327c2`; all **144 manifest entries** verify. `runs/{fb256,fb4096,time1024}/COMMIT.txt:1`, `MANIFEST.sha256:1`. |
| Sbatch | **PASS** | GPU partition, isolated paralab paths, f64/highest, GPU preflight; A100-80G/H200/H200; limits match reservations. `runs/{fb256,fb4096,time1024}/run.sbatch:3`. |
| Budget / concurrency | **PASS, static** | **2.5722 spent + 0 reserved + 14 pending = 16.5722 / 35 GPU-h**; no duplicate fallback allowances. Submission enforces lane cap **3**. Live queue not checked. `runs/GPU-LEDGER.json:113`; `cluster/submit.sh:18`. |
| time1024 subjects | **PASS** | 37 subjects, including registered fallback; doubled-work fixture present. `configs/time1024.json:390`, `:404`. |
| A2.2 terminal-state condition | **SATISFIED** | acc1024 and fb1024 completed. Fallback archive is complete, output checksums verify, log confirms GPU/f64/highest. Other meshes need not finish. `DESIGN.md:484`; `runs/GPU-LEDGER.json:41`, `:101`; `runs/fb1024/archive/logs/5397596.out:6`. |

Ranked must-fix:

1. **Compute and persist fb1024’s qualification labels before restaging time1024.** Terminal completion alone closes A2.2’s state condition; the prior audit also required computed labels, consistent with `DESIGN.md:406`. Current analysis lacks `1024|R128|LSPG|CNR|2.5|fb`. Preserve the original failure and report the fallback independently. `checks/analysis.json:12375`; `analyze.py:197`.

Minor: `all([])` still treats missing split/fused evidence as passing; explicitly reject empty evidence for fail-closed analysis. `analyze.py:384`.