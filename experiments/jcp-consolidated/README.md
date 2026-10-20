# JCP consolidated branch (2026-10-20)

`exp/2026-10-20-jcp-consolidated` is the fork point `21c175a1b` (`exp/2026-10-01-ns3d-coordnet-bank`) with every JCP-campaign lane merged in, without conflicts. Each lane keeps its own folder under `experiments/`.

| Lane branch | Tip merged |
|---|---|
| exp/2026-10-01-quadrature-study | 84d28e303 |
| exp/2026-10-01-quadrature-burgers3d | aab6b6927 |
| exp/2026-10-06-jcp-fom-pareto | 73bab9c31 |
| exp/2026-10-06-jcp-references | cc04f4b54 |
| exp/2026-10-06-jcp-mechanism | 712d682d8 |
| exp/2026-10-06-jcp-hr-baselines | 2e3abac47 |
| exp/2026-10-08-jcp-time2 | a7a12528f |
| exp/2026-10-08-jcp-wide-bank | f050dffa6 |
| exp/2026-10-08-jcp-smooth-bank | 12f620782 |
| exp/2026-10-13-jcp-combined | a3c9e170d |
| exp/2026-10-14-jcp-speed3d | 739154488 |
| exp/2026-10-17-jcp-p1-scaling | 5b782026b |
| exp/2026-10-18-jcp-qualify | 250b20d91 |
| exp/2026-10-18-jcp-fom2d | 12103d331 |
| exp/2026-10-18-jcp-heads | 45d28fac3 |
| exp/2026-10-18-jcp-dims | a99906fe6 |
| exp/2026-10-18-jcp-budget | d70608191 |
| exp/2026-10-19-jcp-regimes | 70ce5f0f2 |
| exp/2026-10-19-jcp-robust | cbf574373 |

**Checks.** Every lane tip is an ancestor of this branch. All 16,506 files changed by the lanes are byte-identical to their lane tips.

**Not in git** (kept on disk or on the cluster):
- Raw arrays over the size limits stay in each lane's worktree, listed by SHA256 manifests (for example `experiments/jcp-wide-bank/runs/j1/LARGE_FILES.sha256`).
- Campaign raw archives are on the cluster at `/cluster/tufts/paralab/tawal01/burgers-archive/<lane>/`.

**Kept on `main`:** the campaign plan, the final report and its PDF, and the slide deck, under `reports/2026-10-17-full-experiment-plan/`. The lab log is also on `main`.

**Backup.** Backed up to `origin/codeonly/exp/2026-10-20-jcp-consolidated` with `bash experiments/jcp-consolidated/sync_github.sh`.
