**f2kA: FIX-FIRST. b2f: FIX-FIRST.** Both inventories pass; submission caps and audit-contract defects remain.

| Item | f2kA | b2f |
|---|---|---|
| Config vs design | **PASS:** 96 settings, 228 arms; κ={1,2,4}, all specified schemes/rungs | **PASS:** 246 arms; both panels, all family ladders/draws, two badlat rollouts |
| Populations and links | **PASS:** all38, tighter dev6, correct `rho_only`/`qp`; top rung without `qp` | **PASS:** same checks, including draw-specific successors |
| Q3 flags | **PASS:** 36 gauss256 arms enabled | **PASS:** disabled |
| Fixgate/fixture | N/A | **PASS:** two blocks, 18 rules each; fixture hash and coefficients match the pinned source |
| Analyzer keys/discovery | **PASS:** keys match manifest; `JOBS` includes f2kA | **PASS:** family keys and fixgate blocks consumed |
| Retry resolution | **FIX:** A9.3 checks incomplete | **FIX:** same |
| Certificate | **PASS arithmetic:** 10.857144 h → 16.5 h | **FIX:** 1.888120 h excludes fixgate work |
| Memory | **PASS:** accepted smoke peak 29.66 GB, below 120 GB | **PASS:** within smoke-tested table sizes |
| Staging/provenance | **PASS** | **PASS** |
| Submission now | **BLOCKED:** GPU-hour and concurrency caps | **BLOCKED:** concurrency caps |

Required corrections and scheduling conditions:

1. **f2kA exceeds the current reservation budget.** Using A10.1’s pending projections:
   `6.6179 spent + 28.25 live limits + 16.5 requested + 5.95 other pending = 57.3179 h > 55 h`.
   Wait for completed jobs to release reservations, then reconcile and recheck. b2f alone totals **52.7879 h**; both together total **58.4279 h**.

2. **Both concurrency caps are full.** The read-only queue check found **3/3 lane jobs and 8/8 account jobs**, including running m3b, k3a, and k3b. Neither audited namespace exists remotely yet.

3. **b2f’s certificate omits both fixgate blocks.** [resources.py:92](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/resources.py:92) prices settings and returns without pricing blocks. Restore justified block costs and regenerate the certificate/pending allowance. The previous model’s 120 seconds per block would yield **1.954786 h**, whose 1.5× allowance still fits **3 h**. The certificate note also incorrectly identifies the old A7.1 model.

4. **Retry merging does not enforce A9.3.** [analyze2d.py:104](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze2d.py:104) resolves numbered retries newest-first, but compares only seven code hashes—not matching inputs, arm configuration, or GPU type. It stores the selected attempt internally but omits it from published configuration records. Enforce those checks and persist the attempt identity.

The production calibration is broadly plausible: observed square-cell maxima were **1.18, 9.30, 16.07, and 17.90 seconds/case** for R128–R512. The registered R384 allowance of 15 seconds is slightly below its observed maximum, so it should not be described as a production maximum.

Both staged manifests verified **176/176 entries**, matched current sources, and matched available git content at `f48a017ac46be19a42a3ec384871a7186e78fd45`. Sbatch settings correctly specify H200, GPU preflight, f64/highest precision, isolated paralab paths, and the requested limits.

No files changed, no submissions, no scratch copies, and no LAB-LOG write.