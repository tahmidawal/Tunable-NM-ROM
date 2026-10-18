**Overall: NEEDS-FIX.** Several round-1 fixes work, but confirmation and incomplete-evidence paths still permit false acceptance.

Read-only audit completed: no files changed, jobs submitted, production results opened, or scratch files created. Python checks used CPU-only execution with bytecode writes disabled.

“CORRECT” below means the round-1 finding is resolved; “NEEDS-FIX” means it remains wholly or partly unresolved.

| Round-1 finding | Disposition | Round-2 assessment |
|---|---|---|
| 1. Adaptive Boolean crash | **CORRECT** | Comparisons now use per-case validity. |
| 2. Registered populations | **NEEDS-FIX** | 2D missing-check regression fixed; 3D validity inventories remain incomplete. |
| 3. 2D timing qualification | **NEEDS-FIX** | G-cert, dev6 coverage and mandatory successors fixed; invocation inventory still trusted indirectly. |
| 4. H2q certification/front | **NEEDS-FIX** | Screening certification, front gating, pair identity and tighter-case generation fixed; confirmation certification remains unchecked. |
| 5. Independent checks | **NEEDS-FIX** | 2D ρ agreement is enforced on fresh computation; 3D recomputation remains incomplete. |
| 6. Shapley/elasticity | **NEEDS-FIX** | Shapley guards and elasticity population fixed; elasticity uses 2,000 instead of 10,000 resamples. |
| 7. Family/spec completeness | **CORRECT** | Manifest ladders preserve missing draws/specs; “not reached” is separated from “none.” |
| 8. Bad-vector/Gauss gates | **CORRECT** | Control gating and explicit target-self evidence added. |
| 9. LMF stability | **CORRECT** | Fresh execution gates scoring and LMF verdicts on stability; see the cache defect below. |
| 10. Q2 state/identity | **NEEDS-FIX** | Top-two precedence and 2D identity checks fixed; 3D q++ remains insufficiently checked. |
| 11. B6 blocking | **NEEDS-FIX** | Secondary blocking improved; primary transitions and comparison-control mapping remain incomplete. |
| 12. Required 3D outputs | **NEEDS-FIX** | Secondary transitions and open inventory added; qualified-comparator ratios remain unfinished. |

Ranked remaining defects follow. Paths are under `experiments/jcp-dims/`.

1. **H2q can pass despite failed confirmation-job certification.**  
   [analyze3d.py:483](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze3d.py:483) reads confirmation timing validity and pair identity but never evaluates `t3c` certification blocks. The runner’s panel-valid predicate checks identity, execution validity and finite fields—not numerical certificates or anchor distances.

   **Reproduced:** the actual `after_screen` function returned `pass` with a confirmation block reporting 100% non-certified steps and an anchor distance of 1. Apply fresh confirmation-job Q1 to both selected subjects on all eight cases before accepting the panel.

2. **3D Q2 accepts inconsistent q++ evidence.**  
   [analyze3d.py:228](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze3d.py:228) checks q++ case IDs but consumes the successor’s stored `dist_next` and `next_valid`. It neither validates q++ setting/spec/M/mode identity nor independently recomputes the check distance.

   **Reproduced:** changing q++’s setting, M, spec and coefficients while retaining the stored distances still returned `pass`. Recompute q+→q++ on cases 0–7 and validate both successors’ identities.

3. **Incomplete populations still receive terminal labels; 3D Q1 can accept empty validity evidence.**  
   [analyze3d.py:128](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze3d.py:128) validates case names and distance count, but not the lengths of production/tight/tighter validity arrays.

   **Reproduced:** with registered case IDs and distances, empty `valid`, `adt_valid` and `adt_tighter_valid` arrays produced an all-true Q1 result.

   Separately, `analyze2d.py:122–134` and `analyze3d.py:291–304` classify a present entry with missing registered cases as “not qualified,” rather than A1.5’s **missing**. That lets `mstar` advance beyond an incomplete lower candidate. Enforce complete inventories and propagate missing-population status before Q labels.

4. **B6 control blocking remains incomplete.**  
   [analyze2d.py:417](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze2d.py:417) and `analyze3d.py:344–352` retain ordinary primary transition verdicts when their control is unavailable or does not fire; only range summaries are blocked.

   Additionally, `analyze2d.py:432` uses LSPG-CN’s control for both GAL comparisons, including GAL versus LSPG-CN-R. `analyze3d.py:375` consults only adaptive LSPG’s control. Propagate the applicable A2.1 form controls to each verdict, retaining raw statistics separately.

5. **2D timing inventory validation is still insufficient.**  
   [analyze2d.py:228](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze2d.py:228) trusts `counts_complete`; the driver computes that flag from aggregate split-invocation counts, not exact fused `(subject, case, repetition)` coverage.

   **Reproduced:** six fused invocations—one per dev6 case instead of two—with an empty split inventory and `counts_complete=True` returned both `usable=True` and `qualified=True`. Validate the exact registered inventories from the records.

6. **Independent 3D checks remain short of the contract.**  
   [analyze3d.py:145](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze3d.py:145) recomputes ρ only on case 0, only for passing ladder Q2 candidates, and never for round jobs. A6.2 specifies two cases per job where feasible.

   Also, adaptive Q1 at `:134–139` trusts stored tight/tighter distances despite archived coefficient trajectories being available. Complete the independent checks and make disagreement block acceptance.

7. **Qualified-comparator reporting is still selection-only.**  
   [analyze3d.py:503](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze3d.py:503) labels `Cq` output “screening ratio only,” but emits a selection result without a ratio. `make_configs_late.py:101` generates confirmation panels only from `h2q`, excluding `h2q_Cq`.

   Thus §6.5’s secondary qualified-versus-qualified ratio is unavailable. Carry those selected pairs into the required measurement/reporting path.

8. **New defect: cached results can bypass the repaired analysis and LMF gate.**  
   [analyze2d.py:55](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze2d.py:55) accepts a cache solely by NPZ hash. Changes to analysis code, expected populations, JSON evidence or references do not invalidate it. A cached scored LMF entry is also returned when `score=False`.

   Version the complete analysis dependencies and enforce the LMF scoring restriction on cache hits.

9. **Elasticity bootstrap count differs from registration.**  
   [analyze2d.py:499](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/analyze2d.py:499) performs 2,000 resamples; §5 specifies 10,000. A1.8 changes the guards and resampling population, not that count.

| File | Verdict |
|---|---|
| `dscore.py` | **READY** for the reviewed fixes |
| `analyze2d.py` | **NEEDS-FIX** |
| `analyze3d.py` | **NEEDS-FIX** |
| `score3.py` | **READY** for the mapping edit |
| `make_configs_late.py` | **NEEDS-FIX** |
| `make_round.py` | **READY** |
| `waitjobs.sh` | **READY** |
| `d3run.py` | **READY** for the adaptive `Wstore` change |

Passing checks included zero missing 2D manifest/config keys, the restricted `score3.py` diff, adaptive scoring-key consistency, predecessor-label coverage, and `bash -n`. On the permitted smoke archive, the original Q1 passed; deleting its tighter-check population now correctly failed Q1.