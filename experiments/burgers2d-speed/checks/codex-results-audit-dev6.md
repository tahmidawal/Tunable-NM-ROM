**The frozen selections and Table-1 speedups are confirmed.** I independently recomputed all 56 candidates’ worst errors and both compile-mode medians from `quick` and `invocations`, and certificate statuses from saved `rho`/`meta`. I found one substantive reporting bug affecting six auxiliary engineering comparisons, not the selected rows.

No files were changed; no GPU or cluster work ran.

## CONFIRMED

All selected knobs are uncapped linear rungs with **no exact first step**: accurate \(R'=384\), fast \(R'=128\). Times below are medians over A1 ∪ A2; bold indicates the selected mode.

| Mesh | Pick | Worst evolved error (%) | Default ms | Graphs ms | Table-1 speedup | Own-accuracy speedup |
|---|---|---:|---:|---:|---:|---:|
| 256² | Accurate | 0.1656174305 | 46.015996 | **45.705815** | **0.369136×** | 0.369136× |
| 256² | Fast | 1.5967685048 | 17.473703 | **17.442230** | **0.967289×** | 0.503331× |
| 512² | Accurate | 0.1946610948 | **45.911947** | 46.130815 | **0.603422×** | 0.603422× |
| 512² | Fast | 1.7479877169 | **17.390011** | 17.417963 | **1.593115×** | 0.820638× |
| 1024² | Accurate | 0.2108198098 | **49.519309** | 49.556280 | **1.456826×** | 1.456826× |
| 1024² | Fast | 1.8280675824 | 21.317557 | **21.305735** | **3.385990×** | 1.769265× |

The fastest eligible Table-1 FOM is **`lean_nt3e-3_l3e-3_dt005__graphs` at every mesh**, requiring `nonlinear_converged` on all six cases.

| Mesh / job | FOM ms | FOM worst error (%) | Same-job q=0 fast bar (%) | Confirmed candidates |
|---|---:|---:|---:|---:|
| 256² / 4241033 | 16.871676 | 0.0484852973 | 1.8891722546 | 13/22 |
| 512² / 4241035 | 27.704285 | 0.0497105474 | 2.1375178937 | 20/22 |
| 1024² / 4241031 | 72.141014 | 0.0483020287 | 2.2883561038 | 12/12 |

Applying §6 independently reproduces all `selection-*.json` picks and summary speedups exactly. Selection files’ recorded summary hashes also match.

**Certificates:** all six selected knobs pass every certification draw and the confirmation draw under \(k\ge\max(j,1)=1\). Each has 400 eligible states per certification draw and 800 in confirmation.

| Mesh | Accurate: max ρ certification / confirmation | Fast: max ρ certification / confirmation |
|---|---:|---:|
| 256² | 0.088823 / 0.093533 | 0.080032 / 0.102042 |
| 512² | 0.068034 / 0.068382 | 0.026414 / 0.052610 |
| 1024² | 0.059492 / 0.058437 | 0.022094 / 0.049229 |

These are below 0.116. This independently checks saved certificate values and eligibility; it does not recompute the underlying advection projections.

**Parity and timing gates:**

| Mesh | ROM pairs / maximum relative difference | A2/A1 range | Maximum case-controlled neighbour ratio | Evaluable / not evaluable |
|---|---|---|---:|---:|
| 256² | 30 / 5.3316e−15 | 0.985237–1.021121 | 1.014793 | 126 / 2 |
| 512² | 30 / 6.0837e−12 | 0.972274–1.001056 | 1.014087 | 126 / 2 |
| 1024² | 20 / 2.2226e−15 | 0.981056–1.005017 | 1.006054 | 87 / 3 |

Every stored ROM parity case passes the \(10^{-10}\) field/state bar and integer checks. All 15 FOM mode pairs per mesh have zero relative field difference and identical Newton iterations. Repetitions are uniformly 5 per case per ROM phase and 3 per FOM case; all invocation records report SHA checks against quick outputs passing.

I recomputed the A0 neighbour statistic using each invocation’s subject/case/phase median for normalization. All evaluable comparisons pass. The unevaluable comparisons are FOM subjects with fewer than three observations on one predecessor side, as disclosed by the summaries.

**Paper §6.3, 1024²:**

- General: **58.548400 ms**, worst error **1.8278119446%**, iterations `[60,68,60,59,55,59]`.
- Parent-text fast: **31.590800 ms**, worst error **1.8280675824%**, iterations `[51,55,52,52,50,51]`.
- **X = 1.853337×**.
- Relative worst-error difference: **0.013984%**, comfortably inside the registered 5% criterion.

## DISCREPANCIES

1. **Six incorrect “LM budget 1” engineering comparisons.**  
   In `checks/b{256,512,1024}-summary.json`, key  
   `selection.parent_settings_this_job[...budget1__parent].engineered_same_knob`  
   points to **uncapped** engineered arms instead of `cap1` twins.

   The matching predicate in [audit_b2speed.py:481](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-23-burgers2d-speed/experiments/burgers2d-speed/audit_b2speed.py:481) uses knob metadata that does not distinguish the parent’s budget override. The generator reproduces these incorrect pairings.

   Correct comparisons are:

   | Mesh | Budget-1 parent | Correct engineered default / graphs ms | Correct factor | Reported factor |
   |---|---|---:|---:|---:|
   | 256² | R128 linear | 16.996841 / 17.116515 | **1.539313×** | 1.50× |
   | 256² | R512 q=0 | 21.104698 / 20.980991 | **1.182139×** | 0.78× |
   | 512² | R128 linear | 17.054003 / 17.191381 | **1.543065×** | 1.51× |
   | 512² | R512 q=0 | 22.475885 / 22.298208 | **1.162054×** | 0.77× |
   | 1024² | R128 linear | 17.926482 / 18.091705 | **1.733728×** | 1.46× |
   | 1024² | R512 q=0 | 24.500094 / 23.822079 | **1.177051×** | 0.81× |

   Thus the apparent q=0 budget-1 slowdowns are false. Parity comparisons use the correct twins; headline selections are unaffected.

2. **The report omits the explicit fast near-tie listing required by §6.**  
   `checks/b256-summary.json:selection.fast_near_ties` correctly contains R128 x1 graphs: **18.215264 ms**, **4.431%** slower than the selected fast knob. It appears in the full candidate table but is not identified as a near-tie. No eligible near-ties occur at 512² or 1024².

3. **`PROGRESS.md` is stale:** its job table still says all three dev panels are “submitted,” despite archived results and the completed report.

## WORDING ISSUES

- **Before/after is a combined deployment-and-engineering comparison.** At 256²/512², “after” removes x1; parity establishes iterate preservation only at the **same setting**. The 512² accurate iteration vector actually changes from `[53,53,49,55,45,51]` to `[53,53,49,54,45,51]`.

- **The x1/no-x1 accurate errors are not exact ties.** No-x1 wins the strict error ordering by only **1.1926e−8** and **5.6848e−9 percentage points** at 256²/512². Selection follows the rule; these differences should not be presented as meaningful accuracy improvements.

- **The headline fast speedups use the accurate row’s FOM comparator.** They do not establish superiority over the cheapest equally accurate FOM for the fast row. Those ratios are **0.503×, 0.821×, 1.769×**. In particular, 512²’s headline **1.59×** is not an own-accuracy speedup.

- **Before timings are valid same-job replays**, not the historical times in DESIGN §1. Parent text uses default compilation; engineered candidates and FOM use both-mode minima. This distinction should remain beside the main comparison.

- In the report glossary, **“\(M=4R'\) here” applies only to linear rungs**; q=0 uses \(M=64\). Calling \(k\ge j+1\) a “stricter reading” is also misleading: for x1 it excludes additional states and makes passing easier.

- The opening says settings are “then run once” on hold64, while the held-out section says **“Not yet run.”** Use future tense. Certification populations participate in selection; they are not fresh held-out confirmation.

- **No evidence of post-hoc candidate or selection-rule changes was found.** The archived jobs identify the A0 commit, and the later DESIGN change only corrects a timestamp. Choosing modes and knobs from dev6 is explicitly registered, but remains development selection pending hold64.