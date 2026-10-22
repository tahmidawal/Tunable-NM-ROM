**DO NOT SUBMIT f1024.** The staged scope and provenance are correct, but not every audit-8 production blocker is closed, and the resource certificate exceeds the job’s effective driver timeout.

Reviewed changes from `011f38036` through `da29cc86d`, the archived smoke outputs, and the untracked f1024 stage. No files changed, no SSH, no submissions, and no lab-log writes.

| Item | Verdict | Finding |
|---|---|---|
| Pooled selection | **CORRECT** | `dev_qualified` now pools stationarity counts. An extracted-function probe accepts 1 bad step among 300 and rejects 4 among 300. |
| Offline native qualification | **WRONG — incomplete** | `_native_recomputed` is recorded separately without checking agreement with the row predicates used downstream. Native anchor reconstruction omits state finiteness, damping, known exit reasons, schedules, and decoded distances. Bank tight–tighter distances also remain row-derived. See [checks.py:317](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/checks.py:317). |
| C19 qualification replay | **WRONG — incomplete** | The new replay computes κ2→κ8 rollout distances, but not the complete qualification label: successor certification and own-state force ratios are missing. Its result does not gate `C['C19']`, which remains the synthetic successor-map check. See [checks.py:424](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/checks.py:424). |
| f4096 control link | **CORRECT structurally** | The configured SHA-256 matches `checks/smk1.json`; the checker requires its `accepted` flag and retains per-job timing/scoring checks. Its evidentiary strength still depends on repairing smoke acceptance. |
| Refinement validity | **NEEDS-RESTATEMENT** | Audit-8 omissions remain: LSPG refinement uses a reduced validity predicate; POD refinement still omits explicit stage-finiteness checking. Resolved diagnostic classifications therefore lack the complete production validity checks. |
| Diagnostic/reporting outputs | **WRONG — unfinished** | Same-rank POD picks exist, but the registered shared-integrator comparison is still absent. Smoke F5 output still emits `order=pass/fail` and `gain_fired`, despite its descriptive flag. |
| Independent φ documentation | **NEEDS-RESTATEMENT** | DESIGN still says 12 Taylor terms below 0.2; implementation uses 40 below 0.5. |

The smoke evidence supports the following narrower conclusions:

| Item | Verdict | Evidence |
|---|---|---|
| Archive integrity and execution | **CORRECT** | Both archive checksum sets pass. Both logs show H200, GPU backend, f64, highest precision, successful completion, and empty stderr. |
| Real-size in-job mutant controls | **CORRECT** | Raw witnesses support the must-fire detectors at both smk1 ranks. Examples: C2m rejects 50/50 steps; C3m reconstruction mismatches are approximately 0.00106 and 0.00142; C7m fires using the prescribed escalation. |
| C9 self-timing | **CORRECT** | Recomputed from saved repetitions: **1.00958** at r=128 and **0.99637** at r=384, both within [0.9, 1.1]. |
| Complete control acceptance | **WRONG — overstated** | `accepted=true` is genuinely stored, but does not establish closure of the incomplete C19 and offline-verification requirements above. |
| A6 | **NEEDS-RESTATEMENT** | Recomputed C3m slopes are **3.75021**, not 3.77, and **2.02772**. Both reference-convergence ratios satisfy the registered threshold. A6 transparently records a post-smoke relaxation, but it does rescue a failed positive control. “Known stiff order reduction” is stronger than this evidence establishes; describe it as a plausible explanation and retain the original failure explicitly. |

A6 can define a revised check before production, but it cannot make the original all-rank order check pass retroactively. The observed second-order behavior is specific to this case and tested step-size range.

The resource and job findings are decisive:

| Item | Verdict | Finding |
|---|---|---|
| Accounting repairs | **CORRECT** | Setup/control double counting is repaired, and an H200 compatibility check exists. Both stored projections reproduce exactly in an in-memory calculation. |
| Cost certificates | **WRONG as sufficient coverage evidence** | Classification allowance prices only **two tripped runs per arm/rank**, while production classifies every unstable case. POD conditional refinements lack a separately justified allowance. POD marginal pricing uses the median of all logged member increments, despite the comment claiming second-step measurements. The single-case f4096 marginal assumption, `elapsed/3`, is not an established conservative bound. |
| f1024 timeout coverage | **WRONG** | Certified projection: **7,108.77 s**. Batch driver timeout: **6,960 s**. `covered=true` compares against the two-hour allocation, overlooking the four-minute deduction. See [run.sbatch:32](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/runs/f1024/stage/run.sbatch:32). |
| Budget sequencing | **CORRECT** | Recorded spend is **0.6303 GPU-h**; reserving 2.0 h leaves **1.3697 h**. A subsequent 1.5-hour reservation requires f1024 actual usage ≤ **1.8697 h**, absent other charges. Waiting for actual usage is necessary. The f4096 deferrals follow the brief. |
| Staged scope and provenance | **CORRECT** | All manifest hashes and provenance entries match recorded git content. Configuration requests all38, both ranks, all specified bank arms, POD ladder/native picks, controls, and headline six-case × three-repetition timing. GPU partition, H200, precision, preflight, and output namespace are correct. |

Before submission: close the remaining offline/control and diagnostic requirements, restate A6 accurately, justify the conditional-work pricing, and make the certified projection fit the **actual driver timeout**. Then regenerate and re-audit the stage.