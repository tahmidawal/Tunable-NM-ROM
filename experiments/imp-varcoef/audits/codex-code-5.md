**Both Round 4 gate fixes are CORRECT. The resource certificates are WRONG as conservative budget projections.**

Paths below are relative to `experiments/imp-varcoef/`.

| Item | Verdict | Evidence |
|---|---|---|
| Kernel reproduction | **CORRECT** | `gate.py:103–104` now requires both `kernel_query_ok is True` and `repro is True`, in addition to existing coverage, validity, veto and timing checks. CPU fixtures accepted the passing kernel and rejected failed reproduction. |
| Paired PSI numerator | **CORRECT** | `gate.py:245–252` requires complete timing, absence from the veto set, and `scored_ok()` for C100. Invalid numerators now produce an unavailable ratio. |

Resource findings:

1. **WRONG — reference projection ignores requested levels.** `resources.py:65–69` never reads `cfg['levels']`. The registered fallback at 2048/4096/8192 therefore receives exactly the same estimate as 1024/2048/4096 for equal case counts. A schema-faithful CPU fixture returned **91 seconds for both**. Price the requested levels and retry allowance explicitly, or refuse an unsupported projection. Also, `seconds` already includes compilation (`vref.py:75–85`, `vfd.py:295–300`), so adding `compile_seconds` double-counts it; that conservative overcount does not establish a bound for larger levels.

2. **WRONG — ladder/test steady cost uses the fastest case and misclassifies variation as compilation.** `resources.py:57–59` takes `min(ts)` and treats `max(ts) - min(ts)` as compile overhead. The producer records compilation-inclusive executions across different cases, not controlled repeated measurements of one workload (`vladder.py:135–157`). A fixture with observations of **1 and 100 seconds** yields a recurring cost of **1 second** and a one-time compile cost of **99 seconds**. This can severely underprice subsequent cases. Both `proj_vladder` and `proj_vtest` inherit this defect (`resources.py:78–79,97`).

3. **WRONG — ladder qualification and screening allowances are not upper bounds.** `resources.py:80–82` assumes half the configurations are screened and prices qualification using 36 candidates at the **mean** configuration cost. Actual screening can include every surviving configuration (`vladder.py:412–416`). Qualification keeps trying candidates until enough **pass**, potentially traversing the entire ranked list (`vladder.py:422–434`); 36 bounds selected winners, not attempted qualifications. Variant compilation and increased quadrature/tighter-solver costs also lack explicit pricing. The 1.3 safety factor does not bound these omissions.

4. **WRONG — timing projection ignores workload differences.** `resources.py:89–92` scales solely by subject count and repetitions. Replacing a small smoke solver with a much larger production solver leaves the estimate unchanged. It also reads the denominator configuration from mutable `configs/smk-time.json` instead of the smoke result’s recorded `config` (`vtime.py:90`). Use recorded configuration and costs appropriate to the production subjects, stages and retry workload; check resident-bank memory compatibility separately.

5. **WRONG — budget-plan completeness is unchecked.** `resources.py:120–131` accepts any supplied list, including an empty list, and labels the total `fits` without requiring fallback references, test references/runs, or a timing retry reserve. Thus an incomplete plan can pass the numerical check despite A2.20/A3.20. Require an explicit accounting of every mandatory category, including justified deferrals.

**Schema/crash assessment:** The readers correctly match the producers’ nested `cases[*].levels`, `crosscheck[*].stats`, `runs[*].seconds`, `elapsed_seconds`, and `memory.bank.build_seconds`. Schema-faithful fixtures exercised these successfully; I found no inevitable schema-mismatch crash on completed outputs. No real smoke result files were available under this worktree’s `runs/`, so this was not a replay against actual smoke artifacts. Missing family timing observations can still cause `KeyError` at `resources.py:78,97`.

No files changed, jobs submitted, or GPU computations performed.

**Overall verdict: WRONG — gate fixes CORRECT; resource projections and budget-plan validation are not conservative.**