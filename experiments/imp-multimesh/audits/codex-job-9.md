**J0: FIX-FIRST. J5: FIX-FIRST.**

| Blocking item from codex-job-8 | Round 3 finding |
|---|---|
| J0 resource certificate | **NOT FIXED.** Smoke inputs are now used, but the estimate substitutes unexplained `120`-second allowances for construction phases and omits explicit contribution costs. It does not scale the measured POD/code phases to production workloads. |
| J5 resource certificate | **NOT FIXED.** The estimate applies R128-BE timings to all ranks/schemes with unsupported mesh multipliers and excludes the new offline checker’s cost. Its own 4096 estimate exceeds that command’s timeout. |
| J0 incremental artifacts and terminal states | **FIXED.** Rules use temporary-file writes followed by `os.replace`; active/pending ranks are persisted as `UNRESOLVED`. |
| J0 missing `seconds` crash | **FIXED.** `analysis.py` now guards access to that field. |
| J0 path versus outer budget | **NOT FIXED.** Halving remaining time accounts for two NNLS attempts, but the allowance is still computed before contribution construction and reused for both attempts. |
| J0 construction burn-ins | **FIXED**, retained. |
| J5 finer-rule non-convergence | **FIXED.** The checker returns the finer-rule “unverified” label, mapped to `UNRESOLVED` when Q1 passes. |
| J5 independent acceptance and descriptor identity | **FIXED.** Staged `p1score` and `npcheck` are byte-identical to their source copies; qualification now checks saved coefficients, validity, distances, certificate agreement and independent rho. Descriptor tables are compared by value. |

The remaining fixes are concrete:

- **J0:** In [resources.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/resources.py:25), replace unexplained allowances with documented phase/workload projections. In `mmbuild.py` and its ECSW callees, recompute remaining time after contributions and share a deadline across initial NNLS, retry and persistence. The current calculation still consumes the intended reserve.
- **J5:** [resources.py:27](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/resources.py:27) predicts **3,705 seconds before the safety factor**, or **5,557.5 seconds with ×1.5**, for the 4096 workload alone. Its staged command timeout is **2,400 seconds**. Reconcile phase estimates, checker costs, command timeouts and allocation before submission.

Small CPU fixtures confirmed that the previous NaN/invalid-check exploit no longer qualifies, while finer-rule-only non-convergence remains unverified. S1c records every acceptance control passing. All checked manifest files ≤2 MB matched: **139 J0, 143 J5**; certificate config hashes and batch syntax passed.

No files changed, jobs submitted, large arrays loaded, or lab-log/main writes performed.