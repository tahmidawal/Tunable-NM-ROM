**Overall: DO NOT SUBMIT as the registered acceptance smoke.** Staging and launch configuration pass, but audit 2’s required acceptance binding remains unfixed.

Read-only audit: no files changed, jobs submitted, SSH, GPU execution, or LAB-LOG writes. No scratch files created.

| Item | Verdict | Finding |
|---|---|---|
| Committed-tree equality | **CORRECT** | All 102 staged files match commit `c29eea265`; provenance sizes/hashes match. Expected staging set is complete, including vendor, checkpoint inputs and all three EXTRA files. |
| Manifest | **CORRECT** | All 106 entries verify; no missing or extra payload files. |
| E5 provenance | **CORRECT** | Staged `e5/fscore.py` matches the recorded SHA256 `3626350c…cbec5`. Audit fixes are in `eccb533a7`; the subsequent HEAD commit only initializes the ledger. |
| Batch allocation | **CORRECT** | `gpu` partition, H200, no preempt, 8 CPUs, 200G host memory, `01:30:00`; allocation agrees with RESERVATION.json. |
| GPU/precision guards | **CORRECT** | Preflight prints `jax_backend`, exits 42 on CPU; x64 and `highest` exported and asserted by the driver. |
| Paths/imports | **CORRECT** | Absolute cluster venv and paralab output/cache/tmp paths. Staged CPU imports and Python syntax checks pass; shell syntax passes. Both runtime checkpoint paths and rotation resolve inside staging. |
| Offline E0 paths | **CORRECT** | Local absolute paths in `p1score.py` are outside the batch execution path. Scoring/C8 require the local checkout and E0 store after collection. |
| A8 code fixes | **CORRECT** | Cache keys cover consumed arrays, dropped cache entries are evicted, eigen/orthogonality gates enforced, P2 requires valid tolerance exits, Q1 shape/anchor checks added, production Q2 population checked, per-fit E5 hash recorded, submission Python fixed. |
| Timing configuration | **CORRECT** | Smoke uses dev6, five repetitions and three burn-ins; repetition arrays, outliers and timed/scored comparisons are saved. |
| Acceptance binding | **WRONG — blocker** | `smoke_accept.py:70–77` never verifies that analysis belongs to this run/config/source, nor requires complete analysis populations. `source_commit` actually contains the `gstep.py` hash. In-memory execution of this final gate accepts unrelated analysis with empty Q1/Q2 dictionaries when preceding control predicates pass. This remains audit 2 required fix #1. |
| Independent control acceptance | **WRONG** | Several predicates still trust summaries: C4, C7, P1/P2/P4/P6; C3 trusts saved `expected`/`ok`; C2 trusts `fixture_valid`. Thus “every predicate recomputed from raw outputs” remains inaccurate. |
| Output completeness | **NEEDS-RESTATEMENT** | Successful execution writes result, arrays, phases, memory records, COMPLETE and output checksums. Acceptance and resource certificates are separate offline steps. P3 failure omits bank phases; unreached κ=8 prevents C2/P4 acceptance. Completion is conditional, as A3.5 permits. |

Control-by-control readiness—**CORRECT means wired to execute, not demonstrated at real size**:

| Controls | Verdict | Finding |
|---|---|---|
| C1 sine/bank | **CORRECT** | Registered coarse-rule fallback and both rollout/ρ failure predicates exist; all-invalid fixtures remain uncredited. |
| C2 | **CORRECT** | F3b κ=8, fallback F3a κ=8; permutation-change threshold and Q2 failure required. No reached support means failure of acceptance. |
| C3 | **CORRECT** | Terminal/interior/multiple perturbations, adjacent residual checks, NumPy cross-check and pooled failure implemented; corrupted states saved. |
| C4 | **CORRECT** | NaNs reach E5 through tested rows as well as P. New eigen gates depend on unchanged linear operators, so they do not inherently intercept this fixture. |
| C5/C6 | **CORRECT** | Changing-state Jacobian benchmark, synchronized repetitions and ≥1.5 ratio; paired A–A bound implemented. |
| C7/C8 | **CORRECT** | Leakage assertion is exercised in-job; swapped identity is exercised offline. |
| P1/P2 | **CORRECT** | Parent-weight equivalence and tight lane/E5 comparisons implemented; P2 covers R and bank Gauss96. |
| P3 | **CORRECT** | A5’s 1024²/768² comparison, 1280² fallback, amended tolerance and positivity checks implemented. |
| P4/P5 | **CORRECT** | Required projection/fitted Jacobians and A6 NumPy, AD-Laplacian, energy and host-assembly checks implemented across dev6 outputs. Mesh comparisons are descriptive. |
| P6 | **CORRECT** | Budget-one anchor independently checked; NaN successor/check fixtures must yield unverified labels. |
| All controls firing | **NEEDS-RESTATEMENT** | Static inspection cannot establish numerical failures, equivalence tolerances, timing ratios, or support attainment. These are smoke outcomes. |

| Resource item | Verdict | Finding |
|---|---|---|
| F3b allocation | **NEEDS-RESTATEMENT** | C alone is **12 GiB**; rounded 8192-column QR buffer **6 GiB**, plus R, refactorization workspaces, copies, bank and tables. H200 is plausible; these are not peak-memory estimates. |
| Gauss mass tables | **CORRECT** | Construction accumulates in 16,384-point chunks rather than retaining full 1024² value/gradient tables. |
| P5 host arrays | **NEEDS-RESTATEMENT** | Each 1023²×384 f64 array is **2.994 GiB**. Gm, padded Gp, lapG and expression temporaries coexist; 200G appears reasonable, but measured peaks remain necessary. |
| A3.5 memory fallback | **WRONG** | `p1common.mem()` returns −1 when counters are unavailable; required per-phase `nvidia-smi` fallback is absent. `resources.py` then fails on an empty `max()`. |
| Production resource certificate | **WRONG; previously deferred blocker** | Still lacks strong evidence/device/source binding, omits controls from projected phases, and implements only Fibonacci deferral. Do not use it to authorize production unchanged. |

Fix the acceptance binding before this registered smoke; retain the resource-certificate defects as explicit production blockers.