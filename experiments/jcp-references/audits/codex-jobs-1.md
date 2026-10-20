**Overall: SUBMIT these three gate/calibration jobs only.** HEAD repairs the blocking-now NaN defect, and I found no execution blocker in the staged jobs. However, **the latest commit introduces G5 audit errors**, so it does not justify clearing the blocking-later findings or releasing references.

Audited `491c66544986f37476870ec98dc32c8df5867c44`. All reproductions were in memory, using actual extracted functions and controlled doubles. No GPU solves, cluster commands, submissions, or file modifications were performed—including no lab-log append.

| Item | Verdict | Evidence / qualification |
|---|---|---|
| `u_transforms`: blocking-now NaN aggregation | **CORRECT** | Finite control passes with worst discrepancy `1.21e-15`. NaNs confined to initial transforms, nonlinear output, lattice interior, or lattice boundary each return `pass=False`. |
| JSON-only FD resume leftover | **CORRECT** | Actual resume condition returns false without reading the missing NPZ; cleanup/recomputation is reachable. |
| Sealed rerun refusal before result overwrite | **CORRECT** | Actual `main()` refuses existing ciphertext before gates or `save()`. |
| Required calibration artifact collection | **CORRECT** | Collector now requires the dimension-specific NPZ and checks its hash against both the result and output manifest. |
| G5 non-finite comparison repair | **WRONG** | `relclose` handles serialized nulls, but FD `umax_running=null` still raises `TypeError` during Péclet computation. Existing blocking-later defect remains. |
| G5 FD per-time vector lengths | **CORRECT** | Explicit five-element checks now cover `c`, `d12`, `d23`, `eps_t`, and `p`. |
| G5 missing PS evidence | **CORRECT** | Missing evidence now gives `match=False`. |
| G5 stability recomputation | **WRONG** | New code processes running maxima before checking failed-run validity. A serialized `umax=null` raises `TypeError`. **New blocking-later regression.** |
| G5 order/effectivity comparison | **WRONG** | Finite control matches; corrupting `q` or `eta` is detected. But truncated `q` raises `KeyError`, and zero uncertainty raises `ZeroDivisionError`. **New blocking-later regressions.** |
| G5 perturbation detection | **NEEDS-RESTATEMENT** | Outcome-only detection is removed correctly. But requiring finite relative uncertainty change also rejects an independently sufficient change in `p` when uncertainty is zero/undefined. |

The new G5 failures are reproducible at [audit_ref.py:263](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/audit_ref.py:263), [audit_ref.py:284](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/audit_ref.py:284), and [audit_ref.py:295](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/audit_ref.py:295). None executes in these three jobs.

| Staged-job item | Verdict | Evidence / qualification |
|---|---|---|
| All three `run.sbatch` scripts | **CORRECT** | `gpu` partition; absolute cluster venv activation and interpreter; f64; `JAX_DEFAULT_MATMUL_PRECISION=highest`; GPU preflight exits 42 on CPU; shell syntax passes. |
| Output isolation | **CORRECT** | Logs, results, cache and temporary directory are under the respective `/cluster/tufts/paralab/tawal01/jcp-ref/<job>/`; distinct task roots; staged log directories exist. |
| Commit, provenance and manifests | **CORRECT** | Checked **all 16 source entries and 19 manifest entries per job**. Every source byte matches its committed blob at `COMMIT.txt`; sizes, SHA256 values and provenance commits agree. All pin HEAD. |
| Sample source hashes | **CORRECT** | Across all jobs: `gates.py` starts `2aee40f286835ac5…`; `refjob.py` starts `fce9d65e31ed8471…`. Full hashes were checked. |
| Configs | **CORRECT** | Respectively `gates`, `calib` dimension 2, and `calib` dimension 3. No CPU allowance or smoke override. `gates` runs both dimensions despite its config’s `dim: 2`. |
| `gates1` resources | **CORRECT** | A100-80G, 120G host, 4 h. Exercises G3b/G3c at escalation sizes: 2D `N=3072`, 3D `N=384`. It does not execute full production PS ladders at those sizes. |
| `calib2d1` resources | **CORRECT** | A100-80G, 120G host, 12 h; real base levels through `L=8192`, with no refinement level. Consistent with A5.6. |
| `calib3d1` resources | **CORRECT** | H200, 240G host, 12 h; real base levels through `L=512`—513 nodes. Meets even the stricter production resource requirement. |
| Runtime and memory sufficiency | **NEEDS-RESTATEMENT** | These are plausible allocations, not demonstrated fits. Tight/tighter solves, compilation, repeated solves at the finest level, full-grid outputs and cached solvers contribute cost. Host memory does not establish device fit. |
| Timing interpretation | **NEEDS-RESTATEMENT** | First-use `solve_seconds` includes solver construction/setup beyond A6.5’s stated execution-plus-transfer scope. Device peak is process-lifetime. Useful budgeting evidence, not isolated solver benchmarks. |

| Scientific coverage | Verdict | Evidence / qualification |
|---|---|---|
| Floor calibration: A2.1, A3.2 | **CORRECT** | Each dimension covers two prescribed dev cases × two kinds × three base levels × prod/tight/tighter: **36 solves and 12 comparison rows**. Both proxy inequalities are checked; all three solves must succeed with finite proxies. |
| Evidence retention: A5.4, A6.2 | **CORRECT** | Harness retained **63 attempts / 189 arrays in 2D**, **56 / 168 in 3D**, with matching attempt records. Injected failed solves make `proxy_ok_all=False`. Status and field hash are produced; refinement levels remain explicitly uncalibrated. |
| D1: A1.8 | **CORRECT** | Prescribed cases, time-only ladder at finest space, space-only ladder at finest timestep, per-time orders and normalized differences. Eight additional solves per dimension; shared finest result reused. |
| X1: A1.14 | **CORRECT** | Prescribed upwind+BE ladders cover all six 2D and four 3D dev cases. Outcomes and orders are retained, including accidental passes. |
| X1b: A2.8 | **NEEDS-RESTATEMENT** | Tests exact synthetic first-order sequences on both grid triples plus a second-order control. Uses seeded random fields, rather than the specified stored dev6 case-0 fields. Same algebraic test, different stated construction. |
| X4: A1.14 | **CORRECT** | 2D dev6 case 0, middle timestep doubled; unchanged endpoints reused. Actual residual proxies, step flags and running maxima feed the verdict. Outcome/order are retained for reporting the sensitivity warning. |
| Job-side gate list: §7, A1.13, A2.7, A3.4, A4, A5.7 | **CORRECT** | Units, G1/G2, G3a–c, G4 spatial/temporal/half-step checks, and projection test are dispatched with amended settings. Calibration jobs run units and both G1 checks. |
| G5–G7 and per-run PS checks | **CORRECT** | Deferred appropriately: G5–G7 require the collected store under A5.9; per-run PS checks belong to later `ps` jobs. These jobs do not establish full release validation. |

The calibration harness counted **18,154 scheduled FD stages in 2D** and **8,752 in 3D**, excluding preliminary gates. These counts help explain why this wave is a substantive cost calibration. Reissue the production estimate from its measured results before production, as A1.15 requires.

| Submission protection | Verdict | Evidence |
|---|---|---|
| Duplicate queued job | **CORRECT** | Mocked execution refuses matching `jr_gates1` with exit 3 before transfer/submission. |
| Existing remote directory | **CORRECT** | Exclusive `mkdir` failure stops execution before transfer/submission. |
| Lane cap of four | **CORRECT** | Under the shared local lock, three queued lane jobs allow one submission; four refuse with exit 5. Pending jobs count too. |
| Extra flags | **CORRECT** | Dependency syntax accepted; `--partition=preempt` rejected with exit 6. |
| Concurrency scope | **NEEDS-RESTATEMENT** | The lock serializes callers using this worktree’s lock file. It cannot enforce an account-wide atomic cap against another checkout, host, or direct `sbatch`. |

**Submission scope:** these three jobs may collect gate and calibration evidence. Fix the remaining and newly introduced G5 defects before store validation or reporting reference results; do not treat this verdict as production or release approval.