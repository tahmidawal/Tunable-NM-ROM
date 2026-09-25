# poisson-lshape3d-test — design and pre-registration

Written and committed **before any test job of this lane was submitted** (commit `4b08ccb1`, 2026-09-25 03:07 EDT). Lane branch
`exp/2026-09-25-poisson-lshape3d-test`, forked from `exp/2026-09-23-poisson-bank-knob-3d` @ `59b67ae2`
(sparse worktree `worktrees/2026-09-25-poisson-lshape3d-test`, never pushed). Cluster namespace
`/cluster/tufts/paralab/tawal01/pl3test_20260925/`. Hard stop 2026-09-25 18:00 EDT.

**Question.** The paper's Table 1 rows for the L-shaped Poisson problem ($256^2$–$2048^2$) and Poisson 3D
($128^3$, $256^3$) are measured on development cohorts (32 and 16 cases). What are the same rows — same frozen
models, same settings, same full-order-model (FOM) rule, same timing protocol — on held-out **test** cases that
were never used to train, select or develop anything?

Nothing is trained, tuned or selected in this lane. The only new inputs are the test cohorts.

## Frozen models (unchanged from the parent lane)

| problem | checkpoint | $K$ | $R$ | weak tests $M$ | rotation |
|---|---|---:|---:|---:|---|
| L-shape | `hires-poisson/lshape/head_sdf_R512_K16` (sha256 `d43082e2…`) | 16 | 512 | 257 L-shape eigenmodes | `poisson-bank-knob-3d/runs/prep_lshape.npz` |
| Poisson 3D | `paper-p3d/runs/final08/checkpoints/{bank,head_K16}.pkl` (sha256 `6c229877…`, `258d2845…`) | 16 | 128 | 512 sine modes | `poisson-bank-knob-3d/runs/prep_cube.npz` |

## Frozen Table-1 rows (chosen on development; `frozen-test-lshape.json`, `frozen-test-cube.json`)

| problem | meshes | accurate | fast | also reported | time scope |
|---|---|---|---|---|---|
| L-shape | $256^2, 512^2, 1024^2, 2048^2$ | span $R'=128$ (linear rung, `R128_linear`) | span $R'=64$ (`R64_linear`) | — | GPU query (`fused_device_seconds`); complete query also reported |
| Poisson 3D | $128^3, 256^3$ | span $R'=128$ (`R128_linear`) | span $R'=16$ (`R16_linear`) | span $R'=32$ (`R32_linear`) | GPU query; complete query also reported |

$R'=32$ is reported at $128^3$/$256^3$ because it is the fast width of the paper's $32^3$/$64^3$ rows. It is **not**
a substitute: the fast column of these rows stays $R'=16$ whatever the test numbers show. Choosing between the two
widths for the paper is the user's decision, and this lane reports both in every case.

**FOM rule (same job, same scope):** the Table-1 FOM of a row is the fastest tested CG tolerance whose worst
**test** error is $\le$ the accurate arm's worst test error. Every arm of the row is divided by that FOM's median
time. The rule is deterministic given the grid; it chooses no model setting.

## Test cohorts (fixed here, before opening)

- **L-shape.** A fresh seed, **20260925**, drawn with the same generator as every earlier L-shape cohort
  (`lsh_core.cohort(seed, 64, 32)`: `source_params` Gaussian family, centre-in-$\Omega$ rejection, first 32 accepted).
  54 of 64 draws are accepted, and the first 32 are used. Parameters sha256
  `c9230b402088fa3eeeba52d27883b2956fd84f4edae118a9eef1a7a1404f3204`. The seeds used before for this family are
  0 (training, 3072 of 4608), 20260916 (common selection, 256 of 512, and the fit/validation split seed) and
  20260917 (development, 32 of 64). No other L-shape cohort was drawn anywhere in the repository; that check covers the
  lshape, hires-poisson, poisson-bank-knob-3d and ops-all p2d designs. Disjointness from all three is asserted in the
  driver, which aborts on overlap. The minimum $L^\infty$ parameter distance is recorded: computed while staging, it
  is 0.0143 to training, 0.0260 to selection and 0.0626 to development. The count of 32 matches the development
  cohort, so the worst-of-$n$ statistic is compared at equal $n$.
- **Poisson 3D.** The paper's 64-case test cohort at $32^3$/$64^3$ (paper-p3d `reserved_final_seed` **920499**,
  `common.family(920499, 64)`, sha256 `27ec2cf52d2eb0ad84d6b3e83c46f504a67c0dd56a7a1f968263d3179430be60`,
  identical to `config-cube-{32,64}-final.json` of the parent lane). The source family is mesh-free (five
  parameters), so the same cases are generated at $128^3$ and $256^3$. The driver asserts disjointness from the
  training (seed 920410, 512) and development (seed 920411, 16) cohorts.
  - **Disclosure: this cohort is not entirely unused for choosing.** Its $32^3$/$64^3$ records (parent jobs 4200246 and
    4202245) were opened by the parent lane under settings frozen on development. The 2026-09-24 "5 % fast rule" edit
    of the paper then picked the $32^3$/$64^3$ fast width $R'=32$ from those same records (LAB-LOG 2026-09-24,
    "Overleaf: 5 % fast rule"). The cohort was never used to choose anything at $128^3$ or $256^3$, and this lane
    chooses nothing on it.

## Arms, timing and gates (identical to the parent lane, `poisson-bank-knob-3d/DESIGN.md` + amendment A1)

- **Drivers.** `plt_lshape.py` and `plt_cube.py` are byte copies of `pbk3_lshape.py` and `pbk3_cube.py`, except for
  the cohort block (`diff` against the parent is in the commit). `pbk3_core.py` and `pbk3_audit_np.py` are staged
  from the parent unchanged.
- **Arms.** The parent's full ladder runs unchanged in every job:
  - L-shape: $R'\in\{32,64,128,256,384,512\}$ × $q$, the linear rungs for $R'<M$, and the parity parents.
  - Poisson 3D: $R'\in\{16,32,48,64,96,128\}$ × $q$, the linear rungs, and the parity parents.

  The full ladder is kept so that the randomised interleaving, the burn-in and the per-invocation guard are those
  of the development jobs. **Every arm's test numbers are reported**, but only the frozen rows above enter
  Table 1. The rule re-applied on test (accurate = lowest worst error, and so on) is printed as *descriptive only*.
- **CG grid** as in the parent. L-shape: rtol {0.3, 0.2, 0.1, 0.03, 0.01, 0.003} main and {1e-3, 1e-4} slow, matrix-free GPU CG.
  Poisson 3D: {0.3, 0.1, 0.03, 0.01, 0.003} main and {1e-3, 1e-4} slow, `iterative_cg.engine(retain_history=False)`.
- **Timing.** Randomised order within each (case, repetition); 0.1 s burn-in and a GPU-UUID guard before every
  invocation; 5 retained repetitions (3 for slow CG); medians over all repetitions × cases; one GPU and one allocation
  per mesh. Every ratio is taken within one job.
- **Gates.** A job's numbers enter the summary as usable only if all of these pass:
  - parity ($\le 10^{-10}$);
  - determinism;
  - CG converged on every call;
  - full-$R$ stationarity;
  - the A1 paired order gate (16 cases × 2 rounds; pooled $\le 1.10$, after/main $\le 1.10$, each arm $\le 1.25$);
  - profile consistency;
  - the device guard;
  - the independent NumPy audit (`pbk3_audit_np.py`: own truth, recomputed errors, re-hashed fields, controls that
    must fail).

  The order gate is evaluated in the Table-1 scope, which is now GPU query for **both** problems (the parent's L-shape
  configs gated in complete query; the paper switched the L-shape series to GPU query on 2026-09-23). `make_tables.py`
  recomputes the gate from the raw rows in both scopes and reports both.
- **GPUs.** Chosen to match the development jobs where possible: A100-80GB for L-shape $256^2$–$1024^2$ (the $256^2$
  development job ran on an A100-40GB), and H200 for L-shape $2048^2$, $128^3$ and $256^3$. Absolute milliseconds
  across jobs are not compared; only same-job ratios are.
- **Field storage** (an infrastructure change only). Saved fields for the audit go to node-local `/tmp` when it has
  room, else to the share. On the H200 nodes `/tmp` is tmpfs charged to the job, so `--mem` is raised. At $256^3$
  with 64 cases the fields are about 230 GB. The audit deletes them, and an EXIT trap removes the local copy.

## What is reported (generated; no hand-typed numbers)

`make_tables.py` → `reports/summary.json` (sha256 recorded in the report and the lab log) →
`make_report.py` → the report. For each (problem, mesh) there is one row in Table-1 format: accurate worst %,
accurate speedup, fast worst %, fast speedup, FOM worst %, ms per arm, FOM setting and ms, job id, GPU and result
sha256. It sits beside the **development** values regenerated by the same code from the parent lane's archived
results. That regeneration must reproduce the paper's printed development values, or the discrepancy is reported:
L-shape 3.04/47.1×/5.73/53.4×/2.40, 3.03/39.4×/5.72/45.5×/1.51, 3.03/48.9×/5.72/57.9×/1.05, 3.03/84.2×/5.72/104×/0.76;
Poisson 3D 128³ 0.14/13.7×/4.50/36.3×/0.075 and 256³ 0.14/23.2×/4.50/83.5×/0.049. Medians and the number of
cases above the development worst error are reported as well as the worst.

## Integrity rules

- No tuning on test cases. Any improvement would have to be found on the development or selection sets, written
  here as a dated amendment, and then evaluated on test once. None is planned.
- Every test evaluation is reported, including failed or worse ones.
- A job that fails for infrastructure reasons (GPU preflight exit 42, OOM, node or disk failure) may be resubmitted
  unchanged; this is recorded in `attempts.json` and the lab log. A job that completes with a failed gate is reported
  as such. A rerun after a failed gate is allowed only after a dated amendment here that states the diagnosis, and
  both runs are reported.
- If a frozen fast arm exceeds the paper's "< 5 %" fast criterion on test, it is reported as exceeding it. It is not
  replaced.

## Amendment A1 (commit `39144077`, 2026-09-25 03:11 EDT, after jobs 4319385, 4319382, 4319381, 4319374) — L-shape test-cohort hash

All four L-shape test jobs (`lt256` 4319385, `lt512` 4319382, `lt1024` 4319381, `lt2048` 4319374) stopped within
seconds, at the driver's cohort assertion. The parameters sha256 recorded above
(`c9230b40…`) was computed on the local GB10, which is ARM. The x86 cluster gives `366e6597…` for the same seed,
draw and acceptance: 54 of 64 accepted, the same 32 kept. The cause is last-ulp differences in `np.exp` between the
two platforms. The development cohort shows the same thing: the local recompute of seed 20260917 hashes to
`4164cfd1…`, while the cluster record in parent job 4199770 is `525e1951…`, with a maximum relative parameter
difference of 1.4e-16. So the cohort is the same draw, and only the platform hash differs.

**Change.** The canonical test-cohort hash is now the cluster value,
`366e659729ec83baf5cd725391cd81cdfff34909927926db03b9ada2c4d7fc03`, in the four configs and in `frozen-test-lshape.json`.
Seed, draw and count are unchanged. Nothing else changes.

**No test case was opened.** Each job stopped before `save()` wrote a `result.json`, and before any reference
solve, field or error existed. The logs are kept under `runs/lt*-failed-cohort-sha/`. The resubmissions are
`lt256b`, `lt512b`, `lt1024b` and `lt2048b`. The two cube jobs (`ct128` 4319378, `ct256` 4319372) passed their cohort
check, because `common.family` uses no transcendental functions, and they continue unchanged.
