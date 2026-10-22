# Codex audit 1 of make_mesh_fig.py (final message; full log trimmed)

Adversarially audit reports/2026-10-22-method-presentation/figs/make_mesh_fig.py and its output mesh_cost_iters.json (run from repo root /home/tahmid/Dev/Tunable-NM-ROM-Claude). Independently recompute from the source result.json (path in the script): per (bank width, scheme, mesh), frozen-mode valid invocations only: median t_evolve and t_decode in ms, and mean/min/max of sum(it) per run. Check: (1) the numbers match; (2) 'it' really counts Levenberg-Marquardt iterations per time step (read the vendored solver code under worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/vendor/quad2d/qcore.py and the p1time driver) and whether that equals Jacobian builds or not -- an earlier report claims about 2.12 Jacobian assemblies per step (reports/2026-10-20-spectral-investigation/why-spectral/report.md); reconcile the two; (3) whether excluding invalid or nominal invocations biases anything; (4) whether t_evolve excludes init and decode. Also verify these scheme facts against worktrees/2026-10-20-jcp-consolidated/experiments/jcp-combined/t2core.py: residual R_n(c)=A(a0 c+a1 c_n+a2 c_{n-1})+dt(b0 F(c)+b1 F(c_n)); BE (1,-1,0,1,0, 0 startup), CN (1,-1,0,.5,.5,0), CNR (CN + 2 BE startup), BDF2 (1.5,-2,.5,1,0, 1 startup), TH06 (1,-1,0,.6,.4,0). End with VERDICT: CONFIRMED or NEEDS-FIX and a list of issues.

The numerical aggregation is correct. I independently recomputed all 20 groups from the source `result.json` using Python’s standard-library statistics, without executing the plotting script. Every stored value matches exactly, including counts and metadata.

| Width | Scheme | Mesh | Evolve median, ms | Decode median, ms | Σit mean | Min | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| 128 | BE | 256 | 22.098623 | 0.134535 | 51.833333 | 50 | 55 |
| 128 | BE | 1024 | 22.034395 | 0.395714 | 51.666667 | 49 | 55 |
| 128 | BE | 4096 | 21.118428 | 5.735625 | 51.666667 | 49 | 55 |
| 128 | CNR | 256 | 21.905725 | 0.130422 | 51.500000 | 48 | 55 |
| 128 | CNR | 1024 | 21.941811 | 0.394129 | 51.500000 | 48 | 55 |
| 128 | CNR | 4096 | 21.864130 | 5.709255 | 51.500000 | 48 | 55 |
| 256 | BE | 256 | 51.717008 | 0.154988 | 53.833333 | 52 | 56 |
| 256 | BE | 1024 | 51.557232 | 0.671795 | 54.166667 | 52 | 57 |
| 256 | BE | 4096 | 51.755582 | 10.005335 | 54.166667 | 52 | 57 |
| 256 | CNR | 256 | 51.153864 | 0.157895 | 53.333333 | 52 | 55 |
| 256 | CNR | 1024 | 51.635963 | 0.667013 | 53.666667 | 52 | 56 |
| 256 | CNR | 4096 | 51.328087 | 10.014144 | 53.666667 | 52 | 56 |
| 384 | BE | 256 | 87.766949 | 0.178947 | 54.333333 | 52 | 57 |
| 384 | BE | 1024 | 86.627100 | 0.953856 | 54.333333 | 52 | 57 |
| 384 | BE | 4096 | 88.282339 | 14.274901 | 54.333333 | 52 | 57 |
| 384 | CNR | 256 | 87.219605 | 0.176554 | 54.000000 | 52 | 56 |
| 384 | CNR | 1024 | 88.748251 | 0.949040 | 54.000000 | 52 | 56 |
| 384 | CNR | 4096 | 87.875430 | 14.256284 | 54.000000 | 52 | 56 |
| 512 | BE | 1024 | 224.520710 | 2.174667 | 54.666667 | 52 | 58 |
| 512 | CNR | 1024 | 223.801599 | 2.048887 | 54.166667 | 52 | 57 |

Each row contains 18 invocations: six cases × three repeats. Every `it` array contains 50 nonnegative integers. Values above are rounded for display; comparisons used full precision.

**Iteration meaning and the 2.12 reconciliation.** The vendored [qcore.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/vendor/quad2d/qcore.py:311) delegates to `hfast.make_fused_lm`. The actual timing path uses `p1common.T2 = p1core`, whose [LM implementation](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/p1core.py:60) has the same counting convention:

- One initial `evalJ` before the loop.
- Each loop pass performs a damped solve, evaluates a trial residual/Jacobian, and increments `it`, including rejected trials.
- An initially converged predictor can return `it=0`.

Thus, for these implementations,

$$
N_J^{\text{run}}=50+\sum_{n=1}^{50}\mathrm{it}_n.
$$

The earlier report uses `dev6|0`, R384/CNR/1024: **56 iterations → 106 Jacobian builds → 2.12 per step**. The current six-case average is **54 iterations → 104 builds → 2.08 per step**. Both are correct. Neither means one Jacobian per step.

**Exclusions and bias.** There are 360 frozen and 72 nominal invocations; **all 432 are valid**. No invalid invocation was dropped. One frozen invocation required a drift retry; substituting its first attempt changes that group’s evolve median from 21.941811 to 21.977201 ms, about 0.16%, with identical iterations.

Nominal means a mesh-dependent test-mode list; frozen fixes that list across meshes. Excluding nominal is appropriate for this controlled comparison. In the four available nominal R384 groups, iteration means equal their frozen counterparts and evolve medians differ by less than 0.7%. There is no evidence that the exclusion manufactures the conclusion.

Caveat: invocation `valid` checks timing drift and identity, not numerical convergence. I separately checked the subject certifications: all report valid runs, zero budget exits and zero nonstationary steps.

**Timing scope.** [p1time.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/p1time.py:229) times and synchronizes init, evolve and decode separately. `t_evolve` excludes initialization, decode and burn-in. It includes predictors, history work, LM solves, statistics and trajectory recording. These are warmed execution timings, not complete query or setup costs. Decode reconstructs six fields.

**Scheme verification.** The supplied residual matches [t2core.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-jcp-consolidated/experiments/jcp-combined/t2core.py:116), with
$F(c)=N(c)+\nu\Lambda Ac$. All supplied tuples match exactly:

| Scheme | $(a_0,a_1,a_2,b_0,b_1)$ | BE startup steps |
|---|---|---:|
| BE | $(1,-1,0,1,0)$ | 0 |
| CN | $(1,-1,0,.5,.5)$ | 0 |
| CNR | $(1,-1,0,.5,.5)$ | 2 |
| BDF2 | $(1.5,-2,.5,1,0)$ | 1 |
| TH06 | $(1,-1,0,.6,.4)$ | 0 |

Startup uses full steps at the configured `dt`; CNR does not use BE half-steps. LSPG minimizes the diagonally scaled residual, $DR_n$, with $D=(a_0+dt\,b_0\nu\Lambda)^{-1}$.

I could not append this audit to `LAB-LOG.md` because this session has read-only filesystem permissions.

**VERDICT: NEEDS-FIX**

Issues:

1. **Output location:** [lines 18 and 33](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-22-method-presentation/figs/make_mesh_fig.py:18) write relative to the working directory. Running from repository root writes root-level files, leaving the audited `figs/` outputs unchanged. Anchor outputs to `Path(__file__).resolve().parent`.
2. **Figure scope:** the rendered title omits “frozen test modes” and implies mesh-flat evidence for all plotted widths. R512 has only one mesh. State that mesh scaling is demonstrated for R128–R384.
3. **Iteration interpretation:** label the middle panel “LM trial iterations” and explain the additional initial Jacobian per step. The numbers themselves need no correction.
