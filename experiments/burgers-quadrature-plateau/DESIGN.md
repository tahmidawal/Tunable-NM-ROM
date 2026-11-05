# burgers-quadrature-plateau — pre-registration (written 2026-11-05, before any cluster job)

Lane `exp/2026-11-05-burgers-quadrature-plateau`, worktree `worktrees/2026-11-05-burgers-quadrature-plateau`, forked from
`93f399faa` (`exp/2026-10-20-jcp-consolidated`). Mirror `origin/codeonly/exp/2026-11-05-burgers-quadrature-plateau`.
Cluster namespace `/cluster/tufts/paralab/tawal01/bqplateau-20261105/`. GPU cap **4 GPU-h** (allocated wall time × GPUs,
all jobs including failures). At most 2 lane jobs running at once.

## 1. Question

The October off-mesh quadrature studies (`exp/2026-10-01-quadrature-study`, 2D; `exp/2026-10-01-quadrature-burgers3d`, 3D)
found an accuracy plateau in the number of rule points $m$, scored against first-order references (provisional):

- 2D (test64, `acc` setting): Gauss $32^2$ ($m=1024$) worst 9.0 %; every rule from $m\approx4096$ up sits at about 2.6 % worst.
- 3D (923901 × 32, $R'=512$): lattice-256 and Smolyak-8 ($m=2559$) fail; lattice-4096, Gauss $24^3$, $32^3$ reach 2.8 % worst / 1.1 % median.

This lane (a) re-scores the same frozen models and rules against the E0 second-order references
(`exp/2026-10-06-jcp-references`, store version `3ec3b5492748…`), and (b) resolves the knee with a dense ladder of $m$,
(c) places the ROM's plateau accuracy against the full-order solvers at every mesh, with solve time vs $m$ and FOM time
measured in the same job.

## 2. Frozen inputs (no retraining)

Vendored byte-identically under `vendor/tree/<original repo path>` (so every vendored module's own path logic resolves
inside the vendor tree), listed with source branch, commit and sha256 in `vendor/PROVENANCE.json`:

- 2D model: checkpoint `experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl`
  (sha256 `18f0266a…`) + `experiments/quadrature-study/inputs/rotation_R512.npz`; code `quadrature-study` @`84d28e303`
  (`qcore.py`, `vendor/{arms,hops,hfast,bkfast,hari_quadrature}.py`) and its dependencies `mr-burgers2d/{engines,iterative_paths}.py`,
  `separable-decoder/sep_common.py`.
- 3D model: `experiments/quadrature-burgers3d/inputs/model_M2/bank.pkl` (sha256 `6687f259…`); code `quadrature-burgers3d`
  @`aab6b6927` (`offmesh.py`, `qpanel.py`, `rules.py`, `npcore.py`, vendor `burgers3d-span/common.py`,
  `burgers3d-retry/tables.py`, `paper-b3d/vendor/b3d_common.py`).
- References and FOMs: `jcp-references` @`cc04f4b54` (`refstore.py`, `verdict.py`, `refcore.py`, `pscore.py`, `cohorts.py`,
  `store/MANIFEST.sha256`, `store/STATUS.json`); `jcp-fom2d` @`12103d331` (`fomrun2.py`, `psjax2d.py`); `jcp-speed3d`
  @`739154488` (`fomrun.py`, `fomcore.py`, `psjax.py`).
- The lane's own drivers (`plateau2d.py`, `plateau3d.py`) are new files that import the vendored modules unchanged;
  they reproduce the rollout code paths of `qstudy.py` phase 1 and `qpanel.py` phase 4 (same query builders, same solver
  settings) and drop the parts not needed here (rho ladders, dense arm, same-grid tight reference, Richardson).

Settings (unchanged from the October lanes):
- 2D `acc`: linear rung, $R'=384$, $M=1536$ sine tests, backward Euler $\Delta t=0.005$, LM gtol $10^{-3}$, step budget 600,
  off-mesh *point* (gradient) form. Only `acc` is run (the 2.6 % plateau is the `acc` setting); `fast` and `head` are not.
- 3D: $R'=512$, $M=4R'$ completed (`common.complete_M`), $\Delta t=0.01$, gtol $10^{-3}$, trust fraction 0.05, cached-predictor
  fixed-sweep LM (`offmesh.make_fsc_rule`). $R'=256$ is not run.

## 3. Cohorts

- 2D: **test64** = `engines.params_draw(20260916, 64)`, physical sha256 `cd058fd2…` — the same held-out cohort scored in
  October (reused, not sealed). Development/smoke: **dev6** only.
- 3D: **test32** = seed 923901 × 32 (table sha256 `f20a1aac…`) — the same held-out cohort. Development/smoke: **dev4** (923651 × 4).
- **No sealed or OOD cohort is touched** (2D sealed128/ood32, 3D sealed64/ood16; `refstore` raises on them).
- Nothing is selected on test64/test32 that is then confirmed on them: $M^*$ and the crossover are *descriptive*
  measurements on the held-out cohort, as the October numbers were. They are reported as such.

## 4. Meshes and rule ladders

2D meshes $L^2$, $L \in \{256, 512, 1024, 2048, 4096\}$ (512² and 2048² are new). Scoring on the 257² lattice $x=i/256$.

| family | rungs ($m$) |
|---|---|
| Gauss tensor $n^2$ | $n$ = 24, 28, 32, 36, 40, 45, 50, 55, 60, 64, 72, 80, 96, 128, 192, 256 (576 … 65,536) |
| Fibonacci lattice | 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368 |
| Sobol (scrambled, seed 0) | 1024, 2048, 4096, 8192, 16384, 32768, 65536 |
| control | Smolyak Clenshaw–Curtis level 8 (must fail) |

3D meshes $n^3$ nodes, $n \in \{65, 129, 257\}$ ($64^3$, $128^3$, $256^3$ cells). Scoring on the $63^3$ lattice $x=k/64$.

| family | rungs ($m$) |
|---|---|
| CBC rank-1 lattice (shift seed 0) | 256, 512, 768, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 6144, 8192, 16384, 32768 |
| Gauss tensor $p^3$ | $p$ = 8, 10, 12, 14, 16, 17, 18, 19, 20, 22, 24, 28, 32 (512 … 32,768) |
| Sobol (scrambled, seed 0) | 512, 1024, 2048, 4096, 8192, 16384, 32768 |
| Korobov lattice | 1021, 2039, 4093, 8191, 16381 |
| control | Smolyak CC level 8 ($m=2559$, must fail); the incumbent mesh tensor is run as a non-ladder reference arm |

3D rules are generated in-job with the vendored `rules.py` generators (deterministic); the sha256 of every rule array is
committed in `rules3d_manifest.json` (computed locally before submission) and checked in-job (gate G-R). 2D rules come from
the vendored `qcore.offmesh_rule` (Hari's generators), sha256 likewise committed in `rules2d_manifest.json`.

## 5. Metrics

Error of one rollout on case $c$ against reference kind $\kappa\in\{ST, S\}$ (E0 `refstore.score`, unchanged):
$$e_c = \max_{k=1..5} \frac{\lVert f(t_k) - R_\kappa(t_k)\rVert_2}{\lVert R_\kappa(t_0)\rVert_2}, \quad t_k = 0.05k,$$
on the scoring lattice. **ST** (central + Crank–Nicolson, three-level, primary) and **S** (central + backward Euler at
$\Delta t=0.005$, space-only; secondary). Statistics over the **eligible** cases (E0 eligibility: 2D test64 63/64 for ST,
55/64 for S; 3D test32 32/32): worst $W(m)=\max_c e_c$ and typical $T(m)=\operatorname{median}_c e_c$. All-case values are
reported beside them.

Rollout distance to the converged rollout: $d_c(m) = \max_{k\ge1}\lVert f_m(t_k)-f_{\rm conv}(t_k)\rVert/\lVert R(t_0)\rVert$,
with conv = Gauss $256^2$ (2D) and CBC lattice 32768 (3D), reported as $\max_c d_c$.

## 6. Plateau and $M^*$ (definitions fixed before any data)

For one mesh and one family with rungs $m_1<\dots<m_K$:
- family plateau $W_F = W(m_K)$, $T_F = T(m_K)$ (largest rung of the same family);
- $M^*_W = \min\{m_i : |W(m_j)/W_F - 1| \le 0.05 \ \forall j \ge i\}$; $M^*_T$ the same with $T$; **$M^* = \max(M^*_W, M^*_T)$**;
- reported as the bracket $(m_{i-1}, m_i]$, since the ladder is discrete;
- global version: the same with $W_{\rm conv}, T_{\rm conv}$ (the conv anchor of §5) in place of $W_F, T_F$; a family that
  never satisfies it inside its ladder is reported "not reached" (Sobol is expected to be slow);
- sensitivity (descriptive only): the same with a 1.01 band, and the rollout-distance rule $\max_c d_c \le 10^{-3}$.
- E0's $\epsilon_{\rm ref}$ (worst 1.2e-5 in 2D, 1.0e-4 in 3D, as a fraction) is far below the 5 % band of a ~2–4 % plateau,
  so reference uncertainty cannot move $M^*$; this is checked and stated with the generated numbers.
Primary statistic: ST. S is reported in the same form.

## 7. Full-order comparators and the crossover

Run in every ROM job, on the same cases, scored with the same `refstore.score` (existing code only; no new FOM):

2D (job at mesh $L$; `fomrun2.FOMs`, E0 `refcore` Newton–BiCGStab and `psjax2d` ETDRK4):
- same-mesh **UB** (upwind + backward Euler, $\Delta t=0.005$, Newton $10^{-6}$/BiCGStab $10^{-7}$) — the discretisation
  the ROM was trained on (the named iterative solver);
- same-mesh **CC** (central + Crank–Nicolson with Rannacher start, $\Delta t=0.005$, same tolerances) — second order;
- coarse-grid **CC** at $L\in\{32,64,128\}$ and **UB** at $L=128$, mapped to the lattice by sine interpolation;
- spectral **PS** (sine pseudo-spectral ETDRK4), $N\in\{12,16,20,24,32,48,64\}$, $\Delta t=0.0125$.

3D (job at $n$; `fomrun.FOMs`, `fomcore` = E0 refcore 3D, and `psjax`):
- same-mesh **UB** and **CC**, $\Delta t=0.01$, tolerance level 1 ($10^{-6}$/$10^{-7}$); the historical Newton–BiCGStab
  `common.make_fom` UB at $(\Delta t, n_{tol}, l_{tol}) \in \{(0.005,10^{-3},0.1), (0.01,10^{-3},0.1), (0.025,10^{-3},0.1)\}$;
- coarse-grid **CC** at $L=64$ (in the 129/257 jobs it is coarse; at 65 it is same-mesh);
- **PS** $N\in\{8,12,16,20,24,32\}$, $\Delta t=0.0125$.

Every FOM solve must pass its runner's convergence predicate (`FOMs.converged`); non-converged configurations are reported
and excluded from the crossover.

**Crossover definitions.** For FOM configuration $F$ at mesh $L$: $W_F(L)$, $T_F(L)$ on the same eligible cases.
- *ROM plateau vs FOM*: the ratios $W_{\rm conv}(L)/W_F(L)$ and $T_{\rm conv}(L)/T_F(L)$.
- *Matching point* $m_\times(F, \text{family}, L)$: the smallest rung such that $W(m_j)\le W_F(L)$ and $T(m_j)\le T_F(L)$
  for it and every larger rung of the family; "none" if the plateau itself is less accurate than $F$.
- *Crossover mesh* for the same-mesh UB FOM (the comparator that improves with $L$): the bracket of meshes between which
  $W_{\rm UB}(L)$ falls below $W_{\rm conv}(L)$ (and the same for $T$).
- *Cheapest plateau rule* at $L$: among families, the rule at its $M^*$ with the lowest same-job median query latency.
- *Same-job speed-up* $= t_F / t_{\rm ROM}$ for (i) the same-mesh UB FOM vs the cheapest plateau rule, (ii) the cheapest
  FOM configuration (any family) whose $W_F\le W_{\rm conv}$ and $T_F\le T_{\rm conv}$ vs the cheapest plateau rule.
  A value below 1 means the FOM is faster. No cross-job time is ever compared.

## 8. Timing protocol (same job)

- Subjects: every ROM rule arm (end-to-end query: initial field on the mesh → six output fields on the full mesh),
  the decode alone (six fields, so solve = query − decode), and every FOM configuration of §7 (prepared input → six lattice
  fields).
- Timing cases: the first 8 cases of the cohort; 3 repetitions; all subjects compiled and warmed on every timing case first;
  a 30 s GPU burn before the first timed call and a pre-compiled 0.1 s burn before every invocation; randomised interleaved
  order (seeded); `block_until_ready` on the whole output.
- Each timed ROM output is checked against the accuracy-phase output of the same case (max abs difference / max abs
  ≤ 1e-12); each FOM output likewise.
- Latency = median of the 24 invocations; every invocation time is persisted. Drift gate: per subject, median of rep 2 /
  median of rep 0 within [1/1.10, 1.10] for the worst subject; if it fails, costs are reported as provisional.

## 9. Gates

| gate | check | on failure |
|---|---|---|
| G-env | log has `jax_backend=gpu`; x64; `JAX_DEFAULT_MATMUL_PRECISION=highest` | job void, resubmit |
| G-coh | cohort hashes equal the October values (2D `cd058fd2…`, 3D `f20a1aac…`) | job void |
| G-ref | every `refstore` load passes its sha256 check; store version equals the vendored manifest | job void |
| G-R | every rule's sha256 equals the committed manifest | job void |
| G-eval (3D) | qpanel's G1 derivative vs FD < 1e-6, G2 mesh-node assembly vs tensor < 1e-12, G3 Jacobian vs jacfwd < 1e-12 | job void |
| G-par | parity with October rollouts for shared rules/meshes: 2D f257 of test64 cases 0, 2 vs the old audit files; 3D output coefficients vs the old `coefficients.npz` (rel ≤ 1e-8), offline | investigate before reporting; numbers marked |
| G-ctl | Smolyak-8 (2D and 3D) and lattice-256 (3D) worst ST error > 2 × $W_{\rm conv}$ in every job | report; controls not used for $M^*$ |
| G-fin | every ROM rollout finite; 3D no reason-3 exit | arm excluded and reported |
| G-fom | FOM convergence predicate | configuration excluded and reported |
| G-time | timed outputs equal the accuracy phase; drift within 1.10 | costs provisional |
| G-audit | NumPy recomputation of E0 errors (2D: saved f257 of audit cases; 3D: saved coefficients decoded by `npcore`) ≤ 1e-10 relative | job void |

Integrity checklist of CLAUDE.md on every job (commit, isolated directory, no duplicate submit, logs complete, pulled with
checksums, namespace deleted).

## 10. Budget and order

Estimated from the October job logs: 2D 0.2–0.6 h per mesh, 3D 0.15–0.5 h per mesh, ≈ 3 GPU-h total. Order: local GB10
smoke (2D 256² dev6 2 cases, 3D 65 dev4 1 case, a few rungs, against the local E0 store copy); then 2D 256² and 3D 65 on the
cluster (these double as cluster smokes), then the rest. Wall limits are set so that GPU-h used + the next job's limit
≤ 4.0; if the remaining budget cannot cover a job, it is not submitted and the report states what did not run.
GPU: A100-80G/H100 for 2D ≤ 2048² and 3D 65/129; H200 for 2D 4096² and 3D 257 (mesh bank ≈ 68 GB).

## 11. Outputs

Per job `result.json` (per arm × case: m, errors ST/S per time, eligibility, LM iterations/exits, first-call seconds,
distance to conv; FOM rows; timing invocations; gates), `fields/` (2D: f257 of audit cases 0, 2 for every arm; 3D: output
coefficients of every arm × case). Report `reports/2026-11-0X-burgers-quadrature-plateau.md` generated by
`reports/make_2026-11-05-burgers-quadrature-plateau.py` from the pulled JSONs only; October provisional numbers are read
from the October lanes' committed JSONs (vendored copies with provenance), never typed.
