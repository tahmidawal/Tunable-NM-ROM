# jcp-hr-baselines (C3-minimal): POD-LSPG with ECSW and Q-DEIM hyper-reduction against the off-mesh bank ROM

**Status:** pre-registration, 2026-10-17, phase I (design → Codex design audit → code → code audit → real-size smoke).
No production job runs before the coordinator releases phase II. Amendments are appended as dated sections at the
end; nothing above them is rewritten. Governing documents: `reports/2026-10-17-full-experiment-plan/PLAN-v2.md`
group C3, `PLAN-v3-amendments.md` V3-1/V3-5, `PLAN-v4-amendments.md` V4-2/V4-5, the campaign
`briefs/COMMON-BRIEF.md` and `briefs/jcp-hr-baselines.md`. Lane `exp/2026-10-06-jcp-hr-baselines`, namespace
`/cluster/tufts/paralab/tawal01/jcp-hrb/`, concurrent-job cap 2, GPU-hour cap 30 (contingency included).

## 1. The question

Can a standard, well-tuned projection ROM with classical hyper-reduction — POD-LSPG with ECSW or with Q-DEIM — match
or beat the off-mesh coordinate-bank ROM ("ours") on viscous Burgers 2D and 3D, at equal tuning budget, in online time
per query and in E0 error? This is **H-base** (PLAN-v2 §1). It can falsify the paper's framing: if a POD baseline is
both faster and more accurate than a qualified configuration of ours on the same mesh, the claim is withdrawn for that
cell.

H-base is evaluated **per dimension and mesh**, against **each numerically qualified configuration of ours** on that
mesh (priority-1 lane `jcp-p1-scaling` in 2D, `jcp-speed3d` in 3D; §8).

## 2. Problem, cohorts, references, scoring

- **PDE.** $u_t + u\sum_{a=1}^{d} \partial_a u = \nu \Delta u$ on
  $(0,1)^d$, $u=0$ on the walls, $t\in[0,0.25]$, outputs at $t_k = 0.05k$, $k=0..5$.
  - 2D: Gaussian bump $(c_x,c_y,w,a,\nu)$, IC `engines.initial(L, desc)`, $\nu\in[0.01,0.1]$ log-uniform.
  - 3D: 1–3 masked blobs (`common.table`, IC `common.initial_interior(n, tab, j)`), $\nu\in[0.01,0.1]$.
- **Meshes (C3-minimal).** 2D $L\in\{256, 1024\}$ intervals per axis (state $(L-1)^2$); 3D $L=64$ (state $63^3$,
  i.e. the $65^3$-node mesh of ours). 2D $4096^2$ and 3D $L>64$ are **deferred to C3-full** (§12): the POD snapshot
  matrix alone would be $576\cdot12\cdot4095^2\cdot8\,\mathrm{B}\approx 0.93$ TB in 2D.
- **Cohorts** (development/validation only; never test as selection, never sealed/OOD):
  - **Selection cohort** (baseline tuning): 2D `dev6`; 3D `923801` cases 0–5.
  - **Verdict cohort** (H-base): 2D `dev6`+`val32` (38 cases); 3D `923801` all 64 cases. These are the cohorts on which
    our configurations were selected and qualified (combined/P1/speed3d), so both methods are compared in-sample on the
    same cases.
  - **Out-of-selection view** (secondary, reported beside the verdict): 2D `val32` (32); 3D `923801` cases 6–63 (58).
  - Historical test cohorts (2D `test64`, 3D `923901`) are not used in C3-minimal.
- **References and metric.** E0 ST (`refstore.py`, `JCPREF_STORE`, store version and STATUS hash pinned), metric
  $e = \max_{k\ge1}\|F_k - R_k\|/\|R_0\|$ on the scoring lattice (2D $257^2$ with walls, 3D $63^3$ interior).
  Eligibility is read **per case-kind** from the store (`reference_status`); ineligible case-kinds are scored and shown,
  never ranked. ε_ref is the store's **conditional Richardson estimate** and is labelled so. The 2D S reference is used
  only for the BE arms at Δt = 0.005 (time-matched); every other S value is an "unmatched diagnostic". 3D S is never
  used. Post-hoc labels carried forward: E0 A10; combined A10, A11 and the retracted A15.
- The t0 field of each reference is checked against our IC sampled on the lattice (≤ 1e-12), as speed3d `score3.py`.

## 3. The baselines

### 3.1 The full-order operator the baselines project

A projection ROM inherits its full-order discretisation. The baselines project the **central-difference** FOM of the
E0 references (`refcore.adv_central`, `refcore.lap`, arithmetic identical), on the baseline's own mesh $L$:

$$F_h(u) = N_h(u) - \nu\,\Delta_h u,\qquad N_h(u)_i = u_i \sum_{a=1}^{d}\frac{u_{i+e_a}-u_{i-e_a}}{2h},$$

$\Delta_h$ the 5-/7-point Laplacian, zero ghost values at the walls. This is the stronger of the two FD operators in the
repository (speed3d f1, in-job and unaudited: central-CN median 0.16–0.47 % at $64^3$ against upwind 3.7–4.1 %). The
sign-upwind operator is **not** used (first-order; it would hand the baseline a floor of several percent). A
pseudo-spectral FOM is not projected: its nonlinear term is global, so no sampling-based hyper-reduction applies.
**Baseline floor.** The un-reduced FOM with the same operator, mesh, scheme and Δt is run on the verdict cohort and
scored against E0; it is the error floor any POD-LSPG of that FOM approaches as $r\to n$, and is reported beside every
baseline number.

### 3.2 Time discretisation (same options as ours)

One residual per step, two-step linear multistep form as `t2core.SCHEMES`:

$$\mathcal R_n(u) = a_0 u + a_1 u_n + a_2 u_{n-1} + \Delta t\,\big(b_0 F_h(u) + b_1 F_h(u_n)\big).$$

| dimension | schemes | Δt₀ |
|---|---|---|
| 2D | BE $(1,-1,0,1,0)$; CN-R = 2 BE start-up steps, then CN $(1,-1,0,\tfrac12,\tfrac12)$ (the t2core `CNR`) | 0.005 |
| 3D | BE; CN without start-up (as speed3d's `fs/CN/0.01` arms) | 0.01 |

### 3.3 Snapshots and POD (training distribution only)

- **Training draws** (the ones the banks were trained on; reused, not redrawn):
  - 2D: `burgers2d_film.sample_params(seed=0, m=576)` — the 576 trajectories of the R512 bank's training run
    (`separable-decoder/runs/push_r3a/out/sep_burgers_r3_N256_K16_R512.json`: `n_traj=576, data_seed=0`). The draw is
    re-implemented in-lane and asserted equal to the vendor function; its sha256 is recorded.
  - 3D: `common.table(923701, 1536)` — the training table of both 3D banks (`model_W1024/training.json`).
  - Disjointness from every development, validation and test cohort is asserted (no row within 1e-12).
- **Generator (stated plainly).** The banks' own generator was the sign-upwind BE FOM at Δt = 0.005 (2D at $256^2$,
  3D at $33^3/65^3/129^3$); its fields were not stored. The baselines need snapshots of the operator they project, so
  the fields are **regenerated** with the §3.1 central FOM on the baseline's mesh, scheme CN with 2 BE start-up steps
  at Δt = 0.005, saving the bank's snapshot steps $\{0,1,2,4,7,10,15,20,25,30,40,50\}$ (12 per trajectory):
  2D 6,912 snapshots, 3D 18,432. Newton–BiCGStab with the exact DST preconditioner, relative residual ≤ 1e-10 per step
  (asserted; a failed trajectory aborts the job).
- **POD.** Method of snapshots in f64 without centring: $G = S S^\top$ (computed in column blocks), eigen-decomposition,
  $\Phi = S^\top V_r\Sigma_r^{-1}$, then two passes of Householder QR. Gates: $\|\Phi^\top\Phi - I\|_{\max}\le 10^{-10}$;
  $\sigma_{r_{\max}}/\sigma_1 \ge 10^{-6}$ (otherwise the method of snapshots is inaccurate and the job switches to the
  thin SVD of $S$ — logged). Reported: singular values, energy, training and selection-case projection errors at each rank.
- **Ranks.** 2D $r\in\{32, 64, 128, 256, 384\}$; 3D $r\in\{64, 128, 256, 512\}$ (nested prefixes of one basis per mesh).
- **IC and output.** $c_0 = \Phi_r^\top u_0$ (full-field orthogonal projection on the mesh); outputs $\Phi_r c$ on the
  mesh, sampled to the lattice.

### 3.4 Arms

Every arm solves, per time step, a least-squares problem $\min_c \|\hat r_n(c)\|_2$ (LSPG: Carlberg et al.) with plain
Euclidean weighting of the FOM residual (no preconditioning), differing only in how $\hat r_n$ is formed.

- **`none` — POD-LSPG without hyper-reduction** (diagnostic and the HR arms' own target): $\hat r_n(c) = \mathcal
  R_n(\Phi_r c)$, all $n$ rows; Jacobian $J = J_F\Phi_r$ by stencil algebra. Online cost $O(nr^2)$ per iteration.
- **`ecsw` — POD-LSPG with ECSW** (energy-conserving sampling and weighting for Petrov–Galerkin ROMs, Grimberg, Farhat
  et al. 2021): $\hat r_n(c) = \operatorname{diag}(\sqrt{\xi_E})\,\mathcal R_n(\Phi_r c)\big|_E$ over a sample set
  $E$ with weights $\xi\ge0$. Training: for training state $s$ (projected pair $c_s=\Phi_r^\top u_s$,
  $c_{s,n}=\Phi_r^\top u_{s-1}$) the per-node contribution to the LSPG gradient is
  $g_i(s) = (\partial\mathcal R_{n,i}/\partial c)^\top \mathcal R_{n,i} \in\mathbb R^r$; the target is the exact sum over
  **all** nodes $b(s) = J(c_s)^\top \mathcal R_n(c_s)$. Weights solve
  $\min_{\xi\ge0}\|G\xi - b\|$ by a greedy Lawson–Hanson NNLS (§3.5) stopped at $\|G\xi-b\|\le\tau\|b\|$.
  - Training states: 24 training trajectories (seeded permutation, seed 20261017) run with the central FOM at the arm's
    Δt₀ and scheme; 4 steps each (2D steps $\{1,5,15,40\}$, 3D $\{1,3,8,20\}$) → 96 states. A CN-R arm's set is the
    CN states plus the two BE start-up states of each trajectory (so its weights cover both residuals).
  - Each state's block is scaled by $1/\|b(s)\|$.
  - Candidates: 32,768 interior nodes, half uniform at random, half drawn ∝ $\sum_s\|g_i(s)\|$ (seed 20261018),
    without replacement (fewer if the mesh has fewer nodes; all nodes when $n\le 32768$).
  - τ ∈ {1e-2, 1e-3, 1e-4}, read off **one** greedy path per (mesh, scheme, r); support cap 8,192. A τ not reached
    under the cap is recorded "not reached" and that configuration does not exist.
  - Online cost $O(|E^+| r)$ for values and Jacobian rows ($E^+$ = sample nodes plus stencil neighbours), $O(|E|r^2)$
    for the normal equations.
- **`qdeim` — POD-LSPG with Q-DEIM on the nonlinear term** (DEIM, Chaturantabut–Sorensen 2010; point selection by
  pivoted QR, Drmač–Gugercin 2016), linear terms exact. $U_p$ = leading $p$ POD modes of the advection snapshots
  $N_h(u_s)$ of the same training states as §3.3; sample points $P$ = the first $p$ pivots of the column-pivoted QR of
  $U_p^\top$. With $\tilde N(c) = (P^\top U_p)^{-1} N_h(\Phi_r c)\big|_P$,
  $$\mathcal R_n(\Phi_r c) \approx \Phi_r(a_0c+a_1c_n+a_2c_{n-1}) - \Delta t\,\nu\,(\Delta_h\Phi_r)(b_0c+b_1c_n) +
  U_p\,\Delta t\,(b_0\tilde N(c)+b_1\tilde N(c_n)) = B z(c),$$
  $B = [\Phi_r,\ \Delta_h\Phi_r,\ U_p]$. With the thin SVD $B = W S V^\top$ (offline), $\|Bz\| = \|S V^\top z\|$
  exactly, so the online residual is $\hat r_n(c) = S V^\top z(c)$, of length $\le 2r+p$; no $n$-sized operation online.
  Singular values of $B$ below $10^{-13}\sigma_1$ are dropped (rank recorded). $p/r\in\{1,2,4\}$.
- The residual formulation actually solved, the sample/point counts, $|E^+|$, the retained rank of $B$ and the NNLS
  residual reached are recorded per configuration and reported (brief: "report each baseline's actual residual and
  sampling formulation").

### 3.5 NNLS

Greedy Lawson–Hanson on the normal system: maintain $A_{SS}=G_S^\top G_S$, $G^\top G_S$ and $G^\top b$ incrementally;
add the largest positive gradient entries in batches of $\max(1,\lceil 0.02|S|\rceil)$; refit by Lawson–Hanson on the
support (Cholesky of the passive block; negative weights stepped back and removed); record every τ crossing (support,
weights, relative residual). Unit-tested against `scipy.optimize.nnls` on small problems (identical support and
weights to 1e-10 when run to the same tolerance).

## 4. Solvers (native and common)

Each step starts from a predictor and stops on the same tests as ours (`t2core.tol_of('LSPG')`): **prod**
(γ ≤ 1e-3 or $\|\hat r\|\le 10^{-9}\,\|\Phi c_0\|$), **tight** (γ ≤ 1e-7), **tighter** (γ ≤ 1e-8), with
$\gamma = \|J^\top \hat r\|/(\|J\|_F\|\hat r\|+10^{-300})$.

- **Native = Gauss–Newton** (the LSPG solver of Carlberg et al.): predictor $c_n$; full step
  $\Delta = \arg\min\|J\Delta+\hat r\|$ by Householder QR of $J$; at most 30 iterations; exit codes: converged,
  budget, tiny step ($\|\Delta\|\le 10^{-14}(1+\|c\|)$), non-finite.
- **Common = ours** (`t2core.make_lm`, copied byte-identical with provenance): Levenberg–Marquardt with diagonal
  damping, Cholesky normal equations, trust clip, budget 600, and the 3-candidate predictor of `t2core.make_evolve`
  (previous, linear and quadratic extrapolation, smallest residual). Trust radius
  $0.01\max_s\|c_s-\bar c\|$ over the training projected coefficients at rank $r$ (the definition of
  `qcore.Model.trust_linear`).
- The common-solver ablation runs every configuration of the grid with both solvers.

## 5. Q1 numerical qualification (applies to every baseline configuration; V4-2)

Certified **in-job at every step with a fresh Jacobian** (r̂ and J recomputed at the returned state):
1. every run valid: finite coefficients, no non-finite step, no damping saturation (LM λ at its 1e14 limit) and no
   GN non-finite exit;
2. stationarity: a step is non-stationary if γ > 1e-3 or non-finite; non-stationary steps ≤ 1 % **pooled over all
   steps** of the configuration on the cohort;
3. distance to the adaptive tight solve ≤ 2.5e-4 on **every** case, where the tight solve is the common LM solver at
   the tight level on the **same** discrete problem (same Φ_r, HR data, scheme, Δt), and the distance is
   $\max_k\|F_k-F_k^{\rm tight}\|/\|F^{\rm tight}_0\|$ on the scoring lattice;
4. tight vs tighter agreement ≤ 2.5e-5 on the prescribed subset = the first two cases of the cohort, per configuration.

Selection-stage Q1 uses the selection cohort; the verdict uses Q1 recomputed on the verdict cohort. An unqualified
configuration is reported (with its numbers) and never ranked. Labels kept separate (V4-2): numerical qualification
(Q1), role qualification (the pick rules of §6), frontier usefulness (non-dominated). There is no quadrature in the
baselines; the HR analogue of Q2 — the distance between an HR arm and the `none` arm of the same rank on the arm's
own cases — is **reported, not gated**.

## 6. Tuning grid, budget and selection (frozen before any data)

**Grid per (dimension, mesh):** ranks (§3.3) × HR ∈ {none, ecsw τ∈{1e-2,1e-3,1e-4}, qdeim p/r∈{1,2,4}} × 2 schemes ×
2 solvers: 2D 140 configurations per mesh, 3D 112. Every configuration runs on the selection cohort at prod level, plus
the tight anchor (one per discrete problem, shared by both solvers) and the tighter anchor on 2 cases.

**Tuning budget.** All baseline GPU-hours spent on the selection cohort (snapshots, POD, HR training, the grid, its
anchors) are the baseline's tuning budget, measured from `sacct` in `runs/GPU-LEDGER.json`. Ours' validation tuning
budget is read from the combined and speed3d ledgers (2D: combined `c2grid` and the P1 selection-relevant jobs; 3D:
combined `c3desc*`, `c3grid`, speed3d `g1–g3`) and reported beside it. If the baseline's budget is the smaller one, the
baseline is labelled **"tuned with less budget than ours"** in every verdict (this weakens only claims in our favour).

**Selection (on the selection cohort; deterministic):**
- Screening statistics per configuration: Q1 pass/fail, median and worst E0 ST error over eligible cases, median
  full-query time per case (in-job, warm, single runs; not the verdict timing).
- **Baseline front:** Q1-qualified configurations not dominated in (median query time, worst error).
- For each of our qualified configurations X on that mesh (§8), two **picks**, both from the baseline front:
  - **B_acc(X)** (equal accuracy): the fastest front configuration whose selection worst error ≤ X's selection worst
    error; ties → smaller median error, then smaller r.
  - **B_cost(X)** (equal cost): the most accurate (worst error) front configuration whose median query time ≤ X's
    median time measured in the same tuning job; ties → faster.
  - A pick may be "none" (no configuration qualifies for the role); reported, never filled in.
- The selection is written to `checks/FROZEN-SELECTION-<cell>.json` with sha256, Codex-audited, before the verdict job.

## 7. Timing contract

Full-field contract as P1c/speed3d: one query = initialisation from the full-field IC on the mesh + time stepping to
$t=0.25$ + decode of the six full-field outputs on the mesh, `block_until_ready` on the result. Ours keeps its own
initialisation (2D: IC sampled at the cold quadrature points, reduced projection, V4-4; 3D: `TB.project`, the mesh L2
projection); the baselines use $\Phi_r^\top u_0$. Solve-only (stepping) time, initialisation and decode are also timed
separately and reported. **Primary time objective: full-query time.**

**Paired A–B–A in the verdict job, one GPU:** for each tested pair (X, B), on the 6 timing cases (2D `dev6`, 3D
`923801` 0–5) × 3 repeats, the order A=X, B, A=X with a 0.1 s burn before each call and warm JIT; ratio
$\rho = t_B/\tfrac12(t_{A,1}+t_{A,2})$. Per pair: the median ρ over the 18 values and its 95 % percentile interval
from a case-level bootstrap (10,000 resamples, seed 0, cases resampled with their repeats). Drift check: within each
triple $|t_{A,1}/t_{A,2}-1|\le 0.25$, else the triple is repeated once and flagged. The timed outputs of A and B are
the scored outputs (asserted equal to the separately scored runs to 1e-12).

## 8. Ours: which configurations, and how their numbers enter

- **2D:** every configuration that `jcp-p1-scaling` reports numerically qualified (Q1+Q2) at $256^2$ or $1024^2$, with
  its per-case E0 errors on the verdict cohort from its `runs/EXPORT.json` (expected: r ∈ {128, 256, 384} × {LSPG-BE,
  LSPG-CN-R} at Δt₀ with the frozen rules).
- **3D:** every configuration that `jcp-speed3d` reports Q1-qualified at $65^3$ in its `runs/EXPORT.json`
  (full-field contract arms only; m1's mesh-free contract is not used).
- The exact export files (branch, commit, sha256) are pinned in a dated amendment at phase-II release, after those
  lanes' results audits pass. If a lane has not passed its results audit, that cell is **not compared** (reported).
- Ours' queries are re-timed in the verdict job (A of the A–B–A) with code copied byte-identically from the owning lane
  (`ours/INHERITED.json`: source branch, commit, sha256). The re-timed run's lattice fields must reproduce the exported
  errors to 1e-10 relative on the timing cases (gate G-ours); a mismatch stops the comparison for that X.

## 9. Decision rule (H-base; V3-5 as amended by V4-5) — frozen

Objectives and directions: full-query time (lower), offline cost (lower), median E0 error (lower), worst E0 error
(lower). For each tested pair (X, B) with B ∈ {B_acc(X), B_cost(X)} on the verdict cohort's eligible ST case-kinds:

- **Error intervals** (endpoint propagation, as combined): per case $[e_i-\varepsilon_i, e_i+\varepsilon_i]$ with
  ε_i = ε_ref; worst $[\max_i(e_i-\varepsilon_i), \max_i(e_i+\varepsilon_i)]$; median likewise.
  "B better in worst error" iff B's upper endpoint < X's lower endpoint (and conversely).
- **Time:** "B faster" iff the upper end of the 95 % interval of median ρ < 1; "X faster" iff its lower end > 1.
- **Pair labels:**
  - **baseline dominates ours** — B faster **and** better in worst error;
  - **ours dominates** — X faster **and** better in worst error;
  - **neither dominates** — each better on one of the two objectives (fronts cross); never "matches or beats";
  - **matches** — only via TOST: the 90 % case-bootstrap interval of the mean of
    $x_i = (e_{B,i}-e_{X,i})/\delta_i$, $\delta_i=\max(2\varepsilon_i, 0.1\,e_{X,i})$, lies inside $(-1,1)$, **and** the
    90 % interval of median $\log\rho$ lies inside $\pm\log 1.10$;
  - **unresolved** — everything else (overlapping intervals are never "matches").
- Offline cost and median error are reported beside every label.
- **Cell verdict (per dimension, mesh, X):** "H-base falsified at X" iff at least one of the two picks dominates X.
  The dimension-level statement lists every X; it never aggregates "falsified at some X" into "holds".
- Unqualified (Q1-fail) baseline configurations cannot dominate anything; if every candidate for a pick is
  unqualified, the pick is "none".

## 10. Controls (V3-7): positive gates, negative controls, sensitivity probes

| id | kind | check | must | blocks if it misbehaves |
|---|---|---|---|---|
| G-fom | positive | in-lane central FOM vs `refcore.make_solver(adv='central')` on one case per dimension at the real mesh, schemes BE and CN | equal to 1e-10 relative (lattice) | baseline floor, snapshots, every baseline |
| G-full | positive | complete basis ($r=n$) and all nodes (`none`; `ecsw` with ξ=1 on all nodes; `qdeim` with p=n) reproduces the FOM, 2D L=16 and 3D L=8, both schemes, both solvers | ≤ 1e-9 relative (to solver tolerance) | every baseline |
| G-all | positive | at real size: `ecsw` with ξ=1 on all nodes equals `none` at the same rank on one case | ≤ 1e-10 | ECSW implementation |
| G-fd | positive | each arm's analytic $\hat r$ Jacobian vs central finite differences (5 random directions, step $10^{-6}\|c\|$) and vs `jax.jacfwd`; and the arm's row values vs the independent `refcore` residual on the sample rows | FD ≤ 1e-6 rel., jacfwd ≤ 1e-11, rows ≤ 1e-12 | every baseline |
| N-fd | negative | the same check on a mutant Jacobian (advection Jacobian term dropped) | FD check fails | G-fd |
| G-nnls | positive | NNLS vs `scipy.optimize.nnls` on small random problems | equal to 1e-10 | ECSW |
| G-qdeim | positive | Q-DEIM interpolation is exact on the span of $U_p$ (random combination) and the interpolation constant $\|(P^\top U_p)^{-1}\|$ is finite and reported | ≤ 1e-10 | Q-DEIM |
| G-pod | positive | orthonormality ≤ 1e-10; projection of a training snapshot at full available rank reproduces it to ≤ 1e-8 | | every baseline |
| N-under | negative | under-sampled HR: ECSW with support capped at ⌈r/4⌉ nodes, Q-DEIM with p = ⌈r/4⌉, at the smoke rank, on the smoke cases | median E0 error ≥ 2× the `none` arm's at the same rank, **or** a Q1 failure (invalid run or > 1 % non-stationary) | the HR arms' accuracy claims |
| N-grad | negative | Q1 certifier on a wrong-gradient solver (LM with the Jacobian scaled by 0.5 in the step) | non-stationary fraction > 1 % or distance > 2.5e-4 → Q1 fail | Q1 |
| N-swap | negative | scoring with case swap, time shift, ST↔S swap | each breaks agreement with the true score by > 10 ε_ref | scoring |
| G-score | positive | offline CPU re-decode of saved coefficients reproduces every in-job score | ≤ 1e-12 | every reported error |
| N-time | negative | A–B–A harness: B = A run twice per call | median ρ in [1.8, 2.2], label "X faster" | timing |
| N-dom | negative | decision rule on synthetic data: B = X with errors × 0.5 and times × 0.5 must give "baseline dominates"; errors × 1.0 and times × 1.0 with small noise must give "matches" or "unresolved", never "dominates" | as stated | §9 |
| G-ours | positive | ours re-timed reproduces its exported errors | ≤ 1e-10 relative | ours' side of every pair |

Every control is shown to fire in the phase-I real-size smoke (2D $1024^2$ and 3D $65^3$, reduced training set) or,
where marked small-mesh (G-full, G-nnls), in the CPU unit tests; a control that cannot fire on real data is a bug.
**Sensitivity probes (reported only):** GPU type of the timing job; the NNLS candidate seed (one alternative seed at
one configuration per cell).

## 11. Offline cost

Reported per cell, in GPU-seconds measured in-job: snapshot generation, POD (Gram, eigen-decomposition,
orthonormalisation), advection POD and Q-DEIM pivoting, ECSW training trajectories, contributions and NNLS paths,
compilation. Ours: bank training (sunk; 2D R512 and 3D W1024 1.85 H200-h job total / 1.14 h bank fit, from the
lanes' logs) and per-mesh table/rule setup (from P1f / speed3d). Shared versus per-configuration costs are kept apart.
Break-even query counts are computed only where one method is accuracy-qualified and has a positive per-query saving.

## 12. Jobs, memory, GPU-hours, defer order

```mermaid
flowchart LR
  D[DESIGN + Codex audit]:::doc --> C[code + Codex code audit]:::doc --> S[smoke job: 2D 1024², 3D 65³, reduced training set, every control]:::job
  S --> R{{coordinator release, phase II}}:::gate
  R --> T1[t2d256 tuning]:::job & T2[t2d1024 tuning]:::job & T3[t3d64 tuning]:::job
  T1 & T2 & T3 --> F[offline CPU scoring + frozen selection + Codex audit]:::doc
  F --> E1[e2d256 verdict]:::job & E2[e2d1024 verdict]:::job & E3[e3d64 verdict]:::job
  E1 & E2 & E3 --> A[results audit, make_report.py, report audit]:::doc
  classDef doc fill:#e5e7eb,stroke:#374151; classDef job fill:#dbeafe,stroke:#1d4ed8; classDef gate fill:#fde68a,stroke:#b45309;
```

- One job per directory under `jcp-hrb/<job>/`, ≤ 2 concurrent, account total ≤ 8 checked before each submit.
- **Memory.** Snapshots on the host (f64): 2D $1024^2$ 57.9 GB, 3D 36.9 GB; streamed in column blocks to the device for
  the Gram. Host memory request 200 GB for those two cells. Device: $\Phi$ (2D $1024^2$, r=384: 3.2 GB), ECSW
  contribution matrix (≤ $96\cdot512\times32768$ f64 = 12.9 GB in 3D), the `none` arm's $J$ (2D $1024^2$, r=384:
  3.2 GB). A100-80 GB suffices; H200 if the smoke's measured peak exceeds 60 GB.
- **Estimate** (to be replaced by a smoke-derived certificate, speed3d `resources.py` pattern): smoke ≤ 1;
  tuning 2D256 0.5, 2D1024 2.5, 3D64 2.5; verdict 1.0 + 1.5 + 1.5; total ≈ 10.5 GPU-h, × 1.25 contingency ≈ 13 GPU-h,
  inside the 30 GPU-h cap. A job is submitted only if ledger-spent + certified projection ≤ 30.
- **Defer order** (first dropped first): `none` arms at r ≥ 256 (diagnostic) → the 2D $256^2$ cell → the 3D r = 512
  rank → common-solver runs of non-front configurations. Never dropped: the 2D $1024^2$ and 3D $65^3$ ECSW and Q-DEIM
  grids with the native solver, their picks and the verdict jobs. Dropped items are reported as deferred.
- Cluster space: the tuning job's $\Phi$ and HR data (≤ 5 GB per cell) stay in its job dir until the verdict job has
  read them (sha256-checked), then both dirs are pulled and deleted.

## 13. Deliverables

`DESIGN.md` (+ amendments), `audits/` (every Codex file), `runs/GPU-LEDGER.json`, `runs/EXPORT.json` (per pair: picks,
labels, intervals, offline costs; per configuration: Q1, errors, times), `make_report.py` → `report.md` with plots:
baseline fronts against ours per cell (time vs worst error, with Q1 failures shown separately), POD singular values and
projection floors, ECSW/Q-DEIM sample count vs error, native vs common solver, offline-cost bars. Every number in the
report is generated. Glossary at the end.

## 14. Appendix: C3-full (priority 4), design sketch only — not implemented in phase I

- **GNAT** (Carlberg et al. 2013): residual and Jacobian gappy POD with oversampled sample mesh ($n_s > n_r \ge r$),
  Q-DEIM/greedy sampling; reuses §3.4's stencil machinery.
- **POD + exact tensor:** the quadratic advection term precomputed as $T_{ijk} = \phi_i^\top(\phi_j\odot D\phi_k)$ for
  the central operator ($O(r^3)$ storage, exact, no sampling); feasible for r ≤ 256 (2D: 128 MB at r = 256).
- **Best-practice EQ on our bank:** NNLS empirical quadrature (Yano/Patera style) on the bank's weak residual with
  the same candidate/target recipe as §3.4, replacing the classical rules.
- **Continuum ECSW:** weights on the off-mesh points of ours rather than mesh nodes.
- **CROM-style sampler:** a continuous reduced-order model's sampled-point network (Chen et al. 2023) with the same
  training data.
- **Larger meshes:** 2D $4096^2$ and 3D $L\in\{128,256\}$ via a streaming randomised range finder (two passes,
  regenerating trajectories) instead of the in-memory method of snapshots.
