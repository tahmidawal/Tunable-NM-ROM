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

---

## Amendment A1 (2026-10-18) — answers to Codex design audit 1 (`audits/codex-design-1.md`, verdict WRONG)

Each item below replaces the text it names; where A1 and the sections above differ, A1 governs. Item numbers are the
audit's.

**A1.1 (items 1, 2) Residual and Jacobian, explicit.** The baselines solve weighted strong-residual LSPG ("ECSW" names
the sampling-and-weighting construction; no energy-conservation property is claimed for this Burgers discretisation).
With $u=\Phi_r c$ and $s(u)_i=\tfrac L2\sum_a(u_{i+e_a}-u_{i-e_a})$,
$$J(c)=\partial_c\mathcal R_n(\Phi_rc)=\Big[a_0 I+\Delta t\,b_0\big(\operatorname{diag}(s(u))+\operatorname{diag}(u)\,D_h-\nu\Delta_h\big)\Big]\Phi_r,$$
$D_h$ the central derivative-sum operator ($s(u)=D_hu$). The same expression (restricted to rows, times $\sqrt{\xi}$) is
used for ECSW online rows and for ECSW training contributions; G-fd checks it.

**A1.2 (item 3) Q-DEIM objective without truncation.** The SVD with a singular-value cut-off is withdrawn. Offline, the
Householder QR $B=[\Phi_r,\ \Delta_h\Phi_r,\ U_p]=Q_BR_B$ is formed in f64 (Householder QR is column-wise backward
stable, so the very different column scales of $\Phi_r$ and $\Delta_h\Phi_r$ are harmless); online
$\hat r_n(c)=R_Bz(c)$, so $\|\hat r_n(c)\|=\|Bz(c)\|$ exactly for every $c$ and no linear term is approximated. $R_B$ is
never inverted, so rank deficiency of $B$ does not matter. The DEIM approximation is the only approximation; G-fd
compares $\|R_Bz\|$ and $J^\top\hat r$ with the independently assembled full-field DEIM residual
$\Phi_r(\cdot)-\Delta t\nu\Delta_h\Phi_r(\cdot)+U_p(P^\top U_p)^{-1}(\cdot)$ and its gradient on random states,
including a near-cancellation state (the residual at a converged step).

**A1.3 (items 4, 5) Snapshot provenance, restated.**
- Reused: the **parameter draws** of the banks' training (2D `sample_params(0, 576)`, 3D `table(923701, 1536)`). Not
  reused: the banks' **trajectory generator** (sign-upwind BE, fields never stored) and, in 2D, the bank's **state
  selection** (all 51 states of 576 trajectories, of which all states with step ≤ 5 plus a random sample, 16,384 in total;
  `sep_burgers_r3.py:321–332`). The baselines' snapshots are the §3.1 central FOM's own trajectories; this is stated as
  "same parameter distribution and draws, different trajectory generator", with its offline cost reported.
- 3D: the 3D banks' 12 snapshot steps (`model_W1024/training.json`) are reused: 18,432 snapshots.
- 2D: the same 12 steps are a **lane choice** (memory: all 51 steps at $1024^2$ would be 246 GB), giving 6,912 snapshots.
  **Training-only saturation check** at $256^2$ (where all 51 steps fit, 15 GB): POD from the 12-step set versus POD from
  all 29,376 states; statistic = RMS relative projection error of all 29,376 training states at each rank. Adequate iff the
  12-step value ≤ 1.10 × the all-step value at every registered rank. If not adequate, the $256^2$ cell uses the all-step
  POD and the $1024^2$ cell is labelled "snapshot-limited (12-step set; saturation check failed)"; nothing else changes.

**A1.4 (item 6) Same time-stepping options as ours.** 2D: LSPG-BE and LSPG-CN-R at Δt₀ = 0.005 (P1's registered
options). 3D: **speed3d's full registered ladder**: schemes BE, CN (no start-up), BDF2 (one BE start-up step,
t2core `BDF2`) × Δt/Δt₀ ∈ {1, 1.25, 5/3, 2.5} (Δt₀ = 0.01), 12 options. Scheme coefficients, step and step count are
traced, so options add runs, not compilations.

**A1.5 (item 7) Tuning budget, frozen accounting.** Four categories, measured from `sacct` and in-job timers, for both
methods: (C-model) model construction — ours: bank training; baseline: snapshots + POD; (C-hr) hyper-reduction
construction — ours: rule tables and rule selection; baseline: advection POD, Q-DEIM pivoting, ECSW training runs and
NNLS; (C-search) validation search — runs whose purpose is choosing a configuration; (C-cert) numerical certification
(anchors, Q1 passes). **Equal tuning budget applies to C-search**: the baseline may spend on C-search up to ours' C-search
GPU-hours for that dimension (2D: combined `c2grid` + `c2fbgrid`; 3D: combined `c3desc*` + `c3grid` and speed3d
`g1–g3`, from their ledgers; read and pinned at phase-II release), within this lane's cap. The registered grid is run
first. **Budget-driven extension (pre-registered):** while budget remains and a screening-front configuration sits on a
grid boundary, the grid is extended one step in that direction (rank +1 rung: 2D 512, 3D 768; ECSW τ = 3e-5; Q-DEIM
p/r = 8) and screened again. If ours' C-search budget cannot be matched inside the cap, H-base for that cell is
restated as an **unequal-budget descriptive comparison** (named in every verdict line). C-model, C-hr and C-cert are
reported for both methods; ours' C-model is sunk (shared bank) and reported as such, never omitted.

**A1.6 (items 8, 17 in part) Cohorts, restated: one in-sample developmental comparison.** Ours was selected in-sample
on the verdict cohorts (2D dev6+val32; 3D all 64 of 923801), with post-hoc steps (combined A10/A11) and against earlier
references. The baseline is now selected on the **same full cohort**:
- Stage A (screening, cheap): the whole grid on the selection subset (2D dev6; 3D 923801 cases 0–5).
- Stage B: the screening front plus every configuration within a factor 1.5 of the front in worst error or in median
  time (at most 24, ranked by the total order of A1.12) is run on the **full verdict cohort** with Q1 certification.
- Picks (§6 rules) are made on the full cohort's Stage-B data. The verdict is read on the same data.
The result is labelled "**developmental comparison, both methods selected in-sample on the same cases, with unequal
selection histories**". The "out-of-selection view" of §2 is withdrawn. No generalisation claim is made; that needs a
jointly frozen cohort later (F1/F2).

**A1.7 (item 9) Q1 restated exactly.** Distance = $\max_{k\ge1}\|F_k-F_k^{\rm tight}\|/n_0$ on the scoring lattice,
$n_0$ = the canonical physical IC norm on the lattice (the E0 metric's $n_0$; checked equal to the store's
`judge.n0` to 1e-12), not the POD-projected IC. Tight and tighter anchors must themselves be **valid** (every step
stationary at their level or at residual tolerance, finite, no damping saturation); an invalid anchor makes the
configuration unqualified ("anchor invalid"), never qualified. The tolerance table is `c2common.tol_of('LSPG', ·)`
(the earlier citation `t2core.tol_of` was wrong). Certification is a separate untimed pass over the saved trajectory
$c_0..c_N$ (speed3d's pattern): for every step it recomputes $\hat r$ and $J$ at $c_{n+1}$ with the history of that
step and records $\gamma_n$; a step counts as stationary iff $\gamma_n\le10^{-3}$, or its residual is at the solve's
residual tolerance ($\|\hat r\|\le$ tol, an exact root; t2core A1.5). Timed queries do not certify (no extra work in
the timing); the certified trajectory equals the timed one (asserted, ≤ 1e-12).

**A1.8 (item 10) ECSW training, restated.**
- Training trajectories: 24 (seeded permutation of the training draw, seed 20261017), each assigned round-robin to one
  time-stepping option (2D: 12 per option; 3D: 2 per option), run with the central FOM; 4 states per trajectory at steps
  $\{k_0, \lceil0.15N\rceil, \lceil0.4N\rceil, N\}$ of that option ($N$ its step count; $k_0$ = 1 for BE and CN, 3 for
  CN-R, 2 for BDF2, i.e. the first step with the option's main coefficients; duplicates removed). 96 states in all,
  pooled over options, so one weight set per (rank, τ) serves every option.
- Block scaling $1/\max(\|b(s)\|,\ 10^{-8}\,\mathrm{median}_s\|b(s)\|)$; a block with $\|b(s)\|<10^{-8}$ median is dropped
  and recorded.
- Every attempted (rank, τ) is reported with the achieved relative residual and support. A τ not reached under the
  8,192 cap is retried once with cap 16,384 (training-only fallback); if still not reached it is a recorded **ECSW
  construction failure** at that τ (reported in the results; the other τ values still run).
- Candidates as in §3.4 (32,768; all nodes when $n\le$ 32,768).

**A1.9 (items 11–14) Controls, restated.** Replaces the table of §10 where they differ.

| id | kind | fixture | must |
|---|---|---|---|
| G-fd-a | positive | raw stencil residual and Jacobian (hrcore) vs `refcore` F (byte-identical copy) and `jax.jacfwd` of it, random state, real mesh | rows ≤ 1e-12 rel.; Jacobian ≤ 1e-11 |
| G-fd-b | positive | each arm's online $\hat r$, $J$: directional FD (5 random unit directions $v$, step $h=10^{-6}\max(1,\|c\|)$, central) and `jacfwd` | FD ≤ 1e-6 rel.; jacfwd ≤ 1e-11 |
| G-fd-c | positive | ECSW rows = $\sqrt\xi$ × independently computed full residual rows at E; Q-DEIM $\|R_Bz\|$ and $J^\top\hat r$ = assembled full-field DEIM residual norm and gradient (A1.2), incl. a near-cancellation state | ≤ 1e-10 rel. |
| N-fd | negative | G-fd-b on the mutant with the advection Jacobian dropped | fails (> 1e-3 rel.) |
| G-full | positive | 2D L = 16 and 3D L = 8: completed orthonormal basis $\Phi=Q$ of $[\Phi_{\rm POD}, I]$ ($r=n$), `none`, `ecsw` with ξ = 1 on all nodes, `qdeim` with $U=I_n$ (p = n, P = all nodes); every scheme option; both solvers at **tight** level; FOM at ntol 1e-13 | lattice distance to the FOM ≤ 1e-8 |
| G-all | positive | real mesh: `ecsw` with ξ = 1 on all nodes vs `none`, same rank, one case, BE | ≤ 1e-10 |
| G-nnls | positive | greedy NNLS run without τ-stop vs `scipy.optimize.nnls` on random problems: objective equal to 1e-10 rel. and KKT (ξ ≥ 0, gradient ≤ 1e-9 on the active set and ≥ −1e-9 off it); at a τ-stop: ξ ≥ 0 and residual ≤ τ | as stated |
| G-qdeim | positive | Q-DEIM interpolation exact on span($U_p$) (random combination, ≤ 1e-10); GPU pivoting equals `scipy.linalg.qr(pivoting=True)` pivots on a small problem; $\|(P^\top U_p)^{-1}\|$ finite, reported | as stated |
| G-pod | positive | $\|\Phi^\top\Phi-I\|_{\max}\le10^{-10}$; a training snapshot projected onto the full computed rank reproduces itself to ≤ 1e-8 relative | as stated |
| N-under | negative | fixtures fixed now, without outcomes: in each smoke cell the two selection-subset cases with the smallest ν (ties → larger amplitude). Ladder ECSW support cap ⌈r/4⌉, ⌈r/8⌉; Q-DEIM p = ⌈r/8⌉, ⌈r/16⌉, …, 1, at the smoke rank; for each level report Q1 (complete predicate incl. distance to its own tight solve) and E0 error | fires iff, at some ladder level, the median E0 error on the fixtures ≥ 2 × the `none` arm's at the same rank (degraded accuracy, the brief's requirement); Q1 failure is reported separately as solver failure, not counted as degradation. If no level degrades, that is recorded as an insensitivity finding and the HR arms' "sampling matters" claim is blocked |
| N-grad | negative | fixture: the N-under fixture case with the smallest ν, ECSW at the smoke rank, BE; the solver uses the Jacobian **without its advection part** while certification recomputes the correct $J$ | Q1 fails (non-stationary > 1 % or distance > 2.5e-4) |
| N-swap | negative | scoring: case swap, time shift (both > 10 ε_ref change), and kind control: requesting S for a 3D case or for a non-BE / non-0.005 arm raises | as stated |
| G-score | positive | offline CPU re-decode of saved coefficients reproduces every in-job score | ≤ 1e-12 |
| N-time | negative | A–B–A with B = A run twice per call | median ρ ∈ [1.8, 2.2] and label "X faster" |
| N-dom | negative | decision rule, synthetic: 38 cases, $e_X\sim$ LogU[1e-3, 1e-2], ε_i = 1e-6; (a) $e_B=0.5e_X$, ρ = 0.5 ± 1 % → "baseline dominates"; (b) $e_B=e_X(1\pm0.5\%)$, ρ = 1 ± 1 % → "matches"; (c) $e_B=0.5e_X$ except one case at 2 $e_X^{\max}$, ρ = 0.5 → not "baseline dominates" (worst error) | as stated |
| G-ours | positive | ours re-timed reproduces its exported errors | ≤ 1e-10 rel. |

**A1.10 (item 15) Envelopes.** Lower endpoints are clipped at 0. The intervals of §9 are **conditional
reference-error envelopes** (conditional on ε_ref, a Richardson estimate), not statistical confidence intervals for a
population worst case.

**A1.11 (item 16) "Matches", restated.** "Matches" requires all three: (i) **mean standardized error equivalence** —
$x_i=(e_{B,i}-e_{X,i})/\delta_i$ with $\delta_i=\max(2\varepsilon_i,0.1e_{X,i})$ fixed per case (not recomputed in
resampling), 90 % case-bootstrap interval of mean $x$ inside $(-1,1)$; (ii) **worst-error equivalence** — B's worst
envelope lies inside $[W_X-\delta_W, W_X+\delta_W]$, $\delta_W=\max(2\varepsilon_{i^*},0.1W_X)$, $i^*$ = X's
worst case; (iii) **time equivalence** — 90 % case-bootstrap interval of the median $\log\rho$ inside $\pm\log1.10$,
cases resampled with all their repeats. Bootstraps: 10,000 resamples, seed 0, percentile intervals.

**A1.12 (item 17) Exhaustive decision table and total order.**
- Total order on baseline configurations (ties anywhere in §6 and A1.6): worst error, median error, median time, r,
  HR kind (none < ecsw < qdeim), HR size (fewer points first), scheme name, Δt, solver (native < common).
- Pair label precedence: (1) **baseline dominates** / **ours dominates** (both objectives resolved in the same
  direction; annotated "within equivalence margins" if A1.11 also holds); (2) **neither dominates** (both resolved, in
  opposite directions); (3) **matches** (A1.11); (4) **unresolved**.
- Cell at X: qualified baseline front empty → "**baseline incomplete** at this mesh" (not assessed; never support). A
  pick "none" → "no baseline in role". **"H-base falsified at X"** iff a registered pick dominates X. Otherwise
  "not falsified by the registered picks" — a narrow statement: it is not proof that no baseline dominates.
- **Exploratory:** every Stage-B Q1-qualified configuration whose worst envelope is not entirely above X's is A–B–A
  timed against X and labelled with the same table, marked **exploratory, unadjusted**. Intervals are unadjusted for the
  number of X and of pairs; dimension-level statements list every X and are labelled exploratory.
- X with no eligible case-kind or missing exported errors → "not assessed".

**A1.13 (items 18, 19) Memory, disk and workload inventory.** Peak live arrays (f64), with the phase they live in:

| phase | 2D $1024^2$ | 3D $65^3$ |
|---|---|---|
| snapshots (host) | 6,912 × 1,046,529: 57.9 GB | 18,432 × 250,047: 36.9 GB |
| POD Gram / eigen (device) | 0.38 GB | 2.7 GB |
| advection snapshots (host, overwrite the state snapshots after POD and the trust spread are done) | 57.9 GB (same buffer) | 36.9 GB |
| $\Phi$ (r_max) + $\Delta_h\Phi$ + $D_h\Phi$ (device) | 3 × 3.2 GB | 3 × 1.0 GB |
| $U_{p_{\max}}$ (p ≤ 4 r_max) | 12.9 GB | 4.1 GB |
| Q-DEIM pivoting residual matrix ($p\times n$) | ≤ 12.9 GB | ≤ 4.1 GB |
| $B$ for one (r, p) + QR workspace | ≤ 2 × 19.3 GB | ≤ 2 × 6.1 GB |
| ECSW contribution matrix ($96r\times32768$) + $G^\top G_S$ ($32768\times$ support) | 9.7 + 2.1 GB | 12.9 + 2.1 GB |
| `none` arm $J$ + QR copy | 2 × 3.2 GB | 2 × 1.0 GB |

Phases run sequentially and free their arrays (`del` + the device's own accounting); the device peak is the largest
phase (2D $1024^2$: $B$ QR ≈ 39 GB + resident $\Phi$ blocks ≈ 10 GB), host peak ≈ 60 GB (2D) / 40 GB (3D). **H200**
(143,771 MiB) for the 2D $1024^2$ and 3D cells; A100-80 GB for $256^2$. Measured peaks from the smoke replace these.
Disk (cluster, per cell): $\Phi_{r_{\max}}$ (3.2 / 1.0 GB) and the compact online HR data (per configuration: stencil
rows ≤ 8,192 × 7 × r, $R_B$, $(P^\top U)^{-1}$; ≤ 1.5 GB in all) are kept for the verdict job; full $U$, $B$ and
snapshots are never written. Local: coefficients and lattice-row bases only (≤ 2 GB per cell).

Workload inventory per cell (prod runs on cases; anchors per discrete problem; certification passes = prod runs):

| item | 2D $256^2$ | 2D $1024^2$ | 3D $65^3$ |
|---|---|---|---|
| FOM snapshot trajectories | 576 | 576 | 1,536 |
| ECSW training FOM runs | 24 | 24 | 24 |
| full-rank limiting comparator (FOM) runs | 38 × 2 | 38 × 2 | 64 × 12 |
| configurations (ranks × HR × options × solvers) | 5×7×2×2 = 140 | 140 | 4×7×12×2 = 672 |
| Stage A prod runs (6 cases) | 840 | 840 | 4,032 |
| Stage A tight anchors / tighter (2 cases) | 420 / 140 | 420 / 140 | 2,016 / 672 |
| Stage B (≤ 24 configs × cohort) prod + tight + tighter | ≤ 912 + 912 + 48 | same | ≤ 1,536 + 1,536 + 48 |
| compiled query shapes (kind × r × HR size × solver) + cert passes | 70 + 35 | 70 + 35 | 56 + 28 |
| Q-DEIM pivoting runs / $B$ QRs / NNLS paths | 9 / 15 / 5 | 9 / 15 / 5 | 9 / 12 / 4 |
| verdict job: A–B–A triples | (picks + exploratory) × X × 6 × 3 | same | same |

The GPU-hour projection is computed from the smoke's measured per-item times by `resources.py` (speed3d pattern) and
must fit: spent + projection ≤ 30. If the registered core (A1.14) does not fit, the cell is declared **incomplete**.

**A1.14 Defer order, restated** (first dropped first; each dependency-safe): exploratory A–B–A pairs → the 2D $256^2$
cell → the 3D options with Δt/Δt₀ = 2.5 → the 3D r = 512 rank → common-solver runs of configurations that are not on
the native-solver screening front (the front is decided from native runs plus the common runs already made; a dropped
common run is reported as "not run"). Never dropped: `none` with the native solver at every rank (the HR-error
diagnostic), the ECSW and Q-DEIM grids with the native solver, Stage B, picks, the verdict jobs.

**A1.15 (item 20) Solvers, comparator naming, spectral.**
- **Native solver = damped Gauss–Newton**: the QR least-squares step with backtracking (step halved up to 8 times
  until $\|\hat r\|$ decreases; if no halving decreases it, exit reason 3), budget 30, previous-state predictor. Its
  convergence sensitivity is reported through the native-vs-common ablation.
- **Common solver trust per dimension, as ours:** 2D $0.01\max_s\|c_s-\bar c\|$ (`qcore.trust_linear`); 3D
  $0.05\sqrt{\mathrm{mean}_s\|c_s-\bar c\|^2}$ (speed3d trust = 0.05 × the bank's `coefficient_rms_spread`,
  `train2w.py:439–441`), over the training projected coefficients at rank r.
- The un-reduced FOM is named the **full-rank limiting comparator**, not a floor: a reduced model can beat it by error
  cancellation.
- Spectral FOM projection is deferred for scope and cost (a fixed-basis spectral derivative can be precomputed, so it is
  not impossible); it joins the C3-full appendix.

---

## Amendment A2 (2026-10-18) — answers to Codex design audit 2 (`audits/codex-design-2.md`, verdict WRONG)

A2 governs where it differs from A1 and the sections above. Item numbers are audit 2's.

**A2.1 (item 3) Q-DEIM checks with cancellation-safe tolerances.** $\|R_Bz\|=\|Bz\|$ holds algebraically (up to
roundoff). G-fd-c compares with a mixed tolerance: $|\,\|R_Bz\|-\|Bz_{\rm ass}\|\,|\le10^{-10}\,\Sigma$ and
$\|J^\top\hat r-g_{\rm ass}\|\le10^{-10}\,\Sigma\,\|J\|_F$, where $\Sigma = \|\Phi_r(\cdot)\|+\|\Delta t\nu\Delta_h\Phi_r(\cdot)\|+\|U_p(\cdot)\|$
is the sum of the assembled terms' norms (never divided by a vanishing residual or gradient). The cancellation fixture
is frozen: the state reached by 8 Gauss–Newton iterations on the assembled DEIM residual from the random start of the
generic fixture.

**A2.2 (item 4) Snapshot adequacy, both meshes, fail-closed.** In addition to A1.3's $256^2$ saturation check, an
**in-cell adequacy check** runs in every cell on the actual basis: 32 training trajectories (the first 32 of the
seeded permutation) are re-run saving the held-out steps $\{3,5,6,8,12,17,22,35,45\}$ (2D) / $\{3,5,6,8,12,17,22,35,45\}$
(3D, same indices at Δt = 0.005); adequate iff, at every registered rank, the RMS relative projection error of these
held-out states ≤ 1.10 × that of the in-set snapshot states of the same trajectories. If either check fails, the cell is
labelled "**basis possibly under-sampled**" and, fail-closed, any statement that ours is **not** dominated by the
baseline in that cell is withheld (observed dominations by the baseline are still reported).

**A2.3 (item 5) Trajectory generator: a recorded deviation, with a registered probe.** Decision (lane agent,
2026-10-18, autonomous per the campaign prompt; **to be ratified by the coordinator at phase-II release**): the primary
snapshots are operator-matched (§3.3, central FOM), because the bank's generator (sign-upwind BE) adds a numerical
viscosity of order $|u|h/2$ that, at $65^3$ ($h=1/64$, $|u|\lesssim2$), is comparable to ν itself, so a POD of those
fields would describe smoother dynamics than the operator being projected and would handicap the baseline. This departs
from the literal wording of the brief ("reuse the training-trajectory generator ... or its stored data"); only the
parameter draws are reused. **Generator probe (registered, cheap):** in each cell, the same 12-step snapshots are also
generated with the bank's own generator (sign-upwind BE, Δt = 0.005, `refcore.adv_upwind` arithmetic), a second POD is
built, and the `none` arm with the native solver runs on the screening subset at every rank and every option with each
basis; reported as projection errors and E0 errors side by side. If the coordinator rejects the deviation, the probe's
basis becomes the primary and the HR construction is rerun on it (cost in the inventory).

**A2.4 (item 6) Deferred options restrict the comparator set.** If any baseline option is deferred, every X using that
option is "not assessed (baseline option deferred)". t2core `CNR` (two full BE steps) and refcore `cnr` (four half BE
steps) are different schemes; this lane uses the t2core convention everywhere and never compares against refcore `cnr`.

**A2.5 (item 7) Task-level budget ledger.** Categories are assigned per **task** by in-job phase timers (device-synchronised
wall time per phase, written to the result JSON), not per job; `sacct` totals are reported beside the sum of phases
(the difference is overhead, reported). Ours' mapping, frozen: C-model = bank training jobs; C-hr = mesh table builds and
quadrature-rule selection/escalation (combined `c3desc*`, the rule-selection part of `c2grid` where its log separates
it, P1e); C-search = scheme/step/solver grids (combined `c2grid` remainder, `c2fbgrid`, `c3grid`; speed3d `g1–g3` grid
phase); C-cert = anchors and certification passes (where logged separately; otherwise inside C-search and stated). The
rule is **equal allocation**, not equal expenditure: the baseline's C-search allocation equals ours' C-search total per
dimension. Cases: (a) registered grid ≤ allocation → run it, then the bounded extension (A2.18) while allocation remains;
(b) registered grid > allocation → the grid still runs in full (it is registered) and every verdict line of that cell
says "**baseline search exceeded ours' allocation**"; (c) allocation cannot be met inside the 30 GPU-h cap → every pair
label in that cell is prefixed "**unequal-budget, descriptive**" and no H-base wording is used for it.

**A2.6 (item 8) Stage B membership, algebraic.** Screening front $F$ = configurations not dominated in (median time on
the 6 timing cases, worst error on the 6 screening cases) among Q1-qualified ones. A configuration $q$ is **near-front**
iff $\exists f\in F$: $W_q\le1.5W_f$ **and** $t_q\le1.5t_f$. Stage B = all of $F$ (never capped) ∪ up to 24 near-front
configurations, chosen alternately from the fastest and the most accurate remaining ones (deterministic, ties by the
unique ID of A2.17). Coverage of $F$ is therefore complete.

**A2.7 (item 9) Q1 predicate exactly as COMMON-BRIEF §3.** A step is non-stationary iff $\gamma>10^{-3}$ or $\gamma$ is
non-finite — **no residual-tolerance exception** (A1.7's exception is withdrawn). Solver termination (which may stop on
the residual tolerance) is separate from certification. Certification runs in the same job as the runs it certifies.
Anchor validity: every step of the anchor has $\gamma\le g_{\rm tol}$ of its level (1e-7 tight, 1e-8 tighter), finite,
no damping saturation; otherwise "anchor invalid" and the configuration is unqualified.

**A2.8 (item 10) ECSW training states and scaling, restated.**
- Per trajectory of option (scheme, Δt) with $N$ steps the state steps are the deduplicated set
  BE, CN: $\{1,\lceil0.15N\rceil,\lceil0.4N\rceil,N\}$; CN-R: $\{1,2,3,\lceil0.15N\rceil,\lceil0.4N\rceil,N\}$;
  BDF2: $\{1,2,\lceil0.15N\rceil,\lceil0.4N\rceil,N\}$. Each state carries its own coefficients (start-up BE or main),
  $u_{k-1}$ and $u_{k-2}$ (BDF2). The state count is whatever the deduplicated schedule gives; it is generated from the
  schedule and reported per option with each state's training FOM residual.
- Scaling floor $\phi=\max(10^{-8}\,\mathrm{median}_s\|b(s)\|,\ 10^{-14}\max_s\|b(s)\|,\ 10^{-300})$; blocks with
  $\|b(s)\|<\phi$ are dropped and recorded; if every block is dropped, ECSW construction fails for that rank (recorded).

**A2.9 (item 12) N-grad, two parts.** (a) **Binding certifier control:** take a certified-valid trajectory from the
smoke, replace $c_{n+1}$ at every 5th step by $c_{n+1}+\delta$, $\delta$ a fixed random direction of norm
$10^{-2}\|c_{n+1}\|$; the certifier must flag every perturbed step (γ > 1e-3, checked independently with a `jacfwd`
Jacobian) and none of the unperturbed ones that were stationary before. (b) **Wrong-gradient solver (reported):** the
fixture of A1.9 with the solver's Jacobian lacking its advection part, native solver, budget 30; its Q1 outcome is
reported; if it passes Q1 that is recorded as "mutation harmless on this fixture", not as a certifier failure.

**A2.10 (item 13) N-under is binding per HR family.** If no ladder level degrades a family (A1.9 predicate), that
family's control is **unfulfilled** and every comparison using that family is withheld from acceptance (reported as
"control unfulfilled") unless the coordinator amends the requirement.

**A2.11 (item 14) Control acceptance tests, corrected.**
- G-full uses a dedicated recovery level (gtol 1e-12, tolf 1e-14; LM budget 600, GN 30) and passes iff the lattice
  distance to the FOM (ntol 1e-13) ≤ 1e-8; it is a recovery test, not Q1.
- G-nnls KKT: $|g_i|\le10^{-9}\|G\|_F\|b\|$ on the support (two-sided) and $g_i\ge-10^{-9}\|G\|_F\|b\|$ off it.
- G-pod: numerical rank $\rho=\#\{\sigma_i>10^{-10}\sigma_1\}$; the reproduction test uses the POD at rank $\rho$
  (small-mesh test only); real-size G-pod = orthonormality plus the optimality check: Frobenius projection error of the
  snapshot set equals the POD tail $\sqrt{\sum_{i>r}\sigma_i^2/\sum_i\sigma_i^2}$ within 1 % at every rank.
- N-swap fixture, frozen: case swap = screening cases 0 ↔ 1; time shift = output index $k\to k+1$ (k = 1..4); fires iff
  the score changes by > 10 max ε_ref; kind control: S requested for 3D, or for a non-BE / Δt ≠ 0.005 arm, raises.
- N-dom synthetic data are seeded (seed 0) and their expected labels are computed by the frozen decision table
  (A2.17), for (a) dominance, (b) equivalence — $e_B=e_X(1+0.005\,\sigma_i)$, $\sigma_i=\pm1$ alternating, ρ values
  $1+0.01\,\sigma_{ij}$ alternating so the median is exactly 1 → "matches", (c) one bad case → not dominance.

**A2.12 (item 16) Worst-error equivalence with both envelopes.** With envelopes $[L_B,U_B]$, $[L_X,U_X]$:
equivalence iff $[L_B-U_X,\ U_B-L_X]\subset(-\delta_W,\delta_W)$, $\delta_W=\max(2\max_i\varepsilon_i,\ 0.1\,L_X)$; if
$\delta_W=0$ equivalence fails. Worst-case ties are irrelevant (envelopes use all cases).

**A2.13 (item 18) Bounded branches only.** The budget extension (A1.5) is bounded to one rung per direction with these
memory rules: 2D rank 512 (B at p ≤ 2048: 25.7 GB), Q-DEIM p/r = 8 only where $p\le2048$ (2D) / 4096 (3D); ECSW cap
retry 16,384 ($G^\top G_S$ buffer 32,768 × 32,832 f64 = 8.6 GB host). The live-memory and disk inventory is **generated
by `manifest.py` from the executable configuration list, for every allowed branch** (base, extension, retry, probe,
saturation), and printed into each job's certificate; the smoke measures the largest phases at full size (A2.19).

**A2.14 (item 19) Feasibility from the manifest.** The ≈ 13 GPU-h estimate of §12 is **withdrawn**. The projection is
generated from `manifest.py` (every run, anchor, certification pass, FOM run, NNLS path, pivoting, QR, compilation
shape) times the smoke's measured unit costs (`resources.py`). Q-DEIM pivoting runs = number of distinct $p$ in the
manifest (each $U_p^\top$ pivoted independently). Ours' comparator compilations (fs3's static schemes) are counted in the
verdict job's projection. If the registered core does not fit the cap, the cell is reported **incomplete**.

**A2.15 (item 20) The common-solver ablation is complete.** Every configuration runs with both solvers; the common-solver
runs are removed from the defer list (A1.14). Defer order now: exploratory A–B–A pairs → the generator probe at
$256^2$ → the 2D $256^2$ cell → the 3D Δt/Δt₀ = 2.5 options (A2.4 applies) → the 3D r = 512 rank.

**A2.16 (item 21) Central LMM FOM, registered and independently checked.** The FOM is `hrcore.make_fom` (central
operator, generic LMM with t2core coefficients and start-up). **G-lmm** (small mesh, CPU): for every scheme (BE, CN,
CNR, BDF2) the saved consecutive states satisfy the scheme's step equation evaluated with `refcore`'s operators
(independent arithmetic) to ≤ 1e-11 relative at every step, start-up steps with BE coefficients (CNR: steps 1–2; BDF2:
step 1); and a temporal self-convergence check gives observed order in [0.9, 1.1] (BE) and [1.8, 2.2] (CN, CNR, BDF2)
on Δt ∈ {0.01, 0.005, 0.0025} against Δt = 0.000625. BE and CN additionally equal `refcore.make_solver` (G-fom).

**A2.17 (item 17) Executable decision table (`stats.py`), fail-closed.**
- Unique configuration ID (string): `kind|r|hr-param|scheme|dt|solver|generator` (hr-param = τ or p or `-`); final
  tie-breaker everywhere.
- Pair outcome, evaluated in this order: invalid inputs (non-finite statistics, fewer than 3 cases with valid A–B–A
  triples after the drift policy, missing eligible cases, $\delta_i=0$ for any case) → "**not assessed**"; both objectives
  resolved in one direction → "baseline dominates" / "ours dominates" (annotated if A1.11 also holds); resolved in
  opposite directions → "neither dominates"; A1.11 (with A2.12) holds → "matches"; else "unresolved".
- A pick that fails replay or certification in the verdict job → role "none (pick failed)" — never support.
- **Wording.** Every pair label is "developmental, in-sample, unadjusted". A cell line reads "observed domination of X by
  a registered pick" or "by a non-registered Stage-B configuration" (both reported with equal prominence), or "no
  observed domination among the N configurations compared" — never "H-base holds" and never "falsified" as a formal
  test result.

**A2.18 Extension rule, precise.** After Stage A, if a front configuration has the largest registered rank, or the
smallest τ, or the largest p/r for its rank, and allocation remains (A2.5), the corresponding single rung (A2.13) is
added for that configuration's (HR kind, options), screened, and the front recomputed; no second extension.

**A2.19 (item 22) Timing population and drift.** The time objective for both role selection (B_cost) and the verdict is
measured on the same 6 timing cases (2D dev6, 3D 923801 cases 0–5). Drift policy: a triple with
$|t_{A,1}/t_{A,2}-1|>0.25$ is re-measured once; if it fails again it is discarded; a pair with fewer than 3 cases
retaining ≥ 1 valid triple is "not assessed (timing drift)". N-time's B makes **two distinct calls** (case $j$ then case
$j'$, each `block_until_ready`), so no computation can be elided. The smoke runs the largest phases at full size: the
2D $1024^2$ $B$-QR at (r, p) = (384, 1536), the ECSW contribution matrix at r = 384, and the 3D NNLS at r = 512, and
records peak device and host bytes.
