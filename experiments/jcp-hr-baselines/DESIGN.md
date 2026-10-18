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

---

## Amendment A3 (2026-10-18) — answers to Codex design audit 3 (`audits/codex-design-3.md`, verdict WRONG)

A3 governs where it differs from A1, A2 and the sections above. Item numbers are audit 3's.

**A3.1 (item 7) Validation-informed budget, both methods, symmetric.** Two categories only:
(T) **training-only construction** — work that uses no development/validation case (ours: bank training, per-mesh
tables; baseline: snapshots, POD, advection POD, Q-DEIM pivoting, ECSW training runs and NNLS); and (V) **all
validation-informed work** — every run on a development/validation case whose outcome selected or qualified a
configuration, certification included, on both sides (ours: combined `c2grid`, `c2fbgrid`, `c3desc*` rule selection
and escalation, `c3grid`, and speed3d `g1–g3`, plus P1d/P1e where they qualify an imported X; baseline: Stage A,
Stage B, anchors and certification, the generator probe's E0 runs, the extension). Whole-job GPU-hours are split by the
in-job phase timers; where ours' logs do not separate phases the whole job counts as V (stated). **Equal budget** is
declared iff baseline V ∈ [0.8, 1.25] × ours' V for that dimension; otherwise every pair label of that dimension carries
"**unequal-budget, descriptive**" (overrun or shortfall alike). T is reported for both, never omitted. Equal allocation is
never reported as equal expenditure.

**A3.2 (item 12) N-grad (a) on isolated tuples with an independent oracle.** For a certified-valid smoke trajectory and
every 5th step $n$: keep the history $(c_n, c_{n-1})$, replace the returned state by $c_{n+1}+\delta_n$ with
$\delta_n = t_n v_1$, $v_1$ the top right singular vector of $J(c_{n+1})$ and $t_n$ such that
$\|J\delta_n\| = 100\,\|\hat r(c_{n+1})\|$ (then $\gamma\approx\sigma_1/\|J\|_F\ge r^{-1/2}\gg10^{-3}$). The certifier
evaluates each tuple **in isolation** (no history propagation). Expected labels come from an independent oracle
(γ recomputed with a `jax.jacfwd` Jacobian): every perturbed tuple non-stationary, every unperturbed tuple unchanged.
Part (b) (wrong-gradient solver) stays reported-only.

**A3.3 (item 14) Control specifications.**
- G-pod: $\|S-S\Phi_r\Phi_r^\top\|_F/\|S\|_F$ equals $\sqrt{\sum_{i>r}\sigma_i^2/\sum_i\sigma_i^2}$ within 1 % at
  every constructible rank.
- N-swap: (i) reference fixture — case 0's E0 field scored against case 1's reference and with the time shift (Codex
  measured 1.12/0.13 in 2D and 0.68/0.17 in 3D, far above 10 max ε); (ii) ROM fixture — the smoke's first certified
  ECSW native BE run at the smoke rank on screening case 0; both must exceed 10 max ε_ref.
- N-dom: labels are hand-derived (not produced by the table): (a) halved errors and times → "baseline dominates"; (b)
  symmetric ±0.5 % errors, ρ with median exactly 1 → "matches"; (c) one bad case → not "baseline dominates";
  (d) identical arms with equal ε → "unresolved" (A3.4); (e) fewer than 6 timing cases → "not assessed".

**A3.4 (item 16) Identical-arm degeneracy is intended.** With equal worst errors and envelope width ε the difference
envelope is $[-2\varepsilon, 2\varepsilon]$, not strictly inside a margin $2\varepsilon$: the outcome is "unresolved"
by design (strict inclusion is frozen; tested by N-dom (d)).

**A3.5 (item 17) Cell accounting.** Exported per cell and X: counts of pairs attempted, assessed, baseline-dominates,
ours-dominates, neither, matches, unresolved, not assessed (each reason), deferred, control-blocked (A2.10), and picks
failed. The statement "no observed domination among the N configurations compared" is allowed only with N = assessed
pairs, printed beside the other counts and the incompleteness labels (budget, basis adequacy, ECSW coverage, rank
coverage, option deferral).

**A3.6 (items 18, 24) Rank grids, maximal shapes, smoke.**
- 3D ranks are now $\{64,128,256,512,768,1024\}$ (covering every rank of ours, W1024 included); the 3D rank extension
  is withdrawn. 2D ranks stay $\{32,64,128,256,384\}$ with the one-rung extension to 512.
- Q-DEIM $p/r\in\{1,2,4\}$; the $p/r=8$ extension is withdrawn. Maximal shapes: 2D $1024^2$ extension
  $(r,p)=(512,2048)$, $B$ 25.7 GB; 3D $(1024,4096)$, $B$ 12.3 GB; ECSW contribution matrices 2D 120 states × 512 ×
  32,768 = 16.1 GB, 3D 102 × 1024 × 32,768 = 27.4 GB; $G^\top G_S$ host buffer $32{,}768\times(2\cdot\mathrm{cap}+64)$
  (cap 16,384 on retry → 8.6 GB; the factor 2 is slack for columns of dropped nodes before compaction).
- `manifest.py` generates every run, factorisation and array of every allowed branch; the phase-I smoke runs the
  **maximal** shapes of each phase type (2D: $B$-QR at (512, 2048), ECSW $G$ at r = 512, the `none` arm at r = 512;
  3D: $B$-QR at (1024, 4096), ECSW $G$ and NNLS at r = 1024) on reduced snapshot sets and records peak device/host
  bytes. These are shape and memory pilots; their accuracy is not interpreted.

**A3.7 (item 19) Feasibility.** Stage-B budget bound before screening: $|F|\le40$ plus 24 near-front per cell; if the
measured $|F|$ exceeds 40 the certificate is regenerated after Stage A and the cell runs only if it fits, else it is
**incomplete**. The certificate includes both generators, the ratification fallback (a second HR construction),
saturation and adequacy checks, every control, warm-ups, certification, compilation and NNLS retries.

**A3.8 (item 21) Order fixture, frozen.** Case 0 of the screening cohort (2D dev6[0] at L = 16; 3D 923801[0] at L = 8)
with ν overridden to 0.1, T = 0.25, final state, discrete L2 norm, self-convergence estimator
$p=\log_2(\|u_{\Delta t}-u_{\Delta t/2}\|/\|u_{\Delta t/2}-u_{\Delta t/4}\|)$ for Δt ∈ {0.01, 0.005} (states at
Δt ∈ {0.01, 0.005, 0.0025, 0.00125}); pass ranges as A2.16. The step-equation residual check is separate.

**A3.9 (item 22) Timing: N-time and complete coverage.** N-time's B is the **same case twice** (two host calls, each
synchronised): expected ratio ∈ [1.8, 2.2]. Per pair and case, up to 3 re-measurements of a drifting triple; the pair is
assessed only if **all 6** timing cases keep ≥ 1 valid triple, else "not assessed (timing drift)". Case weighting: the
per-case statistic is the median of that case's valid ratios; the pair statistic is the median of the 6 per-case
medians; the bootstrap resamples cases. The same estimator (per-case median, then median over the 6 cases) is used for
the single-run screening times of Stage A/B and for role selection.

**A3.10 (item 23) Imported X need Q1 and Q2.** An X enters only with both its Q1 certificate and its own-state Q2
(quadrature) certificate from the owning lane's audited export; Q1-only X are diagnostic and not compared.

**A3.11 (item 25) ECSW coverage.** Attempted versus constructed (r, τ) are reported separately. If a registered
boundary construction (smallest τ at a rank, or the largest rank) fails after the cap retry, the cell's ECSW statements
carry "**ECSW coverage incomplete**" and no "adequately tuned ECSW" or exhaustive non-domination wording is used.

**A3.12 (item 26) Terminology and costs.** "$L$" always counts intervals: the lane's $L=256$ has $255^2$ unknowns;
the historical 2D bank's `N=256` counts **nodes** ($254^2$ unknowns), so the parameter draws are reused but not the
bank's mesh. Bank costs per bank from their provenance: 2D R512 bank fit 10,684.65 s (its training JSON); 3D W1024 job
total 1.85 H200-h, bank fit 1.14 h.

**A3.13 (item 27) Offline numerical failures, fail-closed.**
- Both PODs: numerical rank $\rho=\#\{\sigma_i>10^{-10}\sigma_1\}$ (state) / $\#\{\sigma_i>10^{-10}\sigma_1^{N}\}$
  (advection); a registered rank $r>\rho$ or $p>\rho_N$ is "**not constructible**" (reported; never padded or regularised).
- NNLS: when the passive-set Cholesky fails or its solve residual exceeds $10^{-8}\|g\|$, a minimum-norm least-squares
  solve is used; an index added and immediately dropped is banned from re-entry; the number of banned indices and the
  stop reason are reported per path (`nnls.py`).
- DEIM: the solve $T_{3E}=R_3(P^\top U_p)^{-1}$ must have backward error
  $\|T_{3E}(P^\top U_p)-R_3\|_F/(\|T_{3E}\|_F\|P^\top U_p\|_F)\le10^{-12}$, else that (r, p) is a construction failure
  (reported); the interpolation constant $\|(P^\top U_p)^{-1}\|_2$ is reported for every p.
- The Q-DEIM cancellation fixture of A2.1 is a cancellation **candidate**; its ratio $\|Bz\|/\Sigma$ is recorded.

---

## Amendment A4 (2026-10-18) — answers to Codex design audit 4 (`audits/codex-design-4.md`, verdict WRONG)

A4 governs where it differs from earlier text. Item numbers are audit 4's.

**A4.1 (item 12) N-grad (a): ladder plus independent oracle; binding test is agreement.** For each tested step $n$
(every 5th) of the frozen fixture run: perturbations $\delta = \pm t_k v_1$, $t_k = t_0 2^k$, $k=0..6$,
$t_0=\max(100\|\hat r(c_{n+1})\|/\sigma_1,\ 10^{-6}\|c_{n+1}\|,\ 10^{-12})$ ($v_1,\sigma_1$ the top right singular
pair of $J(c_{n+1})$; a step with $\sigma_1=0$ is skipped and counted). For every tuple the certifier's γ and the
oracle's γ (`jax.jacfwd` Jacobian, same isolated history) are recorded. **Binding:** on every tested tuple
$|\gamma_{\rm cert}-\gamma_{\rm or}|\le10^{-9}\gamma_{\rm or}+10^{-12}$ and both fall on the same side of $10^{-3}$, **and**
at least one tuple is oracle-discriminating ($\gamma_{\rm or}>10^{-3}$) and is flagged. If no tuple is
oracle-discriminating the result is "**fixture unavailable**" (reported; not a certifier defect).

**A4.2 (item 16) N-dom identical-arm fixtures, both regimes.** (d1) reference-limited: 38 cases, $e_X=e_B=5\times10^{-5}$,
$\varepsilon_i=5\times10^{-5}$, identical ρ = 1 → $L_X=0$, $\delta_W=10^{-4}$, difference envelope
$[-10^{-4},10^{-4}]$ not strictly inside → **"unresolved"**. (d2) relative-margin-limited: $e_X=e_B=10^{-2}$,
$\varepsilon_i=10^{-6}$, identical ρ = 1 → **"matches"**. A3.4's general statement is narrowed to (d1)'s regime. The
decision table is unchanged.

**A4.3 (item 7) Budget band wording.** A baseline/ours V ratio in [0.8, 1.25] is reported as "**within the registered
factor-1.25 band (ratio = x)**", not as "equal budget", unless the coordinator ratifies the band as meeting the brief's
equal-budget requirement (recorded at phase-II release with A2.3). Outside the band: "unequal-budget, descriptive". The
actual ratio is printed in every case.

**A4.4 (item 14) N-swap ROM fixture, frozen.** ID `ecsw|r_s|0.001|BE|Δt₀|native|central` with $r_s$ = 128 (2D) /
256 (3D), screening case 0 against case 1's reference and with time shift +1. Expected:
$|e_{\rm swap}-e_{\rm true}|>10\max\varepsilon$ and $|e_{\rm shift}-e_{\rm true}|>10\max\varepsilon$ ($\varepsilon$ over
the two cases). If that configuration does not exist or does not certify, the fixture is `none|r_s|-|BE|Δt₀|native|central`;
if neither exists: "fixture unavailable" (the reference fixture, A3.3 (i), remains binding).

**A4.5 (item 19) Projection is conditional.** Every pre-run projection is labelled conditional. Stage-B membership is
frozen after Stage A and certified by `resources.py` before any Stage-B run; the full 3D base grid is
6 × 7 × 12 × 2 = 1,008 configurations (6,048 Stage-A production runs plus anchors and certification). If a certificate
does not fit, the scope is reported incomplete; nothing is shrunk silently.

**A4.6 (item 21) Order check with a bounded refinement ladder.** The estimators of A3.8 are computed for the pairs
(0.01, 0.005) and (0.005, 0.0025); the check passes if the finest estimator lies in the band; if only the coarser one
fails it is recorded "pre-asymptotic at 0.01"; if the finest fails, a fifth level Δt = 0.000625 is added and the next
estimator decides; if that also fails the scheme is "order not confirmed" and the scheme's use is blocked until
investigated (the step-equation residual check is separate and always binding).

**A4.7 (item 27) Offline robustness, explicit.**
- POD fallback (both state and advection PODs): if the optimality check (A3.3 G-pod) fails at any registered rank, the
  basis is recomputed by a blocked randomised SVD of $S^\top$ (oversampling 64, two power iterations with
  re-orthonormalisation, column blocks streamed from the host) and the check rerun; still failing → that rank
  "not constructible". The numerical-rank cut-off uses the fallback's singular values when it was used.
- NNLS passive-set fallback solves the least-squares problem on the **original contribution columns** $G_S$ (device QR),
  not on the normal matrix. Paths that stop by reason `kkt` are checked against **all** candidates: the largest positive
  gradient over the whole pool (banned indices included) must be ≤ $10^{-9}\|G\|_F\|b\|$, else the path is
  "incomplete construction (KKT violated)". τ-stopped records need only $\xi\ge0$ and exact relative residual ≤ τ.

**A4.8 (item 28) One timing estimand.** Role selection uses the same estimand as the verdict: per case the paired ratio
$t_B/t_X$ (single warm runs in the tuning job, both timed on the same GPU in the same job), median over the 6 timing
cases. B_cost(X) requires that median ratio ≤ 1. The verdict re-measures it with A–B–A triples.

**A4.9 (item 29) G-full metric.** G-full compares on the small mesh's own nodes:
$\max_{k\ge1}\|F_k-F_k^{\rm FOM}\|/\|u_0\|$ (no lattice). Pass ≤ 1e-8.

**A4.10 (items 17, 22) Small fixes.** Assessed pairs = valid inputs receiving a substantive label (unresolved is a
reported subset); a configuration that is both picks is counted once. "Up to 3 re-measurements" is per original triple;
every rejected measurement and retry count is kept in the output.

---

## Amendment A5 (2026-10-18) — answers to Codex design audit 5 (`audits/codex-design-5.md`, NEEDS-RESTATEMENT, no WRONG)

**A5.1 (item 14) N-swap fallback semantics.** The ROM fixture (A4.4) and its fallback count only if their run is
**certified-valid** (A2.7 validity); if neither is, the ROM fixture is "fixture unavailable". Scoring acceptance then
rests on the reference fixture (A3.3 (i)) plus G-score; an unavailable ROM fixture is reported, and the scoring gate is
not called complete without the note.

**A5.2 (item 27) POD certification and NNLS sign convention.**
- **Certificate** for a basis $\Phi_r$: $e_r=\|S-S\Phi_r\Phi_r^\top\|_F/\|S\|_F$ against the Gram tail
  $\tau_r=\sqrt{(\|S\|_F^2-\sum_{i\le r}\lambda_i)/\|S\|_F^2}$ ($\lambda_i$ Gram eigenvalues): pass iff
  $|e_r-\tau_r|\le0.01\,\tau_r+10^{-6}$. The Gram eigenvalue sums are accurate to about $r\,\epsilon\,\lambda_1$, so the
  certificate is informative whenever $\tau_r\gg10^{-6}$; below that the rank is **saturated** (further modes change
  the relative projection error by less than $10^{-6}$, far below any E0 error) and is constructible if $e_r\le10^{-6}$.
- The blocked randomised SVD fallback (sketch $r_{\max}+64$, seed 0, two power iterations, QR re-orthonormalisation
  after every product) is triggered when the method-of-snapshots basis fails the certificate at any registered rank;
  a rank is "not constructible" only if **both** bases fail it. The numerical rank from Gram eigenvalues is reported
  only and never used to reject a rank (A3.13's cut-off rule is withdrawn).
- NNLS sign convention: $w=G^\top(b-G\xi)$ (the negative least-squares gradient). KKT at an optimum: $\xi\ge0$,
  $|w_i|\le10^{-9}\|G\|_F\|b\|$ on the support, $w_i\le10^{-9}\|G\|_F\|b\|$ off it (A2.11 written with
  $g=-w$ is the same condition).

**A5.3 (item 28) X-specific fronts join Stage B.** Stage B = $F$ (absolute-time front, A2.6) ∪ near-front additions ∪,
for every X, the front of Q1-qualified configurations in (median paired ratio to X over the 6 timing cases, worst
error) computed from the tuning job's single warm runs of X and of every configuration on the same GPU. Roles (A4.8) are
selected from the X-specific front. The absolute-time front is a screening convenience only.

**A5.4 (item 30) Smoke fixtures, frozen, and unavailable controls.**

| | 2D | 3D |
|---|---|---|
| mesh | L = 1024 | L = 64 |
| training subset (snapshots) | rows 0–191 of `sample_params(0, 576)` (2,304 snapshots) | rows 0–341 of `table(923701, 1536)` (4,104 snapshots) |
| maximal-shape pilots | POD r = 512; Q-DEIM (r, p) = (512, 2048); ECSW G at r = 512 (NNLS to support 1,024) | POD r = 1024; Q-DEIM (1024, 4096); ECSW G at r = 1024 (NNLS to support 1,024) |
| smoke rank / HR | r_s = 128; ECSW τ ∈ {1e-2, 1e-3} (runs at 1e-3); Q-DEIM p = 256; `none` | r_s = 256; τ as 2D; p = 512; `none` |
| generator / solvers | central; native and common | central; native and common |
| options | BE 0.005, CN-R 0.005 | BE 0.01, CN 0.01, BDF2 0.025 |
| grid cases | dev6 {0, 1} | 923801 {0, 1} |
| N-under / N-grad fixtures | dev6 {2, 3} (smallest ν first: N-grad uses 2) | 923801 {5, 0} (N-grad uses 5) |
| ECSW training | 24 trajectories (seeded permutation, A2.8) | same |
| adequacy / probe | 8 / 4 trajectories | 8 / 4 trajectories |
| ours' adapter | combined R384 gauss128 LSPG-CN-R Δt₀ | combined W256 gl24 FS-K1 BE Δt₀ |

A required control whose fixture is unavailable leaves its acceptance gate **unfulfilled**: it is neither a pass nor
evidence of an implementation defect, and phase I is not declared complete while a required gate is unfulfilled
(it is reported to the coordinator with the reason).

---

## Amendment A6 (2026-10-18) — design audit 6 (`audits/codex-design-6.md`, NEEDS-RESTATEMENT, no WRONG) and the design-level consequences of code audit 1 (`audits/codex-code-1.md`, NOT READY)

**A6.1 POD certificate, final form (audit 6 item 2; code audit item 13).** With the Gram eigenvalues $\lambda_i$ and the
exact $\|S\|_F^2$: $\tau_r^2=(\|S\|_F^2-\sum_{i\le r}\lambda_i)/\|S\|_F^2$. A negative radicand within $-10^{-12}$
(roundoff) is clipped to 0; a more negative one fails the certificate. If $\tau_r<10^{-6}$ (saturated) the rank passes
iff $e_r\le10^{-6}$; otherwise iff $|e_r-\tau_r|\le0.01\tau_r+10^{-6}$. Orthonormality ≤ 1e-10 and finiteness are
required. The wording "far below any E0 error" is withdrawn: saturation means "below the registered
training-projection certification tolerance", nothing more. **One basis per cell:** the method-of-snapshots basis
unless the randomised fallback certifies strictly more registered ranks; ranks that fail under the chosen basis are not
constructible (this replaces A5.2's "both bases fail" rule, which would mix two bases' prefixes). Construction is
enforced in code: a non-constructible rank, an uncertified p, or a DEIM solve with backward error > 1e-12 raises a
recorded construction failure; nothing is used silently.

**A6.2 Randomised fallback dimensions (audit 6 item 3).** Sketch $\ell=\min(q_{\max}+64,\,n_{\rm snapshots},\,
n_{\rm spatial})$ with $q_{\max}=r_{\max}$ (state POD) or $p_{\max}$ (advection POD); a requested rank or p larger than
the snapshot count is not constructible (no completion of the basis). A non-finite method-of-snapshots factor triggers
the fallback before use.

**A6.3 N-under ladder, restated (code audit item 9).** An under-sampled ECSW support below $r$ makes the LSPG problem
underdetermined, so its solution is not unique and it cannot be Q1-qualified; such levels show solver/qualification
failure, not "sampling-induced accuracy degradation". The ECSW ladder is therefore support caps
$\{r,\lceil r/2\rceil,\lceil r/4\rceil,\lceil r/8\rceil\}$ (r points is already far below the registered τ supports),
the Q-DEIM ladder $p\in\{\lfloor r/8\rfloor,\lfloor r/16\rfloor,\dots,1\}$ (each p certified by A6.1 on the advection
POD). Every level is run on both fixtures with its own tight and tighter anchors (complete Q1). A family's control
**fires** iff some level is Q1-qualified **and** its median E0 error on the fixtures ≥ 2 × the `none` arm's; Q1-failing
levels are reported as solver failures. If no level fires the control is unfulfilled (A2.10 applies).

**A6.4 Native solver for underdetermined Jacobians (code audit item 2).** The native Gauss–Newton step is the
minimum-norm least-squares solution when the reduced residual has fewer rows than unknowns (QR of $J^\top$), and the
Householder-QR solution otherwise; a non-finite step (rank deficiency) is exit reason 3.

**A6.5 Validity includes damping saturation (code audit item 5).** A run is invalid if, at any live step, the LM damping
is non-finite or at its limit $10^{14}$, independently of the exit reason.

**A6.6 Timing estimands, exact (code audit items 6, 7, 8).** For a pair: per timing case, $v_c$ = median of that case's
valid ratios; the statistic is $T=\mathrm{median}_c\,v_c$ over the six registered timing cases (numpy median: the
mean of the two middle values); its 95 % interval and the 90 % interval of $\log T$ come from resampling cases (10,000,
seed 0). A pair is assessed only if all six registered timing cases have ≥ 1 valid triple. Role selection uses the same
$T$ with single warm runs (A4.8, A5.3). N-time runs on all six timing cases and passes iff $T\in[1.8,2.2]$ and the
lower 95 % bound exceeds 1 ("X faster"). Every timed output is compared with its separately scored output (≤ 1e-12);
every attempt (including rejected triples) is kept; screening runs are preceded by a 0.1 s burn.

**A6.7 Acceptance is schema-driven (code audit item 3).** `smoke_accept.py` requires an explicit list of phases and
gates per cell and recomputes each predicate from numbers (certificates from projection errors, Q1 from per-run
evidence and from distances recomputed out of the saved fields, timing statistics from raw triples, G-score from saved
coefficients); a missing, skipped, errored or non-finite required item fails; an unavailable fixture is "unfulfilled".
The CPU gate file `checks/test_core.json` (all 13 gates) is a required input.

**A6.8 Metadata pin (code audit item 17).** Jobs run with `JCPREF_META` unset; `score.check_store` asserts the loader's
metadata directory is the lane's pinned `e0/store` and hashes the exact STATUS file the loader reads.

---

## Amendment A7 (2026-10-18) — design-level consequences of code audit 2 (`audits/codex-code-2.md`, NOT READY)

**A7.1 GN backtracking.** A1.15 stands: when no halving decreases $\|\hat r\|$ (or the step is non-finite) the native
solver exits with reason 3 and the run is invalid. (The interim implementation returned reason 2; corrected.)

**A7.2 NNLS fallback numerics.** The passive-set fallback (A4.7) solves the least-squares problem on the original
contribution columns with an SVD-based least-squares routine (`jnp.linalg.lstsq`, rcond 1e-13) rather than a QR; this
is at least as stable as the QR named in A4.7 and is the registered method.

**A7.3 Phase-I adapter check is not G-ours.** The smoke executes ours' adapters at real size (A5.4) and times them
A–B–A; it does not replay exported errors. G-ours (§8, A3.10) is a phase-II gate on the pinned exports and is not
claimed by phase I.

**A7.4 N-under evidence.** Every ladder level persists its per-step certification lists and its production, tight and
tighter lattice fields; acceptance recomputes Q1 from them. ECSW levels whose actual support is below r are forced to
"not qualified (underdetermined)" in-job and in acceptance.

**A7.5 G-fd-a fixture.** At real size: a random state $u=u_0+0.1\max|u_0|\,\xi$ ($\xi$ standard normal, seed 0);
stencil values at 4,096 sampled nodes and **all stencil-Jacobian entries of 64 sampled rows** (rows of refcore's
Jacobian by vector–Jacobian products) against the lane's stencil algebra; bars as A1.9.

**A7.6 N-grad classification.** In addition to γ agreement (A4.1), the production classifier (`rom.certify`) is run on
each perturbation variant's γ array (tested steps replaced by the perturbed tuples' γ); its non-stationary count must
equal the count from the oracle's γ. Part (b) reports complete Q1 on both fixtures.

**A7.7 Timing evidence.** Each timed call's output is compared with its separately scored output immediately after
the timer stops (every call of every attempt, both calls inside N-time's B); acceptance recomputes drift validity from
the raw times and requires all six registered cases (the six are hard-coded in `stats.TIMING_CASES`; a caller cannot
shorten them).

---

## Amendment A8 (2026-10-18) — solver tolerances and anchors, from the local CPU dry run (no verdict data)

Evidence (local CPU dry run of `smoke.py` at reduced size: 2D L = 256, r = 16, 24 training trajectories; a code-path
test, not a result): with ours' production stopping rule γ ≤ 1e-3, the native Gauss–Newton stopped after one iteration
per step and its rollouts were 2.4–3.3e-4 from the tight solve (Q1 bar 2.5e-4); the common LM stopped at the
extrapolated predictor on 40 % of steps (1.3–1.7e-4); and an LM (Cholesky normal equations) **tighter** anchor on an
ECSW problem stalled at γ ≈ 2e-8 > 1e-8 (round-off of the squared normal equations), making the anchor invalid.

**A8.1 Native production tolerance.** The native Gauss–Newton (QR, backtracking) stops at γ ≤ **1e-6** (residual
criterion unchanged). Ours' γ ≤ 1e-3 is a setting tuned for ours' LM with a quadratic-extrapolation predictor; the
native solver is entitled to its own convergence rule, and Gauss–Newton's quadratic convergence makes the extra
iteration cheap. The common solver keeps ours' settings exactly (γ ≤ 1e-3), as an ablation. Q1's non-stationarity
threshold (1e-3) is unchanged for both.

**A8.2 Anchors.** The tight and tighter anchors of every baseline configuration (both solvers) are the **native**
QR-based Gauss–Newton at the tight (1e-7) and tighter (1e-8) levels: an adaptive solve to tolerance whose round-off floor
is not squared. A1.7's "common LM solver at the tight level" is replaced. Distances and validity rules are unchanged.

**A8.3 N-under ECSW ladder.** A support equal to r makes the weighted residual square, so its minimiser is an exact root
and γ is round-off noise (every step non-stationary in the dry run); such a level can never be Q1-qualified. The ECSW
ladder is therefore overdetermined: support caps $\{4r,\ 2r,\ \lceil1.25r\rceil\}$ (still far below the registered τ
supports at real size). The Q-DEIM ladder is unchanged. The firing rule (A6.3) is unchanged.

---

## Amendment A9 (2026-10-18) — anchor round-off floor (CPU dry run evidence, no verdict data)

Evidence: after A8, on ECSW problems the QR-based native Gauss–Newton at the **tighter** level (γ ≤ 1e-8) stalled at
γ ≈ 2–5e-8 on 2–11 of 50 steps (no residual decrease possible at working precision: exit reason 3), while its tight vs
tighter distance was ≤ 1e-7. The ratio γ has a round-off floor near $\epsilon\,\|c\|/\|\hat r\|$, which is large when the
sampled residual is small (ECSW supports close to r).

**A9.1 Anchor validity.** An anchor step passes if γ ≤ gtol(level), **or** it is a round-off stall — the native solver
exhausted its backtracking (exit reason 3) — with γ ≤ 100 × gtol(level). Round-off stalls are counted and reported per
configuration. Everything else in A2.7 stands (finite trajectory, no damping saturation, every step accounted for), and
the binding converged-anchor check remains the tight/tighter agreement ≤ 2.5e-5 on the prescribed subset. This
changes only the lane's own anchor-validity rule (A1.7/A2.7), not the COMMON-BRIEF Q1 predicates for the arm's runs
(where exit reason 3 still makes a run invalid).

**A10.1 (2026-10-18, dry-run evidence) N-time fixture.** N-time times the `none` arm at the smoke rank (BE, native):
its calls are long enough for stable timing, whereas the ECSW calls in the dry run lasted a few milliseconds and gave
T = 1.79 on one CPU cell. Ours' A–B–A pairs with the ECSW fixture as before. The band [1.8, 2.2] is unchanged.

---

## Amendment A11 (2026-10-18) — code audit 3 (`audits/codex-code-3.md`, NOT READY) design-level items

**A11.1 Exit reason 5.** The native solver reports a **non-finite least-squares step** as exit reason 5 (always
invalid, never an anchor round-off stall); exit reason 3 is reserved for a finite step whose backtracking is exhausted
(invalid for production runs; an anchor round-off stall under A9.1 when γ ≤ 100 gtol). Unknown codes are invalid.

**A11.2 Evidence persistence.** Every grid configuration, every N-under level, the N-grad fixture and ours' case-0
field are written as separate array files when each block completes (append-only), together with the coefficient
trajectories of every production and anchor run. Acceptance validates per-step lists against the registered step
count of the option, checks trajectory finiteness from the saved trajectories, rescores every E0 number from saved
fields against freshly loaded pinned references (including N-under and the N-swap ROM fixture, whose selection is
re-derived), re-derives the N-grad oracle counts from the tuples, checks the exact repeat/retry inventory of every
timing case and every per-output discrepancy, and compares the saved $n_0$ with the reference's.

**A11.3 ECSW pilot workload frozen.** The maximal-shape ECSW pilot runs the NNLS to a support of 1,024 (`pilot_support`,
part of the frozen A5.4 inventory); acceptance requires that support with stop reason "cap", or an earlier certified KKT
stop, and positive measured device peaks for every maximal-shape pilot.

---

## Amendment A12 (2026-10-18) — code audit 4 (`audits/codex-code-4.md`, NOT READY)

**A12.1 Evidence completeness.** Acceptance additionally validates exact evidence shapes (trajectories of shape
(steps + 1, r), 1-D integer reason lists), that each saved output row equals the corresponding scheduled trajectory row,
the discrepancy cardinality of every timed call (N-time's B returns two outputs), that an unsuccessful timing repeat
exhausts its retries, that a non-finite run is classified invalid and carries no score (G-score counts decoded plus
invalid runs against the full inventory), the exact N-grad Cartesian inventory (every tested step × k ∈ 0..6 × sign ±,
skipped steps only with a recorded zero singular value), the complete Q1 evidence of N-grad part (b) (recomputed; any
outcome allowed), the raw KKT diagnostics of the ECSW pilot, the finiteness of the maximal `none` pilot's outputs, and
G-all's validity and distance from saved trajectories and fields. Any errored N-under level makes both families'
controls "evidence broken" (not fired). Every reduction of discrepancies propagates non-finite values.

**A12.2 Checkpointing.** The grid's JSON summary is written after every configuration and N-under after its `none`
level, alongside the append-only array chunks (A11.2).

**A12.3 Q-DEIM pivots, stated narrowly.** The implementation is greedy modified Gram–Schmidt with Businger–Golub
column pivoting (geqp3's pivot rule, not its Householder implementation); pivot-sequence equality with LAPACK is
established only on the registered small fixtures.

**A12.4 Forgery-only findings.** Per COMMON-BRIEF §7, findings that require deliberately malformed evidence that the
registered code cannot produce are "noted", not blocking; they were nevertheless fixed where cheap (A12.1).

**A12.5 (code audit 5) Residual-only autodiff.** Wherever a residual is differentiated or only evaluated (FD and
`jacfwd` checks, the N-grad oracle, the Gauss–Newton line search, the LM predictor), the arm's residual-only function
(`rom.make_res`) is used, so the analytic Jacobian is never formed under autodifferentiation (its tangent would be an
(r, n, r) array: 137 GB at 2D $1024^2$, r = 128). Invalid runs' saved outputs are compared with their trajectory rows
NaN-aware.

**A12.6 (code audit 6) Small-mesh G-pod and non-finite discrepancies.** The small-mesh G-pod reproduction test uses
the lane's `pod()` at rank $\rho_8=\#\{\sigma_i>10^{-8}\sigma_1\}$ (no cap) with tolerance 1e-6 relative: the
method of snapshots resolves singular values only above about $\sqrt{\epsilon}\,\sigma_1$, so A2.11's
$10^{-10}$/1e-8 reproduction is not attainable by the registered algorithm; the binding POD check is the production
certificate (A6.1), unchanged. Output discrepancies (timed vs scored) treat matching non-finite entries as agreement and
any mismatch of the non-finite pattern as non-finite, so an honestly invalid run reaches the invalid-run accounting
(A12.1) instead of an evidence error.

**A12.7 (code audit 7) Timing outputs must be finite.** Every timed call's output and its scored comparison field must
be entirely finite (else the discrepancy is non-finite and acceptance fails); the NaN-aware agreement of A12.6 applies
only to the grid's invalid-run accounting.

---

## Amendment A13 (2026-10-18) — design audit 7 (`audits/codex-design-7.md`, NEEDS-RESTATEMENT, no WRONG)

**A13.1 (A6.3 restated).** Rejecting ECSW ladder levels whose actual support is below r is a **registered
control-admissibility restriction** of this lane, not a consequence of the COMMON-BRIEF Q1 predicates (which test
stationarity, validity and distance, not uniqueness). The overdetermined ladder (A8.3) stands.

**A13.2 (A8 restated).** The justification "Gauss–Newton converges quadratically, so the extra iteration is cheap" is
withdrawn (for non-zero-residual LSPG, Gauss–Newton is in general only linearly convergent with a residual-dependent
rate). The evidence is empirical: in the CPU dry run the native solver used 1 iteration per step at γ ≤ 1e-3 and 2 at
γ ≤ 1e-6, with rollout distances to the anchor of 2.4–3.3e-4 and about 1e-7 respectively; the cost of the setting is
measured in the smoke and in production (iteration counts are reported). Likewise, the failure of the square ECSW
support is an **observed dry-run outcome** (round-off-level residuals made γ noisy), not a general property of square
problems. These settings are frozen for production and are not relaxed after any verdict data.

**A13.3 (A9 restated).** A9.1 is a **small-gradient backtracking-stall allowance**: an anchor step whose bounded line
search (eight halvings) is exhausted with a finite step is accepted when γ ≤ 100 gtol (effective limits 1e-5 tight,
1e-6 tighter). Round-off is the proposed explanation, not a proven one. Nominal steps (γ ≤ gtol) and allowance steps are
counted and reported separately for every anchor (`roundoff_stalls` counts the allowance steps). The COMMON-BRIEF Q1
bars are unchanged, including the tight/tighter agreement ≤ 2.5e-5.

**A13.4 (A11 storage scope).** The field-and-trajectory retention of A11.2 is a **phase-I smoke** requirement. For
phase II the storage contract is registered in the phase-II release amendment: evidence is verified on the cluster in
the job (and by an offline checker on the pulled per-configuration summaries), only the selection-relevant fields
(Stage-B candidates and picks) are pulled, every pulled artifact is checksummed, and the local footprint is bounded by a
manifest-derived peak replacing §12's obsolete "≤ 2 GB per cell" statement. No required evidence is dropped silently.

**A13.5 (job audit smk1-1) Smoke job mechanics.** The smoke driver runs under a hard `timeout` of the wall time minus
4 minutes, so the per-phase checkpointed evidence is always checksummed; caches (XDG, CUDA, JAX compilation) are
confined to the job directory; `smoke_accept.py` exits non-zero unless ALL passes, and "ALL-DONE" in the log means job
completion only. The staged config differs from the HEAD blob only by the added `commit` field (by design of
`stage.py`). Whether the full workload fits the hour is unproven; phases not reached are skipped and fail acceptance
(fail-closed), and the phase-I hand-back then reports exactly which controls were not demonstrated.

---

## Amendment A14 (2026-10-18) — findings of the phase-I cluster smoke `smk1` (job 5354834, H200)

**A14.1 Smoke p_max is a shape pilot only.** The reduced smoke snapshot sets (2D 2,304; 3D 4,104 snapshots) cannot be
expected to certify an advection basis at p_max = 2,048 / 4,096 (in smk1 the 2D state POD's numerical rank was 1,652).
The maximal Q-DEIM B-QR in the smoke is therefore a memory/shape pilot that may use an uncertified p_max (recorded),
and acceptance certifies the advection POD only at the p the smoke actually runs (p_s and the N-under ladder). In
production, A6.1 applies unchanged: a p not certified on the full snapshot set is not constructible.

**A14.2 Memory-lean POD fallback.** smk1's 2D advection POD failed its certificate at some registered p, the
randomised fallback then ran out of device memory (16.5 GiB allocation) with both candidate bases resident. Now only one
candidate basis is on the device at a time (the other is parked on the host), the sign normalisation works on column
chunks, and the method-of-snapshots divisor is floored at $10^{-14}\sigma_1$ so columns beyond the numerical rank stay
finite (and are rejected by the certificate, never used).

---

## Amendment A15 (2026-10-18) — smoke `smk1` results (job 5354834, H200, 0.88 GPU-h) and the changes they force

Evidence (`runs/smk1/archive/output/ACCEPTANCE.json`, recomputed offline): every real-size implementation check that
ran passed (G-fom ≤ 1e-16; G-pod orthonormality 7e-15; G-fd-a rows 5e-16 and Jacobian entries exact; G-fd-b FD
≤ 6e-9 and `jacfwd` ≤ 3e-15; ECSW rows 4e-15; Q-DEIM Σ-scaled norm/gradient ≤ 1.2e-15 including the cancellation
candidate; N-fd fired, 2e-3–8e-2; G-all; N-grad (a) in 2D; N-time T = 2.000 in both cells; ours' adapters executed and
timed). Failures and what they show:

1. **ECSW construction at the registered τ is ill-posed (design defect).** In 2D at r = 128 the greedy NNLS reached
   τ = 1e-2 with 32 points and τ = 1e-3 with 64 points — fewer than r — so the weighted LSPG problem was
   underdetermined: E0 errors 0.51–0.77 against 0.008–0.018 for `none` at the same rank, every step non-stationary.
   The N-under ladder on the fixtures shows support 4r near the `none` accuracy (median 0.037 vs 0.031) and 2r
   already failing Q1. **A15.1:** the ECSW knob becomes the support target $s=\kappa r$, κ ∈ {2, 4, 8} (extension 16),
   read off one greedy path per (mesh, rank); the τ reached at each target and the τ crossings are reported, not used as
   the knob. ECSW configuration IDs carry `k<κ>`. The smoke fixture is κ = 4.
2. **N-under ladder (A15.2).** ECSW caps {3r, 2.5r, 2r, 1.5r, 1.25r} (all overdetermined, all below the 4r fixture); the
   firing rule (a Q1-qualified level with median E0 error ≥ 2× `none`) is unchanged. In 3D both families fired in smk1
   (ECSW 4r: 0.198 vs 0.050; Q-DEIM p = 32: 0.277).
3. **NNLS banning (A15.3).** The 3D r = 256 path banned 20,989 of 32,768 candidates (every index dropped once by a
   Lawson–Hanson refit was banned), stopped at relative residual 0.078 with KKT violated and took 1,174 s, so no ECSW
   fixture existed in 3D and the 3D grid was skipped by the deadline. An index is now banned only after its third drop.
4. **Memory (A15.4).** The 3D maximal ECSW pilot failed because `G.T @ G[:, S]` materialised the transpose of the
   27 GB contribution matrix; the 2D advection-POD fallback ran out of memory (A14.2). Products are now formed as
   $(G_S^\top G)^\top$ and $b^\top G$, and $G$ is assembled on the host and transferred once.
5. **N-swap ROM fixture (A15.5).** The registered ECSW fixture was certified-valid but its error (0.51) swamped the
   time shift, so the shift control did not discriminate. The ROM fixture must now be **Q1-qualified** (in-job and
   recomputed); the `none` fallback is unchanged.
6. **Phase I is not complete.** The brief allows one cluster smoke in phase I; smk1 was it. A second smoke `smk2`
   (≤ 1 GPU-h, same inventory with A14/A15) is needed to show every control firing at real size, and is requested from
   the coordinator in the phase-I hand-back.

**A15.6 (code audit 9) Corrections and conventions.** Corrections to A15's evidence paragraph: the 3D state-POD
orthonormality deviation was 1.3e-14 (2D 7.1e-15); "every step non-stationary" held for the native ECSW runs (the
common-solver runs had 46–49 of 50 non-stationary); the 3D failing expression was `G.T @ b` (it and `G.T @ G[:, S]` are
now transpose-free); the executed G-fd subchecks passed but the complete G-fd gates failed in both cells because arms
were missing. Support targets are **first crossings**: the recorded state is the first on the path with at least κr
positive weights; its actual size is reported; one path per (mesh, rank) runs to the largest registered target, and an
extension target (16r) continues that same path without rebuilding earlier checkpoints. A14.2 restated: numerical rank
is diagnostic; a basis prefix is admitted only through the A6.1 certificate, which may admit a saturated prefix.
`collect.sh` now refuses non-terminal jobs, verifies the transfer against a freshly computed remote manifest of outputs
and logs, and deletes the cluster directory only after all of that succeeds. The randomised fallback releases its
sketch before the power iterations.

**A15.7 (code audit 10).** `collect.sh` decides terminality from a successful `sacct` query of the job allocation
(terminal states only); any failed or ambiguous query refuses collection. The drop-count ban is a cycle guard: the
regression test shows that dependent (duplicated) columns reach the NNLS optimum with no ban; a test that forces a ban is
not provided (noted), and every path reports its ban count.

---

## Amendment A16 (2026-10-18) — design audit 9 (`audits/codex-design-9.md`, NEEDS-RESTATEMENT, no WRONG)

**A16.1 Name and claims.** The ECSW arm is "**ECSW with support-targeted greedy NNLS**": the Petrov–Galerkin ECSW
gradient-matching NNLS of Grimberg et al., sampled at support targets $\kappa r$ instead of stopped at a training-residual
tolerance (the published stopping rule). For every configuration the actual support, the training relative residual at
the checkpoint, construction failures and Q1 are reported. No claim that this grid guarantees a strong or unbiased
baseline is made.

**A16.2 Migration of the extension and coverage rules.** A2.18: the ECSW boundary is κ = 8 and its single extension is
κ = 16 (replacing "smallest τ"). A3.11: ECSW coverage is reported per attempted/constructed (r, κ); a failed κ = 8 or
κ = 16 construction makes the cell's ECSW statements carry "ECSW coverage incomplete". The cap-retry rule of A1.8/A3.11
is withdrawn. **Continuation:** every greedy path is built through 16r from the start (checkpoints 2r, 4r, 8r, 16r; the
batch size depends only on the current support, so earlier checkpoints are identical to those of a shorter path except
for the final batch at the cap); the 16r checkpoint is used only when the extension rule triggers. Its construction
cost is in the resource certificate.

**A16.3 Banning consequence.** If a target is unreached and more than 1 % of the candidates were banned, the path is
rebuilt once with banning disabled (iteration-bounded); the path reaching more targets is used, both are reported, and a
target still unreached is a construction failure (A16.2 coverage label). smk2 exercises this on the 3D case that failed
in smk1; if the retry is not triggered or does not help, that is reported.

**A16.4 Accounting.** The development-case evidence used to choose the knob and the fixtures (smk1 and the CPU dry runs)
is validation-informed and counts in the baseline's V budget (A3.1). The discarded τ-knob evidence is retained in
`runs/smk1`. The amended grid (κ ∈ {2, 4, 8}; extension 16) is frozen before production.

**A16.5 (design audit 10, code audit 12) Retry rule frozen.** The ban-free retry runs **whenever more than 1 % of the
candidates were banned** on the initial path (regardless of target attainment). Both paths use the same iteration bound
(20,000 NNLS iterations). One whole path is selected by a training-only order: more reached targets; then the smaller sum
of $\log_{10}$ training residuals over the targets both paths reached; then the initial path. Selection happens at
construction, before any validation run, and both attempts (diagnostics, checkpoint supports and weights) are kept and
reported; acceptance recomputes the trigger and the selection and counts supports from the saved weights. If the
selected path still banned more than 1 % of the candidates (only possible when the retry was not better), that rank's
ECSW statements carry "**ECSW construction quality limited (banning)**", blocking "adequately tuned ECSW" and exhaustive
non-domination wording for that cell.

---

## Amendment A17 (2026-10-18) — NNLS algorithm, from the CPU dry run (training-only fixture, no verdict data)

On a training-only fixture (2D L = 256, r = 16, the ECSW contribution matrix 1,920 × 32,768) the path to 16r = 256
needed 6,444 iterations and banned 7,563 candidates; a reference single-index Lawson–Hanson in numpy needed about 300.
Two causes: (i) the normal-equation refit (Cholesky of $G_S^\top G_S$) loses the positivity of a newly entered index to
round-off, and (ii) batch additions (2 % of the support) add correlated indices that the refit then drops, so they
re-enter and cycle. **A17.1:** every passive-set least-squares problem is solved on the original columns by Householder
QR on the device, and the gradient is recomputed exactly from the residual each iteration. **A17.2:** classic
single-index entry (batch fraction 0): on the fixture this reaches 128 points in 147 iterations with no bans, the same
path as the reference. The iteration bound becomes 40,000 (≈ 1.15 × the largest target at 3D r = 1,024 plus margin),
replacing A16.5's 20,000 for both attempts; the ban-free retry rule is otherwise unchanged. The cost of one path is
about $m\sum_k k^2$ flops for the QR solves (3D smoke r = 256: about $10^{15}$); for production ranks this is large
(3D r = 1,024: about $10^{17}$), so phase II will need an updating QR (append/delete columns, $O(mk)$ per step); this is
recorded as a phase-II implementation item with its own code audit, and the smoke measures the cost at r = 256.

**A18 (2026-10-18) — design audit 12 and code audit 15 on A17.**
- **A18.1 Evidence.** Every NNLS attempt now returns, and the smoke saves, each checkpoint's residual vector, $\|b\|$
  and the per-candidate drop counters; acceptance recomputes checkpoint residuals and ban counts from them, and checks
  both KKT inequalities for any KKT-terminated pilot (in addition to the pilot's saved support, weights and residual).
- **A18.2 Rank-deficient passive sets.** The passive-set least squares uses Householder QR on the original columns; if
  the support has more columns than rows or $\min_i|R_{ii}|\le10^{-13}\max_i|R_{ii}|$, the minimum-norm SVD least
  squares (A7.2) is used. Column padding (for compilation) never exceeds the row count.
- **A18.3 Corrections to A17.** The iteration bound is 40,000 outer iterations (one entry attempt and its full
  Lawson–Hanson refit per iteration) for each attempt, about 2.44 × the largest support target (16 × 1,024); the "1.15 ×"
  statement was wrong. The dry-run comparison was at different endpoints: the old algorithm needed 6,444 iterations for
  256 points, the repaired one 147 iterations for 128 points with no bans, the same trajectory as a numpy reference at
  the checkpoints compared; the two mechanisms in A17 are diagnoses, consistent with but not isolated by these runs. The
  fixture record (scripts and logs) is archived in `runs/nnls_fixture/`. All four checkpoints (2r, 4r, 8r, 16r), both
  saved attempts, the A16.5 selection order and the coverage/banning labels stand; the batch-dependent caveat of A16.2
  is retired (single-index entry).
- **A18.4** Path diagnostics pair each support size with the residual of the same iterate.
- **A18.5 Feasibility.** Repeated full QR is cubic in the attained support; the smoke measures its cost at the smoke
  ranks only. Before phase II, an updating QR (append/delete with periodic refactorisation and gradient refresh, its
  tolerances frozen and its checkpoints compared with the full-QR path on the smoke fixtures) or a reduced production
  scope is registered, and the resource certificate is regenerated for the final algorithm including the 16r
  continuation and the retry. The 30 GPU-h cap and the incomplete-coverage labels stand.

**A18.6 (code audit 16, design audit 13).** A18.1's persistence applies to the **support checkpoints** (the registered
knob); tolerance crossings are reported as scalars only. The fixture archive `runs/nnls_fixture/` is completed by the
logs `t4.log`, `t7.log` and `G.sha256` (generated on the local CPU after A18; committed when present), and `t_old.py`
is pinned to the pre-A17 `nnls.py` (commit c57098e00) that produced `old_algorithm_t.log`. Until those logs are
committed, the repaired-path versus reference comparison of A18.3 is "incompletely archived".

---

## Amendment A19 (2026-10-18) — coordinator decisions (coordinator, 2026-10-18)

1. **smk2 released** (coordinator, 2026-10-18): one more phase-I cluster smoke, ≤ 1 GPU-h.
2. **A2.3 ratified** (coordinator, 2026-10-18): POD snapshots come from the central-difference FOM on the training draws;
   rationale: it gives the baseline its best data, which is conservative for our method. The registered generator probe
   (sign-upwind vs central) is reported in the report.
3. **A4.3 ratified** (coordinator, 2026-10-18) with the wording "within the registered factor-1.25 band"; "equal
   budget" is never used.
4. **Production preconditions** (coordinator, 2026-10-18): before any production job, (a) design audit round 14 must be
   clean, (b) the NNLS updating QR (A18.5) is implemented and code-audited, (c) the resource certificate is regenerated
   for the final algorithm.
5. **Phase II released** (coordinator, 2026-10-18): if smk2's acceptance is ALL = true and item 4 holds, the lane
   continues autonomously into phase II (job audits → production → results audit → `make_report.py` → report audit →
   final hand-back with `runs/EXPORT.json`). Caps: 2 concurrent jobs; 30 GPU-h total including the 0.878 spent;
   campaign cap 8.
6. **Comparators (ours)** (coordinator, 2026-10-18): 2D from `runs/FRONT-2D.json` of the jcp-qualify lane
   (`worktrees/2026-10-18-jcp-qualify`, used once committed and audited there; it carries Q1 + Q2 labels) and the 2D
   configurations certified in jcp-p1-scaling `runs/EXPORT.json`; 3D from jcp-speed3d `runs/EXPORT.json` (Q1
   certificates; **mesh-free contract — labelled so**; its reduced-rule arms are called qualified only with jcp-qualify's
   3D Q2). If FRONT-2D.json is not ready when needed, the P1-certified configurations are used first and the front
   members added later. This supersedes §8's "full-field contract only" for 3D: 3D pairs are labelled "ours under the
   mesh-free query contract".

---

## Amendment A20 (2026-10-18) — NNLS updating QR (A18.5, coordinator precondition A19.4b)

- **A20.1 Algorithm.** The greedy single-index Lawson–Hanson NNLS (A17) keeps a thin QR factorisation of the passive
  columns $G_S=QR$ in fixed-shape device buffers ($Q$: m × K, $R$: K × K, $Q^\top b$; K = the next power of two ≥ the
  path's cap). Entering column: classical Gram–Schmidt with one reorthogonalisation (CGS2). Removing column j: shift
  R's columns, restore triangularity with Givens rotations (applied to R, $Q^\top b$ and Q). Passive-set solve: back
  substitution. The gradient $w=G^\top(b-G_Sx)$ is recomputed exactly from the residual every iteration (it never uses
  the factorisation). Full Householder refactorisation every 512 appends; the orthogonality drift
  $\max|Q^\top Q-I|$ before each refactorisation is recorded and reported.
- **A20.2 Equivalence check (binding, CPU gate G-nnls).** On random problems with frequent refactorisation (every 16)
  the updating path reproduces the full-QR (A17) path: identical supports at every checkpoint and residuals equal to
  1e-10 relative; drift ≤ 1e-10. On the A17 training-only fixture (`runs/nnls_fixture`): identical 32/64/128-point
  checkpoints, 147 iterations both, 3.3 s versus 64.9 s on CPU.
- **A20.3 Rank policy.** An entering column whose orthogonal remainder is ≤ 1e-12 of its norm is numerically dependent
  on the support: it is skipped (not banned) until the support next loses a column; the count is reported. A18.2's SVD
  fallback remains in the full-QR reference path only.
- **A20.4 Cost.** Per iteration $O(mK)$ for the append and $O(mN)$ for the gradient (3D r = 1,024: about $3.5\times10^{9}$
  flops each), i.e. about $10^{14}$ flops for a full 16r path — no longer cubic in the support. The phase-II resource
  certificate uses GPU timings of this path measured in the first phase-II job's NNLS stage (the certificate for that
  job uses a conservative projection, stated in its job audit).

---

## Amendment A21 (2026-10-18) — design audit 14 (`audits/codex-design-14.md`, NEEDS-RESTATEMENT, items 4–6)

**A21.1 2D comparator admission (item 4).** A 2D configuration of ours is compared only if its time-stepping option is
one of the baseline's registered 2D options (BE or CN-R at Δt = 0.005). Front members with other schemes or steps
(e.g. BE/CN-R at 0.01 or 0.0125, GAL-CN) are listed as "**not assessed (no registered baseline option)**"; registering
further baseline options would need a new amendment with regenerated construction, controls and resources.
**Batch 1 (frozen now):** the jcp-p1-scaling `runs/EXPORT.json` (lane commit 5b782026b, file sha256 prefix
8e3b120855c702f2) configurations marked qualified at the baseline meshes: 256² {R128, R256, R384, R384nom} × {BE, CNR}
and 1024² {R128, R256, R384, R512} × {BE, CNR}, all at Δt₀. Their per-case errors are re-derived by replay (G-ours).

**A21.2 3D comparators and query contract (item 5).** The 3D comparison uses the **full-field mesh-path** contract of
speed3d's grid (65³ rollout; `contract.grid`), the same contract as the baselines' query; the mesh-free m1 contract is
reported only as "not assessed (different query contract)". A19's "mesh-free" wording is superseded by this mapping
(the coordinator's instruction was to label the contract; this lane's baseline is a full-field query). **Batch 1
(frozen now):** speed3d `runs/EXPORT.json` (lane commit 691465d88, sha256 prefix 209f3f49924161e2) deployed-rule
configurations `M2_512|gl32`, `W256|gl24`, `W512|lat32768`, `W768|lat32768`, `W1024|lat65536`, each `fs|BE|0.01|K1`,
all "eligible" (Q1, jobs 5301580/5301264/5301270), with the finalists' converged-rule δ_quad as their quadrature
evidence; they are labelled "Q1 + deployed-rule δ_quad". Reduced-rule arms (e.g. the H2 winners on gl24 for M2_512)
enter only in a later batch once jcp-qualify's 3D Q2 for that exact configuration is committed and audited. Timed inputs:
ours' query = `TB.project` of the full-field IC + stepping + decode (speed3d `p1` component definitions; table
preparation excluded and reported separately as setup), the baseline's = $\Phi^\top u_0$ + stepping + full-field decode.

**A21.3 Versioned comparator batches (item 6).** Each batch is a dated amendment listing the exact X identities, source
commits and file hashes. For a new batch: (i) X's same-job paired ratios against every screened baseline configuration
are measured in a new tuning-job timing pass, (ii) X's specific front is recomputed and any new Stage-B members are run
with full Q1, (iii) its picks are frozen (sha256) before its verdict job, (iv) the added cost is certified and charged
to the baseline's V ledger; ours' V ledger adds the qualification/mesh-free work that selected or certified the
imported X, counting inherited evidence once.

---

## Amendment A22 (2026-10-18) — design audit 15 (WRONG) and code audit 17 (NOT READY)

**A22.0 3D quadrature evidence (design audit 15 item 2).** A21.2's attribution of the finalists' converged-rule δ_quad
to the five deployed-rule configurations is **withdrawn** (that evidence belongs to other configurations, and δ_quad
alone is not the own-state Q2 of COMMON-BRIEF §3). The five 3D batch-1 configurations are **diagnostic (Q1 only)**:
their pairs are run and reported with that label but take no H-base label. A 3D configuration enters H-base only in a
later batch with an exact-configuration own-state Q2 certificate (jcp-qualify's 3D Q2, both ρ and rollout-distance
bars), frozen with its rule, solver, initialisation path, certificate, population and source hash.

**A22.1 NNLS refactorisation cost (design 15 item 4, code 17 item 1).** The refactorisation factors only the active
columns at a bucket width (the next multiple of 512 ≥ k) and writes the factors back into the fixed buffers; its
interval is 1,024 appends. (Cost model: see A24.2, which supersedes the figures first written here.)

**A22.2 Memory (code 17 item 2).** Append, reset and delete donate the Q, R and $Q^\top b$ buffers (updated in place);
a rejected append restores its slot. The refactor builds bucket-width factors while the old buffers are live (peak
≈ G + Q + R + the bucket factors: 3D r = 1,024 ≈ 27.4 + 13.7 + 2.1 + 13.7 + 2.1 GB ≈ 59 GB plus QR workspace).
The maximal-shape GPU peak is measured in the first phase-II job before any 3D r = 1,024 path runs; production admission
requires it below 120 GB (decimal).

**A22.3 Stability gate (design 15 item 5).** Binding CPU gate G-nnls now includes the production interval: a path with
refactorisation every 1,024 appends, crossing two refactorisations with deletions, must reproduce a path refactored
every 256 appends (identical supports at every checkpoint, residuals to 1e-10) with drift ≤ 1e-8. In production, a
drift above 1e-8 at a refactorisation is a recorded **stability event**; any event makes that path's construction
"quality limited (orthogonality)" with the A16.5 consequences. Dependent-column skips, drift values, drift events,
interval and buffer capacity are persisted per attempt (`build.ecsw_weights`).

**A22.4** The buffer capacity is clamped to the row count (rectangular small problems).


---

## Amendment A23 (2026-10-18) — smk2 result and the "optimum before target" rule

smk2 (job 5422908, H200, 38:02) passed every acceptance gate under the staged checker except
`2d_L1024|ecsw-smoke-rank`: the 2D r = 128 NNLS path terminated at a KKT-certified NNLS optimum (reason `kkt`, 1,677
iterations, 13 bans, support below 16r = 2,048) after reaching 2r, 4r and 8r (training residuals 1.7e-4, 6.7e-5,
2.5e-6), so the 16r checkpoint did not exist and the staged checker (which required all four) failed. This is not a
construction defect: the training problem was solved exactly with fewer points than the target.
**Rule (A23):** when a path ends at a KKT-certified optimum (both KKT inequalities, A5.2) before reaching a support
target, that target's configuration **is the optimum** (actual support and residual reported, flagged `optimum`), not a
construction failure; for runs made before this rule (smk2) the absent checkpoint is accepted under the same condition.
Acceptance of smk2 is recomputed with the amended checker (`runs/smk2/ACCEPTANCE-A23.json`); the staged result
(ALL = false, this gate only) is kept unchanged beside it. smk2 ran the A17 full-QR NNLS; the production algorithm is the
A20/A22 updating QR, equal to it by the binding gate G-nnls.


---

## Amendment A24 (2026-10-18) — code audit 18 / design audit 16, cost model, phase-II job structure

**A24.1 A23 wording and evidence.** "Solved exactly" is withdrawn: a path that stops early **satisfies the registered
numerical KKT certificate** (the solver's KKT flag, both gradient inequalities with tolerance $10^{-9}\|G\|_F\|b\|$,
and — from A24 on — finite non-negative terminal weights whose count equals the reported support and whose saved
residual vector reproduces the reported residual). The full-QR and updating-QR paths **agree on the binding regression
fixtures** (not universally). An optimum used as a target remains subject to Q1, support disclosure and the
construction-quality labels. The terminal support, weights and residual vector of every attempt are saved; smk2 predates
this and is accepted under the legacy rule with that provenance stated.

**A24.2 Cost model (corrected).** One NNLS path to support s with m rows, N candidates, buffer width K: appends
$8mK$ flops each ($8mKs$ in all; memory traffic $32mKs$ bytes), gradient refreshes $2mN$ flops per iteration,
refactorisation at k columns ≈ $2mk^2$ (Householder) $+ 2mk^2$ (explicit Q) $+ 2mk^2$ (drift product) flops, every
1,024 appends. For 3D r = 1,024 (m = 104,448, N = 32,768, s = K = 16,384): appends ≈ 2.2e14 flops / 9.0e14 bytes,
gradients ≈ 1.2e14 flops (≈ 1.9e4 iterations) and 4.2e15 bytes (reading G each time), refactorisations
$\sum_{j\le16}6m(1024j)^2$ ≈ 9.8e14 flops; total ≈ 1.3e15 flops and ≈ 5e15 bytes (memory-bound: ≈ 20 min on an H200);
the retry rule can double it. Refactorisation still grows cubically in the attained support (A20.4's "no longer cubic"
is withdrawn); at these sizes it is not dominant.

**A24.3 Construction-quality labels (code 18 item 2).** Each rank's ECSW construction carries a quality label:
"ok", "limited (orthogonality)" (any non-finite or > 1e-8 drift at a refactorisation), "limited (banning)" (> 1 % of the
candidates banned on the selected path), or both; the label is recomputed offline from the saved drift values and ban
counters and propagates to every ECSW configuration of that rank (A16.5 consequences).

**A24.4 Phase-II jobs.** Per cell (2D 256², 2D 1024², 3D 65³) a **tuning job** (`tune.py`): offline construction from
training data, saturation (256² only), adequacy, generator probe (basis + `none`/native screening runs), Stage A with
in-job certification and anchors, ours' batch-1 replay and same-job warm timing on the 6 timing cases, and the
full-rank limiting comparator on the verdict cohort. Its artifacts (Φ, per-configuration HR data, ECSW paths) stay in
the job directory for the next job. After offline Stage-B freezing (A2.6, A5.3) and a regenerated certificate, a
**Stage-B job** per cell runs the frozen Stage-B members on the full verdict cohort with Q1, and the A–B–A timing of
every (X, B) pair admitted by A1.12 on the 6 timing cases; picks and labels are then computed offline from that job's
data (A1.6) and frozen with sha256 before the report. Memory admission (A22.2): the 3D tuning job records the device
peak before and after the r = 1,024 ECSW path (JAX reports the process-cumulative peak, so the path's own peak is
known only when it sets a new one); a new peak > 120 GB (decimal) makes that rank "not constructible (memory)".

**A24.5 G-ours inputs.** 2D: the per-case E0 of each batch-1 key from jcp-p1-scaling `checks/analysis.json`
(`inputs/p1_analysis.json`, keys `dev6|0..5`). 3D: the per-case E0 of each batch-1 key on the selection cohort
(seed 923801, cases 0–5) extracted from speed3d `checks/scores-g{1,2,3}.json` at commit 691465d88 into
`inputs/speed3d_batch1_e0.json` (source sha256 recorded). Ours' 3D settings (bank, R', adaptive start a) are speed3d
`configs/r1.json`'s: M2_512 = (M2, 512, 3), W256/W512/W768/W1024 = (W1024, R', 6). G-ours is evaluated in the tuning
job (screening cases = timing cases) and again in the Stage-B job; a failure in either removes that X (reported).

---

## Amendment A25 (2026-10-18) — extension in the tuning job; C-search allocations pinned

**A25.1 Where the A2.18 / A16.2 extension runs.** The extension is decided and screened inside the tuning job, right
after the registered Stage-A grid, by the frozen rule: compute the screening front (Q1-qualified, finite, non-dominated
in median warm time and worst error on the screening cases); for each front member with κ = 8 add κ = 16 at its rank;
in 2D, for each front member at r = 384 add r = 512 with its HR kind and knob (κ, or p/r) — both options, both solvers,
same Q1, anchors, warm timing and scoring as Stage A (phase tag `EXT`). It runs only if the job's elapsed C-search
time (from the start of Stage A) is below the cell's allocation (A25.2); one rung per direction, no second extension.
The basis is built to r = 512 in 2D so that the rung needs no new snapshots; the r = 512 basis is certified like every
rank and its ECSW path is built only if triggered. Extension configurations enter the front, Stage B and picks exactly
like registered ones and are marked `stage = EXT` in every table.

**A25.2 Allocations (A2.5), pinned from `sacct` (ElapsedRaw × 1 GPU).** 2D: combined `c2grid` (job 5209526, 13,596 s)
+ `c2fbgrid` (5228777, 658 s) = 3.960 GPU-h, split evenly over the two 2D cells (1.98 h each; the rule-selection part
of `c2grid` is not separable in its log, so the allocation is an upper bound). 3D: combined `c3grid` (5219449,
15,949 s) + speed3d `g1`–`g3` (1.747 + 2.409 + 3.629 GPU-h, whole jobs, an upper bound on their grid phase) =
12.216 GPU-h. Ours' C-hr jobs (`c3desc*`, `c3extW1024`) are not in the allocation (A2.5 mapping). The baseline's
C-search time is the tuning job's Stage-A + extension phase time plus the Stage-B job's selection-relevant runs; each
verdict line states it beside the allocation (A2.5 cases a–c).

**A25.3 Resource certificate (A19.4c).** `resources.py` → `checks/RESOURCES-tune.json`, from smk2's measured phase
times, N-time per-run times and grid-block times, with stated scaling models (`none` ∝ r² above the smoke rank, NNLS
from A24.2 at an effective 2 TB/s with the retry doubling, mean step-count factor over the options) and a 1.5
contingency: tuning 2D 256² 0.94, 2D 1024² 3.15, 3D 65³ 9.81 GPU-h certified; with a Stage-B projection of half the
tuning total the lane projects 22.4 of 30 GPU-h (1.512 spent). Wall-time requests: 2 h, 5 h and 15 h. A job is
submitted only if spent + its certified projection + the remaining projections ≤ 30; the certificate is regenerated
from each completed tuning job before the Stage-B jobs are submitted.

---

## Amendment A26 (2026-10-18) — design audit 17 (WRONG) and code audit 19 (NOT READY)

**A26.1 NNLS cost and memory admission (design 17 items 2, 4; code 19 items 4, 14).** A24.2's gradient traffic is
corrected: reading $G$ ($8mN$ bytes) on each of $I$ iterations is $8mNI \approx 5.2\times10^{14}$ bytes for the
**assumed** $I = 19{,}000$ (the bound is 40,000 per attempt), not $4.2\times10^{15}$; appends plus gradient reads total
≈ $1.42\times10^{15}$ bytes (≈ 5 min at the H200's 4.8 TB/s peak; the projection uses an effective 2 TB/s, ≈ 12 min,
doubled for the A16.5 retry). Refactorisations are ≈ 73 % of the listed flops; deletions, residual formation, triangular
solves, compilation and refactor temporaries are not in the arithmetic and are measured in-job (phase timers). Nothing
in this arithmetic is a certificate (A26.4). Memory admission (replaces A24.4's rule): after the 3D r = 1,024 path the
**process-cumulative** device peak must be finite and ≤ 120 GB (decimal); a higher or missing value — whatever phase
caused it — makes that rank's ECSW "not constructible (memory)"; an out-of-memory error during the path does the same
(the phase fails, its error is recorded). The path is built after basis, advection POD and the probe basis, before any
Q-DEIM construction.

**A26.2 Evidence integrity (code 19 items 2, 3, 10, 11, 12).** (a) Offline optimum acceptance requires the saved
terminal solution (unique in-mesh integer nodes, finite strictly positive weights, count = final support, residual
vector reproducing the final residual) and that every optimum checkpoint equals it (alias); terminal arrays may be
absent only for the pinned legacy attempt smk2 (`smoke_accept.LEGACY_NO_TERMINAL`, by attempt name). (b) The ECSW
construction-quality label is stored on every ECSW configuration and is required for post-A24 attempts. (c) The tuning
job predeclares the registered inventory (every rank × HR knob × option × solver) with statuses not_run /
construction_failed / run_failed / complete / deferred, checkpoints each configuration and each comparator case as it
completes, and isolates failures per solver, per comparator and per FOM case. (d) A configuration whose warm timed
output differs from its scored output (non-finite or > 1e-12) is `timing_ok = false` and never enters an absolute-time
or X-specific front; ours' timed output is compared with its preceding (scored) call the same way. (e) Every full-rank
limiting-comparator solve is checked for convergence (finite residuals ≤ the Newton tolerance on every step); a failed
solve is a failed entry, and the full E0 record is kept.

**A26.3 Registered consequences implemented (code 19 items 6, 7).** Saturation (A1.3) runs before adequacy; if it fails
at any registered rank, the all-step POD (certified on all states) replaces the 256² basis (with its trust radii and
constructible set) before any downstream construction, and the 1024² cell is labelled "snapshot-limited". Adequacy
(A2.2) uses the first 32 trajectories of the seeded permutation (seed 20261017) on the basis actually used. Generator
probe runs are made only at ranks the probe basis certifies; other ranks are "construction_failed"; per-case E0 records
and output coefficients are kept. Order of the tuning job: comparators (ours, G-ours) → construction → the HR grids
(never deferred) → `none` at r < 256 → full-rank limiting comparator → deferrable work in the registered defer order
(`none` at r ≥ 256, then the extension, then the probe runs), started only while elapsed < 0.85 × the job budget;
deferred items are listed as such.

**A26.4 Resource certificate restated (code 19 item 15; design 17 item 6).** The per-item projections of A25.3 are
conditional models, not a certificate, and are relabelled "projected". The certificate is the hard bound: a 1-GPU job
cannot spend more than its Slurm wall limit. Wall limits: tuning 2D 256² 2 h, 2D 1024² 5 h, 3D 65³ 12 h (budget for
deferrable work = wall − 0.25 h); Stage-B reserve 7.5 h; spent 1.512 → bound 28.0 ≤ 30 GPU-h. If a tuning job is
truncated by its limit, the unfinished configurations remain `not_run` and the cell's coverage statement says so. The
Stage-B jobs' limits are set from the frozen Stage-B inventory and the tuning job's measured per-configuration times,
inside the reserve (re-certified before submission).

**A26.5 Stage B and picks (design 17 item 4).** A24.4 supersedes §6/A21.3's "picks frozen before the verdict job":
Stage-B **membership** is frozen (sha256, audited) before the Stage-B job; picks are computed afterwards from that job's
full-cohort data (A1.6) and frozen before the report. To cover every possible role candidate, the Stage-B job A–B–A
times **every** Stage-B member that is Q1-qualified on the verdict cohort against **every** admitted X (G-ours passed in
both jobs), not only A1.12's exploratory envelope-filtered set.

**A26.6 Staging (code 19 item 8).** The M2 bank (`vendor/quad3d/inputs/model_M2/bank.pkl`) is staged; `stage.py`
runs `ours.check_assets(cfg)` in the staged tree for the tune driver (every bank, rule, checkpoint and exported error
the comparators load lazily, with pinned hashes).

**A26.7 A19 wording.** A19.4's "design audit round 14 clean" refers to the accepted successor of round 14: the first
later design audit returning CLEAN (or NEEDS-RESTATEMENT with no WRONG item and every restatement applied).
