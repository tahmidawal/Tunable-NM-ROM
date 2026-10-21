# imp-multimesh (E7 / ideation W4, F14b): full multi-mesh cost against a fair continuous-POD control

Status: **pre-registration, 2026-10-21** (before any code or job). Amendments are appended at the end, dated; the
body is never rewritten.

## 1. Question

Our method uses **one** ordered coordinate-network bank and **one** off-mesh quadrature rule for every mesh, so its
offline work is done once and its evolution cost is flat in the mesh (jcp-p1-scaling H-cost-N). Standard POD +
ECSW must be rebuilt on each mesh. The paper's remaining cost claim is that this makes us cheaper *in total* when a
user needs several meshes. The fair control is a **continuous POD**: one POD basis built once on one mesh,
resampled to the other meshes by the exact sine series, with either an ECSW rule refitted on each mesh or the
source rule reused.

**Question.** On registered workloads (1–3 meshes from $\{256^2, 1024^2, 4096^2\}$ × $q \in \{10, 100, 1000\}$
queries per mesh), is the **full cost** (offline setup + all queries) of our method lower than that of the best
continuous-POD arm at equal or better worst error? Per-mesh POD + ECSW is reported beside it.

**Stop rule (verbatim from the brief, pre-registered):**
> If the continuous-POD control matches us (no more cost at equal or better worst error) on every workload,
> **drop the comparative multi-mesh cost claim** (mesh-independent evolution and mesh-free evaluation remain
> properties of the method, not comparative advantages).

## 2. Problem, cohorts, scoring (inherited, unchanged)

- Viscous Burgers 2D, $u_t + u(\partial_x u + \partial_y u) = \nu\Delta u$ on $(0,1)^2$, zero Dirichlet walls,
  $t\in[0,0.25]$, outputs at $t_k = 0.05k$, Gaussian initial condition $(c_x, c_y, w, a, \nu)$ (jcp-hr-baselines
  `hrcore.py`, `mr-burgers2d/engines.py`). Mesh $L$ = intervals per axis, $(L-1)^2$ interior unknowns.
- **Cohorts:** dev6 + val32 = 38 cases (development / validation; selection happens here; labelled
  *developmental, in-sample*). test64 is not opened. Sealed / OOD cohorts are never touched.
- **Reference:** E0 ST (`refstore.load(2, cohort, case, 'ST')`, store version pinned as in jcp-hr-baselines
  `score.py`), on the $257^2$ lattice; every mesh is restricted by stride $L/256$.
- **Error:** $e = \max_{k\ge1}\|F_k - R_k\|_2/\|R_0\|_2$ on the lattice (metric id
  `2|lattice=256|times=0:0.05:0.25|norm=fixed_t0|max_k>=1`). Eligibility per case-kind from the E0 store; ineligible
  cases are listed and excluded from worst/median. $\varepsilon_{\rm ref}$ (conditional Richardson estimate) from
  the store.
- Worst error of an arm on a workload $w$: $W(w) = \max_{L\in w}\max_c e(L,c)$; median: median of $e(L,c)$ pooled
  over $L\in w$ and eligible cases. Both are reported; the stop rule uses the worst.
- **Tie:** "equal or better worst error" means $W_{\rm ctrl}(w) \le W_X(w) + 2\varepsilon^{*}$, with
  $\varepsilon^{*}$ the largest $\varepsilon_{\rm ref}$ over the eligible cases (the 2 ε_ref rank rule; labelled
  conditional).

## 3. Arms

Time step $\Delta t_0 = 0.005$, 50 steps, everywhere. f64, `JAX_DEFAULT_MATMUL_PRECISION=highest`.

### 3.1 Ours (X)
The six jcp-p1-scaling settings, frozen mode policy (one hashed 2048-mode list taken at $1024^2$, reused at every
mesh, COMMON-BRIEF §4): $R'\in\{128, 256, 384\}$ × LSPG $\{$BE, CNR$\}$, rules R128 `fib4181`, R256/R384
`gauss128` (successor/check rules as registered in jcp-p1-scaling). Byte-identical `p1common` / `p1core` /
vendor code and the dn256b checkpoint + `rotation_R512.npz`.

- **Accuracy and Q1/Q2:** the jcp-p1-scaling accuracy jobs (5352982 at $256^2$, 5352981 at $1024^2$, 5353180 at
  $4096^2$) certified these exact configurations in-job (Q1 and Q2) on all 38 cases, **except** $4096^2$ R128 and
  R256, which ran only 16 cases. Job **J5** reruns `p1acc.py` byte-identically for those two settings at $4096^2$ on
  all 38 cases (in-job Q1 + Q2). The reused results carry the label "certified in-job by jcp-p1-scaling (same code,
  same configuration)". Replication gate: J5's per-case errors on the 16 shared cases agree with p1's
  `e0_per_case` to $\le 10^{-8}$ absolute; the timing job's own scored runs agree with the accuracy runs likewise.
- **Offline cost** (§5): bank data generation + bank fit; per-mesh decode-table build, sine/operator tables,
  setting setup and compilation.

### 3.2 Per-mesh POD + ECSW (PM)
The jcp-hr-baselines recipe, rebuilt on each mesh: FOM snapshots of the 576-trajectory training family (CNR,
$\Delta t = 0.005$), POD by the method of snapshots in f64 with its certificate, ECSW training states (24
trajectories), contributions on 32,768 candidates, greedy NNLS path. Snapshot sets: all-step (29,376 snapshots)
at $256^2$ (the registered remedy installed there by jcp-hr-baselines), 12-step (6,912) at $1024^2$ (as
jcp-hr-baselines). Configuration grid: $r\in\{128, 256, 384\}$ × support $s=\kappa r$, $\kappa\in\{4, 8\}$ ×
scheme CNR × solver `common` (LM). The NNLS path is run to $\kappa = 8$ only (the deployed targets).

At $4096^2$ the registered recipe is **not constructible** (the 12-step snapshot matrix is $6912\times 4095^2$
f64 ≈ 0.93 TB of host memory). PM at $4096^2$ is reported as a **cost lower bound for the registered
recipe**: 576 × the measured mean FOM trajectory time at $4096^2$ (measured in J4 on the 24 ECSW-state
trajectories, which run the same 50-step solver), with POD, ECSW and everything else counted as zero. Its accuracy
is not measured; PM rows on workloads containing $4096^2$ are "cost ≥ bound". (A practitioner could use fewer
training trajectories; the bound is for the recipe that was tuned at $256^2$ and $1024^2$.)

### 3.3 Continuous POD (CP) — the fair control
Source mesh $s\in\{256, 1024\}$: the PM basis and rules of that mesh (identical construction, built once).
**Resampling** of each mode to mesh $L$ by the exact sine series of the Dirichlet problem: DST-I of the
$(s-1)^2$ nodal values, zero-padding (prolongation, $L > s$) or truncation (restriction, $L < s$) of the
coefficients, inverse DST-I at $L$, scaled by $s/L$. Prolongation preserves Euclidean orthonormality exactly in
exact arithmetic (discrete sine orthogonality); restriction does not, so restricted bases are re-orthonormalised
in mode order (two Householder QR passes, sign-fixed). Every resampled basis must pass orthonormality
$\|\Phi^\top\Phi - I\|_{\max}\le 10^{-10}$.

Two rule variants:
- **CP-refit (CPR-s):** ECSW refitted on each target mesh with the PM recipe (training states from 24 FOM
  trajectories on the target mesh, contributions, NNLS path to $\kappa=8$), same $(r,\kappa)$ grid.
- **CP-reuse (CPU-s):** the source mesh's ECSW nodes and weights reused on each finer target mesh ($L > s$, nested
  meshes: node $(i,j)$ of mesh $s$ is node $(iL/s, jL/s)$ of mesh $L$), weights unchanged (LSPG is invariant to a
  common weight scale). On target meshes coarser than the source, reuse is undefined and CPU-s falls back to the
  refit there (counted as refit cost).

Trust radii for the LM solver: recomputed per target basis with the jcp-hr-baselines rule (`trust_radii`, from the
source snapshot codes, which are mesh-independent coefficients and are therefore reused).

Grid for every CP variant: as PM ($r\in\{128,256,384\}$, $\kappa\in\{4,8\}$, CNR, LM). At the source mesh,
CPR-s = CPU-s = PM-s.

Limitations, stated: the controls' grid omits Q-DEIM, BE, $r=512$ and $\kappa\in\{2,16\}$ (jcp-hr-baselines' best
accurate-end pick was a Q-DEIM $r=512$ configuration). Omitting configurations can only make the controls weaker,
i.e. it biases this lane **against** the stop rule firing.

### 3.4 Qualification of the competitor configurations
Every PM / CP configuration counted in a cost comparison must pass **Q1 in-job** (jcp-hr-baselines
`pipeline.q1_config`): every run valid; non-stationary steps ($\gamma > 10^{-3}$ with a fresh Jacobian at every
step, `rom.make_certifier`) ≤ 1 % pooled; distance to the tight solve ($g_{\rm tol} = 10^{-7}$) ≤ 2.5e-4 on every
case; tight/tighter ≤ 2.5e-5 on the cohort's first two cases. A **Q2 analogue** for ECSW (rollout distance between
$\kappa=4$ and $\kappa=8$ at the same $r$, and between each hyper-reduced configuration and its unreduced POD-LSPG
at $256^2$ and $1024^2$ where the unreduced arm fits in memory) is **reported, not gated**, as in jcp-hr-baselines
(gating it would only remove competitors). Configurations failing Q1 are excluded from the comparisons and listed.

## 4. Workloads (registered)

Mesh sets: all seven non-empty subsets of $\{256^2, 1024^2, 4096^2\}$; queries per mesh $q\in\{10,100,1000\}$.
**21 workloads.** A workload with mesh set $\mathcal M$ and $q$ means: $q$ queries on each mesh of $\mathcal M$
(the query parameters drawn from the training range; the per-query cost is the mean measured cost over the 38
cases).

## 5. Cost model (seconds on one H200; every item measured in this lane on an H200 unless stated)

$$C_{\rm arm}(w) = O_{\rm arm} + \sum_{L\in\mathcal M}\big(P_{\rm arm}(L) + q\,\bar t_{\rm arm}(L)\big)$$

- **Ours.** $O$ = bank training-data generation (576 FOM trajectories at $256^2$) + bank fit (300,000 steps). The
  bank fit was measured on an A100-PCIE-40GB (10,684.65 s, separable-decoder job 2835788). Re-measured on the H200
  by a **rate probe** (job J2): the byte-identical training script (manifest of job 2835788 reconstructed by
  sha256) run on an H200 until step 15,000; H200 total = A100 total × (H200 / A100 seconds per step over steps
  5,000 → 15,000), and data generation measured directly. Label: "projected from a measured rate". Validation
  (CPU, from the A100 log): the same window rate × 300,000 must reproduce the A100 recorded training seconds
  within 5 %. The dn256b head refit is **not** counted: the LSPG linear-span configurations use only the bank
  parameters (`g`, `B`), which are bitwise identical in the r3a and dn256b checkpoints (checked). Not measured
  anywhere and counted as 0 (favours ours): the bank rotation `rotation_R512.npz`, the selection of the frozen
  rules. $P(L)$ = decode-table build + sine/operator tables + setting setup + compilation of the fused query
  (cold first call − warm call) at mesh $L$ (measured in J6).
- **PM.** $O = 0$; $P(L)$ = snapshots + POD + POD certificate + ECSW states + contributions + NNLS path of the
  configuration's $r$ + arm data + compilation (measured in J0/J6).
- **CPR-s.** $O$ = snapshots + POD + certificate at $s$ (+ ECSW at $s$ if $s\in\mathcal M$); $P(L)$ for
  $L\ne s$ = resampling (+ QR) + ECSW states + contributions + NNLS path at $L$ + arm data + compilation.
- **CPU-s.** $O$ = snapshots + POD + certificate + ECSW states/contributions/NNLS path at $s$ (always);
  $P(L)$ for $L > s$ = resampling + arm data + compilation; for $L < s$ as CPR-s.
- Shared items are charged once per arm per workload (one basis, one ECSW path per distinct $r$ used).
- **Per-mesh configuration choice.** For a given target worst error, each competitor arm may choose its
  configuration independently on each mesh (cheapest qualified configuration whose worst error on that mesh is
  within the bar); the total is minimised over these choices by enumeration. Ours uses one configuration X on every
  mesh (its worst error is mesh-independent, H-acc-N).
- $\bar t(L)$ = mean warm full-query time (init + 50 steps + decode of the six full-field outputs on the mesh,
  `block_until_ready`) over the 38 cases (J6).
- **Accounting variants.** *Primary (A_dep):* the items above. *Sensitivity (A_sel):* adds the competitors'
  measured selection/validation run time (their dev6 tuning runs) and ours' 0 (not measured; favours ours);
  *sensitivity (A_100):* ours' bank fit at the A100-measured 10,684.65 s instead of the H200 projection. The stop
  rule is evaluated under A_dep; A_sel and A_100 are reported beside it, and any disagreement is stated.
- Memory (peak device bytes per arm per mesh) is reported, not costed.

## 6. Decision rule

For each workload $w$ and each of our six configurations $X$ (each must be Q1+Q2-qualified on every mesh of $w$):
the CP control **matches X on w** if some CP variant (CPR-256, CPR-1024, CPU-256, CPU-1024) has a choice of
qualified configurations with $W_{\rm CP}(w) \le W_X(w) + 2\varepsilon^{*}$ and $C_{\rm CP}(w) \le C_X(w)$.
The control **matches us on w** if it matches every X. **The stop rule fires** iff the control matches us on all
21 workloads. Reported per workload: for each X, the cheapest matching CP variant, the cost ratio
$C_{\rm CP}/C_X$, $W$ and median of both, and the same against PM (with PM at $4096^2$ as a lower bound: if the
bound already exceeds $C_X$, PM "loses regardless of accuracy"; otherwise "unresolved").

Also reported (descriptive): the break-even query count $q^{*}$ per mesh set and X (the $q$ above which the cheaper
arm changes), from the measured $O$, $P$, $\bar t$.

## 7. Must-fail controls (each must be shown to fire on real data in the smoke, S1)

1. **GPU preflight:** exit 42 when the backend is not `gpu` (tested with `JAX_PLATFORMS=cpu`).
2. **Resampling correctness:** prolong a $1024^2$ mode to $4096^2$ and restrict back: identity to $\le 10^{-12}$.
   *Must fail:* a mutated prolongation (coefficient index shifted by one) fails the orthonormality certificate
   ($> 10^{-10}$) and the round-trip check.
3. **Rule reuse mapping:** the correctly mapped reused rule's dev6 worst error is within a factor 2 of the refit.
   *Must fail:* the source flat node indices used unmapped on the fine mesh (wrong positions) give a dev6 worst
   error > 2 × the mapped rule's, or fail Q1.
4. **Q1 certification:** *must fail:* the same configuration with LM budget 1 (not converged) fails Q1
   (non-stationary steps > 1 %).
5. **Blocked contributions:** the blocked ECSW contribution builder (needed at $4096^2$, where the dense
   `ecsw_contrib` does not fit) equals jcp-hr-baselines `build.ecsw_contrib` at $1024^2$ (same candidates, $G$ and
   $b$ to $\le 10^{-10}$ relative). *Must fail:* a mutation dropping the advection term of $J$ breaks the equality.
6. **Scoring:** *must fail:* a field scored against the wrong case fails the E0 initial-condition check
   (`score.Refs`, $\le 10^{-12}$).
7. **Timing drift:** ours' A1/A2 median ratio per configuration within $[1/1.1, 1.1]$ (else one retry, then
   invalid). *Must fail:* an injected 20 ms host sleep inside the timed region of A2 trips it.
8. **Decision logic (real data, CPU):** with the smoke's measured numbers, inflating every CP worst error to
   $2W_X$ must make "matches" false everywhere; setting CP cost to 0 must make it true everywhere.
9. **Replication:** ours' J5 errors vs p1 `e0_per_case` agree (≤ 1e-8); *must fail:* comparing R128 against R256's
   p1 errors.

## 8. Jobs (H200 only, gpu partition, namespace `/cluster/tufts/paralab/tawal01/imp-multimesh/<job>/`)

| job | content | depends on | est. GPU-h |
|---|---|---|---|
| S1 smoke | must-fail controls at $1024^2$ (one $r$, dev6) + $4096^2$ memory pilot ($\Phi_{384}$ + blocked contributions + one query) | – | 0.4 |
| J0 sources | PM/CP sources: $256^2$ all-step and $1024^2$ 12-step snapshots, POD, certificate, ECSW states, NNLS paths $r\in\{128,256,384\}$ to $\kappa=8$; timed phases; saves bases + rules | S1 | 0.6 |
| J1 m256 | PM-256 accuracy + Q1; CPR-1024 at $256^2$ (restrict, refit) accuracy + Q1 | J0 | 0.4 |
| J2 probe | bank-fit rate probe (to step 15,000) | – | 0.15 |
| J3 m1024 | PM-1024 accuracy + Q1; CPR-256 / CPU-256 at $1024^2$ | J0 | 0.5 |
| J4 m4096 | CPR/CPU-256 and CPR/CPU-1024 at $4096^2$ (prolong, blocked refit), accuracy + Q1; FOM trajectory times | J0 | 1.3 |
| J5 ours4096 | `p1acc.py` R128, R256 at $4096^2$, 38 cases | – | 0.3 |
| J6 time | per mesh: A1 ours (bank, setup, compile, 38 cases) → B every qualified competitor configuration → A2 ours | J1, J3, J4 | 0.9 |

Total ≈ 4.6 GPU-h projected; **cap 8**; lane concurrent-job cap 2; campaign lock for every sbatch. A smoke-derived
cost certificate gates each production job (spent + projection ≤ cap). Defer order if the cap binds: (1) CPU-1024
at $4096^2$, (2) CPR-1024 at $4096^2$, (3) $r=384$ refits at $4096^2$, (4) A_sel accounting.

## 9. Deliverables
`DESIGN.md`, `audits/`, code (`mmcore.py`, `mmbuild.py`, `mmtime.py`, `cost.py`, `make_report.py`, cluster
scripts), `runs/GPU-LEDGER.json`, `report.md` (generated), `EXPORT.json`, PNG plots (cost vs $q$ per mesh set;
cost ratio map over workloads), `HEARTBEAT.json`.
