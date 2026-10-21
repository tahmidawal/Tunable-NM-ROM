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

---

## Amendment A1 (2026-10-21, before any code ran; answers `audits/codex-design-1.md`)

A1 governs wherever it conflicts with sections 1–9.

**A1.1 Decision rule (audit item 1).**
- *Joint selection.* For each workload and each competitor arm, the configuration choice is made **jointly** over
  the meshes of the workload: every combination of per-mesh configurations (≤ 6 per mesh) is enumerated, and its
  cost charges every shared artefact (source basis, source ECSW path of each distinct $r$, ECSW states of a mesh)
  exactly once (§A1.2 ledger). The cheapest admissible combination is used.
- *Tie (casewise).* "Equal or better worst error" is decided on the case–mesh pair where the competitor attains its
  worst error, $(L^\dagger, c^\dagger)$: admissible iff $W_{\rm CP} \le W_X + 2\varepsilon_{\rm ref}(c^\dagger)$
  (the binding 2 ε_ref rank rule applied to the case that defines the competitor's worst; ε_ref labelled
  conditional). The strict variant $W_{\rm CP}\le W_X$ is reported beside it; any verdict that differs between the
  two is stated.
- *Median.* The stop rule uses the worst error (as worded). A second, reported verdict requires the competitor to
  match on worst **and** median (pooled median ≤ ours' pooled median under the same tie rule). Per-mesh worst and
  median are reported for every arm.
- *Accuracy population.* All 38 cases on every mesh. Reference-ineligible cases (E0 store) are listed and excluded
  from worst/median for **both** sides; a solver failure (invalid run) is never excluded: a configuration with an
  invalid run is not Q1-qualified.
- *Missing / deferred / unqualified.* A competitor configuration that is missing, deferred, not constructible, or
  not Q1-qualified cannot be used (it never matches). If one of our configurations X is not qualified (Q1+Q2) on a
  mesh of $w$, X is not part of "us" on $w$; if no X is qualified on $w$, $w$ counts as matched (there is no
  qualified configuration of ours to claim an advantage with) and this is flagged. If any comparison needed for the
  verdict is missing, the outcome is "stop rule **not evaluable**" (never "not fired").
- *Interpretation.* "Not fired" means only that on at least one workload no tested continuous-POD configuration
  matched one of our configurations; it is reported as "comparative multi-mesh cost advantage survives against the
  strongest **tested** continuous-POD control on workloads …", never as proof of a general advantage.
- *Cost uncertainty.* Costs are compared with $\le$ as worded. A comparison with
  $|C_{\rm CP}/C_X - 1| < 0.05$ is labelled a **near-tie** (5 % = the timing drift band); if the verdict on any
  workload depends only on near-ties, this is stated.

**A1.2 Cost model and ledger (audit item 2).**
- *Bank fit.* The A100 log (job 2835788) has two regimes: steps 1–290,000 sub-sampled (≈ 0.0265 s/step) and the last
  `FULL_LAST` = 10,000 full-batch steps (≈ 0.300 s/step; 7,683 s at step 290,000, 10,681 s at 300,000). The
  projection is phase-wise:
  $T_{\rm fit} = t_1 + 289{,}999\,\rho_{\rm sub} + 10{,}000\,\rho_{\rm full} + t_{\rm done}$, with $t_1$ the
  step-1 timestamp (compilation and first step), $\rho_{\rm sub}$ the rate over the logged window 5,000→15,000,
  $\rho_{\rm full}$ the rate over a logged full-batch window of 5,000 steps, $t_{\rm done}$ the time from the last
  step to the "done" line. *Validation (CPU, from the A100 log alone):* the formula with A100 rates (full window
  290,000→300,000) must reproduce the recorded 10,684.65 s within 5 %. *H200 probe (J2):* the byte-identical script
  (the 26 staged files of job 2835788, each resolved by its sha256) with `STEPS=20000 FULL_LAST=5000` and every
  other environment value of job 2835788; $\rho_{\rm sub}$ from 5,000→15,000, $\rho_{\rm full}$ from
  15,000→20,000, data generation (576 FOM trajectories) and $t_1$ measured directly; the process is stopped after the
  step-20,000 line. Label: "H200 bank-fit cost projected phase-wise from measured H200 rates".
  The deployed checkpoint's other dependencies were checked: LSPG linear-span settings use the bank parameters
  `g`, `B` (bitwise equal in r3a and dn256b), `rotation_R512.npz`, the generated rules and the cold-start Gauss
  rule built in `Bank` setup (counted in $P(L)$); the head `h` and `Z_tr` are not used by `p1core` LSPG (to be
  confirmed in the code audit by grep).
- *Ledger.* Every arm's cost is a sum over named artefacts, each with its measured seconds and the job that measured
  it: `snap[L]`, `pod[L]` (incl. certificate), `codes[L]` (trust radii), `states[L]`, `contrib[L,r]`,
  `nnls[L,r]` (incl. any registered retry), `resample[s→L,r_max]` (incl. QR), `armdata[L,cfg]`,
  `compile[L,cfg]`; ours: `bankdata`, `bankfit`, `table[L]`, `ops[L]`, `setting[L,X]`, `compile[L,X]`. A workload
  charges the union of the artefacts it needs, each once. Source-mesh preparation of CP (states, contributions,
  NNLS, arm data, compilation) is charged whenever the source rule is used (CPU, or $s\in\mathcal M$).
- Omissions and their direction, stated next to the numbers: rotation and our rule selection counted 0 (favour
  ours); competitor Q2 not gated (favours competitors); competitor grid omissions (favour ours); A_sel charges
  competitors' validation runs while ours' selection is unmeasured (favours ours).

**A1.3 Resampling, codes and reuse (audit item 3).** Corrections to §3.3:
- The transform is the repository's orthonormal self-inverse DST-I (`engines.dst1`, odd-extension FFT). With
  orthonormal 2D transforms $U_s, U_L$ and coefficient embedding $E$ (zero-pad) or truncation $T$:
  $\Phi_L = U_L E U_s\Phi_s$ (prolongation) or $\tilde\Phi_L = U_L T U_s \Phi_s$ (restriction). **No extra
  $s/L$ factor**: the orthonormal transforms already give Euclidean orthonormality, and the nodal values of
  $\Phi_L$ equal $(s/L)$ × the source sine interpolant. (The "scaled by $s/L$" clause of §3.3 is withdrawn.)
- Restriction: $\tilde\Phi_L = Q R$ (two QR passes, signs fixed, mode order kept). **Rank loss**: if
  $\min_i |R_{ii}| / \max_i |R_{ii}| < 10^{-10}$ the rank $r$ is not constructible on that mesh. Restriction is a
  low-pass projection, not evaluation of the source series at the coarse nodes; this is stated.
- *Codes and trust radii.* The same physical field has codes $c_L = (L/s)\,c_s$ under prolongation and
  $c_L = (L/s)\,R\,c_s$ under restriction ($R$ from the QR above). The LM trust radii are recomputed from the
  transformed source snapshot codes $C_L$ (rows) with the jcp-hr-baselines rule (`trust_radii`); the source codes
  $C_s$ are saved by J0.
- *Node mapping (zero-based flat index $p$ on mesh $s$):* $i = \lfloor p/(s-1)\rfloor + 1$,
  $j = (p \bmod (s-1)) + 1$, $p' = (iL/s - 1)(L-1) + (jL/s - 1)$.
- Reuse keeps the same sampling measure (nodes, weights); the FD stencil and residual at those nodes are the
  target mesh's, so reuse accuracy is a **measured** property, not assumed. LM damping/stopping behaviour under
  reuse is covered by the in-job Q1 certification.

**A1.4 Qualification (audit item 4).**
- *Ours:* jcp-p1-scaling's certificates are prior evidence only (Q1 covered all 38 cases, Q2 only `sub16`). **J5 is
  replaced** by one job that runs byte-identical `p1acc.py` at $256^2$, $1024^2$ and $4096^2$ (three sequential
  processes, one H200) for the three settings × {BE, CNR} with `cases = all38` and **`q2_cases = all38`**, and
  successor/check rules as registered by jcp-p1-scaling. In-job Q1 + Q2 verdicts are computed from its outputs by
  the lane's checker with the COMMON-BRIEF §3 bars (non-stationary ≤ 1 % pooled, distance to tight ≤ 2.5e-4 every
  case, tight/tighter ≤ 2.5e-5 on dev6, ρ ≤ 0.116 and successor-rule rollout distance ≤ 2.5e-4 on every case).
  The p1 errors are a replication check only. The timing job J6 additionally checks that every timed run of ours
  reproduces J5's coefficients (max relative difference ≤ 1e-10), every internal state not being recorded there.
- *Competitors:* labelled **Q1-qualified** (not "numerically qualified"); the Q2 analogue (rollout distance
  between $\kappa=4$ and $\kappa=8$ at the same $r$ and mesh) is reported. A sensitivity verdict restricted to
  competitor configurations whose Q2 analogue is ≤ 2.5e-4 on every case is reported beside the primary one.
- Conclusions are "against the strongest **tested** competitor" (grid of §3.2; the omissions favour ours).

**A1.5 Scoring (audit item 5).** Fields are restricted with zero walls to the $257^2$ lattice (`Cell.lattice` =
`hrcore.lattice_nodes` + `to_lattice`; ours: `p1common.restrict` of the padded full field). Per-mesh results are
reported beside the pooled ones; the pooled median counts each physical case once per mesh (stated).

**A1.6 Must-fail controls, replaced (audit item 6).** Each is run in S1 on real data with its expected predicate:
1. GPU preflight with `JAX_PLATFORMS=cpu` exits 42.
2. *Physical consistency of prolongation:* on the 255² scoring-lattice interior nodes (common to every mesh),
   $(L/s)\,\Phi_L = \Phi_s$ to $\le 10^{-12}$ relative (sine interpolation is exact at the source nodes), and
   orthonormality ≤ 1e-10. *Must fail:* (a) a coefficient index shift (`mutate`) and (b) an x/y-transposed reshape
   each violate the lattice equality; (c) a restricted basis **without** the QR fails orthonormality.
   Trust-radius consistency: the trust radii computed from the transformed codes equal $(L/s)$ × source radii
   under prolongation to $\le 10^{-12}$ relative.
3. *Rule mapping:* mapped node coordinates equal the source coordinates exactly, and
   $(L/s)\,\Phi_L[p'] = \Phi_s[p]$ at every sampled node (≤ 1e-12). *Must fail:* unmapped indices violate both.
   Reuse accuracy versus refit is measured and reported, not gated.
4. *Q1:* the same configuration with LM budget 0 (the predictor state returned unsolved) fails Q1 on dev6
   (non-stationary steps > 1 %); shown on real cases.
5. *Blocked contributions* at $1024^2$, $r=128$, all 120 states: identical candidate sets drawn independently by
   each builder, kept-state mask identical, $G$ and $b$ equal to ≤ 1e-10 relative (Frobenius). *Must fail:* the
   `drop_adv_jac` mutation exceeds 1e-6 relative; shown.
6. *Case identity:* for every scored run, the $t_0$ output (projected IC) is within $e_0 \le 0.1$ of the reference
   $t_0$ of the same case; *must fail:* scoring dev6 case 0 against dev6 case 1's reference gives $e_0 > 0.1$.
7. *Timing drift:* sentinel and A1/A2 medians within $[1/1.1, 1.1]$; *must fail:* a host sleep of 0.25 × the
   measured median, inside the timed region of one A2 block, trips it.
8. *Decision logic on the smoke's real numbers:* separately, (a) competitor errors set to $+\infty$, (b) competitor
   costs × 10⁶, (c) competitor Q1 flags false, (d) one competitor measurement removed → no match ((d): "not
   evaluable"); (e) our costs × 10⁶ with real competitor numbers → match on every workload the competitor has
   admissible configurations for.
9. *Replication fixture:* p1acc for R128 at $1024^2$ on dev6 (BE): errors equal p1 `e0_per_case` (≤ 1e-8);
   *must fail:* compared against p1's R256 errors.

**A1.7 Feasibility (audit item 7).** S1 runs a $4096^2$ memory and time pilot: resample $\Phi_{384}$ from
$1024^2$, blocked contributions for 12 states at $r=384$, the $r=384$ NNLS path to $\kappa=8$ (time recorded), one
ECSW query, and ours' bank table at $4096^2$ (sequentially, not co-resident). J0 records the NNLS path times at the
source meshes. A smoke-derived certificate (phase seconds × counts × 1.5) gates every production job; each job
has a wall-clock limit and writes partial results after every configuration. Timing protocol (J6): per mesh,
sequential blocks with the arm's arrays loaded only for its block and freed after (`del` + `gc` + device memory
check), `burn(0.05)` before every timed call, compile measured as cold first call − warm mean (persistent
compilation cache disabled), ours A1 → competitors B (in blocks of ≤ 6 configurations, each followed by an ours
**sentinel**: R128-BE dev6 case 0, 3 repetitions) → ours A2; each configuration timed once per case on all 38
cases (A1 and A2 for ours). Drift: every sentinel and the A2/A1 median ratio within $[1/1.1, 1.1]$, else the
mesh's blocks are repeated once, then labelled invalid. Timed outputs must reproduce the accuracy jobs'
coefficients (≤ 1e-10 relative). A deferred competitor makes the verdict "not evaluable" on the workloads it
affects, never evidence for ours.

**A1.8 PM at $4096^2$ (audit item 8).** Relabelled: **estimated snapshot-only cost** for the registered recipe =
576 × mean measured time of 8 randomly chosen training trajectories (seed 20261021) run with the snapshot recipe
(CNR, Δt = 0.005, saved steps 0…50 of the 12-step list) at $4096^2$, with the min–max of the 8 times × 576 as the
range. POD and ECSW are not costed (the dense snapshot matrix does not fit; an out-of-core build is possible but
not run). It never enters the CP stop rule; PM rows on workloads containing $4096^2$ say "PM ≥ estimated snapshot
cost (accuracy not measured)".

**A1.9 Interpretation (audit item 9).** The conclusion is scoped to this implementation, this problem, this
query contract (full-field Gaussian input on the mesh, six full-field outputs on the mesh) and the tested
competitors. The CP control pays fine-grid initial projection and FD residual evaluation at sampled nodes; a
stronger mesh-transferable POD (e.g. precomputed reduced operators, cheaper initialisation) may exist and is not
tested. Break-even is reported as the lower envelope of cost lines in $q$ (possibly several switches, or none in
the range), not a single $q^\*$.

**A1.10 Jobs and budget (replaces §8 table).** S1 (0.5 h), J0 sources (0.7 h), J1 m256 (0.4 h), J2 probe (0.4 h),
J3 m1024 (0.5 h), J4 m4096 (1.4 h incl. the 8 PM-estimate trajectories), J5 ours (1.2 h, three meshes), J6 time
(0.8 h): ≈ 5.9 GPU-h, cap 8. Defer order if the certificates exceed the remaining cap: (1) CPU-1024 at $4096^2$,
(2) CPR-1024 at $4096^2$, (3) $r=384$ refits at $4096^2$; each deferral makes the affected workloads "not
evaluable" for that arm only (the other CP arms still count).
