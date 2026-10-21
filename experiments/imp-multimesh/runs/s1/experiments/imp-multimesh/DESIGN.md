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

---

## Amendment A2 (2026-10-21, before any code ran at size; answers `audits/codex-design-2.md`)

A2 governs wherever it conflicts with sections 1–9 or A1.

**A2.1 Accuracy matching (replaces the tie and median clauses of A1.1).**
- *Primary (the stop rule's own metric):* the **aggregate worst-error comparison** $W_{\rm CP}(w) \le W_X(w)$,
  strict, no allowance. This is what "equal or better worst error" names; it is labelled an aggregate comparison,
  not the COMMON-BRIEF casewise rank rule.
- *Reported beside it:* (i) the aggregate comparison with allowance $2\max_c\varepsilon_{\rm ref}(c)$
  (sensitivity); (ii) **casewise dominance**: on every (mesh, case) pair of $w$,
  $e_{\rm CP} \le e_X + 2\varepsilon_{\rm ref}(c)$ (the binding rank rule, case by case); (iii) the pooled median,
  strict ($M_{\rm CP}\le M_X$). The lane brief's "accuracy-matched (worst and median)" claim is the
  conjunction primary ∧ (iii); it is reported as a separate verdict from the verbatim stop rule.
- *Three-valued logic per (workload $w$, configuration X).* MATCH if a measured, Q1-qualified CP combination is
  admissible (accuracy) and costs $\le C_X(w)$ — regardless of other missing arms. NO-MATCH if every CP combination
  of every registered CP variant is measured and none is both admissible and no more costly. Otherwise UNRESOLVED.
  X's own status: QUALIFIED (Q1+Q2 in-job on every mesh of $w$), NOT-QUALIFIED (measured failure), or UNRESOLVED
  (evidence missing — never treated as a failure).
- *Stop rule outcome.* FIRES iff every (w, X) with X QUALIFIED is MATCH and no X is UNRESOLVED. NOT FIRED iff some
  (w, X) with X QUALIFIED is NO-MATCH. Otherwise NOT EVALUABLE. A workload where every X is NOT-QUALIFIED is
  reported as "vacuous" and does not block firing. These three outcomes and "vacuous" are reported separately.
- *Language.* NOT FIRED is reported as "the stop rule did not fire", followed by the specific trade-off for every
  NO-MATCH: the cheapest CP combination and why it fails (accuracy or cost), and the most accurate CP combination
  and its cost. No sentence states a comparative cost advantage unless a CP combination that is accuracy-admissible
  exists and costs more than X on that workload.
- *Near-ties.* A cost comparison with $C_{\rm CP}/C_X \in [1/1.1, 1.1]$ is a **near-tie** (the registered timing
  drift band); if an outcome depends only on near-ties this is stated. The bank-fit and head-fit projections carry
  their own sensitivity variants (A2.2) and are not covered by this band.

**A2.2 Our offline cost (replaces the bank-fit and dependency clauses of A1.2).**
- *Head dependency (audit finding confirmed).* `qcore.Model.trust_linear` evaluates the checkpoint head on `Z_tr`
  to set the LSPG trust radius (`p1common.Bank.cold_trust`), so the deployed configurations depend on the dn256b
  head fit. Its producing work, from the dn256b log (job 2837431, A100 80GB PCIe): coefficient extraction 915.24 s
  + the `full` head-fit variant that emitted the deployed pickle (log line 391) 2,675.56 s. The other two head-fit
  variants (2,636.95 s, 2,696.54 s) are selection work (counted in A_sel). `Z_tr` comes from the bank job.
- *Bank fit (H200).* $T_{\rm fit} = t_1 + 289{,}999\,\rho_{\rm sub} + \tau_{\rm tr} + 10{,}000\,\rho_{\rm full}
  + t_{\rm done}$, with $\tau_{\rm tr}$ the one-off sub→full transition overhead. J2 runs the byte-identical script
  with `STEPS=25000 FULL_LAST=10000` (full phase 15,000→25,000): $\rho_{\rm sub}$ from the logged window
  5,000→15,000, $\rho_{\rm full}$ from the steady window 20,000→25,000, $\tau_{\rm tr}$ = (15,000→20,000 window) −
  5,000 $\rho_{\rm full}$, $t_1$ and the 576-trajectory data generation directly, and $t_{\rm done}$ from the
  step-25,000 line to the `done:` line (the process is stopped after that line). Validation on the A100 log with the
  same decomposition ($\tau_{\rm tr}$ from 290,000→295,000 minus 295,000→300,000) must reproduce 10,684.65 s within
  5 %.
- *Head fit on the H200.* Not re-run. Primary: the A100-80GB seconds × the measured H200/A100-40GB ratio of
  $\rho_{\rm sub}$ (a cross-GPU projection, labelled). Sensitivities: unscaled A100-80GB seconds; excluded.
- *Still uncosted (counted 0, favour ours):* the rotation `rotation_R512.npz` (its producing job is in an
  unmerged earlier branch and was not measured) and the selection of the frozen rules. Because of these, the total
  is called **"full cost of the measured items"**, never simply "full cost". A sensitivity adds a stand-in equal to
  the competitors' own selection work (A_sel) to ours, to show whether a selection cost of that size would change
  the verdict.
- *Sensitivities reported for every verdict:* A_100 (bank fit at the A100-measured 10,684.65 s, head at
  A100-80GB seconds), A_nohead (head excluded), A_sel (competitor selection work added, and the stand-in for ours).

**A2.3 Ledger with dependency keys (refines A1.2).** Artefacts are keyed by everything they depend on:
`states[L]` (FOM trajectories of the 24 ECSW training cases on mesh L; basis-independent, shared by every arm
that needs them), `snap[s]`, `pod[s]`, `codes[s→L, r]`, `resample[s→L]`, `contrib[basis, L, r]`,
`nnls[basis, L, r]` (basis = PM-s or CP-s), `armdata[basis, L, r, κ]`, `compile[arm, L, cfg]`; ours
`bankdata`, `bankfit`, `head`, `table[L]`, `ops[L]`, `setting[L, X]`, `compile[L, X]`. Rule construction for
CPU-s needs `states[s]`, `contrib[CP-s, s, r]`, `nnls[CP-s, s, r]` only; `armdata`/`compile` at $s$ are charged
only when $s\in\mathcal M$. Compilation is charged once, from an explicit ahead-of-time measurement
(`jax.jit(...).lower(...).compile()` on a fresh jit object, wall seconds); setup/build timers exclude it.

**A2.4 Resampling implementation details (A1.3 accepted; frozen here).** $R$ is the combined reconstruction factor
of both sign-fixed QR passes ($\tilde\Phi = Q R$ with $R = S_2R_2S_1R_1$ as computed). Codes are transformed per
deployed rank: $C_{L,r} = (L/s)\,C_s[:, :r]\,R[:r,:r]^\top$ (the prefix of an upper-triangular factorisation is
the factorisation of the prefix). Rank acceptance: the condition number of $R[:r,:r]$ (from its singular values)
≤ 1e10.

**A2.5 Qualification and identity (refines A1.4).** The historical statement is corrected: jcp-p1-scaling covered
Q1 on all 38 cases except $4096^2$ R128/R256 (16 cases) and Q2 on `sub16` only; this is prior evidence. J5 sets
`cases = all38` and `q2_cases = all38` in **each** configuration, with the registered successor/check rules.
Competitors: "Q1-qualified"; the κ=4 vs κ=8 comparison is a **rollout-agreement sensitivity**, not Q2.
Timing identity (J6): ours is timed through the recording program (nrec = 51, jcp-p1-scaling's production
program), and every timed run must reproduce the accuracy run's six outputs **and** every internal state, per-step
exit reason and iteration count; competitors: the timed run's full trajectory $C$, per-step reasons and iteration
counts must equal the accuracy run's (max relative difference ≤ 1e-10 on $C$, exact on integers).

**A2.6 Controls (replaces A1.6 items 2, 4, 6–9).**
- 2. Lattice/node consistency of prolongation and the mutations (a) shift, (b) transpose as in A1.6; (c) replaced:
  the orthonormality checker must fail on the real restricted basis with one column scaled by $1+10^{-6}$; the raw
  (pre-QR) defect of the real restriction is reported. QR reconstruction $\|QR-\tilde\Phi\|/\|\tilde\Phi\|\le
  10^{-12}$ and the code transformation (an in-span field $\Phi_{s,r}c$ restricted: its codes in $\Phi_{L,r}$ equal
  $(L/s)R[:r,:r]c$ to ≤ 1e-10) are checked on the real bases.
- 4. Frozen fixture: CPR at $1024^2$ from the S1 source, $r=128$, $\kappa=4$, CNR, dev6. Mutation: LM budget 0. If
  it does not trip Q1 on the fixture, the registered fallback mutation is a stepper that returns the previous state
  unchanged ($c_{n+1}=c_n$), which is non-stationary wherever the dynamics move; the control result is recorded.
- 6. Case identity: every run record carries (cohort, index, the full descriptor row $(c_x,c_y,w,a,\nu)$); scoring
  asserts the descriptor row equals the cohort table row of the reference's case and that the reference $t_0$
  equals the IC built from that row (`score.Refs`); *must fail:* a dev6 case-0 run presented as case 1 trips the
  descriptor assertion (and its $e_0$ is reported).
- 7. Timing injection: a host sleep of 0.25 × that configuration's A1 median added to **every** timed call of one
  A2 block; the A2/A1 median ratio must leave the band.
- 8. Decision logic on the smoke's real numbers, each against an expected outcome: (a) CP errors → +∞: NO-MATCH;
  (b) every CP cost set to $C_X(w)\cdot 1.5 + 1$: NO-MATCH; (c) CP Q1 flags false: NO-MATCH; (d) with two
  witnesses present, one witness removed: still MATCH; (e) every witness removed while unmeasured combinations
  remain: UNRESOLVED; (f) X status removed: stop rule NOT EVALUABLE.
- 9. Replication fixture: p1acc R128 BE at $1024^2$, dev6: errors equal p1 `e0_per_case` (≤ 1e-8). *Must fail:*
  against p1's R256 BE errors, which differ by ≥ 1.9e-5 on every dev6 case (smallest gap, dev6 case 1:
  0.0133533 vs 0.0133342; read from `jcp-p1-scaling/checks/analysis.json`).

**A2.7 Smoke inputs and projections (refines A1.7).** S1 is self-contained: it builds a reduced source at
$1024^2$ from 64 training trajectories (12 saved steps, 768 snapshots), POD to $r=384$, codes, 24-trajectory ECSW
states at $1024^2$ (the production recipe), and uses it for every control and for the $4096^2$ memory pilot (whose
NNLS uses 12 states, for memory only). Production NNLS run time at $4096^2$ is projected from J0's measured per-rank
path times at $1024^2$ (same candidate count and state count; ×1.5); if a path hits its time limit the support
reached is recorded and the configurations above it are "not constructible" (recorded, never silently dropped).
Timing repetitions: competitors 2 timed observations per case (76 per configuration), ours one per case in A1 and
one in A2 (76); every observation is saved; medians and means are reported, outliers (> 3 × median) counted.
Input/output boundary, identical for every arm: the query starts from the full-field initial condition already on
the device in the arm's native layout and ends when its six full-field outputs on the mesh are on the device
(`block_until_ready`); host transfers and storage are excluded for every arm.

**A2.8 PM at $4096^2$ (replaces A1.8 wording).** Reported as "**estimated** snapshot-only cost (576 × the mean of 8
sampled trajectory times; sample range shown descriptively); remaining setup and accuracy not measured". No
inequality or "loses regardless" statement is made from it.

**A2.9 Budget.** J2 grows to 0.5 h (25,000 steps). Total ≈ 6.0 GPU-h projected, cap 8.

---

## Amendment A3 (2026-10-21, before any job; answers `audits/codex-design-3.md`)

A3 governs wherever it conflicts with sections 1–9, A1 or A2.

**A3.1 Configuration terminal states and searches.** Every registered competitor configuration ends in exactly one
state: MEASURED-Q1 (measured, Q1 passed), MEASURED-FAIL (measured, Q1 failed), NOT-CONSTRUCTIBLE (demonstrated:
basis certificate or rank check failed, or the NNLS path ended at a certified KKT optimum without a usable support
— jcp-hr-baselines A23 fills a target from such an optimum, so this case is rare), or UNRESOLVED (NNLS time limit
reached before the target, deferral, crash, missing file). Only MEASURED-Q1 configurations can be witnesses;
MEASURED-FAIL and NOT-CONSTRUCTIBLE configurations are resolved non-witnesses; UNRESOLVED configurations keep a
NO-MATCH from being proven. Each verdict (primary worst; worst with allowance; casewise; worst ∧ median;
rollout-agreement sensitivity) runs its **own** enumeration of admissible combinations.

**A3.2 Head dependency, corrected.** The deployed `Z_tr` is refitted by the head job (`sep_hfit_run.py` writes the
fitted codes as `Z_tr`), so the head dependency is: coefficient extraction (915.24 s on A100-80GB, which includes
its own data generation of 4,608 trajectories) + the producing `mid` arm of the `full` head-fit experiment
(360.85 s recorded arm seconds). The `wide` and `k32` arms of `full` (2,305.51 s) and the `f576` / `f4096`
experiments are selection work (A_sel). **Re-measured on the H200 in J2**: after the bank-fit probe, J2 runs the
byte-identical dn256b code tree (its 34 staged files resolved by sha256; checkpoint = the r3a pickle, sha256
`acc959e3…`) for the extraction with the dn256b environment, then `sep_hfit_run.py` with `ARMS=mid EMIT=mid` and
otherwise the `full` environment; the H200 seconds are the extraction's `total_seconds` and the `mid` arm's seconds.
The cross-GPU ratio projection of A2.2 is withdrawn; the A100-80GB values are kept only for the A_100 sensitivity.

**A3.3 Compilation.** Build-phase timers are wall clock and include the compilation of their own kernels.
Query compilation is the ahead-of-time compile of the deployed fused query (`lower().compile()`) measured in the
timing process with the persistent compilation cache disabled; in-process autotuning caches may be warm, so this
is labelled "warm-process compile". Every verdict is recomputed with all query-compile seconds × 0 and × 3; any
verdict that changes is reported as compile-sensitive.

**A3.4 Ledger canonicalisation.** PM-s and every CP-s artefact at mesh $s$ are the same artefacts (same keys); no
cost is charged twice because of a label. A_sel adds, each once: the competitors' dev6 configuration runs that are
not deployed (their tuning work), the non-producing head arms and head experiments (A3.2), and, for ours, the
stand-in equal to the competitors' selection work (a scenario, not a bound on the rotation or rule-selection
cost).

**A3.5 Controls 4 and 8 (refined).**
- 4: the control must fire (Q1 fails under the mutation) before any production job is submitted; if neither the
  budget-0 nor the fallback mutation fails Q1 on the fixture, the Q1 checker is unverified and production stops.
- 8: run on fixtures derived from the smoke's measured records, each with its stated precondition: (a)–(c) on a
  fixture whose declared universe is exactly the measured configurations (so it is exhaustive): errors +∞ /
  costs $1.5C_X+1$ / Q1 flags false → NO-MATCH; (d) a fixture with a measured witness duplicated under a second
  identifier (two witnesses): one removed → MATCH; (e) every witness removed while one configuration is UNRESOLVED →
  UNRESOLVED; (f) one X set UNRESOLVED while no other (w, X) is NO-MATCH → NOT EVALUABLE, and the same with another
  X NO-MATCH → NOT FIRED; (g) every X NOT-QUALIFIED → "vacuous" flagged.

**A3.6 NNLS memory and runtime.** The S1 $4096^2$ pilot builds the production-shaped contribution matrix (its 12
real states tiled to 120 states, $r=384$, 32,768 candidates: 12.1 GB) and runs the NNLS on it for memory (device
peak recorded with $\Phi_{384}$ resident). Run time: J0's measured path times × 1.5 is a planning estimate only;
every path has `max_seconds`, and a path that stops on time leaves the unreached configurations UNRESOLVED (A3.1).

**A3.7 Wording.** A "vacuous" outcome is reported with its reason (no qualified configuration of ours), never as an
empirical CP match. A comparative cost advantage of X on $w$ is stated only when NO-MATCH is proven for that
(w, X) and the least-cost admissible competitor combination (if any) is named with its cost under the stated
accounting variant. The verdict is labelled "full cost of the measured items" and the uncosted items are listed
next to it.

**A3.8 Budget.** J2 adds the H200 head re-measurement (≈ 0.35 h): total ≈ 6.4 GPU-h projected, cap 8.

---

## Amendment A4 (2026-10-21, before any job; answers `audits/codex-design-4.md`)

A4 governs wherever it conflicts with sections 1–9 or A1–A3.

**A4.1 Default terminal state.** The only registered impossibility certificates are: the basis certificate failing
(orthonormality / POD certificate), and the rank check of A2.4 failing. **Every other unsuccessful construction**
(NNLS `time`, `max_iter`, `cap`, `incomplete (KKT violated)`, a quality-limited path without a usable checkpoint, a
crash, a missing rule) leaves the configuration UNRESOLVED. A target filled from a KKT-certified optimum (A23 of
jcp-hr-baselines) is constructible.

**A4.2 Head boundary.** The head dependency is the **complete producing invocations**: on the H200 (J2), the wall
seconds of the extraction process and of the single-arm `sep_hfit_run.py` process (`ARMS=mid EMIT=mid`), each from
process start to exit (input loading, whitening, fitting, evaluation, emission). For A_100 the same boundary is
approximated from the historical records as extraction `total_seconds` (915.24 s) + `mid.seconds` (360.85 s) + the
`full` run's time outside every arm timer (9.20 s, all charged to the dependency).

**A4.3 Query compilation.** The deployed query's jit object is created immediately before its ahead-of-time
`lower().compile()`, so that executable has never been compiled in the process; in addition a
`jax.monitoring` duration listener records JAX's own backend-compile events during that call, and the charged
compile cost is the measured wall time of `lower().compile()` only if at least one backend-compile event was
recorded (otherwise the entry is flagged "cache hit suspected" and the verdict is reported with and without it).
The ×0/×3 compile scenario of A3.3 is kept as a descriptive sensitivity, not a bound.

**A4.4 A_sel, exactly.** A_sel adds, each once and each with its hardware provenance: (i) every competitor
configuration run of the target jobs (all 38-case prod/tight/tighter runs: measured H200 `run_seconds`), (ii) the
construction work of every non-deployed competitor configuration (other ranks' contributions and NNLS paths:
measured H200), (iii) the non-producing head arms and head experiments (`wide`, `k32`, `f576`, `f4096`:
historical A100-80GB seconds, labelled cross-GPU, A_sel only), (iv) our J5 qualification runs (measured H200), and
(v) the stand-in of A3.4 for our unmeasured selection. A_sel is descriptive.

**A4.5 NNLS memory certificate.** Device bytes at $4096^2$, $r=384$: $\Phi$ 51.5 GB + $G$ 12.1 GB +
NNLS workspaces bounded by the fixed capacity of `nnls.UpdQR` (rows × capacity × 8 B for $Q$ and capacity² × 8 B for
$R$, with at most two copies alive during a refactorisation): for 46,080 rows and capacity 8,192, ≤ 2 × (3.0 + 0.5)
GB; total ≤ 71 GB, below the 130 GB usable on the H200 at memory fraction 0.92. Host: 120 states × 3 × 134 MB +
the host copy of $G$ ≈ 61 GB, below the 240 GB requested. The S1 pilot records the observed device peak; the
certificate holds if the observed peak is below the bound.
The **unreduced POD-LSPG diagnostics are withdrawn** (not run, not reported); the only competitor rollout check is
the κ=4 vs κ=8 rollout-agreement sensitivity.

**A4.6 Claim rule (restores A2's existence condition).** A comparative cost advantage of X on $w$ is claimed only
if (a) an accuracy-admissible, Q1-qualified competitor combination exists for that verdict, (b) the search is
resolved (no UNRESOLVED configuration of any CP variant at the meshes of $w$), and (c) the minimum cost over the
admissible combinations exceeds $C_X(w)$. A NO-MATCH without an admissible competitor is reported as "competitor
not accuracy-matched; cost advantage undetermined".

---

## Amendment A5 (2026-10-21, before any job; answers `audits/codex-design-5.md`)

**A5.1 Compilation (replaces A4.3's acceptance rule).** J6 adds **fresh-process** compile measurements: for each
mesh, four separate Python processes (persistent compilation cache disabled), each building exactly one deployed
query and timing its first `lower().compile()`: ours R128-BE and R384-CNR, competitor $r=128$/κ=4 and
$r=384$/κ=8 (script `mmcompile.py`). The charged compile cost of a configuration is
max(in-process AOT measurement, the fresh-process measurement of the same mesh, side and nearest $r$). Verdicts
are also reported with compile × 0; a verdict that changes is labelled compile-sensitive. No verdict uses a
compile value known to be a cache hit.

**A5.2 A_sel (replaces A4.4).** A_sel(w) = [union of all selection work] − [items already in A_dep(w)], each item
once, where the union is: every competitor configuration's target-job wall time (`run_seconds`: prod, tight,
tighter and certification calls including their compilation; plus its `armdata_seconds`), every competitor
construction item of the arm not used by the deployed combination (other ranks' contributions/NNLS paths, codes),
our J5 job's p1acc wall time, the historical non-producing head arms (A100-80GB, labelled cross-GPU), and the
A3.4 stand-in. Timer boundaries: `run_seconds` is wall clock around `run_config` (it contains its own compiles);
deployment compile (A5.1) is a separate measurement and is never added to A_sel. A_sel is descriptive and mixes
hardware; it never decides the stop rule.

**A5.3 NNLS memory (replaces A4.5's certificate).** No complete a-priori allocation bound is claimed. Feasibility is
empirical: the S1 pilot records the device and host peaks of the production-shaped matrix and its NNLS run; the
production jobs write results after every configuration; a configuration whose construction fails for memory is
UNRESOLVED (A4.1) and is retried once in a fresh job if the GPU-hour cap allows; a remaining failure stays
UNRESOLVED and is reported (it can only prevent a NO-MATCH or FIRES verdict from being proven, never create one).
