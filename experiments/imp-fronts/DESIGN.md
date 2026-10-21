# imp-fronts DESIGN (pre-registration): steep fronts — E8 (W2), E9 (out-of-regime indicator), E10 (conditional)

Status: pre-registration, written 2026-10-21 before any code or run. Amendments are appended at the end, dated; this
text is never rewritten. Brief: main `reports/2026-10-21-improvement-campaign/campaign/briefs/imp-fronts.md`
(binding, with `COMMON-BRIEF.md`). Sources: ideation `reports/2026-10-20-improvement-ideation/report.md` §2.3, F7,
F11 (+F13, F16), F12, F15, §5 row W2; jcp-regimes `report.md` (cohorts N0s, C1, C2).

## 0. Problem

The ROM fails on fronts steeper than training. On the jcp-regimes cohorts the r = 384 bank floor (best lattice fit
by the first 384 bank functions) has median 3.47 % on C1 and 9.88 % on C2 against 0.0266 % on val32, the in-span
part of the ROM error is as large or larger, and widening the prefix 384 → 512 barely helps (C1 1.1×). Phase A asks
four cheap CPU questions that decide what Phase B retrains are worth running; Phase B runs single-change retrains
against a control retrain.

## 1. Data and conventions (both phases)

- **Bank.** Incumbent = `separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl` + `rotation_R512.npz`,
  evaluated by the vendored NumPy evaluator (`jcp-regimes/npcheck.py`, read-only import from the consolidated
  worktree, sha256 of both files recorded in every output). $G_r$ = first $r$ ordered bank functions on the
  257² scoring lattice $x_i=i/256$ (wall nodes zeroed by `decode257`).
- **References.** dev6 / val32: E0 ST store (`jcp-references/refstore.py`, local store copy of
  `worktrees/2026-10-06-jcp-references/.../store`, manifest-verified, field `finest` = `a2_L8192`, eligibility from
  `reference_status`). Regime cohorts N0, N0s, Ap, Am, C1, C2, B1, B2: jcp-regimes E0-style ST store
  (`worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/store/2d/<cohort>/ST/cNNN.npz`, array `a2_L8192`),
  eligibility = `checks/analysis.json` → `configs.R384|LSPG_CNR_1.per_case[*].eligible`. ε_ref is a conditional
  Richardson estimate; every accuracy number carries that label.
- **Error metric** (project convention, metric id `2|lattice=256|times=0:0.05:0.25|norm=fixed_t0|max_k>=1`):
  $e(v,u)=\max_{k=1..5}\lVert v(t_k)-u(t_k)\rVert_2/\lVert u(t_0)\rVert_2$ on the 257² lattice.
- **Floor** of a field $u$ in a basis $B$ (orthonormalised on the lattice, $P_B$ the lattice projector):
  $f_B(u)=\max_{k=1..5}\lVert(I-P_B)u(t_k)\rVert/\lVert u(t_0)\rVert$.
- **Laplacian sines.** $S_K$ = first $K$ products $\sin(\pi k_1x)\sin(\pi k_2y)$, $1\le k_i\le255$, ordered by
  $k_1^2+k_2^2$ (stable argsort, ties by flattened index), orthonormal on the lattice interior via DST-I (Parseval).
- **Cases are the statistical unit**; snapshots are never bootstrapped. Ineligible references are excluded and
  counted. Invalid ROM runs are counted and reported, never silently dropped.
- Phase A uses only dev/val and the jcp-regimes regime cohorts (all already opened by earlier lanes; none sealed,
  none reserved-OOD). Phase A results select which Phase B arms run; Phase B scores only on **fresh** cohorts that
  this lane generates from new seeds (§B3), so Phase A selection cannot leak into Phase B scoring.

## 2. Phase A — CPU diagnostics (0 GPU-h)

All Phase A computations are NumPy/SciPy on CPU; the only JAX use is the incumbent upwind generator in A1, run with
`JAX_PLATFORMS=cpu` under the campaign slot guard. Each diagnostic has a positive control (the detector must fire
on a case where the effect is present by construction) and a negative / identity control (it must not fire where
the effect is absent by construction). **If a control does not behave as required, that diagnostic's verdict is
"INCONCLUSIVE (control failed)"** and it is not used for any Phase B decision.

### A1. Smearing test with a sine control (ideation F11 Step 0; a diagnostic, not a gate)

*Question.* Is the bank's large floor on steep in-range cases consistent with having been trained on smeared
(first-order upwind, BE) data?

- **Upwind fields.** The incumbent generator, byte-identical copy of `burgers2d_film.py` (sha256 `c521b36a…`):
  `make_rollout(n)`, BE $\Delta t=0.005$, 50 steps, first-order sign-upwind advection, Newton 8 + preconditioned
  BiCGStab, f64; IC `blob_ic(n, cx, cy, w, a)` from the case descriptor; output steps 10, 20, …, 50
  ($t_k=0.05k$). **Primary $n=257$** ($h=1/256$, nodes coincide with the scoring lattice, so no interpolation; the
  numerical viscosity $ah/2$ differs from the training grid's $h=1/255$ by 0.4 %). **Sensitivity $n=256$** (the
  training grid itself) on the target set, with bank and sines evaluated at that grid's nodes $x_i=i/255$.
- **Gates.** (G1) IC identity: $\lVert\text{blob\_ic}(257)-u_{\rm ref}(t_0)\rVert/\lVert u_{\rm ref}(t_0)\rVert
  \le10^{-12}$ on every case (else the generator is not the reference's problem; stop A1). (G2) generator's own
  bar: every step's relative Newton residual $\le10^{-8}$ and every field finite.
- **Cases.** Target set T6 = val32|19 plus the five other eligible dev6 + val32 cases with the largest bank-384
  floor on E0 references (rule computed in code, before any upwind run is looked at). Extended set X = every
  eligible dev6 + val32 case (38) + every eligible N0s case (8).
- **Statistic.** For a basis $B$: $\rho_B=f_B(u_{\rm ref})/f_B(u_{\rm upw})$; smearing excess
  $D_B=\rho_B/\rho_{S}$ with $S$ the Laplacian-sine basis of the **same size**. Computed at sizes 128 and 384 for
  the bank. $D>1$ means the basis represents the smeared field better than the sharp one by more than a
  data-agnostic basis of the same size does.
- **Controls (both at size 128, leave-one-case-out over X, built from the same six output times $t_0..t_5$ of the
  other 45 cases, 270 snapshots, top-128 left singular vectors of the lattice snapshot matrix).**
  - C-A1+ (must fire): POD of the **upwind** fields of the other cases. Required: median over T6 of
    $D_{\rm POD,upw}\ge2$.
  - C-A1− (must not fire): POD of the **E0 reference** fields of the other cases. Required: median over T6 of
    $D_{\rm POD,ref}\le1.25$.
  - Resolution: median over T6 of $D_{\rm POD,upw}/D_{\rm POD,ref}\ge2$.
- **Reading (pre-registered, descriptive).** With controls passed: bank **smearing-consistent** if median over T6
  of $D_{\rm bank}$ (size 384) $\ge2$ and $D_{\rm bank}>1$ on at least 5 of 6; **not smearing-consistent** if the
  median is $\le1.25$; otherwise **ambiguous**. The same statistic is reported on X (median, and Spearman of
  $D_{\rm bank}$ with $q=ah/(2\nu)$). Alternatives that remain open even when consistent (ideation): generator
  mismatch and BE time error; this is not a proof of the fidelity hypothesis — the Phase B fidelity arm is the test.
- **Also reported.** $e(u_{\rm upw},u_{\rm ref})$ per case (the upwind generator's own error against E0).

### A2. Wall curvature and sine-tail convergence of decoded fields (ideation F11 "before Step 1", F16)

*Question.* Do decoded ROM fields violate the Dirichlet compatibility that the true solution satisfies, so that a
boundary-compatible bank (Phase B arm c) could shrink their sine tail and quadrature need?

Burgers with $u=0$ on the wall gives $u_t=0$ and $u\,(u_x+u_y)=0$ there, hence $\Delta u=0$ on the wall, and since the
tangential second derivative vanishes, $\partial_n^2u=0$: the odd reflection of $u$ is $C^2$. A field with
$\partial_n^2v\ne0$ on a wall has an odd reflection with a jump in the second derivative, which slows sine-coefficient
decay.

- **Fields.** (i) decoded ROM fields $G_{384}W$ of the saved r = 384 LSPG-CN-R run
  (`runs/rom1024/output/qacc_rom1024/R_R384_LSPG_CNR_1.npz`, valid runs only, invalid counted); (ii) the bank
  projection $P_{384}u_{\rm ref}$ (span effect without dynamics); (iii) the E0 reference $u_{\rm ref}$.
  Cohorts: dev6 + val32 (in range, primary), N0s, C1, C2 (secondary).
- **Wall-curvature ratio.** $\kappa(v)=\dfrac{\big(\operatorname{mean}_{k,\ \text{wall nodes}}(\partial_n^2v)^2\big)^{1/2}}
  {\big(\operatorname{mean}_{k,\ \text{interior}}(\Delta_hv)^2\big)^{1/2}}$ over $t_1..t_5$, with
  $\partial_n^2v$ at each wall node (corners excluded) by the one-sided second-order stencil
  $(2v_0-5v_1+4v_2-v_3)/h^2$ and $\Delta_h$ the 5-point Laplacian, $h=1/256$. Because `decode257` zeroes the wall
  nodes, the raw bank wall value $\max|G_{384}W|_{\rm wall}|/\max|G_{384}W|$ is also reported (evaluated with the
  wall-zeroing switched off); a non-zero raw wall value would itself be an incompatibility.
- **Sine tail.** $\tau_K(v)=\max_k\lVert(I-S_K)v(t_k)\rVert/\lVert u_{\rm ref}(t_0)\rVert$ for
  $K\in\{1024,2304,4096,9216,16384,36864\}$, and the decay slope $\beta$ = least-squares slope of
  $\log\tau_K$ against $\log K$ over $K\in[2304,16384]$.
- **Controls.**
  - C-A2+ (must fire): $v_+=u_{\rm ref}+0.01\max|u_{\rm ref}(t_0)|\cdot16\,x(1-x)y(1-y)$ (zero on the walls,
    $\partial_n^2\ne0$). Required on every in-range case: $\kappa(v_+)\ge10\,\kappa(u_{\rm ref})$ and median
    $\beta(v_+)\ge\beta(u_{\rm ref})+0.5$ (slower decay; $\beta<0$).
  - C-A2− (must not fire): $S_{4096}u_{\rm ref}$ (exactly compatible: every sine has zero even derivatives at the
    walls). Required: median $\kappa(S_{4096}u_{\rm ref})\le2\,\kappa(u_{\rm ref})$ (the reference's own $\kappa$
    is FD truncation plus true content, both small).
- **Pre-registered decision for Phase B arm (c).** Run the boundary-compatible arm only if, on the in-range cases
  (dev6 + val32), **either** median $\kappa(G_{384}W)\ge5\times$ median $\kappa(u_{\rm ref})$ **or** median
  $\tau_{4096}(G_{384}W)/\tau_{4096}(u_{\rm ref})\ge3$ (the arm's own endpoint is a 3× tail reduction, which is
  impossible if the decoded tail is already the true solution's tail). Otherwise arm (c) is not run, the decision
  and numbers are recorded, and its budget is released. If A2's controls fail, arm (c) runs (no evidence either
  way) only if budget allows after arms (a) and (b).

### A3. F7 truncation screen (ideation F7, CPU part only)

*Question.* Can the full-field decode be replaced by a $K$-sine synthesis without materially changing the ROM's
error?

- Decode every saved coefficient output in `runs/rom1024/output/qacc_rom1024/R_*.npz` (nine configurations, r =
  128 … 512; primary `R384|LSPG_CNR_1`) on the 257² lattice: $v=G_rW$; truncated $v_K=S_K v$ (DST-I, keep the first
  $K$ Laplacian-ordered coefficients), $K\in\{2304,4096,9216\}$. Score $e(v,u_{\rm ref})$, $e(v_K,u_{\rm ref})$ on
  every eligible case with a valid run.
- **Stop rule (verbatim from ideation F7):** *Stop if the worst error rises by more than 0.1 percentage points at
  K = 9216.* Operationalised: $\Delta_{\rm worst}(K)=\max_c e(v_K)-\max_c e(v)$, evaluated (primary) over dev6 +
  val32 and (secondary) over all regime cohorts; also reported: $\max_c[e(v_K)-e(v)]$ (per-case rise).
- **Controls.** C-A3= (identity, must hold): $K=65025$ (all sines) gives $|e(v_K)-e(v)|\le10^{-12}$ on every case.
  C-A3+ (must fire): $K=16$ must trip the stop rule ($\Delta_{\rm worst}>0.001$). Reproduction: $e(v)$ equals the
  jcp-regimes per-case error (`analysis.json`) to $\le10^{-9}$ on every regime case.
- The F7 **timing gate** (GPU, ≤ 0.5 GPU-h) is not in this lane's brief; it is recorded as deferred to the
  coordinator, with the passing $K$ values.

### A4. F15 hybrid (sine-enriched) floors with controls

- Bases: $H$ = span(bank $G_{512}$ ∪ $S_{512}$) (1024 unknowns); $S_{1024}$ (equal-dimension control); $B_{512}$
  (largest available bank; unmatched baseline). Also, for replication of the ideation number on E0 references:
  span($G_{384}\cup S_{512}$).
- Floors $f(u_{\rm ref})$ on every eligible case of dev6, val32, N0, N0s, Ap, Am, C1, C2, B1, B2.
- **Stop rule (verbatim from ideation F15):** *Stop if the hybrid's C1 worst floor is not ≥ 2× below the sine
  control and below the bank-512 floor.* i.e. continue only if $\max_{C1}f_H\le\tfrac12\max_{C1}f_{S_{1024}}$
  **and** $\max_{C1}f_H<\max_{C1}f_{B_{512}}$.
- **Controls.** C-A4= (containment, must hold): $f_H\le\min(f_{B_{512}},f_{S_{512}})+10^{-12}$ at every case and
  time. C-A4− (must fail the rule): the same hybrid with the 512 sines of Laplacian rank 40 000 … 40 511 (content
  far above the fields' spectrum) in place of $S_{512}$ must be stopped by the rule.
- A pass says only that missing content can be added by sines at the floor level; online cost (1024 unknowns) and
  dynamics are not tested here.

### A5. Phase A deliverables

`a1_smear.py`, `a2_wall.py`, `a3_trunc.py`, `a4_hybrid.py` (each writes `checks/<name>.json` with sources, hashes,
per-case rows, control outcomes), `make_report.py` → `report.md` (interim), `EXPORT.json`, PNG plots in `plots/`.
Local CPU time estimate: A1 ≈ 46 cases × 20 s + sensitivity 6 × 20 s ≈ 20 min; A2–A4 ≈ 30 min. GPU-h: 0.

## 3. Phase B — single-change bank retrains (wave 2; nothing is submitted before the coordinator's "release Phase B")

### B0. Question and stop rules (verbatim from the brief)

Does one change to the bank's training recipe fix the steep-front failure, against a control retrain?

- (a) **data fidelity** — stop if the N0s floor is not ≥ 3× below the control's;
- (b) **steep / low-ν coverage** — stop if the C1 ROM median is not ≥ 2× lower;
- (c) **boundary-compatible bank** — stop if $m^*$ is not ≥ 2× smaller **and** the tail not ≥ 3× smaller.
- Every arm also stops if the in-range worst floor is more than 2× the control's.

Operationalisation (all on the fresh cohorts of §B3, eligible references only, r = 384 prefix):
"N0s floor" = median over fresh N0s cases of $f_{384}$; "C1 ROM median" = median over fresh C1 cases of $e$ of the
§B5 policy result (invalid runs counted at $e=\infty$); "$m^*$" = node count of the rule selected at R384 by the
§B4 procedure; "tail" = median over fresh in-range cases of $\tau_{4096}$ (A2) of the decoded §B5 ROM fields;
"in-range worst floor" = max over fresh in-range cases of $f_{384}$. "≥ k× below/lower/smaller" means arm ≤ control / k.
Arm (c) therefore continues if either of its two endpoints improves by its factor.

### B1. Banks

- **K0** archived incumbent (seed 0; deployed rule R384 gauss128). Unmatched historical baseline.
- **K1 = the control retrain**: the jcp-robust seed-1 retrain of the incumbent recipe
  (`jcp-robust/inputs/banks/s1/`, hashes in its `BANK.json`; procedure-selected R384 rule gauss160, jcp-robust E2).
  It is exactly "an incumbent-recipe retrain with a new seed", already trained and rule-selected, so it costs 0 GPU-h
  here. Every arm is trained with **seed 1** and otherwise the identical pipeline.
- **K2, K3**: jcp-robust seeds 2, 3 — the retraining-variation band. Floors are CPU-only and always reported; their ROM
  scoring is in the defer list. An arm's result is called **seed-robust** only if it meets its bar against each of K1–K3
  (reported, not gating; the stop rule is judged against K1).
- Pipeline: a copy of `jcp-robust/train2d/` (`run_seed.sh`: Stage A `smoothtrain.py` 300k steps, Stage B
  `sep_coeff_extract.py` + `sep_hfit_run.py`, rotation `make_rotation.py`) into `experiments/imp-fronts/train2d/`, with
  `PROVENANCE.json` updated for exactly the files each arm changes; the vendored-file guard of `run_seed.sh` stays on.
  Retrains are not bitwise reproducible; seeds, data fingerprints and output hashes are recorded.
- **Arm (a) fidelity.** Only change: the training-data generator. Every place the incumbent calls `make_rollout(256)`
  (both stages) receives instead the jcp-fom2d sine pseudo-spectral ETDRK4 solution ($N=128$ modes per axis,
  $\Delta t=0.001$; jcp-fom2d measured median 3.76e-4 %, worst 0.00251 % against E0 on dev6 + val32, conditional E0
  estimates) sampled at $t=0.005k$, $k=0..50$, by exact sine synthesis at the same 256-node grid. Same parameter draws,
  same state pick, same everything else.
- **Arm (b) coverage.** Only change: the viscosity range of `sample_params`. Draws of $c_x,c_y,w,a$ are unchanged
  (same RNG stream order); $\log\nu\sim U(\log\max(0.0016,a/1000),\log0.1)$, so $s=a/\nu\le1000$ (C2's range) and the
  in-range box is a subset. Same generator (upwind + BE at 256 nodes), same 576 trajectories, same state count.
- **Arm (c) boundary-compatible** (runs only if A2 decides "RUN arm (c)" **and** the certified projection fits the cap
  after (a), (b)). Only change: the output construction
  $u=\text{out\_scale}\,[m(x,y)-m(-x,y)-m(x,-y)+m(-x,-y)]$, $m$ the incumbent MLP on 2-periodic Fourier features
  $[\sin\pi Bx,\cos\pi Bx]$ with $B$ an integer matrix (frozen: $B=\operatorname{round}(2B_0)$ from the incumbent's
  seed-1 initial draw), replacing the envelope $16x(1-x)y(1-y)$. Freezing $B$ to integers is part of the single
  change (2-periodicity requires it) and is disclosed. Needs the JAX ROM path, the NumPy evaluator and the hand
  gradient to support the new form; a field-identity test (JAX vs NumPy values and gradients ≤ 1e-12 relative) gates it.
  If A2 decides "DO NOT RUN", arm (c) is recorded as not run and E10 inherits its question.

### B2. Gates and must-fail controls (each shown to fire in the smoke, §B9)

- C-B1 bank identity: decoding a bank's saved coefficients with another bank must give an error ≥ 10× the matched
  decode on every smoke case (catches bank mix-ups); the hash guard must reject a mismatched `BANK.json`.
- C-B2 stop-rule evaluator: unit tests with synthetic inputs on both sides of every bar (including ties at exactly
  the factor) must give the pre-registered stop / continue outcome.
- C-B3 fidelity data: on dev6, the arm-(a) generator's fields on the 257² lattice must be within 0.01 % (worst, project
  metric) of E0; the incumbent upwind generator's fields must **fail** the same bar (must-fail).
- C-B4 coverage data: the arm-(b) draws must have max $s\ge900$, all $s\le1000$, min $\nu\ge0.0016$; the incumbent
  draws must fail max $s\ge900$ (must-fail); every arm-(b) state finite with relative Newton residual ≤ 1e-8.
- C-B5 freshness: the fresh-cohort seed root differs from every earlier cohort's; no fresh IC's sha256 equals any t0
  sha256 in the E0 store or the jcp-regimes store; a known val32 IC injected into the check must be flagged (must-fail).
- C-B6 Q1 certificate: the r2acc in-job certificate (fresh-Jacobian stationarity γ, tight distance) is reused unchanged;
  its existing must-fail control (a perturbed solve flagged non-stationary) is re-run once in the smoke.
- G-B7: every job log contains `jax_backend=gpu`; f64; `JAX_DEFAULT_MATMUL_PRECISION=highest`.

### B3. Fresh cohorts (generated, hashed and committed before any Phase B job; all **test-once**)

`fcohorts.py`, families copied from jcp-regimes `rcohorts.py`, rows drawn from
`numpy.random.default_rng([0x494D5046, g, j])` (new seed root "IMPF"; $g$ = cohort index, $j$ = case index):

| cohort | $c_x,c_y$ | $w$ | $\nu$ | $a$ | cases |
|---|---|---|---|---|---|
| FI (in range) | U(.15,.85) | U(.05,.20) | logU(0.01, 0.1) | U(0.5, 2) | 12 |
| FN0s (stiff corner) | U(.15,.85) | U(.05,.20) | logU(0.01, 0.0126) | U(1.5, 2) | 8 |
| FC1 ($s\in[250,500]$) | U(.15,.85) | U(.05,.20) | $10^{U[-2.5,-2.15]}$ | U(250ν, min(2, 500ν)) | 12 |
| FC2 ($s\in[500,1000]$) | U(.15,.85) | U(.05,.20) | $10^{U[-2.8,-2.45]}$ | U(500ν, min(2, 1000ν)) | 8 |

(FC1 is drawn independently, not paired with an A− row.) Each fresh cohort is scored **once**, at the end; nothing is
selected on it. Development and debugging use dev6 + val32 (E0) and the already-opened jcp-regimes cohorts.
References: the jcp-regimes E0-style ST recipe (`refjob.py` mode `fd`, CN-R levels 2048² / 4096² / 8192² with
refinement, the same acceptance checks, tight-path allowance 0), labelled conditional; eligibility per case from the
recipe's outcome. Never the sealed or reserved-OOD cohorts.

### B4. Rule selection per new bank ($m^*$)

The jcp-robust E2 procedure, unchanged (`r2acc.py` config pattern `e2s1.json`): mesh 1024², 38 dev6 + val32 cases,
ladder fib4181 → fib6765 → gauss96 → fib10946 → gauss128 → gauss160 → gauss192, first rule qualified under both
LSPG-BE and LSPG-CN-R (Q1 ≤ 2.5e-4, Q2 against its successor ≤ 2.5e-4 with ρ ≤ 0.116, distance to gauss384
≤ 2.5e-4). K0's and K1's selections are taken from jcp-robust (not rerun). A bank with no qualified rule at R384 is
reported as such; its $m^*$ is "> gauss192".

### B5. ROM scoring on the fresh cohorts — the same adaptive rule and step policy for every bank

r = 384, LSPG-CN-R, mesh 1024², the bank's selected rule $m_0$. **Adaptive policy (identical for all banks):** per
case, run rule $m_j$ (start $j=0$) and its ladder successor; if the case fails Q2 (rollout distance > 2.5e-4 or
ρ > 0.116 on its own states), escalate $j\to j+1$ along gauss160 → gauss192 → gauss256 → gauss384, stop at the first
Q2-qualified rule or at gauss384 ("not Q2-qualified"). At that rule, run step factors 1, 2, 4 ($\Delta t$, $\Delta t/2$,
$\Delta t/4$); the scored field is the coarsest factor whose distance to the next finer one is ≤ 2.5e-4, else factor 4
("not step-converged"). Q1 is certified in-job (r2acc certificate) on every run. Reported per bank and cohort:
median / worst $e$, counts of Q2-unqualified, step-unconverged and Q1-failing cases, and the three labels
(numerical, role, frontier) kept separate. Each policy level is a separate r2acc run (one config per bank).

### B6. Floors on the fresh cohorts (CPU)

$f_{384}$ and $f_{512}$ for K0–K3 and every trained arm on every eligible fresh case; medians and worst per cohort.

### B7. E9 — out-of-regime indicator $s=a/\nu$

- Frozen before any fresh case is scored, on dev data (dev6 + val32 + the jcp-regimes cohorts, K0's deployed
  `R384|LSPG_CNR_1` errors from jcp-regimes `analysis.json`): threshold $\tau_s$ = 95th percentile of $s$ over the
  in-range cases dev6 + val32 + N0 (the ideation rule); targets A ($e>2\times$ the dev in-range median) and B ($e>1\,\%$).
- Scored once on the fresh cohorts with K0's and K1's §B5 errors: AUC, detection and false-alarm rates with one-sided
  95 % Clopper–Pearson bounds; per F12, fewer than 30 good or 20 bad cases under a target → "inconclusive" (expected
  with 40 fresh cases; reported as such, never as a pass).
- FOM-fallback cost: (flagged fraction) × (accuracy-matched FOM time) + (ROM time) per query, using the jcp-fom2d PS
  FOM and the ROM timings measured on the same GPU type in the same job (jcp-fom2d / jcp-budget timing records); the
  GPU type is stated next to every time; no cross-GPU comparison.
- Note carried forward: on this family $s$ is a known input, so the indicator is a safety feature, not a speed one.

### B8. GPU-hour projection (to be certified by the smoke; cap 15 for E8 + E9; lane concurrent-job cap 2)

| item | basis | GPU-h |
|---|---|---|
| arm (a) retrain | jcp-robust measured 4.18 GPU-h/seed (A100-80G); PS data generation added | 4.3 |
| arm (b) retrain | same; low-ν Newton may cost more (certified in smoke) | 4.3 |
| fresh references, 40 cases | jcp-regimes measured 2.9–8.5 min/case | 4.0 |
| rule selection (a), (b) | jcp-robust measured 0.45/bank | 0.9 |
| §B5 scoring K0, K1, (a), (b) | jcp-regimes rom1024 0.84 GPU-h for 158 cases × 9 configs; policy levels added | 0.6 |
| smoke | | 0.5 |
| **total** | | **14.6** |

Submit only if spent + certified projection ≤ 15. **Defer order** if the certified projection exceeds the cap:
(1) arm (c); (2) K2 / K3 ROM scoring; (3) FC2 8 → 4 cases; (4) step factor 4; (5) FI 12 → 8 cases. Each deferral is
recorded in `runs/GPU-LEDGER.json` and the report. Arm (c) also needs ≈ 4.8 GPU-h; it runs only if A2 says so and
(a), (b) leave room (they will not under this projection, so (c) is expected to be deferred to E10's design).

### B9. Smoke plan (before any production job; every must-fail control of §B2 shown to fire)

1. Local CPU: arm-(a) and arm-(b) generators on 2 trajectories each; C-B3, C-B4, C-B5, C-B2 locally.
2. One cluster smoke job (A100, ≤ 0.5 GPU-h): arm (b) full 576-trajectory data generation (timed: the cost
   certificate for low ν), Stage A 2 000 steps, Stage B tiny, rotation; r2acc on the smoke bank with 2 cases of
   jcp-regimes C1 (opened cohort); refjob on 1 jcp-regimes C2 case (reference timing); C-B1 and C-B6 fire.
3. The cost certificate (`resources.py` pattern) converts measured rates into the production projection.

### B10. E10 (only if E8 falls short)

If every E8 arm stops (or none meets its bar), write `E10-DESIGN.md` for a transport/shift-aware bank or adaptive
quadrature and time steps near fronts (F8, F13), Codex-audit it and hand it back design-only (no GPU). If an arm meets
its bar, write a one-paragraph note that E10 was not needed.

### B11. Deliverables

`report.md` (generated by `make_report.py`), `EXPORT.json`, PNG plots, `runs/GPU-LEDGER.json`, every audit in `audits/`.

---

## Amendment 1 (2026-10-21, after `audits/codex-design-1.md`; supersedes the items named, everything else stands)

Disclosure: before this amendment the A1 upwind generation (byte-identical incumbent generator, deterministic, no
verdict) was started, and A3/A4 were executed once as debugging runs of draft code; those outputs are discarded and
regenerated after the code audit. A4's debug output was seen (C1 worst floors); no A4 threshold or control is changed
below except by adding the guaranteed null control Codex requested, which cannot change the verdict.

**A1 (restated).** The two LOO POD controls are **empirical comparators**, not effects by construction; their
thresholds (D_pos ≥ 2, D_neg ≤ 1.25, ratio ≥ 2) stay frozen, and failure still makes A1 "INCONCLUSIVE (control
failed)". Added **constructed detector checks (must hold)** at both sizes 128 and 384: $B^{+}$ = orth(first $r-5$
Laplacian sines ∪ the case's own upwind fields $t_1..t_5$) must give $D\ge10$ on every T6 case, and $B^{-}$ = the same
with the case's own E0 fields must give $D\le0.1$. Floors are clipped below at $10^{-14}$ (relative) before any ratio;
a clipped floor is flagged. Orthonormalisation by SVD with relative singular-value cut $10^{-13}$. The $n=256$
sensitivity compares the E0 floor computed on the 257² lattice with the upwind floor computed on the 256-node grid
(each relative to its own $\lVert u(t_0)\rVert$, bases evaluated at that grid's nodes, **no interpolation**); it is a
sensitivity, not a primary statistic. T6 is enriched for large floors by design; A1 stays descriptive.

**A2 (replaced).** The FD-stencil wall curvature and its control are dropped (they fail on real data: 24/38).
- *Analytic wall curvature* (descriptive): for bank-decoded fields $v=G_{384}W$ and the projections $P_{384}u$,
  $\partial_n^2v$ on the 4 × 255 interior wall nodes by a central difference of the bank's **analytic gradient**
  (`npbank.features_grad`) at $\pm\delta$, $\delta=10^{-4}$, normalised by the RMS over the interior of the 5-point
  Laplacian of $u_{\rm ref}$: $\kappa_a(v)$. The true solution has $\kappa_a=0$ (the compatibility argument, confirmed by
  the audit). Identity checks (must hold): the same code reproduces the closed form
  $\partial_x^2[16x(1-x)y(1-y)]_{x=0}=-32y(1-y)$ to $10^{-6}$ relative, and gives $|\partial_n^2|\le10^{-8}$ (relative
  to the interior scale) on a sum of 32 random Laplacian sines.
- *Tail and its wall localisation* (decision quantities): $\tau_K$ as before with times fixed to $t_1..t_5$ and floors
  clipped at $10^{-14}$; additionally the **wall share** $\omega(v)$ = energy of the high-pass part $v-S_{4096}v$ within
  the 8-node band next to the walls divided by its total energy (max over $t_1..t_5$ of the per-time share is not
  used; the share is pooled over $t_1..t_5$). A uniformly spread high-pass field has $\omega\approx$ 0.12.
- *Empirical sensitivity controls (must behave on the in-range cases, checked at real data before any verdict)*:
  C-A2w+ the manufactured incompatibility $h_+=0.01\max|u_{\rm ref}(t_0)|\,16x(1-x)y(1-y)$ must have median
  $\omega(h_+)\ge0.5$; C-A2w− an interior bump $h_-=0.01\max|u_{\rm ref}(t_0)|\exp(-|x-x_c|^2/(2\cdot0.03^2))$ at the
  centre must have median $\omega(h_-)\le0.25$. If either fails, A2's decision is INCONCLUSIVE.
- *Decision (a budget-prioritisation heuristic, not an impossibility argument)*, on dev6 + val32 with the r = 384
  LSPG-CN-R decoded fields: **RUN** if median $\tau_{4096}(v)/\tau_{4096}(u_{\rm ref})\ge3$ **and** median
  $\omega(v-S_{4096}v)\ge0.5$; **DEFER** otherwise; **INCONCLUSIVE** if a control or identity check fails. RUN makes
  arm (c) eligible (it still needs budget, §B8); DEFER and INCONCLUSIVE both leave arm (c) unrun and hand its
  question to E10. An $m^*$-only benefit of arm (c) is possible and is not excluded by DEFER.

**A3 (restated).** Decision population frozen: configuration `R384|LSPG_CNR_1`, cohorts dev6 + val32. All other
configurations and scopes are descriptive. $K=16$ is an empirical stress test (reported, not a control). Added
**constructed fixture (must trip)**: for every in-range case, $u^\*=u_{\rm ref}+h$ with $h$ a fixed seeded combination of
sines of Laplacian rank 9217–20000 scaled to $\lVert h(t_k)\rVert=0.005\lVert u(t_0)\rVert$, and $v=u^\*$ scored against $u^\*$:
$\Delta_{\rm worst}(9216)$ must exceed 0.001. Identity $K=65025$ and reproduction stay. A screen pass is reported as
"CPU truncation screen passed; F7 timing gate deferred", never as an F7 success.

**A4 (restated).** C-A4− (high-frequency sines) becomes an **empirical comparator**. Added **guaranteed null
enrichment (must be stopped)**: $H_0$ = orth($G_{512}$ ∪ $G_{512}$) (duplicate columns, rank-revealing SVD, cut
$10^{-13}$) has span $B_{512}$, so $\max_{C1}f_{H_0}<\max_{C1}f_{B_{512}}$ is impossible beyond $10^{-12}$; the rule must
stop it. Strict "<" with a $10^{-12}$ tolerance against $B_{512}$.

**B0 (replaced operationalisation).** Unqualified or invalid evidence can never make an arm succeed.
- (a): N0s floor median over eligible fresh FN0s test cases; requires ≥ 5 eligible cases (else INCONCLUSIVE).
- (b): let $Q$ = the case is Q1- and Q2-qualified at the policy's terminal state (§B5). Arm median $\tilde e_b$ =
  median over eligible FC1 test cases with unqualified or invalid cases set to $+\infty$; control diagnostic median
  $\hat e_{K1}$ = median of the control's finite policy errors regardless of qualification (invalid = $+\infty$). Arm (b)
  **continues** iff $\tilde e_b<\infty$ and $\tilde e_b\le\hat e_{K1}/2$ and $\hat e_{K1}<\infty$; if $\hat e_{K1}=\infty$ the
  comparison is INCONCLUSIVE. Requires ≥ 6 eligible FC1 cases.
- (c): $m^*$ = node count of the smallest **certified tested** rule at R384 (§B4); if none is certified the $m^*$
  criterion fails ("no certified rule in the tested ladder"). Tail over FI test cases with valid finite policy runs
  (≥ 6 needed, else that criterion fails).
- Guard: in-range worst floor over eligible FI test cases (≥ 6 needed, else INCONCLUSIVE).
- The pairwise $2\varepsilon_{\rm ref}$ rule is applied per case and reported as resolved wins / ties / losses next to
  every median comparison; it does not change the medians.
- Outcomes per arm: CONTINUE, STOP, INCONCLUSIVE, NOT RUN.

**B1.** "Seed-robust" is withdrawn; the label is "passes against all three incumbent retrains (K1–K3)".
- (a) adapter: jcp-fom2d `psjax2d` ETDRK4, $N=128$ convention (modes $1..127$ per axis), $\Delta t=0.001$, state
  captured every 5 steps → 51 states at $t=0.005k$; synthesis at the 256-node grid $x_i=i/255$ by evaluating the sine
  series exactly; state $k=0$ is the incumbent's nodal Gaussian `blob_ic(256)` (unchanged), and the PS evolution
  starts from the PS solver's own projection of that IC. Applied to **every** generator call in Stage A and Stage B
  (576 seed-0 + 4 032 seed-1000 trajectories), draws and state indices preserved. The upwind residual gate R0 is
  replaced for this arm by: all states finite, and on all dev6 cases C-B3 (worst ≤ 0.01 % vs E0 on the 257² lattice)
  plus PS self-convergence ($N=128,\Delta t=0.001$ vs $N=192,\Delta t=0.0005$, distance ≤ 1e-4).
- (b): the viscosity transformation is applied to **both** populations (576 seed-0 and 4 032 seed-1000), state picks
  frozen as in the incumbent; finite fields and Newton residual ≤ 1e-8 certified on all states of both.
- (c): described as one *architecture intervention* (boundary-compatible output **and** frozen integer frequencies);
  it cannot isolate compatibility alone. Added tests: oddness and 2-periodicity in each variable, wall value ≤ 1e-12,
  analytic wall $\partial_n^2$ ≤ 1e-10 (relative), JAX vs NumPy values and gradients ≤ 1e-12. Initialisation,
  normalisation, optimiser masking of $B$ and the rotation step are frozen in its code before training.

**B2.** C-B1: deliberate checkpoint / rotation / hash mismatch must be rejected (mandatory, deterministic); wrong-bank
decode degradation is supplementary. C-B2 covers invalid, infinite, missing and inconclusive inputs. C-B3 runs on all
dev6 (aggregate worst). C-B4 on the complete parameter tables (CPU). **C-B5 (replaced):** freshness against an
explicit allow-list of descriptor tables only — the E0 dev6/val32 descriptors (via `refstore`), the jcp-regimes cohort
descriptors (`rcohorts`), the incumbent's training draws (seed 0 × 576, seed 1000 × 4 032) and each arm's draws;
canonical descriptor $(c_x,c_y,w,a,\nu)$ match with relative tolerance $10^{-9}$; within-fresh duplicates across
splits checked; no directory outside the allow-list is traversed; injected val32|0 descriptor must be flagged.
C-B6: fixture = one converged smoke run's final state perturbed by a seeded relative $10^{-3}$ coefficient change;
the in-job fresh-Jacobian γ must exceed $10^{-3}$ (shown in the smoke); the full common-brief Q1 contract (validity,
pooled non-stationary fraction ≤ 1 %, every-case tight distance ≤ 2.5e-4, tight/tighter ≤ 2.5e-5 on the prescribed
subset) is applied unchanged.

**B3 (replaced split).** Per fresh cohort, cases $j=0,1$ = **dev**, $j=2,3$ = **val** (both reference-free: used only
for plumbing, policy dry-runs and cost certification), $j\ge4$ = **test-once** (references built, scored once at the
end). Draw counts: FI 4 + 10 test, FN0s 4 + 8, FC1 4 + 10, FC2 4 + 6 (34 test references). Cohort index
$g$ = 0 FI, 1 FN0s, 2 FC1, 3 FC2; draw order within a row: $c_x,c_y,w,\nu$-variable, $a$-variable (as `rcohorts`).
Deterministic reduced subsets, if the deferral schedule needs them, are the lowest test indices. The reference
adapter allow-lists exactly the four fresh cohort names. The scoring policy (§B5) is frozen before any test case runs.

**B4 (restated).** Rule selection uses jcp-robust `select2d.py` **unchanged** (its target certification, exact
candidate→successor map, and three-state CERTIFIED / FAILED / UNDETERMINED semantics; an uncertified gauss384 target
gives only a diagnostic pick). $m^*$ = node count of the selected certified rule, reported as "smallest certified
tested", never as a bound. Departure from F11's Gauss 64²–192² search: the E2 mixed ladder is used, as for K1.

**B5 (replaced policy).** Driver factors are **1, 0.5, 0.25** (`r2acc` uses $dt=DT0\cdot$factor). Escalation chain =
the select2d ladder order after the bank's selected rule, extended by gauss256 (successor gauss384). State machine per
case, starting at (selected rule, factor 1): run rule and successor at the factor, and the rule at factor/2 (unless
factor = 0.25). If Q2 fails → next rule, same factor. Else if the step check fails (distance to factor/2 > 2.5e-4) →
same rule, factor/2, and Q2 is re-checked **at the new factor on its own reached states**. Terminal: both pass
(qualified; scored at that run), or rule chain exhausted ("not Q2-qualified", scored at the last run), or factor 0.25
with step check unavailable ("not step-converged", scored at that run). Q1 is certified on every run; a Q1 failure at
the terminal run makes the case unqualified. Implemented as rounds (≤ 4) of `r2acc` jobs whose per-case lists are
generated by `policy.py` from the previous round; every run, including rejected ones, is charged to the ledger. This is
discrete rollout refinement, not F13's embedded time controller.

**B7 (restated).** E9 is **descriptive**: with 34 test cases the F12 minimum (30 good + 20 bad) cannot be met, so it is
necessarily inconclusive under that rule. Target A uses one frozen absolute threshold (2 × K0's dev in-range median of
`R384|LSPG_CNR_1`) for every bank. Invalid runs count as bad and alarmed. Percentile = `numpy.percentile` linear.
Routing is reported both ways: ROM-then-fallback and direct-to-FOM for flagged queries (s is known before solving).
Fallback cost: one paired timing job (A100, same job, warm JIT, A–B–A): K1 policy query vs the accuracy-matched PS FOM
from jcp-fom2d at the in-range target, 0.2 GPU-h.

**B8 (replaced projection; certified after the smoke).**

| item | GPU-h |
|---|---|
| arm (a) retrain incl. PS data for 576 + 4 032 trajectories | 4.4 |
| arm (b) retrain incl. low-ν upwind data | 4.4 |
| fresh test references, 34 cases (E0-style, 6 min/case mean) | 3.4 |
| rule selection (a), (b) | 0.9 |
| §B5 policy rounds, K1 + (a) + (b), 34 test + 16 dev/val cases | 0.6 |
| E9 paired timing | 0.2 |
| smoke (all expensive phases at real dimensions, shortened iterations) | 0.8 |
| **total** | **14.7** |

K0 is not ROM-scored on fresh cohorts (not needed for any stop rule; floors only). Frozen alternative schedules, used in
order only if the certified projection plus spent exceeds 15: **S2** test references FI 8, FN0s 8, FC1 8, FC2 4 (28)
and E9 timing dropped; **S3** S2 with arm (a) not run (arm (b) addresses the brief's primary C1 failure); **S4** no
Phase B. Arm (c) is outside every schedule unless A2 says RUN and (a) or (b) is not run; it is otherwise deferred to E10.

**B9 (replaced smoke).** One cluster smoke job directory per branch (≤ 2 concurrent): (i) arm (a) data generation at
full size (576 + 4 032 PS trajectories) + Stage A 2 000 steps including its final full-spatial phase
(`--steps 2000`, `full_last` capped as in the script) + Stage B at full data, head fit capped + rotation; (ii) arm (b)
the same with the low-ν upwind generator; (iii) `r2acc` round 1 of the §B5 policy on the 2 FC1 dev cases with a forced
too-small rule (fib4181) to exercise escalation and step paths, plus C-B1 and C-B6; (iv) `refjob` on FC2 dev case
j = 0. Measured per-phase rates feed `resources.py` (jcp-speed3d pattern) with 1.25× contingency on training and 1.5×
on references.

---

## Amendment 2 (2026-10-21, after `audits/codex-design-2.md`; supersedes the items named)

**A1.** The own-field constructions become **empirical self-fit checks** (reported, not gating). Added an **exact D
fixture (must hold)** at $r=128$ and $384$: with $B$ = bank-$r$, $S$ = sine-$r$ orthonormal bases, $y$ = a fixed sine of
$S$ (Laplacian rank 5) and $z$ a seeded unit vector orthogonal to span($B\cup S$), $c=\lVert(I-P_B)y\rVert$:
$u_{\rm upw}^{\rm syn}(t_k)=\alpha z$, $u_{\rm ref}^{\rm syn}(t_k)=\alpha z+\gamma y$ ($k=1..5$; $t_0$ any fixed field) gives
$D=\sqrt{1+\gamma^2c^2/\alpha^2}$ exactly; with $\gamma$ chosen so that $D=10$, and the swapped construction $D=0.1$,
the code's $D$ must match to $10^{-10}$ relative. The gating A1 controls remain the two empirical LOO POD comparators
with their frozen thresholds.

**A2.** The curvature is a **gradient-difference estimate** (not analytic): computed at $\delta=10^{-4},\,5\cdot10^{-5},\,
2.5\cdot10^{-5}$; the value at $2.5\cdot10^{-5}$ is reported if its relative change from $5\cdot10^{-5}$ is ≤ $10^{-3}$, else
"unconverged" (descriptive only; not used in the decision). The identity fixtures are frozen as coded (envelope closed
form; 32 sines drawn by `default_rng(7)` among Laplacian ranks < 2000, normal coefficients; denominator = RMS interior
5-point Laplacian of that sine sum; tolerance $10^{-8}$) plus the independent bank-values path (second difference of
bank values at $\pm10^{-3}$, agreement ≤ $10^{-3}$ relative). $\omega$ is **unavailable** for a field whose high-pass
energy is below $10^{-24}$ of its total energy over $t_1..t_5$; medians are over available cases, and if fewer than
half the in-range cases are available the decision is INCONCLUSIVE. The controls use the perturbations $h_\pm$ alone
(not $u_{\rm ref}+h_\pm$).

**B0 (replaced again).** Diagnostic (∞-penalised) medians and **qualified success** are separate. An arm can CONTINUE
on a ROM endpoint only if its terminal policy runs are numerically qualified (Q1 ∧ Q2 ∧ step-converged, §B5) on
**every** eligible test case of the endpoint's cohort (the configuration-level contract); the medians compared are then
the arm's errors against the control's diagnostic finite median.
- (b) CONTINUE iff the arm is qualified on every eligible FC1 test case, $\hat e_{K1}<\infty$, and median
  $e_b\le\hat e_{K1}/2$. Not qualified on every case → STOP (endpoint not certified). $\hat e_{K1}=\infty$ → INCONCLUSIVE.
- (c) $m^*$ route: CONTINUE-eligible iff the arm has a certified selected rule and $m^*_c\le m^*_{K1}/2$ ($m^*_{K1}$ =
  gauss160, from jcp-robust). Tail route: CONTINUE-eligible iff the arm is qualified on every eligible FI test case,
  K1 has ≥ 6 valid FI policy runs, and median $\tau_{4096}$(arm) ≤ median $\tau_{4096}$(K1) / 3. Either route suffices.
- (a) floor endpoint: no ROM qualification needed (bank-only); ≥ 5 eligible FN0s test cases.
- **Precedence per arm:** (1) NOT RUN; (2) INCONCLUSIVE if a population minimum (FN0s ≥ 5, FC1 ≥ 6, FI ≥ 6 eligible)
  or a control prerequisite is unavailable; (3) STOP (guard) if in-range worst floor > 2 × K1's; (4) STOP (endpoint)
  if the endpoint bar is not met or not certified; (5) CONTINUE.

**B1(c).** Arm (c) is expected to be deferred (B8). If A2 says RUN and a schedule funds it, an amendment registering
initialisation, normalisation, $B$ masking, the rotation step and the derivative-test denominators (relative to the
RMS interior Laplacian of the decoded field, zero-scale → test unavailable) precedes any arm-(c) code.

**B2.** C-B6 fixture frozen: from one converged smoke LSPG-CN-R step, perturb the coefficients along the top right
singular vector of the fresh Jacobian with magnitude giving $\lVert J\delta\rVert\ge100\lVert r^\*\rVert$; then
$\gamma\approx\sigma_1/\lVert J\rVert_F\ge1/\sqrt{384}\gg10^{-3}$; the certificate must flag it (shown in the smoke).
Tight/tighter anchors for the fresh populations: FI test $j=4,5$ and FC1 test $j=4,5$ (non-empty, frozen).

**B3.** Development reference permitted and budgeted: FC2 dev $j=0$ and FC1 dev $j=0$ get E0-style references in the
smoke (reference-path cost certification and adapter test); they are development data, never scored with the test
split. Counts (schedule S1 below): draws FI 12, FN0s 12, FC1 12, FC2 8 (dev $j=0,1$; val $j=2,3$; test $j\ge4$) → test
FI 8, FN0s 8, FC1 8, FC2 4 (28 test references).

**B5 (replaced state machine).** State $(j,f)$: rule index $j$ on the bank's escalation chain (select2d ladder order
after the selected rule, then gauss256 with successor gauss384), factor $f\in\{1,0.5,0.25\}$. Round $R(j,f)$ runs
prod$(j,f)$, succ$(j,f)$ = rule $j+1$ at $f$, and half$(j,f/2)$ if $f>0.25$ (reused as prod$(j,f/2)$ later). Transitions:

| condition at $(j,f)$ | next |
|---|---|
| prod invalid (non-finite, damping saturation, solver failure) | terminal INVALID, $e=\infty$ |
| succ invalid / unavailable, or Q2 fails | $(j+1,f)$ if rule $j+1$ has a successor; else terminal Q2-FAIL (scored at prod) |
| Q2 passes, $f=0.25$ | terminal STEP-UNCONVERGED (no finer check; scored at prod) |
| Q2 passes, half invalid | terminal STEP-UNAVAILABLE (scored at prod) |
| Q2 passes, step distance ≤ 2.5e-4 | terminal CONVERGED (scored at prod) |
| Q2 passes, step distance > 2.5e-4 | $(j,f/2)$ |
| no selected (certified) start rule | start at the chain's first rule (fib4181), labelled "uncertified start" |
| round 10 reached | terminal ROUND-CAP (scored at the latest valid prod) |

The longest path is (chain length ≤ 8) + 2 halvings = 10 rounds, so the cap is never binding for a complete path and
the whole path is funded. A case is **numerically qualified** iff terminal CONVERGED and its terminal prod passes the
Q1 certificate. Labels are reported separately (Q1, Q2, step). A Q2 check is "on own reached states" because succ is
replayed from the prod run's own initial state at the same factor.

**B7.** E9 is scored on **K1**'s terminal policy errors (arms descriptive). Paired fallback timing is run only if the
certified projection leaves ≥ 0.3 GPU-h; it would time K1's policy queries and a PS FOM tuned to the declared target on
the flagged families (FC1, FC2) in one job; otherwise E9 reports flagged fractions only and says the cost is
unmeasured.

**B8 (replaced).** Schedule **S1** (baseline):

| item | GPU-h |
|---|---|
| arms (a), (b) retrains incl. data | 8.8 |
| test references 28 cases × 6 min | 2.8 |
| rule selection (a), (b) | 0.9 |
| §B5 policy, 3 banks, worst-case full paths on FC1 + FC2 (12 test + 8 dev/val), typical paths elsewhere | 1.2 |
| smoke incl. 2 dev references | 0.9 |
| **total** | **14.6** |

**S2:** S1 with test FC2 4 → 2 and FI 8 → 6 (−0.4). **S3:** S2 without arm (a) (−4.9). **S4:** no Phase B. Arm (c) is
in no schedule (deferred to E10) unless A2 says RUN and S3 applies, in which case an "S3c" schedule (S3 + arm (c) at
4.4 + 0.45 selection) is costed by an amendment before use. Spent smoke GPU-h are counted separately from the
remaining projection.

**B9 (replaced smoke inventory).**
- CPU, local (no GPU): C-B2 stop evaluator and `policy.py` unit tests with synthetic run records driving **every**
  transition and terminal state of the B5 table (injected failures); C-B4 on the complete parameter tables; C-B5
  freshness with the injected val32|0; A1–A4 fixtures.
- Cluster job SM-a (A100): arm (a) PS data for 576 + 4 032 trajectories (timed), C-B3 on all dev6 and PS
  self-convergence, Stage A 2 000 steps incl. its final full-spatial phase, Stage B on full data with the head fit
  capped, rotation; C-B1 hash-mismatch rejection.
- Cluster job SM-b (A100): arm (b) upwind low-ν data for 576 + 4 032 trajectories (timed, Newton residuals), Stage A
  2 000 steps, Stage B capped, rotation; then `r2acc` policy round 1 on 2 FC1 dev cases with the smoke bank and with
  K1 (real solves), the C-B6 perturbation fixture; `refjob` on FC1 dev $j=0$ and FC2 dev $j=0$.
- Certificate: measured per-phase rates × production sizes, contingency 1.25 (training), 1.5 (references and policy),
  written by `resources.py` to `runs/COST-CERT.json`; submission only if spent + certified ≤ 15.

---

## Amendment 3 (2026-10-21, after `audits/codex-design-3.md`; supersedes the items named)

**A1 fixture prerequisites.** Frozen: seed 424242, $t_0$ field = all-ones (nonzero), $\alpha=1$, $y$ = sine of Laplacian
rank 5, $z$ orthogonalised twice against orth($B\cup S$). Prerequisites asserted: $c\ge10^{-3}$ and all four floors
$>10\cdot$CLIP. If a prerequisite fails, the fixture **fails** (A1 controls fail → INCONCLUSIVE); no fallback.

**A2 comparisons.** All curvature numbers are already normalised by $L$ = RMS interior 5-point Laplacian of
$u_{\rm ref}$; a case with $L=0$ or non-finite is unavailable. Refinement test: $|\kappa_{2.5}-\kappa_{5}|\le
10^{-3}\max(|\kappa_5|,10^{-6})$. Bank-path check (labelled a **fixed-spacing discretisation check**, not refined):
$\max|a-b|\le10^{-3}\max|b|+10^{-6}L$ over the wall nodes and $t_1..t_5$. $\omega$: computed on the original field
$v$ from its own DST coefficients beyond rank 4096; **unavailable** if the total energy of $v$ over $t_1..t_5$ is zero or
non-finite, or its tail energy is below $10^{-24}$ of that total. The decision quantity is $\omega(v)$ (Amendment 1's
"$\omega(v-S_{4096}v)$" means exactly this). An unavailable control statistic ($\omega(h_\pm)$ median) → INCONCLUSIVE.

**B0 precedence (replaced).** Per arm, endpoints and the guard are evaluated separately as PASS / FAIL / UNAVAILABLE:

| step | rule |
|---|---|
| 1 | arm not trained → NOT RUN |
| 2 | guard evaluable (≥ 6 eligible FI test cases) and violated (in-range worst floor > 2 × K1's) → STOP (guard) |
| 3 | every endpoint route FAIL → STOP (endpoint) |
| 4 | guard PASS and at least one route PASS → CONTINUE |
| 5 | otherwise → INCONCLUSIVE |

Routes and their own prerequisites (missing prerequisite → that route UNAVAILABLE): (a) floor route: ≥ 5 eligible
FN0s test cases, K1 floors on the same cases. (b) ROM route: ≥ 6 eligible FC1 test cases, $\hat e_{K1}$ finite; PASS iff
the arm is numerically qualified on every eligible FC1 test case and median $e_b\le\hat e_{K1}/2$; not qualified on every
case → FAIL. (c) $m^*$ route: $m^*_{K1}$ = gauss160 (25 600 nodes); PASS iff the arm has a certified selected rule with
$m^*_c\le12\,800$; no certified rule → FAIL. (c) tail route: ≥ 6 eligible FI test cases with K1 valid runs; PASS iff
the arm is qualified on every eligible FI test case and median $\tau_{4096}$(arm) ≤ median $\tau_{4096}$(K1)/3.

**C-B6 fixture (replaced).** Frozen: the smoke K1 run on FC1 dev $j=0$, LSPG-CN-R, the accepted state of step 10.
Search, in order, $\epsilon\in\{10^{-3},10^{-2},10^{-1}\}\cdot\lVert c\rVert$, signs $+$ then $-$, along the top right
singular vector of the fresh Jacobian there; at each candidate recompute the **actual** nonlinear residual and a fresh
Jacobian and measure $\gamma$. The fixture is the first candidate with measured $\gamma>10^{-3}$, and the in-job
certificate code must flag it. If none of the six candidates has $\gamma>10^{-3}$, Q1 certification is blocked
(no production ROM job).

**B5 (repaired).** Two maps, both from `select2d.py`'s registered ladder, extended by gauss256 (Q2 successor gauss384):
`next_candidate` (ladder order: fib4181 → fib6765 → gauss96 → fib10946 → gauss128 → gauss160 → gauss192 → gauss256)
and `q2_successor` (the registered successor of each rule: fib4181→fib6765, fib6765→fib10946, gauss96→gauss128,
fib10946→fib17711, gauss128→gauss160, gauss160→gauss192, gauss192→gauss256, gauss256→gauss384). Q2 at $(j,f)$ uses
`q2_successor[j]` at the same factor, replayed from prod's own initial state, and checks **both** the rollout distance
(≤ 2.5e-4) and $\rho$ evaluated at prod's own reached states (≤ 0.116), as `r2acc` does. Check precedence at each
state: (1) prod missing / invalid / non-finite → INVALID; (2) Q2 evidence missing or non-finite, or Q2 fails →
escalate to `next_candidate`, or Q2-FAIL if none; (3) $f=0.25$ → STEP-UNCONVERGED; (4) half missing / invalid /
non-finite distance → STEP-UNAVAILABLE; (5) step distance ≤ 2.5e-4 → CONVERGED; (6) else $(j,f/2)$. Round 10 is
evaluated fully before the cap applies; the cap can then only label states left non-terminal (none, by the bound
1 + 7 + 2 = 10). The policy loop runs **inside one job per bank** (no per-round queueing), with a pre-registered
GPU-time ceiling of 0.4 h per bank: cases still non-terminal when it is reached get terminal BUDGET-CAP (unqualified,
scored at their latest valid prod), counted and reported.

**B8 (replaced): ceilings, not estimates.** Each production job has a hard time limit equal to its line below; work
not finished inside it gets the pre-registered budget-exhaustion outcome, so the cap cannot be exceeded:

| item | ceiling GPU-h | exhaustion outcome |
|---|---|---|
| arm (a) training (1 GPU, A100/H100) | 4.4 | arm NOT RUN |
| arm (b) training | 4.4 | arm NOT RUN |
| test references (one job, ≤ 2 GPUs) | 2.8 | unfinished cases ineligible (counted) |
| rule selection (a), (b) | 0.45 each | no certified rule → $m^*$ route FAIL; arm's ROM routes UNAVAILABLE |
| §B5 policy, K1 + (a) + (b) | 0.4 each | BUDGET-CAP cases (unqualified) |
| smoke (SM-a + SM-b) | 0.9 | Phase B not submitted until the smoke completes |
| **total** | **14.6** | |

The smoke's cost certificate (`resources.py`) projects every production item from measured rates: data generation
and Stage A/B per-step rates from SM-a/SM-b at full data size; selection and policy from the smoke `r2acc` runs scaled
by rule size (node count) and factor (step count) to the largest rule (gauss384 for tight/tighter targets) and the
finest factor; references from the two dev references and jcp-regimes' measured per-case maxima (8.5 min/case
A100, 5.4 H200) with refinement at the measured ratio; compilation per distinct (rule, factor) shape. If any
projected item exceeds its ceiling, the frozen schedules S2 / S3 / S4 apply in order (S3 saves 4.85 GPU-h). Smoke
GPU-h are counted once, in the smoke line.

---

## Amendment 4 (2026-10-21, after `audits/codex-design-4.md`, which has no WRONG items; supersedes the items named)

Note: the Codex `-o` file of audit 4 was not written; `audits/codex-design-4.md` is the final answer block copied
verbatim from its stdout log (disclosed in the file header).

**A2 bank-path check.** $a$ and $b$ are **raw** normal second derivatives (not normalised by $L$); the comparison is
the absolute infinity norm over the interior wall nodes and $t_1..t_5$: $\lVert a-b\rVert_\infty\le10^{-3}\lVert b\rVert_\infty
+10^{-6}L$.

**B0.** The guard and the (a) floor route use one common case set: eligible test cases where **both** the arm's and
K1's floors are finite; missing or non-finite evidence for a case removes it from the set; the minima (FI ≥ 6, FN0s ≥ 5)
apply to that common set, else UNAVAILABLE. The (c) tail route uses one common set: eligible FI test cases where both the
arm and K1 have valid finite terminal policy runs (≥ 6, else UNAVAILABLE); both medians are over that set, while the
arm's qualification requirement still covers every eligible FI test case. The (b) route's $\hat e_{K1}$ is over all
eligible FC1 test cases (invalid = ∞), as Amendment 2.

**B5 / B8 consistency.** If selection certifies no rule, the policy still runs from the chain's first rule (fib4181),
labelled "uncertified start", and its cases are evaluated and qualified per case exactly as any other; the $m^*$ route
is FAIL; the ROM routes are **not** made UNAVAILABLE (Amendment 3's B8 clause "arm's ROM routes UNAVAILABLE" is
withdrawn). The synthetic transition tests cover this start.

**B8 operational ceilings.** Each job's Slurm wall limit = (ceiling GPU-h) / (GPUs requested): references 2 GPUs →
1.4 h wall; all other jobs 1 GPU. Smoke split: SM-a 0.45, SM-b 0.45. Every driver runs a self-timer that stops new
work at 90 % of its wall limit and writes its partial result plus exhaustion records (the last 10 % is the
finalisation reserve); if a job is killed anyway, the collector writes the exhaustion records from the job's
persisted per-case outputs. Failed attempts and retries are charged to the same 15 GPU-h total in
`runs/GPU-LEDGER.json`; a retry is submitted only if spent + certified remaining ≤ 15.

**B9 cost inventory.** The certificate enumerates solve multiplicities per production item: for each bank's policy,
44 cases × path length ≤ 10 rounds × {prod, succ, half, tight} + tighter on the 4 anchors, at the rule and factor of
each round; per-solve rates are the smoke's measured rates at its largest measured rule and finest factor, scaled by
node count and step count, and multiplied by (solver iteration cap / measured mean iterations) as a conservative bound
on adaptive tight/tighter iterations. References: per-case bound = the maximum measured per-case time in the
jcp-regimes reference jobs that exercised refinement (their per-case `seconds` records), on the same GPU type, × 1.5.
Any item without a measured or recorded bound is marked **uncertified** and deferred under S2–S4. The job ceilings of
B8 hold the cap regardless of the certificate; the certificate predicts whether each item will complete.

---

## Amendment 5 (2026-10-21, after `audits/codex-design-5.md`; supersedes the items named)

**B0 guard and floor route (replaced Amendment 4's dropping rule).** Guard PASS requires finite floors for both the
arm and K1 on **every** eligible FI test case (≥ 6); any missing or non-finite floor makes the guard UNAVAILABLE,
except that an observed violation on the available cases still gives STOP (guard). Route (a) requires finite arm and
K1 floors on every eligible FN0s test case (≥ 5), else UNAVAILABLE. (Floors are CPU quantities computable for every
eligible reference, so UNAVAILABLE indicates a defect, which is reported.)

**B8 admission.** A job (first attempt or retry) is submitted only if spent GPU-h + the hard ceilings of every running
or pending job + the proposed job's ceiling + the ceilings of every remaining scheduled item ≤ 15. Otherwise the
ceilings are shortened (with their exhaustion outcomes) or work is deferred under S2–S4.

**B9 / B5 Q1 scope.** Full Q1 certification (fresh-Jacobian stationarity per step **and** the tight-solve distance) is
applied to every **prod** run (production candidate). Successor and half runs carry the in-job per-step
fresh-Jacobian certificate (validity and γ fraction) but no tight comparison; when a half run is promoted to prod at
$(j,f/2)$ its tight solve is run then. Tighter-anchor solves: one per anchor case (FI test 4, 5; FC1 test 4, 5) at its
terminal state. Solve inventory per visited state: prod + its tight (1 + 1), succ (1), half (1, if $f>0.25$); per
anchor: 1 tighter. The certificate's scaled timings are projections; the B8 ceilings and admission rule supply the
enforceable protection.

---

## Amendment 6 (2026-10-21, after the Phase-A code audit `audits/codex-code-A-1.md`; before any A2 verdict was read)

**A2 bank-values path check — respecified.** The audit found the registered fixed-spacing check fails on real data
(dev6|0: gap 234.6 vs bound 1.41). Diagnosis (CPU): at $dv=10^{-3}$ the second difference of bank **values** carries a
≈ 3 % truncation error from the bank's high-frequency Fourier content (both paths agree to ≈ 1e-4 relative at
$dv=10^{-4}$ and to ≈ 1e-6 at $10^{-5}$). The check was mis-specified, not the estimator. Replaced by: value second
difference at $dv\in\{10^{-4},5\cdot10^{-5}\}$; the check passes iff, at $dv=5\cdot10^{-5}$,
$\lVert a-b\rVert_\infty\le10^{-3}\lVert b\rVert_\infty+10^{-6}L$ with $b$ the gradient-difference estimate at
$\delta=2.5\cdot10^{-5}$, and the two value spacings agree to the same bound. Fixture identity frozen: case dev6|0,
times $t_1..t_5$. The original failure is reported in the A2 output.

**Phase A accounting (from the code audit, no threshold changes).** Tails are summed directly from the retained DST
coefficients (no cumulative subtraction). A2's availability denominator is the number of **eligible** in-range cases
(invalid ROM runs count as unavailable for ROM statistics; reference and projection statistics are still computed for
them). Every script records excluded-reference and invalid-run counts and the reference label ("E0 ST / E0-style ST;
ε_ref is a conditional Richardson estimate"); an empty population gives UNAVAILABLE, not an error. A1's $q$ uses
$h=1/256$ for the primary 257-node fields ($q_{\rm train}$ with $h=1/255$ is reported beside it). A1's G2 requires a
finite residual at every step of every run at both resolutions (a non-finite state propagates into that step's
residual norm, so finite residuals at all 50 steps imply all 51 states finite). A1 re-verifies the run manifest, the
embedded generator hash, the descriptors and current eligibility of T6 and X before analysis. A3's secondary verdict
uses the regime scope (not "everything") and inherits INCONCLUSIVE from failed controls; its reproduction check fails
if any scored regime case lacks the lane's recorded error. A4 orthonormalises unions by SVD (cut $10^{-13}$) and records
ranks and singular-value margins.

**Amendment 7 (2026-10-21, same session, before any A2 decision was computed).** Testing Amendment 6's value-path
check on dev6|0 showed the value second difference at $dv=10^{-4}$ still carries $O(dv^2)$ truncation (gap between
the two spacings 2.96 vs bound 1.41, while the $5\cdot10^{-5}$ value agrees with the gradient estimate: 0.50). Replaced:
the value path is Richardson-extrapolated, $a_R=(4a_{5\cdot10^{-5}}-a_{10^{-4}})/3$, and the check passes iff
$\lVert a_R-b\rVert_\infty\le10^{-3}\lVert b\rVert_\infty+10^{-6}L$. Scope change: $\kappa_a$ is descriptive and never
entered the arm-(c) decision, so the curvature identity and path checks now gate **only the reporting of $\kappa_a$**
(withheld if any fails); the decision's controls are C-A2w±, the $\omega$ availability rule and the tail computation.

**Amendment 8 (2026-10-21, after `audits/codex-design-6.md`, whose only open item was B0).** Early STOP (guard) on
incomplete evidence is allowed only when the observed arm maximum floor exceeds 2 × the **complete** eligible-population
K1 maximum (K1 floors are CPU quantities and must be complete for every eligible FI test case); otherwise incomplete
arm evidence gives guard UNAVAILABLE. The population minimum (FI ≥ 6) stays explicit.
