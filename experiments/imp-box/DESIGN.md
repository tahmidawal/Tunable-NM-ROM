# imp-box (E1, spectral pilot Part A′): family width × steepness on the box (pre-registration)

**Status:** pre-registration, 2026-10-22, before any code or run. Amendments are appended below, dated, never rewritten.
Brief: `main:reports/2026-10-21-improvement-campaign/campaign/briefs/imp-box.md` (binding with `COMMON-BRIEF.md`).
GPU-hour cap 87; lane concurrent-job cap 3; namespace `/cluster/tufts/paralab/tawal01/imp-box/`.

## 0. Why (what is known)

On the box a coarse sine pseudo-spectral FOM is faster than our reduced models at equal accuracy (campaign K2; 2D 4096²
$S = 0.05$–$0.71\times$). The spectral pilot (`worktrees/2026-10-20-spectral-pilot`, report §3, §6.3, PILOT, one CPU core)
found a POD–Galerkin stand-in 5–13× slower than the spectral FOM with the wide family ($W=0.4$), but 1.7–2.9× faster at
loose targets with a narrow family (post-hoc control AB). This lane tests family width $W$ × steepness, on GPU, with
tuned competitors and our real bank.

**Question.** On the unit box, for which family width $W$ and steepness level does any reduced model reach
$S = t_{\rm FOM}^\ast / t_{\rm ROM}^\ast \ge 1$ (and $\ge 2$) at matched median and worst error $\tau\in\{1,3,10\}\,\%$, against
the tuned sine spectral FOM, on one H200 in the same job?

## 1. Problem (pinned; follows the pilot's Part A)

$$u_t + u\,(u_x+u_y) = \nu\,\Delta u \ \text{ in } (0,1)^2,\qquad u=0 \text{ on the boundary},\qquad t\in[0,0.25],$$
outputs at $t_k = 0.05k$, $k=0,\dots,5$; $u_0 = a\exp(-|x-c|^2/(2w^2))$ (the pilot's IC; its wall value is
$\le 1.6\times10^{-8}a$ for every family below because $c_i\in[0.3,0.7]$ and $w\le0.05$).

**Families** $F(W)$, $W\in\{0.08, 0.16, 0.28, 0.40\}$: $c_x, c_y \sim U[0.5-W/2,\,0.5+W/2]$, $w\sim U[0.035,0.05]$,
$a\sim U[1,2]$. $W=0.4$ is the pilot's Part-A family exactly.

**Steepness levels** (paired $\nu$, factor 3 per level, pilot `LEVELS`): $\nu=\nu_{\rm lo}2^{u_\nu}$, $u_\nu\sim U[0,1]$,
$\nu_{\rm lo} = 0.06,\ 0.02,\ 0.02/3,\ 0.02/9$ for A1–A4; steepness $s=a/\nu$ spans 8–33, 25–100, 75–300, 225–900.

**Pairing.** Row $j$ of a split draws five base uniforms $(u_x,u_y,u_w,u_a,u_\nu)$ from `default_rng([seed, j])`;
a cell $(W,\text{A}k)$ maps them by $c = 0.5 + W(u-0.5)$, $w = 0.035+0.015u_w$, $a = 1+u_a$, $\nu=\nu_{\rm lo}2^{u_\nu}$.
Every cell therefore sees the same base rows; cells differ only in $W$ and $\nu_{\rm lo}$. Cases are the statistical unit
within a cell; cells are not pooled.

**Cells:** 4 widths × 4 levels = 16. Order of execution (brief): $W=0.4$ cells, then $W=0.28$, then the stop rule
(§9), then (if not stopped) $W=0.16$ and $W=0.08$.

## 2. Cohorts (split by seed before any run)

| split | seed | cases | use |
|---|---|---|---|
| train | 4 100 001 | 96 (second data size 384, deferred item 4) | POD, bank, ECSW training |
| train-c | 4 100 001 (same rows, $c$ fixed at $(0.5,0.5)$) | 96 | transport arm only (§4.5) |
| dev | 4 200 001 | 8 | Q1 tight/tighter anchor subset |
| val | 4 300 001 | 32 | all selection (ladders, finalists, timing on val[0:8]) |
| test | 4 400 001 | 32 | test-once per cell, opened once at the end with frozen finalists |

None of these seeds is a campaign seed (asserted). No E0 entry exists for these cohorts; no sealed or reserved cohort is
touched. Historical cohorts are not used.

## 3. References (spectral recipe of the pilot, conditional)

Sine pseudo-spectral ETDRK4 (E0 cross-check kernels `pscore.py`, byte-identical; 3/2 padding) at $N_{\rm ref}=128, 192,
384, 768$ for A1–A4 and $n_{\rm ref}$ = `ref_cfl_steps(N_ref, 2)` steps. Gates per cell on val32 and dev8 (lattice metric,
§5): space check ($3N_{\rm ref}/4$, same $n$) $\le 2\times10^{-5}$; time check ($2n$) $\le 2\times10^{-6}$; padding check
($M=2N$ vs $3N/2$) on the 4 steepest val cases $\le 2\times10^{-6}$. A cell whose gates fail is not scored (its
$N_{\rm ref}$ may be raised once by a dated amendment before any arm runs).

**Reference error estimate** (per case, labelled **conditional**): $\varepsilon_{\rm ref} = d_{\rm space} + d_{\rm time}$ (the
3/4-resolution and half-step differences; for an exponentially convergent space discretisation the 3/4 difference
over-estimates the $N_{\rm ref}$ error). Cell allowance $U = 2\max_{\rm case}\varepsilon_{\rm ref}$ (used by the three-way rule);
two arms tie on a case if they differ by $\le 2\varepsilon_{\rm ref}$. **Deviation (stated):** the E0 recipe is second-order
FD with Richardson; on the box at A3/A4 that recipe would need $h\le 1/2048$; the spectral recipe is stronger and is the one
the pilot used. Test references are generated with the others but not read before the test stage.

Training data: the same solver at $N_{\rm ref}$, 21 snapshots ($t = 0.0125k$), sine coefficients; checked on 4 training
trajectories at $3N_{\rm ref}/4$ (must be $\le 10^{-4}$; else labelled).

## 4. Arms (each tuned on val32 over its own knobs; one cell = one job = one GPU)

### 4.1 F-SPEC: tuned sine spectral FOM, fixed step
$N\in\{8,12,16,24,32,48,64,96,128,192,256\}$; ETDRK4 step ladder $n=5\cdot2^j$ until the trajectory change
$\delta_n\le10^{-4}$ or $n=20480$ (imp-geometry step rule). Query = IC sine coefficients by the separable quadrature table
(case-independent table offline; per-case Gaussian evaluations charged), the $\nu$-dependent ETDRK4 coefficients
(charged), all steps, synthesis to the scoring lattice (charged).

### 4.2 F-SPEC-AD: the same, adaptive step
Step-doubling ETDRK4 on dyadic levels $\Delta t_j = 0.05/2^j$, $j=0,\dots,J$ ($J$ = 9; output times are hit exactly).
Local error $e = \lVert v_{\Delta t} - v_{2\times\Delta t/2}\rVert/\lVert v\rVert$; accept the two-half-step state if $e\le{\rm tol}$,
else refine $j$; after an accept with $e\le{\rm tol}/32$ coarsen when the time is aligned with the coarser level.
Knobs: $N$ as 4.1, ${\rm tol}\in\{10^{-3},10^{-4},10^{-5},10^{-6},10^{-7}\}$. Coefficients of all levels are charged.

### 4.3 R-BANK: our bank, retrained per cell (cells $W\in\{0.4, 0.08\}$)
imp-geometry R-BANK (dn256b Stage-A recipe: Fourier features, $g$-MLP, $K=16$ head, $R=512$, 300k Adam steps; code
copied byte-identically, `PROVENANCE.json`) with the box envelope $16x(1-x)y(1-y)$ (the incumbent `bc_poly`), trained on
the cell's 96 training trajectories × 21 snapshots sampled on the $1/128$ (A1, A2) or $1/256$ (A3, A4) lattice; ordered
span by QR + SVD in the $L^2$ metric of the reference rule. Off-mesh rule: composite $q\times q$ Gauss on $k_q\times k_q$
square elements, $q\in\{4,6,8,10,12\}$ (+14 as the last successor), $k_q$ per level fixed in the smoke (smallest
$k_q\in\{4,8,16,32,64\}$ whose $q=12$ rule integrates the training snapshots' squared norm to $10^{-8}$ relative, against
Parseval). Size $r\in\{16,24,32,48,64,96,128,192,256,384\}$. Steppers (both, as the brief): **B-LSPG** = incumbent CN-R LSPG
(prod tolerance, test prefix $4r$, point form); **B-E4** = Galerkin ETDRK4 (imp-faststep arm 3a; tensor form precontracted,
and point form), step ladder from $n=10$ (includes $\Delta t=0.0125$, $n=20$).

### 4.4 R-PODG: the pilot's POD–Galerkin stand-in (tensor / grid ETDRK4)
POD ($R_{\max}=512$) of the training snapshots in sine-coefficient space, $L^2$ inner product (exact by Parseval). Modes
are sine series, so values and exact derivatives are evaluated at any rule point. On the shared reduced engine
(imp-geometry `rom.py`): Galerkin ETDRK4 in **tensor** form (exact precontraction of the rule) and in **point** form
("grid ETDRK4") on the same rule ladder as 4.3; sizes as 4.3; step ladder from $n=10$. **Deviation (stated):** the pilot
used its own sine-space code; here the same model is implemented on the shared engine (same POD span, same Galerkin
ETDRK4), so all reduced arms share one implementation.

### 4.5 R-TRANS: transport-aware (shifted-frame) POD–Galerkin (cells $W=0.4$ only)
Burgers in the interior is translation invariant. R-TRANS is a POD–Galerkin model trained on `train-c` (the same
$(w,a,\nu)$ rows with $c=(0.5,0.5)$). A query with centre $c$ solves the reduced model for the centred IC (shift
$\delta = c-(0.5,0.5)$ removed) and decodes $u(x,t)=\tilde u(x-\delta,t)$ by phase-shifted sine synthesis
($\sin k\pi(x-\delta)$, separable, charged). It ignores the walls (the odd-periodic extension stands in for them); its error
from wall interaction is measured, not modelled. Stepper Galerkin ETDRK4 tensor form; sizes as 4.3.

### 4.6 R-ECSW: POD + ECSW (imp-geometry / jcp-hr-baselines greedy NNLS, imp-faststep ETDRK4 variant)
The R-PODG span; ECSW node weights fitted on the PODG reference rule's points (imp-geometry `ecsw_fit`, supports
$\kappa r$, $\kappa\in\{2,4,8\}$) on projected training states of 12 trajectories × 11 snapshots; Galerkin ETDRK4 point form
on the selected nodes; IC projected on the parent rule (charged); step ladder from $n=10$.

## 5. Metric, matching and speed-up
- Scoring lattice $x_i = i/256$ (257² with walls). $e = \max_{k\ge1}\lVert F_k-R_k\rVert/\lVert R_0\rVert$ per case; median and
  worst over val32 (selection) or test32 (test-once).
- Three-way rule with the cell allowance $U$: meets if ${\rm stat}+U\le\tau$; misses if ${\rm stat}-U>\tau$; else unresolved.
- Finalists per (cell, arm, $\tau$, stat): the imp-geometry `time` stage rule: candidates = valid visited rows meeting $\tau$,
  ordered by a warm one-case screen time; the screen-fastest is re-run unbatched on val32, must still meet $\tau$ and pass
  Q1 (and Q2 for reduced arms), else the next one. The FOM finalist set is F-SPEC ∪ F-SPEC-AD.
- **Speed-up:** paired A–B–A triplets (FOM finalist, arm finalist, FOM finalist) on val[0:8] × 5 repetitions, warm JIT,
  burn-in 3, batch 1, f64, field contract; $S$ = median ratio; $S_{\rm arm} = \min$ over FOM finalists (F-SPEC and F-SPEC-AD
  finalists of that $(\tau,{\rm stat})$). The FOM ladder is finite, so every $S$ is an **upper bound on our speed-up**
  ("against the strongest tested competitor"). One H200 per cell job; times are never compared across GPUs.
- **Query-cost decomposition** per finalist: IC/initial projection, $\nu$-dependent setup, evolution, decode (each timed
  separately in the same job, median of 5); offline costs (data, POD, bank training, rules, tensors, ECSW fits) reported
  separately.
- **Crossing width** per (level, $\tau$, stat, arm): the bracket $[W_i, W_{i+1}]$ of measured widths across which $S$ goes
  from $\ge c$ to $<c$ ($c=1,2$), or "below / above the measured range"; censored cells (arm never meets $\tau$) are
  reported as such and count as "$S<c$".
- Equal-cost size $r_\times$ (bank): the largest $r$ in the ladder whose cheapest qualified bank configuration's screen time
  is $\le$ the FOM finalist's screen time for that $(\tau,{\rm stat})$; its projection floor = `rom.floor_err` (lattice
  least squares on the first $r_\times$ ordered columns) over val32, median and worst.

## 6. Qualification (COMMON-BRIEF §3)
- **Q1, B-LSPG:** imp-geometry `finalize.q1` (fresh-Jacobian $\gamma$ at every step, $\le1\%$ non-stationary pooled,
  distance to the tight solve $\le2.5\times10^{-4}$ every val case, tight/tighter $\le2.5\times10^{-5}$ on dev8, no damping
  saturation, no non-finite step).
- **Q1, ETDRK4 steppers (B-E4, PODG, TRANS, ECSW):** the imp-geometry A3.5 analogue certificate (no non-finite stage, no
  growth-bound violation, contour coefficients agree with the closed-form φ-functions to $10^{-10}$, update defect
  $\le10^{-9}$, certificate replay identical), **plus** distance to the same configuration at $4n$ steps $\le2.5\times10^{-4}$
  on every val case and $4n$ vs $8n$ $\le2.5\times10^{-5}$ on dev8 (the time-converged anchor of a non-Newton stepper).
- **Q1, FOMs:** validity (finite outputs); F-SPEC-AD additionally: no level-cap hit, no step-count cap hit.
- **Q2 (reduced arms):** rule $q$ vs its successor on the arm's own reached states: $\rho\le0.116$ and rollout distance
  $\le2.5\times10^{-4}$ on every case (`finalize.q2_bank`); tensor form: the tensor of rule $q$ vs successor; ECSW: the
  fitted rule vs the full parent rule (imp-geometry `ecsw_q2`).
- Labels kept separate: numerical (Q1+Q2), role (none pre-declared here beyond "finalist"), frontier.

## 7. Must-fail controls (each shown to fire in the real-size smoke; a control that cannot fail is a bug)
| id | fault injected | must |
|---|---|---|
| SW1a / SW1b / MF-REF | score fixtures (imp-geometry `score.selftest`) | pass / reject |
| MF-REF-real | real reference perturbed by $5\times$ the space bar | space gate rejects |
| MF-PAD | reference run with $M=N$ (no dealiasing) on the steepest val case | padding gate rejects |
| MF-IC | IC table at too few points (8 Gauss points) | IC check vs the fine table fails ($>10^{-8}$) |
| MF-ENV | bank features without envelope | wall values $>10^{-8}$ (with envelope $\le10^{-12}$) |
| MF-TENS | zeroed tensor slice / zeroed test column | G4 (tensor = point form, $\le10^{-10}$) fails |
| MF-Q1 | certificate on a perturbed iterate (LSPG) / perturbed ETDRK4 coefficient (E4) | Q1 fails |
| MF-Q2 | rule $q=2$ against its successor | Q2 fails |
| MF-ECSW | ECSW weights scaled by 1.5 | Q2 (vs parent rule) fails |
| MF-AD | adaptive stepper with the error estimate forced to 0 | misses the target its enabled version meets, or uses fewer steps by ≥4× with error ≥10× |
| MF-TRANS | transport arm with the shift removed ($\delta=0$) | worst val error ≥10× the shifted model's |
| MF-FLOOR | `floor_err` on a field in the span / a random field | $\le10^{-10}$ / $\ge0.5$ |
| T-id / T-dbl | timing harness | timed outputs identical; doubled query 0.9–1.1 |
| MF-LEAK | test stage without the committed frozen manifest | refuses |
| MF-STOP | stop-rule script on fixtures (all $S<0.5$; one $S=0.6$; censored arm) | fires / not fired / fires |

## 8. Pass / fail readouts (no "win" claim beyond what is stated)
Per cell, arm, $\tau$, stat: $S$ (upper bound), qualification labels, finalists, query-cost decomposition. Per level and
$\tau$: the crossing width for $c=1,2$. A cell result is "reduced model faster" only if $S\ge1$ for a numerically qualified
finalist; "≥2× faster" if $S\ge2$. Test-once results are reported next to val selection results, never used to select.

## 9. Stop rules (verbatim from the brief, then the operational reading)
> Stop the line if every reduced model has $S < 0.5$ at every target on the wide family (W ≥ 0.28 cells).
> Also stop the bank arm if its projection floor at its own equal-cost size $r_\times$ is above τ in every cell.
> Run the wide-family cells (W = 0.4, then 0.28) **first** so the stop rule can fire before the narrow cells spend.

**Operational.** (i) *Line stop* fires iff there is no (reduced arm $\in$ {B-LSPG, B-E4, PODG, TRANS, ECSW}, cell with
$W\ge0.28$, $\tau$, stat $\in$ {median, worst}) with a numerically qualified finalist and $S\ge0.5$; a reduced arm that never
meets $\tau$ in a cell has no $S$ there and cannot avert the stop. If it fires: the $W=0.16$ and $W=0.08$ cells are not run;
the test-once stage still runs on the wide cells' frozen finalists. (ii) *Bank stop* fires iff for every bank cell run
and every ($\tau$, stat) the bank's projection floor (stat over val32) at $r_\times(\tau,{\rm stat})$ is $>\tau$ (no $r_\times$
in the ladder = floor of the smallest $r$ = counted as above); if it fires, the bank is not run on the $W=0.08$ cells.
Both are evaluated by an audited script (`stoprule.py`) from the cell JSONs at the coordinator checkpoint.

## 10. Compute plan and GPU-hour estimate (planning; replaced by the smoke cost certificate before release)
Per cell job (H200, f64): references + data A1–A3 ≈ 0.2 h, A4 ≈ 1.5 h; bank training ≈ 0.8 h (W=0.4, 0.08); ladders
≈ 0.5–1.5 h; timing ≈ 0.3 h. Wide cells (8): ≈ 22 GPU-h; narrow (8): ≈ 18 GPU-h; smoke + memory pilot ≈ 3; test-once ≈ 3;
second data size ≈ 8. Total ≈ 54 GPU-h < 87. Submission only if spent + certified projection ≤ 87; defer order (brief):
transport arm → A4 → $W=0.16$ → second data size → bank on $W=0.08$.

## 11. Deliverables
`report.md` (generated by `make_report.py`), `EXPORT.json`, PNG plots ($S$ vs $W$ per level and $\tau$; cost
decomposition), `runs/GPU-LEDGER.json`, per-cell `final.json`, ladders, audits in `audits/`.

## Amendment A1 (2026-10-22, after Codex design audit 1 = NOT-READY; before any code or run)

Each item answers the audit item of the same number (`audits/codex-design-1.md`). Where A1 and §0–§11 differ, A1 governs.

### A1.1 Cohorts
Descriptor arrays of every split and cell are hashed (sha256) into `cohorts.json` at the first stage and asserted pairwise
disjoint across splits (G6). All retained cells are frozen (code commit, descriptor hashes, artifact hashes, finalists) and
committed before any test split is opened. Inference policy: cells are correlated through the shared base rows; any
uncertainty on a difference between cells resamples whole base rows jointly across the cells compared; no snapshot is a
unit. Interpretation: $W$ changes both translation variability and wall proximity (bumps with $c$ near 0.7 reach the far
walls by $t=0.25$); the readout is their combined effect. The IC is the Gaussian imposed in the interior with zero
boundary trace (mismatch $\le1.6\times10^{-8}a$, negligible at the targets).

### A1.2 References (contract)
**Contract basis.** COMMON-BRIEF §2 says the lane brief wins where they conflict; the lane brief names the spectral pilot's
references as the base and fixes "A4 needs $N_{\rm ref}\approx768$ on GPU", i.e. the spectral recipe. The reference is
therefore the pilot's spectral recipe, with these gates (all per case; every split; selection splits gated in the `ref`
stage, the test split gated only in the `test` stage, failing test cases reported unscorable):
- **Metric:** both the coefficient-space continuous $L^2$ difference (the pilot's `gate_diff`) and the lattice difference;
  each gate uses the larger.
- **Space sequence:** $N_{\rm ref}/2$, $3N_{\rm ref}/4$, $N_{\rm ref}$ at the reference step. $d_{3/4}$ = difference
  $3N/4$ vs $N$, $d_{1/2}$ = $N/2$ vs $N$. Gate: $d_{3/4}\le2\times10^{-5}$ **and** decay $d_{3/4}\le d_{1/2}/4$ (else the case
  is ineligible: excluded from scoring, counted, reported).
- **Time:** $2n$ vs $n$ at $N_{\rm ref}$: $d_t\le2\times10^{-6}$.
- **Padding:** $M=2N$ vs $3N/2$ at $N_{\rm ref}$: $d_p\le2\times10^{-6}$; on every dev/val case at A1–A3, on the 8 steepest val
  and all 8 dev cases at A4 (others take the cell maximum).
- **IC rule:** $N\le384$: 64 panels × 16 Gauss points; $N>384$: 256 × 32 (pilot `ic_panels`); every table is checked against
  a 512 × 32 table: IC coefficient difference $\le10^{-12}$ relative (G7).
- $\varepsilon_{\rm ref} = d_{3/4} + d_t + d_p$, labelled a **conditional estimate** (not a bound; it assumes the observed decay
  continues). $U = 2\max\varepsilon_{\rm ref}$ over eligible selection cases. No independent-family cross-check is run
  (stated limitation; the E0 second-order recipe would need $h\le1/2048$ at A3–A4).
- An estimator self-test (`refgate.selftest`): a synthetic geometric sequence is eligible; a non-decaying one is not.

### A1.3 Arms
**F-SPEC.** Step rule exactly: $n_0=5$, $n\to2n$; $\delta_n=\max_{\rm case}\max_{k\ge1}\lVert u_n(t_k)-u_{n/2}(t_k)\rVert_{\rm lat}/
\lVert R_0\rVert_{\rm lat}$; stop at the first valid $n$ with $\delta_n\le10^{-4}$, or $n>20480$, or two consecutive invalid
rungs after a valid one (`ladder.step_ladder`, unchanged). **Endpoint rule:** if for some ($\tau$, stat) no $N\le256$ meets
$\tau$, or the smallest feasible $N$ is the ladder maximum, $N=384$ is added in the same job; if still at the boundary the
FOM is labelled "censored at the ladder top" for that target. FOM coverage (rows run, censoring) is reported.

**F-SPEC-AD (complete).** Levels $j=0..J$, $J=9$, $\Delta t_j=0.05/2^j$; coefficients for $j=0..J+1$ computed in the query
(charged). Time is an integer tick count in units of $\Delta t_{J+1}$. Start $j_0=2$. An attempt at level $j$: big =
ETDRK4($v$, $\Delta t_j$); half = two ETDRK4 steps at $\Delta t_{j+1}$; $e=\lVert{\rm big}-{\rm half}\rVert_2/\max(\lVert{\rm
half}\rVert_2,10^{-30})$ in coefficient space (raw difference, no 1/15 factor: a conservative indicator); non-finite $e$
= reject. Accept iff $e\le{\rm tol}$: $v\leftarrow$ half, advance by $\Delta t_j$; then if $e\le{\rm tol}/32$, $j>0$ and the
tick is a multiple of $\Delta t_{j-1}$, set $j\leftarrow j-1$. Reject: if $j<J$, $j\leftarrow j+1$; at $j=J$ the half state is
accepted and a **cap-hit** is counted. Output times are multiples of every $\Delta t_j$, so they are hit exactly; states are
recorded when the tick reaches $k\cdot0.05$. Attempt cap 8192 (exhaustion = invalid run). A run with any cap-hit is invalid.
Charged: every trial (rejected and accepted), the controller, all coefficients.

**R-BANK.** Box adapter = lane `geom.py` (unit box, `bc_poly` envelope, pilot IC); `bank.py` byte-identical. Training seed
$7\,400\,000+10\,i_W+i_A$ (recorded with the artifact sha256). Training samples on the $1/256$ lattice (A1–A3; 65,025 points,
the full-batch phase uses all) and the $1/512$ lattice (A4; full-batch phase on the recipe's fixed 65,536-point subset).
Ordering on the frozen reference rule ($k_q$, $q=16$). Test-space policy is the code's: $M_t=\min(4r, R_{\rm eff})$, so
$M_t/r = 4$ for $r\le128$ and $<4$ above (reported per row); no extra test functions. LSPG tolerances are scoped to the
incumbent's (`prod`; `tight`/`tighter` for Q1 only), stated as a scope limit. $k_q$: smallest $k_q\in\{4,8,16,32,64\}$ such that,
on 64 training snapshots, the $q=12$ rule's squared norms match Parseval to $10^{-8}$ **and** the bank Gram matrix $M$ on
$q=12$ vs $q=16$ differs by $\le10^{-8}$ (relative Frobenius). Operator convergence on reached states is Q2.
Representation is reported: training reconstruction error, projection floors per $r$ (bank and POD) on val32.

**R-PODG.** New adapter `podsine.py`: POD by the method of snapshots on sine coefficients with $\langle u,v\rangle=\tfrac14\sum
b\,b'$ (Parseval, f64 Gram); modes evaluated with exact derivatives by separable sine/cosine synthesis on the rule and
lattice. Checks (gates, smoke and every cell): $\lVert M-I\rVert\le10^{-10}$ on the reference rule; $K$ vs the exact
$\tfrac{\pi^2}{4}\Phi^\top{\rm diag}(k^2+l^2)\Phi$ $\le10^{-10}$ relative; tensor form = point form on the same rule (G4,
$\le10^{-10}$); **G-PILOT:** at $r\in\{16,32\}$ on 10 states, the engine's reduced nonlinear vector equals an independent
sine-space evaluation (3/2-padded pseudo-spectral product projected onto the modes) to $10^{-8}$. Described as the pilot's
mathematical model with a changed implementation (performance limitations disclosed).

**R-TRANS** is labelled a **static translation approximation**. Rule, size and step ladders as R-PODG (tensor and point
forms). Query: centred descriptor $d'=(0.5,0.5,w,a,\nu)$ → IC projection on the rule → Galerkin ETDRK4 → $z(t_k)$; decode
$B_k=\Phi_r z_k$ (sine coefficients, $N_{\rm ref}-1$ per direction) and $u(x_i,y_j)=\sum B_{kl}\sin k\pi(x_i-\delta_x)\sin
l\pi(y_j-\delta_y)$ with the shifted tables built in the query (charged). Diagnostics: boundary defect (max over outputs of
the decoded trace on the four walls / $\lVert R_0\rVert_\infty$). Q1 on the centred dynamics; Q2 on its reached states; error
against the box reference.

**R-ECSW.** Parent rule = the R-PODG reference rule ($k_q$, $q=12$); fits on the projected states of 12 training
trajectories × 11 snapshots; supports $\kappa r$, $\kappa\in\{2,4,8\}$; fit budget 1800 s; saturation / failures recorded.
**ECSW Q2 (new, gated):** on ECSW's own reached states (val32, 6 outputs): $\rho = \lVert N_{\rm ecsw}(z)-N_{\rm parent}(z)\rVert/
\lVert N_{\rm parent}(z)\rVert\le0.116$ (max) and rollout distance ECSW vs the parent-rule PODG (same $r$, stepper, $n$)
$\le2.5\times10^{-4}$ every case; the parent rule must itself pass PODG Q2 against its successor.

### A1.4 Selection, timing, crossings, $r_\times$
- Screening: up to 6 candidates per (arm, $\tau$, stat), ordered by ladder screen time, re-screened on val[0:4] (median of 3
  warm calls per case, mean over the 4 cases); the fastest that, re-run unbatched on val32, still meets $\tau$ and passes
  Q1 (+Q2 for reduced arms) is finalist 1; the next passing one is finalist 2 (≤2 finalists).
- Timing: every query is warmed for ≥ 2 s of repeated calls and 3 burn-in calls before its first timed call; paired A–B–A
  triplets on val[0:8] × 5; raw times kept; T-id on every timed output; a compile counter over the timed region must stay
  0 (else the pair is invalid). Components (IC, setup, evolution, decode) are timed separately; their sum vs the fused query
  is reported; the fused query is authoritative.
- $S_{\rm arm}(\tau,{\rm stat}) = \max_{b}\min_{a} S(a,b)$ over arm finalists $b$ and FOM finalists $a$ (`finalize.maxmin`);
  undefined if either side has no finalist (reported as "arm censored", "FOM censored" or "missing", never as a number).
- Crossing width: reported only for arms measured at ≥ 2 widths (TRANS: not identifiable). Each observed transition between
  adjacent measured widths is listed (several transitions are listed as such). Censored or missing cells break the bracket
  and are listed, never imputed.
- $r_\times(\tau,{\rm stat})$: $t_F$ = the median paired time of the fastest FOM finalist for that ($\tau$, stat) in the same
  job; $t_B(r)$ = the screen time (val[0:4], same job) of the cheapest **valid** bank configuration with that $r$ (any stepper,
  form, rule, $n$; target feasibility not required, so the floor test is not circular). $r_\times$ = largest $r$ with $t_B(r)\le
  t_F$; if none, $r_\times=0$ and its floor is the zero-field error (measured).

### A1.5 Qualification (symmetric)
Q1 = **algebraic / update correctness**, required of every selectable configuration of every arm; time-discretisation error
is not part of Q1 for any arm (it is in the error against the reference; steps are a tuned knob). Hence the $4n$/$8n$ anchors
of §6 are **withdrawn**.
- B-LSPG: `finalize.q1` (γ with a fresh Jacobian every step ≤1 % pooled; tight solve distance ≤2.5e-4 every val case;
  tight/tighter ≤2.5e-5 on dev8; same time grid — these are algebraic-solve anchors).
- Every ETDRK4 reduced arm (B-E4, PODG, TRANS, ECSW): every stage finite; no growth-bound violation; contour coefficients
  vs closed-form φ-functions ≤1e-10; every stage and update equal to an independent transcription ≤1e-9 at every step;
  certificate replay identical ≤1e-12 (the non-Newton analogue of the stationarity certificate).
- F-SPEC and F-SPEC-AD: every output finite; the same contour-vs-closed-form coefficient check on the FOM's coefficients
  ≤1e-10 (every level for AD); AD: no cap-hit and no attempt-cap exhaustion.
- Q2 (every reduced arm): population = reached states at the 6 output times of all val cases; states with
  $\lVert N_{\rm succ}\rVert\le10^{-12}\max$ skipped (≤5 % allowed, else fail); ρ ≤0.116 and rollout distance ≤2.5e-4 every
  case; tensor form: nonlinear vectors (not tensor entries) of tensor($q$) vs the point form on the successor rule; a
  configuration whose successor rule is not built fails Q2. The selector gates every reduced arm on Q1+Q2 and every FOM on
  Q1; the test stage re-checks Q1 on test32.

### A1.6 Controls (calibrated fault injections are mandatory; physical ablations are diagnostics)
Mandatory, each must fire on real data in the smoke: MF-REF-real (5× bar perturbation); MF-PAD-chk (padding checker on the
reference plus a 3× bar perturbation); MF-IC (2 panels × 8 points vs the fine table: $>10^{-8}$); MF-ENV (no envelope: wall
values $>10^{-8}$; with envelope $\le10^{-12}$); MF-TENS (slice of the dominant test mode scaled by 1.01: G4 fails);
MF-Q1-LSPG (`perturb_cert` 1e-3: γ fails); MF-Q1-E4 (coefficient $E$ perturbed by $10^{-6}$ relative in the update path: the
defect certificate fails); MF-Q2 (successor weights ×1.3: ρ fails); MF-ECSW (weights ×1.5: ECSW Q2 fails); MF-AD-rej (first
trial forced to $e=10\,{\rm tol}$: it is rejected and the level refined); MF-AD-cap (tol $10^{-30}$, $J=3$: run invalid);
MF-TRANS-id (zero shift: shifted decoder = plain decoder ≤1e-12) and MF-TRANS-sign (shifted decoder of a known field vs its
direct evaluation at $x-\delta$ ≤1e-12; with $+\delta$ it must fail); MF-NAN (descriptor with $\nu=$ NaN: run invalid, not
selectable); MF-NU (two $\nu$ give different outputs in the same compiled query: $\nu$-dependent setup is inside the query);
MF-FROZEN (one changed byte in a frozen artifact: test stage refuses); MF-LEAK (no committed manifest: refuses); MF-QUALEX
(a candidate forced to fail Q1 is never a finalist); MF-NOMATCH (no FOM finalist: $S$ undefined); MF-STOP (stop script
fixtures incl. $S=0.5$ exactly → not fired, $S=0.4999$ → fires, missing → unresolved); T-id; T-dbl (0.9–1.1); SW1a/SW1b/MF-REF
(`score.selftest`); `refgate.selftest`; G-MMS (FOM and engine RHS of the field $\sin\pi x\sin\pi y$ vs the analytic
$-2\pi^2\nu b$ plus an independently computed nonlinear term, ≤1e-10).
Diagnostics (reported, not asserted): no padding, $q=2$ rules, no shift (TRANS), error estimate forced to 0 (AD).

### A1.7 Stop rules (three outcomes)
Evidence per (reduced arm $a$, wide cell $c$, $\tau$, stat): **match** (both $a$ and the FOM have numerically qualified
finalists, timing valid: $S$ measured); **arm-censored** ($a$'s ladder fully evaluated in $c$, no qualified row meets $\tau$,
FOM has a finalist); **missing** (job failed, timing invalid, reference unscorable, FOM censored, ladder incomplete).
Deferred arms/levels (defer order) are excluded from coverage and listed.
- *Line stop:* **NOT FIRED** iff some match has $S\ge0.5$. **FIRES** iff no match has $S\ge0.5$, the missing set over the
  required coverage (every non-deferred arm × every run wide cell × 3 τ × 2 stats) is empty, and at least one match exists;
  if there is no match at all and everything is arm-censored it FIRES labelled "by censoring only". Otherwise
  **UNRESOLVED** (with the missing list).
- *Bank stop* (over the bank cells run before the narrow cells, i.e. $W=0.4$): per (cell, τ, stat) the floor at $r_\times$
  (val32 statistic) is **above** if floor $-U>\tau$, **below** if floor $+U\le\tau$, else unresolved. FIRES iff every entry is
  above; NOT FIRED iff any is below; else UNRESOLVED. If it fires, the bank is not run on $W=0.08$.
- If the line stop FIRES, the narrow cells are not run; test-once still runs on the wide cells. UNRESOLVED goes to the
  coordinator with the missing list.

### A1.8 Budget and the second data size
The smoke measures, on one H200 in one job: an A4 reference case (all gates), a bank training throughput (steps/s at the
real recipe and A4 point count) and its full-batch phase, every arm's ladder-row cost at the largest $r$, an ECSW fit, Q1/Q2
replays, timing pairs and compile times; `resources.py` turns them into a per-cell certificate. Release requires spent +
committed (running/queued) + certified remaining + 15 % headroom ≤ 87. **Second data size** (defer item 4): the $W=0.4$ cells
at A2 and A3 only; PODG, ECSW and bank retrained on 384 training trajectories (seed 4 100 001, rows 0–383, a prefix-superset
of the 96) in a separate job per cell; reported as a data-sufficiency diagnostic with its own finalists and $S$; it does not
enter the stop rules. Deferrals propagate into coverage as "deferred", never as failures.

### A1.9 Other
- Finer-lattice diagnostic at A3/A4: each finalist's error on the $1/512$ lattice on val[0:8] (references synthesised there),
  and the FOM finalists' continuous coefficient-space error; reported beside the scored lattice error.
- One trained bank per cell: findings are about these fitted models, not retraining robustness (seeds and hashes recorded).
- Every status (invalid, unresolved, unqualified, censored, missing, deferred) is kept separate in the JSONs and the report.

## Amendment A2 (2026-10-22, after Codex design audit 2 = NOT-READY; before any code or run)

Answers `audits/codex-design-2.md` item by item. A2 governs where it differs from A1 or §0–§11.

### A2.1 Cohorts
Disjointness is asserted row by row (no row of one split `isclose` to a row of another, all five base uniforms), not by
array hashes. Every scored row keeps its base-row id; eligibility masks are stored per cell and split; cross-cell paired
comparisons use the rows eligible in every cell compared.

### A2.2 References
- **Contract (flagged to the coordinator).** COMMON-BRIEF §0 says the lane brief wins where they conflict; this lane reads its
  brief (spectral-pilot references as the base, "A4 needs $N_{\rm ref}\approx768$") as authorising the spectral conditional
  estimator of A1.2 in place of the second-order Richardson recipe of COMMON-BRIEF §2. This is a **labelled lane deviation**,
  stated in the report's status line and raised at the coordinator checkpoint; the lane does not claim it satisfies §2.
- **Populations.** Selection splits (dev8, val32) are gated in `ref`; the test split (test32) is gated in `test` with the same
  gates, and has its own allowance $U_{\rm test}$ and eligibility. Padding: every dev/val (and, in `test`, test) case at
  A1–A3; at A4 the 8 steepest cases plus every dev case per split, the split maximum extrapolated to the others
  (labelled "extrapolated").
- **Decay test with a round-off branch:** if $d_{1/2}\le10^{-11}$ the decay ratio is waived (eligible, labelled
  "round-off").
- **Failure policy (per split):** a case failing any gate is re-run once at $N=\lceil4N_{\rm ref}/3\rceil$ rounded to a multiple
  of 32 (all three space levels scaled, same gates). Still failing → ineligible, reported. If more than 2 cases of a split
  are ineligible the cell is **unscorable** on that split. $N_{\rm ref}=768$ at A4 is provisional until its gates pass.
- $U$ and $\varepsilon_{\rm ref}$ are conditional estimates, never certified bounds.

### A2.3 Rules and ranks (independent per representation and cell)
- POD rules: $k_q^{\rm pod}$ = smallest $k_q\in\{4,8,16,32,64\}$ such that (i) on 64 training snapshots the $q=12$ squared
  norms match Parseval to $10^{-8}$ and (ii) the POD Gram matrix $M$ (all $R_{\rm pod}$ modes) on $q=12$ vs $q=16$ differs by
  $\le10^{-8}$. Used by PODG, TRANS (its own POD, own $k_q$) and ECSW. Bank rules: $k_q^{\rm bank}$ by (i) and the bank Gram
  criterion, computed after training; used by the bank only.
- Named rules per representation: **reference/ordering rule** $(k_q, 16)$ (bank ordering; $M$/$K$ checks); **ladder rules**
  $q\in\{4,6,8,10,12\}$ on $k_q$ with successor $q+2$ (14 built for the successor of 12); **ECSW parent rule**
  $(k_q^{\rm pod},12)$, its successor $(k_q^{\rm pod},14)$.
- POD numerical rank: modes with Gram eigenvalue $\lambda_i\ge10^{-13}\lambda_1$ (f64), $R_{\rm pod}=\min(512,\#)$. Bank:
  $R_{\rm eff}$ from `bank.rotation` (rank tol $10^{-10}$). Ladder ranks above $R$ are skipped and logged ("rank absent").
- **G-PILOT** is a convergence comparison between the reference-rule engine and the 3/2-padded pseudo-spectral product
  (exact for band-limited modes); bar $10^{-8}$; a miss is reported and the reference rule's $k_q$ doubled once.

### A2.4 FOM search boundaries
For each ($\tau$, stat) the selected FOM finalist's knobs are inspected: $N$ at the ladder maximum → add $N=384$ (then 512)
in the same job; $n$ at the step cap → labelled "step-capped"; AD tol at $10^{-7}$ → add $10^{-8}$; AD level cap $J$ hit by
the selected run is impossible (invalid). The same extension applies to F-SPEC and F-SPEC-AD. A knob still binding after
its extension is labelled "search truncated at <knob>". Step ladders ending by two invalid rungs are labelled. AD: a
rejected trial at $j=J$ terminates the run immediately (invalid).

### A2.5 Finalist search coverage
Candidates are screened in ladder-screen-time order, in batches of 6, until a finalist pair is found or the visited rows
meeting $\tau$ are exhausted, with a cap of 18 qualification attempts per (arm, $\tau$, stat). Statuses: **finalist**;
**arm-censored** (no visited valid row meets $\tau$; the ladder was completed); **arm-unqualified** (rows meet $\tau$ but every
one fails Q1/Q2, all attempted); **search-incomplete** (cap reached: counted as missing). Ties are broken by (time, key).

### A2.6 Q1 / Q2 (completed)
- **Audited non-Newton substitution.** COMMON-BRIEF §3's tight/tighter anchors certify an algebraic solve. Explicit ETDRK4
  (FOM and reduced) has none; for every ETDRK4 configuration (F-SPEC, F-SPEC-AD at every accepted and rejected trial, B-E4,
  PODG, TRANS, ECSW) Q1 is: every stage finite; contour coefficients vs closed-form φ-functions ≤1e-10; every stage and the
  update equal to an independent transcription with the closed-form coefficients ≤1e-9 relative at every step;
  certificate replay identical ≤1e-12; for reduced arms no growth-bound violation; for AD no cap-hit. B-LSPG keeps
  `finalize.q1`, and its tight and tighter anchor solves are run **with the certificate** (γ, fresh Jacobian, every step,
  ≤1 % non-stationary pooled, no damping saturation, no non-finite step) — `finalize.py` is modified for this (recorded in
  `PROVENANCE.json`).
- **Test stage:** every frozen finalist re-checks Q1 and Q2 on test32 reached states; a failure is a failure of that frozen
  finalist (reported), never a trigger for reselection.

### A2.7 $r_\times$ (named, evaluation-cost contract of the pilot)
**$r_\times^{\rm eval}$** (used by the bank stop): $t_F$ = median of 20 warm calls of one FOM nonlinear evaluation
(`pscore.nonlinear`) at the FOM finalist's $N$ for that ($\tau$, stat); $t_b(r)$ = median of 20 warm calls of one bank
reduced nonlinear evaluation at $r$, the cheaper of the tensor form ($M_t=r$) and the point form on the ladder rule $q=12$;
same job, same statistic. $r_\times^{\rm eval}$ = largest ladder $r\le R_{\rm eff}$ with $t_b(r)\le t_F$. If none: "no
affordable tested rank" (reported separately; excluded from the bank-stop predicate). **$r_\times^{\rm query}$** (diagnostic
only): the same with whole-query screen times on val[0:4] for both sides (FOM finalist; cheapest valid bank configuration
per $r$, validity = `stats_valid`, numerical qualification not required, stated). "Upper bound" means: a stronger FOM can
only lower $S$; it is not a bound on the globally optimal ROM-vs-FOM ratio.

### A2.8 Stop rules (replace A1.7)
**Required set** (frozen now): FOM, PODG, ECSW on the 8 cells $W\in\{0.28,0.4\}$ × A1–A4; B-LSPG, B-E4, TRANS on the 4 cells
$W=0.4$; × $\tau\in\{0.01,0.03,0.1\}$ × stat ∈ {median, worst}. Entries removed only by the brief's defer order (listed).
Entry status: **match** ($S$ measured), arm-censored, arm-unqualified, FOM-censored, missing.
- **SPEED-STOP (the brief's criterion):** NOT FIRED iff some match has $S\ge0.5$. FIRES iff no match has $S\ge0.5$, no entry
  is missing or FOM-censored, and **every arm has at least one match** (non-vacuity). Otherwise UNRESOLVED (with the
  list). Censored / unqualified entries are not speed evidence and are listed.
- **Scoped reading (reported beside it, not a substitute):** the same predicate restricted to arms with ≥1 match, and the
  list of arms censored or unqualified at every required entry ("infeasible at every target on the wide cells"). Whether
  that infeasibility also stops the line is the coordinator's decision at the checkpoint (not pre-authorised here).
- **BANK-STOP:** entries (cell $W=0.4$, $\tau$, stat) with a defined $r_\times^{\rm eval}$: floor (val32 statistic, `rom.floor_err`
  on the ordered span) **above** if floor $-U>\tau$, **below** if floor $+U\le\tau$, else unresolved. FIRES iff at least one
  entry is defined and every defined entry is above; NOT FIRED iff any is below; else UNRESOLVED. "No affordable tested
  rank" entries are listed separately.

### A2.9 Controls (calibration made explicit)
- Each fault injection is applied to a specified real state (smoke cell, val[0]); the control is accepted only if the
  measured effect exceeds the checker threshold by ≥10× (the magnitude is raised ×10 at most 3 times, recorded); each
  checker's **positive control** (unperturbed configuration passes) is recorded beside it.
- MF-NU is replaced by **MF-SETUP**: query builders take only case-independent inputs (asserted: the compiled query's
  closed-over constants are fixed before any descriptor is seen); a deliberately leaky variant with coefficients frozen at
  $\nu_0$ must differ from the official query at $\nu\ne\nu_0$ by $>10^{-6}$ (detected), and timing uses only official queries.
- **MF-NAN-stage**: a NaN injected into the second ETDRK4 stage at step 1 (FOM and reduced; hook added to `rom.py` and the
  lane FOM) makes the run invalid and unselectable. MF-AD-cap uses tol $10^{-20}$ with $J=3$.
- Selector fixtures: six failing candidates then a qualifying seventh → selected; a forced test-Q2 failure → reported, no
  reselection; partial reference eligibility → masks propagate.

### A2.10 Resource certificate scope
The smoke job measures and `resources.py` certifies, per cell type: reference generation (all gates and populations,
including `train-c`), training data, POD Gram/eigensolve, rule construction and $k_q$ search, bank training (steps/s, full
phase), bank ordering and feature derivatives, tensor construction (largest $r$, $M_t$), ECSW fits, every arm's ladder
rows (all ranks, forms, rules, steps — from per-row timings at two ranks, extrapolated by the measured exponent, labelled),
compilation, Q1/Q2 searches, timing, test stage, host/device memory peaks; plus 15 % headroom and one failed-job
allowance per wave. The second data size covers only two cells (A2, A3 at $W=0.4$): a stated scope reduction from the
pilot's per-cell pair.

### A2.11 Reporting
Eligibility masks; test-once accuracy and test re-timed $S$ reported separately from val-selected $S$; finer-lattice
disagreement qualifies any field-accuracy statement (the 257² lattice stays authoritative); timing outliers (>20 % above
the median) counted; provenance (code commit, artifact sha256) per result.

## Amendment A3 (2026-10-22, after Codex design audit 3 = NOT-READY; before any code or run)

Answers `audits/codex-design-3.md`. A3 governs where it differs.

### A3.1 References and training data
- Round-off branch: decay is waived only if **both** $d_{1/2}\le10^{-11}$ and $d_{3/4}\le10^{-11}$.
- Training gates (`train` and `train-c` alike): rows 0–3 re-run at $3N_{\rm ref}/4$ and at $2n$; the 21-snapshot difference
  (coefficient and lattice metrics) must be $\le10^{-4}$. A failure labels that training set "unqualified data" (reported
  next to every result of the arms trained on it); arms still run.
- Every statistic computed after reference exclusions is labelled "conditional on the eligible subset (k of n)" at the
  claim, including worst-error statistics and stop-rule entries.

### A3.2 Operator check (replaces G-PILOT and G-MMS of A1/A2)
**G-OPS:** basis = the six sine modes $(1,1),(2,1),(1,2),(2,2),(3,1),(1,3)$, orthonormalised; 10 random states $z$. The
exact Galerkin vector $\int\psi_k\,u\,(u_x+u_y)$ and the exact $M$, $K$ are computed from 1-D integrals of products of
sines/cosines by a 200-point Gauss–Legendre rule on $[0,1]$ (exact to round-off for these trigonometric degrees;
independent of the engine's composite 2-D rule). Gate: the engine (point and tensor forms on the reference rule) matches
to $\le10^{-10}$; the FOM's `pscore.nonlinear` ($N=8$, $M=12$) is compared with the same exact coefficients and its
difference is **reported** (a finite pseudo-spectral quadrature difference, not gated). Also gated: the FOM linear part
$-\nu\pi^2(k^2+l^2)$ and the engine's $K$ vs exact $\le10^{-12}$. A failure of a gated item blocks the release.

### A3.3 Bank stop and $r_\times^{\rm eval}$ (replace the A2.7/A2.8 bank parts)
- Comparator: for each ($\tau$, stat) the FOM finalist with the smallest median paired time in the same job (ties by key);
  $t_F$ = its one nonlinear evaluation at its $N$.
- $t_b(r)$ = the minimum over every selectable bank evaluation form at $r$: tensor form for Galerkin ($M_t=r$), tensor form
  for LSPG ($M_t=\min(4r,R_{\rm eff})$, see A3.6), point form on each ladder rule $q\in\{4,\dots,12\}$ (with the LSPG and the
  Galerkin test spaces). Kernel inputs: the projected IC state of val[0] (bank) and the IC coefficients of val[0] (FOM);
  5 burn-in calls immediately before each block of 20 timed calls; median.
- **BANK-STOP:** required entries = 4 cells $W=0.4$ × 3 τ × 2 stats (minus deferrals). NOT FIRED iff any entry is below.
  FIRES iff **every** required entry has a defined $r_\times^{\rm eval}$ and is above. Otherwise UNRESOLVED ("no affordable
  tested rank" entries listed; a separate stop on affordability is the coordinator's decision).

### A3.4 ECSW details and Q2 validity
Training rows 0–11 of `train`, snapshots 0, 2, …, 20 (11 per row); candidates = all parent-rule points if $m\le32768$, else
the imp-geometry importance sample (32768 points, seed $7\,199\,001+r$). **Every Q2 path** (bank, PODG, TRANS, ECSW) requires
all compared rollouts (configuration, successor/parent) to be valid (`stats_valid`) and computes ρ on the configuration's
own reached states.

### A3.5 Controls (calibration criteria per control; replaces the uniform 10× of A2.9)
- Calibrated numerical faults must move the checker's statistic past its threshold by a factor ≥ 2 on the specified real
  state (val[0] of the smoke cell): MF-REF-real, MF-PAD-chk, MF-IC, MF-ENV, MF-TENS, MF-Q1-LSPG, MF-Q1-E4, MF-ECSW, and
  **MF-Q2** = rule-$q$ weights ×1.5 (ρ = 1/3, 2.9× the bar).
- Boolean controls fire iff the decision flips: MF-NAN-stage, MF-AD-rej, MF-QUALEX, MF-NOMATCH, MF-FROZEN, MF-LEAK,
  MF-STOP fixtures, selector fixtures. **MF-AD-cap** (deterministic): the error estimate is forced to $+\infty$ for every trial
  at $j=J$ → the run must be invalid.
- **Setup placement (structural, MF-SETUP):** each timed callable is `jax.jit` of a function of the descriptor $d$ alone,
  built from case-independent arrays; $\nu$ enters only through $d$, so all $\nu$-dependent work is inside the compiled,
  timed computation (asserted: the builder never receives a descriptor). The leaky-variant detection of A2.9 is kept.

### A3.6 Arms: small completions
- F-SPEC/F-SPEC-AD: if **no** $N\le256$ meets $\tau$ for some ($\tau$, stat), $N=384$ then 512 are added (A1's rule kept);
  termination reasons (cap, invalid rungs, convergence) are recorded per step ladder.
- B-LSPG gains the **tensor form** (precontracted on the ladder rule $q=12$, $M_t=\min(4r,R_{\rm eff})$) as a knob beside
  the point form on the rule ladder; the cheaper charged is selected by timing as for every knob.
- A single qualifying finalist at search exhaustion is a valid finalist set of one.
- Test stage (B-LSPG): tight/tighter anchors on test[0:8], tight-solve distance on test32, normalised by test $\lVert R_0\rVert$.

### A3.7 Claims and the test stage
- Frozen at the end of `time`: per (cell, arm, τ, stat) the finalists, and the val **pair** $(a^\ast,b^\ast)$ attaining
  $S$. The test stage evaluates those frozen configurations only: test feasibility (meets τ on test32 by the three-way rule
  with $U_{\rm test}$) and Q1/Q2 are reported first; a "matched test speed-up" is attached only to a pair whose both members
  are test-feasible and qualified, re-timed on test[0:8]; otherwise the entry reads "not matched on test" (no reselection,
  no maximisation over finalists on test data).
- Truncated-search, eligible-subset and deferred-coverage labels are printed at each claim they qualify.

## Amendment A4 (2026-10-22, after Codex design audit 4 = READY with two non-blocking control corrections)

- **MF-Q2:** the configuration's own (production) rule weights are scaled by 1.5, giving $\rho\approx0.5$ against the
  successor (not 1/3 as A3.5 stated); the control asserts the **measured** margin $\rho\ge2\times0.116$.
- **MF-AD-cap:** the error estimate is forced to $+\infty$ for **every** trial, so the controller refines from $j_0$ to $J$ and
  then terminates; evidence required: cap-hit = 1, rejects = $J-j_0+1$, run invalid.
- MF-STOP fixtures additionally cover: one above + one undefined bank entry → UNRESOLVED; one below + one undefined →
  NOT FIRED; an arm censored at every required entry → SPEED-STOP UNRESOLVED.
- Test-stage speed-ups are called "speed-up of the validation-selected pair"; they do not establish the strongest
  competitor on test data. Numerical qualification and training-data qualification are reported separately.

## Amendment A5 (2026-10-22, implementation record; before the Codex code audit and before any cluster run)

- **A5.1 Hopeless-rung prune (step ladders, every family):** a step ladder stops early when, at a valid rung after the
  first, both statistics exceed $0.3=3\times$ the largest target **and** the step change is below a tenth of the worst
  error (the error is not time-dominated), so finer steps cannot reach any target (`ladder.step_ladder`).
- **A5.2 Flop screen for point forms:** a point-form reduced configuration is not run when its per-evaluation flop count
  exceeds 10× that of the tensor form at the same $(r, M_t)$ (LSPG, Jacobian included: $m>10r$; Galerkin: $4m>10r^2$); it
  cannot be the cheaper charged form. Skipped keys are listed in `ladder_coverage.json`.
- **A5.3 Rule search cap:** $k_q\in\{4,8,16,32,64\}$ for POD and bank (memory: $(64\cdot16)^2\approx10^6$ points); if no
  $k_q$ meets the A2.3 criteria the largest is used and labelled `k_q_capped` (Q2 still decides qualification).
- **A5.4 $r_\times^{\rm eval}$ comparator:** paired medians exist only per (FOM, arm) pair; the comparator is the FOM finalist
  with the smallest val[0:4] re-screen time (the same statistic for every FOM finalist, measured in the same job).
- **A5.5 Components:** IC, $\nu$-dependent setup and decode are timed separately; evolution is derived as (the fused query
  without decode) − IC − setup and labelled "derived"; (no-decode query + decode)/fused is reported.
- **A5.6** The `controls` stage runs in every cell job (not only the smoke); MF-LEAK / MF-FROZEN are fixtures in that
  stage (the test stage itself refuses without the committed manifest under `code/runs/frozen/<W>_<lev>.json`).
- **A5.7** Test references are generated (and gated) in the test stage only. A restarted ladder stage re-runs its rows.
- **A5.8 Smoke plan:** two real-size smoke jobs on H200, each a full cell run with every stage including `controls`:
  `smk-A2` = cell (W40, A2) and `smk-A4` = cell (W40, A4) (the most expensive cell type: memory pilot and cost
  certificate). Their results are smoke evidence; a production cell may adopt a smoke cell's outputs only if the code tree
  that produced them is byte-identical (by git tree hash) to the production commit, stated in the report.

## Amendment A6 (2026-10-22, after Codex code audit 1 = NEEDS-FIX; implementation record, before any cluster run)

- **A6.1 Scoring lattice** is the full 257² lattice with walls (§5); every arm except R-TRANS is exactly zero there,
  R-TRANS's wall values are scored. Bank training samples stay interior (A1.3).
- **A6.2 Statuses** (selection): `operator-check-failed` (the POD representation failed its $M$/$K$ checks: its arms
  are not selectable; counts like arm-unqualified), `missing-construction` (no valid ladder row at all), and
  `finalist-partial` (≥1 finalist, attempt cap reached before exhaustion). In the stop rules a `finalist-partial` entry
  counts as a match only if its $S\ge0.5$ ($S$ is a max over finalists, so a partial search can only under-state it);
  otherwise missing. An incomplete search of **either** FOM family makes the entry missing.
- **A6.3 Unscorable cells** (selection splits failing the A2.2 policy) write `final.json` with `scorable=false` and run no
  selection; the stop rule counts their entries missing. A cell's results enter the stop rule only if its controls
  registry passed (`controls.json: all_fire`).
- **A6.4** FOM boundary extensions are re-checked in `time` on the **qualified** finalists, followed by re-selection.
- **A6.5** Tensor-form Q2 compares the production tensor contractions $T_{12}zz$ vs $T_{14}zz$; the MF-Q2 / MF-ECSW faults
  scale the configuration's production weights (rollout and vectors). T-dbl failure invalidates the pairs of that
  finalist. Test-stage $S$ is published only with valid timing.
- **A6.6 Controls registry:** every mandatory control (ref/pod checks included) has a required outcome; `all_fire` is
  their conjunction. MF-ECSW's positive control (clean ECSW Q2 pass) is reported but not required, since ECSW at small
  $\kappa$ may legitimately fail Q2 (which only removes it from selection).
- **A6.7 Freeze authentication:** the test stage accepts only a frozen manifest whose git blob hash appears in the
  staged commit's tree listing (`code/.git_tree.txt`, written by `submit.sh`) and which is listed in the committed
  campaign freeze index `runs/frozen/INDEX.json`; a cell's test job refuses if the archive already holds `test.json`.
- **A6.8 Release gate:** `submit.sh` accepts only H200 jobs listed in the committed `cluster/RELEASE.json` (written after
  the job audit from the `resources.py` certificate); the job refuses a non-H200 GPU (exit 43).

## Amendment A7 (2026-10-22, after Codex code audit 2 = NEEDS-FIX; implementation record, before any cluster run)

- **A7.1 Construction coverage:** an arm with any expected-but-failed construction (ladder build error, ECSW fit without
  support) and no finalist is `missing-construction`; with finalists it is `finalist-partial` (A6.2 rule applies).
- **A7.2** The AD tolerance extension runs for every distinct binding finalist until closure; each finalist carries its
  step-ladder termination reason.
- **A7.3** MF-REF-real / MF-PAD-chk use the first val (padded) case accepted at its first-pass resolution, i.e. its
  accepted reference; if none exists the control is unavailable and the registry fails.
- **A7.4** The query-defining source set (`geom`, `spec`, `rom`, `podsine`, `bank`, `builders`, vendored kernels) is
  hashed at the freeze and verified in the test stage, with the descriptor hashes and the metadata files (`pod.json`,
  `bank.json`, `data.json`, `ref.json`, `cohorts.json`). The freeze index lists the retained cells; the test stage refuses
  unless every retained cell is frozen and committed. Test opening is reserved atomically (`mkdir
  burgers-archive/imp-box/.test-opened/<cell>`) before any test reference is generated; an unscorable test split writes
  `test.json` with `scorable=false` and evaluates nothing.
- **A7.5** Failed mandatory controls stop the job (non-zero exit; the cell is not accepted). MF-ECSW is required in
  every cell (no ECSW fit = failure); MF-ENV in every bank cell; the eligibility control injects an exclusion.
- **A7.6** $r_\times^{\rm query}$ (diagnostic): per $r$ the three cheapest valid bank rows by ladder screen are re-screened on
  val[0:4] and the minimum taken. Continuous coefficient errors zero-pad both arrays to the common size.
- **A7.7 Release:** `cluster/RELEASE.json` approves an exact job specification (body sha256, GPU, memory, time); the
  certificate validates the smoke (H200, COMPLETED, full size, every stage done, references scorable, controls passed)
  and adds one failed-job allowance per wave; the second data size is bounded by 2× the matching cell bound
  (registered conservative factor).
- **A7.8 Memory:** calculated payloads at A4 (training coefficients 9.5 GB host, 512 POD modes 2.4 GB, r = 384 ECSW
  matrix 13.3 GB host + device, rule spaces ≈ 20 GB host for one representation) fit an H200 node at `--mem 240G`; the
  smoke `smk-A4` is the memory pilot and records host and device peaks per stage; an out-of-memory failure is a failed
  job (missing), never a scientific outcome. Timing on GPU uses representative queries for the harness controls.

## Amendment A8 (2026-10-22, after the smoke logs of smk-A2 / smk-A4, before any production job; no selection result read)

- **A8.1 POD operator check bars.** smk-A2 recorded $\lVert M-I\rVert_F=1.1\times10^{-10}$ (512 modes, $k_q=16$, $q=16$),
  failing the A1.3 bar $10^{-10}$ and thereby blocking the PODG and ECSW arms for a benign reason: the Frobenius norm over
  $512^2$ entries accumulates round-off and the rule is, by the A2.3 $k_q$ criterion, converged only to $10^{-8}$. The bars
  are restated consistently with A2.3: relative $\lVert M-I\rVert_F/\lVert I\rVert_F\le10^{-8}$ and relative
  $\lVert K-K_{\rm ex}\rVert_F/\lVert K_{\rm ex}\rVert_F\le10^{-8}$ over all retained modes. The smoke cells ran under the old bar;
  their PODG/ECSW selection entries are smoke evidence only (costs and controls are unaffected), and smk-A2 / smk-A4 are
  not adopted as production cells.

## Amendment A9 (2026-10-22, after the first smokes; before any production job; no selection result read)

The first smokes (smk-A2 6186322, smk-A4 6186325) showed that every ladder row cost ≈ 22 s of wall time, almost all of it
two XLA compilations (the vmapped cohort run and the separate single-query screen), which would put a full cell at
≈ 8 h and the wide cells beyond the cap. Both smokes were cancelled after their ref/data/pod/bank stages (those stages'
costs and memory are recorded) and are re-run as `smk2-A2` / `smk2-A4` with:
- **A9.1** one compilation per ladder row: the cohort is run case by case with the same compiled query that then gives the
  screen time (the official unbatched outputs are thereby identical to the ladder outputs);
- **A9.2** a persistent XLA compilation cache in the staged code directory (excluded from collection and archives): closed-
  over arrays are executable parameters, so the time stage's rebuilt queries load instead of recompiling; the AD tolerance
  is a hoisted constant so all tolerances of one $N$ share an executable;
- **A9.3** a relative step-convergence stop for every step ladder: a ladder stops when the step change is $\le5\,\%$ of the
  row's own (smaller of median and worst) error; finer steps are more expensive and cannot change which targets that size
  meets, so this removes only dominated rows.
The cost certificate uses the `smk2-*` smokes. The ref/data/pod/bank costs of the cancelled smokes are reported.

### A9 addendum (after code audit 7 and job audit smk2-1; before submission)
- **Correction:** smk-A4 was cancelled during bank training (log at step 295,000 of 300,000; no `.done_bank`); only its
  ref/data/pod stages completed. smk-A2 completed ref/data/pod/bank.
- **A9.2 restated:** no executable sharing across AD tolerances is claimed (the tolerance is lowered as a literal); the
  persistent cache serves the time stage's rebuilt queries of identical configurations. The cache must be the job-local
  `code/.jaxcache` (enforced), is capped at 30 GiB (LRU), and a job refuses to start with < 200 GiB free there.
- **A9.3 replaced** by a target-aware stop for explicit steppers only (F-SPEC and every Galerkin ETDRK4 arm, whose cost is
  exactly proportional to the step count): a step ladder stops when no (τ, stat) three-way classification of the row would
  change if both statistics improved by twice the last step change (conditional on the observed convergence; labelled
  `converged-target-aware`). LSPG and AD ladders keep only the registered absolute rule.
- Collection verifies non-empty checksum sets only (an empty selection is valid); the ledger takes the GPU type from the
  job's `allocated_gpu.txt` when sacct's AllocTRES carries none.

### A9 addendum 2 (after code audit 8 / job audit smk2-2; before submission)
- **A9.3 withdrawn entirely.** Neither the relative nor the target-aware stop is a domination guarantee (counterexamples in
  `audits/codex-code-8.md`); step ladders keep only the registered absolute rule ($\delta_n\le10^{-4}$), the A5.1 prune
  and the invalid-rung rule. The compile-cost reduction rests on A9.1 (one compile per row) and A9.2 (persistent cache).
- The job-local cache path is enforced for every run (local debugging needs an explicit override); reported speed-ups carry
  the cell's controls acceptance, FOM search status and timing exclusions beside them.

## Amendment A10 (2026-10-22, after the re-smokes' FOM ladders and four diagnostic jobs; before any production job)

Both H200 re-smokes flagged every fixed-step F-SPEC row at $N=12$ as non-finite while all other sizes (and the adaptive
$N=12$ rows) were valid. Diagnostics (`diag_jobs/`, jobs 6188855 A100, 6188970 / 6189104 / 6189294 H200; no scientific
use; `audits/NOTE-diag-n12h.md`): on H200 / JAX 0.10.2 the compiled fixed-step loop at $N=12$ produces non-finite states
from step 2 on every case, independently of the compilation cache (off, populated, replayed), of the loop form
(dynamic/static `fori_loop`, `scan`, unroll 2, an optimisation barrier) and of the padding ($M=3N/2$ or $2N$), while a
Python loop of a separately compiled step is finite and its nonlinear term equals an independent dense evaluation to
$2.5\times10^{-15}$; on A100 the same code is correct. It is a Hopper code-generation fault specific to this size.
**Change:** the F-SPEC/F-SPEC-AD size ladder replaces $N=12$ by $N=10$ and $N=14$ (both verified correct on H200 against the
eager reference to $\le1.3\times10^{-15}$): $N\in\{8,10,14,16,24,32,48,64,96,128,192,256\}$ (+384, 512 extensions). This
densifies the competitor ladder near the gap (it can only strengthen the FOM). Safeguards against silent mis-compilation
of other configurations: non-finite rows are invalid (flag), and every finalist's Q1 replays its outputs through a
separately compiled certificate program (≤ 1e-12), which a fault in one of the two programs would fail. The two re-smokes
keep their $N=12$ rows (invalid) as smoke evidence; their FOM selection is not production evidence.

## Amendment A11 (2026-10-22, from the smk2-A2 time-stage log, before any production job): competitor strengthening

The smk2-A2 paired timings showed the reduced models 1.6–2.6× faster than the fixed-step FOM on the H200 (batch 1). At these
small sizes the FOM's cost is kernel-launch latency of its FFT-based sine transforms (several transforms per nonlinear
evaluation), not arithmetic; a GPU-tuned implementation of the same discretisation would use dense transform matrices.
§5 requires every competitor to be tuned over its own knobs, so the transform implementation becomes an F-SPEC /
F-SPEC-AD knob: `xform ∈ {fft, dense}` for $N\le128$ (dense = the same $3N/2$ padded grid and trapezoid sine analysis via
$u=SbS^\top$, $g=CbS^\top+SbC^\top$, $-A(u\circ g)A^\top$; identical to the FFT version to $\le10^{-14}$, checked in every cell
by the new mandatory control **G-XFORM**, bar $10^{-12}$). The ladder runs both implementations; selection takes the faster
qualified one. The smoke's speed-ups (FFT-only FOM) are smoke evidence only and are not reported as results.

### A10/A11 addendum (after code audit 10)
- A10 is narrowed to the demonstrated failure: only the fixed-step **FFT** configurations at $N=12$ are excluded; adaptive
  $N=12$ rows and the dense-transform $N=12$ rows stay in the ladder (a fault there would be flagged as non-finite or caught
  by the Q1 replay). $N=10$ and $14$ are added. This densifies the ladder; it is not claimed to dominate the excluded rows.
- The implementation (`xform`) is part of every FOM configuration key, including the AD tolerance extensions. G-XFORM
  covers every dense size and fails on non-finite values (NaN mutation fixture). Measured FFT-vs-dense agreement on CPU
  reached 3.8e-14 at $N\le128$ (bar 1e-12). Smoke cells are never substituted for production cells in the report.

## Amendment A12 (2026-10-22, after smk2-A2 completed: every stage, controls registry all fire; before production)

- **Cost certificate under post-smoke changes.** A10/A11 changed policy and query sources after the smokes (dense FOM rows,
  $N=10,14$ rows). The certificate (`resources.py`) therefore multiplies the smoke's measured elapsed time by a registered
  factor 1.5 (the added FOM rows are ≈ 100 of ≈ 560 ladder rows at ≈ 10 s each, i.e. ≈ +5 % of the cell); the changed files
  are listed in the certificate. Wave-1 certificate (`runs/certificate-wave1.json`): spent 5.79 + committed 24 (smk2-A4 wall
  limit) + 1.15 × 4 × 6.23 (W40_A1, W40_A2, W28_A1, W28_A2) + one failed-job allowance 6.23 = 64.7 GPU-h ≤ 87.
- Production jobs: `<W>_<lev>` bodies (`cluster/bodies/`), H200, 240G, wall 10 h at A1/A2; A3/A4 are certified from smk2-A4.

## Amendment A13 (2026-10-22, after the coordinator's checkpoint release; before any narrow-cell job)

- Order (brief's defer order): wide A1–A3 cells, then W = 0.08 (A1–A3, bank arm included), then W = 0.16; A4 production
  cells only if smk2-A4 certifies them within the cap (smk2-A4 is cancelled at 14 h by `cluster/cancel_at.sh`).
- Downward ladder extensions are **not** run: re-running finished cells' time stages is not cheap; finalists at the bottom
  of their ladders (FOM N = 8 or n = 5; reduced r = 16 or n = 10) are labelled "truncated at the ladder minimum" in the report.
- Wave-2 certificate `runs/certificate-wave2.json` (W28_A1, W08_A1, W08_A2): 63.2 GPU-h projected ≤ 87 (smk2-A4 counted at
  14 h; running cells at their 6.2 h bounds).
- A13 correction (job audit wave2-1): the wave-2 certificate counts smk2-A4 at its full 24 h limit and the running cells
  at their unrounded 6.2275 h bounds: 73.3 GPU-h <= 87 (supersedes the 63.2 above); cancel_at.sh validates the job and
  verifies the cancellation. Incident: a pkill by pattern during this fix killed only this agent's own shell and its old
  cancel_at guard (no other lane's process), recorded here.

## Amendment A14 (2026-10-22, after smk2-A4; before any A3 production job)

- **smk2-A4 cancelled at ≈ 9.8 h** (coordinator's guidance, defer order): ref/data/pod/bank/ladder completed (6.4 h; host
  peak 60 GiB, device peak 36 GiB), but its time stage made no reported progress for > 2 h in the ECSW selection (with
  persistent-cache serialisation warnings). **A4 production cells are deferred** (defer order item 2; the transport arm is
  cheap and kept). The cause of the A4 time-stage stall is not diagnosed (stated limitation).
- **A3 cells (W40_A3, W28_A3) are released as exposure-bounded jobs** at their 10 h wall limit (no completed A3/A4 smoke
  exists to certify a tighter bound). Wave-3 certificate `runs/certificate-wave3.json`: spent 23.8 + committed 32.5 (two
  running A1/A2 cells at 6.2275 h + two A3 cells at 10 h) + 1.15 × 2 × 6.2275 (W08_A1, W08_A2) + 6.2 allowance = 76.8 ≤ 87.
  An A3 job that hits its wall limit is reported as missing (not a scientific outcome). Order: W40_A3, W28_A3, W08_A1,
  W08_A2; W16 cells and W08_A3 only if measured costs leave room.
- A14 corrections (job audit wave3-1): the failed-job allowance is the largest exposure in the wave (10 h, A3), giving
  80.6 GPU-h ≤ 87 (`projected_total_corrected`). W08_A1 (approved and audited in wave 2) started before the A3 cells
  because three duplicate releaser processes were live; they were stopped, and a single releaser now runs W40_A3, W28_A3,
  then W08_A2. The order deviation does not change any job's specification.
