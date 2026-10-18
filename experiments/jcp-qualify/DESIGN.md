# jcp-qualify — which 2D/3D reduced-model configurations are numerically qualified, and what is the qualified error–cost front? (pre-registration)

**Status:** pre-registration, 2026-10-18, written before any code or data of this lane. Amendments are appended at the
end, dated; nothing above them is rewritten. Every accuracy number of this lane is scored against the E0 second-order
reference $R_{ST}$, whose per-case uncertainty $\varepsilon_{\rm ref}$ is a **conditional Richardson estimate** (not a
rigorous bound; the E0 cross-check rule was amended after data, E0 A10). Post-hoc labels carry forward unchanged: E0 A10;
combined A10, A11 and the retracted A15.

Governing documents (later supersedes earlier): `reports/2026-10-17-full-experiment-plan/PLAN-v2.md` (Group Q, B1, B4,
B5), `PLAN-v3-amendments.md` (V3-1, V3-4, V3-5), `PLAN-v4-amendments.md` (V4-1, V4-2, V4-5, V4-6);
`campaign/P1-RESTATEMENT.md` + `campaign/P1-RESTATEMENT-AMENDMENTS.md` (AM-1 … AM-8: the Q1/Q2 and statistics contracts
inherited here); `campaign/briefs/COMMON-BRIEF.md` (binding); `campaign/briefs/jcp-qualify.md` (lane brief, wins on
conflict); `reports/2026-10-08-jcp-lane-agent-rules.md`.

Lane: worktree `worktrees/2026-10-18-jcp-qualify`, branch `exp/2026-10-18-jcp-qualify` (fork `21c175a1b`, setup
`775eef470`), lane dir `experiments/jcp-qualify` (here), cluster namespace `/cluster/tufts/paralab/tawal01/jcp-qualify/`,
lane job cap 3 (campaign cap 8, checked on the account before every submit), GPU-hour cap **35** including contingency,
with time-limit reservations.

Lanes built on (read, cited, code copied with provenance; their certified results are **cited, not re-run**):
`jcp-p1-scaling` (P1, `worktrees/2026-10-17-jcp-p1-scaling`, final, 1.91 GPU-h, results+report audit 4 confirmed);
`jcp-combined` (2D grid of schemes × Δt, finalist table); `jcp-speed3d` (3D grid, Q1 certificates, screening front;
final, report audit READY); `jcp-time2` (temporal-order machinery; LSPG-CN is not established second order).

## 0. Questions and outcomes

| item | question | outcome labels |
|---|---|---|
| **Q1–Q3 (2D)** | for every 2D configuration that can enter the B5 front or the C1/C2 comparison: is it numerically qualified (solver Q1 + quadrature Q2), and how is it conditioned (Q3)? | per configuration: Q1 pass/fail; Q2 pass / fail / unverified; numerically qualified yes/no; Q3 numbers |
| **Q2-3D** | are speed3d's 3D front members and H2 winners quadrature-qualified on their own reached states? | per configuration: Q2 pass / fail / unverified |
| **B1** | error against $r$ and $N$ for every qualified 2D configuration; H-acc-N and H-acc-r (AM-4/AM-7 rules) applied to the new (scheme, Δt) families | H-acc-N pass / fail / inconclusive; H-acc-r per transition improves / does not improve / inconclusive, range monotone / not monotone / inconclusive |
| **B4** | observed temporal order of each scheme against a same-form converged anchor, with solver and quadrature contributions verified smaller | per (scheme, r): observed order $\hat p$, interpretable yes/no, integer-order label or "no order label" |
| **B5** | the qualified Pareto front of error (median and worst, E0 ST) against online cost (paired A–B–A timing, one job per mesh) parameterised by $r$, at each 2D mesh; the 3D front assembled from speed3d's audited outputs | front membership per configuration and mesh; failed/unqualified configurations shown separately |

Three labels are kept separate everywhere (V4-2): (1) **numerical qualification** (Q1 + Q2), (2) **role qualification**
(§5.4, pre-declared criteria), (3) **frontier usefulness** (not dominated, §6). The front need not be monotone in $r$.

## 1. The reduced model (unchanged from P1 §1)

Bank coefficients $c\in\mathbb R^{r}$, $u=\hat G_r(x)\,c$ (frozen checkpoint `dn256b`, $R=512$, $K=16$, rotation
`rotation_R512`, ordered nested prefixes). Test space $M=4r$ discrete sine modes, $A=\Phi^{\mathsf T}\hat G_r$,
$\Lambda$ the discrete eigenvalues; tested advection by an off-mesh point rule $(x_q,w_q)_{q=1}^m$:
$$N(c)=\Psi^{\mathsf T}\big[(G_qc)\odot(G_sc)\big].$$
Two-step LMM residual (combined `t2core.py`, P1 `p1core.py`):
$$\mathcal R_n(c)=A(\alpha_0c+\alpha_1c_n+\alpha_2c_{n-1})+\Delta t\,(\beta_0F(c)+\beta_1F(c_n)),\qquad F(c)=N(c)+\nu\Lambda Ac.$$
- **LSPG** minimises $\lVert D\mathcal R_n\rVert$, $D=(\alpha_0+\beta_0\Delta t\,\nu\Lambda)^{-1}$; tolerances
  $(g_{\rm tol},{\rm tol}_f)$: prod $(10^{-3},10^{-9})$, tight $(10^{-7},0)$, tighter $(10^{-8},0)$.
- **GAL** solves $Q^{\mathsf T}\mathcal R_n(c)=0$ ($A=QR_m$, thin QR): the same LMM on the Galerkin ODE; tolerances
  ${\rm tol}_f$ (relative to $\lVert Ac_n\rVert$): prod $10^{-10}$, tight $10^{-13}$, tighter $10^{-15}$ (`c2common.tol_of`).
- Adaptive Levenberg–Marquardt (fused LM, Cholesky, clipped step, damping carried, quadratic predictor), budget 600.
- Initialisation contract (V4-4, kept): "full-field input sampled at quadrature points; full-field decode".
- Horizon 0.25; outputs at $t=0,0.05,\dots,0.25$; every step size divides 0.05 exactly (asserted).

Code: P1's `p1core.py` / `p1common.py` are copied unchanged (provenance: P1 commit and sha256 in `PROVENANCE.json`);
this lane's changes live in new files (§10). The only new solver-side code is the Galerkin certifier (§4.1b), the
Galerkin conditioning Jacobian (§4.3) and the generalisation of the drivers to arbitrary $\Delta t$ and form.

## 2. Configurations (frozen before data)

### 2.1 The 2D matrix

A configuration is the P1 tuple (bank + rotation sha256, $r$, ordered mode identities and hash, rule name / $m$ /
generator identity, mesh, scheme, form, $\Delta t$, tolerance level, budget, initialisation, weighting, output times,
metric; P1 §2.1, A6.2). The registered matrix:

| axis | values |
|---|---|
| $r$ (bank width) | 128, 256, 384, 512 |
| frozen rule per $r$ (V4-3, P1 §2.2) | R128 `fib4181` (seed-0 shift, one frozen realisation); R256 `gauss128`; R384 `gauss128`; R512 `gauss192` |
| mode list | frozen $F_{4r}$ = `modes_lean(1024, 4r)` reused at every mesh (V4-1; P1 §2.3) |
| bank construction | full R512 bank, edges $[0,128,256,384,512]$, at every mesh (P1 A6.3: 4096² pilot peak 113 GB < 120 GB) |
| schemes | **LSPG-BE**; **LSPG-CN-R** (CN with two BE start-up steps); **GAL-CN** (Galerkin CN, no start-up; *diagnostic* scheme per the brief, fully qualified and eligible for the front) |
| step $\Delta t$ | $\Delta t_0=0.005$; $2\Delta t_0=0.01$; one larger step per scheme $L_s$ (§2.2) |
| meshes $N$ | $256^2$, $1024^2$, $4096^2$ |
| cases | **all 38** (dev6 + val32) at every mesh for every configuration (front and B1 population) |

$4\times3\times3\times3=108$ configurations. **Inherited from P1 (cited, not re-run):** (R128, R256, R384) × (BE, CN-R) ×
$\Delta t_0$ × three meshes, and R512 × (BE, CN-R) × $\Delta t_0$ × $1024^2$ — 20 configurations, each Q1+Q2 qualified in
P1 (`p1-scaling/checks/analysis.json` `configurations`). For R128 and R256 at $4096^2$ P1 ran only the 16-case subset;
this lane runs the **remaining 22 cases** of those four configurations (prod + tight, recorded and certified in-job),
so that every front configuration has the 38-case population. Their Q1 is then judged on the union (P1's 16 rows + 22
new rows: all predicates of §4.1 on the pooled population, P1's tight/tighter subset unchanged); their Q2 is P1's
(population sub16 unchanged). Equality of the numerical configuration with P1's is checked by manifest (P1 AM-1 rule:
numerical fields equal; provenance recorded). **New:** 88 configurations.

### 2.2 The larger step per scheme (rule frozen now, from combined's grid; no lane data seen)

Rule: per scheme, the **largest** factor $f\in\{2.5, 5, 10\}$ of combined's $1024^2$ grid (`c2grid`, E0 analysis
`checks/analysis-c2grid-2d_e0.json`) at which the R384 (`gauss128`, the frozen R384 configuration) arm's **median** E0
$R_{ST}$ error is at most the **worst** E0 error of the deployed LSPG-BE R384 arm at $\Delta t_0$ (3.01 %). Read from that
file before any lane job:

| scheme (combined key) | $f=2.5$ median | $f=5$ median | $f=10$ median | $L_s$ |
|---|---|---|---|---|
| LSPG-BE (`R384|LSPG|BE`) | 2.29 % | 4.16 % | 7.19 % | **2.5** ($\Delta t=0.0125$) |
| LSPG-CN-R (`R384|LSPG|CNR`) | 1.10 % | 3.79 % | 7.14 % | **2.5** ($\Delta t=0.0125$) |
| GAL-CN (`R384|GAL|CN`) | 0.36 % | 1.95 % | 5.23 % | **5** ($\Delta t=0.025$) |

(The table is regenerated by `make_configs.py` from that file and must equal these choices; a mismatch stops the lane.)

### 2.3 Rule ladders, successors and the registered fallback

Successor $q^+$ and check $q^{++}$ (P1 §2.2): R128 `fib4181`→`fib6765`→`fib10946`; R256/R384
`gauss128`→`gauss160`→`gauss192`; R512 `gauss192`→`gauss256`→`gauss320`.

**Registered fallback (one level, pre-declared):** a configuration whose Q2 label is not "pass" is re-run once as the
**fallback configuration** with main rule $q^+$ (successor $q^{++}$, check $q^{+++}$: `fib17711`, `gauss256`, `gauss384`
respectively), everything else equal, in a follow-up job, with the full Q1/Q2/Q3 treatment and all 38 cases. The original
configuration stays reported as failed/unverified; the fallback is a new configuration (named with suffix `fb`) and
enters the front only on its own qualification. A fallback that is not qualified is reported; there is no second level.
Fallback timing enters the per-mesh timing jobs only if the fallback is known before they are submitted; otherwise a
dedicated fallback timing job for that mesh runs with the same A and protocol (§7), and the fallback's front membership
is computed against the other configurations' timings **from the original job** through the common reference A
(labelled "cross-job, A-normalised").

### 2.4 3D: Q2 for speed3d's front members and winners

**3D front (for Q2 selection), frozen now from speed3d's audited outputs** (`jcp-speed3d/checks/analysis-grid.json`,
eligible configurations with a valid screening panel; screening cost = ratio × $t_A$, speed3d's own plot axis), using
the dominance rule of §6 with that file's per-case $\eta$ as $\varepsilon_{\rm ref}$. Union of the median and worst
fronts, plus the two H2 winners (speed3d `EXPORT.json` `H2`):

| # | configuration (setting\|rule\|solver spec) | front | rule → $q^+$ → $q^{++}$ |
|---|---|---|---|
| 1 | W256\|gl16\|fs\|BE\|0.016666667\|K1 | median, worst | gl16 → gl20 → gl24 |
| 2 | W256\|gl16\|fs\|BDF2\|0.025\|K2 | worst | gl16 → gl20 → gl24 |
| 3 | W256\|gl16\|fs\|CN\|0.0125\|K1 | median | gl16 → gl20 → gl24 |
| 4 | W256\|gl20\|fs\|CN\|0.0125\|K1 | median | gl20 → gl24 → gl32 |
| 5 | W256\|gl20\|fs\|CN\|0.01\|K1 | median | gl20 → gl24 → gl32 |
| 6 | W256\|lat8192\|fs\|CN\|0.01\|K1 | median | lat8192 → lat16384 → lat32768 |
| 7 | W512\|gl24\|qn\|CN\|0.025\|k4 | median | gl24 → gl32 → gl40 |
| 8 | M2_512\|gl24\|qn\|BDF2\|0.025\|k4 | worst | gl24 → gl32 → gl40 |
| 9 | W512\|lat16384\|qn\|CN\|0.025\|k4 | median | lat16384 → lat32768 → lat65536 |
| 10 | W512\|gl24\|qn\|CN\|0.016666667\|k4 | median | gl24 → gl32 → gl40 |
| 11 | M2_512\|gl24\|fs\|BDF2\|0.0125\|K1 | worst; **H2 winner W_C_old / W_C_W1024** | gl24 → gl32 → gl40 |
| 12 | M2_512\|gl24\|qn\|BDF2\|0.016666667\|k4 | worst | gl24 → gl32 → gl40 |
| 13 | M2_512\|gl24\|fs\|CN\|0.01\|K1 | median | gl24 → gl32 → gl40 |
| 14 | W512\|gl24\|fs\|CN\|0.01\|K1 | median | gl24 → gl32 → gl40 |
| 15 | W768\|lat16384\|fs\|CN\|0.01\|K1 | median | lat16384 → lat32768 → lat65536 |
| 16 | W1024\|lat32768\|fs\|CN\|0.01\|K1 | median | lat32768 → lat65536 → lat131072 |
| 17 | M2_512\|gl24\|qn\|CN\|0.016666667\|k4 | **H2 winner W_C_W768** (dominated in the screening front) | gl24 → gl32 → gl40 |

(Regenerated by `front3d.py` from the speed3d file; a mismatch stops the lane.) Successors are the next entry of the
speed3d rule ladders (Gauss gl12, gl16, gl20, gl24, gl32, gl40, gl48, gl56, gl64, gl80; lattice lat2048 … lat131072;
`rules.npz` + `rules_extra.npz`, sha256 pinned by speed3d `s3common.RULE_FILES`).

**Q2-3D predicate** (AM-2 transposed to 3D, speed3d A2.3 lattice metric): population = cohort 923801, all 64 cases (the
cohort speed3d selected on). (a) rollout bar: replay of the configuration with $q^+$ (same setting, solver spec,
$\Delta t$, $K$, initialisation, everything else equal); lattice distance $\max_k\lVert u_k-u^{+}_k\rVert/\lVert u_0\rVert$
on the $63^3$ scoring lattice ≤ $2.5\times10^{-4}$ on every case; (b) ρ bar: on the configuration's own six output states
of every case, $\rho=\lVert N_q(c)-N_{q^+}(c)\rVert/\lVert N_{q^+}(c)\rVert$ of the tested advection (`c3common.nl3`),
$\rho_{\max}\le0.116$; (c) finer-rule convergence on cases 0–7: $q^+$ vs $q^{++}$ meets the rollout bar, else "unverified
(finer rule not converged)". Edge cases as P1 A1.13 / A2.4. A reproduction gate first: the lane's rerun with rule $q$
must reproduce speed3d's archived coefficients of the same configuration (`runs/g{1,2,3}/archive`) to
$\le10^{-10}$ relative on every case (both H200); a failing gate blocks that configuration's Q2 label.
3D Q1 is speed3d's (eligibility E-a/E-b/E-c + ADT gate; `EXPORT.json` `q1`), cited.

### 2.5 Cohorts, references and scoring (common brief §2; P1 §3)

dev6 and val32 only in 2D (training disjointness asserted); 3D 923801 only. No test64/923901 run, no sealed/OOD access;
`sealed/` and `custody/` never touched. Scoring offline on CPU through `refstore` (E0 store version pinned in the
manifest): $e=\max_{k=1..5}\lVert u(t_k)-R(t_k)\rVert/\lVert R(t_0)\rVert$ on the $257^2$ lattice; eligibility per
case-kind read from the store at scoring time; $R_S$ only for LSPG-BE at $\Delta t_0$ (labelled), every other S use
"unmatched diagnostic"; ties $|\Delta e|\le2\varepsilon_{\rm ref}$; every scored record asserts its identity (cohort,
case, time tag, reference kind, reference sha256, store version, metric id). Cases are the statistical unit.

## 3. B1: error against $r$ and $N$ (H-acc-N, H-acc-r on the new families)

Families = (scheme, $\Delta t$) — 9 families; P1 owns (BE, $\Delta t_0$) and (CN-R, $\Delta t_0$) for $r\le384$ and
R512 at $1024^2$; its verdicts are cited.

- **H-acc-N** per (r, family), procedure exactly P1 §4.3 + A1.9 (AM-4): P256 = $256^2$ vs $1024^2$ on the 38 cases
  eligible at both; P4096 = $4096^2$ vs $1024^2$ on the **16-case subset** (registered population, identical to P1); trend
  on the 16-case subset; $\delta_i=\max(2\varepsilon_{{\rm ref},i},0.1\,e_i(1024^2),10^{-12})$; TOST = two one-sided
  Wilcoxon tests at α = 0.05; pass / fail / inconclusive as AM-4; "inconclusive (unqualified configuration)" if any of the
  three configurations is not numerically qualified. Sensitivity (reported, no verdict): P4096 on all 38 cases.
  New families: every (r, family) except P1's.
- **H-acc-r** per (mesh, family), procedure exactly P1 §4.4 / AM-7 with precedence: transitions 128→256, 256→384,
  384→512 at **every** mesh (R512 now runs at every mesh); range verdicts over 128–384 (primary, as P1) and 128–512.
  For P1's two Δt₀ families the new transitions are 384→512 at $256^2$ and $4096^2$ (and the 128–512 range there).
  Every transition reported with improve / worsen / tie / unqualified counts, median paired change, worst change.
- Tables and plots: median and worst error against $r$ per mesh and family; against $N$ per $r$ and family.

## 4. Qualification

### 4.1 Q1 — solver qualification, certified in-job at every step with a fresh Jacobian (V4-2; P1 §5.1, A1.3, A1.4)

Every production rollout records its internal trajectory $c_0,\dots,c_S$ ($S=0.25/\Delta t$). Configuration-level Q1:
1. every run valid: finite outputs, no non-finite step, no damping-saturation exit (reason 3); budget exits counted and
   judged by the certificate (A1.4);
2. per-step certificate, non-certified steps ≤ 1 % **pooled over all steps of all cases** (4.1a, 4.1b);
3. scoring-lattice relative distance to the adaptive **tight** solve of the same configuration ≤ $2.5\times10^{-4}$ on
   **every** case (tight anchors on every case; an invalid tight anchor makes the case unqualified);
4. tight/tighter agreement ≤ $2.5\times10^{-5}$ on dev6 (tighter on dev6 cases 0–5);
5. G-cert: the in-job certificate agrees with the solver's own recorded per-step quantity (tolerances below); offline
   recomputation of every distance agrees with the in-job value (A1.3: $10^{-9}$ relative + $10^{-13}$).

**4.1a LSPG** (exactly P1): $\gamma_n=\lVert J^{\mathsf T}r\rVert/(\lVert J\rVert_F\lVert r\rVert+10^{-300})$ from the
rebuilt LSPG residual and a **forward-mode autodiff** Jacobian (not the solver's analytic `nlJ`); non-stationary if
$\gamma_n>10^{-3}$ or non-finite; G-cert $|\gamma_{\rm cert}-\gamma_{\rm rec}|\le10^{-7}\gamma_{\rm rec}+10^{-12}$.

**4.1b GAL (form-appropriate certificate; deviation from the literal brief, stated):** a Galerkin step is a root of the
square system $Q^{\mathsf T}\mathcal R_n(c)=0$; the brief's least-squares stationarity ratio is not a convergence measure
for a root (at a converged root it is a ratio of round-off quantities and is $\ge\sigma_{\min}(J)/\lVert J\rVert_F$
whenever $r\neq0$, so it cannot certify anything). The GAL certificate is therefore the **relative Galerkin residual**
rebuilt from the saved states (scheme coefficients, history $Ac_n$, $Ac_{n-1}$, $F(c_n)$), with $Q$ the thin-QR factor of
$A$:
$$\eta_n=\frac{\lVert Q^{\mathsf T}\mathcal R_n(c_{n+1})\rVert}{\lVert Ac_n\rVert},$$
non-converged if $\eta_n>10^{-8}$ (100 × the production tolerance $10^{-10}$, margin for recomputation round-off) or
non-finite; ≤ 1 % pooled (predicate 2). The Jacobian $J_G=\partial(Q^{\mathsf T}\mathcal R_n)/\partial c$ is formed by
autodiff at every step and its condition number recorded (a singular step Jacobian would make the root ill-posed; reported
with Q3). G-cert: $|\eta_{\rm cert}-\eta_{\rm rec}|\le10^{-6}\eta_{\rm rec}+10^{-13}$ with
$\eta_{\rm rec}=$ (recorded tolerance ratio) × ${\rm tol}_f$. The literal $\gamma_n$ is also computed and reported for
GAL, without a verdict. Independence: $\lVert Q^{\mathsf T}x\rVert=\lVert P_{\mathrm{range}(A)}x\rVert$ does not depend on
the QR sign convention, so the offline NumPy check (`npcheck`, own QR) is basis-independent.

### 4.2 Q2 — quadrature qualification (AM-2; P1 §5.2, A1.13, A2.4)

Population: dev6 + val32[0:10] (16 cases) per configuration at every mesh. Rollout bar: replay with $q^+$, max over the
five evolved outputs of the scoring-lattice relative distance ≤ $2.5\times10^{-4}$ on all 16 cases. ρ bar: own reached
states (6 outputs × 16 cases), $\rho_{\max}\le0.116$ (borrowed threshold, quadrature-study `DESIGN.md:234`; a
successor-rule / output-state check). Finer-rule convergence on dev6: $q^+$ vs $q^{++}$ rollout bar, else "unverified
(finer rule not converged)". Labels pass / fail / unverified; the other 22 cases are never described as individually
Q2-tested. **Numerically qualified** = Q1 pass and Q2 pass.

### 4.3 Q3 — conditioning (P1 §5.3)

Per configuration: singular values of $A$ (per $r$, mesh); at output indices 1, 3, 5 of each dev6 trajectory, the
singular values of the step Jacobian of the scheme's own solve — LSPG: $J=A+\beta_0\Delta t\,S\,\partial N(c)$ (history-
free, autodiff, P1 `make_jac`), so $\mathrm{cond}(J^{\mathsf T}J)=\mathrm{cond}(J)^2$; GAL: $J_G=\alpha_0R_m+\beta_0\Delta
t\,(Q^{\mathsf T}\partial N+\nu Q^{\mathsf T}\Lambda A)$. Median and max condition number and rank at $10^{-13}$
relative. No fix (orthogonalisation, regularisation) is applied; Q3 is descriptive.

### 4.4 Role qualification (pre-declared, descriptive; never merged with the other labels)

For each numerically qualified configuration $X$ at mesh $N$, against the deployed LSPG-BE at $\Delta t_0$ with the same
$r$ and mesh ($B_r$), from the same timing job: **fast-equal-accuracy** iff cost ratio $c_X/c_{B_r}\le0.6$ and $X$ ties
or beats $B_r$ on both median and worst error (ties within $2\varepsilon_{\rm ref}$ of the case that defines each
statistic); **accurate-equal-cost** iff $c_X/c_{B_r}\le1.1$ and the median error is lower by more than the tie margin.
Configurations meeting neither are "no role" (not a failure).

## 5. B4: observed temporal order per scheme (AM-7 rule; P1 §6 CN-R row with A1.7)

Fixture: $1024^2$, $r\in\{128,384\}$ (frozen rules), schemes LSPG-BE, LSPG-CN-R, GAL-CN, dev6 cases 0–5, **tight**
tolerance. Step factors $h/\Delta t_0\in\{L_s, 2, 1, \tfrac12, \tfrac14\}$, same-form anchor $h_a=\Delta t_0/64$, anchor
check $\Delta t_0/128$ (one anchor rule for all schemes, so the anchor is not chosen by the scheme's name). Error
$e(h)=\max_k\lVert u_h(t_k)-u_{h_a}(t_k)\rVert/\lVert u_0\rVert$ (scoring lattice). Per case: observed order $\hat p$ =
least-squares slope of $\log e$ against $\log h$ over $\{1,\tfrac12,\tfrac14\}$; pairwise orders reported over the whole
ladder (the large steps are pre-asymptotic context, no verdict). Per case the estimate is **interpretable** iff
(i) asymptotic: the two pairwise orders on $\{1,\tfrac12,\tfrac14\}$ differ by ≤ 0.3; (ii) anchor:
$d(h_a,h_a/2)\le0.1\,e(\tfrac14)$; (iii) solver: $d({\rm tight},{\rm tighter})$ at $\tfrac14$ ≤ $0.1\,(e(\tfrac12)-e(\tfrac14))$;
(iv) quadrature (A1.7): $\max_k\lVert(u_q(\tfrac14)-u_q(h_a))_k-(u_{q^+}(\tfrac14)-u_{q^+}(h_a))_k\rVert/\lVert u_0\rVert\le
0.1\,(e(\tfrac12)-e(\tfrac14))$ and $d(u_{q^+}(\tfrac14),u_{q^{++}}(\tfrac14))\le2.5\times10^{-4}$; (v) every run valid.
**Label per (scheme, r):** if ≥ 5 of 6 cases are interpretable, $\hat p$ = median over interpretable cases (range
reported); the integer label "order 1" / "order 2" is awarded iff **every** interpretable $\hat p$ lies in
$[0.8,1.2]$ / $[1.8,2.2]$; otherwise "observed order $\hat p$, no integer label". With < 5 interpretable cases:
"order not established (not interpretable)". Never awarded by name.

Controls for B4: **positive gate** (CPU, independent oracle): the manufactured LMM test (`test_lmm.py`, copied from
jcp-time2, SciPy oracle) on the lane's `p1core` copy: GAL-CN and GAL-CN-R order ∈ [1.8, 2.2], GAL-BE ∈ [0.8, 1.2].
**Must-fail control (real data):** GAL-TH06 (θ = 0.6, first order; `SCHEMES['TH06']`) through the same B4 procedure at
R384 on dev6 must **not** receive the "order 2" label (expected observed order ≈ 1). If it does, B4 is blocked.

## 6. B5: the qualified front

**2D cost.** One timing job per mesh (H200), A–B–A with P1's reference A (r = 384, $1024^2$, LSPG-CN-R, $\Delta t_0$,
frozen list, `gauss128`) on dev6 × 3 repeats, triplet order randomised; P1 protocol (§7 of P1 + A1.10, A3.2, A4.1):
split triplet (A evolve | B init | B evolve | B decode | A evolve, each burned and synchronised; drift band
$[1/1.1,1.1]$, ≤ 2 retries) and fused triplet (A | B fused query | A); every timed execution reproduces its scoring run's
identity hash (W, every internal state, per-step exit reasons and iteration counts). Subjects: all 36 configurations of
the mesh. Cost of a configuration: **primary** $c$ = median over its valid fused triplets of $t_{\rm fused}/\bar t_A$
("online cost, full-field contract": input sampling, evolution, full-field decode); secondary $c_{\rm ev}$ = median
evolution / $\bar t_A$; absolute seconds reported. A configuration's timing is usable iff ≤ 1/3 of its triplets are
invalid and it has no identity failure (P1 A4.1).

**Timing-job qualification (P1 A2.3/A3.2):** each subject's untimed scoring run is the recording program, certified
in-job (§4.1), with tight, tighter, $q^+$, $q^{++}$ replays and own-state ρ on its six dev6 cases. A configuration is a
front **candidate** at mesh $N$ iff its accuracy-job label (38/16 populations) **and** its timing-job label (dev6) are
both "numerically qualified" and its timing is usable.

**Dominance (frozen):** for two candidates $X\ne Y$ at the same mesh with costs $c$ and error statistic $e$ (median, or
worst, of $e_i$ over the 38 cases), with $\tau_{XY}=\max\big(2\max(\bar\varepsilon_X,\bar\varepsilon_Y),\,0.02\max(e_X,e_Y)\big)$
($\bar\varepsilon$ = median over cases of $\varepsilon_{{\rm ref},i}$):
$$X\succ Y\iff c_X\le1.05\,c_Y\ \wedge\ e_X\le e_Y+\tau_{XY}\ \wedge\ \big(c_X<c_Y/1.05\ \vee\ e_X<e_Y-\tau_{XY}\big).$$
(Symmetric margins: no pair dominates each other.) The **median front** $F_{\rm med}(N)$ and **worst front**
$F_{\rm worst}(N)$ are the candidates not dominated by any candidate. Front members are reported with their (r, scheme,
Δt) — the front is parameterised by $r$ and need not be monotone in $r$. **Failed or unqualified** configurations are
plotted in a separate panel with their failing label, never on the headline front. Secondary fronts on $c_{\rm ev}$ are
reported.

**`runs/FRONT-2D.json` (deliverable for jcp-fom2d).** Written only after the 2D results audit passes, then frozen
(sha256 recorded in `runs/FRONT-2D.sha256`; any later change is a new versioned file with a dated amendment). Per mesh
and per member of $F_{\rm med}\cup F_{\rm worst}$: the full configuration manifest (r, M, mode-list hash, rule, m,
generator identity, scheme, form, Δt, tolerance level and values, budget, trust, initialisation contract, weighting,
output times, metric id, bank checkpoint/rotation sha256, bank edges, code commit and sha256 of `p1core.py`/
`p1common.py`), the three labels, per-case E0 ST errors and $\varepsilon_{\rm ref}$ (38 cases, case ids), median and
worst, timing (cost $c$, $c_{\rm ev}$, absolute medians, A's identity, GPU type, job id), and which front(s). The
non-member qualified configurations and the unqualified ones are listed too (with labels), so jcp-fom2d can see the
whole candidate set.

**3D front (B5-3D), assembled, no new timing.** From speed3d's audited `analysis-grid.json`: the same dominance rule on
eligible configurations with a valid screening panel; cost = screening ratio × $t_A$ ("speed3d screening contract:
full-field 65³ mesh-path rollout; ratios from separate jobs against C_old, not paired in one job" — the brief's
"mesh-free" wording does not describe the grid's contract; speed3d's mesh-free m1 path covers only four configurations
and is cited separately). Reported: the 3D front of §2.4, each member's Q2-3D label; and the **qualified 3D front** =
the same rule restricted to configurations with Q1 (speed3d) and Q2-3D pass, plus the speed3d configurations whose rule
is the setting's deployed or post-hoc converged rule (cited as "not Q2-tested here"). Failed members in a separate
panel.

## 7. Controls (each with fixture, predicate, and the claim it blocks)

Every control is shown to behave as registered in the real-size smoke (synthetic ones in CPU tests) before any verdict
is read; a misbehaving control blocks its listed claim until fixed and re-audited. Inherited P1 controls are re-run
where the lane's code differs from P1's (drivers), and cited otherwise.

| control | kind | fixture and predicate | blocks |
|---|---|---|---|
| under-resolved rule (2D) | negative | P1 A1.1 fixture (R384 `gauss32` → `gauss64`, LSPG-BE $\Delta t_0$, $1024^2$, dev6, independent evidence `p1-scaling/checks/ctrl-fixture-under-resolved.json`): the lane's Q2 code must return **fail** (rollout > 2.5e-4 on ≥ 1 case, ρ_max > 0.116) | Q2, all fronts |
| under-resolved rule (3D) | negative | W512 `lat8192` → `lat16384`, FS-K1 BE $\Delta t=0.01$ ($a=6$), 923801: independent evidence from combined `SELECTION-3D-POSTHOC.json` (W512 fs\|BE\|0.01\|K1 worst distance to the converged rule: lat8192 4.21e-3, lat16384 2.01e-4, same population, so by the triangle inequality $d(8192,16384)\ge4.0\times10^{-3}$ on the worst case of lat8192): the lane's Q2-3D must return **fail** | Q2-3D |
| Q1 negative, LSPG | negative | P1 A1.6: R384 `gauss128` CN-R $1024^2$ $\Delta t_0$, LM budget 0, dev6, 300 pooled steps; independent NumPy γ > 1e-3 on > 3 steps before use; the lane's Q1 must report pooled > 1 % | Q1 (LSPG) |
| Q1 negative, GAL | negative | GAL-CN R384 `gauss128` $1024^2$ $\Delta t_0$, LM budget 0 (returns the selected predictor), dev6, 300 pooled steps; an independent NumPy Galerkin residual (own QR) must give $\eta_n>10^{-8}$ on > 3 steps before use; the lane's GAL Q1 must report pooled > 1 % and agree per step with NumPy (A1.3) | Q1 (GAL) |
| zero-update | negative | P1: $c_{n+1}=c_n$ on dev6\|0 R384 CN-R (motion of the verified tight trajectory > 100 × 2.5e-4); Q1 distance must exceed 2.5e-4; and the same for GAL-CN | Q1 |
| larger-step solver probe | probe | at $L_s$ the budget-exit and certificate counts per configuration are reported (no required outcome) | — |
| wrong-mesh table | negative | P1: a $256^2$ table fed to the $1024^2$ query must raise | all |
| case / time / reference-kind swap | negative | P1 (CPU, real records): a swapped record must raise; plus a **Δt swap**: a record whose stored $\Delta t$ differs from its configuration key must raise | all accuracy |
| reproduction gate (2D) | positive | the lane's driver at $\Delta t_0$, R384 `gauss128` CN-R and BE, $1024^2$, dev6: coefficients equal P1's `acc1024` archive to ≤ 1e-10 (same numerical configuration; A100) — shows the generalised driver is P1's arithmetic | inherited P1 labels, every new configuration |
| reproduction gate (Δt, form) | positive | combined `c2grid` archives hold prod runs of R128 `fib4181`, R384 `gauss128`, R512 `gauss192` (not R256: archive rule `gauss96`) for LSPG-BE, LSPG-CN-R, GAL-CN at every factor of §2.2 on all 38 cases at $1024^2$ (`W_R128/R384/R512.npz`, keys `<form>|<scheme>|<dt>|prod`). Preliminary in the smoke (dev6, R128 and R512); registered in `acc1024`: **every** new $1024^2$ configuration with $r\in\{128,384,512\}$ equals its archive run to ≤ 1e-10 relative ($\max_k\lVert\Delta W_k\rVert/\lVert W_0^{\rm arch}\rVert$) on all 38 cases, after numerical-manifest equality (P1 `manifest_equal` generalised to Δt and form; A100, the archive's GPU type) | every new Δt / GAL configuration (R256 configurations have no archive gate; they rely on the R128/R384/R512 gates of the same code path) |
| reproduction gate (3D) | positive | §2.4: rerun at rule $q$ equals speed3d's archive ≤ 1e-10 | Q2-3D |
| split = fused, timed = scored, identity, drift | positive / negative | P1 A2.3, A3.2, A4.1 (injected wrong identity hash makes every triplet of that subject invalid) | B5 cost |
| doubled work | negative | P1 AM-8: two separately dispatched, synchronised evolutions of one subject (R384 CN-R $1024^2$) give B/A ∈ [1.8, 2.2] with identical iteration counts | B5 cost |
| front dominance | negative / positive | CPU synthetic: a candidate 10 % costlier and 10 % less accurate than another must be dominated; a pair within both tie bands must both stay; an unqualified configuration that would dominate must be excluded from the headline front | B5 |
| H-acc-N / H-acc-r | negative | P1 CPU synthetics (trend → fail; equal → pass; precedence cases) re-run on the lane's copy | B1 |
| B4 order | positive / negative | §5: manufactured oracle (CPU) and GAL-TH06 must-fail | B4 |
| rank-deficient fixture | negative | P1: duplicated column → Q3 reports rank deficit and cond > 1e12 (CPU) | Q3 |

## 8. Jobs, inventory, resources

| job | GPU | content |
|---|---|---|
| `smkA` | a100-80G | real-size smoke at $1024^2$: generalised accuracy driver on dev2 for one configuration per (scheme, Δt) at R128 and R512 (all Q1/Q2/Q3 paths), both reproduction gates (dev6), every 2D numerical control (under-resolved, Q1-LSPG, Q1-GAL, zero-update, wrong-mesh), B4 mini (one case, both r, all three schemes + TH06), cost probe of the 38-case blocks |
| `smkB` | h200 | timing smoke at $1024^2$ (A + 6 subjects covering every scheme and Δt, dev2, 2 reps; doubled work; injected identity fault) and the $4096^2$ memory pilot (full bank, R512 tables `gauss192/256/320/384`, the $1024^2$ bank for A, one rollout per scheme at R512; peak ≤ 120 GB decimal, else the R512 $4096^2$ configurations use the prefix-retaining builder is **not** available for R512 → they are reported "not reached (memory)") |
| `smk3` | h200 | 3D smoke: speed3d context with W512 and W1024, reproduction gate and Q2-3D on cases 0–3 for configurations 14 and 16, the 3D under-resolved control (all 64 cases: cheap), memory pilot with W1024 `lat131072` |
| `acc256`, `acc1024` | a100-80G | 2D accuracy + Q1/Q2/Q3, all new configurations at the mesh (38 cases) |
| `acc4096` | h200 | same at $4096^2$, plus the 22 extra cases of P1's R128/R256 Δt₀ configurations |
| `time256`, `time1024`, `time4096` | h200 | per-mesh timing (36 subjects) with timing-job certification |
| `order` | a100-80G | B4 (§5) |
| `q3d` | h200 | Q2-3D for the 17 configurations of §2.4 (+ the 3D control) |
| `fb<mesh>` | per mesh | registered fallbacks (§2.3), only if needed |

**Budget.** Hard cap 35 GPU-h including contingency; P1's `ledger.py` pattern: a job is submitted only if spent +
Σ time limits of queued/running lane jobs + this job's time limit ≤ 35 (A1.11, A2.6), the lane has < 3 jobs
queued/running, and `squeue -u tawal01 -h | wc -l` < 8. Time limit = 1.5 × a smoke-derived projection (P1
`resources.py` pattern; projections are estimates, the scheduler limit is the enforced bound). A-priori estimate
(re-priced by the certificate): smokes 1.5, acc256 1.5, acc1024 1.5, acc4096 2, time ×3 2.5, order 0.5, q3d 1.5,
fallbacks 2 ≈ 13 GPU-h.

**Defer order** (brief; first dropped first): $4096^2$ larger-step arms → GAL-CN diagnostic → 3D Q2 for non-winner front
members → $256^2$ larger-step arms. **Never dropped:** the front at $1024^2$ and $4096^2$ for $\Delta t_0$/$2\Delta t_0$
CN-R and BE. Deferred items are reported, never silently omitted.

## 9. Analysis and deliverables

`qscore.py` (CPU): E0 scoring with identity assertions (incl. Δt), offline recomputation of every Q1/Q2 quantity with the
independent NumPy decode, configuration labels (generalised P1 `p1score.qualify`). `qstats.py`: P1 `p1stats` copy
(H-acc-N/H-acc-r, TOST, bootstrap) + the dominance function. `analyze.py` → `runs/EXPORT.json` (every label, verdict,
front, key number; `contract` field on every cost number). `make_report.py` → `report.md` + PNG plots: per-mesh front
(median and worst), failed/unqualified panel, error vs $r$ per mesh and family, error vs $N$ per $r$, observed orders,
conditioning vs $r$, 3D front with Q2-3D labels. `runs/FRONT-2D.json` as §6. `runs/GPU-LEDGER.json`. No hand-typed
number; Codex audits at every gated step (design, code, each job, results, report).

## 10. Code (copy, then edit in the lane; `vendor/` untouched)

Copied unchanged from P1 (`exp/2026-10-17-jcp-p1-scaling` @ `5942f63ef`; sha256 in `PROVENANCE.json`): `p1core.py`,
`p1common.py`, `p1stats.py`, `npcheck.py`, `ledger.py` (cap and namespace edited), `cluster/{stage,submit,collect}`
(namespace, job-name prefix `jq_`, cap edited), `make_fixtures`-style helpers as needed. New: `qcert.py` (Galerkin
certifier and Galerkin Jacobian), `qacc.py` (generalised accuracy driver: any Δt, LSPG and GAL, 38 cases, controls,
B4 blocks), `qtime.py` (generalised per-mesh timing driver), `qscore.py`, `qstats.py` (dominance), `front3d.py`,
`q3d/` (copies of the speed3d modules needed for the 3D replays — `c3common`, `s3common`, `s3ctx`, `fs3`, `fs3p`, `qn3`,
`t2core`, `timing`, `replay`, rule files — with provenance from `exp/2026-10-14-jcp-speed3d` @ `691465d88`, plus the
new `q3run.py`), `test_q.py` (CPU controls), `test_lmm.py` (jcp-time2 copy), `manifest.py`, `resources.py`,
`analyze.py`, `make_report.py`, `make_configs.py`.

## 11. Integrity rules carried

Never quote a pre-September Burgers speed-up; never mention the heat POD-128 comparison; retractions stated plainly.
Every GPU number from the cluster; local GB10 only for CPU analysis (`JAX_PLATFORMS=cpu`) and at most one `jaxrun`.
Codex audits: read-only prompts, no large copies (≤ 2 GB scratch under `/tmp/jcp-qualify-audit-scratch/`, deleted after
each round), `df -h /home` ≥ 60 GB before each audit. Waits on PIDs with an overall deadline.
