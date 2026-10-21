# imp-hrcost (E6, ideation W3): a cheaper nonlinear term — bank-tested Galerkin with classical rules (F2) and fitted continuous ECSW weights on our off-mesh points (F3) — pre-registration

Status: pre-registration, written 2026-10-21 before any lane code or data. Binding sources: campaign
`COMMON-BRIEF.md`, lane brief `briefs/imp-hrcost.md`, ideation report `reports/2026-10-20-improvement-ideation/report.md`
§4 (F2 + F3) and §5 row W3. Every accuracy number this lane produces is a **conditional E0 estimate** (E0 second-order
reference $R_{ST}$, conditional Richardson $\varepsilon_{\rm ref}$; E0 A10, combined A10/A11 and the retracted A15 labels
carry forward unchanged). Amendments are appended at the end, dated; this text is never rewritten.

## 0. Question

Problem 2 of the campaign (tuned POD + ECSW dominates our qualified 2D configurations): after E5 (lane imp-faststep)
replaces the LSPG Newton loop by a precontracted Galerkin step, the remaining per-step cost of our ROM is the tested
advection evaluated on $m$ off-mesh quadrature points ($m = 16384$, Gauss $128^2$, at $r=384$, against 3072 ECSW nodes).
**Can either (F2) changing the test space from $M=4r$ sines to the $r$ bank functions themselves, or (F3) replacing
the classical rule by sparse non-negative weights fitted on our own off-mesh candidate points, cut $m$ — and the
end-to-end query time — while keeping full quadrature (Q2) and solver (Q1) qualification and the worst E0 error?**

Outcomes are reported for F2, F3 and F2+F3 separately. "Fixed" is not claimed by this lane for Problem 2 as a whole;
it reports whether each lever passes its own pre-registered stop rule.

## 1. The reduced model and the four arms

Bank coefficients $c\in\mathbb R^{r}$, $u(x)=G(x)c$, $r=384$, frozen deployed bank (checkpoint `dn256b`, $R=512$, $K=16$,
rotation `rotation_R512`, ordered nested prefix, exactly as in jcp-qualify/E5). Viscous Burgers
$u_t + u(u_x+u_y) = \nu\Delta u$ on $(0,1)^2$, homogeneous Dirichlet, horizon $0.25$, outputs at
$t=0,0.05,\dots,0.25$. Write $u_q=(G_qc)$, $s_q=(G^{x}_q+G^{y}_q)c$ at the rule points $x_q$, $q=1..m$.

Every arm is a **precontracted Galerkin ODE**
$$\dot c = f(c) = -P^{\mathsf T}\big[(G_qc)\odot(G_sc)\big] + \mathcal L\,c ,$$
with an arm-specific $m\times r$ contraction table $P$ and $r\times r$ linear operator $\mathcal L$, all built offline:

| arm | test space | rule (weights) | $P$ | $\mathcal L$ |
|---|---|---|---|---|
| **R** (reference = E5's precontracted arm) | $M=4r$ discrete sines, $A=QR_m$ | classical Gauss $128^2$ | $\Psi QR_m^{-\mathsf T}$, $\Psi = L\,w\,\psi$ | $-\nu R_m^{-1}Q^{\mathsf T}\Lambda A$ |
| **F2** bank-tested, classical | the $r$ bank functions | classical Gauss or Fibonacci rule, smallest passing $m^\*$ | $\operatorname{diag}(w)\,G_q\,\mathcal M^{-1}$ | $-\nu\,\mathcal M^{-1}\mathcal K$ |
| **F3** sine-tested, fitted | as R | sparse NNLS weights $\hat w\ge0$ on the Gauss $128^2$ candidates, support $\kappa r$ | $\operatorname{diag}(\hat w)\,(L\psi)\,QR_m^{-\mathsf T}$ (support rows only) | as R |
| **F2+F3** | as F2 | sparse NNLS weights on Gauss $128^2$ candidates, support $\kappa r$ | $\operatorname{diag}(\hat w)\,G_q\,\mathcal M^{-1}$ (support rows only) | as F2 |

- **Bank mass and stiffness** $\mathcal M=\int_\Omega G^{\mathsf T}G$, $\mathcal K=\int_\Omega\nabla G^{\mathsf T}\nabla G$ are
  computed once offline with Gauss $384^2$ (bank values and gradients by forward-mode AD, `qcore.Model.values_grads`)
  and checked against Gauss $320^2$: relative Frobenius difference must be $\le10^{-10}$ (else the next rule up,
  $448^2$, is used and the check repeated). The linear terms are therefore exact to rule convergence ("exact linear
  terms", lesson of 2026-08-26); only the advection is hyper-reduced. The bank is Dirichlet-exact through its envelope,
  so the weak Laplacian needs no boundary term. $\operatorname{cond}(\mathcal M)$ is recorded (ideation: 2.65 at r = 384).
- R's linear operator uses the mesh-discrete eigenvalues $\Lambda$ and the mesh-projected $A$ at the run mesh, as the
  deployed configuration; F2's uses continuum $\mathcal M,\mathcal K$. This is a change of test space and is part of
  what F2 tests.
- **Mode list:** R and F3 use the mode list and policy E5 hands off for its precontracted arm (expected: jcp-qualify's
  `frozen` list, $M=1536$, hashed). The hash is recorded in every output.
- **Initialisation and decode are identical in every arm** (the deployed contract: full-field input sampled at the
  cold-start points, least-squares fit; full-field decode on the run mesh). Only $P$, $\mathcal L$ and the point set differ.
- Cost per advection evaluation: $2mr$ ($G_qc$) + $2mr$ ($G_sc$) + $2mr$ ($P^{\mathsf T}$) $=6mr$, the same formula for
  every arm, so any F2/F3 saving must come from a smaller $m$ (point count of the arm's own rule or support).

### 1.1 Time integrator (dependency on E5)

F2, F3 and F2+F3 run in **exactly the stepper code of E5's precontracted Galerkin arm**, copied into this lane with a
provenance note (source commit + sha256), with only the tables $(G_q, G_s, P, \mathcal L)$ swapped. The primary integrator
is the one E5 hands off as arm 2 (**precontracted GAL-CN at $\Delta t_0=0.005$**, 50 steps). If E5's hand-off also
includes its ETDRK4 arm (arm 3) at $\Delta t=0.0125$, a **secondary** panel repeats R/F2/F3 with it if the GPU-hour
projection allows (defer order §9); stop rules are evaluated on the primary integrator only, and the secondary is
reported as a result.

If E5's stepper interface cannot take generic tables, a generalisation is written in a new lane file and must reproduce
E5's own code on arm R (same inputs) to $10^{-12}$ relative in every output state (positive control P1, §6); E5's code
is then still timed on R inside the same job as a provenance check.

If E5's precontracted arm fails its own stop rule, it is still R here, and the report says so.

## 2. Rules, candidate pool, fitting

### 2.1 Classical ladders (F2, and R's own Q2)

- Gauss tensor rules $p^2$, $p\in\{64,80,96,112,128\}$ as candidates for $m^\*$, with successors/checks from
  $\{160,192\}$: ladder $64\to80\to96\to112\to128\to160\to192$; successor $q^+$ = next rung, check $q^{++}$ = the rung
  after.
- Fibonacci lattices (`fib<n>`, seed 0) $n\in\{4181, 6765, 10946\}$ as candidates, ladder
  $4181\to6765\to10946\to17711\to28657$.
- **$m^\*$** = the smallest point count among candidate rules ($m\le16384$) whose F2 configuration is **numerically
  qualified** (Q1 and Q2 pass, §3). Ties in $m$ do not occur in these ladders. Non-monotone passes are reported.
  If no candidate with $m\le16384$ qualifies, $m^\*>16384$ and the F2 stop rule fires.
- R's own Q2 is recomputed in-job with $q^+=$ Gauss $160^2$, $q^{++}=$ Gauss $192^2$ (as jcp-qualify).

### 2.2 Fitted rules (F3, F2+F3)

- **Candidate pool:** the Gauss $128^2$ points ($m_c=16384$) — the R rule's own points.
- **Fitting data — training trajectories only.** 16 training draws (`hrcore.train_draw_2d(seed=0, count=576)`, rows
  0–15, the bank's training distribution; disjointness from dev6/val32/test64 asserted, control C7) are rolled out with
  the **parent classical arm of the same projection** (R for F3; F2 at Gauss $128^2$ for F2+F3) at 1024²; the states at
  $t=0,0.025,\dots,0.25$ (11 per draw, 176 snapshots $c_j$) are the fitting snapshots. No dev6/val32 state is used.
- **ECSW system.** Per candidate point $q$ and snapshot $j$, the contribution to $\dot c$ is
  $g_{qj}=p_q\,u_q(c_j)s_q(c_j)\in\mathbb R^r$, with $p_q$ the unweighted row of $P$ (R: $L\psi(x_q)^{\mathsf T}QR_m^{-\mathsf T}$;
  F2: $G(x_q)\mathcal M^{-1}$). Target $b_j=\sum_q w^{\rm G}_q g_{qj}$ (the parent rule). Each snapshot block is scaled
  by $1/\lVert b_j\rVert$ (so small-amplitude late states count). Solve
  $\min_{\hat w\ge0}\lVert C\hat w-b\rVert$ with $C\in\mathbb R^{(176r)\times m_c}$ by the greedy Lawson–Hanson NNLS of
  jcp-hr-baselines (`nnls.py`, copied with provenance), recording the **first crossing** of each support target
  $s=\kappa r$, $\kappa\in\{4,8,16\}$ (1536 / 3072 / 6144 points), as jcp-hr-baselines A15.1/A16. Fit residual
  $\lVert C\hat w-b\rVert/\lVert b\rVert$ is recorded but never used to certify a rule (lesson 2026-08-25).
- **κ\*** = the smallest κ whose fitted arm is numerically qualified (Q1 + Q2).
- Optional (defer order §9): a **sketching control** — the same NNLS on a row sketch (Gaussian sketch of $C$ to
  $4\kappa r$ rows) — reported descriptively.

## 3. Qualification (common brief §3)

### 3.1 Q1 — solver qualification

For the primary integrator (GAL-CN, a root-finding step) the certificate is the **form-appropriate relative Galerkin
residual** of jcp-qualify §4.1b transcribed to the precontracted coordinate form, rebuilt from the saved internal
trajectory with a **fresh forward-mode Jacobian at every step**: $\eta_n = \lVert\rho_n(c_{n+1})\rVert/\lVert c_n\rVert$
with $\rho_n(c)=c-c_n-\tfrac{\Delta t}{2}[f(c)+f(c_n)]$; non-converged if $\eta_n>\eta_{\rm bar}$ or non-finite,
where $\eta_{\rm bar}$ is **100 × the production tolerance E5 registers** (jcp-qualify: $10^{-8}$ for $10^{-10}$). If E5
registers a different, audited certificate for its arm, that certificate is used verbatim and this paragraph is
superseded by an amendment. The literal stationarity ratio $\gamma_n$ is also reported, without a verdict (it does not
certify a root; jcp-qualify 4.1b). The step Jacobian's condition number is recorded (Q3, descriptive).

Configuration-level Q1 (every arm, all 38 cases of dev6 + val32 at 1024²):
1. every run valid: finite outputs, no non-finite step, no damping-saturation exit;
2. non-converged steps ≤ 1 % pooled over all steps of all cases;
3. scoring-lattice relative distance to the **tight** solve of the same configuration ≤ $2.5\times10^{-4}$ on every case
   (tight anchors on every case);
4. tight/tighter agreement ≤ $2.5\times10^{-5}$ on dev6;
5. offline NumPy recomputation of each in-job distance agrees to $10^{-9}$ relative + $10^{-13}$.

For a secondary explicit integrator (ETDRK4), Q1 uses E5's audited certificate for it; without one, the secondary panel
is reported as "Q1 not applicable / unqualified" and not used for any verdict.

### 3.2 Q2 — quadrature qualification on the arm's own reached states

Population: dev6 + val32[0:10] (16 cases) at 1024².
- **Rollout bar:** replay with the successor rule $q^+$ (classical rules: next ladder rung; fitted rules: the parent
  Gauss $128^2$ rule of the same projection), everything else equal; the max over the five evolved outputs of the
  scoring-lattice relative distance ≤ $2.5\times10^{-4}$ on **all 16** cases; every successor run valid.
- **ρ bar:** on the arm's own six output states × 16 cases,
  $\rho=\lVert v_q(c)-v_{q^+}(c)\rVert/\lVert v_{q^+}(c)\rVert$ with $v(c)=P^{\mathsf T}[(G_qc)\odot(G_sc)]$ the advection
  contribution to $\dot c$ (the only part of the rule that enters the dynamics); $\rho_{\max}\le0.116$. For the
  sine-tested arms the unprojected $M$-vector $\rho$ (jcp-qualify's) is also reported, without a verdict.
- **Check-rule convergence (dev6):** $q^+$ vs $q^{++}$ rollout bar (classical: next-next rung; fitted: Gauss $128^2$ vs
  $160^2$ of the parent projection); otherwise "unverified (finer rule not converged)".
- Labels pass / fail / unverified. **Numerically qualified = Q1 pass and Q2 pass.** The other 22 cases are not described
  as individually Q2-tested.

### 3.3 Three labels kept separate
(1) numerical qualification (Q1+Q2); (2) role qualification: "cheaper at equal worst error" (the stop-rule predicates
of §5); (3) frontier usefulness: not dominated in (median end-to-end time, worst E0 error) among R, F2(m\*), F2 at
Gauss $128^2$, F3(κ), F2+F3(κ), descriptive.

## 4. Metrics

- **Error:** $e=\max_{k=1..5}\lVert u(t_k)-R_{ST}(t_k)\rVert/\lVert R_{ST}(t_0)\rVert$ on the $257^2$ lattice, through
  `refstore` (E0 store version pinned), eligibility per case-kind from the store; $R_S$ never used (no BE arm here);
  every record asserts its identity (cohort, case, reference kind, sha256, store version, metric id, arm, rule).
  Worst error = max over the eligible cases of dev6 + val32 (38 cases); median reported too.
- **"No worse" (equal worst error):** arm X is no worse than arm Y iff
  $e_{\rm worst}(X)-e_{\rm worst}(Y)\le2\,\varepsilon^\*$, $\varepsilon^\*=\max(\varepsilon_{\rm ref}$ of X's worst case,
  $\varepsilon_{\rm ref}$ of Y's worst case$)$. Per-case comparisons use the 2ε_ref tie rule.
- **Timing (common brief §5):** one job, one GPU, f64, `JAX_DEFAULT_MATMUL_PRECISION=highest`, warm JIT, 3 burn-in
  queries, then for every one of the 38 cases a paired **A–B–A** sequence (A = R, B = the arm), 5 repetitions per
  sequence, `block_until_ready` on outputs; per-case time = median over repetitions; per-case ratio
  $S_{\rm case}=\tfrac12(t_{A1}+t_{A2})/t_B$. Reported: median over cases of $S_{\rm case}$ and a case-bootstrap 95 %
  interval (2000 resamples, seed 0). Two quantities: **end-to-end query** (init + evolution + full-field decode at
  1024²; primary for the stop rules, as the ideation's kill criterion) and **evolution only** (descriptive).
  Setup cost (table builds, $\mathcal M,\mathcal K$, rollouts for fitting, NNLS) is recorded and reported, never folded
  into the query time.
- **"Faster":** median $S_{\rm case}>1$ **and** the bootstrap lower bound $>1$.
- Times are not compared with any other job or GPU type. The R arm is timed in this job; E5's numbers are not reused.

## 5. Stop rules (verbatim from the brief, then their operational form)

Verbatim (lane brief):
- *Stop F2 if $m^\* > 16384$, accuracy is worse, or it is not faster.*
- *Stop F3 if no κ ≤ 16 passes Q2 at equal worst error with a ≥ 1.5× gain.*

Operational:
- **F2 stop fires** iff (a) no candidate rule with $m\le16384$ makes F2 numerically qualified, or (b) F2 at $m^\*$ is not
  "no worse" than R in worst error (§4), or (c) F2 at $m^\*$ is not "faster" than R end to end (§4).
- **F3 stop fires** iff no κ ∈ {4, 8, 16} gives an F3 arm that simultaneously (i) is numerically qualified (Q2 and Q1),
  (ii) is "no worse" than R in worst error, and (iii) has median end-to-end $S_{\rm case}\ge1.5$ with bootstrap lower
  bound $>1$.
- **F2+F3** has no stop rule of its own; it is reported against R with F3's three predicates, and against F2($m^\*$).
- If F2's stop fires through (a), F2+F3 still runs (its pool is Gauss $128^2$ regardless).

## 6. Controls (each must be shown to fire in the real-size smoke, 1024², r = 384)

| id | control | predicate that must fire | claim it protects |
|---|---|---|---|
| C1 | coarse rule: F2 and R with Gauss $16^2$ (successor $24^2$) | Q2 rollout bar **and** ρ bar fail | Q2 can fail on real data |
| C2 | shuffled weights: the κ = 8 F3 rule with its weights randomly permuted over its support | Q2 fails | a fitted rule is not passed by construction |
| C3 | corrupted state: one saved $c_{n+1}$ perturbed by $10^{-6}\lVert c_{n+1}\rVert$ before certification | η flags that step (and only that step) | the Q1 certificate reads the states |
| C4 | NaN injected in one entry of $P$ | run flagged invalid, Q1 fails | validity gate |
| C5 | duplicated rule: R's points each repeated twice with half weight | outputs equal R's to $10^{-12}$; timed $S_{\rm case}$ median $\le 1/1.3$ | the timer resolves a 2× cost change |
| C6 | A–A timing: R against itself | median $S_{\rm case}\in[0.9,1.1]$ (positive) | no order bias in A–B–A |
| C7 | fitting leakage: a fit set containing a dev6 draw | assertion raises | fitting uses training draws only |
| C8 | scoring identity: a field scored under a swapped case id | `IdentityError` raised | E0 records match their request |
| P1 | equivalence: F3 machinery with the full parent weights ($\hat w=w^{\rm G}$, all 16384 points) | equals R to $10^{-12}$ relative | F3 code path = R code path |
| P2 | equivalence: lane stepper vs E5's code on R (only if a generalisation was needed, §1.1) | $\le10^{-12}$ relative | the tables are the only change |
| P3 | $\mathcal M,\mathcal K$ rule convergence (§1) | $\le10^{-10}$ | exact linear terms |

A control that cannot fire on real data is a bug; the smoke acceptance script recomputes each predicate from raw outputs.

## 7. Cohorts and scoring contract

dev6 + val32 at 1024² only (selection of $m^\*$ and κ\* on these dev/validation cohorts, as the brief allows). No
test64, no sealed or reserved-OOD cohort, never `sealed/` or `custody/`. Fitting on training draws only (C7). Scoring on
CPU, offline, through the E0 loader. Cases are the statistical unit; time snapshots are never bootstrapped.

## 8. Jobs, GPU-hours

- **Smoke** (1 job, H200 or A100-80G): tables for R, F2 at Gauss $96^2$ and $128^2$, F3 κ = 8 fit, all controls,
  dev6 only, timing on 6 cases; memory pilot from device bytes. Estimate 0.3 GPU-h.
- **Production** (1 job, same GPU type as available, all arms in the same job so timings are paired): ladders and Q2
  (16 cases × ≈ 12 rule configurations × rollouts at $q,q^+,q^{++}$), Q1 tight/tighter anchors, 38-case production runs
  for R, F2($m^\*$), F2 Gauss $128^2$, F3 κ ∈ {4, 8, 16}, F2+F3 κ ∈ {4, 8, 16}; paired timing for each against R.
  Estimate 0.8–1.2 GPU-h (rollouts are $\lesssim1$ s each; NNLS fits minutes).
- Cap 3 GPU-h, lane concurrent-job cap 1; a smoke-derived cost certificate gates production submission (spent +
  projection ≤ cap).
- **Defer order** if the cap binds: (1) sketching control; (2) secondary ETDRK4 panel; (3) F2+F3 κ = 4; (4) Fibonacci
  ladder.
- Namespace `/cluster/tufts/paralab/tawal01/imp-hrcost/<job>/`; arrays > 100 MB to
  `/cluster/tufts/paralab/tawal01/burgers-archive/imp-hrcost/<job>/` with a committed sha256 listing.

## 9. Deliverables

`DESIGN.md` (this), `audits/`, lane code, `runs/GPU-LEDGER.json`, `checks/*.json`, `make_report.py` → `report.md`
(level-1 title, status, LaTeX, mermaid, inline provisional labels, glossary), `EXPORT.json` (m\*, κ\*, worst/median
errors, $S$ end-to-end and evolution, setup costs, stop-rule outcomes, source paths), PNG plots (error vs end-to-end
time for all arms; m-ladder Q2 distances).

## 10. Hypotheses stated in advance (for the record, not verdicts)

- F2 may need a large $m$ (jcp-dims A4: fewer tests needed more points); if so its stop fires at (a).
- F3's κ r points (1536–6144) are 2.7–11× fewer than 16384; the kernels are near the latency floor, so the end-to-end
  gain will be smaller than the point ratio (ideation estimate, not a result).
- Fitted rules have historically won only at coarse $m$ and were checkpoint-dependent (2026-08-25/26); a Q2 failure of
  F3 at every κ is a plausible outcome.

## Amendment A1 (2026-10-21, before any lane code ran; answers `audits/codex-design-1.md`, verdict WRONG)

Item numbers refer to the audit. Where A1 and the text above conflict, A1 governs.

**A1.1 (item 2) Cost statement.** Replace "any F2/F3 saving must come from a smaller $m$" by: *the leading
advection-value arithmetic ($6mr$) differs between arms only through $m$; projection changes can also change
conditioning, Newton iterations and convergence.* Every arm reports separately: Newton iterations per step (mean, max),
Jacobian evaluations, failed steps, evolution time, end-to-end time, and a step-Jacobian assembly microbenchmark
(A1.6, C5).

**A1.2 (item 3) Stop-rule reading.**
- "Equal worst error" is operationalised as **no distinguishable worsening under the registered conditional tolerance**
  ($2\varepsilon^\*$, §4), not literal equality. The two defining cases and the per-case worsen / tie / improve counts
  (2ε_ref rule, common eligible set) are reported with every such verdict.
- F3's "≥ 1.5× gain" is a **point-estimate criterion** (median paired ratio ≥ 1.5) plus evidence of some speed-up
  (bootstrap lower bound > 1); it does not establish a statistically supported 1.5×. Because three κ are searched, all
  per-arm intervals are labelled **exploratory, in-sample, unadjusted for selection**.
- Each stop rule reports one of: **not fired**; **fired (tested failure)**; **fired (incomplete evidence)** — the
  last when a registered candidate could not be built, fitted or run (construction failure, unreached NNLS support,
  budget). Incomplete evidence is never described as "no such rule exists".
- **$m^\*$ is defined among the registered candidate rules only** (§2.1). Both minima are recorded: $m^\*_{Q2}$ (Q2
  only) and $m^\*$ (Q1 + Q2, the one the stop rule uses), so a solver failure is not misread as a quadrature-size need.
  If the Fibonacci ladder is deferred (§8), the claim is restricted to the Gauss ladder.

**A1.3 (item 4, Q1) Certificate — a registered deviation, not a transcription.** For every arm the CN step residual in
coefficient form is $\rho_n(c)=c-c_n-\tfrac{\Delta t}{2}[f(c)+f(c_n)]$. The certificate is the **relative Galerkin
residual in the arm's own trial-space $L^2$ Gram norm**
$$\eta_n=\frac{\lVert\rho_n(c_{n+1})\rVert_{\mathcal G}}{\lVert c_n\rVert_{\mathcal G}},\qquad
\lVert v\rVert_{\mathcal G}=\sqrt{v^{\mathsf T}\mathcal Gv},\quad
\mathcal G=\begin{cases}A^{\mathsf T}A=R_m^{\mathsf T}R_m & \text{sine-tested (R, F3)}\\ \mathcal M & \text{bank-tested (F2, F2+F3)}\end{cases}$$
For the sine-tested arms this **equals** jcp-qualify's $\eta=\lVert Q^{\mathsf T}\mathcal R_n\rVert/\lVert Ac_n\rVert$
(since $Q^{\mathsf T}\mathcal R_n = R_m\rho_n$ and $\lVert Ac_n\rVert=\lVert R_mc_n\rVert$); for the bank-tested arms it is
the same quantity (relative $L^2$ norm of the residual field $G\rho_n$) in the bank's Gram. This is registered here as
this lane's certificate. Bar $\eta_{\rm bar}=10^{-8}$; non-finite or $\lVert c_n\rVert_{\mathcal G}=0$ → not certified
(counted). Certification runs **in the allocated GPU job**, from every saved internal state $c_0..c_S$ of every
production run, with a **fresh forward-mode AD Jacobian** of $\rho_n$ at every step (its condition number recorded,
Q3). **G-cert** is restored: the in-job certificate must agree with the solver's own recorded per-step residual
(converted to the same norm, or recomputed by the certifier in the solver's own norm when the solver uses another one)
to $10^{-6}$ relative $+10^{-13}$. Solver settings (if the E5 stepper does not register its own): Newton on $\rho_n$
with the exact Jacobian, predictor $2c_n-c_{n-1}$, budget 50 iterations, production tolerance $10^{-10}$ in the same
relative Gram norm; **tight** $10^{-13}$, **tighter** $10^{-15}$ (jcp-qualify GAL levels); a tight or tighter anchor
that is invalid (non-finite, budget exit without tolerance) makes its case Q1-unqualified. If E5 registers and audits
a different certificate for its stepper, Amendment A2 states which one is used; this definition is then computed too and
reported.

**A1.4 (item 4, Q2) ρ gate.** The gating ρ is measured on the **Galerkin-tested advection**, in the same Gram norm:
$$\rho=\frac{\lVert v_q(c)-v_{q^+}(c)\rVert_{\mathcal G}}{\lVert v_{q^+}(c)\rVert_{\mathcal G}},\qquad v(c)=P^{\mathsf T}[(G_qc)\odot(G_sc)],$$
which for the sine-tested arms is $\lVert Q^{\mathsf T}(N_q-N_{q^+})\rVert/\lVert Q^{\mathsf T}N_{q^+}\rVert$ (the
tested vector projected on the orthonormal Galerkin test basis; no $R_m^{-1}$ weighting), and for the bank-tested arms
the relative $L^2$ error of the projected advection. **Stated deviation:** jcp-qualify gated on the full $M$-vector
$N(c)$, because its LSPG arms use all $M$ tests. A Galerkin arm uses only $Q^{\mathsf T}N$; an error in
$\ker Q^{\mathsf T}$ does not enter its dynamics. The bar 0.116 is a borrowed threshold and is not claimed to transfer
with a calibrated meaning; the rollout bar ($2.5\times10^{-4}$, every case, own states) is the dynamic check. The
inherited full-$M$-vector ρ is computed for R and F3 and **reported next to the gate**, and any arm that passes the gate
but has inherited ρ > 0.116 is flagged "passes Galerkin-tested ρ only". ρ is recomputed offline with independent NumPy
tables on the dev6 states (A1.3-style agreement $10^{-9}$ rel). Zero-denominator states: both zero → excluded and
counted; denominator zero, numerator non-zero → +inf; non-finite → inf. Successor and check runs must be valid, else
the label is "fail" (successor) or "unverified" (check). For a fitted rule, Q2 certifies **compression relative to the
parent rule**; the parent's own Gauss $128^2$ → $160^2$ (rollout) and $160^2$ → $192^2$ (check) convergence is required
in addition (a fitted arm cannot pass Q2 if its parent is not converged).

**A1.5 (item 5) Fitting, frozen now.** "Support-targeted greedy NNLS" (Lawson–Hanson path of `nnls.greedy_nnls`,
single-index additions, `frac = 0`, `max_drops = 3`, `refactor_every = 1024`, rank tolerance $10^{-12}$, cap 6400,
deadline 40 min per fit; first crossings are not global optima). Draw table: the full
`train_draw_2d(seed=0, count=576)` (sha256 recorded), rows 0–15 (hashed); disjointness checked on the descriptors
(C7). If a parent rollout of a draw is invalid it is replaced by the next unused row (16, 17, …), recorded; more than 4
replacements → that fit is "not built" (incomplete evidence). Snapshot blocks with $\lVert b_j\rVert\le10^{-12}\max_j\lVert b_j\rVert$
are dropped (counted); any non-finite entry aborts the fit (incomplete evidence). Rank-deficient appends are refused by
the solver's rank test. A κ whose support target is not reached within the cap/deadline is "not reached" (incomplete
evidence for that κ). No retry with other settings; no fitting choice is revisited after validation data are seen. Two
fits in total (sine-tested pool for F3, bank-tested pool for F2+F3), each one path to $s=6144$, giving the κ = 4, 8, 16
first crossings. Each fit records rows, columns, bytes, wall time, final residual and support.

**A1.6 (item 6) Controls, revised.**
- **C1** fixture: the coarsest of Gauss $\{16^2,24^2,32^2\}$ (successor: the next of $\{24^2,32^2,48^2\}$) whose own
  and successor runs are valid on all 16 cases, for R and F2. Must fire: the rollout bar **and** the ρ gate, each
  shown separately. If either does not fire, the fixture is defective and is reported; the code is investigated
  before any verdict.
- **C2**: κ = 8 F3 weights permuted over the support (numpy seed 0); fixture validity: permuted weights differ from the
  fitted ones by ≥ 50 % relative $\ell_2$. Must fail Q2 (rollout or ρ). It tests sensitivity to the weight–point
  association, not independence from fitting.
- **C3** (replaces the old C3): (a) perturb the **final** saved state $c_S$ by $10^{-6}\lVert c_S\rVert$ along a fixed
  random unit direction (seed 0): exactly step $S-1$ must be flagged; (b) perturb an interior state $c_{k}$, $k=25$:
  steps $k-1$ and $k$ must be flagged and no others.
- **C4**: NaN in $P[0,0]$ → the production validity path marks the run invalid and Q1 fails.
- **C5** (replaces the old C5): duplicated rule (every R point twice, half weight): outputs equal R's to $10^{-12}$
  relative; a **step-Jacobian assembly microbenchmark** ($P^{\mathsf T}[\operatorname{diag}(s)G_q+\operatorname{diag}(u)G_s]$,
  $4mr^2$ flops, 200 timed calls after 20 warm calls) shows duplicated / original median time ≥ 1.5. Query-time
  ratios of the duplicate are reported descriptively only.
- **C6** A–A: R against R, median paired ratio in [0.9, 1.1] (sanity check only).
- **C7**: a fit set containing the descriptor row of dev6 case 0 → assertion raises (descriptor-level check).
- **C8**: a field scored under a swapped case id (requested identity kept) → `IdentityError`.
- **P1**: F3 machinery with all 16384 parent points and parent weights equals R to $10^{-12}$ relative (outputs).
- **P2**: (only if a lane generalisation of E5's stepper is used) equals E5's code on R to $10^{-12}$.
- **P3**: $\mathcal M,\mathcal K$ rule convergence ≤ $10^{-10}$; symmetry (exact by construction) and positive
  definiteness ($\lambda_{\min}(\mathcal M)>0$, $\lambda_{\min}(\mathcal K)>0$) checked. Ceiling: Gauss $448^2$; if the
  bar still fails, F2 and F2+F3 are "not built".
- **P4** (new): the precontracted sine RHS equals the uncontracted jcp-qualify form
  $-R_m^{-1}Q^{\mathsf T}[N(c)+\nu\Lambda Ac]$ (vendored `tested_value`, `form_data`) to $10^{-12}$ relative at the six
  dev6 output states; analytic step Jacobian equals the AD Jacobian to $10^{-10}$ relative.
- **P5** (new): independent bank operators: $\mathcal M$ and $\mathcal K$ against the run-mesh sums
  $h^2\sum_{ij}G^{\mathsf T}G$ and $h^2\sum G^{\mathsf T}(-\Delta_h G)$ (five-point Laplacian on the decoded bank,
  1024²) to relative $10^{-4}$ ($\mathcal M$) and $10^{-3}$ ($\mathcal K$); and the device $P$ of a bank-tested rule
  against a host NumPy solve from the same $G_q$, $w$, $\mathcal M$ to $10^{-12}$.
- **P6** (new): an invalid tight anchor (budget 1 iteration at tight tolerance) is rejected by the Q1 code.
"A control that cannot fire" is reported as either an implementation bug or a defective fixture, after investigation.

**A1.7 (item 7) Timing.** Burn-in of 3 queries of each arm of a pair **immediately before each timed A–B–A block**, and
all fitting finished before the first timed block. Every repetition time, the A/B order, device name/UUID and the count
of repetitions > 3× the case median are saved. The timed invocations use the same tables (fingerprint) as the scored
production runs; the timed outputs are checked equal to the scored outputs (to $10^{-12}$). The statistic is the
**median over cases of per-case paired ratios** $S_{\rm case}$; ratios of separate medians are also reported, labelled.
Worst and median errors use a **common eligible case set** (E0 eligibility ∩ the 38 cases); an invalid run on an eligible
case makes that arm's worst error $+\infty$. **Scope (COMMON-BRIEF §5):** this lane is an ablation *inside our method*
against E5's arm; it makes **no claim against any competitor** (POD-ECSW, spectral FOM). Any later comparison with
E5's POD-ECSW result is cross-job and labelled so.

**A1.8 (item 8) E5 dependency.** HANDOFF-E6.md is polled at least every 30 min with an overall deadline
**2026-10-23T16:30Z** (48 h after lane start); until it exists the heartbeat says
`"waiting_on": "handoff: imp-faststep HANDOFF-E6.md"`. On hand-off, **Amendment A3 (mandatory, Codex-audited; numbered A2 before A2 was taken)**
reconciles the interface, tolerances, certificate and mode list before any dependent code runs at real size. If the
deadline passes without a hand-off, the lane's own precontracted GAL-CN stepper (A1.3 settings) is R, the report
says "E5 hand-off not received; R is the lane's own precontracted GAL-CN", and P2 does not apply. §1.1's "positive
control P1" reads "P2". The "defer order §9" references read §8.

**A1.9 (item 8) Resources.** The smoke measures each phase (table builds, $\mathcal M/\mathcal K$, parent training
rollouts, one NNLS fit to the κ = 16 target with its actual device bytes, Q2 replays, certification, timing per query)
and a cost certificate (`resources.py`) projects production from those measurements; production is submitted only if
spent + projection ≤ 3 GPU-h, reserving all mandatory items (R, F2 ladder Gauss, F3, controls) first; deferrable items
(§8 defer order) are dropped in that order. NNLS memory: $C$ ≈ 8.9 GB plus a QR buffer ≈ 4.4 GB (cap rounded to 8192)
per fit; one fit is resident at a time.

**A1.10 (item 9) Identities and edge cases.** Every output records: checkpoint sha256, rotation sha256, mesh hash,
mode-list hash, rule sha256 (points+weights), cold-start table sha256, $\mathcal M/\mathcal K$ sha256, table
fingerprints, stepper source sha256. **If R is not numerically qualified**, F2/F3 verdicts against it are reported as
"diagnostic: reference unqualified" and no stop rule is reported as "not fired". The optional "sketching control" is
renamed **row-sketched fit (additional fitting experiment, optional)**; it is not the ideation's sketched-LSPG control
and is first in the defer order.

## Amendment A2 (2026-10-21, before any lane code ran; answers `audits/codex-design-2.md`, verdict WRONG on full Q2)

The E5 reconciliation amendment that A1.8 calls "A2" is renumbered **A3**.

**A2.1 Binding Q2 = the inherited tested-vector contract (replaces A1.4's gate).** For every arm, Q2 (binding, used by
every stop rule) requires, on the 16-case population: the rollout bar ($2.5\times10^{-4}$, every case, valid successor),
the check-rule convergence on dev6, and the **inherited tested-vector ρ gate** $\rho_{\max}\le0.116$ on the arm's own six
output states, where the tested vector is
- sine-tested arms (R, F3): the full $M$-vector $N(c)=\Psi^{\mathsf T}[(G_qc)\odot(G_sc)]$ (exactly jcp-qualify's ρ);
- bank-tested arms (F2, F2+F3): the bank-tested vector $T(c)=G_q^{\mathsf T}W[(G_qc)\odot(G_sc)]\in\mathbb R^r$ (the same
  object with the $r$ bank functions as the test functions); Euclidean norms in both cases.
A1.4's Galerkin-tested (Gram-norm projected) ρ is kept as a **non-binding** indicator, label "Q2-projected". The report
gives every stop-rule outcome under binding Q2, and, separately and labelled non-binding, what it would be under
Q2-projected. A1.4's justification sentence is withdrawn as incomplete (jcp-qualify gated its GAL arms on the full
vector too).

**A2.2 F3 is two fitted families (so that a fitted rule can meet the binding gate).** The NNLS target norm is matched to
the gate norm:
- **F3a** (projected target): rows per snapshot $=Q^{\mathsf T}(L\psi(x_q))\,u_qs_q\in\mathbb R^r$ (orthonormal Galerkin
  tests); 176 snapshots (16 draws × 11 times), $C\in\mathbb R^{67584\times16384}$.
- **F3b** (full tested-vector target): rows per snapshot $=L\psi(x_q)\,u_qs_q\in\mathbb R^M$; 64 snapshots (16 draws ×
  times $0, 0.05, 0.15, 0.25$), $C\in\mathbb R^{98304\times16384}$ (12.9 GB).
- **F2+F3** (bank tested vector): rows per snapshot $=G(x_q)\,u_qs_q\in\mathbb R^r$; 176 snapshots.
Each snapshot block is divided by its target norm $\lVert b_j\rVert$. The F3 stop rule is evaluated over the union of
F3a and F3b candidates (κ ∈ {4, 8, 16} each). Three fits in total per job.

**A2.3 Outcome precedence (replaces the A1.2 list).**
- **F3:** *not fired* if at least one fully evaluated candidate (F3a or F3b, any κ) satisfies all three predicates;
  otherwise *fired (tested failure)* if every registered candidate was fully evaluated; otherwise *fired (incomplete
  evidence)*.
- **F2:** (a) is *not fired* if some candidate with $m\le16384$ is numerically qualified; $m^\*$ is **identified** only if
  every smaller registered candidate is resolved (fully evaluated and not qualified); otherwise the report says
  "a qualified rule exists at $m$; $m^\*$ not identified ($\le m$)". (b) and (c) are evaluated on the smallest qualified
  candidate. The F2 outcome is *fired (tested failure)* if (a) fails with every candidate resolved, or (b) or (c) fails;
  *fired (incomplete evidence)* if (a) fails with unresolved candidates; *not fired* otherwise.
- **Reference unqualified:** if R is not numerically qualified, every outcome above is reported as
  "**undecided: reference unqualified**" (a fourth category), with the predicate values given as diagnostics.
- **Unresolved, not failed:** a candidate whose successor or check run is invalid, whose fixture could not be built, or
  whose ρ population is all-trivial (every indicator excluded) is **unverified**, not failed; it counts as unresolved.
  Only a demonstrated bar exceedance (valid runs) is a quadrature *fail*.
- Fibonacci deferred → the F2 claim reads "Gauss ladder only".

**A2.4 Q1 restatements.** $A^{\mathsf T}A$ is the Gram of the **retained sine projection** of the bank, not the full
$L^2$ Gram of $G$ ($\mathcal M$ is the latter, up to rule convergence). Zero scale follows qcert: $\lVert c_n\rVert_{\mathcal G}=0$
and residual exactly 0 → certified; zero scale with non-zero residual → not certified. The equality with jcp-qualify's
η is algebraic, not bitwise.

**A2.5 Fitting provenance and acceptance.** Recorded and hashed per fit: the draw table, the rows actually used (with
replacements), the snapshot array, the parent configuration (rule, tables fingerprint, stepper sha). QR drift: a support
checkpoint recorded after a drift event with `drift_ok = False` is **not built (drift)** (incomplete evidence for that
κ). Every retained support is checked: weights finite and $>0$, size $\ge\kappa r$ (first crossing), and the fit residual
is recomputed by an independent device path (fresh matvec of the support columns built from the raw tables, not the
solver's internals) to $10^{-8}$ relative agreement. All blocks dropped → not built. **The smoke fits are not reused**:
production recomputes the three fits by the identical frozen procedure; the smoke measures their cost for the
certificate.

**A2.6 Controls are credited only when demonstrated.** A control counts as passed only after the smoke shows it firing
on real data with its fixture independently verified; a non-firing control **blocks** the qualification claim it
protects until a replacement fixture is registered by amendment and demonstrated. Revisions:
- **C1:** both the rollout bar and the ρ gate must fire (each shown); fixture list Gauss $\{16^2,24^2,32^2\}$ as in A1.6.
- **C2:** credited only if it actually fails Q2 in the smoke; a non-failure is reported (not a code bug by itself).
- **C3:** fixture = a certified production run (every η ≤ bar before perturbation); the code verifies independently
  (NumPy recomputation) that each intended residual crosses the bar after the perturbation; (a) and (b) as A1.6;
  **(c)** perturb states $k\in\{10,20,30,40\}$ of every dev6 case: the pooled non-converged fraction must exceed 1 % and
  configuration Q1 must fail.
- **C5:** the microbenchmark draws a different state $c$ from a stored stack on every call, sums each output into an
  accumulator returned at the end, synchronises with `block_until_ready`; ≥ 1.5 required; a failure is reported as an
  unsuitable fixture, not as evidence about the timer.
- **P3** is renamed "linear terms converged under the registered check" (not "exact").
- **P4** covers both projections: analytic vs AD Jacobian of $f$ for R, F2 (Gauss $96^2$) and one fitted rule.
- **P5** adds a **direct bank mass-form RHS comparison**: (i) $f_{\rm bank}(c)$ against the host NumPy evaluation
  $\mathcal M^{-1}[-T(c)-\nu\mathcal Kc]$ with $T$ from `npcheck` tables (independent bank evaluator), $10^{-10}$
  relative; (ii) against the mesh Galerkin projection of the finite-difference PDE right-hand side,
  $M_h^{-1}h^2G_h^{\mathsf T}[-u(D_xu+D_yu)+\nu\Delta_hu]$ (central differences, 1024²), relative $\le10^{-3}$ at the six
  dev6 output states — a sign/assembly check whose threshold is a fixture assumption (discretisation error $O(h^2)$).
- **P6:** (a) a tight anchor run with budget 1: the code first verifies independently that it missed its tolerance
  (reason budget, recomputed η > tight tol), then the Q1 code must reject it; (b) a successor table with a NaN → Q2 label
  "unverified" (not pass); (c) a check-rule table with a NaN → "unverified".

**A2.7 Resources and fallback.** The smoke is submitted with a bounded reservation (time limit 1.25 h, ledgered) covering
three full fits to the κ = 16 target with their deadlines and the NNLS tail, all controls and the measurement phases. The
smoke records **peak device bytes** (`memory_stats` peak, reset per phase where supported) per phase, including the bank,
tables, $C$, the QR buffers and temporaries; F3a, F3b and F2+F3 fit costs are measured separately. If mandatory work
cannot fit under the cap after the §8 deferrals, the lane stops with incomplete evidence; qualification coverage is never
reduced silently. If the E5 deadline passes without a hand-off, results against the lane's own R are labelled
"**local ablation; equivalence to E5's arm not established**".

## Amendment A3 (2026-10-21, before any lane code ran; answers `audits/codex-design-3.md`, verdict WRONG on the F2 outcome)

The E5 reconciliation amendment (called "A2" in A1.8 and "A3" in A2) is named **Amendment E5R** from here on.

**A3.1 F2 outcome uses only an identified $m^\*$ (replaces the F2 bullet of A2.3).** $m^\*$ is identified only when every
registered candidate with smaller $m$ is resolved (fully evaluated, not numerically qualified). Then: *not fired* iff
F2 at $m^\*$ is "no worse" and "faster" than R; *fired (tested failure)* otherwise. If no candidate with $m\le16384$ is
qualified and all are resolved: *fired (tested failure)* through (a). **If any smaller candidate is unresolved, the F2
verdict is *fired (incomplete evidence)***, whatever larger candidates show; the smallest qualified larger rule (if
any) is reported descriptively as "a qualified rule exists at $m$ (upper bound on $m^\*$)", with its accuracy and speed
predicates, labelled non-binding. "Reference unqualified" (A2.3) still overrides.

**A3.2 Fitting targets.** F3a's target matches the **non-binding** Q2-projected indicator; F3b's and F2+F3's targets match
the **binding** tested-vector gates. F3a and F3b use different snapshot schedules, so their comparison is a comparison of
fitted procedures, not an isolated test of the target norm. The F3 search has **six candidates across two families**
(F3a, F3b × κ ∈ {4, 8, 16}); all per-arm intervals stay labelled exploratory, in-sample, selection-unadjusted.

**A3.3 NNLS drift.** `nnls.greedy_nnls` returns path-level drift evidence only, so the policy is conservative: if a fit's
path reports any drift event (`drift_ok` false or a recorded drift above the solver's tolerance), **every support
checkpoint of that path is "not built (drift)"** (incomplete evidence for those κ).

**A3.4 Fixtures named.** C2 uses F3b κ = 8 (binding family); if that support is not reached, F3a κ = 8 (recorded). The
fitted fixture of P4 is F3a κ = 8 (else F3b κ = 8).

**A3.5 Resources (replaces A1.5's deadline and A2.7's reservation).** NNLS deadline **12 min per fit** (three fits ≤ 36 min
plus the measured cooperative overshoot and final-residual tail). Smoke reservation **1.5 h** (ledgered); production
is submitted only with a smoke-certified projection such that spent + projection ≤ 3 GPU-h. The runtime is bounded by
the reservation; completion is not guaranteed — work that cannot complete is reported as incomplete evidence. Peak device
bytes come from `memory_stats` (peak per phase where resettable, else the running peak with the phase order recorded);
if the counters are unavailable, `nvidia-smi` memory.used is polled per phase. Host bytes of the system construction
(the $b$ vector and host copies) are recorded from `resource.getrusage`.

**A3.6 Wording carried into the report.** The report distinguishes: binding vs projected fitting targets and Q2 labels;
an identified minimum vs a qualified upper bound; bounded runtime vs guaranteed completion; local-fallback evidence vs
E5-equivalent evidence.

## Amendment A4 (2026-10-21; answers `audits/codex-design-4.md`, verdict NEEDS-RESTATEMENT, no WRONG — design accepted)

**A4.1 Missing F2 branches.** (i) No candidate with $m\le16384$ is qualified and at least one is unresolved →
*fired (incomplete evidence)*. (ii) $m^\*$ is identified but its accuracy or timing evidence is missing (run not
completed, scoring impossible) → *fired (incomplete evidence)*. A3.1's "otherwise" means: with both predicates
evaluated, at least one false.

**A4.2 Memory semantics.** `memory_stats` peaks are allocator high-water marks (phase-attributed only where reset is
supported; otherwise the running maximum with the phase order); `nvidia-smi memory.used` samples are observed sampled
maxima, not guaranteed transient peaks; `ru_maxrss` is a process-lifetime host high-water mark (kB), not an isolated
measurement of construction copies. The report and the cost certificate state these units and limitations.
