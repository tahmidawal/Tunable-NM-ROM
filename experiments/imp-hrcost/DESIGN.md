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
