# jcp-budget — where does the reduced model's error come from, how does it grow beyond the training horizon, and does the scoring lattice see it? (pre-registration)

**Status:** pre-registration, 2026-10-18, written before any code or data of this lane. Amendments are appended at the
end, dated; nothing above them is rewritten. Accuracy numbers are scored against order-verified second-order
references: the E0 store for $t\le0.25$ and this lane's own long-horizon references (same E0 code, own namespace)
beyond. Every $\varepsilon_{\rm ref}$ is a **conditional Richardson estimate**, not a rigorous bound. Post-hoc labels
carry forward unchanged: E0 A10; combined A10, A11 and the retracted A15.

Governing documents (later supersedes earlier): `reports/2026-10-17-full-experiment-plan/PLAN-v2.md` (B2, B7, B8),
`PLAN-v3-amendments.md` (V3-1, V3-6), `PLAN-v4-amendments.md` (V4-1, V4-2); `campaign/briefs/COMMON-BRIEF.md`
(binding); `campaign/briefs/jcp-budget.md` (lane brief, wins on conflict); `reports/2026-10-08-jcp-lane-agent-rules.md`.

Lane: worktree `worktrees/2026-10-18-jcp-budget`, branch `exp/2026-10-18-jcp-budget` (fork `21c175a1b`, setup
`a218a56dc`), lane dir `experiments/jcp-budget`, cluster namespace `/cluster/tufts/paralab/tawal01/jcp-budget/`, lane
job cap **3** (campaign cap 8, checked on the account before every submit), GPU-hour cap **45** including contingency,
with time-limit reservations. Never written: the E0 store, `sealed/`, `custody/`, another lane's namespace or worktree.

Built on (read, cited, code copied with provenance into this lane; their certified results are **cited, not re-run**):
`jcp-qualify` (FRONT-2D v1, sha256 `a51ac1a2…`, lane commit `4399491eb`; the 3D qualified subset; Q1/Q2 machinery),
`jcp-p1-scaling` (frozen mode lists, certifier, split query), `jcp-combined` (the budget method: `budget2d.py`,
`budget3d.py`, $\delta$ definitions, gate B0 and its fixtures), `jcp-references` (E0 generator `refcore.py`, order
judge `verdict.py`, pseudo-spectral `pscore.py`, store loader `refstore.py`), `jcp-speed3d` (3D solver specs).

## 0. Questions and outcome labels

| item | question | outcome labels |
|---|---|---|
| **B2** error budget | for each qualified front configuration, which ingredient (bank floor, quadrature, time, test space/form, solver, initial-condition projection, scoring) moves the solution or the error most, and do ingredients interact? | per configuration and mesh: dominant ingredient (or "no single dominant"); per pair: interacting / additive / unresolved; mesh-invariance of the dominant ingredient: same / differs |
| **B7** longer horizons | past the training horizon ($t\le0.25$), do qualified configurations stay numerically qualified and stable, and how fast does their error grow? | per configuration: Q1/Q2 at $T$ pass/fail; stability pass/fail; growth label (§5.3); front-order preserved yes/no |
| **B8** physical diagnostics | do the reduced solutions respect boundary values, energy decay and the maximum principle; how large is the IC-projection error; does the $257^2$ / $63^3$ scoring lattice miss error a finer observation route sees? | sampling adequacy per configuration: adequate / inadequate / mixed; descriptive tables for the rest, each with a control |

Three labels from V4-2 stay separate: numerical qualification (Q1+Q2), role qualification, frontier usefulness. This lane
**does not change** any configuration's numerical-qualification label at $T=0.25$ (that is jcp-qualify's); at long
horizons it issues new, horizon-specific Q1/Q2 labels.

```mermaid
flowchart LR
  F[FRONT-2D v1 + 3D qualified subset]:::data --> B2[B2 factorial ablations]:::run
  E0[(E0 store, t <= 0.25)]:::ref --> B2
  R[New references T = 0.5 / 1.0, 257^2 + 1025^2]:::ref --> B7[B7 long-horizon rollouts + Q1/Q2]:::run
  R --> B8[B8 diagnostics + fine route]:::run
  F --> B7
  F --> B8
  B2 --> REP[make_report.py -> report.md]:::doc
  B7 --> REP
  B8 --> REP
  classDef data fill:#dbeafe,stroke:#1d4ed8; classDef ref fill:#fef3c7,stroke:#b45309;
  classDef run fill:#dcfce7,stroke:#15803d; classDef doc fill:#f3f4f6,stroke:#6b7280;
```

## 1. The reduced model and inherited contracts (unchanged)

2D: frozen checkpoint `dn256b` ($R=512$, $K=16$), rotation `rotation_R512`, ordered nested prefixes; test space
$M=4r$ frozen mode list $F_{4r}$ (`modes_lean(1024, 4r)`, V4-1); tested advection by an off-mesh point rule;
two-step LMM residual $\mathcal R_n(c)=A(\alpha_0c+\alpha_1c_n+\alpha_2c_{n-1})+\Delta t(\beta_0F(c)+\beta_1F(c_n))$,
$F(c)=N(c)+\nu\Lambda Ac$; LSPG ($\min\lVert D\mathcal R_n\rVert$) or GAL ($Q^{\mathsf T}\mathcal R_n=0$); adaptive LM,
budget 600; tolerance levels prod / tight / tighter as `p1common.tol_of`. Initialisation contract (V4-4): full-field
input sampled at the cold quadrature points, reduced projection solve, full-field decode. $\Delta t_0=0.005$.
3D: speed3d settings at $n=65$ (mesh = scoring lattice $63^3$ interior), solver specs `fs` / `qn` / `ad`
(`s3common.py`), mesh least-squares initialisation (Step 2), $\Delta t_0=0.01$.

The bank was trained on trajectories with $t\in[0,0.25]$ (checkpoint config: `num_steps` 50, `dt` 0.005). **B7 is
therefore a test of temporal extrapolation of the bank**, and the floor $e_P(t)$ for $t>0.25$ is reported as such.

Scoring (common brief §2): $e(t_k)=\lVert u(t_k)-R(t_k)\rVert/\lVert R(t_0)\rVert$ on the scoring lattice, fixed-$t_0$
normalisation; window error $e_W=\max_{t_k\in W}e(t_k)$. Windows: $W_1=(0,0.25]$ (the E0 metric, identical to every
earlier lane's $e$), $W_2=(0.25,0.5]$, $W_3=(0.5,1.0]$. Cases are the statistical unit. Ties: two quantities that differ
by at most $2\varepsilon_{\rm ref}$ are not ranked. $R_S$ is not used by this lane except where an earlier lane's
number is quoted with its label.

## 2. Configurations and populations (frozen now)

### 2.1 2D (FRONT-2D v1)

The members of FRONT-2D v1 (union of the median and worst fronts, cost axis `c`) at each mesh, read by
`make_configs.py` from `jcp-qualify/runs/FRONT-2D.json` (sha256 pinned; a mismatch stops the lane):

| mesh | members (r\|form\|scheme\|factor) | count |
|---|---|---|
| $256^2$ | R128 LSPG BE 2.5; R128 GAL CN 5; R128 GAL CN 2; R128 LSPG BE 1; R128 GAL CN 1; R256 GAL CN 2; R384 LSPG CNR 2; R256 GAL CN 1; R384 LSPG CNR 1; R384 GAL CN 1 | 10 |
| $1024^2$ | the ten above + R256 LSPG CNR 2.5 | 11 |
| $4096^2$ | the eleven above + R128 LSPG BE 2 | 12 |

Rules as frozen in FRONT-2D (R128 `fib4181`, R256/R384 `gauss128`). **GAL-CN members are diagnostic** (coordinator
decision recorded in LAB-LOG 2026-10-18, jcp-qualify entry): they are budgeted and reported in separately labelled
rows "GAL-CN (registered diagnostic scheme)"; every lane-level summary is given for the LSPG members alone and for all
members.

Populations: $256^2$ and $1024^2$ all 38 cases (dev6 + val32); $4096^2$ the 16-case subset `sub16` (dev6 +
val32[0:10], the P1/qualify subset, labelled). No other cohort is run.

### 2.2 3D (qualified subset of jcp-qualify)

The four configurations with Q2 "pass" in jcp-qualify `EXPORT.json` `threeD.tested` (frozen, sha256 pinned):
`W256|gl24|fs|CN|0.01|K1`, `W256|gl24|fs|CN|0.0125|K1`, `W256|lat8192|qn|BDF2|0.025|k4`,
`W512|lat16384|qn|BDF2|0.025|k4`. Label: "qualified subset of tested candidates" (qualify's label carried; jcp-dims is
completing the 3D front — later qualified members are **not** in this lane's scope unless added by a dated amendment).
Population: cohort 923801 (E0 `val64`) cases 0–15 (`v16`), pre-declared now (the first 16; no case selection on data).

### 2.3 Which item runs where

| item | 2D $256^2$ | 2D $1024^2$ | 2D $4096^2$ | 3D $65^3$ |
|---|---|---|---|---|
| B2 single ablations + pairwise | 38 cases | 38 cases | sub16 | v16 (3 trajectory ingredients) |
| B2 full factorial (all $2^5$ corners) | — | 38 cases | — | v16 ($2^3$) |
| B7 $T=0.5$ | — | 38 cases | sub16 (deferrable, §9) | v16 |
| B7 $T=1.0$ | — | sub16 (deferrable, last) | — | — |
| B8 diagnostics + fine route | 38 cases | 38 cases (primary) | sub16 | v16 (deferrable, §9) |

## 3. New references (B7, B8)

### 3.1 Generator

The E0 code (`refcore.py`, `verdict.py`, `cohorts.py`, `pscore.py` from `exp/2026-10-06-jcp-references` @ `cc04f4b54`)
copied into `experiments/jcp-budget/ref/` with sha256 provenance. **The only changes:** (i) a horizon parameter
$T\in\{0.25,0.5,1.0\}$ with outputs at $t_k=0.05k$, $k=0..T/0.05$ (the six-output buffers, the per-stage evidence buffer
length and the per-time loops of the order judge are sized from $T$); (ii) an additional output route: the finest
level's fields sampled on the **fine lattice** $1025^2$ (2D, nodes $x=i/1024$ incl. walls; stride 8 at $L=8192$) or
$127^3$ (3D interior, $x=k/128$; stride 4 at $L=512$), and per-time level differences $d_{12},d_{23}$ on the fine
lattice computed in-job at all three levels (so $\varepsilon_{\rm ref}$ is available on both lattices; only the finest
level's fine-lattice field is stored). With $T=0.25$ and no fine route the code must be the E0 code up to these
parameters (gate G-R1).

Kind: **$R_{ST}$ only** (central differences + Crank–Nicolson with Rannacher start), the E0 levels: 2D $L=2048,4096,8192$
($\Delta t=1.25{\rm e}{-3},6.25{\rm e}{-4},3.125{\rm e}{-4}$), refinement $L=12288$ ($\Delta t=0.05/240$) on order or
Péclet failure (**at $T=0.5$ only**; at $T=1.0$ a failing case is terminal and non-rankable — the refinement level would
cost about 4800 stages at 151 M unknowns); 3D $L=128,256,512$ ($\Delta t=2.5{\rm e}{-3},1.25{\rm e}{-3},6.25{\rm e}{-4}$),
refinement $L=768$. Tolerances, floor-proxy, Newton/BiCGStab rules, tight retry, retry policy and the acceptance
procedure are E0's (`refjob.solve_case`), applied with the judge at **every output time** $k=1..T/0.05$:
$p\in[1.8,2.2]$, Péclet $<2$, floor margin, direction consistency $c\le0.5$, shared IC. Per case: outcome (E0 labels),
$\varepsilon_{\rm ref}(t_k)$ on both lattices, window $\varepsilon_W=\max_{t_k\in W}\varepsilon_{\rm ref}(t_k)$.
$R_S$ is not generated.

Cohorts: 2D dev6 + val32 at $T=0.5$ (38 cases); 2D sub16 at $T=1.0$; 3D v16 at $T=0.5$. ICs and $\nu$ come from the
E0 cohort descriptors (`cohorts.py`, with the E0 cohort sha256 asserted). Storage: `/cluster/tufts/paralab/tawal01/
jcp-budget/refs/<dim>d/<cohort>/T<T>/c<case>.npz` + `.json` (per-file sha256 in a lane `REFS-MANIFEST.sha256`); never
the E0 store.

### 3.2 Eligibility

A case is **eligible at horizon $T$** for windows $W_2$, $W_3$ iff its new reference outcome is `accepted`. For $W_1$
eligibility is read from the E0 store (the primary rule; the new reference must also reproduce E0 there, G-R2). A
non-eligible case is shown, never ranked or counted in a verdict.

### 3.3 Pseudo-spectral cross-check (subset)

E0's sine pseudo-spectral + ETDRK4 solver (`pscore.py`, E0 sizes: 2D $N=1024,1536,2048$, $\Delta t=0.05/1600$;
3D $N=128,192,256$, $\Delta t=0.05/320$), horizon generalised as in 3.1, on **2D val32 cases 0–3** at $T=0.5$ and **3D
val64 cases 0–1** at $T=0.5$ (pre-declared; these are E0's own cross-check cases). Both E0 rules are evaluated and
reported, as E0 does: the original rule and the A10-amended rule (agreement with the finest FD level within
$2\varepsilon_{\rm ref}$ at every $t_k$, PS self-difference $\le\varepsilon_{\rm ref}/2$). Outcome per window. A failed
cross-check does not invalidate the order verification; it is reported as "cross-check failed" next to every number
of that dimension and window.

## 4. B2: the error budget

### 4.1 Ingredients and corners

Fix a configuration $a$ (deployed: rule $q$, scheme $s$, step $h$, form $f$, solver level prod, deployed
initialisation). Five **trajectory ingredients** each switch exactly one element of the configuration tuple to its
"ideal" value:

| ingredient | symbol | deployed → ideal | notes |
|---|---|---|---|
| quadrature | Q | rule $q$ → $q_\infty$ = Gauss $320^2$ (2D); Gauss $56^3$ (3D, combined's converged rule for W256/W512) | gated by G-B4 |
| time | T | $(s,h)$ → (BDF2, $\Delta t_0/8$), a **package** (scheme, step, start-up, LSPG weights change together) | combined's definition |
| form / test space | F | 2D: form $f$ → the other form (LSPG ↔ GAL), with that form's own prod tolerance; the test functions are unchanged | for GAL members the switch is GAL→LSPG ("reverse", labelled) |
| solver | S | prod → tight tolerance (2D); deployed spec (fs/qn) → `ad` tight adaptive LSPG, same scheme and step (3D) | |
| initial condition | I | deployed initialisation → **full-grid least-squares projection** of the mesh IC on the span (2D, $\min_c\lVert G_r^{(L)}c-u_0\rVert$ over the interior nodes of the ROM's own mesh) | 3D: the deployed init already is the mesh LS projection → $I$ is structurally zero, reported, not run |

The ideal values are chosen to remove numerical error (converged rule, small step, tight solve, best IC); the form
switch is not "ideal" in that sense and is labelled a **form sensitivity**, as in combined.

**Corners.** For every subset $S$ of the five ingredients, $u_S$ = the trajectory with exactly the ingredients in $S$
switched. $u_\varnothing=u_a$ (deployed). 2D: pairwise design (all $|S|\le2$: $1+5+10=16$ runs per case) at every mesh;
full factorial (all 32 subsets) at $1024^2$. 3D: ingredients Q, T, S; full factorial ($2^3=8$); the form switch is run
once at the far corner (GAL, $q_\infty$, BDF2 $\Delta t_0/8$, `ad` tight), as combined's 3D $\delta_{\rm form}$.

**Effect vectors (Möbius inversion, exact).** $I_S=\sum_{T\subseteq S}(-1)^{|S|-|T|}u_T$ (per output time, on the
lattice). Main effect $I_{\{x\}}=u_x-u_\varnothing$; pairwise interaction $I_{\{x,y\}}=u_{xy}-u_x-u_y+u_\varnothing$.
With the full factorial, $u_{\rm all}-u_\varnothing=\sum_{S\neq\varnothing}I_S$ **exactly** (an identity of vectors,
checked by gate G-B2); norms are **not** additive and are never stacked.

**Ideal end.** $u_\star$ = the all-switched corner of the LSPG-deployed members' factorial with the form switched to GAL,
i.e. (GAL, $q_\infty$, BDF2 $\Delta t_0/8$, tight, full-grid LS init) — combined's ideal model plus the ideal IC. It
depends only on $(r,\text{mesh},\text{case})$ and is shared by all members with that $r$. Its own uncertainty
$U_\star(t)=\lVert u_\star^{h_0/8}-u_\star^{h_0/16}\rVert+\lVert u_\star^{h_0/8}-u_\star^{h_0/8,\rm tighter}\rVert
+\lVert u_\star^{h_0/16}-u_\star^{h_0/16,\rm tighter}\rVert$ (combined A2.8).

### 4.2 Reference-side components

With $\Pi$ the orthogonal projection onto the span of the bank prefix at the scoring-lattice interior nodes and $u_{\rm ref}$
the E0 $R_{ST}$ field:
- **floor** $e_P(t)=\lVert(I-\Pi)u_{\rm ref}(t)\rVert$;
- **ideal dynamics** $e_{\star,\rm dyn}(t)=\lVert u_\star(t)-\Pi u_{\rm ref}(t)\rVert$;
- for every in-span trajectory $v$ (every corner), $e_v^2=e_P^2+\lVert v-\Pi u_{\rm ref}\rVert^2$ **at each output time**
  (gate G-B0, relative defect $\le10^{-10}$);
- **scoring**: $\Delta_{\rm score}=e^{\rm fine}-e^{257}$ (§6.5) and $\varepsilon_{\rm ref}$, reported beside the budget;
- **IC projection** at $t_0$: $e(t_0)$ of the deployed init, of the full-grid LS init, and the floor $e_P(t_0)$.

All quantities are relative to $n_0=\lVert u_{\rm ref}(t_0)\rVert$ on the same lattice; per case, each component is
summarised as its maximum over $k=1..5$ (and $e(t_0)$ separately); medians and worst over the population.

### 4.3 Decision rules (frozen)

Per case, the **candidate components** are: $e_P$, $e_{\star,\rm dyn}$, $\lVert I_x\rVert$ for each ingredient $x$, and
$\max_{x<y}\lVert I_{xy}\rVert$ (2D: 5 main effects, 10 pairs; 3D: 3 + 3, plus $\delta_{\rm form}$).
- **Unresolved:** a trajectory-side component $<3U_\star$ is "unresolved" on that case; a reference-side component
  ($e_P$, $e_{\star,\rm dyn}$) $\le2\varepsilon_{\rm ref}$ is "reference-limited".
- **Largest component** of a case: the largest resolved candidate; if none is resolved, "none resolved".
- **Dominant ingredient** of a (configuration, mesh): the component that is largest on **more than 50 %** of eligible
  cases; otherwise "no single dominant" (the counts are always reported).
- **Pair label** (per case): $\{x,y\}$ is *interacting* if $\lVert I_{xy}\rVert>0.25\max(\lVert I_x\rVert,\lVert I_y\rVert)$
  and $\lVert I_{xy}\rVert\ge3U_\star$; *additive* if $\lVert I_{xy}\rVert\le0.25\max(\cdot)$ and the larger main effect
  is resolved; otherwise *unresolved*. Per (configuration, mesh, pair): "interacting" if interacting on $>25\,\%$ of
  eligible cases, "additive" if additive on $\ge75\,\%$, else "mixed". The two largest ingredients (by median main
  effect) are named in the report, and their pair is always shown (brief: pairwise at least for the two largest).
- **Mesh invariance of the budget:** for the ten members present at all three meshes, "same" iff the dominant ingredient
  is identical at $256^2$, $1024^2$, $4096^2$ (sub16 at $4096^2$, and the $256^2$/$1024^2$ dominant recomputed on sub16
  for this comparison); otherwise "differs". Per component, the median ratio $4096^2/1024^2$ and $256^2/1024^2$ with a
  case-bootstrap 95 % interval (10,000 resamples, percentile), reported.
- **Closure (reported):** $D_a=\lVert u_a-u_\star\rVert$ against $\sum_x\lVert I_x\rVert$ and
  $(\sum_x\lVert I_x\rVert^2)^{1/2}$; at $1024^2$ also against the full Möbius sum (exact, G-B2).

## 5. B7: longer horizons

### 5.1 Runs

Every B7 configuration (§2.3) runs to $T$ with outputs every 0.05: prod (internal trajectory recorded and certified
in-job at every step with a fresh autodiff Jacobian — LSPG $\gamma$, GAL $\eta$ exactly as jcp-qualify §4.1a/b), tight
anchor (every case), tighter anchor (dev6 cases 0–5 in 2D; v16 cases 0–5 in 3D), successor rule $q^+$ (FRONT-2D's
successor; 3D qualify's) on sub16 (2D) / cases 0–7 (3D), and $q^{++}$ on dev6 / cases 0–3. The first six outputs of
the prod run must reproduce jcp-qualify's archived $T=0.25$ coefficients of the same configuration and case (gate G-L1).

### 5.2 Long-horizon qualification (horizon-specific labels)

Q1 at $T$: the common-brief contract over **all steps up to $T$** (valid; non-certified steps $\le1\,\%$ pooled;
distance to the tight anchor $\le2.5\times10^{-4}$ on every case, the distance being the max over **all** outputs to
$T$; tight/tighter $\le2.5\times10^{-5}$ on the tighter subset). Q2 at $T$: successor-rule rollout distance
$\le2.5\times10^{-4}$ (max over outputs to $T$) on every case of the Q2 population, and $\rho_{\max}\le0.116$ on the
configuration's own output states to $T$ (qualify A2.4 edge cases). Label "numerically qualified at $T$" iff both pass;
a configuration qualified at 0.25 but not at $T$ is reported as such (no fallback rule at $T$).

### 5.3 Metrics and labels

- **Stability at $T$ (pass/fail):** pass iff every case is valid to $T$ and the ROM's lattice energy
  $E(t_k)=\tfrac12\lVert u(t_k)\rVert^2$ satisfies $E(t_k)\le(1+10^{-3})E(t_0)$ for every $k$ on every case (the exact
  solution's energy is non-increasing under zero Dirichlet walls). Monotonicity violations
  $E(t_k)>(1+10^{-3})E(t_{k-1})$ are counted and reported (no verdict), with the reference's own counts beside them.
- **Error growth:** per eligible case $g=e_{W_2}/e_{W_1}$ (and $e_{W_3}/e_{W_2}$ at $T=1$). Configuration label from the
  case-bootstrap 95 % interval of the median $g$: "no growth" if the upper bound $\le1$; "bounded growth" if the
  interval lies in $(1,2]$; "accumulating" if the lower bound $>2$; otherwise "between labels" with the interval.
  The current-time-normalised error $\lVert u-R\rVert/\lVert R(t_k)\rVert$ is reported beside it (the solution
  decays, so fixed-$t_0$ errors can fall while relative errors grow).
- **Floor over time:** $e_P(t_k)$ for $t_k\le T$ (bank extrapolation), median and worst per $r$.
- **Front order:** at $1024^2$, Kendall's $\tau$ between the members' median $e_{W_1}$ and median $e_{W_2}$, and
  whether each pair's order is preserved, reversed or tied (ties within the larger window $2\varepsilon$); "front order
  preserved" iff no resolved pair reverses (reported, used in the report's wording only).

## 6. B8: physical diagnostics

All on the **fine lattice** unless stated; ROM fields decoded through the bank at the fine-lattice nodes
(coefficients are mesh-independent; the decode at $1025^2$ is the full-mesh decode for the $1024^2$ ROM, a finer decode
for $256^2$, and a stride-4 subsample of the full mesh for $4096^2$ — labelled per mesh).

### 6.1 Boundary

Wall values of every decoded ROM field and every reference field are exactly zero by construction (checked: max
$|u|$ on the walls $=0$ exactly; a non-zero value fails gate G-D1). The **near-wall error density**
$b=\big(\lVert e\rVert_{\rm band}^2/|{\rm band}|\big)\big/\big(\lVert e\rVert_{\rm all}^2/|{\rm all}|\big)$ with the band
= interior nodes within distance $1/32$ of a wall; reported per configuration (median, worst), labelled
"boundary-concentrated" when the median $b>2$ (descriptive).

### 6.2 Energy and dissipation

$E(t_k)=\tfrac12h^2\sum u^2$ and $\mathcal D(t_k)=\nu h^2\sum|\nabla_hu|^2$ (central differences on the fine lattice,
$h=1/1024$), for ROM and reference. Reported per time: $E_{\rm rom}/E_{\rm ref}-1$, $\mathcal D_{\rm rom}/\mathcal D_{\rm ref}-1$,
and the **balance defect** $\beta_k=B^{\rm rom}_k-B^{\rm ref}_k$ with $B_k=E(t_k)-E(t_{k-1})+\tfrac{\Delta t_{\rm out}}2
(\mathcal D(t_k)+\mathcal D(t_{k-1}))$, normalised by $E_{\rm ref}(t_0)$. (The common time-quadrature error of $B$
cancels to leading order in $\beta$; the reference's own $|B^{\rm ref}_k|$ is reported to show its size.) Descriptive,
with control C-E (§7).

### 6.3 Maximum principle

The ICs are non-negative Gaussians, so the exact solution satisfies $0\le u\le\max u_0$. Per case: undershoot
$\max(0,-\min u_{\rm rom})/\max u_0$ and overshoot $\max(0,\max u_{\rm rom}-\max u_0)/\max u_0$ over all outputs; counts
of cases above $10^{-3}$, with the reference's own values beside them. Descriptive.

### 6.4 IC projection

$e(t_0)$ for the deployed init, the full-grid LS init and the floor $e_P(t_0)$ on both lattices; the ratio
deployed/floor; and whether the IC ingredient's effect $\lVert I_I\rVert$ (B2) at $t_5$ exceeds its value at $t_1$
(IC error persists vs is damped) — reported.

### 6.5 Output-sampling adequacy (never dropped)

Per case: $e^{257}$ (the campaign metric) and $e^{\rm fine}$ (same definition on $1025^2$, normalised by the
reference's $t_0$ norm on $1025^2$), window $W_1$ (and $W_2$ at $1024^2$). Ratio $s=e^{\rm fine}/e^{257}$.
- Per configuration and mesh: **adequate** iff $|s-1|\le0.10$ on $\ge90\,\%$ of eligible cases **and** the ranking of the
  members' median errors on the fine route equals that on $257^2$ up to ties (two medians tie if they differ by at most
  $2\times$ the larger median $\varepsilon_{\rm ref}$ of the two routes); **inadequate** iff the median $|s-1|>0.10$ or a
  resolved rank reversal occurs; otherwise **mixed**.
- Lane-level statement: "the $257^2$ lattice is adequate for the qualified 2D front" only if every LSPG member at
  $1024^2$ is adequate.
- 3D (deferrable): the same on $127^3$ (decode of the $65^3$ coefficients at $x=k/128$) vs $63^3$.

## 7. Gates (must pass) and controls (must fail), each shown at real size in the smoke

| id | kind | check | blocks if it misbehaves |
|---|---|---|---|
| G-R1 | gate | new generator with $T=0.25$, no fine route, reproduces E0's stored level fields of one dev case to $\le10^{-10}$ relative per time (same code path; cross-GPU rounding allowed) | every new reference |
| G-R2 | gate | the $T=0.5$ references' first six outputs at each level reproduce the E0 store's level fields: per time relative difference $\le10^{-10}$ (cross-GPU rounding allowed) | B7, B8 |
| G-R3 | gate | fine-lattice and $257^2$ fields of the same level agree at the shared nodes bitwise | B8 |
| C-R1 | control | the E0 first-order ladder (upwind + BE, `X1_LEVELS`) on one dev case at $T=0.5$ must be judged `order-fail` | B7 reference validity |
| C-R2 | control | a time-misaligned comparison (new reference at $t_{k+1}$ vs ROM at $t_k$) must break G-L1-style reproduction (difference $>10^{-6}$) | every long-horizon score |
| G-B0 | gate | per time $|e_v^2-e_P^2-\lVert v-\Pi u_{\rm ref}\rVert^2|/e_v^2\le10^{-10}$ for $u_\star$ and every corner; analytic fixtures of combined A3.5 and the real-data mismatched-normaliser fixture (A12.1) must break it ($>10^{-6}$) | B2 floor attribution |
| G-B1 | gate | sham switches (rule $q\to q$; level prod → prod; init deployed → deployed) reproduce $u_\varnothing$ bitwise | every effect vector |
| G-B2 | gate | full factorial: $\lVert u_{\rm all}-u_\varnothing-\sum_{S\ne\varnothing}I_S\rVert\le10^{-12}\lVert u_{\rm all}-u_\varnothing\rVert$ | Möbius tables |
| G-B3 | gate | $u_\varnothing$ reproduces jcp-qualify's archived coefficients of the same configuration/case (fields on the lattice, $\le10^{-8}$ relative, cross-job) | every B2/B7/B8 number of that configuration |
| G-B4 | gate | converged rule: $q_\infty$ vs the check rule (2D Gauss $640^2$; 3D Gauss $64^3$) for the Q-corner and $u_\star$ on dev6 (2D) / cases 0–3 (3D): $\le2.5\times10^{-5}$ | quadrature attribution (else "withheld: converged rule unverified") |
| C-Q | control | deployed rule replaced by an under-resolved rule (2D Gauss $32^2$; 3D `lat256`): $\lVert I_Q\rVert>2.5\times10^{-4}$ on most check cases | quadrature attribution |
| C-T | control | LSPG-BE at $\Delta t_0$: $\lVert I_T\rVert>10^{-3}$ on most cases | time attribution |
| C-S | control | LM budget 0 (premature exit) as the deployed solver: $\lVert I_S\rVert>2.5\times10^{-4}$ on most check cases | solver attribution |
| C-I | control | deployed init perturbed, $w_0\to1.01\,w_0$: $\lVert I_I\rVert$ (perturbed init → LS init) $>10^{-3}$ at $t_1$ on every check case | IC attribution plumbing |
| C-F | control | manufactured floor: reference $+\eta v$ with $v\perp$ span (unit lattice norm, built by orthogonalising a fixed high-frequency field), $\eta=10\times$ the configuration's median $e_a$: $e_P$ must equal $\sqrt{e_{P,0}^2+\eta^2}$ to $10^{-10}$ relative **and** the largest component must flip to $e_P$ on $\ge90\,\%$ of cases | floor attribution, decision rule |
| C-score | control | case swap, time shift and ST↔S swap must each change the reproduced error by $>10^{-6}$ (combined G-score) | every score |
| C-A | control | injected sampling alias: ROM field $+\,a\sin(256\pi x)\sin(256\pi y)$, $a=$ the case's $e^{257}\,n_0^{\rm fine}/\lVert\sin\sin\rVert_{\rm fine}$: $e^{257}$ changes by $\le10^{-12}$ (it vanishes on the lattice) while $s>1.10$ on every check case | B8 sampling verdict |
| C-E | control | injected energy: ROM field scaled by $(1+0.02\,t/T)$: the balance defect $\max_k|\beta_k|$ must rise by $>10\times$ its uninjected value on every check case | B8 energy table |
| G-L1 | gate | B7 prod run's first six outputs reproduce jcp-qualify's archived $T=0.25$ coefficients ($\le10^{-8}$ relative fields) | B7 numbers of that configuration |
| G-Q | gate/control | the B7 certifier: P1/qualify's premature-exit control (budget 0) must give non-certified steps $>1\,\%$; the wrong-mesh table control must raise | B7 Q1 |
| G-D1 | gate | ROM and reference wall values exactly zero | B8 boundary |

"Most" means more than half of the check cases (dev6 in 2D, cases 0–3 in 3D). A control that cannot fire on real data
in the smoke is a bug: the smoke acceptance (`smoke_accept.py`) recomputes every predicate from raw outputs and the
corresponding verdict is withheld until fixed.

## 8. Statistics

Cases are the unit; no bootstrap over time snapshots. Medians with case-bootstrap 95 % percentile intervals (10,000
resamples, seed 0). All verdict thresholds above are frozen here. Failures and ineligible cases are listed, never
dropped silently. Analysis is offline on CPU (`JAX_PLATFORMS=cpu`) from pulled arrays, except fine-lattice decodes and
fine-lattice scoring, which run in-job (the fine reference fields stay on the cluster; dev6's fine fields are pulled
for an offline recomputation of every in-job fine-route number on those cases, bar $10^{-9}$ relative).

## 9. Jobs, GPU-hours and defer order

| job | content | GPU | estimate (GPU-h) |
|---|---|---|---|
| `smk1` | 2D smoke: reference pilot (dev6 case 0 to $T=0.5$, G-R1/G-R2/G-R3, C-R1 at reduced cost), ROM at $1024^2$ on 2 configurations × dev6 cases 0–1 with every 2D gate and control | H200 | ≤ 2 |
| `smk2` | 3D smoke: reference pilot (val64 case 0 to $T=0.5$), 3D B2/B7 on 2 configurations × 2 cases, 3D controls | H200 | ≤ 1.5 |
| `ref2a/b/c` | 2D references $T=0.5$, 38 cases in three jobs | H200 | 6 |
| `ps2` | 2D PS cross-check, 4 cases, $T=0.5$ | A100-80G | 1.5 |
| `ref3` | 3D references $T=0.5$ v16 + PS 3D 2 cases | H200 | 2.5 |
| `b1024` | 2D B2 (full factorial) + B7 $T=0.5$ + B8 at $1024^2$ | H200 | 3 |
| `b256` | 2D B2 pairwise + B8 at $256^2$ | A100-80G | 1.5 |
| `b4096` | 2D B2 pairwise + B8 at $4096^2$ (sub16) [+ B7 $4096^2$ sub16, deferrable] | H200 | 2.5 |
| `b3d` | 3D B2 + B7 $T=0.5$ [+ B8 3D, deferrable] | H200 | 3 |
| `ref2t1`, `b1024t1` | 2D references $T=1.0$ sub16; B7 $T=1.0$ at $1024^2$ | H200 | 5.5 |
| | **total** | | **≈ 29**, with 25 % contingency ≈ 36 ≤ 45 |

Every production job needs a smoke-derived cost certificate (`resources.py`, speed3d/qualify pattern); the ledger
(`runs/GPU-LEDGER.json`, measured `sacct` elapsed) admits a job only if spent + certified projection ≤ 45. The
references precede the B jobs that score against them. A 4096² memory pilot is part of `smk1` only if the certificate
cannot bound the $4096^2$ peak from jcp-qualify's measured 96.6 GB pilot (same bank edges); otherwise the measured
number is cited.

**Defer order** (brief; first item dropped first): B7 at $4096^2$ → B8 3D → B2 interactions beyond pairwise (the
$1024^2$ full factorial and the 3D $2^3$ corners beyond pairs) → B7 $T=1.0$. **Never dropped:** B2 2D at $1024^2$
(pairwise), B7 $T=0.5$ at $1024^2$, B8 2D sampling adequacy at $1024^2$. A deferred item is reported as deferred with
its reason.

## 10. Code plan (new files; copies with provenance in `PROVENANCE.json`)

- `budcore.py` — `p1core.py` plus an output-count parameter (`nout`, default 6 ⇒ byte-identical stepping and outputs,
  test `test_budcore.py`) and an evolve-from-given-$w_0$ entry (split query already exists).
- `budcommon.py` — `p1common.py` plus the full-grid LS initialiser, the fine-lattice decoder, the corner/run registry.
- `b2run.py` — the 2D job driver (B2 corners, B7 rollouts with in-job certification, B8 in-job fine-route and diagnostics,
  controls), one mesh per job; `b3run.py` — the 3D driver on qualify's `s3ctx` / `s3common` (horizon generalised).
- `ref/` — the E0 generator copies (§3.1) and `refrun.py` (the reference job: levels, judge, fine route, gates G-R1–R3,
  C-R1), `psrun.py` (cross-check).
- `budget.py`, `b7.py`, `b8.py` — offline analyses (CPU) recomputing every predicate from raw arrays;
  `smoke_accept.py`; `make_configs.py`; `resources.py`, `ledger.py`, `cluster/{stage.py,submit.sh,collect.py}` (qualify's,
  namespace changed); `make_report.py` → `report.md`; `plots/`.

## 11. Deliverables

`report.md` (generated; level-1 title, status line, LaTeX, mermaid, inline provisional labels, plain-language
glossary), plots (PNG): budget bars per member and mesh (main effects and the largest pair, never stacked), interaction
heat maps, error-vs-time to $T$ with $\varepsilon_{\rm ref}$ bands, energy/dissipation curves, $s$ histograms;
`runs/EXPORT.json` for the coordinator; `runs/GPU-LEDGER.json`; `audits/` (every Codex audit); `REFS-MANIFEST.sha256`
for the new references.

## Glossary

- **r (R′)**: number of bank coefficients solved for. **M**: number of sine test functions (here $4r$). **m**: quadrature points.
- **LSPG / GAL**: least-squares (minimise the weighted residual) / Galerkin (residual orthogonal to the test space) reduced equations.
- **BE, CN-R (CNR), GAL-CN, BDF2**: time schemes (backward Euler; Crank–Nicolson with two BE start-up steps; Galerkin Crank–Nicolson; second-order backward differentiation).
- **factor**: step size in units of $\Delta t_0$.
- **E0 / $R_{ST}$**: the order-verified second-order reference (central + Crank–Nicolson); **$\varepsilon_{\rm ref}$**: its Richardson uncertainty estimate (conditional).
- **floor $e_P$**: distance from the reference to the bank span; no reduced solution can beat it.
- **ingredient, corner, effect vector, Möbius inversion**: one numerical choice switched to its ideal value; a trajectory with a given set of switches; the inclusion–exclusion combination of corners that isolates one ingredient or one interaction.
- **$u_\star$, $U_\star$**: the ideal-end trajectory and its own numerical uncertainty.
- **Q1 / Q2**: solver qualification (stationarity, distance to a tight solve) / quadrature qualification (agreement with a finer rule).
- **window $W_1,W_2,W_3$**: output-time ranges (0, 0.25], (0.25, 0.5], (0.5, 1].
- **fine route**: scoring on the $1025^2$ (2D) / $127^3$ (3D) lattice instead of $257^2$ / $63^3$.
- **sub16, v16, dev6, val32**: the case subsets named in §2.
- **must-fail control**: a deliberately broken input that a check must reject; if it does not, the check is a bug.
