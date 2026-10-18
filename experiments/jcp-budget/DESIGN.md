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

---

## Amendment A1 (2026-10-18, after `audits/codex-design-1.md`, NEEDS-RESTATEMENT; before any code ran at real size)

Every numbered fix of the audit is answered here. Where this amendment and the text above differ, this amendment wins.

**A1.1 Coverage.**
- *Candidate components and "two largest".* Trajectory ingredients (2D: Q, T, F, S, I; 3D: Q, T, S plus the form panel of
  A1.2) are the only switchable ingredients; every pair of them is run (2D: all 10 pairs; 3D: all pairs of Q, T, S, and
  F×Q, F×T in the form panel). The floor is **exactly non-interacting** with every in-span trajectory in squared norm
  ($e_v^2=e_P^2+\lVert v-\Pi u_{\rm ref}\rVert^2$, G-B0). The ideal dynamics interacts with the numerics through the cross
  term $\chi_v(t)=2\langle v-u_\star,\,u_\star-\Pi u_{\rm ref}\rangle$ in
  $\lVert v-\Pi u_{\rm ref}\rVert^2=\lVert v-u_\star\rVert^2+e_{\star,\rm dyn}^2+\chi_v$; $\chi_{u_a}$ is reported per case
  (negative = cancellation between numerical and dynamics error). So whichever two components are largest, their
  interaction is measured.
- *Form vs test space.* F is a **form sensitivity** (same test functions). The test-space *size* is jcp-dims' B6; this
  lane adds one single-ingredient ablation **M** (2D only, not in the factorial): $M=4r\to M'=2048$ (the full frozen list
  $F_{2048}$; $M'/r=16,8,5.33$ for $r=128,256,384$), deployed everything else, every case at every mesh; labelled
  "test-space enlargement sensitivity", not an ideal value.
- *Deferral.* Never deferred: $u_\star$ and its $U_\star$ replays, every corner with $|S|\le2$, and every run a gate or
  control needs. Only corners with $|S|\ge3$ (and the 3D $|S|=3$ corner other than the closure path of A1.2) follow
  defer item 3.

**A1.2 Endpoints.** The factorial far corner $u_{\rm all}$ switches the form; the common ideal is $u_\star$ = (GAL,
$q_\infty$, BDF2 $\Delta t_0/8$, tight, LS init). For LSPG members $u_{\rm all}=u_\star$. For GAL members $u_\star$ is the
corner $\{Q,T,S,I\}$ (form unchanged), and $D_a=\lVert u_a-u_\star\rVert$ closes on the Möbius sum over subsets of
$\{Q,T,S,I\}$ (vector identity); the full five-ingredient closure is reported separately as closure to $u_{\rm all}$.
3D: the factorial over $\{Q,T,S\}$ ends at `ad`-tight LSPG ($u_{QTS}$); $u_\star=u_{QTS}$ with the form switched to GAL,
and $u_\star-u_a=(u_{QTS}-u_a)+(u_\star-u_{QTS})$ exactly. **3D form panel:** at the `ad`-tight end, corners
{LSPG, GAL} × {$q$, $q_\infty$} × {$(s,h)$, (BDF2, $\Delta t_0/8$)} (8 runs, 4 new), giving the form effect at fixed tight
solver and its pairs with Q and T (labelled "at the tight-solver end", a context-specific sensitivity).
$U_\star$ is an **uncertainty indicator, not a bound**; its time aggregation is $\max_{k\ge1}U_\star(t_k)$, as for the
components.

**A1.3 Switch semantics.**
- *Form.* Two estimands, both reported: (i) the factorial switch F = form **plus its own production stopping rule**
  ("form-plus-stopping-policy sensitivity"); (ii) the **controlled form comparison** at tight tolerance,
  $\delta_{\rm form}^{\rm tight}=\lVert u_{\{S,F\}}-u_{\{S\}}\rVert$ (both corners are in the pairwise design) — combined's
  $\delta_{\rm form}$.
- *Time* is a "temporal-discretisation package sensitivity" (scheme, step, start-up steps, LSPG weights; in 3D also the
  adaptive-start length $a_{\rm eff}$ of `fs3.a_effective` at the new step).
- *IC (2D).* Full-grid LS by **blocked accumulation** over the row blocks of the rotated bank (`Bank.Grot`, column prefix
  $r$): $H=\sum_bG_b^{\mathsf T}G_b$, $g=\sum_bG_b^{\mathsf T}u_{0,b}$ in f64, Cholesky solve plus one step of iterative
  refinement; recorded: ${\rm cond}(H)$, the orthogonality residual $\lVert G^{\mathsf T}(Gc-u_0)\rVert/(\lVert G\rVert_F
  \lVert Gc-u_0\rVert)$ accumulated blockwise (gate G-B5: $\le10^{-8}$), and the peak memory (one block temporary; $H$ is
  $r\times r$). **The LSPG stopping scale is preserved:** every corner runs init → (optional $w_0$ replacement) → evolve,
  with the scale returned by the deployed init; only $w_0$ changes. All corners, including $u_\varnothing$, use this split
  path (so the IC sham is bitwise by construction); $u_\varnothing$ is tied to qualify's fused run by G-B3.
- *3D IC.* The deployed mesh path already solves the mesh normal equations (`TB.project`), so the **initialisation switch**
  is zero; the IC-projection **error** $e(t_0)$ is still reported against the floor $e_P(t_0)$.

**A1.4 Decision-rule completions.**
- *Validity.* A corner is valid iff `run_valid` (finite, no non-finite step, no damping saturation; `ad`: no failed step).
  An effect vector, pair label or component is **available** only if every constituent corner is valid; otherwise it is
  "unavailable", counted, and never enters "largest", "resolved" or "unresolved". Non-finite values are unavailable.
- *Ties in "largest".* Two candidates $a,b$ tie if $|a-b|\le\tau_{ab}$ with $\tau_{ab}=3U_\star$ (both trajectory-side),
  $2\varepsilon_{\rm ref}$ (both reference-side) or $2\varepsilon_{\rm ref}+3U_\star$ (mixed), all per case. A tie at the
  top makes the case "tied" (it counts for no component). $e_{\star,\rm dyn}$ is resolved only if it exceeds
  $2\varepsilon_{\rm ref}$ **and** $3U_\star$.
- *Mesh invariance.* "same" only if one identified component is dominant at all three meshes; if any mesh is
  "no single dominant", the comparison is **inconclusive**; otherwise "differs".
- *Growth.* Population = cases eligible for both windows with $e_{W_1}>2\varepsilon_{W_1}$ and $e_{W_2}>2\varepsilon_{W_2}$
  (resolved numerator and denominator); excluded cases are counted. Fewer than 8 cases → "inconclusive (population)".
- *Sampling adequacy.* Per case "consistent" iff $|e^{\rm fine}-e^{257}|\le0.10\,e^{257}+2(\varepsilon^{\rm fine}+
  \varepsilon^{257})$; cases with $e^{257}\le2\varepsilon^{257}$ are "reference-limited" and excluded (counted). Per
  configuration: **adequate** iff $\ge90\,\%$ of the remaining cases are consistent; **inadequate** iff $<50\,\%$; else
  **mixed**. Separately per mesh, **front-ranking adequacy**: no resolved reversal of the members' median order between
  the two routes (resolved = medians differ by more than twice the larger median $\varepsilon$ on **both** routes). The
  lane-level statement requires every LSPG member at $1024^2$ adequate and front-ranking adequacy at $1024^2$; it is
  stated **relative to the $1025^2$ route** (a finer feature could still be missed).
- *Stability* is "at the recorded output times" (transient growth between outputs is not excluded).

**A1.5 References.**
- *Fine route judged independently:* the fine-lattice fields of all three final levels are judged in-job with the full
  E0 judge (order, direction consistency from the difference vectors, floor, Péclet, IC); only the finest level's fine
  field is stored, with the fine judge record. A case's fine-route score is eligible iff its fine judge accepts **and**
  its lattice judge accepts.
- *Window eligibility:* the judge is applied per window to the fields restricted to $\{t_0\}\cup W$ (order, direction,
  floor per window time; solver, IC and Péclet over the whole run). $W_2$ eligibility comes from the $T=0.5$ references;
  $W_3$ from the $T=1.0$ references, whose $W_1\cup W_2$ outputs must agree with the $T=0.5$ ones within the G-R2 bar.
- *G-R1/G-R2 matching and bars:* comparisons are level-by-level at equal $(L,\Delta t)$ **and** equal final purpose
  (prod/retry/tight); purpose-mismatched levels are reported, not compared; at least the finest base level must be
  matched or G-R2 is "not evaluable" for that case (investigated, case-level). Bars: G-R1 runs on E0's GPU type for that
  case and must reach $\le10^{-12}$; on another GPU type the bar is G-R2's. G-R2: per time
  $\lVert\Delta\rVert/n_0\le\min(10^{-8},\,0.01\,\varepsilon_{\rm ref}(t_k))$ — agreement far below the reference
  uncertainty; a failure stops the reference production until explained in a dated amendment.
- *Refinement only at $T=0.5$* is a disclosed **resource restriction**; failed $T=1$ references stay non-rankable.
- *Pseudo-spectral cross-check, full E0 procedure:* per case three spatial runs at $\Delta t$ ($N=1024,1536,2048$ /
  $128,192,256$), a half-step run at $N_3$, one stability rerun at $\Delta t/2$ if needed, E0's PS gates (G3/G4) in the same
  job; E0's `crosscheck` (spatial-order indicator, $s_x+s_t\le\varepsilon/2$, $x_{\rm PS}\le2\varepsilon$) and the
  pre-A10 rule, per window. Cases (pre-declared, a subset of E0's pinned roster): **2D val32 {0, 1, 5, 19}** (two lowest
  indices, two smallest $\nu$); **3D val64 {0} and the smallest-$\nu$ case of E0's pinned 3D list**. Dimension status per
  window as E0 (A2.5) scaled to the subset: **validated** iff all PASS; **suspended** if any FAIL; else **unvalidated**.
  Suspended → that dimension's errors in that window are non-rankable (descriptive only, no B7 verdict); unvalidated →
  every number of that window carries "provisional (cross-check not validated)".

**A1.6 Controls (replacing the table rows of §7 where named).** Controls are demonstrated in the smoke on the smoke
population with the predicate "fires on at least one smoke case" and re-run in production on the stated check population
with the stated predicate. A control that does not fire means the **fixture or the check is unsuitable or wrong**; the
verdict it protects is withheld until a dated amendment resolves it.
- **C-F (exact).** Per time $k\ge1$: $r_k=(I-\Pi)u_{\rm ref}(t_k)$; $h$ = the lattice field $\sin(255\pi x)\sin(255\pi y)$
  (2D) / $\sin(63\pi x)\sin(63\pi y)\sin(63\pi z)$ (3D); $v_k=(I-\Pi)h-\langle(I-\Pi)h,\hat r_k\rangle\hat r_k$,
  normalised ($v_k\perp$ span **and** $\perp r_k$). Inject $u_{\rm ref}+\eta_kv_k$ with physical amplitude
  $\eta_k=10\,n_0\max(\text{all candidate components of that case at }t_k)$; $n_0$ and $t_0$ unchanged. Predicates: $e_P$
  equals $\sqrt{e_{P,k}^2+(\eta_k/n_0)^2}$ to $10^{-10}$ relative, $e_{\star,\rm dyn}$ and every trajectory component
  unchanged to $10^{-12}$, and the decision code labels the floor largest on **every** available case (guaranteed by the
  construction; the control tests the labelling code and the identity, not the physics).
- **C-E (closed form).** (i) ROM field := reference field → balance defect $\beta_k=0$ exactly; (ii) ROM field := reference
  field scaled by $(1+0.01k)$ at output $k$ → the diagnostic must reproduce the closed-form prediction
  $\beta_k=[(1+0.01k)^2-1]E_k-[(1+0.01(k-1))^2-1]E_{k-1}+\tfrac{\Delta t_{\rm out}}2([(1+0.01k)^2-1]\mathcal D_k+
  [(1+0.01(k-1))^2-1]\mathcal D_{k-1})$ (reference $E,\mathcal D$) to $10^{-10}$ relative, and the prediction is non-zero.
- **C-A (guaranteed).** At each $k\ge1$ add $a_k\varphi$, $\varphi=\sin(256\pi x)\sin(256\pi y)$ (2D; 3D
  $\sin(64\pi x)\sin(64\pi y)\sin(64\pi z)$), with $a_k\lVert\varphi\rVert_{\rm fine}=\lVert e_k\rVert_{\rm fine}+
  1.2\,e^{257}n_0^{\rm fine}$, so $\lVert e_k+a_k\varphi\rVert_{\rm fine}\ge1.2\,e^{257}n_0^{\rm fine}$ by the triangle
  inequality (sign-independent). Predicates: $|\Delta e^{257}|\le10^{-10}e^{257}$ (the alias is zero at the lattice nodes up
  to rounding) and $s\ge1.2>1.10$ on every check case; the case must then be "inconsistent" under A1.4.
- **C-I (plumbing).** $w_0\to1.01\,w_0$: the IC effect at $t_0$ must equal the independently computed
  $\lVert G_{\rm lat}(w_0^{\rm LS}-1.01w_0)\rVert/n_0$ to $10^{-12}$ relative and differ from the unperturbed value; its
  propagation to $t_1..t_5$ is reported only.
- **C-S.** Fixture: R384 LSPG-CNR $\Delta t_0$ (front member) with LM budget 0, cases dev6 0–5; comparator: the valid tight
  corner. Predicate: distance $>2.5\times10^{-4}$ on more than half of the cases. The budget-0 run is exempt from validity.
- **C-T.** Fixture: R128 LSPG-BE $\Delta t_0$ (front member), dev6: $\lVert I_T\rVert>10^{-3}$ on more than half (prior
  evidence: combined's BE time sensitivities ≈ 1 %).
- **C-Q.** Fixture: R384 LSPG-BE $\Delta t_0$ with deployed rule Gauss $32^2$ (qualify's under-resolved fixture), dev6:
  predicate **at least one case** $>2.5\times10^{-4}$ (qualify's established criterion), count reported. 3D: `lat256` as the
  deployed rule of W256 CN 0.01, cases 0–3, same predicate.
- **C-R1.** The first-order ladder must have `solver`, `floor`, `t0_shared` true and `order` **false** (checked
  explicitly); if a prerequisite fails the control is "not evaluable" and investigated.
- **C-R2 / C-score (identity-based).** The offline scorer carries identity tags (cohort, case, horizon, output-time list,
  reference kind and sha256, metric id, route); a case swap, a one-step time shift and (W1, LSPG-BE $\Delta t_0$ rows
  only, where E0 S exists) an ST↔S swap must each **raise an identity error**; additionally the numerical time-shifted
  score must differ from the aligned one (reported).
- **G-B6 (factorial truth).** Every corner records its configuration tuple (rule name and table fingerprint, scheme,
  $\Delta t$, form, tolerance level, init id, $M$); an in-job assertion checks that exactly the elements of $S$ differ
  from $u_\varnothing$'s tuple; G-B2 (algebraic closure) stays as a bookkeeping check.

**A1.7 Scoring and qualification completions.**
- Long-horizon Q2 adds the finer-rule convergence: $q^+$ vs $q^{++}$ on the $q^{++}$ cases meets the rollout bar to $T$,
  successor and check runs valid; otherwise "unverified (finer rule not converged)". Q2 labels name their population
  ("Q2 on sub16" in 2D, "on cases 0–7" in 3D) — not individual evidence for other cases.
- GAL certification is qualify's residual criterion ($\eta\le10^{-8}$, qualify A1.3/A2.5), a named deviation from the
  LSPG stationarity ratio.
- Front-order ties: a pair of members is ranked only if their medians differ by more than twice the larger of the two
  medians' $\varepsilon_{W}$.
- G-B4 extended: **every** converged-rule-dependent corner type (every corner containing Q, $u_\star$ and its $U_\star$
  replays) is replayed with the check rule on dev6 (2D) / cases 0–3 (3D); quadrature attribution is claimed only for
  corner types that pass, on the checked population.

**A1.8 Compute.**
- Smokes: the coordinator's rule is **one smoke job at a time, each ≤ 2 GPU-h** (not an aggregate); `smk1` (2D) then
  `smk2` (3D), sequentially. Smoke populations: 2D dev6 cases 0–1, 3D v16 cases 0–1. Production waits for the
  coordinator's release (which follows the C3 report).
- Memory: `smk1` includes a **combined-path memory pilot** at $4096^2$ on an H200: bank edges $[0,128,256,384]$, Gauss
  $320^2$ and $640^2$ tables at $r=384$, the $M'=2048$ test tables, the LS Gram accumulation, a fine-route decode and one
  B2 corner; peak device bytes recorded. If the peak exceeds 120 GiB, the $640^2$ check tables are built and dropped one
  role at a time (A16-style lifetime) and the pilot repeats. Reference certificates include the output buffers
  ($L=8192$, $T=1$: about 10.5 GiB; $512^3$, $T=0.5$: about 10.9 GiB).
- Ledger: qualify's `ledger.py` reserves each job's **time limit** at submission and keeps it reserved until `sacct`
  settles it, so spent + all outstanding reservations + the new projection must be ≤ 45.
- Retention: the new references live in `/cluster/tufts/paralab/tawal01/jcp-budget/refs/` (a data directory, not a job
  directory), with `REFS-MANIFEST.sha256` committed; the $257^2$/$63^3$ lattice fields and all judge records are pulled
  into the lane (small); dev6's fine fields are pulled for the offline recomputation; the remaining fine fields stay on the
  cluster until the coordinator decides (listed in the hand-back). Job directories are deleted after collection.

**A1.9 Other.**
- *Discrete diagnostics.* 2D: $E=\tfrac12h^2\sum u^2$ over the $1023^2$ interior fine nodes, $h=1/1024$; $\nabla_h$ by
  central differences at interior nodes using the zero wall values. 3D: $E=\tfrac12h^3\sum u^2$ over the $127^3$
  interior nodes, $h=1/128$, three central-difference components with zero values outside the interior. The balance
  quantity is a **difference of discrete balance defects** $\beta_k$; no cancellation claim is made.
- *Pins (complete):* FRONT-2D.json sha256 `a51ac1a250f00d4b0006a84f2434ab3c96407b1f575f4ffff7a3eb25b7a8b77d` (its own
  producing-commit field `ef1a73134981…`); qualify EXPORT.json sha256
  `6cb394d149be8852260b75a585db8102791ef52e9963ab6934f67d536c374431`; code copied from qualify at `4399491eb` and from
  references at `cc04f4b54` (`PROVENANCE.json`).
- *New gate G-B5* (LS init orthogonality, A1.3) and *G-C1* (`test_budcore.py`: `budcore` with nout = 6 equals `p1core`
  bitwise; the nout = 11 run's first six outputs equal it bitwise) join §7.

## Amendment A2 (2026-10-18, after `audits/codex-design-2.md`, NEEDS-RESTATEMENT; before any real-size run)

(`codex-design-2.md` was recovered verbatim from the `codex exec` stdout log: the `-o` file was not written.)

**A2.1 Component universe and interactions.**
- The **candidate set** of the "largest component" contest is exactly: floor $e_P$, ideal dynamics $e_{\star,\rm dyn}$,
  and the main effects of the factorial ingredients (2D: Q, T, F, S, I; 3D: Q, T, S, and F at the tight-solver end).
  The M ablation (A1.1) is **outside** the contest: it is a test-space-size probe (owned by jcp-dims B6), reported beside
  the budget with its own validity, and no interaction with it is claimed.
- **Ingredient–dynamics interaction**, per ingredient $x$ and time: $\chi_x=2\langle I_x,\,u_\varnothing-\Pi u_{\rm ref}\rangle$,
  so that $\lVert u_x-\Pi u_{\rm ref}\rVert^2=\lVert u_\varnothing-\Pi u_{\rm ref}\rVert^2+\lVert I_x\rVert^2+\chi_x$
  exactly; $\chi_x<0$ means switching $x$ partly cancels the deployed in-span error. Reported per case (max over $t_k$ of
  $|\chi_x|$ with its sign at that time), normalised by $n_0^2$. The floor needs none (exactly orthogonal).
- **3D F×S is not defined:** the deployed fixed-sweep / QN solvers have no Galerkin counterpart, so the form switch exists
  only with the adaptive solver. The 3D claim is narrowed accordingly: in 3D, "the pair of the two largest components
  is measured" holds for every pair except one containing both F and S, which is reported as "not defined". (Lane
  decision; recorded for the coordinator.)

**A2.2 Controls: per-dimension inventory (supersedes the §7 rows and A1.6 where named).** "Fires" predicates are
evaluated on production check populations; the smoke demonstrates each on the smoke population (≥ 1 case).

| control | 2D fixture and predicate | 3D fixture and predicate |
|---|---|---|
| C-Q | R384 LSPG-BE $\Delta t_0$, deployed rule → Gauss $32^2$; $\lVert u(q_\infty)-u(32^2)\rVert>2.5\times10^{-4}$ on ≥ 1 dev6 case. **Changed comparator** ($q_\infty$ = Gauss $320^2$, qualify compared with Gauss $64^2$): qualify's result is supporting evidence only | W256 `fs` CN 0.01 K1, deployed rule → `lat256`; vs gl56; ≥ 1 of cases 0–3 |
| C-T | R128 LSPG-BE $\Delta t_0$ (member): $\lVert I_T\rVert>10^{-3}$ on > half of dev6 | W256 gl24 `fs` **BE** 0.01 K1 (fixture, not a member): $\lVert u-u_T\rVert>10^{-3}$ on > half of cases 0–3 |
| C-S | R384 LSPG-CNR $\Delta t_0$, LM budget 0 vs the tight corner: $>2.5\times10^{-4}$ on > half of dev6 (budget-0 run exempt from validity) | W256 lat8192 `qn` BDF2 0.025 with mutation `q0` (no iterations) vs `ad` tight: same predicate on cases 0–3 |
| C-I | perturbed init $w_0\to1.01w_0$: the decoded $t_0$ difference perturbed − unperturbed equals $0.01\lVert G_{\rm lat}w_0\rVert/n_0$ to $10^{-12}$ relative (guaranteed non-zero); propagation reported | not applicable (no IC switch in 3D, A1.3) |
| C-F | as A1.6 with $\eta_k=n_0\,(10\max_c+2\varepsilon_{\rm ref}+3U_\star)$ ($\max_c$ = largest candidate of that case at $t_k$), which exceeds every tie threshold, so the floor is labelled largest on every available case by construction; seed fallback: if $\lVert(I-\Pi)h\rVert<10^{-8}\lVert h\rVert$ or the twice-orthogonalised seed is below $10^{-8}$ of it, the next seed $\sin(254\pi x)\sin(254\pi y)$, …; if $r_k=0$ the second orthogonalisation is skipped | same on $63^3$ with seed $\sin(63\pi x)\sin(63\pi y)\sin(63\pi z)$, fallback $62\pi$, … |
| C-E | tested on the **raw** (un-normalised) balance: (i) ROM := reference → $\beta_k=0$ exactly; (ii) scaled reference → closed form of A1.6 to $10^{-10}$ relative (absolute floor $10^{-15}E_{\rm ref}(t_0)$); normalised values then follow by the stated division | only with B8 3D (deferrable); otherwise "not applicable (deferred)" |
| C-A | amplitude against the **actual rejection threshold**: $a_k\lVert\varphi\rVert_{\rm fine}=\lVert e_k\rVert_{\rm fine}+n_0^{\rm fine}\,(1.2\,e^{257}+2(\varepsilon^{\rm fine}+\varepsilon^{257})+10^{-6})$ with the window $\varepsilon$'s, which guarantees $e^{\rm fine}-e^{257}>0.10\,e^{257}+2(\varepsilon^{\rm fine}+\varepsilon^{257})$, i.e. "inconsistent"; and $|\Delta e^{257}|\le10^{-10}e^{257}$ | only with B8 3D, $\varphi=\sin(64\pi x)\sin(64\pi y)\sin(64\pi z)$ |
| C-R1 | first-order ladder on dev6 case 0 at $T=0.5$, prerequisites true and `order` false | X1 3D ladder on val64 case 0 at $T=0.5$, same |
| C-score | identity errors for case swap, time shift, ST↔S (W1 LSPG-BE $\Delta t_0$ rows) | identity errors for case swap and time shift (no S in 3D) |
| G-Q | budget-0 LSPG and GAL runs: non-certified steps > 1 %; wrong-table guard raises | `qn` `q0` run: non-stationary steps > 1 % under `qn3.make_certify` |

**A2.3 References.**
- **Retry and refinement are driven only by the whole-horizon lattice judge** (E0 procedure, $257^2$ / $63^3$, all output
  times). The window judges and the fine-route judges are evaluated on the resulting final triple; they never trigger
  more runs.
- **Reference pairing:** B8 compares the two routes on the **same FD realisation** (this lane's final triple for both
  $e^{257}$ and $e^{\rm fine}$). B7 growth ratios use this lane's references for **both** windows (same realisation);
  $W_1$ errors of the budget tables (B2) and of the B7 qualification tables are scored against E0 (primary, as every
  lane). The per-case E0-vs-new $W_1$ difference is reported (G-R2).
- **G-R2:** $t_0$ fields must be bitwise equal (shared IC); evolved times use the A1.5 bar. A case whose G-R2 is "not
  evaluable" or fails has **non-rankable** long-horizon windows (descriptive only); G-R2 is a declared stopping criterion
  for cross-GPU differences, not an established guarantee.
- **3D cross-check case:** E0's smallest-$\nu$ pinned 3D case is **val64 case 43** ($\nu\approx0.010006$), outside v16; its
  FD ladder to $T=0.5$ is added to `ref3` (it is used only for the cross-check, never in B2/B7 populations).
- **PS escalation:** E0's one-step escalation ladder (2D $N=1536,2048,3072$; 3D $192,256,384$) runs once for a case whose
  attempt-0 verdict is UNRESOLVED for spatial order or resolution (E0 A1.7); its cost is in the certificate reserve.
- **Eligibility tiers** per dimension and window, from the cross-check status: **validated** → ranked comparisons and
  verdicts; **unvalidated** → provisional comparisons, labelled, **no** verdicts; **suspended** → descriptive only. A
  designated cross-check case is ranked only if its own cross-check PASSes. **$W_3$:** if B7 $T=1.0$ runs, PS
  cross-checks to $T=1.0$ on 2D val32 {0, 19}; without them $W_3$ is "not cross-checked" → descriptive only.

**A2.4 Decision-rule completions.**
- *Incomplete candidate sets:* per case the largest is computed only if every candidate is available; otherwise the
  case is "incomplete" (counted; "largest among available" reported descriptively). Dominance is evaluated over complete
  cases and is "inconclusive (incomplete)" if fewer than 75 % of eligible cases are complete.
- *Minimum populations:* any label computed over cases (dominance, pair label, growth, sampling adequacy, front order)
  needs ≥ 8 cases after all exclusions (≥ 6 for sub16 at $4096^2$), otherwise "inconclusive (population)".
- *Growth* rule applies to every adjacent window pair ($W_2/W_1$, $W_3/W_2$) with both windows resolved and eligible.
- *Front order:* common population = cases eligible in every compared window/route for **all** members; Kendall
  $\tau_b$ (ties allowed); "preserved" iff ≥ 1 resolved pair and no resolved reversal; "inconclusive" if no pair is
  resolved.
- *Zero denominators:* near-wall density with zero total error → "zero error"; IC deployed/floor ratio with zero floor →
  "undefined (zero floor)"; mesh component ratios with a zero denominator → case excluded and counted.

**A2.5 Anchors must be converged; quadrature attribution population.**
- **Numerical anchors** (every tight, tighter, $u_\star$-family and check-rule run) must be valid **and** have every step
  verified by the solver's own stopping test (`stats.nfail == 0`: stationarity ≤ gtol or residual ≤ tol from the
  returned state, no budget or damping exit). Deployed-tolerance corners use `run_valid` (budget exits allowed, as
  qualify) and record their exit counts. An anchor that fails this makes every contrast using it unavailable.
- **Quadrature attribution:** G-B4 is evaluated on dev6 (2D) / cases 0–3 (3D) for every Q-containing corner type. The
  Q component on other cases is labelled "extrapolated from the dev6 check" (the converged rule is a property of the rule,
  checked on a sample); if G-B4 fails for a corner type, the Q component is **withheld on every case** and the dominance
  labels are computed without it and marked "Q withheld".

**A2.6 Switch semantics completions.**
- **M ablation:** the deployed configuration with the frozen $F_{2048}$ list (same rule, form, scheme, step, tolerance,
  init); labelled "test-space enlargement under the deployed quadrature and solver package"; its validity and exit
  counts are reported; no Q1/Q2 claim is made for it.
- **LS init by blocked TSQR** (replaces the normal equations of A1.3): per row block $G_b=Q_bR_b$ (QR of the block, once
  per $r$); $Z=[R_1;\dots;R_n]=Q_ZR$; per case $Q^{\mathsf T}u=Q_Z^{\mathsf T}[R_b^{-\mathsf T}G_b^{\mathsf T}u_b]_b$ and
  $c=R^{-1}Q^{\mathsf T}u$. Conditioning is that of $G$ (not squared); ${\rm cond}(R)$ recorded; G-B5 orthogonality
  $\le10^{-8}$; ${\rm cond}(R)>10^{10}$ is flagged in the report.
- **G-B6 compares logical ingredients** (rule role, form, (scheme, step), tolerance level, init kind); derived changes
  (the form's stopping tolerance, the scheme's start-up steps and weights) are not asserted. Inherited operator checks
  stay: table fingerprints per role (`BSetting.check`), the wrong-table guard, and G-B3's reproduction of qualify's runs.

**A2.7 Compute.** The smoke allowance is the coordinator's launch instruction to this lane (SendMessage of 2026-10-18,
quoted): "at most ONE cluster smoke job at a time, ≤ 2 GPU-h each; generating new long-horizon references counts as
production unless it is a ≤ 2 GPU-h pilot". Repriced inventory (estimates; certificates decide):

| job | GPU-h |
|---|---|
| smk1, smk2 (sequential) | 2 + 1.5 |
| ref2a/b/c (38 cases, $T=0.5$) | 6 |
| ps2 (4 cases + escalation reserve) | 2 |
| ref3 (v16 + case 43, $T=0.5$; PS 2 cases) | 3 |
| b1024 (full factorial, checks, M ablation, B7, B8) | 4 |
| b256, b4096 | 1.5 + 2.5 |
| b3d (B2 + form panel + B7) | 4 |
| ref2t1 + ps $T=1$ + b1024t1 | 6.5 |
| **total** | **≈ 33.5**, ×1.25 ≈ 42 ≤ 45 |

Memory certificates include table builds, the 3D construction peaks (`Tables` and Gauss $56^3$/$64^3$ at W512), the
reference output buffers and refinement levels, and the retained fine arrays, measured in the smokes.

## Amendment A3 (2026-10-18, after `audits/codex-design-3.md`, NEEDS-RESTATEMENT; before any real-size run)

**A3.1 Interaction coverage (audit-3 fix 1).**
- **M joins the 2D contest and the pairwise design.** The 2D ingredient set becomes {Q, T, F, S, I, M} with M: $4r\to2048$
  test modes (A2.6 semantics). Pairwise corners now include the five pairs with M (2D pairwise design: $1+6+15=22$
  corners per case; the $1024^2$ full factorial stays over {Q, T, F, S, I} (32 corners) plus the M main effect and its
  five pairs). So in 2D the interaction of **any** two candidate ingredients is measured.
- **Ideal-dynamics interaction, exactly:** $\chi^\star_x=2\langle I_x,\,u_\star-\Pi u_{\rm ref}\rangle$ per ingredient and
  time (the ingredient's effect against the ideal-dynamics vector), reported separately from A2.1's
  $\chi_x=\chi^\star_x+2\langle I_x,u_\varnothing-u_\star\rangle$, whose second term is labelled "numerical background".
  The 3D form effect's background is the tight end (its own row).
- **3D F×S.** An interaction needs all four corners; "GAL with the deployed fixed-sweep / QN solver" does not exist in the
  code or the method (those solvers minimise the LSPG residual), so F×S is **undefined**, not unmeasured. The 3D
  coverage statement reads: "every pair of defined ingredients is measured; F is defined only at the adaptive-solver end
  and its pairs with Q and T are measured there". This is the brief's contract restricted to defined estimands; it is
  flagged in the hand-back for the coordinator. No other narrowing remains.

**A3.2 Missing or unverified quadrature (fix 2).** If any candidate is unavailable on a case — including Q withheld by
G-B4 — the case is "incomplete" and dominance follows A2.4 (inconclusive below 75 % complete); "largest among available"
is descriptive only. A G-B4 failure for the $u_\star$ family marks $u_\star$ **quadrature-unverified**: then
$e_{\star,\rm dyn}$, $\chi^\star_x$, $D_a$ closure and every $U_\star$-based resolution are labelled unverified and
excluded from verdicts (dominance → inconclusive). This supersedes A2.5's "computed without it".

**A3.3 $T=1$ cross-check FD case (fix 3).** The $T=1.0$ FD references cover sub16 **plus val32 case 19** (17 cases);
case 19 is used only for the $W_3$ cross-check (never in B7 populations). Cost +0.4 GPU-h (inventory: `ref2t1` 6.9).

**A3.4 Stable LS (fix 4).** The LS initialisation applies $Q_b^{\mathsf T}$ directly: one streamed pass over the row
blocks computes the reduced QR $G_b=Q_bR_b$ of each block, applies $Q_b^{\mathsf T}$ to the stacked right-hand sides of
**all** cases of the job ($u_b$ columns), keeps $R_b$ and $Q_b^{\mathsf T}U_b$, and discards $Q_b$; then
$[R_1;\dots;R_n]=Q_ZR$ and $C=R^{-1}Q_Z^{\mathsf T}[Q_b^{\mathsf T}U_b]_b$. Householder QR is stable for rank-deficient
blocks (no local inverse is formed). Acceptance: ${\rm cond}(R)\le10^{10}$ and G-B5 orthogonality $\le10^{-8}$ per case;
otherwise the I ingredient (and every corner containing it, and $u_\star$) is **unavailable** for that $r$ and mesh.

**A3.5 Check-rule acceptance is solver-specific (fix 5).** Check-rule replays keep the corner's own solver. Adaptive LM
runs (all 2D runs; 3D `ad`/`adg`): valid and `stats.nfail == 0`. 3D fixed-sweep / QN runs: `S.run_valid` (finite, no
reason-3 step, no damping saturation) **and** the speed3d per-step certificate (fresh contraction,
$\gamma\le10^{-3}$), non-stationary steps ≤ 1 % pooled over the check run's steps — the same acceptance as the deployed
runs of that solver. Deployed-tolerance 3D corners are certified the same way.

**A3.6 C-A (fix 6).** Two predicates: (i) **raw alias detection** on every check case: $|\Delta e^{257}|\le10^{-10}e^{257}$
and $e^{\rm fine}_{\rm injected}-e^{257}>0.10\,e^{257}+2(\varepsilon^{\rm fine}+\varepsilon^{257})$ (guaranteed by the
A2.2 amplitude); (ii) the **production classifier** labels the injected case "inconsistent" on every check case that is
resolved ($e^{257}>2\varepsilon^{257}$); unresolved check cases are counted, not classified.

**Inventory change:** 2D pairwise corners 16 → 22 per case (+6 short runs; b1024/b256/b4096 +15 %), `ref2t1` +0.4:
total ≈ 34.5 GPU-h, ×1.25 ≈ 43 ≤ 45. If a certificate exceeds the cap, the defer order of §9 applies.

## Amendment A4 (2026-10-18, after `audits/codex-code-1.md`, NEEDS-FIX; no real-size run yet)

Clarifications and decisions made while fixing the code (none changes a registered estimand or bar):
- **B8 3D is deferred now** (defer order item 2): its fine-route decode at $129^3$ is not implemented in this lane's code;
  B8 3D, C-A 3D and C-E 3D are reported as "deferred (defer item 2)". All other 3D items stay.
- **Smoke references are not promoted** into `refs/`: production regenerates dev6 case 0 (`ref2a`), so no smoke
  artefact can enter a production verdict; the smoke's fine files are pulled for the smoke acceptance only.
- **PS escalation** is a separate attempt (`attempt = 1`, E0 `esc` sizes) staged after `anaref.py` reports an
  UNRESOLVED (spatial-order or resolution) verdict, exactly as E0 ran it; its cost is in the `ps2`/`ps3` reserve.
- **Smoke predicates** for the empirical controls are "fires on ≥ 1 smoke case"; production uses the registered
  predicates on the check populations.
- **Campaign-wide job cap:** `submit.sh` refuses at ≥ 8 account jobs observed at submission (squeue before and after);
  a race with another lane submitting in the same seconds is possible and is the coordinator's to arbitrate (residual
  risk, recorded).
- **Dominance denominators:** dominance is computed over complete cases and needs ≥ `minpop` complete cases and ≥ 75 %
  of eligible cases complete (A2.4); pair labels use the eligible-case denominator of A1.4 with ≥ `minpop` available.
- **Identities vs effects:** G-B0 and G-B2 are evaluated on every run-valid corner (they are identities); effects,
  pair labels and dominance use corners that are also quadrature-verified (G-B4, A3.2).
- (A4 addendum, 2026-10-18 14:30 UTC) **Campaign-wide submission lock:** following COMMON-BRIEF §14, `submit.sh` now runs
  every `sbatch` under the shared cluster flock `/cluster/tufts/paralab/tawal01/.campaign-sbatch.lock` with the account
  count (< 8) checked inside the lock; exit 75 (at cap) releases the reservation. The residual race recorded above is
  closed. Local `jaxrun` jobs go through `campaign/jaxslot.sh` (§13).

## Amendment A5 (2026-10-18): production release recorded

The coordinator released production for jcp-budget (SendMessage, 2026-10-18, quoted): "production is released for
jcp-budget (including the long-horizon reference generation) as soon as your design/code/smoke gates are clean. Submit
through the shared campaign sbatch lock (§14; exit 75 = account at cap 8 → retry later). Lane cap 3 concurrent jobs,
45 GPU-h." Production jobs are staged only after the smoke acceptance passes (`checks/smoke-smk1.json`,
`checks/smoke-smk2.json` all PASS) and each job's Codex audit says SUBMIT; certificates come from the measured smokes.
Job-audit 1 of `smk1` (FIX-FIRST) added two acceptance rows before submission: an offline G-B6 check of every saved
corner tuple against the configuration, and C-A evaluated through the production classifier (≥ 1 resolved case
classified "inconsistent").

## Amendment A6 (2026-10-18, after the `smk1` smoke, before any production job)

- **G-B2 bar restated (a defect of the check, found on real data).** The registered bar "$\le10^{-12}\lVert u_{\rm all}-
  u_\varnothing\rVert$" mixes scales: the Möbius sum adds 32 corners of norm about $n_0$, so its rounding is about
  $10^{-15}n_0\cdot32$, while $\lVert u_{\rm all}-u_\varnothing\rVert$ is about $10^{-3}n_0$; the smoke gave relative
  residuals up to $3.4\times10^{-12}$ (bookkeeping exact, bar too tight). The bar becomes
  $\lVert u_{\rm all}-u_\varnothing-\sum_SI_S\rVert\le10^{-12}\max_S\lVert u_S\rVert$ (same in 3D). G-B2 remains a
  bookkeeping check; no estimand changes.
- **Reference identity across numpy versions.** The cohort sha256 recorded on the cluster (numpy 2.5) differs from the
  local one (numpy 2.4) because the log-uniform viscosity draw passes through `np.exp`, whose last bit differs across
  versions (the shared IC differs by $2.2\times10^{-16}$ for the same reason). The offline identity check therefore
  compares the recorded case descriptor with the local cohort descriptor to $10^{-13}$ relative instead of the hash;
  the in-job check (same numpy as the generator) keeps the hash.

## Amendment A7 (2026-10-18, after `audits/codex-job-refs2d-1.md` and `codex-job-smk2-1.md`, before production submission)

- **Gating of production by dimension:** 2D production (references, PS, b-jobs) is gated on `smk1` (2D smoke, ALL PASS,
  `checks/smoke-smk1.json`) only; 3D production on `smk2`. A5's "both smoke files" wording is restated accordingly (the
  2D jobs never execute 3D code).
- **Reference certificates repriced:** per job, nominal smoke case time × cases, plus a reserve for up to two cases taking
  the full retry path (tight re-run ×1.5 and the refinement level ≈ 3.4 × the finest-level time in 2D, 5.1× in 3D),
  plus 10 min; memory notes record the smoke's measured peaks and the refinement estimate. **PS** is priced from E0's
  0.17 A100-h per case at $T=0.25$ × 2 (horizon) × 3 (worst-case stability rerun), no H200 credit, and split into two
  jobs (`ps2a` cases 0, 1; `ps2b` cases 5, 19). An escalation attempt, if needed, is a separate certified job.
- **Collection procedure:** promotions run **serially**; `ref2a` (contains dev6) is collected **in full** (its dev6
  fine fields are needed for the offline fine recomputation, A1.8); `ref2b`, `ref2c` with `--no-fine` after promotion;
  `anaref.py` runs on all collected `archive/output` trees together (FD and PS).

## Amendment A8 (2026-10-18): b-job split and cost model (no estimand change)

- `b1024` is split by bank prefix into `b1024a` (R128 + R256 members) and `b1024b` (R384 members + the 2D controls) to
  run concurrently (each its own job directory); the two analyses are merged member-wise for the mesh comparison.
- `resources.py` prices b-jobs from the smoke's phase times (star/setup, corners per member-case, the check phase per
  member-check-case, B7 and B8 per member-case; an unmeasured $r$ takes the slowest measured $r$), not from one
  aggregate overhead.
