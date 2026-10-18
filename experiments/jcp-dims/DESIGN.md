# jcp-dims — how do representation size $r$, test-space size $M$ and quadrature size $m$ each drive accuracy and cost, how many points does each rule family need, and does a quadrature-qualified 3D speed-up exist? (pre-registration)

**Status:** pre-registration, 2026-10-18, written before any code or data of this lane. Amendments are appended at the
end, dated; nothing above them is rewritten. Every accuracy number of this lane is scored against the E0 second-order
reference $R_{ST}$, whose per-case uncertainty $\varepsilon_{\rm ref}$ is a **conditional Richardson estimate** (not a
rigorous bound; the E0 cross-check rule was amended after data, E0 A10). Post-hoc labels carry forward unchanged: E0 A10;
combined A10, A11 and the retracted A15.

Governing documents (later supersedes earlier): `reports/2026-10-17-full-experiment-plan/PLAN-v2.md` (A3, A4, B3, B6),
`PLAN-v3-amendments.md` (V3-1, V3-3, V3-5), `PLAN-v4-amendments.md` (V4-1, V4-2, V4-5); `campaign/P1-RESTATEMENT.md` +
`P1-RESTATEMENT-AMENDMENTS.md` (AM-2 Q2, AM-4/AM-7 verdict style); `campaign/briefs/COMMON-BRIEF.md` (binding);
`campaign/briefs/jcp-dims.md` (lane brief, wins on conflict); `reports/2026-10-08-jcp-lane-agent-rules.md`.

Lane: worktree `worktrees/2026-10-18-jcp-dims`, branch `exp/2026-10-18-jcp-dims` (fork `21c175a1b`, setup `2045e5908`),
lane dir `experiments/jcp-dims` (here), namespace `/cluster/tufts/paralab/tawal01/jcp-dims/`, lane cap **3** concurrent
jobs (campaign cap 8, checked on the account before every submit), **55 GPU-h** including contingency, with reservation.

Built on (read, cited; code copied with provenance; their certified results are **cited, not re-run** except where a
reproduction gate needs them): `jcp-qualify` (`worktrees/2026-10-18-jcp-qualify`, final: Q1/Q2 machinery, 2D driver
`qacc.py`, timing driver `qtime.py`, FRONT-2D, the 3D Q2 rounds that stopped at the round limit with 16 exposed and 146
open candidates); `jcp-p1-scaling` (frozen mode lists, `p1core`/`p1common`, kernel harness, statistics); `jcp-speed3d`
(3D context `s3ctx`, solvers `fs3`/`qn3`, screening and confirmation timing, comparators C_old, C_W768, C_W1024, its 436
eligible screened configurations); `jcp-wide-bank` (configurable $M$: `w2d.py`, `w3d.py`; first-order M/r = 2, 3, 4
results, PROVISIONAL); the 2026-10-01 quadrature lanes (rule generators: 2D `qcore.offmesh_rule`, Hari's
`hari_quadrature.py`; 3D `rules.py`); `jcp-mechanism` (the gain comes from the exact gradient, not point placement).

Prior facts that shape the questions (cited, not re-derived): K2 fired in 2D and 3D (a spectral FOM wins at equal
accuracy); speed3d's H2 winners are **not** quadrature-qualified (qualify: successor-rule distances 8.1e-4, 9.5e-4 >
2.5e-4); in 3D only 4 of 55 tested speed3d configurations passed Q2, all at W256 or W512; wide-bank's 2D/3D trims to
$M/r<4$ were "not acceptable" against first-order references and its predicted "smaller $M$ needs fewer points" was
contradicted. The low mixed-frequency idea comes from the paper plan (`2026-10-06-jcp-offmesh-paper-plan/02`,
`07` C4): Smolyak fails because the sine tests have high mixed frequencies.

## 0. Questions and outcome labels

| item | question | outcome labels |
|---|---|---|
| **A3** | in the factorial $r\times M\times m$ (2D at $1024^2$, 3D at $65^3$), how do accuracy and cost change along each axis with the other two held fixed, and how does the change along the deployed path ($M=4r$, $m=m^\*$) split into representation, test-space and quadrature parts? | per transition: improves / does not improve / inconclusive (AM-7, §6.1); per axis cost elasticity with interval (descriptive); decomposition shares with intervals (descriptive) |
| **A4** | $m^\*(r,M)$: the smallest qualifying point count per rule family on a pre-declared candidate list, own reached states, both Q2 bars | $m^\*$ = a candidate (stable / non-monotone) or **none**, or "not reached" (§6.2) |
| **A4-3D front** | complete the 3D Q2 search (the 16 exposed candidates first) so every width W256–W1024 has qualified members; re-time qualified 3D configurations against C_old, C_W768, C_W1024 in paired panels | per width: covered yes/no; H2q per comparator: pass / fail / no qualified tie-or-beat configuration (§6.5) |
| **B3** | do rule families (Gauss, lattice, Sobol, Smolyak, 2D Halton; then MC and 3D Halton) reach qualification with different point counts, and do they converge to the same answer? | per family pair: fewer points / comparable / qualifies-vs-not (§6.3); cross-family limit agreement pass / flagged |
| **B6** | does accuracy change with the test-space ratio $M/r\in\{1,2,3,4,8\}$; does the weak control $M=r$ degrade; is Galerkin ($Q^{\mathsf T}\mathcal R=0$) more or less accurate than LSPG at each ratio; does a low mixed-frequency test set help Smolyak? | per transition: improves / does not improve / inconclusive; control fired / not fired (blocks B6); GAL vs LSPG: GAL more accurate / LSPG more accurate / neither / inconclusive (§6.4) |

Three labels are kept separate everywhere (V4-2): (1) **numerical qualification** (Q1 + Q2), (2) **role
qualification** (§6.5 only: "ties-or-beats comparator C"), (3) **frontier usefulness** (not dominated). An unqualified
cell is reported with its failing label and never enters a verdict, a front or an $m^\*$.

## 1. The reduced model (unchanged from qualify §1 / speed3d §1)

**2D.** Bank coefficients $c\in\mathbb R^{r}$, $u=\hat G_r(x)\,c$ (frozen checkpoint `dn256b`, $R=512$, $K=16$,
rotation `rotation_R512`, ordered nested prefixes, full R512 bank with edges $[0,128,256,384,512]$). Test space: the
first $M$ entries of an explicit ordered list of discrete sine modes, $A=\Phi_M^{\mathsf T}\hat G_r$, $\Lambda$ the discrete
eigenvalues; tested advection by an off-mesh point rule $(x_q,w_q)_{q=1}^m$, $N(c)=\Psi^{\mathsf T}[(G_qc)\odot(G_sc)]$;
two-step LMM residual
$$\mathcal R_n(c)=A(\alpha_0c+\alpha_1c_n+\alpha_2c_{n-1})+\Delta t\,(\beta_0F(c)+\beta_1F(c_n)),\qquad F(c)=N(c)+\nu\Lambda Ac.$$
LSPG minimises $\lVert D\mathcal R_n\rVert$, $D=(\alpha_0+\beta_0\Delta t\,\nu\Lambda)^{-1}$; GAL solves
$Q^{\mathsf T}\mathcal R_n(c)=0$ with $A=QR_m$ (thin QR). Tolerances, budget, solver, initialisation contract
("full-field input sampled at quadrature points; full-field decode"), horizon 0.25 and outputs every 0.05: exactly
qualify §1 (`p1core.py`, `p1common.py` copied unchanged; the only change is the explicit $M$, §2.1).

**3D.** Speed3d's model at $65^3$ (W1024 bank prefixes W256/W512/W768/W1024, M2 bank M2_512), LSPG on the $M$ lowest
mesh sine tests with shell completion `complete_M` by default, off-mesh rule tables $B=\hat G(X)$, $D$, $P$; solvers
FS-K$k$ (`fs3`), QN-$k$ (`qn3`) and the adaptive reference ADT (`t2core`, level tight); cohort 923801 (64 cases); the
A2.3 lattice metric on $63^3$. No 3D Galerkin solver exists; B6 in 3D is LSPG only (scope statement, not a deferral).

## 2. Configurations (frozen before data)

### 2.1 2D: the factorial $\mathcal F_2$ at $1024^2$ (A3, A4, B6)

**Mode lists.** One explicit ordered list $F_{4096}$ = `modes_lean(1024, 4096)`; a cell with test count $M$ uses its
prefix $F_M$ (nested by the stable sort; V4-1 "fixed-mode" list at $1024^2$, the only mesh of this lane, so policy
"frozen" and "nominal" coincide here). Positive gate G-modes: the first 2048 entries of $F_{4096}$ equal P1's frozen
$F_{2048}$ (`list_hash_2048` recorded by qualify's `acc1024`), else stop.

**Cells.** $r\in\{128,256,384,512\}$; for each $r$ the test counts
$$\mathcal M(r)=\{\kappa r:\kappa\in\{1,2,3,4,8\}\}\cup\{4r':r'\in\{128,256,384,512\},\,4r'\ge r\}\cup\{2048\},$$
i.e. R128: {128, 256, 384, 512, 1024, 1536, 2048}; R256: {256, 512, 768, 1024, 1536, 2048}; R384: {384, 512, 768, 1024,
1152, 1536, 2048, 3072}; R512: {512, 1024, 1536, 2048, 4096} — **26 $(r,M)$ cells**. The union contains every panel:
fixed $(M,m)$ ($M=2048$, any rung, every $r$), fixed $m$ with $M=4r$, fixed $(r,m)$ over $\kappa$ (B6), fixed $(r,M)$ over
$m$ (A4), and every corner of the deployed-path decomposition (§6.1c).

**Gauss ladder** $\mathcal L_G$ = gauss64, gauss96, gauss128, gauss160, gauss192, gauss256, gauss320, gauss384
($m=p^2$: 4,096 … 147,456). Every $(r,M)$ cell runs every rung: **208 cells** per scheme.
**$m^\*$ candidates** (each needs $q^+$ and $q^{++}$ on the ladder): gauss64 … gauss256.

**Schemes per cell.**
- **LSPG-CN-R, $\Delta t_0=0.005$** (registered accurate LSPG scheme, primary in fom2d): all 208 cells.
- **GAL-CN, $\Delta t_0$** (Galerkin; GAL-CN stays a diagnostic per the coordinator decision of 2026-10-18 — its
  B6 comparison is reported as such): the 20 $\kappa$-cells ($r\times\kappa$) × all 8 rungs = 160 cells.
- **LSPG-CN, $\Delta t_0$** (the same LMM as GAL-CN, so GAL vs LSPG compares the solve equation alone): the 20
  $\kappa$-cells × all 8 rungs = 160 cells.
- **LSPG-BE, $\Delta t_0$** (deployed BE): the 20 $\kappa$-cells at gauss256 plus their $q^+$/$q^{++}$ (gauss320,
  gauss384) = 60 cells.

### 2.2 2D: rule families (B3) and the low mixed-frequency test set

**Family panel** at $(r,M)\in\{(128,512),(384,1536)\}$ (deployed nominal $M=4r$), LSPG-CN-R $\Delta t_0$. Ladders (rule
names of `qcore.offmesh_rule`; random families repeat the whole ladder for draws $s=0..4$; draw 0 is Hari's default):

| family | ladder | $m^\*$ candidates | draws |
|---|---|---|---|
| Gauss (deterministic) | $\mathcal L_G$ (from $\mathcal F_2$) | gauss64 … gauss256 | 1 |
| Fibonacci lattice, random shift | fib2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393 | fib2584 … fib46368 | 5 (shift seed $s$) |
| Sobol, scrambled | sobol$2^{k}$, $k=11..17$ | $k=11..15$ | 5 (scramble seed $s$) |
| Halton, scrambled (2D) | halton$2^{k}$, $k=11..17$ | $k=11..15$ | 5 |
| Smolyak (Clenshaw–Curtis, boundary nodes removed) | smolyak8 … smolyak14 | smolyak8 … smolyak12 | 1 |
| **stage (ii)** Monte Carlo, uniform | mc$2^{k}$, $k=11..17$ | $k=11..15$ | 5 |

Stage (ii) (MC) runs only after its generator (a new `offmesh_rule` name `mc<m>` with a seed suffix, wrapping Hari's
`uniform_mc`) passes the Codex code audit and the CPU generator tests of §7. Seeds are carried in the rule name
(`fib4181s3`, `sobol8192s2`; no suffix = seed 0 = the deployed realisation), so every table carries its draw identity.

**Target (cross-family reference).** The gauss384 run of the same $(r,M)$ cell (from $\mathcal F_2$). For every family
rung the lattice distance to the target is recorded (the family's quadrature error against an independent family).

**Low mixed-frequency test set (B6-LMF; defer order position 2).** Selection rule (frozen): order all modes
$(k_x,k_y)$, $1\le k_x,k_y\le1023$, by the product $k_xk_y$ ascending, ties by the discrete eigenvalue $\lambda$, then
by $k_x$ (a hyperbolic cross); take the first $M=1536$ ($r=384$). Cells: Gauss ladder and Smolyak ladder, LSPG-CN-R
$\Delta t_0$, all 38 cases, full Q1/Q2. Stability checks (must pass before any accuracy number of the arm is read):
(i) $\mathrm{rank}(A)=r$ at $10^{-13}$ and $\mathrm{cond}(A)\le10\times$ the cond of the default $(384,1536)$ cell;
(ii) every run valid; (iii) Q1 pass at gauss256. A failed stability check makes the LMF result "unstable test space"
(reported, no verdict). Prediction registered (from the plan's C4): Smolyak's $m^\*$ under LMF is smaller than under the
default list; Gauss's is not larger.

### 2.3 3D: front completion and the $m^\*$ ladders (A4-3D, B6-3D, A3-3D, B3-3D)

**Rule ladders (3D)** — Gauss $\mathcal G_3$: gl16, gl20, gl24, gl32, gl40, gl48, gl56, gl64; lattice (CBC rank-1, shift
seed 0) $\mathcal T_3$: lat8192, lat16384, lat32768, lat65536, lat131072, **lat262144** (new; generated once locally by
the vendored, unchanged `cbc_lattice_vector3` + `rank1_lattice3`, committed with sha256 and its $P_2$ merit, exactly as
speed3d's `rules3d/make_rules.py` generated lat65536/lat131072). $m^\*$ candidates: gl16 … gl48; lat8192 … lat65536.

**(3R) Rounds.** Continue qualify's round procedure (its A1.1/A2.1, unchanged predicate and candidate universe: speed3d's
436 eligible configurations with a valid screening panel; dominance rule qualify §6 with speed3d's per-case $\eta$;
Q2-failed or unverified configurations removed after each round). Round 4 = qualify's 16 exposed candidates
(`exposed_untested`). Rounds 5–8: the current open candidates that are front members of the untested-plus-passed set,
ranked by screening cost, ≤ 20 per round. Lattice members whose $q^{++}$ was missing (lat65536 → lat131072 → none) now
use lat262144. Labels as qualify A2.1: always "qualified subset of the tested candidates", never "complete"; open list
reported.

**(3M) $m^\*$ ladders.** Settings $w\in\{$M2_512, W256, W512, W768, W1024$\}$ (deployed $M$ = `complete_M(65,4r)`)
× specs $\sigma\in\{$`fs|CN|0.01|K1` (accurate, Δt₀), `qn|BDF2|0.025|k4` (fast, 2.5Δt₀)$\}$ × both ladders, every rung,
all 64 cases. Plus the three comparators' own Q2 (C_old `M2_512|gl32|fs|BE|0.01|K1`, C_W768 `W768|lat32768|fs|BE|0.01|K1`,
C_W1024 `W1024|lat65536|fs|BE|0.01|K1`: $q^+$ on 64 cases, $q^{++}$ on cases 0–7). Configurations already in speed3d's
grid are rerun (reproduction gate ≤ 1e-10 against the speed3d archive, as qualify §2.4).

**(3B) Test-space and factorial panels (3D).** W512 (W1024 bank, $r=512$), spec `fs|CN|0.01|K1`, lattice ladder
lat16384 … lat262144, all 64 cases. Test lists: $\kappa=1$ **explicit lowest-512 list** (the first 512 entries of the
stable eigenvalue order; this breaks the degenerate shell and is labelled "shell-breaking control"); $\kappa=2,3,4,8$:
`complete_M(65, κ·512)` (shell-completed; $\kappa=4$ is the deployed 2052). Fixed-$(M,m)$ panel: the frozen $65^3$ list of
$r=512$ ($M=2052$, hashed) for $r\in\{256,512,768,1024\}$ on the lattice ladder (W1024 at $M=2052$ is $\kappa\approx2$,
labelled). The fixed-$m$ panel ($M$ nominal, lat65536, $r\in\{256,512,768,1024\}$) is contained in (3M) and costs
nothing extra.

**(3F) Rule families (3D, B3).** W512, $\kappa=4$, `fs|CN|0.01|K1`: Gauss and lattice from (3M); lattice shift draws
$s=1..4$ (lat8192 … lat262144); Sobol (`sobol3`, scrambled, $m=2^{13}..2^{18}$, draws $s=0..4$); Smolyak (`smolyak3`,
levels with $m\le262144$; `smol8` stays the registered must-fail control). Stage (ii): MC (`mc3` from Hari's `dim3.py`,
copied) and 3D Halton (new: `scipy.stats.qmc.Halton(d=3, scramble=True, seed=s)`), $m=2^{13}..2^{18}$, 5 draws, only
after the generator code audit. Target = the gl64 run of W512 (3M).

### 2.4 Cohorts, references, scoring (common brief §2; qualify §2.5)

2D: dev6 and val32 only (training disjointness asserted); 3D: 923801 only. No test64/923901 run, no sealed/OOD access;
`sealed/` and `custody/` never touched. Scoring offline on CPU through `refstore` (E0 store version pinned):
$e=\max_{k=1..5}\lVert u(t_k)-R(t_k)\rVert/\lVert R(t_0)\rVert$ on the $257^2$ / $63^3$ lattice; eligibility per case-kind
read from the store at scoring time; S only for LSPG-BE at $\Delta t_0$ in 2D (labelled), never in 3D; ties
$\lvert\Delta e\rvert\le2\varepsilon_{\rm ref}$; every scored record asserts its identity (cohort, case, time tag,
reference kind, reference sha256, store version, metric id, rule name including draw, $M$ and mode-list hash).

## 3. Qualification (inherited contract; nothing weakened)

**Q1 (2D)** — qualify §4.1 + A1.3 + A2.5 exactly: valid runs; in-job per-step certificate with a fresh autodiff
Jacobian (LSPG: $\gamma_n\le10^{-3}$; GAL: $\eta_n\le10^{-8}$, the registered Galerkin residual predicate);
non-certified steps ≤ 1 % pooled over all steps of all cases; lattice distance to the adaptive **tight** solve
≤ 2.5e-4 on **every** case; tight/tighter ≤ 2.5e-5 on dev6; G-cert and offline recomputation as qualify.
**Q1 (3D)** — speed3d eligibility (E-a valid, E-b certified non-stationary ≤ 1 % pooled, E-c Gram-lattice distance to
ADT ≤ 2.5e-4 on every case) plus ADT tight/tighter ≤ 2.5e-5 on cases 0–7 (the brief's anchor item, which speed3d
covered by its ADT gate); configurations certified in this lane's jobs.

**Q2 (both)** — AM-2 with the ladder: for rung $q_k$, $q^+=q_{k+1}$ and $q^{++}=q_{k+2}$ of the **same family and the
same draw**. Rollout bar: max over the five evolved outputs of the lattice relative distance between the $q_k$ and
$q_{k+1}$ runs ≤ 2.5e-4 on **every** case of the population (2D: all 38 — stricter than qualify's 16; 3D: all 64);
ρ bar: $\rho_{\max}\le0.116$ over the six own output states of every case, with $N$ the tested advection of the cell
(length $M$); finer-rule convergence: $q_{k+1}$ vs $q_{k+2}$ meets the rollout bar on dev6 (2D) / cases 0–7 (3D), else
"unverified (finer rule not converged)". Because the ladder neighbours are full production runs of the same cell, the
$q^+$ "replay" is the next rung's production run (same configuration in everything but the rule; asserted by manifest).
Labels pass / fail / unverified; the top two rungs of a ladder have no Q2 label ("ladder top").
**Numerically qualified** = Q1 pass and Q2 pass.

**Q3** — $\mathrm{cond}(A)$ and rank per $(r,M)$ cell (and per LMF/shell-breaking list); step-Jacobian singular values
at output indices 1, 3, 5 of dev6 for the $\kappa$-cells at gauss256 (descriptive; qualify §4.3).

## 4. Cost (timing)

**2D (job `t2`, H200, one job).** qualify's `qtime` protocol unchanged except the subject key adds $M$: reference A =
P1/qualify's (r = 384, $1024^2$, LSPG-CN-R, $\Delta t_0$, $F_{1536}$, gauss128); split and fused triplets, burn,
synchronisation, drift band $[1/1.1,1.1]$ (≤ 2 retries), dev6 × 2 repetitions, randomised order, doubled-work control,
injected identity fault. Cost $c$ = median over valid fused triplets of $t_{\rm fused}/\bar t_A$ (online cost,
full-field contract); $c_{\rm ev}$ = evolution only. Subjects: all 208 LSPG-CN-R cells, and for GAL-CN, LSPG-CN and
LSPG-BE the $\kappa$-cells at gauss256 and at their cell's $m^\*$ rung. **Timing-job identity** (replaces qualify's
in-job re-certification, because accuracy and timing run on the same GPU type): each subject's untimed scoring run must
equal the accuracy job's archived production run of the same cell to ≤ 1e-10 relative ($\max_k\lVert\Delta W_k\rVert/
\lVert W_0\rVert$) on every dev6 case, else that subject's timing is "unusable (identity)"; timed executions must
reproduce the scoring run's identity hash (qualify A4.1).

**3D (jobs `t3s`, `t3c`, H200).** Screening `t3s`: every numerically qualified 3D configuration of this lane and
qualify, plus C_old, C_W768, C_W1024, in speed3d's screening panel (A = C_old, cases 0–3 × 2 repetitions, A–B–A,
hashes, drift; `Ctx.time_panel` unchanged). Confirmation `t3c` (after the CPU selection of §6.5): speed3d's confirmation
panel for each (comparator C, winner) pair — A = C, B = winner, cases 0–7 × 3 repetitions (24 invocations), the full
mesh-path query (projection, step loop, decode), H200.

## 5. Cost model statements (descriptive, A3)

Per-iteration leading-order costs (plan §1): $C_N=O(mr+mM)$, $C_{J_N}=O(mMr)$, $C_{J^{\mathsf T}J}=O(Mr^2)$,
$C_{\rm chol}=O(r^3)$. For the 2D factorial the elasticities of $c$ are fitted as
$\log c=\beta_0+\beta_r\log r+\beta_M\log M+\beta_m\log m$ (least squares over the qualified LSPG-CN-R cells, case
bootstrap of the per-cell medians, 10,000 resamples, 95 % percentile intervals), reported beside the operation-model
values (1, 1, 1 if $J_N$ formation dominates). Iteration counts per cell are reported (cost can move through them).
No verdict: P1 already found kernel slopes below the operation model (its 7/8 fits failed).

## 6. Decision rules (frozen before data; three-way outcomes, precedence, actual eligible populations)

Throughout, a **resolved** case of a paired comparison is an eligible case (E0 eligibility of its case-kind, both
configurations numerically qualified, both runs valid) with $\lvert e_X-e_Y\rvert>2\varepsilon_{\rm ref}$; the others
are ties. Counts improve / worsen / tie / unqualified / ineligible are always reported, with the median paired change and
the worst change.

### 6.1 A3 (2D and 3D)

**(a) Transition rule (AM-7, verbatim style).** For a pair of cells $X\to Y$ differing in one axis: if either cell is
not numerically qualified → "inconclusive (unqualified cell)". Else if resolved cases < 50 % of the eligible cases →
"inconclusive (resolution)". Else **"improves"** iff > 50 % of resolved cases improve **and** the median paired change
$e_Y-e_X$ over eligible cases is < 0; **"does not improve"** otherwise. Precedence: unqualified, then resolution, then
the improvement test.

**(b) Panels and their transitions.**
- *Representation size* at fixed $(M,m)$: $M=2048$, rung gauss256: $r$ 128→256→384→512 (2D); $M=2052$ frozen, lat65536:
  $r$ 256→512→768→1024 (3D).
- *Representation size* at fixed $m$, $M=4r$: gauss256 (2D); lat65536 (3D) — reported, same rule.
- *Test space* at fixed $(r,m)$: §6.4.
- *Quadrature* at fixed $(r,M)$: §6.2.
- A **range** claim ("monotone over the panel") requires every transition of the panel to "improve"; otherwise the
  claim is limited to the named transitions; any inconclusive transition makes the range "inconclusive" unless another
  transition already "does not improve" (then "not monotone"). Every transition is reported, never only the passing ones.

**(c) Deployed-path decomposition (2D, descriptive).** Path points $P(r)=(r,4r,m^\*_G(r,4r))$ with $m^\*_G$ the
Gauss $m^\*$ of §6.2 (LSPG-CN-R). For consecutive $r_a\to r_b$ the change of median error $\Delta=f(P(r_b))-f(P(r_a))$
and of cost is split into the three one-axis steps by the Shapley average over the six axis orderings (every corner
cell exists in $\mathcal F_2$). A share is computed only if all eight corner cells are numerically qualified; otherwise
"decomposition not reached (unqualified corner)". Case-bootstrap 95 % intervals (cases resampled jointly across cells);
a component is labelled "reduces error" / "increases error" if its interval excludes 0, else "indistinguishable from
zero". Descriptive: no pass/fail.

### 6.2 A4 — $m^\*(r,M)$ per rule family

For a ladder (family, draw, cell, scheme) with candidates $q_1<\dots<q_J$: $m^\*$ = the point count of the **first**
candidate that is numerically qualified. Outcomes: **"$m^\*=m(q_j)$, stable"** if every larger candidate with a Q2
label is also qualified; **"$m^\*=m(q_j)$, non-monotone"** otherwise (the failing larger candidates listed); **"none
(within the candidate list)"** if no candidate qualifies; "not reached (budget / memory)" for an unrun ladder. A
candidate whose Q2 is "unverified" is not qualified. Random families: per-draw $m^\*_s$; the **family $m^\*$** is the
smallest candidate at which **all five draws** are numerically qualified ("none" if there is none); the median-draw
$m^\*$ and the range over draws are reported beside it. The 3D accurate and fast specs give separate $m^\*$ values; the
3D "width $m^\*$" is the larger of the two (reported with both).

### 6.3 B3 — rule families

At each family-panel cell and for the 2D/3D scheme of §2.2/§2.3:
- **Point efficiency** for families $X,Y$ with family $m^\*$ values: "**X needs fewer points**" iff
  $m^\*_X\le0.8\,m^\*_Y$; "**Y needs fewer points**" symmetric; "**comparable**" otherwise; "**X qualifies, Y does not
  (within its list)**" when exactly one is "none"; "neither qualifies" when both are.
- **Cross-family limit agreement (gate, per family):** the finest rung of each family (each draw) must be within the
  rollout bar (2.5e-4, every case) of the target (gauss384 in 2D; gl64 W512 in 3D). A family that fails is
  **"flagged: limit disagrees with an independent family"**, and its $m^\*$ is reported but not used in a point-efficiency
  verdict. (This is the plan's E1 requirement that a bad generating vector be caught by an independent family; the
  must-fail control is §7.)
- **Error against $m$** (descriptive): distance to the target against $m$ per family, with the min–max range over
  draws; E0 median and worst error against $m$. Intervals across draws are ranges over the 5 draws (labelled; with 5
  draws a percentile interval is not claimed).
- Prediction (registered, from `07-literature-check.md` C4; reported as confirmed / not confirmed, not as a verdict):
  Gauss geometric; lattice and Sobol algebraic; Smolyak fails on the mixed-frequency sine tests; MC slowest.

### 6.4 B6 — test space

Per $r$ and scheme, at **fixed rung gauss256** (primary; the same $m$ for every $\kappa$, so only the test space moves)
and at each cell's own $m^\*$ (secondary):
- Transitions $\kappa$: 1→2, 2→3, 3→4, 4→8 by the §6.1a rule; range verdict over 2–4 (primary, the plan's ratios) and
  over 1–8.
- **Must-degrade control (M = r):** for LSPG-CN-R at every $r$, the transition 1→4 must "improve" (equivalently $M=r$
  is worse than $M=4r$) by the §6.1a rule; if it does not at some $r$, the control "did not fire" there and every B6
  verdict at that $r$ is blocked ("test-space metric not shown to be sensitive"). If the $M=r$ cell is not numerically
  qualified (e.g. invalid runs), the control has fired by failure (reported as such).
- **GAL vs LSPG** at each $(r,\kappa)$, same rule: GAL-CN against LSPG-CN (same LMM; primary) and against LSPG-CN-R
  (the deployed LSPG scheme; secondary). Paired per case: "**GAL more accurate**" iff > 50 % of resolved cases favour GAL
  and the median paired change $e_{\rm GAL}-e_{\rm LSPG}<0$; "**LSPG more accurate**" symmetric; "**neither**" otherwise;
  "inconclusive" under the §6.1a precedence. At $\kappa=1$ the two forms solve the same square system when $A$ is
  invertible; their agreement there is a positive gate (≤ 1e-8 lattice distance on every case where both are valid).
- **B6-LMF:** Smolyak and Gauss $m^\*$ under the LMF list against the default list at $(384,1536)$, by the §6.3
  point-efficiency rule; E0 accuracy of LMF vs default at gauss256 by the §6.4 GAL-vs-LSPG rule with "LMF" and
  "default" in place of the forms.
- **3D:** the same transitions at W512 on the lattice ladder rung lat65536 (primary) and at $m^\*$; the $\kappa=1$
  shell-breaking list is labelled; the must-degrade control applies to 1→4.

### 6.5 A4-3D front and qualified re-timing (H2q)

- **Per-width coverage:** width $w$ is "covered" iff at least one numerically qualified configuration with that width
  exists among all tested (qualify's 4 passes, rounds 4–8, ladders); else "not covered", with the closest failing
  configurations listed.
- **Qualified 3D front:** qualify §6 dominance on the qualified configurations, cost = this lane's `t3s` screening
  ratio × $t_A$ (one job, so all costs share one panel), error median/worst over 64 cases. Labelled "qualified subset of
  the tested candidates" (never complete); open candidates listed.
- **H2q.** For each $C\in\{$C_old, C_W768, C_W1024$\}$: the winner $W_C$ = the cheapest (by `t3s` cost; ties within 3 %
  to the lower median then worst error, speed3d's `pick`) numerically qualified configuration whose median **and** worst
  E0 errors tie-or-beat $C$'s (speed3d §5 tie rule: interval overlap with endpoint propagation). $S_C$ = 1/(median
  confirmation ratio $t_W/\bar t_C$) from `t3c`. **Pass** iff $S_C\ge1/0.6$ and the confirmation panel is valid;
  **fail** iff $S_C<1/0.6$; "**no qualified tie-or-beat configuration**" if no winner exists; "inconclusive (panel
  invalid)". Each verdict carries the comparator's own Q2 label (§2.3); if $C$ itself is not quadrature-qualified the
  speed-up is labelled "qualified configuration vs the deployed (unqualified) comparator". Secondary (descriptive): the
  same against $C^q_w$ = the BE FS-K1 $\Delta t_0$ configuration of width $w$ at its smallest qualified rule
  (if it exists), so a qualified-vs-qualified ratio is available.
- Scope label on every 3D speed: "full-field $65^3$ mesh-path query, H200; not a comparison with the best FOM (K2
  already fired)". Speed3d's 2.01×/5.37×/17.2× are not re-labelled here; H2q is a new, separate quantity.

## 7. Controls (fixture, predicate, what it blocks)

Every control is shown to behave as registered in the real-size smokes (synthetic ones in CPU tests) before any verdict
is read; a misbehaving control blocks its claims until fixed and re-audited.

| control | kind | fixture and predicate | blocks |
|---|---|---|---|
| G-modes | positive | first 2048 entries of $F_{4096}$ hash-equal to P1's `list_hash_2048` | everything 2D |
| reproduction (2D, $M=4r$) | positive | the lane driver with explicit $M=4r$ at R384 gauss128 and R128 fib4181, LSPG-CN-R and LSPG-BE $\Delta t_0$, dev6: coefficients equal qualify's `acc1024`/P1 archives ≤ 1e-10 (H200 vs A100 archive: the bar is ≤ 1e-10, not bitwise) | all 2D cells |
| reproduction (3D, $M$ override) | positive | `Setting` with explicit $M$ = `complete_M(65,4r)` reproduces speed3d's archived coefficients of `W512|lat16384|fs|CN|0.01|K1` and `M2_512|gl24|fs|BDF2|0.0125|K1` ≤ 1e-10 on cases 0–3 | all 3D cells |
| under-resolved rule (2D) | negative | P1 A1.1 fixture (R384 $M=1536$ gauss32 → gauss64, LSPG-BE $\Delta t_0$, dev6; independent evidence `p1-scaling/checks/ctrl-fixture-under-resolved.json`): lane Q2 must return fail | Q2, A4, B3 |
| under-resolved rule (3D) | negative | qualify's fixture W512 lat8192 → lat16384 FS-K1 BE 0.01 (independent evidence: combined `SELECTION-3D-POSTHOC.json`); lane Q2 must return fail | Q2-3D, 3D front |
| Smolyak must-fail (3D) | negative | `smol8` at W512 κ=4: Q2 fail or Q1 fail (registered in quadrature-burgers3d as a must-fail rule) | B3-3D |
| bad generating vector (2D, 3D) | negative | rank-1 lattice with $z=(1,1)$ (2D, $n$ = fib ladder sizes) / $z=(1,1,1)$ (3D, lattice sizes), shift seed 0: the cross-family gate (§6.3) must flag it on ≥ 1 case, whatever its own Q2 says | B3 point-efficiency verdicts |
| draw identity | positive / negative | same seed twice → identical tables (sha256); seeds 0 and 1 → different tables; seed 0 equals the deployed rule's sha256 (fib4181, lat-family) | B3 draws |
| M = r must-degrade | negative (scientific control) | §6.4 | B6 at that $r$ |
| GAL = LSPG at κ = 1 | positive | §6.4 | B6 GAL vs LSPG |
| Q1 negatives | negative | qualify's LSPG and GAL budget-0 fixtures (independent NumPy evidence first), zero-update (motion > 100 × 2.5e-4) | Q1 |
| wrong-mesh / wrong-list table | negative | a table built for $M=1536$ fed to an $M=1024$ setting must raise; a $256^2$ table to $1024^2$ must raise | all |
| case / time / reference / rule-draw swap | negative | a swapped record (case, time tag, reference kind, Δt, rule draw suffix, $M$) must raise in scoring | all accuracy |
| timing | positive / negative | doubled work B/A ∈ [1.8, 2.2] with identical iteration counts; injected identity fault invalidates every triplet of that subject; speed3d T1 (C_old twice → ratio ∈ [1.8, 2.2]) in `t3s` | A3 cost, H2q |
| decision-rule synthetics | negative / positive | CPU fixtures through the actual functions: transition rule (all-improve → improves; 40 % resolved → inconclusive; unqualified → inconclusive), $m^\*$ (stable, non-monotone, none, all-draw), point efficiency (0.8 boundary), Shapley (additive synthetic → exact shares), GAL-vs-LSPG (symmetric cases) | all verdicts |
| generator tests (stage ii) | positive | CPU: MC/Halton3 points in $(0,1)^d$, weights sum 1, seeds distinct; integration of $\prod_i\sin(\pi x_i)$ and a smooth Gaussian bump: MC error within 4 standard errors over 32 seeds; Halton error < MC error at $m=2^{16}$ | B3 stage (ii) |
| memory pilot | positive | the largest 2D cell (R512, $M=4096$, gauss384 + its $q$ tables) and the largest 3D table (W1024 gl64; W1024 lat262144) built in the smoke; peak ≤ 120 GB (decimal), else the cell is "not reached (memory)" | the cells concerned |

## 8. Jobs, resources, defer order

| job | GPU | content |
|---|---|---|
| `smk2` | h200 | real-size smoke at $1024^2$: G-modes, 2D reproduction gates, a dev2 slice of every scheme on 4 cells (incl. R512 $M=4096$ gauss384 memory pilot and an $M=r$ cell), every 2D control, one family ladder per random family (2 draws, 2 rungs), the LMF stability checks, cost probes for the certificate |
| `smk3` | h200 | 3D smoke: $M$-override reproduction, the 3D under-resolved, smol8 and bad-vector controls, one ladder (W512 lattice, 2 rungs, cases 0–3), memory pilot W1024 gl64 and lat262144, cost probes |
| `f2a`, `f2b` | h200 | $\mathcal F_2$ LSPG-CN-R (`f2a`: R128, R256, R384; `f2b`: R512) |
| `f2k` | h200 | $\kappa$-cells: GAL-CN, LSPG-CN, LSPG-BE (+ LMF arm if not deferred) |
| `b2f` | h200 | 2D families stage (i) |
| `q3r` | h200 | 3D rounds 4–8 (one job per round, ≤ 20 members) |
| `m3a`, `m3b` | h200 | 3D ladders (`m3a`: M2_512, W256, W512; `m3b`: W768, W1024) + comparator Q2 |
| `k3` | h200 | 3D test-space and fixed-$(M,m)$ panels |
| `b3f` | h200 | 3D families stage (i) |
| `b2m`, `b3m` | h200 | stage (ii) MC / Halton3 (after the generator audit) |
| `t2` | h200 | 2D timing panel |
| `t3s`, `t3c` | h200 | 3D screening, then confirmation |

**Budget.** Hard cap 55 GPU-h including contingency; qualify's `ledger.py` (cap, namespace, job prefix `jd_` edited):
a job is submitted only if spent + Σ time limits of queued/running lane jobs + this job's time limit + the projected time
limits of the not-yet-submitted **never-dropped** jobs ≤ 55, the lane has < 3 jobs queued/running and the account < 8.
Time limit = 1.5 × a smoke-derived projection (`resources.py`; estimates, not bounds). A-priori estimate (re-priced by
the certificate): smokes 2; `f2a`+`f2b` 3; `f2k` 3; `b2f` 2.5; `q3r` 1; `m3a`+`m3b` 5; `k3` 1.5; `b3f` 2; `b2m`+`b3m` 2;
`t2` 2; `t3s`+`t3c` 1.5 ≈ 26 GPU-h.

**Defer order** (brief; the first listed is dropped first): (1) B3 stage (ii) (`b2m`, `b3m`); (2) the low
mixed-frequency test set; (3) the A3 3D fixed-$m$ panel (here contained in `m3`; only its separate reporting would be
dropped); (4) B6 $M/r=8$ (2D: the κ=8 cells of every scheme; 3D: κ=8). **Never dropped:** A4 2D (the Gauss ladders of
$\mathcal F_2$ and the B3 family $m^\*$ at the two cells, stage (i)), A4 3D front completion (`q3r`, `m3a`, `m3b`) and the
qualified 3D re-timing (`t3s`, `t3c`), B6 $M/r\in\{2,4\}$ and $M=r$ (2D and 3D). Deferred items are reported "deferred
(budget)", never silently omitted. If a never-dropped job cannot be reserved after every permitted deferral, the outcome
is "core incomplete (budget)" with the shortfall reported.

```mermaid
flowchart LR
  D[DESIGN + Codex audit]:::doc --> C[code + Codex code audit]:::doc --> S2[smk2]:::job & S3[smk3]:::job
  S2 --> F2[f2a f2b f2k b2f]:::job --> T2[t2 timing 2D]:::job
  S3 --> Q3[q3r rounds]:::job & M3[m3a m3b ladders]:::job & K3[k3 B6-3D]:::job & B3F[b3f families]:::job
  Q3 & M3 --> SEL[score, qualify, select winners - CPU]:::doc --> T3[t3s then t3c]:::job
  GA[generator audit]:::doc --> ST2[b2m b3m stage ii]:::job
  F2 & T2 & T3 & K3 & B3F & ST2 --> R[analysis, results audit, make_report]:::doc
  classDef doc fill:#e5e7eb,stroke:#374151;
  classDef job fill:#dbeafe,stroke:#1d4ed8;
```

## 9. Analysis and deliverables

`dscore.py` (CPU): E0 scoring with identity assertions (incl. $M$, mode-list hash, rule draw), offline recomputation of
every Q1/Q2 quantity (independent NumPy decode as qualify `npcheck`), labels. `dstats.py`: transition rule, $m^\*$,
point efficiency, cross-family gate, Shapley decomposition, GAL vs LSPG, elasticity fit (all with the CPU fixtures of §7).
`analyze.py` → `runs/EXPORT.json` (every label, verdict, $m^\*$, front, key number; a `contract` field on every cost);
`make_report.py` → `report.md` + PNG plots: error and cost heat maps over $(r,M)$ per rung; error vs $m$ per family with
draw ranges; $m^\*$ table; $\kappa$ curves per $r$ and form; Shapley bars; 3D per-width $m^\*$ and the qualified front;
H2q panel. `runs/GPU-LEDGER.json`. No hand-typed number; Codex audits at every gated step.

## 10. Code (copy, then edit in the lane; `vendor/` untouched)

Copied from qualify (`exp/2026-10-18-jcp-qualify` @ `4399491eb`, sha256 in `PROVENANCE.json`): `p1core.py`,
`p1common.py` (edited: explicit $M$ and $M_{\max}=4096$, mode-list policies `frozen`/`lmf`; recorded as an edit with
the new sha256), `qcert.py`, `qacc.py` → `dacc.py` (ladder- and $M$-aware driver), `qtime.py` → `dtime.py`,
`qscore.py`, `qstats.py`, `npcheck.py`, `ledger.py`, `resources.py`, `cluster/{stage,submit,collect}`; from speed3d via
qualify: `c3common`, `s3common`, `s3ctx`, `fs3`, `fs3p`, `qn3`, `t2core`, `timing`, `replay`, `score3`, rule files,
`q3run.py` → `d3run.py` (ladder, $M$ override, ADT certification, screening and confirmation panels), `front3d.py`.
New: `rules3d/make_rules_dims.py` (lat262144, 3D Sobol/MC/Halton draws, bad-vector fixtures), the 2D rule-name parser
(draw suffix, `mc`, bad-vector `badlat<n>`), `dscore.py`, `dstats.py`, `analyze.py`, `make_report.py`, `make_configs.py`,
`test_dims.py` (CPU controls).

## 11. Integrity rules carried

Never quote a pre-September Burgers speed-up; never mention the heat POD-128 comparison; retractions stated plainly.
Every GPU number from the cluster (H200 throughout); local GB10 only for CPU analysis (`JAX_PLATFORMS=cpu`) and at most
one `jaxrun`. Codex audits read-only, no large copies (≤ 2 GB scratch under `/tmp/jcp-dims-audit-scratch/`, deleted
after each round), `df -h /home` ≥ 60 GB before each audit. Waits on PIDs with an overall deadline. Big arrays stay on the
cluster until pulled into `runs/` (local disk ~130 GB free; large `.npz` listed by sha256 in `LARGE_FILES.sha256`).
