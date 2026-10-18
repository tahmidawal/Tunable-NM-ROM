# jcp-p1-scaling — does the reduced model's cost and accuracy depend on the bank width $r$ and not on the mesh $N$? (pre-registration)

**Status:** pre-registration, 2026-10-17, written before any code or data of this lane. Amendments are appended at the
end, dated; nothing above them is rewritten. Every accuracy number this lane produces is scored against the E0
second-order reference $R_{ST}$, whose uncertainty $\varepsilon_{\rm ref}$ is a **conditional Richardson estimate** (not a
rigorous bound; E0 A10 cross-check rule amended post data); combined A10, A11 and the retracted A15 labels carry forward.

Governing documents (later supersedes earlier): `reports/2026-10-17-full-experiment-plan/PLAN-v2.md`,
`PLAN-v3-amendments.md`, `PLAN-v4-amendments.md`; the coordinator's restatement `campaign/P1-RESTATEMENT.md` and its
amendments `campaign/P1-RESTATEMENT-AMENDMENTS.md` (AM-1 … AM-8, Codex round 4: READY); the common brief
`campaign/briefs/COMMON-BRIEF.md`; the lane rules `reports/2026-10-08-jcp-lane-agent-rules.md`. This file implements
them and freezes the items the four Codex restatement audits delegated to the lane ("safe lane work", §12).

Lane: worktree `worktrees/2026-10-17-jcp-p1-scaling`, branch `exp/2026-10-17-jcp-p1-scaling` (fork `21c175a1b`,
setup `8bd0d20b6`), lane dir `experiments/jcp-p1-scaling` (here), cluster namespace
`/cluster/tufts/paralab/tawal01/jcp-p1-scaling/`, concurrent-job cap 4 (campaign cap 8), GPU-hour cap 20 including
contingency.

## 0. Question and hypotheses

Does the cost and accuracy of the 2D off-mesh coordinate-bank Burgers ROM depend on the bank width $r=R'$ and **not**
on the mesh size $N$ (points per axis)? "k" is read as $r$ (campaign default; the head size $K$ is priority-3 C5).

| hypothesis | claim tested | evidence | verdict outcomes |
|---|---|---|---|
| **H-cost-N** | at fixed configuration the reduced solve (evolution) time does not grow with $N$ | P1c, 2D, $256^2/1024^2/4096^2$ | pass / fail / inconclusive / "verdict not reached" |
| **H-cost-r** | per-component kernel cost follows the leading-order operation model in $r$ | P1a (2D), P1b (3D), fixed-$(M,m)$ panel | per component: pass / fail (op-model or hardware) / inconclusive |
| **H-acc-N** | at fixed configuration the error against E0 $R_{ST}$ does not depend on $N$ | P1d | pass / fail / inconclusive |
| **H-acc-r** | error decreases with $r$ (per transition; majority of resolved cases) | P1d | per transition: improves / does not improve / inconclusive; range: monotone / not monotone / inconclusive |

Prior evidence against unconditional versions (recorded, not tested here as such): combined H1, 2D $R'=384\to512$
worsened on 22 of 38 cases; 3D worst errors 4–7 %.

## 1. The reduced model (unchanged)

Bank coefficients $c\in\mathbb R^{r}$, $u=\hat G_{r}(x)\,c$ (frozen checkpoint `dn256b`, $R=512$, $K=16$, rotation
`rotation_R512`, ordered nested prefixes). Test space: $M=4r$ discrete sine modes $\Phi$ (mesh-orthonormal), $A=\Phi^{\mathsf T}\hat G_r$ (exact discrete projection), $\Lambda$ the discrete eigenvalues. Tested advection by an off-mesh
point rule $(x_q,w_q)_{q=1}^m$ (Hari's point form, exact bank gradient):
$$N(c)=\Psi^{\mathsf T}\big[(G_qc)\odot(G_sc)\big],\qquad \Psi=L\,\mathrm{diag}(w)\,\psi(x_q),\quad G_q=\hat G(x_q),\ G_s=(\partial_x+\partial_y)\hat G(x_q).$$
Two-step LMM residual (combined `t2core.py`):
$\mathcal R_n(c)=A(\alpha_0c+\alpha_1c_n+\alpha_2c_{n-1})+\Delta t\,(\beta_0F(c)+\beta_1F(c_n))$, $F(c)=N(c)+\nu\Lambda Ac$;
LSPG minimises $\lVert D\mathcal R_n\rVert$ with $D=(\alpha_0+\beta_0\Delta t\,\nu\Lambda)^{-1}$; adaptive Levenberg–Marquardt
(fused LM, Cholesky, clipped step, damping carried, quadratic predictor), budget 600. Schemes in P1: **LSPG-BE** and
**LSPG-CN-R** (CN with two BE start-up steps) at $\Delta t_0=0.005$, horizon 0.25 (50 steps), outputs at
$t=0,0.05,\dots,0.25$.

**Initialisation contract (V4-4, kept):** the full-field initial condition is sampled at the cold quadrature points and the
reduced projection is solved (`t2core.make_query.initialize`); output fields are decoded on the full mesh. Label:
"full-field input sampled at quadrature points; full-field decode".

Tolerances (`c2common.tol_of`): LSPG prod $(g_{\rm tol},\ {\rm tol}_f)=(10^{-3},10^{-9})$, tight $(10^{-7},0)$, tighter
$(10^{-8},0)$; GAL prod $(0,10^{-10})$, tight $(0,10^{-13})$, tighter $(0,10^{-15})$.

## 2. Frozen configurations

### 2.1 Configuration identity and the development manifest (AM-1, "item 0")

A **configuration** is the tuple (bank checkpoint sha256, rotation sha256, $r$, ordered mode identities $(k_x,k_y)_{1..M}$
with their hash, rule name and $m$ and the sha256 of its points and weights, mesh $L$, scheme, $\Delta t$, form, tolerance
level, budget, initialisation contract, test weighting $D$, output times, scoring metric). `manifest.py` writes
`runs/MANIFEST-dev.json` (hashed) **before any job**: bank and rotation hashes; for every (r, mesh, mode-policy) the
mode identities, order and hash (eigenvalues and sine tables recomputed per mesh from the frozen identities); every
rule's name, $m$ and sha256; scheme, $\Delta t$, solver and tolerances (prod/tight/tighter, LSPG and GAL); initialisation
contract; weighting; output times; metric id; E0 store version and per-case-kind eligibility for dev6 and val32; cohort
case identities (physical-descriptor sha256); code commit. Reuse or reproduction is accepted only on equality of the
**numerical** configuration; provenance fields (commit, scoring version) are recorded old vs new, not required equal (AM-7).

### 2.2 Bank and rules

- Bank: 2D R512 checkpoint + rotation (`T2.CKPT`, `qcore.build_rotated_bank`), nested column blocks with edges
  $[0,128,256,384,512]$ at $256^2$ and $1024^2$ (the combined construction, so the $1024^2$ reproduction gate compares the
  same arithmetic). At $4096^2$ the full-bank construction is used unless the memory pilot (§9.2) exceeds the trigger; then
  the **prefix-retaining** construction (edges $[0,128,256,384]$, only $T_{:,1:384}$ applied; ≈48 GiB) is used and every
  $4096^2$ number carries the label "prefix-retaining bank".
- Frozen rule per rank (V4-3): R128 `fib4181` (seeded random shift, seed 0, one frozen realisation, AM-7), R256
  `gauss128`, R384 `gauss128`, R512 `gauss192`. Each is re-certified (Q1+Q2) at every mesh it runs on.
- Historical rule (P1g only): `acc` = R384 `gauss96` (the 2026-10-01 quadrature study's post-hoc accurate rule and the
  jcp-mechanism comparison rule).
- Rule ladders (successor $q^+$ and check $q^{++}$), frozen: Gauss orders $\{32,64,96,128,160,192,256,320\}$ (successor
  = next entry); Fibonacci $\{4181,6765,10946\}$. Hence R128 `fib4181`→`fib6765`→`fib10946`; R256/R384
  `gauss128`→`gauss160`→`gauss192`; R512 `gauss192`→`gauss256`→`gauss320`; acc `gauss96`→`gauss128`→`gauss160`;
  under-resolved control `gauss32`→`gauss64`; P1e failing `gauss64`→`gauss96`.

### 2.3 Mode lists (V4-1), with the finding recomputed before data

Frozen 2D lists $F_M=$ `modes_lean(1024, M)` for $M\in\{128,256,512,1024,1536,2048\}$; they are nested prefixes of
$F_{2048}$ (checked). At every mesh $L$ the operators are built for the ordered list $F_{2048}$ (eigenvalues
$\lambda=4L^2(\sin^2\frac{\pi k_x}{2L}+\sin^2\frac{\pi k_y}{2L})$ and sine tables recomputed at $L$) and a configuration
with $r$ uses the prefix $M=4r$ — at $1024^2$ this is exactly combined's construction ($M_{\max}=2048$).

Recomputed on CPU before writing this design (the coordinator's "differs at 4096²" is corrected here; the lane's
historical-claims search found the same):

| $M$ | $256^2$ vs $F_M$: set difference (out / in) | $4096^2$ vs $F_M$: set difference | order identical? |
|---|---|---|---|
| ≤ 1024 | 0 / 0 | 0 / 0 | sets equal; order differs at $256^2$ from $M=512$ |
| 1536 | 2 / 2 (out (28,35),(35,28); in (1,45),(44,9)) | 0 / 0 | no (both meshes) |
| 2048 | 4 / 4 | 0 / 0 | no |

So the **test space differs only at $256^2$** (2+2 modes at $M=1536$, 4+4 at $M=2048$); at $4096^2$ the nominal list is a
**permutation** of the frozen one. The LSPG solve depends on the set (row permutations change only round-off), so the
$4096^2$ nominal arm is a permutation check whose expected effect is round-off-level; it is still run and reported.
Primary P1c/P1d arms use the frozen lists. **Nominal-policy** arms (builder list per mesh), labelled "deployment policy
(mode set varies with mesh)", run for R384 at $256^2$ and $4096^2$ (both rules R384-`gauss128` and acc-`gauss96`, §2.6).
The symmetric difference against the frozen list is reported for every (M, mesh) (generated table).

### 2.4 Kernel fixtures (AM-3)

- **P1a:** `jcp-combined/runs/c2grid/archive/output/W_R512.npz`, run index 15 = `LSPG|CNR|0.005|prod`, cases
  `dev6|0`–`dev6|5`, output indices 1 and 5 → 12 states $c\in\mathbb R^{512}$, truncated to the first $r$ entries.
  The archived configuration (mesh 1024, R512, M 2048, `gauss192`) is verified from `c2grid.json` and the archive's
  `result.json` setup block before use. Label: "kernel fixtures, not reached states of the smaller-r solver".
- **P1b:** `jcp-combined/runs/c3grid/archive/output/W_grid.npz`, key `W1024|lat65536|fs|BE|0.01|K1` (the deployed
  fixed-sweep BE on W1024), cases 0–5 of 923801, output indices 1 and 5, truncated to each $r$. Same label.

### 2.5 Panels

- **P1a (2D kernels, $1024^2$)**, $r\in\{32,64,128,256,384,512\}$: (i) **fixed $(M,m)$** $=(F_{1536},$ `gauss96` $=9216)$;
  (ii) fixed $m=9216$ (`gauss96`) with $M=4r$ ($F_{4r}$); (iii) nominal $M=4r$ with the frozen rule per $r$ ($r\ge128$).
- **P1b (3D kernels, $65^3$)**, W1024 prefixes $r\in\{64,128,256,512,768,1024\}$: (i) **fixed $(M,m)$** = (frozen
  $65^3$ shell list of $r=512$, i.e. `C.mode_order(65)` prefix of length `complete_M(65, 2048)`, Gauss $24^3=13{,}824$);
  (ii) nominal $M=$ `complete_M(65, 4r)` (actual $M$ recorded) with the W1024-prefix rule per width: $r=256$ `gl24`,
  512 `lat32768`, 768 `lat32768`, 1024 `lat65536` (speed3d `DESIGN.md:50–54`, `configs/g2.json:15–23`); $r\in\{64,128\}$
  `gl24` (the $r=256$ rule).
- **P1c (2D timing)**: subjects $r\in\{128,256,384\}$ × {BE, CN-R} × $\{256^2,1024^2,4096^2\}$ (frozen lists) = 18; R512
  diagnostic at $1024^2$ (2); nominal-policy R384 `gauss128` at $256^2$, $4096^2$ (4). Cases dev6 (6), 3 repeats.
- **P1d (2D accuracy)**: R128, R256, R384 × {BE, CN-R}: $256^2$ on 38 cases (dev6 + val32), $1024^2$ on 38, $4096^2$ on
  the 16-case subset dev6 + val32[0:10] (labelled subset); R512 diagnostic × {BE, CN-R} at $1024^2$ on 38.
- **P1e (quadrature slice)**: 2D $r=256$, $1024^2$, BE, dev6: failing `gauss64`, deployed `gauss128`, finer `gauss160`,
  check `gauss192`. 3D $r=512$ (W1024 prefix), $65^3$, BE (deployed fixed-sweep FS-K1, $\Delta t=0.01$, $a=6$), cases
  0–5 of 923801: failing `lat8192` (combined `SELECTION-3D-POSTHOC.json`: W512 `lat8192` under FS-BE has
  dist_to_conv $4.2\times10^{-3}>\tau=2.5\times10^{-4}$), deployed `lat32768`, finer **explicit Gauss comparator
  `gl56`** (the post-hoc converged rule; speed3d `DESIGN.md:415–418`), check `gl64`. Memory projection: off-mesh tables
  $8m(2r+M)$ bytes — `gl64` ($m=262{,}144$, $M\approx2052$): $8\cdot262144\cdot3076\approx6.5$ GB; fits an H200.
- **P1f (memory, offline)**: per configuration, live table bytes (device `bytes_in_use` delta when the configuration's
  arrays are built), the compiled query's `memory_analysis()` (argument, output, temporary bytes), and the process-lifetime
  peak after each phase (labelled process-lifetime, never per-panel). Offline inventory from logs: bank training (sunk),
  per-mesh bank build, operators, rule tables, compilation, each from this lane's jobs (new) or archived logs (sunk).
- **P1g (frozen-mode recheck, §2.6)** and **Q3 (conditioning, §5.3)**.

### 2.6 P1g: historical claims re-checked (AM-6, AM-7)

The historical cross-mesh 2D accurate-setting claims (located before data; only comparisons that include $256^2$ are
affected, by §2.3) all concern **`acc` = R384, M=1536, BE vendor LM query** against the first-order $8192^2$ references:

| # | source (file:line) | claim | rule(s) |
|---|---|---|---|
| H1 | `reports/2026-10-01-burgers2d-offmesh-quadrature.md:102–129` | per-case spread of the worst ST error over $256^2/1024^2/4096^2$, dev6∪val32: acc Gauss $64^2$ 0.014 pp, gref 0.012, Fib 6765 0.012 | gauss64, gref, fib6765 |
| H2 | same file `:70–79` | test64 B2 max/min: acc Gauss $96^2$ 1.0034 (and other acc rows) | gauss96 et al. (test64; replication-only, not re-run) |
| H3 | same file `:49,:57` | "largest per-case spread 0.020 / 0.011 pp" (maxima from `fast`, M=512, unaffected) and acc gref/dense worst ST per mesh | — |
| H4 | `worktrees/2026-10-06-jcp-mechanism/experiments/jcp-mechanism/report.md:94` | 2D per-case mesh spread $256^2$ vs $1024^2$: acc gauss96 0.008 pp, gref 0.008 pp | gauss96 |
| H5 | restatements: `reports/2026-10-06-jcp-offmesh-paper-plan.md:27` ("≤ 0.011 pp", rated solid); `…/01-evidence-inventory.md:17,78`; `…/03-theory.md:100,210` ("only h-dependence is in A and Λ" — omits the test set) | — | — |

P1g therefore runs, at $256^2$ and $4096^2$ on all 38 dev6∪val32 cases, frozen **and** nominal lists for (a) R384
`gauss128` × {BE, CN-R} (the P1 configuration; the frozen-list runs at $256^2$ (38) and $4096^2$ (16 of 38) are shared
with P1d) and (b) acc `gauss96` × LSPG-BE (the historical rule; generic LSPG-BE equals the vendor query to $10^{-8}$,
combined G1a). At $1024^2$ frozen = nominal. Quantities, per case: the frozen–nominal trajectory distance
(scoring-lattice, reference-free), $e_{\rm frozen}-e_{\rm nominal}$ against E0 $R_{ST}$ **and** against the historical
first-order ST reference (`quadrature-study/runs/refdv`, local), and the per-case cross-mesh spread
$\max_N e-\min_N e$ under each list. Reported next to H1 and H4 (the historical numbers quoted with file:line); H2 is
listed as not re-run (test64 replication-only). gauss64/gref/fib6765 are not re-run; their moved values are not claimed.

## 3. Cohorts, references, scoring (common brief §2)

- Cohorts: dev6 (`params_draw(7090702,4)` ∪ `params_draw(911702,2)`) and val32 (`params_draw(20260927,32)`), disjoint
  from training (asserted). 3D: 923801 cases 0–5 only. No test64/923901 run; no sealed/OOD access; `sealed/` and
  `custody/` never touched.
- Reference: E0 $R_{ST}$ through `refstore` (store version pinned in the manifest; local copy for scoring, cluster copy
  never needed since scoring is offline on CPU). $e=\max_{k=1..5}\lVert u(t_k)-R(t_k)\rVert/\lVert R(t_0)\rVert$ on the
  $257^2$ nodes $x=i/256$. Eligibility per case-kind from the store (`reference_status`); at present all 38 ST case-kinds
  are eligible and S is non-eligible for dev6|1, val32|21, val32|24 (re-read at scoring time, never hard-coded).
  $R_S$ only for LSPG-BE at $\Delta t_0$, labelled; every other S comparison "unmatched diagnostic".
- $\varepsilon_{\rm ref,i}$ = the store's eps_ref of case $i$, kind ST. Ties: $|\Delta e|\le2\varepsilon_{\rm ref}$.
- Every scored record carries (cohort, case, time index set, reference kind, reference sha256, store version, metric
  id); the scorer asserts the identity of every record against the rollout's own identity (swap control, §6).
- Statistics: cases are the unit; no time snapshots bootstrapped; failures reported.

## 4. Decision rules (frozen before data)

### 4.1 H-cost-N (P1c; V3-3, V4-4, AM-4)

Timed quantity: **evolution only** (`evolve(w0, nu, scale, data, sch)`, jitted, `block_until_ready`). A = fixed reference
configuration (r=384, $1024^2$, LSPG-CN-R, frozen list, `gauss128`), interleaved A–B–A on the same case in **one job on
one H200** that holds all three meshes (§7). Per triplet: $\rho=t_B/\tfrac12(t_{A1}+t_{A2})$. Drift: if
$t_{A2}/t_{A1}\notin[1/1.1,1.1]$ the triplet is repeated (≤2 retries); still drifting → invalid. A subject with
$>1/3$ invalid triplets on any mesh → "panel invalid; verdict not reached".

Per (r, scheme, mesh): $m_N=$ median of $\rho$ over 6 cases × 3 repeats; $R=\max_N m_N/\min_N m_N$. 95 % percentile
interval from a case bootstrap (10,000 resamples, `numpy.random.default_rng(20261017)`; each resample draws 6 case indices
with replacement and carries every mesh, repeat and configuration of the drawn cases jointly; $R$ recomputed per
resample). **Pass** if upper ≤ 1.25; **fail** if lower > 1.25; else **inconclusive**. A missing mesh → "verdict not
reached"; with $4096^2$ dropped: "two-mesh evidence, registered three-mesh verdict not reached". Primary verdicts: the
6 (r, scheme) frozen-list subjects. The nominal-policy R384 subjects get the same statistic, labelled "deployment
policy". Init, decode and the full fused query are timed in the same triplets (§7) and reported as $t$ and as ratio to A;
they are **expected** to grow with $N$ and are not part of the verdict. Each verdict carries the numerical-qualification
label of its three configurations (§5); a pass on an unqualified configuration is reported as such, not suppressed.

### 4.2 H-cost-r (P1a, P1b; AM-4)

Components (each its own jitted function, timed with `block_until_ready`):

| component | definition | pre-declared $p$ at fixed $(M,m)$ |
|---|---|---|
| `N` | $\Psi^{\mathsf T}[(G_qc)\odot(G_sc)]$ (3D: $P^{\mathsf T}[(Bc)\odot(Dc)]$) | 1 |
| `J` | $\Psi^{\mathsf T}[\mathrm{diag}(G_sc)G_q+\mathrm{diag}(G_qc)G_s]$ (3D analogue) | 1 |
| `JTJ` | $J^{\mathsf T}J$, $J\in\mathbb R^{M\times r}$ | 2 |
| `chol` | Cholesky factor + solve of $J^{\mathsf T}J+\lambda\,\mathrm{diag}$ | 3 |
| descriptive only | full residual `res`, full residual+Jacobian `evalJ`, gradient `g` $=J^{\mathsf T}r$, LM iteration body | — |

Fit $t(r)=a+b\,r^p$ by nonlinear least squares (`scipy.optimize.least_squares`, bounds $a\ge0$, $b>0$, $p\in[0,4]$, on
per-$r$ means of the fixture medians) over $r=32$–512 (2D) / 64–1024 (3D). Interval: case bootstrap (10,000; the 6 cases
resampled with replacement, each case carrying its 2 fixtures × all $r$; seed 20261017), percentile 95 % on $p$.
**Verdict only on the fixed-$(M,m)$ panel, per component:** fit failure or a boundary solution ($p\le10^{-6}$,
$p\ge4-10^{-6}$, or $a\le10^{-9}$ s) of the point fit → inconclusive; > 5 % of bootstrap fits failing/boundary →
inconclusive; **pass** = pre-declared $p$ inside the interval **and** half-width ≤ 0.5; **fail** = $p$ outside the
interval; **inconclusive** otherwise. Secondary label of a fail: compute the operation-model exponents $p_F$, $p_B$ by
the same fit applied to the per-component FLOP and byte counts on the same $r$ grid (counts in §4.2a); **"operation-model"**
if $|p_F-p|>0.25$ or $|p_B-p|>0.25$ **and** that departing model exponent lies inside the measured interval; otherwise
**"hardware scaling"**. Fixed-$m$ and nominal panels: descriptive, plotted against the FLOP model. Fixed-cost
cross-check: $\hat a$ vs the measured time at the smallest rank of the panel (2D $r=32$, 3D $r=64$), reported.

**4.2a FLOP/byte counts (f64, 8 B):** `N`: $4mr+m+2mM$ flops, $8(2mr+mM)$ B; `J`: $2mMr+4mr$ flops, $8(2mr+mM+Mr)$ B;
`JTJ`: $2Mr^2$ flops, $8(Mr+r^2)$ B; `chol`: $r^3/3+2r^2$ flops, $8r^2$ B.

### 4.3 H-acc-N (P1d; V4-5, AM-4)

Per (r, scheme), primary on frozen lists, separately for each pair: P256 = ($256^2$ vs $1024^2$) on the cases eligible at
both (≤ 38); P4096 = ($4096^2$ vs $1024^2$) on the 16-case subset eligible at both. Per case
$d_i=(e_i(N)-e_i(1024^2))/\delta_i$, $\delta_i=\max(2\varepsilon_{{\rm ref},i},\,0.1\,e_i(1024^2))$; absolute errors
reported beside. **TOST** = two one-sided Wilcoxon signed-rank tests (`scipy.stats.wilcoxon`, `zero_method='wilcox'`,
`method='auto'`, i.e. exact for $n\le50$ without zeros/ties) of $d_i+1>0$ (`alternative='greater'`) and $d_i-1<0$
(`'less'`), α = 0.05 each; equivalence iff both reject. Estimand label: "normalised signed-rank equivalence", not a test
of the mean error difference. Within-margin fraction $=\#\{|d_i|\le1\}/n$ of the pair's population. Trend: per-case OLS
slope of $e$ vs $\log_2N$ over the three meshes on the 16-case subset (three-mesh eligible intersection); two-sided
Wilcoxon of the slopes vs 0, α = 0.05; zero slopes dropped; fewer than 6 non-zero slopes → "not testable".
**Pass** = TOST equivalence for both pairs **and** within-margin ≥ 90 % in each pair's actual population.
**Fail** = TOST fails for at least one pair **and** (trend rejects **or** < 50 % of some pair within margin).
**Inconclusive** otherwise. Empty pair population (n = 0) → inconclusive; a pair with $n<6$ cannot reject in TOST (exact
Wilcoxon minimum p = $2^{-n}$ one-sided) and is reported as such. If any of the three configurations of the (r, scheme)
is numerically unqualified (§5): outcome **inconclusive (unqualified configuration)**, raw procedure result reported
beside. If $4096^2$ is dropped: P256 only, "two-mesh evidence, registered three-mesh verdict not reached". The
nominal-policy R384 arm gets the same procedure, labelled.

### 4.4 H-acc-r (P1d; AM-4, AM-7 with precedence)

Transitions 128→256, 256→384 at every mesh (16-case subset at $4096^2$), 384→512 at $1024^2$. Per transition and
(mesh, scheme), population = cases eligible (ST). A case is **qualified** under a configuration iff the configuration
passes its configuration-level Q1 predicates and Q2 is "pass", and the case's own run is valid with distance to its
tight solve ≤ $2.5\times10^{-4}$.
1. If > 25 % of the population lack qualification on either rung → **inconclusive** (stop).
2. Resolved = qualified on both rungs and $|\Delta e|>2\varepsilon_{\rm ref}$. If resolved < 50 % of the population →
   **inconclusive** (stop).
3. Else **improves** iff > 50 % of resolved improve ($\Delta e<0$) and the median paired change over resolved < 0;
   otherwise **does not improve**.
Range verdict per (mesh, scheme): **monotone** iff every transition "improves"; **not monotone** iff any "does not
improve"; else **inconclusive**. Registered ranges: 128–384 at each mesh (primary) and 128–512 at $1024^2$ (registered
through 512 per V3-5; the known 384→512 regression is expected to give "not monotone"). Label: "majority improvement on
resolved cases" — never "every case" or "worst error". Every transition is reported with improve/worsen/tie/unqualified
counts, the median paired change, and the worst-case change (largest $\Delta e$). The Pareto front (error vs P1c cost per
mesh) is reported separately and is not a monotonicity test.

## 5. Qualification (V4-2, AM-2, AM-6)

### 5.1 Q1 — numerical solver qualification, every rollout configuration (BE included)

Certified **in-job, at every step, with a fresh Jacobian**: every production rollout records its full internal
trajectory $c_0,\dots,c_{S}$; an untimed certifier (`p1core.make_certifier`) rebuilds each step's LSPG residual from the
saved states (scheme coefficients incl. BE start-up, history $Ac_n$, $Ac_{n-1}$, $F(c_n)$) and forms the Jacobian
**independently by forward-mode autodiff** (`jax.jacfwd` of the residual, not the analytic `nlJ`), giving
$\gamma_n=\lVert J^{\mathsf T}r\rVert/(\lVert J\rVert_F\lVert r\rVert+10^{-300})$. Gate G-cert: the certified $\gamma_n$
equals the solver's own recorded ratio to $10^{-6}$ relative on every step (else Q1 is blocked for that configuration as
"certifier mismatch"). Configuration-level Q1 predicates:
1. every run valid: all fields finite, no non-finite step, no damping-limit exit (reason 3), no budget exit (reason 0);
2. non-stationary steps ($\gamma_n>10^{-3}$ or non-finite) ≤ 1 % **pooled over all steps of all cases**;
3. scoring-lattice relative distance to the adaptive **tight** solve of the same configuration (tolerance level
   swapped only), $\max_{k=1..5}\lVert u_k-u_k^{\rm tight}\rVert/\lVert u_0\rVert_{257}$, ≤ $2.5\times10^{-4}$ on
   **every** case (tight anchors run on every case of every configuration; a tight solve that is itself invalid makes
   that case unqualified);
4. tight/tighter agreement ≤ $2.5\times10^{-5}$ on dev6 cases 0–5 (tighter run on those cases).

### 5.2 Q2 — quadrature qualification (AM-2), validation population of 16 cases

Population: dev6 + val32[0:10] (16 cases) at every mesh, for every configuration (P1d frozen, nominal, acc, R512).
For rule $q$ with successor $q^+$ and check $q^{++}$ (§2.2):
- **rollout bar (primary):** replay with $q^+$ (everything else equal); max over the five evolved outputs of the
  scoring-lattice relative distance ≤ $2.5\times10^{-4}$ on every one of the 16 cases;
- **ρ bar (secondary):** on the arm's own reached states (6 output times × 16 cases),
  $\rho(c)=\lVert N_q(c)-N_{q^+}(c)\rVert/\lVert N_{q^+}(c)\rVert$ of the tested advection vector (length $M$);
  $\rho_{\max}\le0.116$ (threshold borrowed from quadrature-study `DESIGN.md:234`, B5; this is a successor-rule /
  output-state check, not a reproduction of B5's certificate); the distribution is reported;
- **finer-rule convergence (dev6):** the $q^+$ replay against $q^{++}$ must meet the rollout bar on dev6; if not, Q2 =
  "unverified (finer rule not converged)".
Q2 label ∈ {pass, fail, unverified}. P1c subjects inherit the P1d label of the identical numerical configuration. The
other 22 cases are never described as individually Q2-tested.

**Numerically qualified** = Q1 pass and Q2 pass. Three labels kept separate (V4-2): numerical qualification; role
qualification (none is pre-declared in P1 — every P1 configuration is reported as "role not assessed"); frontier
usefulness (non-dominated on the error–cost plane among qualified configurations at the same mesh and scheme).

### 5.3 Q3 — conditioning

For every P1 rollout configuration: singular values of $A$ (per (r, mesh, list)), and of the weighted LSPG Jacobian
$J(c)=A+\beta_0\Delta t\,S\,\partial N(c)$ (and so $\mathrm{cond}(J^{\mathsf T}J)=\mathrm{cond}(J)^2$) at output indices
1, 3, 5 of each dev6 case's own trajectory. Report median and max condition number per configuration, and rank at
relative tolerance $10^{-13}$ (`T2.a_conditioning` convention).

## 6. Controls (AM-5, AM-7, AM-8), each with its fixture, predicate and the claim it blocks

Every control is shown to behave as required in the real-size smoke at $1024^2$ (synthetic ones in CPU tests,
`test_p1.py`) before any verdict is read; a misbehaving control blocks the listed claim until fixed and re-audited.

| control | kind | fixture and predicate | blocks |
|---|---|---|---|
| synthetic timing fault | negative | `default_rng(0)`, 6 cases × 3 meshes × 3 repeats, $t=1.00(1+0.01\,z)$; one mesh × 2.0 → H-cost-N **fail**; unmodified → **pass** (CPU) | H-cost-N |
| production timing fault | probe | smoke data, every case × 3.0 on one mesh; outcome reported, no requirement | — |
| doubled work | negative | B = two separately dispatched, separately synchronised executions of the same compiled evolution from the same initial state, timed as one interval; A = one execution; both inside the same A–B–A protocol, warm, 0.1 s burn before each timed interval; predicate $B/A\in[1.8,2.2]$ **and** identical iteration counts in both executions; run for R384-CN-R $1024^2$ and R128-BE $256^2$ on dev6 (smoke: 2 cases) | H-cost-N, H-cost-r |
| component sum vs fused | positive | matched scope: for (r ∈ {128, 384}, $1024^2$, nominal rule, CN-R), $\sum_n[3t_{\rm res}+t_{\rm evalJ}+i_n(t_{\rm JTJ}+t_{\rm chol}+t_{\rm evalJ}+t_g)]$ with $i_n$ the recorded LM iterations of step $n$ of the same query (rejected iterations counted, since they execute the same body), vs that query's evolution time: ratio ∈ [0.9, 1.6] | H-cost-r |
| kernel dispatch sync | positive | each verdict component at the largest $r$ of the fixed panel: two separately synchronised dispatches / one ∈ [1.8, 2.2]; reference-kernel drift start vs end of each panel ∈ [1/1.1, 1.1] (else re-run once, then inconclusive) | H-cost-r |
| split-query equivalence | positive | init + evolve + decode vs fused query: fields ≤ $10^{-12}$ relative and identical per-step iteration counts, every subject, case dev6\|0 | H-cost-N |
| timed = scored | positive | in the timing job every timed W has the sha256 of that job's own untimed scoring run of the same (subject, case); those scoring runs are scored against E0 (dev6) and compared with the accuracy job's run of the same configuration (distance reported; the accuracy label is inherited by configuration identity) | all timing claims |
| reproduction gate | positive | R128 `fib4181` and R384 `gauss128` at $1024^2$, BE and CN-R, all 38 cases: $\max_k\lVert W_k-W_k^{\rm arch}\rVert/\lVert W_0^{\rm arch}\rVert\le10^{-10}$ vs combined `W_R128.npz`/`W_R384.npz`, on a100-80G (the archive's GPU type), after numerical-manifest equality | H-acc-N, H-acc-r |
| wrong-mesh table | negative | every data dict carries (L, mesh hash, mode-list hash); a $256^2$ table fed to the $1024^2$ query wrapper must raise | all |
| under-resolved rule | negative | R384 `gauss32` (successor `gauss64`) on the R384-`gauss128` CN-R $1024^2$ own reached states, dev6, output index 5: an independent CPU NumPy ρ (`npbank`, no JAX) of gauss32 vs gauss64 must exceed 0.116 **before** use; then in the smoke the actual Q2 predicate of the gauss32 configuration (rollout vs gauss64 replay) must fail on ≥ 1 case and its in-job $\rho_{\max}>0.116$ | Q2, all fronts |
| Q1 stationarity | negative | premature-exit solver: LM budget 0 (returns the quadratic predictor) for R384-CN-R $1024^2$ dev6; independent NumPy $\gamma$ at its states (dev6\|0, all 50 steps) must exceed $10^{-3}$ on > 1 % of pooled steps before use; then Q1 must report pooled non-stationary > 1 % | Q1 |
| sign-flipped Jacobian | probe | advection Jacobian sign flipped, linear part intact: reported only | — |
| zero-update solver | negative | $c_{n+1}=c_n$ (trajectory constant $=c_0$) on dev6\|0 R384-CN-R $1024^2$, whose verified tight trajectory moves by $\lVert u_5^{\rm tight}-u_0\rVert/\lVert u_0\rVert>100\times2.5\times10^{-4}$ (checked, same scoring contract); Q1 distance must exceed $2.5\times10^{-4}$ | Q1 |
| rank-deficient fixture | negative | $A$ with a duplicated column: Q3 must report rank deficit at $10^{-13}$ and cond $>10^{12}$ (CPU) | Q3 |
| case / time / reference-kind swap | negative | a swapped record (case, time index, or kind) fed to the scorer must raise (CPU, real records) | all accuracy |
| synthetic accuracy trend | negative | 16 cases, $\varepsilon_{\rm ref}=0$, $e(1024^2)=1\%$, $e(256^2)=0.5\%$, $e(4096^2)=1.5\%$, + $0.01\%\cdot z$ (`default_rng(0)`) → H-acc-N **fail**; all three equal ± noise → **pass** (CPU) | H-acc-N |
| H-acc-r precedence | negative | synthetic: 38 cases, 10 unqualified, 28 improving → **inconclusive**; 0 unqualified, all improve → improves; all worsen → does not improve (CPU) | H-acc-r |
| H-cost-r fit | negative | synthetic $t=a+br^{2}$ (+1 % noise) for a component declared $p=3$ → **fail**; with $p=2$ declared → **pass** (CPU) | H-cost-r |
| CN-R temporal order | positive | GAL-CN-R tight, dev6\|0, R384 $1024^2$: $\Delta t_0,\Delta t_0/2,\Delta t_0/4$ vs the anchor $\Delta t_0/16$ (same scheme, form, rule). Order $=$ LS slope of $\log e$ vs $\log h$ over the three steps ∈ [1.8, 2.2]. Interpretable only if (i) asymptotic: the two pairwise orders differ by ≤ 0.3; (ii) anchor: $d(\Delta t_0/16,\Delta t_0/32)\le0.1\,e(\Delta t_0/4)$; (iii) solver: $d({\rm tight},{\rm tighter})$ at $\Delta t_0/4\le0.1\,(e(\Delta t_0/2)-e(\Delta t_0/4))$; (iv) quadrature: $\lvert e_q(\Delta t_0/4)-e_{q^+}(\Delta t_0/4)\rvert\le0.1\,(e(\Delta t_0/2)-e(\Delta t_0/4))$ ($q^+$ = gauss160, all runs repeated). Else "order check not interpretable". LSPG-CN-R order measured the same way, reported, no gate | scheme labels |
| sensitivity probes | probe | GPU type (H-cost-N ratios on A100 from the accuracy jobs' untimed seconds vs H200, dimensionless, reported); repeat seed; one frozen Fibonacci shift (no shift sweep, E1) | — |

## 7. Jobs, inventory and timing protocol

| job | GPU | content |
|---|---|---|
| `smk` | a100-80G | real-size smoke at $1024^2$: every control of §6, the reproduction gate on dev6, a 2-case version of every P1d/Q1/Q2/Q3 path for R128/R384, a mini timing panel (A + 2 subjects, $256^2$ and $1024^2$), the kernel harness at $r\in\{128,384\}$, the CN-R order check (full) |
| `pilot` | h200 | $4096^2$ memory pilot (§9.2) with the $256^2$ and $1024^2$ banks co-resident (the timing job's footprint) |
| `acc256` | a100-80G | P1d $256^2$ (R128/R256/R384 × BE/CN-R × 38), P1g nominal R384 `gauss128` × 2 × 38, acc `gauss96` BE frozen + nominal × 38; tight on all, tighter dev6, Q2 16 cases, Q3 |
| `acc1024` | a100-80G | P1d $1024^2$ (R128/R256/R384/R512 × 2 × 38), reproduction gate, acc `gauss96` BE × 38, P1e-2D, CN-R order check, Q1/Q2/Q3 |
| `acc4096` | h200 | P1d $4096^2$ (R128/R256 × 2 × 16; R384 frozen × 2 × 38), nominal R384 × 2 × 38, acc `gauss96` BE frozen + nominal × 38; Q1/Q2/Q3 |
| `time` | h200 | P1c all meshes (banks co-resident), doubled work, split-query, timed = scored, component-sum check, memory per subject |
| `kern2d` | h200 | P1a, all panels |
| `kern3d` | h200 | P1b |
| `p1e3d` | h200 | P1e-3D |

Timing protocol (P1c): every subject is compiled and run once per timing case untimed (warm; the untimed run is the
"scoring run" of §6); then the triplet order over (subject, case, repeat) is randomised (`default_rng(20261017)`); before
every timed interval a 0.1 s GPU burn; each interval is the wall time from dispatch to `block_until_ready` of the
evolution output. In each triplet B's init, evolve and decode are timed as three consecutive synchronised intervals,
then (after the second A) a separate triplet times B's fused query. Every timed W's sha256 is compared with the scoring
run's. Iteration counts per step are recorded for every timed call.

Kernel protocol (P1a/P1b): per (panel, $r$, fixture, component): 3 warm calls, then 30 individually synchronised calls,
median recorded; the order of $r$ within each fixture is randomised; a reference kernel (fixed-panel `chol` at the
largest $r$) is timed at the start and end of each panel (drift, §6).

Run inventory (exact counts generated by `p1acc.py --dry-run`; summary): P1d new production rollouts
$3\times2\times(38+38+16)=552$ (+76 R512); P1g extra: R384 frozen $4096^2$ +22×2, nominal R384 $2\times2\times38=152$,
acc `gauss96` BE $5\times38=190$; a tight anchor per production rollout; tighter on dev6 per configuration; Q2 $q^+$
16 cases and $q^{++}$ 6 cases per configuration; P1e-2D 6; controls; P1c $24\times6\times3$ triplets ×2 (split +
fused) + doubled work; kernels seconds each.

## 8. Analysis and deliverables

`p1score.py` (CPU): E0 scoring (and first-order scoring for P1g), Q1/Q2 recomputation from the saved fields and states
(acceptance: in-job and offline values agree to $10^{-9}$ relative), identity assertions. `p1stats.py`: bootstrap, NLS,
Wilcoxon/TOST, verdict functions (unit-tested with the synthetic controls). `analyze.py` → `runs/EXPORT.json` (every
verdict per $r$, scheme, panel; key numbers; a `contract` field on every cost number: "full-field 2D query" or
"kernel"). `make_report.py` → `report.md` + PNG plots: cost vs $r$ per component (2D, 3D) with FLOP model; solve time
vs $N$ per $r$; error vs $r$ per mesh; error vs $N$ per $r$; P1g before/after; memory vs $r$; conditioning vs $r$. No
number is hand-typed. The hand-back follows common brief §10.

Speed3d imports (coordinator's final report; AM-6 map): g1–g3 → 3D Q1 for those arms; f1 → 3D FOM set (finite ladder:
upper bound on the optimal FOM cost); m1 → 3D cost vs N under the **mesh-free / scoring-lattice** contract (never
substituted for this lane's full-field contract); p1 → batching. This lane cites them only if speed3d's results audit
has passed; it runs none of them.

## 9. Resources

### 9.1 Cap, ledger, certificates (AM-6)

Hard cap **20 GPU-h** including contingency. `runs/GPU-LEDGER.json` records every job (id, GPU, `sacct` elapsed
GPU-h, projection, state). A job is submitted only if spent + Σ certified projections of queued/running jobs + this
job's projection ≤ 20, the lane has < 4 jobs queued/running, and `squeue -u tawal01 -h | wc -l` < 8. Smoke and pilot
are charged their wall-time allocation as projection (smoke 2 h, pilot 1 h). Production projections come from
`resources.py` using the smoke's measured setup, compile and per-rollout times (A100) and the pilot's (H200), scaled by
rollout counts, × 1.5; the sbatch `--time` must cover the projection. Failed attempts, compilation, certification,
anchors, Q2 replays and controls all count.

### 9.2 $4096^2$ memory pilot

On an H200 (actual device bytes 143,771 MiB): build the full R512 bank at $4096^2$, operators for $F_{2048}$, rule
tables of R384 `gauss128`/`gauss160`, the $256^2$ and $1024^2$ banks co-resident, one BE and one CN-R rollout (prod and
tight) per $r\in\{128,256,384\}$; record `peak_bytes_in_use`. If peak > 120 GB (decimal) the $4096^2$ arms switch to the
prefix-retaining builder (re-piloted in the first production job's setup), labelled so. No $4096^2$ production run
before the pilot. If even the prefix-retaining builder is infeasible, the outcome is "P1 incomplete" (P1g not reached).

### 9.3 Defer order and core (V3-2, V4)

First dropped first: P1f-offline → P1e 3D → P1b → R512 diagnostic → P1d $4096^2$ subset → P1c $4096^2$.
**Never dropped:** P1a, P1c at $256^2/1024^2$, P1d at $256^2$, P1g (including $4096^2$). If the core cannot finish
inside the cap the outcome is **"P1 incomplete"** (no verdict; shortfall reported). Deferred items are reported, never
silently omitted. Rough a-priori estimate (re-priced by the certificate): smoke 1, pilot 0.5, acc256 1, acc1024 1.5,
acc4096 2, time 1, kern2d 0.3, kern3d 0.3, p1e3d 0.5 GPU-h ≈ 8 GPU-h.

## 10. Code (copy, then edit in the lane; `vendor/` untouched)

`p1core.py` (copy of combined `t2core.py` + optional internal-state recording, certifier, ρ, conditioning),
`p1common.py` (copy of combined `c2common.py` adapted: explicit mode lists, table metadata, prefix-retaining bank,
split query), `p1acc.py`, `p1time.py`, `p1kern.py`, `p1kern3d.py`, `p1e3d.py` (3D via combined `c3common.py`/`fs3.py`
copies), `npcheck.py` (independent NumPy residual/γ/ρ), `p1score.py`, `p1stats.py`, `test_p1.py`, `manifest.py`,
`resources.py`, `ledger.py`, `cluster/` (stage, sbatch template, collect), `analyze.py`, `make_report.py`. Provenance of
every copied file (source branch, commit, sha256) in `PROVENANCE.json`.

## 11. Integrity rules carried

Never quote a pre-September Burgers speed-up; never mention the heat POD-128 comparison; retractions stated plainly
(P1g exists to measure one: the mode-set dependence of earlier cross-mesh claims). Every GPU number from the cluster;
local GB10 only for CPU analysis (`JAX_PLATFORMS=cpu`) and at most one `jaxrun`.

## 12. Audit-delegated "safe lane work", frozen here

| item (source) | where frozen |
|---|---|
| discriminating fixtures and aggregate predicates for under-resolved / Q1 / zero-update (AM-7, AM-8, round 3) | §6 rows; under-resolved fails the **actual Q2 predicate** vs its successor; Q1 negative fails the **pooled** criterion; zero-update motion measured on the verified tight trajectory under the Q1 contract |
| component accounting for sum vs fused (round 3) | §6, per-step invocation counts incl. predictor and rejected iterations |
| bootstrap, numerical edge cases, empty populations (round 2) | §4.1–4.4 |
| configuration-level Q2 labels, 16-case scope (round 2) | §5.2 |
| temporal anchor and asymptotic checks (round 3) | §6 CN-R row |
| historical `acc` vs `R384` mapping (round 2) | §2.6 |
| explicit 3D Gauss comparator (round 2) | §2.5 P1e: `gl56`, check `gl64` |
| revised cost certificate and reservations (round 1–2) | §9.1 |
| matched timing boundaries and warm-up for doubled work (round 4) | §6, §7 |
| frozen Fibonacci realisation (round 2) | §2.2, §6 probes |

---

## Amendment A1 (2026-10-18, before any lane job; answers `audits/codex-design-1.md`, NEEDS-RESTATEMENT)

Supersedes the text above wherever they differ. Numbered by the audit's blockers, then its remaining items.

**A1.1 Under-resolved-rule control (blocker 1), re-fixtured with independently established expected results.**
Fixture: R384, $M=1536$ (frozen list, identical to the $1024^2$ builder list), `gauss32` with successor `gauss64`,
**LSPG-BE** $\Delta t_0$, $1024^2$, dev6 (6 cases). Independent evidence, computed and committed **before any lane job**
(`ctrl_fixture_ur.py` → `checks/ctrl-fixture-under-resolved.json`), from the historical quadrature-study job dv1024
(4735709; vendor BE query on the same numerical configuration):
- rollout bar: per case $d(g_{32},g_{64})\ge d(g_{32},g_{\rm ref})-d(g_{64},g_{\rm ref})$ (triangle inequality, same
  scoring lattice and normalisation, the job's per-case `vs_gref_restricted_evolved`): lower bounds
  $1.05\times10^{-3}$ … $3.36\times10^{-2}$ on all six dev6 cases, every one $>2.5\times10^{-4}$;
- ρ bar: NumPy ρ(gauss32 vs gauss64) with numpy Gauss–Legendre rules and the NumPy bank (`npcheck.py`), on
  **gauss32's own** reached output states (historical audit trajectories dev6|0, dev6|2): 1.27–1.58, $>0.116$.
Required in the smoke: the actual Q2 predicate of this configuration (rollout vs `gauss64` replay, and in-job ρ on its
own states) must **fail** — rollout distance $>2.5\times10^{-4}$ on ≥ 1 case and $\rho_{\max}>0.116$; and the offline
NumPy ρ on the smoke's own gauss32 states must agree with the in-job ρ (tolerance A1.3). Disposition if the smoke
does not fail although the independent evidence says it must: the Q2 code is wrong (blocks Q2 and every front). The
G1a-type vendor-vs-generic bridge (A1.5) is run on this fixture's configuration too. The earlier CN-R / gauss128-state
fixture is withdrawn.

**A1.2 Component sum vs fused (blocker 1), restated as a sanity band with complete accounting.** Matched scope, for
$(r\in\{384,512\}$, $1024^2$, nominal rule, LSPG-CN-R$)$ on dev6|0 (small $r$ excluded: their per-dispatch overhead is
comparable to the work). Components timed as separately synchronised dispatches of the solver's own sub-computations:
`hist` ($Ac_k$, $Ac_{k-1}$ and, on CN steps, $F(c_k)$), `pred` (the batched `vmap` of the three candidate residual norms
and the argmin), `init` (initial `evalJ` + gradient $J^{\mathsf T}r$ + ratio), `iter` (one LM body: $J^{\mathsf T}J$,
damped Cholesky solve, clip, `evalJ` at the trial point, gradient, ratio, accept/reject select). Model time
$\sum_n[t_{\rm hist}(n)+t_{\rm pred}+t_{\rm init}+i_n\,t_{\rm iter}]$ with $i_n$ the recorded LM iterations of step $n$ of
the same query (rejected iterations included; CN-R steps 0–1 are BE, so $t_{\rm hist}$ without $F$). Predicate:
model / evolution time ∈ **[0.5, 2.0]** (a factor-2 sanity band: separate dispatches pay launch costs the fused loop
does not, so equality is not expected). Outside the band → H-cost-r blocked ("components do not represent the solver's
work"); inside → the ratio is reported.

**A1.3 Agreement tolerances (blocker 3).** All in-job vs independent agreements use absolute + relative tolerances:
G-cert $|\gamma_{\rm cert}-\gamma_{\rm rec}|\le10^{-7}\gamma_{\rm rec}+10^{-12}$; offline recomputation of distances,
errors and ρ: $|x_{\rm off}-x_{\rm job}|\le10^{-9}|x_{\rm job}|+10^{-13}$; NumPy γ vs certifier γ:
$\le10^{-6}\gamma+10^{-12}$.

**A1.4 Validity follows the governing contract (blocker 3).** A run is **valid** iff all outputs finite, no non-finite
step, and no damping-saturation exit (reason 3; a non-finite residual at step entry is reason 3). Budget exits
(reason 0) are **not** invalid by themselves; they are counted and reported per configuration, and such steps are judged
by their certified γ (non-stationary if $\gamma>10^{-3}$). This replaces §5.1 item 1's budget clause. The same
validity applies to tight/tighter anchors; an invalid anchor makes its case unqualified (unchanged).

**A1.5 Cross-hardware qualification transfer (blocker 3).** P1c subjects inherit the P1d numerical label of the same
configuration only through a **transfer gate**: for every subject and dev6 case, the timing job's scoring-run W
(H200) decoded at the scoring lattice vs the accuracy job's W of the same configuration and case: distance
$\le2.5\times10^{-5}$ (the tight/tighter bar). A subject failing it gets the label "qualification not transferred"; its
timing is reported, its numerical label is not. (At $4096^2$ both are H200 jobs; the gate is applied identically.)
Also, vendor-vs-generic bridge: in `acc1024` (and the smoke) the vendor BE query (`qcore.make_linear_query`, `point`)
is run on dev6 for every rule used by a historical claim (gauss32, gauss64, gauss96, fib6765, gauss640) and for
R384-gauss128; generic LSPG-BE vs vendor: $\max_k\lVert\Delta W_k\rVert/\lVert W_0\rVert\le10^{-8}$ and
$w_0$ to $10^{-12}$ (combined G1a). A failure blocks the historical comparisons that rely on that rule.

**A1.6 Q1 negative fixture (blocker 4).** Population: the six dev6 cases × 50 steps = 300 pooled steps of
R384-`gauss128` LSPG-CN-R $1024^2$ run with LM budget 0 (the solver returns its **selected initial candidate**: the
lowest-residual of $c_k$, linear and quadratic extrapolation — not necessarily the quadratic predictor). Independent
establishment: NumPy γ (`npcheck.gammas`, hand-written residual and Jacobian, A and Λ saved by the job, rule tables from
numpy Gauss–Legendre + NumPy bank) on **all 300** steps must give > 3 non-stationary steps (> 1 % of 300). Required: the
in-job certifier's pooled non-stationary fraction > 1 %, and per-step agreement with NumPy (A1.3).

**A1.7 CN-R order check: quadrature contamination as a trajectory distance (blocker 4).** Check (iv) becomes
$\max_k\lVert(u_q(h)-u_q(h_a))_k-(u_{q^+}(h)-u_{q^+}(h_a))_k\rVert/\lVert u_0\rVert\le0.1\,(e(\Delta t_0/2)-e(\Delta t_0/4))$
at $h=\Delta t_0/4$, $h_a=\Delta t_0/16$ (the change of the error **vector**, not of its magnitude), plus finer-rule
convergence $d(u_{q^+}(h),u_{q^{++}}(h))\le2.5\times10^{-4}$ at $h=\Delta t_0/4$ ($q^{++}$ = gauss192). Checks (i)–(iii)
unchanged ((iii) is a direct trajectory distance already).

**A1.8 P1g historical claims (blocker 5): every H1/H4 rule is re-checked.** P1g's historical-rule arms run, LSPG-BE
$\Delta t_0$, on all 38 dev6∪val32 cases: rules `gauss64`, `fib6765`, `gauss640` (= gref, $m=409{,}600$) for H1 and
`gauss96` for H4; frozen + nominal at $256^2$ and $4096^2$, frozen at $1024^2$; tight anchors on every case, tighter on
dev6 (Q1). Q2 is not run for these historical-rule arms: their claim is a within-rule frozen-vs-nominal difference
(rule error common to both arms); this scope is stated next to every P1g number. The moved numbers reported next to H1
and H4 are the per-case spreads (max over meshes − min over meshes of the worst ST error) under frozen vs nominal,
against the historical first-order reference (to compare with 0.014 / 0.012 / 0.012 / 0.008 pp) and against E0.
Reproduction diagnostic: the nominal-list runs at $256^2$, $1024^2$ and $4096^2$ re-create the historical
configuration; their per-case first-order ST errors are compared with the historical job rows (dv256 4735696, dv1024
4735709, dv4096 4735717) and the largest difference reported (no gate; it shows that "nominal" is the historical
construction). H2 (test64) and H3 stay "not re-run". Memory: gauss640 tables at R384
$8\cdot409600\cdot(768+1536)\approx7.5$ GB, fine on a100-80G with the $1024^2$/$256^2$ banks and on the H200.

**A1.9 Verdict procedures made total (blocker 2).**
- *H-cost-N.* Invalid triplets are excluded. A bootstrap resample in which any (subject, mesh) of the statistic has no
  valid triplet is **undefined**; if > 2.5 % of resamples are undefined the outcome is "inconclusive (insufficient valid
  timings)", else the percentile interval is taken over the defined resamples.
- *H-acc-N.* The false statement "a pair with $n<6$ cannot reject" is **withdrawn** (with $n=5$ and all $d_i=0$ the exact
  one-sided p is $2^{-5}=0.031$). Margin floor: $\delta_i=\max(2\varepsilon_{{\rm ref},i},0.1\,e_i(1024^2),10^{-12})$.
  Wilcoxon specification (implemented in `p1stats.py`, verified against SciPy 1.17.1, the pinned analysis version):
  drop zeros (`zero_method='wilcox'`), no continuity correction; exact null distribution when the remaining $n\le50$
  and the $|d|$ have no ties, otherwise the normal approximation with tie-corrected variance; a one-sided test whose
  shifted sample is all zeros (n = 0 after dropping) **does not reject** ($p=1$). An empty pair population → inconclusive.
- *H-acc-r.* Empty eligible population → inconclusive (before step 1).
- *H-cost-r.* Fit $t=a+e^{\beta}r^p$ (so $b=e^\beta>0$), `least_squares` with bounds $a\ge0$, $p\in[0,4]$, $\beta$ free,
  `x_scale='jac'`, start $a_0=\min_r t$, $p_0$ = declared $p$, $\beta_0$ from a log-linear fit of $t-a_0$; failure = status
  ≤ 0, non-finite parameters, or relative RMS residual > 0.2. Boundary of a **measured** fit: $p\le10^{-6}$,
  $p\ge4-10^{-6}$, or **unidentified exponent** $b\,r_{\max}^p<0.01\,a$ (cost flat in $r$); $a\to0$ is **not** a boundary
  (pure power laws are legitimate). Operation-count fits use the same functional form, keep their exponent at any
  boundary, and never go through the boundary rule. Secondary label of a fail, **as AM-4 literally**: "operation-model"
  iff $|p_F-p|>0.25$ **or** $|p_B-p|>0.25$ (the FLOP or byte model itself departs from the declared $p$ over the range);
  otherwise "hardware scaling". $p_F$, $p_B$ and the measured interval are always reported beside it.
- *Synthetic H-cost-r control* (CPU): $t=a+br^2$ with $a=2\times10^{-5}$ s, $b$ such that $b\cdot512^2=10^{-2}$ s, 6 cases ×
  2 fixtures, multiplicative noise $1+0.01z$ (`default_rng(0)`), $r\in\{32,\dots,512\}$: declared $p=3$ → **fail**; declared
  $p=2$ → **pass**.

**A1.10 Timing and kernel protocol details (item 6).** Every burn is outside the measured interval. Kernels: a 0.1 s
GPU burn before every (panel, $r$, fixture, component) block, then 3 warm calls, then 30 synchronised calls; **all 30 raw
times are persisted**, the median is the fixture value. The smoke's kernel run includes the largest ranks of each fixed
panel (2D $r=512$; and a 3D mini-run at $r\in\{64,1024\}$ of the P1b fixed panel) so the dispatch-sync control is shown
at the ranks it governs. Hardware sensitivity probe (reported only): a matched mini A–B–A panel (A plus R128/R384 CN-R
subjects at $256^2$ and $1024^2$) in the a100-80G smoke vs the same subjects' ratios in the H200 timing job.

**A1.11 Enforceable budget (item 6).** The reservation counts each queued/running job's **sbatch time limit** (an
enforceable bound), not its projection: submit only if spent + Σ time limits of queued/running lane jobs + this job's
time limit ≤ 20 GPU-h (1 GPU per job). Time limit = certified projection × 1.5, rounded up to 0.25 h; the certificate
expands every executed call (A flanks, retries at the worst-case 2 per triplet, warm/scoring calls, burns, split and fused
calls, doubled-work, controls, certification, tight/tighter/Q2/Q3, compilation per distinct shape, setup). On completion
the ledger records `sacct` elapsed and releases the reservation.

**A1.12 P1e-3D scope (P1e row).** P1e-3D is descriptive and carries no qualification label: population = 923801 cases
0–5; rollout validity per A1.4 (finite, no damping saturation, using the fixed-sweep solver's own flags); distances on the
$63^3$ interior scoring lattice relative to $\lVert u_0\rVert$ there, max over the five evolved outputs; ρ of the tested
advection on each rule's own six output states against the next rule of the slice (lat8192→lat32768→gl56→gl64);
failing / deployed / finer judged against the same two bars (rollout $2.5\times10^{-4}$, ρ 0.116) and reported.

**A1.13 Q2 edge cases.** ρ with $\lVert N_{q^+}(c)\rVert=0$ is undefined and excluded (counted); a successor replay
that is invalid (A1.4) makes Q2 "unverified (successor invalid)" for that configuration.

**A1.14 Wording.** §2.3 table: "out" = in the frozen list, not in the nominal $256^2$ list; "in" = in the nominal list,
not in the frozen list. §12's claim of completeness is replaced by this amendment's items.

---

## Amendment A2 (2026-10-18, before any lane job; answers `audits/codex-design-2.md`, NEEDS-RESTATEMENT)

**A2.1 H-cost-r fit, executable (blocker 1).** Data per $r$: mean over the resampled fixtures of the fixture medians.
*Flat-cost rule first:* if $\max_r t/\min_r t<1.10$ over the panel's ranks (cost changes by < 10 % over a 16× range of $r$)
the exponent is **unidentified** → inconclusive ("cost flat in r"). Otherwise fit $t=a+e^{\beta}r^p$ with
`least_squares`, bounds $a\ge0$, $p\in[0,4]$, $\beta$ free, `x_scale='jac'`, from the **interior** start
$a_0=\tfrac12\min_rt$, and $(p_0,\beta_0)$ = slope and intercept of the least-squares line of $\log(t-a_0)$ on
$\log r$ (all $t-a_0\ge\tfrac12\min t>0$), with $p_0$ clipped to $[0.05,3.95]$. Failure = status ≤ 0, non-finite
parameters, or relative RMS residual > 0.2. Boundary (measured fits only): $p\le10^{-6}$, $p\ge4-10^{-6}$, or
$e^{\beta}(r_{\max}^p-r_{\min}^p)<0.01\,a$. Operation-count fits: same start and form, no boundary rule. Everything else
in A1.9 unchanged (bootstrap, pass/fail/inconclusive, literal AM-4 secondary label).

**A2.2 Component sum vs evolution: a reported diagnostic, not a gate (blocker 2).** No independently validated
acceptance interval exists for "separately dispatched components vs one fused loop" (launch overheads, the loop's
bookkeeping — post-solve projections, alternation diagnostics, statistics and output updates — and cross-block compiler
fusion all separate the two), so a mandatory band could reject a correct implementation. The A1.2 measurement is kept
and reported (model, evolution time, ratio, with the bookkeeping named as unaccounted), and **blocks nothing**. Deviation
from AM-5 (which listed it as a positive control blocking H-cost-r), recorded for the coordinator. Consequence for the
claim: the H-cost-r verdict is about **component-kernel scaling** at fixed $(M,m)$ and is labelled so; it is not a
statement about the solver loop's total time (which P1c measures directly).

**A2.3 P1c qualification is certified on the timing job's own trajectories (blocker 3).** The transfer gate of A1.5 is
replaced. In the timing job, per subject and dev6 case: the untimed scoring run (split query, no recording) gives the
hash every timed run must match; a **certified run** of the same configuration with internal-state recording
(`nrec` = 51) must reproduce it ($\max_k\lVert\Delta W_k\rVert/\lVert W_0\rVert\le10^{-12}$, else "certification not
transferable" for that subject) and is certified in-job (autodiff γ every step, G-cert); a tight and (all six dev6
cases) a tighter run, and a successor-rule replay plus own-state ρ, are run in the same job. The subject's **timing-job
numerical label** = Q1 (§5.1 as amended, on its six dev6 trajectories) and Q2 (rollout and ρ bars on the same six;
finer-rule convergence via $q^{++}$ on the same six). A P1c subject is called numerically qualified only if **both** its
timing-job label and the accuracy-job label of the identical configuration (population 38/16 cases) pass; the two are
reported separately. The H200-vs-accuracy-job distance is reported as a reproducibility number only.

**A2.4 Q2 indicator edge cases (blocker 4).** Per state: both $N_q(c)=0$ and $N_{q^+}(c)=0$ → excluded as trivial
(counted); $N_{q^+}(c)=0\ne N_q(c)$ → $\rho=+\infty$ (fails the bar); non-finite → fails. If no ρ indicator remains
after exclusions, the ρ bar is "not evaluable" and Q2 = "unverified (no ρ indicators)".

**A2.5 P1g statistics mapped exactly to the historical claims (blocker 5).** H1 (quadrature study): the per-case
three-mesh spread $\max_N e-\min_N e$ over $256^2/1024^2/4096^2$ of the worst first-order-ST error, per rule (gauss64,
gref = gauss640, fib6765), on dev6∪val32 — the historical statistic. H4 (jcp-mechanism): the per-case **two-mesh**
spread $|e(256^2)-e(1024^2)|$ for gauss96 (and gref), the historical pair; any three-mesh extension is reported
separately. Both under frozen and nominal lists, against the first-order reference (to set beside the historical
numbers) and E0. The historical-rule P1g arms are **descriptive**: they never receive a numerical-qualification,
role or frontier label. The claim "rule error is common to both arms" is withdrawn: changing the test space can change
the quadrature error, so the frozen–nominal difference is reported as the total effect of the list change under that
rule, not as a list-only effect.

**A2.6 Reservations (A1.11, completed).** A reservation is held from submission until `sacct` reports a terminal state
and the elapsed time is entered in the ledger; every attempt counts (requeues, failed starts, resubmissions on another
GPU type); raising a job's time limit requires a new reservation check before `scontrol update`.

---

## Amendment A3 (2026-10-18, before any lane job; answers `audits/codex-design-3.md`, NEEDS-RESTATEMENT)

**A3.1 H-cost-r fit and its synthetic control, frozen as code (blocker 1).** The fit minimises **relative residuals**
$(a+e^{\beta}r^p-t)/t$ (each rank weighted equally on a relative scale; times span three decades) — this weighting was
unstated before and is the cause of the round-3 discrepancy (an absolute-residual fit of the same fixture gives
$\hat p\approx2.017$). Data per rank: mean over the (resampled) fixtures of the fixture medians. The executable
specification is `p1stats.fit_power` / `p1stats.h_cost_r` (bootstrap: `default_rng(20261017)`, cases drawn with
replacement, each case carrying its 2 fixtures × all ranks) and the synthetic control is `test_p1.py::t_h_cost_r`
(`default_rng(0)`, one draw of shape (case 6, fixture 2, rank 6), $t=(a+br^2)(1+0.01z)$), committed with this amendment.
Verified before any lane job with 10,000 resamples: point $\hat p=1.9992$, interval [1.9963, 2.0023] → declared 3
**fail**, declared 2 **pass**; the rank-major draw order gives [1.9976, 2.0068], same outcomes (the control does not
depend on the axis order under the frozen weighting). No statistical rule was changed to obtain these outcomes.

**A3.2 P1c times the recording executable, so the certified trajectory is the timed one (blocker 2).** A2.3's separate
recording run is withdrawn. Every P1c evolution (A and B, scoring and timed) is the internal-state-recording program
(`nrec` = 51, the same program as the P1d production runs). The untimed scoring run of each (subject, case) is certified
in-job (autodiff γ every step, G-cert, validity per A1.4); every timed run must reproduce the scoring run's W **and its
full internal-state array and per-step exit reasons and iteration counts** (sha256 equality), so the certificate is a
certificate of the timed execution. Tight, tighter, successor and check replays and the own-state ρ use the scoring run's
own outputs. Both labels (timing-job dev6, accuracy-job population) must pass, as in A2.3. Cost of recording: the smoke
times the recording vs non-recording evolution (paired A–B–A, R128 and R384 at $1024^2$, dev6 cases 0–1) and reports
the overhead; all P1c timings are labelled "solver with internal-state recording (one row write per step)".

**A3.3 Scope carried to the coordinator.** H-cost-r is a statement about **component-kernel scaling** at fixed
$(M,m)$; it does not establish total solver-time scaling in $r$ (A2.2). Component arithmetic is tied to the solver by
numerical equivalence (the kernel `J`/`N` of `p1comp` equal the solver's `nlJ`/`nl` on the fixtures to $10^{-12}$
relative, checked in the smoke), not by a wall-time band.
