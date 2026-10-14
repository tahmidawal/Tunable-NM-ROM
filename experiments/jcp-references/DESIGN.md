# jcp-references (E0) — order-verified second-order references for Burgers 2D and 3D (pre-registration)

Lane E0 of the JCP campaign (`reports/2026-10-06-jcp-offmesh-paper-plan.md` on `main`, E0 row, kill criterion K0,
"Sealing is global"; detail in its appendices `04-experiment-design.md` §1.1–1.2 and `06-codex-audit.md`). Status:
**pre-registration, 2026-10-13, before any code.** Amendments are appended at the end, dated; nothing above them is
rewritten.

## 1. Question

Every accuracy number of the campaign is measured against a **first-order** reference: the project's own full-order
model (sign-upwind advection, backward Euler (BE), Newton–BiCGStab) on a finer grid — $8192^2$ with
$\Delta t_0/16$ (ST) and $\Delta t_0$ (S) in 2D, 513 nodes per axis with $\Delta t_0/2$ in 3D. Such a reference is biased
toward upwind arms, and in 3D its own error is about 1.7–2.3 % (its distance to the 257-node solution and the post-hoc
Richardson estimate). The question is:

> **Are second-order references, verified for order per case and carrying a per-case uncertainty $\epsilon_{ref}$, good
> enough to rank arms, and how far were the old first-order references off?**

The lane produces, for the *same* PDEs, viscosities, initial conditions (ICs), output times and cohorts the campaign
uses: (i) a primary space+time reference $R_{ST}$; (ii) a space-only reference $R_S$ at the reduced model's BE step;
(iii) an independent pseudo-spectral cross-check; (iv) a store and loader that every other lane scores against;
(v) the old-versus-new comparison.

## 2. The PDEs (unchanged from the campaign; every function named below is reused, not re-derived)

Viscous Burgers on the unit box $\Omega=(0,1)^d$, homogeneous Dirichlet walls, $t\in[0,0.25]$, outputs at
$t_k = 0.05k$, $k=0,\dots,5$:

$$ u_t + u\,\textstyle\sum_{a=1}^{d}\partial_a u = \nu\,\Delta u,\qquad u|_{\partial\Omega}=0,\qquad u(\cdot,0)=u_0 . $$

- **2D** (`experiments/mr-burgers2d/engines.py`): descriptors $(c_x,c_y,w,a,\nu)$ from `engines.params_draw(seed, n)`
  ($c\sim U[.15,.85]$, $w\sim U[.05,.20]$, $a\sim U[.5,2]$, $\log\nu\sim U[\log .01,\log .1]$);
  $u_0 = a\exp(-|x-c|^2/2w^2)$ sampled at the nodes with the wall nodes set to zero (`engines.initial`). This IC does
  **not** vanish on the walls (it jumps there by up to $\approx 0.75a$), so the continuum problem has incompatible data;
  every discretisation here samples it the same way, and the order check (§4) tests whether second order survives it.
- **3D** (`vendor/quad3d/vendor/burgers3d-span/common.py`, `table`, `initial_interior`): 1–3 Gaussian blobs times the
  wall mask $64\prod_a x_a(1-x_a)$, peak-normalised to amplitude $A\sim U[.5,2]$ on the 257-node grid, $\log\nu\sim
  U[\log .01,\log .1]$. Compatible with the walls.
- Grids: $L$ intervals per axis, $h=1/L$, nodes $x=i/L$ (in 3D "$n$ nodes" means $L=n-1$).

## 3. The references

### 3.1 $R_{ST}$ — primary: second-order central differences + Crank–Nicolson

Space: the 5-/7-point Laplacian (as the FOM) and **central** advection
$N_h(u)_i = u_i\sum_a (u_{i+e_a}-u_{i-e_a})/(2h)$ (non-conservative form, the same continuum operator as the FOM's
$u\sum_a\partial_a u$). Time: Crank–Nicolson (CN) on $F_h(u)=N_h(u)-\nu\Delta_h u$,

$$ u^{n+1}-u^n+\tfrac{\Delta t}{2}\big(F_h(u^{n+1})+F_h(u^n)\big)=0, $$

with a **Rannacher start**: the first two CN steps (the interval $[0,2\Delta t]$) are replaced by four BE steps of
$\Delta t/2$ (damps the stiff modes that the 2D wall jump excites, which CN alone would carry undamped; the start adds
$O(\Delta t^2)$, so the scheme stays second order). Each step: Newton, matrix-free Jacobian by JVP, BiCGStab
preconditioned by the exact inverse of $(1+\tfrac{\Delta t}{2}\nu\Lambda_h)$ (resp. $1+\tfrac{\Delta t}{2}\nu\Lambda_h$ with
$\Delta t/2$ for the BE start) in the discrete sine basis (FFT DST-I, as the FOM). Predictor: the previous state.
Newton tolerance: relative nonlinear residual $\le 10^{-11}$ (relative to $\lVert u^n\rVert$), BiCGStab tolerance
$10^{-9}$, at most 20 Newton iterations; a step is accepted iff its final relative residual is finite and
$\le 2\times10^{-11}$ (the old 2D rule). f64 throughout, `JAX_DEFAULT_MATMUL_PRECISION=highest`.

Three nested levels with $\Delta t\propto h$ (both errors are second order, so the total error scales as $h^2$):

| dim | level $L$ (intervals) | $\Delta t$ | steps | CFL $2\Delta t/h$ ($\lvert u\rvert\le2$) |
|---|---|---|---|---|
| 2D | 2048, 4096, 8192 | $1.25\times10^{-3}$, $6.25\times10^{-4}$, $3.125\times10^{-4}$ | 200, 400, 800 | 5.12 |
| 3D | 128, 256, 512 (129/257/513 nodes) | $2.5\times10^{-3}$, $1.25\times10^{-3}$, $6.25\times10^{-4}$ | 100, 200, 400 | 0.64 |

(The CFL number is a property of the implicit solve's conditioning, not a stability limit; the old ST reference ran at
CFL 5.1.) **Refinement level** (only for cases that fail §4): 2D $L=12288$ with $\Delta t = 2.0833\dots\times10^{-4}
= 0.05/240$; 3D $L=768$ (769 nodes) with $\Delta t = 0.05/120$. Both contain the scoring lattice (§5).

### 3.2 $R_S$ — space-only, at the reduced model's step

Same central space discretisation and solver, but time is **BE with $\Delta t_0 = 0.005$** (the deployed reduced
model's step, 50 steps), no Rannacher start, levels as §3.1 (2D 2048/4096/8192, 3D 128/256/512) at fixed $\Delta t_0$.
It converges in $h$ to the semi-discrete-in-time BE solution and isolates spatial and quadrature effects from the BE
time error. Order is verified in $h$ exactly as for $R_{ST}$ (§4). Refinement levels as §3.1.

### 3.3 $R_{PS}$ — independent pseudo-spectral cross-check

A different family in both space and time: sine pseudo-spectral collocation with ETDRK4 (exponential time differencing,
fourth order; Cox–Matthews, coefficients by the Kassam–Trefethen contour integral, 32 points).
- State: orthonormal sine coefficients $\hat u_{\mathbf k}$, $1\le k_a\le N-1$. Linear part exact:
  $\hat L_{\mathbf k} = -\nu\pi^2|\mathbf k|^2$ (continuum eigenvalues).
- Nonlinear term $-u\sum_a\partial_a u$: $u$ (DST-I synthesis) and $\partial_a u$ (DCT-I along $a$, DST-I along the other
  axes) are evaluated on the **3/2-padded** grid $M=3N/2$, multiplied pointwise, and projected back by DST-I; modes
  $k_a\ge N$ are discarded (the 3/2 rule, equivalent to 2/3 truncation).
- IC: the **exact** sine coefficients of the continuous $u_0$ (not of its nodal samples), by tensor Gauss–Legendre
  quadrature of the separable factors (the 2D Gaussian and every 3D masked blob are products of 1D functions), 4096
  points per axis. Using nodal samples would put the FD-type $O(h^2)$ IC error into the spectral solver.
- Resolutions: 2D $N\in\{1024, 2048\}$ with $\Delta t = 5\times10^{-5}$; 3D $N\in\{128, 256\}$ with
  $\Delta t=2.5\times10^{-4}$ (explicit-advection limit for $|u|\le2$: $\Delta t\lesssim 2.8/(2d\pi N)$, i.e.
  $1.1\times10^{-4}$ in 2D at $N=2048$ and $5.8\times10^{-4}$ in 3D at $N=256$). A $\Delta t/2$ rerun of one case per
  dimension at the fine $N$ bounds the time error. Output: nodal values at the lattice (all lattice nodes are grid
  nodes for these $N$).
- Its expected spatial order is about 4 (the solution's sine coefficients decay algebraically because the PDE forces
  nonzero even derivatives at the walls), so its self-difference between $N$ and $2N$ estimates its own error.

## 4. Order verification, Richardson uncertainty and the refinement protocol

Per case, per output time $t_k$, $k\ge1$ ($t_0$ is the shared sampled IC, identical at every level), on the scoring
lattice (§5), with levels $u_1,u_2,u_3$ (coarse to fine, mesh widths $h_1>h_2>h_3$):

$$ d_{12}(t)=\lVert u_1-u_2\rVert,\qquad d_{23}(t)=\lVert u_2-u_3\rVert,\qquad n_0=\lVert u_3(t_0)\rVert, $$

and the observed order $p(t)$ solves $d_{12}/d_{23}=(h_1^{p}-h_2^{p})/(h_2^{p}-h_3^{p})$ (for ratio 2 this is
$p=\log_2(d_{12}/d_{23})$; solved by bisection on $[0.25, 8]$ for the non-uniform refinement level). Then

$$ \epsilon_{ref}(t)=\frac{d_{23}(t)}{n_0}\,\frac{h_3^{p}}{h_2^{p}-h_3^{p}}\;\Big(=\frac{d_{23}}{n_0\,(2^{p}-1)}\ \text{for ratio 2}\Big),\qquad
\epsilon_{ref}=\max_{k\ge1}\epsilon_{ref}(t_k), $$

the Richardson estimate of the error of the finest level in the campaign's metric (§5). The Richardson-extrapolated
field $u_R = u_3 + (u_3-u_2)\,h_3^{p}/(h_2^{p}-h_3^{p})$ is stored as a secondary field.

**Acceptance (per case, per reference kind).** A case is accepted iff (a) every step of every level met the Newton rule
(§3.1), (b) $p(t_k)\in[1.8,2.2]$ for **every** $k\ge1$, (c) the cell Péclet number $\max|u|\,h_1/\nu<2$ at the coarsest
level (max over stored output fields and the IC; the running maximum over steps is also recorded), and (d) $d_{23}/n_0
\ge 10^{-10}$ at every $k\ge1$ (above the solver-tolerance floor, so $p$ is meaningful).

**Refinement (never silent dropping).** A case that fails (b), (c) or (d) is rerun with the refinement level of §3.1
added and is re-judged on the finest three levels. A case that still fails is reported, by cohort and case, as a
**K0-affected case**: its reference is stored and flagged `accepted=false`, the loader returns the flag, and no accuracy
ranking may be made on it (plan K0: only $\rho$ and distance from dense there). There is **no fallback** to two
fourth-order levels. Failure (a) is a solver failure, not an order failure; it is rerun once with Newton budget 40 and
BiCGStab tolerance $10^{-10}$, then treated as (b).

**Uncertainty use.** $\epsilon_{ref}$ is a conditional estimate (it assumes the asymptotic regime the order check
tests), not a rigorous bound. Since any arm's error moves by at most $\epsilon_{ref}$ when the true solution replaces
$u_3$ (triangle inequality in the metric of §5), the difference of two arms' errors moves by at most $2\epsilon_{ref}$:
**$2\epsilon_{ref}$ is the resolution limit for ranking two arms** on that case.

## 5. Scoring lattice and metric

- Lattice: 2D the $257^2$ nodes $x=i/256$ (stored with the zero wall rows, shape $(6,257,257)$, the old `f257`
  convention); 3D the $63^3$ interior nodes $x=k/64$ (flat, $x$ slowest, shape $(6,63^3)$, the old `ref_<seed>.npz`
  `c<j>` convention). Each is a node subset of every compared mesh — 2D $256^2,512^2,1024^2,2048^2,4096^2$; 3D
  $64^3,128^3,256^3$ cells (and $384^3$) — and of every reference level above, so **no interpolation** is used
  anywhere. A mesh outside these sets (e.g. $128^2$, $48^3$) is scored on the common sub-lattice by restriction (the
  loader provides `restrict`), never by interpolation; that is outside this lane's pre-registered mesh set.
- Metric (the campaign's): $e(t_k)=\lVert f(t_k)-R(t_k)\rVert/\lVert R(t_0)\rVert$ on the lattice, and
  $e=\max_k e(t_k)$. $\epsilon_{ref}$ is in the same units.

## 6. The cross-check (K0, second half)

On 8 validation cases per dimension — the 4 with the smallest $\nu$ (hardest) and the 4 lowest-index remaining cases of
val32 (2D) / 923801 (3D):

$$ x_{PS}=\max_{k\ge1}\frac{\lVert u_{PS}^{(2N)}(t_k)-u_3(t_k)\rVert}{n_0},\qquad
s_{PS}=\max_{k\ge1}\frac{\lVert u_{PS}^{(2N)}(t_k)-u_{PS}^{(N)}(t_k)\rVert}{n_0}. $$

- **PASS** iff $x_{PS}\le 2\epsilon_{ref}$ and $s_{PS}\le\epsilon_{ref}/2$ (the cross-check is itself resolved).
- **FAIL** (K0 fires for that dimension) iff $x_{PS} > 2\epsilon_{ref} + s_{PS}$ with $s_{PS}\le\epsilon_{ref}/2$.
- **INCONCLUSIVE** iff $s_{PS}>\epsilon_{ref}/2$: rerun the pseudo-spectral pair at $(2N, 3N)$ (2D 3072, 3D 384), once.
- Also reported (not gated): the effectivity $\eta=x_{PS}/\epsilon_{ref}$ (≈1 if the Richardson estimate is sharp) and
  $\lVert u_{PS}-u_R\rVert/n_0$ (should be $\ll\epsilon_{ref}$).

## 7. Gates (code correctness), each checked at one real size before any verdict is read

| gate | what | bar |
|---|---|---|
| G1 parity | the lane's solver, switched to upwind advection + BE (no Rannacher), reproduces the vendor FOM (`engines.make_fom` 2D at $L=256$, `common.make_fom` 3D at 65 nodes) on one dev case | fields max rel. diff $\le10^{-12}$ |
| G2 MMS order | manufactured smooth solution with forcing (forcing path used only here), CN-R central at 3 levels (2D $L=128,256,512$; 3D $L=32,64,128$), true error vs exact | true order $\in[1.9,2.1]$ at every $t_k$; Richardson effectivity $\epsilon_{ref}/e_{true}\in[0.5,2]$ |
| G3 PS linear | pseudo-spectral with the nonlinear term off vs the exact sine-series heat solution | $\le10^{-12}$ relative |
| G4 PS MMS | pseudo-spectral solver on a manufactured solution with forcing | error decreases with $N$ at order $\ge3$ |
| G5 audit | independent NumPy recomputation (`audit_ref.py`) of $p$, $\epsilon_{ref}$, acceptance, cross-check numbers from the stored fields, with two fault injections (swapped case; one record perturbed by 1 %) that must be detected | exact agreement ($\le10^{-12}$); both faults detected |
| G6 loader | sha256 manifest check; fault injections: swapped case file and a 1 %-perturbed field must raise | both raise |
| G7 sealing | the loader refuses a sealed cohort without the coordinator's opening token; a mismatched token must also refuse | both refuse |

## 8. Controls that must fail (each checked on real data)

| control | construction | must |
|---|---|---|
| X1 first-order arm | the old scheme (upwind + BE, $\Delta t\propto h$) pushed through the same three-level order pipeline (2D dev6, levels 512/1024/2048; 3D dev4, 65/129/257) | be **rejected** by the $[1.8,2.2]$ gate on every case ($p\approx1$) |
| X2 cross-check discrimination | the cross-check of §6 applied to the **old** first-order reference instead of $u_3$ | **FAIL** on every cross-check case |
| X3 must-fire (plan) | $\lVert R_{old}-u_3\rVert/n_0$ vs $\epsilon_{ref}$ | exceed $\epsilon_{ref}$ on $\ge 90\,\%$ of cases; if not, the first-order bias was immaterial — recorded as a finding, not a failure |
| X4 Richardson must notice a broken level | a run with the middle level's $\Delta t$ doubled (time no longer refined with $h$) on one dev case | move $p$ outside $[1.8,2.2]$ at some $t_k$, or the case is recorded as an undetected-defect warning |

X4 is a sensitivity probe; it may legitimately pass if the time error is negligible at that level, and in that case the
report says so instead of claiming discrimination.

## 9. Cohorts and sealing

| cohort | 2D | 3D | use | stored |
|---|---|---|---|---|
| dev | dev6 (`7090702`×4 ∪ `911702`×2) | dev4 (`923651`×4) | smoke, gates, controls | local + cluster |
| validation | val32 (`20260927`×32) | `923801`×64 | selection by other lanes | local + cluster |
| historical test | test64 (`20260916`×64) | `923901`×32 | replication only | local + cluster |
| **sealed test** | sealed128 (new secret seed ×128) | sealed64 (new secret seed ×64) | headline numbers, opened once by the coordinator | **cluster only**, sealed dir |
| **OOD** | ood32 (new secret seed ×32) | ood16 (new secret seed ×16) | stress, reported separately | **cluster only**, sealed dir |

OOD distributions (pre-registered): as the training family but $\log\nu\sim U[\log .01, \log .01 + 0.1\log 10]$ (the
lowest 10 % of the log-range, $\nu\in[0.01, 0.01259]$) and widths below the training minimum: 2D $w\sim U[.030,.050)$,
3D every blob $w\sim U[.070,.100)$; all other draws as the family, consumed in the same order (prefix-stable).

**Sealing protocol.** Sealing is global, and this lane is generation and quality control only.
1. Secret seeds are drawn on the cluster login node by `secrets.randbits(31)` (one per new cohort) and written to
   `/cluster/tufts/paralab/tawal01/jcp-ref/sealed/SEEDS.json`; the file is made read-only and only its sha256 is
   committed (a commitment; the descriptors are therefore not readable from git).
2. Sealed jobs read the seeds from that file on the node; their fields go to the sealed store
   `/cluster/tufts/paralab/tawal01/jcp-ref/sealed/` (read-only after QC, `chmod a-w`), never to the local disk.
3. Only **QC aggregates** are pulled locally and reported (counts accepted, refined, K0-affected; distributions of
   $p$, $\epsilon_{ref}$, Péclet; Newton statistics) plus per-file sha256. No ROM, FOM or other arm is scored on a sealed
   or OOD case by this lane; no per-case field leaves the cluster.
4. The loader refuses sealed/OOD cohorts unless called with an opening token equal to the sha256 of a committed
   `selection.json` that the coordinator registers in `SEALED_OPENING.json` (an empty list today). Opening is the
   coordinator's/user's decision.

## 10. Metrics, bars and verdict

- Per case and kind ($R_{ST}$, $R_S$): $p(t_k)$, $\epsilon_{ref}$, $d_{12},d_{23}$, Péclet, Newton statistics,
  acceptance, refinement used.
- **E0 passes** for a dimension iff (i) every case of every cohort is accepted, possibly after refinement, and (ii) the
  cross-check PASSes on all 8 cases; otherwise K0 fires for the affected cases (reported by case, never dropped).
- Bar from the plan (for consumers): a claimed between-arm difference is resolved only if it exceeds
  $2\epsilon_{ref}$ on that case; the report states worst and median $\epsilon_{ref}$ per cohort so lanes can check
  "worst $\epsilon_{ref}\le 0.1\times$ smallest claimed difference".
- **Old vs new** (non-sealed cohorts): per case $\delta_{old}=\max_{k\ge1}\lVert R_{old}-u_3\rVert/n_0$ for 2D ST
  ($8192^2$, $\Delta t_0/16$, upwind) vs $R_{ST}$, 2D S ($8192^2$, $\Delta t_0$, upwind) vs $R_S$, 3D (513 nodes,
  $\Delta t_0/2$, upwind) vs $R_{ST}$. By the triangle inequality, $\delta_{old}$ bounds how far any provisional error
  number of that case can move when re-scored; the report gives its median and maximum per cohort and dimension.

## 11. Deliverables

- `refcore.py` (solvers, order/Richardson), `pscore.py` (pseudo-spectral), `refjob.py` (cluster job),
  `gates.py` (G1–G4, X1, X4), `audit_ref.py` (G5), `refstore.py` (**loader API**, G6–G7), `make_report.py` →
  `report.md`, plots in `plots/` (order histograms, $\epsilon_{ref}$ per cohort, cross-check, old-vs-new).
- Store layout (cluster primary `/cluster/tufts/paralab/tawal01/jcp-ref/store/`, local mirror of non-sealed cohorts in
  `experiments/jcp-references/store/`, git-ignored, sha256 manifest `store/MANIFEST.sha256` committed):
  `store/<dim>/<cohort>/<kind>/c<case>.npz` with keys `u1,u2,u3` (lattice fields of the three accepted levels, f64)
  and `meta` (JSON: levels, $\Delta t$, $p(t_k)$, $\epsilon_{ref}(t_k)$, acceptance, Péclet, job id, commit).
- Loader: `refstore.load(dim, cohort, case, kind='ST', field='finest'|'richardson')` → `(fields, meta)`;
  `refstore.score(field, dim, cohort, case, kind='ST')` → `{e, e_t, eps_ref, rankable, accepted}`;
  `refstore.restrict(fields, lattice)`. Verifies sha256 against the committed manifest on every load.

## 12. Compute estimate (GPU-hours), cap 4 concurrent jobs

Per-case costs scaled from the old references (2D $8192^2$ BE, 800 steps: 345–356 s mean on A100; 3D 513 nodes BE,
100 steps: 57–61 s on H100), assuming CN costs about the same per step as BE (to be measured in the smoke and amended):

| item | cases | per case | GPU-h |
|---|---|---|---|
| 2D $R_{ST}$ (8192 dominates; +1/8 +1/64) | 262 | ≈ 7.5 min | 33 |
| 2D $R_S$ | 262 | ≈ 2 min | 9 |
| 3D $R_{ST}$ (513, 400 steps) | 180 | ≈ 5.5 min | 17 |
| 3D $R_S$ | 180 | ≈ 1.5 min | 4.5 |
| pseudo-spectral (2×8 cases, 2 resolutions, + 2 $\Delta t/2$ runs) | 16 | 2D ≈ 0.3 h, 3D ≈ 0.6 h | 9 |
| gates, controls, smoke | — | — | 4 |
| refinement contingency (≈10 % of cases at the finer level) | — | — | 15 |
| **total** | | | **≈ 92** (plan budget 120) |

2D $8192^2$ fits an A100-80G (the old reference ran there); 3D 513 runs on H100/H200 (≈1.07 GB per state vector);
refinement levels (2D 12288, 3D 769 nodes) run on H200 with `--mem 240G`.

## 13. Flow

```mermaid
flowchart LR
  P[cohort descriptors<br/>dev, val, test, sealed, OOD]:::data --> ST[R_ST: central + CN-R<br/>3 nested levels]:::solve
  P --> S[R_S: central + BE at dt0<br/>3 nested levels]:::solve
  P --> PS[R_PS: sine pseudo-spectral + ETDRK4<br/>8 val cases per dim]:::solve
  ST --> O{order check per case<br/>p in 1.8..2.2, Peclet, residual}:::gate
  S --> O
  O -- fail --> RF[add refinement level, re-judge]:::solve --> O
  O -- still fails --> K0[K0-affected case: stored, flagged, never dropped]:::kill
  O -- pass --> E[eps_ref, Richardson field]:::data
  E --> X{cross-check: PS vs finest within 2 eps_ref}:::gate
  PS --> X
  E --> STORE[(read-only store + sha256 manifest<br/>loader refstore.py)]:::data
  STORE --> OLD[old vs new per case]:::data
  classDef data fill:#dbeafe,stroke:#1d4ed8;
  classDef solve fill:#dcfce7,stroke:#15803d;
  classDef gate fill:#fef3c7,stroke:#b45309;
  classDef kill fill:#fee2e2,stroke:#991b1b;
```

## 14. Process

Codex adversarial audits gate the design and each amendment, every code file before real-size runs, every cluster job
before submission, every result and the report (`audits/codex-<step>-<n>.md`). One job per cluster directory under
`/cluster/tufts/paralab/tawal01/jcp-ref/<job>/`, `squeue` before and after every submit, at most 4 concurrent jobs, at
most one local `jaxrun`. Commit small and often; `bash experiments/jcp-references/sync_github.sh` after every commit.

---

## Amendment A1 (2026-10-13, after Codex design audit 1, `audits/codex-design-1.md`; before any code)

Item numbers refer to the audit table. Every WRONG is repaired here; NEEDS-RESTATEMENT items are restated. Where A1
conflicts with §1–§14, A1 governs.

**A1.1 Common continuum IC and its representations (items 3, 26).** The continuum problem has IC $u_0$ (2D: the
Gaussian, nonzero on the walls; 3D: the masked blobs). Finite-difference (FD) levels and both FD references use the
nodal samples with zero wall nodes (`engines.initial`, `common.initial_interior`, with the single fixed 257-grid
$s_\star$ in 3D at every resolution). The pseudo-spectral (PS) solver uses numerically integrated sine coefficients of
the continuous $u_0$ (A1.6). The **canonical $t_0$ field** is the lattice restriction of the nodal samples; it defines
$n_0$ for every comparison, including PS. The PS $t_0$ field is stored separately and is never overwritten. All order,
$\epsilon_{ref}$ and cross-check quantities use $t_k$, $k\ge1$ only.

**A1.2 Amplitude, CFL and Péclet (items 5, 10, 19).** No a-priori bound $|u|\le2$ is assumed. Each solver records the
running maximum of $|u|$ over all steps (FD) or stages (PS). The CFL numbers of §3.1 are conditioning diagnostics, not
stability claims (summed over directions they are 10.24 in 2D, 1.92 in 3D). The Péclet check (c) of §4 uses the
running maximum. It is a sanity check that is not expected to fire on these families (nominal coarsest-level values
0.098 in 2D, 1.56 in 3D); its rejection path is tested by a unit test with an injected $\nu$.

**A1.3 Startup and preconditioner wording (items 7, 8).** The Rannacher start is used because it is the standard
remedy for CN with non-smooth data; second order at positive times is **verified per case**, not claimed from theory.
Each implicit stage solves $u - u^{n} + \theta\,\tau\,F_h(u) + (1-\theta)\,\tau\,F_h(u^n)=0$ with step length $\tau$ and
implicitness $\theta$: CN stages $\tau=\Delta t$, $\theta=\tfrac12$; startup BE stages $\tau=\Delta t/2$, $\theta=1$.
The preconditioner is the exact DST-basis inverse of $I+\theta\tau\nu\Lambda_h$, which is $I+\tfrac{\Delta t}{2}\nu
\Lambda_h$ in both cases.

**A1.4 Numerical floor (items 17, 18; WRONG → repaired).** The $10^{-10}$ floor is withdrawn. A **measured tolerance
floor** $\tau_{tol}$ replaces it: on dev cases (2D dev6 cases 0 and 4, 3D dev4 cases 0 and 2) every level of $R_{ST}$
and $R_S$ is re-solved at tight tolerances (Newton $10^{-13}$, BiCGStab $10^{-12}$, budget 40) and
$\tau_{tol}=\max\lVert u^{tight}-u^{prod}\rVert/n_0$ over cases, levels and $t_k$ (per dimension and kind). Condition
(d) of §4 becomes $d_{23}(t_k)/n_0\ge 10\,\tau_{tol}$ for every $k\ge1$. A case failing (d) is **not** refined in
space (that shrinks $d_{23}$); it is re-solved once at the tight tolerances and $\tau_{tol}$ is re-measured for that
case; if it still fails, it is classified **unresolved-floor** (its reference is extremely accurate, $\le d_{23}/n_0 +
2\tau_{tol}$, but its order is unverified) and treated as K0-affected (no ranking), reported separately from order
failures. Refinement (§4) is used only for failures of (b) or (c).

**A1.5 Order solve and failure outcomes (items 13, 16, 20).** $p$ is solved by bisection on $[0.25,8]$ only when the
ratio is bracketed; zero, NaN or infinite differences, or no bracket, give $p=$ `null` and the case fails (b) — never
clipped, never replaced by 2. Outcomes are distinct and all recorded: `accepted`; `order-fail` (b); `peclet-fail` (c);
`unresolved-floor` (A1.4); `solver-fail` (a, after the one retry of §4, which keeps the nonlinear target $10^{-11}$ and
changes only the Newton budget to 40 and the BiCGStab tolerance to $10^{-10}$). A rejected case after refinement is
"order unverified", not "inaccurate". Every attempt's level fields and metadata are kept (refinement history), not only
the final three levels.

**A1.6 PS spatial method restated (items 22, 23, 25, 27, 28, 29, 41, 42; item 23 WRONG → repaired).** The products are
**not** band-limited in the tensor-sine basis, so the 3/2 rule is *oversampling*, not exact de-aliasing: the DST of the
padded-grid product is a trapezoid-rule quadrature of the sine projection whose error is $O(M^{-4})$ for these
integrands (Codex's counterexample: $7.96\times10^{-5}$ at $M=12$, $4.91\times10^{-6}$ at $M=24$, ratio 16.2). The PS
solver is therefore a **high-order algebraic** method whose accuracy is measured, not assumed:
- Conventions (tested mode by mode in unit tests): orthonormal DST-I of size $M-1$; derivative of
  $\sin(k\pi x)$ is $k\pi\cos(k\pi x)$, synthesised by DCT-I on nodes $0..M$ with the standard endpoint halving.
- Lattice evaluation by direct synthesis of the sine series at the lattice coordinates (so $N$ need not contain the
  lattice). Resolutions: three per dimension, 2D $N\in\{1024,1536,2048\}$, 3D $N\in\{128,192,256\}$, and the observed
  order $q$ from them (A1.7).
- IC coefficients: tensor Gauss–Legendre, 4096 points per axis, **checked** against 8192 points (relative coefficient
  difference $\le10^{-13}$, gate G3b).
- ETDRK4 contour: 32 points, checked against 64 (coefficient difference $\le10^{-13}$, gate G3c).
- Time step as §3.3; every cross-check case is also run at the finest $N$ with $\Delta t/2$ (A1.7), and the running
  maximum of $|u|$ and finiteness of every stage are recorded.

**A1.7 Cross-check decision table (items 30, 33, 34, 35; WRONG → repaired).** Per cross-check case:
$x_{PS}=\max_{k\ge1}\lVert u_{PS}-u_3\rVert/n_0$ (finest $N$, $\Delta t$); spatial self-error
$s_x=\max_k\lVert u_{PS}^{(N_3)}-u_{PS}^{(N_2)}\rVert/n_0$, accepted as a conservative bound only if the observed
three-level PS order $q\ge2.5$ at every $t_k$ (for ratio $4/3$, $q\ge2.41$ makes $s_x\ge$ the Richardson error of
$N_3$); temporal self-error $s_t=\max_k\lVert u_{PS}^{(\Delta t)}-u_{PS}^{(\Delta t/2)}\rVert/n_0$; total
$\sigma_{PS}=s_x+s_t$ (IC and contour errors are gated to $\le10^{-13}$ separately). Then, exhaustively:

| condition | verdict |
|---|---|
| PS solver non-finite, or $q<2.5$ / undefined at some $t_k$ | **UNRESOLVED** |
| $\sigma_{PS}>\epsilon_{ref}/2$ | **UNRESOLVED** |
| $\sigma_{PS}\le\epsilon_{ref}/2$ and $x_{PS}\le2\epsilon_{ref}$ | **PASS** |
| $\sigma_{PS}\le\epsilon_{ref}/2$ and $x_{PS}>2\epsilon_{ref}$ | **FAIL** (the plan's K0: disagreement beyond $2\epsilon_{ref}$) |

An UNRESOLVED case is escalated once: the PS triple becomes $(N_2,N_3,\tfrac32N_3)$ (2D 3072, 3D 384) with the
fine-$N$ $\Delta t/2$ run repeated, ratio $N_3/N_4=2/3$, and the table is reapplied; a case still UNRESOLVED is
terminal UNRESOLVED (never promoted to PASS). **Dimension policy:** the cross-check validates the Richardson estimator
for the dimension. Any FAIL **suspends every reference of that dimension** (loader `dimension_status='suspended'`,
nothing rankable) until diagnosed in a dated amendment. With no FAIL, terminal-UNRESOLVED cases are K0-affected
individually, and the dimension is `validated` iff at least 6 of 8 cases PASS, else `unvalidated` (all its cases
non-rankable). Case selection (item 32): ties in $\nu$ broken by lower index; the selection is not a worst-case claim.

**A1.8 Vector consistency and separated refinements (item 15).** Reported for every case (not gated):
$c=\max_k\lVert(u_1-u_2)-\rho(u_2-u_3)\rVert/\lVert u_1-u_2\rVert$, $\rho=(h_1^{p}-h_2^{p})/(h_2^{p}-h_3^{p})$ at the
observed $p$ (small $c$ means the leading error is a common vector, as Richardson assumes). Diagnostic D1 on 2D dev6
cases 0, 4 and 3D dev4 cases 0, 2: a time-only ladder (finest $h$, $\Delta t$ of the three levels) and a space-only
ladder (finest $\Delta t$, three $h$) of $R_{ST}$, each with its observed order and magnitude, to show neither
component is cancelling the other.

**A1.9 Uncertainty wording and bars (items 21, 56, 57, 58).** $2\epsilon_{ref}$ is an **estimated** resolution
threshold (the triangle-inequality bound holds for the true reference error $\delta$; $\epsilon_{ref}$ estimates
$\delta$ under the asymptotic assumption the order check tests). Two bars are distinct and both are reported: (i) the
case-level ranking rule "no ranking inside $2\epsilon_{ref}$" (implemented by the loader's pairwise function); (ii) the
plan's design target "worst $\epsilon_{ref}\le0.1\times$ the smallest claimed difference" (checked by the consuming
lane). The old-vs-new $\delta_{old}$ bound holds for re-scoring the **same arm fields** on the same lattice with the
fixed-$t_0$ normalisation; the cohort maximum of $\delta_{old}$ bounds changes of cohort maxima/medians of paired
errors; the median of $\delta_{old}$ is descriptive only. Item 59: the old 3D "1.7–2.3 %" is restated as a 1.70 %
probe discrepancy (513 vs 257 nodes, case 923651/0) and a worst held-out Richardson displacement of 2.28 %, neither a
verified true error; the old 3D reference also used loose tolerances ($10^{-6}/10^{-7}$). $R_S$ (item 11) is the
BE-matched space-only reference; it is not time-matched to CN/BDF2 arms.

**A1.10 New 2D cohorts are row-wise, prefix-stable (item 50; WRONG → repaired).** Historical cohorts keep
`engines.params_draw(seed, count)` with their exact counts (it draws whole columns and is not prefix-stable). New 2D
cohorts (sealed128, ood32) use a row-wise generator: row $j$ draws from `np.random.default_rng([seed, j])`, in the
order $c_x,c_y,w,a,\log\nu$, with the family's (or OOD) ranges. New 3D cohorts use `b3d_common.draw_param_table`
(row-wise already) with the OOD ranges substituted for OOD. A unit test checks prefix stability and that every new
seed differs from every historical seed.

**A1.11 Sealing restated (items 51, 52, 53; WRONG → repaired as far as a shared account allows).** All lanes run as one
cluster user, so **no file permission is an access boundary**; this is stated plainly in the report. What is done:
- A 256-bit key is generated on the cluster login node (`openssl rand`) into
  `/cluster/tufts/paralab/tawal01/jcp-ref/custody/sealed.key` (mode 0400). The seeds file and **every sealed/OOD
  artifact** (fields and per-case metadata) are written AES-256 encrypted (`openssl enc -aes-256-cbc -pbkdf2`, piped
  from memory; no plaintext file is ever written). This protects against accidental reading (a lane globbing `*.npz`),
  not against a deliberate one. **Open question for the user:** whether to move the key off the cluster to a custodian
  (e.g. the user), which would make the encryption a real boundary.
- Seeds: drawn by `secrets.randbits(31)`, rejected and redrawn only if they collide with a historical or other new seed
  (redraws recorded); never replaced because of an outcome. Only the sha256 of the encrypted seed file and of the
  plaintext seed JSON is committed.
- QC aggregates pulled and reported for sealed/OOD cohorts follow a frozen schema: counts per outcome (A1.5),
  refinement count, quantiles (min/median/max) of $p$, $\epsilon_{ref}$, Péclet and Newton totals, GPU-seconds. No
  per-case values, no descriptors, no fields.
- **The loader cannot read sealed/OOD data at all.** Opening is a coordinator act under the plan's global-freeze rule:
  `open_sealed.py` (written by this lane, run only by the coordinator) refuses unless given a global freeze manifest
  that lists every lane's committed selection hash and this lane's reference-store version, writes an immutable
  opening record, and only then decrypts into the store. Post-opening method changes need a fresh cohort (plan).
  G7 tests the refusal paths (missing manifest, manifest without the store version, wrong key).

**A1.12 Store and loader restated (items 38, 54, 55; item 55 WRONG → repaired).** Per case and kind,
`c<case>.npz` holds every attempted level's lattice field (`u_L<L>`), and `meta` JSON holds: dimension, cohort,
case, descriptors and table/cohort hash, lattice id, output times, canonical $t_0$ field hash and $n_0$, levels and
$\Delta t$ used, solver configuration and tolerances, per-step Newton evidence (max relative residual, total and max
Newton count, step failures), running max $|u|$, Péclet, $p(t_k)$, $\epsilon_{ref}(t_k)$, $c$ (A1.8), outcome
(A1.5), refinement history, job id, GPU, commit, and the cross-check disposition (cross-check cases) and dimension
status. $u_R$ is **derived** by the loader, labelled diagnostic, with no uncertainty attached. API:
- `load(dim, cohort, case, kind='ST', field='finest'|'richardson')` → `(fields, meta)`; sha256-checked.
- `reference_status(dim, cohort, case, kind)` → `{accepted, outcome, eps_ref, dimension_status, eligible}`;
  `eligible` = accepted and dimension validated.
- `score(field, dim, cohort, case, kind)` → `{e, e_t, eps_ref, eligible}` (one arm; no ranking claim).
- `resolved(e_a, e_b, eps_ref)` → `|e_a-e_b| > 2 eps_ref` (pairwise ranking rule).
- `restrict(dim, cohort, case, kind, stride)` → reference levels restricted to a coarser sub-lattice **with $n_0$,
  $p$, $\epsilon_{ref}$ recomputed there**; arms must be restricted to the same sub-lattice (item 38).

**A1.13 Gates restated (items 39–44).**
- G1 parity: identical $\Delta t$ (0.005), tolerances (2D: ntol $10^{-10}$, ltol $10^{-11}$ passed at query time;
  3D: the same passed to the constructor), max Newton 20, nodal IC, outputs at $t_k$; bar $10^{-12}$ relative.
- G2 MMS (positive times only): $u^*=(1+\tfrac12\sin 4\pi t)\,\prod_a\sin(\pi x_a)\,\exp(\prod_a x_a)$, $\nu=0.05$;
  forcing $f=u^*_t+u^*\sum_a\partial_a u^*-\nu\Delta u^*$ computed by JAX automatic differentiation of the analytic
  $u^*$ (independent of the FD operators), applied as $\tfrac12(f^{n+1}+f^n)$ in CN stages and $f^{n+1}$ in BE stages.
  Levels 2D $L=128,256,512$ ($\Delta t=0.05/8,\ /16,\ /32$), 3D $L=32,64,128$ ($\Delta t=0.05/4,/8,/16$). Bars: true
  order in $[1.9,2.1]$ at every $t_k$, $k\ge1$; effectivity $\epsilon_{ref}/e_{true}\in[0.5,2]$ on the finest level.
  The forcing path is a separate argument, `None` in every production call.
- G3a PS linear: nonlinear term off, IC a finite set of sine modes; vs the exact decay at $t_k$, $k\ge1$, error
  $\le10^{-12}\lVert u(t_k)\rVert$. G3b/G3c: IC quadrature and contour checks (A1.6).
- G4 PS MMS: the G2 $u^*$ (not band-limited in sines) with forcing projected by the same oversampled DST; spatial
  ladder (three $N$ at $\Delta t$ small enough that $s_t$ is $<10^{-3}$ of the spatial error) with observed order
  $\ge3$, temporal ladder (three $\Delta t$ at large $N$) with observed order in $[3.5,4.5]$; and the projection unit
  test on Codex's $\sin(2\pi x)\sin^2(\pi y)$ example (observed quadrature order in $[3.5,4.5]$).
- G5 audit: `audit_ref.py` (NumPy, no JAX) recomputes $p$, $\epsilon_{ref}$, $c$, outcomes, cross-check numbers and
  $\delta_{old}$ from stored fields; recomputes the canonical $t_0$ field **from the descriptors** (independent code
  path) and compares it with the stored one (detects a swapped case even when field and metadata are swapped
  together); fault injection 1: cases 0 and 1 of 2D val32 swapped (fields+meta) must be detected by the $t_0$ check;
  fault injection 2: $u_{L=4096}(t_3)$ of 2D val32 case 0 multiplied by 1.01 must change the recomputed $p$ or
  $\epsilon_{ref}$ relative to the stored metadata by more than $10^{-6}$ (detected). Expected values are never
  regenerated after injection. Newton acceptance is audited from the stored per-step evidence.
- G6 loader: manifest pinned by commit; swapped files and a 1 %-perturbed field must raise.

**A1.14 Controls restated (items 45–48).**
- X1: upwind + BE with $\Delta t\propto h$: 2D dev6 at $L=512,1024,2048$ with $\Delta t=2.5,1.25,0.625\times10^{-3}$;
  3D dev4 at $L=64,128,256$ with $\Delta t=5,2.5,1.25\times10^{-3}$. Expected rejection on every case; a case that
  passes is reported as an accidental-order case (not hidden). X1b (deterministic): take accepted $R_{ST}$ levels of
  2D dev6 case 0 and add $C h_i\,\phi$ ($\phi$ the normalised $t_k$ field, $C$ such that the injected term is
  $10\times d_{23}$ at $h_3$); the gate must reject.
- X2: the old first-order reference in place of $u_3$, judged with the **new** $\epsilon_{ref}$ and the same PS
  resolution condition; expected FAIL on every cross-check case (a hypothesis; any non-FAIL is reported).
- X3: kinds $R_{ST}$ (2D ST, 3D) and $R_S$ (2D S) on the non-sealed cohorts only; failing the 90 % threshold is
  reported as "bias not resolved above $\epsilon_{ref}$", not "immaterial".
- X4 is a **sensitivity probe**, not a must-fail control: 2D dev6 case 0, $R_{ST}$, the middle level's $\Delta t$
  doubled.

**A1.15 Compute (items 62, 63).** Solvers stream: only the six output states (plus the current/previous state and
solver workspace) are held on the device; nothing per step is retained except scalar evidence. 3D factor
$1+1/16+1/256$. The PS row becomes 3 resolutions + one $\Delta t/2$ run per case, re-estimated after the smoke
(allocated 15 GPU-h; the 9 GPU-h row is withdrawn). Floor measurement (A1.4), D1 (A1.8), X1 and G-runs add ≈ 4 GPU-h.
All per-case costs are **calibrated in the real-size smoke** and the table is re-issued as an amendment before the
production jobs; production starts only if the calibrated total is $\le 120$ GPU-h (otherwise the user is asked).
`--mem` is host memory; device memory is measured (`peak_bytes_in_use`) in the smoke.

---

## Amendment A2 (2026-10-13, after Codex design audit 2, `audits/codex-design-2.md`; before any code)

Where A2 conflicts with A1 or §1–§14, A2 governs.

**A2.1 Numerical floor, case-specific (items 17, 18).** A1.4 is replaced. For every case, kind and level the solver
records the achieved final relative residual $r_n$ of every step. The **floor proxy** of a level is
$\tau_L = 10\sum_n r_n$ (residual accumulation with a stability factor of 10; it is a conditional proxy, not a bound).
The proxy is **validated** on the dev cases of A1.4 (every level re-solved at Newton $10^{-13}$, BiCGStab $10^{-12}$,
budget 40): the measured $\max_k\lVert u^{tight}-u^{prod}\rVert/n_0$ must be $\le\tau_L$ on every dev case and level;
if it is not, the factor 10 is raised to $10\times$ the worst observed ratio by a dated amendment before production.
Condition (d) of §4 becomes $d_{23}(t_k)/n_0\ge10(\tau_{L_2}+\tau_{L_3})$ for every $k\ge1$. A case failing (d) is
re-solved **at all three levels** at the tight tolerances (attempt `tight`), $\tau$ and every difference, order and
$\epsilon_{ref}$ are recomputed from the tight attempt, and (d) is re-judged with the tight proxies. A case that still
fails is `unresolved-floor`: order unverified, **no accuracy claim of any kind** (the bound "$\le d_{23}/n_0+2\tau$" of
A1.4 is withdrawn), K0-affected.

**A2.2 Direction consistency has a consequence (item 15).** $c$ (A1.8) stays a diagnostic for acceptance, but if
$c>0.5$ at some $t_k$, that case's uncertainty is replaced by the Richardson-free value
$\epsilon_{ref}^{cons}=\max_{k\ge1}d_{23}(t_k)/n_0$ (the last difference; larger than $\epsilon_{ref}$ whenever
$p\ge1$ and ratio 2), and the case is flagged `direction-inconsistent`. D1 is a diagnostic on four cases only, not
evidence about other cases.

**A2.3 PS uncertainty is a conditional indicator; the compared PS field is the half-step one (items 28, 30, A1-q).**
The compared field is $u_{PS}=u_{PS}^{(N_3,\Delta t/2)}$. $s_t=\max_k\lVert u^{(N_3,\Delta t)}-u^{(N_3,\Delta
t/2)}\rVert/n_0$ (under a fourth-order model the error of the half-step field is $s_t/15$; under any order $\ge1$ it is
$\le s_t$). $s_x$ (A1.7, spatial ladder at $\Delta t$) with $q\ge2.5$. $\sigma_{PS}=s_x+s_t$ is a **conditional
indicator** of the PS error under a single-leading-error model; a bias common to all PS resolutions is invisible to it.
What guards against common bias is that PS and FD share no discretisation (only the PDE, the descriptors and the
continuum IC).

**A2.4 Ordered cross-check procedure (items 33, 34, 29, PS vs $R_S$).** For each cross-check case, in this order:
1. **FD validity.** If the case's $R_{ST}$ outcome is not `accepted` (A1.5/A2.1) or $\epsilon_{ref}$ is undefined, the
   verdict is `FD-INVALID` (no replacement case; it counts as not-PASS).
2. **PS validity.** If any PS run is non-finite, the PS gates G3/G4 failed in that job, the spatial order $q$ is
   undefined (A1.5 policy: no bracket, zero, NaN) or $<2.5$ at some $t_k$, or $s_t$ fails to be at least 4× smaller
   than the $(\Delta t, 2\Delta t)$-free bound $\max_k\lVert u^{(N_3,\Delta t)}\rVert$ … (withdrawn; see next
   sentence) — the temporal check is: the running max $|u|$ must satisfy
   $\Delta t\le0.66\cdot2.8/(2d\pi N_3\max|u|)$; if not, the PS triple is rerun at $\Delta t/2$ once; still violated →
   `UNRESOLVED(stability)`. Any of the other failures → `UNRESOLVED(<reason>)`.
3. **Resolution.** $\sigma_{PS}>\epsilon_{ref}/2$ → `UNRESOLVED(resolution)`.
4. **Discrepancy.** $x_{PS}\le2\epsilon_{ref}$ → `PASS`; otherwise `FAIL`.
`UNRESOLVED` (any reason) is escalated once as A1.7; terminal `UNRESOLVED` and `FD-INVALID` are never PASS. The
§3.3 "explicit-advection limit" is withdrawn as a guarantee; it is the heuristic in step 2.
The cross-check compares PS with $R_{ST}$ only. $R_S$ (fixed-step BE) has a different target and gets no PS check; it
shares the spatial code with $R_{ST}$, so it inherits the dimension status.

**A2.5 Status truth table (A1-vs-§10, item 55).**
- *Dimension status:* `suspended` if any cross-check case FAILs; else `validated` if $\ge6$ of 8 PASS; else
  `unvalidated`.
- *Case eligibility* (`eligible`, per kind): outcome `accepted` **and** dimension `validated` **and** (the case is not
  a cross-check case, or its verdict is PASS).
- *E0 per dimension:* `PASS` iff the dimension is `validated` and every case of every cohort is `accepted`;
  `PASS-WITH-K0-CASES` iff `validated` and some cases are not eligible (listed by cohort and case); `FAIL` iff
  `suspended` or `unvalidated`. §10's "all 8 PASS" is replaced by this table. E0 overall = the pair of dimension
  verdicts (no single pooled verdict).

**A2.6 Coefficient checks and forcing at stages (items 25, 27).** IC coefficients: $\lVert\hat u^{4096}-\hat
u^{8192}\rVert_2\le10^{-13}\lVert\hat u^{8192}\rVert_2$ and $\max|\hat u^{4096}-\hat u^{8192}|\le10^{-15}$, at the
largest $N$ used (escalation included). ETDRK4 coefficients: for each of the four coefficient arrays,
$\max|C^{32}-C^{64}|\le10^{-12}\max|C^{64}|$; this is a coefficient check, not a trajectory bound. With forcing (G4
only) the forcing is evaluated at the stage times $t_n,\ t_n+\Delta t/2,\ t_n+\Delta t/2,\ t_n+\Delta t$.

**A2.7 Gates G2/G3a/G4 frozen (items 41, 42; item 42 WRONG → repaired).** The manufactured solution of A1.13 is
replaced (for G2 and G4) by the separable, wall-compatible
$u^*=A(t)\prod_a\varphi(x_a)$, $\varphi(x)=\sin(\pi x)\exp(\tfrac12\sin^2\pi x)$, $A(t)=1+\tfrac12\sin4\pi t$,
$\nu=0.05$: $\varphi$ is odd and 2-periodic and analytic, so $u^*$ has an infinite but spectrally decaying sine
expansion and satisfies every even-derivative wall condition. Each forcing term is a product of 1D functions
($u^*_t$; $u^*\partial_au^*=A^2\varphi\varphi'(x_a)\prod_{b\ne a}\varphi^2(x_b)$; $\Delta u^*$), with $\varphi',\varphi''$
by JAX automatic differentiation of $\varphi$. FD (G2) evaluates the forcing at the nodes. PS (G4) gets the forcing's
sine coefficients by tensor Gauss–Legendre (4096 points per axis) of the 1D factors — independent of the DST and free
of the wall-trapezoid error Codex demonstrated. G4 bars: spatial ladder ($N=32,48,64$ in 2D and 3D, $\Delta t=10^{-4}$)
observed order $\ge3$ at every $t_k$; temporal ladder ($N=64$, $\Delta t=0.01,0.005,0.0025$) order in $[3.5,4.5]$;
projection unit test on $\sin2\pi x\sin^2\pi y$ with order in $[3.5,4.5]$ (A1.6). G3a frozen: $\nu=0.01$, 2D modes
$(1,1),(3,2),(17,5),(40,40)$ with coefficients $1,0.5,-0.3,0.1$; 3D modes $(1,1,1),(2,3,1),(9,4,7),(30,30,30)$ with
$1,0.5,-0.2,0.05$; bar $\lVert u-u_{exact}\rVert/\lVert u_{exact}\rVert\le10^{-12}$ on the lattice at every
$t_k$, $k\ge1$, at $N=128$.

**A2.8 G5 identity and X1b (items 43, 45; WRONG → repaired).** G5 computes the expected descriptors from the
pinned (dimension, cohort, case-index) mapping of the committed cohort definitions (`cohorts.py`), never from the
stored metadata, and compares the canonical $t_0$ field so a swap of fields and metadata together is detected. X1b
becomes a synthetic exact first-order sequence $u_i=U+C\,h_i\,\phi$ on the stored $t_k$ fields of 2D dev6 case 0,
for both the dyadic triple and the refined triple $(3,1.5,1)$: $p=1$ exactly, so the gate must reject (a test of the
gate's code path, not of a solver).

**A2.9 Sealing classified (items 51–53, encryption, opening).** The sealing is **cooperative and unenforced** on the
shared cluster account: encryption prevents accidental reading only. The key file holds a 64-character hex line used
by `openssl enc -aes-256-cbc -pbkdf2 -md sha256 -pass file:<key>`; CBC is unauthenticated, so the sha256 of every
ciphertext **and** of every plaintext is recorded (the plaintext hashes only in the encrypted manifest; the ciphertext
hashes and the encrypted manifest's hash in git), and every decryption is verified against them. Failed generation
attempts are recorded in the QC aggregates (counts per outcome, including solver failures); quantiles exclude undefined
values and state the denominator; QC aggregates are published once per completed sealed job. Opening (coordinator
only, `open_sealed.py`): (1) refuse unless the global freeze manifest lists every lane of a trusted roster (the
plan's lane table, read from a commit given on the command line), each with a commit hash and a selection file whose
sha256 is verified by `git show`, plus this lane's store version; (2) refuse if an opening record already exists for
that cohort; (3) write an opening-intent record; (4) decrypt into a temporary directory and verify every plaintext
hash; (5) atomically rename into the store and append the manifest; (6) write the completion record. **Before a
verified global opening**, the loader cannot read sealed/OOD data; after it, the loader reads them like any cohort
and checks that a completion record exists.

**A2.10 Store schema and loader (items 38, 54, 55).**
- Fields are keyed by attempt and level, `a<k>_L<L>` (attempt table in `meta`: purpose `prod`/`tight`/`refine`/
  `retry`, tolerances), with per-step arrays `res_a<k>_L<L>` (final relative residual) and `nwt_a<k>_L<L>` (Newton
  count). `meta.final` names the attempt/levels used for the verdict. PS ladders of cross-check cases are stored in
  `store/<dim>/<cohort>/PS/c<case>.npz` (all $N$ and the half-step field at the lattice, plus PS $t_0$). Floor
  validation evidence (A2.1) is stored in `store/<dim>/dev*/floor/`.
- `score(...)` returns a dict carrying `(dim, cohort, case, kind, metric_id, store_version, e, e_t, eps_ref,
  eligible)`; `resolved(score_a, score_b)` raises unless both are eligible and agree on dim, cohort, case, kind,
  metric_id and store_version, and returns $|e_a-e_b|>2\epsilon_{ref}$ (using the case's $\epsilon_{ref}$ or
  $\epsilon_{ref}^{cons}$, A2.2).
- `restrict(dim, cohort, case, kind, factor)` restricts coordinate-wise (3D fields reshaped to $(6,63,63,63)$; the
  sub-lattice is $x=k\cdot factor/64$, resp. $/256$), recomputes $n_0$, $p$, $\epsilon_{ref}$, $c$ **and the outcome**
  on the sub-lattice, and, for cross-check cases, the cross-check verdict from the stored PS fields; the dimension
  status is unchanged.

**A2.11 Row-wise RNG pinning (item 50).** NumPy's `default_rng` (PCG64) with the seed sequence `[seed, j]`; the NumPy
version is recorded in every job's result JSON, and the unit test pins the first rows of each new cohort's **public
dev-check seed** (not the secret seeds) to golden values.

**A2.12 Erratum to A2.4 step 2 (same day, before audit).** Step 2 contains an abandoned half-sentence ("$s_t$ fails to
be at least 4× smaller … (withdrawn; see next sentence)"); it has no force. Step 2 reads, in full: the PS validity
check fails, with verdict `UNRESOLVED(<reason>)`, iff (i) any PS run is non-finite, (ii) a PS gate (G3a–c, G4) failed
in that job, (iii) the spatial order $q$ is undefined or $<2.5$ at some $t_k$, or (iv) the stability heuristic
$\Delta t\le0.66\cdot2.8/(2d\pi N_3\max|u|)$ with the recorded running maximum is violated after the one $\Delta t/2$
rerun it triggers (`UNRESOLVED(stability)`).

---

## Amendment A3 (2026-10-13, after Codex design audit 3, `audits/codex-design-3.md`; before any code)

Where A3 conflicts with earlier text, A3 governs.

**A3.1 Direction-inconsistent cases are ineligible (items 15, counterexample $u_3=U+100v$).** A2.2's fallback
$\epsilon_{ref}^{cons}$ is withdrawn for ranking. A case with $c>0.5$ at some $t_k$ gets outcome
`direction-inconsistent` (added to A1.5's list): stored, reported, **not eligible**, K0-affected. $d_{23}/n_0$ is still
reported for it, as a descriptive number only.

**A3.2 Floor proxy: assumptions, both differences, arithmetic floor (items 17, floor-gate counterexample).**
- Proxy per level: $\tau_L=10\sum_n\max(r_n,10^{-14})$, $r_n$ the achieved final relative residual of step $n$
  (relative to $\lVert u^n\rVert$ on the full solver grid); $10^{-14}$ is an arithmetic floor so a computed zero
  residual never implies zero numerical error. **Assumptions** (stated in the report): each step's residual perturbs
  the trajectory by at most about its own size (stability factor $\le10$ over the run), the perturbation is spread
  over the grid so the lattice-relative size matches the full-grid-relative size, and $\lVert u^n\rVert\le\lVert
  u^0\rVert$ (Burgers' $L^2$ energy is non-increasing with Dirichlet walls), so $\tau_L$ is in units of $n_0$. These
  are calibrated, not proven.
- Condition (d) of §4 becomes: $d_{12}(t_k)/n_0\ge10(\tau_{L_1}+\tau_{L_2})$ **and**
  $d_{23}(t_k)/n_0\ge10(\tau_{L_2}+\tau_{L_3})$ for every $k\ge1$.
- Calibration (A2.1) compares production vs tight (Newton $10^{-13}$) **and** tight vs tighter (Newton
  $3\times10^{-14}$, budget 60; the achieved residuals are recorded if the target is not reached) on the four
  calibration cases; each measured difference must be $\le$ the proxy of the less accurate solution. The scope of this
  calibration (four dev cases) is stated in the report; every case still uses its own residuals.

**A3.3 Restricted metrics are diagnostic only (items 38, restriction metric identity).** `restrict` returns the
recomputed numbers (A2.10) with `eligible=False` always; the dimension validation and the cross-check apply to the
canonical lattice only, and no ranking may be made on a restricted metric in this lane's release. Valid factors: 2D
$\{2,4,8,16,32,64,128\}$ on the 256-interval lattice; 3D $\{2,4,8,16,32\}$ on the 64-interval lattice.
`metric_id` is the string `"<dim>|lattice=<intervals>|times=0:0.05:0.25|norm=fixed_t0|max_k>=1"`.

**A3.4 G4 temporal ladder replaced (item 42; Codex's prototype: 3D non-finite at $\Delta t=0.01$ and $0.005$; 2D
exact-error order contaminated by the spatial floor).** The temporal order is measured from **self-differences** at
fixed $N=64$, which cancel the (identical) spatial error: $\Delta t\in\{2.5\times10^{-3},1.25\times10^{-3},
6.25\times10^{-4}\}$ in 2D and 3D (Codex's prototype completed 3D at $2.5\times10^{-3}$), three-level observed order
in $[3.5,4.5]$ at every $t_k$. The spatial ladder ($N=32,48,64$, $\Delta t=10^{-4}$, exact-error order $\ge3$) also
records a $\Delta t/2$ rerun at $N=64$ in both dimensions; the ladder is interpreted only if the half-step difference
is $\le10^{-3}\times$ the $N=64$ spatial error.

**A3.5 E0 statuses are mutually exclusive (status overlap).** Required case-kinds: every case of every cohort ×
$\{R_{ST},R_S\}$. Per dimension: `FAIL` iff the dimension is `suspended` or `unvalidated`; else `PASS` iff **every**
required case-kind is eligible; else `PASS-WITH-K0-CASES` (at least one required case-kind ineligible; listed).

**A3.6 PS run identity (item 54, escalation bookkeeping).** Every PS run has an immutable id
`N<N>_dt<k>_a<attempt>` with $\Delta t=\Delta t_{base}/2^k$ and its record: $N$, $\Delta t$, attempt, purpose
(`spatial`/`half`/`stability`/`escalate`), running max $|u|$ over stages, finiteness, the gate results of its job, and
its lattice field. The cross-check record names the selected spatial-triple run ids and the selected half-step run id
(after a stability rerun the triple is at $\Delta t_{base}/2$ and the half-step run at $\Delta t_{base}/4$).

**A3.7 Loader uncertainty source (`resolved`).** `resolved(score_a, score_b)` re-reads $\epsilon_{ref}$ from the store
record of the shared case (it must be finite and $\ge0$) and ignores any value passed in the dicts.

**A3.8 Opening is out of this release (items 52, opening transaction, post-opening loader; B-open).** This lane
generates, encrypts, hashes and quality-controls the sealed and OOD cohorts; it does **not** ship `open_sealed.py` in
this release, and the loader has no code path for sealed/OOD cohorts. The opening requirements (an authoritative
current roster committed by the coordinator with explicit exclusions; each listed selection's content and hash
verified; the store version; an exclusive lock; resumable or aborted intents; a completion record bound to the
published manifest hash and freeze-manifest hash; loader checks of all of these) are recorded here as the
specification for the coordinator's opening tool, to be built and audited when the coordinator decides to open.

---

## Amendment A4 (2026-10-13, during code development, after small local gate runs; before any real-size case)

Local GB10 runs of the gates (`jaxrun`, f64, highest precision; no production case had been run) gave:
G1 parity **bitwise identical** to both vendor FOMs (max relative difference 0, Newton counts equal, 2D $L=256$ and
3D 65 nodes); G2 3D pass (true order 2.03–2.05, effectivity 0.94–0.99); G3a pass ($\le10^{-15}$); G4 pass in both
dimensions (spatial order 3.96–4.00, temporal self-difference order 3.97–4.02, half-step ratio $\sim2\times10^{-6}$);
projection test order 4.02/4.00. Three gates failed, each for a reason that is not a solver defect:

**A4.1 G2 2D levels (pre-asymptotic).** With levels 128/256/512 the 2D true order at $t_1=0.05$ was 2.118 (bar
$\le2.1$). A diagnostic coupled ladder 128→1024 gave true orders at $t_1$ of 2.209, 2.118, 2.063 for successive pairs
(converging to 2 from above), and a space-only ladder (fixed fine $\Delta t$) gave 1.999 / 1.997: the excess is a
higher-order transient of the Rannacher start at coarse $\Delta t$, which disappears under refinement. G2 2D therefore
uses levels $L=256,512,1024$ with $\Delta t=0.05/16,0.05/32,0.05/64$ (bars unchanged). **Consequence noted for
production:** an early-time observed order slightly above 2 is a known pre-asymptotic signature of this scheme; the
production order gate $[1.8,2.2]$ is not changed, and a case that exceeds it is refined (§4), as pre-registered.

**A4.2 G3b / G3c bars were below the f64 round-off floor.** IC coefficients (2D dev6 case 0, $N=2048$): differences
between composite rules of 64/128/256/512/1024 panels were $2.1\times10^{-4}$, then $2.0, 1.6, 1.1\times10^{-13}$
(relative $\ell_2$; max absolute $3.1$–$3.4\times10^{-14}$ with $\max|\hat u|=0.49$): converged at 128 panels, then a
round-off plateau. ETDRK4 contour coefficients: 32 vs 64 points differ by $8\times10^{-12}$ relative and 64 vs 128 by
$3.8\times10^{-11}$ — the difference **grows** with more points, the signature of cancellation where the radius-1
contour passes close to the origin (worst at $h\hat L\approx-0.99$), not of an unconverged quadrature. New bars: G3b
relative $\ell_2\le10^{-12}$ and max absolute $\le10^{-13}$; G3c $\le10^{-10}$. Both are far below anything that can
affect a comparison at the $\epsilon_{ref}$ scale (expected $\gtrsim10^{-8}$).

**A4.3 Composite Gauss–Legendre (A1.6/A2.6/A2.7 wording).** "Tensor Gauss–Legendre, 4096 points per axis" is
implemented as a **composite** rule: 256 equal panels × 32 points (8192 points per axis), checked against 512 panels
(G3b); the same rule integrates the G4 forcing factors. A single 4096-point rule would need an $O(n^3)$ eigen-solve
and resolves the highest retained sine ($k=3071$) less safely than 32 points per panel of width 1/256.

**A4.4 Implementation details fixed here.** (i) FD state is flat ($x$ slowest), as in both vendor FOMs, so G1 can be
bitwise. (ii) The running maximum $|u|$ (Péclet, A1.2) is the maximum over all stages of the solver state. (iii) Tight
attempts use Newton $10^{-13}$ / BiCGStab $10^{-12}$ / budget 40 and the same per-step **acceptance** rule
$\le2\times10^{-11}$ (the tolerance target differs, the acceptance rule does not); their proxy uses their own
achieved residuals. (iv) Every job first runs the unit gates and G1 and stops (exit 4) if any fails; PS jobs also run
G3a–c and G4 for their dimension in the same job (A2.4 step 2). (v) `refjob.py` accepts a `smoke_override` config
block (reduced levels) **for local functional smoke tests only**; any result carrying `SMOKE_OVERRIDE` is never
collected into the store (collect refuses it) and never reported as a reference. (vi) The cross-check case list is
computed by `cohorts.crosscheck_cases` (2D val32: the 4 smallest $\nu$ by index tie-break, plus the 4 lowest-index
others; 3D val64 likewise). (vii) Store files are hard links of the pulled job outputs (no duplication on disk).

---

## Amendment A5 (2026-10-13, after Codex code audit 1, `audits/codex-code-1.md`; before any real-size run)

**A5.1 Solver failure is terminal after the one retry.** §4's "then treated as (b)" is withdrawn: after the retry of
A1.5 a level that still fails its steps gives the case the terminal outcome `solver-fail` (no refinement: a finer
level is harder to solve, not easier). Reported, never dropped.

**A5.2 Outcome precedence and the shared-IC check.** Outcomes, in precedence: `ic-mismatch` (a level's $t_0$ lattice
field differs from the canonical one by more than $10^{-13}$ relative — new; it blocks acceptance), `solver-fail`,
`unresolved-floor` (precedes `peclet-fail`, so a case whose floor fails after the tight attempt is terminal
`unresolved-floor`, as A2.1 says), `peclet-fail`, `order-fail`, `direction-inconsistent`, else `accepted`. The
procedure acts on the individual checks (floor → tight; order/Péclet → refine), not on the precedence label.

**A5.3 Order solve.** The bracket $[0.25,8]$ applies to the dyadic closed form too (orders outside it are undefined);
this removes the $p=0$ division by zero Codex reproduced.

**A5.4 Calibration evidence and production refusal.** The calibration job saves every field and per-step residual /
Newton array (base levels × prod/tight/tighter, D1, X1, X4) with a sha256, and writes `calib_status` (proxy_ok on
every row, worst ratios). The result is committed as `checks/calib_<d>d.json`; every `fd` job refuses to run unless
that file exists, passed, and records the FLOOR_STAB the job uses. The refinement levels are **not** calibrated
(cost: 2D $L=12288$ × three tolerances); they use the same case-specific proxy, a stated limitation.

**A5.5 Sealed jobs.** Silent: no per-case or per-attempt line reaches stdout (only gate results and "item k/n
written"). Never resumed (a sealed job directory with existing ciphertext refuses to run). Ciphertexts are written
atomically and read-only. The seed file is decrypted only after its committed ciphertext hash matches, and its
plaintext hash is verified after decryption. Seeds come from `make_seeds.py` (login node, once): `secrets.randbits(31)`,
redrawn only on collision with a historical, public-check or other new seed (redraw count recorded). The QC aggregate
is per kind, with every outcome counted (zeros included) and the denominators of the quantiles stated.

**A5.6 Resources and submission.** 3D `fd` jobs must be staged on an H200 with $\ge200$ GB host memory (the 769-node
refinement level); 2D on an 80 GB+ GPU. Every attempt records compile time (ahead-of-time compilation), solve time and
device `peak_bytes_in_use`; the smoke's measured costs re-issue §12 before production. Submission: local `flock`,
exclusive remote `mkdir`, the lane cap of 4 jobs, `squeue` before and after; the only extra sbatch flag accepted is
`--dependency=afterok:…`.

**A5.7 Pseudo-spectral attempts and per-run checks.** PS output files are keyed by ladder attempt
(`c<case>_a<k>`), and run ids by $(N,\Delta t,$ attempt$)$; the collector uses the latest ladder. Every run carries its
own IC-quadrature check (256 vs 512 panels at that $N$) and contour check (32 vs 64 points at that $N,\Delta t,\nu$),
plus the job's gate result; a selected run failing any of them makes the case `UNRESOLVED(gates)`. G3a now compares
with an exact solution synthesised by direct NumPy sums (independent of `pscore`) and fails on any non-finite value;
G4 gates both spatial pairs (32→48 and 48→64); a new unit gate checks the FFT-based sine/cosine synthesis, the
analysis, the 2D nonlinear term and the lattice synthesis against direct $O(n^2)$ sums. The G4 forcing closure captures
three small coefficient tensors (G4 sizes only); production PS captures none.

**A5.8 Collection and loader.** The collector links only files of complete, gate-passing, non-smoke jobs, each
verified against the job's `OUTPUTS.sha256`, refuses if any ciphertext is present locally, computes the dimension
status from all eight pre-registered verdicts with FAIL taking precedence over MISSING, applies X2 only when
$\sigma_{PS}\le\epsilon_{ref}/2$, reports X3 per kind with its coverage, and includes the sealed/OOD cohorts in the
E0 verdict through their QC counts (`INCOMPLETE` until they exist). `STATUS.json` records the store version and the
pre-registered cross-check case list; the loader refuses a status written for another store version, takes
cross-check membership from its own pinned list, marks the Richardson field as carrying no uncertainty, and computes
the restricted cross-check from the stored PS runs.

**A5.8b** For 3D new cohorts, A2.11's "`default_rng` with `[seed, j]`" applies to the **2D** generator only; the 3D
generator stays the vendor's sequential `draw_param_table` (A1.10), which is prefix-stable because every row consumes
the same number of draws.

**A5.9 G5–G7 timing.** G5 (`audit_ref.py`), G6 and G7 (loader fault injections and refusals, `test_refstore.py`) run
on the collected store, before any result is reported; the job-side gates are the pre-run gates (units, G1–G4).

---

## Amendment A6 (2026-10-13, after Codex code audit 2, `audits/codex-code-2.md`; before any real-size run)

**A6.1 Defects repaired (reproduced by the audit).** Loader: the STATUS cross-check list is compared as a set (a
lexical-vs-numeric sort made every status read fail); 3D `restrict` keeps the time axis of a one-frame IC. Gates: the
transform unit gate and every contour comparison fail on any non-finite value (they could pass on NaN).
Calibration: a row passes only if all three solves (prod, tight, tighter) met the per-step acceptance rule with a
finite proxy, as well as both proxy inequalities. `make_seeds.py` writes the seed ciphertext to a temporary file,
verifies it and links it atomically.

**A6.2 Evidence kept.** Every calibration-job attempt record (base levels, D1, X1, X4) is stored in the result
(`attempt_records`), X4 uses the recorded running maximum, and the calibration field file is linked into
`store/<dim>d/<dev cohort>/floor/` and listed in the store manifest after its hash is checked against both the job's
`OUTPUTS.sha256` and the job's own record.

**A6.3 Resume and collection.** An ordinary `fd` job resumes a case only if both its files match the hashes the same
job directory recorded when writing them (read before the job's first save); anything else is deleted and
recomputed. A sealed job refuses to start in any directory that already holds ciphertext. The collector verifies each
job's `result.json` against its `OUTPUTS.sha256`, creates `store/` if needed, and for sealed cohorts records the case
indices of each job (indices are not secret; descriptors are), refusing overlapping jobs and requiring the union to
be the whole cohort before E0 can leave `INCOMPLETE`. A cross-check case that is UNRESOLVED on its first ladder is
reported `UNRESOLVED-PENDING-ESCALATION` (not terminal; it counts as not-PASS) until its one escalation ladder
(`attempt: 1`, refused beyond 1) has been run.

**A6.4 G5 audit.** Agreement is $|a-b|\le10^{-12}\max(|a|,|b|)+10^{-15}$ (pure relative agreement failed on
round-off of quantities near $10^{-12}$); NaN semantics match production; the audit compares every stored per-time
number ($n_0$, $d_{12}$, $d_{23}$, $\epsilon_{ref}(t_k)$, $c$, Péclet, floor flag, the descriptor $\nu$), re-checks
acceptance of every attempt (superseded ones too), recomputes the cross-check numbers ($s_x,s_t,\sigma,x_{PS}$) and
validity flags from the run evidence, recomputes the old-vs-new distances from the old files, requires every
historical case × kind, measures the perturbation fault against the $10^{-6}$ bar of A1.13, and exits non-zero unless
everything passes. G6's stale-status test runs on an isolated copy of the metadata (`$JCPREF_META`), and G6/G7 test
3D restriction too.

**A6.5 Timing and memory are for budgeting, not benchmarking.** `solve_seconds` is the wall time of the compiled
execution including the host-to-device transfer of the IC (compilation is timed separately, ahead of time);
`peak_bytes` is the process-lifetime device high-water mark at the end of the attempt. Both serve the GPU-hour
re-estimate and the memory-fit check of §12; no timing claim about the references is made.

---

## Amendment A7 (2026-10-13, after the first cluster submission)

**A7.1 RNG golden comparison.** The first `calib2d` job (5209647) and the `gates` job refused to run (exit 4,
as designed) because the unit gate `Urng` failed on the cluster: its NumPy is 2.5.0, the local one 2.4.4. The raw
generator streams are identical (every family row, the 3D integer blob counts and all uniform-derived values match
exactly); one OOD viscosity, computed through `exp`, differs by $1.7\times10^{-18}$ (one ulp). The golden comparison
is now exact for integer arrays and relative $\le10^{-14}$ for floating values. The cohort descriptors are therefore
defined up to an ulp of `exp`/`log` across NumPy builds; every job records its NumPy version, and an ulp in $\nu$
changes a reference by far less than any tolerance here.

---

## Amendment A8 (2026-10-13, after Codex job audit 3, before any pseudo-spectral run)

**A8.1 PS base steps.** With the pre-registered steps ($5\times10^{-5}$ in 2D, $2.5\times10^{-4}$ in 3D) the stability
heuristic of A2.12 holds only for running maxima below about 1.44 (2D) and 1.53 (3D), so every cross-check case with
amplitude near the family maximum of 2 would predictably rerun its whole ladder at $\Delta t/2$ (Codex counted 2 of 8
cases in 2D and 4 of 8 in 3D from the initial amplitudes; the 3D job could not fit 12 h). The base steps become
$\Delta t=0.05/1600$ (2D) and $0.05/320$ (3D), for which the heuristic holds up to a running maximum of about 2.3
(2D) and 2.45 (3D). Everything else in A1.6/A2.3/A2.4 is unchanged (half-step run, one stability rerun, escalation).
Each dimension's eight cross-check cases are split over two jobs of four.

**A8.2** The stability check rejects any non-finite running maximum before taking the maximum (Python's `max`
could drop a NaN).

---

## Amendment A9 (2026-10-13, during the calibration jobs)

**A9.1 Tighter-solve budget.** In the first real calibration (job 5209713, 2D, $L=2048$, $R_{ST}$ case 0) the tighter
solve (Newton target $3\times10^{-14}$) cannot be reached in 2D: the residual stagnates at the round-off floor
(achieved $3.8\times10^{-14}$, tight $7.2\times10^{-14}$, prod $7.1\times10^{-12}$), and every step spent its whole
budget of 60 Newton iterations (12,120 iterations, 232 s against 6.5 s for the tight solve). At $L=8192$ this would
exceed the 12-hour job limit, so the job was cancelled after its first row. The tighter budget becomes **8** Newton
iterations per step (A3.2 already allows the target not to be reached; the achieved residuals are recorded and the
proxy uses them). The 3D calibration job (5209714) keeps budget 60: it reaches the tighter target in a few iterations
per step (306 and 586 Newton iterations in total at $L=128$ and $256$), so the budget never binds there.

---

## Amendment A10 (2026-10-13, after the first PS jobs, BEFORE any 2D finite-difference reference or any PS-vs-FD comparison)

**What was seen.** Only pseudo-spectral self-convergence data (jobs 5209854, 5209856; no FD reference of these
cases existed, so no $x_{PS}$ had been or could be computed). 3D: spatial order $q\approx5.6$–$5.9$, spatial
self-difference $s_x$ between $1.4\times10^{-10}$ and $9.4\times10^{-9}$. 2D val32 cases 0 and 2 (Gaussians that do not
vanish on the walls; the PS initial field differs from the nodal one by $\sim2\times10^{-3}$ there): a stable **first**
order, $q=1.01$–$1.02$ at every $t_k$, with $s_x=2.7\times10^{-9}$ and $8.3\times10^{-10}$. 2D case 1: $s_x=1.4\times
10^{-15}$, at round-off, so $q$ is undefined at $t_1$. Case 3: $q\approx4$.

**Why the rule changes.** A1.7/A2.3 required $q\ge2.5$ so that $s_x$ (the raw last difference) would be a
conservative indicator for the ratio-4/3 ladder. That requirement turns accurate PS solutions into `UNRESOLVED`
whenever the PS converges slowly but with a tiny constant, or is already at round-off — it tests the *form* of the PS
convergence rather than its *size*. The purpose (a conservative, conditional PS error indicator) is served directly by
the Richardson multiplier at the observed order.

**A10.1 New PS spatial indicator.** Per $t_k$, with the selected triple $N_1<N_2<N_3$ and $d_{12}^{PS},d_{23}^{PS}$:
- if $d_{23}^{PS}(t_k)/n_0\le10^{-12}$ (round-off level), the spatial indicator at $t_k$ is $10^{-12}$ and no order is
  required at that time;
- otherwise the observed order $q(t_k)$ must be defined (bracket $[0.25,8]$, A1.5 policy) and $q(t_k)\ge0.5$, and the
  indicator is $d_{23}^{PS}/n_0\cdot\max\!\big(1,\ N_3^{-q}/(N_2^{-q}-N_3^{-q})\big)$ (the Richardson estimate of the
  error of $N_3$, never smaller than the raw difference); else `UNRESOLVED(spatial-order)`.
$s_x=\max_k$ of the indicator. Everything else in A2.3/A2.4/A3.6 is unchanged: $\sigma_{PS}=s_x+s_t$ must be
$\le\epsilon_{ref}/2$ (resolution), then PASS iff $x_{PS}\le2\epsilon_{ref}$. With $q=1$ and ratio $4/3$ the multiplier
is 3, so the 2D first-order cases above carry indicators of order $10^{-8}$; the resolution test against
$\epsilon_{ref}$ still decides whether they can count.

**A10.2 Disclosure.** This amendment was written after seeing PS self-convergence data and is reported as such. It was
written before any 2D FD reference existed and before any $x_{PS}$ was computed in either dimension, so no PS-vs-FD
outcome could inform it. The report lists, for every cross-check case, the verdict under A10.1 and the verdict the old
rule ($q\ge2.5$, raw $s_x$) would have given.

**A10.3 (same day, after Codex audit `audits/codex-design-5.md`).** (i) *Chronology, narrowed:* A10 was written after
inspecting PS self-convergence and before any comparison with the **new second-order** FD references of the
cross-check cases (none existed). Older first-order references of these cases exist in sibling worktrees; this lane
made no comparison of PS with them before A10 (repository history records none; it cannot prove a negative).
(ii) *Floor, no abrupt drop:* at a time with $d_{23}^{PS}/n_0\le10^{-12}$ the indicator is
$\max\big(10^{-12},\ d_{23}^{PS}/n_0\cdot m\big)$ with $m$ the Richardson multiplier if $q\ge0.5$ is defined there,
else $m=10$ (above the largest multiplier the $4/3$ and $3/2$ ladders admit, 6.46 and 4.45). The floor rests on an
explicit assumption — that any remaining PS spatial error is negligible at the $10^{-12}$ scale — which small
self-differences alone do not prove. (iii) *Validity from the arrays:* the cross-check rejects any non-finite PS or FD
array directly instead of trusting the stored `finite` flag, and rejects a non-finite $\epsilon_{ref}$. (iv) *Disclosure:*
every cross-check record carries `old_rule_verdict`, the verdict of the pre-A10 rule ($q\ge2.5$ at every time, raw
$s_x$) **on the same selected ladder** (not the full old procedure, which could have required an escalation), computed
on every path; the audit re-derives it and the report prints it beside the A10 verdict. (v) *Meaning of a PASS:*
conditional agreement on the scoring lattice at $t_1..t_5$; it does not validate the PS initial representation, early
times, or the absence of a bias shared by all PS resolutions, and it does not make $\epsilon_{ref}$ a rigorous bound.

---

## Amendment A11 (2026-10-14, while the 2D calibration job 5209895 runs)

**A11.1 Calibration cost.** At $L=8192$ the round-off floor of the relative residual exceeds the tight target
$10^{-13}$ (it scales with $\Delta t\,\nu/h^2$; achieved $8.2\times10^{-14}$ already at $L=4096$), so the tight solve
spends its 40-iteration budget every step: the first $L=8192$ calibration row took 3.5 h (the production solve 282 s).
The row passed with a wide margin (production-vs-tight change $3.1\times10^{-14}$ against a proxy of
$1.2\times10^{-9}$). The remaining rows need about 6 more hours. Production tight attempts (taken only after a floor
failure) would cost the same; none is expected given these margins, and any that occurs is recorded.

**A11.2 Provisional start, held release.** To use idle capacity, 2D production may start under a **provisional**
calibration status (`checks/calib_2d.json` with `provisional: true`, written from the completed rows, all passing).
Every case produced under it carries `calib_status: provisional`. The collector refuses to link such jobs into the
store until `checks/calib_2d.json` is replaced by the final status of the completed calibration job and that status
passes. If the final calibration fails, the floor check of every held case is re-judged from its stored per-step
residuals with the recalibrated factor (no rerun is needed for that), as A2.1 prescribes, and the result is reported.
