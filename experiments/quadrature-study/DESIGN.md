# DESIGN — quadrature-study: off-mesh quadrature for the tested advection of the full-scale Burgers 2D model

Pre-registered before any GPU job (2026-10-01). Lane `exp/2026-10-01-quadrature-study` (worktree
`worktrees/2026-10-01-quadrature-study`, forked from `exp/2026-10-01-ns3d-coordnet-bank` @ `21c175a1b`). Commits are
mirrored to `origin/codeonly/exp/2026-10-01-quadrature-study` by `sync_github.sh` after every commit; the lane branch
itself is never pushed. Cluster namespace `/cluster/tufts/paralab/tawal01/quad2d_20261001/`, one sub-directory per
attempt, never reused, deleted after a checksum-verified pull. A sibling lane runs the Burgers 3D version in another
worktree and namespace; nothing here touches it.

Status line convention: anything below marked *amendment* was added after this file was first committed, with its
date and the reason; nothing above an amendment is edited after a GPU number exists.

## 1. Question

Hari's study (`external/quadrature-study-2026-09-30/quadrature/`, read-only; `results/SUMMARY.md`) replaced the
data-fitted empirical quadrature (EQ) of the tested nonlinear term by fixed classical rules evaluated *off the mesh*,
by decoding the coordinate-network bank and its analytic gradient at the rule's points. On scaled-down models
($R=128$, $k=32$, CPU, meshes to $1024^2$) he found: tensor Gauss is the best 2D rule; the Fibonacci lattice
converges super-algebraically; scrambled Sobol/Halton converge like $1/m$; Smolyak sparse grids fail; off-mesh rules
reproduce the dense solve against a refined reference and are mesh-invariant; per-step cost is flat in $N$ and set by
the point count $m$.

This lane asks the same questions of **our full-scale frozen Burgers 2D model** — the $R=512$, $K=16$
coordinate-network bank and head behind the Table-1 Burgers 2D rows, at $256^2$–$4096^2$:

1. Do Gauss and lattice rules reproduce the dense solve against the refined reference at every mesh,
   mesh-invariantly?
2. How does their $\rho$ ladder compare with our deployed $63\times63$ mesh lattice and with the fitted EQ rule at
   matched $m$?
3. Is per-query cost flat in $N$, and how does it compare with the $63^2$ lattice at equal accuracy?
4. Does the narrow-early-bump worst case show up, as in Hari's study?

## 2. Frozen model and settings (nothing is trained or refitted here)

| object | file | sha256 |
|---|---|---|
| checkpoint `dn256b` ($K=16$, $R=512$) | `experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl` | `18f0266a…` |
| bank rotation (burgers-bank-knob) | `inputs/rotation_R512.npz` | `51149166…` |
| fitted EQ rule for the head (b-eqtop, $m=1024$ at $256^2$ nodes) | `inputs/rule_q0_m1024_qrg304_reachable.npz` | `6a32568f…` |
| solver modules (unchanged copies) | `vendor/{arms,hops,hfast,bkfast}.py` from `exp/2026-09-25-burgers2d-test` @ `e479a8932` | `vendor/PROVENANCE.json` |
| Hari's rules (unchanged copy) | `vendor/hari_quadrature.py` | `vendor/PROVENANCE.json` |

Three settings, as in the user's brief:

| setting | reduced state | tests $M$ | solver (the Table-1 / Table-2 text) |
|---|---|---|---|
| `acc` (accurate) | linear rung: span of the first $R'=384$ rotated bank columns, $384$ unknowns | 1536 | `bkfast.make_linear_query`: fused LM, Cholesky, clipped step, damping carried, quadratic predictor, closed-form Gauss-48 initial fit |
| `fast` | linear rung, $R'=128$ | 512 | same |
| `head` | the $k=16$ head, $q=0$, on the full rotated bank ($R'=512$ folded into the head, parity $\le 2.4\times10^{-13}$ in burgers-bank-knob) | 64 | `hfast.make_query`: fused LM, LU, clip, damping carried, quadratic predictor, LM initial fit |

Step budget 600, LM stopping tolerance $g_{\rm tol}=10^{-3}$ for every arm at every mesh (Table-1's value at
$\ge 1024^2$; one value everywhere so that mesh-to-mesh differences are not tolerance differences — *deviation from
Table 1 at $256^2$/$512^2$, where it used $10^{-2}$*; parity arms with $10^{-2}$ are run at $256^2$, §7).

## 3. The hybrid formulation (the user's choice; no continuum linear-term arm)

One backward-Euler step solves, for the coefficient vector $c$ of the rotated bank (linear rung: $c=w$; head:
$c=h(z)$),

$$ r(c) = S\Big(A c - p + \Delta t\,\big(N(c) + \nu\Lambda A c\big)\Big),\qquad S=(1+\Delta t\,\nu\Lambda)^{-1}, $$

with $A=\Phi^{\mathsf T}G'$ the exact discrete projection of the bank on the $M$ mesh-orthonormal sine tests
$\Phi_{(i,j),ab}=\tfrac{2}{L}\sin(a\pi x_i)\sin(b\pi y_j)$ and $\Lambda$ their discrete eigenvalues. The linear terms
are the exact discrete ones of the project's solver and are **not** changed. Only the tested advection $N(c)$ depends
on the arm:

| kind | $N(c)$ | cached online block |
|---|---|---|
| `dense` | $\Phi^{\mathsf T}a_h(G'c)$, the FOM's sign-upwind stencil at every mesh node | the mesh bank |
| `mesh` | $P_q^{\mathsf T}a_h(G_5c)$, the upwind stencil at the rule's $m$ nodes, $P_q=\Phi[\text{nodes}]\,w$ | $(m,5,R')$ |
| `point` (Hari's gradient form) | $L\sum_q w_q\,\psi_{ab}(x_q)\,u\,(u_x+u_y)(x_q)$ | $u$ and $u_x+u_y$ blocks, $(m,R')$ each |
| `flux` (Hari's integrated-by-parts form) | $-L\sum_q w_q\,(\partial_x\psi_{ab}+\partial_y\psi_{ab})(x_q)\,\tfrac12u(x_q)^2$ | $u$ block, $(m,R')$ |

Here $\psi_{ab}=2\sin(a\pi x)\sin(b\pi y)$ are the $L^2$-orthonormal continuum sines of the **same** $(a,b)$ as
$\Phi$, $\sum_q w_q=1$, and $u,u_x,u_y$ at $x_q$ come from the bank by forward-mode differentiation (partial decoding).
Since $\Phi^{\mathsf T}F\approx L\int\psi F$, the off-mesh forms target the continuum advection; the mesh forms target
the upwind stencil. Jacobians are analytic for `mesh`/`point`/`flux` and by chunked linearisation for `dense` (the
vendored text). Code: `qcore.py`.

## 4. References and cohorts

**Cohorts** (project definitions, asserted disjoint from the training draws `params_draw(0,128)` in every job):

| cohort | definition | role |
|---|---|---|
| dev6 | `params_draw(7090702,4)` ∪ `params_draw(911702,2)` | development, parity targets |
| val32 | `params_draw(20260927,32)` (burgers-heldout's `sel32`) | validation |
| test64 | `params_draw(20260916,64)` (hold64 = burgers2d-test's test64; physical sha256 `cd058fd2…` gated) | run once, at the end, everything frozen |

Rule sizes and the recommended rule (§8) are chosen on dev6 ∪ val32 only.

**Refined references** (`refjob.py`, Newton–BiCGStab of the project's FOM at $L_{\rm ref}=8192$, ntol $10^{-11}$,
ltol $10^{-9}$, accepted iff every step's relative residual $\le 2\times10^{-11}$ — hires-burgers' reference rule):

- **ST** (primary "refined reference"): $\Delta t=0.005/16$, refined in space and time (hires-burgers' definition).
- **S** (secondary): $\Delta t=0.005$, refined in space only. It isolates the spatial discretisation that separates the
  off-mesh rules (continuum advection) from the mesh rules (upwind stencil); under ST every arm also carries the
  backward-Euler time error of $\Delta t=0.005$, which is the same for every arm.

Refined-reference errors are measured on the $257\times257$ nodes shared by every evaluation mesh (stride $L/256$),
normalised by the initial field on the same nodes; the reference is saved there for every case (and at $1025^2$ for
two audit cases). Same-grid errors use the full mesh.

**Same-grid truth**: Newton–BiCGStab `lean_tight` (ntol $10^{-6}$, ltol $10^{-8}$) on the evaluation mesh, in the same
job.

## 5. Arms

**Rollout arms**, every setting, every mesh, every case of the job's cohorts:

| group | arms |
|---|---|
| mesh | `dense` (exact; see §6 for its case coverage), `lat64` (deployed $63\times63$ lattice, $m=3969$), `q0scaled` (head only: the fitted NNLS EQ rule, $m=1024$; no fitted rule exists for the linear rungs) |
| tensor Gauss (point) | $32^2$, $48^2$, $64^2$, $96^2$, $128^2$, $192^2$, $256^2$ |
| Fibonacci lattice (point, random shift seed 0) | 1597, 4181, 6765, 17711, 46368 |
| scrambled Sobol (point, seed 0) | 4096 (the $1/m$ reference) |
| flux form | Gauss $64^2$, Fibonacci 6765 |
| continuum rollout `gref` | point form with the converged continuum rule (§5.1) — the off-mesh analogue of `dense` |
| must-fail controls | Smolyak Clenshaw–Curtis level 8 (Hari's failing rule), Gauss $8^2$ ($m=64$, far too few) |

Gauss $96^2$–$256^2$ and Fibonacci 17711/46368 extend Hari's ladder because our test space is larger ($M=1536$ at
`acc`, test wave numbers up to about 44, against Hari's $M\le 320$) and our bank's bandwidth is larger (§5.1).

**$\rho$-ladder rules** (no rollout; tested advection on the state population of §6.2): `dense`; mesh lattices
`lat16`, `lat32`, `lat64`, `lat128`; `q0scaled` (head); Gauss $p^2$ for
$p\in\{8,16,24,32,48,64,80,96,128,160,200,256,320,400,512\}$; Fibonacci
$\{987,1597,2584,4181,6765,10946,17711,28657,46368,75025,121393\}$; Sobol $\{1024,4096,16384,65536\}$; Halton
4096; Smolyak CC levels $\{6,7,8,9\}$; flux form for Gauss $\{64,128,256\}$, Fibonacci $\{6765,17711\}$, Sobol 4096;
and `gref` in flux form (gate G6). Large rules are summed in point chunks (never an $(m,M)$ table above 32768 rows).
The rule lists and arm lists of every job are generated by `make_configs.py` (`configs/*.json`).

### 5.1 The continuum target

`gref` = tensor Gauss $640^2$ (409 600 points), point form; `gref_check` = Gauss $768^2$. Gate G6: on every
setting's population, $\rho_{\max}(\texttt{gref},\texttt{gref\_check})\le10^{-5}$ and
$\rho_{\max}(\texttt{gref}_{\rm flux},\texttt{gref})\le10^{-5}$.

Why so many points (calibration before this design was frozen; `calib_gauss.py`, local GB10, $256^2$ tests, the six
dev6 initial bumps fitted by Gauss-200 least squares — the narrowest states; `checks/calib-gauss-*-256.txt`): our bank
converges much more slowly than Hari's. Relative difference of the tested advection of Gauss $p^2$ from the finest
rule:

| $p$ | 64 | 128 | 200 | 256 | 320 | 400 | 512 |
|---|---|---|---|---|---|---|---|
| `acc` ($M=1536$), vs $640^2$ | 4.5e-2 | 1.2e-2 | 1.2e-3 | 3.1e-4 | 5.8e-5 | 1.8e-5 | 2.4e-6 |
| `fast` ($M=512$), vs $400^2$ | 1.9e-2 | 3.9e-3 | 4.3e-4 | 2.1e-4 | 2.8e-5 | — | — |

Point vs flux form at the finest rule: 1.9e-7 (`acc`, $640^2$), 1.1e-5 (`fast`, $400^2$). Hari's $R=128$ bank was
converged at Gauss $64^2$ (3.9e-6 against $300^2$). Our bank has 128 Fourier features of up to 11 cycles per unit
length (Hari's: 32 at scale 1.5) feeding two SiLU layers of width 1024, so its effective bandwidth is several times
larger; this is a property of the frozen model and is reported as a finding, not tuned away.

## 6. Metrics

### 6.1 Rollouts (per setting, arm, mesh, case; worst and median over cases reported)

- **Primary**: (a) worst relative $L^2$ error against the refined reference ST, $\max_{t\in\{.05,\dots,.25\}}
  \lVert u-u_{\rm ST}\rVert/\lVert u_0\rVert$ on the shared $257^2$ nodes; (b) the distance from our own dense
  rollout, $\max_t\lVert u-u_{\rm dense}\rVert/\lVert u_0\rVert$ (full mesh) — the hyper-reduction error of the mesh
  rules proper.
- **Secondary**: same-grid error against `lean_tight` (full mesh); error against reference S.
- Also: distance from the continuum rollout `gref` (the hyper-reduction error of the off-mesh rules proper); LM
  iterations (total, max per step), exit reasons, rejected steps; ms per query.
- `dense` case coverage: every case at $256^2$ and $1024^2$; at $4096^2$ dev6 in the development job and the first
  six test cases in the test job (its Jacobian is a full-mesh linearisation, ~minutes per query at $4096^2$).

### 6.2 $\rho$ (the paper's eq. 13) on held-out reached states

Population: the internal states $k=1,\dots,50$ reached by the deployed `lat64` arm on every case of the job (dev6 ∪
val32, or test64). $k=0$ (the initial fit) is excluded because backward Euler never evaluates the advection there.
For every $\rho$-ladder rule, against both targets: the **continuum target** (`gref`, point form) and the **mesh
target** (`dense`, upwind stencil at this mesh):

$$ \rho(c) = \frac{\lVert N_{\rm rule}(c) - N_{\rm target}(c)\rVert}{\lVert N_{\rm target}(c)\rVert}, $$

worst, p95 and median over the population, with the argmax state (case, $k$). Reference bars: 0.116 (the paper's
primary bar) and 0.06 (tight).

### 6.3 Cost

Timed in each job, randomised order, 0.1 s burn-in before every invocation, `block_until_ready`, median over 3
repetitions × the first 6 cases of the job's first cohort: every rollout arm except the controls (full query: dense
input field on the GPU → six dense output fields), the six-field **decode** alone per setting, and two full-order
settings (`lean_tight`, and `lean_nt3e-3_l3e-3_dt005` — Table 1's comparator). **Solve time** := query − decode.

## 7. Gates (a job's numbers are used only if all pass; failures are reported, never hidden)

| id | gate | bar |
|---|---|---|
| G1 | backend / x64 / precision | log has `jax_backend=gpu`; asserted in the job |
| G2 | cohort | test64 physical sha256 = `cd058fd2a297c202…`; dev6/val32 hashes recorded and equal across jobs |
| G3 | parity with the Table-1 / Table-2 records (dev6) | `lat64` reproduces the worst evolved same-grid error of the source job to $\le10^{-6}$ relative: $256^2$ ($g_{\rm tol}=10^{-2}$ parity arms) `acc` 0.16561743047161853 %, `fast` 1.5967685047599642 % (b256, 4241033); $1024^2$ `acc` 0.21081980982218235 %, `fast` 1.8280675823814023 % (b1024 parent, 4241031); $4096^2$ `acc` 0.2239874024132454 %, `fast` 1.8940229488900593 % (bk4096b, 4197473); head `q0scaled` at $1024^2$ 2.288356103816248 % (p1024, 4204019) |
| G4 | truth converged | every step's Newton residual $\le$ ntol |
| G5 | references accepted | §4 rule, every case |
| G6 | continuum target | `gref` vs `gref_check` and `gref` vs `gref_flux`: $\rho_{\max}$ on the population below the bar of §5.1 |
| G7 | controls must fail (checked at one real mesh before any verdict: $1024^2$, dev6 ∪ val32) | Smolyak-8 and Gauss-$8^2$ fail B3 **and** have continuum $\rho_{\max}>0.116$ |
| G8 | independent NumPy audit (`audit_qs.py`) | refined/same-grid/vs-dense errors recomputed from saved restricted fields of the audit cases; $\rho$ recomputed for sampled states with an independent NumPy bank + gradient (`npbank.py`, parity $10^{-13}$ to JAX) and NumPy tests; every median and ratio recomputed from raw invocations; injected controls (swapped case, $10^{-6}$-perturbed field, ×1.2 time) detected |

Local smoke `smoke.py` (before staging): S1 the `mesh` path equals `bkfast.make_linear_query` bitwise (done at
$128^2$: fields 0.0, identical iterations and exits); S3 point vs flux; S4 all-node lattice = dense ($5.6\times10^{-16}$
at $128^2$).

## 8. Pre-registered bars and the verdicts they feed

| bar | statement | used for |
|---|---|---|
| B1 reproduce dense | worst ST error of the arm within $\max(0.02\ \text{pp},\ 2\,\%)$ of `dense`'s worst ST error, on the cases where `dense` ran, at every mesh | (i) |
| B2 mesh invariance | for an off-mesh arm, $\max_L/\min_L$ of its worst ST error over $256^2,1024^2,4096^2$ (same cohort) $\le 1.02$ | (i) |
| B3 hyper-reduction | worst distance from the continuum rollout `gref` (off-mesh arms) or from `dense` (mesh arms) $\le$ 0.05 % (`acc`), 0.2 % (`fast`), 0.5 % (`head`) — about a quarter of each setting's dev6 model error (0.21 / 1.83 / 2.29 %) | (i), (iii) |
| B4 flat cost | solve-time ratio $4096^2/256^2\in[0.8,1.25]$ | (iii) |
| B5 $\rho$ | continuum $\rho_{\max}\le0.116$ on the population | (ii) |

**Recommended off-mesh rule per setting** (fixed on dev6 ∪ val32 before the test job, written here as an amendment):
the smallest-$m$ Gauss or Fibonacci rollout arm that passes B1 (where `dense` ran), B3 and B5 at all three meshes;
none if no arm passes. Question (iii)'s "equal accuracy" comparison is that rule against `lat64`.

**Question (iv)**: for each rule the argmax state of continuum $\rho$ is reported with its case's bump width
$w$, amplitude $a$, viscosity $\nu$ and step $k$; plus the Spearman correlation between a case's maximum $\rho$ and
$w$, and the share of the top-1 % states with $k\le5$.

## 9. Jobs (≈10 at most)

| attempt | what | cohorts | GPU |
|---|---|---|---|
| `refdv` | references ST + S at $8192^2$ | dev6, val32 | A100-80G |
| `reft` | references ST + S at $8192^2$ | test64 | A100-80G |
| `dv256`, `dv1024` | all arms, $\rho$, timing | dev6, val32 | A100-80G |
| `dv4096` | same (bank of 512 columns = 69 GB) | dev6, val32 | H200 |
| `t256`, `t1024`, `t4096` | frozen arms, once | test64 | as above |

The `dv*` jobs depend (`afterok`) on `refdv`, the `t*` jobs on `reft`. $512^2$/$2048^2$ only if the budget allows.
An infrastructure failure may be resubmitted once as a new attempt directory; a code bug found after submission is
fixed, committed and resubmitted; all recorded here and in the lab log.

## 10. Report

`reports/2026-10-01-burgers2d-offmesh-quadrature.md`, generated by `reports/make_report.py` from the audited
summaries; no hand-typed numbers; title + status line, LaTeX, mermaid, glossary (CLAUDE.md). Codex audits of this
file and the code before submission, and of the final report; kept in `results/`.
