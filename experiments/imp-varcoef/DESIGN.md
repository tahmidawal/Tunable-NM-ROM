# imp-varcoef (E3) — variable-coefficient viscous Burgers 2D: FOM-only benchmark and the 2× gate

**Status:** pre-registration, written 2026-10-21 before any code or job. Lane `exp/2026-10-21-imp-varcoef`, namespace
`/cluster/tufts/paralab/tawal01/imp-varcoef/`. Binding rules: `reports/2026-10-21-improvement-campaign/campaign/briefs/COMMON-BRIEF.md`
and `.../briefs/imp-varcoef.md` (lane brief wins on conflict). Amendments are appended at the end, dated; this text is never rewritten.

## 1. Question

On the box, a coarse sine pseudo-spectral FOM beats the bank ROM (plan K2) partly because it integrates the
constant-coefficient Laplacian exactly. A high-contrast variable diffusion coefficient $\kappa(x)$ removes that exact
integration. **Does the fastest tuned, accuracy-matched full-order solver for variable-coefficient Burgers become at
least 2× slower than our deployed bank ROM's box cost?** Only if it does is a ROM trained (Phase 2).

## 2. Problem

$$u_t + u\,(u_x + u_y) = \nabla\cdot\big(\kappa(x;\mu)\,\nabla u\big),\qquad x\in(0,1)^2,\quad u=0 \text{ on the walls},\quad t\in[0,0.25],$$

outputs at $t_k = 0.05k$, $k=0,\dots,5$ (the campaign's six fields). Initial condition: the campaign Gaussian family
$u_0 = a\,e^{-|x-c|^2/(2w^2)}$ (walls zeroed), $c_x,c_y\sim U[0.15,0.85]$, $w\sim U[0.05,0.20]$, $a\sim U[0.5,2]$
(`experiments/mr-burgers2d/engines.py`, `initial`).

**Coefficient family.** $\kappa(x;\mu) = \nu\,\exp\big(s\,\hat g(x)\big)$ with $s=\ln C$, $C$ the contrast, and

$$g(x,y)=\sum_{\substack{0\le j,k\le 3\\ (j,k)\ne(0,0)}}\frac{\xi_{jk}}{1+j^2+k^2}\cos(j\pi x)\cos(k\pi y),\qquad \xi_{jk}\sim\mathcal N(0,1),$$

$\hat g = (g-\min g)/(\max g-\min g) - \tfrac12$ with the extrema taken on the $1025^2$ nodal grid, so that
$\kappa_{\max}/\kappa_{\min}=C$ on that grid and the geometric mean of the extremes is $\nu$ (the convention of
`where-we-win/analysis/varcoef_pcg.py`). The realised continuous contrast (checked on $4097^2$) is recorded per case.
$\nu = \exp(U[\ln 0.02,\ln 0.1])$ per case. **The lower end is raised from the campaign's 0.01 to 0.02** so that
$\kappa_{\min}\ge 0.002$ at $C=100$: this keeps the reference's cell Péclet number $|u|h/\kappa$ below 2 at $L=1024$
(with $|u|\le 2$) and keeps the minimum diffusivity within a factor 5 of the campaign's stiffest training corner,
so the test probes the exact-diffusion mechanism rather than the already-documented steep-front failure.
Contrasts: **$C=100$ (primary, gate)** and $C=10$ (secondary); $C=1$ ($\kappa\equiv\nu$) only as a code-identity control.
The same physical cases (descriptors $c,w,a,\nu,\xi$) are used at every contrast (paired design).

## 3. Cohorts (split by seed before any run)

Row-wise generation, row $j$ from `numpy.random.default_rng([seed, j])`, drawing in order $c_x,c_y,w,a,\ln\nu,\xi$ (15 values).

| split | seed | cases | use |
|---|---|---|---|
| dev | 2026102101 | 8 | selection (ladder) and the FOM timing workload |
| val | 2026102102 | 24 | selection (pooled with dev, as dev6+val32 are pooled for the ROM targets) |
| test | 2026102103 | 16 | generated, **never run in Phase 1**; reserved for Phase 2 test-once |

No historical, sealed or reserved-OOD cohort is touched. The E0 store is not used for these cases (new problem); it is
used only through the inherited ROM targets (constant-viscosity cohorts dev6 + val32).

## 4. Reference (E0 recipe, levels shifted one step coarser)

The E0 recipe (`jcp-references/refcore.py`, `refjob.py`, `verdict.py`) with the diffusion operator replaced by the
conservative 5-point flux form $\nabla_h\cdot(\kappa_f\nabla_h u)$, face coefficients $\kappa$ evaluated at face midpoints,
and the DST preconditioner built on $\bar\kappa=\nu$:
central advection, Crank–Nicolson with Rannacher start (four BE half steps), Newton + DST-preconditioned BiCGStab,
prod tolerances $(10^{-11},10^{-9},20)$. Levels $(L,\Delta t) = (1024, 2.5\cdot10^{-3}), (2048, 1.25\cdot10^{-3}), (4096, 6.25\cdot10^{-4})$
(ratio 2, $\Delta t\propto h$). The cost of the E0 levels ($2048$–$8192$) is not affordable within the 8 GPU-h phase cap.
Richardson as E0: $p=\log_2(d_{12}/d_{23})$ per $t_k$; $\epsilon_{\rm ref}=\max_k d_{23}/(n_0(2^p-1))$; reference field = finest
level restricted to the $257^2$ lattice. **Acceptance per case:** $p\in[1.8,2.2]$ at every $t_k$, cell Péclet $<2$ at the
coarsest level, every step converged, and a tolerance-floor check on dev cases 0–1 at $C=100$ (rerun of level 1 at the
tight tolerances $(10^{-13},10^{-12},40)$, distance $\le 0.1\,\epsilon_{\rm ref}$). **Eligibility:** accepted and
$\epsilon_{\rm ref}\le 10^{-3}$ (one tenth of the tightest target). $\epsilon_{\rm ref}$ is a **conditional** Richardson
estimate and is labelled so. **Fallback (pre-registered):** if fewer than 75 % of the 32 cases at a contrast are eligible,
that contrast's reference is rerun at the E0 levels $(2048,4096,8192)$ on those cases (an amendment records it).

Scoring (as E0): $e=\max_{k\ge1}\lVert u(t_k)-R(t_k)\rVert_2/\lVert R(t_0)\rVert_2$ on the $257^2$ lattice (walls included);
median and worst over eligible cases. A case on which two arms differ by less than $2\epsilon_{\rm ref}$ is a tie.

## 5. Our box cost (inherited, re-measured in the timing job)

`rom_box.py` → `checks/rom-box.json`: from the qualified 2D front at $4096^2$ (`jcp-fom2d/inputs/FRONT-2D.json`,
primary LSPG members, errors conditional E0 on dev6 + val32) and the C2 paired timing (H200, job 5456288), the cheapest
member with worst error $\le\tau$:

- $\tau=1\%$: **no qualified ROM** reaches it (best worst error 1.77 %); the gate is undefined there and FOM costs are reported only.
- $\tau=3\%$: R256 gauss128 LSPG-CN-R $\Delta t=0.0125$ (worst 2.71 %), 32.4 ms.
- $\tau=10\%$: R128 fib4181 LSPG-BE $\Delta t=0.0125$ (worst 6.73 %), 14.7 ms.

(These values are generated by `rom_box.py`; the report reads them from the JSON.) The ROM's cost and accuracy are
those of the **constant-viscosity** family: no variable-coefficient ROM exists before Phase 2, and assuming its accuracy
carries over is optimistic for the ROM (the gate can only be too permissive, never too strict, on that account).
The reduced variable-coefficient diffusion operator would be assembled once per query (about $2mMr$ flops), so its
per-step cost is unchanged; this is an assumption, stated next to every gate number.

**Re-measurement in the same job (primary timing).** The timing job (§8) loads the 4096² bank with the jcp-fom2d code
(unchanged, imported read-only from `experiments/jcp-fom2d/`, same checkpoint `sep_hfit_dense_mid_N256_dense.pkl`,
sha256 recorded) and times ROM A (R384 gauss128 LSPG-CN-R $\Delta t=0.005$, the C2 baseline) and the two target ROMs on
dev6 constant-viscosity inputs, interleaved with the FOM subjects, all as ratios to the same A. The inherited C2 value is
the consistency check (in-job ROM ratio within $\pm20\%$ of C2's, else flagged).

## 6. Competitors (each tuned over its own knobs)

All in JAX, f64, `JAX_DEFAULT_MATMUL_PRECISION=highest`, one jitted program per configuration with $\kappa$, $\nu$ and the
initial data as traced inputs.

| family | discretisation | time stepping | solver | knobs (ladder) |
|---|---|---|---|---|
| **PSI** spectral IMEX + warm CG | sine–Galerkin, modes $1..N-1$; diffusion $-4G^\top W\kappa G$ by tensor Gauss–Legendre quadrature with $Q=2N$ points/axis (symmetric negative semidefinite); advection pseudo-spectral 3/2-padded (`pscore.nonlinear`) | SBDF2 or CNAB2 (diffusion implicit, advection extrapolated; first step SBDF1 / CN–FE) | CG, preconditioner = exact constant-coefficient inverse ${\rm diag}(\alpha+\theta\Delta t\,\nu\pi^2(k^2+l^2))^{-1}$, warm start $2b^n-b^{n-1}$, $\lVert r\rVert\le{\rm tol}\,\lVert {\rm rhs}\rVert$, maxiter 200 | $N\in\{16,24,32,48,64,96,128\}$, $\Delta t\in\{0.025,0.0125,0.00625,0.003125\}$, tol $\in\{10^{-3},10^{-5},10^{-8}\}$, scheme $\in\{$SBDF2, CNAB2$\}$ (168) |
| **ETD** exponential, constant-coefficient split | sine–Galerkin as PSI | ETDRK4 (Kassam–Trefethen coefficients of `pscore`) with linear part $-\bar\kappa\pi^2(k^2+l^2)$ and remainder $\nabla\cdot((\kappa-\bar\kappa)\nabla u)$ + advection explicit | none | $N$ as PSI, $\Delta t\in\{0.025,\dots,0.0015625\}$ (5), $\bar\kappa\in\{\kappa_{\max},\tfrac12(\kappa_{\max}+\kappa_{\min})\}$ (70) |
| **FDMG** finite differences + multigrid | 5-point flux-form diffusion, central advection, $(L-1)^2$ interior nodes | SBDF2 IMEX (diffusion implicit) | PCG with a geometric-multigrid V(2,2) preconditioner (damped Jacobi $\omega=0.8$, full weighting, bilinear prolongation, coarse operators re-discretised from $\kappa$ at coarse faces, dense solve on $L=8$) or with the constant-coefficient DST inverse; warm start; tol as PSI | $L\in\{64,128,256,512\}$, $\Delta t\in\{0.0125,0.00625,0.003125,0.0015625\}$, tol $\in\{10^{-4},10^{-7}\}$, preconditioner $\in\{$MG, DST$\}$ (64) |
| **CCv** E0 scheme as a FOM | the reference scheme (§4) | CN-R | Newton + DST-preconditioned BiCGStab, tolerance levels t1–t3 of jcp-fom2d | $L\in\{64,128,256,512\}$, $\Delta t\in\{0.0125,0.00625,0.003125\}$, t1–t3 (36) |

Not built (budget), so every speed-up is "against the strongest tested competitor": Krylov/Leja full-operator
exponential integrators, reused sparse factorisations, coefficient-aware algebraic multigrid, POD-ECSW.

**Output contract (as jcp-fom2d C2):** one call = prepared input → six full fields on the $4097^2$ nodes of the
$4096^2$ ROM mesh (spectral: sine-series synthesis, `psjax2d.synth_mesh`; FD: `fomrun2.fd_map` sine interpolation).
Input preparation (IC coefficients / nodal samples, $\kappa$ at quadrature points or faces, preconditioner setup) is
outside the timer for every arm, as for the ROM in C2; its seconds are recorded. Scoring uses the lattice restriction of
the same program's output.

**Admissibility:** every output finite, no CG/BiCGStab non-convergence at maxiter, running $\max|u|\le 10$ (blow-up guard).

## 7. Ladder job(s)

Per contrast: every configuration on the 8 dev cases (lattice output, scored against §4). **Lossless pruning:** a
configuration whose dev worst error exceeds 10 % cannot match any target on the pooled 32 cases, so only configurations
with dev worst $\le 10\%$ run on the 24 val cases. Screening cost: warm single-call full-field ($4096^2$) latency,
median of 3 reps on dev cases 0 and 1 (ranks candidates only).

## 8. Timing job (paired, one H200, one job)

Protocol of jcp-fom2d `c2time.py`: triplets burn | A | burn | B | burn | A, ratio $t_B/\tfrac12(t_{A1}+t_{A2})$, drift band
$[1/1.1,1.1]$ with ≤ 2 retries, warm JIT and a warm-up call of every subject on every case first; A = ROM A on dev6
case $i \bmod 6$. Subjects: ROM A, the two target ROMs (dev6 inputs), and per (family, $\tau$, $C$) the three
screening-cheapest admissible configurations matching $\tau$ (worst over eligible pooled cases $\le\tau$).
Stage 1 (selection): A–B–A on the 8 dev cases × 1 rep, selected = smallest median ratio. Stage 2 (inference): every
ROM and selected FOM, fresh triplets, 8 cases (ROM: 6) × 3 reps, randomised order. Cost = median over cases of the
per-case median ratio; absolute seconds kept. Identity: each timed output's lattice restriction is re-scored and must
reproduce the ladder error within $10^{-9}$ (FOM) / the C2 target error (ROM).

## 9. Metrics and the gate

$t^*_{\rm FOM}(\tau,C)$ = stage-2 cost of the fastest selected configuration over all four families matching $\tau$ at $C$;
$t^*_{\rm ROM}(\tau)$ = stage-2 cost of the target ROM of §5. $G(\tau,C)=t^*_{\rm FOM}/t^*_{\rm ROM}$, with a paired
case-bootstrap 95 % interval (descriptive; FOM and ROM workloads differ, so the bootstrap resamples each independently).

**Gate (verbatim from the brief, pre-registered):** *"Train a ROM only if the fastest FOM is ≥ 2× slower than our box
cost. If not, stop and report (no ROM trained). If the gate passes, Phase 2: retrain the bank on the variable-coefficient
family (single change), score vs the fastest FOM with the scoring contract; write Phase 2's DESIGN amendment and get it
audited before any training. Phase 2 must fit in the remaining cap."*

**Operational form:** the gate **passes** iff $G(\tau,100)\ge 2$ (point estimate) for at least one $\tau\in\{3\%,10\%\}$.
$\tau=1\%$ has no qualified ROM and cannot pass. $C=10$ is reported, not decisive. If the point estimate is $\ge2$ but the
interval's lower end is $<2$, the gate passes with the label "marginal". The where-we-win kill criterion is the same rule.
Secondary readings (reported, not gating): per-family costs at every $(\tau,C)$; mean CG / multigrid iterations per
solve; the cost ratio $C=100$ vs $C=1$ for PSI at the same $N,\Delta t$ (how much the lost exact diffusion costs).

## 10. Controls

Must-fail (each shown to fire in the real-size smoke, at real meshes):
- **T-kappa:** the constant-coefficient solver ($\kappa\equiv\nu$, PS ETDRK4 $N=48$) scored against the $C=100$ reference
  must have worst error $>10\%$ on the smoke cases (the coefficient matters; otherwise the problem is not variable-coefficient in effect).
- **T-loose:** PSI $N=48$ with tol $=10^{-1}$ and maxiter 1 must differ from tol $10^{-8}$ by more than $10^{-3}$ (the
  tolerance knob is live).
- **T-ref:** Richardson acceptance on (level 1, level 2, level 3 + $10^{-3}n_0$ perturbation) must reject the case.
- **T-nan:** a configuration with an injected non-finite output must be inadmissible.
- **T-dbl / T-fault / T-id** (timing, as C2): B = A executed twice gives ratio in $[1.8,2.2]$; a drift fault is caught by the band;
  a wrong identity reference vetoes the subject.
- **T-gate:** the gate function on synthetic costs ($3\times$ → pass, $1\times$ → fail, $1.9\times$ → fail).
Must-pass identities:
- **I-C1:** at $C=1$, ETD with $\bar\kappa=\nu$ reproduces `psjax2d` PS ETDRK4 to $\le10^{-10}$ relative; PSI's CG takes ≤ 1 iteration per solve; CCv
  reproduces `refcore.make_solver` to $\le10^{-10}$.
- **I-sym:** the discrete PSI and FDMG diffusion operators are symmetric to $10^{-12}$ and negative definite (random probes); the
  MG V-cycle preconditioner is symmetric to $10^{-10}$.
- **I-quad:** PSI with $Q=2N$ vs $Q=3N$ differs by $\le10^{-8}$ at $C=100$, $N=64$.

## 11. Budget and stop rules

GPU-hour cap 15; **this FOM-only phase ≤ 8**; lane concurrent-job cap 2; campaign cap 8 via the shared lock. Estimate:
real-size smoke 0.5; references 2 jobs × ≈1.5 (A100/H100/H200); ladder 1.5 (H200); timing 1.2 (H200, the 4096² bank needs
69 GB). Total ≈ 6.2 GPU-h. Each production job needs a smoke-derived cost certificate (`resources.py` pattern) and
$\text{spent}+\text{projection}\le 8$. Defer order if short: (1) $C=10$ timing, (2) $C=10$ val ladder, (3) CCv at $L=512$,
(4) FDMG DST-preconditioned variant.

**Stop rules (verbatim):** Gate as §9. Phase 2 only if the gate passes, with its own Codex-audited DESIGN amendment and
within the remaining cap (15 − spent); if it cannot fit, stop and report.

## 12. Deliverables

`checks/*.json` (references, ladders, timing, gate), `make_report.py` → `report.md` (every number generated), `EXPORT.json`,
plots (`plots/fom_cost_vs_error.png`, `plots/gate.png`), `runs/GPU-LEDGER.json`, `audits/`, HEARTBEAT.
