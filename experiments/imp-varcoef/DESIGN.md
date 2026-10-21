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

---

## Amendment A1 (2026-10-21, after Codex design audit 1, `audits/codex-design-1.md`; before any code)

Each item answers the audit item of the same number; the original text above stands where not changed.

- **A1.2 Gate interpretation.** The operational form in §9 (pass at either 3 % or 10 %, point estimate, "marginal" label
  when the interval crosses 2) is a **lane-specific operational choice**, not a consequence of the quoted gate. The
  where-we-win §7.3 Phase-2 criteria (equal median **and** worst error, union of competitors, bank floor ≤ 2× the
  constant-$\kappa$ floor) are carried into any Phase 2 amendment. A missing reference coverage (A1.6), or a target with
  no qualified admissible FOM, makes that target **inconclusive** (never an infinite-cost pass). If both 3 % and 10 % are
  inconclusive at $C=100$ the gate outcome is **unresolved** and no ROM is trained.
- **A1.4 ROM population.** `rom_box.py` searches FRONT-2D 4096² `members` **and** `other_qualified`, keeps entries with both
  `numerically_qualified_accuracy_job` and `numerically_qualified_timing_job`, LSPG only, and checks C2 completion
  (`complete`), absence of a veto for the subject, and 6 cases × 3 reps of valid stage-2 triplets. The E0 A10 and combined
  A10/A11/A15 labels are carried forward next to every ROM number.
- **A1.5 Reference acceptance = E0 predicates.** The reference job calls the E0 `verdict.judge` (imported unchanged from
  `experiments/jcp-references/verdict.py`) with per-level `tau = verdict.floor_proxy(res)`, `steps_ok = verdict.steps_ok`, and
  `nu` := the case's $\kappa_{\min}$ (Péclet uses the minimum diffusivity and the running $\max|u|$ at the coarsest level).
  This gives shared-$t_0$, solver, accumulated-floor, Péclet, order ($k=1..5$ only) and direction-consistency checks.
  **Independent cross-check:** on dev cases 0–3 at each contrast, a spectral PSI solve at $N=256$, SBDF2,
  $\Delta t=0.05/128$, tol $10^{-12}$; its lattice distance to the reference must be $\le 2.5\cdot10^{-4}$; a failure flags
  the contrast's reference as "cross-check failed" (reported; gate inconclusive at that contrast).
- **A1.6 Eligibility mask.** The eligibility mask per contrast (accepted by `judge`, $\epsilon_{\rm ref}\le10^{-3}$, cross-check
  not failed) is computed from the reference job alone and **frozen (sha256 in `checks/eligibility.json`) before any
  ladder output is scored**. Every exclusion is reported with its descriptors and outcome. Fallback: the ineligible cases
  of a contrast are rerun at the E0 levels $(2048,4096,8192)$ if the cap allows, else stay excluded. **Coverage rule:**
  the gate is decided at $C=100$ only if ≥ 24 of 32 pooled cases are eligible; otherwise it is unresolved.
- **A1.7 ν range.** The narrowed $\nu\in[0.02,0.1]$ is a **scope choice**, not a Péclet necessity (the audit is right that
  $\nu_{\min}=0.01$ gives Péclet 1.95 < 2 at $L=1024$ with $|u|\le2$). Reasons: (i) margin for the asymptotic regime of the
  coarser reference ladder, (ii) the minimum diffusivity ($\ge 0.002$) stays within a factor 5 of the campaign's stiffest
  corner (0.01), so the benchmark is less confounded with the documented steep-front failure. It does **not** isolate the
  loss of exact diffusion from changed steepness; the paired $C=1$ runs of PSI are the reading of that cost. Péclet is
  measured (running $\max|u|$, $\kappa_{\min}$ on the $4097^2$ grid, which bounds the $1025^2$ normalisation extremes from outside).
- **A1.8 PSI SPD.** With $G=[\partial_x;\partial_y]$ evaluated at the quadrature points, $W$ the positive tensor weights and
  $\kappa>0$, the implicit matrix $A=\alpha I+4\theta\Delta t\,G^\top W_\kappa G$ is SPD ($\alpha\in\{1,\tfrac32\}$,
  $\theta\in\{\tfrac12,1\}$), since $x^\top A x=\alpha|x|^2+4\theta\Delta t\sum_q w_q\kappa_q|(Gx)_q|^2>0$; the transpose is
  implemented as the exact matrix transpose of the evaluation matrices. **Quadrature change:** tensor **midpoint** rule with
  $Q=2N$ points per axis (exact for trigonometric products whose total frequency is below $2Q$; better than Gauss–Legendre
  for sine/cosine integrands), checked by I-quad ($Q=2N$ vs $3N$).
- **A1.9 MG contract.** Fixed linear V(2,2) cycle; prolongation $P=4R^\top$ with $R$ full weighting (adjoint up to a constant);
  pre- and post-smoothing the same damped Jacobi (ω = 0.8, 2 sweeps), which is symmetric; coarse operators re-discretised
  from $\kappa$ (SPD); dense Cholesky solve on $L=8$. Tests: symmetry $|x^\top My-y^\top Mx|\le10^{-10}|x||My|$ and
  positivity $x^\top Mx>0$ on 8 random probes per level count. If either fails the MG-PCG arm is replaced by the same cycle inside
  flexible CG (Notay's FCG(1)) and that is recorded.
- **A1.10 Integrators.** Temporal/stability behaviour is read from the ladder (the $\Delta t$ dimension at fixed $N$). **Boundary
  extension rule:** if, after the dev ladder, the cheapest qualified configuration of a (family, $\tau$, $C$) sits on the
  cheap edge of a knob (smallest $N$/$L$, largest $\Delta t$, loosest tol), one extension step beyond that edge is run
  (N 12 / L 32, Δt 0.05, tol $10^{-2}$) on dev and, if it survives, val. `pscore.nonlinear` is oversampled (3/2) collocation,
  not exact dealiasing; stated.
- **A1.11 Competitor strength.** Every gate ratio carries "against the strongest tested competitor; finite-ladder upper bound on
  the ROM speed-up". If the gate passes, the Phase 2 amendment must first add a full-operator Krylov exponential competitor
  (Lanczos $\varphi$-functions on the symmetric diffusion part, exponential-Euler/ETD2-type) and re-evaluate the gate before
  any training.
- **A1.12 FOM numerical qualification (Q1 analogue).** Every implicit stage records its iteration count and an independently
  recomputed true relative residual $\lVert A x-b\rVert/\lVert b\rVert$; a run is *valid* iff all outputs are finite, every
  stage's true residual $\le 10\,$tol (CCv: E0's `steps_ok`), the schedule completed, and running $\max|u|\le10$. A
  configuration is **numerically qualified** iff (i) valid on every pooled case (ineligible ones included), (ii) its
  lattice distance to the same $(N/L,\Delta t)$ solved at tol $10^{-10}$ (CCv: E0 tight level) is $\le2.5\cdot10^{-4}$ on
  every pooled case, and (iii) tol $10^{-10}$ vs $10^{-12}$ agree to $\le2.5\cdot10^{-5}$ on dev cases 0–1. (ii)–(iii) are run
  for every configuration that is a timing candidate. ETD has no solver: (i) only. Only qualified configurations enter the gate.
  Timed ROM subjects are revalidated in-job: their lattice errors reproduce the C2/FRONT `target_e` to $10^{-9}$, and the
  checkpoint, rule and mode-list hashes are recorded.
- **A1.13 Identity.** The timing job keeps C2's identity checks: per subject, the sha256 of the lattice restriction plus the full-output
  fingerprint (FOM) and W, w0, per-step exit reasons/iterations plus output fingerprint (ROM) from the scoring run, compared on every
  timed execution; a mismatch vetoes the subject. T-dbl uses two separately dispatched, separately synchronised A executions.
- **A1.14 Timed operations.** Excluded from the timer (prepared input, as C2): the IC (FD nodal samples, spectral sine
  coefficients). **Included** in every timed FOM call: evaluation of $\kappa$ from its 15 coefficients at quadrature points / faces /
  coarse-grid faces, multigrid hierarchy and coarse Cholesky factor, ETD coefficients, preconditioner diagonals, all time
  stepping and the six-field $4096^2$ synthesis. So the FOM pays all of its per-query setup.
- **A1.15 Selection.** Stage 1 selects and freezes, per $(\tau,C)$, both the per-family winners and the **union winner**
  (FROZEN-SELECTION written with sha256 before stage 2); stage 2 only re-measures. The gate uses the union winner's stage-2 cost.
- **A1.16 Bootstrap and anchors.** FOM timing workload = dev cases 0–5 (six); FOM case $i$ and ROM subjects are both timed
  against A on dev6 case $i$. Estimand: $G=\mathrm{med}_i\,\mathrm{med}_{\rm rep}\,r^{\rm FOM}_i/\mathrm{med}_i\,\mathrm{med}_{\rm rep}\,r^{\rm ROM}_i$.
  Interval: bootstrap over anchor index $i$ (resampling the whole record of index $i$: its FOM and ROM triplets jointly),
  2000 draws, percentile 95 %; called "anchor-paired case bootstrap" (descriptive of the timing workload).
- **A1.17 Pruning.** Val runs are skipped only for configurations whose worst error over the **frozen eligible** dev cases exceeds
  10 %, or which are invalid on any dev case (validity is checked on all cases, eligible or not).
- **A1.18 Test split.** Phase 1 result is **developmental evidence** (dev + val). The test split stays unopened, reserved
  for a Phase 2 test-once or a later lane; no claim beyond dev + val is made.
- **A1.19 Controls, restated.**
  *Implementation must-fail controls* (deterministic, real meshes): **T-loose** = PSI/FDMG with maxiter 0 (no solve; the
  stage keeps the warm start) must be invalid (true residual check) and differ from tight by $>10^{-3}$; **T-ref-a**: `judge` on
  levels $(u_1,u_2,u_2)$ must not be accepted; **T-ref-b**: $(u_1,u_2,2u_2-u_1)$ must not be accepted; **T-ref-c**: CCv level run
  with max_newton 1 at loose tolerance must give `steps_ok` false; **T-nan**: injected NaN output → invalid; **T-dbl/T-fault/T-id**
  as C2; **T-gate**: synthetic costs 3× → pass, exactly 2× → pass, 1.9× → fail, missing ROM target → inconclusive, empty FOM set →
  inconclusive, coverage < 24 → unresolved, success at $C=10$ only → fail.
  *Implementation must-pass checks:* I-C1, I-sym (now also positivity), I-quad, and **I-mms**: operator-level manufactured
  solution $u=\sin(\pi x)\sin(2\pi y)$ with the case $\kappa$: the discrete diffusion operators (PSI/ETD Galerkin projection, FD flux form)
  applied to $u$ must match the analytic $\nabla\cdot(\kappa\nabla u)=\kappa\Delta u+\nabla\kappa\cdot\nabla u$ (projected /
  sampled) with error decreasing (FD: observed order in $[1.8,2.2]$ over $L=64,128,256$; spectral: error $\le10^{-8}$ at $N=64$).
  *Scientific diagnostics (reported, not controls):* T-kappa (constant-$\nu$ solver vs the $C=100$ reference).
- **A1.20 Budget.** Every production job needs a smoke-derived certificate (reference: per-level seconds on the hardest smoke
  case at $C=100$; ladder: per-config compile + run seconds; timing: memory with the bank resident). Fallback reference cost is
  reserved before the ladder is submitted. Mandatory reference/qualification work is never deferred to preserve a positive gate.

---

## Amendment A2 (2026-10-21, after Codex design audit 2, `audits/codex-design-2.md`; before any job)

Numbers refer to the round-2 items. A2 overrides A1 where they differ.

- **A2.2 Gate outcome table.** Per decisive target $\tau\in\{3\%,10\%\}$ at $C=100$ the target outcome is *pass* ($G\ge2$),
  *fail* ($G<2$) or *inconclusive* (no qualified admissible FOM, union winner vetoed or with incomplete stage-2 coverage,
  or reference coverage/cross-check failure at $C=100$). **Gate = pass** if any decisive target passes; **fail** only if
  both fail; otherwise **unresolved**. A reference failure at $C=100$ (coverage < 24 eligible, or the cross-check fails)
  overrides all costs: unresolved. Unresolved → no ROM trained.
- **A2.4** `rom_box.py` now asserts the C2 gates (G-RR, A-valid, G-FR), pilots, every control fired, the exact registered case set,
  repetitions 0–2 per case and identity hashes for each subject.
- **A2.5 Cross-check (replaces A1.5's single run).** On dev cases 0–3 at each contrast, PSI SBDF2, tol $10^{-12}$:
  $(N,\Delta t,Q) = (192,\,0.05/128,\,2N)$, $(256,\,0.05/128,\,2N)$, $(256,\,0.05/256,\,2N)$, $(256,\,0.05/128,\,3N)$.
  The spatial, temporal and quadrature differences against $(256, 0.05/128, 2N)$ must each be $\le 2.5\cdot10^{-5}$ (cross-check
  converged); then its distance to the reference must be $\le\max(2.5\cdot10^{-4},\,3\epsilon_{\rm ref})$. Not converged or
  too far → "cross-check failed" for that contrast. API as E0: `floor_proxy(res, n_stages)`, `steps_ok(res, n_stages)` with
  `n_stages = refcore.n_stages(dt, 'cnr')`. The mask is frozen after all permitted fallback attempts; conclusions are stated as
  conditional on the eligible population.
- **A2.7 Discrete Péclet.** `judge`'s `nu` := the minimum of the coarsest reference level's face coefficients (the discrete
  operator's own minimum), with the running $\max|u|$ of that level. Sampled continuous extrema are reported as "sampled".
- **A2.9 MG.** A failed symmetry/positivity probe is treated as a bug to be fixed before production; no automatic method switch.
  If it cannot be fixed, the MG preconditioner variant is dropped (recorded) and FDMG runs with the DST preconditioner only.
- **A2.10 Stability and temporal checks.** (i) Fixed-case temporal check: on dev case 0 at each contrast, for each family at its
  largest $N$/$L$, the finest ladder $\Delta t$ and $\Delta t/2$ are compared (reported). (ii) Small-step extension: an
  $(N/L)$ row with no valid configuration at any ladder $\Delta t$ gets $\Delta t/2$ and $\Delta t/4$ once. (iii) A family with no
  qualified configuration at a decisive $\tau$ after the extensions is reported "unresolved for that family"; a positive gate in
  that situation carries the label "family X unresolved".
- **A2.12 ROM denominators.** Timed ROM subjects carry Q1/Q2 **by inheritance** (jcp-qualify FRONT-2D labels, prior evidence);
  the job verifies trajectory identity in-job (lattice error equal to `target_e` within $10^{-9}$ on dev6, W, w0, per-step exit
  reasons and iteration counts hashed and compared on every timed execution as C2). This is labelled next to every ROM number;
  no fresh in-job stationarity / quadrature requalification is run (it would repeat a deterministic computation on the same code
  and GPU type; the denominator is in any case an optimistic constant-$\kappa$ proxy, §5).
- **A2.14 Cost contract.** The gate uses the **setup-inclusive** query cost (A1.14), which is the C2 convention for the
  spectral FOM (ETD coefficients inside the timed program) and for the FD FOM (preconditioner denominators inside the solve).
  For each union winner a setup-only program (all of the winner's per-query setup, no stepping) is also timed and reported.
- **A2.16 Estimand.** $G=\mathrm{cost}_{\rm FOM}/\mathrm{cost}_{\rm ROM}$ with cost = median over anchor indices $i$ of the per-index
  median of $t_B/\tfrac12(t_{A1}+t_{A2})$ (the C2 A-normalised convention, the same as `rom_box.py`); the ratio of median
  absolute seconds is reported beside it.
- **A2.17** If no dev case is eligible at a contrast, no pruning: every configuration runs on val.
- **A2.18 Test-once (COMMON-BRIEF §2 is binding; A1.18 withdrawn).** At the end of Phase 1, the frozen union and per-family
  winners at each decisive $\tau$ for $C=100$ are run **once** on the 16 test cases with references built by §4 (test references
  are generated only after the selection is frozen). Their test median/worst errors are reported; the gate decision stays the
  dev+val one (test is not used for selection or for the gate).
- **A2.19 Controls with demonstrable power.** Implementation must-fail controls, each with its fixture:
  *T-loose*: PSI and FDMG at maxiter 0 on a real case; the first stage's initial residual is nonzero by construction
  ($\lVert A b^n-{\rm rhs}\rVert=\Delta t\lVert\theta D b^n+N(b^n)\rVert>0$); the validity flag must be false (true residual
  $>10\,$tol), and the residual value is reported.
  *T-ref-solver*: `judge` with one level's `steps_ok` = False → `solver-fail`. *T-ref-floor*: `tau` = 1 → `unresolved-floor`.
  *T-ref-peclet*: `nu` = $10^{-9}$ → `peclet-fail`. *T-ref-order*: $u_3=2u_2-u_1$ → `order-fail`. *T-ref-direction*: per $t_k$,
  $u_3=u_2-\tfrac14\lVert u_1-u_2\rVert\,\hat v$ with $\hat v\perp(u_1-u_2)$ (unit), giving $p=2$ but $c=\sqrt2>0.5$ →
  `direction-inconsistent`. *T-ref-ic*: $u_3(t_0)$ perturbed → `ic-mismatch`. All on the real smoke reference levels.
  *T-nan*, *T-dbl/T-fault/T-id*, *T-gate* (now also: one target pass + other inconclusive → pass; one fail + other inconclusive →
  unresolved; reference failure → unresolved) as before.
- **A2.21 Extension order.** After the dev ladder: provisional winners per (family, $\tau$, $C$) = screening-cheapest valid
  configuration with eligible-dev worst $\le\tau$; cheap-edge (A1.10) and small-step (A2.10) extensions run on dev and then
  val; only then are timing candidates (top 3 by screening cost per family/$\tau$/$C$ among valid configurations with pooled
  eligible worst $\le\tau$) chosen and qualified (A1.12, A2.23).
- **A2.22 CCv validity.** CCv competitors use a configuration-specific threshold: every stage's Newton residual
  $\le 2\,n_{\rm tol}$ (its own level's Newton tolerance), complete schedule, finite. E0's $2\cdot10^{-11}$ stays for references.
- **A2.23 Quadrature qualification (Q2 analogue).** Every spectral timing candidate (PSI, ETD) is rerun with $Q=3N$ on all pooled
  eligible cases; rollout distance $\le2.5\cdot10^{-4}$ on every case, else not qualified. The ROM ρ-bar (0.116) is a statistic of
  residual quadrature on reduced states and has no Galerkin-FOM analogue; stated. FD arms have no quadrature (exact face sampling).
- **A2.24 κ normalisation.** The extrema $\min g,\max g$ on $1025^2$ are part of the case definition (the coefficient's parameters),
  computed once per case like the IC, excluded from every arm's timer (and from Phase 2's ROM), cost reported.
- **A2.25 Timing cases.** Timing workload at contrast $C$ = the first six *eligible* dev cases, completed in index order with eligible
  val cases if fewer than six dev cases are eligible; frozen with the eligibility mask before the ladder.
- **A2.26 Stage-2 failure.** A frozen union winner that is vetoed, or lacks valid triplets for every timing index × 3 reps, makes the
  target inconclusive. No replacement.
- **A2.20 Budget ledger.** After the smoke a `checks/budget-plan.json` lists every remaining job with its certified projection, including
  the worst-case fallback references and the test references; production starts only if the total ≤ 8 GPU-h.

---

## Amendment A3 (2026-10-21, after Codex design audit 3, `audits/codex-design-3.md`; before any job)

A3 overrides A1/A2 where they differ. Numbers refer to the round-3 items.

- **A3.5 Cross-check certificate (a departure from E0's PS cross-check recipe, labelled so).** On dev cases 0–3 per contrast:
  PSI SBDF2 base $(N,\Delta t,Q,{\rm tol})=(256,0.05/128,2N,10^{-12})$ and four refinements: space $N\in\{128,192\}$ (three
  spatial levels; the differences to base must decrease monotonically, $d_{128}>d_{192}$), time $\Delta t/2$, quadrature $Q=3N$,
  solver tol $10^{-14}$. Cross-check uncertainty $U_{xc}=d_{192}+d_{\Delta t/2}+d_{3N}+d_{\rm tol}$; **converged** iff
  $U_{xc}\le2.5\cdot10^{-5}$, monotone spatial differences, all solves valid. **Passed** iff converged and
  $\lVert{\rm base}-R\rVert\le \max\big(2.5\cdot10^{-4},\,3(\epsilon_{\rm ref}+U_{xc})\big)$. The $2.5\cdot10^{-4}$ floor is the
  Q1 distance bar: the cross-check certifies the reference to the resolution relevant to targets $\ge1\%$ (40× below), not to $\epsilon_{\rm ref}$.
- **A3.10 Temporal/stability acceptance.** A row = (family, $N$/$L$, scheme/$\bar\kappa$/preconditioner, tol). A row is extended once by
  $\Delta t/2$ and $\Delta t/4$ if (a) it has no valid configuration, or (b) its finest-$\Delta t$ eligible-dev worst error is more
  than 20 % below the second-finest (time-error-limited). Fixed-case temporal check (A2.10 i) is reported. **A required family
  (PSI, ETD, FDMG: the brief's three categories) without a qualified configuration at a decisive $\tau$ after extensions makes
  that target inconclusive**, unless every one of that family's rows is invalid at every tested $\Delta t$ (then "family X
  unstable on the tested ladder" is reported and the target is decided on the remaining families).
- **A3.12 ROM denominators and COMMON-BRIEF §3.** The ROM configurations are **not** requalified in this lane; their labels are
  prior evidence and are stated as such next to every ROM number. This does not affect a **fail** verdict: requalification can
  only remove ROM configurations, which can only make the cheapest qualified ROM at a target slower, which can only lower $G$.
  A **pass** verdict is provisional: the Phase-2 amendment must, before any training, requalify (fresh in-job Q1/Q2 with the
  jcp-qualify machinery) every ROM configuration whose cost is used, and recompute $G$.
- **A3.14 Kernel timing.** Every competitor is implemented as `setup(kp) -> S` (κ evaluation, preconditioner/hierarchy, ETD
  coefficients) and `core(b0 or u0, S) -> fields` (stepping + synthesis). Timed programs: **query** = `core(·, setup(kp))` jitted as
  one program (the gate cost); for each union winner also **kernel** = `core` with prepared `S`, and **setup** = `setup` alone,
  each as its own A–B–A subject. Reported side by side; not subtracted.
- **A3.16** The gate statistic is named the **anchor-normalised cost ratio**.
- **A3.19 Control fixtures.** T-loose: the fixture is dev case 0 at $C=100$ (frozen); the control fires iff the run is invalid, and the
  measured margin $\max_{\rm stage}\,{\rm res}/(10\,{\rm tol})$ is reported (expected ≫ 1). Reference predicates: fixtures start
  from the first **accepted** real case; $t_0$ is preserved in every fixture except T-ref-ic, and mutations touch $k\ge1$ only; each
  fixture asserts that its target predicate is false **and every other predicate of `judge.checks` stays true**.
- **A3.20 Budget.** Production starts only if spent + all remaining certified work (references, fallback reserve, ladder, extensions,
  qualification, timing with one retry reserve, test-once references and runs) ≤ 8 GPU-h; otherwise the §11 defer order applies.
- **A3.23 Quadrature qualification (Q2 analogue, replaces A2.23).** For every spectral timing candidate, on every pooled case
  (eligible or not): (i) own-state operator discrepancy $\rho=\max_k\lVert D_{2N}b_k-D_{3N}b_k\rVert/\lVert D_{3N}b_k\rVert$ over the
  six reached states $b_k$ of its trajectory, $\rho\le0.116$ (the binding ρ bar applied to the quadrature-dependent operator), and
  (ii) rollout distance between $Q=2N$ and $Q=3N$ trajectories $\le2.5\cdot10^{-4}$. Both on every case.
- **A3.25 Timing slots.** Six slots $s=0..5$, frozen with the eligibility mask: slot $s$ = (FOM case: the $s$-th eligible dev case at
  that contrast, filled in index order from eligible val cases if needed; anchor and ROM case: dev6 case $s$). Bootstrap resamples slots.
- **A3.27 Test cohorts.** The current test split (seed 2026102103) is **Phase-1 only** (FOM test-once, A2.18). Phase 2, if any, must
  register a separately seeded, unopened cohort (seed 2026102104) before any training or tuning.

---

## Amendment A4 (2026-10-21, after Codex design audit 4, `audits/codex-design-4.md`; before any job)

- **A4.5 Cross-check convergence evidence (replaces A3.5's acceptance).** Runs: base $(256,\,0.05/128,\,2N,\,10^{-12})$; space
  $N=128, 192$; time $\Delta t/2$ and $\Delta t/4$; quadrature $3N$; solver $10^{-14}$. *Spatial evidence:* $d_{192}\le0.1\,d_{128}$
  (geometric decay consistent with spectral convergence; tail estimate $t_s=0.11\,d_{192}$) or $d_{128}\le10^{-7}$ (resolved; $t_s=d_{128}$).
  *Temporal evidence:* with $e_1=\lVert F_{\Delta t}-F_{\Delta t/2}\rVert$, $e_2=\lVert F_{\Delta t/2}-F_{\Delta t/4}\rVert$ (lattice
  distance), observed order $\log_2(e_1/e_2)\in[1.8,2.2]$ (base time-error estimate $t_t=\tfrac43e_1$) or $e_1\le10^{-7}$ ($t_t=e_1$).
  $U_{xc}=t_s+t_t+d_{3N}+d_{\rm tol}$ is a **conditional indicator**, not a certified bound. **Converged** iff both evidence rules hold,
  $U_{xc}\le2.5\cdot10^{-5}$ and all solves are valid. **Passed** iff converged and $\lVert{\rm base}-R\rVert\le\max(2.5\cdot10^{-4},3(\epsilon_{\rm ref}+U_{xc}))$.
  This validates the reference at the $2.5\cdot10^{-4}$ resolution relevant to targets $\ge1\%$; it does not validate a smaller $\epsilon_{\rm ref}$.
- **A4.10 Temporal acceptance and families.** Fixed-case temporal check (dev case 0, each contrast, each family at its largest $N/L$
  with its tightest tolerance): accepted iff $\lVert F_{\Delta t_{\min}}-F_{\Delta t_{\min}/2}\rVert\le10^{-3}$ (one tenth of the tightest
  target); if not accepted, every row of that family gets the A3.10 small-step extension ($\Delta t/2$, $\Delta t/4$). After extensions, a
  required family (PSI, ETD, FDMG) with no qualified configuration at a decisive $\tau$ is **unresolved**. Direction of the effect:
  adding competitors can only make the fastest FOM faster, so it can only lower $G$. Hence an unresolved family **blocks a pass**
  at that target (target inconclusive for passing) but does **not** block a fail decided by the resolved families (a budget-limited
  omission under COMMON-BRIEF §5, labelled "against the strongest tested competitor; family X unresolved").
- **A4.12 Outcome labels (COMMON-BRIEF §3 compliance).** No exception to §3 is claimed. The gate outcomes are renamed:
  **PASS-PROVISIONAL** ($G\ge2$; cannot authorise training until the Phase-2 amendment performs fresh in-job Q1/Q2 of the ROM
  denominators and recomputes $G$), **FAIL-SCREEN** (a conservative historical-proxy screen: the denominators are inherited,
  not §3-qualified in this lane; presented as a screening result, not a §3-qualified fail), **UNRESOLVED**. Either non-pass outcome means
  no ROM is trained. To make the denominator an in-job minimum rather than a historical winner, the timing job times **every** ROM
  configuration of `rom-box.json` with worst $\le\tau$ whose C2 cost is within 1.5× of the C2 cheapest at that target; the denominator is
  the in-job cheapest of these (anchor-normalised, stage 2). If a target loses all its ROM configurations (none valid in-job), it is
  inconclusive, not a fail. The lack of an authorised exception is raised in the hand-back as an open question.
- **A4.19 T-ref-order fixture (replaces A2.19's).** $u_3=u_2-\tfrac12(u_1-u_2)$ at $k\ge1$ ($t_0$ preserved): order $p=1$ (outside
  [1.8, 2.2]) while floor, direction ($c=0$), Péclet, solver and $t_0$ predicates stay true. Failure of any must-fail control to fire
  blocks production.
- **A4.23 ρ zero denominators.** $\rho_k=0$ if $\lVert D_{3N}b_k\rVert=\lVert D_{2N}b_k\rVert=0$; $\rho_k=\infty$ if only the denominator is zero.

---

## Amendment A5 (2026-10-21, after Codex design audit 5, `audits/codex-design-5.md`; before any job)

- **A5.5 Cross-check extrapolation (replaces A4.5's evidence rules).** Adjacent differences. Space: $\delta_1=\lVert F_{128}-F_{192}\rVert$,
  $\delta_2=\lVert F_{192}-F_{256}\rVert$ (lattice distance, same $\Delta t,Q$/N-ratio, tol); with $r=\delta_2/\delta_1$, the
  equal-spacing geometric (spectral) model gives the base's remaining spatial error $t_s=\delta_2\,r/(1-r)$; accepted iff $r\le0.1$,
  or iff **both** $\delta_1,\delta_2\le10^{-7}$ (resolved; $t_s=\delta_2$). Time: $e_1=\lVert F_{\Delta t}-F_{\Delta t/2}\rVert$,
  $e_2=\lVert F_{\Delta t/2}-F_{\Delta t/4}\rVert$, $p=\log_2(e_1/e_2)$; accepted iff $p\in[1.8,2.2]$ with the base's time error
  $t_t=e_1/(1-2^{-p})$, or iff **both** $e_1,e_2\le10^{-7}$ ($t_t=e_1+e_2$). $U_{xc}=t_s+t_t+d_{3N}+d_{\rm tol}$, still a conditional
  indicator; converged/passed rules as A4.5.
- **A5.10 Temporal check after extension.** After the small-step extension, the fixed-case temporal check is repeated on the two finest
  (extended) steps; if it still fails ($>10^{-3}$) the family is **unresolved** (blocks a positive target verdict; does not block a
  negative one, A4.10).
- **A5.12 ROM denominator — the lane brief is the binding exception.** COMMON-BRIEF's header: "Where they conflict, your lane brief
  wins." The lane brief defines the denominator: *"Our box cost = the deployed bank ROM per-query cost on the box at the same targets
  (from jcp-p1-scaling / jcp-fom2d, same GPU type; re-measure in the same job if possible)."* So the denominator is, by the lane brief,
  the inherited deployed configuration's cost, with its inherited qualification (prior evidence, stated next to every ROM number);
  no fresh Q1/Q2 is required for the Phase-1 gate. **Selection/inference separation:** the ROM denominator per target is selected
  from the **whole** registered population (all 21 inherited qualified LSPG configurations in `rom-box.json`) using the historical C2
  stage-2 costs — this plays the role of stage 1 and is frozen in `checks/rom-box.json` (sha256 recorded) before any job of this lane —
  and the timing job re-measures only that frozen configuration (stage 2, fresh triplets). It is described as "the frozen selected
  configuration", not as an established population minimum. The A4.12 1.5× shortlist is withdrawn. The inherited C2 cost is reported
  beside the in-job one; if the in-job re-measurement fails (vetoed, invalid, coverage), the inherited C2 cost is the denominator,
  labelled "inherited, not re-measured". Outcome labels: **PASS-PROVISIONAL** (before any training the Phase-2 amendment must still
  requalify the denominators in-job and recompute $G$; a prudential rule of this lane), **FAIL** (per the lane-brief gate; denominators
  with inherited qualification), **UNRESOLVED**.

---

## Amendment A6 (2026-10-21, after Codex code audits 1–2; before any job)

- **A6.1 Cheap-edge extension (restates A1.10/A2.21 for code audit 1 item 8).** Every dev-valid configuration that matches some
  target on the eligible dev cases and sits on a cheap edge of a knob is extended one step beyond that edge (N 12 / L 32,
  $\Delta t=0.05$, tol $10^{-2}$; CCv: a new tolerance level `4` = $(10^{-3},10^{-4})$, one step looser than t3, registered here). This is a
  superset of the provisional-winner rule (it contains every extension that rule could trigger) and uses no latency to choose.
  Its extra cost is included in the ladder certificate.
- **A6.2 CCv setup.** Every CN-R stage has $\theta\tau=\Delta t/2$, so CCv's per-query setup (face coefficients and the one DST
  denominator $1+\tfrac{\Delta t}{2}\nu\Lambda_h$) is prepared in `setup` and timed separately as A3.14 requires.
- **A6.3 C = 1 reading.** The paired PSI cost reading at $C=1$ uses the same physical cases as the $C=100$ slots (κ ≡ ν); it is
  validity-checked only (no reference at $C=1$) and reported, never gating.
- **A6.4 Findings noted, not blocking (COMMON-BRIEF §8: forgery-only or single-submitter concurrency).** Replay of the test-once step
  by a deliberate second invocation, concurrent staging by several submitters, and hand-edited result files: this lane has one
  submitter acting sequentially; the opening record is created exclusively once, and every acceptance check recomputes from raw outputs.
