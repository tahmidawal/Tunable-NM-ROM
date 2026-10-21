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
