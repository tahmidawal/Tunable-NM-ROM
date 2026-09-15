# Outline — *Nonlinear Manifold Reduced Models with Precomputed Weak Operators*

One paper, not two. The spine is the nonlinear manifold ROM with **precomputed weak
operators** and the mesh-independence of its *cached reduced evaluation*.
Fixed-checkpoint tunability is **one section**, not the headline. Losses are reported as
losses. Every number arrives from a generator script reading run JSONs; the LaTeX carries
`\gen{...}` placeholders until it does.

This document is the claim ledger: for each claim, the decisive experiment, its acceptance
criterion, and its evidence status.

---

## 0. Status vocabulary, and an important correction

The brief for this session said four experiments were "running concurrently right now". The
on-disk state of the sibling worktrees at the time of writing does **not** match that, and
this outline records what is actually on disk. A claim ledger that overstates its own
evidence is the failure mode this project has written down most often.

| Status label used below | Meaning |
|---|---|
| **exists** | Collected, audited run JSONs are on disk and a generator can read them today. |
| **collected (negative)** | Same, and the result went against the method. Retained, not hidden. |
| **prepared, not run** | Driver and config written; no run directory, no job, no output. |
| **not started** | Worktree or branch exists; no design document, no code, no output. |
| **not run** | Named in the plan; nothing on disk at all. |

Per-experiment on-disk status (checked in the sibling worktrees):

| Experiment | Status per the session brief | Status on disk |
|---|---|---|
| Burgers fixed-checkpoint tuning (GN cap / tolerance / EQ rule) | running | **prepared, not run** — plan committed at `2026-09-14-no-audit/.../checks/fixed-checkpoint-tuning-plan.json`; driver `tuning.py` + `tuning-config.json` present but **untracked** in `2026-09-14-no-burgers`; no `runs/tuning*`, no job |
| Head ablation (linear / quadratic / neural / POD-LSPG at matched *k*) | running | **not started** — `2026-09-14-head-ablation` is a clean checkout of the consolidated base `02ff0f1f`; no design doc, no code, no output |
| Frozen-checkpoint mesh ladder 64–1024 vs DST / Newton FOM | running | **not started** — `2026-09-14-mesh-ladder` likewise a clean checkout. A *prior* 64/256/1024 ladder does exist as `reports/2026-09-11-iterative-fom-multiresolution.{md,json}` and can carry a provisional figure |
| Burgers FNO common-data baseline | running | **prepared, not run** — lane committed in `2026-09-14-no-audit` (`stage_burgers.py`, `worker_burgers.py`, `configs/burgers/*`); no `runs/fno_burgers01`, no job. The *Burgers ROM-vs-FOM* diagnosis in `2026-09-14-no-burgers` **is** complete and collected, and is negative |
| Poisson matched-data neural-operator screen | (not named) | **collected (negative)** — `2026-09-14-no-poisson/.../runs/matched01/matched-summary.json`, jobs 3701314 / 3702473, complete and audited |

Multi-seed repetition and sealed-cohort confirmation are **not run** for any claim in this
paper.

---

## 1. Introduction

**Framing.** Repeated PDE solves with a discretisation available. Learned nonlinear
manifolds compress the solved dimension; the question is what that compression costs and
buys once the reduced operators are assembled exactly rather than sampled.

**Contributions, stated at the level the evidence can carry.**

| # | Contribution | Decisive experiment | Acceptance criterion | Evidence status |
|---|---|---|---|---|
| C1 | A separable frozen decoder whose bank admits both exact preassembly of the linear and polynomial weak operators and sampled evaluation of what resists it | E1 (operator parity) | Preassembled residual **and** Jacobian match the full-grid weak evaluation to the recorded floating-point tolerance at every mesh, on decoded states that satisfy the sign condition | **exists** — the assertion is executed in-job (`run_pilot.assemble`, heat; `b2d_tensor_common` gates T0/TB, Burgers) |
| C2 | A precisely stated weak least-squares solve with recorded stationarity and exit reasons, replacing the conflation of tangent Galerkin with residual least squares | E0 (definition + per-query diagnostics) | Every reported solve carries its exit reason and its normalized gradient; stalls are never reported as converged | **exists** — implemented in all three cells; the diagnostics are already in the collected JSONs |
| C3 | Removing fitted empirical quadrature from the repeated reduced evaluation, and the conditions under which that is exact | E1, E2 | Stated in E1/E2 below | **partly exists** (parity); **not run** (matched offline/online cost comparison at one frozen checkpoint) |
| C4 | Characterisation of when the learned coefficient nonlinearity pays at matched reduced dimension | E3 (head ablation) | Stated in E3 | **not started** |
| C5 | Cached reduced evaluation has no explicit mesh loop; the complete query does not inherit that property | E4 (mesh ladder) | Stated in E4 | **provisional evidence exists** (prior ladder); the frozen-checkpoint ladder with DST and Newton in one job is **not started** |
| C6 | Deployment-time controls at a fixed checkpoint, reported as an error–cost relation with its saturation and its stalls | E5 (fixed-checkpoint tuning) | Stated in E5 | **prepared, not run** |

**Explicit non-claims, in the introduction and not deferred to a limitations paragraph:**
no speedup at matched accuracy against a tuned classical solver; no dominance over neural
operators; no accuracy–cost frontier; no continuum error bound; no convergence theory.

---

## 2. Related work
`related-work.tex`. Positions against tensorial POD (Ştefănescu & Sandu 2014), quadratic
manifolds (Geelen et al. 2022; Barnett & Farhat 2022; Weder, Schwerdtner & Peherstorfer
2024), NM-ROM/LSPG (Lee & Carlberg 2020; Carlberg et al. 2011; Kim & Choi 2022),
hyper-reduction (Hernández et al. 2017; Yano & Patera 2019; Farhat et al. 2014;
Chaturantabut & Sorensen 2010), and neural operators (Li et al. 2021; Lu et al. 2021;
Kovachki et al. 2023). States plainly that the tensor construction is **not** new, that
the old CP decoder admits the same preassembly, and that the burden is on the empirical
comparisons.

---

## 3. Method
`methods.tex`. Sections and the code each was checked against:

| Subsection | Content | Code checked against |
|---|---|---|
| 3.1 Setting | $n$, $R$, $k$, $M$, $m$; asserted $M\ge 4k$, $m\ge 4M$ | `engines.build_rom` |
| 3.2 Trial manifold | $u=u_g+Gh_\theta(z)$; smooth boundary factor; exact rank bound $\operatorname{rank}J_u\le\min\{k,\operatorname{rank}G\}$ and when it is attained | `sep_common.features/head/SeparableDecoder` |
| 3.3 Weak residual and solver | tests $P$, row scaling, the minimised objective, $\eta=\lVert J^\top r\rVert/(\lVert J\rVert_F\lVert r\rVert)$, the damped LM update, exit-reason table, what stationarity does not certify, cold-start initialisation | `accuracy_paths.make_stationary_lm`, `kernel_solver.make_lm_kernel`, `ctol_tol.lm_tau_poisson`, `heat_core.make_lm`, `engines.build_gauss_cold` |
| 3.4 Precomputed operators | $B=PAG$, $b$; $T_{iab}$, $Q=T+T^\top_{ab}$, $O(MR^2)$ storage and contraction; the three conditions incl. the Burgers sign restriction; what is and is not mesh-independent | `core.assemble`, `run_pilot.assemble`, `b2d_tensor_common.build_T/symmetrize/q_of` |
| 3.5 Poisson | tested-error identity $r_w=P(u-u^\star)$; variable projection with a **constant** projector, so the elimination is exact and only the Gauss–Newton head curvature is dropped | `correction_core.prepare_correction/correction_query` |
| 3.6 Heat | fully discrete Crank–Nicolson residual, nonlinear in $z_{n+1}$; the decoded-current-state right-hand side; Cholesky and contracted-Gram paths | `heat_core.cn_factor/make_rollout`, `cp_algebra_paths.make_gram_lm/build` |
| 3.7 Burgers | backward Euler, modal Helmholtz row scaling, EQ vs tensor for the advection only | `engines.weak/spatial/residual/modes` |
| 3.8 Empirical quadrature | offline NNLS on decoder outputs, requested vs achieved vs fit-snapshot counts, selection among stored rules, what node evaluation does not save | `engines.build_rom/nnls_capped` |
| 3.9 Fixed-checkpoint controls | the three controls and everything held fixed | `accuracy_paths.make_rom`, `tuning-config.json` |
| 3.10 Intrusive vs non-intrusive | stated before any comparison | — |
| 3.11 Limits | seven explicit non-guarantees | — |

---

## 4. Experimental setup
Problem families, discretisations, data generation and seeds, the query contract (host
input in, requested dense fields out, everything between it charged), the timing protocol
(per-block GPU burn-in, device synchronisation around every measured region, medians over
retained repetitions, never a cross-job ratio), and the named FOM comparators: **direct
DST** wherever the operator is separable, tolerance-tuned **CG**, and
**Newton–BiCGStab** with modal Helmholtz preconditioning for Burgers.

---

## 5. Results

### E0 — Solver definition and per-query diagnostics (supports C2)
*What it is.* Not a separate run: a reporting discipline applied to every panel. Each
solve emits its exit reason, its normalized gradient, its residual, and (Poisson) the rank
and singular values of the projected Jacobian and the backward error of the linear
recovery.
*Acceptance criterion.* No table row may report a physical error without its exit-reason
distribution beside it; stalls are labelled `stalled`, never `converged`.
*Status.* **exists** — these fields are already present in the collected JSONs.

### E1 — Operator parity: preassembled vs full-grid weak evaluation (supports C1, C3)
*Arms.* full-grid weak projection · corrected EQ · exact precomputed algebra, at one
frozen checkpoint per PDE, with tests, row scaling, initialisation, time discretisation,
stopping and output contract held fixed.
*Measured.* Residual and Jacobian parity; accepted **and rejected** solver states; the
positivity audit (fraction of negative decoded nodes, and the tensor-vs-upwind departure
it produces); offline assembly cost charged separately.
*Acceptance criterion.* (i) For the linear terms, parity at the recorded floating-point
tolerance — the in-job assertion already demands $<10^{-11}$ relative for both operator
and Jacobian. (ii) For the quadratic term, parity at that tolerance **on sign-satisfying
states**, with the departure on sign-violating states measured and reported, never
assumed small.
*Status.* Linear parity **exists** (asserted in-job). Quadratic parity on positive states
**exists** (`b2d_tensor` gate T0). The matched three-arm comparison at one frozen
checkpoint with charged offline cost is **not run**.

### E2 — Preassembly against a fitted quadrature rule, same checkpoint (supports C3)
*Arms.* exact tensor · EQ at each stored $m$ · full-grid oracle.
*Measured.* Physical error, complete device-query cost with its components, offline
assembly and NNLS fitting cost, memory ($MR^2$ against the rule's cached blocks), and the
node-selection sensitivity (a rebuilt rule selects a different admissible support).
*Acceptance criterion.* A claim that preassembly is preferable requires **both** no
physical-error regression at the same checkpoint **and** a stated total-cost account
including offline assembly. Comparable online cost alone is not a result; it is the
already-known outcome recorded in the strategy document.
*Status.* **not run** (the archived 1D/2D tensor ladders are the closest prior evidence
and are explicitly not a matched-checkpoint comparison).

### E3 — Head ablation at matched latent dimension (supports C4)
*Arms.* linear head · quadratic head · neural head · unrestricted bank coefficients ·
properly solved linear POD-LSPG — all at matched $k$, matched bank where applicable, the
same weak objective and time solver, each with an adequately overdetermined system.
*Measured.* Bank projection error and best-found nonlinear fitting error reported
**separately** from rollout error; solver work; cost at matched error and error at matched
cost.
*Acceptance criterion.* The neural head is preferred only if, at matched $k$ and matched
cost, it reduces deployed error against **both** the quadratic head and a correctly solved
linear POD-LSPG on at least one family, with the failing families reported. Instability of
a *different* Galerkin rollout is not evidence for decoder nonlinearity.
*Status.* **not started**. Closest prior design: `experiments/mr-heat2d/HEAD-PILOT-DESIGN.md`.

### E4 — Frozen-checkpoint mesh ladder, 64–1024 (supports C5)
*Arms.* one frozen checkpoint per PDE, weights unchanged across meshes, reduced dimensions
and solver policy fixed; against direct **DST** (where separable) and
**Newton–BiCGStab** (Burgers), all in one job per mesh on one GPU.
*Measured.* Cached reduced cost against mesh size; complete device-query cost with input,
setup, solve and output components separated; iteration counts (hardware-free); physical
error; offline re-assembly cost at each mesh.
*Acceptance criterion.* The mesh-independence claim is about the **cached reduced
evaluation's operation count**, and is accepted only if stated that way: flat reduced cost
does **not** license a claim of flat complete-query cost, and the input/output components
must be shown growing. If accuracy degrades with refinement at fixed $(k,R,M)$, that is
reported as the cost of the claim.
*Status.* **not started** as a frozen-checkpoint ladder with DST and Newton in one job. A
prior 64/256/1024 ladder exists (`reports/2026-09-11-iterative-fom-multiresolution.json`)
and can carry a **provisional** figure, labelled as development evidence in which Poisson
and waves miss their accuracy target and Burgers exits are stalls.

### E5 — Deployment-time controls at a fixed checkpoint (supports C6)
*Arms.* one frozen Burgers checkpoint; EQ rule $\in$ {stored rules, full-grid control} ×
GN cap × stopping tolerance; identical weights, latent dimension and bank rank throughout.
*Measured.* Physical error against complete query cost over the control grid, with the
exit-reason distribution at every point; saturation; non-monotonicity between weak residual
and field error; the representation floor of the frozen decoder drawn on the same axes.
*Acceptance criterion.* Report the measured relation and its saturation. Do **not** claim
a frontier: a frontier claim requires that the relation be monotone and that no operating
point be dominated, and neither has been measured. Early-stopped arms keep that label.
*Status.* **prepared, not run**. The plan is committed; the driver is untracked and must be
committed before any run (work that exists only in an uncommitted file does not exist).

### E6 — Common-data comparison against a neural operator (supports the framing)
*Arms.* FNO at several capacities against the ROM, **identical dataset and split**, index
hashes recorded; timing only in a single job containing both.
*Acceptance criterion.* Report the outcome whichever way it falls, with the unmatched
training/tuning budgets stated. No accuracy-per-cost verdict without a paired latency panel
in one job.
*Status.* Poisson accuracy screen **collected (negative)**: the neural operator is the more
accurate model, neither ROM arm is selected, and no paired latency panel exists. The
Burgers arm is **prepared, not run**. This belongs in the paper as a reported result, not
as an omission.

### E7 — Multi-seed repetition and sealed-cohort confirmation
*Acceptance criterion.* Every headline number repeated over $\ge 3$ training seeds, then a
final evaluation on a cohort opened once, with all choices frozen beforehand.
*Status.* **not run.** Until it is, every reported endpoint is single-seed development
evidence and must say so inline, not only in a preamble.

---

## 6. Claim-to-figure allocation

| Fig. | Content | Supports | Generator (to be written under `paper/figures/`) | Reads | Status |
|---|---|---|---|---|---|
| F1 | Architecture and data flow: trained / frozen / solved | C1, C2 | `figures/architecture.mmd` (hand-authored diagram; no numbers) | — | **exists** |
| F2 | Operator parity: preassembled vs full-grid residual and Jacobian, per mesh; positivity audit beside it | C1, C3 | `gen_fig_operator_parity.py` | E1 JSONs; in-job assertion records in `run_pilot.assemble` / `b2d_tensor` gates | parity **exists**, matched arms **not run** |
| F3 | Exact tensor vs fitted EQ at one frozen checkpoint: physical error, online cost, offline cost, memory | C3 | `gen_fig_preassembly_vs_eq.py` | E2 JSONs | **not run** |
| F4 | Head ablation at matched $k$: bank projection, best-found fit, rollout error, and cost — as three stacked panels so the error sources are not conflated | C4 | `gen_fig_head_ablation.py` | E3 JSONs | **not started** |
| F5 | Mesh ladder: cached reduced cost, complete query cost with components, iteration counts, error — against DST and Newton | C5 | `gen_fig_mesh_ladder.py` | E4 JSONs; provisionally `reports/2026-09-11-iterative-fom-multiresolution.json` | **provisional** |
| F6 | Fixed-checkpoint controls: error against cost with exit reasons encoded as marker style, representation floor drawn as a horizontal line | C6 | `gen_fig_fixed_checkpoint_controls.py` | E5 `index.json` (`summaries/effort/*`, `summaries/quadrature/*`) | **prepared, not run** |
| F7 | Common-data ROM vs FNO: accuracy by capacity, with the missing latency panel shown as absent rather than omitted | framing | `gen_fig_common_data_operator.py` | `runs/matched01/matched-summary.json`, `runs/fno_poisson01/field-audit.json` | **collected (negative)** |
| T1 | Per-cell configuration: $N$, $n$, $k$, $R$, $M$, $m$, $\Delta t$, checkpoint SHA, job id, GPU, backend, precision | reproducibility | `gen_tab_configuration.py` | every run JSON's metadata block | **exists** |
| T2 | Exit-reason distribution per panel | C2 | `gen_tab_exit_reasons.py` | every run JSON's solver records | **exists** |

Every generator writes a paired `.json` beside its figure so the figure can be re-derived
and checked. No figure may be produced from a number typed into a script.

---

## 7. What must happen before submission, in order

1. Commit the untracked Burgers `tuning.py` and its config (E5). Untracked work is
   unreproducible work.
2. Run E1/E2 — the matched three-arm operator comparison at one frozen checkpoint. This is
   the paper's spine and is the cheapest missing piece.
3. Design and run E3 (head ablation). Without it, C4 is unsupported and the paper's answer
   to "when is the nonlinear manifold needed" is the same non-answer the reviewers rejected.
4. Run E4 as a frozen-checkpoint ladder with DST and Newton inside one job per mesh.
5. Run E5 and report the relation with its saturation.
6. Complete E6's Burgers arm and, separately, a paired latency panel, or drop the
   accuracy-per-cost framing for that comparison.
7. E7: seeds, then the sealed cohort. Only then may any endpoint drop its single-seed label.

---

## 8. Glossary

Written for a reader who knows none of this project's vocabulary.

- **PDE / FOM / ROM / NM-ROM** — the partial differential equation being solved; the full
  discretised solver; a solver using far fewer unknowns; a ROM whose unknowns parameterise a
  curved (nonlinear) set of states rather than a flat subspace.
- **Decoder / spatial bank / coefficient map / latent coordinates** — the map from few
  numbers to a full field; the fixed spatial functions it combines ($G$, width $R$); the
  network producing their weights ($h_\theta$); the few unknowns solved per query ($z$,
  dimension $k$).
- **Weak tests / row scaling** — smooth functions used to average the equation error, $M$
  of them; the diagonal weights setting the relative importance of those averaged
  equations. Both change *which* solution is selected, so both are part of the method.
- **Residual / Jacobian / stationarity** — how badly the equation is violated; its first
  derivatives with respect to the latent coordinates; the condition that the gradient of
  the least-squares objective vanishes. Stationarity is *not* a guarantee of physical
  accuracy.
- **Stall / early-stopped** — a solve that ran out of iteration budget, took a vanishing
  step, or hit the damping limit, rather than meeting a tolerance. Reported as such.
- **Galerkin / least-squares Petrov–Galerkin (LSPG)** — setting projections of the residual
  to zero; minimising the size of a specified residual in a specified norm. They are
  different conditions and give different equations.
- **Empirical quadrature (EQ) / NNLS** — approximating a sum over all mesh nodes by a
  weighted sum over a few, with the weights fitted offline by non-negative least squares.
- **Preassembly / quadrature-free / tensor $T$** — computing the reduced operator exactly
  and once, offline; removing the fitted sampling from the repeated online evaluation;
  the cached array encoding a quadratic operation.
- **Upwind / sign restriction / undershoot** — choosing a difference stencil according to
  the direction of transport; the condition ($u\ge0$) under which that choice becomes a
  fixed stencil and the algebra becomes polynomial; a decoded field dipping below zero and
  breaking that condition.
- **$m$ vs $M$** — $m$ is the number of quadrature nodes; $M$ the number of weak equations.
  The code requires $m\ge 4M$ and $M\ge 4k$.
- **Held-out / development cohort / sealed cohort** — cases not used for fitting; cases
  already opened and therefore usable for choosing settings; cases opened once at the end,
  after every choice is frozen.
- **Tripwire / gate / acceptance criterion** — a condition declared before a run that
  decides whether its result may be used; a check that must pass; the bar a claim must
  clear.
- **DST / CG / Newton–BiCGStab** — a direct transform solver available when the operator
  separates on a rectangle (the fastest comparator where it applies); an iterative solver
  for symmetric positive-definite systems; the nonlinear iteration used for Burgers, with a
  preconditioned Krylov inner solve.
- **Rung / arm** — one setting on a ladder of increasing effort; one variant in a
  controlled comparison.
- **Checkpoint / frozen transfer / seed** — saved network weights; evaluating those same
  weights on a different mesh; the recorded randomness of a training run.
- **Device query / complete query cost** — a computation timed from input arrival to
  output departure on the accelerator, including projection, solve and dense output — not
  the latent solve alone.
