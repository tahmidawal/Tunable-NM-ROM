# Outline — *What is and is not tunable in a nonlinear-manifold ROM*

One paper, not two. The spine is the nonlinear manifold ROM with **precomputed weak
operators**, the mesh-independence of its *cached reduced evaluation*, and — as the
organising frame for every result — the **three-layer error decomposition**
(representation / reduction / solver). Inference-time tunability is **one section**, not the
headline, and it is now stated in the exact form the evidence supports. Losses are reported
as losses. Every number arrives from a generator script reading run JSONs; the LaTeX carries
`\gen{...}` placeholders until it does.

This document is the claim ledger: for each claim, the decisive experiment, its acceptance
criterion, and its evidence status, with the report path and SHA256 of the evidence beside
it.

---

## What changed since 2026-09-14

This outline and `REVIEWER-RESPONSE-MAP.md` were written on 2026-09-14, **before** the
campaign's jobs finished, and their statuses described an empty disk. Between then and
2026-09-16 eleven GPU jobs and three local diagnostics completed, were audited, and were
archived in their worktrees. This section is the diff; the ledger below carries the detail.

1. **E3 head ablation: `not started` → `exists`.** Jobs `3711424` (Burgers) / `3711736`
   (Poisson). The nonlinear head wins per dimension on both PDEs; on Poisson POD-128 buys
   it back at essentially the same cost.
2. **E4 frozen-checkpoint mesh ladder: `not started` → `exists`.** Jobs `3711388` /
   `3711389`. Cached reduced cost is flat in mesh; there is no crossover against an
   efficient same-job FOM at any rung on either PDE.
3. **E5 fixed-checkpoint tuning: `prepared, not run` → `collected (negative)`.** Jobs
   `3711134` / `3712269`. Solver-side tuning is a step in cost, not a slope in accuracy, and
   no tuned setting beats the efficient FOM on both axes.
4. **E6 Burgers FNO arm: `prepared, not run` → `exists`.** Jobs `3710790` / `3710846`. On
   Burgers the ROM is *more* accurate than a tuned FNO — the reverse of the completed Poisson
   screen — and no speed ratio is admissible because the timings come from different jobs.
5. **Four new ledger rows, all from 2026-09-15**, none of which existed when this file was
   written: the prior dial (E8), per-query head refinement (E9), cheap corrections (E10), and
   the zero-start / cold-start-cap diagnostics (E11). Three of the four are negative and are
   retained as negatives.
6. **The tunability claim (C6 / E5) is rewritten.** The old wording promised a controls
   experiment whose outcome was unknown. The evidence now says something narrower and
   defensible, and §5 E5 states it in the words the data supports.
7. **A new explanatory frame, C0**, the three-layer error decomposition, now organises every
   panel. It is what makes the campaign's negatives legible rather than a list of
   disappointments.
8. **One pending user decision is now an explicit open item**: which Burgers error metric the
   paper reports (§9, item 1). Both numbers are recorded there.

Nothing previously recorded in this file is retracted; §0's on-disk table was correct on
2026-09-14 and is superseded, not withdrawn.

---

## 0. Status vocabulary, and the on-disk state

| Status label used below | Meaning |
|---|---|
| **exists** | Collected, audited run JSONs are on disk and a generator can read them today. |
| **collected (negative)** | Same, and the result went against the method. Retained, not hidden. |
| **prepared, not run** | Driver and config written; no run directory, no job, no output. |
| **not started** | Worktree or branch exists; no design document, no code, no output. |
| **not run** | Named in the plan; nothing on disk at all. |

Per-experiment on-disk status, re-checked in the sibling worktrees on 2026-09-16. Every
SHA256 in this table was computed from the file on disk during this session.

| Experiment | Status on 2026-09-14 | Status on 2026-09-16 | Evidence |
|---|---|---|---|
| E3 head ablation | not started | **exists** | `worktrees/2026-09-14-head-ablation/experiments/head-ablation/reports/2026-09-14-head-ablation.md` · SHA256 `aac6536b8a2f080f528492990d707017517b35c5d47fe64ed69144d66cbefb50` · jobs 3711424, 3711736 |
| E4 mesh ladder | not started | **exists** | `worktrees/2026-09-14-mesh-ladder/experiments/mesh-ladder/reports/2026-09-14-frozen-checkpoint-mesh-ladder.md` · SHA256 `028b0753978b1d0fc0d40ff65335cb94006ecd003f5d79a2d41ed959cc7e035a` · jobs 3711388, 3711389 |
| E5 fixed-checkpoint tuning | prepared, not run | **collected (negative)** | `worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/reports/2026-09-14-fixed-checkpoint-tuning.md` · SHA256 `53a8003a3c64ace86a4ff3cdc8418c17fd75061dac900964f22c6e3fda3cb6a5` · jobs 3711134, 3712269 |
| E6 Burgers FNO arm | prepared, not run | **exists** | `worktrees/2026-09-14-no-audit/experiments/neural-operator-audit/reports/2026-09-14-burgers-fno-baseline.md` · SHA256 `87b9a55bfc0a4ca1acfd0e105032a08c3b3a6235cb2f64f557d29bdfc8dbff4c` · jobs 3710790, 3710846 |
| E6 Poisson screen | collected (negative) | **collected (negative)** (unchanged) | `worktrees/2026-09-14-no-poisson/experiments/neural-operator-poisson/runs/matched01/matched-summary.json` · SHA256 `4e6b23d835de48b52de69dd4422be03125782477e314b8829622b565a1132ae3` · jobs 3701314, 3702473 |
| E8 prior dial | — (did not exist) | **collected (negative)** | `worktrees/2026-09-15-prior-dial/experiments/prior-dial/reports/2026-09-15-prior-dial.md` · SHA256 `3813a1c2c40b9c5e63ffb5d9d62f20db4c1c43a7a8913ec8a6a582b779100a47` · jobs 3733929, 3733930 |
| E9 per-query head refinement | — (did not exist) | **collected (negative)** | `worktrees/2026-09-15-head-refine/experiments/head-refine/reports/2026-09-15-per-query-head-refinement.md` · SHA256 `8127f6b4682a5aa38e5ac45f9e06e1645900254af86f354ca2d1fda6e140b015` · jobs 3733978, 3733979 |
| E10 cheap corrections | — (did not exist) | **exists** (mixed: target passes, knob criterion fails) | `worktrees/2026-09-15-cheap-corrections/experiments/cheap-corrections/reports/2026-09-15-cheap-corrections.md` · SHA256 `dae651250a1899e155ec71893a5579257663e129f9cd9e8c89b076e7e03f0024` · jobs 3734098, 3734084 |
| E11a Poisson zero-start | — (did not exist) | **exists** | `worktrees/2026-09-14-head-ablation/experiments/head-ablation/checks/zero-start/poisson_zero_start.json` · SHA256 `7ba36b13bef9f5dbdb02595d308d82cc13bcbb3d0c7c659563b01a95c2c444d0` · local GB10, no job |
| E11b cold-start cap sweeps | — (did not exist) | **exists** | `.../checks/cold-start-caps/poisson_cap_curve.json` SHA256 `f1c76d6fad430498d4643c8fa6d69b15535078a08f66f7e963027f758748c02e`; `.../burgers_cap_curve.json` SHA256 `77d0ab1bd5a0e1b50700e80f894db091090d3152c75eaf67bc2e2e3ef83e8fcd` · local GB10, no job |
| E1 operator parity (matched three-arm) | partly exists | **partly exists** (unchanged) | in-job assertions only; the matched comparison is **not run** |
| E2 preassembly vs fitted quadrature | not run | **not run** (unchanged) | — |
| E7 seeds + sealed cohort | not run | **not run** (unchanged) | — |

Multi-seed repetition and sealed-cohort confirmation are **not run** for any claim in this
paper. Every endpoint below is single-seed, single-checkpoint, development-cohort evidence
and must say so inline.

---

## 1. Introduction

**Framing.** Repeated PDE solves with a discretisation available. Learned nonlinear
manifolds compress the solved dimension; the question is what that compression costs and
buys once the reduced operators are assembled exactly rather than sampled — and which of the
knobs a deployed ROM appears to offer are real.

**Contributions, stated at the level the evidence can carry.**

| # | Contribution | Decisive experiment | Acceptance criterion | Evidence status |
|---|---|---|---|---|
| C0 | A three-layer decomposition of deployed error — **representation** (what the frozen bank can express), **reduction** (what the coefficient map can reach inside it), **solver** (what the iteration actually finds) — measured separately in every panel, and used to explain which levers can move which layer | E3, E5, E8, E9, E10 (each reports all three columns) | Every panel reports bank-projection error, best-found fit error and solved error as three separate numbers, never one | **exists** — the three columns are present in the collected JSONs of all five cells |
| C1 | A separable frozen decoder whose bank admits both exact preassembly of the linear and polynomial weak operators and sampled evaluation of what resists it | E1 (operator parity) | Preassembled residual **and** Jacobian match the full-grid weak evaluation to the recorded floating-point tolerance at every mesh, on decoded states that satisfy the sign condition | **exists** — the assertion is executed in-job (`run_pilot.assemble`, heat; `b2d_tensor_common` gates T0/TB, Burgers) |
| C2 | A precisely stated weak least-squares solve with recorded stationarity and exit reasons, replacing the conflation of tangent Galerkin with residual least squares | E0 (definition + per-query diagnostics) | Every reported solve carries its exit reason and its normalized gradient; stalls are never reported as converged | **exists** — implemented in all cells; the diagnostics are in the collected JSONs |
| C3 | Removing fitted empirical quadrature from the repeated reduced evaluation, and the conditions under which that is exact | E1, E2 | Stated in E1/E2 below | **partly exists** (parity); **not run** (matched offline/online cost comparison at one frozen checkpoint) |
| C4 | Characterisation of when the learned coefficient nonlinearity pays at matched reduced dimension | E3 (head ablation) | Stated in E3 | **exists** — it pays per dimension on both PDEs; on Poisson a POD rank of $8k$ buys it back at the same cost |
| C5 | Cached reduced evaluation has no explicit mesh loop; the complete query does not inherit that property, and neither confers a crossover against an efficient FOM in 2D | E4 (mesh ladder) | Stated in E4 | **exists** — flat cached cost measured; **no crossover at any rung on either PDE** |
| C6 | A single trained decoder exposes a **monotone family of offline-prepared inference-time operating points** — correction rungs, empirical-quadrature rules, stopping tolerance — selected at run time without retraining | E5, E8, E9, E10, E11 | Stated in E5 | **exists, with the negative half reported**: the family dominates the POD rank ladder on Burgers; it does **not** beat an efficient FOM; Poisson's rungs are nearly free and Poisson is the correctness case |

**Explicit non-claims, in the introduction and not deferred to a limitations paragraph:**
no speedup at matched accuracy against a tuned classical solver; no dominance over neural
operators; no Pareto frontier; no continuum error bound; no convergence theory; and — new
since 2026-09-14 — **no claim that the operating-point family is an accuracy/cost frontier
in the pre-registered sense**: on Burgers its converged non-dominated set spans 3.61× in cost
but only 1.41× in error (E10), short of the declared 2× error span.

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
| 3.9 Inference-time operating points | the offline-prepared family and everything held fixed while selecting within it | `accuracy_paths.make_rom`, `tuning-config.json`, `cheap_ladder.py`, `varpro.py` |
| 3.10 Intrusive vs non-intrusive | stated before any comparison | — |
| 3.11 Limits | seven explicit non-guarantees | — |

**Methods work still owed** (not a status change, a to-do): §3.9 was written as three solver
controls. It must be widened to the family actually measured — correction rank $q$ with the
corrections eliminated from the nonlinear iteration, the test count $M$, the stored EQ rule,
the stopping tolerance and the iteration cap — and must state which of these are *offline*
preparations selected at run time (the EQ rules and the correction directions) versus pure
runtime numbers (tolerance, cap).

---

## 4. Experimental setup
Problem families, discretisations, data generation and seeds, the query contract (host
input in, requested dense fields out, everything between it charged), the timing protocol
(per-block GPU burn-in, device synchronisation around every measured region, medians over
retained repetitions, never a cross-job ratio), and the named FOM comparators: **direct
DST** wherever the operator is separable, tolerance-tuned **CG**, and
**Newton–BiCGStab** / spectral Burgers solvers at several tolerances.

**Now measurable and previously not:** every completed cell interleaves its full-order
controls in the *same* job as the ROM arms, so the FOM comparisons in E3, E4, E5, E8 and E10
are same-job. The one place this is violated is any comparison *between* cells — notably the
E6 FNO timings against the E5 ROM timings — and the paper must say so at the point of use.

---

## 5. Results

### E0 — Solver definition and per-query diagnostics (supports C2)
*What it is.* Not a separate run: a reporting discipline applied to every panel. Each
solve emits its exit reason, its normalized gradient, its residual, and (Poisson) the rank
and singular values of the projected Jacobian and the backward error of the linear
recovery.
*Acceptance criterion.* No table row may report a physical error without its exit-reason
distribution beside it; stalls are labelled `stalled`, never `converged`.
*Status.* **exists.**
*Correction carried forward from E3 (2026-09-14).* The normalized-gradient stationarity test
is scale invariant and can only fall below tolerance once the residual is orthogonal to the
reduced tangent space. An arm whose reduced fit is *attainable* drives the residual to
round-off while that ratio stays of order one, and exits by the small-step rule with a better
fit and a worse-looking stationarity number. **The stationarity column is therefore not a
quality ranking across arms of different reduced dimension**; the separate completion status
is the honest one. Both are recorded for every invocation.

### E1 — Operator parity: preassembled vs full-grid weak evaluation (supports C1, C3)
*Arms.* full-grid weak projection · corrected EQ · exact precomputed algebra, at one
frozen checkpoint per PDE, with tests, row scaling, initialisation, time discretisation,
stopping and output contract held fixed.
*Acceptance criterion.* (i) For the linear terms, parity at the recorded floating-point
tolerance — the in-job assertion already demands $<10^{-11}$ relative for both operator
and Jacobian. (ii) For the quadratic term, parity at that tolerance **on sign-satisfying
states**, with the departure on sign-violating states measured and reported, never assumed
small.
*Status.* Linear parity **exists** (asserted in-job). Quadratic parity on positive states
**exists** (`b2d_tensor` gate T0). The matched three-arm comparison at one frozen checkpoint
with charged offline cost is **not run**.

### E2 — Preassembly against a fitted quadrature rule, same checkpoint (supports C3)
*Arms.* exact tensor · EQ at each stored $m$ · full-grid oracle.
*Acceptance criterion.* A claim that preassembly is preferable requires **both** no
physical-error regression at the same checkpoint **and** a stated total-cost account
including offline assembly.
*Status.* **not run.** Two pieces of the cost account now exist and should be reused rather
than remeasured: E5 gives the accuracy-neutrality of quadrature (m=256, m=512, m=1024 and
the quadrature-free full-grid control span only **0.147890 percentage points** of worst
error, while the full grid costs **5.36×** the m=512 arm), and E10 gives the per-rung offline
NNLS fit cost (13.4 s to 307.4 s per rule, each charged outside every query timing).
Sources: `2026-09-14-fixed-checkpoint-tuning.md` (SHA256 `53a8003a…`);
`2026-09-15-cheap-corrections.md` (SHA256 `dae65125…`).

### E3 — Head ablation at matched latent dimension (supports C0, C4)
*Arms.* linear head · quadratic head · neural head · unrestricted bank coefficients ·
classical POD-LSPG at ranks 8/16/32/64/128 — all at matched $k$, matched bank where
applicable, the same weak objective, test modes, time discretisation, initializer policy,
stopping rule and output contract.
*Acceptance criterion.* The neural head is preferred only if, at matched $k$ and matched
cost, it reduces deployed error against **both** the quadratic head and a correctly solved
linear POD-LSPG on at least one family, with the failing families reported.
*Status.* **exists.** Jobs `3711424` (Burgers, A100 80 GB) and `3711736` (Poisson).
Report: `2026-09-14-head-ablation.md`, SHA256 `aac6536b8a2f080f528492990d707017517b35c5d47fe64ed69144d66cbefb50`.
Machine-readable source for every number below: `checks/abl01-audit.json` → `checks.arm_table`
and `checks/pabl01-audit.json` → `checks.arm_table` in the head-ablation worktree.

*Burgers, 256 intervals, worst same-grid error / median complete-query GPU ms* — neural head
(EQ) **2.562872 % at 47.649 ms**; linear head 56.929573 % at 19.948 ms; quadratic head
31.827588 % at 28.027 ms; POD-16 61.650255 % at 16.328 ms; POD-128 10.119784 % at
328.344 ms; unrestricted bank coefficients 0.602667 % at 2417.608 ms. **No POD rank up to
$8k$ matches the neural head**, and the POD rungs are run with exact dense advection, which
favours them, so the matching rank is a conservative lower bound.

*Poisson, 1024 intervals, worst error / median query ms* — neural head **6.092722 % at
5.843 ms**; linear head 17.296602 % at 5.163 ms; quadratic head 16.630250 % at 5.774 ms;
POD-16 20.054797 % at 4.659 ms; **POD-128 4.345242 % at 5.930 ms**; unrestricted bank
2.327435 % at 5.995 ms; direct DST 0.000735 % at 4.341 ms. The head wins per dimension and
loses per millisecond: on Poisson the query is dominated by projection and decode, not by the
reduced solve, so an eightfold larger linear subspace is nearly free and is *more* accurate.
**This is the honest split and it is a contribution, not a caveat.**

*Three layers, read straight off the same table.* Burgers: bank projection floor
**0.391845 %**, best-found nonlinear fit **2.544663 %**, solved **2.562872 %** — the
representation layer is 6.5× below the reduction layer, and the solver layer costs 0.018
percentage points. Poisson: floor **2.314808 %**, best-found **6.092634 %**, solved
**6.092722 %**. On both PDEs the gap that matters is *reduction*, not solver.

### E4 — Frozen-checkpoint mesh ladder, 64–1024 (supports C5)
*Arms.* one frozen checkpoint per PDE, weights unchanged across meshes, reduced dimensions
and solver policy fixed; against direct **DST** (Poisson) and same-job spectral/Newton
comparators (Burgers), every arm interleaved in **one job per PDE on one GPU**.
*Acceptance criterion.* The mesh-independence claim is about the **cached reduced
evaluation's operation count** and is accepted only if stated that way: flat reduced cost
does **not** license a claim of flat complete-query cost, and the input/output components
must be shown growing. If accuracy degrades with refinement at fixed $(k,R,M)$, that is
reported as the cost of the claim.
*Status.* **exists.** Jobs `3711388` (Burgers, A100-PCIE-40GB) and `3711389` (Poisson,
A100 80 GB). Report `2026-09-14-frozen-checkpoint-mesh-ladder.md`, SHA256
`028b0753978b1d0fc0d40ff65335cb94006ecd003f5d79a2d41ed959cc7e035a`.

- Burgers cached reduced solve **44.156 → 44.720 → 44.730 → 45.116 → 43.577 ms** across
  64/128/256/512/1024 intervals (0.987× end to end) while interior unknowns grow **264×**.
  Complete device query 48.504 → 50.093 ms (1.033×). Worst physical error **improves** under
  frozen-weight transfer, 10.8570 % at 64 to 3.8847 % at 1024, passing the 5 % target from
  256 upward.
- Poisson cached reduced solve **2.094 → 2.021 ms** (0.965×); complete device query 2.252 →
  3.004 ms (1.334×); worst physical error pinned at its representation floor,
  6.1119 % → 6.1106 %, missing the 5 % target at **every** rung, so none of its timing ratios
  is a qualifying speedup.
- **No crossover at any rung on either PDE.** FOM/ROM by rung, Burgers: 0.265×, 0.320×,
  0.325×, 0.495×, 0.443×. Poisson against direct DST: 0.063×, 0.065×, 0.067×, 0.079×,
  0.112×.
- The Poisson ROM-vs-same-grid-FOM discrepancy at 1024 intervals equals its physical error to
  four figures, so **the error is reduction error, not discretisation error** — the C0 frame
  again.

*Explicitly not retracted, explicitly not a headline:* the historical 10× and larger "wins"
against over-solved Newton or unpreconditioned CG at 1e-6 are the inflated comparison
`AGENTS.md` forbids. They are recorded in the lab log and are not carried into this paper.

### E5 — Inference-time operating points at a fixed checkpoint (supports C0, C6)

**The claim, in the words the evidence supports.** *A single trained decoder exposes a
monotone family of offline-prepared inference-time operating points — correction rungs,
empirical-quadrature rules, stopping tolerance — selected at run time without retraining.
The family dominates the POD rank ladder on Burgers. It does not beat an efficient FOM.
Poisson's rungs are nearly free, and Poisson is the correctness case.*

That sentence is the section. Everything below is the evidence for each of its four clauses,
and the criteria that the family **fails**.

*Acceptance criteria, pre-registered in each cell's `DESIGN.md` before any implementation:*
error monotone in the knob; at least 3 non-dominated points spanning $\ge 2\times$ in cost
**and** $\ge 2\times$ in error; none early-stopped. Report the measured relation, its
saturation and its stalls. Do **not** claim a frontier.

**Clause 1 — a monotone family exists, and it is the correction rank $q$.** E10 (jobs
`3734098`, `3734084`; report SHA256 `dae651250a1899e155ec71893a5579257663e129f9cd9e8c89b076e7e03f0024`).
With the corrections eliminated from the nonlinear iteration (block-damped variant), the test
count decoupled from $q$, and one EQ rule fitted per rung, the Burgers non-dominated
**converged** set is five points:

| arm | $q$ | worst same-grid % | median GPU ms |
|---|---:|---:|---:|
| `q0_m4_eq_varpro` | 0 | 2.5629 | 49.128 |
| `q16_m256_eq_block` | 16 | 2.4806 | 83.235 |
| `q32_m256_eq_block` | 32 | 2.3534 | 98.287 |
| `q64_m256_eq_block` | 64 | 2.1489 | 121.765 |
| `q128_m256_eq_block` | 128 | 1.8116 | 177.291 |

The pre-registered target — $q=64$ **converged** at $\le 3\times$ the $q=0$ cost — **passes**
at ratio **2.479**. The knob criterion **fails on the error span**: 5 non-dominated converged
points (≥ 3 ✓) spanning **3.61×** in cost (≥ 2× ✓) but only **1.41×** in error (≥ 2× ✗).
$q=256$ and $q=512$ still do not converge under the shared rule, and the $q=512$
empirical-quadrature arm is visibly broken (27.8120 % against 0.6027 % for its dense twin),
its rule walltime-truncated. Closing the error span needs converged rungs that reach the
0.3918 % bank-projection floor, which is where $q\ge 256$ would have to land.

**Clause 2 — the family dominates the POD rank ladder on Burgers.** Cross-job, therefore
**provisional for cost** (E10 ran on an A100 80 GB, E3's `abl01` on an A100 80 GB in a
different allocation; error is same-metric, same-cohort and comparable, cost is not). At
matched worst same-grid error the POD ladder needs rank 128 to reach 10.119784 % at
328.344 ms, while every rung of the correction family sits below 2.6 % at under 180 ms. The
**same-job** version of this panel is the one the paper must print and it has not been run.

**Clause 3 — it does not beat an efficient FOM.** Same-job, so this one is clean. In E10 the
full-order `nt1e-2` control reaches 3.7127 % worst same-grid at 16.264 ms and `fft_tight`
0.0000 % at 90.917 ms; **no rung of the ladder dominates either on both axes**. In E5's own
held-out pass (32 validation cases, jobs 3711134/3712269) the best tuned setting
`m512_gtol0.001` reaches **6.701206 % worst error at 41.641916 ms** while
`same_nt1e-4_dt005` reaches **6.170513 % at 22.389024 ms** — better on median error, worst
error and cost simultaneously.

**Clause 4 — Poisson's rungs are nearly free, and Poisson is the correctness case.** E10
Poisson (job `3734084`, 1024 intervals, twelve development sources): because the weak
residual is linear in the coefficients the corrections are eliminated exactly and the
nonlinear iteration stays 16-dimensional at every $q$. On the fixed `m256` rule worst
physical error falls monotonically **6.0927 % ($q=0$) → 4.1781 % ($q=64$)** for a cost factor
of **1.08×** (3.8285 → 4.1173 median device ms). The $q=128$ endpoint is degenerate ($q=R$, the full bank) and is
flagged `all_solver_valid` false by construction, so the informative range is $q\le 64$. The
same-job `dst_direct` costs 0.3323 ms with 0.000735 % error: Poisson is where the algebra is
**correct**, not where the method is fast, and the paper should use it that way.

**What the solver-side controls actually do (E5's own cell, negative).** Report SHA256
`53a8003a3c64ace86a4ff3cdc8418c17fd75061dac900964f22c6e3fda3cb6a5`.
- Loosening the evolution stopping tolerance 1e-6 → 1e-3 removes **22.847 %** of GPU time and
  changes worst physical error by **+0.000012 percentage points**. Cost is tunable; accuracy
  is not.
- The iteration cap is a **cliff, not a dial**: cap 8 → 4.442939 % worst; caps 4 and 2 → no
  accepted step at all and the trajectory collapses to 74.896500 %.
- Quadrature is a **large cost lever at zero accuracy cost**: m=256/512/1024/full-grid span
  0.147890 pp of worst error while the full grid costs **5.36×** the m=512 arm. E10 measures
  the same effect per rung at **4.66×–6.42×** across the converged pairs.
- The reason accuracy is flat is **representation**: the decoder's own compression of the
  supplied initial field is already 1.867068 % at worst, comparable to the whole trajectory
  error.
- **Retracted inside that cell:** the calibration-stage reading that `same_nt1e-2_dt005` is
  the accuracy bar. On the 32 held-out cases it reaches **35.357022 %** worst error with 3
  cases above 10 %, despite a 1.135606 % median. Its loose Newton tolerance is not reliable
  across the wider family. The dominance conclusion is unchanged, because other controls
  dominate every tuned setting on the held-out cases too.
- **Recorded measurement limit:** two full-order controls re-timed in the second pass of the
  same job drifted **+13.335 %** and **+15.936 %** in median GPU time, so arms measured in
  different passes must not be separated more finely than that.

**Three levers that are NOT knobs — reported as negatives, because that is the paper.**

| Lever | Cell | Verdict | Why it fails |
|---|---|---|---|
| Trust in the neural prior ($\lambda$ on the bank-coefficient penalty) | E8 | **not a knob on Burgers, on either metric**; on Poisson an accuracy lever that is *nearly free* but still fails the criterion | Burgers all-times metric: 1 non-dominated point, 1.00× cost and 1.00× error span — the $t=0$ output is the head's compression of the supplied field and cannot depend on $\lambda$ at all. Burgers evolved-time metric: 6 non-dominated points spanning 25.33× cost but only **1.23×** error, and the points that buy accuracy do not converge. Poisson at 1024: 3 non-dominated points, error span **2.62×** (passes) but cost span **1.16×** (fails) |
| Per-query refinement of the frozen head's own weights | E9 | **not a knob on either PDE**, all six variant × anchor ladders | Burgers V1/loose and V1/tight: non-monotone, 2 non-dominated points, 1.11× cost and ≤1.05× error. Burgers V2/loose and V2/tight: non-monotone, 2 points, ≤1.00× error. Poisson V1/loose: 6 points spanning 3.54× cost and 2.16× error but **non-monotone**; V1/tight fails all four legs |
| Where the solve starts (zero start vs nearest training code) | E11a | **changes cost, never the answer** | All four Poisson starts give worst 6.0931 % / median 1.2421 % at 1024 intervals, per-case agreement $\le$ 1e-9, all stationary; the zero start takes ~3× the iterations and ~2× the local wall time |

**The old paper's knob regime, reproduced and dominated (E11b).** From $z=0$ the Poisson LM
iteration cap *does* trace a monotone curve — cap 6: 91.9 % worst / 54.4 % median; cap 8:
30.0 / 5.96; cap 10: 8.19 / 2.10; cap 12: 6.09 / 1.46; cap ≥ 20 converged 6.09 / 1.24 — which
is the regime the old manuscript's Tables t2a/t2b lived in. **Every point on it is dominated
by starting well**: from the nearest training code, cap 2 reaches what the zero start needs
cap 12 for, at a third of the cost, and every sub-convergence point is early-stopped. On
Burgers the warm-start cap gives a three-point step (3.80 → 2.39 → 1.90 % evolved) inside a
12 % cost band, and cold starts only add wasted iterations (88 % at every cap up to 20,
3.07 % only at cap 180 costing 34×). GB10 wall times; only ratios transfer. This is the
single cleanest explanation of why the old paper saw a frontier and this one does not.

**Also corrected here (E11b):** the earlier reading that the initial fit is the dominant fixed
cost. Measured on Burgers, the initial fit is ~12 % of the local query time (cap 1: 1100 ms
vs cap 400: 1249 ms); the fixed cost is **decode plus per-step overhead**.

**Structural point that must not be misread (E8, E9).** For the unrefined Burgers head the
maximum over the six output times is attained at $t=0$ — the model's own compression of the
supplied initial field — at 2.5629 % against 1.9002 % over the evolved times. Any arm whose
initializer starts the correction at zero therefore *cannot* move the worst-over-all-times
number, and several rows sit at exactly the $q=0$ / $n=0$ value for that structural reason,
not because the lever did nothing. The correction ladder (E10) moves that metric only because
its initializer fits the augmented vector too. **The two cells' all-times headline numbers
are therefore not comparable**, and this is recorded as the main interpretive correction of
the 2026-09-15 round.

### E6 — Common-data comparison against a neural operator (supports the framing)
*Arms.* FNO at several capacities against the ROM, **identical dataset and split**, index
hashes recorded; timing only in a single job containing both.
*Acceptance criterion.* Report the outcome whichever way it falls, with the unmatched
training/tuning budgets stated. No accuracy-per-cost verdict without a paired latency panel
in one job.
*Status.* **exists** on both PDEs, and the two go opposite ways.

- **Poisson — collected (negative for the ROM).** `matched01/matched-summary.json`, SHA256
  `4e6b23d835de48b52de69dd4422be03125782477e314b8829622b565a1132ae3`, jobs 3701314 /
  3702473. The fresh ROMs generalise worse than the FNOs on one-seed common data; neither ROM
  arm is selected; no paired latency panel exists.
- **Burgers — the ROM is the more accurate model.** Jobs 3710790 / 3710846, report SHA256
  `87b9a55bfc0a4ca1acfd0e105032a08c3b3a6235cb2f64f557d29bdfc8dbff4c`. Best
  validation-selected FNO (`fno-large`, 17 877 317 real parameters): **1.805424 % median /
  6.382471 % worst** on 32 held-out cases, 7.314731 ms device query. On the matched
  eight-case cohort: FNO **2.482867 %** worst against ROM **1.867068 %** and efficient FOM
  `same_nt1e-2_dt005` **0.997803 %**.
- **No speed ratio is stated anywhere**, because the FNO timings are same-job only and the
  ROM/FOM timings come from job 3702709 in another allocation. The interleaved same-job
  ROM/FNO/FOM panel is the only admissible route to a speed statement and is **not run**.
- The FNO is a *direct multi-time* operator: one forward pass emits the five evolved fields
  and returns the supplied initial state bitwise, so it has no rollout accumulation and its
  output time set cannot be extended. That asymmetry must be stated beside the comparison.

### E7 — Multi-seed repetition and sealed-cohort confirmation
*Acceptance criterion.* Every headline number repeated over $\ge 3$ training seeds, then a
final evaluation on a cohort opened once, with all choices frozen beforehand.
*Status.* **not run.** Until it is, every reported endpoint is single-seed development
evidence and must say so inline, not only in a preamble.

### E8 — The prior dial (supports C0, C6; negative)
*What it varies.* $\min_{z,c}\lVert r_w(c)\rVert^2+\lambda\lVert R_G(c-h_\theta(z))\rVert^2$,
so $\lambda=\infty$ is the pure head and $\lambda\to0$ with $M\ge R$ is the free bank.
*Status.* **collected (negative).** Jobs `3733929`, `3733930`; report SHA256
`3813a1c2c40b9c5e63ffb5d9d62f20db4c1c43a7a8913ec8a6a582b779100a47`. Verdicts in the E5 table
above. Its lasting content is two structural facts: (i) $\lambda$ acts entirely in the
reduction/solver layer, because the penalty restricts the **solver** and not the reachable
set, so layers 1 and 2 coincide at every finite $\lambda$; (ii) with the bank free the weak
residual has only $M$ equations, so at $M<R$ the small-$\lambda$ end is a *regularized
underdetermined* solve and not the free bank — only $M>R$ reaches that limit.

### E9 — Per-query refinement of the frozen head's weights (supports C6; negative)
*What it varies.* $n$ Adam steps on $\theta$ inside the timed query, anchored to the trained
weights with weight $\mu$; V1 once against the supplied initial field, V2 after every step.
*Status.* **collected (negative)** on both PDEs, all six ladders. Jobs `3733978`, `3733979`;
report SHA256 `8127f6b4682a5aa38e5ac45f9e06e1645900254af86f354ca2d1fda6e140b015`. Five
fidelity gates passed, including an independent NumPy re-decode of every saved refined weight
vector. Recorded deviation: the anchor weights were moved from $\{10^{-3},10\}$ to
$\{10^{2},10^{5}\}$ after the local smoke and before any evaluation run, on gradient scales
alone, and the Poisson step size sits at the edge of its pre-registered grid.

### E10 — Cheap corrections (supports C0, C6; the target passes, the knob criterion fails)
*What it changes, each isolated:* eliminate the correction coefficients from the nonlinear
iteration; decouple the test count from $q$; fit one EQ rule per rung.
*Status.* **exists.** Jobs `3734098`, `3734084`; report SHA256
`dae651250a1899e155ec71893a5579257663e129f9cd9e8c89b076e7e03f0024`.
*What each change bought.* Change 1 buys **convergence** — at matched $q$, test count and
quadrature it leaves the error unchanged to the printed digits but turns the audited joint
solver's budget exits at $q=64$ and $q=128$ into clean stationary exits, at the same median
3 iterations per step the $q=0$ rung needs. Change 2 is a **cost** lever only at the larger
rungs and is never an accuracy lever above the identifiability threshold; at $q=0$ too few
tests visibly costs accuracy (3.8946 % against 2.5629 %). Change 3 buys the **speed**:
4.66×–6.42× over the paired dense arm at identical error.
*Negatives recorded inside the cell, against its own hypotheses.* (i) Flattening the doubly
vectorised LM `while_loop` in the offline direction fit did **not** save time — it cost about
124 s more on this node while producing a bitwise identical matrix, so the
"compilation-bound setup" reading of the audited ladder is wrong as stated. (ii)
`directions_sha256` and the retained reference fields do **not** match `qlad01` bitwise
across jobs (different A100 SKUs); the scientific gates that carry the weight all pass to
1e-12 and 9.4e-07, and the two hash checks are reported as failed rather than relaxed.
(iii) Plain variable projection, predeclared as the primary solver, is the **slowest** thing
in the sweep at large $q$; the block-damped variant was added after the local smoke and
pre-registered before submission, with both run on the cheap rungs so the comparison is in
the data.

### E11 — Where the solve starts (supports C6; both negative for tunability)
*E11a, Poisson zero start.* Local GB10 diagnostic, no cluster job, no source change.
`checks/zero-start/poisson_zero_start.json`, SHA256
`7ba36b13bef9f5dbdb02595d308d82cc13bcbb3d0c7c659563b01a95c2c444d0`. Four starts — nearest
training code, mean code, $z=0$ with the trust radius, $z=0$ without — give **identical**
answers at 64, 256 and 1024 intervals. The head's surface has a single basin for the weak
objective on this checkpoint and cohort; the start changes only how many steps it takes.
*E11b, cold-start iteration-cap sweeps.* `checks/cold-start-caps/poisson_cap_curve.json`
(SHA256 `f1c76d6fad430498d4643c8fa6d69b15535078a08f66f7e963027f758748c02e`) and
`burgers_cap_curve.json` (SHA256 `77d0ab1bd5a0e1b50700e80f894db091090d3152c75eaf67bc2e2e3ef83e8fcd`).
Numbers and reading in E5 above. **This is the experiment that explains the old paper.**

---

## 6. Claim-to-figure allocation

| Fig. | Content | Supports | Generator (under `paper/figures/`) | Reads | Status |
|---|---|---|---|---|---|
| F1 | Architecture and data flow: trained / frozen / solved | C1, C2 | `architecture.mmd` (hand-authored; no numbers) | — | **exists** |
| F2 | Operator parity: preassembled vs full-grid residual and Jacobian, per mesh; positivity audit beside it | C1, C3 | `gen_fig_operator_parity.py` | E1 JSONs; in-job assertion records | parity **exists**, matched arms **not run** |
| F3 | Exact tensor vs fitted EQ at one frozen checkpoint: physical error, online cost, offline cost, memory | C3 | `gen_fig_preassembly_vs_eq.py` | E2 JSONs; partial substitutes in E5/E10 (see E2) | **not run** |
| F4 | Head ablation at matched $k$: bank projection, best-found fit, solved error, and cost — three stacked panels so the C0 layers are not conflated | C0, C4 | `gen_fig_head_ablation.py` | `checks/abl01-audit.json`, `checks/pabl01-audit.json` → `checks.arm_table` | **exists** — generator **not written** |
| F5 | Mesh ladder: cached reduced cost, complete query cost with components, iteration counts, error — against DST and the same-job FOM | C5 | `gen_fig_mesh_ladder.py` | mesh-ladder aggregates JSON (job 3711388/3711389) | **exists** — generator **not written** |
| F6 | **The operating-point family**: error against GPU ms with correction rungs labelled, the POD rank ladder faint behind it, FOM verticals, converged vs early-stopped encoded as filled vs hollow | C0, C6 | **`gen_fig_tunability_family.py`** | `cheap-corrections` `artifacts/{cclad01,ccpoi01}/result.json` + `checks/{cclad01,ccpoi01}-audit.json`; `head-ablation` `checks/{abl01,pabl01}-audit.json`; `no-burgers` `artifacts/tuning02/output-index.json` | **exists, PROVISIONAL for cost** — written this session; cross-job, so the same-job envelope is still owed |
| F7 | Common-data ROM vs FNO on both PDEs, with the missing latency panel shown as absent rather than omitted | framing | `gen_fig_common_data_operator.py` | `matched01/matched-summary.json`; FNO baseline JSONs | **exists** — generator **not written** |
| F8 | The old paper's knob regime: cold-start cap curve against the warm-start points that dominate it | C6 | `gen_fig_cold_start_regime.py` | `checks/cold-start-caps/*.json`, `checks/zero-start/*.json` | **exists** — generator **not written** |
| T1 | Per-cell configuration: $N$, $n$, $k$, $R$, $M$, $m$, $\Delta t$, checkpoint SHA, job id, GPU, backend, precision | reproducibility | `gen_tab_configuration.py` | every run JSON's metadata block | **exists** |
| T2 | Exit-reason distribution per panel | C2 | `gen_tab_exit_reasons.py` | every run JSON's solver records | **exists** |

Every generator writes a paired `.json` beside its figure so the figure can be re-derived
and checked. No figure may be produced from a number typed into a script. **F6 is the only
generator that exists today**; it is labelled PROVISIONAL on its cost axis because it reads
three different jobs.

---

## 7. What must happen before submission, in order

1. **The same-job envelope panel**: correction rungs, the POD rank ladder and the full-order
   controls interleaved in one job on one GPU. F6 is cross-job and cannot carry the C6 claim
   as printed. (Owned by the `b-ladder-top` lane as of 2026-09-16.)
2. **Settle the Burgers metric** (§9 item 1) before any number is typeset.
3. Write the six remaining figure generators listed in §6.
4. Run E1/E2 — the matched three-arm operator comparison at one frozen checkpoint. This is
   the paper's spine and remains the cheapest missing piece.
5. Complete E6's paired latency panel, or drop the accuracy-per-cost framing for that
   comparison.
6. E7: seeds, then the sealed cohort. Only then may any endpoint drop its single-seed label.
7. Rewrite `methods.tex` §3.9 to the family actually measured (see §3 above).

---

## 8. Explicit non-claims, restated for the introduction

- No speedup at matched accuracy against a tuned classical solver, on any PDE, at any mesh.
- No dominance over neural operators: the ROM wins on Burgers accuracy, loses on Poisson.
- No Pareto frontier. The operating-point family fails the pre-registered error-span test.
- No claim that any inference-time lever other than correction rank moves accuracy.
- No continuum error bound, no convergence theory, no cold-start GN convergence guarantee.
- Every endpoint is single-seed, single-checkpoint, development-cohort.

---

## 9. Open items requiring the user's decision

1. **The Burgers error metric — pending, and it changes the headline.** The two candidates,
   both from audited runs, both at $q=0$ / $M=64$ / EQ on the frozen `sep_hfit_dense_mid_N256_dense.pkl`
   checkpoint at 256 intervals:
   - **(a) worst over all six output times = 2.5629 %.** This is the number every 2026-09-14/15
     table prints. Its maximum is attained at $t=0$, which is the decoder's compression of the
     *supplied* initial field, not a dynamics error. Because arm (a)'s initializer starts the
     correction at zero, this metric is **pinned by construction** for the prior dial and for
     every V2 refinement arm — no setting of those levers can move it
     (`2026-09-15-prior-dial.md`, SHA256 `3813a1c2…`, confirmed to 0.00e+00 relative across
     every $\lambda$, test count and quadrature).
   - **(b) worst over the evolved times = 1.9002 %, with the $t=0$ compression reported
     separately** (same source table, `M64_eq_laminf` row). This is what the dynamics
     actually cost, and it is the metric under which the prior dial shows 6 non-dominated
     points and the cold-start cap shows its three-point step.
   The recommendation of this session is **(b) with (a) printed beside it**, because (a)
   silently conflates compression with evolution and has already caused one cross-cell
   comparison to be withdrawn. **The decision is the user's and nothing is typeset until it
   is made.**
2. **Paper framing**: diagnostic ("what is and is not tunable") versus reviving the mechanism
   story with an engineering fix. The evidence supports the first as written; the second
   needs the enrichment-elimination job.
3. **Worktree merges**: eleven `2026-09-1x` worktrees are unmerged. Ask, do not merge.
4. **GitHub**: see `tools/mirror-code-only.sh` and the mirror plan for the code-only route,
   measured this session at a 170.07 MiB pack against the repository's 212 GiB.

---

## 10. Glossary

Written for a reader who knows none of this project's vocabulary.

- **PDE / FOM / ROM / NM-ROM** — the partial differential equation being solved; the full
  discretised solver; a solver using far fewer unknowns; a ROM whose unknowns parameterise a
  curved (nonlinear) set of states rather than a flat subspace.
- **Decoder / spatial bank / coefficient map / latent coordinates** — the map from few
  numbers to a full field; the fixed spatial functions it combines ($G$, width $R$); the
  network producing their weights ($h_\theta$); the few unknowns solved per query ($z$,
  dimension $k$).
- **The three layers (representation / reduction / solver)** — the error you could not
  remove even with a perfect coefficient map, because the bank cannot express the answer
  (measured as *bank projection*); the extra error from restricting to what the coefficient
  map can reach (measured as *best-found fit*, an oracle over many restarts); the extra error
  the actual iteration leaves behind (*solved* minus *best-found*). Reported as three
  separate numbers everywhere, because a lever can only move the layer it acts on.
- **Weak tests / row scaling / test count $M$** — smooth functions used to average the
  equation error; the diagonal weights setting the relative importance of those averaged
  equations; how many of them are used. All three change *which* solution is selected, so all
  three are part of the method.
- **Residual / Jacobian / stationarity** — how badly the equation is violated; its first
  derivatives with respect to the latent coordinates; the condition that the gradient of
  the least-squares objective vanishes. Stationarity is *not* a guarantee of physical
  accuracy, and across arms of different reduced dimension it is not even a quality ranking.
- **Stall / early-stopped / budget exit / converged** — a solve that ran out of iteration
  budget, took a vanishing step, or hit the damping limit, rather than meeting its tolerance;
  "converged" is reserved for solves that met the shared stopping rule. Figures encode this
  as filled versus hollow markers.
- **Galerkin / least-squares Petrov–Galerkin (LSPG)** — setting projections of the residual
  to zero; minimising the size of a specified residual in a specified norm. They are
  different conditions and give different equations.
- **Empirical quadrature (EQ) / NNLS / rule / support / $m$** — approximating a sum over all
  mesh nodes by a weighted sum over a few; the non-negative least squares fit that chooses
  the weights offline; one fitted instance of that; the nodes it actually uses; the requested
  node count. $m$ is quadrature nodes, $M$ is weak equations; the code requires $m\ge 4M$ and
  $M\ge 4k$.
- **Correction rank $q$ / rung / bank directions** — extra fixed linear directions solved
  alongside the neural head's output, $u = G(h_\theta(z) + C_q y)$; one value of $q$ on the
  ladder; the columns of $C_q$, chosen offline by POD of the head's own residual.
- **Block-damped / variable projection / joint / alternating** — four ways to solve for $z$
  and $y$ together: damp the two blocks separately inside one Levenberg–Marquardt step;
  eliminate $y$ in closed form and iterate on $z$ alone; treat $(z,y)$ as one vector;
  alternate between them. Block-damped is the one kept.
- **Prior dial ($\lambda$)** — a penalty pulling the bank coefficients towards the neural
  head's own prediction; $\lambda=\infty$ is the pure head, $\lambda\to0$ the free bank.
- **Head refinement ($n$, $\mu$)** — taking $n$ gradient steps on the *network's own weights*
  inside a single query, anchored to the trained weights with strength $\mu$.
- **Warm start / cold start / zero start / nearest code** — starting a solve from the
  previous time step; from a fixed query-independent point; from $z=0$; from the training
  latent code whose weak prediction is closest. Only query-independent starts are legitimate.
- **Iteration cap / stopping tolerance / trust radius** — the maximum number of solver steps
  allowed; the gradient bar below which the solve stops; the maximum step length.
- **Same-grid error vs reference error vs physical error** — discrepancy against the
  converged full-order solve on the *same* mesh (isolates the reduction); discrepancy against
  a much finer reference (contains this mesh's discretisation error too); the cell's own
  declared accuracy metric. Which one is quoted always matters.
- **Worst over all times vs worst over evolved times** — the maximum error over every
  requested output time, including $t=0$ (which is the decoder's compression of the *given*
  initial field); the same maximum excluding $t=0$. See §9 item 1.
- **Held-out / development cohort / sealed cohort** — cases not used for fitting; cases
  already opened and therefore usable for choosing settings; cases opened once at the end,
  after every choice is frozen.
- **Tripwire / gate / acceptance criterion / pre-registered** — a condition declared before a
  run that decides whether its result may be used; a check that must pass; the bar a claim
  must clear; declared in `DESIGN.md` before implementation, not after seeing the numbers.
- **Non-dominated / dominates / cost span / error span** — a point that nothing else beats on
  both axes; beating on both axes; the ratio of the most to the least expensive non-dominated
  point; the same ratio for error. The knob criterion asks for $\ge 3$ non-dominated points
  and $\ge 2\times$ on **both** spans.
- **DST / CG / Newton–BiCGStab / `fft_tight` / `nt1e-2`** — a direct transform solver
  available when the operator separates on a rectangle (the fastest comparator where it
  applies); an iterative solver for symmetric positive-definite systems; the nonlinear
  iteration used for Burgers with a preconditioned Krylov inner solve; the tight and loose
  same-job full-order Burgers controls.
- **FNO / neural operator / direct multi-time output** — a learned map between function
  spaces; the model class; an architecture that emits every requested output time from one
  forward pass rather than stepping, so it has no rollout accumulation and a fixed output
  time set.
- **Checkpoint / frozen transfer / seed / SHA256** — saved network weights; evaluating those
  same weights on a different mesh; the recorded randomness of a training run; the content
  hash that pins which file was used.
- **Device query / complete query cost / cached reduced cost** — a computation timed from
  input arrival to output departure on the accelerator, including projection, solve and dense
  output; the same thing including host transfers; the reduced solve alone with all operators
  already assembled — the only quantity that is mesh-independent.
- **Job id / same-job / cross-job** — the cluster's identifier for one allocation; two
  measurements taken inside one allocation on one GPU, which may be divided; two taken in
  different allocations, which may **not**.
