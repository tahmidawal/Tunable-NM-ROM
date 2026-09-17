# Reviewer complaint → what answers it → status

Complaints from the NeurIPS 2026 reviews of *Tunable Non-linear Manifold ROMs for Elliptic
and Parabolic PDEs via Matrix-Free Galerkin Projection* (submission 25242; reviewers
**fxe8**, **GwrW**, **5mgh**), read together with the author rebuttals and the AC-confidential
follow-ups.

Status vocabulary matches `OUTLINE.md` §0: **answered in text** (the rewrite settles it),
**answered by experiment (exists)**, **answered by experiment (collected, negative)**,
**answered by experiment (not run)**, **conceded as a stated limitation**, **withdrawn** (the
claim is gone).

## What changed on 2026-09-16

This map was written on 2026-09-14, before the campaign's jobs finished, so several rows said
"experiment not run" or "not started" about experiments that have since completed and been
audited. The rows updated are **F1, F2, F9, F11, G4, M1, M4, M5, M8, M11** plus new rows
**F16** and **M12** for evidence that did not exist on 2026-09-14. Each updated row now names
the report and its SHA256, computed from the file on disk during this session, and quotes the
number the reviewer's complaint turns on. Nothing previously written here is retracted; the
statuses are superseded.

Evidence files cited below, with the SHA256 computed on 2026-09-16:

| Short name | Path (relative to the repository root) | SHA256 |
|---|---|---|
| head ablation | `worktrees/2026-09-14-head-ablation/experiments/head-ablation/reports/2026-09-14-head-ablation.md` | `aac6536b8a2f080f528492990d707017517b35c5d47fe64ed69144d66cbefb50` |
| mesh ladder | `worktrees/2026-09-14-mesh-ladder/experiments/mesh-ladder/reports/2026-09-14-frozen-checkpoint-mesh-ladder.md` | `028b0753978b1d0fc0d40ff65335cb94006ecd003f5d79a2d41ed959cc7e035a` |
| fixed-checkpoint tuning | `worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/reports/2026-09-14-fixed-checkpoint-tuning.md` | `53a8003a3c64ace86a4ff3cdc8418c17fd75061dac900964f22c6e3fda3cb6a5` |
| Burgers FNO | `worktrees/2026-09-14-no-audit/experiments/neural-operator-audit/reports/2026-09-14-burgers-fno-baseline.md` | `87b9a55bfc0a4ca1acfd0e105032a08c3b3a6235cb2f64f557d29bdfc8dbff4c` |
| Poisson operator screen | `worktrees/2026-09-14-no-poisson/experiments/neural-operator-poisson/runs/matched01/matched-summary.json` | `4e6b23d835de48b52de69dd4422be03125782477e314b8829622b565a1132ae3` |
| prior dial | `worktrees/2026-09-15-prior-dial/experiments/prior-dial/reports/2026-09-15-prior-dial.md` | `3813a1c2c40b9c5e63ffb5d9d62f20db4c1c43a7a8913ec8a6a582b779100a47` |
| head refinement | `worktrees/2026-09-15-head-refine/experiments/head-refine/reports/2026-09-15-per-query-head-refinement.md` | `8127f6b4682a5aa38e5ac45f9e06e1645900254af86f354ca2d1fda6e140b015` |
| cheap corrections | `worktrees/2026-09-15-cheap-corrections/experiments/cheap-corrections/reports/2026-09-15-cheap-corrections.md` | `dae651250a1899e155ec71893a5579257663e129f9cd9e8c89b076e7e03f0024` |
| correction ladder (audited, superseded by cheap corrections on convergence) | `worktrees/2026-09-14-head-ablation/experiments/head-ablation/reports/2026-09-14-burgers-correction-ladder.md` | `5926ea3d9883aa25bc1064b1763ef848929a21214bac73483cb51260fdbb82d8` |
| zero start | `worktrees/2026-09-14-head-ablation/experiments/head-ablation/checks/zero-start/poisson_zero_start.json` | `7ba36b13bef9f5dbdb02595d308d82cc13bcbb3d0c7c659563b01a95c2c444d0` |
| cold-start caps (Poisson) | `worktrees/2026-09-14-head-ablation/experiments/head-ablation/checks/cold-start-caps/poisson_cap_curve.json` | `f1c76d6fad430498d4643c8fa6d69b15535078a08f66f7e963027f758748c02e` |
| cold-start caps (Burgers) | `worktrees/2026-09-14-head-ablation/experiments/head-ablation/checks/cold-start-caps/burgers_cap_curve.json` | `77d0ab1bd5a0e1b50700e80f894db091090d3152c75eaf67bc2e2e3ef83e8fcd` |

---

## fxe8 — mathematics, internal consistency, and evidence

| # | Complaint | What answers it now | Status |
|---|---|---|---|
| F1 | "A single trained model produces an entire Pareto frontier" is contradicted by the paper's own tables: best-accuracy and best-speed points used different $k$ and different CP ranks, and one cell used a different training recipe | §3.9 states exactly what is held fixed (weights, bank, $R$, $k$, time discretisation, query contract) and exactly what may vary. Every 2026-09-14/15 cell holds ONE SHA-pinned checkpoint (`18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589`, $k=16$, $R=512$) fixed and varies one declared knob. **The frontier claim does not survive even under that discipline**: the only lever that moves accuracy is the correction rank $q$, and its non-dominated **converged** set spans 3.61× in cost but only **1.41×** in error, failing the pre-registered $\ge 2\times$ error span [cheap corrections]. Three further levers — the prior dial, per-query head refinement, and the starting point — are measured and are **not knobs at all** [prior dial; head refinement; zero start] | **answered by experiment (collected, negative)**: the complaint is upheld, and the paper now says so |
| F2 | The word "Pareto frontier" is not earned | Removed from the paper, and now measured rather than merely disclaimed. `OUTLINE.md` E5 states the surviving claim in full: *a single trained decoder exposes a monotone family of offline-prepared inference-time operating points — correction rungs, EQ rules, tolerance — selected at run time without retraining; the family dominates the POD rank ladder on Burgers; it does not beat an efficient FOM; Poisson's rungs are nearly free and Poisson is the correctness case.* The pre-registered frontier test is applied and **fails** on the error span in every cell that ran | **withdrawn**, and replaced by a measured, weaker statement |
| F3 | Internal numeric inconsistencies between Table 1 and Table 2; a wrongly bolded Pareto-dominance cell | No number is typed by hand anywhere. Every table and figure is emitted by a generator reading run JSONs (`OUTLINE.md` §6), and each writes a paired `.json` so the figure can be re-derived | **answered in text** (process fix) |
| F4 | **Eq. (4), the heat step, is invalid for a nonlinear decoder**: it is a linear system in $z_{n+1}$ obtained by replacing $\tilde u(z)$ with $J_D z$, dropping the decoder bias, the operating point, and the linearisation remainder | §3.6 derives the step from the **fully discrete** residual $(I+\tfrac{\Delta t\kappa}{2}A)u(z)-(I-\tfrac{\Delta t\kappa}{2}A)u(z_n)$, which is nonlinear in $z_{n+1}$, then projects and row-scales to $B_0h_\theta(z)-D\,B_0h_\theta(z_n)$ and solves it by damped LM warm-started at $z_n$. The scheme is **Crank–Nicolson**, not backward Euler — a second correction the review did not catch (DISCREPANCY D3a in `methods.tex`) | **answered in text** |
| F5 | Inconsistent algorithmic description: the method section says each heat step is solved by CG, the appendix and tables move along the speed axis with GN cap and tolerance | §3.6 and §3.9: no path applies CG to the projected operator. The $k\times k$ damped normal system is solved directly — Gauss–Jordan with symmetric diagonal scaling and a backward-error gate with a charged general-solve fallback, or Cholesky. The heat controls are the LM cap and the stationarity tolerance (DISCREPANCY D5) | **answered in text** |
| F6 | Symmetry: with row weights the projected operator $J^\top W K J$ need not be symmetric, yet CG is applied | Moot, and we say why it is moot: no CG is applied to the projected operator. The damped normal matrix $J^\top J+\lambda\operatorname{diag}$ is SPD by construction, which is what licenses the Cholesky path in §3.6 | **answered in text** |
| F7 | EQ is under-specified: how is an arbitrary target sample count specified, are the rules nested, is NNLS re-solved, does a smaller rule need new offline work? | §3.8 specifies the offline NNLS on decoder-output snapshots, the greedy Lawson–Hanson support growth capped at $m$ with exact non-negative refit, the padding rule that makes the achieved count equal the requested one, and states plainly: **rules are not nested, changing $m$ re-solves NNLS offline, and deployment selects among stored rules** | **answered in text** |
| F8 | Three irreconcilable quadrature counts (minimum support 300; $n_{\rm eq}\in\{4..64\}$; $N_{\rm eq}\in[512,2560]$) | §3.8 separates the three quantities that were being conflated: requested count $m$, achieved support size, and number of fit snapshots $n_{\rm fit}$ (DISCREPANCY D4) | **answered in text** |
| F9 | Speedups measured only against unpreconditioned CG do not demonstrate superiority over competitive solvers (FFT/DST, multigrid, preconditioned Krylov, reused factorisations) | §3.11 and §4 name **direct DST** as the comparator wherever the operator separates. E4 has now run with DST and the tuned full-order controls interleaved in the same job as the ROM at 64/128/256/512/1024 intervals: **no crossover at any rung on either PDE**. FOM/ROM by rung, Burgers 0.265×, 0.320×, 0.325×, 0.495×, 0.443×; Poisson against direct DST 0.063×, 0.065×, 0.067×, 0.079×, 0.112× [mesh ladder]. The historical 10×-and-larger wins against over-solved Newton at 1e-6 are the inflated comparison `AGENTS.md` forbids and are not carried into this paper | **answered by experiment (collected, negative)**; the limitation is now quantified, not just conceded |
| F10 | The 324× claim | Gone. No speedup claim survives into this draft; `OUTLINE.md` forbids one without a paired same-job latency panel at matched accuracy | **withdrawn** |
| F11 | Asymmetric tuning: per-cell tuning of the method, canonical fixed configs for FNO/DeepONet | E6 uses an identical dataset and split with recorded index hashes, four FNO capacities at an **equal wall budget** with validation selection and early stopping, and reports the outcome whichever way it falls. Poisson: the FNOs generalise **better** than the fresh ROMs; neither ROM arm is selected [Poisson operator screen]. Burgers: the ROM is the more accurate model — on the matched eight-case cohort FNO `fno-large` 2.482867 % worst against ROM 1.867068 % and efficient FOM 0.997803 %; on the 32 held-out cases `fno-large` is 1.805424 % median / 6.382471 % worst at a 7.314731 ms device query [Burgers FNO]. **No speed ratio is stated**, because the FNO timings are same-job and the ROM/FOM timings come from another allocation | **answered by experiment (exists; one arm negative, one positive)** |
| F12 | Timing methodology unspecified: JIT compilation, synchronisation, warm-ups, repetitions, precision, variance | §4 states the protocol: per-block GPU burn-in, device synchronisation around every measured region, medians over retained repetitions with all repetitions kept, f64 with highest matmul precision, and **no ratio across jobs or GPUs** | **answered in text** |
| F13 | The $\kappa$-as-runtime-argument effect (14.7× → 191×) suggests results dominated by recompilation | Diagnosed in the rebuttal as a compilation-caching artefact; the protocol in §4 makes it impossible for compilation to enter a measurement. No inherited speedup number is carried forward | **withdrawn** |
| F14 | No code or checkpoints, so the claims cannot be verified | Every panel records checkpoint SHA, config, commit, job id, GPU, backend and precision (table T1), and each figure ships its source JSON | **answered in text** |
| F15 | Originality is system-level integration | Accepted verbatim in related work, which states that the projection, the hyper-reduction algorithm and the tensor construction are all **not new**, and that the old CP decoder admits the same preassembly | **conceded as a stated limitation** |
| F16 | *Not raised, added 2026-09-16 because F14 invites it:* what exactly can a reader verify, given that results cannot be reproduced bitwise? | Stated plainly. Within one job, repetitions are byte-identical and every recorded error is recomputed from the retained output fields by an independent NumPy audit that imports neither the driver nor JAX (worst disagreement 5.65e-16 Burgers, 1.21e-15 Poisson) [head refinement]. **Across jobs on different A100 SKUs, bitwise identity does not hold and is not claimed**: the cheap-corrections cell reports `directions_sha256` and the retained reference fields as **failing** their bitwise checks while the scientific gates that carry the weight pass — $q=0$ reproducing the audited ladder's `q0_eq` and the head-ablation arm (a) to 1e-12, and $q=16$ agreeing with the audited `q16_dense` field to 9.4e-07 [cheap corrections]. The paper states what is certified (agreement to a declared tolerance on the same stationary point) and what is not (byte equality) | **answered in text**, with the failing checks reported rather than relaxed |

---

## GwrW — what is actually being solved, and against what

| # | Complaint | What answers it now | Status |
|---|---|---|---|
| G1 | Unclear which PDEs, domains, boundary and initial conditions are used | §3.1 and §4 give the domain, the discretisation, the boundary treatment (§3.2: homogeneous Dirichlet enforced structurally by a smooth vanishing factor folded into the bank), the families and the seeds | **answered in text** |
| G2 | Only well-conditioned, diffusion-dominated problems | Burgers 2D with sign-upwind advection is a first-class instance (§3.7), not an appendix. The elliptic cell is retained as the **algebraic correctness and limitation case**, and related work says so | **answered in text**; harder-family extension **not run** |
| G3 | Ask for genuinely harder PDEs (Navier–Stokes, Euler, Schrödinger) | Not addressed by this draft. Stated as scope, not silently omitted | **conceded as a stated limitation** |
| G4 | Compare against preconditioned CG or a direct solver, not plain CG | Same as F9, and now measured: direct DST and tolerance-tuned full-order controls interleaved in the same job at every mesh rung, with no crossover anywhere [mesh ladder]; the same-job full-order controls also appear in E3, E5, E8 and E10 [head ablation; fixed-checkpoint tuning; prior dial; cheap corrections] | **answered by experiment (collected, negative)** |
| G5 | Conditioning drives any speedup comparison, and the elliptic cells are the weakest evidence | Adopted as the paper's own position in related work and §3.11 | **answered in text** |
| G6 | Reviewer's closing assessment: "the direct solver significantly outperforms the methods described"; the paper needs a bigger rethink than a rebuttal allows | Accepted. This is the reason the paper is rebuilt around precomputed weak operators and their conditions rather than around a speed claim, and the reason the non-claims are in the introduction | **conceded as a stated limitation** |
| G7 | The re-entrant (L-shaped) domain result, where DST stops applying | Not in this draft. It is the natural next family and is recorded as such, unclaimed | **not run** |

---

## 5mgh — baselines, the case for nonlinearity, and projection

| # | Complaint | What answers it now | Status |
|---|---|---|---|
| M1 | The Pareto-dominance claim rests on broken (DeepONet), untuned (FNO width 32) or oracle-warm-started (SMA) baselines | Dominance claim withdrawn entirely. E6 is a matched-data comparison at four FNO capacities (1.19 M to 17.9 M real parameters) on an equal wall budget, whose Poisson arm is **negative for the ROM** and whose Burgers arm is positive, both reported [Poisson operator screen; Burgers FNO]. The FNO comparison stays accuracy-only until a same-job latency panel exists | **withdrawn** + **answered by experiment (exists; both directions reported)** |
| M2 | Single seed, no variance reported | E7. Until it runs, every endpoint carries its single-seed label **inline, beside the number**, not only in a preamble | **conceded as a stated limitation**; **not run** |
| M3 | Intrusive vs non-intrusive is not an apples-to-apples comparison | §3.10 states the asymmetry and the single condition under which the comparison is meaningful, **before** any comparison appears | **answered in text** |
| M4 | The paper never exhibits a regime where the nonlinear manifold is the enabling ingredient rather than an expensive alternative to linear POD | E3 has run on both PDEs at matched $k=16$ with a correctly solved POD-LSPG ladder (ranks 8/16/32/64/128) and an unrestricted-coefficient arm, through the same weak objective, test modes, time discretisation, initializer, stopping rule and output contract [head ablation]. **Burgers 256 intervals, worst same-grid error:** neural head 2.562872 % at 47.649 ms; linear head 56.929573 %; quadratic head 31.827588 %; POD-16 61.650255 %; POD-128 10.119784 % at 328.344 ms — **no POD rank up to $8k$ matches the head**, and the POD rungs run with exact dense advection, which favours them, so that is a conservative lower bound. **Poisson 1024 intervals:** neural head 6.092722 % at 5.843 ms but **POD-128 4.345242 % at 5.930 ms** — the head wins per dimension and loses per millisecond, because the Poisson query is dominated by projection and decode rather than the reduced solve. Both directions are printed. §3.2's structural limit stands: the manifold lies in $\operatorname{range}G$, so nonlinearity buys a smaller solved dimension, not an escape from the bank | **answered by experiment (exists; positive on Burgers, negative on Poisson)** |
| M5 | The natural intrusive baseline (linear POD-Galerkin + DEIM with the same knobs) is present but not foregrounded | It is a foregrounded arm of E3 and it has run: classical POD-LSPG at five ranks, at matched $k$, through the same weak objective, the same test modes and the same solver, with the same empirical-quadrature rule where the arm admits one [head ablation]. Poisson is the case where it **wins on cost at equal accuracy** (POD-128, 4.345242 % at 5.930 ms against the head's 6.092722 % at 5.843 ms), and the paper reports that as a result rather than burying it | **answered by experiment (exists)** |
| M6 | Galerkin is unstable for non-symmetric / advection-dominated operators; is LSPG required? | §3.3: the implemented condition **is** a least-squares Petrov–Galerkin one with an explicit, latent-independent test space and an explicit row norm — not tangent Galerkin, and overdetermined ($M>k$). The old manuscript's tangent-Galerkin description did not match the code (DISCREPANCY D2). The rebuttal's claim that residual least squares and Galerkin coincide for a linear decoder is **not** repeated: a comment in `methods.tex` records that $V^\top A^\top(AVz-b)=0$ and $V^\top(AVz-b)=0$ are different conditions | **answered in text** |
| M7 | Report SMA-NM-ROM under the same cold-start protocol, not oracle warm start | §3.3 *Initialisation* states that no query uses the solution it predicts, and describes each cell's cold start exactly. A cold-start SMA arm is not in this draft's scope; the requirement that any baseline be run under the same protocol is stated | **conceded as a stated limitation** |
| M8 | Does changing the EQ count require re-solving the NNLS — i.e. is the frontier really free from one model? | §3.8: **yes**, it re-solves offline; rules are not nested; deployment selects among stored rules. The offline cost is now measured rather than asserted: per-rung NNLS fits took **13.4 s to 307.4 s** each, every one charged outside every query timing, and two of the fitted rules were walltime-truncated [cheap corrections]. What the rules buy online is also measured: **4.66×–6.42×** cost over the paired dense arm at identical same-grid error across the converged pairs [cheap corrections], and 5.36× against the quadrature-free full-grid control for 0.147890 pp of worst error [fixed-checkpoint tuning]. The one rule that broke — $q=512$, walltime-truncated, 27.8120 % against 0.6027 % for its dense twin — is reported as broken | **answered in text and by experiment (exists)** |
| M9 | No theoretical guarantees: cold-start GN convergence, EQ error as a function of $N_{\rm eq}$ on unseen instances, latent backward-Euler stability | §3.11 lists all three as open, with the additional statement that the NNLS fit residual on its own snapshots is **not** a fidelity certificate for unseen states | **conceded as a stated limitation** |
| M10 | A small projected residual is treated as sufficient for accuracy | §3.3 *What stationarity does and does not certify*: it is a first-order condition on $M$ tested components of an $n$-component residual; it bounds neither the untested components, nor the representation error, nor the physical error. Weak stationarity and physical error are reported as separate quantities throughout | **answered in text** |
| M12 | *Not raised, added 2026-09-16:* the review has no vocabulary for separating "the model cannot express it", "the reduced parameterisation cannot reach it" and "the solver did not find it", which is why M4 and M10 keep colliding | The paper's organising frame, C0: every panel reports **bank projection**, **best-found fit** and **solved** error as three separate numbers. Burgers at 256 intervals: 0.391845 % / 2.544663 % / 2.562872 % — the solver layer costs 0.018 percentage points and the gap that matters is the reduction layer. Poisson at 1024: 2.314808 % / 6.092634 % / 6.092722 % [head ablation]. The frame is what makes the negative levers legible: the prior dial acts entirely in the reduction/solver layer because the penalty restricts the solver and not the reachable set [prior dial], and refining the head's weights moves the reachable set but not the frontier [head refinement] | **answered by experiment (exists)** |
| M11 | Originality is incremental; the tunable frontier is inherent to any solver-based ROM and is novel only relative to neural operators | Accepted, and strengthened into the paper's actual contribution. Tunability is **one section, not the headline**, and the campaign's finding is that most of the tunability a solver-based ROM *appears* to offer is not there: of six inference-time levers tested on one frozen checkpoint, **only the correction rank moves accuracy, only hyper-reduction moves cost, and nothing does both** [cheap corrections; fixed-checkpoint tuning; prior dial; head refinement; zero start; cold-start caps]. The old manuscript's apparent frontier is reproduced and explained: it is the **cold-start iteration-cap curve**, which is real but dominated point-for-point by simply starting from the nearest training code (Poisson cap 2 reaches what the zero start needs cap 12 for, at a third of the cost) [cold-start caps (Poisson); cold-start caps (Burgers)]. That diagnosis, not the frontier, is what is novel | **conceded, and converted into the paper's framing** |

---

## Not raised by any reviewer, but corrected here

These came out of reading the code against the manuscript and are recorded because the
project's rule is that retractions matter more than successes.

| # | What the old manuscript said | What the code does | Where it is corrected |
|---|---|---|---|
| D1 | Dirichlet enforced by a **binary** mask plus a nonzero lift, $\tilde u=m\odot\mathcal D(z)+u_g$ | A **smooth polynomial** factor $16x_1(1-x_1)x_2(1-x_2)$ folded into the bank, with $u_g=0$; enforcement is still exact but by a different mechanism, and as implemented covers only homogeneous data | §3.2 |
| D3a | Backward Euler for heat | Crank–Nicolson | §3.6 |
| D6 | ViT encoder feeding a LinearCPDecoder | **No encoder at all** in the deployed path: latent codes come from an auto-decoder fit during training and a least-squares fit at query time. The "ViT for global coupling" argument does not apply to the method described | §3.2 |
| D8 | "Gauss–Newton" throughout, with an undamped update | Damped Levenberg–Marquardt with diagonal damping, a monotone accept/reject test, an explicit trust radius, and no line search | §3.3 |
| D9 | A latent ODE $J^\top J\dot z=-\kappa J^\top K\tilde u(z)$ | No latent mass matrix is ever formed and no ODE in $z$ is integrated; each step is a minimisation | §3.6 |
| — | Poisson ground truth built analytically | Verified discrete solution (already conceded in rebuttal; retained here because the older Poisson numbers are unreproducible against it) | §4 |

---

## Glossary

- **Reviewer id (fxe8, GwrW, 5mgh)** — anonymous reviewer labels from OpenReview.
- **Rebuttal / AC-confidential comment** — the authors' public reply during discussion; a
  reply visible only to the area chair, which several of the late corrections went into.
- **Pareto frontier / dominance** — a set of operating points none of which is worse than
  another on both axes; the claim that one method is at least as good everywhere. Both
  claims are withdrawn from this paper.
- **Cold start / oracle warm start** — beginning a solve from a fixed, query-independent
  starting point; beginning it from information derived from the answer. Only the first is
  a legitimate deployment protocol.
- **Nested quadrature rules** — a family of rules in which the smaller is a subset of the
  larger, so one can be truncated into another. The rules here are **not** nested.
- **SMA-NM-ROM / DEIM / LSPG** — the shallow-masked-autoencoder nonlinear manifold ROM used
  as a prior baseline; discrete empirical interpolation, an alternative hyper-reduction;
  least-squares Petrov–Galerkin, the projection family this method belongs to.
- **DST** — the direct discrete-sine-transform solver, exact and fast for separable
  constant-coefficient operators on a rectangle, and the comparator this project must always
  name.
- **DISCREPANCY Dn** — the label used in `methods.tex` comments to mark a place where the
  previous manuscript and the implemented code disagree and the code was followed.
- **The three layers (representation / reduction / solver)** — the error that would remain
  even with a perfect coefficient map, because the fixed spatial bank cannot express the
  answer (reported as *bank projection*); the extra error from restricting to what the
  coefficient map can reach (*best-found fit*, an oracle over many restarts); the extra error
  the actual iteration leaves behind (*solved* minus *best-found*). A lever can only move the
  layer it acts on, which is why most of the levers tested move nothing.
- **Correction rank $q$ / rung** — extra fixed linear directions solved alongside the neural
  head's output, $u = G(h_\theta(z)+C_q y)$; one value of $q$ on the resulting ladder. The
  only inference-time lever measured to move accuracy.
- **Prior dial ($\lambda$) / head refinement ($n$, $\mu$) / zero start** — three further
  inference-time levers, each tested against pre-registered criteria and each found not to be
  a usable knob: a penalty pulling the bank coefficients towards the head's own prediction;
  gradient steps on the network's own weights inside a single query, anchored to the trained
  weights; beginning the latent solve from $z=0$ rather than the nearest training code.
- **Non-dominated / cost span / error span** — a point nothing else beats on both axes; the
  ratio of the most to the least expensive such point; the same ratio for error. The
  pre-registered knob test asks for at least three non-dominated points spanning $\ge 2\times$
  on **both**, with none early-stopped.
- **Same-job vs cross-job** — two measurements taken inside one cluster allocation on one
  GPU, which may be divided; two taken in different allocations, which may not. Every timing
  ratio in this paper is same-job; anything cross-job is labelled provisional.
- **Worst over all times vs worst over evolved times** — the maximum Burgers error over every
  requested output time including $t=0$, where the model is merely compressing the *supplied*
  initial field; the same maximum excluding $t=0$. Which one is reported is an open decision,
  recorded in `OUTLINE.md` §9.
