# Reviewer complaint → what answers it → status

Complaints from the NeurIPS 2026 reviews of *Tunable Non-linear Manifold ROMs for Elliptic
and Parabolic PDEs via Matrix-Free Galerkin Projection* (submission 25242; reviewers
**fxe8**, **GwrW**, **5mgh**), read together with the author rebuttals and the AC-confidential
follow-ups.

Status vocabulary matches `OUTLINE.md` §0: **answered in text** (the rewrite settles it),
**answered by experiment (exists)**, **answered by experiment (not run)**, **conceded as a
stated limitation**, **withdrawn** (the claim is gone).

---

## fxe8 — mathematics, internal consistency, and evidence

| # | Complaint | What answers it now | Status |
|---|---|---|---|
| F1 | "A single trained model produces an entire Pareto frontier" is contradicted by the paper's own tables: best-accuracy and best-speed points used different $k$ and different CP ranks, and one cell used a different training recipe | §3.9 *Deployment-time controls* states exactly what is held fixed (weights, bank, $R$, $k$, tests, $M$, time discretisation, query contract) and exactly what may vary (GN cap, stopping tolerance, stored EQ rule). E5 varies only those three at one SHA-pinned checkpoint | **answered in text**; **experiment prepared, not run** |
| F2 | The word "Pareto frontier" is not earned | Removed from the paper. §3.9 says we *measure* an error–cost relation, report its saturation and its stalls, and explicitly do not claim monotonicity or non-domination. `OUTLINE.md` E5 makes "no frontier claim" the acceptance criterion | **withdrawn** |
| F3 | Internal numeric inconsistencies between Table 1 and Table 2; a wrongly bolded Pareto-dominance cell | No number is typed by hand anywhere. Every table and figure is emitted by a generator reading run JSONs (`OUTLINE.md` §6), and each writes a paired `.json` so the figure can be re-derived | **answered in text** (process fix) |
| F4 | **Eq. (4), the heat step, is invalid for a nonlinear decoder**: it is a linear system in $z_{n+1}$ obtained by replacing $\tilde u(z)$ with $J_D z$, dropping the decoder bias, the operating point, and the linearisation remainder | §3.6 derives the step from the **fully discrete** residual $(I+\tfrac{\Delta t\kappa}{2}A)u(z)-(I-\tfrac{\Delta t\kappa}{2}A)u(z_n)$, which is nonlinear in $z_{n+1}$, then projects and row-scales to $B_0h_\theta(z)-D\,B_0h_\theta(z_n)$ and solves it by damped LM warm-started at $z_n$. The scheme is **Crank–Nicolson**, not backward Euler — a second correction the review did not catch (DISCREPANCY D3a in `methods.tex`) | **answered in text** |
| F5 | Inconsistent algorithmic description: the method section says each heat step is solved by CG, the appendix and tables move along the speed axis with GN cap and tolerance | §3.6 and §3.9: no path applies CG to the projected operator. The $k\times k$ damped normal system is solved directly — Gauss–Jordan with symmetric diagonal scaling and a backward-error gate with a charged general-solve fallback, or Cholesky. The heat controls are the LM cap and the stationarity tolerance (DISCREPANCY D5) | **answered in text** |
| F6 | Symmetry: with row weights the projected operator $J^\top W K J$ need not be symmetric, yet CG is applied | Moot, and we say why it is moot: no CG is applied to the projected operator. The damped normal matrix $J^\top J+\lambda\operatorname{diag}$ is SPD by construction, which is what licenses the Cholesky path in §3.6 | **answered in text** |
| F7 | EQ is under-specified: how is an arbitrary target sample count specified, are the rules nested, is NNLS re-solved, does a smaller rule need new offline work? | §3.8 specifies the offline NNLS on decoder-output snapshots, the greedy Lawson–Hanson support growth capped at $m$ with exact non-negative refit, the padding rule that makes the achieved count equal the requested one, and states plainly: **rules are not nested, changing $m$ re-solves NNLS offline, and deployment selects among stored rules** | **answered in text** |
| F8 | Three irreconcilable quadrature counts (minimum support 300; $n_{\rm eq}\in\{4..64\}$; $N_{\rm eq}\in[512,2560]$) | §3.8 separates the three quantities that were being conflated: requested count $m$, achieved support size, and number of fit snapshots $n_{\rm fit}$ (DISCREPANCY D4) | **answered in text** |
| F9 | Speedups measured only against unpreconditioned CG do not demonstrate superiority over competitive solvers (FFT/DST, multigrid, preconditioned Krylov, reused factorisations) | §3.11 and §4 name **direct DST** as the comparator wherever the operator separates, and state that it is faster than every reduced model measured on those cells. Related work: "the question is not whether it beats a fast direct solver on a rectangle — it does not." E4 puts DST and Newton in the same job as the ROM | **conceded as a stated limitation**; **E4 not run** |
| F10 | The 324× claim | Gone. No speedup claim survives into this draft; `OUTLINE.md` forbids one without a paired same-job latency panel at matched accuracy | **withdrawn** |
| F11 | Asymmetric tuning: per-cell tuning of the method, canonical fixed configs for FNO/DeepONet | E6 uses an identical dataset and split with recorded index hashes, and reports the outcome whichever way it falls — the completed Poisson screen is negative for the ROM and is in the paper. The unmatched training/tuning budgets are stated beside the result | **answered by experiment (exists, negative)** |
| F12 | Timing methodology unspecified: JIT compilation, synchronisation, warm-ups, repetitions, precision, variance | §4 states the protocol: per-block GPU burn-in, device synchronisation around every measured region, medians over retained repetitions with all repetitions kept, f64 with highest matmul precision, and **no ratio across jobs or GPUs** | **answered in text** |
| F13 | The $\kappa$-as-runtime-argument effect (14.7× → 191×) suggests results dominated by recompilation | Diagnosed in the rebuttal as a compilation-caching artefact; the protocol in §4 makes it impossible for compilation to enter a measurement. No inherited speedup number is carried forward | **withdrawn** |
| F14 | No code or checkpoints, so the claims cannot be verified | Every panel records checkpoint SHA, config, commit, job id, GPU, backend and precision (table T1), and each figure ships its source JSON | **answered in text** |
| F15 | Originality is system-level integration | Accepted verbatim in related work, which states that the projection, the hyper-reduction algorithm and the tensor construction are all **not new**, and that the old CP decoder admits the same preassembly | **conceded as a stated limitation** |

---

## GwrW — what is actually being solved, and against what

| # | Complaint | What answers it now | Status |
|---|---|---|---|
| G1 | Unclear which PDEs, domains, boundary and initial conditions are used | §3.1 and §4 give the domain, the discretisation, the boundary treatment (§3.2: homogeneous Dirichlet enforced structurally by a smooth vanishing factor folded into the bank), the families and the seeds | **answered in text** |
| G2 | Only well-conditioned, diffusion-dominated problems | Burgers 2D with sign-upwind advection is a first-class instance (§3.7), not an appendix. The elliptic cell is retained as the **algebraic correctness and limitation case**, and related work says so | **answered in text**; harder-family extension **not run** |
| G3 | Ask for genuinely harder PDEs (Navier–Stokes, Euler, Schrödinger) | Not addressed by this draft. Stated as scope, not silently omitted | **conceded as a stated limitation** |
| G4 | Compare against preconditioned CG or a direct solver, not plain CG | Same as F9: direct DST named wherever it applies, tolerance-tuned CG and Newton–BiCGStab as the iterative comparators, all in-job | **conceded as a stated limitation**; **E4 not run** |
| G5 | Conditioning drives any speedup comparison, and the elliptic cells are the weakest evidence | Adopted as the paper's own position in related work and §3.11 | **answered in text** |
| G6 | Reviewer's closing assessment: "the direct solver significantly outperforms the methods described"; the paper needs a bigger rethink than a rebuttal allows | Accepted. This is the reason the paper is rebuilt around precomputed weak operators and their conditions rather than around a speed claim, and the reason the non-claims are in the introduction | **conceded as a stated limitation** |
| G7 | The re-entrant (L-shaped) domain result, where DST stops applying | Not in this draft. It is the natural next family and is recorded as such, unclaimed | **not run** |

---

## 5mgh — baselines, the case for nonlinearity, and projection

| # | Complaint | What answers it now | Status |
|---|---|---|---|
| M1 | The Pareto-dominance claim rests on broken (DeepONet), untuned (FNO width 32) or oracle-warm-started (SMA) baselines | Dominance claim withdrawn entirely. E6 is a matched-data comparison whose completed Poisson arm is **negative for the ROM** and is reported as such | **withdrawn** + **answered by experiment (exists, negative)** |
| M2 | Single seed, no variance reported | E7. Until it runs, every endpoint carries its single-seed label **inline, beside the number**, not only in a preamble | **conceded as a stated limitation**; **not run** |
| M3 | Intrusive vs non-intrusive is not an apples-to-apples comparison | §3.10 states the asymmetry and the single condition under which the comparison is meaningful, **before** any comparison appears | **answered in text** |
| M4 | The paper never exhibits a regime where the nonlinear manifold is the enabling ingredient rather than an expensive alternative to linear POD | E3, the head ablation at matched $k$, including a correctly solved linear POD-LSPG arm and an unrestricted-coefficient arm. Its acceptance criterion forbids using the instability of a *different* rollout as evidence for decoder nonlinearity. §3.2 also states the structural limit: the manifold lies in $\operatorname{range}G$, so nonlinearity buys a smaller solved dimension, not an escape from the bank | **answered in text (structural)** + **answered by experiment (not started)** |
| M5 | The natural intrusive baseline (linear POD-Galerkin + DEIM with the same knobs) is present but not foregrounded | It is an arm of E3, at matched $k$ with the same weak objective and the same solver | **answered by experiment (not started)** |
| M6 | Galerkin is unstable for non-symmetric / advection-dominated operators; is LSPG required? | §3.3: the implemented condition **is** a least-squares Petrov–Galerkin one with an explicit, latent-independent test space and an explicit row norm — not tangent Galerkin, and overdetermined ($M>k$). The old manuscript's tangent-Galerkin description did not match the code (DISCREPANCY D2). The rebuttal's claim that residual least squares and Galerkin coincide for a linear decoder is **not** repeated: a comment in `methods.tex` records that $V^\top A^\top(AVz-b)=0$ and $V^\top(AVz-b)=0$ are different conditions | **answered in text** |
| M7 | Report SMA-NM-ROM under the same cold-start protocol, not oracle warm start | §3.3 *Initialisation* states that no query uses the solution it predicts, and describes each cell's cold start exactly. A cold-start SMA arm is not in this draft's scope; the requirement that any baseline be run under the same protocol is stated | **conceded as a stated limitation** |
| M8 | Does changing the EQ count require re-solving the NNLS — i.e. is the frontier really free from one model? | §3.8: **yes**, it re-solves offline; deployment selects among stored rules. Answered as a correction, not a defence | **answered in text** |
| M9 | No theoretical guarantees: cold-start GN convergence, EQ error as a function of $N_{\rm eq}$ on unseen instances, latent backward-Euler stability | §3.11 lists all three as open, with the additional statement that the NNLS fit residual on its own snapshots is **not** a fidelity certificate for unseen states | **conceded as a stated limitation** |
| M10 | A small projected residual is treated as sufficient for accuracy | §3.3 *What stationarity does and does not certify*: it is a first-order condition on $M$ tested components of an $n$-component residual; it bounds neither the untested components, nor the representation error, nor the physical error. Weak stationarity and physical error are reported as separate quantities throughout | **answered in text** |
| M11 | Originality is incremental; the tunable frontier is inherent to any solver-based ROM and is novel only relative to neural operators | Accepted. Tunability is **one section, not the headline**, and related work says the observation is relative to neural operators only | **conceded as a stated limitation** |

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
