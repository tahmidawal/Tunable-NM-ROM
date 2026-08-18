# Lab log

Append-only. Newest at the bottom. One `##` per date, one `###` per session.
Every session appends before it ends — including what was retracted.

---

## 2026-08-16

### Wave 1 — the objective fix, Poisson and Burgers

Two agents, one worktree each: `exp/2026-08-16-poisson2d-rom-objective` (namespace `pobj/`) and
`exp/2026-08-16-burgers2d-rom-latent-stepping` (`blat/`).

**The problem.** The latent solve stalled about 8× above the decoder's own accuracy ceiling.
Traced to the objective: minimising the pointwise FD residual amplifies grid-scale decoder error
by ~N², and the minimiser it finds has a *lower* residual than the true latent while producing
8× worse fields.

**The fix.** Minimise the residual projected onto M low smooth sine test modes, `Λ⁻¹`-weighted,
evaluated at NNLS empirical-quadrature points fitted on decoder-output snapshots. Reaches the
ceiling and is insensitive to the initial guess. Verified by giving it 5× the budget and seeing
nothing change.

Established: `M > k` comfortably or the objective collapses; `m ≈ 4M` is the knee; meshfree
candidate pools work as well as grid nodes; hyper-reduce the cold start or the online path stays
grid-bound.

### Wave 2 — Heat and Wave ports

`exp/2026-08-16-heat2d-rom-latent-stepping` (`hlat/`), `exp/2026-08-16-wave2d-rom-latent-stepping`
(`wlat/`).

Held-out error at k=8, N=64 — ceiling / ours / POD: Poisson 7.11e-3 / 7.65e-3 / 1.77e-1;
Heat 1.16e-2 / 1.87e-2 / 1.29e-1; Burgers 1.15e-2 / 1.65e-2 / 2.09e-1;
Wave 1.72e-1 / 8.78e-1 / 3.42e-1.

**Wave fails structurally.** On a fixed subspace the Newmark/CN recurrence is time-reversible
and therefore conservative; on a nonlinear manifold that symmetry is gone. End-time energy ratio
0.27 against POD's 1.000003. Refining the ROM time step 5× lowers the time-discretisation floor
28.9× and *raises* error 7.5%. Not fixable by tuning — the open problem is structure-preserving
latent stepping.

**Heat is accurate but does not pay**: a direct reduced POD-Galerkin solve is 13–38× faster than
the FOM, so nothing iterative competes on a linear parabolic problem.

---

## 2026-08-17

### Consolidation

Merged the four completed cells into `exp/2026-08-17-inr-rom-consolidated` with the reports
pipeline. Pruned `__pycache__`, smoke artefacts, and staged code copies.

**Retracted later:** this tree pruned the trained decoders, which is how they ended up tracked
nowhere. See 2026-08-18.

### Two experiments launched

`exp/2026-08-17-cost-to-tolerance` (`ctol/`) and `exp/2026-08-17-rom-warmstart-fom` (`wsfom/`),
one agent each.

**Incident — `scancel` killed both fleets.** A `scancel --name=a,b,c` matched nothing, degraded
to an empty selector, and cancelled every job on the shared account. ~16 min of compute lost, no
corruption. `cluster/cancel.sh` now refuses names, globs and user selectors.

**Correction to the record — every published speedup was inflated.** Both classical baselines did
more work than their accuracy required. Burgers ran a fixed 8 Newton iterations per step (400
Newton steps and 400 BiCGStab solves per rollout, ~4× over-converged); Poisson ran CG at
`tol=1e-13`, the tolerance used to *manufacture the truth data*.

- Burgers ladder 0.72 / 1.57 / 4.46 / 7.96× becomes **0.19 / 0.36 / 0.93 / 1.83×**. N=128 moves
  from clearly winning to break-even.
- Poisson depends on the assumed deployment tolerance and **the tolerance must always be named**:
  ~1.16× at 1e-10, ~1.31× at 1e-8, **~1.56× at 1e-6**. At 1e-6 the archived N=256 figure changes
  *sign*, 1.40× → 0.91×.
- Poisson's factor is near-constant in N so one scalar is defensible per tolerance; Burgers'
  varies 28% across meshes and must be applied per mesh or not at all.

**Retracted mid-session:** ~1.16× was first published as "the" Poisson factor. It is the tightest
of three columns and the one most flattering to the archived numbers. Erratum issued.

**The warm-start hybrid does not pay.** No Poisson crossover in any of 75 configurations, best
0.933×; Burgers 0.15–0.49×, losing 12/12 on wall clock. Linear extrapolation from the previous
two time steps beats it and costs nothing. Different mechanisms: on Poisson our guess is poor in
the norm CG actually contracts (4.97e-2 against a 9.27e-3 field error); on Burgers the guess is
good but costs 273 ms against a 48–228 ms solve.

**A crossover that did not exist.** The pre-fix panel showed 1.02× at N=512. It was a 17%
device-clock-ramp bias after a long host-bound NNLS fit. Burn in before every timed block.

**Bugs caught by the two Codex passes.** A missing BiCGStab alpha half-step test that *discarded
converged iterates*; `N_POD_TRAJ=128` pinned in the batch environment, which would have silently
reinstated a POD handicap the audit had just removed; invalid commit provenance (`git -C` walking
into an unrelated ancestor repo); unpersisted timing repetitions, which forced withdrawing a
"best configuration" claim rather than defending it.

**My grid-spec error.** I specified `M=256` with `m=256` for k≥32, landing on the `m = M` corner
and violating the project's own `m ≈ 4M` rule. Quadrature 660× worse, worst-row error 8.5e+05.
The entire k=32 column was an artefact of our own settings until fixed.

---

## 2026-08-18

### The k-spike retraction

A dedicated investigation (`experiments/k-stall-diagnosis/`) overturned the "our solver stalls at
particular latent dimensions" finding.

**It was a mean-over-16-cases artefact** dominated by 1–5 diverging solves. The median sits within
0.96–1.24× of the ceiling at *every* k, and k=32's median is the best on the ladder. Running all
64 held-out cases shows k=8 and k=16 failing too — the 16-case panel simply drew none of them.
The powers-of-two pattern was coincidence read off eight noisy points.

**Root cause: the latent LM has no globalisation.** `lam0=1e-6` gives an essentially undamped
Gauss–Newton first step, 10–350× the norm of the latent itself, and any residual decrease is
accepted. The iterate leaves the region the decoder was trained on and converges to a spurious
stationary point. Discriminator, exact on 9/9 traced cases: iterate norm after the first accepted
step, relative to the training-cloud radius — 0.77–1.32× succeeds, 1.73–4.60× fails.

**Fix**: trust region at the training-cloud radius. Ratios across the ladder go from
1.11/7.61/1.10/7.85/1.26/3.40/15.49 to **1.09/1.05/1.09/1.18/1.28/1.45/1.50**, nothing regresses,
iterations fall at large k. Not yet measured through the EQ + timing path.

**The same optimiser is used on every PDE**, so the project's accuracy numbers are likely broadly
pessimistic by an unquantified amount.

### Consolidation for Codex

Created `exp/2026-08-18-codex-handoff` with all seven cells, the reports, `AGENTS.md` and
`CODEX-START-HERE.md`.

**Two things were nearly lost, and are why the lab-log rule now exists:**

1. The k-stall investigation existed only in a `/tmp` scratchpad — session-scoped, and the most
   actionable result of the two days.
2. The trained decoders were `.gitignore`d and tracked on no branch. The Burgers k-ladder existed
   in git nowhere at all. A clone could be read but not rerun. All 16 are now tracked.

### Open

- **`cost-to-tolerance` is unfinished.** The single-GPU consolidation run never happened, so every
  ROM-vs-FOM wall-clock ratio pairs an A100-80GB against an A100-40GB — a 3.7× apparent speed-up
  from hardware alone. §7 Verdict and §8 Caveats are placeholders; no results audit. **Until this
  lands no speed number in the tree is publishable.**
- The Burgers correction denominator is formally open — the two cells ladder different knobs
  (a Newton tolerance vs the testbed's fixed `NEWTON_ITERS`). A pre-registered cross-check with a
  stated sign convention is set up.
- Apply the trust-region fix upstream and re-measure all four PDEs.
- Decoding is now 84% of online cost at N=512 while the latent solve is 9%.
- On 2-D Poisson a direct sparse solve is 494× faster than the iterative baseline. Say it before
  a reviewer does.
