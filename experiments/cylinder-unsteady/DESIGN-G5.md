# G5 — Unsteady flow past a cylinder (DFG 2D-2, Re ≈ 100): prospective release

Status: **prospective design, 2026-11-03, before any G5 execution.** No transient job has
run. This document fixes the settings of the G5 stage defined in
`exp/2026-10-27-ns-obstacle:experiments/ns-obstacle/DESIGN.md` §8 and must be committed and
independently reviewed (Codex) before the first allocation. Later changes are dated
amendments that preserve the original text, the original result and the reason; no gate is
relaxed after its result is seen.

## 0. Lanes, forks, namespace, user decisions

| Lane (worktree / branch `exp/<same>`) | Forked from (backed-up head → code-only mirror) | Content |
|---|---|---|
| `2026-11-03-cylinder-unsteady` | `exp/2026-10-27-ns-obstacle` `f463a08ec` → `eddd11586` | design, sources, integration, report evidence |
| `2026-11-03-cylinder-unsteady-ref` | `exp/2026-10-27-ns-obstacle-2d` `091afa488` → `36fce0d1d` | transient FE reference (R1) |
| `2026-11-03-cylinder-unsteady-rom` | `exp/2026-10-27-ns-obstacle-rom` `57680db12` → `e2d4b08bf` | projection floors, bank, ROM, baselines (R2–R5) |

The obstacle worktrees themselves are read-only for this campaign. Cluster namespace:
`/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103/<attempt>/`, one directory per job.

User decisions of 2026-11-03 (recorded verbatim in spirit, LAB-LOG): total budget
**12 allocated GPU-hours** ("feel free to extend further if needed" — any extension is a
recorded amendment with its reason, never silent); start from the Re 20 steady Newton
state with a sin² ramp; a small deterministic asymmetric kick; a Reynolds-number band
rather than a single-Re time split.

## 1. Question and hypotheses

Can the neural velocity bank that solved the steady DFG 2D-1 family represent and roll out
periodic vortex shedding at Re ≈ 100, and how does it compare with POD-Galerkin at matched
dimension in accuracy and in cost against the FE solver? Failure is an admissible outcome.

- H-ref: a BDF2 Taylor–Hood reference on the admitted curved mesh passes time and space
  refinement and reproduces the published DFG 2D-2 quantities.
- H-rep0: the frozen steady bank (trained on steady Re 15–25) does **not** represent the
  shedding snapshots (expected failure, measured, not assumed).
- H-rep: a bank retrained on transient snapshots with the steady recipe approaches the POD
  projection floor at matched dimension.
- H-dyn: a BDF2 Galerkin rollout of the bank (and of POD) tracks the phase-sensitive
  reference trajectory on development and held-out Reynolds numbers.
- H-cost: the reduced rollout is cheaper per simulated second than the FE stepper on the
  same CPU cores; offline cost is reported separately (break-even only if saving > 0).

## 2. Physics and geometry (DFG 2D-2)

Primary source (saved byte copy + SHA in `sources/`):
https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html .
Domain $\Omega=[0,2.2]\times[0,0.41]\setminus B_{0.05}(0.2,0.2)$, $\rho=1$, kinematic
pressure. No slip on both walls and the cylinder. Inlet
$u=(4U_{\max}(t)\,y(H-y)/H^2,\,0)$ with $U_{\max}=\tfrac32 U_{\rm mean}$; outlet
$(\nu\nabla u-pI)n=0$ (gradient form, as in the steady campaign; this fixes the pressure
level). $D=0.1$, $\mathrm{Re}=U_{\rm mean}D/\nu$; the family varies **viscosity only**:
$U_{\rm mean}=1$, $\nu=0.1/\mathrm{Re}$. Coefficients $C=2F/(U_{\rm mean}^2D)$ per unit span;
$\Delta p=p(0.15,0.2)-p(0.25,0.2)$ evaluated by FE interpolation at the exact points.
Force sign and normal conventions are those of the steady lane (force on the body, fluid
outward normal verified by `oriented_normals.py`).

**Start and ramp (primary trajectory, Re 100).** Initial state: the steady Newton solution
at $U_{\rm mean,0}=0.2$, $\nu=10^{-3}$ (exactly DFG 2D-1, Re 20) on the production
discretization. Inflow mean $U_{\rm mean}(t)=U_0+(U_1-U_0)\sin^2(\pi t/(2t_r))$ for
$t<t_r=1$, then $U_1=1$. Since $\dot U(0)=0$ the start is consistent with $\partial_tu=0$.

**Kick.** To make the shedding onset and its phase a deterministic function of the
discretization rather than of round-off, a body force
$f=(0,\,A\,\phi(t)\,e^{-|x-x_k|^2/(2\sigma^2)})$ with $A=1$ (i.e. $0.1\,U^2/D$),
$x_k=(0.3,0.2)$, $\sigma=0.02$, $\phi=\sin^2(\pi(t-1)/0.5)$ on $t\in[1,1.5]$, zero
otherwise, acts once on the primary spin-up. It is zero in every recorded window, every
continuation and every ROM. (The off-centre cylinder already breaks symmetry; the kick only
shortens the linear growth phase.)

## 3. Parameter population and data access

| Split | Re | Use |
|---|---|---|
| train | 90, 95, 100, 105, 110 | bank/POD construction, supremizers, lifts, hyper-reduction fits |
| development | 97.5 | all R2–R4 decisions, plots, animation |
| held-out (closed) | 102.5, 107.5 | opened once in R5 with frozen settings |

Justification: periodic shedding at a single Re lies on one limit cycle, so a time split
would test almost nothing beyond the training orbit; a ±10% band changes frequency and
amplitude (a genuine parametric test) while staying inside the documented periodic regime.
Interpolation-only conclusions; nothing is claimed outside [90, 110].

**Reduced fallback population** (used only if the predeclared cost rule in §7 triggers):
train 90, 100, 110; development 95; held-out 105.

Held-out trajectories are generated in R1 by the same jobs but written to a `heldout/`
subtree whose arrays are not read by any R2–R4 code; R1 audits check only reference-quality
gates (residuals, periodicity, divergence, hashes), never ROM-relevant statistics. ROM code
loads data only through a manifest that lists train/dev files.

## 4. Reference (R1)

### 4.1 Discretization

- Mesh: the admitted steady mesh **by bytes**, `mesh.vol.bin` SHA
  `528ffff38bb0f4d6d542f1abd4cba7c9ca2ba0427871caca806527c63ae6b092` (maxh 0.01,
  obstacle maxh 0.0025, curved geometry order 4), loaded, never regenerated or re-curved.
- Production spaces: Taylor–Hood **P3/P2** on that mesh. Spatial refinement check:
  **P4/P3** on the same mesh, which is exactly the admitted steady (FD1) discretization.
  Rationale: the admitted P4/P3 has 437,140 DOFs; P3/P2 is the cheapest admitted-geometry
  setting and the gate below certifies it against the admitted order.
- Weak form identical to the steady lane (gradient form, advective convection
  $(\nabla u)u\cdot v$, assembly bonus order 6), plus the mass term.
- Time: BDF2, one linear solve per step. Two implementations, both formally second order:
  **S1 `sbdf2`** — implicit Stokes/viscous, explicit second-order extrapolated convection
  $2N(u^n)-N(u^{n-1})$, constant matrix factored once (UMFPACK);
  **S2 `oseen_bdf2`** — linearly implicit convection with $w=2u^n-u^{n-1}$, matrix
  reassembled each step, LU refreshed every 10 steps and used as GMRES preconditioner.
  Selection rule (fixed now): S1 if it is stable in the smoke at the candidate dt and passes
  the time-refinement gate; otherwise S2. The choice and its evidence are recorded; the
  rejected scheme's failure is retained.
- Startup from any single state: 4 first-order substeps spanning one dt (IMEX/Oseen Euler),
  then BDF2. **Startup check**: from the common state S\*, runs with 4 and 16 substeps at
  the production dt must differ by at most 0.2 × each R1b tolerance over the window W.
- Linear solves: free-DOF scaled relative residual ≤ 1e-10 (checked on every refresh and
  every 200 steps for S1, every step for S2). Steady Newton: relative residual ≤ 1e-10.

### 4.2 Time levels, discard, horizon, cadence

- dt ladder $\{1/800, 1/1600, 1/3200, 1/6400\}$; production $dt_0$ chosen by R1b.
- Save cadence $\Delta t_s=1/400$ (≈ 132 snapshots per period; divisible by every ladder
  dt). ROM training uses every snapshot of the recorded windows.
- **Primary spin-up** (Re 100, kick on) at $dt_{\rm spin}$ = largest ladder dt stable in
  the smoke: run until the developed-flow rule holds, cap $t\le15$ (fail → R1 stops).
- **Developed-flow rule** (`cycles.periodic`): the last 3 consecutive pairs of complete
  cycles (lift minimum to lift minimum) agree in $C_{L,\max}$ and $C_{L,\min}$ within 1e-4
  absolute, in $C_{D,\max}$ and in period within 1e-4 relative.
- Then 0.7 s at $dt_0$ (≈ 2 periods, discards the dt-switch transient) → common state
  **S\***, saved with hash.
- **Recorded horizon:** 1.7 s (≈ 5.1 periods at Re 100) from the start of recording, for
  every trajectory, saved at $\Delta t_s$ (681 snapshots including both ends). The
  primary trajectory records from S\*.
- **Continuations** (all other Re): from S\* (or the developed state of the Re neighbour
  nearer 100), switch ν, run at $dt_0$ until the developed-flow rule holds (cap 4 s; fail
  → that trajectory fails and R1 stops), then record 1.7 s. Starting the band from a
  developed neighbour instead of the Re 20 ramp is a cost decision: the periodic attractor
  does not depend on the initial state, and only post-rule windows are used.

### 4.3 Diagnostics saved

Every step: time, reaction (variational, including the BDF2 inertia term) and surface
(gradient stress, order-12 rule) drag and lift, both probe pressures, kinetic energy, wall
time. Every other save: divergence $D\|\nabla\cdot u\|/(U\sqrt{|\Omega|})$, inlet/outlet
flux mismatch, and the gradient-form energy balance with $dE/dt$ written as
$\int u\cdot\partial_t u$ (BDF2 derivative) plus dissipation, boundary convective flux,
traction work, finite-divergence and body-force terms, normalized by the sum of absolute
terms. Snapshots: full FE coefficient vectors (float64), chunked NPZ with SHA-256.
Reaction drag/lift are the primary force histories; surface forces are a supplemental
consistency diagnostic (non-gating).

### 4.4 Reference gates (prospective; a failing gate stops the dependent stage)

Field metrics on the FE spaces with exact assembled mass matrices: velocity
$e_u=\|u_a-u_b\|_{L^2}/\|u_b\|_{L^2}$; pressure characteristic-scaled
$e_p=\|p_a-p_b\|_{L^2}/(U^2\sqrt{|\Omega|})$ (unshifted outlet convention). Each is taken
as the **maximum over all common save times** of the window (phase-sensitive). Forces:
$e_D=\max_t|C_D^a-C_D^b|/\max_t|C_D^b|$, $e_L=\max_t|C_L^a-C_L^b|$ at common step times.

| Gate | Rule |
|---|---|
| R1a smoke | imports/GPU preflight; steady Newton ≤1e-10; P3/P2 Re 20 drag within 2e-3 relative of the admitted P4/P3 value 5.579548…; linear residuals ≤1e-10; no instability over the smoke windows; per-step cost measured |
| R1b time | from S\*, window W = 0.35 s (> one period) at each ladder dt. Production $dt_0$ = largest dt in {1/800,1/1600,1/3200} whose comparison with dt/2 satisfies $e_u\le2\times10^{-3}$, $e_p\le4\times10^{-3}$, $e_D\le3\times10^{-3}$, $e_L\le2\times10^{-4}$. Empirical order from three successive levels must lie in [1.5, 2.5] for every metric whose differences exceed 100× the solver floor. None passes → R1 stops. Plus the startup check (§4.1) |
| R1c space | from S\* (P3 state embedded exactly into P4/P3), P4/P3 versus P3/P2 at $dt_0$ over W: same four tolerances |
| R1d DFG | Re 100 recorded window, statistics over complete cycles (`cycles.py`, the same code applied to the published logs). Against the Featflow 2014 finest data (Q2/P1disc, level 6, finest dt; values extracted into `sources/dfg2d2_reference.json`): $C_{D,\max}$ ≤2e-3 relative, $C_{L,\max}$ ≤2e-3 absolute, St ≤5e-3 relative, $\Delta p(t_0+1/(2f))$ ≤5e-3 relative. And inside the Schäfer–Turek (1996) published ranges for $C_{D,\max}$ [3.22, 3.24], St [0.295, 0.305], $\Delta p$ [2.46, 2.50]. The 1996 $C_{L,\max}$ range [0.99, 1.01] is reported but not gating, because the high-accuracy reference itself (0.9866) lies outside it; this is fixed now, before any result |
| R1e population | every trajectory reaches the developed-flow rule before its cap; linear residuals pass; divergence ≤1e-3 and mass mismatch ≤1e-7 at every diagnostic time; recorded window complete with hashes |
| R1f audit | an independent Codex audit recomputes, from the saved arrays alone, the R1b–R1e numbers, the cycle statistics and the file hashes, before any ROM stage reads the data |

Tolerances for R1b/R1c are the DESIGN.md transient tolerances (u, drag, lift) as directed;
the pressure tolerance keeps the steady 1:2 velocity:pressure ratio. A one-period window is
used because phase-sensitive lift differences grow linearly in time; it is fixed now.

## 8. Review status (append-only)

- 2026-11-03: the requested independent Codex review could not run: first attempt hung on
  stdin (no output), second returned "usage limit … try again at Nov 4th, 2026 4:37 PM"
  (`reviews/R1-code-codex-01-usage-limit.log`). **Codex review of this design is a missing
  gate, not a pass.**
- User decision 2026-11-03: use Grok (cursor-agent, `grok-4.7-xhigh`, read-only) now and
  Codex later. Grok may release **only the R1a smoke allocation (≤25 min)**; the full
  prospective release (including the ROM section, still to be written) and every later
  stage wait for the Codex review.
- Grok code review 01 (`reviews/R1-code-grok-01.json`): NOT READY — 2 blockers (Re 20
  drag stored at the wrong velocity scale; ineffective fresh-directory guard), 4 majors,
  minors. All addressed in ref-lane commits `3ebefc10d`, `fd1c769c9`. Re-review 02: READY,
  remaining minors: Newton denominator `max(r0,1)` (same as the steady lane), merged-boundary
  energy denominator (stricter than the steady split), smoke diagnostics every 10th save.
- Diagnostic cadence correction to §4.3 before any run: production diagnostics every 4th
  save (Δt = 0.01), smoke every 10th; "every other save" in §4.3 is superseded.

## Amendment G5-A1 — after R1a smoke01, 2026-11-03 (before any reference gate)

Smoke01 (job 6845044, pax024, 7 min 27 s, evidence in ref lane `artifacts/r1a-smoke01/`):
R1a Re 20 drag **passes** (P3/P2 reaction drag 5.579537 vs admitted 5.579548, relative
2.0e-6; four independent runs). **S1 `sbdf2` failed**: unstable once the inflow reached
U_mean = 1, at both dt = 1/1600 (blow-up near t = 0.19) and 1/800 (t = 0.105); retained.
Per the §4.1 rule the fallback is S2, which crashed on an NGSolve GMRes argument assertion
(`pre` and `freedofs` both given): an ordinary infrastructure error, repaired once
(drop `freedofs`). Measured P3/P2 cost 0.53 s/step with a constant matrix (235,231 DOFs);
P4/P3 1.38 s/step (437,140 DOFs).

Because S2 reassembles its matrix every step, a third member of the same family is added
before any gate: **S3 `frozen_oseen`** — BDF2 with implicit convection $(\nabla u)w_f$,
$w_f=u^n$ frozen and the LU refreshed every K = 10 steps, plus the explicit
second-order extrapolated remainder $2R(u^n)-R(u^{n-1})$ with
$R(u)=(\nabla u)u-(\nabla u)w_f$. Same startup, same linear tolerance (checked at every
refresh). Updated selection rule: among S2/S3, the cheaper scheme that is stable in smoke02
at full speed and passes R1b. All R1 gates, tolerances, populations and budgets are
unchanged. Smoke02 also measures the cost of assembly bonus order 2 versus 6; any change of
bonus order would be a further dated amendment justified by measured effect, not adopted
silently. S3 code was not separately reviewed before smoke02 (smoke-only, ≤25 min); it
enters the Codex review before any production use.

## Amendment G5-A2 — after R1a smoke02, 2026-11-03 (before any reference gate)

Smoke02 (job 6845240, pax026, 10 min 37 s, ref lane `artifacts/r1a-smoke02/`):
S3 `frozen_oseen` (K = 10) is stable through full inflow speed at dt = 1/1600 and 1/800
(0.15 s at U_mean = 1, energy bounded, linear residuals ≤ 6.3e-16, energy-balance
residual ≤ 2.5e-4); Re 20 drag again 2.0e-6 from the admitted value. S2 failed its linear
tolerance at the third step (stale-LU GMRES, 1.6e-5 > 1e-10) and cost 7.3 s/step:
**S2 rejected, retained**. Assembly bonus order 2 changes neither the Re 20 drag (2.0e-6)
nor the cost (the direct solve dominates), so bonus order 6 stays; no change.

Fixed now, before R1b data:
1. **Scheme: S3**, with K chosen so the frozen lag K·dt ≤ 1/80 s (the tested 1/800 × 10):
   K = 10 at 1/800, 20 at 1/1600, 40 at 1/3200, 80 at 1/6400. A K that proves unstable is a
   failure of that run, recorded; one retry at half K is allowed, no further tuning.
2. **S\*** is the developed state at the end of the primary spin-up (dt_spin = 1/800, the
   largest tested-stable ladder dt). Every R1b/R1c run starts from S\* with its own startup,
   so the dt-switch transient is part of the measured difference. The primary recorded
   window starts after 0.7 s of settle at $dt_0$ from S\*.
3. **Cost model** (measured, P3/P2, one 16-core node): mean seconds/step ≈ 0.53 (solve) +
   3.3/K (refactor) + 0.03. **Population rule**, applied once after R1b fixes $dt_0$:
   projected reference cost of the full population (spin-up actual + studies + 8
   trajectories of [settle at 1/800 until the developed-flow rule, then 0.7 s settle and
   1.7 s record at $dt_0$]) ≤ 8 GPU-h → full; otherwise reduced population if ≤ 8 GPU-h;
   otherwise stop and ask the user for a budget extension (the user said extension is
   acceptable "if needed"; it is still asked and recorded, not assumed).
4. **Divergence (R1e) pre-decision.** Smoke divergence reached 2.6e-3 during the compressed
   ramp (undeveloped, non-gating). R1e stays ≤ 1e-3. The spin-up records divergence at every
   0.1 s; if the developed P3 value at S\* exceeds 1e-3, R1e cannot pass for P3/P2, and the
   predeclared response is to **stop R1 and report** (the alternative, P4/P3 production at
   1.38 s/step plus refactor, is a budget decision for the user). No grad-div or other
   stabilization is added after seeing data.

## 5. Reduced models (R2–R5) — prospective, written 2026-11-03 before any reference data

### 5.1 Inherited bank (what transfers, what does not)

The frozen bank is the run-09 composite of the steady campaign: file `run09_bank.npz`
(SHA `645d030088ed7bc217b1afb59e3e13787b8f38c8ea5877599cc86d3b8b0a2a60`, the model of
PDE09 job 6661438), schema `ns-obstacle-independent-residual-head-bank-v1`: a frozen
parent MLP (21 Fourier/radial input features → 3 × 128 tanh → 2r, times the bounded
envelope $b(x)$ vanishing on inlet, walls and cylinder) trained to the first four steady
training POD velocity modes (values and $D$-scaled gradients, weights
$\sqrt{\sigma_i/\sigma_1}$; Adam 4000 cosine 1e-3→1e-4 + L-BFGS), plus four residual heads
(21-64-64-2) trained over the frozen parent with supervised divergence (weight 1.0) and
cylinder traction (weight 1.0) terms. Evaluator sources are vendored byte-identically.
**Transfers unchanged:** geometry, envelope, feature map, evaluator, the training
objective terms and optimizer recipe, the FE pressure/supremizer construction.
**Does not transfer:** the steady lift (Re 15–25 Stokes lift at U_mean = 0.2), the steady
weak least-squares Gauss–Newton solve (no time derivative), the steady POD teacher. New:
time-dependent mixed Galerkin rollout, transient POD teacher, phase-aware metrics.

### 5.2 Common spaces and quadrature (identical for every method)

- Quadrature Q: NGSolve order-10 mapped volume rule on every admitted-mesh element
  (points, positive weights saved with hashes); all reduced operators, projections and
  field errors use Q, so bank and POD are integrated identically.
- Lift $\bar u$: the mean of all training velocity snapshots (it carries the U_mean = 1
  inlet trace; fluctuation modes have homogeneous essential traces).
- Pressure basis (A4 rule): the constant, then L2-POD of mean-removed training pressures;
  $p_r=r$ functions in total. Supremizers: one per pressure function, $s_j\in V_{h,0}$ with
  $(\nabla s_j,\nabla v)=(\psi_j,\nabla\cdot v)$ for all $v$ (FE, H1 Riesz).
- POD velocity: L2 (FE mass matrix) method of snapshots on $u-\bar u$ for all training
  snapshots, f64, re-orthogonalized; ranks $r\in\{2,4,8,12,16,20,24\}$.
- Velocity space for a velocity basis $\Phi_r$: $\bar u+\mathrm{span}(\Phi_r\cup S)$,
  L2_Q-orthonormalized. Unknowns $d=r+2p_r=3r$ (the steady formula).

### 5.3 Time discretization of every ROM

Mixed Galerkin (momentum tested by the reduced velocity space, continuity by the reduced
pressure space), gradient-form viscous term, do-nothing outlet (no boundary term), exact
pre-contracted quadratic convection tensor including lift terms, viscous operator affine in
$\nu$. **BDF2 with fully implicit convection**, reduced Newton to relative residual
1e-12 (max 20 iterations, failure recorded), **dt_rom = dt_0** and the same 4-substep
startup as the reference. Initial condition: L2_Q projection of the reference velocity at
the first recorded time. Horizon: the full 1.7 s recorded window. Forces by the same
variational reaction formula as the reference (obstacle test function χ evaluated on Q,
inertia from the BDF2 derivative); probe pressures from basis values at the probe points.
No closure, stabilization, filtering or calibration is added after seeing results.

### 5.4 Stages and predeclared rules

- **R2 projection floors** (development trajectory Re 97.5, all 681 snapshots): relative
  L2_Q velocity error of the best approximation in $\bar u+\mathrm{span}(\Phi)$ and
  characteristic pressure error of the pressure basis. Bases: frozen run-09 bank (its 4
  functions), POD at every rank. **Floor rule:** a velocity basis is
  representation-adequate if the median floor ≤ 5e-3 and the maximum ≤ 1e-2.
- **R3 retraining** only if the frozen bank fails the floor. Rank $r_b$ = smallest POD
  rank in {8, 12, 16, 20, 24} meeting the floor (24 if none does, recorded). Parent: the
  same architecture with $2r_b$ outputs, seed 3112026, teacher = transient training POD
  modes 1..$r_b$ (values + $D$-scaled gradients on Q), weights $\sqrt{\sigma_i/\sigma_1}$,
  Adam 4000 steps (batch 4096, cosine 1e-3→1e-4, full check every 200) then L-BFGS 300
  (history 10, 20 line-search steps); selection by minimum full training objective only.
  Child: frozen parent + $r_b$ residual heads (21-64-64-2, seed 30102026) with base +
  1.0 × supervised divergence + 1.0 × cylinder traction, **L-BFGS 1000 iterations or
  2700 s, whichever first** (run 09 used 10,000; reduced for budget, fixed now). No
  hyperparameter, architecture or seed search; every failure retained.
- **R4 development rollouts**, arms: POD-G at r ∈ {8, 16, 24} and at $r_b$; bank-G
  (retrained, $r_b$); frozen-bank-G (r = 4) for completeness regardless of R2. Optional
  hyper-reduced POD at $r_b$: positive NNLS (Lawson–Hanson) point weights on Q fitted on
  training reduced-convection snapshots to relative 1e-4, m ≤ 4·d points; run only if the
  fit finishes within 15 min, otherwise recorded infeasible. Products: time-series plots,
  an animation (reference / ROM / pointwise error over the shedding cycles), drag and lift
  histories overlaid.
- **R5 held-out**: Re 102.5 and 107.5 opened once; **every R4 arm runs**, no selection.
  Paired timings in one allocation, CPU, 16 threads fixed, GPU burn-in irrelevant
  (CPU-only kernels) but a CPU warm-up repetition is discarded: FE reference stepper at
  production settings over a 200-step window (7 repetitions, alternating order with the
  ROM), ROM full-horizon rollout (7 repetitions) with and without full-field
  reconstruction. Reported: seconds per simulated second, ratio, offline costs (data,
  POD, supremizers, operator/tensor build, training, compilation); break-even only if the
  per-query saving is positive.

### 5.5 Metrics (all from saved arrays by scripts)

Primary, phase-sensitive: $e_u(t)=\|u_{\rm rom}-u_{\rm ref}\|_Q/\|u_{\rm ref}\|_Q$ and
$e_p(t)=\|p_{\rm rom}-p_{\rm ref}\|_Q/(U^2\sqrt{|\Omega|})$ at every saved time;
summaries: time-RMS over the horizon, maximum, last-period mean. Supplemental: C_D/C_L
histories and their max-abs differences; cycle statistics (C_D,max relative, C_L,max
absolute, St relative, Δp relative) via `cycles.py`; phase-aligned error (reference shifted
by the lag in [−T/2, T/2] on the save grid that minimizes RMS lift mismatch over the last
two periods; reported, never substituted for the primary); divergence
$D\|\nabla\cdot u\|/(U\sqrt{|\Omega|})$, inlet/outlet mass mismatch, reduced energy balance
(dE/dt + dissipation + boundary flux/traction terms). Development feasibility targets
(reported pass/fail, not a speed condition): time-RMS $e_u\le10^{-2}$, max $e_u\le3\times10^{-2}$,
time-RMS $e_p\le2\times10^{-2}$, St ≤1e-2 relative, C_D,max ≤2e-2 relative, C_L,max ≤2e-2
absolute. With one development and two held-out Reynolds numbers, every case is reported
individually (median and worst over arms and cases are listed, with every failure).

## 6. Budget, concurrency, stop rules

Total 12 allocated GPU-hours (all failures counted; FE kernels run on CPU but the GPU
allocation counts): reference 8 (smokes ≈0.3 used, spin-up ≤5, studies ≈1, population per
the G5-A2 rule), ROM 4 (R2 ≤0.5, R3 ≤1.5, R4 ≤0.75, R5 ≤0.75, contingency). At most two
allocations at once, one GPU and ≤16 cores each. Stop rules: R1 any failing gate stops all
dependent stages; R2 a failed floor triggers R3, never relaxes; R3 a failed or deadline-
stopped fit is retained and the retrained bank is still evaluated as trained; R4 a
divergent rollout is a recorded failure; R5 runs once. An extension beyond 12 GPU-h is
asked of the user and recorded, with the reason.

## 7. Deliverables

Report `reports/<date>-cylinder-unsteady/` on main with tables/figures generated from
saved JSON, the animation, provenance, the job ledger and a glossary; LAB-LOG appended
throughout with retractions; every lane mirrored via `codeonly/`; no merges.

## Amendment G5-A3 — production discretization P4/P3 (user decision), 2026-11-04

Spin-up job 6845880 (pax020, 1 h 49 min 50 s = 1.83 GPU-h) reached the developed-flow rule
at t = 9.5 (7600 steps at dt = 1/800, 0.861 s/step); S\* = `spinup.npz` SHA
`5fa0e84231f9785984852fa4fe2e35368908438b745d4a0a9a3cc1b15bc316e9`. Its developed P3/P2
divergence is ≈ 2.9e-3 > 1e-3: **R1e fails for P3/P2 (retained)**. Per G5-A2 R1 stopped and
the user was asked; the user chose **P4/P3 production** (the FD1-admitted steady
discretization) with the **reduced population** (train 90/100/110, development 95,
held-out 105) and an extended budget of about 23 GPU-h (≈ 18 reference + ≈ 5 ROM).

Changes, all before any R1b–R1e data: production spaces P4/P3 on the same mesh bytes;
R1c still compares P4/P3 with P3/P2 at $dt_0$ (production is now the finer side, so the
gate is conservative); all tolerances, windows, R1d/R1e rules unchanged. Every P4 run starts
from S\* embedded exactly into P4/P3. Settle phases (discarded, no gate) run at 1/800 with
K = 10 until the developed-flow rule, cap 4 s; continuation order 100 → 95 → 90 and
100 → 105 → 110. Recorded windows (0.7 s settle + 1.7 s record at $dt_0$) wait for R1b.
The ROM quadrature becomes order 12 (P4 convection integrand degree 11). Non-gate settle
jobs proceed under Grok review per the user's earlier authorization for non-gate
preparation runs; R1b and every later gate still wait for the Codex review.
P3 spin-up cycle statistics at dt = 1/800 (non-gating, coarse dt): C_D,max 3.22847,
C_L,max 0.98871, St 0.301934, Δp 2.48361.

## Amendment G5-A4 — settle cap 4 s → 6 s (user decision), 2026-11-04

Settles Re 95 (job 6863601) and Re 105 (job 6863602), P4/P3 from the Re 100 settled state
(SHA b77b6138…), both **failed the developed-flow rule at the 4 s cap** (retained; 2.0 GPU-h
each, 2.25 s/step): the last lift-extremum changes were 1.4e-4 then 7.7e-5 (Re 95) and
1.4e-4 then 7.7e-5 (Re 105), decaying roughly geometrically. Divergence ≤ 3.5e-4. Per the
predeclared rule R1 stopped and the user was asked; the user chose to raise the settle cap
to **6 s total** for every remaining settle (90, 95, 105, 110), with every tolerance
unchanged. Re 95/105 continue from their capped states (`settle.npz`, SHAs 611741ba…,
ebd8beea…) for at most 2 s; the restart (4-substep startup) means the rule must again
see four complete cycles within the continuation. Settle time is discarded and enters no
reported metric. If a continuation fails at 6 s, R1 stops again.

## Amendment G5-A5 — Codex design/code review responses, 2026-11-04 (before any R1b–R5 data)

Codex review 02 (`reviews/DESIGN-codex-02.md`): DESIGN "release with changes", CODE "not
ready". Review 01 was an administrative hold (its sandbox could not start; no files read).
Every tolerance and every recorded failure is unchanged. Responses, numbered as the review:

1. **R1b/R1c decisions fail closed** (`r1_gates.py`). A candidate $dt_0=1/c$, c ∈ (800, 1600,
   3200), qualifies only if the comparison $(1/c$ vs $1/2c)$ passes all four
   tolerances, its frozen stencil gives empirical orders in [1.5, 2.5] for each metric whose
   two differences exceed the noise floor, and the startup check at that c (4 vs 16
   substeps) passes 0.2 × every tolerance. Stencils: c = 800 → (800, 1600, 3200);
   c = 1600 and 3200 → (1600, 3200, 6400); only the candidate's own stencil counts. Noise
   floor 1e-8 for every metric (100 × the 1e-10 linear-solver tolerance). Missing, invalid
   or nonfinite evidence fails. R1c passes iff the P3/P2-vs-P4/P3 comparison at the selected
   $dt_0$ passes all four tolerances. Startup checks run at 800, 1600 and 3200.
2. **Comparison validity** (`compare.py`): both runs completed, identical hash-bound initial
   state, saves exactly covering [S\*, S\* + W] on the 1/400 grid (141 common saves), and
   every eligible coarse force time present; otherwise the comparison is invalid.
3. **Full-horizon reference uncertainty.** W stays 0.35 s. For each phase-sensitive metric
   y ∈ {e_u, e_p, e_D, e_L} the reference uncertainty over the 1.7 s horizon is propagated
   linearly in time (phase drift): $\delta_y=2\max(d_{y,\rm time},d_{y,\rm space})\cdot1.7/0.35$,
   with $d_{\rm time}$ the $dt_0$-vs-$dt_0/2$ gap and $d_{\rm space}$ the R1c gap. Cycle
   statistics use the same recipe without the time factor. A ROM-vs-reference comparison of
   y is **resolved** only if $\delta_y$ ≤ 1/5 of the corresponding §5.5 target; otherwise it
   is reported as unresolved. **R1d** compares the mean over every complete cycle of the
   Re 100 recorded window (every cycle and the min/max also reported); Featflow values are
   extracted by the same `cycles.py` (finest: C_D,max 3.2273927, C_L,max 0.9865830,
   St 0.3018410, Δp 2.4832194 with t₀ at the C_L minimum).
4. **Bank-identity projections**: reference inner products are stored per bank
   (`uv_frozen`, `uv_trained`) and the trained-bank file is SHA-bound (`rom_project.py`).
5. **Consistent initial pressure**: the rollout solves
   $[M,-B^T;B,0][\dot a;c]=[-F(a_0);0]$ at the initial time and records the continuity defect.
6. **R5 timing contract.** Workload: 0.1 s from the start of the held-out Re 105 recorded
   window, $dt_0$, identical outputs for both methods (force/probe history every step and the
   state at every 1/400 save; for the ROM additionally the velocity reconstructed on Q at
   every save, timed separately). Cold (FE factorizations; ROM operator/tensor build) and
   warm accounting reported separately. One CPU warm-up repetition immediately before each
   timed block, 7 paired repetitions in alternating order, 16 threads, one allocation; all
   repetition arrays, result checksums and work counts (steps, factorizations, Newton
   iterations) are saved. **Scope:** speed versus the production reference configuration
   only; no accuracy-matched FOM ladder is tested, so no "fastest FOM" claim is made.
7. **Quadrature check**: before any method ranking, re-export at order 14 and compare with
   order 12: reduced operators (M, K, B, C1, C2, force functionals) relative Frobenius change
   ≤ 1e-6 for POD r16 and every bank arm; R2 floors relative change ≤ 1e-3; R4 primary metrics
   relative change ≤ 1e-2. Failure marks the affected ranking unresolved.
8. **Embedding witness**: the loader measures the L2 change of velocity and pressure under the
   P3→P4/P3 transfer and stops above 1e-10 relative; both R1c arms start from the original P3
   S\* (P3 natively, P4/P3 by the witnessed embedding).
9. **Retention and budget.** Snapshot chunks are kept on the paralab share for the whole
   campaign (no deletion) and copied to checksum-verified local retention directories
   (outside Git; SHA manifests committed); R1f requires both copies to match. Hard budget
   ceiling **25 allocated GPU-h** cumulative (user-approved ≈ 23 plus the A4 increment); before
   each stage, if used + projected exceeds 25, the user is asked before any submission.
10. Phase-aligned search uses one fixed support (saves in [t_end − 1.5T, t_end − 0.5T], lags
   |k| ≤ T/2); the 1996 ranges are verified against Table 4 of the hosted compilation
   (`sources/karlin_compilation/`, page 13), whose Δp convention puts t₀ at the C_L maximum;
   `cycles.py` reports both conventions and each source is compared in its own.
R3 uses SciPy L-BFGS-B (no bounds) for both L-BFGS phases; the steady lane used its own
deterministic L-BFGS with Armijo search — same method class, different implementation.

### G5-A5 revision 2 — Codex review 03 (`reviews/DESIGN-codex-03.md`), 2026-11-04, before R1b data

- (1) The noise floor is **demonstrated**: a roundoff control reruns the 1/1600 P4 study with 8
  instead of 16 threads (only floating-point reduction order differs); floor_y = 10 × that gap,
  per metric. A missing control fails R1b. Every stencil, startup, roundoff and space record
  must carry the same initial-state hash, the same start time and W = 0.35, or R1b fails.
- (2) `compare.py` takes no window arguments: the window starts at the common initial-state
  time, lasts exactly W = 0.35 s (single `study` stage of that duration), needs 141 common
  saves on the 1/400 grid, the exact coarse force grid (all steps except the two startup rows)
  and SHA-verified snapshot chunks.
- (3) Statistic-specific reference uncertainty, from the R1b ($dt_0$ vs $dt_0/2$) and R1c gaps
  $d$: $\delta_{C_{D,\max}}=2\max(d_{e_D})\cdot\max|C_D|$ (a sup-norm bound on a difference of
  maxima of the sampled histories), $\delta_{C_{L,\max}}=2\max(d_{e_L})$,
  $\delta_{\Delta p}=2\max(d_{e_{\Delta p}})$, and $\delta_{St}/St=2\max(d_{e_L})/(2\pi f A_L W)$
  (phase shift implied by the lift gap over W, $A_L$ the lift amplitude). Field/force
  uncertainty over 1.7 s uses the linear propagation factor 1.7/0.35. These are **heuristic
  estimates, not bounds**; comparisons classified with them are labelled "resolved under the
  linear-propagation estimate".
- (7) The quadrature check compares operators in the **raw** basis (the FE POD/supremizer
  coefficient modes are quadrature-independent, the bank functions are fixed), i.e. before
  orthonormalization, so no rotation or sign ambiguity enters; it covers every ranked arm
  (POD 8/16/24/r_b, frozen and retrained bank); relative Frobenius change uses the order-14
  array norm as denominator with floor 1e-14.
- (10) The 1996-convention Δp of a cycle whose evaluation time lies beyond the saved history is
  NaN and excluded from that statistic (counts reported); no extrapolation.
- **A4 is a post-failure extension.** Success of any settle under the 6 s cap qualifies the
  amended 6 s protocol only; the original 4 s protocol failed for Re 95 and Re 105.

## 8b. Review status update — 2026-11-04

Codex reviews 02–05 (`reviews/DESIGN-codex-0{2,3,4,5}.md`) completed the missing Codex gate:
review 04 **DESIGN: RELEASE** for the amended protocol (with the heuristic-uncertainty
qualification of A5 rev.2 retained wherever accuracy conclusions appear); review 05
**CODE: READY FOR R1b** after the cycles.py NaN fix. The R1b–R1f reference stages are
released; R2–R5 code (quadrature check, R5 timing implementation) needs its own review
before those stages execute.
- 2026-11-04, R1b attempt 1 (jobs 6867838/6867839, 35 s each): the embedding witness **stopped
  every run** — `GridFunction.Set` changed velocity by 5.6e-13 but pressure by 5.0e-10 relative
  (> 1e-10; local-projection roundoff). The bound is unchanged. One ordinary repair: exact
  node-by-node coefficient injection exploiting NGSolve's hierarchical H1 bases, still verified
  by the same witness. Attempt 1 is retained.

## Amendment G5-A6 — budget ceiling 42 GPU-h (user decision), 2026-11-05

R1b (jobs 6867873/6867877 + comparison 6868804) **passed with $dt_0=1/3200$**: 1/800 and
1/1600 fail only the lift tolerance (8.5e-4, 2.14e-4 > 2e-4); 1/3200 vs 1/6400 passes all
four (e_u 5.0e-5, e_p 2.3e-5, e_D 6.9e-6, e_L 5.4e-5); empirical orders ≈ 2; startup and
roundoff controls pass. At 1.66 s/step (P4/P3, K = 40) the five recorded trajectories cost
≈ 3.5 GPU-h each; with 12.15 GPU-h used, the projected total is ≈ 40 GPU-h > the 25 GPU-h
ceiling, so the user was asked and chose to **extend the hard ceiling to 42 GPU-h** keeping
every design setting (0.7 s settle + 1.7 s record at $dt_0$). Records depend on R1c.

### G5-A5 revision 3 — Codex ROM-code review 01 (`reviews/ROM-codex-01.md`), 2026-11-05, before any R1d–R5 data

- Bases are eigen-truncated (1e-12) then re-orthogonalized once (Cholesky) and **verified**
  (max |ΦᵀWΦ − I| ≤ 1e-12, else the arm stops); squared errors from norm expansions below
  −1e-12 × scale stop the computation (smaller negatives are roundoff, set to 0 and counted).
- R1e validates the exact population {90, 95, 100, 105, 110}, production configuration,
  settle→record state linkage (record initial SHA = last settle state), cumulative settle
  ≤ 6 s, retained snapshot bytes, exactly 681 saves on the 1/400 grid and 171 diagnostics in
  the recorded window. **Held-out Re 105 is sealed in R1e**: only quality checks are computed
  and exported; no cycle statistics or histories before R5. R1f (independent audit of the
  saved arrays) remains a separate Codex gate.
- R4 refuses to run unless R1b–R1e pass and the uncertainty inputs are bound (same initial
  state and window, R1b/R1c pass, $dt_0$ = 1/3200). Every target is reported with its reference
  uncertainty and classification ("resolved under the linear-propagation estimate" iff δ ≤
  target/5, otherwise "unresolved"). The **reduced energy balance** (kinetic rate, dissipation,
  inlet/outlet convective flux and traction work, finite-divergence terms; walls/cylinder carry
  u = 0) is reported every 4th save.
- **Initial continuity defect**: reported as $\|Ba_0+B_L\|/\|B_L\|$; ≤ 1e-3 is "qualified",
  above it the arm is flagged "large initial defect" (the initialization is never changed after
  results); the primary pressure metric keeps the t₀ sample, and the time-RMS excluding t₀ is
  reported as a supplement.
- Quadrature check: both exports must carry identical FE coefficient modes and the identical
  frozen-bank SHA; the trained bank is re-evaluated on the order-14 rule from the saved
  parameters (evaluation-only replay, identical parameter digest); criteria: operators ≤ 1e-6,
  development floors ≤ 1e-3, R4 time-RMS e_u/e_p ≤ 1e-2 relative change; failure → ranking
  unresolved.
- R5: identical output contract (reaction forces and probes every step, state at every save;
  the FE runs in an opt-in `timing_mode` without the per-step surface integral and kinetic
  energy, persistence after the timed loop; ROM reconstruction timed separately); an untimed
  20-step warm-up of the same method immediately before every timed invocation; input identity
  binding; per-repetition accuracy witnesses; all Newton iterations counted.
- Manifests must state `velocity_order` (4) and `quadrature_order` (12) explicitly; reported
  dimensions are velocity $d$, pressure $d_p$ and total $d+d_p$ with retained ranks; peak RSS
  recorded.
- 2026-11-05: Codex ROM-code reviews 01–04 (`reviews/ROM-codex-0{1..4}.md`) completed: after the
  rev.3 responses and two further fix rounds (evaluated-basis orthonormality with actual Gram
  matrices in all norm expansions; full R1e provenance/configuration checks; quadrature-check
  identity binding incl. the trained bank; symmetric R5 cold timing and force/probe/pressure
  witnesses) the verdicts are **R1d/R1e evaluator, R2, R3, R4, R5: READY**. R1f (Codex audit of
  the saved arrays) remains a separate gate after the population completes.
- 2026-11-06 **R1f PASS** (Codex audit, `reviews/R1F-codex-audit.md`): 107 snapshot chunks and 13
  states (17.15 GB) match both copies' hashes; R1b–R1e numbers independently recomputed;
  held-out arrays not opened. Correction to the G5-A6 narrative (decision unchanged): at 1/800
  and 1/1600 the candidates failed the lift pair tolerance **and** the startup lift check
  (1.85e-4 and 4.61e-5 > 4e-5); the saved gate JSON recorded both. The reference is admitted
  for R2–R5.

## Results record — R2–R4 (2026-11-06), reported against the predeclared rules

- **R2**: frozen bank fails the floor (median 2.55e-1); POD r16 first passing (4.83e-3 /
  5.74e-3) → R3 at r_b = 16. **R3** (job 6924961, 37 min): parent and child both stopped at
  their predeclared iteration limits (not convergence). **R4** development Re 95: POD r16 and r24
  meet all six targets; retrained bank r16 misses the e_u-RMS target (1.45e-2) and has
  divergence 4.6e-2 (POD ≈ 3e-4); frozen bank fails all; POD r8 misses three.
- **Quadrature check (G5-A5 item 7)**: attempt 1 refused at its identity guard (independently
  recomputed FE modes differ at roundoff, 1e-10 > 1e-12); attempt 2 reused the order-12 modes
  by hash. Result: development floors change ≤ 3.8e-9 and R4 primary metrics ≤ 1.6e-7 (pass),
  but the operator criterion (≤ 1e-6) **fails for every arm, solely through `BL`**
  (∫ψ_k ∇·ū, a vector that vanishes in exact arithmetic for Taylor–Hood lifts; relative changes
  3.3e-4–6.3e-4); every other operator changes ≤ 1.2e-7. **By the predeclared consequence the R4
  method ranking is reported as "unresolved under the quadrature check"**, with this evidence
  shown beside it. The rule is not changed.
