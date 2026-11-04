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
