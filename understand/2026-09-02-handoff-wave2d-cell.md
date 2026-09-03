# Handoff — Wave 2D cell: reflective + non-reflective, accuracy first, then cost vs FOM

Paste the block below into a fresh session. Pointer document: numbers live in `LAB-LOG.md`,
the reports, and the run JSONs. Nothing here should be quoted without opening the source.

---

You are resuming a nonlinear model-order-reduction project to run a **Wave 2D cell**. Read, in
order, before doing anything:

1. `/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md` — canonical record.
   "Where things stand" plus the 2026-08-14, 08-16, and 08-27..30 chronology. Read the retractions.
2. `understand/2026-08-30-session-handoff-stokes.md` — the previous session's handoff; project
   conventions, the recurring failure modes, and the carried-forward conditions.
3. `worktrees/2026-08-30-waves-vector/experiments/separable-decoder/WAVES-RETIRED.md` and
   `WAVES-DESIGN-AUDIT-codex-gpt56sol.md` — a **fully worked design and adversarial audit for
   exactly this cell**, written and then shelved. Start from it; do not re-derive it.
4. `CLAUDE.md` at the repo root and in `Tunable-NM-ROM-Claude/` — binding operational rules.

## Why this is being revisited after being retired

Waves was retired on 2026-08-30 on the strength of three arguments. **Two have since fallen.**

- **The n-width argument was a misreading of this project's own data.** I claimed traveling waves
  decay too slowly in Kolmogorov n-width for the decoder to represent them, citing POD floors ~4x
  worse than heat. That bounds *POD*, not us: the 2026-08-14 cell measured the FiLM decoder on
  Wave 2D at **2.80e-2 -> 3.51e-2** across the resolution ladder against POD-6 at 4.41e-1 ->
  4.52e-1 — **12.9-15.8x better than POD, and flat as the mesh refines**. The decoder represents
  wave solutions well. Slow n-width is exactly the regime a nonlinear decoder exists to beat.
- **Architecture independence is unestablished.** The wave ROM failure
  (`exp/2026-08-16-wave2d-rom-latent-stepping`: held-out 8.78e-1 vs decoder ceiling 1.72e-1 and
  POD 3.42e-1; end-time energy ratio 0.27) was measured with a **FiLM/coordinate decoder**. The
  separable decoder did not exist until 2026-08-22 and the exact-residual machinery until 08-27.

**The surviving argument, and its untested competitor.** The failure is isolated to the *latent
solve*, not the representation — the decoder could represent the answer 5x better than the solver
found it. Two candidate causes:

1. **Structural.** On a fixed linear subspace, POD-Galerkin of a wave system gives a reduced
   operator `V^T Delta V` that is still symmetric, so the reduced system is still Hamiltonian and
   Crank-Nicolson preserves its quadratic invariant exactly (this is why POD reads 1.000003). A
   nonlinear manifold has no such reduced operator, so nothing preserves the invariant.
2. **Quadrature error.** That cell used *sampled* NNLS quadrature, so the residual itself was
   inexact. An inexact residual leaks energy on a conservative system with no manifold argument
   needed at all.

**The current method eliminates cause 2 outright for waves.** Waves is linear, so the entire
residual collapses to precomputed matrices and is evaluated **exactly, zero quadrature error**.
If cause 2 dominated, waves was broken by something already fixed.

Also note: "structure-preserving latent stepping is an open problem" was overstated. Symplectic
model order reduction, Hamiltonian ROMs, and structure-preserving/symplectic autoencoders are an
active literature. It is open *for this project*, not for the field.

## The scientific design — why BOTH boundary conditions

This is not two boundary conditions, it is a **controlled test of the mechanism above**:

- **Reflective** (homogeneous Dirichlet) is **conservative** — the case the structural argument
  says must fail.
- **Non-reflective** (Sommerfeld / absorbing) **radiates energy out of the domain**, making the
  system effectively **dissipative** — like Burgers, where this method works.

If the structural argument is right: reflective degrades, non-reflective behaves. If both work,
the argument was wrong and quadrature error was the culprit. If both fail, the limitation is
deeper than either explanation. **All three outcomes are publishable and all three change the
project's scope statement.** Predeclare this prediction before running.

## Phase structure — accuracy before cost, and cost is the goal

**Phase 1 — FOM.** Wave 2D as a first-order `(u,v)` system, Crank-Nicolson (NOT backward Euler —
it damps waves artificially and would contaminate everything). Both boundary configurations. The
retired design and its audit already contain: the correct Sommerfeld signs; the fact that the
absorbing case makes the state carry `n = N` rather than `N-2` nodes because boundary values
become unknowns; energy `E = (dx/2)||v||^2 + (c^2 dx/2)||D_e u||^2` with `D_e^T D_e = -L`, which
CN preserves **exactly in exact arithmetic**, not merely to O(dt^2); and the discrete reflection
coefficient for first-order Sommerfeld, `|R|^2 = tan^2(theta/4) ~ (k_h dx)^2/16`, so a flat
reflection threshold **must** fail at coarse meshes and the gate has to be a convergence gate.
A manufactured-solution gate is mandatory.

**Phase 2 — decoder.** Separable decoder `u = G h(z)` on the `(u,v)` vector state. The Stokes cell
(`exp/2026-08-30-stokes-vector`) already rehearsed vector-valued decoders — reuse that experience.
Report the decoder reconstruction ceiling; the 08-14 cell says to expect roughly 2.8e-2 to 3.5e-2,
so a ceiling far worse than that is a bug, not a finding.

**Phase 3 — ROM accuracy, and this is a PREDECLARED STOP GATE.** Fix the bar and record it in
`LAB-LOG.md` *before* running. Suggested: the ROM must (a) beat plain POD at matched latent
dimension by a stated margin, and (b) come within a stated factor of its own decoder ceiling.
The prior failure was 5x above ceiling and 2.6x *worse* than POD. **Report the end-time energy
ratio alongside the error every time** — it is the mechanism, and it is what distinguishes the
two candidate causes. If the gate fails, STOP and report the negative; do not proceed to cost.
A cost comparison for a method less accurate than POD is not a result.

**Phase 4 — the goal: cost vs FOM across increasing resolution.** Only if phase 3 passes. Ladder
the mesh and measure our online cost against the FOM at matched accuracy, both arms timed under
the same conditions on one GPU in one job.

**The cost comparison is where this cell is most likely to mislead, so read this.** The wave FOM
is **cheap** — an SPD operator with fast CG. This project has already logged a wave "speedup" of
0.80-0.86x as an honest negative for exactly that reason. So: the FOM baseline must be a
**properly preconditioned, well-tuned** solver, not a strawman; the crossover may be late or may
not exist; and both arms must be timed in the same process with a balanced order. The
quadrature-free residual is flat in N by construction, so the interesting question is *where* it
crosses a good FOM, and it is entirely possible the honest answer is "not in the range tested" —
which is a result, not a failure to be tuned away.

## Landmines specific to this cell, already paid for

- **Sommerfeld is a DAE, not an ODE.** The absorbing boundary couples `u` and `v`, so the residual
  does *not* collapse to two matrices. It still collapses exactly — zero sample points survives —
  but needs four: `Phi^T S G`, `Phi^T L_i G`, `Phi^T Q G`, `Phi^T C G`. See the audit.
- **A gate that cannot fail.** Dirichlet sine test modes vanish at the endpoints, so `Phi^T`
  deletes every residual row supported only at the boundary — an exactness gate can pass while the
  ROM enforces no boundary condition at all. The absorbing configuration needs its own test space
  and explicit boundary residual rows.
- **Sign convention.** This repo uses `L Phi = -Phi Lambda`, so `A = Phi^T L G = -Lambda B`. A gate
  written as `||A - Lambda B||` measures ~2, not roundoff.
- **Non-symmetry is not what kills the diagonal shortcut** — a non-symmetric operator still has one
  if the test space holds its LEFT eigenvectors. What kills it is `Phi` not being eigenvectors of
  the absorbing generator.

## Project rules that are not optional

- Experiments live in dated worktrees on their own branches. **Ask before creating one.** Never
  branch from `main`.
- **Never hand-type a number into a report** — generate tables from run JSONs with a script.
- **Verify every conclusion and code file with an independent `codex exec` pass** (defaults to
  `gpt-5.6-sol`, xhigh); audit designs *before* implementation. Launch with `< /dev/null`, omit
  `-s read-only`, and **scope reviews tightly** — broad prompts get killed by a wall-clock cap.
- **The recurring failure mode is gates, not numerics.** In the Stokes cell six separate gates were
  found that compared something to itself (one read exactly `1.000000000000`), five thresholds were
  absolute where the quantity scales with the mesh, and a NaN passed silently because
  `max([finite, nan])` returns the finite value. Normalize every threshold; give every gate a
  negative control that actually fires; treat a suspiciously exact value as tautology.
- Append to `LAB-LOG.md` before the session ends, including what was **retracted**.
