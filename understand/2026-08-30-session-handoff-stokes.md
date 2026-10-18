# Handoff prompt — resuming after the Stokes cell (2026-08-30)

Paste the block below into a fresh session. It is a pointer document: every number lives in a
report or run JSON, and `LAB-LOG.md` is canonical. Nothing here should be quoted without opening
the source.

---

You are resuming a nonlinear model-order-reduction project. Read these three, in order, before
doing anything:

1. `/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md` — canonical record. Read
   "Where things stand" (dated 2026-08-30) and the whole 2026-08-27..30 chronology, including the
   retractions. Several results supersede earlier claims.
2. `understand/2026-08-30-experiment-roadmap.md` — the ordered, gated experiment list.
3. `CLAUDE.md` at the repo root and in `Tunable-NM-ROM-Claude/` — binding operational rules.

## Where things stand

The organizing principle: **score every PDE by the degree of its residual as a polynomial in the
state.** Degree 1 (Poisson, heat, Stokes) and degree 2 (Burgers, NS convection) are sample-free —
the grid folds into one precomputed matrix or 3-index tensor. Non-polynomial terms (`|u|`
upwinding, `1/rho`, limiters) still need sampling.

Done and confirmed: Poisson quadrature-free (~1e-13, flat in N, deletes a 25-37 s NNLS fit);
1D and 2D Burgers sample-free via the precomputed quadratic tensor (2D: tensor flat at ~27 ms
while the full-grid oracle grows 33 -> 670 ms); steady Stokes as a Navier-Stokes rehearsal
(residual exact to 6e-16..7e-15, pressure eliminated, cost flat in N, crossing the FOM at N=128).

Two framing statements that supersede earlier wording:
- **Non-symmetry is not what kills the diagonal shortcut.** A non-symmetric operator still has one
  if the test space holds its LEFT eigenvectors; self-adjointness is sufficient, not necessary.
- **On a linear PDE with affine parameter dependence the nonlinear head can never buy accuracy —
  its entire value is cost.** The solution manifold is literally a linear subspace. This is the
  correct reading of the Poisson result, and it is why Stokes was a rehearsal and not a result.

Waves is **retired, not pending**: it was already run twice here and the ROM fails structurally on
a nonlinear manifold (end-time energy ratio 0.27, "not fixable by tuning"). Read
`worktrees/2026-08-30-waves-vector/experiments/separable-decoder/WAVES-RETIRED.md` before ever
re-proposing it.

## What to do next

Roadmap item 2: **sign-changing / skew-symmetric Burgers (1D)**. Three arms — upwind + plain
tensor (expected failure, to be characterized not avoided), upwind + split form (sample only the
`|u| * Delta_h u` stabilizer), skew-symmetric + plain tensor (expected exact for any sign).
Decides whether the non-negativity caveat is a property of the method or of the upwind stencil.
Methodological guard: grade each arm's ROM against **its own** FOM, and report both FOMs'
discretization error against a resolved reference at matched dx — otherwise "the caveat
disappears" could just mean a worse solver was used.

Then item 3, 2D incompressible Navier-Stokes, which is what the Stokes cell exists to de-risk.

## Carried-forward conditions for the Navier-Stokes cell

- `M >= R` is **necessary but not sufficient** — at M=R=32 the operator `A` was numerically rank
  28-30 and the direct arm degraded to 1e-2.
- The existing `b2d_tensor_common.py` is **not reusable**: it is scalar, collocated, and
  fixed-positive-upwind. MAC NS needs two staggered lattices, cross-component interpolation, and a
  new skew-symmetric bilinear tensor with its own build and Jacobian gates.
- The decoder's `bc(x)` mask must stay removed — it destroys divergence-freeness.
- Use the metric-reorthogonalized psi-route bank. Naive Gram POD amplifies snapshot divergence by
  1/sigma_i (measured up to 9.0e3) and fails under mild gradient contamination.
- NS is unsteady; steady Stokes did **not** rehearse the time/mass block. The risk is bounded by NS
  being dissipative like Burgers rather than conservative like waves, but that is an expectation,
  not a result.

## Four claims not yet verified by an independent model

The `M >= R` insufficiency; the near-collision rank puzzle (family Jacobian collapses 8->4 at zero
blob separation while rank(A * J_h) stays 8 with unmoved conditioning); S7 timing fairness; the
~2x ROM-vs-oracle gap. Two Codex passes were killed by a background wall-clock cap; the three
headline items were verified directly instead.

## How this project works, and the mistakes it keeps making

- Experiments live in dated worktrees on their own branches. **Ask before creating one.** Never
  branch from `main` — it is a frozen baseline whose heat rollout is known broken.
- **Never hand-type a number into a report.** Generate tables from run JSONs with a script in
  `reports/`. Every prose number that has gone wrong here went wrong by being typed by hand.
- **Verify every conclusion and code file with an independent `codex exec` pass** (the box defaults
  to `gpu-5.6-sol`, xhigh). Audit design documents *before* implementation. Launch with
  `< /dev/null` or it hangs on stdin; omit `-s read-only` (bubblewrap is broken on this box) and
  put the guardrails in the prompt. Scope each review tightly — broad prompts get killed by a
  wall-clock cap before producing output.
- **The recurring failure mode is gates, not numerics.** Across the Stokes cell the MAC operators
  were right in the first build and never changed, while six separate gates were found that
  **compared something to itself** (one read exactly `1.000000000000`), five thresholds were
  absolute where the quantity scales with the mesh, and a NaN passed the aggregates silently
  because Python's `max([finite, nan])` returns the finite value. Write every threshold normalized;
  give every gate a negative control that actually fires; treat a suspiciously exact value as
  tautology until proven otherwise.
- **Predeclare stop gates and record them before the result exists.** The Stokes cell's 3x bar was
  written into the lab log and stored in the artifact as baseline/3 before training ran.
- Append to `LAB-LOG.md` before the session ends. Record what was **retracted** — that matters more
  than the successes and is the part that gets silently dropped.

## Open decisions for the user

**Thirteen unmerged experiment branches**, now including `exp/2026-08-30-waves-vector` (retired
archive) and `exp/2026-08-30-stokes-vector`. This has been deferred repeatedly and is a real
liability. Ask.

Working tree on `main` has pre-existing modifications to `reports/2026-08-28-presentation-notes.md`
and untracked `Older Paper /` and `meeting_notes/` — leave them alone. `main` is committed but not
pushed.
