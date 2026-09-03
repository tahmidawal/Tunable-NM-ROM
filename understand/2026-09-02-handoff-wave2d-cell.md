# Handoff: the Wave 2D cell (reflective + absorbing, then a cost ladder vs FOM)

Written 2026-09-02, revised 2026-09-03 after an independent analysis changed the diagnosis
of the 2026-08-16 wave failure. **Paste the block below into a fresh session.** Everything
it refers to is committed on `main` or on the named branches.

---

**STATUS 2026-09-03 (end of the session that ran it): the cell is DONE through phase 3; phase 4 was
not run.** Read `WAVE2D-NOTES.md` on the branch (reading + generated tables + eight retractions) and
the 2026-09-03 lab-log entry before anything else. Summary for a new session:

- Reflective stop gate W3 **fails on every head (`sup`, `auto`, `auto+vc`) and both ROM arms at N=64
  and 128**. The incumbent arm A sits at 2–5× its floor with energy growing 4.5–19×; the variational
  arm C conserves the reduced energy to 1e-4 over 4T and is **no more accurate** (4–6× floor). The
  08-16 energy explanation is refuted; the 08-16 verdict stands, with the mechanism named: the
  tangent-space velocity residual of every reflective head is no better than POD-K's (0.63–0.74 vs
  0.69–0.83), while on the absorbing wave it is 0.23–0.30 vs 0.68–0.79 and `sup` passes W3 on both
  arms at both N (1.05–1.29× floor).
- The predeclared decision row is "G0 pass + reflective arm C fail + W4 pass → INCONCLUSIVE" for
  structural-vs-manifold; what is settled is that energy conservation is not the cure.
- Phase 4 (`wav2d_ladder.py`, written, smoke-tested, PCG + exact DST FOM, bank prolongation across
  N) is gated on a reflective pass and was **not run**; running it on absorbing `sup` is the user's
  call (`cluster/stage_ladder.sh`, `cluster/make_ladder_sbatch.sh`).
- Certified heads/banks/oracles are committed under `runs/wav2d/p2fix_*/cache`; result JSONs and
  logs under `runs/wav2d/p2fix_*/out,logs`; the cluster namespace `wav2d/` is empty.
- Eight retractions, three of them absolute thresholds on mesh-scaling quantities and two of them
  controls that could not fail (caught only by real N=64 numbers, not by two design audits or the
  N=16 smoke). Read them before writing a gate.
- Open: fold the Codex verification of the reading (`WAVE2D-RESULTS-VERIFY-codex-gpt56sol.md` if
  present) into the notes; the FiLM 08-14 comparator on this dataset is not recomputed; a
  tangent-space gate with a demanding bar should precede any conservative-PDE ROM on the roadmap.

The original handoff text follows for context; its "predeclared prediction" was scored in the
notes (wrong for `auto`'s G0 and for the `sup`/`auto+vc` reflective W3).

---


You are resuming a nonlinear model-order-reduction project to run a **Wave 2D cell**. Read,
in order, before doing anything:

1. `LAB-LOG.md` — canonical record. "Where things stand", then the 2026-08-14, 08-16,
   08-30 and 09-03 chronology, **including the retractions**.
2. `worktrees/2026-09-03-wave2d-mechanism/experiments/separable-decoder/WAVE2D-DESIGN.md` (branch `exp/2026-09-03-wave2d-mechanism`) — **the design for exactly this cell**, revision r2,
   Codex-audited (`WAVE2D-DESIGN-AUDIT-r1-codex-gpt56sol.md` and `-r2-` beside it).
   Implement it; do not re-derive it. Where the audit marked an item WRONG the design was
   revised; where it marked NEEDS-RESTATEMENT the restatement is in the design.
3. `understand/2026-09-02-fable-wave-rom-case.md` — the analysis that reopened waves, with
   the 08-16 numbers it rests on and the 1D CPU check that reproduces the phenomenology.
4. `worktrees/2026-08-30-waves-vector/experiments/separable-decoder/WAVES-RETIRED.md` and
   `WAVES-DESIGN-AUDIT-codex-gpt56sol.md` — the shelved 1D design and its audit; the
   surviving parts (sign convention, general block form, Sommerfeld pitfalls) are already
   folded into item 2.
5. `understand/2026-08-30-session-handoff-stokes.md` — project conventions and the
   recurring failure modes. `CLAUDE.md` at both levels — binding.

**Why this is being revisited.** Waves was retired on 2026-08-30 on three arguments. The
n-width argument was a misreading of this project's own data (the 08-14 FiLM decoder
reached 2.80e-2 → 3.51e-2 on Wave 2D against POD-6 at 4.41e-1, flat in N — the slow width
bounds POD, not a manifold). Architecture independence was never established (the 08-16
failure was a FiLM auto-decoder that predates the separable decoder). And the structural
argument — "a nonlinear manifold breaks the CN time-reversal symmetry, energy is destroyed,
not fixable by tuning" — has been **re-diagnosed**: a Lagrangian (tangent-space Galerkin,
variationally integrated) manifold ROM conserves the pulled-back energy for *any* smooth
decoder at two precomputed R×R matrices; in a 1D check it removes the energy drift
completely **and does not recover the accuracy**. The 08-16 failure was a
**manifold-quality** failure (auto-decoder generalising 2.7× worse than it fit; tangent
space wrong), which conservative dynamics expose and dissipative dynamics hide.
**Quadrature error is dead as a cause**: the 08-16 full-grid, no-EQ arms failed identically
to the EQ arm. Do not re-propose it.

**Why both boundary conditions on the same IC family.** Reflective is conservative;
absorbing (first-order Engquist–Majda, ghost-closed, a damped ODE — *not* a DAE in this
formulation) is dissipative. **Predeclared prediction (design r3): the causal claim is an
across-head interaction, not a per-BC contrast** — after the rank gate D2 and the time-step
gate W6 pass, reflective arm C passes W3 **iff** the head passes the manifold gates G0.
Per head: `auto` fails G0 and fails reflective W3; `sup` (on (μ,t)) and `auto+vc` pass both.
The absorbing runs are the dissipative *comparator* (predicted ≤ 2× their pre-exit floor for
every head) and are **not** evidence for the causal claim (the absorbing data may be
intrinsically easier, damping erases error, and the banks differ). Mixed outcomes are
recorded as inconclusive with the alternatives named, never as a verdict.

**Phases, each gated, in order.** (1) FOM, both BCs — **DONE, ALL PASS at N=64 and 128 on
2026-09-03** (`runs/wav2d/wav2d_fom_gates_N{64,128}.json`, table in `WAVE2D-NOTES.md`; three
retractions, all absolute thresholds on mesh-scaling quantities — read them). The absorber's
reflected-energy fraction matches the discrete prediction tan⁴(θ/4) within 2% at N=64…512,
slope 4.01. The inherited family's blobs carry a 1-cell wall discontinuity (hard mask), which
limits convergence order on family trajectories — a data property to keep in mind when
reading manifold quality. (2) Decoder + **G0 manifold
gates** per head arm (`auto`, `sup`, `auto+vc`): held-out oracle vs POD-K, train/held-out
gap, tangent-space velocity residual vs POD-K on the same states, stepdiag from oracle
starts with the `hold` control. G0 is a prediction about phase 3; every head arm proceeds.
(3) **ROM accuracy STOP gate W3** per (BC, head, ROM arm A = incumbent LSPG on the exact
Newmark residual; ROM arm C = variational): ≤1.5× oracle floor and ≤0.5× POD-K at T and 4T,
energy ratio in [0.9,1.1] reflective; plus W4, arm C energy bounded regardless of accuracy
(a W4 failure means the implementation is wrong, not the manifold). **If no pair passes W3 on
the reflective BC, stop, report the negative, do not run the ladder.** (4) Cost ladder
N∈{64,128,256,512}, both BCs, arms FOM-ref / **FOM-tol (the fastest configuration on a
predeclared SUBSTEPS × CG-tol grid that meets the ROM's error, with the best available
solver)** / POD-K / POD-R / ours-A / ours-C, one process, balanced order, raw reps kept, all 16
cases must complete, K/R/M fixed across the ladder; report **both** the latent-only kernel
time and the end-to-end time (cold start + stepping + decode at the 51 output times), the
end-to-end figure being the headline.

**The cost comparison is where this will mislead you.** The linear wave FOM is cheap and a
POD-R CN ROM is a precomputed R×R recurrence with no iteration. The predeclared expectation
is that ours loses to POD-R at every N and to FOM-tol below N≈256, with a possible crossover
at N≥512 because the ROM cost is flat in N. **"No crossover in the range tested" is a result,
not a tuning failure.** The value of the cell is the corrected record, a certified
conservative-stepping arm, and a tangent-space gate — prerequisites for compressible
Euler/acoustics on the roadmap.

**Landmines already paid for.** Repo convention `LΦ = −ΦΛ`, so `A = −ΛB`; a gate written
`‖A − ΛB‖` reads ~2. Dirichlet sine test modes annihilate boundary-supported residual rows;
the absorbing arm uses cosine modes on all N² nodes and its C = ΦᵀD_BG term is what kills
the diagonal shortcut (non-symmetry is not the reason). Never report the kinematic-recursion
energy (the 08-16 "0.27" disagreed with the dynamic estimate by ~100×); report E_r(z,ż) or
the CN-consistent dynamic velocity. Component scaling: u and v differ by up to tens; every
metric is in the energy norm with per-trajectory weights. Wave snapshot norms pass near
zero during kinetic/potential exchange — never per-snapshot relative errors. Absorbing
trajectories decay to ~0 late in the horizon — report errors on a fixed pre-exit window as
well as the full horizon.

**Non-optional rules.** The worktree exists (`worktrees/2026-09-03-wave2d-mechanism`, cut
from `exp/2026-08-30-stokes-vector` with the user's approval) — work in it; do not create
another without asking; one session writes to one worktree; never hand-type a number
into a report — tables from run JSONs by script; verify every conclusion and every code file
with an independent tightly-scoped `codex exec` pass, and audit designs before
implementation; **the recurring failure mode is gates, not numerics** — six gates in the
Stokes cell compared something to itself, five thresholds were absolute for mesh-scaling
quantities, a NaN passed because `max([finite, nan])` returns the finite value, and
`PRECOND` must `raise`, not `assert`; thresholds are frozen before the first run and any
change after a number is seen is a numbered retraction; append to `LAB-LOG.md` before
ending, retractions first.
