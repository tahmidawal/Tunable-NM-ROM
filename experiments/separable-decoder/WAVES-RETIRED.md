# Waves: designed, audited, RETIRED before implementation (2026-08-30)

This branch carries a design (`WAVES-DESIGN.md`) and its adversarial audit
(`WAVES-DESIGN-AUDIT-codex-gpt56sol.md`, Codex `gpt-5.6-sol`, xhigh). **No code was
written and no job was run.** The experiment was retired for the reasons below.
Kept as a record so the next session does not re-propose it.

## Why retired — prior work already answers it, negatively

`LAB-LOG.md` records two earlier wave cells:

- **`exp/2026-08-14-wave2d-coord-rom`** already built `u_tt = c^2 Delta u` as a
  first-order `(u,v)` system with Crank-Nicolson (energy drift <= ~2e-11, observed
  orders 1.19-2.31), and already measured the Kolmogorov-width question that was the
  main motivation here: *"POD floors here are roughly 4x worse than heat's at matched
  rank -- the slow Kolmogorov decay a hyperbolic transport problem is supposed to
  have, confirmed."*
- **`exp/2026-08-16-wave2d-rom-latent-stepping`** ran the ROM and found it **fails
  structurally**. At k=8, N=64: decoder ceiling 1.72e-1, ours 8.78e-1, POD 3.42e-1 --
  the ROM is 5x worse than its own decoder ceiling and 2.6x worse than plain POD.
  End-time energy ratio 0.27 vs POD's 1.000003. Log's conclusion: *"On a fixed
  subspace the Newmark/CN recurrence is time-reversible and therefore conservative;
  on a nonlinear manifold that symmetry is gone. Not fixable by tuning."*

That argument is architecture-independent -- the separable decoder is also a nonlinear
manifold -- so this design would have re-derived a known negative. The audit reached
the same conclusion independently, without being told: *"There is no ROM rollout gate.
The repository already records a wave case where residual operators were correct but
nonlinear-manifold stepping destroyed energy. Otherwise every proposed gate can pass
while the vector ROM fails structurally."*

## What the audit found in the design itself (all pre-implementation)

Verdicts: items 1, 3, 5 **WRONG**; items 2, 4, 6 **NEEDS-RESTATEMENT**. Substantive:

1. **Sign error against the repo convention.** Under `b1d_common.py`'s positive-Lambda
   convention `L Phi = -Phi Lambda`, so `A = Phi^T L G = -Lambda B`. The design's gate
   W1 (`||A - Lambda B||/||A|| <= 1e-12`) would have measured **~2**, not 1e-12. The
   design could not keep both its `r_v` sign and its W1 identity.
2. **Sommerfeld is a DAE, not an ODE.** The absorbing boundary couples `u` and `v`, so
   the residual does *not* collapse to the two matrices `A`, `B`. It still collapses
   exactly -- zero sample points survives -- but needs four:
   `B_i = Phi^T S G`, `A_i = Phi^T L_i G`, `B_b = Phi^T Q G`, `C_b = Phi^T C G`.
3. **A gate that could not fail.** Dirichlet sine test modes vanish at the endpoints,
   so `Phi^T` *deletes every residual row supported only at the boundary*: "W0 can pass
   while the ROM enforces no Sommerfeld boundary equation at all."
4. **F3's threshold was unachievable for valid reasons.** First-order Sommerfeld has
   discrete reflection `|R|^2 = tan^2(theta/4) ~ (k_h dx)^2/16`, so a flat 1e-3 gate
   *must* fail at N=128 and N=256. It needed to be a convergence gate, not a threshold.
5. **W3 was not an n-width measurement.** POD singular tails are training-set Frobenius
   quantities; the decoder number is held-out reconstruction. Not comparable as written.

## What SURVIVES and carries forward

- **The central algebraic claim is TRUE**: `u = G h` implies `P^T L u = (P^T L G) h`
  for any fixed linear `L`. Confirmed by the audit.
- **But the motivation was stated imprecisely.** Non-symmetry is *not* what kills the
  diagonal shortcut: "Even a non-symmetric matrix has a diagonal shortcut if `P^T`
  consists of its left eigenvectors; self-adjointness is sufficient, not necessary."
  What actually kills it is that the Dirichlet sine modes are not eigenvectors of the
  absorbing generator.
- **The right general statement of the linear quadrature-free claim**, which supersedes
  the "one `A` plus one `B`" form and covers DAEs, cross-component coupling, and
  parameter dependence:
  $$R = P^\top\Big[E(\mu)\,\mathcal{G}\,\tfrac{\Delta\eta}{\Delta t} - K(\mu)\,\mathcal{G}\,\bar\eta\Big],\qquad \mathcal{G} = \mathrm{diag}(G,G)$$
  precomputing one matrix `P^T E_q G` and `P^T K_q G` per affine operator component.
  This is what the Stokes / Navier-Stokes work should be built on.

These carry into `2026-08-30-stokes-vector`, which replaces this cell.
