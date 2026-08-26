# Design: quadrature-aware co-training of the separable decoder

Written 2026-08-26, during the post-overnight ideation session. **Status: design only — nothing
here has run.** Origin: Tahmid's question "why can't training the autoencoder optimize the
sample points for the empirical quadrature?" This is the concrete answer. If it is approved,
execution follows the mechanics at the bottom (ask first, worktree from
`exp/2026-08-26-eq-learned`, ladder certification).

## The idea in one paragraph

Today the decoder is trained knowing nothing about the fact that its residual will be sampled
at m points; the quadrature is fit afterwards to whatever nonlinearity the decoder happened to
learn. Stage 3 (2026-08-26 report, F7) showed that with the decoder frozen, learning the node
positions buys ~3% and only where m is small. This design flips the dependency: fine-tune the
decoder **jointly with the node positions** so the manifold itself becomes easy to integrate.
The payoff target is not accuracy but **m** — the last mesh-flavored cost in the online solve.
If a co-trained decoder at m=64 matches the frozen decoder at m=1024, the ROM solve cost drops
by that factor, which attacks the weak paired speedup (1.87× at N=1024) directly and is the
prerequisite for the 2D move.

## Architecture

Unchanged decoder family and unchanged solver:

```
u(x; z) = bc(x) · ⟨ g(x), h(z) ⟩          g: R=512 analytic features, h: MLP  z ∈ R^K → R^512

r_w(z) = wt ⊙ [ A·(h(z) − h(zⁿ)) + Δt·( Φ_qᵀ N(u)|nodes  +  ν λ ⊙ A·h(z) ) ]
          └── exact linear terms, A = ΦᵀG precomputed ──┘   └ sampled: advection ONLY ┘
```

What changes is which objects are trainable in a new **phase-2 fine-tune**:

| object | role | phase-2 status |
|---|---|---|
| feature bank `g` + `bc` mask | spatial features | **FROZEN** (arm-default; trainable bank is a follow-up arm — span-floor risk) |
| `h` (MLP) | z → coefficients | **TRAINED** (warm-started from the committed checkpoint, never from scratch) |
| node positions x₁…x_m | quadrature | **TRAINED** (stage-3 machinery: sigmoid box [dx, 1−dx], min-separation penalty) |
| node weights w | quadrature | **SOLVED** — NNLS re-solve every REFIT_EVERY steps (variable projection); never gradient-trained |
| A = ΦᵀG, λ, wt | linear terms | exact/precomputed; rebuilt only if `g` ever becomes trainable |

Solve time is bit-for-bit the current pipeline: rebuilt banks + new nodes = a drop-in
checkpoint. No new solver code.

## Loss (per training state)

```
L = L_rec + λ_g·L_sob + λ_s·L_samp + λ_j·L_jac        (per-state normalization, stage-3 recipe)
```

- **L_rec** — relative L2 reconstruction against the snapshot. The anchor.
- **L_sob** — relative L2 of ∂x u against the snapshot derivative. Analytic bank derivatives
  (`sc.features` is pure jnp), no finite differences. Trains the decoder on what the residual
  consumes.
- **L_samp** — ‖sampled advection weak term − full-grid advection weak term‖² / ‖full‖².
  Both sides from the **current** decoder. Anti-collusion by construction: the loss is a
  mismatch, so it cannot be reduced by fooling the points — only by genuinely becoming easier
  to integrate.
- **L_jac** — the same mismatch for ∂z of the advection term (K JVPs per state; subsample to
  ~1/4 of the batch for cost). This is the term the LM solver actually steps on; it is the
  rung (c1) that predicted every rollout outcome on 2026-08-25/26.
- **No Hessian term.** c3 stays a diagnostic rung only.

**Training-state distribution:** snapshot LS codes plus small Gaussian latent perturbations
(σ calibrated to measured LM step sizes from existing rollout JSONs). Deliberately light on
solver-path iterates — the stage-2 negative transfer (+13% on N=256 dense_mid) is the measured
warning against training on the fit distribution. A held-out μ slice is never touched by any
loss.

## Training loop

Phase 1 = the existing pretraining, unchanged. Phase 2 (~20k steps):

```
every step     Adam update on (θ_h, node positions) w.r.t. L
every ~200     NNLS re-solve of node weights w on the current decoder (variable projection)
every ~1k      held-out mini-ladder: L2 rung, Sobolev rung, (b) mismatch, (c1) cos
```

**Tripwire (the FREEZE_WDEC guard):** if the held-out L2 rung degrades more than ~3% from the
warm-start value, λ_s/λ_j are too aggressive — reduce them; never trade reconstruction blindly.

**Gates before any rollout evaluation:** gate C at init (continuous node machinery ≡ grid ops,
≤1e-13), gates L/A unchanged from the exlin rule set, N=64 local smoke before any cluster
submission.

## The pilot that decides it

N=256, K=16, **dense_mid** checkpoint — chosen because it is the hostile one (stage-2 refits
*hurt* it), so a win there is meaningful. Budgets m ∈ {64, 256}, where quadrature binds.

| arm | decoder | nodes | loss |
|---|---|---|---|
| 0 | frozen | NNLS (incumbent) | — (already measured) |
| 0n | frozen | learned (stage 3) | — (already measured) |
| i | co-trained | co-trained | L_rec + λ_s L_samp |
| ii | co-trained | co-trained | + λ_j L_jac |
| iii | co-trained | co-trained | + λ_g L_sob |

**Decision number:** arm ii at **m=64** vs arm 0 at **m=1024** on test rollouts. Match or beat
→ m just dropped 16× and the idea carries into 2D. Miss → the ladder attributes which term
failed to move which rung (ii → c1, iii → the derivative rungs), separating "wrong idea" from
"wrong weighting".

Certification: the ladder, never NNLS rel-fit. Rollout validation per checkpoint (stage-2
lesson: held-out fidelity gains do not imply test gains).

## What this buys the NM-ROM, and how we use it

The context: a nonlinear-manifold ROM's accuracy story was never the hard part — the online
cost is. A linear ROM precomputes everything; the moment the decoder is nonlinear, the
nonlinear PDE term must be *evaluated* at every solver step, and hyper-reduction (the m
points) is the patch. That term is why NM-ROM papers show accuracy-per-latent-dimension wins
but weak wall-clock wins — ours included (1.87× paired at N=1024).

1. **It attacks the only cost that doesn't precompute.** After exlin, the online solve is
   cached linear algebra + m advection evaluations. m is the last mesh-flavored thing in the
   loop; shrinking it 16× multiplies the paired speedup directly.
2. **It repairs the actual failure mode of small-m hyper-reduction.** Small-m NM-ROMs fail by
   Jacobian corruption — the sampled gradient points Newton wrong and rollouts drift (measured:
   c1 cosine predicted every rollout outcome). L_jac trains that instability away, which is
   what forces m large in the first place.
3. **It restores the method's own design principle.** The original NM-ROM line chose sparse
   masked decoders *because of* hyper-reduction. The separable decoder gained tunability and
   cached banks but dropped that co-design; this puts it back in learned form. Paper sentence:
   "we train the manifold to be hyper-reducible."
4. **It makes the 2D move viable.** In 2D the FOM cost grows ~N× while the ROM solve stays
   flat — *if* m stays small. Co-design is how m stays small at scale.

**How the result gets used, concretely:**

- The deliverable is a **drop-in checkpoint + node file** — same rollout code, same solver.
  Any downstream experiment (speed protocol, 2D port) consumes it unchanged.
- If the pilot wins: re-run the round-4/5 paired speed protocol with the co-trained
  checkpoint at its smallest certified m — that is the new headline speedup — and make
  phase-2 co-training a **standard final stage of every future decoder training run**,
  including the 2D pipeline (train → co-tune → certify on the ladder → ship checkpoint).
- If the pilot loses: the ladder attribution tells us whether sampleability is un-trainable
  (idea wrong) or just mis-weighted (λ wrong), and the negative goes in the report either way.

## Open questions (to settle before running)

1. Trainable bank `g` in phase 2 — fifth arm or follow-up? (Span-floor risk vs extra freedom.)
2. Perturbation σ and count for the training-state distribution.
3. λ grid — start {0.1, 1, 10} per active term, or calibrate from initial loss magnitudes?
4. REFIT_EVERY sensitivity (200 is a guess).
5. Does the node-position gradient still matter once the decoder moves, or does periodic NNLS
   plus decoder adaptation dominate? (Cheap ablation: arm ii with nodes frozen at NNLS picks.)

## Mechanics if approved

Ask first. Worktree from `exp/2026-08-26-eq-learned` (has the exlin residual + stage-3
machinery), date-prefixed name, own cluster namespace, one job per directory, N=64 smoke
locally, `JAX_DEFAULT_MATMUL_PRECISION=highest`, gates logged in every JSON.
