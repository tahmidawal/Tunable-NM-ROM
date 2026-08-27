# Co-designing the decoder with its quadrature: the N=64 mini-pilot

Answers "can training the autoencoder optimize the empirical-quadrature sample
points?" with a 9-arm controlled experiment at N=64 (branch
`exp/2026-08-26-codesign`, driver `sep_codesign.py`). **Numbers final for
N=64; nothing here has been tested at N≥256 — the project has measured
checkpoint-dependent sign flips before, so no finding below transfers
automatically.** Design: `understand/2026-08-26-codesign-design.md`; chronology:
`understand/2026-08-26-codesign-notes.md`.

## The architecture under test

The decoder is the project's separable EQ-decoder: `u(x; z) = bc(x) · ⟨g(x), h(z)⟩`. All
x-dependence lives in the feature bank `g` (random-Fourier lift → SiLU MLP; at this pilot's
scale R=64 features on the N=64 grid), all z-dependence in the head `h` (SiLU MLP + linear
skip, K=16). Because `g` never sees `z`, the decoder restricted to any fixed point set is a
cached table times `h(z)` — that is what makes an m-point quadrature the whole online cost
story. The residual is the exact-linear (exlin) weak form adopted 2026-08-26: linear terms
computed exactly through the precomputed `A = ΦᵀG` (M=64 sine test modes), so **only the
advection term `ΦᵀN(u)` is sampled** at the m nodes, and the co-design has exactly one term
to serve.

What each arm trains, solves, or freezes:

| object | role | in this pilot |
|---|---|---|
| bank `g`, `bc`, `A = ΦᵀG` | all spatial structure, exact linear terms | **frozen** (every cached identity and the span floor untouched) |
| head `h` (`params["h"]`, `h_lin`) | z → 64 coefficients | **trained** in arms i/ii/iii, frozen in arm n; always warm-started from the committed checkpoint |
| node positions x₁…x_m | where advection is sampled | **trained** (continuous, sigmoid-box reparam, stage-3 machinery) |
| node weights w | quadrature weights | **solved** — NNLS on the exact loss-form rows every 500 steps, never gradient-trained |

The loss is four normalized terms: reconstruction anchor (L_rec, on 256 training
snapshots), optional derivative reconstruction (L_sob), and the two mismatch terms —
sampled-vs-full advection value (L_samp) and its z-Jacobian (L_jac) — both against the
**current** decoder's own full-grid teacher, so the loss is a mismatch that cannot be
reduced by fooling the points, only by the decoder and points genuinely agreeing.
Certification never trusts the training loss: every arm is graded by the same external
instrument (held-out ladder rungs + full ROM rollouts against FOM truth).

## Findings

**F1 — The mismatch is trainable, massively.** Fine-tuning the h-track jointly
with continuous node positions against the full-grid advection teacher drives
the held-out sampled-vs-true gradient rung (c1) down 12× at m=64 (5.43 → 0.43,
cos 0.64 → 0.89) and 7× at m=256 (0.239 → 0.032, cos 0.9595 → 0.9983), on
states excluded from every gradient and NNLS row. The co-design mechanism
works exactly as intended at the mismatch level.

**F2 — And it still loses: decoder drift is net-harmful at every budget
tried (the reportable negative).** Every h-training arm degrades held-out
reconstruction ~+12% (2.733e-2 → 3.04–3.05e-2, tripwire fired), and that
drift costs more rollout accuracy than the mismatch gains buy: at m=64 the
best co-train arm reaches −13.5% vs baseline where nodes-only reaches −17.3%;
at m=256 every co-train arm is +11–12% WORSE than baseline. Raising the
reconstruction anchor 10× (REC_W 10 → 100) left the drift essentially
unchanged (3.054e-2 → 3.047e-2) — the drift is not a smooth trade the anchor
prices, consistent with h satisfying the 256 anchor states while degrading
between them (anchor-set sparsity; open, untested). The pre-committed single
adjustment round is spent; per protocol this closes the co-training question
at N=64 as a clean negative.

**F3 — The new positive: at the truly tight budget m=M, learning node
positions alone is worth −17.3% rollout error, free.** Baseline NNLS at
m=64=M leaves an 80% held-out residual mismatch (quadrature badly binding);
frozen-decoder node learning takes rollouts 7.113e-2 → 5.880e-2 with
reconstruction untouched and no tripwire. At m=256=4M (not binding) the same
arm ties baseline (+0.4%) — consistent with the 2026-08-26 stage-3 finding
that node learning pays only where quadrature binds. **The m=M regime was
never tested in stage 3 (all its arms used m=4M)** — this is the regime that
matters for solve cost, and it is open at N≥256.

**F4 — The Jacobian and Sobolev loss terms do not separate at rollout level
at N=64.** samp-only, +jac, and +jac+sob co-train arms land within 1% of each
other on rollouts at both budgets; jac improves the held-out c1 rung only
marginally beyond samp here. (At the co-train drift level this experiment
cannot distinguish them; F4 says nothing about their value in a drift-free
setting.)

**F5 — Caveats.** One checkpoint, one seed, N=64, 4 test trajectories (one
with a poor IC fit, err_t0 0.36, shared by all arms — it inflates every mean
equally). Wall budget: 2000 steps, LR 3e-5. Gates 0/C/D/H/L/R all passed in
every run; two of them caught real wiring bugs before any training (notes,
2026-08-26 entry).

## Interpretation — why co-training does not work as well as it should

Four readings, ordered from most to least certain; the numbers cited are all in T-C1.

**1. The error budget never favored it.** Rollout error decomposes roughly into decoder
representation error plus quadrature-induced solver error. At m=4M the quadrature term is
already small (base held (b)=3.06e-2 against held recon 2.73e-2), so even a perfect
quadrature cannot buy much — and the co-trained arms indeed improve the mismatch 7×
while rollouts get *worse*, because any decoder degradation lands one-for-one in the
larger term. At m=M the quadrature term IS large (b=0.805) — but nodes-only shows that
most of the removable part is removable *without touching the decoder*. Co-training only
pays where there is quadrature error that node placement alone cannot reach, and this
pilot found none worth its price.

**2. The drift is a generalization failure of the anchor, not a priced trade.** If the
optimizer were smoothly exchanging reconstruction for integrability, multiplying the
reconstruction weight by 10 (REC_W 10 → 100) should shrink the drift roughly
proportionally. It did not move it (3.054e-2 → 3.047e-2). The consistent reading: `h` has
enough capacity to keep the 256 anchored snapshots reconstructed while deforming
*between* them, where the mismatch terms push and the anchor cannot see. The anchor
constrains a measure-zero slice of the manifold; the mismatch terms act everywhere the
fit states live. That is an experimental-design gap (anchor density), not necessarily a
refutation of the idea — but as run, the anchor was structurally unable to protect the
decoder.

**3. There is a cheap direction the mismatch loss likes and physics does not.** The
sampled-vs-full advection gap is largest where `N(u) = u·∇u` varies on scales the m
points cannot resolve. Two ways to shrink it: sample better (what we wanted), or make
`u`'s advection field tamer between sample points (what a flexible `h` can also do). The
second is invisible to a sparse anchor and mildly destructive to reconstruction — which
is exactly the observed signature: large mismatch gains, small uniform recon drift,
rollouts that do not improve in proportion to the rungs. The frozen-denominator
normalization blocks the crudest version (globally shrinking the advection term) but not
this local smoothing.

**4. It is the project's recurring pattern, third instance.** A component sitting at its
data-optimum, re-optimized against a different objective, loses more on the original axis
than it gains on the new one: FREEZE_WDEC (training the POD lift destroyed a 1e-4
optimum), the stage-2 same-target refits (+13% test rollout at N=256 despite large
held-out fidelity gains), and now this. The common mechanism is that the new objective is
evaluated on a finite state set and the component has capacity to overfit that set. The
node positions escape the pattern precisely because they are too low-dimensional (2m
numbers) to overfit — which is arguably the deepest reason nodes-only wins: **it puts the
learnable capacity where it cannot collude.**

What would have to change for co-training to be worth revisiting, in order of cost:
anchor on ALL training codes rather than 256 (removes reading 2's gap; one local run);
a hard trust region on `h`'s drift (constraint, not penalty); or a regime where
quadrature error dominates end-to-end error even after node learning — plausibly 2D at
small m, not demonstrated anywhere yet.

## Recommendation

Close co-training (F2) unless the anchor-sparsity hypothesis motivates a
redesign (anchor on ALL training codes — cheap, untested). Carry F3 forward:
the next experiment worth cluster time is **frozen-decoder node learning at
m=M budgets at N=256/1024** — if learned m=M nodes match NNLS m=4M accuracy,
the sampled part of the online solve shrinks 4× at matched accuracy, which
multiplies the paired speedup directly. Needs sign-off before any submission.

## Generated tables

### T-C1. All arms, held-out rungs and test rollouts

N=64, K=16, R=64, M=64; 2000 steps, LR 3e-5; 4 fresh test trajectories; base = frozen decoder + advection-only NNLS nodes (identical across arms at equal m by construction).

| m | arm | trained | REC_W | held (b) | held (c1) | (c1) cos | (c3) cos | held recon | rollout err | vs base | trip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | base (NNLS) | — | — | 8.050e-01 | 5.428e+00 | 0.6407 | 0.7022 | 2.733e-02 | 7.113e-02 | — | — |
| 64 | i_m64 | h+nodes | 10 | 3.252e-01 | 4.280e-01 | 0.9002 | 0.8662 | 3.050e-02 | 6.383e-02 | -10.3% | Y |
| 64 | ii_m64 | h+nodes+jac | 10 | 3.149e-01 | 4.462e-01 | 0.8752 | 0.8883 | 3.054e-02 | 6.391e-02 | -10.1% | Y |
| 64 | iii_m64 | h+nodes+jac+sob | 10 | 3.192e-01 | 4.640e-01 | 0.8715 | 0.8878 | 3.054e-02 | 6.451e-02 | -9.3% | Y |
| 64 | n_m64 | nodes | 10 | 3.626e-01 | 1.533e+00 | 0.7321 | 0.8354 | 2.733e-02 | 5.880e-02 | -17.3% | n |
| 64 | r100_ii_m64 | h+nodes+jac | 100 | 2.968e-01 | 4.345e-01 | 0.8851 | 0.9053 | 3.047e-02 | 6.154e-02 | -13.5% | Y |
| 256 | base (NNLS) | — | — | 3.062e-02 | 2.393e-01 | 0.9595 | 0.9915 | 2.733e-02 | 5.185e-02 | — | — |
| 256 | i_m256 | h+nodes | 10 | 1.442e-02 | 3.746e-02 | 0.9969 | 0.9997 | 3.043e-02 | 5.800e-02 | +11.9% | Y |
| 256 | ii_m256 | h+nodes+jac | 10 | 1.221e-02 | 3.158e-02 | 0.9983 | 0.9995 | 3.038e-02 | 5.782e-02 | +11.5% | Y |
| 256 | n_m256 | nodes | 10 | 1.405e-02 | 9.296e-02 | 0.9844 | 0.9965 | 2.733e-02 | 5.207e-02 | +0.4% | n |
| 256 | r100_ii_m256 | h+nodes+jac | 100 | 1.250e-02 | 2.816e-02 | 0.9988 | 0.9994 | 3.046e-02 | 5.795e-02 | +11.8% | Y |

### T-C2. Provenance

| arm | backend | gpu | steps | complete |
|---|---|---|---|---|
| i_m256 | gpu | NVIDIA GB10 | 2000 | Y |
| i_m64 | gpu | NVIDIA GB10 | 2000 | Y |
| ii_m256 | gpu | NVIDIA GB10 | 2000 | Y |
| ii_m64 | gpu | NVIDIA GB10 | 2000 | Y |
| iii_m64 | gpu | NVIDIA GB10 | 2000 | Y |
| n_m256 | gpu | NVIDIA GB10 | 2000 | Y |
| n_m64 | gpu | NVIDIA GB10 | 2000 | Y |
| r100_ii_m256 | gpu | NVIDIA GB10 | 2000 | Y |
| r100_ii_m64 | gpu | NVIDIA GB10 | 2000 | Y |

Sources: runs/cd_*/out/sep_codesign_*.json on branch exp/2026-08-26-codesign; base rows deduplicated (bit-identical across arms at equal m).
