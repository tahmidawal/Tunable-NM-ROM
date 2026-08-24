# A separable decoder with co-trained empirical quadrature for the weak NM-ROM

**Status: design proposal, pre-experimental, revised after an independent adversarial
audit (OpenAI Codex, 2026-08-22; two blockers and eight major findings incorporated —
§11).** No number in this document is a new result; the few measurements cited are
transcribed from the job `2689547` log (unaudited — the pull and validation of that job
is still owed to the lab log) and are marked as such.

**The question it answers:** can we build a decoder custom-designed for the weak NM-ROM
latent solve — one whose in-loop cost is provably independent of the mesh and small in
the latent dimension, and whose empirical quadrature is learned jointly with it — and
what do we give up for that?

**The governing invariant (added after audit):** every cached operator below is derived
from, and must be provably identical to, the *existing full-grid discrete weak
Gauss–Newton map* — the same discrete operators, the same test-mode projection, the same
solver. The architecture changes how that map is *evaluated*, never what it *is*. Any
arm that changes the discretization (e.g. continuum derivatives instead of the FOM
stencil) is a different method and must be labeled as such.

---

## 1. The setting: the weak NM-ROM we already have

The PDE solution is parameterised by a latent vector `z ∈ R^k` through a decoder
`u(x; z)`. The solve minimises the PDE residual projected onto `M` smooth sine test
modes (a weak form), evaluated by empirical quadrature at `m` points:

```
R_p(z) = Σ_{j=1}^m w_j ψ_p(x_j) r(u(·;z))(x_j),     p = 1..M
z*     = argmin ½‖R(z)‖²        by trust-region Gauss–Newton (GN).
```

For Burgers the same structure holds per implicit time step with
`r(u^{n+1}) = (u^{n+1} − u^n)/Δt + u^{n+1}·∇_h u^{n+1} − ν Δ_h u^{n+1}`, where `∇_h` is
the FOM's **sign-upwind five-point stencil** and `Δ_h` the discrete Laplacian — *not*
continuum derivatives. Established facts this design must preserve:

- **Weak form is mandatory** (pointwise residual minimisation amplifies grid-scale
  decoder error by ~N²).
- **Mesh-independence of the latent solve is already achieved** (m points, M equations,
  k unknowns).
- **`M > k` comfortably, `m ≈ 4M`** is the working recipe.
- **The Poisson weak form needs no decoder derivatives at all**: with sine test modes,
  `⟨ψ_p, −Δu⟩ = λ_p ⟨ψ_p, u⟩` for fields satisfying the BCs, so the implemented weak
  residual uses decoder *values* only (`pro_common.py`). Any design that reintroduces a
  decoder Laplacian is regressing toward the derivative-sensitive strong form.

## 2. Where the cost actually goes with the current FiLM decoder

The current decoder is a monolithic FiLM/INR network: `z` modulates every layer, so
every GN iteration re-evaluates the full network and its Jacobian at all `m` points —
`O(m·D·W²)` forward plus ~`O(m·D·W²·k)` for k JVPs, paid hundreds of times per solve.
Transcribed from the `2689547` log (unaudited): Burgers at N=64 spends ~340 Jacobian
evaluations at k=4 rising to ~1090 at k=64 with ~100 % of cells censored; end-to-end
cost grows ~170× across a 16× range of k.

The three coupled defects:

- **(D1)** per-iteration deep-network Jacobian cost at `m` points;
- **(D2)** iteration counts in the hundreds — a latent-landscape/solver problem;
- **(D3)** a quadrature fitted *after* training, on on-manifold snapshots only, never
  asked to reproduce the *gradient* of the objective.

**Audit caveat on D3 (finding 6):** the sweep's k=64 row cannot be used as evidence that
NNLS-EQ degrades with capacity — that checkpoint's *decoder training itself* collapsed
(training error ~0.45 vs ~5e-3 at k=48) before its EQ fit degraded. D3 must be
diagnosed on healthy, accuracy-matched checkpoints only.

## 3. The proposed architecture: a separable (two-track) decoder

### 3.1 Form

```
u(x; z) = Σ_{i=1}^r g_i(x) · h_i(z)  =  g(x)ᵀ h(z)
```

- `g : Ω → R^r` — a **coordinate feature bank**: an INR in `x` only, meshfree, with no
  dependence on `z`.
- `h : R^k → R^r` — a small **nonlinear head** in `z` only (2–3 layers, width O(r)).
- `r` is the feature rank, `k ≤ r ≪ n` (we write `n = N²` for the full grid dimension
  throughout; audit finding 5).

**POD special case (with the audit's qualification):** homogeneous uncentered POD is
exactly `r = k`, `h(z) = z`, `g_i` = POD modes; centered/lifted POD (`u = ū + Φz`)
additionally needs one constant-coefficient feature carrying `ū`. Either way the
architecture contains the classical ROM as an initialisation point (§7).

### 3.2 The in-loop algebra, derived from the discrete operators (blocker 1 fixed)

All `x`-dependence factors through `g`, so at any fixed point set the spatial
information is a set of **z-independent cached matrices**. The caches are built by
evaluating `g` at *exactly the points the discrete operators touch*, so the online map
is the incumbent one.

**Poisson (values-only weak form).** With `Φ` the test-mode values at the EQ points,
`W = diag(w)`, `Λ = diag(λ_p)`, and `G = [g_i(x_j)] ∈ R^{m×r}`:

```
B  = Λ Φᵀ W G          ∈ R^{M×r}      cached once
R(z)   = B h(z) − b_μ                  O(M r)
J_R(z) = B J_h(z)                      O(M r k)  after forming J_h
```

The Poisson weak residual is **linear in h(z)**: there is no `m` inside the iteration at
all once `B` is cached. Per iteration: form `J_h` (`O(k r²)` forward-mode through the
tiny head), `B J_h` (`O(Mrk)`), normal matrix (`O(Mk²)`), solve (`O(k³)`).

**Burgers (FOM-exact upwind stencil).** The accepted method differentiates on the grid
stencil, and the upwind operator switches on `sign(u)`. Cache the feature banks at the
stencil points: `C` (center values), `D⁺_x, D⁻_x, D⁺_y, D⁻_y` (one-sided difference
operators applied to the feature bank — built from `g` evaluated at `x_j ± Δx eᵢ`, or
restricted to grid nodes), and `L` (discrete-Laplacian features). With `q = h(z)`,
`H = J_h(z)`, `v = Cq`, and `D_σ` the upwind selection assembled per-point from the
cached one-sided banks by `sign(v)`:

```
d   = D_σ q
R   = A { v − v_n + Δt [ v ⊙ d − ν L q ] }
J_R = A [ C + Δt { diag(d) C + diag(v) D_σ − ν L } ] H
```

Note the term `diag(v) D_σ H` — the derivative of the *difference*, not only of the
center value — and that `D_σ` is piecewise in `v` (nonsmooth at `v = 0`), exactly as in
the FOM. This is the full discrete Fréchet derivative; a "pointwise diagonal ∂r/∂u"
shortcut is wrong here (audit blocker 1). All products remain dense `m×r`-shaped BLAS;
no network is evaluated.

**First-cell constraint:** quadrature nodes are restricted to grid centers (or a proven
stencil-consistent off-grid evaluation of `g(x_j ± Δx eᵢ)`), so the discrete-operator
identity below (§9, gate 0) is checkable exactly. Continuous off-grid nodes are a
*separate continuum-ROM arm*, not this method.

### 3.3 Cost accounting (corrected per audit finding 4)

With `M = 4k`, `m = 16k`, `r = ck`, head width O(r):

| term                                | cost                 | at c=16   |
| ----------------------------------- | -------------------- | --------- |
| form`J_h` (forward-mode)          | O(k r²) = O(c²k³) | ~256·k³ |
| feature-Jacobian products (Burgers) | O(m r k)             | ~256·k³ |
| projection through`A`             | O(M m k)             | 64·k³   |
| normal matrix`JᵀJ`               | O(M k²)             | 4·k³    |
| dense solve                         | O(k³)               | k³       |

So the route is **O(k³) with fixed c, but "small constants" is a hypothesis to be
measured, not a theorem** — at c = 16 the head Jacobian alone is a ~256·k³-scale term
(mitigation: shrink head width below r, or share `J_h` across trial steps). The claim
that *is* structural: no spatial network evaluation appears in the compiled iteration,
and nothing scales with `n`.

Costs the per-iteration figure excludes and the experiment must count separately:
cold-start fitting, source projection, trial residuals and repeated factorizations on
rejected trust steps, convergence checks, guard/output decodes, compilation, and cache
construction. "Per iteration" is reported split into: linearization / attempted step /
accepted step / residual-only trial.

**A stronger baseline this design must face (audit):** for polynomial nonlinearities
like Burgers' quadratic term, precomputed reduced tensors can remove `m` from the online
cost entirely (hyper-reduction-free cached Newton operators; see §10). The sign-switching
upwind operator complicates that route but does not excuse omitting it as a control.

**Final readout and memory.** The unavoidable output is Θ(n) = Θ(N²); as a matmul
against a full-grid `G` it is Θ(n r). At N=2048, r=128, f64 that cache is ~4 GiB — fine;
at r = 16k with k=64 (r=1024) it is ~32 GiB — **not** automatically affordable. Either
evaluate `g` on the fly at readout (once per solve) or gate the cache memory explicitly
(§9, gate 7).

## 4. What we give up: the span ceiling, stated precisely (audit findings 8–9)

> The image `{ g(·)ᵀ h(z) }` is an at-most-k-dimensional set (a manifold where `J_h`
> has full rank) **embedded inside the fixed r-dimensional linear span of {g_i}**.

Consequences, with the audit's corrections to the wording:

1. The accuracy ceiling is an **empirical linear-span representation bound**: the
   held-out projection error onto the best available r-dimensional space, in a stated
   norm, with stated centering. This is *not* the Kolmogorov r-width (an infimum over
   all subspaces under a worst-case norm), and an empirical POD-r number is an
   optimistic reference, not a certified floor. Nonlinear `h` buys parameter efficiency
   (k ≪ r solve unknowns) and a curved coefficient chart — it cannot beat the best
   approximation in `span(g)`.
2. A POD-r pass is **necessary, not sufficient**: `h(R^k)` may occupy a badly
   conditioned sliver of the r-space. Hence the additional *coefficient-manifold oracle
   gate* (§9, gate 2): optimize `z` directly against held-out fields within the trained
   `h`, per time step and per parameter, worst case included.
3. For transport-dominated Burgers the span ceiling is the classical objection. It is
   softened — not removed — by `r` being an order of magnitude cheaper to raise here
   than in POD (O(mr) per iteration, not r solve unknowns). Whether the ceiling binds
   at the ~1e-2–1e-3 targets is measurable from existing snapshot data for zero GPU
   time, *before* any training. Trajectory-average passes can hide shock failure, so
   the gate includes worst-time and worst-viscosity slices.

## 5. The solver angle (D2), with claims trimmed to what is true

Audit finding 7 is accepted: the FiLM Jacobian is already exact via autodiff — the
separable Jacobian is **cheaper, not more exact** — and `JᵀJ` is PSD, so there is no
"negative curvature" to handle in GN, only rank deficiency and small eigenvalues. The
corrected claims:

1. **Cheaper relinearization permits stronger globalization**: full trust-region model
   ratios, LM with per-trial refactorization, k×k eigendecompositions for
   rank/conditioning control — affordable because they cost O(k³), not a network sweep.
   Whether iteration counts actually improve is **gated, not predicted**.
2. **Latent-chart conditioning is monitored, not assumed**: singular values of `J_h`
   and `J_R`, latent whitening, trust-region rejection counts (§9, gate 6). A nonlinear
   `h` can produce a landscape as bad as FiLM's; nothing in the factorization prevents
   that.

**Solver-in-the-loop training is deferred to a later arm** (audit finding 11). It is
not "a differentiable O(k³) layer": backprop through GN needs second derivatives of `h`,
inverse-system sensitivities amplified by squared condition numbers through the normal
equations, and the production trust-region policy is full of nondifferentiable
accept/reject logic. The later arm, if licensed: fixed-T, fixed-damping LM surrogate
with QR (not normal equations), finite-difference gradient checks, checkpointed
iterations; implicit differentiation only as a separate converged-solution arm; never
differentiate through the production trust policy.

## 6. Co-trained empirical quadrature (D3), restructured after audit blockers 2–3

### 6.1 What "gradient fidelity" must mean (blocker 2)

Matching the residual Jacobian `J` in Frobenius norm does **not** preserve the GN step:
the solver consumes `g_GN = JᵀR`, `H_GN = JᵀJ`, and the damped step, and correlated
R/J errors are amplified by the normal equations. The trained-and-gated quantities are
therefore the **GN-map fidelity ladder**:

```
(b)  ‖R_s − R_f‖² / (‖R_f‖² + ε)                        weak-residual fidelity
(c1) ‖J_sᵀR_s − J_fᵀR_f‖² / (‖J_fᵀR_f‖² + ε)            objective-gradient fidelity
(c2) E_v ‖(J_sᵀJ_s − J_fᵀJ_f) v‖²    (fixed probes v)    normal-operator fidelity
(c3) damped/trust-step discrepancy at matched radius      step fidelity
```

with absolute (not only relative) scales near stationary points, where the denominators
vanish. Raw Jacobian discrepancy stays as a diagnostic. The pointwise sample-
reconstruction error (Hari's metric (a)) also stays diagnostic-only. The feature
orthonormality regulariser is `‖G_fᵀ W_f G_f − I_r‖²_F` on the full grid (fixing the
earlier ill-typed expression).

The expectation runs over **(μ, z_n, z_{n+1}, Δt) and off-manifold iterates** — solver
trajectories and perturbed encodings — not only converged snapshot codes: Burgers
residuals depend on viscosity and the previous state, and the GN solve visits states
that are not snapshots.

### 6.2 Degeneracy control (blocker 3) and staging

Joint training of `(g, h, nodes, weights)` from scratch has unexcluded degenerate
minima: near reconstructed snapshots `R_f ≈ 0`, so weights → 0 shrink the integral loss
without learning a quadrature; the decoder can flatten its tangent near training codes
instead of the quadrature improving; `g → gQ, h → Q⁻¹h` and latent reparameterizations
are exact symmetries; duplicate nodes and permutations are redundant; softplus weights
are strictly positive and cannot reproduce NNLS sparsity. Therefore the experiment is
**staged, with each stage frozen before the next**:

1. **Train and freeze the decoder** (recipe in §7).
2. **Quadrature against a frozen full-grid teacher:** first arm is *NNLS on the
   GN-fidelity targets* — at fixed decoder and nodes, `R_s` and `J_s` are linear in the
   weights, so NNLS on a stacked (b)/(c1)/(c2) design matrix is already the optimal
   positive-weight baseline (audit finding 10). Beating the *current* NNLS (fitted to
   different targets) is not a victory; beating *same-target* NNLS is.
3. **Continuous-node optimization with weights solved by NNLS/variable projection**,
   frozen decoder, static nodes per solve. Constraints: `Σw = |Ω|`, exact integration
   of constants and the mandatory low test modes, minimum node separation, box
   reparameterization that can actually reach the boundary.
4. **Only then** a tightly regularized joint fine-tuning arm, with a held-out full-grid
   terminal loss as the anti-collusion control (the teacher never moves during student
   updates).

**Node differentiability caveats (finding 10):** for Burgers the upwind switch is
nondifferentiable at `v = 0`; for any strong-form residual, node gradients need spatial
derivatives one order above the residual (third derivatives of the INR for a Laplacian).
The values-only Poisson weak form avoids the worst of this; Burgers node motion is a
research risk, which is one more reason nodes are static in the first cell.

**Caching interaction (finding 12):** static nodes ⇒ all banks cached once and reused
across all 50 implicit steps. Per-step adaptive nodes ⇒ banks recomputed once per step
(acceptable, O(m·cost(g)) per step). Inside-GN adaptation puts the INR back in the loop
— out of scope. Fixed node cardinality and shapes avoid JAX recompilation; changing
cardinality does not.

## 7. Training recipe (decoder stage)

1. **Init at POD:** `g` pretrained to the first r POD modes (plus a mean feature for
   the centered case), `h(z) ≈ [z, 0, …]`. The model starts as POD-k inside span-r — a
   known quantity. (Inherited `FREEZE_WDEC` lesson: don't let Adam destroy a good
   linear optimum.)
2. **Stage A:** freeze `g`, train `h`.
3. **Stage B:** unfreeze `g`, small LR, reconstruction + orthonormality.
4. Auto-decoder latent codes (no encoder exists at solve time) with **latent whitening**
   fixed after training, so the chart's scale is pinned (degeneracy control).
5. f64 data, `JAX_DEFAULT_MATMUL_PRECISION=highest`, orthonormality checks at ~1e-8 —
   the repo's standing numerics rules.

## 8. What the first preregistered cell is — and is not

Per the audit verdict, the first cell is **not** the all-at-once decoder+nodes+unrolled-
solver experiment. It is:

- separable decoder, staged training (§7), **static grid-restricted nodes**,
- cached operators proven identical to the incumbent discrete weak GN map (gate 0),
- quadrature stage limited to same-target NNLS (arm 1) vs current NNLS,
- both PDEs at N=64, k ∈ {8, 16}, against FiLM, POD-r oracle, and the classical
  controls (spectral Poisson; optimized cubic-history / residual+Helmholtz Burgers).

Node learning, joint fine-tuning, adaptive sampling, and solver-in-the-loop are later
arms, each licensed only by the previous one's gates.

## 9. Go

/no-go gates (preregistered before any training job)

0. **Discrete-operator identity (new, from audit):** on random `z`, full-grid cached
   `R` and `J_R` match the incumbent implementation to ~1e-13 (f64), including the
   Burgers upwind switch. No spatial-network call appears in the compiled online
   iteration (cache audit of the jaxpr).
1. **Span-ceiling gate (free, before training):** held-out POD-r projection error at
   r ∈ {4k, 8k, 16k} below target (~1e-2 Burgers, ~1e-3 Poisson, N=64), reported as
   mean *and* worst trajectory, *and* worst-time / worst-viscosity slices for Burgers.
2. **Coefficient-manifold oracle gate (new):** direct z-optimization against held-out
   fields within the trained `h` must approach the span bound; POD-r alone is
   necessary, not sufficient.
3. **Accuracy gate:** matches FiLM within 1.2× **and** meets absolute mean/worst
   targets with uncensored solves — FiLM itself is censored on Burgers, so "as good as
   FiLM" alone is a false-positive gate.
4. **Speed gate:** same-GPU, same-invocation cost+accuracy, persisted repetitions,
   separate counters for Jacobian evals / rejected trials / trial residuals / decodes /
   compilation / cache build; ≥5× per-accepted-iteration vs FiLM at matched (k, m, M)
   before any crossover claim.
5. **Quadrature gate:** same-target NNLS ladder — (b), (c1), (c2), (c3) on held-out
   parameters, plus final 50-step rollout fidelity for Burgers.
6. **Conditioning gate (new):** σ(J_h), σ(J_R), trust-region rejection counts, no
   latent-chart collapse across ≥3 training seeds.
7. **Memory gate (new):** complete compiled route (all banks + output) within the
   preregistered device budget at the proposed r.

No gate passed ⇒ no next stage. All numbers from run JSONs via a builder script.

## 10. Prior art (expanded per audit; citations to be verified before any paper claim)

The audit found the field considerably closer than the first draft acknowledged. The
factorization `u = trunk(x)ᵀ branch(·)` is DeepONet's; beyond that:

- **Shen et al., high-order differentiable autoencoder (arXiv:2102.11026):** nonlinear
  decoder *with a learned element sampler and nonnegative state-dependent cubature
  weights* — directly defeats any broad "neural decoder + learned quadrature is new"
  claim.
- **CROM (Chen et al., arXiv:2206.02607):** continuous neural fields, sampled PDE
  evolution, GN inversion, robust sample selection — closer than "INR + pointwise
  collocation" suggests.
- **Romor, Stabile, Rozza (arXiv:2203.00360):** nonlinear-manifold ROM with reduced
  over-collocation, 2-D conservation laws.
- **Cocola et al. (arXiv:2303.09630):** hyper-reduced autoencoders trained from
  subsampled fields, incl. 2-D Burgers.
- **Deep-HyROMnet (arXiv:2202.02658):** learns reduced residuals/Jacobians along
  Newton iterations.
- **Jain & Tiso (arXiv:1710.05160):** ECSW hyper-reduction on nonlinear manifolds.
- **Gradient-preserving DEIM (arXiv:2206.01792):** sample selection targeting reduced
  gradient structure — the same fidelity target as §6.1.
- **Hyper-reduction-free cached polynomial Newton operators (reported as
  arXiv:2603.03420, 2026 — verify):** precomputed reduced tensors for quadratic
  nonlinearities; a mandatory Burgers *baseline*, not background.

**The defensible novelty, narrowed accordingly:** a query-only separable neural trial
manifold that preserves an incumbent *discrete weak* GN geometry exactly (upwind stencil
included), with positive stencil-aware quadrature fitted/learned against **GN-map
fidelity** (objective gradient, normal operator, damped step) and validated on
off-manifold iterates and full rollouts. Each element exists somewhere; this
combination — and the fidelity target — appears open, pending citation verification.

## 11. Audit record

Independent adversarial review by OpenAI Codex (read-only, 2026-08-22; full report in
session scratchpad `codex-audit.md`). Verdict: *"worth one bounded, preregistered cell,
but not preregistration-ready as written"* — with the incorporated changes: Burgers
upwind cache algebra (blocker 1), GN-map fidelity losses replacing Jacobian-Frobenius
(blocker 2), staged quadrature training with degeneracy constraints (blocker 3),
corrected cost constants and memory accounting (4–5), k=64 confound flagged (6),
exactness/curvature claims trimmed (7), span-bound wording and oracle gate (8–9),
same-target NNLS baseline and node-differentiability caveats (10), solver-in-the-loop
deferred with a corrected method (11), caching-vs-adaptivity clarified (12), prior art
expanded and novelty narrowed. Claims the audit confirmed: O(k³) asymptotics at fixed c,
mesh-independence and network-free iterations with static nodes, the span-confinement
analysis, the 4 GiB readout-cache figure, and the necessity of off-manifold training.

## 12. Summary judgement

The architecture buys, by construction: network-free, mesh-independent GN iterations in
O(k³) dense algebra; cached discrete operators identical to the incumbent weak form —
for Poisson the weak residual becomes *linear in h(z)*; cheap relinearization that makes
strong globalization affordable; a POD initialisation with staged unfreezing; and a
quadrature that can finally be fitted against the quantity the solver consumes.

It costs, by construction: confinement to an r-dimensional linear span — the transport
objection — softened only by r being cheap to raise; O(c²k³) head constants that must be
measured, not assumed small; and a novelty claim that survives only in its narrowed,
discrete-weak-GN-fidelity form. Gates 0–2 price all of the structural risks before a
single GPU-hour of training is spent.
