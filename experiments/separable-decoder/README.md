# The separable-decoder NM-ROM — exactly how it works

This document explains the system implemented in this directory: what every stage
computes, with the math as it is actually coded, the array shapes, and where each piece
lives. The design rationale and the adversarial audit that shaped it are in
`reports/2026-08-22-separable-eq-decoder-design.md` on `main`; the run configuration and
first results are in `EXPERIMENT.md` and `runs/sepdec_r1/`.

---

## 1. The problem being solved

We want to solve a PDE not by iterating on the full grid (N² unknowns) but by finding a
small latent vector `z ∈ R^k` on a learned solution manifold and decoding it. For that we
need three things:

1. a **decoder** `u(x; z)` mapping a latent vector to a solution field,
2. an **objective** whose minimiser over `z` is the PDE solution's latent code — here the
   established *weak form*: the PDE residual projected onto `M` smooth sine test modes,
   evaluated by empirical quadrature at `m ≪ N²` points,
3. a **solver** — damped/trust-region Gauss–Newton (GN) over `z`.

The project's incumbent decoder is a FiLM coordinate network: `z` modulates every layer,
so each GN iteration must re-evaluate a deep network *and* its Jacobian at all `m`
quadrature points. That per-iteration network cost, paid hundreds of times per solve, is
what this architecture removes.

## 2. The decoder: a separable two-track network

```
u(x; z) = bc(x) · ( g(x)ᵀ h(z) )          g : R² → R^r     h : R^k → R^r
```

implemented in `sep_common.py`:

- **`g` — the feature bank** (`features()`): all x-dependence. The coordinate
  `x = (x₁,x₂)` is lifted to fixed random Fourier features
  `[sin(2πBx), cos(2πBx)]` (`B ∈ R^{2×64}`, entries ~ N(0, 4²), never trained beyond the
  MLP that follows), then through a SiLU MLP `128 → 128 → r`. The output is multiplied by
  the hard Dirichlet factor `bc(x) = 16 x₁(1−x₁) x₂(1−x₂)` — so every decoded field is
  *exactly* zero on the walls, off-grid included — and by a fixed data-scale constant
  `out_scale = RMS(training data)` so the network starts at the data's amplitude
  (without this the model initialises ~100× too large and training stalls).
- **`h` — the nonlinear head** (`head()`): all z-dependence. A SiLU MLP
  `k → 128 → 128 → r` **plus a linear skip** `z ↦ z·W`. The skip eases early
  optimisation; the MLP makes the map genuinely nonlinear in `z`. There is **no POD
  basis, no POD initialisation, and no linear corrector anywhere** — `g` and `h` are
  random-initialised and trained jointly.

**Why this shape matters.** The decoder restricted to *any fixed point set*
`{x_1..x_m}` is

```
u(x_j; z) = G[j,:] · h(z),        G = [bc(x_j)·g(x_j)] ∈ R^{m×r}
```

`G` does not depend on `z`, so it can be computed **once** and cached. After that, every
solver iteration touches only the tiny head `h` and dense matrices — no spatial network
at all. That is the "sampling built in" property: decoding at m samples is a cached
`(m×r)` matrix times `h(z)`.

**What it costs.** The manifold `{g(·)ᵀh(z)}` is a k-dimensional nonlinear set embedded
in the r-dimensional linear span of the learned features `g_1..g_r`. The nonlinearity of
`h` buys parameter efficiency (the solve has k unknowns, not r), not escape from that
span — the accuracy ceiling is the best approximation in span{g_i}. We take `r = 4k` and
rely on `r` being cheap to raise (it enters the iteration as a matrix width, not as
solve unknowns).

## 3. Training: joint auto-decoding (`sep_common.train_autodecoder`)

There is no encoder. Each training snapshot `u_s` (interior grid values, f64) gets its
own latent code `z_s`, and Adam optimises **(g-params, h-params, all codes Z) jointly**:

```
L = mean_s,x ( [H Gᵀ]_{s,x} − U_{s,x} )² / mean(U²)   +   λ_orth ‖GᵀG/(n·s²) − I_r‖²
```

where `G = features(coords) ∈ R^{n_pts×r}`, `H = head(Z) ∈ R^{S×r}`, and the second term
is a *soft* conditioning regulariser on the feature Gram (λ = 1e-4; it cannot linearise
`h`). One optimisation step is: two small-network forwards plus one `(S×r)·(r×n_pts)`
matmul — which is why the whole 40k-step training takes ~20 s (Poisson, 512 snapshots)
to ~47 s (Burgers, 8192 states) on an H200, versus hours for the FiLM decoder. Schedule:
warmup-cosine, peak 1e-3, f64 throughout, `out_scale` masked out of the gradients.

- **Poisson data**: 512 training fields regenerated from the canonical seed
  (`ms_parametric.build_snapshots`), Gaussian-source family.
- **Burgers data**: all (trajectory, time) states of the seed-regenerated training
  trajectories, subsampled to ≤8192 states; each state is a snapshot with its own code.

The per-state codes `Z_tr` are kept: they provide the solver's initial guess, the
EQ-fitting snapshots, and the trust-region radius.

## 4. The weak objective and empirical quadrature

Both PDEs use the project's incumbent weak machinery — this experiment changes *no*
formula in it.

**Poisson** (`sep_poisson.py`). With sine test modes (the eigenmodes of the discrete
Laplacian), the weak residual needs **only decoder values**, because
`⟨ψ_p, −Δu⟩ = λ_p⟨ψ_p, u⟩` for fields obeying the BCs:

```
R_p(z) = λ_p^{1-α} Σ_j w_j ψ_p(x_j) u(x_j;z) − λ_p^{-α}(Φᵀf)_p ,   α=1, p=1..M'
```

(`ctol_tol.lm_tau_poisson`'s `r_of`: `Wl * (PhiT @ (wq * dec(z, pts))) − f_m`.)

**Burgers** (`sep_burgers.py`, via `blat_common.make_weak_ops`). Backward-Euler step,
weak form with the **FOM's sign-upwind stencil** for advection and the mode eigenvalues
for diffusion:

```
R(z) = wt ⊙ [ Φ_qᵀ(u − u_prev) + Δt( Φ_qᵀ(u ⊙ (u_x^upw + u_y^upw)) + ν λ ⊙ Φ_qᵀu ) ]
u_x^upw = (c−x₋)/dx if c>0 else (x₊−c)/dx        (identical branch to the FOM)
```

where `u, x₊, x₋, y₊, y₋` are decoder values at the 5-point stencils of the quadrature
nodes. The upwind switch acts on *decoded values*, exactly as in the full-order model —
the ROM discretises the FOM, not the continuum PDE.

**Empirical quadrature.** The `m` nodes and weights `w_j ≥ 0` come from the incumbent
NNLS fit (`ctol_eq.eq_fit_poisson` / `blat_common.fit_eq_weights`): decode a set of
training latents (plus small perturbations), and solve a nonnegative least-squares
problem asking the m-point rule to reproduce the *full-grid* mode projections of those
fields (and of the advection field, for Burgers). Recipe: `M = 4k` test modes,
`m = 4M = 16k` grid nodes.

## 5. The solver — unchanged, by construction

- **Poisson**: `ctol_tol.lm_tau_poisson` — damped Levenberg–Marquardt on `½‖R(z)‖²`,
  Jacobian by `jacfwd`, trust radius = 1.0 × training-latent radius, stop at
  `‖R‖ ≤ τ‖R(z₀)‖`, init `z₀ = mean(Z_tr)`.
- **Burgers**: `blat_common._finish_ops → lm_step_jit / rollout_jit` — the same damped
  LM per implicit time step inside a `lax.scan` over the 50 steps, trust radius
  0.01 × training radius (the independently accepted Burgers recipe), warm-started from
  the previous step's latent. The initial condition's code is fitted by LM on the data
  misfit to the known `u₀` (`fit_ic`), best of several inits (mean code + training t=0
  codes).

## 6. The two arms, and why the caching is legitimate

Every cell runs the same solve twice:

| arm | what evaluates `u` at the quadrature/stencil points |
|---|---|
| `meshfree` | the network itself, in-loop: `dec(z, pts)` — the incumbent path |
| `cached` | precomputed banks: Poisson `G_q ∈ R^{m×r}`, `u = G_q h(z)`; Burgers `G_st ∈ R^{m×5×r}` at the stencils, `us = einsum('msr,r→ms', G_st, h(z))`; full-field readout `G_all ∈ R^{N²×r}` |

Both arms are passed through the *same* solver factory, so the LM code, acceptance rule,
tolerance rule, and rollout are literally the same lines.

**GATE 0** (asserted before any solve is reported): at random latents, the two arms'
weak residual, Jacobian, and previous-state maps must agree to ≤ 1e-12 relative. In the
committed run they agree to 0.0 (Poisson) and ≤ 2.8e-15 (Burgers — including the
sign-upwind branch, which is exact because the switch is applied to identical decoded
values). This is the audit's governing invariant: *the discretization is never changed,
only how it is evaluated*. Any future "fast path" that fails this gate is a different
method, not an optimisation.

**Per-iteration cost of the cached arm** (no network in the compiled loop):

```
h(z), J_h        O(k·128 + 128·r)  +  O(k · that)        tiny MLP forward + jacfwd
u at points      O(m r)            G_q h(z)  /  einsum(G_st, h)
weak residual    O(m M)            mode projection
GN step          O(M k² + k³)      normal equations, k×k solve
```

Everything is independent of the mesh size N; N appears only in the one-time bank
construction and the final `G_all h(z*)` readout.

## 7. What one cell reports

Per (PDE, K, R): training curve and per-snapshot reconstruction; held-out
representation oracle (direct LM fit of `z` to test truth — the manifold ceiling);
NNLS-EQ fit diagnostics; gate-0 deviation; then for each arm and each τ: solve/rollout
wall time (median of 7 reps after warmups, cost and error **from the same invocation**),
error vs the seed-regenerated FOM truth, Jacobian counts, censoring/blowups; and a
same-job FOM baseline (Poisson: CG tolerance ladder; Burgers: the truth-generating
Newton rollout). Everything lands in one JSON under `runs/sepdec_r1/out/`.

## 8. Result snapshot (job 2802238, H200, N=64, seed 0 — from the committed JSONs)

- **Gate 0**: 0.0 / 0.0 / 1.50e-15 / 2.79e-15 across the four cells.
- **Burgers K=16**: 4/4 fresh-seed trajectories complete 50/50 implicit steps, zero
  blowups; trajectory rel-L2 1.38e-2 / 2.10e-2 / 2.58e-2 (+ one 1.47e-1 outlier from a
  poor IC fit); cached rollout median **54.1 ms** vs the same-job FOM Newton rollout
  **176.1 ms** (~3.3×), with ~315 total Jacobians vs the FiLM k-sweep's ~507.
- **Poisson K=16**: weak-EQ solve 3.75e-2 in 2.06 ms — equal to its representation
  oracle (4.43e-2 mean), i.e. the solver saturates the manifold and accuracy is
  capacity-limited; same-job CG at tol 1e-1 reaches 3.24e-2 in 1.31 ms, so Poisson at
  N=64 remains a classical win.

**Caveats that must travel with these numbers:** the Burgers baseline is the
truth-generating fixed-Newton rollout, *not* the optimized cubic-history solver — no
matched-tolerance strong classical arm ran in this job, so the 3.3× is promising, not a
supported win; 4 trajectories, one GPU, no AB/BA repetition protocol; cross-job
comparisons to the FiLM k-sweep numbers are indicative only (different GPUs); and the
cached-vs-meshfree gap *within* this job is small (~12%) because the separable network
is itself tiny — the large speedup vs FiLM comes from the architecture, and quoting it
properly needs a same-job FiLM arm.

## 9. File map and how to run

```
sep_common.py     model (features/head/SeparableDecoder), auto-decoder training,
                  oracle fit, checkpoint I/O, timing helper
sep_poisson.py    Poisson cell: data → train → oracle → NNLS-EQ → gate 0 →
                  both arms through lm_tau_poisson → FOM CG ladder → JSON
sep_burgers.py    Burgers cell: data → train → NNLS-EQ → incumbent weak ops +
                  cached ops → gate 0 → IC fit → both-arm rollouts (error + timed
                  jitted rollout) → FOM reference rollout → JSON
cluster/          run.sbatch (the exact submitted script) ; stage/ is gitignored
runs/sepdec_r1/   pulled logs + result JSONs + trained checkpoints (committed)
```

Local smoke (GB10, ~2 min each):

```bash
source /etc/profile.d/jax-mem.sh
N=16 K=8 R=32 M=32 MQ=128 STEPS=4000 N_TEST=4 \
  JAX_DEFAULT_MATMUL_PRECISION=highest \
  jaxrun /home/tahmid/Dev/.venv/bin/python sep_poisson.py     # or sep_burgers.py
```

Cluster: copy `cluster/stage/` (code + the `deps/` tree of sibling-experiment modules),
`sbatch run.sbatch` from an isolated paralab directory — the script carries the
mandatory `jax_backend=gpu` preflight and per-cell env (`N=64, K∈{8,16}, R=4K, M=4K,
m=16K, STEPS=40000`).

Dependencies are imported from the sibling experiments, never re-implemented:
`ms_parametric` (Poisson family/FOM), `blat_common` + `burgers2d_film` (Burgers testbed,
weak ops, solver), `ctol_eq`/`ctol_tol` (NNLS-EQ, trust-LM), resolved by
`sep_common._bootstrap()` for both the worktree and the staged `deps/` layouts.
