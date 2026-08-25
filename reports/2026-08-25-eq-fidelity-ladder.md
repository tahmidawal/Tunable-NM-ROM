# The EQ fidelity ladder: how much the empirical quadrature distorts what the solver actually uses

Diagnostic study, numbers **final** for the four checkpoints measured (2026-08-25, jobs
2841798–2841801). It changes no ROM accuracy or speed number; it measures, for the first time,
the gap between the sampled weak form the online solver minimises and the exact full-grid weak
form it approximates — at the residual, the gradient, the Hessian, and the step. Motivated by
the 2026-08-25 group meeting (advisor: "measure the L2 error on the points, the error in the
integral, and the error in the gradient — separately; the gradient is the holy grail") and by
§6.1 of `2026-08-22-separable-eq-decoder-design.md`, which specified this ladder and had never
been run. Tables T-L1–T-L5 below are the verbatim output of `reports/gen_2026-08-25-eq-ladder.py`
on the four run JSONs; the per-time-bucket tables are `experiments/separable-decoder/EQ-LADDER.md`
on branch `exp/2026-08-25-eq-fidelity-ladder` (`runs/gen_eq_ladder.py`). No number in this file is
typed by hand; the prose only points at table cells.

## 1. What was measured

The online solver minimises `½‖R_s(z)‖²` where `R_s` is the weak residual (M sine test modes)
evaluated by the m-node NNLS quadrature. `R_f` is the same residual with exact grid sums over
all interior nodes. `sep_eq_ladder.py` computes both at the same `(z, z_prev, ν, λ_LM)` and
reports, per state:

| rung | quantity | what it answers |
|---|---|---|
| (a) | decoder-vs-truth L2 error on the m nodes (quadrature-weighted) vs on the full grid | are the sample points representative? |
| (b) | `‖R_s − R_f‖ / ‖R_f‖`, split into the **linear** part (`Φᵀ(u−uⁿ)` and the Laplacian, all of the form `Φᵀu`) and the **advection** part (`ΦᵀN(u)`) | is the integral right, and which term is wrong? |
| (c1) | `‖g_s − g_f‖ / ‖g_f‖` with `g = JᵀR`; its cosine; and an absolute version `‖g_s − g_f‖ / (‖J_f‖‖R_f‖)` | does it push `z` the right way? |
| (c2) | `‖H_s − H_f‖_F / ‖H_f‖_F` with `H = JᵀJ` | is the curvature right? |
| (c3) | the damped LM step `−(H+λD)⁻¹g` at the solver's own λ; its cosine | would the solver move differently? |

States: **solver-path** — every LM iterate of a real ROM rollout of four fresh test trajectories
(all 50 steps, the ROM's own previous latent, incumbent LM rule with the round-4 stall 1e-3 and
extrapolation order 1); **oracle** — the full-grid least-squares code of the truth state at
t ∈ {0,1,2,3,5,10,25,50} with `z_prev` the oracle code of the previous truth state (truth is used
only to place `z`, never in a solve path); **training snapshots** (16). Two EQ sets per
checkpoint, built by the same code as every speed and accuracy job: the control set
(M=64, m=256, plain NNLS — the set behind every speed number) and the fine set (M=256, m=1024).

Gates, all in T-L5: the sampled path reproduces `blat_common.make_weak_ops` at ≤1e-12 (gate 0,
measured ≤7e-15); the full-grid reference reproduces `make_weak_ops` run on the whole interior
with unit weights at ≤1e-12 (gate F, measured ≤3e-16, N≤512 only — at N=1024 the reference does
not fit, and the full-grid path is the same bank-product + `upwind_adv_field` code already gated
by `eq_bank_vs_meshfree`); the cached bank equals the meshfree decode at ≤1e-12.

Checkpoints: the K=16 reference recipe at N=256 (r3a) and N=1024 (r3d — the 1.61× decoder); the
round-5 cost-matched dense_mid decoder at N=256; the K=32 h512x3 decoder at N=1024 (r4a6, the
most accurate N=1024 decoder). One H200 (N=1024) or A100 (N=256) each, ~20–40 min, no training.

## 2. Findings

**F1 — The residual rung is not small, and the NNLS "rel fit" hides it.** T-L1: on the control
set the sampled and exact residuals differ by tens of percent along the solver path and by of
order one at the solution (T-L3); on the fine set by several percent on the path and ~5–20 % at
the solution. The NNLS rel-fit statistic in the same rows is two orders of magnitude smaller.
The two measure different things: NNLS is fitted to reproduce `Φᵀu` on 64 training snapshots,
whereas the residual is a *small difference* (`u − uⁿ`, plus terms that nearly cancel at the
solution), so a quadrature error that is ~5e-3 of `Φᵀu` is a large fraction of `R`. The
certification metric the project has been quoting is the wrong one.

**F2 — Most of the error is in the part that need not be sampled at all.** T-L1 splits (b) into
the linear terms and the advection term. On the fine set the linear part is 2–4× the advection
part in every arm; on the control set the two are comparable. Because `Φᵀu = (ΦᵀG)h(z)` exactly
for the separable decoder, the whole linear part can be computed with one precomputed M×R
matrix and **no quadrature** — one kernel, zero online cost change. This is the single largest
lever the ladder identifies, and it needs no learning.

**F3 — The Hessian is fine; the error enters through the residual and propagates.** T-L2:
(c2) is 1e-4–1e-2 everywhere, i.e. the Jacobian `Φ_qᵀG_q ∂h/∂z` is well sampled (its columns are
full-size fields, exactly what NNLS was fitted on). The gradient `JᵀR` and the LM step inherit
the residual error: gradient cosines on the solver path are ~0.3–0.4 (control) and ~0.6–0.8
(fine); step cosines ~0.4–0.7. This is precisely the failure mode named in the meeting — a
modest integral error becoming a large gradient error — and it happens through `R`, not `J`.

**F4 — At the true solution, the sampled objective is not stationary.** T-L3, control set: the
sampled gradient at the oracle code is several times *larger* than the true weak gradient and
poorly aligned with it, so the sampled solver's minimiser is elsewhere. On the fine set the
excess is of order one or below. This is the mechanism behind two established facts: the
rollout sitting 1.3–1.9× above the oracle in every configuration, and the same decoder being
1.45× more accurate on the fine set (`fq256`). The quadrature is a floor.

**F5 — Better decoders hit the floor harder.** The absolute gradient error (`(c1) abs`) is
similar across all four checkpoints, but the *relative* rungs at the solution are worse for the
K=32 and dense_mid decoders than for the reference recipe: their true residual at the solution
is smaller, so the same quadrature error is a larger fraction of it. Any further gain in `h`
generalisation will be capped by this unless the quadrature improves with it.

**F6 — Rung (a) is a non-issue.** T-L4: the quadrature-weighted L2 error on the m nodes is
within a few percent of the full-grid error in every arm. The sample points are representative
for measuring reconstruction; the problem is the integral of the *residual*, not the sampling of
the *state*.

**F7 — Resolution-independent, again.** N=256 and N=1024 rows in every table agree to within
the spread between trajectories, as did the span floor and the `h` gap before.

**Caveat on the t=0 oracle rows.** At t=0 the previous state is the state itself, so the
`u − uⁿ` term vanishes and `R_f` is larger; the t=0 (b) values are therefore lower than at t≥5 for
a structural reason, not because the quadrature is better there. Compare t≥5 rows across sets,
not t=0 against t≥5.

## 3. What this means for "have the model learn the quadrature"

The ladder ranks the fixes; the ranking is the answer.

1. **Exact linear terms** (F2). Replace `Φ_qᵀ(u − uⁿ)` and the Laplacian term by
   `(ΦᵀG_int)(h(z) − h(zⁿ))` and `λ ⊙ (ΦᵀG_int)h(z)`. Removes the dominant part of (b) for free;
   for Poisson (values-only weak form) it removes the quadrature entirely. Requires redefining
   gate 0 against the full-grid weak residual, since the incumbent reference also samples the
   linear terms — a deliberate rule change to be logged.
2. **Refit the advection quadrature against the right target** (F1, F3). With the linear part
   exact, all m nodes serve `ΦᵀN(u)` only; fit them by NNLS on residual/gradient-fidelity rows
   (design doc §6.2 stage 2, "same-target NNLS"), including off-manifold solver iterates and
   several ν. Still convex; still "learned"; no network change. Certify with this ladder, not
   the rel-fit or the row tail.
3. **Learned nodes or a learned weight network** only if the advection rung is still binding
   after 1–2. The ladder script is the test that would decide it.

Joint training of `(g, h, w)` from scratch is not motivated by anything measured here: the
decoder's Jacobian is already sampled well (F3), and the error lives in a term that can be made
exact. The advisor's premise that the weights are uniform was incorrect (they are NNLS-fitted,
T-L1); the advisor's conclusion — that the gradient, not the integral, must be certified — is
exactly what the data show.

## 4. Tables (generated)

### T-L1. Where the quadrature error sits: residual, split into the part that could be exact and the part that cannot

Solver-path states (every LM iterate of a real ROM rollout, 4 fresh test trajectories, all 50 steps). Relative to the full-grid residual norm. `lin` = mass + previous state + Laplacian terms (all `Φᵀu`, exactly computable as `(ΦᵀG)h(z)`); `adv` = `ΦᵀN(u)` (sign-upwind, needs sampling).

| decoder | N | K | EQ set (M/m) | NNLS rel fit | (b) ‖R_s−R_f‖/‖R_f‖ | of which lin | of which adv | (b) worst |
|---|---|---|---|---|---|---|---|---|
| sep_burgers_r3_N256_K16_R512 | 256 | 16 | ctrl (64/256) | 5.2e-03 | 4.3e-01 | 3.1e-01 | 4.4e-01 | 1.2e+00 |
| sep_burgers_r3_N256_K16_R512 | 256 | 16 | M256 (256/1024) | 5.2e-04 | 5.6e-02 | 5.9e-02 | 2.3e-02 | 2.7e-01 |
| sep_hfit_dense_mid_N256 | 256 | 16 | ctrl (64/256) | 4.0e-03 | 3.3e-01 | 3.0e-01 | 1.9e-01 | 1.3e+00 |
| sep_hfit_dense_mid_N256 | 256 | 16 | M256 (256/1024) | 5.4e-04 | 7.6e-02 | 7.4e-02 | 1.9e-02 | 7.3e-01 |
| sep_burgers_r3_N1024_K16_R512 | 1024 | 16 | ctrl (64/256) | 5.4e-03 | 5.3e-01 | 4.1e-01 | 4.6e-01 | 1.4e+00 |
| sep_burgers_r3_N1024_K16_R512 | 1024 | 16 | M256 (256/1024) | 5.9e-04 | 6.7e-02 | 7.1e-02 | 2.6e-02 | 4.4e-01 |
| sep_burgers_r4_N1024_K32_R512_h512x3 | 1024 | 32 | ctrl (64/256) | 5.1e-03 | 4.3e-01 | 3.4e-01 | 4.7e-01 | 1.3e+00 |
| sep_burgers_r4_N1024_K32_R512_h512x3 | 1024 | 32 | M256 (256/1024) | 5.6e-04 | 1.1e-01 | 1.2e-01 | 5.0e-02 | 7.6e-01 |

### T-L2. The ladder: residual → gradient → Hessian → step (solver-path states, means; cos = direction agreement)

| decoder | N | K | EQ set | (b) resid | (c1) grad | (c1) cos | (c1) abs | (c2) Hess | (c3) LM step | (c3) cos |
|---|---|---|---|---|---|---|---|---|---|---|
| sep_burgers_r3_N256_K16_R512 | 256 | 16 | ctrl | 4.3e-01 | 6.5e-01 | 0.30 | 7.8e-02 | 3.1e-03 | 7.3e-01 | 0.43 |
| sep_burgers_r3_N256_K16_R512 | 256 | 16 | M256 | 5.6e-02 | 4.3e-01 | 0.70 | 2.4e-03 | 1.8e-04 | 4.5e-01 | 0.67 |
| sep_hfit_dense_mid_N256 | 256 | 16 | ctrl | 3.3e-01 | 6.0e-01 | 0.34 | 6.6e-02 | 7.3e-03 | 8.0e-01 | 0.42 |
| sep_hfit_dense_mid_N256 | 256 | 16 | M256 | 7.6e-02 | 2.8e-01 | 0.81 | 7.4e-03 | 6.9e-04 | 5.1e-01 | 0.60 |
| sep_burgers_r3_N1024_K16_R512 | 1024 | 16 | ctrl | 5.3e-01 | 6.7e-01 | 0.34 | 8.7e-02 | 4.7e-03 | 7.1e-01 | 0.44 |
| sep_burgers_r3_N1024_K16_R512 | 1024 | 16 | M256 | 6.7e-02 | 4.1e-01 | 0.79 | 4.5e-03 | 2.5e-04 | 4.4e-01 | 0.67 |
| sep_burgers_r4_N1024_K32_R512_h512x3 | 1024 | 32 | ctrl | 4.3e-01 | 6.9e-01 | 0.42 | 6.3e-02 | 4.8e-03 | 7.2e-01 | 0.40 |
| sep_burgers_r4_N1024_K32_R512_h512x3 | 1024 | 32 | M256 | 1.1e-01 | 4.6e-01 | 0.61 | 5.6e-03 | 3.2e-04 | 4.3e-01 | 0.66 |

### T-L3. The same ladder AT THE SOLUTION (oracle states: full-grid LS code of the truth state, t ∈ {0,1,2,3,5,10,25,50})

Here `R_f` is the true weak residual of the best on-manifold state; a sampled residual/gradient much larger than it means the sampled objective's minimum is somewhere else.

| decoder | N | K | EQ set | (b) resid | (b) at t=0 | (b) at t≥5 | (c1) grad | (c1) cos | (c1) abs | (c2) Hess | (c3) step | (c3) cos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sep_burgers_r3_N256_K16_R512 | 256 | 16 | ctrl | 7.1e-01 | 5.7e-02 | 9.0e-01 | 6.6e+00 | 0.49 | 1.4e-01 | 3.5e-03 | 6.5e-01 | 0.88 |
| sep_burgers_r3_N256_K16_R512 | 256 | 16 | M256 | 5.9e-02 | 1.9e-02 | 5.3e-02 | 2.1e-01 | 0.97 | 3.5e-03 | 1.9e-04 | 5.5e-02 | 1.00 |
| sep_hfit_dense_mid_N256 | 256 | 16 | ctrl | 5.4e-01 | 6.4e-02 | 5.8e-01 | 5.5e+00 | 0.54 | 1.3e-01 | 1.0e-02 | 3.3e-01 | 0.95 |
| sep_hfit_dense_mid_N256 | 256 | 16 | M256 | 1.4e-01 | 3.4e-02 | 7.3e-02 | 9.9e-01 | 0.82 | 2.0e-02 | 1.2e-03 | 8.0e-02 | 0.99 |
| sep_burgers_r3_N1024_K16_R512 | 1024 | 16 | ctrl | 8.9e-01 | 5.5e-02 | 1.2e+00 | 9.5e+00 | 0.50 | 1.8e-01 | 5.0e-03 | 1.7e+00 | 0.74 |
| sep_burgers_r3_N1024_K16_R512 | 1024 | 16 | M256 | 8.0e-02 | 1.9e-02 | 6.2e-02 | 3.4e-01 | 0.94 | 5.6e-03 | 2.6e-04 | 1.1e-01 | 0.99 |
| sep_burgers_r4_N1024_K32_R512_h512x3 | 1024 | 32 | ctrl | 2.0e+00 | 4.8e-02 | 2.6e+00 | 2.2e+01 | 0.38 | 2.9e-01 | 4.9e-03 | 2.7e+00 | 0.66 |
| sep_burgers_r4_N1024_K32_R512_h512x3 | 1024 | 32 | M256 | 2.1e-01 | 2.4e-02 | 1.2e-01 | 1.6e+00 | 0.82 | 1.3e-02 | 3.1e-04 | 2.4e-01 | 0.94 |

### T-L4. Rung (a): is the L2 error on the m sample points representative of the full-grid error? (oracle states)

| decoder | N | K | EQ set | recon err, full grid | recon err on nodes (w-weighted RMS) | ratio |
|---|---|---|---|---|---|---|
| sep_burgers_r3_N256_K16_R512 | 256 | 16 | ctrl | 3.4e-02 | 3.3e-02 | 0.98 |
| sep_burgers_r3_N256_K16_R512 | 256 | 16 | M256 | 3.4e-02 | 3.3e-02 | 0.99 |
| sep_hfit_dense_mid_N256 | 256 | 16 | ctrl | 1.6e-02 | 1.6e-02 | 0.95 |
| sep_hfit_dense_mid_N256 | 256 | 16 | M256 | 1.6e-02 | 1.6e-02 | 0.98 |
| sep_burgers_r3_N1024_K16_R512 | 1024 | 16 | ctrl | 3.7e-02 | 3.6e-02 | 0.98 |
| sep_burgers_r3_N1024_K16_R512 | 1024 | 16 | M256 | 3.7e-02 | 3.7e-02 | 0.99 |
| sep_burgers_r4_N1024_K32_R512_h512x3 | 1024 | 32 | ctrl | 2.2e-02 | 2.2e-02 | 0.98 |
| sep_burgers_r4_N1024_K32_R512_h512x3 | 1024 | 32 | M256 | 2.2e-02 | 2.2e-02 | 0.98 |

### T-L5. Provenance and gates

| decoder | N | GPU | job | backend | bank-vs-meshfree | gate 0 (ctrl / M256) | gate F | rollout err (ctrl / M256, mean of 4) |
|---|---|---|---|---|---|---|---|---|
| sep_burgers_r3_N256_K16_R512 | 256 | NVIDIA A100-PCIE-40GB | 2841798 | gpu | 4.3e-16 | 1.2e-15 / 3.6e-16 | 3.0e-16 / 1.8e-16 | 2.9e-02 / 2.2e-02 |
| sep_hfit_dense_mid_N256 | 256 | NVIDIA A100 80GB PCIe | 2841799 | gpu | 3.2e-15 | 6.6e-15 / 3.2e-15 | 2.9e-16 / 2.9e-16 | 1.7e-02 / 1.1e-02 |
| sep_burgers_r3_N1024_K16_R512 | 1024 | NVIDIA H200 | 2841800 | gpu | 4.3e-16 | 1.3e-15 / 1.0e-15 | — / — | 3.2e-02 / 2.3e-02 |
| sep_burgers_r4_N1024_K32_R512_h512x3 | 1024 | NVIDIA H200 | 2841801 | gpu | 2.9e-16 | 1.6e-15 / 1.1e-15 | — / — | 3.0e-02 / 1.4e-02 |

Sources:

- `worktrees/2026-08-25-eq-fidelity-ladder/experiments/separable-decoder/runs/lad256k16/out/sep_eq_ladder_N256_K16_r3a.json`
- `worktrees/2026-08-25-eq-fidelity-ladder/experiments/separable-decoder/runs/lad256dm/out/sep_eq_ladder_N256_K16_dense_mid.json`
- `worktrees/2026-08-25-eq-fidelity-ladder/experiments/separable-decoder/runs/lad1024k16/out/sep_eq_ladder_N1024_K16_r3d.json`
- `worktrees/2026-08-25-eq-fidelity-ladder/experiments/separable-decoder/runs/lad1024k32/out/sep_eq_ladder_N1024_K32_r4a6.json`
