# Multi-stage precision decoders + ROM solve (2026-08-14/15)

Question (user): can the Wang & Lai multi-stage training scheme (JCP 504
(2024) 112865, "Multi-stage neural networks: function approximator of machine
precision") drive a coordinate decoder toward machine precision — and, for an
AUTOENCODER-style decoder, does that precision survive when the decoder is used
inside a reduced-order solve of the simple Poisson-2D problem?

All experiments: Poisson 2D, `-lap u = a*exp(-((x-cx)^2+(y-cy)^2)/2w^2)` on
the unit square, u=0 walls, f64 FD/CG truth (rel residual ~1e-13), same bump
family as the coord-decoder testbed (`exp/2026-08-12-coord-decoder`).
f64 data and f64 networks THROUGHOUT (the whole point is going below the f32
floor ~1e-7). Local GB10, jax_backend=gpu, JAX_DEFAULT_MATMUL_PRECISION=highest,
single seed 0.

## Method (as implemented)

Per stage k (Algorithm 1 of the paper, Fourier-feature variant of the
frequency matching):
  target_k = e_k / eps_k,  eps_k = RMS(e_k),  e_0 = u,  e_{k+1} = e_k - eps_k*net_k;
  combined model  u ~= sum_k eps_k * net_k.
Stage bandwidth n_freq_k is set from the dominant radial frequency of e_k (2D
FFT peak over sample fields, x1.5 + 4 margin), capped at the grid Nyquist
(N-1)//2 — the paper's Fig. 3 shows a Fourier-feature first layer is
equivalent to their kappa-scaled first layer for capturing the residue's
dominant frequency. Adam (warmup-cosine) per stage; L-BFGS polish only in the
single-function experiment.

Scripts:
- `ms_function.py`   — (A0) ONE field, validates the implementation.
- `ms_parametric.py` — (1) parametric FiLM decoder u(x; z), TRUE z given.
- `ms_autodecoder.py`— (2) AUTO-DECODER: learned latents, frozen-latent
  staging, and the Levenberg-Marquardt Gauss-Newton ROM solve on held-out
  sources (phases A/B/C).
- `ms_rom_solve.py`  — GN skeleton for the true-z decoder (superseded by the
  LM solver inside ms_autodecoder.py for the auto-decoder arms).

## Status 2026-08-15 — IMPLEMENTATION READY FOR REVIEW (full run NOT launched)

Control (1), partial (killed at the review gate during stage 2; log in
`runs/parametric/log.txt`; N=64, n_train 512, 20k Adam steps/stage, P_SUB=1024,
hidden 128x4, true z given):

| stage | n_freq | eps_in (abs RMS) | TRAIN fit rel-RMS | VAL rel-L2 | stage-final batch loss (normalized) |
|---|---|---|---|---|---|
| 0 | 16 | 2.36e-3 | 2.65e-3 | 7.76e-3 | 8.3e-6 |
| 1 | 16 | 6.26e-6 | 1.03e-3 | 6.49e-3 | 1.55e-1 |
| 2 | 22 | 2.43e-6 | (killed @5k/20k) | — | 6.8e-1 @5k |

Early read (to be confirmed by the full run + `ms_diag.py`): the per-stage
gain collapses from ~100x (single function, `ms_function_report.json`:
3.1e-4 -> 2.6e-6 -> 1.4e-8 -> 3.3e-11) to ~2.6x on the family, and stage
losses plateau at 0.15-0.7 of the normalized residual variance — a fresh
z-conditioned net cannot represent the family residual. Hypothesis: the
stage-0 error is ROUGH in the z direction (each sample's error is
idiosyncratic), so no smooth function of (x, z) fits it; `ms_diag.py`
quantifies this via nearest-neighbour-in-z residual correlation vs field
correlation. VAL barely moves (7.8e-3 -> 6.5e-3): the generalization floor
at n_train=512 binds first.

Auto-decoder pipeline (`ms_autodecoder.py`) smoke: N=32, n_train 64, 2
stages x 300 steps, 2 held-out ROM solves — runs end-to-end, finite,
`smoke/ms_autodecoder_K4_report.json` (numbers meaningless by design).
