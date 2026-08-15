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
