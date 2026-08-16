# Multi-stage precision decoders + ROM solve (2026-08-14/15)

Question (user): can the Wang & Lai multi-stage training scheme (JCP 504
(2024) 112865, "Multi-stage neural networks: function approximator of machine
precision") drive a coordinate decoder toward machine precision — and, for an
AUTOENCODER-style decoder, does that precision survive when the decoder is
used inside a reduced-order solve of the simple Poisson-2D problem?

Everything here: Poisson 2D, `-lap u = a exp(-((x-cx)^2+(y-cy)^2)/2w^2)` on
the unit square, u=0 walls, f64 FD/CG truth (max rel residual over all samples
~1e-13, asserted), same bump family as the coord-decoder testbed
(`exp/2026-08-12-coord-decoder`), seed 0 (SINGLE seed — no error bars).
f64 data + f64 networks THROUGHOUT (below the f32 floor ~1e-7 is the point).
Local GB10, jax_backend=gpu, JAX_DEFAULT_MATMUL_PRECISION=highest. Precision
is always measured at the FD GRID NODES against the FD solution ("grid-node
precision"), never against a continuum solution.

The pipeline was rebuilt after two independent adversarial reviews (Codex +
a fresh Claude reviewer; consolidated findings F1-F8 in the parent session)
before any full run — every table below comes from the corrected code
(commit `77aadb9`+).

## Method (as implemented)

Per stage k (paper's Algorithm 1, Fourier-feature variant of the frequency
matching): `target_k = e_k/eps_k`, `eps_k = RMS(e_k)`, `e_0 = u`,
`e_{k+1} = e_k - eps_k*net_k`; combined model `u ~= sum_k eps_k net_k`.

* **Units.** Coordinate features are `sin/cos(pi*j*x)`, j = 1..n_freq — j is a
  HALF-CYCLE index (j/2 cycles per unit length). The residual-frequency probe
  (`ms_parametric.dominant_radial_freq`) returns f_d in CYCLES/unit: radial
  MEAN amplitude per unit-width FFT annulus over the whole training residual
  set, DC dropped (a per-annulus SUM is biased to the Nyquist ring — white
  noise returns the last bin). Schedule `n_freq = ceil(2 f_d) + 4`,
  non-decreasing across stages, capped at N-1 (the half-cycle Nyquist).
  Unit-tested on known 2-D sinusoids + white noise: `ms_freq_test.py`.
* **Loss.** Per-sample inverse-energy weights (mean 1): the family's snapshot
  energies span >100x, so an unweighted MSE fits only the loud samples. Every
  arm trains on RELATIVE error. Metrics are reported BOTH as the global
  Frobenius relative error `||E||_F/||U||_F` and the mean per-sample rel-L2,
  on TRAIN and VAL, plus val error by amplitude quartile.
* **Nets.** FiLM coordinate decoder: Fourier features in x -> 4x128 swish
  trunk, FiLM-modulated by a 64-unit embedding of the conditioning vector
  (true parameters z, or the learned latent). `Z_FF=m` optionally adds
  Fourier features `sin/cos(pi j z)` on the conditioning input (spectral-bias-
  in-z control).
* **Optimizer.** AdamW warmup-cosine (peak 2e-3), minibatch of `BATCH`
  samples x `P_SUB` random grid points per step (f64 on the GB10 is ~140
  ms/step at full 64^2 grids); control arms use full batch / all points /
  constant LR / full-batch L-BFGS polish (`LBFGS_STEPS`).
* **Auto-decoder (`ms_autodecoder.py`).** Stage 0 learns per-snapshot latents
  jointly with the decoder using a LAZY per-row Adam (only the batch's rows
  update moments/step counters — no drift on stale momentum). Latents are then
  FROZEN and stages >= 1 fit the residual over (x; z). Held-out latents are
  obtained by the SAME Levenberg-Marquardt solver as the ROM applied to the
  DATA-MISFIT `dec(z) - u` at all grid nodes, multi-start (mean training
  latent / nearest-in-source-parameter training latent), same attempt budget
  as the ROM. This is a **finite-budget inferred-latent error**, NOT a lower
  bound (an Adam-inferred secondary row is also recorded).
* **ROM (`ms_autodecoder.py` phase C).** Online the solver knows only the
  source f and the decoder. Residual = ghost-zero-Dirichlet 5-point FD
  operator on the decoded INTERIOR field minus f (boundary rows DROPPED, so
  the residual is exactly the FOM operator and a perfect decoder at some z
  zeros it); the boundary block `||dec(z, walls)||` is recorded separately.
  Collocation: full interior, or an m-node random EQ-style subset (wall
  neighbours contribute 0). LM: adaptive damping on diag(J^T J), accept/
  reject, NaN guard, stop on rel decrease < 1e-12 or step < 1e-13, budget =
  attempts (each 1 residual eval, accepted ones +1 Jacobian eval); inits:
  mean latent, nearest-source-parameter latent, and STAGED (stage-0 decoder
  for half the budget, full sum for the rest — equal total). Every arm
  records `||r(z_LM)||`, `||r(z_oracle)||`, `||f||` (restricted), the field
  error AT the oracle latent, latent norms and nearest-training-latent
  distances, so "objective floor" vs "solver floor" is decidable.
* **Provenance.** Reports/pkls carry the config manifest and a `complete`
  flag; consumers assert on it (`ms_diag.py`), the summarizer marks
  incomplete runs; held-out-derived val latents live in a `*_TAINTED.pkl`.

## The three "floors" — how they are labeled

1. **Manifold / representation error at the LEARNED training latents** —
   TRAIN fit of the staged decoder (K_LAT-dim manifold; both metrics).
2. **Finite-budget inferred-latent error on held-out fields** — LM on the
   data misfit, multi-start, same budget as the ROM ("oracle" in the tables,
   NOT a bound: it is what a data-fit could reach at this budget).
3. **ROM-solve error** — LM on the PDE residual only, from the source.

Scripts: `ms_function.py` (single field, validation of the staging code),
`ms_parametric.py` (true-z control + shared blocks), `ms_autodecoder.py`,
`ms_diag.py` (residual roughness in the conditioning variable — whitened NN
correlation), `ms_freq_test.py`, `ms_summarize.py`.
