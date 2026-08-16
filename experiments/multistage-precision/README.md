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

---------------------------------------------------------------------------

# RESULTS (2026-08-15; Tufts A100 runs `tnrom-msprec{,2,3}-20260815`, jax_backend=gpu in every log, cluster dirs deleted after checksummed pull; local GB10 for c1/c4/c16)

All numbers: seed 0, single seed. Full tables (every arm, both metrics,
amplitude quartiles, LM accounting): `python ms_summarize.py .` over
`runs/`. Config manifests are inside every report JSON.

## (A0) Single function — the staging code works

`ms_function_report.json` (N=128, tanh 64x3, Adam 20k + L-BFGS 2k, OLD
frequency units): 3.1e-4 -> 2.6e-6 -> 1.4e-8 -> **3.3e-11** over 4 stages.
With the CORRECTED half-cycle schedule (`runs/control/c1`, FiLM 128x4, Adam
only, n_train=1): **1.84e-4 -> 7.6e-11 in ONE stage** (2.4e6x; stage-1
n_freq 28 for f_d = 12 cyc/unit). Machine-precision-class regression per the
paper is reproduced.

## (F6 control) Budget vs representation — the stage-1 gain vs family size

Same code path, true z given, N=64, hidden 128x4, 20k Adam steps/stage
(P_SUB = points/sample/step; 0 = all 4096).

| arm | n_train | batch / P_SUB | extras | stage-0 TRAIN | stage-1 TRAIN | **stage-1 gain** | stage-1 n_freq (f_d cyc) | VAL mean-rel s0 -> s1 |
|---|---|---|---|---|---|---|---|---|
| c1 | 1 | 1 / all | — | 1.84e-4 | 7.58e-11 | **2.4e6x** | 28 (12) | n/a (1 sample) |
| c4 | 4 | 4 / all | — | 3.74e-4 | 2.58e-7 | **1450x** | 26 (11) | n/a |
| c16 (local) | 16 | 16 / 1024 | L-BFGS 300 | 8.87e-4 | 1.65e-5 | **54x** | 22 (9) | 0.51 -> 0.51 |
| c16_lbfgs | 16 | 16 / all | L-BFGS 300 | 8.29e-4 | 1.60e-5 | **52x** | 22 (9) | 0.52 -> 0.52 |
| c16_2x | 16 | 16 / all | 40k steps, const LR | 2.26e-3 | 7.42e-5 | **30x** | 18 (7) | 0.51 -> 0.51 |
| c64 | 64 | 64 / all | — | 1.90e-3 | 1.45e-4 | **13x** | 16 (6) | 0.110 -> 0.110 |
| c64_lbfgs | 64 | 64 / all | L-BFGS 300 | 1.80e-3 | 1.28e-4 | **14x** | 16 (6) | 0.111 -> 0.111 |
| c64_bw | 64 | 64 / all | n_freq >= 32 | 1.66e-3 | 2.93e-4 | **5.7x** | 38 | 0.075 -> 0.076 |
| c256 | 256 | 256 / 1024 | — | 2.65e-3 | 6.20e-4 | **4.3x** | 6 (1) | 9.5e-3 -> 1.0e-2 |
| c256_bw | 256 | 256 / 1024 | n_freq = 32 all stages | 2.14e-3 | 7.01e-4 | **3.0x** | 32 | 5.8e-3 -> 5.7e-3 |
| par512 (= control (1)) | 512 | 32 / 1024 | 3 stages | 3.96e-3 | 1.48e-3 (s2 1.15e-3) | **2.7x** (3.4x @3) | 6 (1) | 4.9e-3 -> 4.5e-3 -> 4.5e-3 |
| par512_full | 512 | 512 / 256 | full-batch samples | 2.83e-3 | 8.98e-4 | **3.1x** | 6 (1) | 4.4e-3 -> 4.1e-3 |
| par512_bw | 512 | 32 / 1024 | n_freq = 32 all stages, 3 stages | 3.93e-3 | 1.74e-3 (s2 1.46e-3) | **2.3x** (2.7x @3) | 32 | 4.7e-3 -> 4.2e-3 -> 4.1e-3 |
| par512_zff | 512 | 32 / 1024 | Fourier features on z (4 freqs) | 5.17e-3 | 1.75e-3 | 3.0x | 6 | **3.0e-2 -> 3.1e-2** (6x WORSE val: overfits) |

TRAIN = global Frobenius rel; VAL = mean per-sample rel-L2 (both metrics in
the JSONs; they agree to ~10% everywhere).

**Verdict: REPRESENTATIONAL, in the conditioning direction — not an
optimizer/budget artifact.**
- The stage-1 gain decays smoothly and monotonically with the number of
  distinct parameter points the residual net must fit: 10^6.4 (1) -> 10^3.2
  (4) -> 10^1.7 (16) -> 10^1.1 (64) -> 10^0.6 (256) -> 10^0.4 (512).
- Budget arms move nothing: 2x steps/constant LR (30x) ~ L-BFGS polish (52x)
  ~ plain (54x) at n_train=16; L-BFGS changes the full-batch loss by <15%
  at every size; full-batch-in-samples at 512 (3.1x) ~ minibatch (2.7x).
- x-bandwidth is not the limiter: forcing n_freq>=32 at every stage
  (`*_bw`) leaves 512/256 unchanged (2.3-2.7x / 3.0x) and makes 64 WORSE (5.7x vs 13x) — the family stages
  are NOT starved of Fourier features even though the peak-frequency probe
  returns 1 cyc/unit while the residual's spectral centroid is 11-17
  cyc/unit (broad spectrum; see diag).
- Fourier features on z (`par512_zff`) do not help train (3.0x) and make val
  6x worse (overfitting 512 samples in 4-D): the naive "spectral bias in z"
  remedy is refuted; the z-pathway is data/dimension-limited, not
  frequency-limited.
- `ms_diag_report.json` (whitened nearest-neighbour correlation of residual
  fields in parameter space, 1-NN / 5-NN): fields 0.95 / 0.91 (smooth in z);
  true-z residual after stage 1: 0.38 / 0.23; after 2: 0.08 / 0.05; after 3:
  -0.01 / 0.00 — the residual becomes uncorrelated between neighbouring
  parameters, i.e. ROUGH in z. No smooth function of (x, z) fits it, so no
  per-stage power law. Spectral centroid of the residual in x rises 2.7 ->
  10.9 -> 15.1 -> 16.7 cyc/unit (rougher in x too, but the bandwidth arms
  show that is not what binds).
- VAL never improves with staging at any family size (generalization floor
  binds first): 4.9e-3 -> 4.5e-3 at 512; identical to 3 digits at <= 64.

## (2) Auto-decoder, K_LAT = 4 and 8 (`runs/autodec/`)

n_train 512, 20k steps/stage, batch 32 x 1024 pts, lazy latent Adam, LM
budget 60 attempts (identical for oracle and ROM), m_eq 512, 16 held-out
sources.

### Floors 1 and 2 — representation and finite-budget inferred latents

| K | stage | n_freq | TRAIN at learned latents (global / mean-rel) | VAL, stage-0 latents fixed | VAL LM-inferred (best / mean-start / nearest-start) |
|---|---|---|---|---|---|
| 4 | 0 | 6 | 1.59e-2 / 1.48e-2 | 2.03e-2 | 2.03e-2 / 7.84e-2 / 2.81e-2 |
| 4 | 1 | 8 | 6.34e-3 / 6.32e-3 | 2.36e-2 | 2.15e-2 / 7.95e-2 / 3.04e-2 |
| 4 | 2 | 8 | 4.88e-3 / 5.02e-3 | 2.35e-2 | 2.13e-2 / 7.99e-2 / 3.02e-2 |
| 8 | 0 | 6 | 1.17e-2 / 9.40e-3 | 8.26e-3 | 8.26e-3 / 1.28e-2 / 8.26e-3 |
| 8 | 1 | 8 | 3.39e-3 / 3.16e-3 | 1.21e-2 | 8.60e-3 / — / — |
| 8 | 2 | 12 | 2.10e-3 / 2.16e-3 | 1.20e-2 | 8.38e-3 |
| 8 (bw, n_freq>=32) | 2 | 36 | 2.41e-3 / 2.33e-3 | 1.63e-2 | 8.81e-3 |
| 4 (z_ff=4) | 2 | 16 | 2.14e-3 / 2.09e-3 | 3.33e-2 | 3.09e-2 |

- Staging on frozen learned latents improves the TRAIN representation 3.3x
  (K=4) / 5.5x (K=8) over 3 stages — same order as the true-z control, no
  power law. The auto-decoder stage 0 is 3-4x WORSE than the true-z decoder
  (1.6e-2 / 1.2e-2 vs 4e-3): learned latents at 20k joint steps do not reach
  the true-parameter conditioning.
* The held-out inferred-latent error is FLAT across stages (K4 ~2.1e-2, K8
  ~8.4e-3) — stages fit the training residual only. K=8 generalizes better
  than K=4 (2.4x) at every stage. Multi-start matters: the mean-latent start
  is 1.5-4x worse than the nearest-source-parameter start; the Adam
  inference (secondary) is 6.2e-2 (K4) / 1.1e-2 (K8) — worse than LM, as the
  review predicted.

### Floor 3 — the ROM solve (held-out sources, LM on the ghost-zero FD residual)

nearest-source-parameter init (mean init is 2-4x worse everywhere: the
residual landscape is multi-modal in z). 60-attempt budget; the 300-attempt
rerun (`ms_rom_rerun_K{4,8}_300.json`) converged 16/16 in every arm and
changed no number by more than a few % — the floor is NOT the solver budget.

| K | colloc | stages | ROM rel-L2 mean (med) | oracle, same start & budget | ‖r(z_LM)‖ / ‖r(z_oracle)‖ / ‖f‖ | boundary block | z-norm LM / oracle, NN-latent dist |
|---|---|---|---|---|---|---|---|
| 4 | full | 1 | 2.13e-1 (1.70e-1) | 3.58e-2 | 1.91 / 2.37 / 5.13 | 8e-5 | 0.30 / 0.32, 0.06 |
| 4 | full | 2 | 2.16e-1 (1.31e-1) | 4.00e-2 | 2.67 / 2.96 / 5.13 | 8e-5 | 0.26 / 0.30, 0.05 |
| 4 | full | 3 | 2.18e-1 (1.45e-1) | 3.98e-2 | 2.69 / 2.96 / 5.13 | 6e-5 | 0.26 / 0.30, 0.05 |
| 4 | m512 | 1 / 2 / 3 | 3.06e-1 / 4.02e-1 / 3.21e-1 | 3.6-4.0e-2 | 0.63 / 0.89 / 1.74 (s1) | 8e-5 | |
| 8 | full | 1 | **6.25e-2 (5.06e-2)** | 7.78e-3 | 0.82 / 1.01 / 5.13 | 8e-5 | 0.47 / 0.42, 0.21 |
| 8 | full | 2 | 7.66e-2 (6.31e-2) | 8.22e-3 | 1.06 / 1.27 / 5.13 | 7e-5 | 0.42 / 0.41, 0.15 |
| 8 | full | 3 | 8.55e-2 (5.91e-2) | 8.03e-3 | 1.04 / 1.30 / 5.13 | 5e-5 | 0.42 / 0.41, 0.17 |
| 8 | m512 | 1 / 2 / 3 | 9.65e-2 / 1.18e-1 / 1.28e-1 | 7.8-8.2e-3 | 0.25 / 0.35 / 1.74 (s1) | 8e-5 | |
| 8 (bw) | full | 1 / 2 / 3 | 1.03e-1 / 1.18e-1 / 1.11e-1 | 7.6-8.3e-3 | 1.18 / 1.31 (s1) | 4e-5 | |
| 8 | full, staged init | 2 / 3 | 8.8e-2 / 1.0e-1 | 8.0e-3 | | | |

**Reading the three floors (K=8, full collocation, nearest init):**
manifold (train, learned latents) 2.1e-3 -> finite-budget inferred-latent
(held-out) 8.4e-3 -> ROM solve 6.3e-2 (1 stage) / 8.6e-2 (3 stages).
K=4: 4.9e-3 -> 2.1e-2 -> 2.1e-1.

- **The ROM floor is the OBJECTIVE, not the solver and not the decoder's
  data precision.** In every nearest-init arm the LM solution has a LOWER
  PDE residual than the oracle latent (e.g. K8 s1: 0.82 vs 1.01, ‖f‖ 5.13)
  yet 8x larger field error. The minimum-residual point on the decoder
  manifold is not the minimum-field-error point: the residual applies the
  FD Laplacian (x(N-1)^2 ~ 4000 gain on grid-scale error) to the decoder
  error, so a 1% field error whose spectrum has grid-scale content produces
  residuals of order ‖f‖/5, and the solver legitimately trades field
  accuracy for residual reduction. This is exactly why decoder precision as
  measured by field error does not translate into ROM precision through a
  strong-form/FD residual, and why more precise stages (whose corrections
  are HIGHER frequency, centroid 9 -> 16 cyc/unit) make the ROM slightly
  WORSE (K8: 6.3e-2 -> 8.6e-2), not better.
- Solver accounting is clean: boundary block ~1e-4 (irrelevant), no NaN
  terminations, 300-attempt rerun converged everywhere with the same
  numbers; the m512 subset arms are 1.5-2x worse than full collocation
  (EQ-sampling loss, expected).
- Latent geometry: ROM latents have the same norm as oracle latents and sit
  0.05-0.2 from a training latent (K8 more spread) — the ROM is not escaping
  the training distribution; it converges to a genuinely different point on
  the manifold.

## Conclusions

1. Wang & Lai staging works as advertised for a FIXED target: 1.8e-4 ->
   7.6e-11 in one stage on one Poisson field (f64, correct half-cycle
   frequency schedule).
2. On a parametric FAMILY the per-stage gain collapses smoothly with family
   size (2.4e6x -> 2.7x from 1 to 512 samples) and no arm — 2x budget,
   L-BFGS, full-batch, wide x-bandwidth, z-Fourier features — restores it.
   The residual after stage 0 is ROUGH in the parameter direction (NN
   correlation 0.95 for fields -> 0.38 -> 0.08 -> 0.0 for residuals): a
   z-conditioned net cannot represent it. Held-out error never improves
   with staging.
3. A multi-stage AUTO-DECODER can be trained (frozen-latent staging, 3-5x
   train gain, K=8 held-out inferred-latent 8.4e-3), but the ROM solve on
   the FD residual lands ~8x above the inferred-latent error and gets NO
   benefit from staging (slightly worse). The floor is the residual
   objective (Laplacian amplification of decoder error), demonstrated by
   ‖r(z_LM)‖ < ‖r(z_oracle)‖ with larger field error and by unchanged
   numbers at 5x the LM budget.
4. What WOULD move the ROM floor (not run here): a residual metric that
   does not amplify grid-scale decoder error — energy-norm/H^-1 (inverse-
   Laplacian-weighted) least squares, or a Galerkin projection with smooth
   test functions; hard-BC decoders; and, for the decoder itself,
   PDE-residual-in-the-loss training (physics-informed staging, which the
   paper does show works for a FIXED PDE instance) rather than data-fit
   staging over a family.

## Confounds / caveats (explicit)

- Single seed everywhere; n_train=512 val gaps and ROM medians vs means
  (K8 ROM mean 6.3e-2 vs median 5.1e-2, max 0.16) show heavy tails — 16
  test sources.
- P_SUB point subsampling (1024/4096) in the family arms and the K runs
  (`par512_full`/`c64`/`c16_*` all-points arms agree, but it is a
  difference from the paper's full-batch regime).
- Frequency probe = peak of the radial-MEAN spectrum; on the broad-spectrum
  family residual it returns 1 cyc/unit while the centroid is 11-17. The
  `_bw` arms show it does not change the conclusions; a centroid-based
  schedule would be the cleaner default.
- The auto-decoder stage 0 (learned latents) is 3-4x worse than the true-z
  decoder at equal budget; more joint steps / latent LR tuning could move
  floor 1 (but not, per the ROM analysis, floor 3).
- Oracle = same LM, same budget, two starts: a finite-budget number, not a
  bound (labelled so throughout).
- Grid-node precision vs the FD solution (data floor ~1e-13); nothing here
  is about continuum accuracy.
- ms_function's stage-0/1 used the OLD (2x-off) frequency units — kept as
  the historical validation; `c1` is the corrected-units single-function
  number.

## Cluster-scale follow-up (proposal)

1. ROM objective study on the SAME K=8 checkpoints: H^-1 / energy-norm
   residual, Galerkin with smooth test functions, and hard-BC decoders —
   the cheapest test of conclusion 4.
2. Physics-informed staging for one PDE instance behind the GN solve
   (paper Sec. 4 regime) to measure the ROM floor when the decoder is
   trained on the residual itself.
3. Multi-seed (3) replication of the F6 ladder and the K=8 auto-decoder
   (A100, ~15 min/arm) to put error bars on the 2.7x/3.4x family gains.
4. Latent-dimension ladder K in {8, 16, 32} at 3x joint steps.
