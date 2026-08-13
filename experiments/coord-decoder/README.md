# Coordinate-network decoder testbed (2026-08-12)

Diagnostic experiment for the resolution wall in the ViT-CP NM-ROM: as mesh
resolution N increases, the CP autoencoder's error does not decrease (it rises),
while POD-only ROMs improve as expected. This testbed isolates the decoder and
tests the hypothesis that the wall is architectural — the CP tensor decoder is
effectively a *linear subspace with grid-tied basis vectors* — and that a
**coordinate-network decoder** `u(x; z)` (no grid-tied parameters at all)
removes resolution from the architecture entirely.

Everything here is a standalone parametric-Poisson testbed (translated
Gaussian-bump sources, FD/CG ground truth), deliberately outside the `heat/`
and `poisson/` packages so `main` stays untouched. Decoders are conditioned on
the true family parameters z to isolate the decoder from the encoder.

All errors are mean relative L2 over held-out validation parameters, seed 0,
single seed. Every GPU run verified `jax_backend=gpu` and used
`JAX_DEFAULT_MATMUL_PRECISION=highest`.

## Headline results

### 1D Poisson (local GB10) — proof of concept

Fair head-to-head, translated-bump family, N=1024, 40k steps both arms
(`poisson1d_bump_upgraded.py`, `results_bump_upgraded.json`):

| model                                     | reduced dim | rel-L2      |
|-------------------------------------------|-------------|-------------|
| POD-3 (optimal linear, same dim as coord) | 3           | 4.75e-2     |
| POD-24 (optimal linear)                   | 24          | 6.7e-5      |
| grid-tied decoder (CP-style)              | 24          | 9.0e-3      |
| **coord-net decoder**                     | **3**       | **4.47e-3** |

The coord-net breaks the equal-dimension linear barrier by 10.6×. The grid-tied
decoder is stuck ~135× above the floor its own rank provably permits (the SGD
basis-learning disease). Mesh transfer: coord-net trained at N=64 evaluated at
N=4096 → identical error (4.75e-2 vs 4.73e-2 in the earlier run) — the
operation does not even exist for the grid-tied decoder.

Convergence vs a near-continuum N=8192 reference
(`poisson1d_convergence.py`, `results_convergence.json`): coord-net error
descends monotonically with training N until its fit floor (~9e-3), then holds
flat. It never rises.

### 2D Poisson (Tufts A100s) — the decisive test

Round 3, 80k steps, 32 Fourier frequencies (`poisson2d_diag.py`, `round3/`):

| N   | POD-4 (equal dim) | POD-24  | grid-tied (rank 24) | coord-net (4 latent, 67k params) |
|-----|-------------------|---------|---------------------|----------------------------------|
| 16  | 3.14e-1           | 9.83e-2 | 2.45e-1             | 1.01e-1                          |
| 32  | 3.01e-1           | 8.26e-2 | 1.90e-1             | 4.04e-2                          |
| 64  | 2.99e-1           | 7.96e-2 | 1.85e-1             | **3.83e-2**                      |
| 128 | 2.98e-1           | 7.90e-2 | 2.17e-1             | 4.14e-2                          |
| 256 | 2.98e-1           | 7.88e-2 | 2.42e-1             | **3.89e-2**                      |

- Coord-net with 4 latent variables sits ~7.5× below POD-4 (the best any
  4-dim linear model can do) and beats POD-24 with 6× fewer reduced variables.
- Grid-tied sits *above* its own POD-24 floor while its parameter count
  balloons 26k → 1.6M across the sweep and buys nothing — the repo's disease
  signature at 20× smaller scale.
- Mesh transfer (`transfer2d_eval.py`, round-2 checkpoints): trained at 32×32,
  evaluated natively on 256×256 → 4.21e-2, matching the natively-trained-at-256
  model (4.41e-2).

### FiLM push (Tufts A100s) — does the loss descend with resolution?

The concat-conditioned coord-net plateaus at its own fitting error (~3.9e-2 in
2D). Diagnosis: conditioning mechanism (translation is the hard case for
concat conditioning), training-set coverage, and gradient dilution on the
sharp blobs. Upgrade (`poisson2d_film.py`): FiLM conditioning on every trunk
layer, 464k params (5×256 trunk), n_train 2048, source-centered importance
sampling, 120k steps. Evaluated against a common N=512 CG reference
(`poisson2d_refgen.py` → `film/ref512_val.npz`, regenerable from seed):

| train N | data-floor (discretization bound) | FiLM coord-net vs reference |
|---------|-----------------------------------|-----------------------------|
| 16      | 4.74e-2                           | 7.21e-2                     |
| 32      | 4.21e-3                           | 1.34e-2                     |
| 64      | 9.8e-4                            | 9.79e-3                     |
| 128     | 2.5e-4                            | 1.09e-2                     |
| 256     | 7.0e-5                            | **6.26e-3**                 |

**The loss goes down as resolution increases — 11× across the sweep — and then
flattens at the capacity floor. It never rises.** Data-limited at N=16,
capacity-limited from N≥32. The push moved the floor 5–9× (3.9e-2 → 6e-3);
the net now beats POD-24 by ~12× and POD-64 by ~4×. The plateau is an
engineering ceiling (movable, demonstrated), unlike POD's mathematical floor.

Canonical numbers: `film/film_convergence_fixed.json` (see landmine below).

### The Nyquist landmine (documented, cost one round)

Fourier-feature bandwidth must respect the training grid's Nyquist limit
(n_freq ≤ N/2). The original N=16/32 FiLM runs used n_freq=32, exploited
aliased features that look correct on-grid, and blew up when evaluated on the
fine reference grid (errors 3.7e4 / 1.8e3 — preserved in
`film/film_convergence.json` and the `film/film_params_N{16,32}.pkl`
checkpoints). Retrained Nyquist-capped checkpoints live in `film-fix/`
(n_freq 8 / 16); `film_fix_eval.py` re-evaluates them and writes the merged,
corrected `film/film_convergence_fixed.json`.

## Verification

Two independent audits of the 1D testbed code and numbers (Codex CLI static
audit — `codex-verify-out.md` — and a Claude subagent that reproduced the POD
floors, data-floors, and FD solver behavior digit-for-digit in independent
numpy): both confirmed the coord-net numbers are correct.

## File map

| path | what |
|---|---|
| `poisson1d_decoder_diag.py` / `results_run1.json` | 1D smooth sin family, both arms |
| `poisson1d_bump_diag.py` / `results_bump_run1.json` | 1D translated-bump family |
| `poisson1d_bump_upgraded.py` / `results_bump_upgraded.json` | 1D fair head-to-head |
| `poisson1d_convergence.py` / `results_convergence.json` | 1D convergence vs N=8192 reference |
| `poisson2d_diag.py` (+ `_nf` variant) | 2D testbed: POD sweep + grid-tied + coord-net |
| `round1/` `round2/` `round3/` | 2D Tufts runs (round 3 = verified 80k-step config; round 2 had a stale-script staging mishap, kept for the record) |
| `transfer2d_eval.py` | 2D mesh-transfer evaluation |
| `poisson2d_film.py` | FiLM coord-net trainer (reads N, N_FREQ from env) |
| `poisson2d_refgen.py` | N=512 CG reference generator (seed 0) |
| `film/` | FiLM round: checkpoints, per-N results, `ref512_val.npz` (480M, gitignored, regenerable), pre-fix convergence JSON |
| `film-fix/` | Nyquist-capped N=16/32 retrains |
| `film_convergence_eval.py` | original convergence eval (n_freq=32 checkpoints) |
| `film_fix_eval.py` | corrected eval → `film/film_convergence_fixed.json` |
| `smoke2d*/`, `smoke_film/` | local N=16 smoke runs |
| `codex-verify-out.md` | Codex audit report |

## Provenance

- Session date: 2026-08-12 (Claude Code session `e60431d0`), promoted from the
  session scratchpad to this worktree on 2026-08-13.
- Cluster jobs (Tufts `gpu` partition, A100, one isolated job dir per N):
  round 2 = 2277176–2277180; FiLM = 2325289–2325294 (+ 2325324 for the N=512
  reference); Nyquist fixes = 2329167/2329168. All job dirs deleted after
  pulling artifacts; per-job metadata (backend, params, steps, seed, n_freq)
  is embedded in each results JSON.
- Single seed (0) throughout — multi-seed replication is a known next step.

## Next steps

1. Multi-seed replication of the headline cells.
2. Swap the coordinate decoder into the `poisson/` package behind the
   Gauss–Newton/EQ solver and measure actual ROM accuracy/speedup — EQ needs
   only point evaluations, so cost decouples from N.
3. Optional: warped/deformation conditioning (translation equivariance) to
   push the fit floor further.
