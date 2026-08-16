# Generalizable cascade NM-ROM (Poisson-2D + Burgers-2D) — 2026-08-16

**Status: IMPLEMENTATION READY FOR REVIEW. No full run has been launched.**
Smoke artifacts under `smoke/` (N=16, a few hundred steps) prove only that
every code path executes and produces finite numbers — the numbers there are
meaningless by design.

## Question

Can we merge the two lines of prior work —

- June's residual **cascade autoencoder** (`ae-research/old-exp/exp-residual-cascade`,
  keeper on the smooth 1D family, dead-end on transport; f32, k=24 bottleneck
  per stage, no frequency scaling), whose encoder makes it **generalizable** to
  unseen instances, and
- Wang & Lai **multi-stage precision** as validated in
  `exp/2026-08-14-multistage-precision` (f64, frequency-scaled stages, no
  bottleneck; 3e-11 on one function, but the per-stage gain collapses with
  family size — representational, in the parameter direction)

— into a decoder that is as accurate as the family allows **and** still
generalizes, and then use it inside the NM-ROM (GN/LM latent solve on the
discrete residual) on held-out Poisson-2D and Burgers-2D instances?

## Method (cn_common.py)

| piece | what | from |
|---|---|---|
| encoder `E` | MLP on the PDE INPUT sampled on a fixed 16x16 lattice (Poisson: source `f`; Burgers: initial condition `u0`) -> `z` (K_LAT) | new; the ROM legitimately knows this input |
| stage 0 | `E` + FiLM coord decoder `D0(x; z)`, joint, f64, inverse-energy weights | FiLM/fit machinery reused from `ms_parametric.py` |
| freeze | `E` (and all latents) frozen after stage 0 -> fixed target per stage | Wang-Lai requirement; auto-decoder lesson |
| stages `k>=1` | fresh f64 FiLM residual decoders `D_k(x; z)`, NO bottleneck, target `u - sum_{j<k} eps_j D_j`, `eps_k` = residual RMS, `n_freq = ceil(2 f_d)+4` (half-cycle idx) from the radial-MEAN spectrum probe, capped `N-1` | `ms_parametric.fit_stage`, `freq_schedule`, `dominant_radial_freq` (unit-tested) |
| gate | before each stage: June POD probe of the residual (effective rank, energy-top-r, POD-r rec. err) + whitened nearest-neighbour-in-z correlation; `GATE=1` stops stacking when `eff_rank > GATE_EFFRANK` or `nn5_corr < GATE_NNCORR` (defaults 64 / 0.2 — to be calibrated on the control) | June `cascade_pod_probe.compressibility`, `ms_diag.nn_corr` |
| lever (a) | `K_EXTRA`: each residual stage gets `c_k = E_k(input)` (small fresh encoder, trained with the stage) as extra conditioning; the ROM solves `z` only, `c_k` are known at query | new |
| lever (b) | `LAT_SMOOTH`: encoder-Lipschitz proxy `||z_i-z_j||^2 / ||inp_i-inp_j||^2` on batch pairs during stage 0 | new (proxy for "leave a smoother-in-z residual") |
| Burgers time | decoder has NO time input; rows = (traj, time); `z_{i,0} = E(u0_i)`, `z_{i,n>=1}` free auto-decoder latents (lazy per-row Adam) + `T_SMOOTH ||z_{i,n}-z_{i,n-1}||^2` | multistage F5 fix |
| ROM Poisson | LM (`ms_autodecoder.lm_solve`: tolerance, NaN guard, accounting) on the ghost-zero-Dirichlet 5-pt FD residual (interior rows; = FOM operator), init `z=E(f)`; collocation full + `M_EQ` subset; objectives `fd` (plain), `hinv` (residual preconditioned by the exact discrete inverse Laplacian via CG), `hinvK` (K CG iterations) | ms_autodecoder + new objective arms |
| ROM Burgers | latent time stepping: per step LM on the FOM's backward-Euler residual (interior rows, decoded walls zeroed) w.r.t. `z_{n+1}`, warm start `z_n`, `nu` known to the ROM (not a decoder input); rollout error vs the FOM trajectory | burgers2d testbed FOM residual (guarded Newton) |
| floors | per number of stages S: **encoded plug-in** (`z=E(inp)`, no solve — the free generalizable answer), **finite-budget LM-inferred latent** on the data misfit (init `E(inp)` and mean; labelled inferred, NOT a lower bound), **ROM solve**; plus `||r||` at the LM solution vs at the encoded latent, boundary block, LM accounting, nearest-training-latent distance | multistage F1/F3 fixes |
| control | `TRUE_Z=1`: identity encoder = the true 4 parameters (Poisson) | multistage true-z arm |

Note on `hinv`: for the LINEAR Poisson operator, `L^{-1}(L u(z) - f) = u(z) - u_FOM`,
so the exact-`hinv` objective IS the field misfit to the FOM solution — it is
an OBJECTIVE DIAGNOSTIC (what a perfect residual norm would buy) at FOM-solve
cost, not a cheaper ROM.  `hinvK` (fixed CG iterations) is the cheap
approximation.  The smoke already shows the expected identity: `hinv` ROM ≈
inferred-latent error.

## Files

- `cn_common.py` — lattice input, encoder, cascade apply, probes, stage-0 joint fit
  (encoder + free latents), residual-stage fit (+K_EXTRA), imports of the
  multistage machinery.
- `cn_poisson.py` — full Poisson pipeline (stages, gate, held-out inference, ROM
  with 3 objectives, provenance/`complete` flag, deployable pkl).
- `cn_burgers.py` — Burgers pilot pipeline (auto-decoder latents in time, stages,
  latent time-stepping ROM).
- `cn_summarize.py` — markdown tables from report JSONs (rejects incomplete).
- `smoke/` — tiny end-to-end artifacts.

## Planned runs (NOT launched; each = one isolated Tufts A100 job dir)

Common: `JAX_DEFAULT_MATMUL_PRECISION=highest`, `jax_backend=gpu` preflight,
cluster venv, f64.

Controls first (cheap):
```
# family-size ladder for the CASCADE gain with a frozen ENCODER (Poisson)
for NT in 8 32 128 512: K_LAT=8 N=64 N_TRAIN=$NT N_VAL=64 N_STAGES=3 STEPS=20000 P_SUB=1024 BATCH=32 N_TEST=16 GN_ITERS=60 OBJECTIVES=fd python cn_poisson.py runs/ladder_nt$NT
# true-z control (same code path)
TRUE_Z=1 N=64 N_TRAIN=512 N_VAL=64 N_STAGES=3 STEPS=20000 P_SUB=1024 python cn_poisson.py runs/truez
# budget arm
K_LAT=8 N=64 N_TRAIN=512 N_VAL=64 N_STAGES=3 STEPS=40000 CONST_LR=1 P_SUB=1024 python cn_poisson.py runs/budget2x
```
Full Poisson arms (parallel):
```
K_LAT=8  ... N_STAGES=4 OBJECTIVES=fd,hinv,hinvK python cn_poisson.py runs/enc_K8
K_LAT=16 ... N_STAGES=4 python cn_poisson.py runs/enc_K16
K_LAT=8 K_EXTRA=4 ... python cn_poisson.py runs/enc_K8_X4          # lever (a)
K_LAT=8 LAT_SMOOTH=0.1 ... python cn_poisson.py runs/enc_K8_S0.1   # lever (b)
K_LAT=8 GATE=1 ... python cn_poisson.py runs/enc_K8_gated
```
Burgers pilot (one or two arms):
```
K_LAT=8 N=32 N_TRAIN=128 N_VAL=16 N_STAGES=3 STEPS=30000 P_SUB=512 BATCH=32 N_TEST=8 GN_ITERS=30 python cn_burgers.py runs/burgers_K8
K_LAT=16 ... python cn_burgers.py runs/burgers_K16
```

## Reviewer scrutiny points

1. Encoder input = the source / IC on a fixed lattice via the analytic family
   formula (`gaussian_on_lattice`) — legitimate for the ROM (it knows the
   input), but note it bakes the family's functional form into the encoder
   input; a general implementation would sample the given field. Any leak of
   the SOLUTION into the encoder path? (Intended: none.)
2. Stage-0 joint fit: encoder + decoder + (Burgers) free latents with lazy Adam
   and temporal smoothness (`fit_stage0`) — check the masking of encoded vs free
   rows, `stop_gradient` on the previous-row latent, and that encoded rows are
   re-filled with `E(inp)` after training (they are; free-latent rows keep the
   Adam values).
3. Frequency schedule / eps / weights reused verbatim from the reviewed
   multistage code — but for BURGERS the probe subsamples rows (`e_tr[::7]`) for
   speed; is that biased?
4. Gate thresholds are uncalibrated defaults; the runs report `stop_suggested`
   always and only stop with `GATE=1`.
5. Lever (a): `c_k = E_k(inp)` per stage — is this a fair "extra latent" or a
   backdoor to more capacity that should be matched in the baseline (a K+extra
   single-stage arm)?  Lever (b): the pair penalty is a crude Lipschitz proxy.
6. ROM Poisson: `hinv` = FOM-cost objective diagnostic (documented); `hinvK`
   with `tol=0.0, maxiter=K` — verify jax CG honours the fixed iteration count.
   Held-out `f_rows` for the `fd` objective are built from the analytic source
   at collocation nodes; `hinv*` use `mp.source_interior` (same formula).
7. ROM Burgers: pure rollout (u_prev = decoded previous latent); interior-rows
   residual with decoded walls zeroed; `nu` used from the family (ROM knows the
   physics parameter). Inferred-latent reference uses the TRUE snapshot (data
   misfit) — clearly not a ROM number; check the labelling downstream.
8. Budget/size: Poisson N=64 STEPS=20000/stage; Burgers pilot N=32,
   128 traj x 51 = 6528 rows — is 30k steps of batch 32 enough for the
   auto-decoder latents (each row visited ~150 times)?
9. Metrics: train = both metrics; held-out plug-in / inferred / ROM = mean
   per-sample rel-L2 (+ global for plug-in). Quartile breakdown for plug-in.
10. Provenance: config manifest in JSON + pkl, `complete` flag, summarizer
    rejects incomplete; deployable pkl contains encoder + lattice stats +
    stages; nothing derived from held-out fields is stored.
