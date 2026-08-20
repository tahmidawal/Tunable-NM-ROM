# Burgers 1e-3 / 10x Phase-8 preregistration

Status: prospective only. No Phase-8 implementation, data generation, smoke, or
cluster job exists. Phase 8 remains pure nonlinear (no POD or linear correction)
and is capped at one diagnostic cell, one G1/q19 training cell, and one
conditionally licensed G2/q32 training cell.

## Provenance, exclusions, and fixed data

Work remains on branch `exp/2026-08-19-burgers-1e3-10x` in worktree
`worktrees/2026-08-19-burgers-1e3-10x`, prospectively based on Phase-7 closure
commit `b52ba3705dda55885b57248cfc19fc2b72894336`. P8-D binds the immutable r3
JSON/NPZ/checkpoint SHA-256 values `a59e9264...`, `fd40d339c...`, and
`11310163...`, plus its passing audit `35c95ee3...` and P5/P6 dependency chain.

The only development populations are regenerated seed-0, all 51 times:
N64 cases `0:512`, N128 `0:128`, and N256 `0:64` for training (35,904
snapshots), and N64 `512:576`, N128 `512:544`, and N256 `512:528` for exposed
selection (5,712 snapshots). Model-validation indices beginning at 576 and
confirmation seed 20261031 remain inaccessible. Seeds 29/47, weak/EQ, scaling,
model validation, and confirmation are outside Phase 8.

A local all-selection trust-region attempt exceeded the repository's one-minute
local-smoke rule and was terminated. Its partial state/output is excluded and
must not be read, copied, cited, or reused. All scientific Phase-8 work is
cluster-only on one requested H200, 8 CPUs, 96 GiB, 16 h per licensed cell, with
GPU backend, f64, and highest matmul precision. Local work after implementation
may only be a synthetic smoke under one minute and is never evidence.

## P8-D: one representation/inference diagnostic

P8-D regenerates the locked FOM truth and independently verifies all immutable
P5/P6/r3 bindings. For snapshot `u` and bounded latent `q`, its sole optimization
objective is the exact discrete full-grid quantity

```
ell(q) = ||D_H1(a,q;G1) - u_FOM||_2^2 / max(||u_FOM||_2^2, 1e-300).
```

Every one of the N-by-N points is used. There is no random-point, coefficient,
or spline-Gram optimization alternative. Decoding each immutable free-H1
coefficient target on the same full grid and comparing it with FOM truth is a
separately labelled representation control, never a latent objective.

P8-D reports exact full-grid metrics for: free-H1 representation; r3 train
encoder handoff and final autolatents; exposed free-target-encoder q; exposed
predictor q; the deployed predictor field; the locked r3 oracle; and the two
trust-region recoveries below. It reports snapshot, trajectory, per-N, and pooled
mean/worst values without inspecting model-validation or confirmation data.

### Fixed bounded-q trust-region Gauss--Newton

The variable is `q` directly, constrained componentwise to `[-1,1]^19`; q-raw
is not optimized. The only starts are the r3 direct-predictor q and the r3
free-target-encoder q. There is no zero/random start or restart. With normalized
residual `r`, matrix-free JVP/VJP products apply the exact full-grid decoder
Jacobian `J=d r/dq`; no dense full-grid Jacobian is formed.

At each attempt solve `(J^T J + lambda I)p=-J^T r` by matrix-free CG, at most 19
iterations and relative residual tolerance `1e-12`. Scale `p` to radius `Delta`,
then use the projected trial `q_trial=clip(q+p,-1,1)` and actual step
`s=q_trial-q`. Predicted decrease is `-g^T s-0.5||Js||^2`; actual decrease is
`0.5(||r||^2-||r_trial||^2)`. A step is accepted only when all values are finite,
predicted and actual decreases are positive, and `rho=actual/predicted >=1e-4`.

The fixed initialization/bounds are `Delta=0.25`, `Delta in [2^-20,1]`,
`lambda=1e-6`, and `lambda in [1e-12,1e12]`. Updates after every attempt are:

- reject or `rho<0.25`: `Delta=max(Delta/4,2^-20)` and
  `lambda=min(10*lambda,1e12)`;
- accept with `rho>0.75` and `||s||>=0.9*Delta`:
  `Delta=min(2*Delta,1)` and `lambda=max(lambda/3,1e-12)`;
- accept with `rho>0.75` but an interior step: retain `Delta` and set
  `lambda=max(lambda/3,1e-12)`;
- otherwise retain both.

Each start has at most 40 total attempts. It may terminate after an accepted
step only if relative actual objective decrease is `<=1e-12` or
`||s||/(1+||q||)<=1e-12`. Nonfinite work, CG breakdown, or exhaustion at both
the minimum radius and maximum damping without an accepted step is unhealthy.
Every attempt persists objective, gradient norm, radius, damping, CG count and
residual, JVP/VJP counts, trial objective, predicted/actual decrease, rho,
acceptance, step norm, bound activity, and termination. Exact full-grid
per-N/pooled work is persisted at attempts 0, 10, 20, 30, 40 and termination.
The lower terminal objective of the two preregistered starts is reported as the
globalized two-start control; both starts remain separately visible.

P8-D is valid only with GPU/f64/highest, healthy regenerated FOM truth, finite
arrays/objectives, exact binary boundary, immutable hashes, K3/Cox field identity
`<=2e-14`, and a fully reproducible work trace. It is diagnostic only and cannot
promote an r3 result. A valid P8-D licenses T1; an unhealthy P8-D hard-stops.

## T1: corrected G1/q19 training

T1 retains H1, G1/q19, the P6 Cox-weak/K3-full route, M=96, m=384, exact boundary,
seed 11, and the existing feature/affine definitions. Immutable free coefficient
grids are inputs to the offline mirrored encoder. They are not the primary target.
For every sampled snapshot the primary loss is the exact full-grid FOM loss
`ell(q)` above. Random point sampling and raw coefficient Euclidean training are
forbidden.

The only coefficient auxiliary is fixed at weight `0.01` and head-balanced:

```
L_coeff = 0.5*mean(((G_c-c_c)/rms_c)^2)
        + 0.5*mean(((G_f-c_f)/rms_f)^2),
L_AE/joint = mean(ell) + 0.01*L_coeff + 1e-6*mean(q_raw^2).
```

`rms_c,rms_f` are immutable train-only P5 head RMS values. This auxiliary may
not select a checkpoint, model, schedule, or gate. The direct predictor uses
`mean(ell)+0.1*mean((state_pred-state_final)^2)` and the existing folded-feature
identity; it has no coefficient loss.

Training is in complete deterministic epochs. Each update contains only one
resolution, cycling exactly `N64,N128,N256`; batch sizes are respectively
`8,2,1`. At each epoch, independent NumPy PCG64 permutations without replacement
cover every snapshot once within each N. Seeds are `311011+epoch` for encoder,
`11+epoch` for joint, and `100011+epoch` for predictor. Each resolution therefore
has exactly 3,264 batches and one complete epoch has 9,792 updates. No partial
epoch is allowed.

The encoder/G1 phase runs 9 epochs (9 visits/snapshot), matching/exceeding the
Phase-7 warmup's 8.91 visits. Its terminal encoder q values are copied bitwise
into independent autolatents. Joint G1/autolatent training runs 27 epochs (27
visits/snapshot), exceeding Phase 7's 26.74. Direct-predictor training runs 18
epochs (18 visits/snapshot), exceeding Phase 7's 17.82. AdamW beta1/beta2/eps,
weight decay, clip, initialization, and cosine endpoints remain respectively
`0.9/0.999/1e-8`, `1e-6`, `1`, f64 Xavier/zero bias, and Phase-7 values:
encoder `1e-3 -> 1e-4`, joint `1e-3 -> 1e-5`, predictor `5e-4 -> 5e-6`.

At every epoch end, frozen weights are evaluated on the complete 35,904-snapshot
training cohort with exact full grids, and the complete metrics/losses are
persisted. Exposed exact-full-grid checkpoints occur only at encoder epoch 9,
joint epoch 27, and predictor epoch 18 and cannot alter training. There is no
early stopping or favorable checkpoint selection: terminal joint epoch 27 and
predictor epoch 18 are binding.

At the terminal joint checkpoint the selection oracle runs the exact P8-D
two-start, 40-attempt protocol with the new predictor and new free-target encoder;
for each snapshot its lower terminal objective is the fixed oracle. The direct
field is the terminal predictor with no latent optimization. At every N and
pooled, the unchanged gates are oracle trajectory mean/worst `<=2e-4/7e-4`,
direct mean/worst `<=3e-4/1e-3`, direct/oracle mean ratio `<=1.5`, finite values,
exact boundary, and K3/Cox identity `<=2e-14`. The same oracle thresholds are
also explicit terminal full-train gates. Passing T1 stops Phase 8 successfully;
T2 is not run.

## Prospective T2 capacity license

After a healthy T1 miss, T2 is licensed only when the terminal G1/q19 full-train
failure is demonstrably stationary and tangent-capacity limited. On all training
snapshots, using exact full grids and q directly, compute the projected gradient
and the unconstrained matrix-free tangent least-squares residual. The latter uses
CG on `(J^T J+1e-12 I)delta=-J^T r`, at most 38 iterations, tolerance `1e-12`, and
defines `eta=||r+J delta||/||r||`.

All of these prospective tests must pass at every N and pooled:

1. terminal full-train oracle misses `2e-4/7e-4`, while every health/identity gate passes;
2. pooled exact-full-grid loss improves by at most 1% from joint epoch 24 to 27;
3. normalized projected-gradient stationarity has median `<=1e-4` and p95 `<=1e-3`;
4. `eta` has median `>=0.9`, p10 `>=0.8`, and residual-energy-weighted ratio `>=0.8`;
5. at most 1% of q components lie within `1e-6` of either bound.

Failure of any item classifies the miss as optimization/inference/saturation rather
than proven capacity and hard-stops Phase 8 without T2. A train pass plus an exposed
selection miss is likewise an inference failure and does not license T2.

## T2: exactly G2/q32, if licensed

T2 changes only G1/q19 to the fixed Phase-5 DualResUpConv32 generator and mirrored
encoder with q32; state dimension is exactly 37 (five affine plus q32). There is no
other width, q, seed, loss, or schedule arm. The generator, encoder, and direct
predictor parameter counts must statically audit as 165,954, 164,384, and 2,533.
T2 otherwise repeats T1's exact epochs, batches, full-grid loss/checkpoints,
two-start inference, and scientific gates.

Before its first weight update, the T2 cell must execute the actual repaired
Cox-weak/K3-full q32/k37 route with deterministic nonconstant G2/predictor weights
and a live same-H200 eligible FOM. It must pass exact boundary/support, field-route
identity `<=2e-14`, exact 50 weak and 51 coefficient-grid evaluations, zero
Jacobian/trial work in the mandatory structural route, zero failures, compiled
memory `<=20 GiB`, paired median speedup `>=10`, and trajectory-clustered 95% lower
bound `>=8`; raw repetition arrays and outlier counts are persisted. Failure stops
pre-update and closes Phase 8.

## Cap and audit

The absolute scientific cap is `1 P8-D + 1 T1 + conditional 1 T2 = 3` cluster
cells. A valid completed cell consumes its slot; there is no automatic retry or
post-result revision. Every result requires independent negative-aware audit,
exact source/stage hashes, isolated directory, queue/disk preflight, persisted
health/work, checksum pull, and exact remote cleanup. No Phase-8 result licenses
weak/EQ, N256/512 deployment scaling, N1024 timing, model validation, confirmation,
or seeds 29/47; any such work requires a later prospective authorization.
