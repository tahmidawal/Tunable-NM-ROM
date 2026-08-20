# Burgers 1e-3 / 10x Phase-7 G1 seed-11 training preregistration

Status: prospectively locked after the audited Phase-6 P6-D positive repair and
before any Phase-7 implementation, smoke, or scientific output. Phase 7 contains
exactly one scientific cell: a complete seed-11 training and exposed-selection
evaluation of the already-fixed G1 nonlinear coefficient generator. It is not an
architecture, latent, loss, optimizer, data, weak-objective, EQ, or kernel sweep.

## Immutable licenses and inputs

Phase 7 binds all of the following path-by-path through their local checksum and
staged root-manifest chains.

- P4-D H1 selection control, job 2668794: JSON/NPZ/AUDIT/root-manifest SHA-256
  `ff425dfa1f73ac2d8559df2d780ef09dc1ade179ed5636458e53e0f389f5d617`,
  `720c5890b22709c83858f46e43f18fa3d28bb1305544f02ad9aea71625a2228f`,
  `f41010b72ad9ddae43409a1c1d2073dc2839edea22e57a7d5bafc8e2359e08a3`,
  and `67a3bf8d0d8c755ec39cd4056a0bca0e852b4ba17493e8f052a0b288e63a201a`.
- P5-D target generation, job 2669249: JSON/NPZ/AUDIT/root-manifest SHA-256
  `97f8bc6bb9e1d67d0baf4652bd57e6fb69dab484fc8f99ce12018e9f6c1d0c96`,
  `5235b81b19c4ed459e7fda4291fe67eb3f4b87ba07413eb36e861a0b147dfe54`,
  `c84ee29e1b9fe84f5e90949e18be26c07a6c54c00320a2f7a82bd1cb8dee0eff`,
  and `6135791d3a5cca08b0ff1c424d93451579f3dd1048bef2a5cf314e2b8bf317d6`.
  Its 16 target filenames and individual hashes are read from the immutable JSON
  and must match the root/local manifests before any training update.
- P6-D actual-route repair, job 2669652: JSON/NPZ/AUDIT/root-manifest SHA-256
  `9fe2d49bbb0324fd08ef5da906c3afab0338a1f3bbfb6dfc1ef73f89603c139a`,
  `9f0372daba8c12e3aff86efde3201d3ae0612aba8cd297aa37559aa286967381`,
  `9e017b37709bf37fc8c8b87bbbb70461cba8afb47603a5b65dd2618c901ad2b4`,
  and `f8932a6a4304a14b93bfdf6783e900a47a9a1d45ba03f9915941bb00770bfb40`.
  Its immutable decision must be `repair_licensed=true`,
  `phase6_hard_stop=false`, and `training_authorized=false`; this document is the
  separate prospective authorization required by that decision.

All 35,904 P5 coefficient targets, their train-only 3,328-value mean and two
head scales, affine states, raw features, draw/time/N/global indices, health, and
normalization are immutable. No coefficient target is regenerated, dropped,
reweighted, or refit. The P4 exposed coefficients are diagnostic controls only
and never enter a weight update or train normalization.

Schema clarification locked after zero-output infrastructure job 2669861 and
before any retry: P5 target chunks store the physical five-value moment transport
returned by `fit_hierarchical_oracle`; the decoder/training state uses its already-
locked `normalized_state_from_affine` image. Both physical values and exact mapped
states are persisted and independently audited. The failed job compared these two
schemas directly before its first training update. This clarification changes no
field, coefficient, transport definition, training method, loss, gate, or cell cap.

Numerical comparison clarification locked after zero-output infrastructure job
2669975 and before any further stage or submission: the immutable P5 targets and
their hashes remain exact, and their physical affine values must still map
bitwise through `normalized_state_from_affine` to the states used by training.
Only the separately regenerated FOM-field moment check changes from bitwise
equality to the fixed finite componentwise rule

```
abs(regenerated_normalized_affine - immutable_normalized_affine) <= 2e-15
```

with no relative tolerance. The regenerated and immutable arrays must have the
same shape, and the maximum absolute difference is persisted and independently
audited. After this check, the regenerated state is replaced by the exact
immutable normalized state before any update, so the tolerance cannot alter a
training input. All feature rules are unchanged: columns 0/1/2/3/5/6 remain
bitwise and only the pre-existing viscosity-derived column-4 ULP rule applies.

This narrow repair is forced by same-node evidence. P5 job 2669249 and Phase-7
job 2669975 both ran on pax011/NVIDIA H200 with JAX 0.10.2 and identical hashes
for `b10_common.py`, `b10_spline.py`, and the inherited Burgers FOM. P5 formed
its target trajectories in batches of four whereas Phase 7 regenerated batches
of eight; the bitwise affine assertion failed before output or training. The
read-only GB10 reproduction against the immutable target finds the first N64
draw-0 normalized-affine difference at snapshot 0/component 3 (`3.1735e-18`),
maximum `4.4409e-16` over that trajectory and `8.8818e-16` over the first 64
trajectories. The fixed `2e-15` ceiling is the already-audited cross-execution
bound in the Phase-7 schema regression, not a data/model/loss/gate change.

## Fixed model

The spatial decoder remains Phase-4 H1: clamped cubic global R=48 plus the fixed
central P=32 fine patch and exact binary boundary. The state is k=24: five affine
transport variables and q=19 bounded nonlinear variables. M=96 and m=384 remain
fixed for later weak work, but no weak rollout is run in this training cell.

G1 is the unchanged 30,594-parameter `DualUpConv16` generator. Each q-to-grid
head maps q19 by a biased dense seed to 6x6x16 (coarse) or 4x4x16 (fine), then
performs three nearest-neighbor 2x upsampling stages, each followed by one
reflection-padded 3x3 16-to-16 convolution and Swish, and one final valid 1x1
16-to-1 convolution. Outputs are denormalized by the immutable train mean/scales.
There is no G2 fallback, dense bypass, POD path, linear correction, or full-grid
FOM correction.

The direct predictor remains `7 -> 32 -> 32 -> 24` with Swish hidden layers and
2,104 parameters. Its first five outputs represent the same decode-safe affine
state and its final 19 values represent q_raw before bounded `tanh` deployment.

## Training data and deterministic truth

The train mix is seed 0, all 51 times: N64 indices `0:512`, N128 `0:128`, and
N256 `0:64`. The cluster job regenerates only these locked inherited-reference
fields for deterministic pointwise field losses; it does not refit coefficients.
Each regenerated parameter, normalized vector, feature row, draw/time/N/global
index, and reference-health record must match the corresponding immutable P5
metadata, using the P5 audit's one-ULP allowance only for the known cross-version
viscosity/normalized-viscosity pair. Returned and independently recomputed
training-reference residual maxima must be finite and `<=1e-8`.

The exposed selection mix is seed 0, all 51 times: N64 `512:576`, N128
`512:544`, and N256 `512:528`. These fields are regenerated by the same locked
healthy reference solely for exposed development selection. Model-validation
indices beginning at 576 and confirmation seed 20261031 may not be generated,
read, staged, or inspected.

## Exact encoder warmup and handoff

Seed 11 initializes G1 and its offline mirrored encoder independently with the
Phase-5 arm-code/index-folded f64 Xavier-normal, bitwise-zero-bias rule. The
encoder has separate normalized coarse/fine coefficient heads. Each applies a
biased 1x1 `1->16` convolution, then repeats three times:

```
reflect-pad 1 -> 3x3 conv 16->16 -> Swish -> 2x2 average pool stride 2.
```

The terminal 6x6x16 and 4x4x16 arrays flatten and concatenate to 832 values,
followed by a biased `832->19` affine map. It has 29,811 parameters and returns
`q_raw_encoder=3*tanh(output/3)` and `q=tanh(q_raw_encoder)`.

The exact 10,000-update warmup jointly optimizes only encoder and G1. Batch size
is 32 snapshots with 512 deterministic target-grid points per snapshot. AdamW
uses beta1/beta2/eps `0.9/0.999/1e-8`, weight decay `1e-6`, global gradient clip
1.0, cosine learning rate `1e-3 -> 1e-4`, and stream seed `11+300000`:

```
L_coeff = 0.5*(mean((G_c(q)-t_c)^2)+mean((G_f(q)-t_f)^2))
L_field = mean_b ||D(a_b,G(q_b))-u_b||^2 / max(||u_b||^2,1e-300)
L_AE = L_field + L_coeff + 1e-6*mean(q_raw_encoder^2).
```

At step 10,000 the frozen encoder evaluates all 35,904 training snapshots once.
Its raw outputs are copied bitwise into an independent per-snapshot q_raw table.
G1 is retained and a fresh optimizer state is initialized for G1 plus that table.
Encoder weights, optimizer state, outputs, timing, and bitwise handoff identity are
persisted; the encoder is then frozen and excluded from every later/deployment path.

## Joint manifold and direct-predictor training

The exact 30,000-update joint phase optimizes G1 and every copied q_raw value with
the same batch/point count, AdamW beta/eps/decay/clip, deterministic stream seed 11,
cosine `1e-3 -> 1e-5`, and

```
L_joint = L_field + 0.1*L_coeff + 1e-6*mean(q_raw^2).
```

The direct predictor is freshly initialized by the unchanged Phase-4 rule and
trained for exactly 20,000 AdamW updates with stream seed `11+100000`, cosine
`5e-4 -> 5e-6`, and

```
L_predictor = L_field + 0.1*mean((state_pred-[affine,tanh(final_q_raw)])^2).
```

All seven predictor features use train-only mean/scale; the standardized first
layer is folded into raw-feature weights/bias, and train/selection identity between
standardized and folded evaluation must be `<=1e-12`. Warmup, joint, and predictor
histories persist step 1 and every 1,000 steps through exactly 10,000/30,000/20,000.
No schedule, initializer, loss weight, batch, point count, regularizer, optimizer,
handoff, early training stop, or retry may change.

## Exposed selection oracle and gates

After training, G1 freezes. All 5,712 exposed q_raw values are optimized against
fields, never coefficient targets, from the exact three starts: zero,
`Normal(0,0.25^2)` from NumPy PCG64 seed 20260825, and its sign reverse. Each uses
Adam beta1/beta2/eps `0.9/0.999/1e-8`, 10,000 updates, batch 64, 512 deterministic
points, cosine `5e-2 -> 1e-3`, and `1e-8*mean(q_raw^2)`. Full-grid/all-time
checkpoints occur at 4,000, 6,000, 8,000, and 10,000; a start may stop only at the
first complete passing checkpoint. The chosen start minimizes pooled equally
snapshot-weighted `mean(snapshot_relative_L2^2)`.

All of the following must pass simultaneously at N64, N128, N256, and pooled:

1. learned-oracle trajectory mean `<=2e-4`, worst `<=7e-4`;
2. direct-predictor trajectory mean `<=3e-4`, worst `<=1e-3`;
3. direct/oracle mean degradation `<=1.5`;
4. every output finite with exact binary boundary.

The job also reports every selection snapshot's K3/Cox full-field identity and
requires worst relative L2 `<=2e-14`, plus unchanged P6 basis/support bindings.
This is an implementation invariant, not a replacement for field accuracy. There
is no N1024 timing or rollout claim in Phase 7.

If every gate passes, G1 is the provisional finalist and Phase 7 licenses only a
separate prospectively audited seeds-29/47 proposal. If any gate fails, Phase 7
hard-stops. There is no G2 fallback, seed averaging, retry, loss revision, k/channel
change, extra architecture, or favorable-N rescue.

## Artifacts, audit, and finite cap

The single artifact persists model/checkpoint arrays, all optimizer states,
encoder outputs and handoff, all autolatents, folded and standardized predictor,
three selection starts/histories, complete metric/error arrays, exact schedule and
point-stream IDs, train/selection metadata/health, P4/P5/P6 bindings, source and
stage hashes, GPU/backend/f64/highest, setup/timing, and explicit false flags for
model-validation/confirmation access. Large target/field arrays are explicit JIT
arguments or host-streamed batches, never captured constants; any captured-large-
constant warning invalidates the run.

An independent negative-aware audit recomputes state, handoff, autolatent,
predictor-fold, schedule/seed, metadata, history/checkpoint, per-N/pooled metric,
identity, gate, and provenance consistency. It accepts an honestly recorded
negative decision rather than demanding promotion.

Phase 7 authorizes exactly one complete scientific cell. Jobs 2669861 and
2669975 both stopped before any output or training update and consume no
scientific cell. After the numerical rule above is implemented, independently
audited on real immutable data, and exact-staged for root review, one final
`p7_g1_s11_r3` submission may be authorized separately. No further
infrastructure retry is allowed. A complete valid cell consumes the cap. Local
smoke is synthetic-only, under one minute, and touches no locked scientific
field or coefficient artifact. No model-validation, weak/EQ, scaling, N1024
learned rollout, or confirmation cell is authorized here.
