# Burgers 1e-3 / 10x adaptive Phase-5 preregistration

Status: prospectively locked after the audited Phase-4 learned-manifold hard stop
and before any Phase-5 generator implementation or scientific job.  Phase 5 is a
finite test of one specific failure mechanism in the selected H1 representation.
It is not a new H1 spatial-resolution, patch, latent-dimension, loss-weight, weak-
objective, EQ, or kernel sweep.  All Phase-1--4 selection results and the rank
checkpoint are exposed development evidence.  Model-validation and seed-20261031
confirmation data remain untouched and are not authorized in Phase 5.

## Fixed hypothesis and non-adaptive bracket

The completed H1/k24 seed-11 hyperdecoder has the exact map

```
q in R^19 -> 32 -> 32 -> c in R^(48^2+32^2)=R^3328.
```

Whatever the upstream nonlinear map `h(q)`, its final layer has the form
`c(q)=b+W h(q)`.  Thus every generated coefficient vector lies in the single
affine set `b+range(W)`, whose dimension is at most 32.  This is an exact
architectural restriction, not an empirical rank estimate.  It is stricter than
the intended curved 19-dimensional coefficient manifold.

The reproducible read-only checkpoint
`b10_phase5_rank_diagnostic.py` binds the immutable P4-D JSON/NPZ/AUDIT/manifest,
checks all 5,712 exposed H1 coefficient fits, and applies a fixed randomized range
finder to their centered coefficient matrix.  Its estimated rank-32 relative
coefficient Frobenius residual is about 0.2522 and its estimated `s33/s1` is about
0.1304.  These are post-hoc coefficient-space diagnostics only.  Randomized SVD is
not a rigorous singular-value bound, the coefficient norm is not the field norm,
and neither value is a field-error lower bound or promotion gate.  Optimizer
failure may be a co-cause of Phase 4.

The Phase-5 bracket below was fixed prospectively from the exact affine-hull
restriction and the already-exposed checkpoint, before P5-D.  P5-D will recompute
and persist train-coefficient diagnostics, but **may not adapt the architectures,
channels, layers, losses, latent dimension, or cell cap from its coefficient SVD**.
No additional generator is authorized.

Every candidate remains a pure nonlinear manifold.  There is no POD basis,
parameter-dependent linear skip, independent coarse/fine state, full-grid FOM
correction/fallback, or learned parameter count that grows with deployment N.
Both H1 coefficient heads are generated from the same bounded `q in R^19`.  The
fixed train-only coefficient mean described below is the neural generator's output
bias, not a POD basis or a parameter-dependent bypass.

## Unchanged H1 decoder and online state

The spatial decoder is bit-for-bit the selected Phase-4 H1 representation:

- five bounded affine transport variables followed by 19 bounded nonlinear
  variables, so `k=24` and `q=19`;
- global R48 clamped cubic spline on aligned `[-4.5,4.5]^2`;
- fixed central P32 clamped cubic fine patch on aligned `[-2,2]^2`;
- the Phase-4 tensor-product C2 quintic window, combined as `S_c+w*S_f`;
- at most 16 global plus 16 fine coefficient contributions per query;
- affine transform first, exact K3 H200/f64 coefficient evaluation next, and the
  exact binary physical boundary mask last;
- `M=96`, `m=384`, deterministic test modes, exact FOM upwind in weak advection,
  decoder-output NNLS EQ, and hyperreduced cold start.

The direct state predictor remains exactly `7 -> 32 -> 32 -> 24`, Swish hidden
activations, all biases, and 2,104 parameters.  Its seven raw features and train-
only standardization/folded-first-layer identity are unchanged.  Predictor output
passes through `tanh`, so all 24 online state components are bounded.  No direct
features-to-coefficients arm is eligible; such an arm would be only a surrogate
control and is not included.

## Fixed nonlinear coefficient generators

Both heads use row-major tensor layout, nearest-neighbor replication for every
2x upsample, one-cell reflection padding before every 3x3 convolution, no batch
normalization or dropout, and a final linear 1x1 convolution.  Every convolution
has a bias.  The generator consumes `q=tanh(q_raw)` and produces normalized global
and fine residual grids, which are denormalized exactly as specified below.

### G1: DualUpConv16

Two affine maps of the same q produce a `6x6x16` global seed and a `4x4x16` fine
seed.  Each head applies three repetitions of

```
nearest-neighbor 2x upsample -> reflect-pad 1 -> 3x3 conv 16->16 -> Swish
```

and therefore reaches `48x48x16` and `32x32x16`.  Separate linear 1x1 convs
produce the R48 and P32 coefficient residuals.  Parameter count, including every
bias, is

```
(19+1)*(6*6*16 + 4*4*16)
+ 6*(3*3*16*16 + 16)
+ 2*(16+1)
= 30,594.
```

### G2: DualResUpConv32

Two affine maps of the same q produce a `6x6x32` global seed and a `4x4x32` fine
seed.  At each of three scales per head, set `u=nearest2x(x)` and apply

```
y = Swish(conv3x3_reflect(u))
x_next = Swish(u + conv3x3_reflect(y)).
```

Separate linear 1x1 convs produce the two coefficient residuals.  Parameter count
is

```
(19+1)*(6*6*32 + 4*4*32)
+ 12*(3*3*32*32 + 32)
+ 2*(32+1)
= 144,322.
```

G1 is the smallest candidate.  G2 is a prospectively fixed nonlinear-capacity
guard, not a post-result width sweep.  There is no G3 and no k32 retry.

## Exact offline encoder and warmup handoff

Each complete retraining has its own offline encoder, initialized independently
with the generator.  It consumes the two normalized free-coefficient grids.  The
global and fine encoder heads are separate until their final flattened features
are concatenated; both use the same channel count as their corresponding generator.

For G1, each head first applies a biased 1x1 conv `1->16`, then repeats three times

```
reflect-pad 1 -> 3x3 conv 16->16 -> Swish -> 2x2 average pool, stride 2.
```

The terminal `6x6x16` and `4x4x16` tensors are flattened, concatenated, and mapped
by one biased affine layer `832->19`.  For G2, the initial biased 1x1 conv is
`1->32`; each downsampling stage first applies the exact two-convolution residual
block

```
y = Swish(conv3x3_reflect(x))
z = Swish(x + conv3x3_reflect(y))
```

and then 2x2 average pooling with stride two.  Its terminal concatenation has 1,664
entries and its final biased affine map is `1664->19`.  Encoder parameter counts are
29,811 for G1 and 142,739 for G2.  These offline-only parameters are persisted and
timed but never counted as online inference.

The encoder predicts **raw** latent values by the exact output map

```
q_raw_encoder = 3*tanh(affine_output/3),
q = tanh(q_raw_encoder).
```

Thus encoder initialization is finite and avoids immediate saturation while the
deployed generator still consumes the unchanged bounded q.  The 10,000-update AE
warmup jointly optimizes only encoder and generator parameters.  At its end, the
encoder evaluates all 35,904 locked training snapshots once.  Those raw outputs are
copied exactly into an independent per-snapshot `q_raw` table.  Generator weights
are retained; a fresh joint-training AdamW optimizer state is initialized for the
generator and copied q_raw table.  The encoder is then frozen, never called by the
30,000-update joint phase, predictor, selection oracle, weak ROM, or online path,
and discarded from deployment.  Its weights, final outputs, optimizer state,
timing, and the bitwise handoff identity remain in the checkpoint.

## Train-only free-coefficient truth and normalization

P5-D regenerates the locked seed-0 training truth on the cluster and fits the exact
Phase-4 S2 H1 projection for all 35,904 training snapshots:

- N64 cases `0:512`, N128 `0:128`, N256 `0:64`;
- all 51 times for every case;
- the unchanged five-variable moment transport;
- the unchanged joint R48/P32 sparse design, ridge, COLAMD SuperLU, and disabled
  equilibration.

Every one of the 35,904 fits must have finite coefficient, prediction, RHS, and
diagnostic arrays; bitwise-zero physical boundary; bitwise-zero recorded POU error;
maximum attained support exactly 32; exact support indices/weights in the common
knot/nextafter basis probe; and independent relative normal residual

```
||A^T(Ac-u)+lambda*c||_2 / max(||A^T u||_2,1e-300) <= 1e-8.
```

The locked training reference must also have finite returned and independently
recomputed residual maxima `<=1e-8`.  One failed fit or reference check hard-stops
P5-D and all training.  P5-D persists chunked coefficient arrays, five affine
states, seven raw features, case/time/N/global-index metadata, all per-fit normal,
boundary/POU/support/finite records, solver work/timing, and checksums.  It may not
silently drop or refit a failed snapshot.

Let `S=35,904`, `n_c=48^2`, and `n_f=32^2`.  For head `h` and control `j`, the
normalization is computed from training coefficients only:

```
mu_h[j] = (1/S) * sum_s c_h[s,j]
r_h = sqrt((1/(S*n_h)) * sum_(s,j) (c_h[s,j]-mu_h[j])^2)
scale_h = max(r_h, 1e-12)
t_h[s,j] = (c_h[s,j]-mu_h[j]) / scale_h.
```

The unprotected `r_h` must be finite and positive; `mu_h`, `r_h`, `scale_h`, their
source ranges, and exact recomputation are persisted.  Online output is
`c_h(q)=mu_h+scale_h*G_h(q)`.  The 3,328-value mean buffer and two scales are stored
and their loads, multiplies, and additions are charged.  The mean is the network's
fixed output bias.  It depends on neither deployment parameters nor time and is not
a POD basis or linear correction path.

The immutable P4-D H1 selection coefficients at seed-0 N64 `512:576`, N128
`512:544`, and N256 `512:528`, all 51 times, remain the exposed free-oracle control.
They may be used for health/no-regression diagnostics and selection reporting, but
**never enter generator/encoder/predictor weight updates or train normalization**.

## Locked losses, optimizers, and schedules

All field losses use the fixed transported H1 decoder and snapshot-relative squared
L2 on 512 deterministic target-grid points drawn without replacement for each
sampled snapshot.  A coefficient loss uses the complete normalized control grids
and equal head weighting:

```
L_coeff = 0.5 * (mean((G_c(q)-t_c)^2) + mean((G_f(q)-t_f)^2))
L_field = mean_b ||D(a_b,G(q_b))-u_b||_2^2 / max(||u_b||_2^2,1e-300)
L_q = mean(q_raw^2).
```

The 10,000-update warmup loss is exactly

```
L_AE = L_field + L_coeff + 1e-6*L_q_encoder.
```

It uses batch 32, 512 points, AdamW beta1/beta2/eps `0.9/0.999/1e-8`, weight decay
`1e-6`, global gradient clip 1.0, and cosine learning rate `1e-3 -> 1e-4`.
For training seed `s`, its deterministic snapshot permutation and point-draw stream
uses seed `s+300000`.

After the exact handoff, the fresh 30,000-update joint phase uses

```
L_joint = L_field + 0.1*L_coeff + 1e-6*L_q,
```

with the same batch/point count, AdamW beta/eps/decay/clip, cosine `1e-3 -> 1e-5`,
and deterministic stream seed `s`.  Generator and every copied per-snapshot q_raw
are optimized; the encoder is frozen.  Warmup and joint histories persist step 1
and every 1,000th step through their exact terminal steps.  No coefficient, field,
regularization, schedule, optimizer, batch, point, or handoff weight may be revised.

Generator/encoder training initialization uses
`jax.random.fold_in(PRNGKey(s),arm_code)` with arm codes G1=1 and G2=2, followed
by a fixed parameter-index fold.  Every weight is f64 Xavier-normal with standard
deviation `sqrt(2/(fan_in+fan_out))` and every training bias starts at bitwise zero.
Every full seed is independent.  The direct predictor retains the Phase-4
Xavier-normal/zero-bias initializer, is freshly initialized, and is trained for
20,000 AdamW updates with the Phase-4 cosine
`5e-4 -> 5e-6`, stream `s+100000`, and loss

```
L_predictor = L_field + 0.1*mean((state_pred-state_target)^2),
```

where the fixed target is `[affine,tanh(final_q_raw)]`.  Its raw-feature
standardization uses the locked training mix only and is folded into the first
affine layer for deployment exactly as in Phase 4.

Training seeds are exactly 11, 29, and 47.  Each seed completely retrains encoder,
generator, all autolatents, and predictor from scratch.  Seeds may not share weights
or be averaged.

## Selection oracle and accuracy gates

For each completed seed, generator weights freeze and all 5,712 exposed-selection
q_raw values are optimized against fields, never coefficient targets.  The exact
three starts remain zero, `Normal(0,0.25^2)` from NumPy PCG64 seed 20260825, and
its sign reverse.  Each uses Adam beta1/beta2/eps `0.9/0.999/1e-8`, 10,000 updates,
batch 64, 512 seeded points, cosine `5e-2 -> 1e-3`, and
`1e-8*mean(q_raw^2)`.  Full-grid/all-time checkpoints occur at 4,000, 6,000,
8,000, and 10,000; an individual start may stop at the first complete passing
checkpoint.  The chosen start minimizes pooled, equally snapshot-weighted
`mean(snapshot_relative_L2^2)`.

At every N=64/128/256 and pooled, the learned oracle must have trajectory mean
`<=2e-4`, worst `<=7e-4`, all finite, and exact binary boundary.  The direct
predictor must have mean `<=3e-4`, worst `<=1e-3`, direct/oracle mean degradation
`<=1.5`, all finite, and exact boundary at every N and pooled.  Passing means all
gates pass simultaneously; no favorable case or pooled mean can rescue a failed N.

Selection is sequential and fixed.  G1 seed 11 runs first if P5-D licenses it.  If
G1 passes every learned-oracle and direct gate, it is the finalist and G2 is never
trained.  If G1 fails any gate, G2 seed 11 may run only if its own P5-D cost,
identity, memory, work, and non-collapse checks passed.  If G2 passes, it is the
finalist.  If G2 fails, Phase 5 hard-stops.  Failure magnitude does not open another
candidate.  The finalist is then completely retrained at seeds 29 and 47; all three
seeds must pass independently.  There is no seed averaging, k32 cell, loss revision,
optimizer retry, or third architecture.

## P5-D geometry, identity, and paired cost

P5-D uses deterministic nonconstant parameters so XLA cannot constant-fold the
generator.  For every dense/conv weight with fan-in/out, it draws f64 Normal values
with standard deviation `sqrt(2/(fan_in+fan_out))`; every bias is f64 Normal with
standard deviation `0.01/sqrt(max(fan_in,1))`.  The base PRNG is JAX seed 20260826,
folded by arm code and parameter index.  The direct predictor uses the same rule.
Representative q comes from the resulting predictor on all 51 raw-feature rows for
the four locked seed-20260822 N1024 cases.  P5-D requires and persists q standard
deviation `>=1e-3` and denormalized coefficient standard deviation `>=1e-3` over
every arm/case/time/control set; a collapse fails the arm.

The generated-output non-collapse probe uses 128 bounded q rows generated as
`tanh(Normal(0,0.5^2))` by NumPy PCG64 seed 20260827.  For their denormalized
coefficient matrix `C`, it computes exact f64 SVD after per-control centering and

```
rank_1e-10 = count(sigma_i/sigma_1 > 1e-10).
```

The 64 fixed consecutive pairs define

```
kappa_i = ||G((q_2i+q_(2i+1))/2)
             - (G(q_2i)+G(q_(2i+1)))/2||_2
          / max(||G(q_2i)-G(q_(2i+1))||_2,1e-12).
```

All values, singular ratios, and kappas are persisted.  An arm must have finite
outputs, `rank_1e-10>=33`, and median kappa `>=1e-8`; otherwise it is an
implementation/nonlinearity collapse and is killed.  These checks only establish
that the new generator escaped the fixed affine-rank-32 implementation.  They do
not substitute for any field-accuracy gate.

P5-D regenerates the Phase-4 cost cases: seed 20260822, draw-count 32, N1024 indices
`0:4`.  Cost/reference truth is the exact Phase-4 tight/tighter and live-FOM
protocol.  Both reference chains must be finite, have zero flags/breakdowns, meet
their returned tolerances, and differ by `<=1e-4`.  The live selected cubic/
exact-Helmholtz FOM uses outer/inner `3e-3/1e-1` and must be healthy with mean
`<=1e-3`, worst `<=3e-3` from the same invocation.

Five timed methods are FOM, G1 mandatory/max-one, and G2 mandatory/max-one.  Each
route charges raw-feature construction, hyperreduced cold recovery, direct state
prediction, train-normalization buffers, all coefficient generation, exactly 50
weak objective/rho evaluations, and final 51xN^2 K3 output.  Max-one additionally
charges one M-by-k Jacobian and every fixed trial at all 50 steps with corrected
previous-state chaining.  Coefficients may be batch-generated only where the actual
data dependency permits; every generator evaluation and recomputation is counted.

After lower/compile/accurate-first-use recording, common warmup, and burn, 20
repetitions per trajectory use the Phase-4 five-method cyclic/reversed schedule,
giving exactly four observations per clock position.  P5-D persists every timing,
order, within-trajectory outlier, per-trajectory median, setup, compiled memory,
work/rho/residual/Jacobian/trial/acceptance record, coefficient-evaluation count,
failure, and 10,000-resample trajectory-clustered interval.  K3 generated outputs,
current/previous stencils, weak residuals, rho, support/weights at knots and
nextafter neighbors, and exact boundaries must agree with the independent clamped
Cox path to relative/absolute tolerance `2e-14` as applicable.  Canonical work is
finite, exactly 50 weak evaluations, and zero failures.

An arm receives a training cost license only if mandatory compiled memory is
`<=20 GB`, paired median speedup is `>=10x`, clustered 95% lower bound is `>=8x`
against the live eligible FOM, and every identity/non-collapse/canonical-work gate
passes.  Max-one is always reported.  If max-one also passes, the arm is
correction-capable; otherwise it is only conditional zero/occasional-attempt.
Random-parameter P5-D timing is a structural screen, not a deployable speed claim.
The final trained actual thresholded rollout remains decisive.

Planning arithmetic, never a pass/fail clock: G1 uses 10,132,928 MAC per state and
516,779,328 MAC per 51-state trajectory (about 1.034 GFLOP at two FLOP/MAC), with
0.245 MB f64 parameters.  G2 uses 80,649,088 MAC per state and 4,113,103,488 MAC
per trajectory (about 8.226 GFLOP), with 1.155 MB parameters.  Batched-51 forward
features are about 22 MB / 44 MB; even two buffers and 24 forward tangents are
roughly below 1.1 GB / 2.1 GB before the already-measured K3 arrays.  Only measured
same-job timing and compiled peak memory can license an arm; prior H200 milliseconds
are context only.

## Exposed full weak/EQ/scaling cell and unchanged final gates

Only after one finalist passes independently at seeds 11, 29, and 47 may the one
final exposed-development cell run.  It uses the exposed seed-0 N256 `512:528`
cohort for reconstruction/full-weak/EQ checks, the unchanged train N256 `0:64`
decoder-output EQ pool with the 256 pairs fixed by seed 20260824, seed-20260821
draw-count 32 indices `0:8` for N256/N512 scaling and N1024 development, and all
three seed models.  NNLS EQ weights are refit for each N/M from decoder outputs and
exact-upwind images.  Base `M=96,m=384`; no next-M cell exists in Phase 5.

Every seed must pass exposed reconstruction mean `<=3e-4`, worst `<=1e-3`; full
weak mean `<=7e-4`; EQ mean `<=1e-3`; `EQ/full<=1.05`; zero failure/censoring;
finite residual/work/acceptance; and exact boundary.  N256/N512 scaling each requires
mean `<=1e-3`, worst `<=3e-3`, zero failure/censoring, and healthy references; speed
and clustered intervals are reported without a threshold.

The N1024 trained policy evaluates the exact weak objective at every step.
Zero attempt is allowed only for finite `rho<=1e-3`; otherwise exactly one
radius-0.25 LM-`1e-6` attempt with factors `[1,0.5,0.25,0]` is charged, and the
accepted or explicitly rejected state becomes the next previous state.  Accuracy,
speed, work, attempts, and failures come from the same invocation.  Every seed must
have mean `<=1e-3`, worst `<=3e-3`, zero failures/censoring, median same-GPU speedup
`>=10x`, and trajectory-clustered 95% lower bound `>=8x` against the fastest healthy
like-for-like FOM satisfying the same accuracy requirement.  Mandatory-only or
random-parameter P5-D timing can never support that claim.

Accepted N256/N512/N1024 truth uses the audited cubic/exact-Helmholtz outer/inner
`1e-12/1e-7` chain and independent `3e-13/3e-8` chain.  Both must be finite with
zero flags/breakdowns, meet returned tolerances, and differ by `<=1e-4`.

## Split locks and prohibition on model validation/confirmation

All Phase-2/4 splits remain exact:

- train: seed 0; N64 `0:512`, N128 `0:128`, N256 `0:64`, all 51 times;
- exposed selection: seed 0; N64 `512:576`, N128 `512:544`, N256 `512:528`,
  all 51 times;
- untouched model validation: seed 0; N64 `576:640`, N128 `576:608`, N256
  `576:592`;
- reserve: seed 0 indices starting at 640 as previously locked;
- development scaling/N1024: seed 20260821, draw-count 32, indices `0:8`;
- untouched confirmation: seed 20261031, draw-count 32, indices `0:8`.

Phase 5 may not generate, read, stage, or inspect model-validation or confirmation
fields.  Even a complete exposed-development pass is provisional and stops for a
new root audit and separate prospective authorization.  This preregistration does
not authorize a model-validation or confirmation cell.

## Finite cell cap and hard stops

Phase 5 authorizes at most six new exposed-development scientific cells.  Earlier
phases and the read-only rank checkpoint are bookkeeping, not Phase-5 cells.

1. P5-D: all 35,904 train S2 coefficient targets plus G1/G2 geometry, identity,
   mandatory/max-one, memory, and live-FOM cost: 1.
2. G1 complete seed-11 training: 1.
3. Conditional G2 complete seed-11 training, only if G1 fails and P5-D licensed
   G2: at most 1.
4. The fixed smallest passing finalist, complete seed-29 and seed-47 retrainings: 2.
5. One joint all-seed exposed full-weak/EQ/N256/N512/N1024 cell: 1.

The maximum is `1+1+1+2+1=6`; if G1 passes, G2 is skipped and the maximum is five.
Infrastructure-only zero-output attempts may be resubmitted once and remain recorded.
Local smokes and the rank checkpoint never count as scientific evidence.

Any P5-D train-fit/reference, identity, non-collapse, cost, memory, or work failure
kills the affected arm; failure of train target integrity hard-stops both.  Failure
of G1 licenses only the already-fixed G2.  Failure of G2, any finalist seed, any
weak/EQ/scaling gate, or the trained N1024 gate hard-stops Phase 5.  No k, channel,
layer, encoder, warmup, loss, optimizer, batch, point, R/P/window, M/m, EQ, weak-
objective, solver, tolerance, block-size, or fusion retry is authorized.  There is
no cherry-picking or averaging.

Every artifact stores commit/source hashes, immutable input hashes, seed/draw/index/
N/viscosity/time grid, complete architectures and parameter counts, coefficient
truth/normalization, all optimizer states/schedules/handoffs, GPU/job/backend,
f64/highest, reference health, per-case accuracy/work/residual/Jacobian/trial/
acceptance, all timing repetitions/orders/outliers, first-use/setup/memory, failures/
censoring, and clustered intervals.  Cluster staging, GPU preflight, one-job-per-
directory, checksummed pull, exact remote cleanup, generated report, and canonical
LAB-LOG updates follow `AGENTS.md`.  Phase 5 can claim success only after a later,
separately authorized untouched program proves every original supporting/headline
gate; this finite phase itself can report only provisional exposed evidence or an
audited negative result.
