# Burgers 1e-3 / 10x Phase-11 preregistration

Status: prospective only. Phase 11 permits exactly one seed-11 G2/q32 cell.
It does not permit a G1 retry, a third architecture, seeds 29/47, weak/EQ
fitting, scaling, model validation, confirmation, or any downstream rollout.

## Independent license and immutable evidence

Phase 10 did not license an architecture increase. Phase 11 creates a new,
independent prospective license from accepted exact-field evidence:

- the free H1 space passes, at pooled trajectory mean/worst
  `5.8054779105e-5 / 3.8589189381e-4`;
- accepted Phase-10 fixed-G1 globalization improves pooled full-train
  mean/worst from `1.3055157123939194e-2 / 4.84066602026945e-2` to
  `9.783142651706258e-3 / 3.46294172981746e-2`, but still misses the
  unchanged `2e-4 / 7e-4` gates by about 49 times in mean;
- G2/q32 is the sole previously defined nonlinear-capacity alternative. Its
  Phase-5 structural mandatory route was `15.057138x` with clustered interval
  `[11.837359,18.486060]` and compiled memory `555273272` bytes.

The Phase-9 capacity calculation is retracted, nonportable, unaccepted, and
forbidden as an input to any Phase-11 gate or decision. The Phase-5 G2 timing
is descriptive prior evidence only: G2 must pass a fresh live same-H200
P6-repaired structural preflight before update 1.

Work remains on branch `exp/2026-08-19-burgers-1e3-10x`, worktree
`worktrees/2026-08-19-burgers-1e3-10x`, prospectively based on accepted
Phase-10 closure commit
`5adcefc18c010cde28601cc62b278b4386cce3d9`. The accepted Phase-10 scientific
JSON/NPZ/work-checkpoint SHA-256 values are
`f0fab7583d1f1590efb58cd97d0128f34cc80cf4ab7ce8e3271066158f239523`,
`35d139fedcfff539aef25a335c556d47f2ec5e8581066a9e1f9b6ad8fcef4664`,
and `5fd879c32f2205d6427b272042c095624d824a8278e1eaf074c5295bcb574d68`.
The accepting Phase-10 audit and LOCAL SHA-256 values are
`695d5a29829764f3660ef606f6104dbb947761bc52d1f48cdf395969dc8983f4`
and `735b03b303b18618d835ed3c2154aa6f7fd5b5316707deee460702f934782a7e`.
The scientific and audit staged-manifest SHA-256 values are respectively
`60cd3e3f2d4946c8671f3cb43c3a852fd72a12b86735f3314bf7e78379edb45d`
and `44877a42a749b608372e29f13dd3f35c2d6a488aa2c824f563b01b08e506b456`.
The complete P4--P10 source/dependency chain must be exact-manifest-bound.

## Population and information boundary

The sole weight-training population is regenerated seed 0 at all 51 times:
N64 cases `0:512`, N128 `0:128`, and N256 `0:64`, totaling 35,904 snapshots.
The exposed selection population remains N64 `512:576`, N128 `512:544`, and
N256 `512:528`, totaling 5,712 snapshots. Model-validation indices beginning
at 576 and confirmation seed 20261031 remain inaccessible.

Only train target grids may update weights, optimizer states, normalization,
schedule, hyperparameters, or checkpoint choice. The 3,328-vector coefficient
mean, two head RMS output scales, and feature mean/scale are computed in fixed
global-index order from train only and persisted with hashes and work. Exposed
selection coefficients may initialize only the explicitly nondeployable
oracle after all train gates pass; they may never update weights or alter a
choice. Selection FOM fields are not generated or read before that gate.

## Fixed architecture and exact loss

The sole arm retains H1 R48+P32/k24, exact physical boundary, M=96, m=384, and
the P6 Cox-weak/K3-full route. It changes only G1/q19 to the already defined
DualResUpConv32 G2 with q32/state dimension 37. Independent parameter-tree
counts before update 1 must be exactly 165,954 generator, 164,384 mirrored
encoder, and 2,533 predictor parameters. There is no POD basis or linear
corrector.

For every training snapshot the primary loss is the exact full-grid Cox value

```
ell = ||u_theta-u_FOM||_2^2 / max(||u_FOM||_2^2,1e-300).
```

All grid points enter each loss. Immutable coefficient grids are encoder input
and may contribute only the fixed head-balanced auxiliary

```
L_coeff = 0.5*mean(((G_c-c_c)/rms_c)^2)
        + 0.5*mean(((G_f-c_f)/rms_f)^2),
L_encoder/joint = mean(ell) + 0.01*L_coeff + 1e-6*mean(q_raw^2).
```

The auxiliary cannot select a checkpoint, epoch, schedule, or gate. Random
point, spline-Gram, and raw coefficient-Euclidean primary losses are forbidden.
K3 is identity/timing only. Predictor loss is
`mean(ell)+0.1*mean((state_pred-state_target)^2)`.

## Fresh same-H200 structural preflight

Before update 1, deterministic nonconstant G2 weights execute the actual
q32/k37 P6 Cox-weak/K3-full route against a fresh live accuracy-eligible FOM in
the same H200 allocation. Four fixed cost trajectories use 24 repetitions each
with exact AB/BA balance and a GPU burn before every timed block. Each timed
record stores same-invocation accuracy, health, work, and cost. Raw repetitions,
per-trajectory medians, burns, position counts, and outlier counts are retained.

The preflight requires K3/Cox identity `<=2e-14`, exact boundary/support,
exactly 50 Cox weak and 51 K3 coefficient-grid/full-field evaluations, zero
unintended weak Jacobian/trial work, zero failures, compiled device memory
`<=20 GiB`, paired median speedup `>=10`, and trajectory-clustered 95% lower
bound `>=8` against the live eligible FOM. Any failure stops before update 1
and makes no training or learned-speed claim.

## Complete deterministic training schedule

Updates are resolution-homogeneous batches of size 8/2/1 in the exact repeating
cycle `N64,N128,N256`. Within each N, a NumPy PCG64 permutation covers every
snapshot once without replacement. Each N supplies 3,264 batches, so each
complete epoch has 9,792 updates. Zero-based permutation seeds are
`311011+e` for encoder epochs, `11+e` for joint epochs, and `100011+e` for
predictor epochs.

AdamW uses beta1/beta2/epsilon `0.9/0.999/1e-8`, weight decay `1e-6`, global
gradient clip 1, f64 Xavier initialization, and zero biases. The fixed schedule
is:

1. Encoder+G2: 18 epochs. Epochs 1--9 use cosine `1e-3 -> 1e-4` and epochs
   10--18 use a fixed fine continuation `1e-4 -> 1e-5`.
2. Copy terminal encoder q bitwise into independent train `q_raw` state.
3. Joint G2+q: 54 epochs. Epochs 1--27 use cosine `1e-3 -> 1e-5` and epochs
   28--54 use a fixed fine continuation `1e-5 -> 1e-6`.
4. Only after the globalized train gate passes, freeze G2 and train the
   predictor for 18 epochs with cosine `5e-4 -> 5e-6`, using the accepted
   globalized train q states as targets.

Generator-bearing training is 72 epochs and 705,024 updates, exactly twice the
Phase-9 G1 exposure. The conditional predictor adds 176,256 updates, so the
maximum is 90 epochs and 881,280 updates. This keeps the old G1 recipe as an
exact prefix and adds only a fixed low-learning-rate convergence continuation.
The accepted Phase-10 result places the floor in the learned generator after
q globalization, while supplying no evidence that more predictor exposure is
needed; hence predictor exposure is unchanged and conditional.

Frozen weights receive a complete full-train Cox evaluation at every epoch
end. All epoch arrays, actual resolution-update order, permutations, optimizer
states, q handoffs, normalization, and atomic work checkpoints are retained.
Only fixed terminal epochs 18/54/18 may decide a gate. No early stopping,
favorable checkpoint choice, or resume is permitted.

## Terminal control and final-q globalization

After joint epoch 54, report the complete terminal joint/autolatent train
control. Then freeze G2 and optimize q directly from that final q only for all
35,904 train snapshots. There is no encoder, predictor, free-target, zero,
random, or restart start.

Each row receives at most 40 matrix-free full-grid Cox GN/LM attempts. CG solves
`(J^T J+lambda I)p=-J^T r` from zero for at most 32 iterations at relative
tolerance `1e-12`. The box is `[-1,1]^32`; `Delta0=0.25` with bounds
`[2^-20,1]`; `lambda0=1e-6` with bounds `[1e-12,1e12]`. A step is accepted
only with finite work, no CG breakdown, positive predicted and actual decrease,
and defined `rho=actual/predicted >=1e-4`. Nonpositive predicted decrease stores
finite `rho=0`, `rho_defined=false`, and rejects without reading rho.

Rejection or defined `rho<0.25` divides Delta by four and multiplies lambda by
ten. Accepted `rho>0.75` divides lambda by three and doubles Delta only when
the projected step norm is at least `0.9*Delta`; other accepted steps retain
both. Accepted relative objective or relative step change `<=1e-12` terminates
that row. Minimum-radius/maximum-damping exhaustion, breakdown, nonfinite work,
false rho/decision, or any transition/work-count mismatch is unhealthy.
Attempts 0--40 persist q/objective/Delta/lambda/active, trial/predicted/actual,
rho/rho-defined, acceptance, termination, gradient, step, CG, JVP/VJP, bound,
exhaustion, and elapsed work.

For case c, the locked trajectory metric is

```
E_c = sqrt(sum_t ||u_theta[c,t]-u[c,t]||_2^2
           / max(sum_t ||u[c,t]||_2^2,1e-300)).
```

Per-N mean/worst are arithmetic mean/maximum over cases; pooled mean/worst are
arithmetic mean/maximum over the concatenated case arrays. The gate-controlling
train result is the final globalized-q full-train result: at every N and pooled,
mean/worst must be `<=2e-4/7e-4`, all values finite, boundary violations zero,
K3/Cox identity `<=2e-14`, and FOM/provenance/trust health true. The terminal
pre-globalization result is a mandatory control, not a second accuracy gate;
this prospectively separates G2 image quality from the q-optimization defect
accepted Phase 10 measured. Any globalized train miss closes G2 negatively.

## Conditional predictor and exposed selection

Only after the globalized train pass may the predictor phase run. Its complete
train direct route must then pass mean/worst `<=3e-4/1e-3` at every N and
pooled, with finite fields, zero boundary violations, and identity `<=2e-14`,
before any selection target is accessed.

Only after both train gates pass, selection evaluates the deployable direct
predictor and the fixed nondeployable two-start oracle. Oracle starts are
predictor-q and free-target-encoder-q only and use the same q32 40-attempt
protocol. At every N and pooled, oracle mean/worst must be `<=2e-4/7e-4`,
direct mean/worst `<=3e-4/1e-3`, and direct/oracle mean ratio `<=1.5`. Both
also require finite fields, zero boundary violations, and identity `<=2e-14`.

A selection pass establishes only an accurate G2 candidate. The structural
preflight is not a final learned-rollout speed claim. No later stage opens
without a separate prospective authorization.

## Runtime projection and cap

The one cell requests exactly one H200, 8 CPUs, 96 GiB host memory, and 16
hours with GPU backend, f64, and highest matmul precision. Before update 1,
after actual data-generation, compilation, and structural-preflight elapsed
time, a no-update throughput preflight measures materialized and host-transferred
encoder, joint, full-cohort Cox/K3, q32 trust, and conditional predictor/
selection units at every N with three warmups and ten retained timings.

The named maximum projection includes 705,024 pre-gate updates; 72 complete
35,904-snapshot epoch evaluations; 35,904x40 train trust attempts; conditional
176,256 predictor updates and 18 complete train evaluations; 5,712 direct
selection fields and 2x5,712x40 selection attempts; terminal identity,
compression, checkpoint, and independent-audit overhead. Require
`1.15*projected_remaining_seconds <= actual Slurm seconds remaining` at the
end of preflight. Failure stops before update 1 with weights and optimizer
states bitwise unchanged. Update 1 consumes the sole arm. No result, failure,
or timeout self-authorizes retry.

## Independent audit and outcomes

The independent negative-aware auditor binds exact P4--P10 artifacts, source
hashes, staged manifest, job, GPU/f64/highest, and explicitly rejects any
retracted capacity input. It independently regenerates train data, features,
and normalization; gated selection is regenerated only after the recorded
train passes. It recomputes parameter counts, exact split/order and hashes,
actual update-resolution telemetry, permutations, all phase handoffs, optimizer
trees, update counts, terminal checkpoint equality, epoch and terminal metrics,
globalized-q trust transitions/work, structural timing/health/identity, FOM
eligibility, information-boundary counters, and every Boolean decision.

Corruption controls must reject provenance, normalization, update order/count,
state handoff, terminal checkpoint, trust rho/transition/work, preflight
projection, structural timing/identity/work, premature selection access, and
false positive/negative gate decisions. Exact checksummed pull and remote
cleanup occur only after audit PASS. Negative outcomes are binding.

## Prospective r1 infrastructure closure and sole retry

Job `2751097` (`p11_g2_s11_r1`) ran the exact commit
`cf6dee04f96110a0e8f7fb9be334f216aac0f4b1` and 69-file manifest
`a14887b2828cde86c5489a1e5d6cf5437c4ea51ed32d6dd453944b11ad32584f`
on pax008/H200/GPU, f64/highest.  It exited `1:0` after 3m46s while writing the
terminal JSON, before independent audit.  The partial report records
`structural_preflight_pass=false` and `updates_started=false`; the NPZ has only
normalization arrays, the checkpoint has no optimizer states and an empty
globalized-q array, and no PROGRESS file exists.  Therefore the arm consumed
zero optimizer updates.  The structural numeric panel was not durably written,
the runtime projection was not executed, and no r1 value or gate is accepted
as scientific evidence.

The exact infrastructure cause is a shared empty dictionary on the pre-update
branch (`permutations`, `arrays`, and `history` aliased), followed by the lack
of recursive NumPy JSON normalization.  The first unserializable path was
`history.coefficient_mean`.  One sole prospective retry
`p11_g2_s11_r2` may use distinct empty dictionaries and generic recursive
normalization of NumPy arrays/scalars only.  It must reproduce the unchanged
architecture, initialization, data, fresh structural panel, runtime kill,
schedule, loss, trust protocol, gates, H200 request, and information boundary.
Regression tests must prove the containers are independent, pre-update history
stays empty, nested arrays/scalars serialize, and nonfinite JSON remains
rejected.  No retry is authorized until its exact commit and manifest pass a
new root audit; r1 never self-authorizes any other arm or downstream work.

## Final disposition and timing retraction

The sole retry job `2754129` completed on pax008/H200 and passed its exact
independent audit under the implemented schema, stopping before runtime
projection/update 1 with bitwise-unchanged weights.  A later source-backed
call-graph reconciliation found that its timed `route` returned both the 51
K3 full fields and a separately computed 51-state full-grid Cox control.
Nevertheless the canonical work and independent auditor recorded only 50 Cox
weak plus 51 K3 full evaluations; they inferred those counts from the residual
and K3 output shapes and never accounted for the Cox control call/output.
Phase 6 had correctly kept that full Cox identity/control outside the timed
actual route.

The reported `0.03205502685159445 s`, `3.047233624534141x`, and clustered
interval `[2.396706675073385, 3.7521239409805687]` are therefore **retracted as
an overcharged, canonical-work-invalid timing negative**, not accepted evidence
that the intended G2 Cox-weak/K3-full route misses the speed gate.  The extra
Cox output is exactly `51*1024^2*8 = 427819008` bytes; the compiled footprint
was `1206594776` bytes versus Phase-5 G2 mandatory `555273272` bytes.  Changing
q19/state24 to q32/state37 adds 21,632 generator and 429 predictor parameters
but leaves the same 32-channel, three-block, depth-two residual convolutional
heads, so this architectural delta does not remove the work-accounting defect.

The immutable bundle remains valid evidence that the FOM, boundary, identity,
and numerical route were healthy and that zero optimizer updates occurred.
It establishes no corrected-route speed, learned G2, predictor, selection,
rollout, third-architecture, or downstream result and licenses none
retrospectively.
