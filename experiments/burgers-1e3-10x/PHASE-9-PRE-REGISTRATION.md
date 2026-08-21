# Burgers 1e-3 / 10x Phase-9 preregistration

Status: prospective only. Phase 9 permits at most one exact-full-field G1/q19
training cell and one conditional G2/q32 capacity cell. No Phase-9
implementation, smoke, stage, or job exists.

## Independent license and provenance

Phase 9 is independently licensed by the valid Phase-4 through Phase-7 evidence:
H1 free representation passes, the P5 target chain is healthy, the P6
Cox-weak/K3-full route passes, and P7 shows a learned G1 training failure. The
immutable P8-D result is descriptive evidence that free H1 still passes while
exact latent globalization bottoms far above the target. P8-D remains invalid
with `health=false`, `p8_d_valid=false`, and every P8 T1/T2 promotion false.
Phase 9 does not promote, repair, or reinterpret Phase 8.

Work remains on branch `exp/2026-08-19-burgers-1e3-10x` in worktree
`worktrees/2026-08-19-burgers-1e3-10x`, prospectively based on result/report
commit `03981815e9f7c7b5c4b76f8d16e9e77712100dc8`. The locked JSON/AUDIT SHA-256
pairs are:

- P4: `ff425dfa1f73ac2d8559df2d780ef09dc1ade179ed5636458e53e0f389f5d617` /
  `f41010b72ad9ddae43409a1c1d2073dc2839edea22e57a7d5bafc8e2359e08a3`;
- P5: `97f8bc6bb9e1d67d0baf4652bd57e6fb69dab484fc8f99ce12018e9f6c1d0c96` /
  `c84ee29e1b9fe84f5e90949e18be26c07a6c54c00320a2f7a82bd1cb8dee0eff`;
- P6: `9fe2d49bbb0324fd08ef5da906c3afab0338a1f3bbfb6dfc1ef73f89603c139a` /
  `9e017b37709bf37fc8c8b87bbbb70461cba8afb47603a5b65dd2618c901ad2b4`;
- P7: `a59e92640aae787003d5753d4614de1b6fe90c68fdba7f2daa81a125a3c0956c` /
  `35c95ee38dea7622f40b6c199f4164b6c27ec3f37ad5561a51de6cd3b0a79322`;
- descriptive P8: `be0ca15e5c36f45a1f9a8fdfe86b79a592b530c069ae6d25c1f573c59c974f4f` /
  `d1e9e7bd238ac9e9320afc018fa5ff85afcd51d6ff714a61eb47c3b32900de17`.

P7 NPZ/checkpoint SHA-256 are respectively
`fd40d339c0746b48c07408d5d017dff8819595350c365f7af5fc0d40673946ae`
and `113101637ef2b4fb75fe2ca0ba0dabe5a90c35d03a5c563b765438385c62db8c`.

## Fixed populations and information boundary

The only weight-training population is regenerated seed 0 at all 51 times:
N64 cases `0:512`, N128 `0:128`, and N256 `0:64`, totaling 35,904 snapshots.
The exposed selection population is N64 `512:576`, N128 `512:544`, and N256
`512:528`, totaling 5,712 snapshots. Model-validation indices beginning at 576
and confirmation seed 20261031 remain inaccessible.

Weights, optimizer states, normalization, hyperparameters, epoch counts,
schedule, checkpoint choice, and the G2 capacity license may use only training
FOM fields and immutable training coefficient grids. Exposed selection target
coefficients may initialize only the explicitly nondeployable terminal oracle;
they may not update weights or change any choice. Selection FOM fields are used
only for the fixed terminal exposed gates after a train pass.

The 3,328-vector coefficient mean and the two centered per-head RMS output
scales are computed in fixed global-index order from the 35,904 training target
grids only. Predictor-feature mean and scale are likewise train-only. These
arrays, definitions, source indices, SHA-256 values, computation elapsed time,
and host/device bytes are persisted. Generator de-normalization is inside every
Cox/K3 field decode and its work is charged; predictor standardization is folded
exactly into the saved predictor and the fold identity is audited. Selection
data never contributes to a normalization statistic.

## Exact field loss and reported trajectory metric

For snapshot truth `u` and Cox-decoded field `u_theta`, the sole primary loss is

```
ell = ||u_theta-u||_2^2 / max(||u||_2^2, 1e-300).
```

Every N-by-N point is included. Batch loss is the arithmetic mean of `ell` over
the resolution-homogeneous batch. Cox--de Boor evaluation produces all
scientific training and accuracy fields. K3/Pallas is only an independently
reported identity and structural-timing route. Random-point, spline-Gram, and
raw coefficient-Euclidean primary losses are forbidden.

For case `c`, the reported 51-state trajectory relative L2 is

```
E_c = sqrt(sum_t ||u_theta[c,t]-u[c,t]||_2^2
           / max(sum_t ||u[c,t]||_2^2, 1e-300)).
```

Per-N mean/worst are the arithmetic mean and maximum of `E_c` over cases at
that N. Pooled mean/worst are the arithmetic mean and maximum over the
concatenated per-case `E_c` arrays from N64, N128, and N256. This exactly
matches the prior report aggregation; no snapshot-weighted or resolution-equal
alternative may select a gate.

## Resources, cap, and infrastructure-only preflight

The fixed GPU type is NVIDIA H200. Each cell requests one H200, 8 CPUs, 96 GiB
host memory, and 16 hours with GPU backend, f64, and highest matmul precision.
Changing to H100 or A100-80GB requires a prospective amendment and root audit
before staging. No absolute timing is compared across jobs.

The scientific cap is `1 P9-T1 + conditional 1 P9-T2 = 2` cells, at most 32
allocated GPU-hours. Before update 1, each arm runs a no-update same-job
throughput preflight. For encoder, joint, predictor, full-cohort evaluation,
and terminal trust work at each N, it compiles with explicit array arguments,
runs three warmups and ten measured no-update repetitions, and uses the median
per fixed unit of work. The deterministic projection sums the exact locked
update/evaluation/trust counts below. If `1.15 * projected_terminal_seconds`
exceeds the remaining allocation, the cell stops before update 1 and records an
infrastructure-only failure. Weights and optimizer state must remain bitwise
unchanged. Update 1 consumes that arm; a timeout, failure, or negative result
cannot self-authorize resumption or retry.

## P9-T1: exact-full-field G1/q19 retraining

T1 retains H1 R48+P32/k24, G1 DualUpConv16, q19/state dimension 24, seed 11,
M=96, m=384, exact boundary, and the P6 Cox-weak/K3-full route. Independent
parameter-tree audits must equal 30,594 generator, 29,811 mirrored encoder, and
2,104 predictor parameters before update 1.

Immutable training coefficients are encoder inputs, not the primary target. The
only coefficient auxiliary is fixed and head-balanced:

```
L_coeff = 0.5*mean(((G_c-c_c)/rms_c)^2)
        + 0.5*mean(((G_f-c_f)/rms_f)^2),
L_encoder/joint = mean(ell) + 0.01*L_coeff + 1e-6*mean(q_raw^2).
```

It cannot select a model, epoch, schedule, or gate. Predictor loss is
`mean(ell)+0.1*mean((state_pred-state_final)^2)` and has no coefficient term.

Training uses complete deterministic epochs. Updates contain one resolution,
cycling exactly `N64,N128,N256`; batch sizes are `8,2,1`. Independent NumPy
PCG64 permutations without replacement cover every snapshot once within each N,
using seeds `311011+epoch`, `11+epoch`, and `100011+epoch` for encoder, joint,
and predictor phases. Each N contributes 3,264 batches, so one epoch is 9,792
updates. Encoder/G1 runs 9 epochs; terminal encoder q is copied bitwise into
independent autolatents; joint G1/autolatent training runs 27 epochs; predictor
training runs 18 epochs. The total is 54 epochs and 528,768 updates.

AdamW beta1/beta2/epsilon are `0.9/0.999/1e-8`, weight decay `1e-6`, global
gradient clip 1, and initialization f64 Xavier with zero biases. Cosine learning
rates are encoder `1e-3 -> 1e-4`, joint `1e-3 -> 1e-5`, and predictor
`5e-4 -> 5e-6`. At every epoch end, frozen weights are evaluated on the complete
35,904-snapshot train cohort with exact full grids. All epoch-end arrays and
metrics are persisted. Terminal encoder/joint/predictor epochs 9/27/18 are
binding; there is no early stopping or favorable checkpoint selection.

## Terminal train and exposed gates

The terminal full-train autolatent must pass at every N and pooled: trajectory
mean/worst `<=2e-4/7e-4`, all values finite, zero boundary violations, and
K3/Cox identity `<=2e-14`. All immutable bindings and FOM residual health must
also pass. A train miss never opens selection for model choice.

Only after a train pass, selection is evaluated with the direct predictor and a
fixed two-start bounded-q oracle. The direct predictor is deployable. The second
start, encoded from exposed free-target coefficients, is nondeployable
oracle-only. Oracle mean/worst must be `<=2e-4/7e-4`; direct mean/worst must be
`<=3e-4/1e-3`; direct/oracle mean ratio must be `<=1.5`, at every N and pooled.
Both require finite fields, zero boundary violations, and identity `<=2e-14`.

The oracle optimizes q directly in `[-1,1]^q` from predictor-q and
free-target-encoder-q only. It uses matrix-free full-grid Cox GN with at most
40 attempts, CG maximum q iterations and tolerance `1e-12`, `Delta0=0.25` in
`[2^-20,1]`, `lambda0=1e-6` in `[1e-12,1e12]`, and acceptance only for finite
positive actual/predicted decrease with `rho>=1e-4`. Rejection or `rho<0.25`
divides Delta by 4 and multiplies lambda by 10; accepted `rho>0.75` divides
lambda by 3 and doubles Delta only when `||s||>=0.9 Delta`; other accepted
steps retain both. Bounds apply after every update.

Every attempted row stores finite objective, trial, predicted, and actual work.
When predicted decrease is positive, it stores finite
`rho=actual/predicted` and `rho_defined=true`. When predicted decrease is
nonpositive, it stores finite `rho=0`, `rho_defined=false`, rejects the step,
and no acceptance/update branch may read rho. NaN/Inf work, a defined-rho
mismatch, false decision, CG breakdown, or unhealthy minimum-radius/
maximum-damping exhaustion fails health. This prospective schema does not
change P8's immutable failed health gate.

A complete T1 train and selection pass closes Phase 9 positively. It does not
license a rollout-speed claim or downstream weak/EQ work.

## Exact conditional G2 capacity license

G2 is licensed only if the healthy terminal G1 train result misses its train
gate at every N and pooled and all conditions below pass. Let
`L_e=mean(ell)` over all 35,904 terminal train snapshots at joint epoch `e`.
The locked late improvement is

```
I_24_27 = (L_24-L_27) / max(L_24,1e-300),
```

and must satisfy `0 <= I_24_27 <= 0.01`.

On every training snapshot at terminal q, define
`r=(u_theta-u)/sqrt(max(||u||_2^2,1e-300))`, `f=0.5||r||^2`,
`g=J^T r`, and the projected-gradient mapping

```
G(q) = q-clip(q-g,-1,1),
gamma = ||G(q)||_2 / max(1,||q||_2).
```

For tangent capacity, solve from zero by matrix-free CG
`(J^T J+1e-12 I)delta=-J^T r`, maximum 38 iterations and relative tolerance
`1e-12`, then define `eta=||r+J delta||_2/max(||r||_2,1e-300)` and

```
eta_E = sqrt(sum ||r+J delta||_2^2
             / max(sum ||r||_2^2,1e-300)).
```

These are computed for the full 35,904-snapshot cohort in fixed global-index
order, not a subset; there is no random seed. Quantiles use NumPy
`quantile(method="linear")`. At every N and pooled, gamma median must be
`<=1e-4` and p95 `<=1e-3`; eta median must be `>=0.9`, p10 `>=0.8`, and eta_E
`>=0.8`. At every N and pooled, at most 1% of q components may satisfy
`min(q+1,1-q)<=1e-6`.

The independent auditor recomputes `L_24`, `L_27`, `I_24_27`, every gamma/eta,
quantile, eta_E, and bound fraction from persisted immutable arrays and requires
every clause by Boolean conjunction. Any failure, any partial-N pass, a train
pass followed by a selection miss, or any subjective alternative classification
hard-stops Phase 9 without G2.

## P9-T2: sole nonlinear capacity escalation, if licensed

T2 changes only G1/q19 to DualResUpConv32 with residual generator and q32/state
dimension 37. It jointly raises nonlinear width/depth and q capacity; no claim
will attribute an effect to either change separately. Independent parameter-tree
audits must equal 165,954 generator, 164,384 mirrored encoder, and 2,533
predictor parameters before update 1. Every other population, normalization,
seed, loss, auxiliary, epoch, permutation, optimizer, M/m, metric, and gate is
identical to T1.

Before update 1, deterministic nonconstant T2 weights execute the actual repaired
Cox-weak/K3-full q32/k37 structural route in the same H200 job as a fresh live
eligible FOM. After GPU burn, both methods run 24 repetitions per each of four
fixed cost trajectories with exactly balanced AB/BA positions. Raw repetitions,
per-trajectory medians, outlier counts, burns, and work are persisted. The route
must pass identity `<=2e-14`, exact boundary/support, exactly 50 Cox weak and 51
K3 coefficient-grid/full-field evaluations, zero unintended weak Jacobian or
trial work, zero failures, compiled device memory `<=20 GiB`, paired median
speedup `>=10`, and trajectory-clustered 95% lower bound `>=8` against the live
eligible FOM. Failure stops before update 1.

This is a structural feasibility preflight only. It is not final learned NM-ROM
speed. Any later speed claim requires a separately preregistered corrected
rollout with accuracy and cost from the same invocation.

## Audit, outcomes, and hard exclusions

Each completed cell requires exact source/dependency manifests, isolated
namespace, queue/disk checks, GPU/f64/highest preflight, finite and persisted
epoch/work arrays, negative-aware independent audit, checksum pull, and exact
remote cleanup. Negative outcomes are binding.

Seeds 29/47, a third architecture or q, POD or any linear corrector, weak/EQ,
deployment scaling/timing, model validation, and confirmation remain sealed.
Even a positive Phase-9 result requires a later prospective authorization for
those stages.
