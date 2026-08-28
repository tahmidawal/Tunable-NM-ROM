# OPTIM-NOTES — 1D Burgers ROM rollout speed study (2026-08-28)

Running log of the GPU-performance work on the `sep_b1d_scale.py` on-device ROM
rollout.  New code: `sep_b1d_fast.py` (driver + harness) and
`b1d_fast_common.py` (loaders, verbatim reference copies, optimized
implementations).  Nothing existing was edited.  All timings below are on the
**local GB10** (shared unified-memory box, slower than the A100 the committed
baselines were measured on) — relative gains are the signal; absolute A100
numbers come from a cluster confirmation run.

## Ground rules applied

- f64 everywhere (`jax_enable_x64`), `JAX_DEFAULT_MATMUL_PRECISION=highest`,
  every GPU run through `jaxrun`, one at a time.
- Hard error parity: optimized path must reproduce the committed
  `runs/b1dqf/b1ds_n512/out/sep_b1d_scale_n512.json` /
  `..._n4096.json` `rollout_err_mean` per arm within 0.2% relative, matching
  latent paths and stop-reason distributions.  Iteration caps / tolerance
  loosening are forbidden except as clearly-labeled ALGORITHMIC arms with
  parity proof.
- Arms are rebuilt bit-identically from the committed artifacts
  (`CKPT_CACHE` .pkl checkpoint, `_nodes.npz` node sets); test data
  regenerated from `TEST_SEED=1` with the same tridiagonal truth generator.

## Committed A100 baseline (from the b1ds JSONs, medians per 50-step traj)

| N | arm | err | ic ms | roll ms | dec ms | e2e ms |
|---|-----|-----|-------|---------|--------|--------|
| 512 | base_tight | 5.194389e-03 | 16.85 | 34.30 | 0.47 | 56.03 |
| 512 | nodes_tight | 4.886140e-03 | 16.90 | 35.94 | 0.47 | 56.76 |
| 4096 | base_tight | 4.888627e-03 | 15.39 | 32.71 | 0.48 | 47.18 |
| 4096 | nodes_tight | 4.529491e-03 | 15.50 | 35.32 | 0.49 | 50.36 |

FOM baseline (tridiag tolerance-Newton): 7.9–8.8 ms (ntol 1e-3),
14.0–15.3 ms (1e-8).  Everything flat in N — launch-bound, as in the 2D
"Round 4 speed" profiling (~0.14 ms per LM iteration there, ~70x above the
bandwidth bound).

mean_njac 3.3–6.3 per step; stop reasons overwhelmingly `2` (stall rule),
~1–5 per 400 steps reason `0` (budget exhausted without stall).

## Planned candidates

a. Unrolled Cholesky for the (8,8) normal-equation solve (kills cuSOLVER
   custom-call + syncs).  `OPT_CHOL`.
b. One-pass residual+Jacobian via `jax.linearize` (J^T carried directly), and
   body restructure: compute (r_new, J_new) unconditionally instead of
   `rn_fn` + `lax.cond` re-evaluation (identical values, removes a dispatch
   boundary and a duplicated primal eval).  `OPT_ONEPASS`.
c. Hoist per-trajectory constants out of the whole rollout: `wt` and
   `DT*nu*lam` are constant for a trajectory (nu fixed) but the reference
   recomputes them in EVERY residual call (≈ 2x per LM iteration).  Hoisted
   values computed with the identical expression → bitwise identical.
   `OPT_HOIST`.
d. XLA command-buffer (CUDA graph) flags for the launch-bound sequential
   program; bitwise parity check after enabling.
e. IC fit: same chol/onepass/no-cond treatment; separately, an ALGORITHMIC
   reduced-init arm if the icdiag study shows all 9 inits reach the same
   optimum (`IC_ARM=init2`, parity proven separately).
f. THROUGHPUT arm: vmap the rollout over the 8 test trajectories (amortized
   cost only; vmapped while_loop pays worst-case iterations — reported
   honestly, never as a latency claim).

## Experiment log

(appended chronologically)

### 2026-08-28 — Diagnosis (MODE=diag, N=512, GB10, base_tight arm)

Scratch: `opt/diag_n512.{log,json}`.

- **Host dispatch dominates single calls**: one jitted call of *anything tiny*
  costs 0.4–1.4 ms wall on the GB10 (r_w eval 0.41 ms, r+J 0.69 ms, 8x8 LU
  solve 0.92 ms, head MLP 0.09 ms) — these micro-numbers are dispatch, not
  compute.
- **Marginal in-loop LM iteration cost** (fixed-T unconditional LM loop,
  diagnosis-only): 0.79 ms @T=1 → 19.84 ms @T=64, slope **~0.26 ms per LM
  iteration** inside the compiled while loop (A100 2D measurement was
  0.14 ms — consistent, GB10 ~2x slower per launch).  A production rollout is
  ~350–400 LM iterations → ~90–100 ms on GB10 (observed: 66 rollouts / 6 s).
- **HLO census (compiled rollout)**: ref = 79 fusions + 19 custom calls
  (cuSOLVER LU among them) + 1 conditional + 2 whiles (scan + LM);
  fast (chol+onepass+hoist) = 66 fusions + 12 custom calls + **0
  conditionals**.  Per LM-loop-body kernel count is the target.
- **SM utilization** while looping the ref rollout: mean **86%**, p90 89%
  (123 samples @50 ms).  Caveat: on this counter "utilization" = fraction of
  time ANY kernel is resident; with dozens of ~µs kernels back-to-back it
  reads high while doing ~0.1 MFLOP/iteration — the 2D bandwidth-bound
  analysis (~70x above roofline) still stands.  The program is
  kernel-count/dispatch bound, not occupancy bound.
- chol-vs-LU solution diff on a representative normal-equation system:
  1.6e-14 (f64) — numerically interchangeable.

### 2026-08-28 — A/B #1: chol+onepass+hoist, no XLA flags (N=512, GB10)

Scratch: `opt/ab_n512_allopt.{log,json}`.  TIME_REPS=7, interleaved.

| arm | err ref | err fast | latdev | ic ref→fast | roll ref→fast |
|-----|---------|----------|--------|-------------|----------------|
| base_tight | 5.194389e-03 | 5.194389e-03 (rel 2.6e-09) | 6.5e-09 | 40.8→51.1 ms | 65.2→88.2 ms |
| nodes_tight | 4.886140e-03 | 4.886140e-03 (rel 2.3e-09) | 6.4e-09 | 40.2→50.5 ms | 67.0→91.1 ms |

- **PARITY PASSES perfectly** (ref vs committed A100 JSON: 3.6e-09 rel; fast
  vs ref: 2.6e-09; max latent path deviation 6.5e-09; identical stop-reason
  pattern) — the chol/onepass/hoist transforms are numerically faithful.
- **REJECTED as a speedup: the combination is ~30–35% SLOWER.**  Interpretation:
  per-while-iteration cost is dominated by fixed loop/dispatch overhead, not
  by the cuSOLVER call or a duplicated primal — and the restructured body
  (unconditional r+J eval; ~200-op scalar Cholesky chain) *adds* kernels/work
  where the reference's `lax.cond` skipped them on rejects.  Flag isolation
  postponed; first test the orthogonal lever (command buffers), then re-test
  each flag under whichever XLA config wins.

### 2026-08-28 — XLA command buffers (CUDA graphs): REJECTED

Scratch: `opt/ab_n512_cmdbuf.log`, `opt/loop_anatomy.py` output.

`--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUDNN,CUSTOM_CALL,`
`CONDITIONAL,WHILE --xla_gpu_graph_min_graph_size=1` (WHILE/CONDITIONAL types
exist in this jax 0.10.1 CUDA-13 plugin):

- On the real rollout: **no effect** (roll_ref 66.5 ms vs 65.2 ms unflagged).
- On loop microbenchmarks: **1.5–2x SLOWER** per iteration (scalar-add fori
  5.1→13.1 µs/it, tiny-matmul while 8.3→12.2 µs/it).  Graph
  capture/update overhead exceeds any launch savings at these sizes on GB10.

### 2026-08-28 — Loop-overhead anatomy (the real diagnosis)

`opt/loop_anatomy.py` (throwaway, medians of 30):

| body (fori/while, T=1000) | µs/iteration |
|---|---|
| scalar add | 5.1 |
| two tiny matmuls (8-dim) | 7.6 (fori) / 8.3 (while) |
| LM-ish body with `jnp.linalg.solve` (8x8) | **117.9** |
| same body, solve replaced by diag division | 9.1 |

**The XLA while/fori loop itself is CHEAP (~5–10 µs/it); the (8,8)
`jnp.linalg.solve` cuSOLVER custom-call pair costs ~110 µs per in-loop
iteration**

### 2026-08-28 — Flag isolation (N=512, base_tight, GB10)

Scratch: `opt/ab_n512_{chol,cholhoist,cho_op,cho_nc}.log`.  All parity-clean
(err rel diff ≤ 3e-9, latdev ≤ 7.2e-9).

| config | roll fast (ref ~64) | ic fast (ref ~40) | verdict |
|---|---|---|---|
| scalar-chol only | 93.6 ms | 61.0 ms | scalar-unrolled chol −30 ms WORSE |
| + hoist | 93.0 ms | 60.9 ms | hoist: neutral (keep — free) |
| + onepass | 93.0 ms | 60.9 ms | onepass: neutral |
| + nocond | 87.8 ms | 50.4 ms | nocond: −5 ms roll, −10 ms ic (GOOD) |

The scalar-unrolled Cholesky (`solve_spd8`) is REJECTED: ~200 scalar HLO ops
compile into slow serial fusions (+90 µs/iter vs cuSOLVER).  `nocond`
(unconditionally evaluating r+J instead of `lax.cond`) is KEPT: with one-pass
rJ the kernel count per iteration goes DOWN vs r-test + conditional re-eval.

### 2026-08-28 — Solve alternatives shootout (`opt/solve_anatomy.out`)

In-loop LM-ish body, 300 iterations, µs/iteration (single / vmapped over 9):

| solver | single | vmap9 | acc vs LU |
|---|---|---|---|
| `jnp.linalg.solve` (cuSOLVER LU) | 119.5 | 99.0 | — |
| scalar-unrolled Cholesky | (slow, see above) | — | 1.2e-16 |
| Cholesky w/ broadcast columns | 69.8 | 150.6 | 1.2e-16 |
| XLA `jnp.linalg.cholesky`+triangular solves | 214.1 | — | 3.1e-16 |
| 8-step CG | 60.7 | — | 2.5e-16 |
| **Gauss-Jordan, no pivoting, broadcast-only** | **28.1** | **72.2** | 3.7e-16 |

**KEPT: `gj_solve`** — Gauss-Jordan without pivoting (SPD normal equations;
non-finite results on a degenerate system are rejected by the existing
`finite` guard exactly like a failed LU).  No dots, no custom calls → fuses
into a few kernels.  Saves ~91 µs per LM iteration single, ~27 µs vmapped.

### 2026-08-28 — A/B #2: gj solver composition (N=512, base_tight, GB10)

Scratch: `opt/ab_n512_gj_{nc,cond}.log`.

| config | roll ref→fast | ic ref→fast | parity |
|---|---|---|---|
| gj+onepass+hoist+**nocond** | 70.2→**43.5 ms** (−38%) | 40.5→**21.1 ms** (−48%) | err rel 6.1e-10, latdev 1.7e-9 |
| gj+onepass+hoist+cond | 65.7→47.4 ms | 39.4→31.7 ms | err rel 1.6e-9, latdev 6.6e-9 |

**KEPT: solver=gj, onepass, hoist, nocond.**  With the solve cheap, the
unconditional one-pass r+J (no `lax.cond`) is clearly better — the cond
carries its own dispatch overhead and a duplicated primal eval.

(Aside: first `OPT_LEAN` run crashed — the head-weight unpack `(W1, b1)`
shadowed the `b1d_common as b1` module inside the closure; renamed to
`bh1/bh2/bh3`.  No numbers affected.)

### 2026-08-28 — A/B #3: lean residual + masked unroll (N=512, base_tight)

`lean` = per-rollout folding of every nu-constant (`wt`, weighted A, Phi_q)
into premultiplied matrices, merge of the head's last layer with `h_lin`
(one (136,R) stacked weight), stacking the A-projection and the G3 node
features into ONE matmul per residual eval, folding `Lt` into the IC head
weights, and computing H,g as a single (K,K+1) dot.  Associativity-only
changes; parity checked.  `unroll=U` = U masked micro-iterations per while
trip (bit-identical state sequence, amortizes loop control).

| config | roll ref→fast | ic ref→fast | parity |
|---|---|---|---|
| gj+nocond (no lean, from A/B #2) | 70.2→43.5 ms | 40.5→21.1 ms | 6.1e-10 |
| + lean (unroll=1) | 64.1→**38.8 ms** | 39.7→22.6 ms | err rel 1.0e-09, latdev 6.8e-9 |
| + lean, unroll=4 | 63.9→49.1 ms | 40.6→24.2 ms | err rel 5.1e-10, latdev 6.3e-9 |

**KEPT: lean (roll −11% on top of gj+nocond).  REJECTED: unroll=4** — the
loop control it amortizes costs ~5–10 µs/iter while the masked extra
iterations pay the full ~60–100 µs body; net loss.  unroll=2 also loses
(41.4 vs 38.8 ms) — **unroll REJECTED at every U**.  The masked-unroll
transform itself is verified value-preserving (identical errors and latent
paths at U=2 and U=4).

### 2026-08-28 — IC multistart study (MODE=icdiag, N=512): reduced-init arm REJECTED

`opt/icdiag_n512.log`.  Running each of the 9 LM inits alone against the
9-init reference best:

- zbar init alone matches the best optimum only on trajs 4, 5, 7; on trajs
  0/1/2/3/6 it lands 30–80% worse (e.g. traj 3: 4.14e-02 vs best 2.35e-02,
  latent distance 0.42).
- No single init (and no pair) covers all 8 trajectories: winners are spread
  over inits {0,1,2,3,4,5,7,8}.

The Gram-space IC landscape is genuinely multimodal; the 9-init multistart is
load-bearing.  **The ALGORITHMIC reduced-init IC arm fails parity by
construction and is rejected** — `IC_ARM=init2` exists in the driver but must
not be used for production numbers.

### 2026-08-28 — Matvec lowering + OPT_NODOT (N=512, base_tight)

Microbench (`opt/matvec_anatomy.out`): full lean r_w chain in-loop —
dot-lowered primal 23.0 µs/it vs broadcast-reduce 30.2, BUT the one-pass
r+J path (what the LM body actually runs): dot 60.7 vs broadcast **54.6**.
On the real rollout, replacing every matvec in the lean residual with
`sum(x[:,None]*W)` broadcast-reduces (no cublas custom calls; everything
fuses into reduce fusions):

| config | roll fast | ic fast | parity |
|---|---|---|---|
| lean (dot) | 38.8 ms | 22.6 ms | 1.0e-9 |
| lean + **nodot** | **31.6 ms** (−18%) | 23.5 ms (IC not yet nodot) | err rel 1.0e-09, latdev 6.8e-9 |

**KEPT: nodot** for the rollout; IC nodot measured next.

With nodot extended to the IC lean head (`opt/ab_n512_nodot_ic.log`):
roll 63.9→**30.4 ms**, ic 41.5→**21.5 ms** (err rel diff 1.3e-9 vs ref,
4.3e-11 vs committed A100 JSON; latdev 1.7e-8).  IC nodot is ~neutral in
time but removes custom calls — kept.  Composition so far (GB10, base_tight):
roll **2.1x**, ic **1.9x** vs the verbatim reference.

### 2026-08-28 — Outer-scan unroll + convergence

`OPT_SCAN_UNROLL=5` (unroll the fixed-length 50-step `lax.scan` 5x — bitwise
identical program, only loop control amortized): roll 30.4→**29.1 ms** (+4%),
ic unchanged.  KEPT (free).  Convergence criterion met: the last two ideas
(IC-nodot ~0%, scan-unroll 4%) were each <5%.

**FINAL CONFIGURATION** (all defaults in `sep_b1d_fast.py`):
`OPT_SOLVER=gj OPT_ONEPASS=1 OPT_HOIST=1 OPT_NOCOND=1 OPT_LEAN=1
OPT_NODOT=1 OPT_SCAN_UNROLL=5` — production A/B at N=512 and N=4096 across
all arms plus the labeled THROUGHPUT arm follow below.

## FINAL A/B (GB10, interleaved reps, TIME_REPS=7, medians over 8 trajs)

Scratch: `opt/final_n512.{log,json}`, `opt/final_n4096.{log,json}`.
"ref" = verbatim copy of the sep_b1d_scale.py device implementation, timed in
the same process, interleaved rep-by-rep with "fast".  Every arm's error
matches BOTH the in-process reference and the committed A100 baseline JSON to
≤ 2.9e-9 relative (limit 0.2% = 2e-3): parity holds everywhere.

| N | arm | err (ref == fast) | vs A100 json | latdev | ic ref→fast | roll ref→fast | dec |
|---|-----|-------------------|--------------|--------|-------------|----------------|-----|
| 512 | oracle | 4.876680e-03 | 7.5e-10 | 1.1e-08 | 40.5→21.2 | 80.1→55.0 | 1.1 |
| 512 | base_tight | 5.194389e-03 | 4.3e-11 | 1.1e-08 | 40.3→**21.4** | 64.1→**28.8** | 1.1 |
| 512 | nodes_tight | 4.886140e-03 | 9.2e-10 | 1.1e-08 | 40.2→22.3 | 67.4→30.3 | 1.0 |
| 512 | base_half | 2.675516e-02 | 4.5e-10 | 1.0e-08 | 40.4→21.6 | 73.1→32.5 | 1.1 |
| 512 | nodes_half | 1.029420e-02 | 6.6e-10 | 1.0e-08 | 40.9→21.0 | 65.1→29.1 | 1.1 |
| 4096 | oracle | 4.485117e-03 | 1.3e-09 | 2.7e-08 | 36.3→19.4 | 81.2→53.9 | 1.4 |
| 4096 | base_tight | 4.888627e-03 | 5.9e-10 | 2.7e-08 | 35.6→**18.9** | 59.6→**26.3** | 1.0 |
| 4096 | nodes_tight | 4.529491e-03 | 1.0e-09 | 3.2e-08 | 35.1→18.9 | 64.1→28.8 | 1.2 |
| 4096 | base_half | 2.293701e-02 | 2.8e-09 | 1.6e-07 | 36.2→18.8 | 76.1→32.6 | 1.1 |
| 4096 | nodes_half | 9.218639e-03 | 2.2e-11 | 1.5e-07 | 34.6→18.5 | 61.0→26.8 | 1.1 |

Sampled arms: **roll 2.2–2.3x, ic 1.9x, e2e ≈ 2.1x** — flat in N (as the
baseline was; nothing in the fast path scales with N except the one-off
IC projection and decode).  Oracle arm (no lean — full-grid advection in the
residual): roll 1.5x from gj+onepass+nocond alone.

Stop-reason distributions and mean_njac per trajectory match ref throughout
(recorded per-arm in the final JSONs under `parity`).

### THROUGHPUT arm (labeled; amortized, NOT latency)

`MODE=thru`: the identical optimized rollout vmapped over the 8 test
trajectories (batched while_loop pays worst-case iterations per batch —
honest amortized number only; per-trajectory VALUES bit-identical to the
sequential fast path, err 5.194389e-03 reproduced exactly at N=512):

| N | batch ic | batch roll | amortized per traj (ic + roll) |
|---|----------|-----------|--------------------------------|
| 512 | 142.1 ms | 137.4 ms | 17.8 + 17.2 = **35.0 ms** |

(vs sequential fast 21.4 + 28.8 = 50.2 ms, i.e. another ~1.4x for
many-query workloads.)

| N | batch ic | batch roll | amortized per traj (ic + roll) |
|---|----------|-----------|--------------------------------|
| 512 | 142.1 ms | 137.4 ms | 17.8 + 17.2 = 35.0 ms |
| 4096 | 110.8 ms | 120.8 ms | 13.9 + 15.1 = 29.0 ms |

## How to run the A100 confirmation (cluster)

One job per N, isolated job dir, standard cluster mechanics (venv
`/cluster/tufts/paralab/tawal01/ae-research/venv`, `gpu` partition,
`jax_backend=gpu` preflight, `JAX_DEFAULT_MATMUL_PRECISION=highest`).  From a
checkout of this branch's `experiments/separable-decoder/` (needs
`b1d_common.py`, `b1d_fast_common.py`, `sep_b1d_fast.py`, and the committed
`runs/b1dqf/b1ds_n{512,4096}/out/` artifacts — .pkl, _nodes.npz, .json):

```bash
cd experiments/separable-decoder
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
for NN in 512 4096; do
  N=$NN MODE=ab TIME_REPS=7 \
  ARMS=oracle,base_tight,nodes_tight,base_half,nodes_half \
  CKPT_CACHE=runs/b1dqf/b1ds_n$NN/out/sep_b1d_scale_n$NN.pkl \
  NODES_NPZ=runs/b1dqf/b1ds_n$NN/out/sep_b1d_scale_n${NN}_nodes.npz \
  BASE_JSON=runs/b1dqf/b1ds_n$NN/out/sep_b1d_scale_n$NN.json \
  OUT=out/fast_a100_n$NN.json \
  JAX_DEFAULT_MATMUL_PRECISION=highest "$PY" sep_b1d_fast.py
  N=$NN MODE=thru ARMS=base_tight ... OUT=out/thru_a100_n$NN.json \
  JAX_DEFAULT_MATMUL_PRECISION=highest "$PY" sep_b1d_fast.py
done
```

No training happens: the decoder checkpoint and node sets are loaded from the
committed artifacts; only the 8 test trajectories are regenerated from
`TEST_SEED=1` with the tridiagonal truth generator (seconds).  All OPT_*
flags default to the final configuration; `OPT_SOLVER=lu OPT_LEAN=0
OPT_NODOT=0 OPT_ONEPASS=0 OPT_NOCOND=0 OPT_HOIST=0 OPT_SCAN_UNROLL=1`
reproduces the reference behavior in the fast slot if ever needed.  Accept
the run iff each arm's `parity.err_rel_diff_fast_vs_base` ≤ 2e-3 (observed
≤ 3e-9) and `lat_dev_max` is small (observed ≤ 1.6e-7).

## Ladder summary (isolated → composed, GB10, base_tight N=512, roll / ic ms)

| step | roll | ic | verdict |
|---|---|---|---|
| reference (verbatim) | 64–70 | 40 | baseline |
| scalar-chol (a) | 93.6 | 61.0 | REJECTED (slow serial fusions) |
| XLA command buffers (d) | 66.5 (no change) | — | REJECTED (2x slower on microbench) |
| gj solve + onepass + hoist + nocond (a,b,c) | 43.5 | 21.1 | KEPT |
| + lean folding (c) | 38.8 | 22.6 | KEPT |
| + nodot broadcast matvecs (g) | 30.4 | 21.5 | KEPT |
| + scan_unroll=5 (g) | 28.8–29.1 | 21.4 | KEPT (~4%) |
| masked LM unroll U=2/4 (g) | 41.4 / 49.1 | — | REJECTED |
| reduced-init IC (e) | — | — | REJECTED (breaks parity; multimodal) |
| THROUGHPUT vmap x8 (f) | 17.2 amortized | 17.8 | separate labeled arm |

Bottom line (GB10): sampled-arm rollout **2.2x**, IC fit **1.9x**, e2e
**~2.1x**, identical results to 1e-9 relative; amortized throughput path
reaches **~3.7x** per trajectory vs the reference sequential path. — i.e. ~40–45% of the measured 257 µs/LM-iteration, and it also
sits (vmapped over 9 inits) in every IC-fit iteration.  The remaining
~130 µs/iter is the residual+Jacobian network evals (several batched matmul +
elementwise kernels at ~5–10 µs each).  Predicate readback is NOT a
bottleneck; CUDA graphs are NOT a lever here.  Conclusion: kill cuSOLVER with
the unrolled Cholesky but KEEP the reference's `lax.cond` (which skips the
Jacobian eval on rejected steps) — the earlier all-opt slowdown is explained
by the nocond restructure paying a full J eval every iteration.
