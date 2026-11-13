# Protocol: how large an operator, and how much data, to reach the ROM's median error (2D viscous Burgers, $256^2$)

Written 2026-11-12 (about 21:00 EST) **before any training or test outcome of this lane exists**. The only
measurements seen so far are a speed/memory benchmark on synthetic inputs (job 7157490, `bench_speed.py`: no data,
no validation, no test). Later changes are appended under "Amendments" with date and reason; nothing above that
heading is edited afterwards.

Lane `exp/2026-11-12-operator-scaling`, forked from `6357834e4` (head of `exp/2026-11-11-burgers-ops-matched`, the
"template lane"; its PROTOCOL.md, amendments A1–A7, code and results are the starting point). Cluster namespace
`/cluster/tufts/paralab/tawal01/rebuttal-op-scaling/`. Answers reviewer 1dYR (ICLR 49310).

## 1. Question

With 4,608 trajectories the template lane's operators reach a median test error of 0.189 % (FNO, validation-selected,
6.60 ms) and 0.203–0.224 % (U-Net, 7.4 ms); the ROM (span width 384) reaches **0.0264 %** median at 45.2 ms
(the full-order solver's own median error at the ROM-accurate tolerance is 0.030 %). The goal is to **scale the
operators (model size and training data) until they reach the ROM's median test error, and report their query
time at that accuracy**. If no configuration reaches 0.0264 % within the deadline, report the closest median
reached, its query time, the measured trends, and (only if the trend is clean, section 7) a clearly labelled
extrapolation. The outcome is reported whichever way it goes.

## 2. Unchanged from the template lane

Problem, generator (`fft_tight`, 256 intervals, $\Delta t=0.005$, Newton $10^{-6}$, linear $10^{-8}$), operator
contract (input $u_0$ and $\nu$ on $257^2$; output the five fields at $t=0.05,\dots,0.25$), loss (mean squared
relative error), optimiser (AdamW, weight decay $10^{-4}$, batch 8, gradient clip 1.0), schedule and stopping rule of
template amendment A2 (validation every 500 steps; ReduceLROnPlateau factor 0.5, patience 6 validations, threshold
$10^{-3}$ relative, floor $10^{-5}$; early stopping after 20 validations without a relative improvement $>10^{-3}$),
selection score (validation mean of per-case max-over-time relative error), precision (networks float32, TF32 off,
features/loss float32, final metric float64), test cohort test64 = `params_draw(20260916, 64)` (sha256
`cd058fd2…`, asserted), metric (per case $\max_k \lVert u(t_k)-u_{\rm ref}(t_k)\rVert_2/\lVert u_0\rVert_2$;
median and worst over 64 cases, %), timing (one A100 80GB, batch 1, timed region from $u_0,\nu$ on the GPU to the
six fields on the GPU, synchronise around each query, 20 untimed warm-up queries, 5 repetitions × 64 cases,
median and 5/95 % percentiles over 320 samples). Learning rate per family as the template's selected arms: FNO
$10^{-3}$, U-Net $3\times10^{-4}$.

## 3. Data (the data axis)

Pool `P = params_draw(0,576) ++ params_draw(1000,4032)` (the template's 4,608 cases, copied byte-for-byte from the
template namespace). Extra draw `X = params_draw(20261112, 32256)` (new seed, disjoint from 0, 1000, 20260916 and
the dev seed 7090702; no repeated parameter vector, asserted), solved with the same generator, stored float32 as the
pool. Generation cost (GPU-seconds, GPU type, job ids) is recorded from the generator's JSON files and reported.

**Validation is the same 256 cases at every data size: `P[4352:4608]`** (the template's 4,608-rung validation set),
so validation scores are comparable across data sizes. Rungs (total trajectories incl. validation → training):

| rung | training cases | layout |
|---|---|---|
| 4,608 (1×) | 4,352 | `P[0:4352]` (bitwise the template's training set) |
| 9,216 (2×) | 8,960 | `P[0:4352] ++ X[0:4608]` |
| 18,432 (4×) | 18,176 | `P[0:4352] ++ X[0:13824]` |
| 36,864 (8×) | 36,608 | `P[0:4352] ++ X[0:32256]` |

Rungs above 4,608 keep the data in pinned host memory and copy each batch to the GPU (numerically identical).

## 4. Model sizes (the size axis) — parameter counts measured (real parameters; FNO complex weights count twice)

| arm | family | configuration | parameters | × template paper arch |
|---|---|---|---|---|
| Fw64m32L4 | FNO | width 64, 32 modes, 4 layers (= template F0) | 17,877,317 | 1.0 |
| Fw96m32L4 | FNO | width 96, 32 modes, 4 layers (= template F2, its selected FNO) | 40,222,181 | 2.25 |
| Fw128m32L4 | FNO | width 128, 32 modes, 4 layers | 71,504,517 | 4.0 |
| Fw128m64L4 | FNO | width 128, 64 modes, 4 layers | 277,025,413 | 15.5 |
| Fw192m48L4 | FNO | width 192, 48 modes, 4 layers | 354,343,877 | 19.8 |
| Ub32 | U-Net | base 32 | 7,763,461 | 0.44 |
| Ub48 | U-Net | base 48 (= template U0, its selected U-Net) | 17,462,021 | 1.0 |
| Ub64 | U-Net | base 64 (= template U1) | 31,038,469 | 1.8 |
| Ub96 | U-Net | base 96 | 69,825,029 | 4.0 |
| Ub128 | U-Net | base 128 | 124,123,141 | 7.1 |
| Ub192 | U-Net | base 192 | 279,254,021 | 16.0 |

Depth (FNO 8 layers, `Fw128m32L8`, 142.9 M) is defined in the code but is run only if a GPU becomes free after
everything below has started (it is not part of the fitted ladder).

## 5. Runs, budget, priority (fixed now)

**Budget per run (counted in gradient steps, hardware-free):** hard cap **200,000 steps** (the template's runs stopped
at 59,500–147,500 steps, so this is at least as long as every template run), plus the stopping rule of section 2.
The two **priority runs** (largest-data, large-model; the configurations most likely to reach the target) get a cap of
**400,000 steps**, and keep a copy of their best checkpoint up to step 200,000 (`best_upto_200000.pt`) so that the
data axis can also be read at the common 200k budget. A run ended by the step cap, the deadline, or a job limit
is flagged **"not converged"** in every table, with its final learning rate and the gain of its validation score over
its last 10/20 validations. One seed (0) per run (no time for more); the template's three-seed spread at 4,608
(FNO F0 median 0.195–0.233 %, U-Net U0 0.203–0.224 %) is quoted as the seed-noise scale.

**Hard deadline:** every training process stops at its first validation after **2026-11-15 10:00 EST** (unix
1794754800; reason `deadline`, flagged not converged). Runs not started by then are reported as not run.
A job that hits the cluster's wall limit is continued exactly from its last 5,000-step state (`last.pt`: model,
optimiser, scheduler, sampler state; the only deviation is that the interrupted epoch's order is redrawn).

Hardware for training: one RTX PRO 6000 (96 GB) or H200 per job (fastest float32 throughput available; benchmark
job 7157490); runs within a job are sequential, in the priority order below. GPU type does not affect a step-capped
run's result beyond floating-point nondeterminism. At most ~5 of this lane's jobs at once (the 12-GPU account cap is
shared with the official-baselines lane).

| priority | runs | rung | cap |
|---|---|---|---|
| 1 | Fw128m64L4, Ub128 | 36,864 | 400k |
| 2 | Fw128m64L4, Fw192m48L4, Ub128, Ub192 | 4,608 | 200k |
| 3 | Fw128m64L4, Ub128 | 18,432 and 9,216 | 200k |
| 4 | Fw128m32L4, Fw96m32L4, Fw64m32L4, Ub96, Ub64, Ub48, Ub32 | 4,608 | 200k |

The data-axis arms (Fw128m64L4 for FNO, Ub128 for U-Net) are fixed now, not chosen from outcomes: they are the largest
configuration of each family whose 400k-step run fits the time to the deadline on one GPU with the 36,864-case set
in host memory (benchmark: 152 and 194 ms per step on the RTX PRO 6000). Fw192m48L4 and Ub192 are run at 4,608 only.

## 6. Selection and the single test evaluation

- Nothing in this lane reads `data/test` before all training has stopped. Then a selection manifest (every run's
  final `best.pt` and, for the priority runs, `best_upto_200000.pt`, with SHA-256, validation scores and stop reasons)
  is committed, and **every frozen checkpoint is evaluated once** on test64 at $256^2$ in one job on one A100 80GB
  (template evaluator, timing protocol of section 2). No $1024^2$ evaluation (not part of this question).
- Headline per family: the run with the **lowest validation score over all runs of that family** (any size, any
  rung) — its test median, worst case and query time. All other runs are reported in the full table as measured.
- "Reaches the ROM's median" means test median $\le 0.0264\,\%$ for a frozen configuration. If any run does, its
  measured query time is reported directly (together with its worst case, beside the ROM's 0.875 %).

## 7. Trends and extrapolation (fixed now)

- **Size trend:** per family, ordinary least squares of $\log_{10}$(test median) on $\log_{10}$(parameters) over the
  4,608-rung size ladder (FNO 5 points, U-Net 6 points). **Data trend:** per data-axis arm, the same on
  $\log_{10}$(training cases) over rungs 4,608 / 9,216 / 18,432 / 36,864 (using the 200k-step checkpoint at 36,864 so
  the budget is equal; the 400k checkpoint is reported beside it).
- Reported for each fit: slope, intercept, $R^2$, residual standard error, 95 % confidence interval of the slope
  ($t$ distribution, $n-2$ degrees of freedom).
- **An extrapolation is given only if** the fit has $\ge 4$ points, $R^2 \ge 0.9$, the slope's 95 % interval excludes
  zero, and the medians decrease monotonically along the axis. It is then the parameter count (or data size) at which
  the fitted line reaches 0.0264 %, with a 95 % interval from the slope interval, plus the query time implied by a
  log–log fit of measured query time on parameters over the same runs. It is labelled **"extrapolation, not a
  measurement"** everywhere it appears. If the criteria fail the report says the trend is not clean enough to
  extrapolate, and gives the measured numbers only.
- The worst-case error is fitted and reported in the same way (no extrapolation claim attached to it).

## 8. Outputs

`reports/tables.md` and `reports/summary.json` generated by script from the run and test JSON files (never typed by
hand), log–log figures of median error against parameters and against training cases, and a short report
`reports/REPORT.md`. Status log: `.../scratchpad/nmrom/push/operator-scaling-status.md`.

## Amendments

### 2026-11-12 ~21:45 EST, A1 — made after about one hour of training (validation histories of the six running runs seen), before any test evaluation

What was seen: the validation curves of the first 0.5–1 h of the six runs (`status.py`). At the FNO's learning rate
$10^{-3}$ the validation score of the large models is noisy, and the plateau scheduler of section 2 halved the rate
six times in the first 34,000 steps of the priority FNO run (rate $1.6\times10^{-5}$, near the floor $10^{-5}$), well
before its 400,000-step budget. A reviewer could argue that this schedule, not size or data, limits the operators.
To give them that chance, **one additional run** is added, without changing any protocol run:

- `Fw128m64L4` at the 36,864 rung, 400,000 steps, **cosine decay** of the learning rate from $10^{-3}$ to $10^{-6}$
  over the 400,000 steps, no plateau scheduler and no early stopping, best-validation checkpoint as before (run name
  suffix `-cosine`). Everything else as section 2. Started when one of this lane's six GPU slots frees (the account
  cap is shared 6/6 with the other lane); if it cannot finish by the deadline it is stopped by the deadline rule and
  flagged.
- If a second slot frees before 2026-11-13 12:00 EST, the same for `Ub128` at the 36,864 rung with 200,000 steps
  (lr $3\times10^{-4}\to10^{-6}$).
- These runs are eligible for the per-family validation selection of section 6 and are reported in the full table,
  labelled; they are **not** points of the fitted size or data ladders of section 7 (those keep one schedule).
