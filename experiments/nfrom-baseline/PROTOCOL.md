# Protocol: neural-field ROM baselines (CROM-style, CoLoRA-style) on the paper's Burgers 2D data

Written 2026-11-11 **before any training or evaluation outcome of this lane exists**. Later changes go under
"Amendments" with the reason; nothing above that heading is edited afterwards.

Lane `exp/2026-11-11-nfrom-baseline`, forked from `93f399faa` (`exp/2026-10-20-jcp-consolidated`).
Cluster namespace `/cluster/tufts/paralab/tawal01/rebuttal-nfrom/`.

## 1. Question (reviewer 1dYR, weakness 1 and question 3)

How does the paper's method compare, on its own Burgers 2D data, cohort and metric, with the neural-field
reduced models it names as its closest relatives: CROM (Chen et al., ICLR 2023) and CoLoRA (Berman & Peherstorfer,
ICML 2024)? And what do the stored ablations say about the separate contributions of the coordinate bank, the
nonlinear head and hyper-reduction? The outcome is reported whichever way it goes.

## 2. Data (same trajectories as the paper's method)

- Parameter vectors: `P = params_draw(0,576) ++ params_draw(1000,4032)` — the 576 bank trajectories and the 4,608
  head trajectories of the paper's Burgers 2D checkpoint, in this order (same as lane
  `exp/2026-11-11-burgers-ops-matched`).
- Discretisation: 256 intervals ($257^2$ nodes), backward Euler, $\Delta t = 0.005$, first-order sign-upwind
  advection, $T = 0.25$; solver `fft_tight` (Newton–BiCGStab, FFT Helmholtz preconditioner, Newton $10^{-6}$,
  linear $10^{-8}$) — the test reference solver. Cross-checked against the operator lane's pool on shared cases.
- Saved states: every second time level, $t = 0, 0.01, \dots, 0.25$ (26 per trajectory; 119,808 states in all,
  float32). The paper's head was fitted on 131,072 states of the same 4,608 trajectories (of 235,008) and its bank
  on 16,384 states of the first 576. Declared difference: the baselines see a different, similar-sized subset of
  states of the same trajectories.
- Split: trajectories `P[0:4352]` train, `P[4352:4608]` validation (carved from the pool, so the baselines train on
  slightly fewer trajectories than the paper's head, never on anything outside `P`).
- Test: `test64 = params_draw(20260916, 64)`, physical sha256 `cd058fd2…` asserted, reference fields
  `rebuttal-burgers-ops/data/test/test64-L256.fields.npy` (read-only; `fft_tight`, float64). The test file is opened
  by exactly one script (`rom.py test`), **once, after every selection below is frozen in `selection.json`**.

## 3. Metric (the paper's)

Per case $\max_{k=1..5} \lVert u(t_k) - u_{\rm ref}(t_k)\rVert_2 / \lVert u_0 \rVert_2$ at
$t_k = 0.05,\dots,0.25$ over the $257^2$ grid, float64. Reported: worst and median over the cohort, in percent.

## 4. Arms

Common: networks in JAX, float32 training; decoder input is the node coordinate through 128 fixed Gaussian Fourier
features (scale 4, the paper's bank embedding); the zero Dirichlet condition is imposed exactly at the discrete
level (boundary nodes are set to zero; the decoder is trained and evaluated on interior nodes only). Loss: squared
error at randomly drawn interior nodes divided by the mean square of that trajectory's initial field (the metric's
normalisation), Adam, linear warm-up then cosine decay to $10^{-5}$.

**Arm C (CROM-style).** Decoder $g_\theta(x, z)$: MLP on $[\gamma(x), z]$ (latent concatenated to the coordinate
features, as in CROM), GELU, scalar output. Encoder $e_\phi$: convolutional network on the state restricted to the
$65^2$ sub-grid (every fourth node), output $z \in \mathbb{R}^k$; trained jointly as an auto-encoder on the training
snapshots. Latent sizes $k = 16$ (the paper's Burgers head latent size) and $k = 32$.
Online, per query: $z_0 = e_\phi(u_0)$, optionally refined by Gauss–Newton against $u_0$ at the collocation nodes;
then for each of the 50 backward-Euler steps the latent is the Levenberg–Marquardt minimiser of the **same discrete
residual as the reference solver** (`burgers_fom.residual`: backward Euler, sign-upwind advection, five-point
Laplacian on the 256 mesh) evaluated only at $M$ fixed randomly chosen interior collocation nodes, with the stencil
values obtained by evaluating the decoder at each collocation node and its four neighbours. The Jacobian with
respect to $z$ is exact (forward mode). No linear bank, no precomputed operators, no encoder after $t=0$.
This is CROM's optimisation-based projection at a sub-sampled set of points, used implicitly (least-squares
Petrov–Galerkin on the collocation residual) because the reference is an implicit discretisation; CROM's original
explicit update followed by a projection is not run unless time allows and, if run, is reported as a separate row.
Output: decoder evaluated on the full grid at the five output times.

**Arm L (CoLoRA-style), data-driven variant (CoLoRA-D).** Decoder MLP on $\gamma(x)$ whose hidden layers are
$W_\ell + A_\ell\,\mathrm{diag}(\alpha_\ell)\,B_\ell$ with low rank $r$; the latent is the stacked
$\alpha \in \mathbb{R}^k$, $k = 16$ and $32$ (four adapted layers, $r = 4$ or $8$). A hyper-network
$h_\psi(t, \mu) \mapsto \alpha$ with $\mu = (c_x, c_y, w, a, \log\nu)$ is trained jointly with the decoder.
Online: one hyper-network evaluation per output time and one decoder pass over the grid; no PDE residual is used.
Declared difference: this arm is given the five generating parameters of the initial condition, which the paper's
method (and arm C) do not use — they start from the initial field. A residual-based variant on the CoLoRA decoder
(same collocation Levenberg–Marquardt as arm C with $z = \alpha$) is an optional extra row.

## 5. Tuning budget and selection (validation only)

- Offline grid (seed 0 each; all trained with the same step budget, a run whose validation curve is still
  improving by more than 2 % over the last 10 % of steps is flagged "not at plateau" and extended if time allows):
  arm C: $k\in\{16,32\}$ × (width, depth) $\in\{(128,4),(256,5),(512,5)\}$ at learning rate $10^{-3}$, plus
  $3\times10^{-4}$ for (256,5) and (512,6) at $k=16$;
  arm L: $k\in\{16,32\}$ × decoder width $\in\{256,512\}$ (5 hidden layers), hyper-network width 256/512, learning
  rate $10^{-3}$, plus $3\times10^{-4}$ at one size.
- Offline monitoring score: validation reconstruction (C) or prediction (L) error in the metric's normalisation on
  a fixed set of validation states.
- Arm C online knobs, tuned on the first 32 validation trajectories: collocation nodes $M\in\{512, 2048, 8192\}$,
  Levenberg–Marquardt iterations per step $\in\{1,2,4\}$, initial refinement on/off, float64. For each $k$ the two
  decoders with the best validation reconstruction error enter this grid (all of them if time allows).
- Selection per (arm, $k$), frozen in `selection.json` before the test file is read:
  **accurate** = lowest validation worst-case rollout error (settings within 5 % of the lowest: the fastest);
  **fast** = the fastest setting with validation worst-case error $\le 3.85\,\%$ (the paper's fast setting's test
  error), if any. Arm L: the configuration with the lowest validation worst-case error per $k$.
- Everything that was run is listed in the report with its validation numbers; only the selected settings are
  evaluated on the test cohort (one invocation).

## 6. Timing

One A100 80GB (the GPU type of the paper's ROM rows), float64 online (as the paper's solves), one process, batch 1.
The timed region starts with $u_0$ (and $\nu$, or $\mu$ for arm L) on the GPU and ends with the five output fields
on the GPU (encoder, latent rollout and full-grid decoding inside); 20 untimed warm-up queries, then 5 repetitions
of each of the 64 cases; median, 5th/95th percentile, min and max. The latent-rollout time without the full-grid
decoding is reported beside it. A float32 online variant is timed and reported as an extra row if its validation
error is within 5 % of the float64 one. Training cost: wall time and steps per run, with the note that tuning runs
share GPUs (so wall time is an upper bound on a dedicated run), plus a dedicated re-timing of steps/s if a slot is free.

## 7. Comparison rows and verdict rule (fixed now)

Paper rows, same cohort and metric, A100 80GB, from stored JSON (not re-run; the comparison of times is across
allocations of the same GPU type, which is declared):
span $R'=384$: worst 0.875 %, median 0.026 %, 45.2 ms; span $R'=96$: 3.85 %, 0.529 %, 14.2 ms
(`2026-09-25-burgers2d-test`, job 4319328); head $k=16$: taken from `2026-09-25-t2-burgers-test` if stored.

For each baseline arm and latent size, on test64:
- it **beats the paper's accurate setting** if its worst-case error is below 0.875 % and its time below 45.2 ms;
- it **beats the paper's fast setting** if worst-case error is below 3.85 % and time below 14.2 ms;
- it **beats the paper's nonlinear head at the same latent size** if both its worst-case error and its time are
  below the head $k=16$ row's;
- otherwise the paper's method is not dominated by that arm; which of accuracy and time it wins is stated.
The median error is reported beside the worst-case one with the same statements.

## 8. Ablation rows of the paper's method

Taken from stored results only (bank + span, bank + head, POD at matched rank, quadrature on/off), each with its
cohort, mesh, job and file; development-cohort rows are labelled as such. No new runs of the paper's method.

## 9. Secondary problem

Navier–Stokes 3D at $64^3$ is attempted only after the Burgers result is complete; if not reached it is reported
as not attempted.

## Amendments

**A1 (2026-11-11, before any training or evaluation outcome; reason: cost of the online knob grid, estimated
from operation counts while writing the solver).** (a) Offline step budget fixed: 120,000 Adam steps, 64 states ×
512 interior nodes per step, warm-up 1,000 steps. (b) Arm L decoder is the input layer plus four adapted hidden
layers (not five). (c) Online knob grid of arm C: the initial refinement (4 Gauss–Newton/LM iterations against
$u_0$ at the collocation nodes) is switched on or off once per decoder, at the middle setting ($M=2048$, 2
iterations), and the better choice is used for the other eight settings (10 settings per decoder instead of 18).
(d) Decoders of width $\ge 512$ are not run at $M = 8192$ (one LM iteration there costs about $10^{12}$ floating-point
operations, i.e. tens of seconds per query). (e) Because online cost grows with decoder width, **every** trained
arm-C decoder enters the knob grid (not only the two with the best reconstruction error), so that small, fast
decoders are not excluded. (f) The float32 online variant is run on validation for each decoder's most accurate
setting and its fastest setting under the 3.85 % gate. (g) Selection times are rough batch-1 timings from the grid
job (2 warm-ups, 2 repetitions of 3 cases); the reported times are from the final timing protocol of section 6.

**A2 (2026-11-11 ~04:30 EST; after the first validation *reconstruction* curves of three arm-C decoders, before any
rollout, any arm-L outcome and any test evaluation; reason: the 120,000-step budget of A1 is not a plateau).**
Evidence (job 7121932, validation monitor only): C16-w128, C16-w256 and C16-w256-lr3 finished 120,000 steps with
validation mean reconstruction error 0.591 %, 0.452 %, 0.570 % and were all flagged "not at plateau" (3.2 %, 5.0 %,
4.6 % improvement over the last tenth of the schedule); the measured throughput (700 / 370 / 155 steps per second at
width 128 / 256 / 512) is about thirty times the estimate A1 was sized for, so a longer schedule is affordable.
Leaving the baselines at 120,000 steps would handicap them. Therefore:
(a) the offline step budget is raised to **600,000 Adam steps** (same batch, warm-up, cosine decay to $10^{-5}$) for
every configuration of both arms; these runs are stored under `models-x/` with the knob grid under `grid-x/`;
(b) the 120,000-step arm-C runs (`models/`, `grid/`) are kept and reported as a short-budget row of the training
table, and their validation knob grid is reported, but the selection of section 5 is made over **all** decoders that
were trained (both budgets), by the same rule — a longer-trained decoder is not preferred by fiat;
(c) the 120,000-step round is not run for $k=32$ and for arm L (their first jobs, 7121934 and 7121935, failed with a
GPU out-of-memory error on 48 GB cards before the first validation; no outcome exists);
(d) if a 600,000-step run is still flagged "not at plateau", that is reported beside its row; a further extension is
run only if time allows and is again an amendment;
(e) C16-w512d6-lr3 is trained last and only if the wall budget allows; if it is not trained this is reported;
(f) training jobs are restricted to GPUs with at least 80 GB (the training set is held on the GPU). Training
hardware is reported per run; the test evaluation and timing remain on an A100 80GB.

**A3 (2026-11-11 ~05:35 EST; after validation rollouts of two arm-C decoders, before any test evaluation; reason:
a robustness defect of the baseline's online solver that would have handicapped it).** Evidence (validation only,
jobs 7121932 and 7122345): with the solver as first written ("v1": Levenberg–Marquardt damping reset to $10^{-6}$ at
every time step, initial guess = previous latent) the worst validation case sat at 75–84 % for every knob setting of
C16-w256 (600k) and at 82 % for C16-w128 (120k) with 1–2 iterations, while the median was 1–6 %: on one trajectory
every trial step was rejected and the few iterations per step never raised the damping far enough, so the latent
froze. The paper's own latent solver carries its damping across time steps and uses an extrapolated predictor
(its settings are named `lamcarry`, `pred2`), so the baseline is given the same two devices ("v2"):
(a) the damping is carried from one time step to the next (still $\times 0.1$ on acceptance, $\times 10$ on rejection);
(b) the initial guess of a step is the linear extrapolation $2z_n - z_{n-1}$ whenever its collocation residual is
smaller than that of $z_n$, otherwise $z_n$.
Nothing else changes (same residual, same knob grid, same selection rule). The v1 validation grids already computed
are kept (`grid-v1/`, `grid-x-v1/`) and reported; every decoder's knob grid is (re)computed with v2 and the
selection is made over the v2 grids only. The network-free solver check (`check_solver.py`) was re-run with v2 and
gives the same errors as before.
