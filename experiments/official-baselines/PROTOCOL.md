# Protocol: the authors' released CROM and CoLoRA code on the paper's Burgers 2D benchmark

Written 2026-11-12, **before any training or validation outcome of either method in this lane, and before the test
cohort is read**. The only outcomes known when it was written are the smoke tests of the two repositories' own examples
and the network-free integrator check of section 5.3 (validation trajectories only). Later changes go under
"Amendments" with the reason; nothing above that heading is edited afterwards.

Lane `exp/2026-11-12-official-baselines`, forked from `93f399faa`. Cluster namespace
`/cluster/tufts/paralab/tawal01/rebuttal-official/` (envs in `env/`, pinned clones in `ext/`, one directory per job
in `jobs/`). Template lane: `exp/2026-11-11-nfrom-baseline` (re-implementations), whose data, split, cohort, metric
and timing protocol are reused unchanged.

## 1. Question

How do the **authors' released implementations** of CROM (Chen et al., ICLR 2023) and CoLoRA (Berman & Peherstorfer,
ICML 2024), adapted only at the data/output boundary, perform on the paper's Burgers 2D (256 mesh) test cohort,
against the re-implementations of lane `nfrom-baseline` and the paper's method? Reported whichever way it goes.

## 2. Code (pinned)

| method | repository | commit | owner |
|---|---|---|---|
| CROM part I (offline) | https://github.com/dc3042/CROM_offline_training | `39333f76ccf878dacc9ae425c185b76ed4e72c5c` | Dong Heon Cho (co-author); "Code (Part I)" on crom-pde.github.io |
| CROM part II (online) | https://github.com/peterchencyc/CROM_online_deployment | `ccb8a6603318492442e32f9d254d420b63c86419` | Peter Yichen Chen (first author); "Code (Part II)" |
| CoLoRA | https://github.com/julesberman/CoLoRA | `736700464b5e8de2d0a8d79c75f518da29a8a139` | Jules Berman (first author) |

Environments (isolated, `env/`): CROM: Python 3.9, torch 1.11.0+cu113, pytorch_lightning 1.6.5, h5py 3.4.0
(README and requirements pins). CoLoRA: Python 3.11, jax/jaxlib 0.4.19 (CUDA 12, cuDNN 8.9), flax 0.7.5,
optax 0.1.9, dm_haiku 0.0.10, numpy 1.24.4 (requirements pins). Every change to the authors' files is a diff in
`diffs/`; new files (adapters) live in `code/`.

What is **not released** and therefore not run: CoLoRA's equation-driven (Neural Galerkin) variant (the repository
contains only the data-driven hyper-network notebooks); CROM implicit time integration (the released online loop
has explicit `linear` and `nonlinear` projections only).

## 3. Data, split, cohort, metric (identical to `nfrom-baseline` PROTOCOL sections 2-3)

- Pool: `rebuttal-nfrom/data/pool26.fields.npy` (read-only), parameters `P = params_draw(0,576) ++ params_draw(1000,4032)`,
  256 intervals ($257^2$ nodes), backward Euler $\Delta t = 0.005$, sign-upwind advection, five-point Laplacian,
  `fft_tight` solver, 26 saved levels $t = 0, 0.01, \dots, 0.25$, float32.
- Train: trajectories `P[0:4352]` (113,152 snapshots). Validation: `P[4352:4608]`; selection uses the first 32
  validation trajectories (as the template).
- Test: `test64 = params_draw(20260916, 64)`, `rebuttal-burgers-ops/data/test/test64-L256.fields.npy` (float64,
  physical sha256 `cd058fd2…` asserted), opened by one script (`evaluate.py test`), **once per frozen selection**.
- Metric: per case $\max_{k=1..5} \lVert u(t_k) - u_{\rm ref}(t_k)\rVert_2 / \lVert u_0 \rVert_2$, $t_k = 0.05,\dots,0.25$,
  $257^2$ grid, float64; worst and median over the cohort, in percent.
- Output adapter (both methods): the model's field on the $257^2$ nodes at the five output times. CROM: boundary nodes
  are zero (the online residual acts on interior nodes with the zero Dirichlet value, as the released Burgers example
  imposes its boundary value on the decoded state). CoLoRA: its raw prediction on all $257^2$ nodes is scored, unchanged.

## 4. CoLoRA (data-driven, CoLoRA-D, as in the released `burgers.ipynb`)

Unchanged: `colora.build.build_colora`, `colora.layers`, `colora.networks`, `colora.adam.adam_opt` (Adam, cosine decay
to 0), the notebook's `relative_loss_fn` (mean over the batch of $\lVert u - \hat u\rVert_2/\lVert u\rVert_2$ over the full
grid), input normalisation of $(\mu, t)$ by train mean/std, `Periodic` first layer with period = domain length (1, 1),
swish activations, `PRNGKey(1)`, final (not best) parameters.

Adapted: (i) data — $\mu = (c_x, c_y, w, a, \log\nu)$ (the five generating parameters; CoLoRA-D needs the parameter,
the paper's method and CROM start from $u_0$ instead — declared difference, as in the template's arm L), $t$ the 26 levels,
$X$ the $257^2$ node coordinates; (ii) batching — 16 snapshots per step drawn as a fresh random permutation each epoch,
gathered from the training array held on the GPU (the released `Dataset` `array_split` copies the whole array, which does
not fit for 113k snapshots of 66k nodes); (iii) a validation monitor every 25k steps (logged only, never used to stop).

Configurations (all at the released learning rate $5\times10^{-3}$; latent size $k$ = number of `alpha`s):
- `D7`: the released Burgers configuration exactly: `u_layers = P + 7 C`, width 25, rank 3, `full=False`,
  `h_layers = 3 D`, width 15; $k = 7$; 100,000 steps (released) — reported as "authors' default".
- `D7-600k`: the same at 600,000 steps.
- $k = 16$ (the paper's head latent size): `P + 8 C`, rank 2, `full=True` (two alphas per layer); 600,000 steps;
  (decoder width, hyper width) $\in \{(25, 15), (64, 64), (128, 64)\}$ → `K16-w25`, `K16-w64`, `K16-w128`.
- `K16-deep`: `P + 16 C`, width 25, rank 3, `full=False` (one alpha per layer, the paper's convention), h width 15; 600,000 steps.

Selection ($k=16$ row): lowest validation worst-case error (float64 evaluation) among the four $k=16$ configurations.
`D7` and `D7-600k` are evaluated on test as well, as the "authors' default" rows (no selection among them beyond
reporting both). Online: one hyper-network call on $(\mu, t_k)$ for the five output times, one decoder pass over the
grid; evaluated in float64 (parameters cast; `jax_enable_x64`) and float32.

## 5. CROM

### 5.1 Offline (manifold construction), released code unchanged except the data module

`cromnet.CROMnet` (conv1d encoder over the point list, ks 6, stride 4; ELU MLP decoder on $[\hat x, x]$ with five hidden
layers of width `i_dim * scale_mlp`; per-point MSE ×1000; Adam; explicit step-wise LR schedule `LambdaLR`),
`callbacks` (checkpoint, TorchScript export), Lightning `Trainer` with the released arguments. Adapter
`code/crom/ours_data.py` serves our snapshots in the released item format (`{'x','q','encoder_input',…}`, $x$ = node
coordinates, $q$ = field, all $257^2$ nodes) with the released ordering (simulations shuffled with `random.seed(0)`,
`shuffle=False`) and the released standardisation statistics. `code/crom/run_ours.py` = `run.py` with only the data
module import and `num_workers` 64 → 8 changed (`diffs/crom_run.py.diff`). float32 training (torch 1.11 default,
TF32 matmuls on Ampere as released).

Hyper-parameters: from the released Burgers checkpoint (`weights/Burger/epoch=3099-step=40300.ckpt`): ELU (no SIREN),
batch 16, ks 6, stride 4, `scale_mlp` 64, LR multipliers $[10,5,2,1,0.5,0.2,0.1,0.05,0.01,0.005,0.001,5\cdot10^{-4},10^{-4}]$
on `initial_lr` $10^{-4}$ (code default) with stage lengths proportional to the checkpoint's
$[100,100,200,200,200,200,300,\dots,300]$ epochs. One of our epochs is 7,072 steps (113,152 snapshots); the budget is
31 epochs (stages $[1,1,2,2,2,2,3,3,3,3,3,3,3]$), ≈219,000 steps of 16 full snapshots ≈ $2.3\times10^{11}$ point
evaluations (about 11× the re-implementation's $600{,}000 \times 64 \times 512$). Latent `lbl` = 16.
Configurations: `scale_mlp` $\in \{64$ (released), $128\}$ → `C16-s64` (width 128), `C16-s128` (width 256);
plus `C16-s20` (README example value, width 40). If a configuration's measured throughput projects beyond 20 h, its
epoch count is reduced by an amendment before its outcome is seen.

### 5.2 Online (latent dynamics), released code

Unchanged and imported: `online/Decoder.py`, `online/Encoder.py`, `online/NonlinearSolver.py` (Gauss-Newton with Armijo
backtracking), the released TorchScript-exported encoder/decoder. New: `code/crom/burgers_ours.py`, an `Experiment`
subclass written after the released `Experiments/Burger.py` (same method names and roles), whose spatial operator is our
discretisation (`burgers_fom.spatial`: sign-upwind advection, five-point Laplacian, zero Dirichlet), and
`code/crom/online_ours.py`, the released `online.py` loop for `exp == 'burger'` (encoder → GN warm start against
$u_0$ with 20 iterations → per step residual, target, GN projection) without file writing, returning the five output
fields. Sampling: `full` (as the released Burgers example).

Two arms:
- **CROM-E (as released):** explicit — per substep `q_target = q_sample + dt_sub * f(q)`, $f(q) = -(\text{adv}(q) - \nu\,\text{lap}(q))$,
  then `NonlinearSolver.solve` (GN). Substeps: $n_{\rm sub} = \lceil 0.005\,(4\nu L^2 + 2\max|u_0| L)/0.9\rceil$ per
  0.005 interval (forward-Euler stability of this operator; a fixed rule, not tuned).
- **CROM-I (connected to our implicit reference; a deviation, reported as a separate row):** per $\Delta t = 0.005$
  step, the released GN solver (same iteration, line search and stopping rule) minimises the backward-Euler residual of
  the reference solver, $q - q_n + \Delta t(\text{adv}(q) - \nu\,\text{lap}(q))$, over the latent, instead of the distance
  to an explicit target. Only `computeLossAndR` and the Jacobian call are swapped (subclass), the residual's Jacobian is
  exact (decoder Jacobian from the released `jacobianPart`, chain rule through the stencil). Reason: the released
  integrator cannot match a backward-Euler reference (5.3).

Knob grid (validation, 32 trajectories, float64), per trained decoder:
- CROM-E: GN stopping (diff, step, max iters) $\in$ {Burgers example $(10^{-15}, 10^{-15}, 50)$; `online.py` defaults
  $(10^{-6}, 10^{-3}, 50)$; defaults with max 5}.
- CROM-I: the `online.py` default thresholds with max iters $\in \{2, 5, 50\}$; sampling full.
- float32 twin of each arm's most accurate setting.
A setting whose projected wall time for the 32 cases exceeds 6 GPU-hours is run on the first 8 validation cases only and
labelled; it is selectable only if no fully evaluated setting exists for that arm.

Selection per arm (rule of the template, section 5): **accurate** = lowest validation worst-case error (ties within 5 %:
fastest); **fast** = fastest with validation worst-case $\le 3.85\,\%$, if any and different. Float32 twin carried if its
validation worst is within 5 % of the float64 one.

### 5.3 Integrator floor (network-free, validation; run before this protocol, job 7156929)

Forward Euler with the substep rule above on the exact full-order field, against the backward-Euler reference, 32
validation trajectories: worst 2.95 %, median 0.82 %. This is a lower bound on CROM-E's error on this benchmark
independent of training, and is reported beside the CROM-E rows.

## 6. Timing (template section 6, unchanged)

One A100 80GB, batch 1, one process; timed region from $u_0$ (CROM) or $\mu$ (CoLoRA) on the GPU to the five output
fields on the GPU; 20 warm-up queries then 5 repetitions × 64 cases; median, p5, p95, min, max. Settings slower than 1 s
per query: 3 warm-up queries, 1 repetition (template A4); slower than 60 s per query: the evaluation pass itself is
timed (1 warm-up case, each of the 64 cases timed once). CROM runs in PyTorch eager mode as released (no compilation);
CoLoRA under `jax.jit`. Offline cost: GPU-hours per run, steps, GPU type.

## 7. Comparison rows (stored, not re-run)

Re-implementations (`nfrom-baseline`, test64, A100 80GB): CROM-style k=16 accurate 22.4 % / 0.43 % / 2,818 ms,
fast 16.9 % / 1.11 % / 507 ms; CoLoRA-style hyper-network k=16 0.240 % / 0.104 % / 49.9 ms. Paper: span $R'=384$
0.875 % / 0.026 % / 45.2 ms; span $R'=96$ 3.85 % / 0.529 % / 14.2 ms; head k=16 4.10 % / 0.67 % / 32.4 ms. The verdict
statements of the template's section 7 are applied to each official row, and each official row is stated as better or
worse than its re-implementation counterpart on worst error, median error and time.

## Amendments
