# Burgers 3D — the sample-free tensor ROM on a new positive multi-blob family, cost ladder vs FOM at $N = 32, 64, 128$ per side

Design r1, 2026-09-03. Branch `exp/2026-09-03-burgers3d-tensor` (worktree
`worktrees/2026-09-03-burgers3d-tensor`, cut from `exp/2026-08-29-b2d-tensor`). To be audited by
an independent Codex pass before any code is written; every number in a results table is generated
from run JSONs; thresholds below are frozen once phase 0 closes, and any later change is a numbered
retraction in `B3D-NOTES.md`.

## What this cell is FOR — read this before the gates

The 2D tensor cell (`TENSOR2D-NOTES.md`, report `reports/2026-08-30-b2d-tensor-ladder.md`) showed
that with a separable decoder the projected advection of Burgers is an exact quadratic in the head
output, so the whole residual needs no quadrature, and that the tensor arm matches the full-grid
oracle to five digits at a cost that does not grow with the grid. What that cell could **not** show
is a compelling cost story: on a $1024^2$ grid the full-order model (FOM) is 1M unknowns and the
matched-accuracy speedup at the finest rung was $3.45\times$ against a preconditioned Newton solve;
below that the FOM wins.

In 3D the FOM has $N^3$ unknowns, so its cost climbs three orders of magnitude between $N=32$ and
$N=128$ while the ROM's latent solve, which never touches the grid, should stay where it is. This
cell asks three predeclared questions, on a **new** initial-condition family (up to three blobs
instead of one) so that the answer is not a rerun of the 2D data:

1. **Exactness.** Does the tensor arm still equal the full-grid oracle in 3D, where the decoded
   fields will undershoot zero at more points than in 2D (the tensor is exact only where the field is
   positive)?
2. **Cost.** Is the ROM's latent solve flat in $N$ from $32^3$ to $128^3$ unknowns, and where does the
   paired matched-accuracy comparison against the FOM tolerance ladder cross 1?
3. **Accuracy.** How far above its own reconstruction floor does the ROM land, with a head wide
   enough for the family's intrinsic dimension (which the 2D cell's $K=16$ was not, for a multi-blob
   family)?

What this cell does **not** claim: a cross-$N$ scaling exponent (each $N$ runs on its own GPU and
only within-job ratios are like-for-like, as in 2D); any result on sign-changing data (the upwind
stencil is not quadratic there; that is a separate roadmap item); multi-seed statistics (one seed,
eight test trajectories, exactly the 2D protocol).

## Governing equations

Scalar viscous Burgers on the unit cube with the non-conservative advection form the 2D and 1D
cells use, homogeneous Dirichlet walls:

$$
u_t + u\,(u_x + u_y + u_z) = \nu\,\Delta u \quad\text{on } (0,1)^3,\qquad u = 0 \text{ on } \partial(0,1)^3,
\qquad u(\cdot, 0) = u_0 .
$$

Viscosity is a per-trajectory parameter, $\log\nu \sim \mathcal U(\log 0.01, \log 0.1)$, exactly the
2D draw.

### The new family — positive, up to three blobs

For each trajectory draw, in this order from one `numpy.random.default_rng(seed)` stream:

- $B \sim \mathcal U\{1, 2, 3\}$ (number of blobs);
- for $b = 1..3$ (always three draws so the stream is independent of $B$): centre
  $c_b \sim \mathcal U(0.15, 0.85)^3$, width $w_b \sim \mathcal U(0.05, 0.20)$, relative amplitude
  $\rho_b \sim \mathcal U(0.5, 1.0)$;
- peak amplitude $A \sim \mathcal U(0.5, 2.0)$;
- $\log\nu \sim \mathcal U(\log 0.01, \log 0.1)$.

$$
s(x) = \sum_{b=1}^{B} \rho_b \exp\!\Big(-\frac{\|x - c_b\|^2}{2 w_b^2}\Big),\qquad
u_0(x) = A\,\frac{s(x)}{\max_{\text{grid}} s}\;\cdot\;\mathbb 1_{\text{interior}}(x).
$$

The rescaling to peak $A$ keeps the amplitude and Reynolds range identical to the 2D family
($aL/\nu \le 200$), so the regime is the same and only the **shapes** are new. The first blob's
draw is the 2D family lifted to 3D, so $B=1$ trajectories are the lifted family and appear in the
same cohort — no separate control arm is needed. Intrinsic dimension of the family including time:
$5B + 2 \le 17$; the head width $K$ is chosen above it (below).

Positivity: $u_0 \ge 0$ by construction, and the implicit sign-upwind scheme preserves
non-negativity (its Jacobian on a non-negative state is an M-matrix). This is asserted, not assumed
(gate F5).

## Discretization — fully specified

Uniform $N^3$ nodes, $\Delta x = 1/(N-1)$, flat index $i N^2 + j N + k$; interior
$n_i = (N-2)^3$ unknowns in interior flat order $i(N-2)^2 + j(N-2) + k$. Boundary nodes carry
residual rows $R = u$ (Dirichlet enforced in the residual, as in 2D).

Backward Euler, $\Delta t = 0.005$, 50 steps, $T = 0.25$, 51 stored slices (unchanged from 2D so the
regimes match). Interior residual, with $u_c$ the centre value and ghost zeros on all six faces:

$$
\begin{aligned}
\text{lap} &= \frac{u_{x+} + u_{x-} + u_{y+} + u_{y-} + u_{z+} + u_{z-} - 6u_c}{\Delta x^2},\\
u_x &= \begin{cases}(u_c - u_{x-})/\Delta x & u_c > 0\\ (u_{x+} - u_c)/\Delta x & \text{else}\end{cases}
\quad\text{and likewise } u_y, u_z \text{ switching on the same } u_c,\\
R &= u_c - u^{\text{prev}}_c + \Delta t\,\big(u_c (u_x + u_y + u_z) - \nu\,\text{lap}\big).
\end{aligned}
$$

Truth generator: Newton on $R(u; u^{\text{prev}}, \nu) = 0$, at most 8 iterations, matrix-free
BiCGStab with $J v$ by `jax.jvp`, `lin_tol = 1e-10`, `maxiter = 2000`, **with the exact Helmholtz
preconditioner in the 3D discrete sine basis from the start** (the 2D generator stalled at
$N \ge 512$ without it):

$$
M^{-1} v = S\,\Big(\frac{S^{\mathsf T} v}{1 + \Delta t\,\nu\,\lambda}\Big),\qquad
\lambda_{pqr} = \frac{4}{\Delta x^2}\Big(\sin^2\tfrac{\pi p}{2(N-1)} + \sin^2\tfrac{\pi q}{2(N-1)} + \sin^2\tfrac{\pi r}{2(N-1)}\Big),
$$

applied as three one-axis DST matmuls (the 3D DST is separable), identity on boundary rows. Newton
step acceptance guard as in 2D (finite step, and stop once $\|R\| \le 10^{-12}\|u^{\text{prev}}\|$).
Every accepted trajectory must satisfy $\max_k \|R(u_{k+1}; u_k)\| / \|u_k\| \le 10^{-8}$ or the job
aborts (gate F3).

The standardised FOM tolerance ladder for the cost comparison: tolerance-terminated Newton
(`MAX_NEWTON = 20`) with the same preconditioned BiCGStab, rungs
`newton_tol ∈ {3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4}` × `lin_frac ∈ {0.05, 0.5}`,
`lin_tol = max(newton_tol · lin_frac, 1e-12)` — the 14 rungs of the 2D cell, unchanged.

Cohorts: `N_TRAIN = 512`, `N_VAL = 64` from `SEED = 0` (validation = last 64 of the draw), 8 test
trajectories from `TEST_SEED = 1`, at $N = 32$ and $64$. At $N = 128$ the training cohort size is
**set in phase 0** from the measured generator cost (rule: the largest of 512 / 256 / 128 / 96
trajectories whose streamed generation fits 8 h on the target GPU), recorded before any ROM run, and
never changed after. The 2D cell at $N=1024$ used 96 and paid for it in decoder accuracy; this cell
records the choice rather than hiding it.

## Decoder — scalar separable, 3D coordinates

$$
u(x; z) = \text{bc}(x)\,\big\langle g(x), h(z)\big\rangle,\qquad \text{bc}(x) = 64\,x(1-x)\,y(1-y)\,z(1-z),
$$

$g$: random Fourier lift $B \in \mathbb R^{3\times n_{ff}}$, $n_{ff} = 64$, scale 4.0 (fixed), then SiLU
MLP $[128, g_{h}, g_{h}, R]$; $h$: SiLU MLP $[K, 128, 128, R]$ plus linear skip. Frozen widths:

| symbol | value | why |
|---|---|---|
| $K$ | 32 | above the family's intrinsic dimension 17; the 2D accuracy champion was $K=32$ |
| $R$ | 128 | bank width; $g_h = 256 \ge 2R$ so the bank is not rank-capped |
| $M$ | 256 | test modes; in 3D the 64 lowest modes reach only wavenumber $\approx 4$ per axis, 256 reaches $\approx 6$; $M = 2R$ keeps $A = \Phi^{\mathsf T}G$ overdetermined |
| $m$ | 1024 | NNLS nodes for the sampled arm, $m = 4M$ as in every earlier cell |

Tensor size $M R^2 \cdot 8$ B = 32 MiB. Bank $G_{\text{all}}$ at $N=128$: $2.1\text{M} \times 128 \times 8$ B
= 2.1 GB, passed as an explicit jit argument, never captured.

Test modes: the $M$ lowest 3D discrete sine modes ranked by discrete eigenvalue (ties broken
stably), triple outer product on the interior, each column normalised to unit 2-norm; they are
exact eigenvectors of the ghost-zero 7-point Laplacian (gate F2), which is what makes the diffusion
term exact.

Training: auto-decoder, global relative MSE plus the $10^{-4}$ feature-Gram orthonormality term,
Adam with warmup-cosine, 60 000 steps, lr $10^{-3}$, **explicit-argument trainer**
(`train_autodecoder_v2`, never a closure over the training array). Snapshot pick: 8192 states drawn
by the `sep_burgers.py` rule from the streamed truth. **Training point set:** all interior points
at $N = 32, 64$; at $N = 128$ a fixed seeded random subset of $262\,144$ interior points (the
$N=64$ interior count), so the training array is 17 GB as in the 2D $N=1024$ cell. The decoder is
meshfree, so the bank is still built on the full $N=128$ grid; gate D2 records the reconstruction on
the full grid versus on the training subset.

One decoder per $N$. Recorded, not gated: the $N=64$ decoder evaluated on the $N=128$ grid
(cross-resolution transfer, tensor arm only), because the coordinate decoder is resolution-flat
and this costs no training.

## ROM arms — identical solver, three residuals

All arms share the exact-linear residual of `sep_burgers_exlin.py` with $A = \Phi^{\mathsf T}G_{\text{int}}$
($M \times R$) precomputed and $\lambda$ the discrete eigenvalues of the chosen modes,

$$
r_w(z) = w \odot \Big[A\big(h(z) - h(z_n)\big) + \Delta t\,\big(q(z) + \nu\,\lambda \odot A h(z)\big)\Big],
\qquad w = (1 + \Delta t\,\nu\,\lambda)^{-1},
$$

and differ only in $q$:

| arm | $q(z) = \Phi^{\mathsf T} N(u(z))$ computed as | cost in $n$ |
|---|---|---|
| `full` | sign-upwind stencil on the full interior grid through the bank (the oracle) | $O(n)$ |
| `ex` | advection sampled at $m = 1024$ NNLS nodes fitted on advection rows only | $O(m)$ |
| `tensor` | $\tfrac12 h^{\mathsf T} Q h$, $Q = T + T^{(j\leftrightarrow k)}$, $T_{ijk} = \sum_x \Phi_{xi} G_{xj} (DG)_{xk}$, $DG = D^-_x G + D^-_y G + D^-_z G$ | $O(M R^2)$, grid-free |

One tensor covers all three axes. $T$ is built in f64, blocked over $x$ with `chunk` rows per block
so that the $(\text{chunk}, R^2)$ intermediate stays at 512 MiB ($\text{chunk} = 4096$ at $R = 128$);
the $n_i \times R^2$ product is never materialised (at $N=128$ it would be 275 GB).

IC fit (encoder init + Gram-space LM), LM step (`make_step_lspg_var` replica, `STALL = 1e-3`,
`TR_FACTOR = 0.01`, `STEP_TOL = 1e-9`, budget 30), warm-start extrapolation (`EXTRAP = 1.0`) and
full-grid decode are byte-for-byte the 2D cell's; gates STEP and ROLL assert bit-identity to the
shared solver. `ex_learned` is dropped (it bought nothing at this budget in 2D).

## Gates — every one with a pass rule, predeclared

Rules: no absolute threshold on a quantity that scales with the mesh (everything is relative);
every asserted gate has a negative control that must fire on real data at $N = 32$ before the
cluster jobs are submitted; NaN anywhere is FAIL; a gate that cannot fire is a defect, not a pass.

### Phase 0 — FOM (new code, self-contained `b3d_common.py`)

| gate | what | pass | control (must fire) |
|---|---|---|---|
| F1 axis symmetry | a 3-blob IC symmetric under every axis permutation, rolled 50 steps | $\max$ over permutations of $\|u - P u\|/\|u\| \le 10^{-12}$ | the same IC with one blob displaced off the diagonal: $> 10^{-3}$ |
| F2 modes are eigenvectors | assembled 7-point ghost-zero Laplacian $L$ (scipy sparse, independent of the stencil code) | $\|L\Phi + \Phi\Lambda\|_F / \|\Phi\Lambda\|_F \le 10^{-12}$ | $\Lambda$ replaced by the continuum $\pi^2(p^2+q^2+r^2)$: $> 10^{-3}$ at $N=32$ |
| F3 truth acceptance | every step of every trajectory | $\max \|R(u_{k+1};u_k)\|/\|u_k\| \le 10^{-8}$ | asserted; the control is F4 |
| F4 preconditioner exactness | $M^{-1}(I + \Delta t\,\nu\,(-L))v$ on random interior $v$ | $\|\cdot - v\|/\|v\| \le 10^{-12}$ | wrong $\nu$ ($2\nu$): $> 10^{-2}$ |
| F5 positivity | train, val and test truth | $\min u \ge -10^{-9}$ | a sign-changing IC (one blob negated) must produce $\min u < -10^{-2}$ |
| F6 stencil vs assembled operator | residual of the stencil code vs $R = u - u^{\text{prev}} + \Delta t(u \odot D^-u - \nu L u)$ assembled with scipy on a positive state | $\le 10^{-12}$ relative | on a state with negative points the two differ $> 10^{-6}$ (records that the switch is live) |
| F7 spatial consistency | one fixed IC solved at $N = 32, 64, 128$, compared on the common $32^3$ nodes | errors decrease with $N$, recorded with the observed order | none (recorded) |
| F8 generator cost | seconds per trajectory at each $N$ on the target GPU | recorded; fixes `N_TRAIN` at $N=128$ | none |

### Phase 1 — decoder and residual (per $N$)

| gate | pass | control |
|---|---|---|
| D1 bank == meshfree | $\max|G_{\text{int}} h(z) - \text{dec}(z, x)| / \max|\cdot| < 10^{-12}$ | a bank built at the wrong coordinates ($x + \Delta x/2$): $> 10^{-3}$ |
| D2 reconstruction | R-lite: first 64 regenerated picked states reconstructed by their codes, mean rel-L2 $< 0.2$ (asserted); full-grid vs training-subset reconstruction at $N=128$ (recorded) | a checkpoint with shuffled codes: $> 0.5$ |
| D3 rank of $A$ | $\sigma_{\min}(A)/\sigma_{\max}(A) > 10^{-8}$ | $M$ truncated to $R/2$ rows: rank-deficient |
| L | `ex` linear part == `full` linear part | $\le 10^{-12}$ | drop the diffusion term: $> 10^{-3}$ |
| A | `ex` advection at its nodes == the incumbent stencil advection at the same nodes, **computed directly, never as residual minus linear part** | $\le 10^{-12}$ | use the central difference instead of upwind: $> 10^{-6}$ |
| FOMR | full-grid weak residual == $w \odot \Phi^{\mathsf T} R_{\text{FOM}}[\text{interior}]$ from the generator's own residual function | $\le 10^{-10}$ | $w$ omitted: $> 10^{-3}$ |
| STEP / ROLL | aux-threaded step and rollout bit-identical to `sep_solvers.make_step_lspg_var` / `make_rollout_v2` | $= 0.0$ | different stall constant: $\ne 0$ |

### Phase 2 — tensor (per $N$)

| gate | pass | control |
|---|---|---|
| TB build order | two chunkings (chunk, reverse) | $\max|T_1 - T_2| / \max|T_1| < 10^{-14}$ | none possible beyond the two orders; recorded |
| TA algebraic identity | over all training head outputs, $\tfrac12 h^{\mathsf T}Qh$ vs $\big((Gh) \odot (DGh)\big)^{\mathsf T}\Phi$ with the fixed backward stencil | $< 10^{-13}$ | $Q$ not symmetrised ($T$ alone, evaluated as $h^{\mathsf T} T h$ is still exact; the control is $\tfrac12 h^{\mathsf T} T h$): $> 10^{-1}$ |
| T0 tensor == oracle on all-positive decoded states | $< 10^{-12}$, and the **count** of such states reported; if the count is 0 the gate is NOT TESTABLE and says so | on the non-positive states the mismatch is $> 10^{-8}$ (records the switch) |
| TQ solve-path fidelity (recorded) | 32 latent states (half training codes, half perturbed), fresh $\nu$: residual, Jacobian, gradient mismatch, `min_u`, `n_neg` | none |
| TR rollout fidelity (recorded, new) | along every test rollout of the `full` arm, per step: tensor-vs-oracle residual mismatch at the accepted latent | none; this is the honest per-step number |

### Phase 3 — the ladder and the decision (per $N$)

| gate | pass |
|---|---|
| E1 exactness in rollout | `tensor` vs `full`: $|\text{err ratio} - 1| \le 10^{-2}$ per $N$, identical stop histograms per trajectory, identical LM attempt counts |
| E2 tensor vs sampled | `tensor` vs `ex`: err ratio recorded; no pass rule (the 2D result was a tie) |
| C1 flat latent solve | within each job: `tensor` latent-solve ms / `full` latent-solve ms, and `tensor` ms itself; recorded, with the GPU named per row; no exponent fitted |
| C2 crossover | matched-accuracy paired speedup of `tensor` vs the FOM ladder at each $N$; predeclared prediction: $< 1$ at $N = 32$, $> 1$ at $N = 128$ |
| A1 accuracy | `tensor` rollout error vs the decoder's held-out reconstruction (the floor): ratio recorded; the claim "the ROM sits on its floor" needs ratio $\le 3$ |
| A2 positivity of decoded fields | fraction of decoded interior points $\le 0$ and of rollout states touching one: recorded (expected larger than 2D's 15–17 %) |

## Frozen contract

`N ∈ {32, 64, 128}`, `K=32`, `R=128`, `g_hidden=256`, `M=256`, `m=1024`, `dt=0.005`, `steps=50`,
`SEED=0`, `TEST_SEED=1`, `N_TEST=8`, `N_TRAIN=512/64` at $N \le 64$ (at 128: phase 0),
`MAX_SNAPS=8192`, `STEPS=60000`, `LR=1e-3`, `POS_TOL=1e-9`, `STALL=1e-3`, `TR_FACTOR=0.01`,
`EXTRAP=1.0`, `STEP_TOL=1e-9`, `BURN=2`, `TIME_REPS=5`, `FOM_REPS=5`, `PAIR_REPS=3`, `T_CHUNK=4096`,
f64 everywhere, `JAX_DEFAULT_MATMUL_PRECISION=highest`. Jobs: `gpu` partition, one directory per
$N$ under `/cluster/tufts/paralab/tawal01/b3dtensor/n<N>/`; $N=32$ and $64$ on A100-80G
(`--mem 160G`), $N=128$ on H200 (`--mem 240G`, `--time 24:00:00`); `jax_backend=gpu` preflight
(exit 42), `MANIFEST.sha256`, `RESULTS.sha256`, pulled then deleted.

## What the cell decides

| outcome | reading |
|---|---|
| E1 pass at every $N$, C2 $> 1$ at $N=128$, A1 $\le 3$ | the sample-free tensor ROM is exact, flat in $N$, and beats the FOM in 3D at the resolution where the FOM cost has climbed — the positive result the 2D cell could not deliver |
| E1 pass, C2 $> 1$, A1 $> 3$ | the ROM is exact and fast but not on its floor: a head-generalisation result, cost numbers are labelled "at the ROM's own error" as in 2D |
| E1 pass, C2 $\le 1$ at $N=128$ | the FOM with an exact Helmholtz preconditioner is still cheaper at 2.1M unknowns; report the crossover as not reached and the within-job cost ratios; no speed claim |
| E1 fail | the undershoot has grown enough that the fixed-branch tensor no longer follows the upwind oracle; TR tells where; the split central-difference form becomes the next cell |
| any phase-0 or phase-1 gate fails | fix and re-run phase 0 or 1 before anything else; a control that does not fire is a design defect and is logged as a retraction |

## Deliverables

- `b3d_common.py` (self-contained, on the `b1d_common.py` pattern: grid, family, FOM with the DST
  preconditioner, tolerance-Newton ladder, modes, upwind field, NNLS), `b3d_tensor_common.py`,
  `sep_b3d_tensor.py`, `b3d_fom_gates.py` (phase 0), cluster `stage/push/pull_b3dtensor.sh` and
  `run_b3dtensor_n{32,64,128}.sbatch`, `runs/b3dtensor/gen_tables.py`.
- `B3D-NOTES.md` with the numbered retractions and the generated tables; a report
  `reports/2026-09-XX-b3d-tensor-ladder.md` on `main` with tables generated from the JSONs.
- Codex verification of every conclusion and every new code file; the lab log entry.
