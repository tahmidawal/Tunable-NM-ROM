# Burgers 3D — the sample-free tensor ROM on a new non-negative multi-blob family, cost ladder vs two classical solvers at 32 / 64 / 128 cells per side

Design **r2**, 2026-09-03. Branch `exp/2026-09-03-burgers3d-tensor` (worktree
`worktrees/2026-09-03-burgers3d-tensor`, cut from `exp/2026-08-29-b2d-tensor`). r1 is kept as
`B3D-DESIGN-r1-superseded.md`; the r1 audit (33 items, Codex `gpt-5.6-sol`) is
`B3D-DESIGN-AUDIT-r1-codex-gpt56sol.md` and every item is folded in below, marked `[A<n>]`. r2 is
itself audited before any code is written. Every number in a results table is generated from run
JSONs; thresholds are frozen once phase 0 closes; any later change is a numbered retraction in
`B3D-NOTES.md`. Scope of every conclusion: **seed 0, test seed 1, eight test trajectories, one
GPU per resolution** `[A32]`.

## What this cell is FOR — read this before the gates

The 2D tensor cell (`TENSOR2D-NOTES.md`, `reports/2026-08-30-b2d-tensor-ladder.md`) showed that
with a separable decoder the projected Burgers advection is an exact quadratic in the head output
on non-negative fields, so the residual needs no quadrature; the tensor arm matched the full-grid
oracle to five digits at a grid-free cost. It could not show a cost story: on a $1024^2$ grid the
matched-accuracy speedup at the finest rung was $3.45\times$ against a preconditioned Newton
solve, and the FOM won below that.

In 3D the FOM has 64× more unknowns at 128 cells per side than at 32 `[A6]`; how its cost grows is
empirical. The ROM's latent solve has $N$-independent reduced dimensions. This cell asks four
predeclared questions on a **new** family (one to three blobs):

1. **Oracle-equivalence.** Does the tensor arm follow the full-grid oracle in 3D, at every LM
   candidate, where decoded fields undershoot zero at more points than in 2D? (Not "exactness":
   that word is reserved for the algebraic identity on the non-negative cone `[A1, A18]`.)
2. **Reduced cost.** Is the latent solve's cost independent of $N$ when all three checkpoints run
   on one GPU `[A27]`, and where does the paired comparison against the **strongest** classical
   arm cross 1, with an accuracy bracket and an uncertainty rule `[A28–30]`?
3. **Accuracy.** What is the decoder's held-out representation oracle, and how far above it does
   the ROM land `[A21, A22]`?
4. **Completion.** Do the rollouts finish at first-order optimality, or do they stall `[A20]`?

Not claimed: any cross-$N$ scaling exponent; sign-changing data; multi-seed statistics.

## Governing equations

$$
u_t + u\,(u_x + u_y + u_z) = \nu\,\Delta u \ \text{on } (0,1)^3,\quad u|_{\partial(0,1)^3} = 0,\quad
\log\nu \sim \mathcal U(\log 0.01, \log 0.1)\ \text{per trajectory}.
$$

### The family — non-negative, one to three blobs, smoothly masked `[A7–A10, A12]`

One raw parameter table of 576 + 8 rows is drawn **once** from `numpy.random.default_rng(seed)`
and persisted (`b3d_params_seed<seed>.npz`, sha256 in the JSON); every $N$ takes exact row prefixes
`[A12]`. Row $j$: $B_j \sim \mathcal U\{1,2,3\}$; for $b = 1..3$ (always three, so the stream does
not depend on $B$): $c_b \sim \mathcal U(0.2, 0.8)^3$, $w_b \sim \mathcal U(0.10, 0.20)$,
$\rho_b \sim \mathcal U(0.5, 1.0)$; $A \sim \mathcal U(0.5, 2.0)$; $\log\nu$ as above.

$$
s(x) = m(x) \sum_{b=1}^{B} \rho_b \exp\!\Big(-\frac{\|x - c_b\|^2}{2 w_b^2}\Big),\qquad
m(x) = 64\,x(1-x)\,y(1-y)\,z(1-z),\qquad u_0(x) = A\,\frac{s(x)}{s^\star},
$$

with $s^\star = \max s$ evaluated on a **fixed reference grid of 257 nodes per side**, so the
normalisation is the same at every $N$ `[A7]`. The smooth mask $m$ removes the wall jump of the
2D family `[A8]`; $w \ge 0.10$ gives at least 3.2 cells per width at the coarsest mesh `[A10]`.
The $B = 1$ rows are the single-blob 3D subfamily, not a lift of 2D `[A9]`. Generic maximum
intrinsic dimension with time: $5B + 2 \le 17$; coincident blobs lower it, and the head width is
chosen by the pilot, not by this count `[A11, A23]`.

Non-negativity of the truth: at a converged root of the backward-Euler sign-upwind system a
negative global minimum is impossible (there the forward branch gives $u\,\nabla u \le 0$ and the
discrete Laplacian is $\ge 0$, contradicting $u^{n+1} < 0 \le u^n$) `[A5]`. Gate F5 checks the
**solver output** at finite tolerance; Newton iterates are not claimed non-negative.

## Discretization — interior-only unknowns `[A4]`

Nodes $N \in \{33, 65, 129\}$ per side (32 / 64 / 128 cells, $\Delta x = 2^{-5}, 2^{-6}, 2^{-7}$,
**nested**, so the spatial-consistency gate compares on common nodes). Unknowns are the interior
$n_i = (N-2)^3$ values in flat order $i(N-2)^2 + j(N-2) + k$; walls are fixed ghost zeros and never
enter the Newton system. Backward Euler, $\Delta t = 0.005$, 50 steps, $T = 0.25$, 51 slices.
Interior residual with $u_c$ the centre value:

$$
\begin{aligned}
\text{lap} &= \frac{u_{x+} + u_{x-} + u_{y+} + u_{y-} + u_{z+} + u_{z-} - 6u_c}{\Delta x^2},\qquad
u_x = \begin{cases}(u_c - u_{x-})/\Delta x & u_c > 0\\ (u_{x+} - u_c)/\Delta x & \text{else}\end{cases}
\ (\text{likewise } u_y, u_z, \text{ same switch}),\\
R(u; u^{\text{prev}}, \nu) &= u_c - u^{\text{prev}}_c + \Delta t\,\big(u_c (u_x + u_y + u_z) - \nu\,\text{lap}\big).
\end{aligned}
$$

**Sine modes.** $\Phi_{pqr} = \phi_p \otimes \phi_q \otimes \phi_r$, $p,q,r = 1..N-2$,
$\phi_p(i) = \sqrt{2/(N-1)}\,\sin(\pi p i/(N-1))$ (orthonormal DST-I). They are exact eigenvectors
of the ghost-zero 7-point Laplacian with $L\Phi = -\Phi\Lambda$,
$\lambda_{pqr} = \frac{4}{\Delta x^2}\sum_{\alpha \in \{p,q,r\}} \sin^2\frac{\pi\alpha}{2(N-1)}$ `[A2]`.

**Truth generator.** Newton on the interior system, at most 8 iterations, matrix-free BiCGStab
($Jv$ by `jax.jvp`, `lin_tol = 1e-10`, `maxiter = 2000`) preconditioned by the exact Helmholtz
inverse $H_\nu^{-1} = S\,(1 + \Delta t\,\nu\,\lambda)^{-1} S^{\mathsf T}$, applied as a forward and an
inverse 3D DST, each three one-axis orthonormal transforms (six axis transforms per application)
`[A3]`. Two DST implementations, dense one-axis matmul and FFT of the odd extension, are both
built; gate F9 asserts they agree and records which is faster per $N$ `[A28]`. Step acceptance as in
2D (finite step; skip once $\|R\| \le 10^{-12}\|u^{\text{prev}}\|$). Every accepted trajectory:
$\max_k \|R(u_{k+1}; u_k)\|/\|u_k\| \le 10^{-8}$ (gate F3).

**Classical arms for the cost comparison `[A28, A29]`.** Two ladders, same rungs, same GPU, same
job, tolerance-terminated on $\|R\| \le \text{ntol}\,\|u^{\text{prev}}\|$:

| arm | one iteration | rungs |
|---|---|---|
| `newton` | Newton step, BiCGStab preconditioned by $H_\nu^{-1}$ (the 2D ladder) | ntol $\in \{1, 3\text{e-}1, 1\text{e-}1, 3\text{e-}2, 1\text{e-}2, 3\text{e-}3, 1\text{e-}3, 1\text{e-}4\}$ × lin_frac $\in \{0.05, 0.5\}$ |
| `picard` | defect correction $u_{k+1} = u_k - H_\nu^{-1} R(u_k)$, i.e. one stencil evaluation and one DST pair; $u_0$ = linear extrapolation from the two previous slices | ntol as above; plus the zero-work rung "extrapolation only" and the one-iteration rung |

`picard` exists because with an exact Helmholtz inverse it is the obvious cheaper competitor; the
matched comparison is made against **whichever arm is cheaper at the matched rung**, and the
report names it. The ladder must **bracket** the ROM: at least one rung less accurate and one more
accurate than each ROM arm, else the comparison is reported as Pareto dominance only, never as
"matched" `[A29]`.

Cohorts: 512 train + 64 val (rows 0–575 of the table), 8 test (test seed 1), at $N = 33$ and $65$.
At $N = 129$ the training cohort is the largest of 512 / 256 / 128 whose streamed generation the
phase-0 timing projects to fit 8 h, fixed and recorded before any ROM run.

## Decoder

$$
u(x; z) = \text{bc}(x)\,\langle g(x), h(z)\rangle,\qquad \text{bc}(x) = m(x) = 64\,x(1-x)\,y(1-y)\,z(1-z),
$$

$g$: random Fourier lift $B \in \mathbb R^{3 \times 64}$, scale 4.0, fixed; SiLU MLP
$[128, g_h, g_h, R]$ with $g_h = 2R$ (bank not rank-capped); $h$: SiLU MLP $[K, 128, 128, R]$ plus
linear skip.

**Capacity pilot at $N = 33$ `[A23]`.** Two frozen configurations, (K, R, M, m) = (16, 64, 128,
512) and (32, 128, 256, 1024), trained by the same recipe; the promoted one is the **smaller** if
its held-out oracle (gate D4) is within 1.2× of the larger's and it passes D3 and the M-stability
check (residual norms at the training codes change by $< 5\%$ when $M$ is doubled); otherwise the
larger. One configuration is then used at every $N$. Tensor size $M R^2 \cdot 8$ B ≤ 32 MiB.

**Training `[A24]`.** Auto-decoder, global relative MSE plus $10^{-4}$ feature-Gram term, Adam with
warmup-cosine, 60 000 steps, lr $10^{-3}$, **explicit-argument trainer** (`train_autodecoder_v2`)
with **the same sampling measure at every $N$**: 16 384 interior points drawn iid per step, no
full-grid finishing steps (`full_last = 0`). The training point pool is the whole interior at
$N = 33, 65$ and a fixed seeded subset of $63^3 = 250\,047$ interior points at $N = 129$ (the
$N = 65$ interior count `[A24]`); the pool is only what the per-step draw samples from. 8192
snapshots by the `sep_burgers.py` pick rule at $N \le 65$; at $N = 129$ the pick is the largest of
8192 / 4096 / 2048 that fits the measured device memory (phase-0 pilot, gate M1). Generalisation
from the pool to the full grid is **gated**, not recorded: D4 computes the held-out oracle on the
full grid and on the pool and requires their ratio $\le 1.5$.

One decoder per $N$. Recorded only: the $N = 65$ decoder on the $N = 129$ grid (tensor arm).

## ROM arms — identical solver, three residuals

Exact-linear residual with $A = \Phi^{\mathsf T} G_{\text{int}}$ ($M \times R$) and $\lambda$ the
eigenvalues of the chosen modes:

$$
r_w(z) = w \odot \Big[A\big(h(z) - h(z_n)\big) + \Delta t\,\big(q(z) + \nu\,\lambda \odot A h(z)\big)\Big],
\qquad w = (1 + \Delta t\,\nu\,\lambda)^{-1},
$$

| arm | $q(z) = \Phi^{\mathsf T} N(u(z))$ | cost in $n$ |
|---|---|---|
| `full` | sign-upwind stencil on the full interior grid (the full-quadrature oracle) | $O(n)$ |
| `ex` | advection sampled at $m = 4M$ NNLS nodes fitted on advection rows only | $O(m)$ |
| `tensor` | $\tfrac12 h^{\mathsf T} Q h$, $Q = T + T^{(j\leftrightarrow k)}$, $T_{ijk} = \sum_x \Phi_{xi} G_{xj} (DG)_{xk}$, $DG = D^-_x G + D^-_y G + D^-_z G$ | $O(M R^2)$, grid-free |

One tensor covers all three axes; $T$ is built in f64, blocked over $x$ with the
$(\text{chunk}, R^2)$ intermediate at 512 MiB; the $n_i \times R^2$ product is never
materialised. IC fit (encoder init + Gram-space LM), LM step (`make_step_lspg_var` replica,
`STALL = 1e-3`, `TR_FACTOR = 0.01`, `STEP_TOL = 1e-9`, budget 30), warm-start extrapolation
(`EXTRAP = 1.0`) and full-grid decode are the 2D cell's, byte for byte. **Every completed step also
records the first-order optimality $\|J^{\mathsf T} r\|/(\|J\|\,\|r\|)$ at its accepted latent; a step
with optimality $> 10^{-4}$ is censored** `[A20]`.

**Same-GPU kernel job `[A27]`.** After the three per-$N$ jobs, one job on one GPU runs the three
tensor kernels (each needs only $A$, $\lambda$, $Q$ and the head: no bank, no grid) on the same
eight initial latents, interleaved, so the reduced-cost claim is made on one card.

## Gates — every one with a pass rule, predeclared `[A13–A17]`

Rules: no absolute threshold on a mesh-scaling quantity; every asserted gate has a negative
control that fires on real data at $N = 33$ **after** $t = 0$ and on admissible initial data
`[A14]`, with the witness state and the crossed threshold recorded; NaN anywhere is FAIL; a
control that cannot fire is a defect. Backward-error normalisation wherever an operator is
applied `[A15]`: for $y \approx Op\,x$ the discrepancy is divided by $\|Op\|_\infty \|x\| + \|y\|$.

### Phase 0 — FOM

| gate | what | pass | control (must fire) |
|---|---|---|---|
| F1 axis symmetry | a 3-blob IC symmetric under all axis permutations, 50 steps; compare $u(t)$ to $Pu(t)$ for $t \ge 1$ | $\max \|u - Pu\|/\|u\| \le 10^{-12}$ | the $z$-advection coefficient scaled by 1.01 in the stencil: $> 10^{-4}$ |
| F2 modes are eigenvectors | $L$ assembled with scipy sparse from the 7-point rule, independent of the stencil code | $\|L\Phi + \Phi\Lambda\|_F / (\|L\|_\infty \|\Phi\|_F + \|\Phi\Lambda\|_F) \le 10^{-14}$ | continuum $\pi^2(p^2+q^2+r^2)$: $> 10^{-3}$ |
| F3 truth acceptance | every step of every trajectory | $\max \|R(u_{k+1};u_k)\|/\|u_k\| \le 10^{-8}$ | the generator with 2 Newton iterations on the first test trajectory: $> 10^{-8}$ (the 8-iteration output is then the pass witness) |
| F4 preconditioner | $y = H_\nu^{-1} v$, then $\|H_\nu y - v\| / (\|H_\nu\|_\infty \|y\| + \|v\|)$ on random interior $v$, $H_\nu$ applied by the assembled $L$ | $\le 10^{-14}$ | $\nu$ replaced by $2\nu$ inside $H^{-1}$: $> 10^{-2}$ |
| F5 non-negativity | train, val and test truth, all $t \ge 1$ | $\min u \ge -10^{-9}$ | the first test IC rolled with the upwind switch **inverted** (downwind stencil) at $\nu = 0.01$: $\min u < -10^{-3}$ for $t \ge 1$ (smoke at $N = 17, 33$: fires by four orders; a central-difference control was tried first and did NOT go negative at cell Péclet 6, so it is not used) |
| F6 stencil vs assembled operator | $R$ from the stencil code vs $u - u^{\text{prev}} + \Delta t(u \odot D^- u - \nu L u)$ assembled with scipy, on a non-negative truth state | $\le 10^{-13}$ backward-normalised | on a state with negative points, the two differ $> 10^{-6}$ |
| F7 spatial consistency | one smooth IC ($w = 0.2$, $B = 1$) at $N = 33, 65, 129$ on the common $33^3$ nodes, at $t = T$; observed order from the two differences | order in $[0.8, 1.3]$ (first-order upwind) | the $N = 65$ solution compared with the wrong-$\nu$ $N = 129$ solution: order outside the band |
| F8 2D-vs-3D consistency `[A17]` | 3D state $v(x,y)\,p(z)$ with $p = 1$ on the three central planes tapering to 0 at the faces; on the middle plane the $z$ terms vanish exactly, so the 3D residual and its JVP there must equal the actual 2D code's (`burgers2d_film.residual`) | $\le 10^{-13}$ backward-normalised | the $z$-Laplacian coefficient scaled by 1.01: $> 10^{-4}$ |
| F9 DST implementations | matmul DST vs FFT-DST on random fields | $\le 10^{-13}$; faster one recorded per $N$ | none (two independent implementations) |
| F10 generator cost | seconds per trajectory at each $N$; fixes the $N = 129$ cohort | recorded | none |
| M1 memory pilot `[A25, A26]` | at $N = 65$: device peak (`jax` memory stats) and host RSS for every phase; the whole pipeline's wall time by phase | recorded; the $N = 129$ job is submitted only if the projected wall time $\le 20$ h and projected device peak $\le 120$ GB | none |

### Phase 1 — decoder and residual (per $N$)

| gate | pass | control |
|---|---|---|
| D1 bank == meshfree | $\max|G_{\text{int}} h(z) - \text{dec}(z, x)| / \max|\cdot| < 10^{-12}$ | bank built at $x + \Delta x/2$: $> 10^{-3}$ |
| D2 lineage | first 64 regenerated picked states reconstructed by their codes, mean rel-L2 $< 0.2$ | shuffled codes: $> 0.5$ |
| D3 rank of $A$ | $\sigma_{\min}(A)/\sigma_{\max}(A) > 10^{-8}$, recorded with $M$-stability | none (recorded) `[A13]` |
| D4 held-out oracle `[A21]` | multi-start (8 starts) full-grid LM fit of the decoder to 32 held-out test states (8 trajectories × $t \in \{0, 10, 25, 50\}$): mean rel-L2 **$\le 5\times10^{-2}$**; pool-vs-full-grid ratio $\le 1.5$ | a decoder with shuffled bank rows: $> 0.5$ |
| L | `ex` linear part vs the **assembled** $\Phi^{\mathsf T}(u - u^{\text{prev}} - \Delta t\,\nu L u)$ through scipy `[A13]` | $\le 10^{-12}$ backward-normalised | diffusion dropped: $> 10^{-3}$ |
| A | `ex` advection at its nodes vs the stencil advection at the same nodes, computed directly | $\le 10^{-12}$ | central difference: $> 10^{-6}$ |
| FOMR | full-grid weak residual vs $w \odot \Phi^{\mathsf T} R_{\text{FOM}}$ from the generator's residual | $\le 10^{-10}$ | $w$ omitted: $> 10^{-3}$ |
| STEP / ROLL | aux-threaded step and rollout bit-identical to the shared solver | $= 0.0$ | stall $10^{-1}$ on a preselected witness state where the LM path has $\ge 2$ accepted steps: $\ne 0$ `[A13]` |

### Phase 2 — tensor (per $N$)

| gate | pass | control |
|---|---|---|
| TB build order | two chunkings | $\max|T_1 - T_2| / \max|T_1| < 10^{-13}$ (recorded with the chunk count) | none |
| TA algebraic identity | over all training head outputs, $\tfrac12 h^{\mathsf T}Qh$ vs $((Gh) \odot (DGh))^{\mathsf T}\Phi$ | $< 10^{-13}$ | $\tfrac12 h^{\mathsf T} T h$ (unsymmetrised): $> 10^{-1}$ |
| T0-truth `[A16]` | on the **truth** snapshots (non-negative by F5): $\Phi^{\mathsf T} N_{\text{upwind}}(u) = \Phi^{\mathsf T}(u \odot D^- u)$ | $< 10^{-13}$ | the same on a sign-changing field (blob minus displaced blob): $> 10^{-3}$ |
| T0-decoded | tensor vs oracle on all-positive decoded training states | **recorded** with the count; zero states is reported, not bypassed |
| TQ | 32 latent states, fresh $\nu$: $r$, $J$, $J^{\mathsf T}r$ mismatch, `min_u`, `n_neg` | recorded |
| TR candidate-path audit `[A19]` | host-loop LM rollout of the `tensor` arm on 2 test trajectories; at **every** candidate (initial, trial, accepted, rejected): tensor-vs-oracle $r$, $J$, $J^{\mathsf T}r$ mismatch, sign counts, and whether the oracle would have made the same accept/reject decision | recorded; decision agreement $< 100\%$ is reported per step |

### Phase 3 — the ladder and the decision (per $N$)

| gate | pass |
|---|---|
| E1 oracle-equivalence `[A18]` | `tensor` vs `full`, per test trajectory: worst per-state field rel-diff $\le 10^{-3}$, latent dev $\le 10^{-6}$, $|\text{err ratio} - 1| \le 10^{-2}$, identical stop histograms and attempt counts |
| E2 | `tensor` vs `ex`: recorded |
| P1 completion `[A20]` | every step of every arm: optimality $\le 10^{-4}$; censored steps $= 0$ for a result row, else the row is "incomplete" |
| C1 reduced cost `[A27]` | same-GPU kernel job: latent-solve ms for the three checkpoints, interleaved; within-job ratios in the per-$N$ jobs recorded with the GPU named |
| C2 crossover `[A28–30]` | paired AB/BA of the `tensor` e2e vs the cheaper classical arm at the matched rung, `PAIR_REPS = 5`; bracket required; a speed win needs per-trajectory speedup $> 1$ on **all 8** trajectories and median $> 1.1$; raw times, outlier counts and the trajectory-clustered 5th percentile reported |
| A1 accuracy `[A21, A22]` | held-out oracle (D4) and the ROM's excess over it, reported separately; "within 3× of its oracle" is the phrase, never "on its floor" |
| A2 | decoded positivity fractions: recorded |

## Frozen contract

`N ∈ {33, 65, 129}`, (K, R, M, m) from the pilot, `g_hidden = 2R`, `dt = 0.005`, `steps = 50`,
`SEED = 0`, `TEST_SEED = 1`, `N_TEST = 8`, cohorts as above, `STEPS = 60000`, `LR = 1e-3`,
`p_sub = 16384`, `full_last = 0`, `POS_TOL = 1e-9`, `STALL = 1e-3`, `TR_FACTOR = 0.01`,
`EXTRAP = 1.0`, `STEP_TOL = 1e-9`, `BURN = 2`, `TIME_REPS = 5`, `FOM_REPS = 5`, `PAIR_REPS = 5`,
`T_CHUNK` = 512 MiB / $R^2$ rows, f64, `JAX_DEFAULT_MATMUL_PRECISION=highest`. Jobs: `gpu`
partition, `/cluster/tufts/paralab/tawal01/b3dtensor/{n33,n65,n129,kernels}/`; $N = 33, 65$ on
A100-80G (`--mem 160G`), $N = 129$ on H200 (`--mem 240G`, `--time 24:00:00`, submitted only after
M1); `jax_backend=gpu` preflight, `MANIFEST.sha256`, `RESULTS.sha256`, pulled then deleted.

## What the cell decides `[A31, A33]`

Preconditions for **any** result row: phase 0 and 1 gates pass with their controls fired; D4
oracle $\le 5\times10^{-2}$ and pool-to-grid ratio $\le 1.5$; T0-truth pass; P1 zero censored
steps; ladder bracket present; C1 measured on one GPU.

| outcome | reading |
|---|---|
| preconditions met; E1 pass at all $N$; C2 speed win at $N = 129$ against the cheaper classical arm; A1 within 3× | the sample-free tensor ROM is oracle-equivalent, its reduced cost is $N$-independent on one GPU, and it beats the stronger of two classical 3D solvers at 128 cells per side, at the stated accuracy, on one seed and eight trajectories |
| as above but A1 $> 3\times$ | oracle-equivalent and fast, but the ROM lands well above what the decoder can represent: an objective / test-space result; cost is labelled "at the ROM's own error" |
| E1 pass, C2 no win at $N = 129$ | the classical solver with an exact Helmholtz inverse is still cheaper at 2M unknowns; report the within-job ratios and the same-GPU kernel cost; no speed claim |
| E1 pass at some $N$ only | oracle-equivalence is resolution-dependent; TR says where the candidates diverge; report per $N$ |
| E1 fail at any $N$ | the undershoot breaks the fixed-branch identity; TR locates it; the split central-difference form is the next cell |
| P1 censored steps $> 0$ in any arm | incomplete; no accuracy or cost row for that arm at that $N$ |
| D4 oracle $> 5\times10^{-2}$ or ratio $> 1.5$ | the decoder does not represent the family on the full grid; stop before residual work; report the pilot |
| no ladder bracket | Pareto dominance only; "matched" is not used |
| M1 projection exceeds the budget | $N = 129$ not run; the cell reports $N = 33, 65$ and the projection |
| any phase-0/1 gate fails or a control does not fire | fix, log as a numbered retraction, re-run that phase |

## Deliverables

- `b3d_common.py` (self-contained: grid, family table, FOM with both DSTs, `newton` and `picard`
  ladders, modes, upwind field, assembled operators for the gates, NNLS), `b3d_tensor_common.py`,
  `b3d_fom_gates.py` (phase 0), `sep_b3d_tensor.py` (phases 1–3), `sep_b3d_kernels.py` (C1),
  cluster `stage/push/pull_b3dtensor.sh`, `run_b3dtensor_{n33,n65,n129,kernels}.sbatch`,
  `runs/b3dtensor/gen_tables.py`.
- `B3D-NOTES.md` (retractions, generated tables); report `reports/2026-09-XX-b3d-tensor-ladder.md`
  on `main`; Codex verification of every conclusion and code file; lab log entry.
