# Burgers 3D — the sample-free tensor ROM on a new non-negative multi-blob family, cost ladder vs two classical solvers at 32 / 64 / 128 cells per side

Design **r3**, 2026-09-03. Branch `exp/2026-09-03-burgers3d-tensor` (worktree
`worktrees/2026-09-03-burgers3d-tensor`, cut from `exp/2026-08-29-b2d-tensor`). r1 and r2 are kept
as `B3D-DESIGN-r{1,2}-superseded.md`; the two Codex `gpt-5.6-sol` audits are
`B3D-DESIGN-AUDIT-r{1,2}-codex-gpt56sol.md` (33 + 17 items). Every item is folded in, marked
`[A<n>]`. Every number in a results table is generated from run JSONs; thresholds are frozen when
phase 0 closes; any later change is a numbered retraction in `B3D-NOTES.md`. Scope of every
conclusion: **seed 0 (train/validation), test seed 1, eight test trajectories, one GPU per
resolution** `[A32]`. Time is written as the stored step index $k = 0..50$, $t = k\,\Delta t$ `[A50]`.

Code already written before r3 and covered by it: `b3d_common.py` (self-contained 3D module with
its **own explicit-argument trainer**, `[A49]`), `b3d_tensor_common.py`, `b3d_fom_gates.py`
(phase 0). The smoke runs that shaped r3's controls are recorded in `B3D-NOTES.md`.

## What this cell is FOR — read this before the gates

The 2D tensor cell (`TENSOR2D-NOTES.md`, `reports/2026-08-30-b2d-tensor-ladder.md`) showed that
with a separable decoder the projected Burgers advection is an exact quadratic in the head output
on non-negative fields, so the residual needs no quadrature; the tensor arm matched the full-grid
oracle to five digits at a grid-free cost. It could not show a cost story: on a $1024^2$ grid the
matched-accuracy speedup at the finest rung was $3.45\times$ against a preconditioned Newton
solve, and the FOM won below that.

In 3D the FOM has 64× more unknowns at 128 cells per side than at 32 `[A6]`; how its cost grows is
empirical. The ROM's latent solve has $N$-independent reduced dimensions. Four predeclared
questions on a **new** family (one to three blobs):

1. **Oracle-equivalence.** Does the tensor arm follow the full-grid oracle in 3D, at every LM
   candidate, where decoded fields undershoot zero at more points than in 2D? ("Exact" is reserved
   for the algebraic identity on the non-negative cone `[A1, A18]`.)
2. **Reduced cost.** Is the latent kernel's cost independent of $N$ on one GPU `[A27, A43]`, and
   where does the paired end-to-end comparison against the **cheaper of two classical arms** cross
   1, with an accuracy bracket and a clustered confidence bound `[A28–30, A42]`?
3. **Accuracy.** What is the decoder's held-out representation oracle, certified as converged, and
   how far above it does the ROM land `[A21, A22, A44]`?
4. **Completion.** Do the rollouts finish at first-order optimality `[A20]`?

Not claimed: any cross-$N$ scaling exponent; sign-changing data; multi-seed statistics.

## Governing equations and the family

$$
u_t + u\,(u_x + u_y + u_z) = \nu\,\Delta u \ \text{on } (0,1)^3,\quad u|_{\partial(0,1)^3} = 0,\quad
\log\nu \sim \mathcal U(\log 0.01, \log 0.1)\ \text{per trajectory}.
$$

**Two immutable parameter tables `[A12, A47]`**, each drawn once and persisted with a sha256:
`train` (seed 0, 576 rows: rows 0–511 training, rows 512–575 validation at every $N$; at $N=129$ the
training prefix is rows $0..N_{\text{train}}-1$ with $N_{\text{train}} \in \{512, 256, 128\}$ fixed by
phase 0, validation stays rows 512–575) and `test` (seed 1, 8 rows, opened only after the pilot has
promoted a configuration `[A41]`). Row $j$ consumes the same draws whatever its blob count:
$B_j \sim \mathcal U\{1,2,3\}$; for $b = 1..3$: $c_b \sim \mathcal U(0.2, 0.8)^3$,
$w_b \sim \mathcal U(0.10, 0.20)$, $\rho_b \sim \mathcal U(0.5, 1.0)$; $A \sim \mathcal U(0.5, 2.0)$;
$\log\nu$.

$$
s(x) = m(x) \sum_{b=1}^{B} \rho_b \exp\!\Big(-\frac{\|x - c_b\|^2}{2 w_b^2}\Big),\qquad
m(x) = 64\,x(1-x)\,y(1-y)\,z(1-z),\qquad u_0(x) = A\,\frac{s(x)}{s^\star},
$$

$s^\star = \max s$ on the fixed 257-node reference grid, which contains every experiment grid
`[A7, A36]`. **Disclosure `[A37]`:** $m$ is the decoder's own boundary factor, so the $k = 0$
states are architecture-aligned; D4 is therefore reported both overall and with $k = 0$ excluded,
and the family is called "easy at $k=0$ for this decoder" in the report. $w \ge 0.10$ gives 3.2 cells
per width at the coarsest mesh `[A10]`; $B = 1$ rows are the single-blob 3D subfamily `[A9]`.
Generic maximum intrinsic dimension with time $5B + 2 \le 17$; the table records, per row, the
minimum pairwise centre distance over the mean width (overlap diagnostic) `[A11]`.

Non-negativity of the truth: at a converged root of the backward-Euler sign-upwind system a
negative global minimum is impossible (forward branch gives $u\,\nabla u \le 0$, discrete Laplacian
$\ge 0$, contradicting $u^{n+1} < 0 \le u^n$) `[A5]`; F5 checks the solver output at $k \ge 1$.

## Discretization — interior-only unknowns `[A4]`

Nodes $N \in \{33, 65, 129\}$ (32/64/128 cells, $\Delta x = 2^{-5..-7}$, nested). Unknowns: the
interior $(N-2)^3$ values, flat order $i(N-2)^2 + j(N-2) + k$; walls are fixed ghost zeros. Backward
Euler, $\Delta t = 0.005$, 50 steps, $T = 0.25$, 51 slices.

$$
\begin{aligned}
\text{lap} &= \tfrac{1}{\Delta x^2}\big(u_{x+} + u_{x-} + u_{y+} + u_{y-} + u_{z+} + u_{z-} - 6u_c\big),\qquad
u_x = \begin{cases}(u_c - u_{x-})/\Delta x & u_c > 0\\ (u_{x+} - u_c)/\Delta x & \text{else}\end{cases}\ (u_y, u_z \text{ likewise, same switch}),\\
R(u; u^{\text{prev}}, \nu) &= u_c - u^{\text{prev}}_c + \Delta t\,\big(u_c (u_x + u_y + u_z) - \nu\,\text{lap}\big).
\end{aligned}
$$

**Sine modes.** $\Phi_{pqr} = \phi_p \otimes \phi_q \otimes \phi_r$, $p,q,r = 1..N-2$,
$\phi_p(i) = \sqrt{2/(N-1)}\sin(\pi p i/(N-1))$; $L\Phi = -\Phi\Lambda$,
$\lambda_{pqr} = \frac{4}{\Delta x^2}\sum_{\alpha} \sin^2\frac{\pi\alpha}{2(N-1)}$ `[A2]`.

**Truth generator.** Newton on the interior system, 8 fixed iterations with the skip guard,
matrix-free BiCGStab (`lin_tol = 1e-10`, `maxiter = 2000`) preconditioned by
$H_\nu^{-1} = S(1 + \Delta t\,\nu\lambda)^{-1}S^{\mathsf T}$ (a forward and an inverse 3D DST, six
one-axis transforms `[A3]`; matmul and FFT implementations, F9). Acceptance: every trajectory
$\max_k \|R(u_{k+1}; u_k)\|/\|u_k\| \le 10^{-8}$ (F3).

**Classical arms `[A28, A29, A34, A35]`.** Same rungs, same GPU, same job, tolerance-terminated on
$\|R\| \le \text{ntol}\,\|u^{\text{prev}}\|$:

| arm | one iteration | rungs |
|---|---|---|
| `newton` | Newton step, BiCGStab preconditioned by $H_\nu^{-1}$ | ntol $\in \{1, 3\text{e-}1, 1\text{e-}1, 3\text{e-}2, 1\text{e-}2, 3\text{e-}3, 1\text{e-}3, 1\text{e-}4\}$ × lin_frac $\in \{0.05, 0.5\}$ |
| `defect` | **safeguarded** Helmholtz defect correction: $d = -H_\nu^{-1}R(u_k)$, step $\alpha \in \{1, \tfrac12, \tfrac14, \tfrac18\}$, the first $\alpha$ with $\|R(u_k + \alpha d)\| < \|R(u_k)\|$ is taken, none → the iteration stops (stall, recorded); predictor = **cubic history extrapolation** $u_0 = 4u^n - 6u^{n-1} + 4u^{n-2} - u^{n-3}$ with order-reducing bootstrap (constant, linear, quadratic for $n = 0, 1, 2$) | ntol as above with at most 60 corrections; plus the fixed-work rungs **zero corrections** (predictor only, its history self-generated, every predictor and output operation charged) and one, two corrections |

`defect` is labelled a *safeguarded heuristic*, not a convergent solver: its linearised iteration
$e_{k+1} = -\Delta t H_\nu^{-1} J_N e_k$ has gain up to $\approx 1.2$ on the fastest modes at $A = 2$,
$\nu = 0.01$ `[A34]`, so the damping is what makes it usable, and every rung's accuracy is measured,
never assumed. The matched comparison is against **whichever arm is cheaper at the matched rung**;
the ladder must **bracket** each ROM arm (a less accurate and a more accurate rung), else the
comparison is Pareto dominance only `[A29]`.

## Decoder

$u(x; z) = \text{bc}(x)\langle g(x), h(z)\rangle$, $\text{bc} = m$; $g$: Fourier lift
$B \in \mathbb R^{3\times 64}$ (scale 4, fixed), SiLU MLP $[128, 2R, 2R, R]$; $h$: SiLU MLP
$[K, 128, 128, R]$ + linear skip. Trainer: `b3d_common.train_autodecoder_3d` — auto-decoder, global
relative MSE + $10^{-4}$ feature-Gram term, Adam warmup-cosine, 60 000 steps, lr $10^{-3}$, **the
same sampling measure at every $N$**: 16 384 pool points iid per step, no full-grid finishing, no
EMA, arrays as explicit jit arguments `[A24, A49]`. Pool: the whole interior at $N = 33, 65$; a fixed
seeded subset of $63^3$ interior points at $N = 129$. Snapshots: 8192 by the `sep_burgers.py` pick
rule at $N \le 65$; at $N = 129$ the largest of 8192 / 4096 / 2048 that the M1 micro-pilot admits.

**Capacity pilot at $N = 33$, on validation only `[A23, A41]`.** Configurations
(K, R, M, m) = (16, 64, 128, 512) and (32, 128, 256, 1024). Both are trained; each is scored by the
held-out oracle D4 on the **64 validation trajectories** (rows 512–575, $k \in \{0, 10, 25, 50\}$) and
must pass D3, D4 and M-stability. Promotion: the smaller if it passes and its validation oracle is
within 1.2× of the larger's; else the larger if it passes; **if neither passes the cell stops
before the test table is opened**. One configuration at every $N$.

## ROM arms

Exact-linear residual, $A = \Phi^{\mathsf T}G_{\text{int}}$, $w = (1 + \Delta t\,\nu\lambda)^{-1}$:

$$
r_w(z) = w \odot \Big[A\big(h(z) - h(z_n)\big) + \Delta t\,\big(q(z) + \nu\,\lambda \odot A h(z)\big)\Big].
$$

| arm | $q(z)$ | cost in $n$ |
|---|---|---|
| `full` | sign-upwind stencil on the full interior grid (full-quadrature oracle) | $O(n)$ |
| `ex` | advection sampled at $m = 4M$ NNLS nodes (advection rows only) | $O(m)$ |
| `tensor` | $\tfrac12 h^{\mathsf T}Qh$, $Q = T + T^{(jk)}$, $T_{ijk} = \sum_x \Phi_{xi}G_{xj}(DG)_{xk}$, $DG = (D^-_x + D^-_y + D^-_z)G$ | $O(MR^2)$, grid-free |

IC fit (encoder init + Gram-space LM), LM step (`STALL = 1e-3`, `TR_FACTOR = 0.01`,
`STEP_TOL = 1e-9`, budget 30), warm-start extrapolation (`EXTRAP = 1.0`) and full-grid decode as in
2D. **Every completed step records $\|J^{\mathsf T}r\|/(\|J\|\|r\|)$ at its accepted latent; $> 10^{-4}$
is censored `[A20]`.** The STEP/ROLL bit-identity gates of the 2D driver compared two jitted paths
through `sep_solvers`; that module imports the 2D FiLM stack, so here STEP/ROLL compare the device
step and rollout against an independently written eager (non-jit) LM on the same states, pass
$\max|\Delta z| \le 10^{-10}$, control stall $10^{-1}$ on a preselected witness with $\ge 2$ accepted
LM steps (recorded as a deviation from 2D).

**Same-GPU kernel job `[A27, A43]`.** One job on one GPU runs the three checkpoints' latent kernels
($A$, $\lambda$, $Q$, head only), each from **its own** IC latents of the eight test trajectories
(saved by its per-$N$ job), interleaved, `TIME_REPS = 5`. Kernel-only, labelled so; C1 pass:
$\max_N / \min_N$ of the median kernel ms $\le 1.5$.

## Gates `[A13–A17, A38–A40, A45, A48]`

Rules: no absolute threshold on a mesh-scaling quantity; every asserted gate has a negative
control that fires, with a **deterministic mutation** wherever one exists and an empirically
**pre-validated** witness otherwise (the witness and its value are stored in the JSON); controls act
on the solver/operator at $k \ge 1$, never on the initial datum; NaN anywhere is FAIL; backward-error
normalisation $\|a - b\|/(\|Op\|_\infty\|x\| + \|b\|)$ for operator applications.

### Phase 0 — FOM (`b3d_fom_gates.py`)

| gate | what | pass | control (must fire) |
|---|---|---|---|
| F1 axis symmetry | permutation-symmetric 3-blob IC, $k = 1..50$, $\max_P \|u - Pu\|/\|u\|$ | $\le 10^{-12}$ | $z$-advection coefficient × 1.01 |
| F2 modes are eigenvectors | scipy-assembled $L$ | $\le 10^{-14}$ backward-normalised | continuum $\pi^2(p^2+q^2+r^2)$ |
| F3 truth acceptance | all trajectories | $\max \|R\|/\|u_k\| \le 10^{-8}$ | 2-iteration generator on test trajectory 0 |
| F4 preconditioner | $\|H_\nu y - v\|$ backward-normalised, $H_\nu$ by assembled $L$ | $\le 10^{-14}$ | $2\nu$ inside $H^{-1}$ |
| F5 non-negativity | train, val, test truth, $k \ge 1$ | $\min u \ge -10^{-9}$ | **downwind** stencil (switch inverted) at $\nu = 0.01$ on test IC 0: $\min u < -10^{-3}$ — pre-validated at $N = 17, 33$ (fires by four orders); a central-difference control was tried and did **not** fire, so it is not used `[A39]` |
| F6 stencil vs assembled | $R$ vs $u - u^{\text{prev}} + \Delta t(u \odot D^-u - \nu Lu)$ (scipy) on a non-negative truth state | $\le 10^{-13}$ | sign-changing state (blob minus shifted blob): the switch is live |
| F7 nested-grid order `[A40]` | datum: $B = 1$, $c = (0.45, 0.5, 0.55)$, $w = 0.2$, $A = 1.5$, $\nu = 0.03$; solutions at $N = 33, 65, 129$ restricted to the common $33^3$ nodes at $k = 50$; $p = \log_2(\|u_{33} - u_{65}\|_2 / \|u_{65} - u_{129}\|_2)$ | recorded, expected near 1; the **band is set by F11** | index mutation: the $N = 129$ solution sampled one fine cell off |
| F8 2D-vs-3D `[A17, A38]` | state $v(x,y)p(z)$, $p = 1$ on the three central planes tapering to the faces, $v$ sign-changing; on the middle plane the 3D residual's **interior rows** and its JVP with the **lifted tangent** $\delta v\,p$ equal the actual 2D code's (`burgers2d_film.make_rollout(n).residual`, interior rows) | $\le 10^{-13}$ backward-normalised | **$x$-advection coefficient × 1.01** (a non-zero term; the $z$-Laplacian mutation of r2 was inert and is retracted) |
| F9 DST implementations | matmul vs FFT DST | $\le 10^{-13}$, faster recorded | FFT with the normalisation factor dropped `[A48]` |
| F10 generator cost | s/trajectory per $N$ | recorded; fixes the $N = 129$ cohort | none |
| F11 manufactured solution `[A40]` | $u_{\text{ex}} = (1 + \tfrac12\sin 2\pi t)\,m(x)\,e^{-\|x - c\|^2/2w^2}$ with $c = (0.5, 0.5, 0.5)$, $w = 0.2$, forcing $f = \partial_t u_{\text{ex}} + u_{\text{ex}}\nabla\!\cdot\!$-form advection $- \nu\Delta u_{\text{ex}}$ by autodiff of the closed form, added to the residual; $\nu = 0.03$; error at $k = 50$ in the 2-norm over the common nodes at $N = 33, 65, 129$ | observed order $p_{\text{mms}} \in [0.7, 1.3]$ (first-order upwind with second-order diffusion); **F7's band is then $[p_{\text{mms}} - 0.3, p_{\text{mms}} + 0.3]$** | forcing sign flipped: error $O(1)$ |
| M1 micro-pilot `[A25, A26, A46]` | on the target H200, the actual $N = 129$ shapes: bank build, tensor build, one e2e compile and run, one `newton` and one `defect` step; device peak and host RSS recorded; the $N = 65$ full pipeline timed by phase | $N = 129$ job submitted only if device peak $\le 120$ GB with $\ge 20$ GB free reported, host RSS $\le 200$ GB, and the projected wall time $\le 20$ h | none |

### Phase 1 — decoder and residual (per $N$)

| gate | pass | control |
|---|---|---|
| D1 bank == meshfree | bank vs an **independent numpy** evaluation of the same network `[A13]` | $< 10^{-12}$ | bank at $x + \Delta x/2$ |
| D2 lineage | 64 regenerated picked states, mean rel-L2 $< 0.2$ | shuffled codes |
| D3 rank of $A$ | $\sigma_{\min}/\sigma_{\max} > 10^{-8}$; M-stability: at 256 training codes the residual energy in modes $M{+}1..2M$ relative to $1..M$ recorded, $\le 0.05$ for promotion | $A$ with row $M$ replaced by row 1 (deterministic rank drop) `[A48]` |
| D4 held-out oracle `[A21, A44]` | per state: 8 starts (zero code, 7 random training codes), full-grid Gram-space LM, budget 200, final optimality $\|J^{\mathsf T}r\|/(\|J\|\|r\|)$ recorded; budget-doubling (400) on 4 states changes the oracle $< 1\%$; pass: mean $\le 5\times10^{-2}$ **and** worst $\le 1.5\times10^{-1}$, reported overall and with $k = 0$ excluded `[A37]`; comparator: oracle $\le 0.5\times$ the POD-$K$ projection floor of the same states (POD from the training snapshots); pool-vs-full ratio $\le 1.5$ | shuffled bank rows |
| L | `ex` linear part vs scipy-assembled $\Phi^{\mathsf T}(u - u^{\text{prev}} - \Delta t\,\nu Lu)$ | $\le 10^{-12}$ backward-normalised | diffusion dropped |
| A | `ex` advection at its nodes vs the stencil advection there, computed directly | $\le 10^{-12}$ | central difference |
| FOMR | full-grid weak residual vs $w \odot \Phi^{\mathsf T}R_{\text{FOM}}$ | $\le 10^{-10}$ | $w$ omitted |
| STEP / ROLL | as above | $\le 10^{-10}$ | stall $10^{-1}$ on the preselected witness |

### Phase 2 — tensor (per $N$)

| gate | pass | control |
|---|---|---|
| TB build order | two chunkings | $\le n_{\text{chunks}} \times 10^{-15}$ `[A15]` | last chunk dropped `[A48]` |
| TA algebraic identity | all training head outputs | $< 10^{-13}$ | unsymmetrised $\tfrac12 h^{\mathsf T}Th$ |
| T0-scope | FOM scope check only `[A45]`: on truth snapshots $\Phi^{\mathsf T}N_{\text{upwind}}(u) = \Phi^{\mathsf T}(u \odot D^-u)$; not a tensor precondition | $< 10^{-13}$ | sign-changing field |
| T0-decoded | tensor vs oracle on all-positive decoded training states, with the count | recorded |
| TQ | 32 latent states, fresh $\nu$: $r$, $J$, $J^{\mathsf T}r$ mismatch, `min_u`, `n_neg` | recorded |
| TR candidate-path `[A19]` | host-loop `tensor` rollout on **all 8** test trajectories; at every LM candidate the oracle's $r$, $J$, $J^{\mathsf T}r$ vs the tensor's, sign counts, and whether the oracle's accept/reject decision agrees | recorded; **concern thresholds** $r$ rel $> 10^{-3}$ or decision disagreement $> 1\%$ of candidates flag the $N$ in the report |

### Phase 3 — the ladder and the decision (per $N$)

| gate | pass |
|---|---|
| E1 oracle-equivalence `[A18]` | `tensor` vs `full`, per trajectory: worst per-state field rel-diff $\le 10^{-3}$, latent dev $\le 10^{-6}$, $|\text{err ratio} - 1| \le 10^{-2}$, identical stop histograms and attempt counts, **and** along the tensor's accepted path the oracle-vs-tensor $r$ rel $\le 10^{-3}$, $J$ rel $\le 10^{-2}$, scaled $J^{\mathsf T}r$ $\le 10^{-3}$ at every step |
| E2 | `tensor` vs `ex`: recorded |
| P1 completion `[A20]` | censored steps $= 0$ for a result row |
| C1 reduced cost `[A43]` | kernel job: $\max_N/\min_N$ median kernel ms $\le 1.5$; kernel-only label |
| C2 crossover `[A29, A30, A42]` | paired AB/BA of `tensor` e2e vs the cheaper classical arm at the matched rung, `PAIR_REPS = 5`, bracket required; speed win needs (i) per-trajectory speedup $> 1$ on all 8, (ii) median $> 1.1$, (iii) the trajectory-clustered bootstrap (2000 resamples of the 8 trajectories' paired speedups) 95 % lower bound $> 1$; raw times and outlier counts stored |
| A1 accuracy `[A22, A44]` | D4 oracle and the ROM's excess reported separately; a positive row also needs the tensor rollout error $\le 5\times10^{-2}$ (absolute usefulness) |
| A2 | decoded positivity fractions: recorded (diagnostic; tensor validity rests on E1/TR, not on positivity) `[A33]` |

## Frozen contract

`N ∈ {33, 65, 129}`, (K, R, M, m) from the pilot, `g_hidden = 2R`, `dt = 0.005`, `steps = 50`,
tables as above, `N_TEST = 8`, `STEPS = 60000`, `LR = 1e-3`, `p_sub = 16384`, `full_last = 0`,
`POS_TOL = 1e-9`, `STALL = 1e-3`, `TR_FACTOR = 0.01`, `EXTRAP = 1.0`, `STEP_TOL = 1e-9`, `BURN = 2`,
`TIME_REPS = 5`, `FOM_REPS = 5`, `PAIR_REPS = 5`, `T_CHUNK = 512 MiB / (8R^2)` rows, f64,
`JAX_DEFAULT_MATMUL_PRECISION=highest`. Jobs: `gpu` partition,
`/cluster/tufts/paralab/tawal01/b3dtensor/{pilot33,n33,n65,micro129,n129,kernels}/`; $N \le 65$ on
A100-80G (`--mem 160G`), `micro129` and `n129` on H200 (`--mem 240G`, `n129 --time 24:00:00`);
`jax_backend=gpu` preflight, `MANIFEST.sha256`, `RESULTS.sha256`, pulled then deleted.

## What the cell decides `[A31, A33]`

Preconditions for **any** result row: phase 0 and 1 gates pass with their controls fired; the pilot
promoted a configuration on validation; D4 pass (mean, worst, comparator, pool ratio); P1 zero
censored steps; ladder bracket present; C1 measured on one GPU; M1 passed before $N = 129$.

| outcome | reading |
|---|---|
| preconditions met; E1 pass at all $N$; C2 speed win (i)–(iii) at $N = 129$; A1 within 3× **and** $\le 5\times10^{-2}$ | the sample-free tensor ROM is oracle-equivalent, its kernel cost is $N$-independent on one GPU, and it beats the cheaper of two classical 3D solvers at 128 cells per side at a stated useful accuracy, on one seed and eight trajectories |
| as above but A1 $> 3\times$ or $> 5\times10^{-2}$ | oracle-equivalent and fast but not useful-accurate: an objective / test-space result; cost labelled "at the ROM's own error" |
| E1 pass, C2 not a win at $N = 129$ | the classical solver with an exact Helmholtz inverse is still cheaper at 2M unknowns; within-job ratios and kernel cost reported; no speed claim |
| E1 pass at some $N$ only | oracle-equivalence is resolution-dependent; TR locates it |
| E1 fail at any $N$ | the undershoot breaks the fixed-branch identity; the split central-difference form is the next cell |
| TR concern flags without E1 failure | reported as a caveat on that $N$'s rows |
| P1 censored $> 0$ | incomplete; no row for that arm and $N$ |
| pilot: neither configuration passes | stop before the test table; report the pilot |
| D4 fails at any $N$ | the decoder does not represent the family on the full grid at that $N$; no ROM rows there |
| no ladder bracket | Pareto dominance only |
| M1 fails | $N = 129$ not run; the cell reports $N = 33, 65$ and the micro-pilot numbers |
| any gate or control fails | fix, numbered retraction, re-run that phase |

## Deliverables

`b3d_common.py`, `b3d_tensor_common.py`, `b3d_fom_gates.py` (with F11, M1 micro-pilot mode),
`sep_b3d_tensor.py` (pilot mode + phases 1–3), `sep_b3d_kernels.py`, cluster
`stage/push/pull_b3dtensor.sh` and `run_b3dtensor_*.sbatch`, `runs/b3dtensor/gen_tables.py`,
`B3D-NOTES.md`, report `reports/2026-09-XX-b3d-tensor-ladder.md` on `main`, Codex verification of
every conclusion and code file, lab log entry.
