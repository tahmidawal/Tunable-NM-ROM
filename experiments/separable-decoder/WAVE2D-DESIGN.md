# Wave 2D — reflective vs absorbing, latent stepping on a separable manifold, cost ladder vs FOM

Design document, **revision r2** (2026-09-03). r1 (`WAVE2D-DESIGN-r1-superseded.md`) was
audited by Codex `gpt-5.6-sol` (`WAVE2D-DESIGN-AUDIT-r1-codex-gpt56sol.md`): items 1–3
CORRECT with fixes, 4 and 8 NEEDS-RESTATEMENT, 5–7 WRONG. Every fix is applied below and
marked **[r2]**. **Status: DESIGN. No numbers here.** Branch `exp/2026-09-03-wave2d-mechanism`,
cut from `exp/2026-08-30-stokes-vector` (2c3f1b6). Supersedes the retired 1D design on
`exp/2026-08-30-waves-vector`.

## What this cell is FOR — read this before the gates

The project retired waves on 2026-08-30 on three arguments. Two have fallen and the third
has been re-diagnosed (full argument and the 08-16 numbers it rests on:
`understand/2026-09-02-fable-wave-rom-case.md` on `main`; 1D CPU check and its 2026-09-03
reproduction: `wav2d_refs/`):

1. *n-width*: the 08-14 FiLM decoder reached 2.80e-2 → 3.51e-2 held-out on Wave 2D
   against POD-6 at 4.41e-1, flat in N. The slow linear width bounds POD, not a manifold.
   **Fallen.**
2. *architecture independence*: the 08-16 failure was on a FiLM auto-decoder predating the
   separable decoder. **Unestablished — this cell establishes it either way.**
3. *structural* ("nonlinear manifold breaks CN time-reversal symmetry; energy destroyed;
   not fixable by tuning"). **Re-diagnosed.** A Lagrangian manifold ROM (tangent-space
   Galerkin of the second-order form, variationally integrated) conserves the pulled-back
   energy for any smooth immersion $g$ at two precomputed $R\times R$ matrices
   (Lall–Krysl–Marsden 2003; Carlberg–Tuminaro–Boggs 2015). In the 1D check it removes the
   energy drift entirely **and does not recover the accuracy**; the accuracy failure tracks
   the *tangent-space* quality of the manifold (a 15% high-frequency wrinkle orthogonal to
   the data leaves the reconstruction floor nearly unchanged and puts every stepping
   scheme at ~7× floor). The 08-16 auto-decoder generalised 2.7× worse than it fit while the
   FiLM decoder on $(\mu,t)$ reached 0.028 on the same data. **Quadrature error is dead**:
   the 08-16 full-grid no-EQ arms failed identically to the EQ arm. The 08-16 energy
   figure 0.27 was the kinematic recursion; the dynamic estimate disagreed by ~100×.

The re-diagnosis: **the 08-16 failure was a manifold-quality failure that a conservative
system exposes and a dissipative one hides.** This cell tests that with three head-training
arms on the same data and two ROM arms, under both boundary conditions.

**Decisive prediction [r2 — the causal claim is an across-head interaction, not a
per-BC contrast].** After the rank gate (D2) and the time-step convergence gate (W6) pass,
**reflective arm C passes W3 if and only if the head passes G0.** Predicted per head:
`auto` fails G0 and fails reflective W3; `sup` and `auto+vc` pass G0 and pass reflective W3.
The absorbing runs are reported beside it as the *dissipative comparator* (predicted: every
head lands ≤ 2× its pre-exit floor), **not** as evidence for the causal claim, because the
absorbing dataset can be intrinsically easier, damping erases accumulated error, and the two
BCs use separately trained banks. Mixed outcomes (G0 fail + reflective pass; G0 pass +
reflective fail with W4 pass) are recorded as **inconclusive**, with the alternative causes
named (time-step, rank loss, initialisation, $4T$ extrapolation, curvature), not as a
verdict either way.

**What this cell is NOT for.** The linear wave FOM is cheap and a POD-$R$ CN ROM is a
precomputed $R\times R$ recurrence with no iteration; nothing iterative on a manifold
competes with it on a *linear* wave. The cost ladder (phase 4) is measured because it must
be, with the predeclared expectation of **no crossover vs POD-$R$ and possibly none vs the
matched-accuracy FOM in the range tested — either is the result.** The value of the cell is
the corrected record plus a certified conservative-stepping arm and a tangent-space gate
for compressible Euler / acoustics later.

## Governing equations

$$u_{tt} = c^2\,\Delta u,\qquad (x,y)\in(0,1)^2,\qquad u(\cdot,0)=u_0,\quad u_t(\cdot,0)=v_0 .$$

- **Reflective:** $u=0$ on $\partial\Omega$.
- **Absorbing:** first-order Engquist–Majda, $u_t + c\,\partial_n u = 0$ on $\partial\Omega$.
  Exact only at normal incidence (oblique reflection $|(\cos\theta-1)/(\cos\theta+1)|$,
  which does **not** vanish with refinement). The cell needs a *dissipative* system with a
  certified energy inequality, not a perfect absorber; F3 measures the absorber on an
  isolated quasi-1D pulse and the blob family's oblique leakage is reported, not gated.

**Family (identical to 08-16 so the reflective arm re-runs the recorded negative on the new
decoder):** $\mu = (c_x,c_y,w,a,\log c)$, $c_x,c_y\sim U(0.15,0.85)$, $w\sim U(0.05,0.20)$,
$a\sim U(1,10)$, $c\sim\log U(0.5,2)$; blob IC `blob_ic` from `wlat_common.py`, $v_0=0$.
Train = first 512 trajectories of seed 0; test = 16 fresh from `TEST_SEED=1`; 51 stored
snapshots on $T=1$; a $4T$ continuation on the 16 test trajectories only. Absorbing arm: the
same blob **without** the hard Dirichlet factor; the initial compatibility defect
$\|v_0 + c\,\partial_n u_0\|$ is recorded (nonzero: a startup wave exists and is reported).

## Discretization — fully specified [r2: every boundary row written out]

Nodes $x_i = i\Delta x$, $\Delta x = 1/(N-1)$, $i=0..N-1$; same in $y$.

**Reflective.** Unknowns on the $(N-2)^2$ interior nodes, ghost-zero 5-point Laplacian
$L_D$ (the 08-14/08-16 FOM, `wave2d_film.make_rollout`, reproduced bit-for-bit — gate V0).
Mass $M = \Delta x^2 I$. Sine modes $\phi_{k\ell}(i,j) = \sin(k\pi x_i)\sin(\ell\pi y_j)$
are exact eigenvectors, $L_D\phi = -\lambda\phi$, $\lambda_{k\ell} = \tfrac{2}{\Delta x^2}[(1-\cos k\pi\Delta x) + (1-\cos\ell\pi\Delta x)]$.
**Repo convention $L\Phi = -\Phi\Lambda$, $\Lambda \succ 0$.**

**Absorbing.** Unknowns on all $N^2$ nodes, state $(u,v)$. Ghost elimination from the BC:
at the $x=1$ face $u_x = -v/c$, so $u_{J+1} = u_{J-1} - 2\Delta x\,v_J/c$ and
$(u_{xx})_J = 2(u_{J-1}-u_J)/\Delta x^2 - 2v_J/(c\Delta x)$; mirrored at $x=0$
($u_{-1} = u_1 - 2\Delta x\,v_0/c$, same damping sign); same in $y$. Rows of the
**Neumann-closed Laplacian $L_N$**:

| node | $L_N u$ row |
|---|---|
| interior | $(u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{ij})/\Delta x^2$ |
| $x$-face, not corner ($i=N-1$) | $(2u_{i-1,j} - 2u_{ij})/\Delta x^2 + (u_{i,j+1}+u_{i,j-1}-2u_{ij})/\Delta x^2$ — **the tangential centred term is kept** |
| corner ($i=N-1$, $j=N-1$) | $(2u_{i-1,j} - 2u_{ij})/\Delta x^2 + (2u_{i,j-1}-2u_{ij})/\Delta x^2$ |

Damping $D_B = \mathrm{diag}(d_{ij})$: $d = 2/\Delta x$ on a face node, $4/\Delta x$ at a
corner, $0$ inside. Semi-discrete system — a **damped ODE**, not a DAE:
$$ \dot u = v,\qquad \dot v = c^2 L_N u - c\,D_B v . $$
Trapezoid mass $M = M_x\otimes M_y$, $M_x = \Delta x\,\mathrm{diag}(\tfrac12,1,\ldots,1,\tfrac12)$.
$ML_N$ is symmetric negative **semi**definite (constants in $\ker L_N$ — the energy is a
seminorm in $u$, so the **mean-field error is reported separately** [r2]). Energy
$$ E = \tfrac12 v^\top M v - \tfrac{c^2}{2} u^\top M L_N u,\qquad \dot E = -c\,v^\top M D_B v \le 0, $$
and for Crank–Nicolson **exactly** $E^{n+1}-E^n = -c\,\Delta t\,\bar v^\top M D_B \bar v$
(gate F1b). At a corner $M_{ii}D_{B,ii} = \Delta x$, equal to the two half-corner face
contributions. Cosine modes $\phi_{k\ell}(i,j) = \cos(k\pi x_i)\cos(\ell\pi y_j)$ (DCT-I) are
exact **right** eigenvectors of $L_N$ including face and corner rows, with the same
$\lambda_{k\ell}$ as above ($k,\ell \ge 0$; the $(0,0)$ mode has $\lambda=0$).
**[r2] $L_N$ is not Euclidean-symmetric; the test operator is $\Phi^\top M$, not $\Phi^\top$.**

**Time.** CN on $(u,v)$, `SUBSTEPS=80` per stored snapshot as in 08-16
($\Delta t_{\rm FOM} = 2.5\times10^{-4}$), f64. Reflective $u$-solve: $(I - aL_D)$, SPD, CG
tol 1e-10 ($a = (c\Delta t/2)^2$). **[r2] Absorbing $u$-solve: $\big(I + \tfrac{c\Delta t}{2}D_B - aL_N\big)$,
SPD only in the $M$-inner product** — solve the $M^{1/2}$-similarity-transformed system
$M^{1/2}(\cdot)M^{-1/2}$ (diagonal scaling, exact) with ordinary CG, or weighted CG; the
solver's SPD-ness is gate F0d. Backward Euler is not acceptable (numerical damping would
fake dissipation in the reflective arm); it appears only as the F1a negative control.

## Decoder — scalar separable, $u$ only

$$u(x;z) = \mathrm{bc}(x)\,\langle g(x), h(z)\rangle = (G\,h(z))(x),\qquad G\in\mathbb R^{n\times R}.$$

`bc` mask in the reflective configuration, **absent** in the absorbing one. No vector-valued
head: $v$ is eliminated exactly (Newmark, arm A; gate V1) or lives in the tangent bundle,
$v = GJ_h(z)\dot z$ (arm C). Bank: $M$-weighted POD of the training snapshots, **$R$ fixed
per ladder** [r2] ($R=64$ primary; a separate $R=96$ ladder only if D0 shows $\sigma_R/\sigma_1$
above the floor of interest). Head: the incumbent SiLU MLP + linear skip $K\to R$ from
`stk2d_head.py`, trained on coefficient targets $C = G^\top M u$ (Cfit).

**Three head-training arms — the experimental variable:**

| arm | latent | prediction |
|---|---|---|
| `auto` | per-snapshot free codes, $K=8$, initialised from top-$K$ POD coefficients (08-16 recipe) | fails G0, fails reflective W3 |
| `sup` | $z=(\mu,t)$ affinely mapped to $[-1,1]^6$, $K=6$, supervised | passes G0, passes reflective W3 |
| `auto+vc` | as `auto` plus velocity-consistency loss $\|GJ_h(z_s)\dot z_s - v_s\|_M^2$, $\dot z_s$ free per snapshot, $v_s$ the FOM velocity, weighted into the energy norm | passes G0, passes reflective W3 |

**Component scaling.** Every metric, loss and residual block is in the **energy norm**
$\|(u,v)\|_E^2 = \tfrac12 v^\top Mv - \tfrac{c^2}{2}u^\top MLu$ with **per-trajectory** inverse
mean-square weights (wave snapshot norms pass near zero during kinetic/potential exchange;
never per-snapshot relative errors). Trajectory error is the 08-16 traj-RMS:
$\sqrt{\mathrm{mean}_t\|e_t\|^2}/\sqrt{\mathrm{mean}_t\|u_t\|^2}$ in the $M$-norm of $u$ for
comparability with 08-14/08-16 (primary), and in the energy norm (secondary).
**[r2] Absorbing errors are additionally normalised by $\sqrt{E^0}$ and reported on a
predeclared pre-exit window** $t \le t_{\rm exit}$, $t_{\rm exit}$ = first time the FOM retains
< 50% of $E^0$; post-exit error separately.

## ROM arms — all exact, zero quadrature [r2: $M$-weighted tables]

Latent state carried: $(z_n, z_{n-1})$. ROM step $\Delta t = \Delta t_{\rm snap}/\mathrm{RS}$,
RS $\in\{8,20,40\}$; the same-$\Delta t$ CN FOM error is reported beside every ROM error
(08-16 convention).

Precomputed tables (all exact, one-time, timed separately):

| | reflective ($M=\Delta x^2 I$, sine $\Phi$) | absorbing (trapezoid $M$, cosine $\Phi$) |
|---|---|---|
| $B$ | $\Phi^\top M G$ | $\Phi^\top M G$ |
| $A$ | $\Phi^\top M L_D G = -\Lambda B$ | $\Phi^\top M L_N G = -\Lambda B$ |
| $C$ | — | $\Phi^\top M D_B G$ (dense, boundary-supported) |
| $\mathsf M$ | $G^\top M G$ | $G^\top M G$ |
| $\mathsf K$ | $-G^\top M L_D G$ | $-G^\top M L_N G$ |
| $\mathsf D$ | — | $G^\top M D_B G$ |

**Arm A — incumbent LSPG on the exact damped Newmark residual, Petrov test space ($M_{\rm modes}=64$).**
Derived from CN on $(u,v)$ by subtracting consecutive kinematic equations and using both
adjacent dynamic equations (audit item 3, CORRECT):
$$ r = B\,(h^{n+1} - 2h^n + h^{n-1}) - a\,A\,(h^{n+1}+2h^n+h^{n-1}) + \tfrac{c\Delta t}{2}\,C\,(h^{n+1}-h^{n-1}),\qquad a=(c\Delta t/2)^2, $$
first step (general $v_0$): $r = B(h^1-h^0) - aA(h^1+h^0) + \tfrac{c\Delta t}{2}C(h^1-h^0) - \Delta t\,\Phi^\top M v_0$,
which for $v_0=0$ drops the last term. $C \equiv 0$ reflective. Solved by LM in $z$
(`lm_solve`), warm start $z_n$, cold start = best-of LM fit to the known $u_0$ (timed as
part of end-to-end cost).

**Arm C — variational (Lagrangian) manifold ROM.** Pulled-back Euler–Lagrange equation
**[r2, $M$-weighted]**: $J_g^\top M\big(J_g\ddot z + H_g[\dot z,\dot z] - c^2 L g(z)\big) = 0$,
conserving $E_r = \tfrac12\dot z^\top J_g^\top M J_g\dot z + \tfrac{c^2}{2}g^\top(-ML)g$ in
continuous time for $g \in C^2$ an immersion along the trajectory; the fixed-step variational
integrator preserves a **nearby modified energy**, not $E_r$ exactly (W4 is stated
accordingly). Forced variational Verlet with the Rayleigh term (audit 4b: consistent,
second order):
$$ F(z_{n+1}) = J_h(z_n)^\top\Big[\mathsf M\,(h^{n+1}-2h^n+h^{n-1}) + c^2\Delta t^2\,\mathsf K h^n + \tfrac{c\Delta t}{2}\mathsf D(h^{n+1}-h^{n-1})\Big] = 0, $$
**[r2] Newton Jacobian** $F'(z_{n+1}) = J_h(z_n)^\top\big(\mathsf M + \tfrac{c\Delta t}{2}\mathsf D\big)J_h(z_{n+1})$
— no stiffness and no $\partial J_h(z_n)$ term, because $J_h(z_n)$ and the potential at $n$
are frozen during the solve. First step from $(z_0, \dot z_0)$ with $\dot z_0$ from
$v_0$ by $M$-weighted least squares in the tangent space ($\dot z_0 = 0$ for $v_0=0$):
$J_h(z_0)^\top[\mathsf M(h^1 - h^0 - \Delta t J_h(z_0)\dot z_0) + \tfrac{c^2\Delta t^2}{2}\mathsf K h^0 + \tfrac{c\Delta t}{2}\mathsf D(h^1-h^0)] = 0$
(gate W7 checks this first step against the CN FOM at RS=80 on a linear head, where it must
be exact). C-Verlet's admissible $\Delta t$ is set by the *reduced* spectrum and is measured
(W6), not assumed. C-mid (variational midpoint) is an optional second integrator, only after
its own derivation is audited; it is not in the frozen contract.

**Controls.** POD-$K$ Galerkin CN (linear, matched $K$, energy ratio $1\pm10^{-9}$
reflective — gate W2; floor proximity *reported*, not gated [r2]). POD-$R$ Galerkin CN (the
cost competitor). Oracle floor: LM projection of every held-out snapshot onto the manifold.
`hold` (frozen latent) as a comparator, plus a **deterministic mutation** (wrong-sign
stiffness) as the negative control [r2].

**Energy reporting.** Reflective: $E_r(z,\dot z)$ for arm C with $\dot z$ the Verlet
central difference; for arm A the CN-consistent dynamic velocity on the decoded fields
(08-16 "dynamic"). **Never** the kinematic recursion. Absorbing: retained energy $E^n/E^0$
and the flux history $-c\Delta t\,\bar v^\top MD_B\bar v$ on the decoded fields, **with the
residual-work term** [r2] — a projected ROM does not obey the uncorrected FOM flux identity;
the balance that closes is the forced discrete variational balance (arm C) or
$E^{n+1}-E^n + c\Delta t\,\bar v^\top MD_B\bar v = \langle \text{residual}, \bar v\rangle$ (arm A).

## Gates — every one with a pass rule, predeclared [r2: rebuilt per audit item 5]

Rules obeyed by every gate: no gate compares a quantity to itself through the same code
path; every threshold on a mesh-scaling quantity is normalised; aggregates run
`np.isfinite` first and a NaN anywhere is FAIL; **every gate names a negative control that
must fire, and a gate whose control does not fire is itself FAIL**; `PRECOND` uses `raise`;
**all 16/16 test rollouts must complete** — "completed rollouts only" is not permitted [r2];
aggregation (mean / median / worst) is stated per gate; thresholds are frozen here and any
later change is a numbered retraction.

**Phase 1 — FOM (both BCs), local at $N=64,128$; all must pass.**

| gate | what | pass (aggregate) | negative control (must fire) |
|---|---|---|---|
| V0 | new reflective FOM reproduces `wave2d_film.make_rollout` on 4 trajectories | max rel diff ≤ 1e-13 | perturb one stencil coefficient by 1e-6 → ≥ 1e-7 |
| F0a | $\|L_D\Phi + \Phi\Lambda\|_F/\|\Phi\Lambda\|_F$, $L_D$ assembled by an **independent stencil routine** (not the solver's), 16 modes | ≤ 1e-13 | hold $\Phi$, perturb one $\lambda$ by 1% → O(1e-3) [r2: not the same wrong $k$ in both] |
| F0b | same for $L_N$ with cosine modes incl. $(0,0)$ ($\lambda=0$: test $\|L_N\phi\|$ directly, no division), corner rows exercised by $k,\ell \ge 1$ | ≤ 1e-13 | same perturbation |
| **F0c** [r2] | face/corner ghost-row coefficients: manufactured $u = $ quadratic in $x,y$, compare $L_N u$ row by row against the closed-form ghost formula with a prescribed $v$ | max abs ≤ 1e-12·$\|L_N u\|_\infty$ | corner coefficient $4/\Delta x \to 2/\Delta x$ fires |
| F0d | $ML_N$ symmetry $\|ML_N-(ML_N)^\top\|_F/\|ML_N\|_F$; absorbing step matrix $M(I + \tfrac{c\Delta t}{2}D_B - aL_N)$ SPD (min eigenvalue > 0 at $N=32$) | ≤ 1e-15; SPD | $M = I$ fires (value reported, no scaling claimed [r2]) |
| F1a | reflective CN relative energy drift over $T$ and $4T$ (CG tol 1e-12 for this gate) | max ≤ 1e-10 | backward Euler, same $\Delta t$: measured drift must exceed 1e-4 (stated, not assumed [r2]) |
| F1b | absorbing identity $E^{n+1}-E^n + c\Delta t\,\bar v^\top MD_B\bar v$, relative to $E^0$, every step, during active boundary flux (blob reaching a wall) | max ≤ 1e-10 | replace $\bar v$ by $v^{n+1}$ → O($\Delta t$)·flux, must exceed 1e-6 |
| F2 [r2] | self-convergence per BC: spatial ($N \in \{64,128,256\}$ vs new $N=512$ reference, $\Delta t$ frozen at the finest), temporal (SUBSTEPS $\in\{10,20,40\}$ vs 320, $N$ frozen) | order $2\pm0.3$ each, per BC | a first-order-in-time scheme (BE) reads $1\pm0.3$ |
| **F3** [r2] | absorber on an **isolated quasi-1D traveling pulse**, $v_0 = -c\,\partial_x u_0$, $y$-uniform (transverse faces then carry zero flux), reflected energy fraction vs $N\in\{64,128,256,512\}$ against the closed-form prediction $\tan^4(\theta/4)$, i.e. $\approx\tfrac{15}{1024}(\Delta x/w)^4$ | slope $4\pm0.5$ **or** agreement with the discrete prediction within a factor 2 | reflective walls → fraction ≈ 1 |
| F4 [r2] | absorbing: $E^n$ monotone non-increasing every step; reflective: $\max_n E^n/E^0 \le 1+10^{-10}$ over $4T$ | as stated | anti-damping $D_B \to -D_B$ → growth |
| F5 [r2] | absorbing generator spectrum at $N=32$: $\max\Re\lambda\,/\,(c/\Delta x)$ | ≤ 1e-12 (normalised) | $D_B\to-D_B$ → positive |
| V1 [r2] | u-only Newmark recurrence (three-level, damped form) at RS=80 vs an **independently assembled block CN solve** at CG tol 1e-13, both BCs | max rel ≤ 1e-11 | absorbing: damping sign flipped → O(1); reflective: $a \to 1.01a$ → O(1e-2) |

**Phase 2 — decoder + manifold quality (G0), per head arm; Tufts.**

| gate | what | pass | negative control |
|---|---|---|---|
| D0 [r2] | bank: $\sigma_R/\sigma_1$ reported; $\|G^\top MG - I\|_F$ ≤ 1e-12; coefficient round-trip $\|G(G^\top M u) - P_R u\|$ ≤ 1e-12 on 32 snapshots | as stated | drop $M$ from the Gram → fails orthonormality |
| D1 [r2] | held-out oracle reconstruction (traj-RMS, $M$-norm) vs POD-$K$ on the same held-out set and metric; **the 08-14 FiLM comparator recomputed on this dataset and metric** (its checkpoint is on `exp/2026-08-14-wave2d-coord-rom`) | ≤ 0.5× POD-$K$ (median over 16); FiLM comparator reported, exceeding it is a *finding* | POD-$K$ with $K\to K/2$ must be worse |
| **D2** [r2] | $J_h$ conditioning: $\sigma_{\min}(J_h)/\sigma_{\max}(J_h)$ at all training codes, oracle projections and every rollout step | min ≥ 1e-8; a rollout that violates it is **incomplete** (no pseudoinverse, no damping rescue) | a head with $K > $ rank forced (duplicate a latent coordinate) reads 0 |
| G0a [r2] | train/held-out oracle gap, median over trajectories; **absolute normalised gap** $e_{\rm held} - e_{\rm train}$ also gated | ratio ≤ 1.5 and abs gap ≤ 0.05 | shuffled-label head reads ≫ |
| G0b [r2] | tangent-space velocity residual $\|(I - P_T)v\|_M/\|v\|_M$ at oracle-projected held-out states with kinetic energy ≥ 10% of $E$ (excludes $v\approx0$ states); $P_T$ rank-revealing $M$-orthogonal projector (QR of $M^{1/2}GJ_h$, tolerance 1e-10); POD-$K$ value on the same states | median ≤ 1.0× POD-$K$ | random-tangent (Gaussian $J$) reads ≈ 1 |
| G0c [r2] | stepdiag from oracle starts, arm C, $H\in\{1,2,5,10\}$ intervals, excess over floor, median; comparators `hold` (reported) and **wrong-sign stiffness mutation** (control) | excess at $H=10$ ≤ 0.5× floor | mutation must diverge (error > 1 at $H=10$) |

**Phase 3 — ROM accuracy STOP gate, per (BC, head, ROM arm); Tufts.**

| gate | what | pass | negative control |
|---|---|---|---|
| W0 [r2] | QF residual vs an **independent full-grid path** (decode, stencil + boundary rows from the F0c routine, project with $\Phi^\top M$) at 32 random states and every captured solve; gradient vs a **finite-difference directional derivative** of the full-grid residual (not autodiff twice); backward-error normalisation at converged states | ≤ 1e-12 (≤ 1e-7 for the FD gradient, step-optimised) | perturb $B$ by 1e-8 → fires |
| W1 [r2] | reflective and absorbing: $\|A + \Lambda B\|/\|A\|$ ≤ 1e-12 with the $M$-weighted tables. Absorbing: on a **manufactured boundary-active state** ($v$ nonzero on faces), the dropped term $\tfrac{c\Delta t}{2}C\Delta h$ vs $\|r\|$ must be ≥ 1e-2 (the term that kills the diagonal shortcut, measured where it is active) | as stated | unweighted $\Phi^\top L_N G$ reads ≫ 1e-12 |
| W2 [r2] | POD-$K$ CN control: reflective energy ratio $1\pm10^{-9}$; floor proximity reported | energy | POD-$K$ with BE reads < 1 |
| **W3 STOP** [r2] | 16/16 complete (finite everywhere, D2 satisfied); traj-RMS vs the 80-substep FOM at $T$ and $4T$, median over 16: ≤ 1.5× oracle floor **and** ≤ 0.5× POD-$K$; reflective: dynamic-velocity energy ratio at $T$ in $[0.9,1.1]$; absorbing: pre-exit window error ≤ 1.5× floor with the $\sqrt{E^0}$ normalisation, post-exit reported | PASS on **reflective** for ≥ 1 (head, ROM arm) pair or the ladder is not run | wrong-sign stiffness mutation → error > 1 |
| W4 [r2] | arm C, **reflective only**, at the RS that passes W6: $|E_r^n/E_r^0 - 1|$ bounded, no secular trend (linear fit slope over $4T$ ≤ 1e-3/T), max ≤ 1e-2; **and** the bound shrinks under RS refinement (W6) before a FAIL is read as an implementation bug | as stated | arm A on the `auto` head (predicted to drift) |
| W5 [r2] | absorbing balance closes **with the residual-work term**: arm C forced discrete variational balance; arm A $E^{n+1}-E^n + c\Delta t\bar v^\top MD_B\bar v - \langle r_{\rm full},\bar v\rangle$, relative to $E^0$ | ≤ 1e-8 | drop the residual work → does not close |
| **W6** [r2] | time-step convergence: RS $\in\{8,20,40\}$, ROM error vs the same-$\Delta t$ FOM error; the ROM–floor excess must be RS-independent within 20% at RS ≥ 20 | as stated | — (reported) |
| **W7** [r2] | arm C first step and recurrence on a **linear head** ($h = Wz$, $W$ the POD-$K$ coefficients): must reproduce POD-$K$ Galerkin CN to 1e-11 | ≤ 1e-11 | first step with $\dot z_0$ dropped → O($\Delta t$) |

**Phase 4 — cost ladder, only if W3 passes; Tufts, one GPU type, one process. [r2 rebuilt per audit 8]**

Resolutions $N\in\{64,128,256,512\}$, **$K$, $R$, $M_{\rm modes}$ fixed across the ladder**;
both BCs. Arms, timed **balanced AB/BA order, all raw repetitions retained, warm-up
discarded, all 16 cases must complete, one-time setup timed separately and reported under
identical amortisation rules for FOM and ROM** (bank + tables vs FOM matrix assembly +
preconditioner setup):

| arm | what |
|---|---|
| FOM-ref | CN, 80 substeps, CG tol 1e-10 |
| **FOM-tol** [r2] | the **fastest** configuration on a **predeclared grid** SUBSTEPS $\in\{1,2,4,8,20,40,80\}$ × CG tol $\in\{10^{-4},10^{-6},10^{-8},10^{-10}\}$ whose traj-RMS vs FOM-ref (same metric, same 16 cases, same invocation for accuracy and time) is ≤ the ROM's; with the **best available solver** for the separable operator: Jacobi-preconditioned CG **and** the exact DST/DCT fast solve (the operator is diagonalised by the very modes we use), whichever is faster |
| POD-$K$ CN, POD-$R$ CN | precomputed $K\times K$ / $R\times R$ recurrence, same 16 cases |
| ours-A, ours-C | latent stepping at the RS that meets W3 |

Reported, per $N$ and BC: **(i) latent-only kernel time** (per step and per trajectory,
labelled as such) and **(ii) end-to-end time** = cold-start fit to $u_0$ + stepping + decode
at the 51 requested output times (the headline, because W3 judges full fields); the ratios
ours/FOM-tol and ours/POD-$R$ on the end-to-end figure; the crossover $N$ if any. No tuning
on the evaluation cases (RS and tolerances fixed from W6 on the training-side diagnostics).
**Predeclared expectation: ours loses to POD-$R$ at every $N$; vs FOM-tol a crossover may
appear at $N \ge 256$ because the latent kernel is flat in $N$ — or not. Either is the
result.**

## Frozen contract

$T=1$, 51 snapshots, SUBSTEPS=80, f64, `JAX_DEFAULT_MATMUL_PRECISION=highest`, seed 0 train
/ seed 1 test, $N_{\rm train}=512$, $N_{\rm test}=16$, $K = 6$ (`sup`) / $8$ (`auto`, `auto+vc`),
$R=64$, $M_{\rm modes}=64$, RS $\in\{8,20,40\}$, head 3×128 SiLU + linear skip. Phase 1 at
$N=64,128$ locally; phases 2–4 on Tufts (`/cluster/tufts/paralab/tawal01/wav2d/`, one job
directory per (N, BC, head) cell). Thresholds frozen 2026-09-03 before any run.

## What the cell decides [r2]

| outcome (after D2 and W6 pass) | reading |
|---|---|
| reflective arm C: W3 pass **iff** G0 pass, across the three heads | manifold-quality diagnosis **confirmed**; "fails structurally" **retracted** |
| G0 pass + reflective arm C W3 pass on any head | universal structural failure **refuted** (this alone suffices for the retraction) |
| G0 pass + reflective W3 fail + W4 pass, all heads | structural diagnosis **not refuted**; **inconclusive** until time-step, rank, initialisation, $4T$ and curvature alternatives are excluded — list which were excluded |
| G0 fail + reflective W3 pass | **inconclusive** — G0 is a proxy; record it |
| W4 fail after W6 refinement | arm C implementation bug; fix before reading anything else |
| W7 fail | arm C wrong on the linear case; fix first |
| absorbing results | reported as the dissipative comparator with pre-/post-exit split and mean-field error; **not** used for the causal verdict |
| phase 4 no crossover | expected; the honest cost verdict for linear waves |

## Deliverables

`wav2d_common.py` (grids, $L_D$/$L_N$/$D_B$/$M$, independent stencil routine, modes, CN
FOM both BCs, energies), `wav2d_fom_gates.py` (phase 1, JSON), `wav2d_bank.py`,
`wav2d_head.py` (three arms), `wav2d_rom.py` (arms A and C, controls), `wav2d_rom_gates.py`
(phases 2–3), `wav2d_ladder.py` (phase 4), `wav2d_tables.py` (every number in
`WAVE2D-NOTES.md` from `runs/wav2d/*.json`), `WAVE2D-NOTES.md` with retractions numbered,
this design and its audits, `wav2d_refs/` (the 1D check). Every code file and every
conclusion gets an independent Codex pass before it is reported. Lab-log entry with the
decision table filled in.
