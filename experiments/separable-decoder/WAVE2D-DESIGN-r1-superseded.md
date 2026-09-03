# Wave 2D — reflective vs absorbing, latent stepping on a separable manifold, cost ladder vs FOM

Design document, revision r1 (2026-09-03). **Status: DESIGN, under audit. No numbers here.**
Supersedes `WAVES-DESIGN.md` on `exp/2026-08-30-waves-vector` (1D, retired before
implementation). Everything from that design's Codex audit that survives is folded in here;
the items that audit marked WRONG are not repeated.

## What this cell is FOR — read this before the gates

The project retired waves on 2026-08-30 on three arguments. Two have fallen and the third
has been re-diagnosed:

1. *n-width*: the 08-14 cell measured the FiLM decoder at 2.80e-2 → 3.51e-2 held-out on
   Wave 2D against POD-6 at 4.41e-1, flat in N. The slow linear width bounds POD, not a
   nonlinear manifold. **Fallen.**
2. *architecture independence*: the 08-16 ROM failure was on a FiLM auto-decoder that
   predates the separable decoder. **Unestablished, not fallen — this cell establishes it.**
3. *structural* ("on a nonlinear manifold the CN time-reversal symmetry is gone, energy is
   destroyed, not fixable by tuning"). An independent analysis (Fable 5.1, 2026-09-02, CPU
   check in 1D, script archived with this cell) found:
   - A Lagrangian manifold ROM — tangent-space Galerkin of the *second-order* form,
     $J_g^\top(u_{tt} - c^2 L u) = 0$, integrated variationally — conserves the pulled-back
     energy $E_r = \tfrac12\|J_g\dot z\|^2 + V(g(z))$ for **any** smooth decoder
     (Lall–Krysl–Marsden 2003; Carlberg–Tuminaro–Boggs 2015). No symplectic constraint on
     $g$ is needed because the $u$-only second-order form puts velocity in the tangent
     bundle. For $u = Gh(z)$ this costs two precomputed $R\times R$ matrices.
   - On a smooth manifold the 08-16 formulation (LSPG on the Newmark residual) *also*
     tracks its oracle floor with energy ratio 0.999 — even at a 45% floor. It drifts only
     when the manifold is **wrinkled**: a 5% high-frequency perturbation orthogonal to the
     data moved the floor 0.115 → 0.124 but the ROM to 1.7× floor with 25% energy loss; at
     15% it reproduced the 08-16 phenomenology (4.5× floor, energy 0.62, dt-insensitive).
   - The variational arm removes the energy drift completely (1.0000) **and does not recover
     the accuracy** (0.87 vs 0.69). Energy loss was a symptom.
   - The 08-16 auto-decoder generalised 2.7× worse than it fit (train 0.070, held-out
     0.189) while the FiLM decoder conditioned on true $(\mu,t)$ reached 0.028 on the same
     data. The manifold *is* low-dimensional and smooth; the auto-decoder did not learn it
     smoothly. **Conservative dynamics integrate tangent-space error instead of damping it.**
   - Quadrature error is **dead** as a cause: the 08-16 full-grid, no-EQ arms
     (`lspg:full:weak64` 0.850, `lspg:full:fd` 0.864) failed identically to the EQ arm
     (0.843).
   - The 08-16 energy number (0.27) is itself not a robust measurement: the kinematic and
     dynamic velocity reconstructions disagreed by ~100× (0.22–0.30 vs 8–50). Report the
     pulled-back energy $E_r(z,\dot z)$ or the CN-consistent dynamic velocity, never the
     kinematic recursion.

So the re-diagnosis is: **the 08-16 failure was a manifold-quality failure that a
conservative system exposes and a dissipative one hides.** That is exactly what this cell
is set up to test, and it is why both boundary conditions are run on the *same* IC family:

| BC | system | prediction under the manifold-quality diagnosis | prediction under the (retracted) structural diagnosis |
|---|---|---|---|
| reflective (Dirichlet walls) | conservative | ROM tracks its floor **iff** the manifold gates G0 pass; fails on a manifold that fails G0 | fails regardless |
| absorbing (first-order Engquist–Majda ghost closure) | dissipative | ROM tracks its floor even on a manifold that fails G0 (error is damped) | works |

**Predeclared prediction:** with the incumbent auto-decoder head, G0 fails, reflective ROM
fails at ~3–5× its floor with energy loss, absorbing ROM lands ≤ 2× floor. With a head that
passes G0 (supervised on $(\mu,t)$, or auto-decoder with the velocity-consistency loss),
reflective tracks its floor with energy ratio ≈ 1 under the variational arm. If reflective
fails **with** a G0-passing manifold under the variational arm, the structural diagnosis is
back and this design is wrong.

**What this cell is NOT for.** The linear wave FOM is cheap (SPD CN system, few CG
iterations; 0.80–0.86× logged honestly on 08-14 for full-field decoding at N=512) and a
POD-$R$ Galerkin CN ROM is a precomputed $R\times R$ recurrence with no iteration. The
08-30 maxim "on a linear PDE the nonlinear head buys parameter efficiency, not accuracy"
transfers as a *cost* verdict: nothing iterative on a manifold competes with POD-$R$ on a
linear wave. The cost ladder (phase 4) is run because the user asked for it and because the
number must be measured, not asserted — **but "no crossover in the range tested" is the
expected outcome and is a result.** The value of the cell is (a) the corrected record,
(b) a certified conservative-stepping arm and a tangent-space gate that are prerequisites
for compressible Euler / acoustics on the roadmap.

## Governing equations

$$u_{tt} = c^2\,\Delta u,\qquad (x,y)\in(0,1)^2,\qquad u(\cdot,0)=u_0,\quad u_t(\cdot,0)=0.$$

- **Reflective:** $u=0$ on $\partial\Omega$.
- **Absorbing:** first-order Engquist–Majda, $u_t + c\,\partial_n u = 0$ on $\partial\Omega$.
  In 2D this is exact only at normal incidence; oblique reflection coefficient
  $|(\cos\theta-1)/(\cos\theta+1)|$. **The cell does not need a perfect absorber** — it
  needs a *dissipative* system with a certified energy inequality. Report the absorbing
  quality (F3) as a convergence gate, not a threshold.

Parameter and IC family: **identical to the 08-16 cell** so the reflective arm is a direct
re-run of the recorded negative on the new decoder. $z=(c_x,c_y,w,a,\log c)$,
$c_x,c_y\sim U(0.15,0.85)$, $w\sim U(0.05,0.20)$, $a\sim U(1,10)$, $c\sim\log U(0.5,2)$;
blob IC `blob_ic` from `wlat_common.py`, $v_0=0$. Train = first 512 trajectories of seed 0;
test = 16 fresh from `TEST_SEED=1`; 51 stored snapshots on $T=1$. Reflective ICs are
already boundary-compatible (the blob is multiplied by the hard Dirichlet factor in
`blob_ic`); for the absorbing arm the same IC is used **without** the factor and the
compatibility defect $\|u_t + c\partial_n u\|$ at $t=0$ is recorded (it is zero because
$v_0 = 0$ only if $\partial_n u_0 = 0$; it is not, so a startup wave exists — measured and
reported, not hidden).

## Discretization — fully specified

Nodes $x_i = i\,\Delta x$, $\Delta x = 1/(N-1)$, $i=0..N-1$.

**Reflective.** Interior unknowns $(N-2)^2$, ghost-zero 5-point Laplacian $L_D$
(the 08-14 / 08-16 FOM, `wave2d_film.make_rollout`, imported verbatim). Mass $M=\Delta x^2 I$.
Discrete sine modes $\phi_{k\ell}$ are exact eigenvectors: $L_D\phi = -\lambda\phi$,
$\lambda>0$ (**repo convention: $L\Phi = -\Phi\Lambda$, so $A = \Phi^\top L G = -\Lambda B$**).

**Absorbing.** Unknowns on all $N^2$ nodes. Eliminate the ghost node through the BC:
at the $x=1$ face $u_x = -u_t/c = -v/c$, so $u_{J+1} = u_{J-1} - 2\Delta x\,v_J/c$ and
$$ (u_{xx})_J \approx \frac{2u_{J-1}-2u_J}{\Delta x^2} - \frac{2}{c\,\Delta x}v_J . $$
Collecting, the semi-discrete system is the **damped ODE** (not a DAE)
$$ \dot u = v,\qquad \dot v = c^2 L_N u - c\,D_B v, $$
with $L_N$ the Neumann-closed (ghost-reflected) 5-point Laplacian and
$D_B = \mathrm{diag}(d_i)$, $d_i = 2/\Delta x$ on a face node, $4/\Delta x$ at a corner,
$0$ inside. With the trapezoid mass $M = M_x\otimes M_y$ (weight $\tfrac12$ on face nodes,
$\tfrac14$ at corners, times $\Delta x^2$), $M L_N$ is symmetric negative semidefinite and
$$ E = \tfrac12 v^\top M v - \tfrac{c^2}{2} u^\top M L_N u,\qquad \dot E = -c\,v^\top M D_B v \le 0 . $$
Crank–Nicolson preserves this inequality discretely:
$E^{n+1}-E^n = -c\,\Delta t\,\bar v^\top M D_B \bar v$ (midpoint identity; **this is gate F1b
and it is an identity to check, not a bound**). Cosine modes $\cos(k\pi x_i)\cos(\ell\pi y_j)$
are exact eigenvectors of $L_N$ with $\lambda_{k\ell} = \tfrac{2}{\Delta x^2}\big[(1-\cos k\pi\Delta x)+(1-\cos \ell\pi\Delta x)\big]$
(gate F0b). They are **not** eigenvectors of the full generator because of $D_B$ — this is
the precise statement of why the diagonal shortcut dies (non-symmetry is not the reason).

**Time.** Crank–Nicolson on $(u,v)$, `SUBSTEPS=80` per stored snapshot as in 08-16
($\Delta t_{\rm FOM} = 2.5\times10^{-4}$), CG on the SPD system $(I - a L)$ with
$a=(c\Delta t/2)^2$, tol 1e-10, f64 throughout. Backward Euler is not acceptable (numerical
damping would fake a dissipative system in the reflective arm).

## Decoder — scalar separable, $u$ only

$$u(x;z) = \mathrm{bc}(x)\,\langle g(x), h(z)\rangle = (G\,h(z))(x),\qquad G\in\mathbb R^{n\times R}.$$

`bc` mask present in the reflective configuration, **absent** in the absorbing one (the
boundary values are unknowns). No vector-valued $(u,v)$ head: velocity is eliminated
exactly (Newmark, arm A — the 08-16 identity V1 already certifies it) or lives in the
tangent bundle $v = G J_h(z)\dot z$ (arm C). Bank $G$: POD of the training snapshots at
$R\in\{64, 96\}$ (fixed sine bank is a roadmap item, not this cell); numerical rank of $G$
checked (gate D0). Head: the incumbent SiLU MLP + linear skip, $K\to R$, from
`stk2d_head.py` (`init_head`, `train_head`, `head_np`, `head_jac_np`), trained on
coefficient targets $C = G^\top M u$ ("Cfit").

**Three head-training arms** — this is the experimental variable:

| arm | latent | why |
|---|---|---|
| `auto` | per-snapshot free codes, $K=8$, initialised from top-$K$ POD coefficients (08-16 recipe) | the incumbent; predicted to fail G0 |
| `sup` | $z = (\mu, t)$, $K=6$, supervised | the 08-14-proven smooth manifold; predicted to pass G0 |
| `auto+vc` | as `auto`, plus velocity-consistency loss $\|G J_h(z_s)\dot z_s - v_s\|_M^2$ with $\dot z_s$ a free per-snapshot variable, $v_s$ the FOM velocity | the modest change that addresses the diagnosed failure |

Component scaling: $u$ and $v$ have different magnitudes ($v$ up to tens of $u$ for
narrow blobs); every metric, loss and residual block is nondimensionalised by the
**energy norm** $\|(u,v)\|_E^2 = \tfrac12 v^\top M v + \tfrac{c^2}{2}u^\top(-ML)u$, per
trajectory (per-trajectory inverse mean-square weights, as 08-16, because wave snapshot
norms pass near zero during kinetic/potential exchange).

## ROM arms — all exact, zero quadrature

Latent state carried: $(z_n, z_{n-1})$ (arm A) or $(z_n, \dot z_n)$ / $(z_n,z_{n-1})$ (arm C).
ROM step $\Delta t = \Delta t_{\rm snap}/\mathrm{RS}$, $\mathrm{RS}\in\{8,20,40\}$ swept; the
same-$\Delta t$ CN FOM error is reported beside every ROM error (08-16 convention), so the
time-discretisation floor is never mistaken for manifold error.

**Arm A — incumbent LSPG on the exact Newmark residual, Petrov test space.**
Reflective, $M$ sine modes ($M=4K$..$64$), $B=\Phi^\top G$, $A=\Phi^\top L_D G = -\Lambda B$:
$$ r = B\,(h^{n+1} - 2h^n + h^{n-1}) - a\,A\,(h^{n+1}+2h^n+h^{n-1}) ,\qquad a = (c\Delta t/2)^2 .$$
Absorbing, $M$ cosine modes, $B=\Phi^\top G$, $A=\Phi^\top L_N G = -\Lambda B$,
$C = \Phi^\top D_B G$ (dense, boundary-supported):
$$ r = B\,(h^{n+1} - 2h^n + h^{n-1}) - a\,A\,(h^{n+1}+2h^n+h^{n-1}) + \tfrac{c\Delta t}{2}\,C\,(h^{n+1}-h^{n-1}) .$$
(The damped Newmark form is derived from CN on $(u,v)$ by the trapezoidal elimination of
$v$; **gate V1b certifies it against the $(u,v)$ FOM at RS=80, and the derivation is part of
the audit.**) Solved by LM in $z$ (`lm_solve`), warm start $z_n$, cold start = best-of LM
fit to the known $u_0$. First step from $v_0=0$: $r = B(h^1 - h^0) - aA(h^1+h^0)$ (+ damping
term $\tfrac{c\Delta t}{2}C(h^1-h^0)$ absorbing).

**Arm C — variational (Lagrangian) manifold ROM, tangent-space Galerkin.**
Precompute $\mathsf M = G^\top M G$, $\mathsf K = -G^\top M L G$ (SPD / SPSD), and for the
absorbing arm $\mathsf D = G^\top M D_B G$ — all $R\times R$. Reduced Lagrangian
$L_r(z,\dot z) = \tfrac12 \dot z^\top J_h^\top \mathsf M J_h \dot z - \tfrac{c^2}{2} h^\top\mathsf K h$,
Rayleigh dissipation $\tfrac{c}{2}\dot z^\top J_h^\top \mathsf D J_h\dot z$. Two integrators:
- **C-verlet** (forced variational, explicit in $u$, implicit $K\times K$ in $z$):
  $J_h(z_n)^\top\big[\mathsf M\,(h^{n+1}-2h^n+h^{n-1}) + c^2\Delta t^2\,\mathsf K h^n + \tfrac{c\Delta t}{2}\mathsf D(h^{n+1}-h^{n-1})\big] = 0$.
  CFL-limited by the *reduced* spectrum (Rayleigh quotient ≤ full), so much larger $\Delta t$
  than the FOM's explicit limit; the admissible RS is measured, not assumed.
- **C-mid** (variational midpoint, unconditionally stable, symmetric): as in the archived
  1D script `run_D`, generalised with the dissipation term.
Solved by square Newton ($K\times K$), Jacobian $J_h^\top(\mathsf M + a c^2\mathsf K + \ldots)J_h$
plus the exact second-derivative term $\partial_z J_h^\top(\cdot)$ (include it; report the
Newton iteration count with and without it once, then keep whichever converges).
Initial $\dot z_0$ from $v_0=0$: $\dot z_0 = 0$.

**Controls.** POD-$K$ Galerkin CN (linear, matched $K$; must sit on its projection floor
with energy ratio $1 \pm 10^{-9}$ reflective — the gate that the *control* is right).
POD-$R$ Galerkin CN (the honest cost competitor). Oracle floor: LM projection of each
held-out snapshot onto the manifold (the accuracy ceiling for both arms). `hold` control:
freeze the latent (the "do nothing" baseline from 08-16 stepdiag).

**Energy reporting.** Reflective: $E_r(z,\dot z)$ for arm C; for arm A the CN-consistent
dynamic velocity $v_{k} = v_{k-1} + \tfrac{c^2\Delta t}{2}(Lu_{k-1}+Lu_k)$ (08-16 "dynamic",
calibrated on an exact trajectory). **Never** the kinematic recursion. Absorbing: energy
*flux* $-c\,\bar v^\top MD_B\bar v\,\Delta t$ summed vs $E^0 - E^n$ (the identity F1b, applied
to the ROM's decoded fields — the ROM's energy bookkeeping must close to the same identity
up to its residual).

## Gates — every one with a pass rule, predeclared

Rules that every gate below obeys (the recurring failure mode of this project is gates,
not numerics): no gate compares a quantity to itself through the same code path; every
threshold on a mesh-scaling quantity is normalised; every aggregate runs `np.isfinite`
first and a NaN anywhere is a FAIL; every gate has a **negative control that must fire**;
`PRECOND` uses `raise`, not `assert`.

**Phase 1 — FOM (both BCs). Must pass before anything else.**

| gate | what | pass rule | negative control |
|---|---|---|---|
| F0a | sine modes are eigenvectors of $L_D$: $\|L_D\Phi + \Phi\Lambda\|/\|\Phi\Lambda\|$ | ≤ 1e-13 | a mode with wrong $k$ reads O(1) |
| F0b | cosine modes are eigenvectors of $L_N$, same form, $\lambda_{k\ell}$ from the closed form above | ≤ 1e-13 | same |
| F0c | $ML_N$ symmetric: $\|ML_N - (ML_N)^\top\|/\|ML_N\|$ | ≤ 1e-15 | $M=I$ (no trapezoid weights) reads O($\Delta x^{-1}$)-relative |
| F1a | reflective CN energy drift over $T$ and $4T$, relative | ≤ 1e-10 (measures CG tol + roundoff; the invariant is exact) | backward Euler at the same $\Delta t$ reads O(1e-2) |
| F1b | absorbing CN energy *identity* $E^{n+1}-E^n + c\Delta t\,\bar v^\top MD_B\bar v$, relative to $E^0$, every step | ≤ 1e-10 | drop the $\bar v$ midpoint (use $v^{n+1}$) → O($\Delta t$) |
| F2 | convergence to the 08-14 N=512 reference, both BCs, spatial with $\Delta t$ frozen fine, temporal with $N$ frozen fine, separately | observed order 2 ± 0.3 in each | — |
| F3 | absorbing reflection convergence: plane-pulse normal incidence, reflected energy fraction vs $N$ | scales as $O((\Delta x/w)^2)$, slope 2 ± 0.3 on the $N\in\{64,128,256\}$ ladder; report the oblique-incidence fraction beside it, not gated | reflective walls read ≈ 1 |
| F4 | no growth over $4T$, both BCs; absorbing energy monotone non-increasing every step | max ratio ≤ 1 + 1e-10 | — |
| F5 | generator spectrum, absorbing: eigenvalues of the $2N^2$ block generator at $N=32$ have real part ≤ 0 | max Re ≤ 1e-12 | $D_B \to -D_B$ (anti-damping) reads > 0 |
| V1a/b | u-only Newmark rollout at RS=80 reproduces the $(u,v)$ CN FOM, both BCs (the $v$-elimination is exact) | ≤ 1e-12 relative | wrong damping sign reads O(1) |

**Phase 2 — decoder and manifold quality (G0, the diagnosed cause). Per head arm.**

| gate | what | pass rule |
|---|---|---|
| D0 | numerical rank of $G$ = $R$; top-$R$ POD floor on held-out (the decoder ceiling) reported | rank = $R$ exactly |
| D1 | held-out oracle reconstruction (energy norm, per trajectory) vs POD-$K$ on the same held-out set, same metric | ≤ 0.5× POD-$K$; **≤ 2× the 08-14 FiLM value (3.5e-2) or the arm is a bug** |
| G0a | train / held-out oracle gap | ≤ 1.5× |
| G0b | tangent-space velocity residual $\|(I-P_T)v\|_M/\|v\|_M$ at oracle-projected held-out states, $P_T$ the $M$-orthogonal projector onto $\mathrm{range}(GJ_h)$; POD-$K$ value on the same states beside it | ≤ 1.0× the POD-$K$ value (i.e. the manifold's tangent space is at least as good as the subspace it was initialised from) |
| G0c | stepdiag from oracle starts, horizons $H\in\{1,2,5,10\}$ snapshot intervals, excess over floor, with the `hold` control | excess at $H=10$ ≤ 0.5× floor; `hold` must be worse than the ROM at every $H$ (negative control) |

**Predeclared:** `auto` fails G0b/G0c; `sup` passes; `auto+vc` passes. Whatever happens,
**every head arm proceeds to phase 3** — G0 is a *prediction* about phase 3, and the
prediction is the result.

**Phase 3 — ROM accuracy STOP GATE. Per (BC, head arm, ROM arm).**

| gate | what | pass rule |
|---|---|---|
| W0 | QF residual vs an **independent** full-grid implementation (decode, apply stencil + boundary rows, project) at ≥ 32 random states and at every captured solve; gradient likewise; backward-error normalisation at converged solutions | ≤ 1e-12 relative |
| W1 | reflective: $\|A + \Lambda B\|/\|A\|$ ≤ 1e-12 (diagonal shortcut valid). Absorbing: $\|A+\Lambda B\|/\|A\|$ ≤ 1e-12 **and** $\|C\|/\|A\|$ reported (the term that kills the shortcut); the $\Lambda$-only residual (dropping $C$) must **fail** W0 visibly (≥ 1e-3) | as stated |
| W2 | POD-$K$ control on its floor (≤ 1.05× projection floor), energy ratio $1\pm10^{-9}$ reflective | must pass or the harness is wrong |
| **W3 STOP** | ROM traj-RMS error vs the 80-substep FOM, 16 held-out, completed rollouts only (a rollout completes iff every value is finite), at $T$ and $4T$: ≤ 1.5× oracle floor **and** ≤ 0.5× POD-$K$; energy ratio (reflective) within $[0.9, 1.1]$ at $T$ | PASS for at least one (head arm, ROM arm) pair **on the reflective BC** or the cost ladder is not run and the cell reports the negative |
| W4 | arm C energy: $E_r$ bounded oscillation, no secular trend, $|\Delta E_r/E_r| \le 10^{-2}$ over $4T$ — **regardless of accuracy** | predicted PASS on every head arm; a FAIL here means the variational implementation is wrong, not the manifold |
| W5 | absorbing flux identity on the ROM's decoded fields closes to ≤ 1e-6 relative | — |

**Phase 4 — cost ladder. Only if W3 PASSES.**

Resolutions $N\in\{64,128,256,512\}$, fixed $K$, $R$, $M$; both BCs. Arms timed in **one
process, balanced AB/BA order**, all raw repetitions retained, warm-up discarded, one-time
setup (bank, $A/B/C$ or $\mathsf M/\mathsf K/\mathsf D$ builds) timed separately:

| arm | what |
|---|---|
| FOM-ref | CN, 80 substeps, CG tol 1e-10 (the reference solution) |
| **FOM-tol** | CN with the **largest** $\Delta t$ and loosest CG tol whose error vs FOM-ref is ≤ the ROM's error — the *fair* baseline (matched accuracy, as the 08-17 cost-to-tolerance cell) |
| POD-$K$ CN, POD-$R$ CN | precomputed $K\times K$ / $R\times R$ recurrence |
| ours-A, ours-C | latent stepping at the RS that meets W3 |

Reported: wall time per trajectory, per step, and the ratio ours/FOM-tol and ours/POD-$R$
at every $N$, with the crossover $N$ (if any). **Predeclared expectation: ours loses to
POD-$R$ at every $N$ and to FOM-tol below $N\approx256$; the ROM cost is flat in $N$ (it never
touches the grid) so a crossover vs FOM-tol may exist at $N\ge 512$ — or not. Either is the
result.** Full-field decoding ($G h$, an $n\times R$ matvec) is timed separately and *not*
included in the ROM solve time (the 08-14 0.80× number included the decoder render, which
is not the ROM).

## Frozen contract

$T=1$, 51 snapshots, SUBSTEPS=80, f64, `JAX_DEFAULT_MATMUL_PRECISION=highest`, seed 0
train / seed 1 test, $N_{\rm train}=512$, $N_{\rm test}=16$, $K\in\{6\ (\texttt{sup}),\ 8\ (\texttt{auto})\}$,
$R=64$ at $N\le128$, $R=96$ at $N\ge256$, $M=64$ test modes (arm A), RS$\in\{8,20,40\}$.
Phase 1 at $N=64$ and $128$ locally (sub-minute FOM gates), everything else on Tufts.
Thresholds above are frozen before the first run; a threshold changed after a number is
seen is recorded as a retraction, with the original.

## What the cell decides, in one table

| outcome | reading |
|---|---|
| G0 fail + W3 fail (reflective) + W3 pass (absorbing), same head | manifold-quality diagnosis **confirmed**; structural retracted |
| G0 pass + W3 pass (reflective, arm C) | waves work on a good manifold; "fails structurally" **retracted** |
| G0 pass + W3 fail (reflective, arm C) + W4 pass | structural diagnosis **back**; this design wrong; record it |
| W4 fail | variational arm implemented wrong; fix before reading anything else |
| phase 4 no crossover | expected; the honest cost verdict for linear waves; the value is the certified arm C + G0 gate for Euler |

## Deliverables

`experiments/separable-decoder/wav2d_*.py` (common/FOM, gates, bank, head, rom, tables),
`runs/wav2d/*.json`, `WAVE2D-NOTES.md` with retractions numbered, all tables generated
from JSONs by `wav2d_tables.py`, this design and its Codex audits, the archived Fable 1D
check script. Commit and push to the cell's branch; lab-log entry with the verdict table
filled in.
