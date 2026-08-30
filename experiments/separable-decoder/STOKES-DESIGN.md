# Steady Stokes 2D — vector-valued, divergence-constrained, quadrature-free

Design for step 1 of the 2026-08-30 roadmap, **replacing the retired waves cell**
(`exp/2026-08-30-waves-vector`, `WAVES-RETIRED.md`).
Branch `exp/2026-08-30-stokes-vector`, cut from `exp/2026-08-29-b2d-tensor`.

**Revision 2**, rewritten against `STOKES-DESIGN-AUDIT-r1-codex-gpt56sol.md` (Codex
`gpt-5.6-sol`, xhigh), which marked revision 1 "should not be implemented as written"
and supplied numerical checks (archived as `STOKES-AUDIT-mac_check.py`,
`STOKES-AUDIT-mac_s5_scaling.py`). **Status: DESIGN, awaiting re-audit. No code yet.**

---

## What this cell is FOR — read this before reading the gates

**This cell is a de-risking rehearsal for 2D incompressible Navier–Stokes. It is not
expected to produce a positive result, and it must not be written up as one.**

The audit made the reason explicit, and the lab log already contains the same finding
for heat: *"a direct reduced POD-Galerkin solve is 13–38× faster than the FOM, so
nothing iterative competes on a linear parabolic problem"* (2026-08-16). Steady Stokes
is linear and $G$ is itself a POD basis, so a **direct linear reduced solve in the
$G$ span is a one-shot projection** — it will very likely be both faster and more
accurate than a nonlinear head driven by an iterative LM solve. That control is gate
S7c below, and it is expected to win.

The deliverable is therefore **verified machinery plus honest gate numbers**, not a
speedup:

1. a **vector-valued** decoder and residual path,
2. a **discretely divergence-free** bank with the `bc(x)` mask removed,
3. **exact analytic pressure elimination** — no pressure in the ROM at all,
4. a **staggered MAC** FOM certified by a manufactured solution,
5. the quadrature-free residual in its general block form, gated against an
   independent implementation.

NS = this machinery + a staggered skew-symmetric convection tensor. Everything above
carries; the tensor does not (see "What does NOT carry to NS").

---

## Governing equations

$$-\nu\,\Delta \mathbf{u} + \nabla p = \mathbf{f}(\mathbf{x};\mu),\qquad
\nabla\!\cdot\mathbf{u}=0,\qquad \mathbf{u}=0 \text{ on } \partial\Omega,\quad \Omega=(0,1)^2.$$

Homogeneous no-slip with a parameterized body force — deliberately not lid-driven,
which would need a divergence-free lift (a second hard problem stacked on the first).

**$\nu$ is not a shape parameter.** In steady Stokes a scalar $\nu$ rescales the
velocity by $1/\nu$ and nothing else. Revision 1 listed it as "the continuous
parameter", which was wrong: all manifold richness must come from $\mathbf{f}$.
$\nu$ is held fixed, or varied only to confirm the exact $1/\nu$ scaling as a cheap
correctness check.

## Discretization — staggered MAC, fully specified

$N$ **cells** per side, $h=1/N$. **This differs from every existing script here**, which
is collocated with $(N-2)^2$ interior points and $h=1/(N-1)$. The two conventions must
never share a reshape, mask, or spacing; the audit flagged this as a silent-corruption
risk.

| quantity | location | count |
|---|---|---|
| $p$ | cell centres | $N^2$, with one gauge null mode |
| $u_x$ | interior vertical faces | $N(N-1)$ |
| $u_y$ | interior horizontal faces | $N(N-1)$ |
| $\psi$ | vertices | $(N-1)^2$ interior, zero on the boundary |

so the active velocity dimension is $n_u = 2N(N-1)$ — **not** the geometric face count
$2N(N+1)$. Boundary-normal velocities are eliminated (identically zero). No-slip on the
tangential components is imposed through **odd ghosts** in $L$; tangential wall values
are not stored degrees of freedom. Revision 1's claim that no-slip is "inherited from
the POD snapshots" was wrong for the tangential components — only the eliminated
normal components are inherited linearly.

Mass matrices $M_u, M_p$ (both $h^2 I$ on this uniform layout). The identity the whole
design rests on is **weighted adjointness**:

$$M_u\,\mathrm{Grad}_h = -D_h^\top M_p.$$

The audit verified numerically at $N=32$ that the standard MAC pair satisfies
$\lVert D+\mathrm{Grad}^\top\rVert_\infty$ **exactly zero**, $\lVert DC\rVert_\infty$
exactly zero, $\operatorname{rank}D = N^2-1$, and $\dim\ker D = \operatorname{rank}C =
(N-1)^2$ — i.e. the vertex-curl space **exactly spans** the discrete solenoidal space
on this simply connected domain.

## Test space — curls of vertex sine stream functions

$$\Phi = C\,\psi,\qquad \psi_{k\ell}=\sin(k\pi x)\sin(\ell\pi y),\qquad
u_{i,j+1/2}=\tfrac{\psi_{i,j+1}-\psi_{i,j}}{h},\quad
v_{i+1/2,j}=-\tfrac{\psi_{i+1,j}-\psi_{i,j}}{h}.$$

Divergence-free because the cell divergence **telescopes exactly**, boundary cells
included, given $\psi$ constant on the boundary (the sines give zero). This is a
property of *this* incidence-difference curl; it does **not** hold for a curl built
from unrelated centred differences, interpolation, or a cell-centred $\psi$.

$\Phi$ is **normalized in the mass-weighted (kinetic) inner product** — curl-sine norms
grow like $\sqrt{\lambda_{k\ell}}$, so unnormalized modes silently up-weight the
high-frequency equations. The same mass-weighted metric is used for the residual, the
POD, the training loss, and every reported error (gate S8).

**Petrov, not Galerkin, and only for the strong residual.** These modes have
$\psi=0$ but generally $\partial_n\psi\neq0$, so their tangential trace is nonzero and
they are *not* admissible $H^1_0$ tests for an integrated-by-parts weak Stokes form —
viscous boundary terms would survive. They are legitimate tests for the **strong
discrete residual** $\Phi^\top M_u(\nu L\mathbf{u}+\ldots)$, where pressure elimination
needs only zero normal trace and no-slip stays hard-enforced in $L$. The design uses
the strong form throughout; this must not drift.

## The bank — divergence-free, no `bc` mask

$$\mathbf{u}(\mathbf{x};z) = \bar{\mathbf{u}} + G\,h(z),\qquad G\in\mathbb{R}^{n_u\times R}.$$

The `bc(x)` mask is removed: multiplying a divergence-free field by a scalar mask
destroys divergence-freeness, since $\nabla\!\cdot(\mathrm{bc}\cdot Gh) = \nabla
\mathrm{bc}\cdot(Gh) + \mathrm{bc}\,\nabla\!\cdot(Gh)$ and the first term is not zero.

**The affine mean $\bar{\mathbf{u}}$ is explicit**, and revision 1 omitted it. With
centred POD the residual acquires a constant term, $-\nu(\Phi^\top M_u L\bar{\mathbf{u}}
+ A h(z)) - b$, precomputed once alongside $A$. Uncentred POD ($\bar{\mathbf{u}}=0$) is
the alternative; whichever is used must be stated and consistent.

**The "POD of div-free snapshots is div-free" argument is exact only in exact
arithmetic, and revision 1 overstated it.** For Gram POD $g_i = Xv_i/\sigma_i$, so
$Dg_i = (DX)v_i/\sigma_i$: a snapshot divergence residual of $10^{-8}$ becomes
$10^{-8}/\sigma_i$ in the **tail modes**. Mean subtraction, truncation, normalization
and QR reorthogonalization all preserve the constraint in exact arithmetic; random
completion, coefficient thresholding and sparsification break it. Mitigation: project
snapshots onto $\ker D$ before POD, and re-gate after *every* normalization or
reorthogonalization step, per mode, with the $1/\sigma_i$ amplification reported
explicitly rather than assumed away.

## The quadrature-free residual

Testing against divergence-free $\Phi$ annihilates the pressure exactly,
$\Phi^\top M_u\mathrm{Grad}_h p = -(D_h\Phi)^\top M_p\,p = 0$ (verified numerically at
$5.91\times10^{-18}$ normalized), leaving

$$r(z) = -\nu\big(\Phi^\top M_u L\bar{\mathbf{u}} + A\,h(z)\big) - b(\mu),
\qquad A = \Phi^\top M_u L\,G,\qquad b(\mu)=\Phi^\top M_u \mathbf{f}(\mu).$$

$A$ is $M\times R$, precomputed once. **Pressure never appears in the ROM.** Cost per
residual is $MR$, independent of $n$.

This is the steady single-component case of the general block form the waves audit
established: $R = P^\top[E(\mu)\mathcal{G}\dot\eta - K(\mu)\mathcal{G}\eta]$, one
precomputed matrix per affine operator component.

**$b(\mu)$ must be affine or the end-to-end cost claim is false.** For moving Gaussian
centres $\Phi^\top M_u\mathbf{f}(\mu)$ cannot be precomputed once — it costs
$O(Mn_u)$ per query, and the incumbent Poisson convention explicitly times source
projection *inside* the timed pipe. So the force family is chosen **affine by
construction** (fixed blob shapes, varying amplitudes):

$$\mathbf{f}(\mu)=\sum_q \theta_q(\mu)\,\mathbf{f}_q \;\Longrightarrow\;
b(\mu)=\sum_q\theta_q(\mu)\,\Phi^\top M_u\mathbf{f}_q$$

with every $\Phi^\top M_u\mathbf{f}_q$ precomputed. The non-affine moving-centre family
is run as a **separate, separately-reported arm** where the $O(Mn_u)$ projection is
timed inside the pipe — never blended into the affine numbers.

## Force family — Hodge content is the design variable

Only the **solenoidal part** of $\mathbf{f}$ drives the velocity. A gradient-dominated
family produces large pressure and almost no velocity; a purely solenoidal family makes
the pressure gate vacuous. So the family is a controlled mixture with independently
varied amplitudes,

$$\mathbf{f} = C q + \mathrm{Grad}_h\chi,$$

and every run reports the solenoidal-force energy fraction, the gradient-force energy
fraction, $\lVert\mathrm{Grad}_h p\rVert/\lVert\mathbf{f}\rVert$, the velocity snapshot
singular spectrum, and numerical rank versus $R$.

---

## Gates

Conventions (`ALLOW_CPU`, cancellation-aware tolerances, balanced timing, held-out +
fresh-seed cohorts) mirror `sep_poisson_qf.py`. **Every threshold below is normalized**
— revision 1 used absolute thresholds that scale with $h^{-1}$ and would have rejected
correct code (measured: raw $\lVert D\Phi\rVert$ = 2.27e-13 / 4.55e-13 / 9.09e-13 at
$N$ = 64 / 128 / 256, against a proposed absolute gate of 1e-14).

- **S0** backend `gpu`, f64, matmul `highest`.

- **S-FOM — manufactured solution. New, and the most important gate here.** Every ROM
  gate can pass against a consistently wrong FOM; the specific hazard is *accidentally
  solving free-slip Stokes*. Use the clamped stream function $\psi=\sin^2(\pi x)
  \sin^2(\pi y)$, which has $\psi=0$ **and** $\partial_n\psi=0$, so its curl is
  divergence-free *and* zero on all four walls — genuine no-slip. Pair it with a
  non-constant pressure. Require second-order convergence in **both** velocity and
  pressure over $N=32,64,128,256$. This exercises gradient/divergence signs, the
  pressure gauge, the tangential ghost coefficients, and force evaluation on both face
  lattices.

- **S-ADJ — weighted adjointness**, $\lVert M_u\mathrm{Grad}+D^\top M_p\rVert /
  (\lVert M_u\mathrm{Grad}\rVert+\lVert D^\top M_p\rVert)$ at roundoff, plus its
  restriction to the rows the test space can actually reach.

- **S1 — bank divergence, normalized and per mode.** $\lVert Dg_i\rVert/(\lVert
  D\rVert\lVert g_i\rVert)$ for every $i$, reported against the FOM's own snapshot
  divergence *and against $1/\sigma_i$*, so tail-mode amplification is visible rather
  than assumed absent.

- **S2 — test-space divergence.** Structural check $\lVert DC\rVert$ exactly zero, then
  the scale-aware field-path check $\lVert D\Phi\rVert/(\lVert D\rVert\lVert\Phi\rVert)$.
  The raw $10^{-14}$ threshold from revision 1 is removed.

- **S3 — pressure elimination, built so it can fail.** Report $\lVert\Phi^\top M_u
  \mathrm{Grad}_h p\rVert$ under backward-error normalization at the true FOM pressure,
  and separately require the pressure to be non-trivial (so the denominator is not
  near-zero). Revision 1's anti-vacuity control was inadequate — $\lVert\Phi^\top(\alpha
  q)\rVert$ can be made $O(1)$ by choosing $\alpha$, which proves only $\Phi\neq0$.
  Replaced by four checks: **(a)** a *matched* non-solenoidal test basis
  $\widetilde\Phi$ with equal column norms and frequencies, required to give a
  non-negligible $\widetilde\Phi^\top\mathrm{Grad}_h p$ for the same pressure;
  **(b)** per-row leverage $\lVert\Phi_{j,:}\rVert_2$, reported separately for
  boundary-adjacent rows; **(c)** deterministic injected residuals on each boundary
  stencil family; **(d)** an $M$-ladder with $M>K$ comfortably, plus the numerical rank
  and conditioning of $A\,J_h(z)$.

- **S4 — exactness against an INDEPENDENT implementation.** Decode $\mathbf{u}$, apply
  the MAC stencil, project — never a path sharing the cached $A$, which would prove only
  self-consistency. $\le10^{-12}$ relative at $\ge32$ seeded random states and at every
  captured solve solution, with a **cancellation-aware normalization on the residual as
  well as the gradient** (relative residual-to-residual error is meaningless when both
  are near zero).

- **S5 — REWRITTEN; revision 1's version could not fail.** Under no-slip MAC the
  curl-sine modes are **not** eigenvectors of the vector Laplacian, and this is settled
  analytically, not empirically: the tangential (cosine) components want even /
  free-slip ghosts while no-slip uses odd ghosts, leaving the defect
  $(L\phi+\lambda\phi) = -\tfrac{2}{h^2}\phi_u$ on boundary-adjacent tangential rows and
  zero elsewhere. **Primary gate:** the direct operator eigen-residual $\lVert
  L\phi+\lambda\phi\rVert/\lVert L\phi\rVert$ must be $O(1)$ — audit-measured
  0.769–0.998 ($N=64$), 0.943–0.999 ($N=128$), 0.988–0.99985 ($N=256$) for
  $k,\ell\le8$. **A roundoff result is a FAILURE**, indicating even/free-slip ghosts, a
  wrong $L$, or omitted boundary terms. Secondary diagnostic: $\lVert A+\Lambda
  B\rVert/\lVert A\rVert$, expected $10^{-1}$–$1$ (audit measured ≈0.34 on surrogate
  banks at all three $N$; the true POD-bank value is bank-dependent). **Note the sign** —
  this repo's convention is $L\Phi=-\Phi\Lambda$, so the comparison is $A+\Lambda B$,
  the error the waves audit caught. Conclusion going in: **dense $A$ is required.**

- **S6 — cost.** QF vs full-grid vs an NNLS-sampled arm, incumbent timing conventions,
  setup timed separately. The NNLS arm must define sampling **separately on the two face
  lattices** and may only sample *after* analytic pressure elimination — sampled
  quadrature does not itself preserve exact gradient annihilation. Affine and non-affine
  force arms reported separately (see $b(\mu)$ above).

- **S7 — three controls, not one.** **(a)** POD-$K$ at matched online dimension;
  **(b)** POD-$R$ at matched trial span; **(c)** a **direct linear reduced solve in the
  $G$ span**, removing the nonlinear head entirely. (c) is expected to win on this
  linear problem — see the framing at the top. Report the decoder's own reconstruction
  ceiling (oracle floor) alongside. Without these, every other gate can pass while the
  method loses, which is exactly how the wave cell failed.

- **S8 — mass-weighted metric stated and used everywhere:** training loss, POD,
  reconstruction error, residual blocks, and all reported errors.

- **S9 — $R$-frontier, deconfounded.** Sweeping $R$ simultaneously changes POD
  truncation, head output dimension and parameter count, tail-mode constraint
  amplification, and how much of $G$ the fixed low-mode $\Phi$ can see. So: **nested**
  banks from one factorization, $K$ held fixed, parameter count reported, an $M$-ladder,
  and a gate on $\sigma_{\min}(A J_h(z))$.

## What does NOT carry to NS — stated now to prevent a false expectation

Revision 1 said NS is "Stokes plus the tensor on the same code". Too strong. The
existing `b2d_tensor_common.py` is **scalar, collocated, and fixed-positive-upwind**.
MAC Navier–Stokes needs two staggered component lattices, cross-component
interpolation, a defined skew-symmetric bilinear convection operator, and a new tensor
$T_{mjk}=\langle\phi_m,\mathcal{B}(g_j,g_k)\rangle$ with its own build-order and
Jacobian gates. Sign-upwind convection is piecewise and admits no single global tensor
for sign-changing velocity — which is precisely why roadmap item 2 (skew-symmetric
convection) exists.

**Target NS is unsteady.** Steady Stokes therefore rehearses the vector, div-free and
pressure machinery but **not** the time/mass block or nonlinear-manifold time stepping.
That residual risk is acknowledged and bounded: the wave failure was a *conservative*
system losing its reversibility symmetry, whereas NS is *dissipative*, like the Burgers
family that works well here. NS is expected to behave like Burgers, not like waves — but
that is an expectation, not a result, and item 3 must gate it.

## Deliverables

Code and run JSONs under `runs/stk2d/`; `STOKES-NOTES.md` with what ran, the numbers,
and anything retracted; every reported number generated from the JSONs by a script;
commit and push to `origin/exp/2026-08-30-stokes-vector`.
