# Experiment roadmap from 2026-08-30 — ordered, gated, executed sequentially

Agreed with the user on 2026-08-30, after the 1D/2D Burgers tensor result closed out.
Supersedes the unordered "Open experiments" list in
`2026-08-30-session-handoff-tensor.md`, which stays valid as a description of the
individual items.

**Execution rules for this roadmap** (user's instruction, 2026-08-30):

- Items run **one after another**, not in parallel — even where they are logically
  independent and the project's default is to parallelize.
- **Every conclusion and every code file is verified by an independent Codex
  (`gpt-5.6-sol`, `model_reasoning_effort = xhigh`) pass before it is accepted.**
  Design documents get audited *before* implementation, not only after.
- Each item gets its own worktree, branch, and cluster namespace, per `CLAUDE.md`.

## The organizing principle

Every PDE choice is scored by one property: **what degree is the residual as a
polynomial in the state?**

| degree | residual cost | sample points | example |
|---|---|---|---|
| 1 (linear) | precomputed $B=\Phi^\top LG$, one matvec | 0 | Poisson, heat, waves |
| 2 (quadratic) | precomputed $T_{ijk}$, $\tfrac12h^\top Qh$ | 0 | Burgers, incompressible NS convection |
| non-polynomial | sampled (NNLS/EQ nodes) | $m>0$ | $\lvert u\rvert$ upwinding, $1/\rho$, limiters |

The tensor's cost is $R^2M$ per residual, **independent of $n$** — so the route only
pays where $n$ is large and $R$ is modest. That is why the $R$-frontier and the move
to 3D matter as much as any new PDE.

## The list

### 1. Waves 1D — reflective and non-reflective  ← IN PROGRESS

Worktree `2026-08-30-waves-vector`, branch `exp/2026-08-30-waves-vector` (from
`exp/2026-08-29-b2d-tensor`), cluster ns `wav1d/`. Design: `WAVES-DESIGN.md` on that
branch.

Decides three things, in the order in which they would kill the NS plan:

1. **Vector-valued state.** Every decoder so far emits one scalar field; the wave
   equation as a first-order system carries $(u,v)$ and NS carries $(u_x,u_y)$. This
   is the cheapest place to make that change, because the residual is entirely
   linear — no quadrature machinery in play while the plumbing is debugged. This is
   the argument that moved waves from "nice risk reduction" to a genuine prerequisite
   for item 3.
2. **The general linear claim.** Poisson QF relies on $\Phi$ being exact Laplacian
   eigenvectors so $\Phi^\top(\Delta u)=\Lambda\Phi^\top u$. An absorbing boundary
   makes the operator non-symmetric and $\Lambda$ dies; the dense $M\times R$ matrix
   $A=\Phi^\top LG$ must replace it. Still exact, still zero-sample — but our
   published linear claim currently assumes self-adjointness without saying so.
3. **Decoder $n$-width.** Traveling waves decay slowly in Kolmogorov $n$-width. If
   $K=8\!-\!32$ cannot represent them, NS fails for reasons unrelated to our
   contribution.

### 2. Sign-changing / skew-symmetric Burgers (1D) — the control

Three arms: upwind + plain tensor (expected failure, to be *characterized* not
avoided), upwind + split form (sample only the $\lvert u\rvert\Delta_h u$ stabilizer),
skew-symmetric + plain tensor (expected exact for any sign).

Decides whether positivity is a limitation of the method or an artifact of the upwind
stencil. Methodological guard: each arm's ROM is graded against **its own** FOM, and
both FOMs' discretization errors are reported against a highly-resolved reference at
matched $\Delta x$ — otherwise "the caveat disappears" could just mean we bought
exactness with a worse solver. Péclet stays $\le 1.2$ on this family, so central
differencing is legitimately stable rather than a convenient choice.

### 3. 2D incompressible Navier–Stokes — the headline

Divergence-free bank (POD of div-free snapshots is automatically div-free, since
divergence is linear), skew-symmetric convection, pressure drops out of the weak form
when tested against div-free modes. Exactly quadratic; no positivity assumption.

**Known architectural obstacle, found 2026-08-30 before any code was written:** the
decoder's scalar boundary mask destroys divergence-freeness —
$\nabla\!\cdot(\mathrm{bc}\cdot Gh)=\nabla \mathrm{bc}\cdot(Gh)+\mathrm{bc}\,\nabla\!\cdot(Gh)$,
and the first term is not zero. So `bc(x)` must go, and the div-free modes have to
satisfy the boundary conditions themselves (POD of BC-satisfying snapshots does this
automatically). The NS decoder is therefore architecturally different from every
decoder trained in this project so far.

Sweep $R$ inside this item rather than as a separate study — that answers the
$R$-frontier question where it actually matters.

### 4. Multi-seed + 32 trajectories, paired CIs (1D N=512, 2D N=256)

A debt, not a bet: the "1–9 % better than NNLS-32" claim is already retracted and
$n=8$ cannot support a replacement either way. Also folds in the two small items from
the handoff — a numeric `min(truth) >= 0` assert in the 1D driver (it currently leans
on a max-principle argument) and per-step $\lVert u_T-u_{or}\rVert$ persisted to the
comparison JSON.

### 5. Fixed sine bank, P=32 then 48

If 32 sines match the learned bank's held-out reconstruction ($\le 3.7\times10^{-3}$),
then $B=I$, $T$ is analytic, and Burgers is quadrature-free in the literal Poisson
sense with no tensor to store at all. One training job; the best upside-per-hour on
the list, and it could reshape the paper.

### 6. 3D linear (Poisson / heat) quadrature-free

$n$ grows as $N^3$ and $R$ does not, so $n$-independence is worth far more in 3D than
in 2D. `heat3d` / `poisson3d` configs already exist. If the crossover is not dramatic
here, it will not be in 3D nonlinear either.

### 7. Compress $T$ itself (CP / Tucker), if the $R$-frontier from item 3 demands it

Compressing the **tensor**, not the head — the head-PCA attempt already failed
(499/512 directions needed at $R=512$).

### 8. 3D incompressible NS — requires item 3 working and item 6's cost story.

### 9. Compressible NS via lifting

$\zeta=1/\rho$ makes Euler/NS exactly quadratic (the Lift & Learn transformation).
Subsonic and smooth only — shocks reintroduce limiters, which are non-polynomial.
Last deliberately: it retrains the decoder on transformed variables, so the entire
stage-1 target changes, and it is the one item that can fail for reasons unrelated to
our contribution.

## Ordering rationale, stated once

Items 1, 2, and 5 are each cheap and can each individually reshape everything below
them. Item 3 is the expensive one and sits behind them on purpose. Two dependencies
are real rather than conventional: waves → NS (vector-valued plumbing plus the
$n$-width risk), and 2D NS → 3D NS.

One correction to an earlier version of this list: the $R$-frontier was originally
gated *before* NS. That was too strong. A dense 2D tensor at $R=512$, $M=64$ is about
134 MB per component block — under a gigabyte for the full vector-valued set, which
fits comfortably on an 80 GB A100. The $R$-frontier decides the *speed claim* and it
decides 3D; it does not block building 2D NS.
