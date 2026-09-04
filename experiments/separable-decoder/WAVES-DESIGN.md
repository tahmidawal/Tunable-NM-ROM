# Waves 1D — vector-valued state, reflective and non-reflective boundaries

Design document for step 1 of the 2026-08-30 experiment roadmap (item 7).
Branch `exp/2026-08-30-waves-vector`, cut from `exp/2026-08-29-b2d-tensor`.
**Status: DESIGN, under audit. No numbers here.**

## Why this experiment, and what it decides

Three things get decided here, in the order in which they would kill the
incompressible-NS plan:

1. **Vector-valued state.** Every decoder trained in this project so far emits one
   scalar field. The wave equation written as a first-order system carries $(u,v)$;
   2D NS carries $(u_x,u_y)$. This is the cheapest possible place to make that
   change, because the residual here is *entirely linear* — no quadrature or tensor
   machinery is in play while the plumbing is debugged. If the vector-valued
   pipeline breaks, we find out in a setting with an exactly known answer.
2. **The general form of the linear claim.** The Poisson quadrature-free result
   leans on $\Phi$ being exact eigenvectors of the discrete Laplacian, so that
   $\Phi^\top(\Delta u)=\Lambda\Phi^\top u$. An absorbing boundary makes the spatial
   operator **non-symmetric**, and $\Lambda$ dies. Claim under test: storing the
   dense $M\times R$ matrix $A=\Phi^\top L G$ instead is *still exact* and *still
   zero-sample*. Right now our published linear claim quietly assumes self-adjointness.
3. **Decoder $n$-width.** Traveling waves have slowly decaying Kolmogorov $n$-width.
   If $K=8\!-\!32$ cannot represent them, NS at any interesting Reynolds number will
   fail for reasons that have nothing to do with our contribution — and we learn it
   in days instead of weeks.

## Governing equations

$$\partial_t u = v,\qquad \partial_t v = c^2\,\partial_{xx}u,\qquad x\in(0,1).$$

- **Reflective:** $u(0,t)=u(1,t)=0$ (homogeneous Dirichlet, hard wall).
- **Non-reflective:** first-order Sommerfeld radiation condition,
  $\partial_t u + c\,\partial_x u = 0$ at $x=1$ (outgoing right) and
  $\partial_t u - c\,\partial_x u = 0$ at $x=0$ (outgoing left).

**Structural consequence to handle explicitly:** under Sommerfeld the boundary
values are *unknowns*, not zeros. The interior-only, ghost-zero convention used by
every existing script here (`b1d_common`, `blat_common`) does not apply — the state
carries $n=N$ nodes in the absorbing case versus $n=N-2$ in the reflective case, and
the decoder's `bc(x)` mask must be dropped for the absorbing case. Treat these as
two configurations, not one with a flag.

## Discretization

- Space: 3-point Laplacian, $\Delta x = 1/(N-1)$. One-sided first-order difference
  in the Sommerfeld boundary rows.
- Time: **Crank–Nicolson** ($\theta=1/2$). Backward Euler is not acceptable here —
  it damps waves artificially, which would contaminate the $n$-width probe (item 3
  above) with numerical dissipation and make a representable solution look
  unrepresentable.
- FOM state $w=(u,v)\in\mathbb{R}^{2n}$.

## Data family

Traveling Gaussian pulses: $u_0=\exp(-(x-x_0)^2/2\sigma^2)$ with
$v_0=\mp c\,\partial_x u_0$ selecting a right- or left-going wave; plus a
two-pulse superposition mode so the family is not rank-1 in disguise.
Parameters: $x_0\sim U(0.25,0.75)$, $\sigma\sim U(0.02,0.08)$, direction $\pm$,
and $c\sim U(0.8,1.2)$ as the continuous parameter (the analogue of $\nu$ in Burgers).
Same train/held-out/fresh-seed cohort convention as the incumbent scripts.

## Decoder

Shared frozen bank $G\in\mathbb{R}^{n\times R}$; head emits $2R$ outputs split into
$h_u(z),h_v(z)\in\mathbb{R}^R$:

$$u(x;z)=\mathrm{bc}(x)\,\langle g(x),h_u(z)\rangle,\qquad v(x;z)=\mathrm{bc}(x)\,\langle g(x),h_v(z)\rangle$$

with the `bc` mask present in the reflective configuration and **absent** in the
absorbing one. A shared bank is the starting point; if held-out reconstruction is
much worse than with two independent banks, record that and say so.

## Quadrature-free ROM residual

Per Crank–Nicolson step, testing against $M$ modes $\Phi$, with $B=\Phi^\top G$ and
$A=\Phi^\top L G$ both precomputed **once**:

$$r_u = \frac{B\,(h_u^{n+1}-h_u^{n})}{\Delta t} - \tfrac12 B\,(h_v^{n+1}+h_v^{n})$$
$$r_v = \frac{B\,(h_v^{n+1}-h_v^{n})}{\Delta t} - \frac{c^2}{2}\,A\,(h_u^{n+1}+h_u^{n})$$

Zero sample points, no NNLS fit, no sampling error — the Poisson argument, but with
a dense $A$ in place of the diagonal $\Lambda$.

## Gates

**FOM gates (phase 1 — must pass before any ROM work begins):**

- **F1** Reflective + CN conserves the discrete energy
  $E=\tfrac12\|v\|^2+\tfrac{c^2}{2}\|Du\|^2$ to $\le 10^{-10}$ relative over the
  full horizon.
- **F2** Convergence against the analytic standing wave
  $u=\sin(k\pi x)\cos(k\pi ct)$ at $N=128,256,512,1024$: observed order $\approx 2$
  in both $\Delta x$ and $\Delta t$.
- **F3** Sommerfeld absorbs: a pulse launched at the right boundary leaves the
  domain with reflected energy $\le 10^{-3}$ of incident (1D normal incidence is
  the case first-order Sommerfeld handles essentially exactly). The reflective run
  at the same time retains $\approx 100\%$ — report both, side by side. If F3 does
  not hold, the absorbing experiment is testing nothing and must be fixed first.
- **F4** No growth over $4\times$ the horizon (CN is unconditionally stable).

**ROM gates (phase 2):**

- **W0** QF residual vs the full-grid residual: $\le 10^{-12}$ relative at $\ge 32$
  seeded random states *and* at every captured solve solution; gradient $J^\top r$
  to $\le 10^{-10}$ relative **or** the cancellation-aware absolute form
  $\|g_{qf}-g_{full}\|/(\|J_{full}\|_F\|r_{full}\|)\le 10^{-12}$ — the same
  convention as gate Q in `sep_poisson_qf.py`.
- **W1 — the point of the experiment.** Reflective: $\|A-\Lambda B\|/\|A\|\le
  10^{-12}$, i.e. the diagonal shortcut is provably valid. Absorbing: report
  $\|A-\Lambda B\|/\|A\|$ (expected $\gg 10^{-12}$), and show that the dense-$A$
  residual still passes W0 **while the $\Lambda$-shortcut residual visibly fails**.
  The general form must be *demonstrated*, not asserted.
- **W2** Cost: QF vs full-grid reference vs an NNLS-sampled arm, under the
  incumbent timing conventions (balanced order, all raw repetitions retained,
  one-time setup cost — NNLS fit vs $A$/$B$ build — timed separately).
- **W3** $n$-width probe: train at $K\in\{4,8,16,32,64\}$ at fixed $R$; report
  held-out reconstruction vs $K$ against the POD singular-value decay of the same
  snapshots as the linear-subspace reference. State plainly whether the decoder
  beats the POD floor at matched $K$ or hits a wall.

## Deliverables

Code and run JSONs in this worktree under `runs/wav1d/`; a `WAVES-NOTES.md`
recording what ran, what was found, and what was retracted; commit and push to
`origin/exp/2026-08-30-waves-vector`. Every reported number generated from the run
JSONs by a script — never hand-typed.
